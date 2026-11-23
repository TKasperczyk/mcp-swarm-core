"""Disk-backed task store, SIGCHLD reaping, cancellation, and sync/async runner."""

import json
import logging
import os
import secrets
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import diagnostics, platform
from .backend import RuntimeHooks, SwarmBackend

# ---------------------------------------------------------------------------
# Process-wide SIGCHLD handler -- reap only tracked async children
# ---------------------------------------------------------------------------
# Values: NOT_REAPED sentinel = not yet reaped; int = real exit code.
# Exit codes: >= 0 for normal exit, negative (-signal) for signal death.
NOT_REAPED = object()
ASYNC_PIDS: Dict[int, Any] = {}  # pid -> NOT_REAPED | int | None
ASYNC_PROCS: Dict[int, subprocess.Popen] = {}  # pid -> Popen (kept alive for returncode)
_sigchld_installed = False
_sigchld_install_lock = threading.Lock()


def sigchld_handler(signum, frame):
    # Strictly lock-free: never invoke logging or anything that acquires locks
    # inside a POSIX signal handler, or SIGCHLD arriving mid-log can deadlock.
    for pid in list(ASYNC_PIDS):
        if ASYNC_PIDS.get(pid) is not NOT_REAPED:
            continue  # already reaped or removed by another thread
        try:
            rpid, status = os.waitpid(pid, os.WNOHANG)
            if rpid == pid:
                if os.WIFEXITED(status):
                    exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    exit_code = -os.WTERMSIG(status)  # negative = killed by signal
                else:
                    exit_code = 127  # unknown abnormal termination
                if ASYNC_PIDS.get(pid) is NOT_REAPED:
                    ASYNC_PIDS[pid] = exit_code
        except ChildProcessError:
            # Already reaped by subprocess internals -- exit code lost.
            # Guard with NOT_REAPED so we never resurrect a key just popped
            # by finalize_dead_task_locked() or cancel_task().
            if ASYNC_PIDS.get(pid) is NOT_REAPED:
                ASYNC_PIDS[pid] = None
        except OSError:
            pass


def install_sigchld_handler() -> bool:
    """Install the POSIX ``SIGCHLD`` reaper explicitly (idempotent, main-thread safe).

    Deliberately NOT called at module import time so importing ``mcp_swarm_core``
    from a worker thread or library caller has no global signal side effects.
    """
    global _sigchld_installed
    if _sigchld_installed or not hasattr(signal, "SIGCHLD"):
        return _sigchld_installed
    if threading.current_thread() is not threading.main_thread():
        return False
    with _sigchld_install_lock:
        if not _sigchld_installed:
            signal.signal(signal.SIGCHLD, sigchld_handler)
            _sigchld_installed = True
    return True


def _open_private_write(path: Path):
    """Open ``path`` for UTF-8 writing with ``0o600`` permissions (owner-only)."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        return os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def write_meta_atomic(
    meta_file: Path,
    meta: Dict[str, Any],
    *,
    is_windows: Optional[bool] = None,
) -> None:
    """Atomically write task metadata via temp file + ``os.replace``."""
    win = platform.IS_WINDOWS if is_windows is None else bool(is_windows)
    meta_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = json.dumps(meta, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{meta_file.stem}.",
        suffix=".tmp",
        dir=str(meta_file.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
        retries = 5 if win else 1
        for attempt in range(retries):
            try:
                os.replace(tmp_path, meta_file)
                break
            except PermissionError:
                if not win or attempt == retries - 1:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class SyncProcRegistry:
    """Track in-flight synchronous child processes for cancellation and shutdown."""

    def __init__(self) -> None:
        self.procs: Dict[Any, subprocess.Popen] = {}
        self.lock = threading.Lock()

    def register(self, request_id: Any, proc: subprocess.Popen) -> None:
        if request_id is None:
            return
        with self.lock:
            self.procs[request_id] = proc

    def unregister(self, request_id: Any) -> None:
        if request_id is None:
            return
        with self.lock:
            self.procs.pop(request_id, None)

    def has(self, request_id: Any) -> bool:
        with self.lock:
            return request_id in self.procs

    def terminate(
        self,
        request_id: Any,
        *,
        is_valid_id_fn: Callable[[Any], bool],
        stop_proc_fn: Optional[Callable[..., None]] = None,
    ) -> bool:
        """Terminate the sync child for ``request_id`` if still running."""
        if not is_valid_id_fn(request_id):
            return False
        with self.lock:
            proc = self.procs.get(request_id)
        if proc is None or proc.poll() is not None:
            return False
        stopper = stop_proc_fn or platform.stop_proc
        stopper(proc, own_group=True)
        logging.info(
            "Terminated sync child pid=%s for request id=%s", proc.pid, request_id
        )
        return True

    def terminate_all(
        self,
        *,
        is_windows: Optional[bool] = None,
        win_kill_tree_fn: Optional[Callable[[int], bool]] = None,
    ) -> None:
        """Kill any surviving sync child process trees on server shutdown."""
        win = platform.IS_WINDOWS if is_windows is None else bool(is_windows)
        win_kill = win_kill_tree_fn or platform.win_kill_tree
        with self.lock:
            items = list(self.procs.items())
        for request_id, proc in items:
            alive = proc.poll() is None
            group_alive = (not win) and platform.posix_group_alive(proc.pid)
            if alive or group_alive:
                logging.info(
                    "Shutdown: killing sync child pid=%s (request %s)",
                    proc.pid,
                    request_id,
                )
                if win:
                    win_kill(proc.pid)
                else:
                    platform.posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)
                if alive:
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass


class TaskManager:
    """Backend-agnostic sync + async task lifecycle engine."""

    def __init__(
        self,
        backend: SwarmBackend,
        hooks: RuntimeHooks,
        *,
        sync_procs: Optional[SyncProcRegistry] = None,
        is_cancelled_fn: Optional[Callable[[Any], bool]] = None,
        send_progress_fn: Optional[
            Callable[[Any, float, str, Optional[str]], None]
        ] = None,
    ) -> None:
        self.backend = backend
        self.hooks = hooks
        self.sync_procs = sync_procs or SyncProcRegistry()
        self._is_cancelled_fn = is_cancelled_fn or (lambda _rid: False)
        self._send_progress_fn = send_progress_fn or (
            lambda _tok, _elapsed, _label, _detail=None: None
        )
        self._task_finalize_locks: Dict[str, threading.Lock] = {}
        self._task_finalize_guard = threading.Lock()

    def _write_meta(self, meta_file: Path, meta: Dict[str, Any]) -> None:
        if self.hooks._write_meta_atomic_fn is not None:
            self.hooks._write_meta_atomic_fn(meta_file, meta)
        else:
            write_meta_atomic(meta_file, meta, is_windows=self.hooks.is_windows())

    def get_task_lock(self, task_id: str) -> threading.Lock:
        with self._task_finalize_guard:
            if task_id not in self._task_finalize_locks:
                self._task_finalize_locks[task_id] = threading.Lock()
            return self._task_finalize_locks[task_id]

    @staticmethod
    def _path_is_within(child: str, parent: str) -> bool:
        try:
            child_r = os.path.realpath(child)
            parent_r = os.path.realpath(parent)
            return os.path.commonpath([child_r, parent_r]) == parent_r
        except (OSError, ValueError):
            return False

    def _touch_worktree_activity(self, cwd: Optional[str]) -> None:
        """Extend ``last_updated_at`` on any owning worktree meta containing ``cwd``."""
        if not cwd:
            return
        task_dir = self.hooks.task_dir()
        if not task_dir.exists():
            return
        now = time.time()
        for meta_file in task_dir.glob("*.meta"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                wt_root = meta.get("worktree_root") or meta.get("worktree_path")
                if not wt_root or not self._path_is_within(cwd, wt_root):
                    continue
                task_id = meta_file.stem
                with self.get_task_lock(task_id):
                    if not meta_file.exists():
                        continue
                    fresh = json.loads(meta_file.read_text(encoding="utf-8"))
                    fresh["last_updated_at"] = max(
                        float(fresh.get("last_updated_at", 0)), now
                    )
                    self._write_meta(meta_file, fresh)
            except Exception:
                continue

    def _collect_protected_worktree_paths(
        self, task_dir: Path, now: float, max_age: int
    ) -> List[Tuple[str, str]]:
        """Return ``[(task_id, path), ...]`` for running or non-expired tasks."""
        protected: List[Tuple[str, str]] = []
        for meta_file in task_dir.glob("*.meta"):
            try:
                tid = meta_file.stem
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                status = meta.get("status")
                is_active = False
                if status == "running":
                    pid = meta.get("pid")
                    if pid is not None and self.hooks.is_alive(
                        pid, expected_start_time=meta.get("pid_start_time")
                    ):
                        is_active = True
                    elif pid is None and now - meta.get("started_at", 0) < max_age:
                        is_active = True
                elif status in ("completed", "failed", "cancelled"):
                    last_act = max(
                        float(meta.get("completed_at", 0)),
                        float(meta.get("last_updated_at", 0)),
                    )
                    if now - last_act < max_age:
                        is_active = True
                if is_active:
                    for key in ("cwd", "worktree_path", "worktree_root"):
                        val = meta.get(key)
                        if isinstance(val, str) and val:
                            protected.append((tid, val))
            except Exception:
                continue
        return protected

    def cleanup_old_tasks(self) -> None:
        """Remove task artifacts older than ``task_max_age`` seconds."""
        now = time.time()
        if now - self.hooks.get_last_cleanup() < 300:  # Run at most every 5 minutes
            return
        self.hooks.set_last_cleanup(now)

        task_dir = self.hooks.task_dir()
        max_age = self.hooks.task_max_age()
        if not task_dir.exists():
            return

        protected_paths = self._collect_protected_worktree_paths(
            task_dir, now, max_age
        )

        for meta_file in task_dir.glob("*.meta"):
            try:
                task_id = meta_file.stem
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                status = meta.get("status")

                # Finalize unpolled "running" tasks whose process has died
                if status == "running":
                    pid = meta.get("pid")
                    pid_start_time = meta.get("pid_start_time")
                    if pid is not None and self.hooks.is_alive(
                        pid, expected_start_time=pid_start_time
                    ):
                        continue  # still running, skip
                    if pid is not None:
                        self.resolve_task_state(task_id)
                    else:
                        # Orphaned sync-worktree pre-launch meta whose server died
                        started = meta.get("started_at", 0)
                        if now - started < max_age:
                            continue
                        meta["status"] = "failed"
                        meta["completed_at"] = started
                        self._write_meta(meta_file, meta)

                lock = self.get_task_lock(task_id)
                with lock:
                    if not meta_file.exists():
                        continue
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    status = meta.get("status")
                    if status not in ("completed", "failed", "cancelled"):
                        continue
                    last_activity = max(
                        float(meta.get("completed_at", 0)),
                        float(meta.get("last_updated_at", 0)),
                    )
                    if now - last_activity < max_age:
                        continue

                    wt_root = meta.get("worktree_root") or meta.get("worktree_path")
                    worktree_branch = meta.get("worktree_branch")
                    if wt_root:
                        # Protect worktree if any other live/recent task (e.g. a
                        # reply running inside the worktree) is still using it.
                        in_use = any(
                            other_tid != task_id
                            and self._path_is_within(other_path, wt_root)
                            for other_tid, other_path in protected_paths
                        )
                        if in_use:
                            logging.debug(
                                "Deferring cleanup of task %s (worktree %s still in use)",
                                task_id,
                                wt_root,
                            )
                            continue
                        if not self.hooks.remove_worktree(
                            wt_root, worktree_branch or ""
                        ):
                            logging.warning(
                                "Deferring cleanup of task %s (worktree removal failed)",
                                task_id,
                            )
                            continue

                    self.backend.invalidate_jsonl_cache(task_dir / f"{task_id}.stdout")
                    for ext in (".meta", ".stdout", ".stderr", ".stdin"):
                        try:
                            (task_dir / f"{task_id}{ext}").unlink(missing_ok=True)
                        except OSError:
                            pass

                with self._task_finalize_guard:
                    self._task_finalize_locks.pop(task_id, None)

                logging.debug("Cleaned up old task %s", task_id)
            except Exception:
                continue

        # Clean up any orphaned Windows sync prompt files left by a hard crash
        for stdin_file in task_dir.glob("sync_*.stdin"):
            try:
                if now - stdin_file.stat().st_mtime >= max_age:
                    stdin_file.unlink(missing_ok=True)
            except OSError:
                continue

    def start_async(
        self,
        params: Dict[str, Any],
        *,
        high_entropy_id: bool = False,
        mcp_task: bool = False,
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Launch a detached async CLI task and persist its metadata atomically."""
        install_sigchld_handler()
        self.cleanup_old_tasks()

        task_id = secrets.token_hex(16) if high_entropy_id else secrets.token_hex(4)
        cmd_params = dict(params)
        worktree_enabled = bool(cmd_params.pop("worktree", False))
        worktree_root: Optional[str] = None
        worktree_path: Optional[str] = None
        worktree_branch: Optional[str] = None
        resolved_model = self.backend.resolve_model(cmd_params)

        # If resuming without explicit cwd, allow backend to recover prior cwd/worktree
        session_param = cmd_params.get(self.backend.session_param_name)
        if session_param and not cmd_params.get("cwd") and not worktree_enabled:
            recovered_cwd = self.backend.resolve_reply_cwd(
                str(session_param), cmd_params
            )
            if recovered_cwd:
                cmd_params["cwd"] = recovered_cwd

        if worktree_enabled:
            worktree_root, wt_cwd, worktree_branch = self.hooks.create_worktree(
                task_id,
                cmd_params.get("cwd"),
                branch_prefix=self.backend.branch_prefix,
            )
            worktree_path = wt_cwd
            cmd_params["cwd"] = wt_cwd

        stdout_f: Optional[Path] = None
        stderr_f: Optional[Path] = None
        stdin_f: Optional[Path] = None
        proc: Optional[subprocess.Popen] = None
        win = self.hooks.is_windows()
        task_dir = self.hooks.task_dir()
        try:
            cmd, cwd = self.backend.build_command(cmd_params)
            cmd = self.backend.prepare_command_for_json(cmd)
            logging.info("Async start [%s]: %s", task_id, shlex.join(cmd))

            task_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            stdout_f = task_dir / f"{task_id}.stdout"
            stderr_f = task_dir / f"{task_id}.stderr"
            stdin_f = (task_dir / f"{task_id}.stdin") if win else None
            stdin_payload = self.backend.format_stdin_payload(cmd_params)

            with _open_private_write(stdout_f) as out, _open_private_write(
                stderr_f
            ) as err, platform.prompt_stdin(
                stdin_payload,
                stdin_path=stdin_f,
                task_dir=task_dir,
                is_windows=win,
            ) as stdin_fh:
                proc = subprocess.Popen(
                    self.backend.exec_argv(cmd),
                    stdin=stdin_fh,
                    stdout=out,
                    stderr=err,
                    cwd=cwd,
                    **platform.TEXT_KW,
                    **self.hooks.popen_kw(detach=True),
                )

            ASYNC_PIDS[proc.pid] = NOT_REAPED
            ASYNC_PROCS[proc.pid] = proc

            started_at = time.time()
            effective_cwd = (
                worktree_path
                or (os.path.abspath(cmd_params["cwd"]) if cmd_params.get("cwd") else None)
                or cwd
            )
            meta: Dict[str, Any] = {
                "task_id": task_id,
                "pid": proc.pid,
                "pid_start_time": self.hooks.get_pid_start_time(proc.pid),
                "status": "running",
                "command": shlex.join(cmd),
                "started_at": started_at,
                "last_updated_at": started_at,
            }
            if not win:
                meta["own_process_group"] = True
            if resolved_model:
                meta["model"] = resolved_model
            if effective_cwd:
                meta["cwd"] = effective_cwd
            if mcp_task:
                meta["mcp_task"] = True
                meta["tool_name"] = tool_name or self.backend.async_tool_name
            if session_param:
                sid_field = self.backend.session_field_name
                meta[sid_field] = str(session_param)
                meta[f"requested_{sid_field}"] = str(session_param)
            if worktree_root:
                meta["worktree_root"] = worktree_root
            if worktree_path:
                meta["worktree_path"] = worktree_path
            if worktree_branch:
                meta["worktree_branch"] = worktree_branch
            extra_meta = self.backend.extra_meta_on_start(cmd_params)
            if extra_meta:
                meta.update(extra_meta)
            self._write_meta(task_dir / f"{task_id}.meta", meta)
        except Exception:
            if proc is not None:
                if callable(getattr(proc, "poll", None)) and proc.poll() is None:
                    self.hooks.stop_proc(proc, own_group=True)
                ASYNC_PIDS.pop(proc.pid, None)
                ASYNC_PROCS.pop(proc.pid, None)
            for path in (stdout_f, stderr_f, stdin_f):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            if worktree_root and worktree_branch:
                self.hooks.remove_worktree(worktree_root, worktree_branch)
            raise

        return {
            "task_id": task_id,
            "worktree_path": worktree_path,
            "worktree_branch": worktree_branch,
            "model": resolved_model,
            "started_at": started_at,
        }

    def terminal_state_dict(
        self, task_id: str, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        started_at = meta.get("started_at", time.time())
        completed_at = meta.get("completed_at", started_at)
        status_inferred = (
            bool(meta["status_inferred"])
            if "status_inferred" in meta
            else bool(meta.get("exit_code_lost"))
        )
        return {
            "status": meta["status"],
            "task_id": task_id,
            "meta": meta,
            "elapsed_seconds": int(completed_at - started_at),
            "exit_code": meta.get("exit_code"),
            "exit_code_lost": bool(meta.get("exit_code_lost")),
            "status_inferred": status_inferred,
        }

    def finalize_dead_task_locked(
        self, task_id: str, meta_file: Path, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize a dead task while holding ``get_task_lock(task_id)``."""
        started_at = meta.get("started_at", time.time())
        pid = meta.get("pid")
        win = self.hooks.is_windows()

        # Priority: SIGCHLD handler > Popen.poll() (with race re-check) > manual waitpid.
        exit_code = None
        sigchld_code = ASYNC_PIDS.get(pid, NOT_REAPED)
        if sigchld_code is not NOT_REAPED and sigchld_code is not None:
            exit_code = sigchld_code
        elif sigchld_code is NOT_REAPED:
            proc = ASYNC_PROCS.get(pid)
            if proc is not None:
                proc.poll()
                raced_code = ASYNC_PIDS.get(pid, NOT_REAPED)
                if raced_code is not NOT_REAPED and raced_code is not None:
                    exit_code = raced_code
                elif proc.returncode is not None:
                    exit_code = proc.returncode
            if exit_code is None and pid is not None and not win:
                try:
                    rpid, wstatus = os.waitpid(pid, os.WNOHANG)
                    if rpid == pid:
                        if os.WIFEXITED(wstatus):
                            exit_code = os.WEXITSTATUS(wstatus)
                        elif os.WIFSIGNALED(wstatus):
                            exit_code = -os.WTERMSIG(wstatus)
                        else:
                            exit_code = 127
                except ChildProcessError:
                    pass
                except Exception:
                    pass

        completed_at = time.time()
        exit_code_lost = exit_code is None
        task_dir = self.hooks.task_dir()
        stdout_path = task_dir / f"{task_id}.stdout"
        stderr_text = self.hooks.safe_read(task_dir / f"{task_id}.stderr")
        analysis, _ = self.backend.parse_jsonl_file(stdout_path)
        final_status, status_inferred = self.backend.decide_run_status(
            exit_code, analysis, stderr_text
        )

        sid_field = self.backend.session_field_name
        observed_sid = analysis.get(sid_field)
        requested_sid = meta.get(f"requested_{sid_field}")
        if requested_sid and final_status == "completed":
            mismatch = self.backend.validate_observed_session(
                str(requested_sid), observed_sid, analysis, stderr_text
            )
            if mismatch is not None:
                final_status = "failed"
                status_inferred = False
                meta["session_mismatch_error"] = mismatch

        meta["status"] = final_status
        meta["completed_at"] = completed_at
        meta["last_updated_at"] = completed_at
        if exit_code is not None:
            meta["exit_code"] = exit_code
        if exit_code_lost:
            meta["exit_code_lost"] = True
        meta["status_inferred"] = status_inferred

        if observed_sid and (
            self.backend.prefer_observed_session_id or not meta.get(sid_field)
        ):
            meta[sid_field] = observed_sid
        if analysis.get("model") and not meta.get("model"):
            meta["model"] = analysis["model"]

        if final_status == "completed":
            effective_sid = meta.get(sid_field)
            effective_cwd = meta.get("worktree_path") or meta.get("cwd")
            if effective_sid and effective_cwd:
                self.backend.record_session_cwd(effective_sid, effective_cwd)

        try:
            self._write_meta(meta_file, meta)
        except Exception as exc:
            logging.warning(
                "Failed to persist final metadata for %s: %s", task_id, exc
            )

        if pid is not None:
            ASYNC_PIDS.pop(pid, None)
            ASYNC_PROCS.pop(pid, None)
        try:
            (task_dir / f"{task_id}.stdin").unlink(missing_ok=True)
        except OSError:
            pass

        return {
            "status": final_status,
            "task_id": task_id,
            "meta": meta,
            "elapsed_seconds": int(completed_at - started_at),
            "exit_code": exit_code,
            "exit_code_lost": exit_code_lost,
            "status_inferred": status_inferred,
        }

    def resolve_task_state(self, task_id: str) -> Dict[str, Any]:
        """Single source of truth for task lifecycle state."""
        if not diagnostics.validate_task_id(task_id):
            return {
                "status": "error",
                "task_id": task_id,
                "error": f"Invalid task ID: {task_id}",
            }

        meta_file = self.hooks.task_dir() / f"{task_id}.meta"
        if not meta_file.exists():
            return {
                "status": "not_found",
                "task_id": task_id,
                "error": f"Task {task_id} not found",
            }

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "status": "not_found",
                "task_id": task_id,
                "error": f"Task {task_id} not found",
            }
        except Exception as exc:
            return {
                "status": "error",
                "task_id": task_id,
                "error": f"Bad metadata: {exc}",
            }

        started_at = meta.get("started_at", time.time())
        if meta.get("status") in ("completed", "failed", "cancelled"):
            return self.terminal_state_dict(task_id, meta)

        pid = meta.get("pid")
        pid_start_time = meta.get("pid_start_time")

        if self.hooks.is_alive(pid, expected_start_time=pid_start_time):
            return {
                "status": "running",
                "task_id": task_id,
                "meta": meta,
                "elapsed_seconds": int(time.time() - started_at),
            }

        lock = self.get_task_lock(task_id)
        with lock:
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {
                    "status": "not_found",
                    "task_id": task_id,
                    "error": f"Task {task_id} not found",
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "task_id": task_id,
                    "error": f"Bad metadata: {exc}",
                }
            if meta.get("status") in ("completed", "failed", "cancelled"):
                return self.terminal_state_dict(task_id, meta)

            return self.finalize_dead_task_locked(task_id, meta_file, meta)

    def check_task(
        self, task_id: str, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check task status and return formatted result dict."""
        state = state if state is not None else self.resolve_task_state(task_id)
        status = state["status"]
        sid_field = self.backend.session_field_name
        task_dir = self.hooks.task_dir()

        if status in ("error", "not_found"):
            return state

        if status == "running":
            meta = state.get("meta", {})
            resp: Dict[str, Any] = {
                "status": "running",
                "task_id": task_id,
                "elapsed_seconds": state["elapsed_seconds"],
            }
            if meta.get("model"):
                resp["model"] = meta["model"]
            if meta.get("worktree_path"):
                resp["worktree_path"] = meta["worktree_path"]
            if meta.get("worktree_branch"):
                resp["worktree_branch"] = meta["worktree_branch"]
            sid = self.backend.extract_session_id_fast(
                task_dir / f"{task_id}.stdout", meta
            )
            if sid:
                resp[sid_field] = sid
            return resp

        stdout_path = task_dir / f"{task_id}.stdout"
        stderr_path = task_dir / f"{task_id}.stderr"
        analysis, _ = self.backend.parse_jsonl_file(stdout_path)
        stdout: Optional[str] = None
        stderr: Optional[str] = None
        if analysis.get("saw_json") and analysis.get("last_assistant_text"):
            result = analysis["last_assistant_text"]
            session_id = analysis.get(sid_field)
        else:
            stdout = self.hooks.safe_read(stdout_path)
            stderr = self.hooks.safe_read(stderr_path)
            result, session_id = self.backend.extract_result(
                stdout, stderr, analysis=analysis
            )
        if stderr is None and self.backend.wants_stderr_on_success:
            stderr = self.hooks.safe_read(stderr_path)
        meta = state.get("meta", {})
        session_id = session_id or meta.get(sid_field)

        resp = {
            "status": status,
            "task_id": task_id,
            "result": result,
            "elapsed_seconds": state["elapsed_seconds"],
        }
        resolved_model = analysis.get("model") or meta.get("model")
        if resolved_model:
            resp["model"] = resolved_model
        if status in ("completed", "failed"):
            resp["exit_code"] = state.get("exit_code")
            resp["exit_code_lost"] = bool(state.get("exit_code_lost"))
            resp["status_inferred"] = bool(state.get("status_inferred"))
        if meta.get("worktree_path"):
            resp["worktree_path"] = meta["worktree_path"]
        if meta.get("worktree_branch"):
            resp["worktree_branch"] = meta["worktree_branch"]
        if session_id:
            resp[sid_field] = session_id

        extra = self.backend.extra_check_task_fields(
            analysis, meta, status, stderr=stderr or ""
        )
        if extra:
            resp.update(extra)

        if status == "failed":
            mismatch = meta.get("session_mismatch_error")
            if isinstance(mismatch, dict):
                failure = dict(mismatch)
                if not failure.get("model") and resolved_model:
                    failure["model"] = resolved_model
            else:
                if stderr is None:
                    stderr = self.hooks.safe_read(stderr_path)
                if stdout is None:
                    stdout = (
                        ""
                        if (analysis.get("terminal_error") or analysis.get("errors"))
                        else self.hooks.safe_read(stdout_path)
                    )
                failure = self.backend.build_failure_info(
                    stdout,
                    stderr,
                    command=meta.get("command"),
                    analysis=analysis,
                    model=resolved_model,
                )
            resp["failure_cause"] = failure["cause"]
            resp["model"] = failure.get("model")
            resp["failure_category"] = failure["category"]
            resp["retryable"] = failure["retryable"]
            resp["failure_action"] = failure["action"]
            resp["failure_detail"] = self.backend.format_failure_info(failure)
            if failure.get("stdout_diagnostics"):
                resp["stdout_diagnostics"] = failure["stdout_diagnostics"]
            if failure.get("stderr_diagnostics"):
                resp["stderr_diagnostics"] = failure["stderr_diagnostics"]
            if analysis.get("last_assistant_text"):
                resp["partial_output"] = analysis["last_assistant_text"]
        return resp

    def build_already_terminal_cancel_resp(
        self, task_id: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        status = state["status"]
        resp: Dict[str, Any] = {
            "status": status,
            "task_id": task_id,
            "message": f"Task already {status}",
            "already_terminal": True,
        }
        meta = state.get("meta", {})
        if meta.get("worktree_path"):
            resp["worktree_path"] = meta["worktree_path"]
        if meta.get("worktree_branch"):
            resp["worktree_branch"] = meta["worktree_branch"]
        if status == "failed":
            failure = self.check_task(task_id, state=state)
            for key in (
                "exit_code",
                "exit_code_lost",
                "status_inferred",
                "model",
                "failure_cause",
                "failure_category",
                "retryable",
                "failure_action",
                "failure_detail",
            ):
                if key in failure:
                    resp[key] = failure[key]
        return resp

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Kill a running async task (and its process tree) and mark it cancelled."""
        if not diagnostics.validate_task_id(task_id):
            return {
                "status": "error",
                "task_id": task_id,
                "error": f"Invalid task ID: {task_id}",
            }

        task_dir = self.hooks.task_dir()
        meta_file = task_dir / f"{task_id}.meta"
        if not meta_file.exists():
            return {
                "status": "not_found",
                "task_id": task_id,
                "error": f"Task {task_id} not found",
            }

        win = self.hooks.is_windows()
        lock = self.get_task_lock(task_id)
        with lock:
            if not meta_file.exists():
                return {
                    "status": "not_found",
                    "task_id": task_id,
                    "error": f"Task {task_id} not found",
                }

            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {
                    "status": "not_found",
                    "task_id": task_id,
                    "error": f"Task {task_id} not found",
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "task_id": task_id,
                    "error": f"Bad metadata: {exc}",
                }

            if meta.get("status") in ("completed", "failed", "cancelled"):
                terminal_state = self.terminal_state_dict(task_id, meta)
                return self.build_already_terminal_cancel_resp(task_id, terminal_state)

            pid = meta.get("pid")
            pid_start_time = meta.get("pid_start_time")

            if not (pid and self.hooks.is_alive(pid, expected_start_time=pid_start_time)):
                final_state = self.finalize_dead_task_locked(task_id, meta_file, meta)
                return self.build_already_terminal_cancel_resp(task_id, final_state)

            if win:
                if not self.hooks.win_kill_tree(pid):
                    return {
                        "status": "error",
                        "task_id": task_id,
                        "error": f"Failed to kill process tree of PID {pid}; task is still running",
                    }
                logging.info("Killed process tree of PID %d (task %s)", pid, task_id)
            else:
                own_group = bool(meta.get("own_process_group"))
                platform.posix_signal_tree(pid, signal.SIGTERM, own_group=own_group)
                logging.info(
                    "Sent SIGTERM to PID %d (group=%s, task %s)",
                    pid,
                    own_group,
                    task_id,
                )

                for _ in range(10):
                    time.sleep(0.05)
                    if not self.hooks.is_alive(pid, expected_start_time=pid_start_time):
                        break

                if self.hooks.is_alive(pid, expected_start_time=pid_start_time):
                    platform.posix_signal_tree(pid, signal.SIGKILL, own_group=own_group)
                    logging.info(
                        "Sent SIGKILL to PID %d (group=%s, task %s)",
                        pid,
                        own_group,
                        task_id,
                    )
                elif own_group and platform.posix_group_alive(pid):
                    time.sleep(0.05)
                    if platform.posix_group_alive(pid):
                        platform.posix_signal_tree(pid, signal.SIGKILL, own_group=True)
                        logging.info(
                            "Sent SIGKILL to surviving process group %d (task %s)",
                            pid,
                            task_id,
                        )

            proc = ASYNC_PROCS.get(pid) if pid else None
            if proc is not None:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            elif pid and not win:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass

            now = time.time()
            meta["status"] = "cancelled"
            meta["completed_at"] = now
            meta["last_updated_at"] = now
            try:
                self._write_meta(meta_file, meta)
            except Exception as exc:
                logging.warning(
                    "Failed to persist cancelled metadata for %s: %s", task_id, exc
                )

            if pid:
                ASYNC_PIDS.pop(pid, None)
                ASYNC_PROCS.pop(pid, None)
            try:
                (task_dir / f"{task_id}.stdin").unlink(missing_ok=True)
            except OSError:
                pass

        resp: Dict[str, Any] = {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Task cancelled",
        }
        if meta.get("worktree_path"):
            resp["worktree_path"] = meta["worktree_path"]
        if meta.get("worktree_branch"):
            resp["worktree_branch"] = meta["worktree_branch"]
        return resp

    def summarize_pending_activity(self, pending: List[str]) -> Optional[str]:
        """Build an enriched progress detail string across pending async tasks."""
        if not pending:
            return None
        task_dir = self.hooks.task_dir()
        if len(pending) == 1:
            _, st = self.backend.parse_jsonl_file(task_dir / f"{pending[0]}.stdout")
            return diagnostics.format_task_activity_brief(st)

        snippets = []
        for tid in pending[:3]:
            _, st = self.backend.parse_jsonl_file(task_dir / f"{tid}.stdout")
            brief = diagnostics.format_task_activity_brief(st)
            if brief:
                snippets.append(f"{tid[:8]}: {brief}")
        if not snippets:
            return None
        if len(pending) > 3:
            snippets.append(f"+{len(pending) - 3} more")
        return "; ".join(snippets)

    def wait_tasks(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
        request_id: Any = None,
        progress_token: Any = None,
    ) -> Dict[str, Any]:
        """Block until all tasks complete, timeout, or request is cancelled."""
        deadline = time.time() + timeout if timeout is not None else None
        results: Dict[str, Any] = {}
        started = time.time()
        last_progress = started
        product = self.backend.product_name

        while True:
            if request_id is not None and self._is_cancelled_fn(request_id):
                for tid in task_ids:
                    if tid not in results:
                        state = self.resolve_task_state(tid)
                        meta = state.get("meta", {}) if isinstance(state, dict) else {}
                        results[tid] = {
                            "status": "cancelled",
                            "task_id": tid,
                            "elapsed_seconds": (
                                state.get("elapsed_seconds", 0)
                                if isinstance(state, dict)
                                else 0
                            ),
                            "error": "Wait cancelled by client",
                            "wait_cancelled": True,
                        }
                        if meta.get("worktree_path"):
                            results[tid]["worktree_path"] = meta["worktree_path"]
                        if meta.get("worktree_branch"):
                            results[tid]["worktree_branch"] = meta["worktree_branch"]
                break

            pending = []
            for tid in task_ids:
                if tid in results:
                    continue
                state = self.resolve_task_state(tid)
                if state["status"] in (
                    "completed",
                    "failed",
                    "cancelled",
                    "not_found",
                    "error",
                ):
                    results[tid] = self.check_task(tid, state=state)
                else:
                    pending.append(tid)

            if not pending:
                break

            if deadline is not None and time.time() >= deadline:
                for tid in pending:
                    state = self.resolve_task_state(tid)
                    meta = state.get("meta", {}) if isinstance(state, dict) else {}
                    results[tid] = {
                        "status": "timeout",
                        "task_id": tid,
                        "elapsed_seconds": (
                            state.get("elapsed_seconds", 0)
                            if isinstance(state, dict)
                            else 0
                        ),
                        "error": "Still running (wait timed out, task NOT killed)",
                    }
                    if meta.get("worktree_path"):
                        results[tid]["worktree_path"] = meta["worktree_path"]
                    if meta.get("worktree_branch"):
                        results[tid]["worktree_branch"] = meta["worktree_branch"]
                break

            now = time.time()
            if now - last_progress >= self.hooks.progress_interval():
                last_progress = now
                waiting_on = len(pending)
                detail = (
                    self.summarize_pending_activity(pending)
                    if progress_token is not None
                    else None
                )
                self._send_progress_fn(
                    progress_token,
                    now - started,
                    f"{waiting_on} {product} task{'s' if waiting_on != 1 else ''}",
                    detail,
                )

            sleep_time = self.hooks.poll_interval()
            if deadline is not None:
                sleep_time = min(sleep_time, max(0.1, deadline - time.time()))
            time.sleep(sleep_time)

        return results

    def wait_proc_detailed(
        self,
        proc: subprocess.Popen,
        deadline: Optional[float] = None,
        request_id: Any = None,
        timeout_msg: Optional[str] = None,
        progress_token: Any = None,
        progress_label: Optional[str] = None,
        command: Any = None,
        model: Optional[str] = None,
        own_group: Optional[bool] = None,
        requested_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wait for a sync Popen child with cancellation, timeout, and progress support."""
        product = self.backend.product_name
        label = progress_label or product
        t_msg = timeout_msg or f"Error: {product} execution timed out"
        sid_field = self.backend.session_field_name
        started = time.time()
        last_progress = started
        poll_interval = self.hooks.poll_interval()

        while True:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.hooks.stop_proc(proc, own_group=own_group)
                    return {
                        "text": t_msg,
                        sid_field: None,
                        "failed": True,
                        "status": "failed",
                        "exit_code": proc.returncode,
                        "model": model or self.backend.model_from_command(command),
                        "failure": None,
                        "partial_output": None,
                    }
            wait_time = (
                min(poll_interval, remaining)
                if remaining is not None
                else poll_interval
            )
            try:
                stdout, stderr = proc.communicate(timeout=wait_time)
                if request_id is not None and self._is_cancelled_fn(request_id):
                    return {
                        "text": "Cancelled by client",
                        sid_field: None,
                        "failed": True,
                        "status": "cancelled",
                        "exit_code": proc.returncode,
                        "model": model or self.backend.model_from_command(command),
                        "failure": None,
                        "partial_output": None,
                    }
                analysis, _ = self.backend.parse_jsonl_text(stdout)
                status, _ = self.backend.decide_run_status(
                    proc.returncode, analysis, stderr
                )
                resolved_model = (
                    analysis.get("model")
                    or model
                    or self.backend.model_from_command(command)
                )
                mismatch = None
                if requested_session_id and status == "completed":
                    mismatch = self.backend.validate_observed_session(
                        requested_session_id,
                        analysis.get(sid_field),
                        analysis,
                        stderr,
                    )
                    if mismatch is not None:
                        status = "failed"
                        if not mismatch.get("model") and resolved_model:
                            mismatch["model"] = resolved_model

                if status == "failed":
                    failure = (
                        mismatch
                        if mismatch is not None
                        else self.backend.build_failure_info(
                            stdout,
                            stderr,
                            command=command,
                            analysis=analysis,
                            model=resolved_model,
                        )
                    )
                    exit_code = proc.returncode if proc.returncode is not None else "?"
                    text = (
                        f"{label} FAILED (exit {exit_code})\n"
                        f"{self.backend.format_failure_info(failure)}"
                    )
                    outcome: Dict[str, Any] = {
                        "text": text,
                        sid_field: analysis.get(sid_field),
                        "failed": True,
                        "status": "failed",
                        "exit_code": proc.returncode,
                        "model": failure.get("model"),
                        "failure": failure,
                        "partial_output": analysis.get("last_assistant_text"),
                    }
                    extra = self.backend.extra_sync_outcome_fields(
                        analysis, outcome, stderr=stderr
                    )
                    if extra:
                        outcome.update(extra)
                    return outcome

                result, session_id = self.backend.extract_result(
                    stdout, stderr, analysis=analysis
                )
                outcome = {
                    "text": result,
                    sid_field: session_id,
                    "failed": False,
                    "status": "completed",
                    "exit_code": proc.returncode,
                    "model": resolved_model,
                    "failure": None,
                    "partial_output": None,
                }
                extra = self.backend.extra_sync_outcome_fields(
                    analysis, outcome, stderr=stderr
                )
                if extra:
                    outcome.update(extra)
                return outcome
            except subprocess.TimeoutExpired:
                if request_id is not None and self._is_cancelled_fn(request_id):
                    self.hooks.stop_proc(proc, own_group=own_group)
                    return {
                        "text": "Cancelled by client",
                        sid_field: None,
                        "failed": True,
                        "status": "cancelled",
                        "exit_code": proc.returncode,
                        "model": model or self.backend.model_from_command(command),
                        "failure": None,
                        "partial_output": None,
                    }
                now = time.time()
                if now - last_progress >= self.hooks.progress_interval():
                    last_progress = now
                    self._send_progress_fn(progress_token, now - started, label, None)

    def run_sync_detailed(
        self,
        params: Dict[str, Any],
        request_id: Any = None,
        progress_token: Any = None,
    ) -> Dict[str, Any]:
        """Run a CLI invocation synchronously with cancellation and optional worktree."""
        product = self.backend.product_name
        sid_field = self.backend.session_field_name
        started_at = time.time()
        cmd_params = dict(params)
        worktree_info: Optional[Dict[str, str]] = None
        run_id: Optional[str] = None
        cmd: List[str] = []
        session_id: Optional[str] = None
        exit_code: Optional[int] = None
        meta_status = "failed"
        resolved_model = self.backend.resolve_model(cmd_params)
        proc_started = False
        proc = None

        worktree_enabled = bool(cmd_params.pop("worktree", False))
        requested_sid_raw = cmd_params.get(self.backend.session_param_name)
        requested_sid = (
            str(requested_sid_raw).strip()
            if isinstance(requested_sid_raw, str) and requested_sid_raw.strip()
            else None
        )

        if request_id is not None and self._is_cancelled_fn(request_id):
            return {
                "text": "Cancelled by client",
                sid_field: None,
                "worktree_info": None,
                "failed": True,
                "status": "cancelled",
                "exit_code": None,
                "model": resolved_model,
                "failure": None,
                "partial_output": None,
            }

        try:
            if requested_sid and not cmd_params.get("cwd") and not worktree_enabled:
                recovered_cwd = self.backend.resolve_reply_cwd(
                    requested_sid, cmd_params
                )
                if recovered_cwd:
                    cmd_params["cwd"] = recovered_cwd

            if worktree_enabled:
                self.cleanup_old_tasks()
                run_id = secrets.token_hex(4)
                wt_root, wt_cwd, worktree_branch = self.hooks.create_worktree(
                    run_id,
                    cmd_params.get("cwd"),
                    branch_prefix=self.backend.branch_prefix,
                )
                cmd_params["cwd"] = wt_cwd
                worktree_info = {
                    "worktree_root": wt_root,
                    "worktree_path": wt_cwd,
                    "worktree_branch": worktree_branch,
                }
                # Persist worktree metadata early so a hard-killed server never
                # leaves an untracked orphan worktree that cleanup misses.
                try:
                    self._write_meta(
                        self.hooks.task_dir() / f"{run_id}.meta",
                        {
                            "task_id": run_id,
                            "status": "running",
                            "started_at": started_at,
                            "last_updated_at": started_at,
                            "worktree_root": wt_root,
                            "worktree_path": wt_cwd,
                            "worktree_branch": worktree_branch,
                            "cwd": wt_cwd,
                        },
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to pre-persist sync worktree metadata for %s: %s",
                        run_id,
                        exc,
                    )

            if request_id is not None and self._is_cancelled_fn(request_id):
                if worktree_info:
                    self.hooks.remove_worktree(
                        worktree_info["worktree_root"],
                        worktree_info["worktree_branch"],
                    )
                    if run_id:
                        try:
                            (self.hooks.task_dir() / f"{run_id}.meta").unlink(
                                missing_ok=True
                            )
                        except OSError:
                            pass
                    worktree_info = None
                return {
                    "text": "Cancelled by client",
                    sid_field: None,
                    "worktree_info": None,
                    "failed": True,
                    "status": "cancelled",
                    "exit_code": None,
                    "model": resolved_model,
                    "failure": None,
                    "partial_output": None,
                }

            cmd, cwd = self.backend.build_command(cmd_params)
            cmd = self.backend.prepare_command_for_json(cmd)
            logging.info("Sync exec: %s", shlex.join(cmd))
            stdin_payload = self.backend.format_stdin_payload(cmd_params)
            with platform.prompt_stdin(
                stdin_payload,
                task_dir=self.hooks.task_dir(),
                is_windows=self.hooks.is_windows(),
            ) as stdin_fh:
                proc = subprocess.Popen(
                    self.backend.exec_argv(cmd),
                    stdin=stdin_fh,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    **platform.TEXT_KW,
                    **self.hooks.popen_kw(new_group=True),
                )
                if stdin_fh is not subprocess.DEVNULL:
                    stdin_fh.close()
                proc_started = True
                self.sync_procs.register(request_id, proc)
                if worktree_info and run_id:
                    try:
                        self._write_meta(
                            self.hooks.task_dir() / f"{run_id}.meta",
                            {
                                "task_id": run_id,
                                "pid": proc.pid,
                                "pid_start_time": self.hooks.get_pid_start_time(
                                    proc.pid
                                ),
                                "status": "running",
                                "started_at": started_at,
                                "last_updated_at": started_at,
                                "worktree_root": worktree_info["worktree_root"],
                                "worktree_path": worktree_info["worktree_path"],
                                "worktree_branch": worktree_info["worktree_branch"],
                                "cwd": worktree_info["worktree_path"],
                            },
                        )
                    except Exception as exc:
                        logging.warning(
                            "Failed to update sync worktree pid metadata for %s: %s",
                            run_id,
                            exc,
                        )
                if request_id is not None and self._is_cancelled_fn(request_id):
                    self.hooks.stop_proc(proc, own_group=True)
                    outcome: Dict[str, Any] = {
                        "text": "Cancelled by client",
                        sid_field: None,
                        "failed": True,
                        "status": "cancelled",
                        "exit_code": proc.returncode,
                        "model": resolved_model or self.backend.model_from_command(cmd),
                        "failure": None,
                        "partial_output": None,
                    }
                else:
                    outcome = self.wait_proc_detailed(
                        proc,
                        request_id=request_id,
                        timeout_msg=f"Error: {product} execution timed out",
                        progress_token=progress_token,
                        progress_label=product,
                        command=cmd,
                        model=resolved_model,
                        own_group=True,
                        requested_session_id=requested_sid,
                    )
            session_id = outcome.get(sid_field)
            exit_code = outcome["exit_code"]
            meta_status = outcome["status"]
            if outcome.get("model"):
                resolved_model = outcome["model"]
            effective_cwd = (
                (worktree_info["worktree_path"] if worktree_info else None)
                or (os.path.abspath(cmd_params["cwd"]) if cmd_params.get("cwd") else None)
                or cwd
            )
            if meta_status == "completed" and session_id and effective_cwd:
                self.backend.record_session_cwd(session_id, effective_cwd)
            outcome["worktree_info"] = worktree_info
            return outcome
        except Exception as exc:
            if (
                proc is not None
                and callable(getattr(proc, "poll", None))
                and proc.poll() is None
            ):
                self.hooks.stop_proc(proc, own_group=True)
            if worktree_info and not proc_started:
                self.hooks.remove_worktree(
                    worktree_info["worktree_root"], worktree_info["worktree_branch"]
                )
                if run_id:
                    try:
                        (self.hooks.task_dir() / f"{run_id}.meta").unlink(
                            missing_ok=True
                        )
                    except OSError:
                        pass
                worktree_info = None
            return {
                "text": f"Error calling {product}: {exc}",
                sid_field: None,
                "worktree_info": worktree_info,
                "failed": True,
                "status": "failed",
                "exit_code": None,
                "model": resolved_model,
                "failure": None,
                "partial_output": None,
            }
        finally:
            self.sync_procs.unregister(request_id)
            if worktree_info and run_id:
                completed_at = time.time()
                meta: Dict[str, Any] = {
                    "task_id": run_id,
                    "status": meta_status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "last_updated_at": completed_at,
                    "worktree_root": worktree_info["worktree_root"],
                    "worktree_path": worktree_info["worktree_path"],
                    "worktree_branch": worktree_info["worktree_branch"],
                    "cwd": worktree_info["worktree_path"],
                }
                if cmd:
                    meta["command"] = shlex.join(cmd)
                if resolved_model:
                    meta["model"] = resolved_model
                if exit_code is not None:
                    meta["exit_code"] = exit_code
                if session_id:
                    meta[sid_field] = session_id
                try:
                    self._write_meta(self.hooks.task_dir() / f"{run_id}.meta", meta)
                except Exception as exc:
                    logging.warning(
                        "Failed to persist sync worktree metadata for %s: %s",
                        run_id,
                        exc,
                    )

    def run_reply_sync_detailed(
        self,
        session_id: str,
        prompt: str,
        extra_params: Optional[Dict[str, Any]] = None,
        request_id: Any = None,
        progress_token: Any = None,
    ) -> Dict[str, Any]:
        """Continue an existing CLI session synchronously with cancellation support."""
        product = self.backend.product_name
        sid_field = self.backend.session_field_name
        params: Dict[str, Any] = dict(extra_params or {})
        params[self.backend.session_param_name] = session_id
        params["prompt"] = prompt
        if not params.get("cwd"):
            recovered_cwd = self.backend.resolve_reply_cwd(session_id, params)
            if recovered_cwd:
                params["cwd"] = recovered_cwd
        resolved_model = self.backend.resolve_model(params)

        if request_id is not None and self._is_cancelled_fn(request_id):
            return {
                "text": "Cancelled by client",
                sid_field: session_id,
                "failed": True,
                "status": "cancelled",
                "exit_code": None,
                "model": resolved_model,
                "failure": None,
                "partial_output": None,
            }

        cmd, cwd = self.backend.build_command(params)
        cmd = self.backend.prepare_command_for_json(cmd)
        logging.info("Reply exec: %s", shlex.join(cmd))
        self._touch_worktree_activity(cwd or params.get("cwd"))
        proc = None
        try:
            popen_kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                **platform.TEXT_KW,
                **self.hooks.popen_kw(new_group=True),
            }
            if cwd is not None:
                popen_kwargs["cwd"] = cwd
            stdin_payload = self.backend.format_stdin_payload(params)
            with platform.prompt_stdin(
                stdin_payload,
                task_dir=self.hooks.task_dir(),
                is_windows=self.hooks.is_windows(),
            ) as stdin_fh:
                proc = subprocess.Popen(
                    self.backend.exec_argv(cmd),
                    stdin=stdin_fh,
                    **popen_kwargs,
                )
                if stdin_fh is not subprocess.DEVNULL:
                    stdin_fh.close()
                self.sync_procs.register(request_id, proc)
                if request_id is not None and self._is_cancelled_fn(request_id):
                    self.hooks.stop_proc(proc, own_group=True)
                    outcome: Dict[str, Any] = {
                        "text": "Cancelled by client",
                        sid_field: session_id,
                        "failed": True,
                        "status": "cancelled",
                        "exit_code": proc.returncode,
                        "model": resolved_model or self.backend.model_from_command(cmd),
                        "failure": None,
                        "partial_output": None,
                    }
                else:
                    outcome = self.wait_proc_detailed(
                        proc,
                        request_id=request_id,
                        timeout_msg=f"Error: {product} reply timed out",
                        progress_token=progress_token,
                        progress_label=f"{product} reply",
                        command=cmd,
                        model=resolved_model,
                        own_group=True,
                        requested_session_id=session_id,
                    )
            if not outcome.get(sid_field):
                outcome[sid_field] = session_id
            if outcome["status"] == "completed" and cwd:
                self.backend.record_session_cwd(outcome.get(sid_field), cwd)
            return outcome
        except Exception as exc:
            if (
                proc is not None
                and callable(getattr(proc, "poll", None))
                and proc.poll() is None
            ):
                self.hooks.stop_proc(proc, own_group=True)
            return {
                "text": f"Error calling {product} reply: {exc}",
                sid_field: session_id,
                "failed": True,
                "status": "failed",
                "exit_code": None,
                "model": resolved_model,
                "failure": None,
                "partial_output": None,
            }
        finally:
            if proc is not None:
                self.sync_procs.unregister(request_id)
