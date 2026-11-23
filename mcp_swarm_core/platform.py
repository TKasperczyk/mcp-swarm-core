"""Cross-platform process, process-tree, liveness, private-dir, and stdin helpers.

Stdlib only. Supports Linux, macOS, and native Windows (via Win32 ctypes).
"""

import contextlib
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

IS_WINDOWS: bool = sys.platform == "win32"

# /tmp does not exist on Windows; use the per-user temp dir there.
DEFAULT_TMP: str = tempfile.gettempdir() if IS_WINDOWS else "/tmp"

# Subprocess text decoding. Always decode as UTF-8 with replacement on every
# platform so a stray non-UTF-8 byte or C/POSIX locale can never abort
# communicate() after a child run has already finished.
TEXT_KW: Dict[str, Any] = {"text": True, "encoding": "utf-8", "errors": "replace"}


def default_user_tmp_dir(app_slug: str) -> str:
    """Return a per-user temporary directory path safe on multi-user hosts.

    On Windows returns ``%TEMP%\\<app_slug>``. On POSIX prefers a user-owned
    ``$XDG_RUNTIME_DIR/<app_slug>`` when available, otherwise
    ``/tmp/<app_slug>-<uid>`` so different local users never collide.
    """
    if IS_WINDOWS:
        return os.path.join(tempfile.gettempdir(), app_slug)
    uid = os.getuid()
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        try:
            st = os.lstat(xdg)
            if (
                stat.S_ISDIR(st.st_mode)
                and not stat.S_ISLNK(st.st_mode)
                and st.st_uid == uid
                and os.access(xdg, os.W_OK | os.X_OK)
            ):
                return os.path.join(xdg, app_slug)
        except OSError:
            pass
    return os.path.join(DEFAULT_TMP, f"{app_slug}-{uid}")


def default_worktree_dir(app_slug: str) -> str:
    """Return the default worktree storage directory outside RAM-capped runtime dirs.

    On Windows returns ``%TEMP%\\<app_slug>\\worktrees``. On POSIX prefers
    ``${XDG_CACHE_HOME:-~/.cache}/<app_slug>/worktrees`` so full git checkouts
    do not fill ``$XDG_RUNTIME_DIR`` (10% RAM tmpfs wiped on logout), falling
    back to ``/tmp/<app_slug>-<uid>/worktrees`` if the home cache dir is
    unavailable.
    """
    if IS_WINDOWS:
        return os.path.join(tempfile.gettempdir(), app_slug, "worktrees")
    uid = os.getuid()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    candidates = []
    if xdg_cache and os.path.isabs(xdg_cache):
        candidates.append(Path(xdg_cache))
    try:
        candidates.append(Path.home() / ".cache")
    except RuntimeError:
        pass
    for cache_root in candidates:
        try:
            if cache_root.exists():
                st = os.lstat(str(cache_root))
                if (
                    stat.S_ISDIR(st.st_mode)
                    and not stat.S_ISLNK(st.st_mode)
                    and st.st_uid == uid
                    and os.access(str(cache_root), os.W_OK | os.X_OK)
                ):
                    return str(cache_root / app_slug / "worktrees")
            elif cache_root.parent.exists():
                pst = os.lstat(str(cache_root.parent))
                if (
                    stat.S_ISDIR(pst.st_mode)
                    and not stat.S_ISLNK(pst.st_mode)
                    and pst.st_uid == uid
                    and os.access(str(cache_root.parent), os.W_OK | os.X_OK)
                ):
                    return str(cache_root / app_slug / "worktrees")
        except OSError:
            continue
    return os.path.join(DEFAULT_TMP, f"{app_slug}-{uid}", "worktrees")


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` with ``0o700`` permissions and verify ownership on POSIX.

    Refuses symlinks or directories owned by a different user on POSIX so a
    local attacker on a shared ``/tmp`` cannot plant metadata or intercept logs.
    """
    if not IS_WINDOWS and path.exists():
        st = os.lstat(str(path))
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f"Refusing symlink directory: {path}")
        if not stat.S_ISDIR(st.st_mode):
            raise PermissionError(f"Expected directory, found non-directory: {path}")
        if st.st_uid != os.getuid():
            raise PermissionError(
                f"Refusing directory {path} owned by uid {st.st_uid} (expected {os.getuid()})"
            )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not IS_WINDOWS:
        st = os.lstat(str(path))
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise PermissionError(f"Refusing non-directory/symlink: {path}")
        if st.st_uid != os.getuid():
            raise PermissionError(
                f"Refusing directory {path} owned by uid {st.st_uid} (expected {os.getuid()})"
            )
        if st.st_mode & 0o077:
            with contextlib.suppress(OSError):
                os.chmod(str(path), 0o700)
    return path


def prepare_runtime_dir(
    path: Path,
    *,
    is_default: bool,
    default_base: Optional[Path] = None,
) -> Path:
    """Prepare a task/worktree directory.

    Default app-owned directories are hardened with ``ensure_private_dir`` (on
    both the app base directory and target child). Explicit user overrides from
    environment variables get a plain ``mkdir(parents=True, exist_ok=True)``
    without ownership checks, symlink refusal, or chmodding user directories.
    """
    if not is_default:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path
    if default_base is not None:
        ensure_private_dir(default_base)
    return ensure_private_dir(path)


def prepare_log_parent(
    log_file: Union[str, Path],
    *,
    is_default: bool,
    default_base: Optional[Path] = None,
) -> Path:
    """Prepare the parent directory for ``log_file``.

    When ``is_default`` is True, hardens the app-owned base directory via
    ``ensure_private_dir``. When the user explicitly configured ``log_file``
    (e.g. ``/tmp/foo.log`` or ``$HOME/foo.log``), only ensures the parent
    exists without refusing root-owned shared dirs or chmodding ``$HOME``/cwd.
    """
    parent = Path(log_file).parent
    if not is_default:
        parent.mkdir(parents=True, exist_ok=True)
        return parent
    if default_base is not None:
        ensure_private_dir(default_base)
    return ensure_private_dir(parent)


def _is_abs_path_entry(entry: str, *, win: bool) -> bool:
    if os.path.isabs(entry):
        return True
    if win and len(entry) >= 3 and entry[0].isalpha() and entry[1] == ":" and entry[2] in ("\\", "/"):
        return True
    return False


def safe_which(
    exe: str,
    *,
    is_windows: Optional[bool] = None,
) -> Optional[str]:
    """Locate ``exe`` on ``PATH`` while avoiding Windows cwd binary hijacking.

    On Windows, ``shutil.which`` and ``CreateProcess`` prepend the current
    working directory even when ``path=`` is supplied, allowing an untrusted
    repository containing ``git.exe``/``node.exe``/``codex.exe``/``agy.exe`` to
    hijack execution. On Windows this helper scans absolute ``PATH`` entries ×
    ``PATHEXT`` directly with ``os.path.isfile``, skipping relative entries and
    any directory resolving to ``os.getcwd()``.
    """
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if not win:
        found = shutil.which(exe)
        return os.path.abspath(found) if found else None

    raw_pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    pathext_list = []
    for ext in raw_pathext.split(";"):
        ext = ext.strip()
        if ext and ext.startswith("."):
            pathext_list.append(ext)
            lower_ext = ext.lower()
            if lower_ext != ext:
                pathext_list.append(lower_ext)
    if not pathext_list:
        pathext_list = [".exe", ".EXE", ".cmd", ".CMD", ".bat", ".BAT", ".com", ".COM"]
    pathext_lower = {e.lower() for e in pathext_list}

    _, exe_ext = os.path.splitext(exe)
    if exe_ext and exe_ext.lower() in pathext_lower:
        candidates = [exe]
    elif exe_ext:
        candidates = [exe] + [exe + ext for ext in pathext_list]
    else:
        candidates = [exe + ext for ext in pathext_list]

    # Explicit path (contains separator) -- check directly without PATH lookup
    if "/" in exe or "\\" in exe:
        base_dir = os.path.dirname(exe) or "."
        base_name = os.path.basename(exe)
        _, b_ext = os.path.splitext(base_name)
        cands = [base_name] if (b_ext and b_ext.lower() in pathext_lower) else [base_name] + [
            base_name + ext for ext in pathext_list
        ]
        for cname in cands:
            full = os.path.join(base_dir, cname)
            if os.path.isfile(full):
                return os.path.abspath(full)
        return None

    raw_path = os.environ.get("PATH", "")
    sep = ";" if (not IS_WINDOWS and ";" in raw_path) else os.pathsep
    try:
        cwd_norm = os.path.normcase(os.path.realpath(os.getcwd()))
    except OSError:
        cwd_norm = ""

    for entry in raw_path.split(sep):
        entry = entry.strip().strip('"')
        if not entry or not _is_abs_path_entry(entry, win=True):
            continue
        try:
            norm = os.path.normcase(os.path.realpath(entry))
        except OSError:
            continue
        if cwd_norm and norm == cwd_norm:
            continue
        for cname in candidates:
            full = os.path.join(entry, cname)
            if os.path.isfile(full):
                return os.path.abspath(full)
    return None


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.GetProcessTimes.restype = wintypes.BOOL
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetSystemDirectoryW.restype = wintypes.UINT
    _kernel32.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5

    # No console window per child and its own process group so a console
    # Ctrl+C aimed at the server does not reach detached tasks.
    POPEN_FLAGS: int = (
        subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    )
else:
    POPEN_FLAGS = 0


def win_open_process(pid: int) -> Tuple[Optional[int], bool]:
    """Return ``(handle, access_denied)``. ``handle`` is None if the PID is gone."""
    if not IS_WINDOWS:
        raise RuntimeError("win_open_process is only supported on Windows")
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        return handle, False
    return None, ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def win_is_running(pid: int) -> bool:
    if not IS_WINDOWS:
        raise RuntimeError("win_is_running is only supported on Windows")
    handle, denied = win_open_process(pid)
    if handle is None:
        return denied  # exists but not ours to inspect -- same as POSIX EPERM
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def win_start_time(pid: int) -> Optional[float]:
    if not IS_WINDOWS:
        raise RuntimeError("win_start_time is only supported on Windows")
    handle, _ = win_open_process(pid)
    if handle is None:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not _kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        created = times[0]
        return ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 1e7
    finally:
        _kernel32.CloseHandle(handle)


def win_system32(exe: str) -> str:
    """Absolute path to a System32 binary (avoids cwd hijacking via CreateProcess)."""
    if not IS_WINDOWS:
        raise RuntimeError("win_system32 is only supported on Windows")
    buf = ctypes.create_unicode_buffer(260)
    if not _kernel32.GetSystemDirectoryW(buf, len(buf)):
        raise OSError(ctypes.get_last_error(), "GetSystemDirectoryW failed")
    return os.path.join(buf.value, exe)


def win_kill_tree(pid: int) -> bool:
    """Kill a process and every descendant on Windows. Returns False if any survived."""
    if not IS_WINDOWS:
        raise RuntimeError("win_kill_tree is only supported on Windows")
    try:
        result = subprocess.run(
            [win_system32("taskkill.exe"), "/T", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        logging.warning("taskkill failed for PID %d: %s", pid, exc)
        return False
    if result.returncode != 0:
        logging.warning("taskkill exited %d for PID %d", result.returncode, pid)
    return result.returncode == 0 or not win_is_running(pid)


def popen_kw(
    detach: bool = False,
    new_group: bool = False,
    *,
    is_windows: Optional[bool] = None,
) -> Dict[str, Any]:
    """Platform-specific Popen kwargs for subprocess children."""
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if win:
        return {"creationflags": POPEN_FLAGS}
    return {"start_new_session": True} if (detach or new_group) else {}


def posix_signal_tree(pid: int, sig: int, own_group: bool = False) -> bool:
    """Send ``sig`` to ``pid``'s process group when it leads its own group, else ``pid``.

    Returns True when the signal was delivered or the target was already gone.
    Guards against signalling PID <= 1 or the server's own process group.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return False
    my_pgrp = os.getpgrp()
    use_pgid: Optional[int] = None
    try:
        pgid = os.getpgid(pid)
        if pgid == pid and pgid != my_pgrp:
            use_pgid = pgid
    except ProcessLookupError:
        if own_group and pid != my_pgrp:
            use_pgid = pid
        else:
            return True
    except Exception:
        pass

    if use_pgid is not None:
        try:
            os.killpg(use_pgid, sig)
            return True
        except ProcessLookupError:
            return True
        except Exception as exc:
            logging.warning(
                "Failed to signal process group %d with %s: %s", use_pgid, sig, exc
            )

    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except Exception as exc:
        logging.warning("Failed to signal PID %d with %s: %s", pid, sig, exc)
        return False


def posix_group_alive(pgid: int) -> bool:
    """Return True if any process remains in POSIX process group ``pgid``."""
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def stop_proc(
    proc: subprocess.Popen,
    own_group: Optional[bool] = None,
    *,
    is_windows: Optional[bool] = None,
    win_kill_tree_fn: Optional[Callable[[int], bool]] = None,
) -> None:
    """Terminate a child process tree, escalating to kill after 5s, and reap it."""
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if win:
        kill_fn = win_kill_tree_fn or win_kill_tree
        kill_fn(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return

    my_pgrp = os.getpgrp()
    if own_group is None:
        in_own_group = False
        try:
            in_own_group = os.getpgid(proc.pid) == proc.pid and proc.pid != my_pgrp
        except Exception:
            in_own_group = False
    else:
        in_own_group = bool(own_group) and proc.pid > 1 and proc.pid != my_pgrp

    if in_own_group:
        posix_signal_tree(proc.pid, signal.SIGTERM, own_group=True)
    else:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if in_own_group:
            posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)
        else:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        proc.wait()

    # If the group leader exited on SIGTERM while grandchildren survived in
    # its process group, sweep the remaining group members with SIGKILL.
    if in_own_group and posix_group_alive(proc.pid):
        time.sleep(0.05)
        if posix_group_alive(proc.pid):
            posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)


def get_pid_start_time(
    pid: int,
    *,
    is_windows: Optional[bool] = None,
) -> Optional[float]:
    """Get process start time from /proc (Linux) or Win32. Returns None if unavailable."""
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if win:
        return win_start_time(pid)
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
        # Field 22 (0-indexed: 21) is starttime in clock ticks
        fields = stat_text.rsplit(")", 1)[-1].split()
        return float(fields[19])  # index 19 after the closing paren
    except Exception:
        return None


def is_alive(
    pid: Optional[int],
    expected_start_time: Optional[float] = None,
    *,
    is_windows: Optional[bool] = None,
    get_start_time_fn: Optional[Callable[[int], Optional[float]]] = None,
) -> bool:
    """Return True if ``pid`` is alive (not zombie) and matches ``expected_start_time``."""
    if pid is None:
        return False
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if win:
        # os.kill(pid, 0) is not a probe on Windows: it terminates processes.
        if not win_is_running(pid):
            return False
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists under another UID; still run start-time guard below.
            pass
        except Exception:
            return False

        # Check for zombie via /proc (Linux)
        try:
            stat_path = Path(f"/proc/{pid}/status")
            if stat_path.exists():
                for line in stat_path.read_text().splitlines():
                    if line.startswith("State:"):
                        if "Z" in line:
                            return False
        except Exception:
            pass

    if expected_start_time is not None:
        start_fn = get_start_time_fn or (lambda p: get_pid_start_time(p, is_windows=win))
        actual = start_fn(pid)
        if actual is not None and actual != expected_start_time:
            return False

    return True


@contextlib.contextmanager
def prompt_stdin(
    payload: Optional[Union[str, bytes]],
    stdin_path: Optional[Path] = None,
    *,
    task_dir: Optional[Path] = None,
    is_windows: Optional[bool] = None,
):
    """Yield a readable binary file handle containing ``payload`` (UTF-8), or DEVNULL."""
    if not payload:
        yield subprocess.DEVNULL
        return

    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    win = IS_WINDOWS if is_windows is None else bool(is_windows)
    if not win:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            tmp.write(raw)
            tmp.flush()
            tmp.seek(0)
            yield tmp
        return

    # Windows: write a regular .stdin file and hand the child a normal read-only
    # handle (no O_TEMPORARY).
    cleanup_on_exit = stdin_path is None
    if stdin_path is None:
        base_dir = task_dir if task_dir is not None else Path(DEFAULT_TMP)
        base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(
            prefix="sync_", suffix=".stdin", dir=str(base_dir)
        )
        path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as wf:
                fd = -1
                wf.write(raw)
        except Exception:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            raise
    else:
        path = stdin_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(fd, "wb") as wf:
                fd = -1
                wf.write(raw)
        finally:
            if fd >= 0:
                os.close(fd)

    try:
        with open(path, "rb") as fh:
            yield fh
    finally:
        if cleanup_on_exit:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
