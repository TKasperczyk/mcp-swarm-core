"""Backend interface and runtime hooks for MCP swarm servers."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import diagnostics, platform, worktree


class RuntimeHooks:
    """Configurable runtime paths, timers, and platform hooks.

    Most backends only pass ``task_dir``, ``worktree_base_dir``,
    ``task_max_age``, ``wait_min_timeout``, ``log_file``, and ``log_level``.
    Optional callable overrides allow host modules to expose live mutable
    globals or monkeypatch points cleanly.
    """

    def __init__(
        self,
        *,
        task_dir: Union[Path, Callable[[], Path]],
        worktree_base_dir: Union[Path, Callable[[], Path]],
        branch_prefix: Optional[str] = None,
        task_max_age: Union[int, Callable[[], int]] = 86400,
        wait_min_timeout: Union[int, Callable[[], int]] = 150,
        client_auto_background_s: int = 120,
        poll_interval: Union[float, Callable[[], float]] = 2.0,
        progress_interval: Union[float, Callable[[], float]] = 20.0,
        log_file: Union[str, Callable[[], str]] = "",
        log_level: Union[str, Callable[[], str]] = "WARNING",
        is_windows_fn: Optional[Callable[[], bool]] = None,
        popen_kw_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        get_pid_start_time_fn: Optional[Callable[[int], Optional[float]]] = None,
        is_alive_fn: Optional[
            Callable[[Optional[int], Optional[float]], bool]
        ] = None,
        win_kill_tree_fn: Optional[Callable[[int], bool]] = None,
        stop_proc_fn: Optional[Callable[..., None]] = None,
        write_meta_atomic_fn: Optional[
            Callable[[Path, Dict[str, Any]], None]
        ] = None,
        safe_read_fn: Optional[Callable[[Path], str]] = None,
        create_worktree_fn: Optional[
            Callable[[str, Optional[str]], Tuple[str, str, str]]
        ] = None,
        remove_worktree_fn: Optional[Callable[[str, str], bool]] = None,
        send_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        start_async_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        get_last_cleanup_fn: Optional[Callable[[], float]] = None,
        set_last_cleanup_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._task_dir = task_dir
        self._worktree_base_dir = worktree_base_dir
        self.branch_prefix = branch_prefix
        self._task_max_age = task_max_age
        self._wait_min_timeout = wait_min_timeout
        self.client_auto_background_s = client_auto_background_s
        self._poll_interval = poll_interval
        self._progress_interval = progress_interval
        self._log_file = log_file
        self._log_level = log_level

        self._is_windows_fn = is_windows_fn
        self._popen_kw_fn = popen_kw_fn
        self._get_pid_start_time_fn = get_pid_start_time_fn
        self._is_alive_fn = is_alive_fn
        self._win_kill_tree_fn = win_kill_tree_fn
        self._stop_proc_fn = stop_proc_fn
        self._write_meta_atomic_fn = write_meta_atomic_fn
        self._safe_read_fn = safe_read_fn
        self._create_worktree_fn = create_worktree_fn
        self._remove_worktree_fn = remove_worktree_fn
        self.send_fn = send_fn
        self.start_async_fn = start_async_fn

        self._last_cleanup_local = 0.0
        self._get_last_cleanup_fn = get_last_cleanup_fn
        self._set_last_cleanup_fn = set_last_cleanup_fn

    def task_dir(self) -> Path:
        return self._task_dir() if callable(self._task_dir) else self._task_dir

    def worktree_base_dir(self) -> Path:
        return (
            self._worktree_base_dir()
            if callable(self._worktree_base_dir)
            else self._worktree_base_dir
        )

    def task_max_age(self) -> int:
        return (
            self._task_max_age()
            if callable(self._task_max_age)
            else self._task_max_age
        )

    def wait_min_timeout(self) -> int:
        return (
            self._wait_min_timeout()
            if callable(self._wait_min_timeout)
            else self._wait_min_timeout
        )

    def poll_interval(self) -> float:
        return (
            self._poll_interval()
            if callable(self._poll_interval)
            else self._poll_interval
        )

    def progress_interval(self) -> float:
        return (
            self._progress_interval()
            if callable(self._progress_interval)
            else self._progress_interval
        )

    def log_file(self) -> str:
        return self._log_file() if callable(self._log_file) else self._log_file

    def log_level(self) -> str:
        return self._log_level() if callable(self._log_level) else self._log_level

    def is_windows(self) -> bool:
        if self._is_windows_fn is not None:
            return bool(self._is_windows_fn())
        return platform.IS_WINDOWS

    def popen_kw(self, detach: bool = False, new_group: bool = False) -> Dict[str, Any]:
        if self._popen_kw_fn is not None:
            return self._popen_kw_fn(detach=detach, new_group=new_group)
        return platform.popen_kw(
            detach=detach, new_group=new_group, is_windows=self.is_windows()
        )

    def get_pid_start_time(self, pid: int) -> Optional[float]:
        if self._get_pid_start_time_fn is not None:
            return self._get_pid_start_time_fn(pid)
        return platform.get_pid_start_time(pid, is_windows=self.is_windows())

    def is_alive(
        self, pid: Optional[int], expected_start_time: Optional[float] = None
    ) -> bool:
        if self._is_alive_fn is not None:
            return self._is_alive_fn(pid, expected_start_time=expected_start_time)
        return platform.is_alive(
            pid,
            expected_start_time=expected_start_time,
            is_windows=self.is_windows(),
            get_start_time_fn=self.get_pid_start_time,
        )

    def win_kill_tree(self, pid: int) -> bool:
        if self._win_kill_tree_fn is not None:
            return self._win_kill_tree_fn(pid)
        return platform.win_kill_tree(pid)

    def stop_proc(self, proc: Any, own_group: Optional[bool] = None) -> None:
        if self._stop_proc_fn is not None:
            self._stop_proc_fn(proc, own_group=own_group)
            return
        platform.stop_proc(
            proc,
            own_group=own_group,
            is_windows=self.is_windows(),
            win_kill_tree_fn=self.win_kill_tree,
        )

    def safe_read(self, path: Path) -> str:
        if self._safe_read_fn is not None:
            return self._safe_read_fn(path)
        return diagnostics.safe_read(path)

    def create_worktree(
        self,
        run_id: str,
        base_cwd: Optional[str],
        *,
        branch_prefix: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        if self._create_worktree_fn is not None:
            return self._create_worktree_fn(run_id, base_cwd)
        prefix = branch_prefix or self.branch_prefix or "mcp-swarm"
        return worktree.create_worktree(
            run_id,
            base_cwd,
            worktree_base_dir=self.worktree_base_dir(),
            branch_prefix=prefix,
            popen_kw_fn=self.popen_kw,
        )

    def remove_worktree(self, worktree_path: str, branch_name: str) -> bool:
        if self._remove_worktree_fn is not None:
            return self._remove_worktree_fn(worktree_path, branch_name)
        return worktree.remove_worktree(
            worktree_path,
            branch_name,
            worktree_base_dir=self.worktree_base_dir(),
            popen_kw_fn=self.popen_kw,
        )

    def get_last_cleanup(self) -> float:
        if self._get_last_cleanup_fn is not None:
            return self._get_last_cleanup_fn()
        return self._last_cleanup_local

    def set_last_cleanup(self, value: float) -> None:
        if self._set_last_cleanup_fn is not None:
            self._set_last_cleanup_fn(value)
        else:
            self._last_cleanup_local = value


class SwarmBackend:
    """Abstract adapter implemented by each CLI backend (Codex, Antigravity, ...)."""

    server_name: str = "mcp-swarm"
    version: str = "0.1.0"
    product_name: str = "CLI"
    tool_prefix: str = "cli"
    uri_scheme: str = "mcp-swarm"
    branch_prefix: str = "mcp-swarm"

    # Parameter & output field names for session continuity
    session_param_name: str = "threadId"
    session_field_name: str = "thread_id"
    session_label: str = "Thread ID"
    prefer_observed_session_id: bool = False
    wants_stderr_on_success: bool = False

    process_check_cmd: str = "`pgrep -af 'cli'`"
    server_instructions: str = ""
    tools: List[Dict[str, Any]] = []
    resources: List[Dict[str, Any]] = []
    reply_extra_param_keys: Tuple[str, ...] = ()

    @property
    def sync_tool_name(self) -> str:
        return self.tool_prefix

    @property
    def async_tool_name(self) -> str:
        return f"{self.tool_prefix}_async"

    @property
    def reply_tool_name(self) -> str:
        return f"{self.tool_prefix}_reply"

    @property
    def status_tool_name(self) -> str:
        return f"{self.tool_prefix}_status"

    @property
    def wait_tool_name(self) -> str:
        return f"{self.tool_prefix}_wait"

    @property
    def cancel_tool_name(self) -> str:
        return f"{self.tool_prefix}_cancel"

    def hints(self) -> Dict[str, str]:
        return diagnostics.make_inferred_hints(
            product_name=self.product_name,
            wait_tool_name=self.wait_tool_name,
            process_check_cmd=self.process_check_cmd,
        )

    def validate_call_params(
        self, args: Dict[str, Any], *, is_reply: bool = False
    ) -> Optional[str]:
        return None

    def build_command(self, params: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        raise NotImplementedError

    def prepare_command_for_json(self, cmd: List[str]) -> List[str]:
        return cmd

    def format_stdin_payload(
        self, params: Dict[str, Any]
    ) -> Optional[Union[str, bytes]]:
        prompt = params.get("prompt", "")
        return prompt if prompt else None

    def exec_argv(self, cmd: List[str]) -> List[str]:
        return cmd

    def resolve_model(self, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return None

    def model_from_command(self, command: Any) -> Optional[str]:
        return None

    def parse_jsonl_file(
        self, stdout_path: Path
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    def parse_jsonl_text(
        self, jsonl_text: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    def invalidate_jsonl_cache(self, stdout_path: Path) -> None:
        pass

    def extract_session_id_fast(
        self, stdout_path: Path, meta: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if (
            meta
            and isinstance(meta.get(self.session_field_name), str)
            and meta[self.session_field_name]
        ):
            return meta[self.session_field_name]
        return None

    def extract_result(
        self,
        stdout: str,
        stderr: str,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def decide_run_status(
        self,
        exit_code: Optional[int],
        analysis: Dict[str, Any],
        stderr: str,
    ) -> Tuple[str, bool]:
        return diagnostics.decide_run_status(exit_code, analysis, stderr)

    def validate_observed_session(
        self,
        requested_id: str,
        observed_id: Optional[str],
        analysis: Dict[str, Any],
        stderr: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a structured failure dict if a session resume silently fell back."""
        return None

    def resolve_reply_cwd(
        self, session_id: str, extra_params: Dict[str, Any]
    ) -> Optional[str]:
        """Optional hook to recover the original cwd/worktree for a resumed session."""
        return None

    def record_session_cwd(
        self, session_id: Optional[str], cwd: Optional[str]
    ) -> None:
        """Optional hook called when a session ID and working directory are observed."""
        pass

    def classify_failure(self, message: str) -> Dict[str, Any]:
        return diagnostics.classify_failure(message, product_name=self.product_name)

    def build_failure_info(
        self,
        stdout: str,
        stderr: str,
        command: Any = None,
        analysis: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if analysis is None:
            analysis, _ = self.parse_jsonl_text(stdout)
        return diagnostics.build_failure_info(
            stdout,
            stderr,
            analysis=analysis,
            product_name=self.product_name,
            command=command,
            model=model,
            model_from_command_fn=self.model_from_command,
            classify_fn=self.classify_failure,
        )

    def format_failure_info(
        self, info: Dict[str, Any], include_diagnostics: bool = True
    ) -> str:
        return diagnostics.format_failure_info(
            info, include_diagnostics=include_diagnostics, product_name=self.product_name
        )

    def get_server_info_extra(self) -> Dict[str, Any]:
        return {}

    def get_config_resource(self) -> Dict[str, Any]:
        return {}

    def extra_meta_on_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def extra_check_task_fields(
        self,
        analysis: Dict[str, Any],
        meta: Dict[str, Any],
        status: str,
        stderr: str = "",
    ) -> Dict[str, Any]:
        return {}

    def extra_sync_outcome_fields(
        self,
        analysis: Dict[str, Any],
        outcome: Dict[str, Any],
        stderr: str = "",
    ) -> Dict[str, Any]:
        return {}

    def format_extra_completion_details(
        self, info: Dict[str, Any], *, is_reply: bool = False
    ) -> List[str]:
        return []

    def format_extra_status_lines(
        self,
        analysis: Dict[str, Any],
        jsonl_status: Dict[str, Any],
        meta: Dict[str, Any],
        status: str,
    ) -> List[str]:
        return []

    def extra_structured_status_fields(
        self,
        analysis: Dict[str, Any],
        jsonl_status: Dict[str, Any],
        meta: Dict[str, Any],
        status: str,
    ) -> Dict[str, Any]:
        return {}

    def extra_structured_wait_fields(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def extra_structured_sync_fields(
        self, outcome: Dict[str, Any], *, is_reply: bool = False
    ) -> Dict[str, Any]:
        return {}

    def extra_resource_task_fields(
        self,
        resolved_meta: Dict[str, Any],
        analysis: Optional[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {}


def parse_env_int(env_name: str, default: int) -> int:
    """Safely parse an integer environment variable, falling back to ``default``."""
    try:
        return int(os.environ.get(env_name, str(default)))
    except (ValueError, TypeError):
        return default
