#!/usr/bin/env python3
"""Unit and integration tests for mcp-swarm-core.

Runnable via ``python3 tests/test_core.py`` or ``pytest tests/test_core.py``.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_swarm_core as core  # noqa: E402


class DummyBackend(core.SwarmBackend):
    """Minimal concrete backend exercising the full mcp-swarm-core stack."""

    server_name = "dummy-mcp-swarm"
    version = "0.1.0"
    product_name = "Dummy"
    tool_prefix = "dummy"
    uri_scheme = "dummy-swarm"
    branch_prefix = "dummy-swarm"
    session_param_name = "session_id"
    session_field_name = "session_id"
    session_label = "Session ID"
    prefer_observed_session_id = True
    process_check_cmd = "`pgrep -af 'dummy'`"
    server_instructions = "Dummy swarm server for testing."
    reply_extra_param_keys = ("cwd", "model")

    def __init__(self, script_path: Path, open_fn=None) -> None:
        self.script_path = script_path
        self._open_fn = open_fn
        self.cache = core.IncrementalJsonlCache(
            core.JsonlParserAdapter(
                new_state=self._new_state,
                clone_state=self._clone_state,
                consume_lines=self._consume_lines,
                finalize_views=self._finalize_views,
            )
        )
        self.tools = [
            {
                "name": "dummy",
                "description": "Run dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}},
                    "required": ["prompt"],
                },
                "outputSchema": core.make_sync_output_schema("session_id"),
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            },
            {
                "name": "dummy_async",
                "description": "Async dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                "outputSchema": core.make_async_output_schema(),
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            },
            {
                "name": "dummy_reply",
                "description": "Reply dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["session_id", "prompt"],
                },
                "outputSchema": core.make_reply_output_schema("session_id"),
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            },
            {
                "name": "dummy_status",
                "description": "Status dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {"task_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["task_ids"],
                },
                "outputSchema": core.make_status_output_schema("session_id"),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "dummy_wait",
                "description": "Wait dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {"task_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["task_ids"],
                },
                "outputSchema": core.make_wait_output_schema("session_id"),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "dummy_cancel",
                "description": "Cancel dummy",
                "inputSchema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
                "outputSchema": core.make_cancel_output_schema(),
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": True,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
        ]
        self.resources = [
            {
                "uri": "dummy-swarm:///server-info",
                "name": "Server Info",
                "description": "Info",
                "mimeType": "application/json",
            },
            {
                "uri": "dummy-swarm:///config",
                "name": "Config",
                "description": "Config",
                "mimeType": "application/json",
            },
            {
                "uri": "dummy-swarm:///tasks",
                "name": "Tasks",
                "description": "Tasks",
                "mimeType": "application/json",
            },
        ]

    @staticmethod
    def _new_state() -> Dict[str, Any]:
        return {
            "session_id": None,
            "model": None,
            "last_assistant_text": None,
            "errors": [],
            "warnings": [],
            "terminal_events": [],
            "terminal_error": None,
            "saw_json": False,
            "saw_turn_started": False,
            "phase": "starting",
            "tools_called": 0,
            "last_tool": None,
            "last_tool_args": None,
            "last_reasoning": None,
            "last_assistant_preview": None,
        }

    @staticmethod
    def _clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
        c = dict(state)
        c["errors"] = list(state["errors"])
        c["warnings"] = list(state["warnings"])
        c["terminal_events"] = list(state["terminal_events"])
        return c

    @staticmethod
    def _consume_lines(state: Dict[str, Any], lines: List[str]) -> None:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            state["saw_json"] = True
            etype = ev.get("event", "")
            if etype == "init":
                state["session_id"] = ev.get("session_id")
                state["model"] = ev.get("model")
                state["phase"] = "running"
                state["saw_turn_started"] = True
            elif etype == "tool":
                state["tools_called"] += 1
                state["last_tool"] = str(ev.get("name", "tool"))
                state["last_tool_args"] = str(ev.get("args", ""))[:150]
            elif etype == "done":
                state["terminal_events"].append("completed")
                txt = ev.get("text", "")
                if isinstance(txt, str) and txt:
                    state["last_assistant_text"] = txt
                    state["last_assistant_preview"] = txt[-300:]
            elif etype == "fail":
                state["terminal_events"].append("failed")
                state["terminal_error"] = core.error_message(ev.get("error"))

    @staticmethod
    def _finalize_views(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t_events = list(state["terminal_events"])
        t_types = set(t_events)
        t_status = next(iter(t_types)) if len(t_types) == 1 else None
        t_conflict = len(t_types) > 1
        analysis = {
            "session_id": state["session_id"],
            "model": state["model"],
            "last_assistant_text": state["last_assistant_text"],
            "errors": list(state["errors"]),
            "warnings": list(state["warnings"]),
            "terminal_events": t_events,
            "terminal_status": t_status,
            "terminal_error": state["terminal_error"],
            "terminal_conflict": t_conflict,
            "saw_json": state["saw_json"],
            "saw_turn_started": state["saw_turn_started"],
        }
        status = {
            "phase": t_status or ("unknown (conflicting terminal events)" if t_conflict else state["phase"]),
            "tools_called": state["tools_called"],
            "last_tool": state["last_tool"],
            "last_tool_args": state["last_tool_args"],
            "last_reasoning": state["last_reasoning"],
            "last_assistant_text": state["last_assistant_preview"],
            "last_error": state["errors"][-1] if state["errors"] else None,
            "last_warning": state["warnings"][-1] if state["warnings"] else None,
            "terminal_error": state["terminal_error"],
            "terminal_conflict": t_conflict,
        }
        return analysis, status

    def build_command(self, params: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        cmd = ["dummy"]
        if params.get("model"):
            cmd.extend(["--model", str(params["model"])])
        if params.get("session_id"):
            cmd.extend(["--session", str(params["session_id"])])
        cwd = params.get("cwd")
        return cmd, os.path.abspath(cwd) if cwd else None

    def exec_argv(self, cmd: List[str]) -> List[str]:
        return [sys.executable, str(self.script_path)] + cmd[1:]

    def resolve_model(self, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        m = (params or {}).get("model")
        return m.strip() if isinstance(m, str) and m.strip() else None

    def parse_jsonl_file(self, stdout_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.cache.parse_file(stdout_path, open_fn=self._open_fn)

    def parse_jsonl_text(self, jsonl_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.cache.parse_text(jsonl_text)

    def invalidate_jsonl_cache(self, stdout_path: Path) -> None:
        self.cache.invalidate(stdout_path)

    def extract_result(
        self, stdout: str, stderr: str, analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[str]]:
        if analysis is None:
            analysis, _ = self.parse_jsonl_text(stdout)
        if analysis.get("saw_json") and analysis.get("last_assistant_text"):
            return analysis["last_assistant_text"], analysis.get("session_id")
        return (stdout.strip() or stderr.strip() or "No output"), analysis.get("session_id")

    def get_config_resource(self) -> Dict[str, Any]:
        return {"mode": "test"}


def _make_server(tmp_dir: Path, script_body: str):
    script = tmp_dir / "fake_cli.py"
    script.write_text(script_body, encoding="utf-8")
    frames: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def capture(frame: Dict[str, Any]) -> None:
        with lock:
            frames.append(frame)

    hooks = core.RuntimeHooks(
        task_dir=tmp_dir / "tasks",
        worktree_base_dir=tmp_dir / "worktrees",
        branch_prefix="dummy-swarm",
        wait_min_timeout=1,
        progress_interval=1,
        poll_interval=0.1,
        send_fn=capture,
    )
    backend = DummyBackend(script)
    server = core.McpSwarmServer(backend, hooks)
    return server, frames


def test_sigchld_handler_lock_free_and_no_import_side_effect():
    src = inspect.getsource(core.sigchld_handler)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "logging":
            raise AssertionError("sigchld_handler must not reference logging")

    # Importing mcp_swarm_core on a worker thread must NOT fail with ValueError
    # (no import-time signal.signal side effect).
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, threading; "
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r}); "
                "err = []; "
                "t = threading.Thread(target=lambda: (__import__('mcp_swarm_core'))); "
                "t.start(); t.join(); "
                "import mcp_swarm_core as c; assert c.install_sigchld_handler() in (True, False)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"worker-thread import failed: {proc.stderr}"


def test_task_id_validation_private_dir_and_win_guard():
    assert core.validate_task_id("deadbeef") is True
    assert core.validate_task_id("a" * 32) is True
    # Trailing newline must be rejected (fullmatch, not match+$)
    assert core.validate_task_id("deadbeef\n") is False
    assert core.validate_task_id("../deadbeef") is False

    # Token count 550 must NOT misclassify as HTTP 5xx
    c = core.classify_failure("Total tokens: 550; unknown error")
    assert c["category"] == "unknown", f"false 5xx match: {c}"
    c503 = core.classify_failure("HTTP 503 Service Unavailable")
    assert c503["category"] == "provider_5xx"

    if not core.IS_WINDOWS:
        # win_* helpers raise RuntimeError (not NameError) on POSIX
        try:
            core.win_kill_tree(12345)
            raise AssertionError("expected RuntimeError on POSIX")
        except RuntimeError:
            pass

        # ensure_private_dir refuses symlinks
        with tempfile.TemporaryDirectory() as td:
            real_dir = Path(td) / "real"
            real_dir.mkdir()
            link = Path(td) / "sym"
            link.symlink_to(real_dir)
            try:
                core.ensure_private_dir(link)
                raise AssertionError("expected PermissionError for symlink dir")
            except PermissionError:
                pass
            priv = core.ensure_private_dir(Path(td) / "private")
            assert priv.is_dir() and (os.lstat(str(priv)).st_mode & 0o077) == 0


def test_request_id_validation_and_cancellation_tracker():
    assert core.is_valid_request_id("abc") is True
    assert core.is_valid_request_id(42) is True
    assert core.is_valid_request_id(1.5) is True
    assert core.is_valid_request_id(True) is False
    assert core.is_valid_request_id(False) is False
    assert core.is_valid_request_id([1]) is False
    assert core.is_valid_request_id({"a": 1}) is False
    assert core.is_valid_request_id(float("nan")) is False
    assert core.is_valid_request_id(float("inf")) is False

    tracker = core.CancellationTracker(max_entries=3)
    for i in range(5):
        tracker.mark_cancelled(i)
    assert len(tracker.cancelled_requests) <= 3
    assert tracker.is_cancelled(4) is True
    assert tracker.consume_if_cancelled(4) is True
    assert tracker.is_cancelled(4) is False


def test_incremental_jsonl_cache_unicode_and_growth():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.stdout"
        backend = DummyBackend(Path(td) / "unused.py")
        msg = "line1\u2028line2\u2029line3\x85end"
        lines = [json.dumps({"event": "init", "session_id": "s-1", "model": "m-1"}, ensure_ascii=False)]
        for i in range(150):
            lines.append(
                json.dumps({"event": "tool", "name": "sh", "args": f"cmd-{i} " + ("x" * 60)}, ensure_ascii=False)
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert p.stat().st_size > backend.cache.min_bytes

        a1, s1 = backend.parse_jsonl_file(p)
        assert a1["session_id"] == "s-1"
        assert s1["tools_called"] == 150

        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "done", "text": msg}, ensure_ascii=False) + "\n")
        a2, s2 = backend.parse_jsonl_file(p)
        assert a2["terminal_status"] == "completed"
        assert a2["last_assistant_text"] == msg
        assert s2["phase"] == "completed"


def test_protocol_discover_initialize_tasks_cancel_and_timeout():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        server, frames = _make_server(
            tmp,
            """
import json, sys
prompt = sys.stdin.read()
print(json.dumps({"event": "init", "session_id": "sess-99", "model": "dummy-v1"}), flush=True)
print(json.dumps({"event": "tool", "name": "read", "args": "foo.py"}), flush=True)
print(json.dumps({"event": "done", "text": "REPLY:" + prompt}), flush=True)
""",
        )

        # initialize negotiation
        server.handle({"jsonrpc": "2.0", "id": 100, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
        assert frames.pop()["result"]["protocolVersion"] == "2024-11-05"
        server.handle({"jsonrpc": "2.0", "id": 101, "method": "initialize", "params": {"protocolVersion": "2099-01-01"}})
        assert frames.pop()["result"]["protocolVersion"] == core.DEFAULT_PROTOCOL_VERSION

        # server/discover
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {}})
        disc = frames.pop()["result"]
        assert disc["resultType"] == "complete"
        assert disc["serverInfo"]["name"] == "dummy-mcp-swarm"
        assert core.TASKS_EXTENSION_ID in disc["capabilities"]["extensions"]

        # unsupported version -> -32022
        server.handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2099-01-01"}},
        })
        assert frames.pop()["error"]["code"] == -32022

        # unknown notification -> zero frames
        server.handle({"jsonrpc": "2.0", "method": "notifications/random"})
        assert frames == []

        # sync tool call
        server.handle({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "dummy", "arguments": {"prompt": "hello core"}},
        })
        res = frames.pop()["result"]
        assert res["isError"] is False
        assert "REPLY:hello core" in res["content"][0]["text"]
        assert res["structuredContent"]["session_id"] == "sess-99"
        assert res["structuredContent"]["model"] == "dummy-v1"

        # Pre-cancelled async request must NOT launch any task
        server.cancellations.mark_cancelled(999)
        before_metas = set((tmp / "tasks").glob("*.meta"))
        server.handle({
            "jsonrpc": "2.0",
            "id": 999,
            "method": "tools/call",
            "params": {"name": "dummy_async", "arguments": {"prompt": "cancelled"}},
        })
        assert set((tmp / "tasks").glob("*.meta")) == before_metas
        assert frames == []

        # MCP Tasks extension lifecycle
        tasks_meta = {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {
                    "extensions": {core.TASKS_EXTENSION_ID: {}},
                },
            }
        }
        server.handle({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "dummy", "arguments": {"prompt": "task hello"}, **tasks_meta},
        })
        create_res = frames.pop()["result"]
        assert create_res["resultType"] == "task"
        tid = create_res["taskId"]
        assert len(tid) == 32

        final = None
        for _ in range(60):
            server.handle({"jsonrpc": "2.0", "id": 5, "method": "tasks/get", "params": {"taskId": tid, **tasks_meta}})
            poll = frames.pop()["result"]
            if poll["status"] != "working":
                final = poll
                break
            time.sleep(0.05)
        assert final is not None and final["status"] == "completed"
        assert final["result"]["isError"] is False
        assert "REPLY:task hello" in final["result"]["content"][0]["text"]

        # tasks/cancel on already-completed -> -32602
        server.handle({"jsonrpc": "2.0", "id": 50, "method": "tasks/cancel", "params": {"taskId": tid, **tasks_meta}})
        assert frames.pop()["error"]["code"] == -32602

        # Resources
        for uri in ("dummy-swarm:///server-info", "dummy-swarm:///config", "dummy-swarm:///tasks"):
            server.handle({"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": uri}})
            r = frames.pop()["result"]
            parsed = json.loads(r["contents"][0]["text"])
            assert parsed is not None


def test_worktree_lifecycle_guard_and_async_cancel_tree():
    if sys.platform == "win32":
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo = tmp / "repo"
        repo.mkdir()
        git_env = dict(os.environ)
        git_env["GIT_AUTHOR_NAME"] = "Test"
        git_env["GIT_AUTHOR_EMAIL"] = "t@example.com"
        git_env["GIT_COMMITTER_NAME"] = "Test"
        git_env["GIT_COMMITTER_EMAIL"] = "t@example.com"
        subprocess.run(["git", "init", "-q", str(repo)], check=True, env=git_env)
        (repo / "f.txt").write_text("v1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True, env=git_env)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
            check=True,
            env=git_env,
        )

        wts_dir = tmp / "wts"
        wt_root, wt_cwd, branch = core.create_worktree(
            "ab12cd34",
            str(repo),
            worktree_base_dir=wts_dir,
            branch_prefix="core-test",
        )
        assert Path(wt_root).exists()
        assert branch == "core-test/ab12cd34"

        # Guard: remove_worktree must refuse removing a path outside worktree_base_dir
        assert core.remove_worktree(str(repo), "main", worktree_base_dir=wts_dir) is False
        assert repo.exists()

        assert core.remove_worktree(wt_root, branch, worktree_base_dir=wts_dir) is True
        assert not Path(wt_root).exists()

        # Process tree cancellation + wait timeout + private transcript permissions
        gc_file = tmp / "gc.pid"
        server, frames = _make_server(
            tmp,
            f"""
import subprocess, sys, time
from pathlib import Path
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path({str(gc_file)!r}).write_text(str(gc.pid))
time.sleep(60)
""",
        )
        server.handle({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "dummy_async", "arguments": {"prompt": "wait"}},
        })
        tid = frames.pop()["result"]["structuredContent"]["task_id"]

        # Verify task transcript files were created 0o600 (owner-only)
        assert (os.lstat(str(tmp / "tasks" / f"{tid}.stdout")).st_mode & 0o077) == 0
        assert (os.lstat(str(tmp / "tasks" / f"{tid}.stderr")).st_mode & 0o077) == 0

        for _ in range(50):
            if gc_file.exists():
                break
            time.sleep(0.05)
        assert gc_file.exists()
        gc_pid = int(gc_file.read_text().strip())
        assert core.is_alive(gc_pid)

        # Wait timeout does NOT kill the task
        server.handle({
            "jsonrpc": "2.0",
            "id": 105,
            "method": "tools/call",
            "params": {"name": "dummy_wait", "arguments": {"task_ids": [tid], "timeout": 1}},
        })
        w_res = frames.pop()["result"]
        assert w_res["structuredContent"]["tasks"][0]["status"] == "timeout"
        assert core.is_alive(gc_pid)

        # Cancel kills leader AND grandchild
        server.handle({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "dummy_cancel", "arguments": {"task_id": tid}},
        })
        assert frames.pop()["result"]["structuredContent"]["status"] == "cancelled"
        for _ in range(40):
            if not core.is_alive(gc_pid):
                break
            time.sleep(0.05)
        assert not core.is_alive(gc_pid)


def test_safe_which_windows_skips_cwd_and_relative_entries():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        cwd_dir = base / "repo"
        real_bin = base / "real_bin"
        rel_bin = cwd_dir / "rel_bin"
        cwd_dir.mkdir()
        real_bin.mkdir()
        rel_bin.mkdir()

        # Plant decoy binaries in cwd and a relative dir, plus clean binary in real_bin
        for d in (cwd_dir, rel_bin, real_bin):
            (d / "git.exe").write_text("bin", encoding="utf-8")
        (cwd_dir / "only_in_cwd.exe").write_text("decoy", encoding="utf-8")

        old_cwd = os.getcwd()
        old_path = os.environ.get("PATH")
        old_pathext = os.environ.get("PATHEXT")
        try:
            os.chdir(str(cwd_dir))
            os.environ["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
            os.environ["PATH"] = f".;{cwd_dir};rel_bin;{real_bin}"

            found_git = core.safe_which("git", is_windows=True)
            assert found_git == os.path.abspath(str(real_bin / "git.exe")), f"got {found_git}"
            assert core.safe_which("git.exe", is_windows=True) == os.path.abspath(
                str(real_bin / "git.exe")
            )
            # Binary existing ONLY in cwd must never be resolved
            assert core.safe_which("only_in_cwd", is_windows=True) is None
        finally:
            os.chdir(old_cwd)
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
            if old_pathext is None:
                os.environ.pop("PATHEXT", None)
            else:
                os.environ["PATHEXT"] = old_pathext


def test_prepare_log_and_runtime_dir_overrides_and_default_worktree_dir():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # 1. Explicit user log override in /tmp (owned by root) must NOT raise or chmod /tmp
        if not core.IS_WINDOWS and Path("/tmp").exists():
            tmp_mode_before = os.lstat("/tmp").st_mode
            core.prepare_log_parent("/tmp/mcp_swarm_core_test.log", is_default=False)
            assert os.lstat("/tmp").st_mode == tmp_mode_before

        # 2. Explicit user log override in a 0755 user dir must NOT chmod it to 0700
        user_dir = base / "proj"
        user_dir.mkdir(mode=0o755)
        os.chmod(str(user_dir), 0o755)
        core.prepare_log_parent(user_dir / "app.log", is_default=False)
        if not core.IS_WINDOWS:
            assert (os.lstat(str(user_dir)).st_mode & 0o777) == 0o755

        # 3. Explicit task dir override pointing at a symlink must work
        real_target = base / "real_tasks"
        real_target.mkdir()
        link_tasks = base / "link_tasks"
        if not core.IS_WINDOWS:
            os.symlink(str(real_target), str(link_tasks))
            core.prepare_runtime_dir(link_tasks, is_default=False)

        # 4. Default paths DO enforce private 0700 ownership on both base and child
        default_base = base / "app-default"
        default_tasks = default_base / "tasks"
        core.prepare_runtime_dir(default_tasks, is_default=True, default_base=default_base)
        if not core.IS_WINDOWS:
            assert (os.lstat(str(default_base)).st_mode & 0o077) == 0
            assert (os.lstat(str(default_tasks)).st_mode & 0o077) == 0

        # 5. default_worktree_dir uses XDG_CACHE_HOME, NOT XDG_RUNTIME_DIR
        cache_dir = base / "cache"
        cache_dir.mkdir()
        rt_dir = base / "runtime"
        rt_dir.mkdir()
        old_cache = os.environ.get("XDG_CACHE_HOME")
        old_rt = os.environ.get("XDG_RUNTIME_DIR")
        try:
            os.environ["XDG_CACHE_HOME"] = str(cache_dir)
            os.environ["XDG_RUNTIME_DIR"] = str(rt_dir)
            wt_default = core.default_worktree_dir("demo-swarm")
            if not core.IS_WINDOWS:
                assert wt_default == str(cache_dir / "demo-swarm" / "worktrees")
                assert str(rt_dir) not in wt_default
        finally:
            if old_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = old_cache
            if old_rt is None:
                os.environ.pop("XDG_RUNTIME_DIR", None)
            else:
                os.environ["XDG_RUNTIME_DIR"] = old_rt


def test_cleanup_defers_worktree_with_active_or_recent_reply():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        task_dir = tmp / "tasks"
        wt_dir = tmp / "wts"
        task_dir.mkdir()
        wt_dir.mkdir()

        wt_path = wt_dir / "11112222"
        wt_path.mkdir()
        removed = []

        backend = DummyBackend(sys.executable)
        hooks = core.RuntimeHooks(
            task_dir=lambda: task_dir,
            worktree_base_dir=lambda: wt_dir,
            task_max_age=lambda: 60,
            wait_min_timeout=lambda: 1,
            poll_interval=lambda: 0.1,
            progress_interval=lambda: 60,
            log_file=lambda: str(tmp / "log"),
            log_level=lambda: "WARNING",
            remove_worktree_fn=lambda p, b: (removed.append((p, b)) or True),
        )
        tm = core.TaskManager(backend, hooks)

        old_ts = time.time() - 3600
        # Expired creator task owning wt_path
        (task_dir / "11112222.meta").write_text(
            json.dumps({
                "task_id": "11112222",
                "status": "completed",
                "started_at": old_ts,
                "completed_at": old_ts,
                "last_updated_at": old_ts,
                "worktree_root": str(wt_path),
                "worktree_path": str(wt_path),
                "worktree_branch": "dummy-swarm/11112222",
            }),
            encoding="utf-8",
        )
        # Recent reply task referencing a cwd inside wt_path
        (task_dir / "33334444.meta").write_text(
            json.dumps({
                "task_id": "33334444",
                "status": "completed",
                "started_at": time.time() - 10,
                "completed_at": time.time() - 5,
                "last_updated_at": time.time() - 5,
                "cwd": str(wt_path),
            }),
            encoding="utf-8",
        )
        tm.cleanup_old_tasks()
        assert removed == [], f"worktree was removed while recent reply referenced it: {removed}"
        assert (task_dir / "11112222.meta").exists()

        # Touching activity via _touch_worktree_activity also bumps last_updated_at
        tm._touch_worktree_activity(str(wt_path))
        refreshed = json.loads((task_dir / "11112222.meta").read_text(encoding="utf-8"))
        assert refreshed["last_updated_at"] > old_ts


def _run_all() -> int:
    tests = [
        fn for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed.append((fn.__name__, exc))
    print(f"\n{passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
