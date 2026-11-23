"""MCP 2026-07-28 + io.modelcontextprotocol/tasks + legacy initialize server engine."""

import json
import logging
import math
import signal
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from . import diagnostics
from .backend import RuntimeHooks, SwarmBackend
from .task_store import SyncProcRegistry, TaskManager, install_sigchld_handler

# ---------------------------------------------------------------------------
# Protocol negotiation constants
# ---------------------------------------------------------------------------
STATELESS_PROTOCOL_VERSIONS = ("2026-07-28",)
SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
ALL_PROTOCOL_VERSIONS = STATELESS_PROTOCOL_VERSIONS + SUPPORTED_PROTOCOL_VERSIONS
TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"
DISCOVER_TTL_MS = 3600000
TASK_POLL_INTERVAL_MS = 2000

# Methods that can block and must be dispatched to worker threads
BLOCKING_METHODS = {"tools/call", "tasks/cancel"}

_CANCELLED_MAX_ENTRIES = 1024


def is_valid_request_id(rid: Any) -> bool:
    """JSON-RPC 2.0 request IDs must be strings or finite numbers (never bool/list/dict)."""
    if isinstance(rid, bool):
        return False
    if isinstance(rid, (str, int)):
        return True
    if isinstance(rid, float):
        return not (math.isnan(rid) or math.isinf(rid))
    return False


class CancellationTracker:
    """Thread-safe bounded set of client-cancelled JSON-RPC request IDs."""

    def __init__(self, max_entries: int = _CANCELLED_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self.cancelled_requests: Set[Any] = set()
        self.order: Deque[Any] = deque()
        self.lock = threading.Lock()

    def mark_cancelled(self, rid: Any) -> None:
        if not is_valid_request_id(rid):
            return
        with self.lock:
            if rid in self.cancelled_requests:
                return
            while (
                len(self.cancelled_requests) >= self.max_entries
                or len(self.order) >= self.max_entries * 2
            ) and self.order:
                oldest = self.order.popleft()
                self.cancelled_requests.discard(oldest)
            self.cancelled_requests.add(rid)
            self.order.append(rid)

    def is_cancelled(self, rid: Any) -> bool:
        if not is_valid_request_id(rid):
            return False
        with self.lock:
            return rid in self.cancelled_requests

    def consume_if_cancelled(self, rid: Any) -> bool:
        if not is_valid_request_id(rid):
            return False
        with self.lock:
            if rid in self.cancelled_requests:
                self.cancelled_requests.discard(rid)
                return True
            return False


def client_supports_tasks(
    client_caps: Dict[str, Any], req_meta: Dict[str, Any]
) -> bool:
    extensions = client_caps.get("extensions")
    if isinstance(extensions, dict) and TASKS_EXTENSION_ID in extensions:
        val = extensions[TASKS_EXTENSION_ID]
        return val is not False and val is not None
    if TASKS_EXTENSION_ID in req_meta:
        val = req_meta[TASKS_EXTENSION_ID]
        return val is not False and val is not None
    return False


def validate_task_ids_param(raw: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    if not isinstance(raw, list) or not raw:
        return None, "task_ids must be a non-empty array of task ID strings"
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            return None, "Every element of task_ids must be a non-empty string"
    return raw, None


class McpSwarmServer:
    """Complete MCP stdio JSON-RPC server driving any ``SwarmBackend``."""

    def __init__(
        self,
        backend: SwarmBackend,
        hooks: RuntimeHooks,
        *,
        sync_procs: Optional[SyncProcRegistry] = None,
        cancellations: Optional[CancellationTracker] = None,
    ) -> None:
        self.backend = backend
        self.hooks = hooks
        self.sync_procs = sync_procs or SyncProcRegistry()
        self.cancellations = cancellations or CancellationTracker()
        self._send_lock = threading.Lock()
        self.tasks = TaskManager(
            backend,
            hooks,
            sync_procs=self.sync_procs,
            is_cancelled_fn=self.cancellations.is_cancelled,
            send_progress_fn=self.send_progress,
        )

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------
    def _emit_frame(self, response: Dict[str, Any]) -> None:
        if self.hooks.send_fn is not None:
            self.hooks.send_fn(response)
        else:
            self.send(response)

    def send(self, response: Dict[str, Any]) -> None:
        rid = response.get("id")
        if (
            rid is not None
            and is_valid_request_id(rid)
            and self.cancellations.consume_if_cancelled(rid)
        ):
            logging.info("Suppressed response for cancelled request id=%s", rid)
            return
        with self._send_lock:
            try:
                out = json.dumps(response, ensure_ascii=True)
                print(out, flush=True)
                logging.debug("Sent id=%s (%d bytes)", rid, len(out))
            except (TypeError, ValueError) as exc:
                logging.error("Serialization failed: %s", exc)
                safe_rid = rid if is_valid_request_id(rid) else None
                err = {
                    "jsonrpc": "2.0",
                    "id": safe_rid,
                    "error": {"code": -32603, "message": f"Serialization error: {exc}"},
                }
                print(json.dumps(err, ensure_ascii=True), flush=True)

    def send_result(
        self,
        rid: Any,
        result: Dict[str, Any],
        *,
        stateless: bool = False,
        force_result_type: Optional[str] = None,
        is_notification: bool = False,
    ) -> None:
        if is_notification:
            return
        payload = dict(result)
        if force_result_type is not None:
            payload["resultType"] = force_result_type
        elif stateless and "resultType" not in payload:
            payload["resultType"] = "complete"
        self._emit_frame({"jsonrpc": "2.0", "id": rid, "result": payload})

    def send_error(
        self,
        rid: Any,
        code: int,
        message: str,
        *,
        data: Any = None,
        is_notification: bool = False,
    ) -> None:
        if is_notification:
            return
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._emit_frame({"jsonrpc": "2.0", "id": rid, "error": err})

    def send_progress(
        self,
        progress_token: Any,
        elapsed: float,
        label: str,
        detail: Optional[str] = None,
    ) -> None:
        if progress_token is None:
            return
        message = f"{label} running ({int(elapsed)}s elapsed)"
        if detail:
            message = f"{message} -- {detail}"
        self._emit_frame({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": progress_token,
                "progress": round(elapsed, 1),
                "message": message,
            },
        })

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------
    def read_resource(self, uri: str) -> Optional[str]:
        b = self.backend
        scheme = b.uri_scheme
        if uri == f"{scheme}:///server-info":
            payload: Dict[str, Any] = {
                "name": b.server_name,
                "version": b.version,
                "supported_protocol_versions": list(ALL_PROTOCOL_VERSIONS),
                "extensions": [TASKS_EXTENSION_ID],
                "tools": [t["name"] for t in b.tools],
                "task_dir": str(self.hooks.task_dir()),
                "worktree_dir": str(self.hooks.worktree_base_dir()),
                "task_max_age_seconds": self.hooks.task_max_age(),
                "log_file": self.hooks.log_file(),
                "log_level": self.hooks.log_level(),
            }
            extra = b.get_server_info_extra()
            if extra:
                payload.update(extra)
            return json.dumps(payload, indent=2)

        if uri == f"{scheme}:///config":
            return json.dumps(b.get_config_resource(), indent=2)

        if uri == f"{scheme}:///tasks":
            task_dir = self.hooks.task_dir()
            tasks = []
            sid_field = b.session_field_name
            for meta_file in sorted(task_dir.glob("*.meta")):
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    task_id = meta.get("task_id", meta_file.stem)
                    state = self.tasks.resolve_task_state(task_id)
                    resolved_meta = (
                        state.get("meta")
                        if isinstance(state.get("meta"), dict)
                        else meta
                    )
                    entry: Dict[str, Any] = {
                        "task_id": task_id,
                        "status": state.get("status"),
                        "elapsed_seconds": state.get("elapsed_seconds"),
                    }
                    if state.get("status") in ("completed", "failed"):
                        entry.update({
                            "exit_code": state.get("exit_code"),
                            "exit_code_lost": bool(state.get("exit_code_lost")),
                            "status_inferred": bool(state.get("status_inferred")),
                        })
                    for key in ("model", "worktree_path", "worktree_branch", sid_field):
                        val = resolved_meta.get(key)
                        if val:
                            entry[key] = val

                    analysis: Optional[Dict[str, Any]] = None
                    if state.get("status") == "failed":
                        mismatch = resolved_meta.get("session_mismatch_error")
                        if isinstance(mismatch, dict):
                            failure = dict(mismatch)
                        else:
                            stdout_path = task_dir / f"{task_id}.stdout"
                            analysis, _ = b.parse_jsonl_file(stdout_path)
                            stdout = (
                                ""
                                if (
                                    analysis.get("terminal_error")
                                    or analysis.get("errors")
                                )
                                else self.hooks.safe_read(stdout_path)
                            )
                            stderr = self.hooks.safe_read(task_dir / f"{task_id}.stderr")
                            failure = b.build_failure_info(
                                stdout,
                                stderr,
                                command=resolved_meta.get("command"),
                                analysis=analysis,
                                model=resolved_meta.get("model"),
                            )
                        entry.update({
                            "model": failure.get("model") or resolved_meta.get("model"),
                            "failure_cause": failure["cause"],
                            "failure_category": failure["category"],
                            "retryable": failure["retryable"],
                            "failure_action": failure["action"],
                        })
                        if failure.get("stderr_diagnostics"):
                            entry["stderr_diagnostics"] = failure["stderr_diagnostics"]

                    extra_fields = b.extra_resource_task_fields(
                        resolved_meta, analysis, state
                    )
                    if extra_fields:
                        entry.update(extra_fields)
                    tasks.append(entry)
                except Exception:
                    continue
            return json.dumps(tasks, indent=2)

        return None

    # ------------------------------------------------------------------
    # Reply hint helper
    # ------------------------------------------------------------------
    def _format_reply_hint(
        self, sid: str, worktree_path: Optional[str] = None
    ) -> str:
        b = self.backend
        if worktree_path:
            return (
                f"{b.session_label}: {sid}\n"
                f'Use {b.reply_tool_name}({b.session_param_name}="{sid}", '
                f'cwd="{worktree_path}", prompt="...") to continue this session.'
            )
        return (
            f"{b.session_label}: {sid}\n"
            f'Use {b.reply_tool_name}({b.session_param_name}="{sid}", prompt="...") to continue this session.'
        )

    # ------------------------------------------------------------------
    # MCP Tasks extension (SEP-2663) result builder
    # ------------------------------------------------------------------
    def build_mcp_task_tool_result(
        self, info: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        b = self.backend
        hints = b.hints()
        sid_field = b.session_field_name
        status = info["status"]
        tool_name = meta.get("tool_name", b.sync_tool_name)
        is_reply = tool_name == b.reply_tool_name
        label = f"{b.product_name} reply" if is_reply else b.product_name

        if status == "completed":
            text = info["result"]
            if info.get("status_inferred"):
                text = f"{hints['unverified_completion_hint']}\n\n{text}"
            details: List[str] = []
            sid = info.get(sid_field)
            if sid and not is_reply:
                details.append(self._format_reply_hint(sid, info.get("worktree_path")))
            if info.get("worktree_path") or info.get("worktree_branch"):
                wt_lines = []
                if info.get("worktree_path"):
                    wt_lines.append(f"Worktree Path: {info['worktree_path']}")
                if info.get("worktree_branch"):
                    wt_lines.append(f"Worktree Branch: {info['worktree_branch']}")
                details.append("\n".join(wt_lines))
            details.extend(b.format_extra_completion_details(info, is_reply=is_reply))
            if details:
                text += "\n\n---\n" + "\n\n---\n".join(details)
            structured_map: Dict[str, Any] = {
                "status": "completed",
                "text": text,
                sid_field: sid,
                "model": info.get("model") or meta.get("model"),
                "exit_code": info.get("exit_code"),
                "worktree_path": info.get("worktree_path"),
                "worktree_branch": info.get("worktree_branch"),
            }
            structured_map.update(
                b.extra_structured_sync_fields(info, is_reply=is_reply)
            )
            structured = diagnostics.clean_none(structured_map)
            return {
                "resultType": "complete",
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured,
                "isError": False,
            }

        exit_code = info.get("exit_code")
        exit_display = exit_code if exit_code is not None else "?"
        detail = info.get("failure_detail") or (
            "Model: unknown\n"
            f"Cause: {info.get('failure_cause', f'Unknown {b.product_name} failure')}"
        )
        if info.get("partial_output"):
            detail += (
                "\n\nPartial output (before failure): "
                f"{info['partial_output'][:300]}"
            )
        extra_details = b.format_extra_completion_details(info, is_reply=is_reply)
        if extra_details:
            detail += "\n\n" + "\n\n".join(extra_details)
        if info.get("status_inferred"):
            detail = f"{hints['unverified_failure_hint']}\n\n{detail}"
        text = f"{label} FAILED (exit {exit_display})\n{detail}"
        failure_obj = diagnostics.clean_none({
            "model": info.get("model"),
            "cause": info.get("failure_cause") or f"Unknown {b.product_name} failure",
            "category": info.get("failure_category") or "unknown",
            "retryable": info.get("retryable"),
            "action": info.get("failure_action"),
            "stdout_diagnostics": info.get("stdout_diagnostics"),
            "stderr_diagnostics": info.get("stderr_diagnostics"),
        })
        structured_map = {
            "status": "failed",
            "text": text,
            sid_field: info.get(sid_field),
            "model": info.get("model") or meta.get("model"),
            "exit_code": info.get("exit_code"),
            "worktree_path": info.get("worktree_path"),
            "worktree_branch": info.get("worktree_branch"),
            "partial_output": info.get("partial_output"),
            "failure": failure_obj,
        }
        structured_map.update(b.extra_structured_sync_fields(info, is_reply=is_reply))
        structured = diagnostics.clean_none(structured_map)
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": True,
        }

    # ------------------------------------------------------------------
    # Request dispatcher
    # ------------------------------------------------------------------
    def handle(self, request: Dict[str, Any]) -> None:
        b = self.backend
        has_id = "id" in request
        rid = request.get("id")
        if has_id and rid is not None and not is_valid_request_id(rid):
            self.send_error(
                None,
                -32600,
                "Invalid request: id must be a string, number, or null",
            )
            return
        is_notification = (not has_id) or (rid is None)

        method = request.get("method")
        if not isinstance(method, str) or not method:
            self.send_error(
                rid,
                -32600,
                "Invalid request: method must be a non-empty string",
                is_notification=is_notification,
            )
            return

        raw_params = request.get("params")
        if raw_params is None:
            params: Dict[str, Any] = {}
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            self.send_error(
                rid,
                -32602,
                "Invalid params: expected object",
                is_notification=is_notification,
            )
            return

        # --- Known notifications ---
        if method == "notifications/initialized":
            return

        if method == "notifications/cancelled":
            cancelled_id = params.get("requestId")
            if is_valid_request_id(cancelled_id):
                logging.info("Client cancelled request id=%s", cancelled_id)
                self.cancellations.mark_cancelled(cancelled_id)
                if self.sync_procs.has(cancelled_id):
                    threading.Thread(
                        target=self.sync_procs.terminate,
                        args=(cancelled_id,),
                        kwargs={
                            "is_valid_id_fn": is_valid_request_id,
                            "stop_proc_fn": self.hooks.stop_proc,
                        },
                        daemon=True,
                    ).start()
            return

        # JSON-RPC 2.0 §4.1: unknown notifications MUST NOT receive a reply.
        if is_notification:
            logging.debug("Ignoring notification method=%s", method)
            return

        # --- Per-request MCP 2026-07-28 metadata ---
        raw_meta = params.get("_meta")
        req_meta = raw_meta if isinstance(raw_meta, dict) else {}
        proto_ver = req_meta.get("io.modelcontextprotocol/protocolVersion")
        raw_client_caps = req_meta.get("io.modelcontextprotocol/clientCapabilities")
        client_caps = raw_client_caps if isinstance(raw_client_caps, dict) else {}

        if method not in ("server/discover", "initialize") and proto_ver is not None:
            if (
                not isinstance(proto_ver, str)
                or proto_ver not in ALL_PROTOCOL_VERSIONS
            ):
                self.send_error(
                    rid,
                    -32022,
                    f"Unsupported protocol version: {proto_ver}",
                    data={
                        "supportedVersions": list(ALL_PROTOCOL_VERSIONS),
                        "supported": list(ALL_PROTOCOL_VERSIONS),
                        "requested": proto_ver,
                    },
                )
                return

        is_stateless = proto_ver in STATELESS_PROTOCOL_VERSIONS

        # --- Discovery & Lifecycle ---
        if method == "server/discover":
            self.send_result(
                rid,
                {
                    "supportedVersions": list(ALL_PROTOCOL_VERSIONS),
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "extensions": {
                            TASKS_EXTENSION_ID: {},
                        },
                    },
                    "serverInfo": {
                        "name": b.server_name,
                        "version": b.version,
                    },
                    "instructions": b.server_instructions,
                    "ttlMs": DISCOVER_TTL_MS,
                    "cacheScope": "public",
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {
                            "name": b.server_name,
                            "version": b.version,
                        },
                    },
                },
                force_result_type="complete",
            )
            return

        if method == "initialize":
            requested = params.get("protocolVersion")
            negotiated = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            if requested != negotiated:
                logging.info(
                    "Client requested protocol %s; responding with %s",
                    requested,
                    negotiated,
                )
            self.send_result(
                rid,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {
                        "name": b.server_name,
                        "version": b.version,
                    },
                    "instructions": b.server_instructions,
                },
            )
            return

        if method == "ping":
            self.send_result(rid, {}, stateless=is_stateless)
            return

        if method == "tools/list":
            self.send_result(rid, {"tools": b.tools}, stateless=is_stateless)
            return

        if method == "resources/list":
            self.send_result(rid, {"resources": b.resources}, stateless=is_stateless)
            return

        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str) or not uri:
                self.send_error(rid, -32602, "uri must be a non-empty string")
                return
            content = self.read_resource(uri)
            if content is not None:
                self.send_result(
                    rid,
                    {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": content,
                        }],
                    },
                    stateless=is_stateless,
                )
            else:
                self.send_error(rid, -32602, f"Unknown resource: {uri}")
            return

        # --- Official MCP Tasks extension (io.modelcontextprotocol/tasks) ---
        if method == "tasks/get":
            task_id = params.get("taskId")
            if not isinstance(task_id, str) or not diagnostics.validate_task_id(
                task_id
            ):
                self.send_error(rid, -32602, f"Invalid taskId: {task_id!r}")
                return
            state = self.tasks.resolve_task_state(task_id)
            status = state["status"]
            if status == "not_found":
                self.send_error(rid, -32602, f"Unknown task: {task_id}")
                return
            if status == "error":
                self.send_error(
                    rid, -32603, state.get("error", "Task metadata error")
                )
                return

            meta = state.get("meta", {})
            started_at = meta.get("started_at", time.time())
            created_at_iso = diagnostics.iso_utc(started_at)
            ttl_ms = self.hooks.task_max_age() * 1000

            if status == "running":
                stdout_path = self.hooks.task_dir() / f"{task_id}.stdout"
                _, jsonl_status = b.parse_jsonl_file(stdout_path)
                last_updated = meta.get("last_updated_at", started_at)
                try:
                    if stdout_path.exists():
                        last_updated = max(last_updated, stdout_path.stat().st_mtime)
                except OSError:
                    pass
                self.send_result(
                    rid,
                    {
                        "taskId": task_id,
                        "status": "working",
                        "statusMessage": diagnostics.format_mcp_task_status_message(
                            jsonl_status,
                            state["elapsed_seconds"],
                            product_name=b.product_name,
                        ),
                        "createdAt": created_at_iso,
                        "lastUpdatedAt": diagnostics.iso_utc(last_updated),
                        "ttlMs": ttl_ms,
                        "pollIntervalMs": TASK_POLL_INTERVAL_MS,
                    },
                    force_result_type="complete",
                )
                return

            info = self.tasks.check_task(task_id, state=state)
            completed_at = (
                meta.get("completed_at") or meta.get("last_updated_at") or time.time()
            )

            if status == "cancelled":
                msg_parts = ["Task cancelled"]
                if info.get("worktree_path"):
                    msg_parts.append(f"worktree preserved: {info['worktree_path']}")
                if info.get("worktree_branch"):
                    msg_parts.append(f"branch: {info['worktree_branch']}")
                self.send_result(
                    rid,
                    {
                        "taskId": task_id,
                        "status": "cancelled",
                        "statusMessage": "; ".join(msg_parts),
                        "createdAt": created_at_iso,
                        "lastUpdatedAt": diagnostics.iso_utc(completed_at),
                        "ttlMs": ttl_ms,
                    },
                    force_result_type="complete",
                )
                return

            tool_result = self.build_mcp_task_tool_result(info, meta)
            if status == "completed":
                status_message = f"Completed in {info['elapsed_seconds']}s"
            else:
                status_message = (
                    info.get("failure_cause") or f"{b.product_name} task failed"
                )

            self.send_result(
                rid,
                {
                    "taskId": task_id,
                    "status": "completed",
                    "statusMessage": status_message,
                    "createdAt": created_at_iso,
                    "lastUpdatedAt": diagnostics.iso_utc(completed_at),
                    "ttlMs": ttl_ms,
                    "result": tool_result,
                },
                force_result_type="complete",
            )
            return

        if method == "tasks/cancel":
            task_id = params.get("taskId")
            if not isinstance(task_id, str) or not diagnostics.validate_task_id(
                task_id
            ):
                self.send_error(rid, -32602, f"Invalid taskId: {task_id!r}")
                return
            cancel_res = self.tasks.cancel_task(task_id)
            if cancel_res["status"] == "not_found":
                self.send_error(
                    rid, -32602, cancel_res.get("error", f"Unknown task: {task_id}")
                )
                return
            if cancel_res.get("already_terminal"):
                self.send_error(
                    rid,
                    -32602,
                    f"Cannot cancel task {task_id}: task is already in terminal state '{cancel_res['status']}'",
                )
                return
            if cancel_res["status"] == "error":
                self.send_error(
                    rid, -32603, cancel_res.get("error", "Failed to cancel task")
                )
                return
            self.send_result(rid, {}, force_result_type="complete")
            return

        # --- Tool calls ---
        if method == "tools/call":
            self._handle_tool_call(
                rid,
                params,
                req_meta=req_meta,
                client_caps=client_caps,
                proto_ver=proto_ver,
                is_stateless=is_stateless,
            )
            return

        self.send_error(rid, -32601, f"Method not found: {method}")

    def _abort_if_launch_cancelled(self, rid: Any, task_id: str) -> bool:
        """If the request was cancelled while launching, stop the just-started task."""
        if rid is not None and self.cancellations.is_cancelled(rid):
            logging.info(
                "Request id=%s cancelled during launch; cancelling task %s",
                rid,
                task_id,
            )
            self.tasks.cancel_task(task_id)
            self.cancellations.consume_if_cancelled(rid)
            return True
        return False

    def _handle_tool_call(
        self,
        rid: Any,
        params: Dict[str, Any],
        *,
        req_meta: Dict[str, Any],
        client_caps: Dict[str, Any],
        proto_ver: Any,
        is_stateless: bool,
    ) -> None:
        b = self.backend
        hints = b.hints()
        sid_param = b.session_param_name
        sid_field = b.session_field_name
        tool = params.get("name")
        raw_args = params.get("arguments")
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            self.send_error(rid, -32602, "arguments must be an object")
            return

        progress_token = req_meta.get("progressToken")
        use_mcp_tasks = (
            (proto_ver is None or is_stateless)
            and client_supports_tasks(client_caps, req_meta)
        )

        # 1. Sync / native MCP Task execution
        if tool == b.sync_tool_name:
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self.send_error(rid, -32602, "prompt must be a non-empty string")
                return
            err_msg = b.validate_call_params(args, is_reply=False)
            if err_msg:
                self.send_error(rid, -32602, err_msg)
                return

            args.pop("timeout", None)  # ignored: sync tools run to completion

            if use_mcp_tasks:
                if rid is not None and self.cancellations.is_cancelled(rid):
                    self.cancellations.consume_if_cancelled(rid)
                    return
                try:
                    launch_info = self.tasks.start_async(
                        args,
                        high_entropy_id=True,
                        mcp_task=True,
                        tool_name=b.sync_tool_name,
                    )
                except Exception as exc:
                    err_text = f"Error starting {b.product_name} task: {exc}"
                    self.send_result(
                        rid,
                        {
                            "content": [{"type": "text", "text": err_text}],
                            "structuredContent": {"status": "failed", "text": err_text},
                            "isError": True,
                        },
                        stateless=True,
                    )
                    return
                if self._abort_if_launch_cancelled(rid, launch_info["task_id"]):
                    return
                now_iso = diagnostics.iso_utc(launch_info["started_at"])
                self.send_result(
                    rid,
                    {
                        "taskId": launch_info["task_id"],
                        "status": "working",
                        "statusMessage": f"Starting {b.product_name} session...",
                        "createdAt": now_iso,
                        "lastUpdatedAt": now_iso,
                        "ttlMs": self.hooks.task_max_age() * 1000,
                        "pollIntervalMs": TASK_POLL_INTERVAL_MS,
                    },
                    force_result_type="task",
                )
                return

            outcome = self.tasks.run_sync_detailed(
                args,
                request_id=rid,
                progress_token=progress_token,
            )
            text = outcome["text"]
            session_id = outcome.get(sid_field)
            worktree_info = outcome.get("worktree_info")
            details: List[str] = []
            if session_id:
                wt_path = worktree_info["worktree_path"] if worktree_info else None
                details.append(self._format_reply_hint(session_id, wt_path))
            if worktree_info:
                details.append(
                    f"Worktree Path: {worktree_info['worktree_path']}\n"
                    f"Worktree Branch: {worktree_info['worktree_branch']}"
                )
            details.extend(
                b.format_extra_completion_details(outcome, is_reply=False)
            )
            if details:
                text += "\n\n---\n" + "\n\n---\n".join(details)
            structured_map: Dict[str, Any] = {
                "status": outcome["status"],
                "text": text,
                sid_field: session_id,
                "model": outcome.get("model"),
                "exit_code": outcome.get("exit_code"),
                "worktree_path": (
                    worktree_info["worktree_path"] if worktree_info else None
                ),
                "worktree_branch": (
                    worktree_info["worktree_branch"] if worktree_info else None
                ),
                "partial_output": outcome.get("partial_output"),
                "failure": diagnostics.failure_structured(outcome.get("failure")),
            }
            structured_map.update(
                b.extra_structured_sync_fields(outcome, is_reply=False)
            )
            structured = diagnostics.clean_none(structured_map)
            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": structured,
                    "isError": bool(outcome["failed"]),
                },
                stateless=is_stateless,
            )
            return

        # 2. Async fan-out launch
        if tool == b.async_tool_name:
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self.send_error(rid, -32602, "prompt must be a non-empty string")
                return
            err_msg = b.validate_call_params(args, is_reply=False)
            if err_msg:
                self.send_error(rid, -32602, err_msg)
                return

            if rid is not None and self.cancellations.is_cancelled(rid):
                self.cancellations.consume_if_cancelled(rid)
                return

            try:
                if self.hooks.start_async_fn is not None:
                    launch_info = self.hooks.start_async_fn(args)
                else:
                    launch_info = self.tasks.start_async(args)
            except Exception as exc:
                err_text = f"Error starting async {b.product_name} task: {exc}"
                self.send_result(
                    rid,
                    {
                        "content": [{"type": "text", "text": err_text}],
                        "structuredContent": {"status": "error", "error": str(exc)},
                        "isError": True,
                    },
                    stateless=is_stateless,
                )
                return

            task_id = launch_info["task_id"]
            if self._abort_if_launch_cancelled(rid, task_id):
                return

            lines = [f"{b.product_name} task {task_id} started."]
            if launch_info.get("worktree_path"):
                lines.append(f"Worktree Path: {launch_info['worktree_path']}")
            if launch_info.get("worktree_branch"):
                lines.append(f"Worktree Branch: {launch_info['worktree_branch']}")
            lines.extend([
                "",
                (
                    f"NOT TRACKED BY YOUR CLIENT. This id is internal to "
                    f"{b.uri_scheme}. No completion notification will arrive and "
                    "nothing will wake this session when the task finishes."
                ),
                "",
                (
                    f'You MUST call {b.wait_tool_name}(task_ids=["{task_id}"]) before this '
                    f"turn ends, or the result is stranded on disk with nobody to "
                    f"collect it. {b.wait_tool_name} is the call your client tracks and the "
                    "only thing that brings the answer back."
                ),
                "",
                (
                    f'{b.status_tool_name}(task_ids=["{task_id}"]) is a progress peek only. '
                    "It does not collect the result and it does not keep the "
                    "session alive."
                ),
            ])
            structured = diagnostics.clean_none({
                "status": "started",
                "task_id": task_id,
                "model": launch_info.get("model"),
                "worktree_path": launch_info.get("worktree_path"),
                "worktree_branch": launch_info.get("worktree_branch"),
            })
            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                    "structuredContent": structured,
                    "isError": False,
                },
                stateless=is_stateless,
            )
            return

        # 3. Session reply / resume
        if tool == b.reply_tool_name:
            session_id = args.get(sid_param)
            prompt = args.get("prompt")

            if (
                not isinstance(session_id, str)
                or not session_id.strip()
                or not isinstance(prompt, str)
                or not prompt
            ):
                self.send_error(
                    rid, -32602, f"{sid_param} and prompt are required strings"
                )
                return
            err_msg = b.validate_call_params(args, is_reply=True)
            if err_msg:
                self.send_error(rid, -32602, err_msg)
                return

            extra_params = {
                k: args[k]
                for k in b.reply_extra_param_keys
                if k in args and args[k] is not None
            }

            if use_mcp_tasks:
                if rid is not None and self.cancellations.is_cancelled(rid):
                    self.cancellations.consume_if_cancelled(rid)
                    return
                reply_params = dict(extra_params)
                reply_params[sid_param] = session_id
                reply_params["prompt"] = prompt
                try:
                    launch_info = self.tasks.start_async(
                        reply_params,
                        high_entropy_id=True,
                        mcp_task=True,
                        tool_name=b.reply_tool_name,
                    )
                except Exception as exc:
                    err_text = f"Error starting {b.product_name} reply task: {exc}"
                    self.send_result(
                        rid,
                        {
                            "content": [{"type": "text", "text": err_text}],
                            "structuredContent": {
                                "status": "failed",
                                "text": err_text,
                                sid_field: session_id,
                            },
                            "isError": True,
                        },
                        stateless=True,
                    )
                    return
                if self._abort_if_launch_cancelled(rid, launch_info["task_id"]):
                    return
                now_iso = diagnostics.iso_utc(launch_info["started_at"])
                self.send_result(
                    rid,
                    {
                        "taskId": launch_info["task_id"],
                        "status": "working",
                        "statusMessage": f"Starting {b.product_name} reply session...",
                        "createdAt": now_iso,
                        "lastUpdatedAt": now_iso,
                        "ttlMs": self.hooks.task_max_age() * 1000,
                        "pollIntervalMs": TASK_POLL_INTERVAL_MS,
                    },
                    force_result_type="task",
                )
                return

            outcome = self.tasks.run_reply_sync_detailed(
                session_id,
                prompt,
                extra_params=extra_params,
                request_id=rid,
                progress_token=progress_token,
            )
            text = outcome["text"]
            extra_details = b.format_extra_completion_details(outcome, is_reply=True)
            if extra_details:
                text += "\n\n---\n" + "\n\n---\n".join(extra_details)
            structured_map = {
                "status": outcome["status"],
                "text": text,
                sid_field: outcome.get(sid_field),
                "model": outcome.get("model"),
                "exit_code": outcome.get("exit_code"),
                "partial_output": outcome.get("partial_output"),
                "failure": diagnostics.failure_structured(outcome.get("failure")),
            }
            structured_map.update(
                b.extra_structured_sync_fields(outcome, is_reply=True)
            )
            structured = diagnostics.clean_none(structured_map)
            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": structured,
                    "isError": bool(outcome["failed"]),
                },
                stateless=is_stateless,
            )
            return

        # 4. Live task status peek
        if tool == b.status_tool_name:
            task_ids, err_msg = validate_task_ids_param(args.get("task_ids"))
            if err_msg or task_ids is None:
                self.send_error(rid, -32602, err_msg or "task_ids is required")
                return

            parts: List[str] = []
            structured_tasks: List[Dict[str, Any]] = []
            error_count = 0
            task_dir = self.hooks.task_dir()

            for tid in task_ids:
                state = self.tasks.resolve_task_state(tid)
                status = state["status"]

                if status in ("error", "not_found"):
                    error_count += 1
                    parts.append(
                        f"=== Task {tid} === {state.get('error', status).upper()}"
                    )
                    structured_tasks.append(
                        diagnostics.clean_none({
                            "task_id": tid,
                            "status": status,
                            "error": state.get("error", status),
                        })
                    )
                    continue

                elapsed = state["elapsed_seconds"]
                stdout_path = task_dir / f"{tid}.stdout"
                stderr_path = task_dir / f"{tid}.stderr"
                analysis, jsonl_status = b.parse_jsonl_file(stdout_path)

                if status == "running":
                    lines = [f"=== Task {tid} ({elapsed}s elapsed) ==="]
                elif status == "failed":
                    exit_code = state.get("exit_code")
                    if exit_code is None:
                        exit_code = "?"
                    lines = [
                        f"=== Task {tid} (FAILED in {elapsed}s, exit {exit_code}) ==="
                    ]
                    if state.get("status_inferred"):
                        lines.append(hints["unverified_failure_hint"])
                elif status == "cancelled":
                    lines = [f"=== Task {tid} (CANCELLED after {elapsed}s) ==="]
                else:
                    lines = [f"=== Task {tid} (COMPLETED in {elapsed}s) ==="]
                    if state.get("status_inferred"):
                        lines.append(hints["unverified_completion_hint"])

                meta = state.get("meta", {})
                if b.prefer_observed_session_id:
                    tid_session = analysis.get(sid_field) or meta.get(sid_field)
                else:
                    tid_session = meta.get(sid_field) or analysis.get(sid_field)
                if tid_session:
                    lines.append(f"{b.session_label}: {tid_session}")
                if meta.get("worktree_path"):
                    lines.append(f"Worktree: {meta['worktree_path']}")
                if meta.get("worktree_branch"):
                    lines.append(f"Branch: {meta['worktree_branch']}")

                display_phase = (
                    status
                    if status in ("completed", "failed", "cancelled")
                    else jsonl_status["phase"]
                )
                lines.append(f"Phase: {display_phase}")
                lines.append(f"Tools called: {jsonl_status['tools_called']}")

                failure_info: Optional[Dict[str, Any]] = None
                if status == "failed":
                    mismatch = meta.get("session_mismatch_error")
                    if isinstance(mismatch, dict):
                        failure_info = dict(mismatch)
                        if not failure_info.get("model") and meta.get("model"):
                            failure_info["model"] = meta["model"]
                    else:
                        stdout_text = (
                            ""
                            if (
                                analysis.get("terminal_error")
                                or analysis.get("errors")
                            )
                            else self.hooks.safe_read(stdout_path)
                        )
                        failure_info = b.build_failure_info(
                            stdout_text,
                            self.hooks.safe_read(stderr_path),
                            command=meta.get("command"),
                            analysis=analysis,
                            model=analysis.get("model") or meta.get("model"),
                        )
                    lines.extend(b.format_failure_info(failure_info).splitlines())
                elif status == "running" and jsonl_status["last_error"]:
                    observed = b.classify_failure(jsonl_status["last_error"])
                    retryable = observed.get("retryable")
                    retryable_text = (
                        "yes"
                        if retryable is True
                        else "no"
                        if retryable is False
                        else "unknown"
                    )
                    lines.append(
                        "Error observed (non-terminal; process still running): "
                        f"{jsonl_status['last_error'][:300]}"
                    )
                    lines.append(f"Category: {observed['category']}")
                    lines.append(
                        "Retryable: "
                        f"{retryable_text} (informational only; the wrapper did not retry)"
                    )

                if jsonl_status["last_warning"]:
                    lines.append(f"Warning: {jsonl_status['last_warning'][:300]}")

                if jsonl_status["terminal_conflict"]:
                    lines.append(
                        "Terminal signals conflicted; lifecycle state used the process exit code."
                    )

                extra_lines = b.format_extra_status_lines(
                    analysis, jsonl_status, meta, status
                )
                if extra_lines:
                    lines.extend(extra_lines)

                if jsonl_status["last_tool"]:
                    tool_info = jsonl_status["last_tool"]
                    if jsonl_status["last_tool_args"]:
                        tool_info += f"({jsonl_status['last_tool_args'][:120]})"
                    lines.append(f"Last tool: {tool_info}")

                if jsonl_status["last_reasoning"]:
                    lines.append(
                        f"Thinking: {jsonl_status['last_reasoning'][:200]}"
                    )

                if jsonl_status["last_assistant_text"]:
                    output_label = (
                        "Partial output (before failure)"
                        if status == "failed"
                        else "Output"
                    )
                    lines.append(
                        f"{output_label}: {jsonl_status['last_assistant_text'][:300]}"
                    )

                parts.append("\n".join(lines))
                st_entry: Dict[str, Any] = {
                    "task_id": tid,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "phase": display_phase,
                    "tools_called": jsonl_status["tools_called"],
                    sid_field: tid_session,
                    "model": (
                        (failure_info.get("model") if failure_info else None)
                        or analysis.get("model")
                        or meta.get("model")
                    ),
                    "worktree_path": meta.get("worktree_path"),
                    "worktree_branch": meta.get("worktree_branch"),
                    "last_tool": jsonl_status.get("last_tool"),
                    "last_tool_args": jsonl_status.get("last_tool_args"),
                    "last_reasoning": jsonl_status.get("last_reasoning"),
                    "last_assistant_text": jsonl_status.get("last_assistant_text"),
                    "last_warning": jsonl_status.get("last_warning"),
                    "last_error": jsonl_status.get("last_error"),
                    "exit_code": (
                        state.get("exit_code")
                        if status in ("completed", "failed")
                        else None
                    ),
                    "exit_code_lost": (
                        bool(state.get("exit_code_lost"))
                        if status in ("completed", "failed")
                        else None
                    ),
                    "status_inferred": (
                        bool(state.get("status_inferred"))
                        if status in ("completed", "failed")
                        else None
                    ),
                    "terminal_conflict": (
                        bool(jsonl_status.get("terminal_conflict")) or None
                    ),
                    "failure": diagnostics.failure_structured(failure_info),
                }
                st_entry.update(
                    b.extra_structured_status_fields(
                        analysis, jsonl_status, meta, status
                    )
                )
                structured_tasks.append(diagnostics.clean_none(st_entry))

            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                    "structuredContent": {"tasks": structured_tasks},
                    "isError": error_count == len(task_ids),
                },
                stateless=is_stateless,
            )
            return

        # 5. Batch wait
        if tool == b.wait_tool_name:
            task_ids, err_msg = validate_task_ids_param(args.get("task_ids"))
            if err_msg or task_ids is None:
                self.send_error(rid, -32602, err_msg or "task_ids is required")
                return

            timeout = args.get("timeout", 1800) if "timeout" in args else 1800
            if timeout is not None:
                if (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, (int, float))
                    or math.isnan(timeout)
                    or math.isinf(timeout)
                    or timeout <= 0
                ):
                    self.send_error(
                        rid, -32602, "timeout must be a positive number or null"
                    )
                    return
                min_wait = self.hooks.wait_min_timeout()
                if timeout < min_wait:
                    logging.info(
                        "Raising %s timeout %ss -> %ss to cross the "
                        "client's %ss auto-background threshold",
                        b.wait_tool_name,
                        timeout,
                        min_wait,
                        self.hooks.client_auto_background_s,
                    )
                    timeout = min_wait

            results = self.tasks.wait_tasks(
                task_ids,
                timeout=timeout,
                request_id=rid,
                progress_token=progress_token,
            )

            parts = []
            structured_tasks = []
            any_error = False

            for tid in task_ids:
                info = results.get(tid, {"status": "unknown", "task_id": tid})
                st = info["status"]
                elapsed = info.get("elapsed_seconds", 0)

                if st == "completed":
                    header = f"=== Task {tid} (completed in {elapsed}s) ==="
                    sid = info.get(sid_field)
                    if sid:
                        header += f"\n{b.session_label}: {sid}"
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = info["result"]
                    extra_details = b.format_extra_completion_details(
                        info, is_reply=False
                    )
                    if extra_details:
                        detail += "\n\n---\n" + "\n\n---\n".join(extra_details)
                    if info.get("status_inferred"):
                        detail = f"{hints['unverified_completion_hint']}\n\n{detail}"
                    parts.append(f"{header}\n{detail}")
                elif st == "failed":
                    any_error = True
                    exit_code = info.get("exit_code")
                    if exit_code is None:
                        exit_code = "?"
                    header = (
                        f"=== Task {tid} (FAILED in {elapsed}s, exit {exit_code}) ==="
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = info.get("failure_detail") or (
                        "Model: unknown\n"
                        f"Cause: {info.get('failure_cause', f'Unknown {b.product_name} failure')}"
                    )
                    if info.get("partial_output"):
                        detail += (
                            "\n\nPartial output (before failure): "
                            f"{info['partial_output'][:300]}"
                        )
                    extra_details = b.format_extra_completion_details(
                        info, is_reply=False
                    )
                    if extra_details:
                        detail += "\n\n" + "\n\n".join(extra_details)
                    if info.get("status_inferred"):
                        detail = f"{hints['unverified_failure_hint']}\n\n{detail}"
                    parts.append(f"{header}\n{detail}")
                elif st == "cancelled":
                    header = f"=== Task {tid} (CANCELLED after {elapsed}s) ==="
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = (
                        info.get("error")
                        if info.get("wait_cancelled")
                        else (info.get("result") or "Task was cancelled")
                    )
                    parts.append(f"{header}\n{detail}")
                elif st == "timeout":
                    header = (
                        f"=== Task {tid} === STILL RUNNING (wait timed out, "
                        f"task is NOT killed -- call {b.wait_tool_name} again to "
                        f"resume waiting)"
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    parts.append(f"{header}\n{hints['still_running_hint']}")
                else:
                    any_error = True
                    header = f"=== Task {tid} === {info.get('error', st)}"
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    parts.append(header)

                w_entry: Dict[str, Any] = {
                    "task_id": tid,
                    "status": st,
                    "elapsed_seconds": info.get("elapsed_seconds"),
                    "result": info.get("result"),
                    sid_field: info.get(sid_field),
                    "model": info.get("model"),
                    "worktree_path": info.get("worktree_path"),
                    "worktree_branch": info.get("worktree_branch"),
                    "exit_code": info.get("exit_code"),
                    "exit_code_lost": info.get("exit_code_lost"),
                    "status_inferred": info.get("status_inferred"),
                    "failure_cause": info.get("failure_cause"),
                    "failure_category": info.get("failure_category"),
                    "retryable": info.get("retryable"),
                    "failure_action": info.get("failure_action"),
                    "partial_output": info.get("partial_output"),
                    "stdout_diagnostics": info.get("stdout_diagnostics"),
                    "stderr_diagnostics": info.get("stderr_diagnostics"),
                    "error": info.get("error"),
                }
                w_entry.update(b.extra_structured_wait_fields(info))
                structured_tasks.append(diagnostics.clean_none(w_entry))

            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                    "structuredContent": {"tasks": structured_tasks},
                    "isError": any_error,
                },
                stateless=is_stateless,
            )
            return

        # 6. Cancel task
        if tool == b.cancel_tool_name:
            task_id = args.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                self.send_error(rid, -32602, "task_id must be a non-empty string")
                return

            result = self.tasks.cancel_task(task_id)
            lines = [
                f"Task {task_id}: {result.get('message', result.get('error', result['status']))}"
            ]
            if result.get("worktree_path"):
                lines.append(f"Worktree preserved: {result['worktree_path']}")
            if result.get("worktree_branch"):
                lines.append(f"Branch preserved: {result['worktree_branch']}")
            if result.get("failure_detail"):
                lines.extend(["", result["failure_detail"]])
            structured = diagnostics.clean_none({
                "task_id": task_id,
                "status": result["status"],
                "message": result.get("message"),
                "error": result.get("error"),
                "worktree_path": result.get("worktree_path"),
                "worktree_branch": result.get("worktree_branch"),
                "exit_code": result.get("exit_code"),
                "exit_code_lost": result.get("exit_code_lost"),
                "status_inferred": result.get("status_inferred"),
                "model": result.get("model"),
                "failure_cause": result.get("failure_cause"),
                "failure_category": result.get("failure_category"),
                "retryable": result.get("retryable"),
                "failure_action": result.get("failure_action"),
            })
            self.send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                    "structuredContent": structured,
                    "isError": result["status"] in ("error", "not_found"),
                },
                stateless=is_stateless,
            )
            return

        self.send_error(rid, -32601, f"Unknown tool: {tool}")

    # ------------------------------------------------------------------
    # Stdio JSON-RPC main loop
    # ------------------------------------------------------------------
    def handle_threaded(self, request: Dict[str, Any]) -> None:
        rid = request.get("id")
        is_notification = ("id" not in request) or (rid is None)
        safe_rid = rid if is_valid_request_id(rid) else None
        try:
            self.handle(request)
        except Exception as exc:
            logging.error(
                "Handler error (thread): %s\n%s", exc, traceback.format_exc()
            )
            self.send_error(
                safe_rid,
                -32603,
                f"Internal error: {exc}",
                is_notification=is_notification,
            )

    def run_stdio_loop(
        self,
        *,
        on_start: Optional[Callable[[], None]] = None,
    ) -> None:
        install_sigchld_handler()
        if (
            hasattr(signal, "SIGTERM")
            and threading.current_thread() is threading.main_thread()
        ):
            def _handle_sigterm(_signum, _frame):
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, _handle_sigterm)

        if hasattr(sys.stdin, "reconfigure"):
            try:
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if self.hooks.is_windows() and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        if on_start is not None:
            on_start()

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                request: Any = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        self._emit_frame({
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32600,
                                "message": "Invalid request: expected JSON object",
                            },
                        })
                        continue
                    method = request.get("method")
                    logging.debug(
                        "Request: method=%s id=%s", method, request.get("id")
                    )

                    if isinstance(method, str) and method in BLOCKING_METHODS:
                        t = threading.Thread(
                            target=self.handle_threaded,
                            args=(request,),
                            daemon=True,
                        )
                        t.start()
                    else:
                        self.handle(request)
                except json.JSONDecodeError as exc:
                    self._emit_frame({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc}"},
                    })
                except Exception as exc:
                    is_notification = isinstance(request, dict) and (
                        ("id" not in request) or (request.get("id") is None)
                    )
                    raw_rid = (
                        request.get("id") if isinstance(request, dict) else None
                    )
                    safe_rid = raw_rid if is_valid_request_id(raw_rid) else None
                    logging.error("Handler error: %s\n%s", exc, traceback.format_exc())
                    self.send_error(
                        safe_rid,
                        -32603,
                        f"Internal error: {exc}",
                        is_notification=is_notification,
                    )
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.sync_procs.terminate_all(
                is_windows=self.hooks.is_windows(),
                win_kill_tree_fn=self.hooks.win_kill_tree,
            )

        logging.info("Server stopped")
