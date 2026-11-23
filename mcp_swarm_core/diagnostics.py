"""Shared failure classification, status formatting, and outputSchema builders."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Accepts both legacy 8-char hex task IDs (*_async) and 32-char high-entropy
# hex task IDs (MCP Tasks extension io.modelcontextprotocol/tasks).
TASK_ID_RE = re.compile(r"[0-9a-f]{8}(?:[0-9a-f]{24})?")


def validate_task_id(task_id: Any) -> bool:
    """Reject task IDs that aren't our generated 8-char or 32-char hex format."""
    return isinstance(task_id, str) and bool(TASK_ID_RE.fullmatch(task_id))


def iso_utc(ts: float) -> str:
    """Format a POSIX timestamp as an ISO 8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Error reading {path}: {exc}]"


def clean_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy omitting keys whose values are None."""
    return {k: v for k, v in mapping.items() if v is not None}


def error_message(value: Any) -> Optional[str]:
    """Return a non-empty message from a string or structured error payload."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str):
            message = message.strip()
            return message or None
    return None


def decide_run_status(
    exit_code: Optional[int], analysis: Dict[str, Any], stderr: str
) -> Tuple[str, bool]:
    """Return ``(status, inferred)`` using terminal events, then the exit code."""
    terminal_status = analysis.get("terminal_status")
    if terminal_status in ("completed", "failed"):
        return terminal_status, False

    # Missing or contradictory terminal signals defer to the observed process.
    if exit_code is not None:
        return ("failed" if exit_code != 0 else "completed"), False

    # Exit-code-lost compatibility fallback. Agent output is useful evidence of
    # completion; stderr is only a weak, explicitly inferred signal.
    if analysis.get("last_assistant_text"):
        return "completed", True
    if stderr.strip():
        return "failed", True
    return "completed", True


def classify_failure(
    message: str,
    *,
    product_name: str = "CLI",
) -> Dict[str, Any]:
    """Conservatively classify a final failure; retryability is information only."""
    lower = message.lower()
    category = "unknown"
    retryable: Optional[bool] = None
    action = "Inspect the structured cause and diagnostics before retrying."

    if any(
        term in lower
        for term in (
            "refresh token",
            "token has been revoked",
            "revoked token",
            "unauthorized",
            "unauthenticated",
            "authentication failed",
            "not authenticated",
            "not logged in",
            "login required",
        )
    ) or re.search(r"\b401\b", lower):
        category, retryable = "authentication", False
        action = f"Reauthenticate {product_name} before starting another run."
    elif any(
        term in lower
        for term in (
            "at capacity",
            "model capacity",
            "high demand",
            "provider capacity",
            "server overloaded",
            "service overloaded",
        )
    ):
        category, retryable = "provider_capacity", True
        action = (
            "Retry later, or ask the server operator to change the configured model."
        )
    elif any(
        term in lower
        for term in (
            "rate limit",
            "rate-limit",
            "too many requests",
            "quota temporarily",
            "resource_exhausted",
            "quota exceeded",
        )
    ) or re.search(r"\b429\b", lower):
        category, retryable = "rate_limit", True
        action = "Retry after the provider limit resets."
    elif any(
        term in lower
        for term in (
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
        )
    ) or re.search(
        r"\b(?:500|502|503|504)\b|(?:http|status|error)[^0-9\n]{0,12}\b5\d\d\b",
        lower,
    ):
        category, retryable = "provider_5xx", True
        action = "Retry later; the provider or an upstream gateway failed."
    elif any(
        term in lower
        for term in (
            "stream disconnected",
            "connection reset",
            "connection failed",
            "network error",
            "timed out",
            "timeout",
        )
    ):
        category, retryable = "transport", True
        action = "Retry after checking network and provider availability."
    elif any(
        term in lower
        for term in (
            "invalid request",
            "bad request",
            "invalid prompt",
            "unsupported",
            "model not found",
            "unknown model",
            "invalid model selection",
            "not recognized as a known model",
        )
    ) or re.search(r"\b400\b", lower):
        category, retryable = "invalid_request", False
        action = "Correct the request or server configuration before retrying."
    elif any(
        term in lower
        for term in (
            "panicked at",
            "panic",
            "segmentation fault",
            "fatal runtime error",
            "agent loop died unexpectedly",
            "signal 11",
        )
    ):
        category = "process_crash"
        action = "Inspect the diagnostics; retryability is unknown."

    return {"category": category, "retryable": retryable, "action": action}


def build_failure_info(
    stdout: str,
    stderr: str,
    *,
    analysis: Dict[str, Any],
    product_name: str = "CLI",
    command: Any = None,
    model: Optional[str] = None,
    model_from_command_fn: Optional[Callable[[Any], Optional[str]]] = None,
    classify_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a clean primary cause plus explicitly secondary diagnostics."""
    cause = analysis.get("terminal_error")
    if not cause and analysis.get("errors"):
        cause = analysis["errors"][-1]

    raw_stdout = stdout.strip()
    stdout_diagnostics = None
    if not cause and raw_stdout and not analysis.get("saw_json"):
        cause = raw_stdout
    elif not cause and raw_stdout:
        stdout_diagnostics = raw_stdout[-1000:]

    fallback_prefix = f"{product_name} exited without"
    if not cause:
        cause = f"{fallback_prefix} a structured failure message."

    classification_input = cause
    if classification_input.startswith(fallback_prefix) and stderr.strip():
        classification_input = f"{classification_input}\n{stderr}"
    classifier = classify_fn or (
        lambda msg: classify_failure(msg, product_name=product_name)
    )
    classification = classifier(classification_input)

    resolved_model = (
        model
        or analysis.get("model")
        or (model_from_command_fn(command) if model_from_command_fn else None)
    )

    info: Dict[str, Any] = {
        "cause": cause,
        "model": resolved_model,
        **classification,
    }
    if stdout_diagnostics:
        info["stdout_diagnostics"] = stdout_diagnostics
    if stderr.strip():
        info["stderr_diagnostics"] = stderr.strip()[-500:]
    return info


def format_failure_info(
    info: Dict[str, Any],
    include_diagnostics: bool = True,
    *,
    product_name: str = "CLI",
) -> str:
    retryable = info.get("retryable")
    retryable_text = (
        "yes" if retryable is True else "no" if retryable is False else "unknown"
    )
    lines = [
        f"Model: {info.get('model') or 'unknown'}",
        f"Cause: {info.get('cause', f'Unknown {product_name} failure')}",
        f"Category: {info.get('category', 'unknown')}",
        f"Retryable: {retryable_text} (informational only; the wrapper did not retry)",
    ]
    if info.get("action"):
        lines.append(f"Action: {info['action']}")
    if include_diagnostics and info.get("stdout_diagnostics"):
        lines.extend([
            "",
            "Output (stdout; no structured cause was found):",
            info["stdout_diagnostics"],
        ])
    if include_diagnostics and info.get("stderr_diagnostics"):
        lines.extend([
            "",
            "Diagnostics (stderr tail; not the failure cause):",
            info["stderr_diagnostics"],
        ])
    return "\n".join(lines)


def failure_structured(failure: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not failure:
        return None
    return clean_none({
        "model": failure.get("model"),
        "cause": failure.get("cause"),
        "category": failure.get("category"),
        "retryable": failure.get("retryable"),
        "action": failure.get("action"),
        "stdout_diagnostics": failure.get("stdout_diagnostics"),
        "stderr_diagnostics": failure.get("stderr_diagnostics"),
    })


def format_task_activity_brief(jsonl_status: Dict[str, Any]) -> Optional[str]:
    """Return a short one-line activity summary from JSONL status, or None."""
    tools_called = jsonl_status.get("tools_called", 0)
    last_tool = jsonl_status.get("last_tool")
    last_tool_args = jsonl_status.get("last_tool_args")
    phase = jsonl_status.get("phase", "starting")

    if last_tool:
        tool_str = last_tool
        if last_tool_args:
            tool_str += f"({last_tool_args[:60]})"
        noun = "tool" if tools_called == 1 else "tools"
        return f"{tools_called} {noun} called, last: {tool_str}"
    if tools_called:
        noun = "tool" if tools_called == 1 else "tools"
        return f"{tools_called} {noun} called"
    if jsonl_status.get("last_assistant_text"):
        return jsonl_status["last_assistant_text"][:100].replace("\n", " ")
    if jsonl_status.get("last_reasoning"):
        return jsonl_status["last_reasoning"][:100].replace("\n", " ")
    if phase and phase != "starting":
        return phase
    return None


def format_mcp_task_status_message(
    jsonl_status: Dict[str, Any],
    elapsed: int,
    *,
    product_name: str = "Session",
) -> str:
    """Build a human-readable ``statusMessage`` for ``tasks/get`` while working."""
    brief = format_task_activity_brief(jsonl_status)
    if brief:
        msg = f"Running ({elapsed}s): {brief}"
    else:
        msg = f"Starting {product_name} session ({elapsed}s elapsed)"
    if jsonl_status.get("last_error"):
        msg += f" [non-terminal error: {jsonl_status['last_error'][:120]}]"
    return msg


def make_inferred_hints(
    product_name: str,
    wait_tool_name: str,
    process_check_cmd: str,
) -> Dict[str, str]:
    """Build consistent inferred-verdict and wait-timeout hint strings."""
    return {
        "unverified_failure_hint": (
            "NOTE: this verdict is inferred, not observed -- the exit code was lost to "
            f"a reaping race, so the status was derived from output alone. {product_name} may "
            "have finished its work. Verify before reporting failure upstream: run "
            f"{process_check_cmd} to see whether it is still alive, then check "
            "file mtimes, the worktree branch, and your own build/tests."
        ),
        "unverified_completion_hint": (
            "NOTE: this completion verdict is inferred, not observed -- neither an "
            "unambiguous terminal event nor the process exit code was available. "
            "Verify the returned output and your own build/tests before relying on it."
        ),
        "still_running_hint": (
            "This is NOT a failure. The task was not killed and is still working. Call "
            f"{wait_tool_name} again with the same task_id to keep waiting, or check "
            f"{process_check_cmd}."
        ),
    }


FAILURE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "cause": {"type": "string"},
        "category": {"type": "string"},
        "retryable": {"type": "boolean"},
        "action": {"type": "string"},
        "stdout_diagnostics": {"type": "string"},
        "stderr_diagnostics": {"type": "string"},
    },
    "required": ["cause", "category"],
}


def make_sync_output_schema(
    session_field_name: str = "thread_id",
    extra_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "status": {"type": "string", "enum": ["completed", "failed", "cancelled"]},
        "text": {"type": "string"},
        session_field_name: {"type": "string"},
        "model": {"type": "string"},
        "exit_code": {"type": "integer"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "partial_output": {"type": "string"},
        "failure": FAILURE_SCHEMA,
    }
    if extra_properties:
        props.update(extra_properties)
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "text"],
    }


def make_async_output_schema(
    extra_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "status": {"type": "string", "enum": ["started", "error"]},
        "task_id": {"type": "string"},
        "model": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "error": {"type": "string"},
    }
    if extra_properties:
        props.update(extra_properties)
    return {
        "type": "object",
        "properties": props,
        "required": ["status"],
    }


def make_reply_output_schema(
    session_field_name: str = "thread_id",
    extra_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "status": {"type": "string", "enum": ["completed", "failed", "cancelled"]},
        "text": {"type": "string"},
        session_field_name: {"type": "string"},
        "model": {"type": "string"},
        "exit_code": {"type": "integer"},
        "partial_output": {"type": "string"},
        "failure": FAILURE_SCHEMA,
    }
    if extra_properties:
        props.update(extra_properties)
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "text"],
    }


def make_status_output_schema(
    session_field_name: str = "thread_id",
    extra_task_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item_props: Dict[str, Any] = {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "elapsed_seconds": {"type": "integer"},
        "phase": {"type": "string"},
        "tools_called": {"type": "integer"},
        session_field_name: {"type": "string"},
        "model": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "last_tool": {"type": "string"},
        "last_tool_args": {"type": "string"},
        "last_reasoning": {"type": "string"},
        "last_assistant_text": {"type": "string"},
        "last_warning": {"type": "string"},
        "last_error": {"type": "string"},
        "exit_code": {"type": "integer"},
        "exit_code_lost": {"type": "boolean"},
        "status_inferred": {"type": "boolean"},
        "terminal_conflict": {"type": "boolean"},
        "failure": FAILURE_SCHEMA,
        "error": {"type": "string"},
    }
    if extra_task_properties:
        item_props.update(extra_task_properties)
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": ["task_id", "status"],
                },
            },
        },
        "required": ["tasks"],
    }


def make_wait_output_schema(
    session_field_name: str = "thread_id",
    extra_task_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item_props: Dict[str, Any] = {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "elapsed_seconds": {"type": "integer"},
        "result": {"type": "string"},
        session_field_name: {"type": "string"},
        "model": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "exit_code": {"type": "integer"},
        "exit_code_lost": {"type": "boolean"},
        "status_inferred": {"type": "boolean"},
        "failure_cause": {"type": "string"},
        "failure_category": {"type": "string"},
        "retryable": {"type": "boolean"},
        "failure_action": {"type": "string"},
        "partial_output": {"type": "string"},
        "stdout_diagnostics": {"type": "string"},
        "stderr_diagnostics": {"type": "string"},
        "error": {"type": "string"},
    }
    if extra_task_properties:
        item_props.update(extra_task_properties)
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": ["task_id", "status"],
                },
            },
        },
        "required": ["tasks"],
    }


def make_cancel_output_schema(
    extra_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "message": {"type": "string"},
        "error": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "exit_code": {"type": "integer"},
        "exit_code_lost": {"type": "boolean"},
        "status_inferred": {"type": "boolean"},
        "model": {"type": "string"},
        "failure_cause": {"type": "string"},
        "failure_category": {"type": "string"},
        "retryable": {"type": "boolean"},
        "failure_action": {"type": "string"},
    }
    if extra_properties:
        props.update(extra_properties)
    return {
        "type": "object",
        "properties": props,
        "required": ["task_id", "status"],
    }
