# mcp-swarm-core

Shared, zero-dependency (Python stdlib only) infrastructure for building parallel MCP CLI swarm servers such as [codex-mcp-swarm](https://github.com/TKasperczyk/codex-mcp-swarm) and [antigravity-mcp-swarm](https://github.com/TKasperczyk/antigravity-mcp-swarm).

## What it provides

- **MCP Protocol Engine (`mcp_swarm_core.protocol`)**
  - Stateless MCP `2026-07-28` (`server/discover`, per-request `_meta["io.modelcontextprotocol/protocolVersion"]` and `_meta["io.modelcontextprotocol/clientCapabilities"]`, `resultType` discriminators, `-32022` unsupported-version rejection)
  - Official MCP Tasks extension (`io.modelcontextprotocol/tasks`, SEP-2663: `CreateTaskResult`, `tasks/get`, `tasks/cancel`, tool-level `isError: true` under `status: "completed"`)
  - Backward-compatible stateful MCP (`2024-11-05` through `2025-11-25`: `initialize`, `instructions`, `ping`, tool `annotations`, `outputSchema` + `structuredContent`, enriched `notifications/progress`, silent unknown notifications per JSON-RPC 2.0 §4.1)
  - Multi-threaded stdio dispatcher with bounded request cancellation tracking and off-thread sync child termination
- **Cross-Platform Process & Tree Management (`mcp_swarm_core.platform`)**
  - POSIX process-group isolation (`start_new_session=True`), group signalling (`os.killpg`), grandchild sweep, and Linux `/proc` PID-reuse + zombie detection
  - Native Windows Win32 `ctypes` process liveness, creation-time PID-reuse protection, `taskkill.exe /T /F /PID` process-tree termination, and `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`
  - Safe stdin payload transport (`prompt_stdin`): unlinked `TemporaryFile` on POSIX and regular `.stdin` files (no `O_TEMPORARY`) on Windows
- **Disk-Backed Task Store & Lifecycle (`mcp_swarm_core.task_store`)**
  - Strictly lock-free POSIX `SIGCHLD` async reaper (no logging or locks inside the signal handler; safe against PID pop/resurrection races)
  - Atomic metadata persistence (`mkstemp` + `os.replace` with Windows retry)
  - Per-task finalization locking, automatic expiry cleanup, batch wait with client auto-backgrounding floor, and sync child tracking
- **Git Worktree Isolation (`mcp_swarm_core.worktree`)**
  - Isolated `HEAD`-based git worktrees (`<prefix>/<id>`) with safe cleanup and branch deletion
- **Incremental Single-Pass JSONL Cache (`mcp_swarm_core.jsonl_cache`)**
  - Parses growing task `.stdout` files incrementally for files above 8 KiB
  - RFC 8259 Unicode line-separator safety (`split("\n")`, never `splitlines()`)
  - `st_dev` + `st_ino != 0` + strict `>` shrink-race guards and isolated partial-tail state cloning
- **Shared Diagnostics & Schema Helpers (`mcp_swarm_core.diagnostics`)**
  - Terminal-event precedence, conservative failure classification, structured failure formatting, live activity briefs, and JSON `outputSchema` builders

## Requirements

- Python 3.8+
- Zero third-party dependencies (stdlib only)

## License

MIT
