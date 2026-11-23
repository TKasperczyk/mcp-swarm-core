"""Incremental single-pass JSONL file cache for growing task output files.

Guards against:
- RFC 8259 raw Unicode line separators (U+2028, U+2029, U+0085) via strict
  ``split("\\n")`` rather than ``splitlines()``.
- In-place shrink/rewrite races via ``st_dev``, ``st_ino != 0``, and strict
  ``>`` size verification before and after seeking.
- Partial trailing JSONL lines via isolated ``clone_state`` views.
- Coarse filesystem mtime granularity on small files (<= ``min_bytes``).
"""

import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class JsonlParserAdapter:
    """Backend-specific JSONL parsing callbacks for ``IncrementalJsonlCache``."""

    def __init__(
        self,
        *,
        new_state: Callable[[], Dict[str, Any]],
        clone_state: Callable[[Dict[str, Any]], Dict[str, Any]],
        consume_lines: Callable[[Dict[str, Any], List[str]], None],
        finalize_views: Callable[
            [Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]
        ],
    ) -> None:
        self.new_state = new_state
        self.clone_state = clone_state
        self.consume_lines = consume_lines
        self.finalize_views = finalize_views


class IncrementalJsonlCache:
    """Thread-safe incremental JSONL parser and cache for growing stdout files."""

    def __init__(
        self,
        parser: JsonlParserAdapter,
        *,
        min_bytes: int = 8192,
        max_entries: int = 128,
    ) -> None:
        self.parser = parser
        self.min_bytes = min_bytes
        self.max_entries = max_entries
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def invalidate(self, stdout_path: Path) -> None:
        with self.lock:
            self.entries.pop(str(stdout_path), None)

    def parse_text(self, jsonl_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-pass parse of JSONL text returning ``(analysis, status)``."""
        state = self.parser.new_state()
        # Use split("\n") rather than splitlines() so legal RFC 8259 unescaped
        # Unicode line separators (\u2028, \u2029, \x85) inside JSON strings
        # never split a valid JSONL event in half.
        self.parser.consume_lines(state, jsonl_text.split("\n"))
        return self.parser.finalize_views(state)

    def parse_file(
        self,
        stdout_path: Path,
        *,
        open_fn: Optional[Callable[..., Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse a task's ``.stdout`` JSONL file in a single pass, incrementally
        reading only newly appended bytes for large growing files.
        """
        opener = open_fn or open
        try:
            st = stdout_path.stat()
        except OSError:
            self.invalidate(stdout_path)
            return self.parser.finalize_views(self.parser.new_state())

        if st.st_size <= self.min_bytes or st.st_ino == 0:
            # Evict any prior large-file entry for the same path so truncating
            # and later regrowing the same inode can never splice stale state,
            # and skip inode-based caching on filesystems reporting st_ino == 0.
            self.invalidate(stdout_path)
            try:
                text = stdout_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return self.parser.finalize_views(self.parser.new_state())
            return self.parse_text(text)

        key = str(stdout_path)
        with self.lock:
            entry = self.entries.get(key)
            if (
                entry is not None
                and entry["dev"] == st.st_dev
                and entry["ino"] == st.st_ino
                and entry["size"] == st.st_size
                and entry["mtime_ns"] == st.st_mtime_ns
            ):
                return self.parser.finalize_views(entry["view_state"])

            if (
                entry is not None
                and entry["dev"] == st.st_dev
                and entry["ino"] == st.st_ino
                and st.st_size > entry["size"]
                and entry["offset"] <= st.st_size
            ):
                try:
                    with opener(stdout_path, "rb") as f:
                        fst_before = os.fstat(f.fileno())
                        if (
                            fst_before.st_dev == entry["dev"]
                            and fst_before.st_ino == entry["ino"]
                            and fst_before.st_size > entry["size"]
                        ):
                            f.seek(entry["offset"])
                            chunk = f.read()
                            fst_after = os.fstat(f.fileno())
                            consumed_size = entry["offset"] + len(chunk)
                            if consumed_size > entry["size"]:
                                last_nl = chunk.rfind(b"\n")
                                if last_nl >= 0:
                                    complete = chunk[: last_nl + 1]
                                    tail = chunk[last_nl + 1 :]
                                    self.parser.consume_lines(
                                        entry["state"],
                                        complete.decode("utf-8", errors="replace").split("\n"),
                                    )
                                    entry["offset"] += len(complete)
                                else:
                                    tail = chunk

                                if tail.strip():
                                    view_state = self.parser.clone_state(entry["state"])
                                    self.parser.consume_lines(
                                        view_state,
                                        [tail.decode("utf-8", errors="replace")],
                                    )
                                else:
                                    view_state = entry["state"]

                                entry["size"] = consumed_size
                                entry["mtime_ns"] = fst_after.st_mtime_ns
                                entry["view_state"] = view_state
                                return self.parser.finalize_views(view_state)
                except FileNotFoundError:
                    self.entries.pop(key, None)
                    return self.parser.finalize_views(self.parser.new_state())
                except OSError:
                    return self.parser.finalize_views(entry["view_state"])

            # Full single-pass read from byte 0 (cache miss, truncation, or new inode)
            try:
                with opener(stdout_path, "rb") as f:
                    raw = f.read()
                    fst = os.fstat(f.fileno())
            except OSError:
                self.entries.pop(key, None)
                return self.parser.finalize_views(self.parser.new_state())

            state = self.parser.new_state()
            last_nl = raw.rfind(b"\n")
            if last_nl >= 0:
                complete = raw[: last_nl + 1]
                tail = raw[last_nl + 1 :]
                self.parser.consume_lines(
                    state, complete.decode("utf-8", errors="replace").split("\n")
                )
                offset = len(complete)
            else:
                tail = raw
                offset = 0

            if tail.strip():
                view_state = self.parser.clone_state(state)
                self.parser.consume_lines(
                    view_state, [tail.decode("utf-8", errors="replace")]
                )
            else:
                view_state = state

            if len(raw) > self.min_bytes and fst.st_ino != 0:
                if len(self.entries) >= self.max_entries and key not in self.entries:
                    oldest_key = next(iter(self.entries))
                    self.entries.pop(oldest_key, None)
                self.entries[key] = {
                    "dev": fst.st_dev,
                    "ino": fst.st_ino,
                    "size": len(raw),
                    "mtime_ns": fst.st_mtime_ns,
                    "offset": offset,
                    "state": state,
                    "view_state": view_state,
                }
            else:
                self.entries.pop(key, None)
            return self.parser.finalize_views(view_state)
