"""Git worktree isolation lifecycle helpers for parallel MCP swarm tasks."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .platform import IS_WINDOWS, TEXT_KW, popen_kw, safe_which

_GIT_TIMEOUT_S = 60
_git_bin_cache: Optional[str] = None


def _git_bin() -> str:
    """Resolve ``git`` safely (avoiding Windows cwd binary hijacking)."""
    global _git_bin_cache
    if _git_bin_cache is not None:
        return _git_bin_cache
    if IS_WINDOWS:
        found = safe_which("git", is_windows=True)
        if not found:
            raise RuntimeError("git.exe not found on PATH")
        _git_bin_cache = found
        return found
    _git_bin_cache = "git"
    return "git"


def create_worktree(
    run_id: str,
    base_cwd: Optional[str],
    *,
    worktree_base_dir: Path,
    branch_prefix: str = "mcp-swarm",
    popen_kw_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Tuple[str, str, str]:
    """Create an isolated ``HEAD``-based git worktree.

    Returns ``(worktree_root, target_cwd, branch_name)``.
    Uncommitted and untracked changes in the source working tree are excluded.
    """
    pkw = popen_kw_fn or popen_kw
    git = _git_bin()
    branch_name = f"{branch_prefix}/{run_id}"
    worktree_base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    worktree_root = worktree_base_dir / run_id

    if base_cwd:
        base_path = Path(base_cwd)
        effective_cwd = (
            base_path if base_path.is_absolute() else Path.cwd() / base_path
        )
    else:
        effective_cwd = Path.cwd()
    effective_cwd = effective_cwd.resolve()

    try:
        proc = subprocess.run(
            [git, "-C", str(effective_cwd), "rev-parse", "--show-toplevel"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_S,
            **TEXT_KW,
            **pkw(),
            check=True,
        )
        repo_root = Path(proc.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)).strip()
        raise RuntimeError(
            f"Failed to find git repo for {effective_cwd}: {detail}"
        ) from exc

    try:
        rel_subdir = effective_cwd.relative_to(repo_root)
    except ValueError:
        rel_subdir = Path(".")

    try:
        subprocess.run(
            [
                git,
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_root),
                "HEAD",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_S,
            **TEXT_KW,
            **pkw(),
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)).strip()
        raise RuntimeError(
            f"Failed to create git worktree {worktree_root}: {detail}"
        ) from exc

    target_cwd = (worktree_root / rel_subdir).resolve()
    logging.info(
        "Created worktree %s (branch %s) for run %s",
        worktree_root,
        branch_name,
        run_id,
    )
    return str(worktree_root), str(target_cwd), branch_name


def remove_worktree(
    worktree_path: str,
    branch_name: str,
    *,
    worktree_base_dir: Optional[Path] = None,
    popen_kw_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> bool:
    """Remove a worktree and its branch. Returns True if cleanup succeeded.

    Refuses to remove an enclosing repository or worktree outside
    ``worktree_base_dir`` (when provided) or above ``worktree_path``.
    """
    pkw = popen_kw_fn or popen_kw
    git = _git_bin()
    worktree_input = Path(worktree_path)
    if not worktree_input.exists():
        logging.debug("Worktree path already removed: %s", worktree_input)
        return True

    resolved_input = worktree_input.resolve()
    if worktree_base_dir is not None:
        resolved_base = worktree_base_dir.resolve()
        try:
            resolved_input.relative_to(resolved_base)
        except ValueError:
            logging.warning(
                "Refusing to remove worktree outside %s: %s",
                resolved_base,
                resolved_input,
            )
            return False

    worktree_root = resolved_input
    repo_root: Optional[Path] = None

    try:
        proc = subprocess.run(
            [git, "-C", str(worktree_input), "rev-parse", "--show-toplevel"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_S,
            **TEXT_KW,
            **pkw(),
            check=True,
        )
        detected_root = Path(proc.stdout.strip()).resolve()
        if worktree_base_dir is not None:
            resolved_base = worktree_base_dir.resolve()
            try:
                detected_root.relative_to(resolved_base)
            except ValueError:
                logging.warning(
                    "Refusing to remove enclosing git toplevel %s outside %s (for input %s)",
                    detected_root,
                    resolved_base,
                    resolved_input,
                )
                return False
        worktree_root = detected_root
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)).strip()
        logging.debug(
            "Unable to resolve worktree root for %s: %s", worktree_input, detail
        )

    try:
        proc = subprocess.run(
            [git, "-C", str(worktree_input), "rev-parse", "--git-common-dir"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_S,
            **TEXT_KW,
            **pkw(),
            check=True,
        )
        common_dir = Path(proc.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (worktree_input / common_dir).resolve()
        if common_dir.name == ".git":
            repo_root = common_dir.parent
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)).strip()
        logging.debug(
            "Unable to resolve git common dir for %s: %s", worktree_input, detail
        )

    if worktree_root.exists():
        remove_cwd = repo_root or worktree_root
        try:
            subprocess.run(
                [
                    git,
                    "-C",
                    str(remove_cwd),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_GIT_TIMEOUT_S,
                **TEXT_KW,
                **pkw(),
                check=True,
            )
            logging.info("Removed worktree %s", worktree_root)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            lowered = detail.lower()
            if not any(
                token in lowered
                for token in (
                    "not a git repository",
                    "working tree not found",
                    "is not a working tree",
                    "does not exist",
                    "no such file",
                )
            ):
                logging.warning(
                    "Failed to remove worktree %s: %s", worktree_root, detail
                )
                return False
        except subprocess.TimeoutExpired as exc:
            logging.warning("Timed out removing worktree %s: %s", worktree_root, exc)
            return False

    if branch_name and repo_root:
        try:
            subprocess.run(
                [git, "-C", str(repo_root), "branch", "-D", branch_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_GIT_TIMEOUT_S,
                **TEXT_KW,
                **pkw(),
                check=True,
            )
            logging.info("Deleted worktree branch %s", branch_name)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)).strip()
            logging.debug(
                "Ignoring worktree branch delete failure for %s: %s",
                branch_name,
                detail,
            )

    return True
