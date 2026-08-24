"""glob_search tool — find files matching a glob pattern."""

from __future__ import annotations

import logging
from pathlib import Path

from frontier_agent.core.tool import tool
from plugins.tools._sandbox import shell_quote as _shell_quote

logger = logging.getLogger(__name__)

_MAX_RESULTS = 200


def _local_glob(pattern: str, path: str, max_results: int) -> str:
    """Fallback: search local allowed directories using pathlib.glob."""
    from plugins.tools._path_auth import _is_path_allowed, task_input_matcher
    from plugins.tools._sandbox import is_spill_path, spill_path_matcher
    from plugins.tools.ignore_rules import discover_repo_root, should_ignore_path

    search_path = Path(path)
    if not search_path.exists():
        return f"Error: path '{path}' does not exist"

    # Security: only search allowed directories
    allowed, reason = _is_path_allowed(str(search_path))
    if not allowed:
        return f"Access denied: {reason}"

    try:
        matches = list(search_path.glob(pattern))
    except Exception as e:
        return f"Error: invalid glob pattern: {e}"

    # Filter to files only, sort by mtime (newest first)
    repo_root = discover_repo_root(search_path if search_path.is_dir() else search_path.parent)
    # Re-authorize every match, not just the search root: ``glob`` follows
    # symlinked descendants, so a link in the workspace would otherwise list
    # (and thereby disclose) names under its external target.
    # Cheap ignore test first: only a file the repo rules would drop is worth
    # the input-mount check, which resolves the candidate's real path.
    is_task_input = task_input_matcher()
    in_spill = spill_path_matcher()
    explicit_spill_search = (
        ".spill" in search_path.parts or is_spill_path(str(search_path))
    )
    files = [
        f for f in matches
        if f.is_file()
        and _is_path_allowed(str(f))[0]
        and (
            explicit_spill_search
            or (
                not in_spill(f)
                and (not should_ignore_path(f, repo_root) or is_task_input(f))
            )
        )
    ]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    if not files:
        return f"No files found matching '{pattern}' in {path}"

    lines = []
    for f in files[:max_results]:
        rel = f.relative_to(search_path) if f.is_relative_to(search_path) else f
        lines.append(str(rel))

    output = "\n".join(lines)
    if len(files) > max_results:
        output += f"\n\n... ({len(files)} total, showing first {max_results})"
    else:
        output += f"\n\n({len(files)} files found)"
    return output


@tool
async def glob_search(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
) -> str:
    """Find files matching a glob pattern, sorted by modification time.

    Use this to quickly locate files by name or extension.
    Supports ** for recursive matching.

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts", "*.md", "test_*.py").
        path: Root directory to search in (default "."). Use absolute paths for sandbox.
        max_results: Maximum number of files to return (default 100).

    Returns:
        List of matching file paths, newest first. Includes total count.
    """
    if not pattern or not pattern.strip():
        return "Error: glob pattern is required."

    pattern = pattern.strip()
    max_results = max(1, min(max_results, _MAX_RESULTS))

    from plugins.tools._sandbox import resolve_runtime_path
    path = resolve_runtime_path(path)

    from plugins.tools._path_auth import _is_path_allowed
    local_allowed, _ = _is_path_allowed(path)

    # Try E2B sandbox first when path is not an allowed local repo path
    try:
        from plugins.tools._sandbox import get_sandbox, sandbox_available
        if sandbox_available() and not local_allowed:
            sandbox = get_sandbox()

            # Use find + stat for mtime sorting, with glob pattern via -name/-path
            # Convert glob to find-compatible pattern
            # The spill store stays out of ordinary discovery: listing the
            # agent's own spilled bodies invites it to read back detail
            # compaction just removed. An explicit path INTO the store still
            # works, matching ``grep_search``'s rule. Only the recursive branch
            # needs the prune — ``-maxdepth 1`` never descends into ``.spill``.
            prune = (
                "-type d -name .spill -prune -o "
                if ".spill" not in Path(path).parts
                else ""
            )
            if "**" in pattern:
                # Recursive pattern: use find with -name on the basename
                basename = pattern.rsplit("/", 1)[-1] if "/" in pattern else pattern.replace("**/", "")
                cmd = (
                    f"find {_shell_quote(path)} {prune}-type f -name {_shell_quote(basename)} "
                    f"-printf '%T@\\t%p\\n' 2>/dev/null | sort -rn | head -n {max_results} | cut -f2"
                )
            else:
                # Simple pattern
                cmd = (
                    f"find {_shell_quote(path)} -maxdepth 1 -type f -name {_shell_quote(pattern)} "
                    f"-printf '%T@\\t%p\\n' 2>/dev/null | sort -rn | head -n {max_results} | cut -f2"
                )

            result = sandbox.commands.run(cmd, timeout=30)
            output = (result.stdout or "").strip()

            # Distinguish "found nothing" from "the command did not run".
            # `find` exits 0 even with no matches, so a non-zero code here means
            # a real failure — and since the per-exec memory cap makes "this
            # command was killed" an ordinary outcome, reporting that as
            # "No files found" would hand the model a confident wrong answer to
            # build on. grep_search already draws this distinction; this path
            # did not. See docs/WORKER_OOM_HARDENING.md (P0-4).
            if result.exit_code not in (0, None):
                err = (result.stderr or "").strip() or f"exit code {result.exit_code}"
                return f"Error: file search failed: {err}"

            if not output:
                return f"No files found matching '{pattern}' in {path}"

            file_count = output.count("\n") + 1
            output += f"\n\n({file_count} files found)"
            return output
    except Exception as e:
        logger.debug("Sandbox glob failed: %s, trying local", e)

    # Local search
    return _local_glob(pattern, path, max_results)
