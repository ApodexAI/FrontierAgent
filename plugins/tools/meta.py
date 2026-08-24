"""ToolMeta — metadata annotations for tools."""

from __future__ import annotations

from dataclasses import dataclass

from frontier_agent.infra.config import get_config

# Inline cap for exec/compute tool results before overflow-to-disk. Configurable
# via TOOL_EXEC_RESULT_MAX_CHARS (default 8K); shared by bash / run_python_code /
# the finance sandbox runners so the budget is tuned in one place.
_EXEC_RESULT_MAX_CHARS = get_config().tool_exec_result_max_chars


@dataclass(frozen=True)
class ToolMeta:
    """Metadata for a tool instance.

    Attributes:
        is_read_only: Tool only reads data, never modifies state.
        is_destructive: Tool can cause irreversible changes (file deletion, etc.).
        concurrency_safe: Safe to run in parallel with other tools.
        timeout: Maximum execution time in seconds.
        category: Logical grouping (web, file, compute, search, vision).
        max_result_chars: Max characters to keep inline. Overflow saved to disk.
            0 means no limit (default for most tools).
        result_is_ranked: The result is ordered by RELEVANCE, best first, so a
            contiguous head is worth more than a head-plus-tail split — the tail
            of a ranked list is its worst entries. False (the default) means the
            result is sequential: a transcript, a document, a file listing, an
            exec log, where the end carries the verdict and is worth keeping.
            Read by ``_overflow`` when TOOL_RESULT_TRUNCATION=auto.
    """

    is_read_only: bool = True
    is_destructive: bool = False
    concurrency_safe: bool = True
    timeout: int = 45
    category: str = ""
    max_result_chars: int = 0
    result_is_ranked: bool = False


# ── Default metadata for built-in tools ──────────────────────────────────

TOOL_META: dict[str, ToolMeta] = {
    # Web tools — read-only, concurrent-safe, longer timeout for network.
    # max_result_chars=0 (no per-tool overflow): content density rewards
    # full bodies (academic papers bury results 30K+ chars in), and the
    # global TOOL_RESULT_MAX_CHARS + ReasoningStripCompactor backstop
    # bound total context. Compaction ages old fetches to URL stubs so
    # the model can re-fetch with info_to_extract if needed.
    "web_search": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="web", result_is_ranked=True),
    "web_fetch": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=60, category="web"),
    "scholar_search": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="web", result_is_ranked=True),

    # File read tools — read-only, concurrent-safe. max_result_chars=0
    # for the same reason as the web tools: source files / long markdown
    # / downloaded papers carry their signal across the whole body, and
    # the global TOOL_RESULT_MAX_CHARS + the workflow compactor bound
    # total context. file_editor_view also accepts a view_range arg the
    # agent can use when it knows what slice it wants.
    # read_file = structured/rich reader (sandbox-backed, may pip-install parsers
    # + recalc spreadsheets), so a longer timeout than the plain-text read_text.
    "read_file": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=120, category="file"),
    # Reads this run's own transcript in-process — no sandbox, no network, so the
    # timeout is a scan of an ~1 MB file. ``max_result_chars=0``: the tool caps and
    # paginates its own slice, and a per-tool overflow on top would truncate the
    # very content the call exists to recover.
    "recover_result": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="file"),
    "read_text": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=15, category="file"),
    "file_editor_view": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=15, category="file"),

    # File write tools — NOT read-only, NOT concurrent-safe (may conflict)
    "create_file": ToolMeta(
        is_read_only=False,
        concurrency_safe=False,
        timeout=120,
        category="file",
    ),
    "write_file": ToolMeta(is_read_only=False, concurrency_safe=False, timeout=15, category="file"),
    "file_editor_create": ToolMeta(is_read_only=False, concurrency_safe=False, timeout=15, category="file"),
    "file_editor_str_replace": ToolMeta(is_read_only=False, concurrency_safe=False, timeout=15, category="file"),

    # Bash — NOT read-only, NOT concurrent-safe, potentially destructive
    "bash": ToolMeta(is_read_only=False, is_destructive=True, concurrency_safe=False, timeout=60, category="compute", max_result_chars=_EXEC_RESULT_MAX_CHARS),

    # Vision — read-only, concurrent-safe
    "view_image": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="vision"),

    # Search tools — read-only, concurrent-safe
    "grep_search": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="search", max_result_chars=8_000),
    "glob_search": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=30, category="search", max_result_chars=8_000),

    # Delegation — spawns sub-agents, long-running, NOT concurrent-safe (shared evidence pool)
    "delegate_subtask": ToolMeta(is_read_only=False, concurrency_safe=False, timeout=1900, category="orchestration"),

    # Python sandbox — isolated code execution, concurrent-safe (each call is independent)
    "run_python_code": ToolMeta(
        is_read_only=False,
        is_destructive=False,
        concurrency_safe=True,
        timeout=120,
        category="compute",
        max_result_chars=_EXEC_RESULT_MAX_CHARS,
    ),

    # Meta — tool discovery
    "tool_search": ToolMeta(is_read_only=True, concurrency_safe=True, timeout=5, category="meta"),

    # Finance sandbox
    "create_finance_sandbox": ToolMeta(
        is_read_only=False,
        concurrency_safe=False,
        timeout=120,
        category="finance",
        max_result_chars=4_000,
    ),
    "run_command_in_finance_sandbox": ToolMeta(
        is_read_only=False,
        concurrency_safe=False,
        timeout=600,
        category="finance",
        max_result_chars=_EXEC_RESULT_MAX_CHARS,
    ),
    "run_python_code_in_finance_sandbox": ToolMeta(
        is_read_only=False,
        concurrency_safe=False,
        timeout=600,
        category="finance",
        max_result_chars=_EXEC_RESULT_MAX_CHARS,
    ),

}


def get_tool_meta(tool_name: str) -> ToolMeta:
    """Get metadata for a tool. Returns safe defaults if not found."""
    return TOOL_META.get(tool_name, ToolMeta())
