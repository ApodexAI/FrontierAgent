"""Tool result overflow — persist large results to disk, return summary inline."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

from plugins.tools.meta import get_tool_meta

logger = logging.getLogger(__name__)

# The store's physical root comes from ``_sandbox.spill_root()``; see there for
# why it lives outside every root the agent can write. What is left here is the
# per-conversation partitioning under it.
_RUN_SUBDIR = "spill"
_SPILL_SEPARATOR = "\n---\n\n"
# Backends whose commands always run on this process's own filesystem, so the
# physical path IS the path a model command can name. Container mode normally
# does too, but may opt into an inner bwrap jail; that case is resolved at run
# time in :func:`_overflow_dir`.
_SAME_FILESYSTEM_BACKENDS = frozenset({"native"})
# Backends whose commands run on another machine entirely. Nothing on this
# filesystem is nameable there, so spill advertises no path and the footer says
# the remainder is unreadable rather than pointing somewhere that cannot resolve.
_REMOTE_BACKENDS = frozenset({"e2b"})
#: Stores this process created, so a discarded session can drop exactly its own
#: recovery files. Replaces walking a directory tree looking for them: we delete
#: only paths we made, which is why this needs no symlink or filesystem-root
#: defence — the previous implementation walked a tree inside the agent's own
#: workspace and had to assume it was hostile.
_created_stores: set[Path] = set()


def _scope_component(task_id: str) -> str:
    """Map an arbitrary task id to one safe, stable directory component."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16] if task_id else ""


def _current_task_id() -> str:
    from frontier_agent.core.execution_context import get_current_execution_scope

    scope = get_current_execution_scope()
    if scope is None:
        return ""
    task_id = str(scope.task_id or "")
    session_id = str(scope.metadata.get("llm_session_id") or "")
    # This is a per-conversation overflow cache, not cross-session memory.
    return f"{task_id}:{session_id}" if session_id else task_id


def _resolved_backend() -> str:
    """The active backend as ``_sandbox`` resolves it, or "" when unresolvable.

    Reading ``SANDBOX_BACKEND`` straight from the environment misses a backend
    supplied only through ``config.yaml`` — the resolver consults ``get_config()``
    for exactly that case — and would then advertise ``/workspace/.spill/...``
    for a run whose commands execute directly on the host filesystem, where that
    literal path may name an unrelated directory. A misconfigured backend must
    not take spill down with it: spill is a diagnostic aid, so fall back to the
    conservative canonical mount rather than raising.
    """
    from plugins.tools._sandbox import _get_sandbox_backend

    try:
        return _get_sandbox_backend()
    except Exception:
        return ""


def _overflow_dir(task_id: str = "", *, create: bool = True) -> tuple[Path, str]:
    """Return the physical write directory and the path visible to the agent.

    ``create=False`` resolves the same location without touching the filesystem,
    for callers that only want to inspect or remove an existing store.

    This used to branch four ways over the workspace, the configured mount dir,
    a run directory and a legacy host-only path, each with its own rule for what
    the agent could name. The store now has one root outside every write root, so
    the only remaining question is how a model command reaches it: directly under
    ``native`` and ordinary ``container`` mode, through the read-only ``/spill``
    mount under bwrap, and not at all from a remote backend.
    """
    from plugins.tools._sandbox import _DEFAULT_SPILL_DIR, spill_root

    scope = _scope_component(task_id)
    target = spill_root() / scope if scope else spill_root()
    if create:
        target.mkdir(parents=True, exist_ok=True)
        # Readable and traversable by others, writable only by the harness. Set
        # explicitly because this is the WHOLE enforcement under ``container``,
        # where model commands are dropped to an unprivileged uid: they may read
        # a 0644 spill file but cannot create or unlink inside a directory they
        # do not own. Leaving it to the ambient umask would make that guarantee
        # depend on whoever launched the process. bwrap gets a read-only mount
        # instead (uid 0 in a user namespace ignores DAC); ``native`` has no
        # isolation to enforce anything with.
        with contextlib.suppress(OSError):
            target.chmod(0o755)
        _created_stores.add(target)

    backend = _resolved_backend()
    if backend in _REMOTE_BACKENDS:
        return target, ""
    if backend in _SAME_FILESYSTEM_BACKENDS:
        return target, str(target)
    if backend == "container":
        # Production CurrentSandbox explicitly runs without a mount namespace;
        # advertising /spill there points at nothing because the physical store
        # is normally under /tmp or the run directory. Only the optional inner
        # bwrap path creates the canonical read-only /spill mount.
        from plugins.tools._sandbox import container_uses_inner_bwrap

        if not container_uses_inner_bwrap():
            return target, str(target)
    return target, f"{_DEFAULT_SPILL_DIR}/{scope}" if scope else _DEFAULT_SPILL_DIR


def body_names_a_spill_file(body: str) -> bool:
    """Whether *body* already carries a spill pointer the agent can act on.

    A presence test against the two roots the store can be named by — the
    canonical mount and the physical path — not a parse of the pointer's prose.
    The wording differs per backend and per caller, and ``7cf9188`` moved
    deliberately away from recognising spill refs by shape.

    Deliberately does NOT go through :func:`agent_visible_spill_dir`, which
    resolves via ``_overflow_dir`` and would CREATE the store as a side effect of
    asking a read-only question.

    Exists so the site-3 recovery footer can stay quiet when it would be
    redundant. Measured on a live agent-team run: every result site 3 shortened
    was a ``bash`` result that already carried a spill pointer, and the spill file
    behind it held the FULL pre-gate-① output — 42,770 chars against the 8,000 the
    model saw. The agent read those files with ``cat`` and never called the tool
    the footer named. Two routes to the same bytes, and the footer lost.
    """
    if not body:
        return False
    from plugins.tools._sandbox import _DEFAULT_SPILL_DIR, spill_root

    roots = [_DEFAULT_SPILL_DIR]
    with contextlib.suppress(Exception):
        roots.append(str(spill_root()))
    # The separator is not cosmetic: a bare ``"/spill" in body`` also fires on
    # ``/spillover``, and a pointer always names a FILE under the store, so the
    # trailing slash is both stricter and exactly what a real pointer contains.
    return any(
        root and f"{root.rstrip('/')}/" in body for root in roots
    )


def agent_visible_spill_dir() -> str:
    """Return the spill directory as tools should name it, or empty if unreadable."""
    return _overflow_dir(_current_task_id())[1]


# Smallest inline preview worth keeping. A cap tighter than
# ``footer + _MIN_PREVIEW_CHARS`` is honoured only approximately: the pointer is
# worth more than the last few hundred characters of body.
_MIN_PREVIEW_CHARS = 500
# Below this, splitting a budget in two leaves two useless slivers.
_MIN_SIDE_CHARS = 200
# Bodies smaller than this are not worth a recovery file of their own.
_SPILL_MIN_CHARS = 1_500


def _truncation_mode(tool_name: str = "") -> str:
    """The preview shape for one tool: ``middle`` (head AND tail) or ``head``.

    Read per call rather than captured at import so an A/B run can flip arms
    through ``TOOL_RESULT_TRUNCATION`` without a rebuild. A malformed value
    falls back to ``middle`` — this is output shaping, not a place to fail.

    ``auto`` decides per tool from ``ToolMeta.result_is_ranked``, because the two
    shapes are not competing for the same kind of output. An exec log states its
    verdict last, so cutting the middle keeps it. A relevance-ranked search
    result is the opposite: its tail is its worst entries, and splitting the
    budget spends half of it on them instead of on more good hits. The first live
    A/B was run on a search benchmark and found no accuracy difference in either
    direction, which is consistent with the two effects cancelling — see
    docs/tool-result-truncation-ab.md.
    """
    from frontier_agent.infra.config import get_config

    try:
        mode = str(get_config().tool_result_truncation).strip().lower()
    except Exception:
        return "middle"
    if mode == "auto":
        if not tool_name:
            return "middle"
        return "head" if get_tool_meta(tool_name).result_is_ranked else "middle"
    return mode if mode in {"middle", "head"} else "middle"


def _elision(removed: int) -> str:
    return f"\n… {removed:,} chars elided …\n"


def _head_end(text: str, budget: int) -> int:
    """Where a head slice of at most ``budget`` chars ends, snapped to a line.

    Snapping back past the halfway point would throw away more than it buys, so
    a single line longer than half the budget is cut mid-line instead.
    """
    if budget >= len(text):
        return len(text)
    newline = text.rfind("\n", budget // 2, budget)
    return newline if newline > 0 else budget


def _tail_start(text: str, budget: int) -> int:
    """Where a tail slice of at most ``budget`` chars starts, snapped to a line.

    Snapping FORWARD (dropping the partial first line) rather than back, so the
    slice never exceeds ``budget`` and never opens mid-token.
    """
    start = max(len(text) - budget, 0)
    if start == 0:
        return 0
    newline = text.find("\n", start, start + max(budget // 2, 1))
    return newline + 1 if newline != -1 else start


def truncate_preview(text: str, budget: int, *, tool_name: str = "") -> str:
    """Cut ``text`` to at most ``budget`` chars, keeping the head AND the tail.

    A head-only cut is the wrong default for tool output: a pytest run states
    its verdict in the last ten lines, a compiler in the last error, a script in
    its exit status. Keeping only the head hides precisely the part the model
    called the tool for, and costs an extra recovery round-trip to get it back.
    Both codex (``truncate_middle_with_token_budget``) and the shape used here
    split the budget evenly and name the gap.

    The marker is sized against an upper bound on the elided count before the
    split, so the assembled preview is never longer than ``budget``.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if _truncation_mode(tool_name) == "head":
        return text[:_head_end(text, budget)]

    room = budget - len(_elision(len(text)))
    if room < 2 * _MIN_SIDE_CHARS:
        return text[:_head_end(text, budget)]
    head_end = _head_end(text, room // 2)
    tail_start = _tail_start(text, room - room // 2)
    if tail_start <= head_end:
        return text[:_head_end(text, budget)]
    return text[:head_end] + _elision(tail_start - head_end) + text[tail_start:]


def budgeted_preview(
    body: str,
    *,
    cap: int,
    ref: str,
    full_len: int | None = None,
    note: str = "",
    tool_name: str = "",
) -> str:
    """Preview plus recovery pointer, together within ``cap``.

    The footer is measured FIRST and charged against the preview budget. Adding
    it afterwards — as every call site used to — makes a tool that advertises an
    8K cap return 8K plus a few hundred characters, on every overflowing call,
    for exactly the results that are already the largest in the turn.
    """
    total = len(body) if full_len is None else full_len
    footer = _spill_footer(ref, full_len=total, note=note)
    return truncate_preview(
        body, max(cap - len(footer), _MIN_PREVIEW_CHARS), tool_name=tool_name,
    ) + footer


def _write_spill(
    tool_name: str, body: str, *, require_visible: bool, task_id: str = "",
) -> tuple[Path, str] | None:
    """Persist ``body`` once and return its path plus the agent-visible ref.

    Named by ``sha256(tool_name, body)`` rather than a fresh uuid so re-spilling
    the SAME body is idempotent. Tier 2 re-spills every protected fan-in result
    on each pass that wins, which under a uuid name left one identical copy per
    compaction on disk and burned a manifest slot each time.

    Returns ``None`` when the store is unreachable, or when the caller requires
    an agent-visible path and this backend cannot name one.
    """
    write_dir, visible_dir = _overflow_dir(task_id or _current_task_id())
    if require_visible and not visible_dir:
        return None
    digest = hashlib.sha256(
        tool_name.encode("utf-8") + b"\x00" + body.encode("utf-8", "replace"),
    ).hexdigest()[:16]
    path = write_dir / f"{digest}.md"
    if not path.exists():
        # Write-then-rename: a crash mid-write would otherwise leave a truncated
        # file under the name the digest resolves to, which ``path.exists()``
        # then treats as a complete spill forever.
        tmp = path.with_name(f".{digest}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(_spill_document(tool_name, digest, body), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Failed to spill %s result: %s", tool_name, exc)
            with contextlib.suppress(OSError):
                tmp.unlink()
            return None
    return path, (f"{visible_dir}/{path.name}" if visible_dir else "")


def _spill_document(tool_name: str, spill_id: str, result: str) -> str:
    """Use grep-friendly markdown and keep the captured body verbatim.

    ``spill_id`` is the content digest, not a call id: the same body spilled
    twice is one file, so no single call owns it.
    """
    return (
        f"# {tool_name} — spilled tool result\n\n"
        f"- id: `{spill_id}`\n"
        f"- captured: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"- length: {len(result):,} chars"
        f"{_SPILL_SEPARATOR}{result}"
    )


def _spill_footer(ref: str, *, full_len: int, note: str = "") -> str:
    """The model-visible pointer back to the full result.

    Kept at a fixed shape so :func:`budgeted_preview` can charge its length to
    the preview budget before deciding where to cut.
    """
    suffix = f" {note}" if note else ""
    if not ref:
        return (
            f"\n\n[... only part of this {full_len:,}-char result is shown; the "
            f"remainder is not readable from this backend.{suffix}]"
        )
    directory = ref.rsplit("/", 1)[0]
    # Name more than one route: ``read_file`` is not bound in every profile (the
    # stateful_react benchmark profile binds bash/grep_search/glob_search and no
    # reader), and an agent that follows the advice literally there gets "unknown
    # tool" instead of its own spilled content.
    return (
        f"\n\n[... only part of this {full_len:,}-char result is shown (head and "
        f"tail; the gap is marked above). Full content is saved read-only at "
        f"{ref}. Only if the elided middle is required, read that path with "
        f"whichever tool you have — read_file, `cat` via bash, or "
        f"grep_search(pattern=\"...\", path=\"{directory}\"). It is read-only; "
        f"do not write there.{suffix}]"
    )


def spill_compacted_body(tool_name: str, body: str) -> str | None:
    """Persist a result immediately before compaction discards its inline body."""
    if len(body) < _SPILL_MIN_CHARS:
        return None
    spilled = _write_spill(tool_name, body, require_visible=True)
    return spilled[1] if spilled else None


def maybe_overflow(
    tool_name: str,
    result: str,
    *,
    task_id: str = "",
    call_id: str = "",
) -> str:
    """Check if result exceeds max_result_chars and overflow to disk if needed.

    Args:
        tool_name: Name of the tool that produced the result.
        result: The full tool result string.
        task_id: Optional task ID for organizing overflow files. Defaults to
            the current execution scope. Pass the SAME composite form the scope
            uses (``f"{task_id}:{llm_session_id}"``) or the store will not be
            the one ``cleanup_overflow`` removes.
        call_id: Accepted for backwards compatibility and no longer used to name
            the file — see :func:`_write_spill` on content-hash naming.

    Returns:
        The original result if within limits, or a head-and-tail preview with a
        reference to the overflow file, together no longer than the cap.
    """
    meta = get_tool_meta(tool_name)

    # 0 means no limit
    if meta.max_result_chars <= 0:
        return result

    if len(result) <= meta.max_result_chars:
        return result

    spilled = _write_spill(
        tool_name, result, require_visible=False, task_id=task_id,
    )
    if spilled is not None:
        logger.info(
            "Tool result overflow: %s result (%d chars) saved to %s",
            tool_name, len(result), spilled[0],
        )
    # A failed write only costs the pointer: the footer then says the remainder
    # is unreadable instead of naming a path that does not exist.
    ref = spilled[1] if spilled else ""
    return budgeted_preview(
        result, cap=meta.max_result_chars, ref=ref, tool_name=tool_name,
    )


# ── Aggregate budget (per-turn total) ───────────────────────────────────

# Max total chars across all tool results in a single ReAct turn.
# Prevents N parallel tools from flooding the context.
# Ref: Claude Code MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200,000
MAX_AGGREGATE_RESULT_CHARS = 200_000
# Floor for one result inside the aggregate pass: a turn of many big results
# must not cut any single one down to nothing.
_MIN_AGGREGATE_KEEP = 2_000


def check_aggregate_budget(
    results: list[str],
    tool_names: list[str] | None = None,
    task_id: str = "",
) -> list[str]:
    """Enforce aggregate budget across multiple tool results in one turn.

    If the total exceeds MAX_AGGREGATE_RESULT_CHARS, the largest results are
    re-cut — spilling first, so what this pass removes stays recoverable — until
    the total fits. A result that already carries a spill pointer keeps it: the
    pointer sits at the very end of the string and the tail half of the preview
    survives, so recovery chains from this file to the original one.

    Args:
        results: List of tool result strings (already individually overflowed).
        tool_names: Optional list of tool names (parallel to results).
        task_id: Unused; the store follows the current execution scope.

    Returns:
        Adjusted list of results, same length as input.
    """
    total = sum(len(r) for r in results)
    if total <= MAX_AGGREGATE_RESULT_CHARS:
        return results

    logger.info(
        "Aggregate tool results (%d chars) exceed budget (%d), re-truncating",
        total, MAX_AGGREGATE_RESULT_CHARS,
    )

    names = list(tool_names or [])
    adjusted = list(results)
    # Largest first: cutting the biggest result is what buys the most room, and
    # leaves the small results in the turn untouched.
    for idx, result in sorted(enumerate(results), key=lambda x: len(x[1]), reverse=True):
        if total <= MAX_AGGREGATE_RESULT_CHARS:
            break
        excess = total - MAX_AGGREGATE_RESULT_CHARS
        cap = max(_MIN_AGGREGATE_KEEP, len(result) - excess)
        if cap >= len(result):
            continue
        name = names[idx] if idx < len(names) else "tool"
        spilled = (
            _write_spill(name, result, require_visible=True)
            if len(result) >= _SPILL_MIN_CHARS
            else None
        )
        replacement = budgeted_preview(
            result,
            cap=cap,
            ref=spilled[1] if spilled else "",
            note="Cut further to fit the per-turn tool-result budget.",
            tool_name=name,
        )
        total -= len(result) - len(replacement)
        adjusted[idx] = replacement

    return adjusted


def get_overflow_content(overflow_path: str) -> str | None:
    """Read the full content from an overflow file.

    Args:
        overflow_path: Path to the overflow JSON file.

    Returns:
        The full tool result content, or None if not found.
    """
    path = Path(overflow_path)
    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        if _SPILL_SEPARATOR in text:
            return text.split(_SPILL_SEPARATOR, 1)[1]
        # Backward compatibility for sessions holding pointers to old JSON spills.
        data = json.loads(text)
        return data.get("content")
    except Exception as e:
        logger.warning("Failed to read overflow file %s: %s", overflow_path, e)
        return None


def cleanup_overflow(
    scope: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> int:
    """Remove the spilled tool results of one finished conversation.

    Args:
        scope: The store to remove, in the SAME composite form the writers use —
            ``f"{task_id}:{llm_session_id}"``, which is what
            :func:`_current_task_id` returns. A bare ``task_id`` hashes to a
            different directory and would silently match nothing, since
            ``llm_session_id`` defaults to ``task_id`` rather than staying empty.
            Omit it to clean up the caller's own current scope. An explicitly
            empty string is always a safe no-op.
        workspace: Ignored, kept so existing teardown calls still type-check.
            The store no longer lives under a workspace.

    Returns:
        Number of files removed.
    """
    del workspace
    if scope == "":
        return 0
    resolved_scope = _current_task_id() if scope is None else scope
    if not resolved_scope:
        return 0

    # ``create=False``: resolving a store in order to delete it must not first
    # bring it into existence, which would also leave a stray empty directory
    # behind for any scope that never spilled.
    return _remove_store(_overflow_dir(resolved_scope, create=False)[0])


def _remove_store(store: Path) -> int:
    """Delete one store directory's files, then the directory. Count the files."""
    if not store.is_dir():
        return 0
    count = 0
    for entry in store.iterdir():
        try:
            entry.unlink()
            count += 1
        except OSError:
            pass
    with contextlib.suppress(OSError):
        store.rmdir()
    _created_stores.discard(store)
    return count


def cleanup_overflow_process() -> int:
    """Remove every store THIS process created.

    For a discarded conversation: the TUI's ``/clear`` and ``/mode`` drop all
    history that could reference a spill path, so the files it named are dead.

    PRECONDITION: one conversation per process, which holds for the only caller —
    the terminal app — and for the benchmark runner's subprocess-per-question. A
    server multiplexing concurrent sessions in one process must NOT use this; it
    would delete a live session's recovery files. Such a caller wants
    :func:`cleanup_overflow` per scope instead.
    Scoped to what this process created rather than to a directory tree, which
    is both safer — another session's store is not ours to delete, whatever it
    is next to — and simpler, since deleting only paths we made needs none of
    the symlink and filesystem-root defences that walking the agent's workspace
    required.
    """
    return sum(_remove_store(store) for store in list(_created_stores))
