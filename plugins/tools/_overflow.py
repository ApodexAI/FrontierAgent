"""Tool result overflow — persist large results to disk, return summary inline."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from agent_core.runtime.spill import SpillStore, scope_component

from plugins.tools.meta import get_tool_meta

logger = logging.getLogger(__name__)

# Persistence is AgentCore's ``SpillStore`` — one implementation, one registry
# of created stores — so nothing here writes, reads or deletes spill files by
# hand any more. What stays product-side is the policy around it:
#
#   * the scope KEY (:func:`current_store_scope`), which is the one identity the
#     store directory, the bwrap mount (``spill_bind_args``) and read
#     authorization (``_path_auth``) all derive from;
#   * the VISIBLE root (:func:`_visible_root`), i.e. whether the backend that
#     actually runs model commands can name the store at all;
#   * preview shaping and the footer, through AgentCore's ``budgeted_preview``.
_SPILL_SEPARATOR = "\n---\n\n"
#: Unscoped writes (no ExecutionScope) go to one store per PROCESS. The pid is
#: part of the key because the default root is shared by every process of a
#: uid: a constant key would merge two processes' unscoped stores, and either
#: one's cleanup would delete the other's files.
UNSCOPED_PROCESS_STORE = "__process__"
#: Registry of the stores this process created — AgentCore's, shared by
#: reference so a reset in tests affects the store's own bookkeeping too.
_created_stores: set[Path] = SpillStore._created_stores
#: parent scope key -> child scope keys entered from inside it (in-process
#: sub-agents). A parent may read its children's stores — a fan-in report can
#: carry a child's spill path back — but siblings never see each other's.
_child_scopes: dict[str, set[str]] = {}
_child_lock = threading.Lock()
#: digest(preview text) -> ref of the FULL body behind it. Lets a later
#: compaction of that preview point at the original full text instead of
#: storing the preview and labelling it "[Full text]". Bounded; exact-match only,
#: so an agent echoing part of a preview cannot claim someone else's ref.
_preview_refs: OrderedDict[str, str] = OrderedDict()
_PREVIEW_REFS_MAX = 4_096


def spill_scope_key(task_id: str, llm_session_id: str = "") -> str:
    """The store key for one conversation: ``task:session`` (or ``task``).

    Built only from stable ids — never ``metadata["session_id"]``, which trace
    plumbing may replace mid-run.
    """
    task_id = str(task_id or "")
    session = str(llm_session_id or "")
    if not task_id:
        return ""
    return f"{task_id}:{session}" if session else task_id


def _scope_key_of(scope: object | None) -> str:
    if scope is None:
        return ""
    metadata = getattr(scope, "metadata", None) or {}
    return spill_scope_key(
        str(getattr(scope, "task_id", "") or ""),
        str(metadata.get("llm_session_id") or ""),
    )


def _current_task_id() -> str:
    """The current ExecutionScope's store key, or "" outside any scope."""
    from frontier_agent.core.execution_context import get_current_execution_scope

    return _scope_key_of(get_current_execution_scope())


def current_store_scope() -> str:
    """The ONE scope key writers, the mount and read authorization share."""
    return _current_task_id() or f"{UNSCOPED_PROCESS_STORE}:{os.getpid()}"


def register_child_scope(parent: object | None, child: object | None) -> None:
    """Record that ``child``'s loop was entered from inside ``parent``'s."""
    parent_key, child_key = _scope_key_of(parent), _scope_key_of(child)
    if parent_key and child_key and parent_key != child_key:
        with _child_lock:
            _child_scopes.setdefault(parent_key, set()).add(child_key)


def readable_scopes(scope: str | None = None) -> list[str]:
    """Scope keys an agent may read: its own and its descendants'."""
    root_key = current_store_scope() if scope is None else scope
    out: list[str] = []
    pending = [root_key]
    with _child_lock:
        while pending:
            key = pending.pop()
            if not key or key in out:
                continue
            out.append(key)
            pending.extend(_child_scopes.get(key, ()))
    return out


def _physical_root() -> Path:
    from plugins.tools._sandbox import spill_root

    return spill_root().expanduser().resolve()


def _resolved_backend() -> str:
    """The active backend as ``_sandbox`` resolves it, or "" when unresolvable.

    Reading ``SANDBOX_BACKEND`` straight from the environment misses a backend
    supplied only through ``config.yaml``. A misconfigured backend must not take
    spill down with it, so an error resolves to "" (decided by bwrap below).
    """
    from plugins.tools._sandbox import _get_sandbox_backend

    try:
        return _get_sandbox_backend()
    except Exception:
        return ""


def _visible_root() -> str | None:
    """How model commands can name the store, or ``None`` when they cannot.

    Decided by the sandbox that ACTUALLY runs commands when one exists, then by
    configuration: ``auto`` with an E2B key goes off-host, and a bwrap backend on
    a host where bwrap cannot run has no ``/spill`` mount. Advertising a path in
    either case would hand the model a ref it can never open — and would make
    the compaction callback promise a recovery that does not exist.
    """
    from plugins.tools import _sandbox as sb

    live = sb.get_existing_sandbox()
    if isinstance(live, sb.BwrapSandbox):
        return sb._DEFAULT_SPILL_DIR
    if isinstance(live, sb.CurrentSandbox):
        return sb._DEFAULT_SPILL_DIR if live._inner is not None else str(_physical_root())
    if live is not None and type(live).__module__.split(".", 1)[0].startswith("e2b"):
        return None

    backend = _resolved_backend()
    if backend == "e2b":
        return None
    if backend == "native":
        return str(_physical_root())
    if backend == "container":
        return sb._DEFAULT_SPILL_DIR if sb.container_uses_inner_bwrap() else str(_physical_root())
    if backend == "auto":
        try:
            use_e2b = sb._resolve_use_e2b()[0]
        except Exception:
            use_e2b = False
        if use_e2b:
            return None
    return sb._DEFAULT_SPILL_DIR if sb.bwrap_available() else None


def _store(scope: str | None = None) -> SpillStore:
    """The SpillStore for ``scope`` (default: the current one)."""
    return SpillStore(
        _physical_root(),
        current_store_scope() if scope is None else scope,
        visible_root=_visible_root(),
    )


def spill_is_recoverable() -> bool:
    """Whether a compaction spill would produce a ref the agent can open."""
    return _visible_root() is not None


def default_compaction_spill() -> Callable[[str, str], str | None] | None:
    """The compaction spill callback, or ``None`` when nothing is recoverable.

    AgentCore cannot introspect a plain function, so it takes any callback as a
    promise that discarded bodies stay recoverable and drops its guard that
    keeps the newest unseen results verbatim. Withholding the callback is how a
    backend without a readable store keeps that guard.
    """
    return spill_compacted_body if spill_is_recoverable() else None


def readable_store_dirs() -> list[Path]:
    """Physical directories of :func:`readable_scopes` this process created."""
    root = _physical_root()
    dirs: list[Path] = []
    for key in readable_scopes():
        directory = root / scope_component(key)
        if directory in _created_stores and directory.is_dir():
            dirs.append(directory)
    return dirs


def spill_bind_args() -> list[str]:
    """bwrap args mounting ONLY the readable scopes under ``/spill``.

    Built per command from the current scope, so a shared or pooled jail still
    shows each agent just its own store (and its sub-agents'). ``--ro-bind-try``
    because a store is created lazily on the first spill.
    """
    from plugins.tools._sandbox import _DEFAULT_SPILL_DIR

    root = _physical_root()
    args = ["--dir", _DEFAULT_SPILL_DIR]
    for key in readable_scopes():
        component = scope_component(key)
        args.extend([
            "--ro-bind-try", str(root / component), f"{_DEFAULT_SPILL_DIR}/{component}",
        ])
    return args


def _overflow_dir(task_id: str = "", *, create: bool = True) -> tuple[Path, str]:
    """Physical directory and agent-visible path of one scope's store."""
    store = _store(task_id or current_store_scope())
    if create:
        store.ensure()
    return store.directory, store.visible_directory


def body_names_a_spill_file(body: str) -> bool:
    """Whether *body* already carries a spill pointer the agent can act on.

    A presence test against the two roots the store can be named by — the
    canonical mount and the physical path — not a parse of the pointer's prose.
    Exists so the trajectory recovery footer can stay quiet when it would be
    redundant: the spill file already holds the full output.
    """
    if not body:
        return False
    from plugins.tools._sandbox import _DEFAULT_SPILL_DIR, spill_root

    roots = [_DEFAULT_SPILL_DIR]
    with contextlib.suppress(Exception):
        roots.append(str(spill_root()))
    # The separator is not cosmetic: a bare ``"/spill" in body`` also fires on
    # ``/spillover``, and a pointer always names a FILE under the store.
    return any(
        root and f"{root.rstrip('/')}/" in body for root in roots
    )


def agent_visible_spill_dir() -> str:
    """Return the spill directory as tools should name it, or empty if unreadable."""
    return _overflow_dir(current_store_scope())[1]


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
    """Persist ``body`` once through the scope's SpillStore.

    Content-addressed (``sha256(tool_name, body)``) so re-spilling the SAME body
    is idempotent. Returns ``None`` when the store is unreachable, or when the
    caller requires an agent-visible path and this backend cannot name one.
    """
    try:
        store = _store(task_id or current_store_scope())
        return store.write(tool_name, body, require_visible=require_visible)
    except OSError as exc:
        logger.warning("Failed to spill %s result: %s", tool_name, exc)
        return None


def _preview_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def remember_preview(preview: str, ref: str) -> None:
    """Record that ``preview`` stands for the full body stored at ``ref``."""
    if not preview or not ref:
        return
    key = _preview_key(preview)
    with _child_lock:
        _preview_refs[key] = ref
        _preview_refs.move_to_end(key)
        while len(_preview_refs) > _PREVIEW_REFS_MAX:
            _preview_refs.popitem(last=False)


def _full_body_ref(body: str) -> str:
    """The ref of the full body ``body`` is a preview of, if it is still readable."""
    with _child_lock:
        ref = _preview_refs.get(_preview_key(body), "")
    if not ref:
        return ""
    store = _store()
    for key in readable_scopes():
        candidate = SpillStore(store.root, key, visible_root=store.visible_root)
        if candidate.contains_path(ref) and candidate.read(ref) is not None:
            return ref
    return ""


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
    """Persist a result immediately before compaction discards its inline body.

    When ``body`` is itself a preview this module produced, the ref of the full
    body behind it is returned instead: storing the preview would label a cut
    copy "[Full text]" and orphan the real one.
    """
    if len(body) < _SPILL_MIN_CHARS:
        return None
    existing = _full_body_ref(body)
    if existing:
        return existing
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
        task_id: Optional store key. Defaults to :func:`current_store_scope`;
            pass the SAME ``spill_scope_key`` form or the store will not be the
            one the mount and read authorization expose.
        call_id: Accepted for backwards compatibility and no longer used to name
            the file — the store is content-addressed.

    Returns:
        The original result if within limits, or a head-and-tail preview with a
        reference to the overflow file, together no longer than the cap.
    """
    del call_id
    meta = get_tool_meta(tool_name)

    # 0 means no limit
    if meta.max_result_chars <= 0:
        return result

    if len(result) <= meta.max_result_chars:
        return result

    # ``require_visible``: a backend that cannot name the store (E2B) gets no
    # file at all — one written on this host would be unreachable forever.
    spilled = _write_spill(
        tool_name, result, require_visible=True, task_id=task_id,
    )
    if spilled is not None:
        logger.info(
            "Tool result overflow: %s result (%d chars) saved to %s",
            tool_name, len(result), spilled[0],
        )
    # A failed write only costs the pointer: the footer then says the remainder
    # is unreadable instead of naming a path that does not exist.
    ref = spilled[1] if spilled else ""
    preview = budgeted_preview(
        result, cap=meta.max_result_chars, ref=ref, tool_name=tool_name,
    )
    remember_preview(preview, ref)
    return preview


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
        # A re-cut preview still stands for the ORIGINAL full body when the
        # input was already one of ours.
        remember_preview(replacement, _full_body_ref(result) or (spilled[1] if spilled else ""))
        total -= len(result) - len(replacement)
        adjusted[idx] = replacement

    return adjusted


def get_overflow_content(overflow_path: str) -> str | None:
    """Read the full content from a spill file this conversation may read.

    Accepts a physical path or the agent-visible ref. Only the current scope's
    store, its sub-agents' stores, or (for a physical path) a store this
    process created are consulted — never an arbitrary file.
    """
    store = _store()
    for key in readable_scopes():
        candidate = SpillStore(store.root, key, visible_root=store.visible_root)
        if candidate.contains_path(overflow_path):
            return candidate.read(overflow_path)
    return SpillStore.read_created(overflow_path)


def cleanup_overflow(
    scope: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> int:
    """Remove the spilled tool results of one finished conversation.

    Args:
        scope: The store key, in :func:`spill_scope_key` form. Omit it to clean
            up the caller's own current scope. An explicitly empty string is
            always a safe no-op.
        workspace: Ignored, kept so existing teardown calls still type-check.

    Returns:
        Number of files removed.
    """
    del workspace
    if scope == "":
        return 0
    resolved_scope = current_store_scope() if scope is None else scope
    if not resolved_scope:
        return 0
    removed = SpillStore(_physical_root(), resolved_scope).cleanup()
    with _child_lock:
        _child_scopes.pop(resolved_scope, None)
    return removed


def cleanup_overflow_tree(scope: str | None = None) -> int:
    """Remove one conversation's store and every sub-agent store under it.

    Only stores this process created are touched, so another process — or an
    unrelated session in this one — keeps its files.
    """
    keys = readable_scopes(scope)
    root = _physical_root()
    removed = 0
    for key in keys:
        if root / scope_component(key) in _created_stores:
            removed += SpillStore(root, key).cleanup()
        with _child_lock:
            _child_scopes.pop(key, None)
    return removed


def cleanup_overflow_process() -> int:
    """Remove every store THIS process created.

    For a discarded conversation in a one-conversation process: the TUI's
    ``/clear`` and ``/mode`` drop all history that could reference a spill path.
    A server multiplexing concurrent sessions in one process must use
    :func:`cleanup_overflow_tree` per conversation instead.
    """
    with _child_lock:
        _child_scopes.clear()
        _preview_refs.clear()
    return SpillStore.cleanup_process()
