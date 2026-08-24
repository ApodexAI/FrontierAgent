"""DuplicateQueryRollbackObserver — pop and re-run a turn that re-issues a search.

**Problem this fixes.** A research agent that has lost the result of an
earlier search (context compaction replaces older tool results with a
placeholder) commonly re-issues the *identical* query instead of
pivoting. Measured on a 200-question deep-research benchmark, the
losing trials averaged ~37 duplicate ``web_search`` queries each and ran
58-122 turns where a healthy trial finishes in ~22: the identical
results pile up in history, inflate the context, and steer the model
into a longer wandering loop. With ``sub_max_turns`` at 100 a sub-agent
can spend its whole allowance this way.

**What this observer does.** Fires on ``on_llm_response`` — after the LLM
emitted an assistant message with tool calls, *before* those tools run.
When a tracked tool call repeats a request already executed in this loop,
it returns an :class:`Intervention` that pops the assistant message and
re-runs the turn without consuming a ``max_turns`` slot (the loop's
``EXTRA_ATTEMPTS_BUFFER`` bounds total iterations). At the sampling
temperatures these profiles use, the next generation almost never
reproduces the same query verbatim.

**Rollback budget.** ``max_consecutive_rollbacks`` (default 5) is the
livelock valve: once 4 rollbacks have fired back-to-back the next
duplicate is allowed through. The counter resets only after a turn that
carried tracked tool calls and had no duplicate — a budget-exhaust
let-through deliberately leaves it at the ceiling, so the model has to
ride through one fully clean turn before detection re-arms.

**What is deduped.** Only search tools (``web_search`` by default).
``web_fetch`` on the same URL is often a legitimate retry after a
transient render failure, and ``bash`` / ``run_python_code`` arguments
are too varied to dedupe usefully.

**Key granularity — batch, not per-query.** ``web_search`` accepts a list
of queries, and the whole call is one dedup unit: ``[a, b]`` and
``[a, c]`` are different searches, so a model refining a batch
incrementally is never rolled back. Keying on individual queries instead
fires an order of magnitude more often and pops legitimate query
evolution.

**A turn holding a terminal tool is never popped.** The rollback discards
the whole assistant message, so every tool call in the batch dies with the
duplicate search. A batch of ``[web_search(dup), submit_report(...)]``
would lose the report permanently: the loop returns before
``_execute_tool_calls``, so ``FinalizeAnswerObserver`` never sees it. The
terminal set comes from the loop's own ``LoopPolicy`` rather than a
hardcoded list, plus a small floor for tools that finalize without being
the policy's terminal.

**A request is only remembered once it actually returned something.**
Bookkeeping happens in ``on_tool_result``, not when the turn is let
through. Search failures arrive as ordinary result *strings*
(``"[ERROR]: …"``, ``"No search results found."``) with ``is_error``
False, and a transient upstream failure is precisely the case where
re-running the same query is correct — the one case an eager recording
would block.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["DuplicateQueryRollbackObserver"]

# Arguments that change how MANY rows come back but not WHICH request was
# made. Everything else (``page``, ``tbs``, ``gl``, ``hl``, ``location``)
# selects a different result set, so it belongs in the key: the aligned
# web_search exposes pagination, and paging through one query must not
# read as re-issuing it.
_COUNT_ARG_KEYS = frozenset({"num", "num_results"})


# Result strings that mean "this search returned nothing usable". A failed
# search must not be remembered as executed — see the module docstring.
_FAILED_RESULT_PREFIXES = (
    "[ERROR]:",
    "No results found for:",
    "Error: search query cannot be empty.",
)
_FAILED_RESULT_EXACT = frozenset({"No search results found."})

# Tools that finalize a loop. Union'd with the active ``LoopPolicy``'s own
# ``terminal_tool_names``, which is authoritative; this floor covers a tool
# that finalizes through an observer instead of the policy.
_TERMINAL_TOOL_FLOOR = frozenset({
    "submit_report",
    "finalize_answer",
    "finish_planning",
})


def _coerce_str_list(value: Any) -> Any:
    """Parse a JSON-list string into a list; pass everything else through.

    Deliberately mirrors ``plugins.tools._coerce.coerce_json_list`` instead of
    importing it: nothing under ``frontier_agent/`` imports from ``plugins/``,
    and this observer is not the place to open that direction. The behaviour
    has to match, because the search tool runs the same coercion on its own
    ``q`` — models emit the SAME logical batch both as a real array and as a
    stringified one, and a key that does not coerce would file the two forms
    under different keys and miss the duplicate.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped.startswith(("[", "{")):
        return value
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value
    if isinstance(parsed, dict):
        return [parsed]
    return parsed


def _normalise_q(value: Any) -> str | None:
    """Render the ``q`` argument as one stable string, or ``None`` if empty."""
    coerced = _coerce_str_list(value)
    if isinstance(coerced, str):
        stripped = coerced.strip()
        return stripped or None
    if isinstance(coerced, list):
        parts = [
            item.strip()
            for item in coerced
            if isinstance(item, str) and item.strip()
        ]
        return "|".join(parts) if parts else None
    return None


def _is_failed_search(result: ToolResult) -> bool:
    """True when a search result carries no usable content."""
    if result.is_error:
        return True
    text = (result.result or "").strip()
    if not text or text in _FAILED_RESULT_EXACT:
        return True
    return text.startswith(_FAILED_RESULT_PREFIXES)


def _dedup_key(tool_call: dict[str, Any]) -> str | None:
    """Build the dedup key for one tool call, or ``None`` when untrackable.

    No-raise by contract — this runs on the critical observer path, where
    an exception costs the current turn. A call without a usable ``q`` is
    reported as untrackable rather than keyed on its remaining arguments:
    keying a search on locale alone would collapse unrelated queries.
    """
    name = str(tool_call.get("name") or "")
    if not name:
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None
    q = _normalise_q(args.get("q"))
    if q is None:
        return None
    rest = {
        key: value
        for key, value in args.items()
        if key != "q" and key not in _COUNT_ARG_KEYS and value is not None
    }
    try:
        suffix = json.dumps(rest, sort_keys=True, default=str)
    except (TypeError, ValueError):
        suffix = repr(sorted(rest))
    return f"{name}_{q}_{suffix}"


class DuplicateQueryRollbackObserver(BaseObserver):
    """Roll back a turn whose search request was already executed.

    ``critical`` so the loop awaits the hook and collects the returned
    ``Intervention`` — a non-critical observer's return value is dropped.
    """

    critical: bool = True

    #: Tools whose requests are deduped. See the module docstring for why
    #: fetch / shell tools are excluded.
    DEFAULT_TOOL_NAMES: frozenset[str] = frozenset({"web_search"})

    def __init__(
        self,
        *,
        tool_names: set[str] | frozenset[str] | None = None,
        max_consecutive_rollbacks: int = 5,
    ) -> None:
        self._tool_names: frozenset[str] = frozenset(
            tool_names if tool_names else self.DEFAULT_TOOL_NAMES,
        )
        # 2 is the floor: 1 would mean "never roll back" (the budget check
        # is ``< max - 1``), which silently disables the observer.
        self._max_consecutive_rollbacks = max(2, int(max_consecutive_rollbacks))
        self._seen: dict[str, int] = {}
        self._consecutive_rollbacks = 0
        self._terminal_tools: frozenset[str] = _TERMINAL_TOOL_FLOOR

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._seen.clear()
        self._consecutive_rollbacks = 0
        # The loop's own policy is authoritative about what finishes it.
        policy_terminals = {
            str(name)
            for name in getattr(
                getattr(config, "loop_policy", None), "terminal_tool_names", (),
            )
            if str(name)
        }
        self._terminal_tools = _TERMINAL_TOOL_FLOOR | policy_terminals

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if not ctx.tool_calls:
            # A think-only / bare-text turn is not a rollback candidate;
            # refresh the budget so the next tracked turn starts clean.
            self._consecutive_rollbacks = 0
            return None

        turn_keys: list[str] = []
        for tool_call in ctx.tool_calls:
            if str(tool_call.get("name") or "") not in self._tool_names:
                continue
            key = _dedup_key(tool_call)
            if key:
                turn_keys.append(key)

        if not turn_keys:
            # Untracked tools only (web_fetch / bash / …). Leave the
            # counter alone and let the next tracked turn manage it.
            return None

        if any(
            str(tool_call.get("name") or "") in self._terminal_tools
            for tool_call in ctx.tool_calls
        ):
            # Never pop a turn that is trying to finish — the terminal call
            # would be discarded with the duplicate and never re-run.
            logger.info(
                "DuplicateQueryRollback turn=%d: batch carries a terminal "
                "tool; letting it through without dedup bookkeeping.",
                ctx.turn,
            )
            return None

        duplicates = [key for key in turn_keys if self._seen.get(key, 0) > 0]

        if duplicates and (
            self._consecutive_rollbacks < self._max_consecutive_rollbacks - 1
        ):
            self._consecutive_rollbacks += 1
            logger.info(
                "DuplicateQueryRollback turn=%d: %d duplicate search call(s) "
                "— popping the assistant message and retrying (rollback "
                "%d/%d). Example: %r",
                ctx.turn,
                len(duplicates),
                self._consecutive_rollbacks,
                self._max_consecutive_rollbacks,
                duplicates[0][:160],
            )
            return Intervention(
                pop_last_message=True,
                continue_to_next_turn=True,
            )

        # Either nothing repeated, or the budget is spent and this turn is
        # let through. Recording happens in ``on_tool_result`` once the search
        # has actually returned something. Reset the streak ONLY on a
        # genuinely clean turn.
        if not duplicates:
            self._consecutive_rollbacks = 0
        elif logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "DuplicateQueryRollback turn=%d: rollback budget spent "
                "(%d consecutive) — letting a duplicate search through.",
                ctx.turn,
                self._consecutive_rollbacks,
            )
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        """Remember a tracked request once it returned usable content."""
        del ctx
        if result is None or str(result.name or "") not in self._tool_names:
            return None
        if _is_failed_search(result):
            logger.info(
                "DuplicateQueryRollback: %s returned no usable content; not "
                "recording it, so a retry is allowed.",
                result.name,
            )
            return None
        key = _dedup_key({"name": result.name, "args": result.args})
        if key:
            self._seen[key] = self._seen.get(key, 0) + 1
        return None
