"""TaskBoardTruthGuard — enforces the task board as the single source of
truth for what remains to be done, blocking RE-delegation to an already-
existing agent once every board item is genuinely resolved, while still
allowing genuinely new agents (like a final_verifier) through.

Real, added 2026-09-08, hardened and corrected multiple times the same
session. Directly motivated by seven real, confirmed findings from this
same session's coordinator-only microtask tests (see prior in-code history
for 1-6). This version fixes a real, confirmed regression found in a live,
direct test: the guard, as previously written, blocked EVERY delegation
tool call once the board was fully resolved -- including a genuinely new,
legitimate spawn_agent("final_verifier", ...) call, part of the real,
intended protocol (spawn a verifier once the main work is done). That is a
real over-block: it conflated "re-dispatching already-completed work" with
"doing genuinely new, additional work", and blocked both.

The fix: only block when the delegation call's target agent NAME already
exists (confirmed, directly, via AgentBus.list_sessions_for_task) -- that
is the real, concrete signal of re-dispatching already-assigned work.
A call naming a genuinely new agent is let through even when the board is
fully resolved, since a new name cannot be "already-completed work" by
definition.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    ToolCallIntervention,
    TurnContext,
)

logger = logging.getLogger(__name__)

_DELEGATION_TOOLS = frozenset({"create_subagent", "spawn_agent", "assign_task"})

_EXPLAIN_AT = 1
_FORCEFUL_AT = 2
_HARD_STOP_AT = 3

_FINAL_ANSWER_TEMPERATURE = 0.1


def _target_agent_names(tool_name: str, tool_call: dict) -> set[str]:
    """Real, direct extraction of which agent name(s) a delegation call
    targets, across the three real, different argument shapes these tools
    actually use (confirmed directly against each tool's real signature)."""
    args = tool_call.get("args") or {}
    names: set[str] = set()
    if tool_name == "spawn_agent":
        n = args.get("name")
        if isinstance(n, str) and n:
            names.add(n)
    elif tool_name == "create_subagent":
        for a in args.get("agents") or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str):
                names.add(a["name"])
    elif tool_name == "assign_task":
        for t in args.get("tasks") or []:
            if isinstance(t, dict) and isinstance(t.get("agent"), str):
                names.add(t["agent"])
    return names


class TaskBoardTruthGuard(BaseObserver):
    critical = True

    def __init__(self) -> None:
        self._blocked_streak: dict[str, int] = {}
        self._force_stop: dict[str, bool] = {}

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        name = str(tool_call.get("name", ""))
        # Real, added 2026-09-19: verification-exhaustion counter. Placed
        # here, at the top, before this function's own early-return
        # guards below (board_size == 0, not fully_resolved) -- this
        # signal is deliberately NOT tied to this observer's own
        # board-resolution logic at all, it just answers "was a real
        # verification task genuinely attempted this run", regardless of
        # board state.
        #
        # Real, corrected architectural finding before this was deployed:
        # cannot use ctx.metadata here at all, despite ctx being available
        # in this method -- confirmed directly, ctx.metadata and
        # ExecutionScope.metadata (read via get_current_execution_scope())
        # are two completely separate dicts (agent_loop.py constructs
        # `scope_meta` as its own fresh dict, entirely distinct from the
        # `metadata` variable passed to TurnContext). The real consumer of
        # this signal is `finalize_answer`, a plain @tool function that
        # never receives `ctx` at all (confirmed directly: its real
        # signature is `async def finalize_answer(content, confidence)`)
        # -- it can only read `get_current_execution_scope().metadata`.
        # Using ctx.metadata here would have made this signal invisible to
        # the one function that actually needs to read it. Reuses this
        # file's own existing `_target_agent_names` helper rather than
        # re-deriving target extraction logic.
        if name == "assign_task":
            targets = _target_agent_names(name, tool_call)
            if any("verif" in t.lower() for t in targets):
                from frontier_agent.core.execution_context import (
                    get_current_execution_scope,
                )
                scope = get_current_execution_scope()
                if scope is not None:
                    scope.metadata["verification_assignment_streak"] = (
                        scope.metadata.get("verification_assignment_streak", 0) + 1
                    )
        if name not in _DELEGATION_TOOLS:
            return None

        from plugins.tools.task_board import board_size, unresolved_task_ids

        if board_size(ctx.task_id) == 0:
            return None

        fully_resolved = len(unresolved_task_ids(ctx.task_id)) == 0
        if not fully_resolved:
            self._blocked_streak[ctx.task_id] = 0
            return None

        # Real, added 2026-09-08: only treat this as re-dispatch of
        # already-completed work if the target agent name(s) already
        # exist. A genuinely new agent name (e.g. a first-time
        # final_verifier) is real, legitimate additional work, not a
        # repeat of anything -- let it through even though the board is
        # fully resolved.
        #
        # Real, widened 2026-09-19: also exempt an EXISTING target with
        # genuinely zero real assigned work so far (total_task_count ==
        # 0). Root-caused directly, via a real, live raw-transcript
        # trace: finalize_gate() requires a verifier session with
        # total_task_count > 0 before allowing FINAL_ANSWER, but this
        # exact code, unwidened, only checked session *existence* -- a
        # freshly spawn_agent'd verifier (e.g. final_verifier, created
        # the immediately prior turn) already "exists" as a session
        # despite never having done any real work at all, so its own
        # first, genuinely legitimate assign_task attempt was being
        # blocked here as if it were a repeat/re-dispatch, creating a
        # real, confirmed logical deadlock between these two independent
        # gates: verification requires assignable work; this guard
        # refused to allow that exact assignment. This widening lets a
        # target that already exists but has done zero real work through
        # exactly once (matching the same "genuinely new work, not a
        # repeat" spirit as the exemption above) -- it does NOT skip or
        # weaken the verification requirement itself in finalize_gate(),
        # it only lets the real, legitimate first assignment to a fresh
        # verifier actually go through, so real verification work can
        # genuinely happen and total_task_count can genuinely become > 0.
        targets = _target_agent_names(name, tool_call)
        if targets:
            from frontier_agent.components.agent_bus import AgentBus
            from frontier_agent.core.runtime.registries import services as registry
            bus = registry.get_optional(AgentBus)
            sessions = bus.list_sessions_for_task(ctx.task_id) if bus else []
            existing = {s.name for s in sessions}
            worked = {
                s.name for s in sessions
                if getattr(s, "total_task_count", 0) > 0
            }
            genuinely_new = targets - existing
            existing_but_unworked = (targets & existing) - worked
            if genuinely_new or existing_but_unworked:
                logger.info(
                    "TaskBoardTruthGuard: allowing %s (task=%s) -- target "
                    "agent(s) %s are genuinely new or exist but have done "
                    "zero real work yet, not a re-dispatch of existing "
                    "work",
                    name, ctx.task_id, sorted(genuinely_new | existing_but_unworked),
                )
                self._blocked_streak[ctx.task_id] = 0
                return None

        streak = self._blocked_streak.get(ctx.task_id, 0) + 1
        self._blocked_streak[ctx.task_id] = streak

        if streak >= _HARD_STOP_AT:
            self._force_stop[ctx.task_id] = True
            metadata_updates = None
            if (ctx.ai_text or "").strip():
                metadata_updates = {"final_answer": ctx.ai_text.strip()}
                logger.info(
                    "TaskBoardTruthGuard: harvesting real, existing plain "
                    "text (task=%s, %d chars) as the final answer",
                    ctx.task_id, len(ctx.ai_text.strip()),
                )
            logger.info(
                "TaskBoardTruthGuard: %s blocked %d time(s) on a fully "
                "resolved board (task=%s) -- forcing a hard finalize",
                name, streak, ctx.task_id,
            )
            return ToolCallIntervention(
                skip_with_result=(
                    "All tasks on the board are resolved. You must now "
                    "produce the final answer as plain text. No further "
                    "tool calls will be executed this turn."
                ),
                metadata_updates=metadata_updates,
            )

        if streak >= _FORCEFUL_AT:
            ctx.metadata["_llm_strip_tools"] = True
            ctx.metadata["_llm_temp_override"] = _FINAL_ANSWER_TEMPERATURE
            logger.info(
                "TaskBoardTruthGuard: %s blocked %d time(s) (task=%s) -- "
                "forceful redirect: tool-calling disabled + temperature "
                "lowered to %.1f for the model's next turn",
                name, streak, ctx.task_id, _FINAL_ANSWER_TEMPERATURE,
            )
            return ToolCallIntervention(skip_with_result=(
                "You are now in FINAL ANSWER MODE. Your tool-calling "
                "ability has been disabled -- you cannot call any tool on "
                "your next turn, no matter what you write. All tasks on "
                "the board are resolved; there is nothing left to plan, "
                "delegate, or verify. Write your final answer as plain "
                "text now."
            ))

        logger.info(
            "TaskBoardTruthGuard: blocking %s (task=%s, attempt %d) -- "
            "task board shows every item genuinely resolved already, and "
            "the target agent(s) already exist",
            name, ctx.task_id, streak,
        )
        return ToolCallIntervention(skip_with_result=(
            f"Blocked: the task board already shows every item resolved, "
            f"and {name} named an agent that already exists -- there is no "
            "real, remaining work for it to do. If you genuinely need a "
            "NEW agent (e.g. a verifier) use a NEW, not-yet-used name; if "
            "you believe there is genuinely more to investigate, add_task "
            "a NEW, real item describing it first; otherwise, deliver "
            "your final answer now."
        ))

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if not self._force_stop.pop(ctx.task_id, False):
            return None
        logger.info(
            "TaskBoardTruthGuard: forcing loop end (task=%s) after repeated "
            "blocked attempts on a fully resolved board",
            ctx.task_id,
        )
        return Intervention(
            stop_reason="task_board_fully_resolved",
            inject_messages=[
                "All tasks on the board are resolved. You must now produce "
                "the final answer as plain text. No further tool calls are "
                "allowed."
            ],
        )


__all__ = ["TaskBoardTruthGuard"]
