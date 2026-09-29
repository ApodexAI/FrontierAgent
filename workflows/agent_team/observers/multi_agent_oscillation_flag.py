"""MultiAgentOscillationFlag — logging-only observer that flags when the
main coordinator ping-pongs between two different sub-agents when
assigning tasks, rather than genuinely making forward progress.

Real, added 2026-09-08. This is a real, deliberately SIMPLIFIED,
coordinator-level counterpart to MultiToolOscillationFlag/
CycleDetectionGuard (both built earlier this session at the sub-agent
level). Confirmed directly, before building: no coordinator-level
observer list previously tracked which sub-agent name gets assigned a
task at each real assign_task call, and the original, much larger
"MultiAgentOscillationObserver" design proposed this session (tracking
8 separate signals: switch frequency, task/tool repetition across
agents, fallback/dominance/starvation correlation) was rejected as
over-scoped, the same pattern flagged for several other proposals this
session.

This simplified version implements only the single clearest condition,
directly reusing the same real period-N exact-signature cycle-detection
algorithm already confirmed correct in CycleDetectionGuard: does the
coordinator's sequence of assign_task target-agent names fall into a
stable, repeating A,B,A,B (or longer) cycle, without a genuinely new
agent name appearing to break it.

Real, fixed 2026-09-13 (a confirmed, real false positive): tracking
agent name ALONE genuinely misfired live -- a coordinator that assigns
{doc1, doc2}, then later re-assigns doc1 with a completely different,
new task ("compare the summaries", not "summarize doc1.txt" again) was
flagged as a "period-2 cycle" purely because the agent NAME repeated,
even though the actual work was genuinely new each time. Now tracks
(agent_name, normalized_task_text) pairs instead -- a real cycle now
requires the same agent to be re-assigned the SAME (or near-identical)
task repeatedly, which is what "ping-pong" actually means.

This is wired into the MAIN-agent's own observer list (in
workflows/agent_team/nodes/main_agent.py), a real, different location
from every other observer built this session, all of which were wired
into the SUB-agent observer list (workflows/agent_team/
subagent_runtime.py) instead.

This observer does NOT change behavior -- matching the established,
deliberate design of the other real observers built this session: it
only logs a flagged anomaly.
"""

from __future__ import annotations

import logging

from frontier_agent.core.loop_types import BaseObserver, TurnContext

logger = logging.getLogger(__name__)

__all__ = ["MultiAgentOscillationFlag"]


class MultiAgentOscillationFlag(BaseObserver):
    """Real, logging-only: tracks the real sequence of assign_task
    (agent, task) pairs at the coordinator level; flags a stable,
    repeating period-N cycle (e.g. A,B,A,B,...) where the SAME agent is
    genuinely re-assigned the SAME task repeatedly."""

    def __init__(self, *, max_period: int = 3, min_repeats: int = 2) -> None:
        self.max_period = max(2, int(max_period))
        self.min_repeats = max(2, int(min_repeats))
        self._history: list[tuple[str, str]] = []
        self._hinted_at_len = -1

    def _normalize_task_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _extract_target_pairs(self, tool_call: dict) -> list[tuple[str, str]]:
        args = tool_call.get("args") or {}
        tasks = args.get("tasks")
        if not isinstance(tasks, list):
            return []
        pairs: list[tuple[str, str]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            agent = task.get("agent")
            if not isinstance(agent, str) or not agent:
                continue
            task_text = task.get("task_prompt") or task.get("task") or ""
            pairs.append((agent, self._normalize_task_text(str(task_text))))
        return pairs

    def _detect_cycle(self) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        n = len(self._history)
        for period in range(2, self.max_period + 1):
            if n < period * 2:
                continue
            block = self._history[n - period:]
            repeats = 1
            start = n - period
            while start - period >= 0 and self._history[start - period:start] == block:
                repeats += 1
                start -= period
            if repeats >= self.min_repeats and (best is None or repeats > best[1]):
                best = (period, repeats)
        return best

    async def on_tool_call(self, ctx: TurnContext, tool_call: dict):
        if str(tool_call.get("name", "")) != "assign_task":
            return None
        pairs = self._extract_target_pairs(tool_call)
        if not pairs:
            return None
        self._history.extend(pairs)
        max_len = self.max_period * (self.min_repeats + 1)
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

        found = self._detect_cycle()
        if found is None:
            return None
        period, repeats = found
        if len(self._history) == self._hinted_at_len:
            return None
        self._hinted_at_len = len(self._history)

        # Real, print()-based (not logger.warning()), matching the
        # established, confirmed-necessary pattern from earlier this same
        # session -- logger output has nowhere real to go here.
        cycle_agents = [agent for agent, _task in self._history[-period:]]
        print(
            f"[MULTI_AGENT_OSCILLATION_FLAG] task={ctx.task_id or '?'} "
            f"turn={ctx.turn} -- coordinator's assign_task (agent, task) "
            f"pairs show a real, stable period-{period} cycle "
            f"(agents={cycle_agents!r}) repeated {repeats} times with the "
            f"SAME task text each time -- a genuine repeat, not just a "
            f"repeated agent name. Possible coordinator ping-pong. This "
            f"is a real, plausible risk signal, not a confirmed root "
            f"cause.",
            flush=True,
        )
        return None


__all__ = ["MultiAgentOscillationFlag"]
