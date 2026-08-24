"""BudgetObserver — critical observer that tracks token usage and stops the loop
when the token budget is exhausted.
"""
from __future__ import annotations

from frontier_agent.core.loop_types import Intervention, LoopConfig, TurnContext


class BudgetObserver:
    """Track cumulative token usage and intervene when the budget is exhausted.

    critical = True  → awaited; Intervention return values are collected.
    """

    critical = True

    def __init__(self, max_tokens: int = 500_000, warn_ratio: float = 0.8) -> None:
        self.max_tokens = max_tokens
        self.warn_ratio = warn_ratio
        self.tokens_used: int = 0
        self._warned: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        """Reset state at the start of each loop."""
        self.tokens_used = 0
        self._warned = False

    async def on_llm_response(self, ctx: TurnContext) -> None:
        """Accumulate token counts from the LLM response usage dict."""
        if ctx.usage is None:
            return
        self.tokens_used += ctx.usage.get("input_tokens", 0) + ctx.usage.get("output_tokens", 0)
        ctx.metadata["budget_tokens_used"] = self.tokens_used
        ctx.metadata["budget_tokens_limit"] = self.max_tokens

    async def on_tool_result(self, ctx: TurnContext, result: object) -> None:
        pass

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        """Return an Intervention if the budget is exhausted or nearly so."""
        ratio = self.tokens_used / self.max_tokens if self.max_tokens > 0 else 0.0

        if ratio >= 1.0:
            return Intervention(
                stop_reason="budget_exhausted",
                inject_messages=[
                    f"Token budget exhausted ({self.tokens_used:,} / {self.max_tokens:,} tokens). "
                    "Stopping the loop to avoid exceeding limits."
                ],
            )

        if ratio >= self.warn_ratio and not self._warned:
            self._warned = True
            pct = int(ratio * 100)
            return Intervention(
                inject_messages=[
                    f"Warning: {pct}% of token budget used "
                    f"({self.tokens_used:,} / {self.max_tokens:,}). "
                    "Consider wrapping up soon."
                ],
            )

        return None

    async def on_loop_end(self, result: object) -> None:
        pass
