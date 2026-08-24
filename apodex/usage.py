"""Token and current-context usage tracking for terminal front ends.

Providers report the size of the complete prompt, but not how that prompt is
distributed across conversation, summaries, and tool traffic. The observer
keeps the provider total authoritative and uses the local tokenizer only for
the labelled breakdown shown by ``/context``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)
from frontier_agent.core.messages import Message, text_of
from frontier_agent.core.runtime.loop.context_budget import estimate_tokens

_WORKFLOW_SECTION_RE = re.compile(
    r"(?m)^\[(Compacted earlier turns|User query|Assistant response|"
    r"Query/tool result(?: \([^\n]*\))?|Current user query|Earlier turn \d+)\]\s*\n?"
)
_SUMMARY_PREFIXES = (
    "[compacted summary",
    "[context compacted]",
    "[compaction failed",
)


def _tool_schema_tokens(tools: Iterable[Any]) -> int:
    schemas: list[Any] = []
    for tool in tools:
        try:
            schemas.append(
                tool.to_openai_schema() if hasattr(tool, "to_openai_schema") else tool
            )
        except Exception:
            continue
    if not schemas:
        return 0
    return estimate_tokens(json.dumps(schemas, ensure_ascii=False, default=str))


def _content_parts(role: str, content: str) -> dict[str, int]:
    """Estimate one message, including workflow history's labelled sections."""
    if not content:
        return {}
    if role == "system":
        return {"system": estimate_tokens(content)}
    if role == "tool":
        return {"tool_results": estimate_tokens(content)}
    if content.casefold().startswith(_SUMMARY_PREFIXES):
        return {"summarized": estimate_tokens(content)}

    matches = list(_WORKFLOW_SECTION_RE.finditer(content))
    if not matches:
        return {"conversation": estimate_tokens(content)}

    out = {"conversation": estimate_tokens(content[:matches[0].start()])}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section = content[match.end():end]
        label = match.group(1)
        if label == "Compacted earlier turns":
            bucket = "summarized"
        elif label.startswith("Query/tool result"):
            bucket = "tool_results"
        else:
            bucket = "conversation"
        out[bucket] = out.get(bucket, 0) + estimate_tokens(section)
    return out


@dataclass(frozen=True)
class ContextBreakdown:
    """Provider-calibrated estimate of the most recently submitted prompt."""

    system: int = 0
    tool_definitions: int = 0
    conversation: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    summarized: int = 0
    other: int = 0
    total: int = 0

    @classmethod
    def estimate(
        cls, messages: Iterable[Message], *, tool_definition_tokens: int = 0,
    ) -> ContextBreakdown:
        counts = {field.name: 0 for field in fields(cls)}
        counts["tool_definitions"] = max(0, int(tool_definition_tokens))
        message_count = 0
        for message in messages:
            message_count += 1
            role = str(message.get("role") or "")
            for bucket, amount in _content_parts(
                role, text_of(message.get("content")),
            ).items():
                counts[bucket] += amount
            calls = message.get("tool_calls") or []
            if calls:
                counts["tool_calls"] += estimate_tokens(
                    json.dumps(calls, ensure_ascii=False, default=str)
                )
        # Chat protocols carry a small role/envelope cost per message.
        counts["other"] = message_count * 4
        counts["total"] = sum(
            value for key, value in counts.items() if key != "total"
        )
        return cls(**counts)

    def calibrated(self, actual_total: int) -> ContextBreakdown:
        """Fit estimated buckets to the exact provider-reported prompt total."""
        actual = max(0, int(actual_total))
        if actual <= 0:
            return self
        names = [field.name for field in fields(self) if field.name != "total"]
        estimated = sum(getattr(self, name) for name in names)
        if estimated <= actual:
            values = {name: getattr(self, name) for name in names}
            values["other"] += actual - estimated
        elif estimated:
            scale = actual / estimated
            values = {name: round(getattr(self, name) * scale) for name in names}
            values["other"] += actual - sum(values.values())
        else:
            values = {name: 0 for name in names}
            values["other"] = actual
        return ContextBreakdown(**values, total=actual)

    def display_categories(self) -> tuple[tuple[str, int], ...]:
        """Stable, compact category set shared by line mode and the modal."""
        return (
            ("System & definitions", self.system + self.tool_definitions + self.other),
            ("Conversation / history", self.conversation),
            ("Tool calls & results", self.tool_calls + self.tool_results),
            ("Summarized history", self.summarized),
        )

    @classmethod
    def from_dict(cls, value: object) -> ContextBreakdown | None:
        if not isinstance(value, dict):
            return None
        try:
            return cls(**{
                field.name: int(value.get(field.name, 0) or 0)
                for field in fields(cls)
            })
        except (TypeError, ValueError):
            return None


def _compact_tokens(value: int, *, binary: bool = False) -> str:
    divisor = 1024 if binary else 1000
    if value < divisor:
        return str(value)
    amount = value / divisor
    digits = 0 if amount >= 10 else 1
    return f"{amount:.{digits}f}k"


@dataclass
class Usage:
    """Running session totals plus the last submitted context snapshot."""

    input: int = 0
    output: int = 0
    cached: int = 0
    last_input: int = 0
    estimated: bool = False
    compactions: int = 0
    breakdown: ContextBreakdown | None = None

    @property
    def total(self) -> int:
        return self.input + self.output

    def context_pct_left(self, window: int) -> int | None:
        if window <= 0 or self.last_input <= 0:
            return None
        return max(0, round((1 - self.last_input / window) * 100))

    def context_status(self, window: int) -> str:
        """Compact ``used/max percent`` text for the responsive status bar."""
        if window <= 0:
            return ""
        maximum = _compact_tokens(window, binary=window % 1024 == 0)
        if self.last_input <= 0:
            return f"--/{maximum}"
        used_pct = min(100, round(self.last_input / window * 100))
        return (
            f"{_compact_tokens(self.last_input)}/"
            f"{maximum} {used_pct}%"
        )

    def summary(self) -> str:
        tag = "≈" if self.estimated else ""
        return (
            f"{tag}{self.total:,} tokens (in {self.input:,} · out {self.output:,}"
            + (f" · cached {self.cached:,}" if self.cached else "") + ")"
        )

    def clear_context(self) -> None:
        """Forget the prompt snapshot while retaining cumulative session usage."""
        self.last_input = 0
        self.breakdown = None

    def context_report(self, window: int, *, output_reserve: int = 0) -> str:
        """Text equivalent of the TUI context visualization."""
        estimate = "≈" if self.estimated else ""
        lines = ["Context"]
        if window > 0 and self.last_input > 0:
            used = min(self.last_input, window)
            available = max(0, window - used)
            used_pct = round(used / window * 100)
            lines.extend((
                f"  Current input     {estimate}{used:,} / {window:,}  {used_pct}%",
                f"  Available         {estimate}{available:,}  {max(0, 100 - used_pct)}%",
            ))
            if self.breakdown is not None:
                lines.append("  Estimated breakdown")
                for label, tokens in self.breakdown.display_categories():
                    pct = round(tokens / window * 100, 1) if window else 0
                    lines.append(f"    {label:<24}{tokens:>10,}  {pct:>5.1f}%")
        elif window > 0:
            lines.extend((
                f"  Current input     unavailable / {window:,}",
                "  Available         unavailable until the first model response",
            ))
        else:
            lines.append("  Current input     unavailable (model window unknown)")
        lines.extend((
            f"  Session total     {estimate}{self.total:,} tokens",
            f"  Input / output    {estimate}{self.input:,} / {self.output:,}",
        ))
        if self.cached:
            lines.append(f"  Cache read        {estimate}{self.cached:,}")
        if output_reserve > 0:
            lines.append(f"  Output reserve    up to {output_reserve:,}")
        compact = (
            f"{self.compactions} time{'s' if self.compactions != 1 else ''}"
            if self.compactions else "not triggered"
        )
        lines.append(f"  Compaction        {compact}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cached": self.cached,
            "last_input": self.last_input,
            "estimated": self.estimated,
            "compactions": self.compactions,
            "breakdown": asdict(self.breakdown) if self.breakdown is not None else None,
        }

    def restore(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        for name in ("input", "output", "cached", "last_input", "compactions"):
            try:
                setattr(self, name, max(0, int(value.get(name, 0) or 0)))
            except (TypeError, ValueError):
                setattr(self, name, 0)
        self.estimated = bool(value.get("estimated", False))
        self.breakdown = ContextBreakdown.from_dict(value.get("breakdown"))


class UsageObserver(BaseObserver):
    critical = False

    def __init__(self, usage: Usage, *, tools: Iterable[Any] = ()) -> None:
        self.u = usage
        self._tool_definition_tokens = _tool_schema_tokens(tools)

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        d = getattr(ctx, "usage", None) or {}
        prompt = int(d.get("prompt_tokens", 0) or 0)
        completion = int(d.get("completion_tokens", 0) or 0)
        self.u.input += prompt
        self.u.output += completion
        self.u.cached += int(d.get("cached_tokens", 0) or 0)
        # The loop appends this response's assistant message before notifying
        # observers. Everything before that final assistant turn is precisely
        # the submitted history; trailing synthetic tool errors belong to the
        # response and are excluded with it. Reconstructing here avoids mutable
        # cross-hook state in this deliberately passive observer.
        request_messages = list(ctx.messages)
        response_index = next(
            (
                index for index in range(len(request_messages) - 1, -1, -1)
                if request_messages[index].get("role") == "assistant"
            ),
            len(request_messages),
        )
        snapshot = ContextBreakdown.estimate(
            request_messages[:response_index],
            tool_definition_tokens=self._tool_definition_tokens,
        )
        actual = prompt or snapshot.total
        if actual:
            self.u.last_input = actual
            self.u.breakdown = snapshot.calibrated(actual)
        if not d or d.get("estimated") or not prompt:
            self.u.estimated = True
        return None


__all__ = ["ContextBreakdown", "Usage", "UsageObserver"]
