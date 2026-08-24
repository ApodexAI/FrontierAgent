"""Build a protocol-clean finalization request from a damaged history."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from frontier_agent.components.observers.leaked_tool_call_retry import (
    LEAKED_TOOL_CALL_NUDGE,
)
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.messages import Message, text_of
from frontier_agent.core.runtime.loop.context_budget import (
    truncate_text_to_tokens as _truncate_text_to_tokens,
)
from frontier_agent.core.runtime.loop.llm_client import estimate_text_tokens

RECOVERY_CONTEXT_MAX_CHARS = 60_000
RECOVERY_ITEM_MAX_CHARS = 8_000

# Framework-injected control messages: they steer the *loop*, and replaying them
# inside a finalization prompt makes them compete with the finalization
# instruction. Workflows extend this with their own nudges.
COMMON_RECOVERY_NUDGE_PREFIXES: tuple[str, ...] = (
    "Please provide your final answer now",
    "Your response was empty",
    "Finalization phase has started.",
    "The execution budget is entering its finalization reserve.",
    "Time budget nearly exhausted",
    "This is your penultimate turn.",
    LEAKED_TOOL_CALL_NUDGE,
)

DEFAULT_RECOVERY_LABELS: Mapping[str, str] = {
    "system": "System guidance",
    "assistant": "Visible agent draft",
    "tool": "Observed tool result",
    "user": "User instruction",
}

_CJK_RE = re.compile(r"[㐀-鿿]")
_TRUNCATION_MARKER = "\n[... older context truncated for reporter input limit ...]"


def is_recovery_nudge(content: str, prefixes: Iterable[str]) -> bool:
    """Return whether a user-role message is framework finalization control."""
    stripped = (content or "").lstrip()
    if stripped.startswith("Warning: ~") and "wall-clock deadline" in stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in prefixes)


def fallback_leg_count(llm: object) -> int:
    """Best-effort number of finite legs available on an LLM fallback chain."""
    entries = getattr(llm, "entries", None)
    return max(1, len(entries)) if isinstance(entries, (list, tuple)) else 1


async def chat_with_fallback_budget(
    llm: LLMClient,
    messages: list[Message],
    *,
    per_leg_timeout_s: float,
) -> Any:
    """Run chat with a per-leg timeout and a finite whole-chain backstop.

    The outer guard exists because a custom client is free to ignore
    ``timeout=``; without it a single hung leg would hang the whole rescue.
    """
    timeout_s = max(float(per_leg_timeout_s), 1.0)
    outer_timeout_s = timeout_s * fallback_leg_count(llm) + 5.0
    chat = llm.chat
    return await asyncio.wait_for(
        chat(messages, timeout=timeout_s),
        timeout=outer_timeout_s,
    )


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Keep the largest safe prefix under ``max_tokens``."""
    return _truncate_text_to_tokens(
        text,
        max_tokens,
        marker=_TRUNCATION_MARKER,
        estimator=estimate_text_tokens,
    )


def build_recovery_context(
    messages: Sequence[object],
    *,
    strip_thinking: Callable[[str], str],
    strip_leaked_tool_calls: Callable[[str], str],
    nudge_prefixes: Iterable[str],
    empty_fallback: str,
    labels: Mapping[str, str] = DEFAULT_RECOVERY_LABELS,
    fixed_prompt_text: str = "",
    context_max_chars: int = RECOVERY_CONTEXT_MAX_CHARS,
    item_max_chars: int = RECOVERY_ITEM_MAX_CHARS,
    context_max_tokens: int | None = None,
) -> str:
    """Flatten history into one labelled plain-text block.

    System guidance is pinned first (behavioural/safety constraints must survive
    truncation pressure), then the most recent entries fill the remaining
    budget. Assistant reasoning is stripped — hidden chain-of-thought must never
    become user-visible recovery context.

    Budgeting has two modes. With ``context_max_tokens`` the budget is measured
    in tokens against the model's input limit and the first entry that does not
    fit is truncated to fill it exactly; otherwise entries are capped at
    ``item_max_chars`` each and selected against ``context_max_chars``.
    """
    system_entries: list[str] = []
    entries: list[str] = []
    prefixes = tuple(nudge_prefixes)
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = text_of(message.get("content"))
        if role == "system":
            pass
        elif role == "assistant":
            content = strip_thinking(content)
        elif role == "user":
            # The original task is supplied authoritatively by the caller's
            # prompt; replaying loop-control nudges here only competes with it.
            if is_recovery_nudge(content, prefixes):
                continue
        elif role != "tool":
            continue
        content = strip_leaked_tool_calls(content).strip()
        if not content:
            continue
        if context_max_tokens is None:
            content = content[:item_max_chars]
        entry = f"[{labels.get(role, role)}]\n{content}"
        if role == "system":
            system_entries.append(entry)
        else:
            entries.append(entry)

    selected_system: list[str] = []
    selected_recent: list[str] = []
    if context_max_tokens is None:
        used = 0
        for entry in system_entries:
            if selected_system and used + len(entry) > context_max_chars:
                break
            selected_system.append(entry)
            used += len(entry)
        for entry in reversed(entries):
            if selected_recent and used + len(entry) > context_max_chars:
                break
            if used + len(entry) > context_max_chars:
                continue
            selected_recent.append(entry)
            used += len(entry)
    else:
        remaining_tokens = max(
            1_024,
            int(context_max_tokens) - estimate_text_tokens(fixed_prompt_text),
        )
        for entry in system_entries:
            entry_tokens = estimate_text_tokens(entry)
            if entry_tokens <= remaining_tokens:
                selected_system.append(entry)
                remaining_tokens -= entry_tokens
                continue
            if not selected_system and remaining_tokens > 128:
                selected_system.append(
                    truncate_text_to_tokens(entry, remaining_tokens),
                )
                remaining_tokens = 0
            break
        for entry in reversed(entries):
            entry_tokens = estimate_text_tokens(entry)
            if entry_tokens <= remaining_tokens:
                selected_recent.append(entry)
                remaining_tokens -= entry_tokens
                continue
            if not selected_recent and remaining_tokens > 128:
                selected_recent.append(
                    truncate_text_to_tokens(entry, remaining_tokens),
                )
            break
    selected_recent.reverse()
    return "\n\n".join([*selected_system, *selected_recent]) or empty_fallback


def has_malformed_tool_protocol(messages: Sequence[object]) -> bool:
    """Return whether replaying ``messages`` would violate tool-call protocol.

    A healthy assistant tool-call turn must contain well-formed calls followed
    immediately by exactly one tool response for every call id. Tool messages
    outside such a block are orphaned. This deliberately stays narrow so
    ordinary runs retain the original role structure.
    """
    expected: set[str] = set()
    seen: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            return True
        role = str(message.get("role") or "")
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            if not call_id or call_id not in expected or call_id in seen:
                return True
            seen.add(call_id)
            continue

        if expected:
            if seen != expected:
                return True
            seen.clear()

        calls = message.get("tool_calls")
        if not calls:
            expected = set()
            continue
        if role != "assistant" or not isinstance(calls, list):
            return True
        call_ids: set[str] = set()
        for call in calls:
            if not isinstance(call, dict):
                return True
            call_id = str(call.get("id") or "").strip()
            function = call.get("function")
            if (
                not call_id
                or call_id in call_ids
                or not isinstance(function, dict)
                or not str(function.get("name") or "").strip()
                or function.get("arguments") in (None, "")
            ):
                return True
            call_ids.add(call_id)
        expected = call_ids

    return bool(expected and seen != expected)


def minimal_best_effort_answer(
    task_description: str,
    stopped_by: str,
    *,
    language: str = "",
) -> str:
    """Always provide a user-facing answer even when the rescue LLM is down.

    ``language`` is the run's already-resolved answer language label (e.g.
    ``"Simplified Chinese"``); it wins over sniffing the task text, so a
    Chinese-speaking user asking an English-worded question still gets Chinese.
    """
    label = (language or "").strip().lower()
    if label:
        chinese = "chinese" in label or label in {"zh", "zh-cn", "zh-hans", "中文"}
    else:
        chinese = bool(_CJK_RE.search(task_description or ""))
    marker = stopped_by or "execution_limit"
    if chinese:
        return (
            "## 当前可交付结果\n\n"
            "执行已按现有进度提前收尾；工作区和输出目录中已经生成的文件均予以保留。"
            "由于最终汇总未能完成，这里不对尚未验证的结果做断言。请以现有交付物为准；"
            "若输出目录为空，则本次任务尚未形成可验证的完整交付——仍然返回这一明确"
            f"状态而不是空答案（终止标记：`{marker}`）。"
        )
    return (
        "## Best available result\n\n"
        "Execution was finalized from the progress available at the limit. "
        "Any files already present in the workspace and output directory have "
        "been preserved. The final synthesis did not yield a reliable verified "
        "conclusion, so no unsupported result is asserted here. If the output "
        "directory is empty, the task did not reach a verifiable complete "
        f"deliverable (stop marker: `{marker}`)."
    )


__all__ = [
    "COMMON_RECOVERY_NUDGE_PREFIXES",
    "DEFAULT_RECOVERY_LABELS",
    "RECOVERY_CONTEXT_MAX_CHARS",
    "RECOVERY_ITEM_MAX_CHARS",
    "build_recovery_context",
    "chat_with_fallback_budget",
    "fallback_leg_count",
    "has_malformed_tool_protocol",
    "is_recovery_nudge",
    "minimal_best_effort_answer",
    "truncate_text_to_tokens",
]
