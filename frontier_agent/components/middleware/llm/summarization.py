from __future__ import annotations

import logging
import re
from typing import Any

from frontier_agent.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.messages import (
    Message,
    system_msg,
    text_of,
    user_msg,
)

logger = logging.getLogger(__name__)


class SummarizationMiddleware(LLMMiddleware):
    """Compresses message history when token count exceeds threshold.

    Token counting strategy (ordered by preference):
    1. tiktoken (cl100k_base) — accurate for GPT-4/Claude-class models
    2. Heuristic fallback — regex-based CJK detection with per-message overhead
    """

    # Broad CJK regex: Unified Ideographs, Ext-A, radicals, strokes,
    # Hiragana, Katakana, CJK compatibility, fullwidth forms
    _CJK_RE = re.compile(
        r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf"
        r"\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
    )

    _MSG_OVERHEAD = 4  # tokens per message for role/formatting markers

    def __init__(
        self,
        threshold: int = 80000,
        keep_recent: int = 6,
        summary_llm: LLMClient | None = None,
    ) -> None:
        self._threshold = threshold
        self._keep_recent = keep_recent
        self._summary_llm = summary_llm  # raw LLM, bypasses proxy

    @property
    def name(self) -> str:
        return "summarization"

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count. Uses tiktoken when available, else heuristic.

        The encoder loads on a daemon thread (see ``tokenizer.py``); until
        it lands ``get_encoding_nonblocking`` returns ``None`` and we use
        the heuristic — the loop thread never blocks on a network fetch.
        """
        from frontier_agent.core.runtime.loop.tokenizer import get_encoding_nonblocking
        encoder = get_encoding_nonblocking("cl100k_base")
        if encoder is not None:
            return self._estimate_tiktoken(encoder, messages)
        return self._estimate_heuristic(messages)

    def _estimate_tiktoken(self, encoder: Any, messages: list[Message]) -> int:
        """Accurate token count via tiktoken cl100k_base encoding."""
        try:
            total = 0
            for m in messages:
                total += len(
                    encoder.encode(
                        text_of(m.get("content")), disallowed_special=()
                    )
                ) + self._MSG_OVERHEAD
            return total
        except Exception:
            return self._estimate_heuristic(messages)

    def _estimate_heuristic(self, messages: list[Message]) -> int:
        """Regex-based heuristic for mixed CJK/English text."""
        total = 0
        for m in messages:
            text = text_of(m.get("content"))
            cjk_count = len(self._CJK_RE.findall(text))
            other_count = len(text) - cjk_count
            total += cjk_count + (other_count // 4) + self._MSG_OVERHEAD
        return total

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message]
    ) -> list[Message]:
        # Per-node config overrides global defaults
        comp = ctx.metadata.get("compression") if ctx.metadata else None
        if isinstance(comp, dict):
            if not comp.get("enabled", True):
                return messages
            threshold = comp.get("threshold", self._threshold)
            keep_recent = comp.get("keep_recent", self._keep_recent)
        else:
            threshold = self._threshold
            keep_recent = self._keep_recent

        token_est = self._estimate_tokens(messages)
        if token_est <= threshold or len(messages) <= keep_recent + 1:
            return messages

        # Split: system + middle (to summarize) + recent (to keep)
        system_msgs = [m for m in messages[:1] if m.get("role") == "system"]
        rest = messages[len(system_msgs):]

        # Advance the split point past any tool message that would
        # otherwise start the kept window — an orphan tool message
        # (without its matching assistant tool_calls) causes Azure to
        # return HTTP 400 "No tool call found for function call output
        # with call_id ...".
        split_idx = len(rest) - keep_recent
        while split_idx < len(rest) - 1 and rest[split_idx].get("role") == "tool":
            split_idx += 1
        to_summarize = rest[:split_idx]
        keep = rest[split_idx:]

        if not to_summarize:
            return messages

        # Build summary
        if self._summary_llm:
            summary_text = await self._generate_summary(to_summarize)
        else:
            # Fallback: truncate without LLM
            summary_text = self._truncate_summary(to_summarize)

        summary_msg = user_msg(
            f"[Previous conversation summary ({len(to_summarize)} messages compressed)]\n{summary_text}"
        )
        result = [*system_msgs, summary_msg, *keep]
        logger.info(
            "SummarizationMiddleware: compressed %d→%d messages (est %d→%d tokens)",
            len(messages), len(result), token_est, self._estimate_tokens(result),
        )
        return result

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Use raw LLM (no proxy) to summarize messages.

        Falls back to truncation if LLM call fails or times out.
        """
        # Callers reach here only under ``if self._summary_llm``, but that guard
        # lives at the call site; binding it locally both narrows the Optional
        # and makes the no-LLM path explicit rather than an AttributeError.
        summary_llm = self._summary_llm
        if summary_llm is None:
            return self._truncate_summary(messages)

        content_parts = []
        for m in messages:
            role = m.get("role", "?")
            text = text_of(m.get("content"))[:500]
            content_parts.append(f"[{role}] {text}")
        joined = "\n".join(content_parts[-20:])  # cap input to summary LLM

        try:
            import asyncio
            resp = await asyncio.wait_for(
                summary_llm.chat([
                    system_msg(
                        "Summarize the following conversation concisely, preserving "
                        "all key findings, tool results, decisions, and pending questions. "
                        "Output as 3-8 bullet points. Be specific — include names, numbers, "
                        "and URLs when present."
                    ),
                    user_msg(joined),
                ]),
                timeout=30,
            )
            return text_of(resp.content)
        except Exception as e:
            logger.warning("SummarizationMiddleware: LLM summary failed (%s), falling back to truncation", e)
            return self._truncate_summary(messages)

    def _truncate_summary(self, messages: list[Message]) -> str:
        """Simple truncation fallback when no summary LLM is available."""
        parts = []
        for m in messages[-5:]:
            role = m.get("role", "?")
            text = text_of(m.get("content"))[:200]
            parts.append(f"- {role}: {text}")
        return "\n".join(parts)
