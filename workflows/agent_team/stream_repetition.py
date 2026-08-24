"""Stream-repetition wiring for agent-team / heavy-mode LLM calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from frontier_agent.core.loop_types import LLMDeltaContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamRepetitionConfig:
    """Parsed ``stream_repetition`` profile section."""

    min_pattern_len: int | None = None
    max_pattern_len: int | None = None
    min_repeats: int | None = None
    min_text_len: int | None = None
    check_interval: int | None = None

    def detector_kwargs(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key in (
            "min_pattern_len",
            "max_pattern_len",
            "min_repeats",
            "min_text_len",
            "check_interval",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = int(value)
        return out


class StreamRepetitionDeltaSink:
    """No-op observer that forces ``run_agent_loop`` onto ``astream``.

    ``StreamRepetitionDetectorMiddleware`` only receives chunks on the
    streaming path. This observer does not emit user-visible deltas; it
    only flips the loop's ``wants_llm_delta`` probe.
    """

    critical = False
    wants_llm_delta = True

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> None:
        del ctx


def parse_stream_repetition_config(agent_cfg: dict[str, Any]) -> StreamRepetitionConfig | None:
    """Return detector config when ``stream_repetition.enabled`` is true."""

    raw = agent_cfg.get("stream_repetition") or {}
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    kwargs: dict[str, int] = {}
    for key in (
        "min_pattern_len",
        "max_pattern_len",
        "min_repeats",
        "min_text_len",
        "check_interval",
    ):
        if key in raw:
            kwargs[key] = int(raw[key])
    return StreamRepetitionConfig(**kwargs)


def wrap_llm_for_stream_repetition(
    llm: Any,
    *,
    config: StreamRepetitionConfig | None,
    role_id: str,
    label: str = "",
) -> tuple[Any, StreamRepetitionDeltaSink | None]:
    """Wrap an LLM in ``LLMProxy`` with chunk-level repetition detection.

    Returns ``(llm, None)`` when disabled. When enabled, callers must add
    the returned observer to the loop observer stack so ``run_agent_loop``
    streams and the proxy's ``on_chunk`` hook fires.
    """

    if config is None:
        return llm, None

    from frontier_agent.components.middleware.llm.base import LLMMiddlewareChain
    from frontier_agent.components.middleware.llm.proxy import LLMProxy
    from frontier_agent.components.middleware.llm.stream_repetition import (
        StreamRepetitionDetectorMiddleware,
    )

    detector_kwargs = config.detector_kwargs()
    chain = LLMMiddlewareChain()
    chain.add(StreamRepetitionDetectorMiddleware(**detector_kwargs))
    logger.info(
        "stream-repetition detector ON for role=%s label=%s kwargs=%s",
        role_id,
        label or "-",
        detector_kwargs or "defaults",
    )
    return LLMProxy(inner=llm, chain=chain, role_id=role_id), StreamRepetitionDeltaSink()


__all__ = [
    "StreamRepetitionConfig",
    "StreamRepetitionDeltaSink",
    "parse_stream_repetition_config",
    "wrap_llm_for_stream_repetition",
]
