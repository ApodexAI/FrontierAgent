"""FrontierAgent adapter for :mod:`agent_core.runtime.loop.tiered_compact`.

Two product defaults on top of the shared mechanics: the trigger ratio is
overridable through ``AGENT_COMPACTION_TRIGGER_RATIO`` (AgentCore takes it as an
explicit argument), and the summary prompt comes from the product config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import agent_core.runtime.loop.tiered_compact as _shared
from agent_core.runtime.loop.tiered_compact import (
    DEFAULT_TRIGGER_RATIO,
    InputTokenGauge,
    InputTokenThresholdPolicy,
)

from frontier_agent.infra.llm.summary_prompt import compaction_prompt

__all__ = [
    "InputTokenGauge",
    "InputTokenThresholdPolicy",
    "TieredCompactor",
    "compaction_trigger_ratio",
    "compaction_trigger_tokens",
]

logger = logging.getLogger(__name__)

_TRIGGER_RATIO_ENV = "AGENT_COMPACTION_TRIGGER_RATIO"


def compaction_trigger_ratio() -> float:
    """The trigger ratio, honouring ``AGENT_COMPACTION_TRIGGER_RATIO``."""
    raw = os.getenv(_TRIGGER_RATIO_ENV)
    if not raw:
        return DEFAULT_TRIGGER_RATIO
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number — using %.2f", _TRIGGER_RATIO_ENV, raw, DEFAULT_TRIGGER_RATIO
        )
        return DEFAULT_TRIGGER_RATIO
    if not 0.0 < parsed < 1.0:
        logger.warning(
            "%s=%r is outside (0, 1) — using %.2f", _TRIGGER_RATIO_ENV, raw, DEFAULT_TRIGGER_RATIO
        )
        return DEFAULT_TRIGGER_RATIO
    return parsed


def compaction_trigger_tokens(max_len: int) -> int:
    """The absolute token threshold tiered compaction should fire at."""
    return _shared.compaction_trigger_tokens(max_len, compaction_trigger_ratio())


class TieredCompactor(_shared.TieredCompactor):
    """Default the summary prompt to FrontierAgent's configured style."""

    def __init__(self, *args: Any, prompt_builder: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, prompt_builder=prompt_builder or compaction_prompt, **kwargs)


def __getattr__(name: str) -> Any:
    """Read-through to the shared module (patch ``agent_core`` to rebind)."""
    return getattr(_shared, name)
