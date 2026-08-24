"""Record bounded, JSON-safe tool-call previews in ``react_steps``.

Caps prevent multi-agent runs from retaining unbounded tool output; raw trace
observers may preserve full payloads separately.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

from frontier_agent.core.loop_types import BaseObserver, ToolResult, TurnContext

logger = logging.getLogger(__name__)

# Per-field caps. Generous enough that a human skimming a step sees the
# useful head of the output, small enough that turns × agents stays in the
# low MB. 0 disables the cap for that field.
DEFAULT_RESULT_MAX_CHARS = 4096
DEFAULT_THINKING_MAX_CHARS = 4096
DEFAULT_ARGS_MAX_CHARS = 8192

# Deprecated API consumers still parse ``tool_args`` and look up these
# locator fields directly.  Keep them at the top level of a truncation
# envelope so valid JSON does not silently hide the artifact path.
_PRESERVED_ARG_KEYS = ("path", "file_path", "image_path_or_url")
_PRESERVED_ARG_VALUE_MAX_CHARS = 1024


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "ReactStepTracker: ignoring non-numeric %s=%r", name, raw,
        )
        return default
    return max(0, value)


# Cached: these are process-lifetime constants, and ``on_tool_result`` runs
# on every tool call of every agent — re-reading os.environ there is pure
# overhead.
@functools.lru_cache(maxsize=1)
def _caps() -> tuple[int, int, int]:
    """``(result_max, thinking_max, args_max)`` in chars; 0 = unbounded."""
    return (
        _env_int(
            "FRONTIER_AGENT_REACT_STEP_RESULT_MAX_CHARS", DEFAULT_RESULT_MAX_CHARS,
        ),
        _env_int(
            "FRONTIER_AGENT_REACT_STEP_THINKING_MAX_CHARS",
            DEFAULT_THINKING_MAX_CHARS,
        ),
        _env_int(
            "FRONTIER_AGENT_REACT_STEP_ARGS_MAX_CHARS", DEFAULT_ARGS_MAX_CHARS,
        ),
    )


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _clip(text: str, limit: int) -> str:
    """Head-slice *text* to *limit* chars with an explicit marker.

    The marker names the dropped byte count so a reader can tell a
    genuinely short result from a truncated one — a bare slice looks like
    the tool simply returned less.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n"
        f"... [truncated {len(text) - limit} of {len(text)} chars]"
    )


def _clip_args(args: Any, limit: int) -> str:
    """JSON-encode *args*, staying parseable even when over *limit*.

    Consumers ``json.loads`` this field, so an over-long value is swapped
    for a valid envelope instead of being sliced into invalid JSON.  Small
    artifact-locator fields remain available at the envelope's top level
    for legacy consumers that call ``args.get("path")``.
    """
    encoded = json.dumps(args, ensure_ascii=False, default=str)
    if limit <= 0 or len(encoded) <= limit:
        return encoded
    envelope: dict[str, Any] = {
        "_truncated": True,
        "_original_chars": len(encoded),
        "_preview": encoded[:limit],
    }
    if isinstance(args, dict):
        for key in _PRESERVED_ARG_KEYS:
            value = args.get(key)
            if (
                isinstance(value, (str, int, float, bool))
                and len(str(value)) <= _PRESERVED_ARG_VALUE_MAX_CHARS
            ):
                envelope[key] = value
    return json.dumps(envelope, ensure_ascii=False)


class ReactStepTracker(BaseObserver):
    """Records each tool result as a react_step dict in metadata."""

    critical: bool = True

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> None:
        result_max, thinking_max, args_max = _caps()
        steps: list[dict[str, Any]] = ctx.metadata.setdefault(
            "react_steps", [],
        )
        step: dict[str, Any] = {
            "turn": ctx.turn,
            "thinking": _clip(ctx.thinking or "", thinking_max),
            "tool_name": result.name,
            "tool_args": _clip_args(result.args, args_max),
            "tool_result": _clip(_safe_str(result.result), result_max),
            "duration_ms": result.duration_ms,
            "is_error": result.is_error,
        }
        # Salvaged thinking from leaked tags — recorded alongside the
        # native thinking field so report-rendering / debugging tools can
        # inspect both without conflating provenance.
        if ctx.leaked_reasoning:
            step["leaked_reasoning"] = _clip(
                ctx.leaked_reasoning, thinking_max,
            )
        steps.append(step)
