"""submit_report — terminal tool for sub-agents (sibling of finalize_answer)."""

from __future__ import annotations

import logging

from frontier_agent.core.tool import tool

logger = logging.getLogger(__name__)


@tool
async def submit_report(content: str, confidence: float = 0.7) -> str:
    """Submit your final research report and end this task.

    Call this **once** when you have gathered enough evidence and are ready
    to return to the main agent. The solver loop terminates after this turn.

    Args:
        content: The complete report in the mandatory format:
            ``Scope: ...\\nFinding: ...\\nEvidence:\\n  - ...\\nConfidence: ...``.
            Include exact values, source URLs, and code results — not just
            conclusions.
        confidence: Self-assessed confidence in [0, 1] that your findings
            are correct and well-sourced. Default 0.7.

    Returns:
        Short acknowledgement. The report body is surfaced to the main
        agent via ``<report agent="NAME">...</report>``.
    """
    text = (content or "").strip()
    if not text:
        return (
            "submit_report rejected: `content` was empty. "
            "Pass the complete Scope/Finding/Evidence report in `content`."
        )

    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.7

    logger.info("submit_report called: len=%d chars, confidence=%.2f", len(text), conf)
    return (
        f"Report submitted ({len(text)} chars, confidence={conf:.2f}). "
        "Task loop will exit after this turn."
    )
