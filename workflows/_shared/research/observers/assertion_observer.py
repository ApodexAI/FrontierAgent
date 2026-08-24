"""AssertionObserver — synthesize assertions from evidence on loop end.

Uses cluster-based grouping: evidence cards grouped by query,
each group becomes one assertion. No LLM call needed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from frontier_agent.core.loop_types import AgentLoopResult, BaseObserver


class AssertionObserver(BaseObserver):
    """Synthesize assertions from evidence_cards at loop end.

    critical=True because callers read metadata["assertions"] from the result.
    """

    critical: bool = True

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        evidence_cards: list[dict] = result.metadata.get("evidence_cards", [])
        assertions = _cluster_assertions(evidence_cards)

        # Fallback: single assertion from final_content if no evidence
        if not assertions and result.final_content:
            assertions = [{
                "id": "as-react-001",
                "statement": result.final_content[:500],
                "confidence": 0.5,
                "supporting_evidence": [],
                "counter_evidence": [],
                "is_disputed": False,
            }]

        result.metadata["assertions"] = assertions


def _default_confidence(avg_rel: float, count: int) -> float:
    """Confidence formula for research assertions."""
    conf = avg_rel * 0.7 + min(count, 10) * 0.02
    return round(max(0.1, min(0.80, conf)), 2)


def _benchmark_confidence(avg_rel: float, count: int) -> float:
    """Simpler confidence formula for benchmark solvers."""
    return round(min(0.75, 0.4 + count * 0.05), 2)


def _cluster_assertions(
    evidence_cards: list[dict],
    confidence_fn: Any = None,
) -> list[dict]:
    """Group evidence by query -> one assertion per cluster."""
    if not evidence_cards:
        return []

    if confidence_fn is None:
        confidence_fn = _default_confidence

    clusters: dict[str, list[dict]] = defaultdict(list)
    for card in evidence_cards:
        key = card.get("query", "").strip().lower()[:80] or "general"
        clusters[key].append(card)

    assertions: list[dict[str, Any]] = []
    for i, (query, cards) in enumerate(clusters.items()):
        claims = [
            c.get("claim", "")[:100]
            for c in cards[:3] if c.get("claim")
        ]
        statement = (
            f"{query}: " + "; ".join(claims) if claims else query
        )
        avg_rel = sum(
            float(c.get("relevance_score", 0.3) or 0.3)
            for c in cards
        ) / len(cards)

        assertions.append({
            "id": f"as-ev-{i + 1:03d}",
            "statement": statement[:300],
            "confidence": confidence_fn(avg_rel, len(cards)),
            "supporting_evidence": [c["id"] for c in cards[:15]],
            "counter_evidence": [],
            "is_disputed": False,
        })
    return assertions


def extract_assertions_from_response(
    response_text: str,
    evidence_cards: list[dict],
) -> list[dict]:
    """Lightweight assertion placeholder for benchmark solvers."""
    if not response_text or not evidence_cards:
        return []
    return _cluster_assertions(evidence_cards, _benchmark_confidence)
