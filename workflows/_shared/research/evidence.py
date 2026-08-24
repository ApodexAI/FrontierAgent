"""Evidence and assertion models for the research pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NewType
from uuid import uuid4

from pydantic import BaseModel, Field

# ── Research-specific identity types (relocated from core.types) ────────────
# These IDs belong to the research domain (pipeline layer), not the kernel
# primitives in core.types. Keep them close to the EvidenceCard / Assertion
# models that actually produce and consume them.

EvidenceId = NewType("EvidenceId", str)
AssertionId = NewType("AssertionId", str)


def new_evidence_id() -> EvidenceId:
    return EvidenceId(f"ev-{uuid4().hex[:8]}")


def new_assertion_id() -> AssertionId:
    return AssertionId(f"as-{uuid4().hex[:8]}")


class Source(BaseModel):
    """A web source where evidence was found."""

    url: str
    title: str = ""
    snippet: str = ""
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reliability_score: float = 0.5  # 0-1, assessed by Critic


class EvidenceCard(BaseModel):
    """A unit of evidence supporting or contradicting a claim.

    Created by Researcher during the Collect phase, evaluated by Critic
    during the Analyze phase.
    """

    id: EvidenceId = Field(default_factory=new_evidence_id)
    claim: str  # the factual claim this evidence supports
    source: Source
    supporting_quotes: list[str] = Field(default_factory=list)
    relevance_score: float = 0.5  # 0-1
    contradiction_notes: str = ""  # notes from Critic if contradictions found
    query: str = ""  # the search query that produced this evidence


class Assertion(BaseModel):
    """A conclusion in the final report, backed by evidence.

    Every assertion must reference at least one EvidenceCard.
    The Critic may add counter-evidence references.
    """

    id: AssertionId = Field(default_factory=new_assertion_id)
    statement: str
    confidence: float = 0.5  # 0-1
    supporting_evidence: list[str] = Field(default_factory=list)  # EvidenceCard IDs
    counter_evidence: list[str] = Field(default_factory=list)  # EvidenceCard IDs
    is_disputed: bool = False  # flagged by Critic
