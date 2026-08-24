"""Two-call, canonically cited reporter for agent-team runs.

One call reviews existing evidence and one writes the report. Failures preserve
the coordinator answer, and no trajectory files are required.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from frontier_agent.utils.language import detect_language
from workflows.agent_team.fast_reporter_v1_evidence import (
    join_reference_rows,
    project_evidence,
    review_candidates,
)
from workflows.agent_team.prompts_fast_reporter_v1 import WRITER_ADDENDUM
from workflows.agent_team.prompts_report_fast import (
    REPORT_FAST_SYSTEM_PROMPT,
    build_user_prompt,
)
from workflows.agent_team.report_fast_guard import (
    DeliverablesLostError,
    deliverable_metrics,
    deliverables_lost,
)
from workflows.agent_team.report_fast_select import prepare_candidates

logger = logging.getLogger(__name__)

CallLLM = Callable[[str, str], Awaitable[str]]

# the citation_contract_cap default. Gate 1 quality-ranks
# before capping, so 30 keeps the whitelist to the sources the reviewer
# rated highest.
DEFAULT_CITATION_CAP = 30

_REPORT_LLM_TIMEOUT_S = 200.0


@dataclass
class FastReportResult:
    """Outcome of one report generation.

    ``preclean`` is the body exactly as the writer submitted it, before
    ``clean_report_v3``.

    ``metrics`` is the guard's ``deliverable_metrics(...)`` output for this
    run, plus three review-call stats folded in: ``candidates_considered``
    (how many survived ``prepare_candidates``), ``references_selected``
    (how many rows the writer actually saw, i.e. ``len(references)``), and
    ``selection_fallback`` (falsy when the review call was used, else a
    short reason string for why it fell back).
    """

    report: str
    references: list[dict[str, str]] = field(default_factory=list)
    repaired: bool = False
    preclean: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class NoAdmissibleEvidenceError(RuntimeError):
    """Raised when no evidence card passes the gate.

    With an empty reference table the writer cannot cite anything, so the
    rewrite would strictly lose to the native answer. Callers fail open.
    """


def resolve_language(language: str, native_draft: str) -> str:
    """Resolve the label Gate 3 uses to pick its References heading.

    ``is_chinese_label`` (``frontier_agent/utils/language.py``) returns False
    for ``"auto"`` — it neither starts with ``zh`` nor contains ``chinese``
    — so passing ``"auto"`` straight through renders ``## References`` on a
    Chinese report. Most real agent_team traces are CJK-majority, so this
    detects rather than defaults. An explicit non-``auto`` label always
    wins.
    """
    label = str(language or "").strip()
    if label and label.lower() != "auto":
        return label
    return detect_language(native_draft or "")


def resolve_citation_cap(metadata: dict[str, Any]) -> int:
    """Citation cap: metadata -> env -> :data:`DEFAULT_CITATION_CAP`.

    Out-of-range or unparseable values fall back to the default rather than
    raising: a bad env var should not take the report node down, since the
    whole chain is fail-open.
    """
    raw = str(
        (metadata or {}).get("fast_report_citation_cap")
        or (metadata or {}).get("fast_reporter_v1_citation_cap")
        or os.environ.get("AGENT_TEAM_FAST_REPORT_CITATION_CAP")
        or os.environ.get("AGENT_TEAM_FAST_REPORTER_V1_CITATION_CAP")
        or "",
    ).strip()
    if not raw:
        return DEFAULT_CITATION_CAP
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("fast_reporter_v1: bad citation cap %r, using default", raw)
        return DEFAULT_CITATION_CAP
    if not 1 <= value <= 500:
        logger.warning(
            "fast_reporter_v1: citation cap %d out of range, using default", value,
        )
        return DEFAULT_CITATION_CAP
    return value


async def generate_fast_reporter_v1_report(
    *,
    question: str,
    native_draft: str,
    evidence_cards: list[dict[str, Any]],
    call_llm: CallLLM,
    language: str = "auto",
    cap: int = DEFAULT_CITATION_CAP,
) -> FastReportResult:
    """Two LLM calls in, canonically-cited report out.

    Raises ``NoAdmissibleEvidenceError`` when the whitelist comes out empty
    (an uncitable rewrite strictly loses to the native answer) and
    ``DeliverablesLostError`` when the rewrite dropped every ``/outputs/``
    path or every code block the draft had. Callers own the fail-open
    decision.
    """
    from workflows._shared.citation_contract import (
        backfill_reference_titles_from_snapshots,
        build_report_references_from_methods_info,
        compose_citation_contract,
        finalize_report_with_canonical_references,
    )
    from workflows._shared.report_clean import (
        clean_report_v3,
        with_code_protected,
    )

    candidates = prepare_candidates(evidence_cards)
    nodes, review_stats = await review_candidates(
        question=question,
        native_draft=native_draft,
        candidates=candidates,
        call_llm=call_llm,
        cap=cap,
    )

    methods_info, run_snapshots = project_evidence(nodes)
    references = build_report_references_from_methods_info(methods_info, cap=cap)
    references = backfill_reference_titles_from_snapshots(references, run_snapshots)
    if not references:
        raise NoAdmissibleEvidenceError(
            f"gate 1 admitted no source among {len(evidence_cards or [])} cards",
        )

    prompt_rows = join_reference_rows(references, nodes, tag_citations=True)
    user_prompt = build_user_prompt(
        question=question,
        native_draft=native_draft,
        references=prompt_rows,
    )
    system_prompt = (
        REPORT_FAST_SYSTEM_PROMPT
        + WRITER_ADDENDUM
        + compose_citation_contract(references)
    )

    raw = await call_llm(system_prompt, user_prompt)
    preclean = raw if isinstance(raw, str) else str(raw or "")

    # internal_node_ids is empty on purpose: this chain mints no DAG ids, so
    # clean_report_v3's node-id scrub is a documented no-op and only its
    # code-fence-aware cleaning applies.
    cleaned, _stats = clean_report_v3(preclean, internal_node_ids=set())

    resolved_language = resolve_language(language, native_draft)
    # Gate 3 renumbers every bracketed ``[N]`` group document-wide with a
    # plain string replace — it has no notion of fenced/inline code. Without
    # this wrapper a numeric literal like ``data[1, 2]`` inside a code block
    # gets split, remapped and rewritten right alongside the real citations
    # (verified against real replay output). The heavy chain always runs
    # Gate 3 through the equivalent ``self._protect_code(...)``
    # elsewhere; this call site is the one that had skipped it.
    report = with_code_protected(
        cleaned,
        lambda body: finalize_report_with_canonical_references(
            body, references=references, language=resolved_language,
        ),
    )

    # Fail-open safety valve only — this dict is the guard's input, never
    # published.
    deliverable_check = deliverable_metrics(report, native=native_draft)
    if deliverables_lost(deliverable_check):
        raise DeliverablesLostError(
            f"rewrite lost all deliverables: {deliverable_check}",
        )

    result_metrics: dict[str, Any] = {
        "candidates_considered": review_stats.get("candidates_considered", 0),
        "references_selected": len(references),
        "selection_fallback": review_stats.get("selection_fallback", ""),
        "language": resolved_language,
    }
    logger.info("fast_reporter_v1 metrics: %s", result_metrics)

    return FastReportResult(
        report=report,
        # The joined rows, not Gate 1's bare {title, url}: the writer prompt
        # and any offline tooling both need each source's snippet,
        # page_title and quality, and Gate 3 already ran against the same
        # URLs and titles.
        references=join_reference_rows(references, nodes, tag_citations=False),
        repaired=False,
        preclean=preclean,
        metrics=result_metrics,
    )


def _build_call_llm(metadata: dict[str, Any]) -> CallLLM:
    """Build the report ``call_llm`` from the agent_team reporter profile.

    Reuses ``heavy_reporter._build_report_llm_for_repair`` so the provider
    fallback chain resolution is identical to the existing heavy chain's,
    and the selected profile's ``report_llm:``
    section works unchanged — no new profile file.

    The per-call timeout is passed down explicitly, in a **local copy** of
    the profile section. ``build_aux_llm`` defaults to ``llm_timeout_s``
    120 (``frontier_agent/infra/llm/aux_builder.py``) and the profile sets no
    such key, so without this the client would abort at 120s while the
    outer ``asyncio.wait_for(..., 200.0)`` never bound. The copy matters:
    ``report_llm:`` is shared with the heavy chain, whose behaviour this
    must not change, so the YAML is not touched and the mutation never
    escapes this function.
    """
    import asyncio

    from frontier_agent.core.messages import system_msg, user_msg
    from frontier_agent.core.runtime.loop.llm_client import extract_usage
    from frontier_agent.infra.llm.aux_builder import build_aux_llm
    from workflows._shared.sdk_shim import record_reporter_usage
    from workflows.agent_team.nodes.reporter import _load_report_profile

    profile_name, profile = _load_report_profile(metadata or {})
    section = profile.get("report_llm") or {}
    if not section:
        raise RuntimeError(
            f"profile {profile_name!r} has no report_llm: section",
        )
    timeout_s = float(section.get("llm_timeout_s") or _REPORT_LLM_TIMEOUT_S)
    llm = build_aux_llm({**section, "llm_timeout_s": timeout_s})
    if llm is None:
        raise RuntimeError(f"could not build report LLM from {profile_name!r}")

    async def call_llm(system: str, user: str) -> str:
        response = await asyncio.wait_for(
            llm.chat([system_msg(system), user_msg(user)]),
            timeout=timeout_s,
        )
        usage = extract_usage(response)
        record_reporter_usage(
            (metadata or {}).get("sdk_protocol_usage_aggregator"),
            usage=usage,
            provider=str(
                (usage or {}).get("provider")
                or section.get("_provider_label")
                or section.get("provider")
                or "",
            ),
            model=str(
                (usage or {}).get("model")
                or section.get("model")
                or "",
            ),
        )
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    return call_llm


async def _run_fast_reporter(state: dict[str, Any], ctx: Any) -> str:
    """Run the fast implementation behind the shared reporter node.

    It uses the same ``agent_id="reporter"`` run envelope, output channel,
    heartbeat, usage aggregator, and trace reporter object as the established
    heavy implementation. Inputs come entirely from state; no trajectory files
    are required.
    """
    import asyncio

    from workflows._shared.cited_report_finalizer import (
        citation_numbers,
        strip_trailing_references,
    )
    from workflows._shared.sdk_shim import ReporterDeltaEmitter, start_heartbeat, stop_heartbeat
    from workflows.agent_team.nodes.reporter import (
        _provider_usage_snapshot,
        _refresh_reporter_usage,
        _set_reporter_extension,
    )

    metadata = state.get("metadata") or {}
    question = str(
        state.get("effective_question")
        or state.get("original_question")
        or state.get("current_query")
        or "",
    ).strip()
    native_draft = str(
        state.get("final_answer") or state.get("final_content") or "",
    ).strip()
    if not native_draft:
        logger.info(
            "fast_reporter_v1: no native draft — keeping main_agent answer",
        )
        return ""

    # Resolve the full reporter profile contract before opening the protocol
    # run. This honors report_profile, swarm_profile, profile_inline, and
    # profile_overrides through reporter._load_report_profile.
    call_llm = _build_call_llm(metadata)
    emitter = metadata.get("sdk_protocol_emitter")
    reporter_stream = ReporterDeltaEmitter(emitter)
    usage_before = _provider_usage_snapshot(metadata)
    reporter_start_iso = datetime.now(UTC).isoformat()
    report_md = ""
    run_status, stop_reason, error_message = "success", "final_answer", ""
    cancelled = False

    reporter_stream.start()
    reporter_stream.reasoning("Reviewing evidence and attaching citations…\n")
    heartbeat = start_heartbeat(
        emitter=emitter,
        agent_id="reporter",
        state="generating_report",
        parent_id=None,
    )
    try:
        result = await generate_fast_reporter_v1_report(
            question=question,
            native_draft=native_draft,
            evidence_cards=state.get("evidence_cards") or [],
            call_llm=call_llm,
            language=str(state.get("language") or "auto"),
            cap=resolve_citation_cap(metadata),
        )
        report_md = result.report.strip()
        reporter_stream.stream_output(report_md)

        try:
            body = strip_trailing_references(report_md)
            reporter_usage = _refresh_reporter_usage(
                metadata,
                before=usage_before,
            )
            _set_reporter_extension(
                metadata,
                report_md=report_md,
                dag_entries=[],
                start_time_iso=reporter_start_iso,
                citation_metadata={
                    "backend": "fast",
                    "citation_count": len(set(citation_numbers(body))),
                    "reference_count": len(result.references),
                    **result.metrics,
                },
                usage_summary=reporter_usage,
                raw_report=result.preclean,
            )
        except Exception as exc:
            logger.warning(
                "fast_reporter_v1: reporter-object trace fold failed: %s",
                exc,
                exc_info=True,
            )
        return report_md
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        run_status, stop_reason, error_message = "error", "exception", str(exc)
        raise
    finally:
        stop_heartbeat(heartbeat)
        if not cancelled:
            reporter_stream.finish(
                final_content=report_md,
                status=run_status,
                stop_reason=stop_reason,
                error_message=error_message,
            )


async def agent_team_fast_reporter_v1(
    state: dict[str, Any], ctx: Any,
) -> dict[str, Any]:
    """Compatibility node wrapper for the fast reporter backend. Fail-open."""
    try:
        report_md = await _run_fast_reporter(state, ctx)
    except Exception as exc:
        logger.warning(
            "fast_reporter_v1: keeping main_agent answer (%s: %s)",
            type(exc).__name__, exc,
        )
        return {}

    if not report_md:
        return {}

    return {
        "final_answer": report_md,
        "final_content": report_md,
        "report_markdown": report_md,
    }


__all__ = [
    "DEFAULT_CITATION_CAP",
    "FastReportResult",
    "NoAdmissibleEvidenceError",
    "_run_fast_reporter",
    "agent_team_fast_reporter_v1",
    "generate_fast_reporter_v1_report",
    "resolve_citation_cap",
    "resolve_language",
]
