"""Optional fast reporter for the agent-team pipelines.

The node fails open: report-generation errors leave the coordinator's final
answer untouched. A stable backend seam is retained for future extensions.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from datetime import UTC, datetime
from typing import Any

from frontier_agent.components.finalization import remaining_phase_budget_s
from frontier_agent.models.node_context import NodeContext

logger = logging.getLogger(__name__)

# Self-contained agent_team profile carrying the reporter's
# report_llm/dag_llm/outline_llm sections (and the main_agent ``llm:``).
# Lives in ``workflows/agent_team/profiles/``.
_DEFAULT_REPORT_PROFILE = "benchmark"


def _set_reporter_extension(
    metadata: dict[str, Any],
    *,
    report_md: str,
    dag_entries: list[dict[str, Any]],
    start_time_iso: str,
    citation_metadata: dict[str, Any] | None = None,
    usage_summary: dict[str, Any] | None = None,
    raw_report: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
) -> None:
    """Fold the reporter run into the final trace as a reporter run entry.

    The shipped (``fast``) reporter backend calls this with
    ``message_history`` / ``tool_trace`` / ``thinking_blocks`` left at their
    ``None`` default and ``dag_entries=[]`` — it has no ReAct loop of its
    own to capture a transcript from. The three transcript fields exist for
    a reporter backend that runs its own tool-calling loop (a
    ``reporter_backend=heavy`` alternative is not included in this OSS
    port; see ``agent_team_reporter`` in this module): such a backend can
    pass its message history / tool trace / thinking blocks through here for
    debug-only inspection of how the report was actually assembled. The
    other payload that matters for training/analysis is ``dag[]`` — the
    captured ``builder="meta_synth_verifier"`` entry (full ``meta`` /
    ``synth`` / ``verifier_result``), shaped like ``reporter.dag[]`` so consumers
    reuse the same lookup
    (``reporter.dag.find(d => d.builder === "meta_synth_verifier")``).
    Best-effort: no worker-trace observer wired ⇒ silent no-op.
    """
    from workflows._shared.sdk_shim import find_trace_observer

    trace_obs = find_trace_observer(metadata.get("sdk_extra_observers"))
    if trace_obs is None:
        return
    end_iso = datetime.now(UTC).isoformat()
    duration_ms = 0
    try:
        start_dt = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        duration_ms = max(0, int((end_dt - start_dt).total_seconds() * 1000))
    except (ValueError, TypeError):
        duration_ms = 0
    reporter_obj = {
        "id": "reporter",
        "run_type": "reporter",
        "agent_id": "reporter",
        "role_id": "reporter",
        "parent_id": None,
        "status": "success" if report_md else "partial",
        "stop_reason": "final_answer" if report_md else "",
        "start_time": start_time_iso,
        "end_time": end_iso,
        "duration_ms": duration_ms,
        "message_history": list(message_history or []),
        "sub_agents": [],
        "llm_call_timings": [],
        "step_logs": [],
        # Debug-only: a heavy-backend reporter's ReAct tool calls
        # (turn/tool/args/result_len), a lighter-weight companion to the
        # full message_history above.
        "tool_trace": list(tool_trace or []),
        # Debug-only: a heavy-backend reporter's extended-thinking blocks,
        # kept out of message_history by that backend's writer loop itself.
        "thinking_blocks": list(thinking_blocks or []),
        "dag": list(dag_entries),
        # The reporter does not build its own run-level DAG snapshot.
        "dag_snapshot": None,
        "citation_metadata": dict(citation_metadata or {}),
        "usage_summary": copy.deepcopy(usage_summary or {}),
        "final_answer": report_md,
        # Writer LLM's report verbatim off ``submit_report``, before
        # ``_clean_report``/Gate2.5/Gate3/memory-gate post-processing —
        # debug-only, for diffing against the published ``final_answer``.
        "raw_report": raw_report or "",
    }
    # Same sidecar-file mechanism heavy_reporter.py uses for its own
    # ``reporter`` block (write_extension_json_value + set_extension_data_
    # from_files): materialise the object through the streaming encoder
    # (which already carries ``default=str``) and register only the file
    # path, so a stray non-JSON value here can't blow up the single
    # in-memory ``json.dumps`` of the whole trace document.
    reporter_path = trace_obs.write_extension_json_value("reporter", reporter_obj)
    trace_obs.set_extension_data_from_files(reporter=reporter_path)


def _provider_usage_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the shared task aggregator's current provider/model snapshot."""
    aggregator = metadata.get("sdk_protocol_usage_aggregator")
    snapshotter = getattr(aggregator, "provider_usage_snapshot", None)
    if snapshotter is None:
        return {}
    try:
        snapshot = snapshotter()
    except Exception as exc:
        logger.debug("agent_team_reporter: usage snapshot failed: %s", exc)
        return {}
    return snapshot if isinstance(snapshot, dict) else {}


def _usage_summary_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Build a v1 UsageSummary for calls made between two task snapshots."""
    count_fields = (
        "prompt_tokens",
        "completion_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "total_tokens",
        "request_count",
    )

    def _rows(snapshot: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row.get("provider") or ""), str(row.get("model") or "")): row
            for row in (snapshot.get("by_provider_model") or [])
            if isinstance(row, dict)
        }

    before_rows, after_rows = _rows(before), _rows(after)
    llm: dict[str, dict[str, Any]] = {}
    totals = {field: 0 for field in count_fields}
    for (provider, model), row in after_rows.items():
        prior = before_rows.get((provider, model), {})
        delta = {
            field: max(
                0,
                int(row.get(field, 0) or 0) - int(prior.get(field, 0) or 0),
            )
            for field in count_fields
        }
        if not any(delta.values()):
            continue
        key = f"{model}@{provider}" if provider else model
        llm[key] = {
            "model_name": model,
            "provider": provider,
            "provider_class": "OpenAIClient",
            "total_prompt_tokens": delta["prompt_tokens"],
            "total_completion_tokens": delta["completion_tokens"],
            "total_cache_read_tokens": delta["cache_read_tokens"],
            "total_cache_write_tokens": delta["cache_write_tokens"],
            "total_cached_tokens": delta["cached_tokens"],
            "total_reasoning_tokens": delta["reasoning_tokens"],
            "total_tokens": delta["total_tokens"],
            "request_count": delta["request_count"],
        }
        for field, value in delta.items():
            totals[field] += value
    return {"llm": llm, "tools": {}, "totals": totals}


def _refresh_reporter_usage(
    metadata: dict[str, Any],
    *,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Refresh top-level usage and return the reporter-only delta."""
    after = _provider_usage_snapshot(metadata)
    if not after:
        return {}
    from workflows._shared.sdk_shim import find_trace_observer

    trace_obs = find_trace_observer(metadata.get("sdk_extra_observers"))
    setter = getattr(trace_obs, "set_provider_usage_snapshot", None)
    if setter is not None:
        try:
            setter(after)
        except Exception as exc:
            logger.warning(
                "agent_team_reporter: top-level usage refresh failed: %s",
                exc, exc_info=True,
            )
    return _usage_summary_delta(before, after)


def _refresh_trace_terminal(metadata: dict[str, Any], report_md: str) -> None:
    """Replace the pre-reporter draft in the worker trace terminal fields."""
    from workflows._shared.sdk_shim import find_trace_observer

    trace_obs = find_trace_observer(metadata.get("sdk_extra_observers"))
    if trace_obs is None:
        return
    trace_obs.set_terminal_state(
        stop_reason="final_answer",
        task_end_time=datetime.now(UTC).isoformat(),
        final_answer=report_md,
    )


def _load_report_profile(
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Load a reporter-capable profile without breaking named precedence.

    An explicit ``report_profile`` (request metadata or deployment env) always
    wins and is loaded by name, even when the main run uses ``profile_inline``.
    Otherwise an inline/run profile is reused only when its effective config
    carries ``report_llm``; profiles that only configure the coordinator fall
    back to the self-contained ``benchmark`` profile.
    """
    from workflows.agent_team.profile import load_swarm_profile

    overrides = metadata.get("profile_overrides")
    explicit_name = (
        str(metadata.get("report_profile") or "").strip()
        or os.environ.get("AGENT_TEAM_REPORT_PROFILE", "").strip()
    )
    if explicit_name:
        return explicit_name, load_swarm_profile(
            explicit_name,
            overrides=overrides,
        )

    inline = metadata.get("profile_inline")
    if isinstance(inline, dict) and inline:
        inline_profile = load_swarm_profile(
            "",
            overrides=overrides,
            inline=inline,
        )
        if inline_profile.get("report_llm"):
            return "__inline__", inline_profile
        return _DEFAULT_REPORT_PROFILE, load_swarm_profile(
            _DEFAULT_REPORT_PROFILE,
            overrides=overrides,
        )

    run_profile_name = str(
        metadata.get("profile") or metadata.get("swarm_profile") or ""
    ).strip()
    if run_profile_name:
        run_profile = load_swarm_profile(
            run_profile_name,
            overrides=overrides,
        )
        if run_profile.get("report_llm"):
            return run_profile_name, run_profile

    return _DEFAULT_REPORT_PROFILE, load_swarm_profile(
        _DEFAULT_REPORT_PROFILE,
        overrides=overrides,
    )


async def agent_team_reporter(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    """Run the fast terminal reporter for either agent-team pipeline.

    Reporter failures preserve the coordinator answer; successful reports
    replace only answer fields and keep evidence state unchanged.
    """
    from workflows.agent_team.profile import (
        REPORTER_BACKEND_FAST,
        REPORTER_BACKEND_HEAVY,
    )

    backend = str(
        state.get("reporter_backend") or REPORTER_BACKEND_FAST,
    ).strip()
    if backend == REPORTER_BACKEND_HEAVY:
        raise RuntimeError(
            "reporter_backend=heavy is not included in the OSS port; use "
            "reporter_backend=fast or disable the reporter. The future heavy "
            "backend should implement generate_report_v3(state, ctx) here."
        )
    raw_timeout = state.get("reporter_wall_time_s", 1800)
    try:
        reporter_wall_time_s = float(raw_timeout)
    except (TypeError, ValueError):
        reporter_wall_time_s = 1800.0
    if reporter_wall_time_s <= 0:
        reporter_wall_time_s = 1800.0
    # main_agent planned this budget as part of its wall reserve, but on a short
    # platform wall the reserve does not survive intact (the research deadline is
    # floored at half the wall). Clamp to the time actually left so this node
    # fails open with the handoff baseline instead of being cancelled mid-report.
    raw_deadline = state.get("reporter_deadline_monotonic_s")
    try:
        deadline_monotonic = (
            float(raw_deadline) if raw_deadline is not None else None
        )
    except (TypeError, ValueError):
        deadline_monotonic = None
    reporter_wall_time_s = remaining_phase_budget_s(
        reporter_wall_time_s, deadline_monotonic,
    )

    try:
        async with asyncio.timeout(reporter_wall_time_s):
            from workflows.agent_team.nodes.fast_reporter_v1 import (
                _run_fast_reporter,
            )

            report_md = await _run_fast_reporter(state, ctx)
    except Exception as exc:
        logger.warning(
            "agent_team_reporter[%s]: failed (%s: %s) — keeping main_agent answer",
            backend,
            type(exc).__name__,
            exc,
        )
        return {}

    if not report_md.strip():
        return {}

    try:
        _refresh_trace_terminal(state.get("metadata") or {}, report_md)
    except Exception as exc:
        logger.warning(
            "agent_team_reporter: failed to refresh terminal trace "
            "(%s: %s)",
            type(exc).__name__, exc,
        )

    return {
        "final_answer": report_md,
        "final_content": report_md,
        "report_markdown": report_md,
        "answer_status": "complete",
        "answer_sentinel": "",
        "final_answer_source": "reporter_llm",
        "final_answer_rescued": False,
        "final_answer_rescue_mode": "",
    }
