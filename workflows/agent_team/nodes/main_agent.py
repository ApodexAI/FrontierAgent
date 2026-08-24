"""Main-agent node — agent-team coordinator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.components.finalization import (
    ResearchWall,
    check_wall_feasibility,
    nonnegative_seconds,
    positive_seconds,
    resolve_research_wall,
)
from frontier_agent.components.observers.budget_observer import BudgetObserver
from frontier_agent.components.observers.context_size_guard import ContextSizeGuard
from frontier_agent.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from frontier_agent.components.observers.finalization_reserve import (
    FinalizationReserveObserver,
)
from frontier_agent.components.observers.last_turn_forcer import LastTurnForcer
from frontier_agent.components.observers.leaked_tool_call_retry import (
    LeakedToolCallRetryObserver,
)
from frontier_agent.components.observers.react_step_tracker import ReactStepTracker
from frontier_agent.components.observers.sse_observer import SSEObserver
from frontier_agent.components.observers.text_repetition_guard import (
    TextRepetitionGuard,
)
from frontier_agent.components.observers.trajectory import TrajectoryFileObserver
from frontier_agent.components.observers.wall_clock_observer import (
    WallClockDeadlineObserver,
)
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
)
from frontier_agent.core.messages import (
    Message,
    ToolCall,
    is_assistant_msg,
)
from frontier_agent.core.runtime import registry
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.runtime.loop.budget_consistency import (
    check_context_budget,
)
from frontier_agent.core.runtime.loop.compact import KeepLastNToolResultsCompactor
from frontier_agent.core.runtime.loop.model_profile import (
    ModelProfile,
    resolve_history_policy,
)
from frontier_agent.core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
    compaction_trigger_tokens,
)
from frontier_agent.core.runtime.loop.tool_exec import PROTECTED_FANIN_TOOLS
from frontier_agent.core.runtime.resources.manager import ResourceManager
from frontier_agent.core.runtime.session_history import build_session_turn
from frontier_agent.models.node_context import NodeContext
from frontier_agent.state.event_store.sqlite import EventStore
from plugins.tools._coerce import coerce_json_list
from plugins.tools._sandbox import (
    BwrapSandbox,
    bwrap_available,
    clear_task_sandbox,
    make_current_sandbox,
    resolve_mount_dirs,
    resolve_sandbox_mode,
    set_task_sandbox,
)
from plugins.tools.assign_task import agent_team_assign_task
from plugins.tools.document_node_toolchain import (
    render_document_node_toolchain_note,
)
from plugins.tools.task_board import (
    build_task_board_observer,
    clear_board,
    force_finish_planning,
    is_planning_allowed,
    render_board,
    start_planning,
)
from workflows.agent_team.citation_url_repair import repair_citation_urls
from workflows.agent_team.identity import (
    MAIN_AGENT_ID,
    MAIN_ROLE_ID,
    SUB_ROLE_ID,
    llm_session_id,
)
from workflows.agent_team.observers.auto_fan_in import AutoFanInObserver
from workflows.agent_team.observers.bare_text_finalize import (
    BareTextFinalizeObserver,
)
from workflows.agent_team.observers.console import RichConsoleObserver
from workflows.agent_team.observers.no_progress_guard import NoProgressGuard
from workflows.agent_team.observers.planning_gate import PlanningGateObserver
from workflows.agent_team.observers.unassigned_nudge import UnassignedAgentNudge
from workflows.agent_team.prompts import (
    get_main_system_prompt,
    render_team_effort,
)
from workflows.agent_team.stream_repetition import (
    parse_stream_repetition_config,
    wrap_llm_for_stream_repetition,
)
from workflows.agent_team.subagent_runtime import (
    SWARM_SCOPE_KEY,
    SwarmSubagentRuntime,
    SwarmToolResultPostProcessor,
    _minimal_best_effort_answer,
    _strip_leaked_tool_calls,
    force_final_answer,
    prepare_report_handoff,
    render_sandbox_fs_note,
)

logger = logging.getLogger(__name__)

# Defaults — overridden by the agent_team profile YAML.
MAIN_MAX_TURNS = 200
MAIN_BUDGET_TOKENS: int | None = None
MAIN_TOOL_TIMEOUT_S = 1900  # must exceed collect_reports' 1800s default.
MAIN_LLM_TIMEOUT_S = 1800
# agent-team has NO terminal tool: the coordinator finishes by ending a turn
# with a plain-text answer and no tool call (BareTextFinalizeObserver). Empty
# string = "no terminal tool" for LastTurnForcer / the loop policy.
MAIN_TERMINAL_TOOL = ""
MAIN_COMPACT_AFTER_TURNS = 100
MAIN_CONTEXT_TOKEN_LIMIT = 180_000
MAIN_KEEP_TOOL_RESULT = 20
SUB_KEEP_TOOL_RESULT = 10

_TEAM_FINALIZATION_MESSAGE = (
    "Finalization phase has started. Stop creating new research branches. "
    "Collect completed work now. If the task requires files, immediately "
    "assign or reuse exactly one capable publisher carrying the exact "
    "/outputs manifest; have it finish, validate, and publish the best "
    "current deliverables. Then return the best complete plain-text answer. "
    "If full completion is impossible, preserve existing artifacts and give a "
    "useful partial answer instead of ending with no deliverable and no answer."
)

# Module-level LLM cache — one client per profile name, shared across tasks.
_llm_cache: dict[str, tuple[dict[str, Any], Any, ModelProfile | None]] = {}


# Wall-clock arithmetic lives in the shared finalization component; these
# aliases keep the workflow's existing private import surface.
_positive_seconds = positive_seconds
_nonnegative_seconds = nonnegative_seconds


def _resolve_research_wall(
    agent_cfg: dict[str, Any],
    *,
    hard_wall_reserve_s: float | None = None,
) -> ResearchWall:
    """Resolve the research deadline plus the hard ceiling it derives from."""
    reserve_s = (
        nonnegative_seconds(
            agent_cfg.get("wall_deadline_reserve_s"),
            default=180,
            label="agent_team wall_deadline_reserve_s",
        )
        if hard_wall_reserve_s is None
        else max(float(hard_wall_reserve_s), 0.0)
    )
    return resolve_research_wall(
        agent_cfg, reserve_s=reserve_s, label_prefix="agent_team",
    )


def _resolve_wall_deadline_s(
    agent_cfg: dict[str, Any],
    *,
    hard_wall_reserve_s: float | None = None,
) -> float:
    """Research-only deadline for :class:`WallClockDeadlineObserver`."""
    return _resolve_research_wall(
        agent_cfg, hard_wall_reserve_s=hard_wall_reserve_s,
    ).research_deadline_s


def _resolve_finalization_timeout_s(
    agent_cfg: dict[str, Any],
    *,
    llm_timeout_s: float,
) -> float:
    value = _positive_seconds(
        agent_cfg.get("finalization_timeout_s"),
        label="agent_team finalization_timeout_s",
    )
    return value or max(float(llm_timeout_s), 1.0)


def _resolve_reporter_wall_time_s(agent_cfg: dict[str, Any]) -> float:
    """Resolve the absolute ceiling for the downstream reporter node.

    Absent or ``0`` falls back to 1800s — unlike the wall-time keys, ``0`` here
    does NOT mean "unlimited". The reporter node clamps this further when a
    platform hard wall leaves less time than this.
    """
    value = _positive_seconds(
        agent_cfg.get("reporter_wall_time_s"),
        label="agent_team reporter_wall_time_s",
    )
    return value or 1800.0


# ── Helpers: profile / trajectory / observers / runtime --------------------

def _resolve_llm_and_profile(
    profile_name: str | None,
    *,
    profile_overrides: dict[str, Any] | None = None,
    profile_inline: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None, ModelProfile | None]:
    """Return ``(llm, profile_dict, model_profile)`` for this run.

    With a named profile → load YAML, build LLM + ModelProfile, cache.
    Without → use ResourceManager's ``agent_team_main`` LLM at ``temperature=0.0``.

    ``profile_overrides`` (deep-merged ``Request.config``) and ``profile_inline``
    (a whole profile shipped in the request) both make the effective config
    per-CALLER, so when either is present the cache is bypassed: the cache key is
    just the profile name, and a cached entry would otherwise be shared across
    callers with different overrides.
    """
    if profile_name or profile_inline:
        bypass_cache = bool(profile_overrides) or profile_inline is not None
        cache_key = profile_name or "__inline__"
        if bypass_cache or cache_key not in _llm_cache:
            from workflows.agent_team.profile import (
                build_swarm_model_profile,
                create_swarm_llm,
                load_swarm_profile,
            )
            profile = load_swarm_profile(
                profile_name or "",
                overrides=profile_overrides,
                inline=profile_inline,
            )
            entry = (
                profile,
                create_swarm_llm(profile),
                build_swarm_model_profile(profile),
            )
            if not bypass_cache:
                _llm_cache[cache_key] = entry
        else:
            entry = _llm_cache[cache_key]
        profile, llm, model_profile = entry
        return llm, profile, model_profile

    llm = registry.get(ResourceManager).get_llm(MAIN_ROLE_ID)
    try:
        # ``bind`` is an optional client capability, probed here rather than
        # declared on the LLMClient protocol; the except branch is the contract.
        llm = llm.bind(temperature=0.0)  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        logger.debug("LLM does not support .bind(temperature=0.0) — skipping")
    return llm, None, None


def _resolve_reporter_enabled(
    profile: dict[str, Any] | None,
    *,
    pipeline_id: str,
    profile_name: str | None,
) -> bool:
    """Resolve the optional reporter while preserving report-pipeline defaults."""
    from workflows.agent_team.profile import resolve_reporter_enabled

    return resolve_reporter_enabled(
        profile,
        pipeline_id=pipeline_id,
        profile_name=profile_name,
    )


def _resolve_reporter_backend(profile: dict[str, Any] | None) -> str:
    """Resolve which implementation the shared reporter node dispatches."""
    from workflows.agent_team.profile import resolve_reporter_backend

    return resolve_reporter_backend(profile)


def _resolve_trajectory_dir(
    state: dict[str, Any], task_id: str, subdir: str = "",
) -> Path:
    """Trajectory directory: ``<trial_dir>/agent/trajectories/`` for Harbor
    runs, ``logs/swarm/<task_id>/trajectories/`` for standalone runs.

    Heavy mode passes a non-empty ``subdir`` (e.g. ``"run_3"``) so each
    parallel run writes its trajectory files under a per-run sub-folder
    instead of overwriting the same ``main_agent.json``. A normal run
    leaves ``subdir=""`` → identical layout to before.
    """
    trial_dir = (state.get("metadata") or {}).get("_trial_dir")
    if trial_dir:
        base = Path(trial_dir) / "agent" / "trajectories"
    elif run_dir := os.environ.get("APODEX_RUN_DIR", "").strip():
        base = Path(run_dir) / "trajectories" / task_id
    else:
        base = Path("logs") / "swarm" / task_id / "trajectories"
    return base / subdir if subdir else base


def _resolve_worktree_root(state: dict[str, Any], task_id: str) -> Path:
    """Per-task worktree root (the agents' *working* dir, distinct from the
    trajectory/log dir).

    Priority:
      1. Harbor ``_trial_dir`` → ``<trial_dir>/sandbox/worktree`` so the sandbox
         (worktree + /outputs) lives INSIDE the trial dir, right next to the
         eval results — one place for everything:
            <trial_dir>/agent/...              (trajectories, final_answer.txt)
            <trial_dir>/sandbox/worktree/      (/workspace; <sub_id>/ per sub)
            <trial_dir>/sandbox/outputs/       (/outputs deliverables)
      2. ``experiment`` + ``bench_task_id`` → ``experiments/<exp>/questions/<id>/worktree``
         (non-harbor benchmark-aware fallback).
      3. explicit ``coding_workspace_root``.
      4. standalone fallback ``logs/swarm/<task_id>/worktree``.

    ``/inputs`` and ``/outputs`` are separate bind mounts (see
    _resolve_sandbox_binds); only ``/workspace`` lives under the returned root.
    """
    md = state.get("metadata") or {}
    trial_dir = md.get("_trial_dir")
    if trial_dir:
        return Path(trial_dir) / "sandbox" / "worktree"
    experiment = md.get("experiment")
    bench_task_id = md.get("bench_task_id")
    if experiment and bench_task_id:
        return (
            Path("experiments") / str(experiment)
            / "questions" / str(bench_task_id) / "worktree"
        )
    coding_root = md.get("coding_workspace_root")
    if coding_root:
        return Path(coding_root)
    return Path("logs") / "swarm" / task_id / "worktree"


def _resolve_sandbox_binds(
    state: dict[str, Any], worktree_root: Path,
) -> tuple[
    tuple[tuple[str, str, bool], ...],
    tuple[tuple[str, str, bool], ...],
    Path,
]:
    """Resolve the bwrap extra-mount sets for this task.

    Returns ``(main_binds, sub_binds, outputs_dir)`` where each *_binds is a
    tuple of ``(host_src, sandbox_dst, read_only)``:

      * ``/inputs`` (read-only) — per-benchmark input files, declared in
        ``metadata["_sandbox_mounts"]`` as ``[{src, dst, mode}, ...]``. A
        relative ``src`` is anchored to ``metadata["_dataset_root"]`` (so the
        benchmark layer can pass portable relative paths). ``dst`` must live
        under ``/inputs``.
      * ``/outputs`` — shared output dir. The host/container bind is RW, while
        each sub-agent task is made read-only by default and receives bounded
        write permission only from the structured ``assign_task``
        ``output_paths`` manifest.
        Scratch/intermediate files go to /workspace; only the declared final
        manifest belongs in /outputs.

    Benchmarks with no input files (OneMillion, browsecomp) yield only the
    /outputs bind.
    """
    md = state.get("metadata") or {}
    outputs_dir = worktree_root.parent / "outputs"

    inputs: list[tuple[str, str, bool]] = []
    dataset_root = str(md.get("_dataset_root") or "")
    for m in md.get("_sandbox_mounts") or []:
        src = str(m.get("src", "")).strip()
        dst = str(m.get("dst", "")).strip()
        if not src or not dst:
            continue
        if not dst.startswith("/inputs"):
            logger.warning("sandbox mount dst not under /inputs, skipped: %s", dst)
            continue
        p = Path(src)
        if not p.is_absolute() and dataset_root:
            p = Path(dataset_root) / p
        ro = str(m.get("mode", "ro")).lower() != "rw"
        inputs.append((str(p.resolve()), dst, ro))

    inputs_t = tuple(inputs)
    main_binds = (*inputs_t, (str(outputs_dir), "/outputs", False))
    sub_binds = (*inputs_t, (str(outputs_dir), "/outputs", False))  # rw for subs too
    return main_binds, sub_binds, outputs_dir


def _shallow_entries(root: str, *, limit: int = 40) -> list[str]:
    """Depth-1 listing of *root* (dirs suffixed ``/``), for fallback probing."""
    try:
        p = Path(root)
        if not p.is_dir():
            return []
        names: list[str] = []
        for e in sorted(p.iterdir()):
            try:
                names.append(e.name + ("/" if e.is_dir() else ""))
            except OSError:
                names.append(e.name)
            if len(names) >= limit:
                names.append("… (truncated)")
                break
        return names
    except OSError:
        return []


def _has_input_files(path: str) -> bool:
    """Best-effort truth signal for prompt routing; never raises."""
    try:
        root = Path(path)
        return root.is_file() or (
            root.is_dir() and any(item.is_file() for item in root.rglob("*"))
        )
    except OSError:
        return False


def _profile_flag(value: Any, *, default: bool = False) -> bool:
    """Parse profile booleans, including env-substituted strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _log_inputs_dir_contents(
    roots: list[tuple[str, str]],
    *,
    fallback_roots: list[tuple[str, str]] | None = None,
    max_files: int = 200,
) -> None:
    """Diagnostic: log what actually lives under each ``/inputs`` root at runtime.

    ``roots`` is ``[(label, host_path), ...]`` where ``host_path`` is the real
    directory the model's read-only tools (``glob_search`` / ``grep_search`` /
    the sub-agents' ``read_file``) will see as ``/inputs`` — in container mode
    the mounted ``/inputs`` itself, in bwrap mode the host ``src`` of each
    ``/inputs`` bind. For each root it logs the resolved absolute path, whether
    it exists, and every file beneath it (name + absolute path + size) so a
    mount/path mismatch (S3 files landing at a path the tools don't read) is
    visible in the worker log stream.

    When no input file surfaces at any expected root, ``fallback_roots`` are
    probed shallowly (depth-1) so a misplaced mount (e.g. files under
    ``/workspace`` or a nested UUID subdir) shows up in the same log burst.
    Never raises — a diagnostic must not break the run.
    """
    found_any = False
    for label, root in roots:
        try:
            p = Path(root)
            exists = p.exists()
            is_dir = exists and p.is_dir()
            logger.info(
                "[agent_team inputs] %s path=%s exists=%s is_dir=%s",
                label, p, exists, is_dir,
            )
            if not is_dir:
                continue
            files: list[Path] = []
            for f in sorted(p.rglob("*")):
                try:
                    if f.is_file():
                        files.append(f)
                except OSError:
                    continue
                if len(files) >= max_files:
                    break
            if not files:
                logger.warning(
                    "[agent_team inputs] %s path=%s is EMPTY — read_file / "
                    "glob_search will find nothing here", label, p,
                )
                continue
            found_any = True
            logger.info(
                "[agent_team inputs] %s path=%s has %d file(s):",
                label, p, len(files),
            )
            for f in files:
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                logger.info(
                    "[agent_team inputs]   name=%r abs=%s size=%s", f.name, f, size,
                )
        except Exception as exc:
            logger.warning(
                "[agent_team inputs] failed to scan %s path=%s: %s", label, root, exc,
            )

    if not found_any and fallback_roots:
        logger.warning(
            "[agent_team inputs] no files at expected input path(s); probing "
            "fallback locations to find where the mounted files landed",
        )
        for label, root in fallback_roots:
            p = Path(root)
            logger.warning(
                "[agent_team inputs] fallback %s path=%s exists=%s entries=%s",
                label, p, p.exists(), _shallow_entries(root) or "(none/not-a-dir)",
            )


def _dedupe_tool_names(raw_names: Any) -> list[str]:
    out: list[str] = []
    for raw in raw_names or []:
        name = str(raw).strip()
        if name and name not in out:
            out.append(name)
    return out


def _resolve_profile_tools(
    resource_mgr: ResourceManager,
    *,
    role_id: str,
    override: Any,
    label: str,
    dynamic_names: Any = None,
) -> tuple[list[Any], list[str]]:
    """Resolve per-run profile tools without mutating AgentRegistry.

    A profile list replaces the role's static tools, but request-scoped SDK
    tools (for example discovered MCP tools) are capabilities granted for this
    run and must remain bound after that replacement.
    """
    if not override:
        tools = list(resource_mgr.get_tools_for_role(role_id))
        return tools, [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]

    requested = _dedupe_tool_names(override)
    # Closed-book has to be enforced on the profile list, which is the list
    # actually bound to the model. Honouring SWARM_NO_WEB only on the role's
    # permission pool made it inert for every shipped profile, because all of
    # them name the web tools explicitly.
    if os.environ.get("SWARM_NO_WEB", "").strip().lower() in ("1", "true", "yes", "on"):
        from workflows.agent_team import WEB_TOOL_NAMES
        dropped = [n for n in requested if n in WEB_TOOL_NAMES]
        if dropped:
            requested = [n for n in requested if n not in WEB_TOOL_NAMES]
            logger.info(
                "closed-book (SWARM_NO_WEB): dropped %s web tools %s", label, dropped,
            )
    # Compatibility migration for pre-download profiles: web-enabled agents
    # receive the bounded binary-file companion to web_fetch automatically.
    if "web_fetch" in requested and "download_file" not in requested:
        requested.insert(requested.index("web_fetch") + 1, "download_file")
    for name in _dedupe_tool_names(dynamic_names):
        if name not in requested:
            requested.append(name)
    policy = resource_mgr.global_tool_policy
    all_tools = resource_mgr.all_tools
    tools: list[Any] = []
    skipped: list[str] = []
    for name in requested:
        if policy is not None and not policy.allows(name):
            skipped.append(name)
            continue
        tool = all_tools.get(name)
        if tool is None:
            skipped.append(name)
            continue
        tools.append(tool)
    names = [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]
    if skipped:
        logger.warning("%s skipped unavailable/denied profile tools: %s", label, skipped)
    logger.info("%s resolved profile tools: %s", label, names)
    return tools, names


def _apply_agent_team_tool_contract(tools: list[Any]) -> list[Any]:
    """Bind workflow-specific schemas without mutating the global registry."""
    return [
        agent_team_assign_task if getattr(tool, "name", "") == "assign_task" else tool
        for tool in tools
    ]


def _replace_profile_tool_impls(
    tools: list[Any], agent_cfg: dict[str, Any],
) -> list[Any]:
    """Select profile-scoped web tool implementations without global mutation."""
    resolved = list(tools)
    if (agent_cfg.get("web_search_impl") or "original") == "aligned":
        from plugins.tools.web_search_aligned import web_search_aligned

        resolved = [
            web_search_aligned if getattr(tool, "name", "") == "web_search" else tool
            for tool in resolved
        ]
    if (agent_cfg.get("web_fetch_impl") or "original") == "aligned":
        from plugins.tools.web_fetch_aligned import web_fetch_aligned

        resolved = [
            web_fetch_aligned if getattr(tool, "name", "") == "web_fetch" else tool
            for tool in resolved
        ]
    return resolved


_COORDINATOR_REPETITION_HINT = (
    "Your last several turns repeat almost the same text. If you are waiting "
    "on sub-agents, that is fine — but say what has changed since your last "
    "turn, or stop narrating the wait: call collect_reports and act on what "
    "comes back. If a sub-agent is genuinely stuck, stop it and reassign its "
    "sub-question. Do NOT finalize while assigned work is still running."
)


def _build_observers(
    *,
    traj_dir: Path,
    task_id: str,
    budget_tokens: int | None,
    max_input_tokens: int | None,
    event_store: Any,
    tool_names: list[str],
    tools: list[Any] | None = None,
    run_id: str = "",
    run_type: str = "",
    traj_filename: str = "main_agent",
    finalization_reserve_turns: int = 8,
    wall_deadline_s: float = 0,
    force_compaction_first: bool = False,
) -> list[Any]:
    observers: list[Any] = [
        BareTextFinalizeObserver(),
        LeakedToolCallRetryObserver(tool_names=tool_names),
        AutoFanInObserver(),
        UnassignedAgentNudge(),
        # Break the create/assign wind-down spin (repeated new sub-agents +
        # trivial tasks, never finalising) — force a synthesised answer.
        NoProgressGuard(),
        # Hint-only: the coordinator IS the run, so a false positive must
        # never end it. Deliberately paired with NoProgressGuard rather than
        # with RepetitionGuard: waiting on running sub-agents means calling
        # collect_reports repeatedly with identical arguments, which is
        # correct behaviour and exactly what an exact-signature guard would
        # punish. The coordinator's own spin pathology is NoProgressGuard's.
        #
        # The hint is overridden for the same reason. A coordinator narrating
        # the same wait ("still waiting on agent_2 …") for four turns trips the
        # prose detector, and the built-in advice ends with "call your
        # terminal/finalize action" — which would push a correctly waiting
        # coordinator into finalizing early, i.e. the exact failure the line
        # above avoids, reached through the text channel instead.
        TextRepetitionGuard(hint_message=_COORDINATOR_REPETITION_HINT),
        build_task_board_observer(),
        FinalizationReserveObserver(
            reserve_turns=finalization_reserve_turns,
            message=_TEAM_FINALIZATION_MESSAGE,
        ),
        LastTurnForcer(terminal_tool=MAIN_TERMINAL_TOOL),
        RichConsoleObserver(),
        TrajectoryFileObserver(
            traj_dir,
            filename=traj_filename,
            tools=tools or [],
            format_env_vars=("SWARM_TRAJECTORY_FORMATS",),
            tool_schema_detail="minimal",
            include_start_tool_names=False,
        ),
        ReactStepTracker(),
    ]
    # Lazy-imported research observers — keep kernel-only test paths free
    # of research-domain modules.
    from workflows._shared.research.observers.assertion_observer import AssertionObserver
    from workflows._shared.research.observers.evidence_observer import EvidenceObserver

    observers.extend([EvidenceObserver(), AssertionObserver()])

    if DuplicateQueryRollbackObserver.DEFAULT_TOOL_NAMES.intersection(tool_names):
        # The benchmark profile gives the coordinator web_search; the TUI one
        # does not. Mount only where it can fire.
        observers.append(DuplicateQueryRollbackObserver())
    if budget_tokens:
        observers.append(BudgetObserver(max_tokens=budget_tokens))
    if max_input_tokens:
        observers.append(ContextSizeGuard(
            max_input_tokens=max_input_tokens,
            force_compaction_first=force_compaction_first,
        ))
    if wall_deadline_s > 0:
        observers.append(WallClockDeadlineObserver(
            deadline_s=wall_deadline_s,
            # The resolver already converted any operational hard wall into
            # this research-only soft deadline.
            reserve_s=0,
            warn_ratio=0.75,
        ))
    if event_store is not None:
        observers.append(SSEObserver(
            event_store=event_store, task_id=task_id,
            run_id=run_id, run_type=run_type,
        ))
    return observers


def _swarm_main_loop_policy() -> LoopPolicy:
    # No terminal tool: BareTextFinalizeObserver ends the loop when the agent
    # produces a plain-text answer with no tool call (and the gate passes).
    # The no-tool nudge below only fires on an EMPTY no-tool turn — a non-empty
    # bare-text turn is handled by the observer before this point.
    return LoopPolicy(
        terminal_tool_names=(),
        no_tool_behavior="nudge",
        no_tool_nudge_message=(
            "You produced no output and called no tool. Either dispatch / "
            "collect from sub-agents (assign_task, collect_reports) to keep "
            "working, or — once every task-board item is resolved — deliver "
            "your COMPLETE answer as plain text (no tool call) to finish."
        ),
    )


def _planning_loop_policy() -> LoopPolicy:
    """Loop policy for the two-loop PLANNING phase (fresh_execution_context).

    ``finish_planning`` is the terminal tool, so calling it ends the planning
    loop and the node then starts the execution loop. If the model never calls
    it, the loop closes on ``max_turns`` (= ``planning_max_turns``) and the node
    force-finishes planning anyway.
    """
    return LoopPolicy(
        terminal_tool_names=("finish_planning",),
        no_tool_behavior="nudge",
        no_tool_nudge_message=(
            "Every turn must end with a tool call. You are PLANNING: decompose "
            "the problem and register sub-questions with `add_task` (read-only "
            "tools — grep_search / glob_search / web_search — may help you "
            "understand it), then call `finish_planning`. Do not reply with "
            "plain text."
        ),
    )


# ── Helper: derive frontend-facing sub-question list ----------------------

def _tool_call_name_args(tc: ToolCall) -> tuple[str, dict[str, Any]]:
    """Read ``(name, args)`` from an OpenAI-wire tool_call dict.

    Native history stores tool calls as
    ``{"type": "function", "id": …, "function": {"name", "arguments"}}``
    where ``arguments`` is a JSON-encoded string. Returns the function
    name and the decoded args object (``{}`` on any decode failure).
    """
    fn = tc.get("function") or {}
    name = str(fn.get("name", ""))
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        return name, raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return name, {}
        return name, parsed if isinstance(parsed, dict) else {}
    return name, {}


def _extract_sub_questions(
    messages: list[Message], bus: AgentBus, task_id: str,
) -> list[dict[str, Any]]:
    """Synthesise ``clarified_questions`` from the main agent's tool calls.

    Swarm has no upstream clarify node — the frontend Plan card reads
    this list to show X/Y progress. Pass 1 harvests assigned tasks;
    pass 2 picks up created-but-unassigned agents; falls back to the
    bus session roster if history was trimmed.
    """
    sub_questions: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    covered: set[str] = set()

    def _add(agent: str, text: str) -> None:
        sub_questions.append({
            "id": f"sq-{len(sub_questions) + 1:03d}",
            "text": text,
            "question": text,
            "agent": agent,
        })

    for msg in messages:
        if not is_assistant_msg(msg):
            continue
        for tc in msg.get("tool_calls") or []:
            name, args = _tool_call_name_args(tc)
            if name != "assign_task":
                continue
            for t in coerce_json_list(args.get("tasks") or []) or []:
                if not isinstance(t, dict):
                    continue
                agent = str(t.get("agent", "")).strip()
                task_text = str(t.get("task") or t.get("prompt") or "").strip()
                if not task_text:
                    continue
                key = f"{agent}::{task_text[:120]}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                covered.add(agent)
                _add(agent, task_text)

    for msg in messages:
        if not is_assistant_msg(msg):
            continue
        for tc in msg.get("tool_calls") or []:
            name, args = _tool_call_name_args(tc)
            if name != "create_subagent":
                continue
            for spec in coerce_json_list(args.get("agents") or []) or []:
                if not isinstance(spec, dict):
                    continue
                agent = str(spec.get("name", "")).strip()
                if not agent or agent in covered:
                    continue
                hint = str(spec.get("system_prompt") or "").strip()[:200]
                covered.add(agent)
                _add(agent, hint or agent)

    if not sub_questions:
        try:
            sessions = bus.list_sessions_for_task(task_id)
        except Exception:
            sessions = []
        for s in sessions:
            name = getattr(s, "name", "") or getattr(s, "session_id", "")
            if name and name not in covered:
                covered.add(name)
                _add(name, name)

    return sub_questions


def _collect_live_followups(observers: list[Any]) -> list[str]:
    """Collect delivered user follow-ups from optional live observers."""
    followups: list[str] = []
    seen: set[str] = set()
    for observer in observers:
        snapshot: Callable[[], list[str] | None] | None = getattr(
            observer, "live_followups_snapshot", None,
        )
        if not callable(snapshot):
            continue
        try:
            values = snapshot()
        except Exception:
            logger.debug("Failed to snapshot live follow-ups", exc_info=True)
            continue
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                followups.append(text)
    return followups


def _compose_effective_question(question: str, followups: list[str]) -> str:
    """Build the complete user-visible scope for downstream DAG nodes."""
    if not followups:
        return question
    additions = "\n".join(f"- {text}" for text in followups)
    return (
        f"{question}\n\n"
        "Additional user requests received during execution:\n"
        f"{additions}"
    )


# ── Node entry point ------------------------------------------------------

async def main_agent_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    """Run the agent-team main coordinator as a ReAct loop."""
    question = state.get("original_question", "")
    if not question:
        raise ValueError("main_agent_node requires 'original_question' in state")

    metadata = state.get("metadata") or {}
    profile_name: str | None = (
        metadata.get("profile") or metadata.get("swarm_profile")
    )
    llm, profile, model_profile = _resolve_llm_and_profile(
        profile_name,
        profile_overrides=metadata.get("profile_overrides"),
        profile_inline=metadata.get("profile_inline"),
    )

    agent_cfg = (profile or {}).get("agent", {})
    # Model's declared per-call output cap (``llm.max_tokens``). Threaded to
    # the sub-agent runtime so ``_bind_sub_agent_llm`` never requests MORE
    # output than the model accepts (over-requesting 400s the sub-agent on
    # turn 1). ``0``/absent → None → sub-agent keeps its legacy desired value.
    llm_max_tokens = int((profile or {}).get("llm", {}).get("max_tokens", 0) or 0) or None
    reporter_enabled = _resolve_reporter_enabled(
        profile,
        pipeline_id=str(metadata.get("pipeline_id") or ""),
        profile_name=profile_name,
    )
    reporter_backend = _resolve_reporter_backend(profile)

    # Heavy mode runs K parallel main agents under one root task_id; it
    # injects a synthetic ``bus_task_id`` (e.g. ``f"{root}.heavy_run_3"``)
    # so AgentBus session ids ``{task_id}::{name}`` don't collide and a
    # ``cleanup_task`` from one run doesn't kill another run's sessions.
    # ``trajectory_subdir`` (e.g. ``"run_3"``) routes per-run trajectory
    # output into its own folder. A normal run leaves both unset → no
    # behaviour change.
    bus_task_id = str(metadata.get("bus_task_id") or ctx.task_id)
    trajectory_subdir = str(metadata.get("trajectory_subdir") or "")

    max_turns = int(agent_cfg.get("main_max_turns", MAIN_MAX_TURNS))
    budget_raw = agent_cfg.get("budget_tokens", MAIN_BUDGET_TOKENS)
    budget_tokens = int(budget_raw) if budget_raw else None
    max_input_raw = agent_cfg.get("max_input_tokens")
    max_input_tokens = int(max_input_raw) if max_input_raw else None
    # One knob for both the coordinator's loop and its sub-agents' — the
    # coordinator's long-blocking tools do not need a looser ceiling here
    # because ``_effective_tool_timeout`` already floors the outer wait for
    # ``_AGGREGATION_TOOLS`` at the tool's own ``timeout`` argument, so
    # ``collect_reports(timeout=1800)`` gets 1805s regardless of this value.
    tool_timeout = float(agent_cfg.get("tool_timeout_s", MAIN_TOOL_TIMEOUT_S))
    llm_timeout = float(agent_cfg.get("llm_timeout_s", MAIN_LLM_TIMEOUT_S))
    reasoning_only_timeout_raw = agent_cfg.get("reasoning_only_timeout_s")
    reasoning_only_timeout_s = (
        float(reasoning_only_timeout_raw) if reasoning_only_timeout_raw else None
    )
    reasoning_only_max_tokens_raw = agent_cfg.get(
        "reasoning_only_max_tokens",
    )
    reasoning_only_max_tokens = (
        int(reasoning_only_max_tokens_raw)
        if reasoning_only_max_tokens_raw
        else None
    )
    logical_call_timeout_raw = agent_cfg.get("logical_call_timeout_s")
    logical_call_timeout_s = (
        float(logical_call_timeout_raw) if logical_call_timeout_raw else None
    )
    finalization_timeout_s = _resolve_finalization_timeout_s(
        agent_cfg,
        llm_timeout_s=llm_timeout,
    )
    reporter_wall_time_s = _resolve_reporter_wall_time_s(agent_cfg)
    finalization_reserve_turns = int(
        agent_cfg.get("finalization_reserve_turns", 8) or 8,
    )
    configured_wall_reserve_s = _nonnegative_seconds(
        agent_cfg.get("wall_deadline_reserve_s"),
        default=180,
        label="agent_team wall_deadline_reserve_s",
    )
    landing_budget_s = (
        reporter_wall_time_s if reporter_enabled else finalization_timeout_s
    )
    # WallClockDeadlineObserver runs at turn end, so a tool launched just
    # before the deadline can consume its full timeout. Keep room for both that
    # overrun and the now-bounded reporter/finalization phase. "Full timeout"
    # is the OUTER wait, which carries a grace over the configured value for
    # budget-aware tools — ask the loop instead of assuming they are equal.
    from frontier_agent.core.runtime.loop.tool_exec import max_tool_wall_time_s
    worst_case_tool_s = max_tool_wall_time_s(tool_timeout)
    wall_deadline_reserve_s = max(
        configured_wall_reserve_s,
        worst_case_tool_s + landing_budget_s,
    )
    research_wall = _resolve_research_wall(
        agent_cfg,
        hard_wall_reserve_s=wall_deadline_reserve_s,
    )
    wall_deadline_s = research_wall.research_deadline_s
    # ``soft_wall_deadline_s`` floors research at half the wall, so on a short
    # wall the reserve above is NOT what actually survives. Publish an absolute
    # instant instead: the reporter node clamps its own ceiling to the time
    # really left and fails open with a real answer, rather than being killed
    # mid-call by the platform ceiling (the live serve ``RenewableWallTimeLease``
    # gives only a few seconds of grace).
    hard_deadline_monotonic = (
        time.monotonic() + research_wall.hard_total_s
        if research_wall.hard_total_s > 0
        else None
    )
    check_wall_feasibility(
        hard_total_s=research_wall.hard_total_s,
        research_deadline_s=wall_deadline_s,
        tool_timeout_s=worst_case_tool_s,
        landing_budget_s=landing_budget_s,
        label_prefix="agent_team",
    )
    history_policy = resolve_history_policy(agent_cfg)
    compact_after_turns = int(agent_cfg.get("compact_after_turns", MAIN_COMPACT_AFTER_TURNS))
    context_token_limit = int(agent_cfg.get("context_token_limit", MAIN_CONTEXT_TOKEN_LIMIT))
    main_keep_tool_result = int(agent_cfg.get("main_keep_tool_result", MAIN_KEEP_TOOL_RESULT))
    sub_keep_tool_result = int(agent_cfg.get("sub_keep_tool_result", SUB_KEEP_TOOL_RESULT))
    # Tiered context compaction (opt-in via profile ``context_compaction``); see
    # stateful_react for the tier semantics. "off" (default) → legacy keep-last-N.
    context_compaction = str(agent_cfg.get("context_compaction", "off")).lower()
    compaction_spill = _profile_flag(agent_cfg.get("compaction_spill"))
    max_len = int(agent_cfg.get("max_len", 0) or 0)
    # config/sglang/README.md states these invariants and sglang-doctor.sh checks
    # them — but only for SGLANG_* on the compose path. These are the values that
    # actually reach the loop, and they can be set straight through OPENAI_* .
    check_context_budget(
        max_len=max_len,
        max_input_tokens=max_input_tokens,
        max_tokens=llm_max_tokens,
        reasoning_only_max_tokens=reasoning_only_max_tokens,
        label="agent_team",
    )
    fs_mode = bool(agent_cfg.get("fs_mode", False)) or bool(metadata.get("fs_mode", False))
    # Sandbox mode: trusted deployment config must explicitly select container;
    # a profile may only tighten auto to bwrap, never attest container isolation.
    #   container — one task = one isolated container; main + every sub share
    #     the mounted /workspace, /outputs, /inputs via CurrentSandbox (prod).
    #   bwrap / auto — local benchmark path (per-sub bwrap namespaces).
    sandbox_mode = resolve_sandbox_mode(agent_cfg)
    fs_enabled = sandbox_mode in ("container", "native") or bwrap_available()
    if sandbox_mode in ("container", "native"):
        _workspace_dir, _outputs_dir, inputs_dir_for_prompt = resolve_mount_dirs()
        inputs_available = _has_input_files(inputs_dir_for_prompt)
    else:
        dataset_root_raw = str(metadata.get("_dataset_root") or "").strip()
        dataset_root = Path(dataset_root_raw) if dataset_root_raw else None
        input_sources: list[Path] = []
        for mount in metadata.get("_sandbox_mounts") or []:
            src_raw = str(mount.get("src") or "").strip()
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.is_absolute() and dataset_root is not None:
                src = dataset_root / src
            input_sources.append(src)
        inputs_available = any(_has_input_files(str(src)) for src in input_sources)
    main_fs_note = render_sandbox_fs_note(
        sandbox_mode=sandbox_mode,
        inputs_available=inputs_available,
        audience="main",
    )
    sub_fs_note = render_sandbox_fs_note(
        sandbox_mode=sandbox_mode,
        inputs_available=inputs_available,
        audience="sub",
    )
    stream_repetition_config = parse_stream_repetition_config(agent_cfg)
    sub_agent_llm = llm
    llm, stream_repetition_observer = wrap_llm_for_stream_repetition(
        llm,
        config=stream_repetition_config,
        role_id=MAIN_ROLE_ID,
        label=str(metadata.get("run_id") or ctx.task_id),
    )

    # Assemble the main agent's compactor + trigger. Tiered reuses KeepLastN
    # (Tier1) + LLMSummaryCompactor (Tier2) behind a real-input-token threshold.
    # ``PROTECTED_FANIN_TOOLS`` (collect_reports / assign_task / submit_report /
    # create_subagent) are pinned out of Tier1 age-based blanking so a sub-agent's
    # findings are never nuked to an OMITTED placeholder; if they fall outside the
    # recent window AND Tier1 didn't free enough, Tier2 can still fold them into a
    # summary (graceful, keeps URLs/entities) — see TieredCompactor's docstring.
    main_keep_recent_msgs = max(6, int(agent_cfg.get("keep_recent_turns", 5)) * 3)
    main_compaction_policy: Any = None
    main_gauge: InputTokenGauge | None = None
    from plugins.tools._overflow import spill_compacted_body

    if context_compaction == "tiered" and max_len > 0:
        main_gauge = InputTokenGauge()
        main_compactor: Any = TieredCompactor(
            keep_tool_result=int(agent_cfg.get("tier1_keep_tool_result", 5)),
            # Tier2 summariser: ``sub_agent_llm`` (== ``llm`` unless a profile
            # gives sub-agents a distinct/cheaper model — in which case the MAIN
            # agent's context is summarised by that sub model, by design here).
            summary_llm=sub_agent_llm,
            relief_target=int(max_len * 0.6),
            protect_tool_names=PROTECTED_FANIN_TOOLS,
            gauge=main_gauge,  # calibrate relief to real tokens (unit-match trigger)
            spill=spill_compacted_body if compaction_spill else None,
            # Bound the whole retry sequence by what ONE summariser call was
            # already allowed to spend, so retrying costs no extra worst case.
            summary_retry_timeout_s=llm_timeout,
        )
        main_compaction_policy = InputTokenThresholdPolicy(
            main_gauge, compaction_trigger_tokens(max_len),
        )
    else:
        main_compactor = KeepLastNToolResultsCompactor(keep_tool_result=main_keep_tool_result)

    # Trajectory layout: per-task split for debugging vs one file per agent.
    # Resolution
    # precedence: agent_cfg → metadata → env (SWARM_TRAJECTORY_PER_TASK)
    # → default True. Each lower step is consulted only when the previous
    # step is missing, so a deployment env-var doesn't override an
    # explicit profile decision.
    if "trajectory_per_task" in agent_cfg:
        trajectory_per_task = bool(agent_cfg["trajectory_per_task"])
    elif "trajectory_per_task" in metadata:
        trajectory_per_task = bool(metadata["trajectory_per_task"])
    else:
        env_val = os.getenv("SWARM_TRAJECTORY_PER_TASK", "1").strip().lower()
        trajectory_per_task = env_val not in ("0", "false", "no", "off")

    resource_mgr = registry.get(ResourceManager)
    dynamic_tool_names = _dedupe_tool_names([
        *(metadata.get("sdk_dynamic_tool_names") or []),
        *(metadata.get("sdk_mcp_tool_names") or []),
    ])
    tools, tool_names = _resolve_profile_tools(
        resource_mgr,
        role_id=MAIN_ROLE_ID,
        override=agent_cfg.get("main_agent_tools"),
        label="main_agent_tools",
        dynamic_names=dynamic_tool_names,
    )
    # ``assign_task`` is registered process-wide with a permissive schema for
    # any workflow that binds it. Agent-team uses a stricter one in its own
    # loop: only the one bounded publisher carries an ``output_paths`` manifest,
    # and carrying it is what authorizes the write. Replacing the same-named
    # Tool object here changes both
    # the model-facing schema and executor entry without mutating the global
    # registry or affecting other bindings.
    tools = _apply_agent_team_tool_contract(tools)
    # ``web_search_impl`` / ``web_fetch_impl`` read as profile-wide switches, so
    # apply them to the coordinator too — benchmark.yaml puts ``web_search`` in
    # ``main_agent_tools``, and a swap that reached only sub-agents would split
    # the two roles across different implementations silently.
    tools = _replace_profile_tool_impls(tools, agent_cfg)
    sub_agent_tools, sub_agent_tool_names = _resolve_profile_tools(
        resource_mgr,
        role_id=SUB_ROLE_ID,
        override=agent_cfg.get("sub_agent_tools"),
        label="sub_agent_tools",
        dynamic_names=dynamic_tool_names,
    )
    sub_agent_tools = _replace_profile_tool_impls(sub_agent_tools, agent_cfg)
    main_document_toolchain_note = render_document_node_toolchain_note(
        sandbox_mode=sandbox_mode,
        tool_names=sub_agent_tool_names,
        audience="coordinator",
    )
    sub_document_toolchain_note = render_document_node_toolchain_note(
        sandbox_mode=sandbox_mode,
        tool_names=sub_agent_tool_names,
    )
    mcp_tool_names = _dedupe_tool_names(metadata.get("sdk_mcp_tool_names") or [])
    if not mcp_tool_names:
        mcp_tool_names = [name for name in tool_names if name.startswith("mcp_")]
    mcp_tool_specs = list(metadata.get("sdk_mcp_tool_specs") or [])
    planning_mode = bool(agent_cfg.get("planning_mode", True))
    # When leaving planning: false = continue the SAME loop (planning reasoning
    # stays in context, turn counter keeps running); true = drop the planning
    # context and run a FRESH execution loop (board injected into the new user
    # message, turns reset). Only meaningful when planning_mode is on.
    fresh_execution_context = bool(agent_cfg.get("fresh_execution_context", False))
    planning_max_turns = int(agent_cfg.get("planning_max_turns", 40))
    date_str = datetime.now(UTC).date().isoformat()

    # Per-benchmark addendum (e.g. OfficeQA's official prompt with paths
    # rewritten to /inputs, or GDPval's "save deliverables to /outputs").
    # MAIN agent only; sub-agent prompts are untouched. Empty for benchmarks
    # that need no addendum (OneMillion, browsecomp).
    _prompt_addendum = str(metadata.get("_sys_prompt_addendum") or "").strip()

    def _decorate(base: str) -> str:
        """Append per-benchmark addendum + sandbox FS note + team_effort tag.

        Shared by every phase prompt (combined / planning / execution) so the
        addendum and the trailing ``<team_effort>`` tag land identically. The
        tag is ALWAYS the last line (stable byte position for the future
        chat-template migration).
        """
        p = base
        if _prompt_addendum:
            p = f"{p}\n\n{_prompt_addendum}"
        if fs_enabled:
            p = f"{p}{main_fs_note}"
        return f"{p}{render_team_effort(agent_cfg.get('team_effort', 'max'))}"

    event_store = registry.get_optional(EventStore)
    traj_dir = _resolve_trajectory_dir(state, ctx.task_id, trajectory_subdir)
    observers = _build_observers(
        traj_dir=traj_dir,
        task_id=ctx.task_id,
        budget_tokens=budget_tokens,
        max_input_tokens=max_input_tokens,
        event_store=event_store,
        tool_names=tool_names,
        tools=tools,
        run_id=str(metadata.get("run_id") or ""),
        run_type=str(metadata.get("run_type") or ""),
        finalization_reserve_turns=finalization_reserve_turns,
        wall_deadline_s=wall_deadline_s,
        force_compaction_first=(context_compaction == "tiered" and max_len > 0),
    )
    # A serve/CLI driver injects its own protocol-stream and worker-trace
    # observers via state.metadata["sdk_extra_observers"] (see
    # ``workflows/_shared/sdk_shim``). The HTTP API path leaves this key
    # unset → no behavior change.
    extra_observers = metadata.get("sdk_extra_observers") or []
    if extra_observers:
        observers = list(observers) + list(extra_observers)
    if stream_repetition_observer is not None:
        observers.append(stream_repetition_observer)
    if main_gauge is not None:
        observers.append(main_gauge)  # feeds the real-token compaction trigger
    # Single-loop planning (planning_mode on, fresh_execution_context off):
    # this observer is the in-loop Planning Mode enforcer — it blocks non
    # read-only / non-board tools while planning and auto-finishes planning at
    # planning_max_turns. NOT needed on the two-loop path (the planning loop is
    # already tool-restricted + max_turns-capped) or when planning is off.
    if planning_mode and not fresh_execution_context:
        observers.append(PlanningGateObserver(planning_max_turns=planning_max_turns))

    # WORKER_TRACE_DIR plumbing: SDK driver puts trace_dir + workflow_id
    # into state.metadata so sub-agents write per-session JSON files.
    worker_trace_dir = metadata.get("worker_trace_dir")
    workflow_id_for_trace = str(metadata.get("workflow_id") or "")

    # Per-task worktree (working dir, NOT the trajectory/log dir).
    #   bwrap mode — main's /workspace is the whole root (sees every sub subdir);
    #     each sub's /workspace is its own subdir; /inputs (ro) + /outputs are
    #     separate bind mounts (see _resolve_worktree_root / _resolve_sandbox_binds).
    #   container mode — main AND every sub SHARE the single mounted /workspace
    #     and /outputs (the container is the isolation), so there are no per-sub
    #     subdirs and no bwrap binds.
    main_binds: tuple[tuple[str, str, bool], ...] = ()
    sub_binds: tuple[tuple[str, str, bool], ...] = ()
    shared_workspace_dir = ""
    if sandbox_mode in ("container", "native"):
        workspace_dir, outputs_dir, inputs_dir = resolve_mount_dirs()
        worktree_root = Path(workspace_dir)
        worktree_root.mkdir(parents=True, exist_ok=True)
        # Pre-create the shared /outputs mount too (parity with the bwrap path,
        # which builds it via _resolve_sandbox_binds). In production it's an
        # existing mount; for a local container run pointing FRONTIER_AGENT_OUTPUTS_DIR
        # at a repo dir this ensures the first deliverable write doesn't fail.
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        shared_workspace_dir = str(worktree_root)
        # /inputs is an external bind-mount; the harness never populates it.
        # Log what actually landed there so a
        # missing/misplaced input file is diagnosable from the worker log.
        _log_inputs_dir_contents(
            [("container /inputs", inputs_dir)],
            fallback_roots=[
                ("workspace", workspace_dir),
                ("root", "/"),
                ("cwd", os.getcwd()),
            ],
        )
    else:
        worktree_root = _resolve_worktree_root(state, ctx.task_id)
        main_binds, sub_binds, _outputs_dir = _resolve_sandbox_binds(
            state, worktree_root,
        )
        # bwrap mode: the model sees each bind's host ``src`` as ``/inputs/…``.
        _log_inputs_dir_contents(
            [
                (f"bwrap bind {dst}", src)
                for (src, dst, _ro) in main_binds
                if dst.startswith("/inputs")
            ]
        )

    swarm_runtime = SwarmSubagentRuntime(
        original_question=question,
        trajectory_dir=traj_dir,
        worktree_root=worktree_root,
        sub_binds=sub_binds,
        sandbox_mode=sandbox_mode,
        shared_workspace_dir=shared_workspace_dir,
        sub_prompt_suffix=(
            (sub_fs_note if fs_enabled else "")
            + sub_document_toolchain_note
        ),
        sub_agent_llm=sub_agent_llm if profile_name else None,
        llm_max_tokens=llm_max_tokens,
        sub_agent_llm_timeout=llm_timeout,
        sub_agent_tool_timeout=tool_timeout,
        reasoning_only_timeout_s=reasoning_only_timeout_s,
        reasoning_only_max_tokens=reasoning_only_max_tokens,
        logical_call_timeout_s=logical_call_timeout_s,
        sub_agent_max_turns=int(agent_cfg.get("sub_max_turns", 100)),
        finalization_reserve_turns=int(
            agent_cfg.get("sub_finalization_reserve_turns", 6) or 6,
        ),
        sub_agent_thinking_in_history=history_policy.thinking_in_history,
        sub_agent_thinking_history_max_tokens=(
            history_policy.thinking_history_max_tokens
        ),
        sub_agent_model_profile=model_profile,
        sub_keep_tool_result=sub_keep_tool_result,
        sub_compact_after_turns=compact_after_turns,
        sub_context_token_limit=context_token_limit,
        context_compaction=context_compaction,
        compaction_spill=compaction_spill,
        max_len=max_len,
        max_input_tokens=max_input_tokens,
        tier1_keep_tool_result=int(agent_cfg.get("tier1_keep_tool_result", 5)),
        keep_recent_turns=int(agent_cfg.get("keep_recent_turns", 5)),
        fs_mode=fs_mode,
        # Sub-agent stop-loss (0 = off). Only sub-agents carry network tools,
        # so the guard lives on them and not on this coordinator.
        stuck_target_hint_after=int(agent_cfg.get("sub_stuck_target_hint_after", 5)),
        stuck_target_escalate_after=int(
            agent_cfg.get("sub_stuck_target_escalate_after", 8),
        ),
        stuck_target_window=int(agent_cfg.get("sub_stuck_target_window", 15)),
        trajectory_per_task=trajectory_per_task,
        worker_trace_dir=worker_trace_dir,
        workflow_id=workflow_id_for_trace,
        # A serve/CLI driver threads its event emitter + usage aggregator
        # through here so sub-agents emit protocol and usage events on the
        # same stdout stream and feed the same final usage aggregator. The
        # HTTP API path leaves both keys unset → no behavior change.
        protocol_emitter=metadata.get("sdk_protocol_emitter"),
        protocol_usage_aggregator=metadata.get("sdk_protocol_usage_aggregator"),
        mcp_tool_names=mcp_tool_names,
        mcp_tool_specs=mcp_tool_specs,
        stream_repetition_config=stream_repetition_config,
        sub_agent_tool_names=sub_agent_tool_names,
        sub_agent_tools=sub_agent_tools,
    )

    logger.info(
        "Swarm main agent starting (task_id=%s, tools=%s)",
        ctx.task_id, [getattr(t, "name", str(t)) for t in tools],
    )

    bus = registry.get(AgentBus)

    # Main agent's task sandbox:
    #   container mode — a CurrentSandbox on the shared mounted /workspace; the
    #     same dir every sub-agent uses, so deliverables land in the one shared
    #     /outputs. No bwrap, no E2B.
    #   bwrap mode — a bwrap sandbox whose /workspace is the WHOLE worktree
    #     (sees every sub subdir), plus /inputs (ro) and /outputs (rw). Sub-agents
    #     override this with their own narrower sandbox (context_setup). Active
    #     bwrap also forces local execution (E2B bypassed).
    main_sb_token = None
    if sandbox_mode in ("container", "native"):
        main_sb_token = set_task_sandbox(make_current_sandbox(worktree_root))
    elif bwrap_available():
        main_sb_token = set_task_sandbox(BwrapSandbox(
            workspace=str(worktree_root), binds=main_binds,
        ))

    # Planning Mode (toggle via profile agent.planning_mode, default on). Mark
    # this run as planning so the gates (PlanningGateObserver, finalize_answer)
    # apply; finish_planning / the turn-cap flip it to execution. When off, no
    # gate; the task board is still used (prompt requires it).
    if planning_mode:
        start_planning(ctx.task_id)

    # Shared scope metadata for every main loop (planning + execution).
    # bus_task_id namespaces AgentBus sessions per-run in heavy mode;
    # root_task_id keeps SSE / event_store keyed on the user-visible task.
    _scope_meta = {
        SWARM_SCOPE_KEY: swarm_runtime,
        "bus_task_id": bus_task_id,
        "root_task_id": ctx.task_id,
        "run_id": metadata.get("run_id"),
        "run_type": metadata.get("run_type"),
        # Forward the SDK protocol emitter + usage aggregator so plugins
        # called inside the loop (notably ``collect_reports``) can emit
        # heartbeat frames on the stdout stream while blocking on
        # sub-agents — without this they no-op and the chunked stream goes
        # silent for minutes, tripping the reverse-proxy idle close (the
        # run then dies with ``status=running`` and no terminal). Mirrors
        # the coordinator node's scope_metadata.
        "sdk_protocol_emitter": metadata.get("sdk_protocol_emitter"),
        "sdk_protocol_usage_aggregator": metadata.get(
            "sdk_protocol_usage_aggregator"
        ),
        # Local TUI observers consume collect_reports progress snapshots while
        # the coordinator loop is blocked inside the tool. Keep them in the
        # execution scope as well as the loop observer list.
        "sdk_extra_observers": metadata.get("sdk_extra_observers") or [],
        "parent_agent_id": metadata.get("parent_agent_id"),
        # Default this eval workflow to the bash command allowlist (enforce).
        # Carried on the loop's ExecutionScope (auto-reset by run_agent_loop's
        # own try/finally — no manual lifecycle to leak). Overridable via
        # BASH_ALLOWLIST_MODE (env wins). Sub-agents set their own mode in
        # subagent_runtime's context_setup.
        "bash_allowlist_mode": "enforce",
        # Container mode: authorize file tools' direct-local access to the
        # mounted /workspace (plugins.tools._path_auth). Empty in bwrap mode.
        "workspace_root": shared_workspace_dir,
    }

    async def _run_main_loop(
        *, system_prompt: str, user_message: str, loop_tools: list[Any],
        loop_observers: list[Any], loop_policy: LoopPolicy, turns: int,
    ) -> AgentLoopResult:
        return await run_agent_loop(
            system_prompt=system_prompt,
            user_message=user_message,
            llm=llm,
            tools=loop_tools,
            config=LoopConfig(
                max_turns=turns,
                task_id=ctx.task_id,
                llm_session_id=llm_session_id(
                    str(metadata.get("session_id") or bus_task_id),
                    MAIN_AGENT_ID,
                ),
                role_id=MAIN_ROLE_ID,
                no_tool_max_retries=3,
                tool_timeout=int(tool_timeout),
                llm_timeout=int(llm_timeout),
                reasoning_only_timeout_s=reasoning_only_timeout_s,
                reasoning_only_max_tokens=reasoning_only_max_tokens,
                logical_call_timeout_s=logical_call_timeout_s,
                max_completion_tokens=(
                    llm_max_tokens or LoopConfig.max_completion_tokens
                ),
                context_token_limit=context_token_limit,
                compact_after_turns=compact_after_turns,
                keep_recent=main_keep_recent_msgs,
                loop_policy=loop_policy,
                compactor=main_compactor,
                compaction_policy=main_compaction_policy,
                tool_result_post_processor=SwarmToolResultPostProcessor(),
            ),
            model_profile=model_profile,
            history_policy=history_policy,
            observers=loop_observers,
            scope_metadata=_scope_meta,
        )

    if planning_mode and fresh_execution_context:
        # ── Two-loop path: PLANNING loop → fresh EXECUTION loop ──────────────
        # Loop 1 sees only read-only + board tools and a planning-only prompt;
        # finish_planning (terminal) or planning_max_turns ends it. Then we drop
        # this loop's context and start a fresh execution loop, handing the
        # task board in via the user message (the board lives outside context,
        # so dropping the messages does not lose it).
        planning_tools = [
            t for t in tools if is_planning_allowed(getattr(t, "name", ""))
        ]
        planning_tool_names = [
            getattr(t, "name", "") for t in planning_tools if getattr(t, "name", "")
        ]
        planning_observers: list[Any] = [
            LeakedToolCallRetryObserver(tool_names=planning_tool_names),
            build_task_board_observer(),
            RichConsoleObserver(),
            TrajectoryFileObserver(
                traj_dir,
                filename="main_planning",
                tools=planning_tools,
                format_env_vars=("SWARM_TRAJECTORY_FORMATS",),
                tool_schema_detail="minimal",
                include_start_tool_names=False,
            ),
            ReactStepTracker(),
            LastTurnForcer(terminal_tool="finish_planning"),
        ]
        if DuplicateQueryRollbackObserver.DEFAULT_TOOL_NAMES.intersection(
            planning_tool_names,
        ):
            # Planning may search to understand the problem before
            # decomposing it; re-running one query there burns turns out of
            # the smaller planning_max_turns budget.
            planning_observers.append(DuplicateQueryRollbackObserver())
        if event_store is not None:
            planning_observers.append(SSEObserver(
                event_store=event_store, task_id=ctx.task_id,
                run_id=str(metadata.get("run_id") or ""),
                run_type=str(metadata.get("run_type") or ""),
            ))
        await _run_main_loop(
            system_prompt=_decorate(get_main_system_prompt(
                date_str, fs_mode=fs_mode, phase="planning",
                sub_agent_tools=sub_agent_tool_names,
                mcp_tool_names=mcp_tool_names, mcp_tool_specs=mcp_tool_specs,
                document_toolchain_note=main_document_toolchain_note,
            )),
            user_message=question,
            loop_tools=planning_tools,
            loop_observers=planning_observers,
            loop_policy=_planning_loop_policy(),
            turns=planning_max_turns,
        )
        # Loop ended on finish_planning OR planning_max_turns — ensure execution.
        # (no-op if already in execution; force_finish_planning self-guards.)
        force_finish_planning(ctx.task_id)
        exec_user = (
            f"{question}\n\n--- YOUR TASK BOARD (from planning) ---\n"
            f"{render_board(ctx.task_id, bus_task_id=bus_task_id)}"
        )
        result = await _run_main_loop(
            system_prompt=_decorate(get_main_system_prompt(
                date_str, fs_mode=fs_mode, phase="execution",
                sub_agent_tools=sub_agent_tool_names,
                mcp_tool_names=mcp_tool_names, mcp_tool_specs=mcp_tool_specs,
                document_toolchain_note=main_document_toolchain_note,
            )),
            user_message=exec_user,
            loop_tools=tools,
            loop_observers=observers,
            loop_policy=_swarm_main_loop_policy(),
            turns=max_turns,
        )
    else:
        # ── Single-loop path ─────────────────────────────────────────────────
        # planning_mode off → no gate; planning_mode on + fresh off → planning
        # and execution share this one loop/context (PlanningGateObserver gates
        # tools while planning and auto-finishes at planning_max_turns).
        result = await _run_main_loop(
            system_prompt=_decorate(get_main_system_prompt(
                date_str, fs_mode=fs_mode, planning_mode=planning_mode,
                phase="combined",
                sub_agent_tools=sub_agent_tool_names,
                mcp_tool_names=mcp_tool_names, mcp_tool_specs=mcp_tool_specs,
                document_toolchain_note=main_document_toolchain_note,
            )),
            user_message=question,
            loop_tools=tools,
            loop_observers=observers,
            loop_policy=_swarm_main_loop_policy(),
            turns=max_turns,
        )

    if main_sb_token is not None:
        clear_task_sandbox(main_sb_token)
    if reporter_enabled:
        # The downstream reporter owns final synthesis and its fallback chain.
        # Avoid spending a duplicate coordinator LLM call here; preserve a
        # deterministic fail-open baseline and hand off the retained history
        # immediately.
        result = prepare_report_handoff(
            result,
            task_description=question,
        )
    else:
        result = await force_final_answer(
            result,
            llm,
            finalization_timeout_s,
            task_description=question,
            structured_report=False,
        )

    aggregate = bus.drain_task_metadata(bus_task_id)
    sub_evidence = list(aggregate.get("evidence_cards", []))
    sub_assertions = list(aggregate.get("assertions", []))

    main_evidence = list(result.metadata.get("evidence_cards", []))
    main_assertions = list(result.metadata.get("assertions", []))

    clarified_questions = _extract_sub_questions(
        result.messages, bus, bus_task_id,
    )
    live_followups = _collect_live_followups(observers)
    effective_question = _compose_effective_question(question, live_followups)

    logger.info(
        "Swarm main done (task=%s): sub_ev=%d sub_as=%d main_ev=%d "
        "main_as=%d sub_questions=%d",
        ctx.task_id, len(sub_evidence), len(sub_assertions),
        len(main_evidence), len(main_assertions), len(clarified_questions),
    )

    try:
        cleaned = await asyncio.wait_for(
            bus.cleanup_task(bus_task_id), timeout=45.0,
        )
        if cleaned:
            logger.info("Swarm cleanup: dropped %d session(s)", cleaned)
    except TimeoutError:
        logger.warning(
            "AgentBus cleanup exceeded 45s (task=%s) — detaching stuck "
            "sessions and proceeding to report", bus_task_id,
        )
    except Exception as exc:
        logger.warning("AgentBus cleanup failed: %s", exc)

    # Drain post-cleanup rescue (see AgentBus.cleanup_task).
    rescued = bus.drain_task_metadata(bus_task_id)
    rescued_ev = list(rescued.get("evidence_cards", []))
    rescued_as = list(rescued.get("assertions", []))
    if rescued_ev or rescued_as:
        sub_evidence.extend(rescued_ev)
        sub_assertions.extend(rescued_as)
        logger.info(
            "Swarm rescued post-cleanup (task=%s): ev=%d as=%d",
            ctx.task_id, len(rescued_ev), len(rescued_as),
        )

    research_notebook: dict[str, Any] = {}
    merged_evidence = sub_evidence + main_evidence

    # Last-line defense: even after the report handoff / force-final strip pass,
    # paranoid double-check that the harness never writes leaked
    # ``<tool_call>`` XML as the final answer (e.g. qwen text-mode
    # tool-call collapse on ``stopped_by={max_turns,no_tool,llm_error}``
    # paths). Both finalization paths already sanitise their baseline, so
    # this is defense-in-depth — covers the rare edge case where
    # ``metadata["final_answer"]`` was set by a different observer
    # earlier in the loop with leaked content.
    final_text = result.metadata.get("final_answer") or result.final_content
    cleaned = _strip_leaked_tool_calls(final_text or "")
    if cleaned != (final_text or "").strip():
        logger.warning(
            "swarm main: stripped leaked tool_call XML from final_answer "
            "(stopped_by=%s, before=%d chars, after=%d chars)",
            result.stopped_by, len(final_text or ""), len(cleaned),
        )
    final_text = cleaned or _minimal_best_effort_answer(
        question, result.stopped_by,
    )
    if not cleaned:
        result.metadata["final_answer_source"] = "deterministic_fallback"
    answer_source = str(result.metadata.get("final_answer_source") or "agent")
    answer_status = (
        "not_found"
        if answer_source == "deterministic_fallback"
        else "best_effort"
        if answer_source in {
            "clean_context_llm",
            "existing_partial",
            "collected_reports",
        }
        else "complete"
    )
    # Citation URLs the coordinator wrote are free text: it has no search
    # tool, so it copies whatever spelling its sub-agents reported, and a
    # sub-agent that "tidied" a URL (dropped a query string, appended
    # ``.html``) has already broken the link by this point. Rewrite each one
    # to the spelling the tools actually returned, from this run's evidence
    # cards. Runs unconditionally: when the reporter IS enabled its citation
    # gate supersedes this text anyway, and when it is off — the common case,
    # since a deployment-level profile override can disable it — this is the
    # only check standing between a mangled URL and the user.
    final_text, url_repair_stats = repair_citation_urls(final_text, merged_evidence)
    if url_repair_stats["repaired"] or url_repair_stats["ambiguous"]:
        logger.info(
            "agent_team citation URLs: repaired=%d ambiguous=%d unmatched=%d "
            "checked=%d",
            url_repair_stats["repaired"], url_repair_stats["ambiguous"],
            url_repair_stats["unmatched"], url_repair_stats["checked"],
        )

    if reporter_enabled and result.metadata.get("report_handoff"):
        logger.info(
            "agent_team: research stopped by %s; advancing to downstream reporter",
            result.metadata.get("report_handoff_reason"),
        )

    # Drop this run's task board so it doesn't leak across trials in a
    # long-lived worker process (no-op if the agent never used add_task).
    clear_board(ctx.task_id)

    return {
        "final_answer": final_text,
        # Mirror to ``final_content`` so the scheduler's terminal-output
        # check (frontier_agent/scheduling/scheduler.py:_has_terminal_output)
        # passes when the workflow runs without a separate ``report`` node.
        "final_content": final_text,
        "session_turn": build_session_turn(
            str(state.get("current_query") or question).strip(),
            result.messages,
            final_text,
            steps=result.metadata.get("react_steps", []),
        ),
        "answer_confidence": result.metadata.get("final_answer_confidence", ""),
        "evidence_cards": merged_evidence,
        "assertions": sub_assertions + main_assertions,
        "clarified_questions": clarified_questions,
        "react_steps": result.metadata.get("react_steps", []),
        "research_notebook": research_notebook,
        "live_followups": live_followups,
        "effective_question": effective_question,
        "answer_status": answer_status,
        "answer_sentinel": (
            "<ANSWER_NOT_FOUND>" if answer_status == "not_found" else ""
        ),
        "final_answer_rescued": bool(
            result.metadata.get("final_answer_rescued", False),
        ),
        "final_answer_rescue_mode": str(
            result.metadata.get("final_answer_rescue_mode") or "",
        ),
        "final_answer_source": answer_source,
        # Conditional edge in both agent-team specs consumes this resolved
        # per-request value. False routes directly to END, preserving the
        # coordinator's answer as the protocol final.
        "reporter_enabled": reporter_enabled,
        "reporter_wall_time_s": reporter_wall_time_s,
        "reporter_deadline_monotonic_s": hard_deadline_monotonic,
        # The reporter node is a stable production surface. Profiles select
        # its implementation without changing pipeline ids, protocol agent ids,
        # terminals, or trace ownership.
        "reporter_backend": reporter_backend,
        # Coordinator run stats — surfaced to state so an external runner can
        # fold them into its own run result (it reads these top-level keys),
        # which serve then reports in the coordinator's ``run_finished`` for
        # agent_team_report. Without these the finished event carried 0/0. The
        # downstream reporter node doesn't return these keys, so they survive
        # the state merge unchanged. (node_function outputs are merged wholesale
        # — ``output_fields`` only filters prompt-template nodes.)
        "turns_used": result.turns_used,
        "tool_calls_count": result.tool_calls_count,
        "stopped_by": result.stopped_by or "",
        # The terminal/TUI must be able to distinguish an infrastructure or
        # provider failure from a completed report. Keep the provider response
        # out of the fallback prose and carry it as structured run metadata.
        "llm_error": str(result.metadata.get("llm_error") or ""),
        "llm_error_reason": str(result.metadata.get("llm_error_reason") or ""),
    }


__all__ = ["main_agent_node"]
