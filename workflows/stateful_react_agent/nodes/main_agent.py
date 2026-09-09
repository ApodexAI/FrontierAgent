"""Native ReAct agent node - single stateful agent."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from frontier_agent.components.finalization import (
    ResearchWall,
    check_wall_feasibility,
    nonnegative_seconds,
    positive_seconds,
    resolve_research_wall,
)
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
from frontier_agent.components.observers.repetition_guard import RepetitionGuard
from frontier_agent.components.observers.sse_observer import SSEObserver
from frontier_agent.components.observers.stuck_target_guard import StuckTargetGuard
from frontier_agent.components.observers.text_repetition_guard import (
    TextRepetitionGuard,
)
from frontier_agent.components.observers.trajectory import TrajectoryFileObserver
from frontier_agent.components.observers.wall_clock_observer import (
    WallClockDeadlineObserver,
)
from frontier_agent.core.loop_types import LoopConfig, LoopPolicy
from frontier_agent.core.messages import text_of, user_msg
from frontier_agent.core.runtime import registry
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.runtime.loop.budget_consistency import (
    check_context_budget,
)
from frontier_agent.core.runtime.loop.compact import KeepLastNToolResultsCompactor
from frontier_agent.core.runtime.loop.llm_client import bind_temperature, extract_model_name
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
from frontier_agent.core.runtime.pause_check import pause_check_from_state
from frontier_agent.core.runtime.resources.manager import ResourceManager
from frontier_agent.core.runtime.session_history import build_session_turn
from frontier_agent.infra.wall_time_lease import WALL_TIME_LEASE_SCOPE_KEY
from frontier_agent.models.node_context import NodeContext
from frontier_agent.state.event_store.sqlite import EventStore
from frontier_agent.utils.history_input import extract_current_query
from frontier_agent.utils.language import (
    detect_language_from_prompt,
    is_language_detect_enabled,
    language_instruction,
    resolve_language,
)
from plugins.tools._bash_policy import reset_policy_mode, set_policy_mode
from plugins.tools._sandbox import (
    BwrapSandbox,
    SandboxUnavailableError,
    bwrap_available,
    clear_task_sandbox,
    make_current_sandbox,
    resolve_mount_dirs,
    resolve_sandbox_mode,
    set_task_sandbox,
)
from plugins.tools.task_board import build_task_board_observer, clear_board
from workflows.stateful_react_agent._runtime import (
    ReactToolResultPostProcessor,
    _minimal_best_effort_answer,
    _strip_leaked_tool_calls,
    _strip_thinking,
    render_system_prompt_notes,
)
from workflows.stateful_react_agent.observers import (
    FinalAnswerSalvageObserver,
    ReporterStreamObserver,
    ReportSynthesisObserver,
    RichConsoleObserver,
)
from workflows.stateful_react_agent.prompts import (
    BOARD_PROMPT_ADDENDUM,
    get_direct_system_prompt,
    get_react_system_prompt,
)

logger = logging.getLogger(__name__)

REACT_MAX_TURNS = 100
REACT_TOOL_TIMEOUT_S = 1800
REACT_LLM_TIMEOUT_S = 1800
REACT_KEEP_LAST_K = 5
REACT_COMPACT_AFTER_TURNS = 0
REACT_CONTEXT_TOKEN_LIMIT = 180_000

_STATEFUL_FINALIZATION_MESSAGE = (
    "Finalization phase has started. Stop new research and implementation "
    "branches. Use the remaining tool-enabled turns to finish the requested "
    "work, copy the best current deliverables to /outputs, and run only the "
    "minimum checks needed to avoid shipping broken files. Then answer the "
    "user in plain text. If full completion is impossible, preserve the best "
    "existing artifacts and give a useful partial answer; never finish with "
    "no deliverable and no answer."
)

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
            label="stateful wall_deadline_reserve_s",
        )
        if hard_wall_reserve_s is None
        else max(float(hard_wall_reserve_s), 0.0)
    )
    return resolve_research_wall(
        agent_cfg, reserve_s=reserve_s, label_prefix="stateful",
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


def _resolve_runaway_guardrails(
    agent_cfg: dict[str, Any],
) -> tuple[float | None, int | None, float | None]:
    """Resolve ``(reasoning_only_timeout_s, reasoning_only_max_tokens,
    logical_call_timeout_s)`` from the profile.

    Absent or ``0`` means off for each — and off matters more than it looks:
    the semantic reasoning watchdog only exists on the *streaming* request
    path, and the loop picks that path precisely because one of the
    ``reasoning_only_*`` values is set. With both unset, a reply that spends
    its whole completion budget inside the reasoning channel can only be
    detected after the fact, once the provider has already billed it.
    """
    timeout_raw = agent_cfg.get("reasoning_only_timeout_s")
    tokens_raw = agent_cfg.get("reasoning_only_max_tokens")
    logical_raw = agent_cfg.get("logical_call_timeout_s")
    return (
        float(timeout_raw) if timeout_raw else None,
        int(tokens_raw) if tokens_raw else None,
        float(logical_raw) if logical_raw else None,
    )


def _resolve_finalization_timeout_s(
    agent_cfg: dict[str, Any],
    *,
    llm_timeout_s: float,
) -> float:
    """Resolve the clean-context rescue timeout per fallback leg.

    Absent or ``0`` falls back to ``llm_timeout_s``; ``0`` does NOT mean
    "unlimited" here.
    """
    value = _positive_seconds(
        agent_cfg.get("finalization_timeout_s"),
        label="stateful finalization_timeout_s",
    )
    return value or max(float(llm_timeout_s), 1.0)


def _resolve_reporter_timeout_s(
    agent_cfg: dict[str, Any],
    *,
    llm_timeout_s: float,
) -> float:
    """Resolve a finite per-leg reporter read timeout.

    Absent or ``0`` falls back to ``llm_timeout_s`` — unlike the wall-time keys,
    ``0`` here does NOT mean "unlimited"; an unbounded leg is never wanted.
    """
    value = _positive_seconds(
        agent_cfg.get("reporter_timeout_s"),
        label="stateful reporter_timeout_s",
    )
    return value or max(float(llm_timeout_s), 1.0)


def _resolve_reporter_phase_timeout_s(
    agent_cfg: dict[str, Any],
    *,
    llm_timeout_s: float,
) -> float:
    """Resolve the absolute ceiling around the complete fallback chain.

    Absent or ``0`` falls back to ``llm_timeout_s * 3``; ``0`` does NOT mean
    "unlimited" here. The runtime clamps the result further when a platform
    hard wall leaves less time than this.
    """
    value = _positive_seconds(
        agent_cfg.get("reporter_phase_timeout_s"),
        label="stateful reporter_phase_timeout_s",
    )
    return value or max(float(llm_timeout_s) * 3, 1.0)


def _language_probe(state: dict[str, Any], question: str) -> str:
    """Return only the latest user instruction for answer-language detection.

    Legacy multi-turn requests may fold prior turns into ``question`` with the
    current query appended last. A long history can otherwise dilute the
    heuristic or crowd out an explicit language instruction. Prefer the clean
    ``current_query`` field; ``extract_current_query`` keeps wrapped callers
    safe.
    """
    return (
        str(state.get("current_query") or "").strip()
        or extract_current_query(question).strip()
        or question
    )


def _resolve_answer_language(state: dict[str, Any], question: str) -> str:
    """Resolve the query language used by the agent and optional reporter.

    The SDK currently seeds every run with legacy ``language="en"`` even when
    the caller supplied no preference.  Treat that default like ``"auto"`` so
    non-English queries are detected, matching the agent_team reporter's
    query-language behavior.  Other explicit language values remain
    authoritative.

    Detection uses the cleaned task text (``question`` has already had the
    protocol ``# Task`` wrapper removed).
    """
    requested = str(state.get("language", "auto") or "auto").strip()
    if requested.lower() == "en":
        requested = "auto"
    language_state = {
        "language": requested,
        "original_question": question,
    }
    return resolve_language(language_state) or "English"


def _flag(value: Any, *, default: bool) -> bool:
    """Coerce a profile/metadata boolean that may have come from env substitution.

    ``_resolve_env_vars`` yields strings, so a profile writing
    ``reporter: ${REPORTER:-false}`` hands this the string ``"false"`` — and
    ``bool("false")`` is True, which silently inverts the operator's intent.
    Vocabulary matches ``workflows/agent_team/nodes/main_agent.py``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


async def _resolve_answer_language_with_llm(
    state: dict[str, Any],
    question: str,
    *,
    answer_language: str,
    reporter_enabled: bool,
    llm: Any,
    llm_timeout: float,
    profile: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> str:
    """Upgrade auto language detection using the already-resolved profile LLM."""
    requested = str(state.get("language", "auto") or "auto").strip().lower()
    if (
        not reporter_enabled
        or requested not in ("", "auto", "en")
        or not is_language_detect_enabled()
    ):
        return answer_language

    async def _detect_ask(prompt: str) -> str:
        resp = await llm.chat([user_msg(prompt)], timeout=llm_timeout)
        usage = dict(getattr(resp, "usage", None) or {})
        provider = str(
            (getattr(resp, "response_metadata", None) or {}).get(
                "provider_actually_used",
            )
            or "",
        )
        model = (
            getattr(resp, "model", "")
            or extract_model_name(llm, profile)
            or ""
        )
        from workflows._shared.sdk_shim import (
            record_language_detect_usage,
        )

        record_language_detect_usage(
            metadata.get("sdk_protocol_usage_aggregator"),
            usage=usage,
            provider=provider,
            model=model,
        )
        return _strip_thinking(text_of(resp.content))

    return (
        await detect_language_from_prompt(question, _detect_ask)
        or answer_language
    )


def _resolve_llm_and_profile(
    profile_name: str | None,
    *,
    profile_overrides: dict[str, Any] | None = None,
    profile_inline: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None, ModelProfile | None]:
    """Return ``(llm, profile_dict, model_profile)`` for this run."""
    if profile_name or profile_inline:
        from workflows.stateful_react_agent.profile import (
            build_react_model_profile,
            create_react_llm,
            load_react_profile,
        )

        bypass_cache = bool(profile_overrides) or profile_inline is not None
        cache_key = profile_name or "__inline__"
        if bypass_cache or cache_key not in _llm_cache:
            profile = load_react_profile(
                profile_name or "",
                overrides=profile_overrides,
                inline=profile_inline,
            )
            entry = (
                profile,
                create_react_llm(profile),
                build_react_model_profile(profile),
            )
            if not bypass_cache:
                _llm_cache[cache_key] = entry
        else:
            entry = _llm_cache[cache_key]
        profile, llm, model_profile = entry
        return llm, profile, model_profile

    llm = registry.get(ResourceManager).get_llm("stateful_react")
    return bind_temperature(llm, 0.0), None, None


def _resolve_trajectory_dir(state: dict[str, Any], task_id: str) -> Path:
    trial_dir = (state.get("metadata") or {}).get("_trial_dir")
    if trial_dir:
        return Path(trial_dir) / "agent" / "trajectories"
    if run_dir := os.environ.get("APODEX_RUN_DIR", "").strip():
        return Path(run_dir) / "trajectories" / task_id
    return Path("logs") / "stateful_react" / task_id / "trajectories"


def _resolve_worktree_root(state: dict[str, Any], task_id: str) -> Path:
    md = state.get("metadata") or {}
    trial_dir = md.get("_trial_dir")
    if trial_dir:
        return Path(trial_dir) / "sandbox" / "worktree"
    experiment = md.get("experiment")
    bench_task_id = md.get("bench_task_id")
    if experiment and bench_task_id:
        return (
            Path("experiments")
            / str(experiment)
            / "questions"
            / str(bench_task_id)
            / "worktree"
        )
    coding_root = md.get("coding_workspace_root")
    if coding_root:
        return Path(coding_root)
    return Path("logs") / "stateful_react" / task_id / "worktree"


def _resolve_sandbox_binds(
    state: dict[str, Any],
    worktree_root: Path,
) -> tuple[tuple[tuple[str, str, bool], ...], Path]:
    """Resolve benchmark-provided ``/inputs`` and shared ``/outputs`` mounts."""
    metadata = state.get("metadata") or {}
    outputs_dir = worktree_root.parent / "outputs"

    binds: list[tuple[str, str, bool]] = []
    dataset_root = str(metadata.get("_dataset_root") or "")
    for mount in metadata.get("_sandbox_mounts") or []:
        src = str(mount.get("src", "")).strip()
        dst = str(mount.get("dst", "")).strip()
        if not src or not dst:
            continue
        if not dst.startswith("/inputs"):
            logger.warning("sandbox mount dst not under /inputs, skipped: %s", dst)
            continue
        src_path = Path(src)
        if not src_path.is_absolute() and dataset_root:
            src_path = Path(dataset_root) / src_path
        read_only = str(mount.get("mode", "ro")).lower() != "rw"
        binds.append((str(src_path.expanduser().resolve()), dst, read_only))

    binds.append((str(outputs_dir), "/outputs", False))
    return tuple(binds), outputs_dir


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


def _log_inputs_dir_contents(
    roots: list[tuple[str, str]],
    *,
    fallback_roots: list[tuple[str, str]] | None = None,
    max_files: int = 200,
) -> None:
    """Diagnostic: log what actually lives under each ``/inputs`` root at runtime.

    ``roots`` is ``[(label, host_path), ...]`` where ``host_path`` is the real
    directory the model's file tools (``read_file`` / ``glob_search`` /
    ``grep_search``) will see as ``/inputs`` — in container mode the mounted
    ``/inputs`` itself, in bwrap mode the host ``src`` of each ``/inputs`` bind.
    For each root it logs the resolved absolute path, whether it exists, and
    every file beneath it (name + absolute path + size) so a mount/path mismatch
    (S3 files landing at a path the tools don't read) is visible in the worker
    log stream.

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
                "[stateful_react inputs] %s path=%s exists=%s is_dir=%s",
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
                    "[stateful_react inputs] %s path=%s is EMPTY — read_file / "
                    "glob_search will find nothing here", label, p,
                )
                continue
            found_any = True
            logger.info(
                "[stateful_react inputs] %s path=%s has %d file(s):",
                label, p, len(files),
            )
            for f in files:
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                logger.info(
                    "[stateful_react inputs]   name=%r abs=%s size=%s", f.name, f, size,
                )
        except Exception as exc:
            logger.warning(
                "[stateful_react inputs] failed to scan %s path=%s: %s",
                label, root, exc,
            )

    if not found_any and fallback_roots:
        logger.warning(
            "[stateful_react inputs] no files at expected input path(s); probing "
            "fallback locations to find where the mounted files landed",
        )
        for label, root in fallback_roots:
            p = Path(root)
            logger.warning(
                "[stateful_react inputs] fallback %s path=%s exists=%s entries=%s",
                label, p, p.exists(), _shallow_entries(root) or "(none/not-a-dir)",
            )


def _loop_policy() -> LoopPolicy:
    return LoopPolicy(terminal_tool_names=(), no_tool_behavior="stop")


def _tools_for_stateful_react(
    resource_mgr: ResourceManager,
    agent_cfg: dict[str, Any],
) -> list[Any]:
    override = agent_cfg.get("agent_tools")
    if not override:
        return resource_mgr.get_tools_for_role("stateful_react")

    names: list[str] = []
    for raw in override:
        name = str(raw).strip()
        if name and name not in names:
            names.append(name)
    # Closed-book has to be enforced *here* too, not only on the role's tool
    # pool. A profile's ``agent_tools`` list wins over the pool, and every
    # shipped profile lists the web tools explicitly — so honouring
    # REACT_NO_WEB only in the AgentDefinition made it silently inert for any
    # real run. This is the list that actually gets bound to the model.
    import os as _os

    from workflows.stateful_react_agent import WEB_TOOL_NAMES
    no_web = _os.environ.get("REACT_NO_WEB", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if no_web:
        dropped = [n for n in names if n in WEB_TOOL_NAMES]
        if dropped:
            names = [n for n in names if n not in WEB_TOOL_NAMES]
            logger.info(
                "closed-book (REACT_NO_WEB): dropped profile web tools %s", dropped,
            )
    # Profiles created before controlled downloads existed commonly list
    # web_fetch explicitly. Preserve that narrowing while adding its new
    # binary-file companion without requiring every deployed profile YAML to
    # change in lockstep.
    if "web_fetch" in names and "download_file" not in names:
        names.insert(names.index("web_fetch") + 1, "download_file")

    policy = resource_mgr.global_tool_policy
    all_tools = resource_mgr.all_tools
    tools: list[Any] = []
    skipped: list[str] = []
    for name in names:
        if policy is not None and not policy.allows(name):
            skipped.append(name)
            continue
        tool = all_tools.get(name)
        if tool is None:
            skipped.append(name)
            continue
        tools.append(tool)
    if skipped:
        logger.warning("stateful_react profile tools skipped: %s", skipped)
    logger.info("stateful_react tools selected by profile: %s", [t.name for t in tools])
    return tools


def _replace_tool_impls(tools: list[Any], agent_cfg: dict[str, Any]) -> list[Any]:
    out = list(tools)
    if (agent_cfg.get("web_search_impl") or "original") == "aligned":
        from plugins.tools.web_search_aligned import web_search_aligned

        out = [web_search_aligned if getattr(t, "name", "") == "web_search" else t for t in out]
    if (agent_cfg.get("web_fetch_impl") or "original") == "aligned":
        from plugins.tools.web_fetch_aligned import web_fetch_aligned

        out = [web_fetch_aligned if getattr(t, "name", "") == "web_fetch" else t for t in out]
    return out


async def react_agent_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    """Run the single stateful ReAct agent."""
    question = state.get("original_question", "")
    if not question:
        raise ValueError("react_agent_node requires 'original_question' in state")
    if question.startswith("# Task\n\n"):
        question = question[len("# Task\n\n") :]
    question = question.rstrip()

    metadata = state.get("metadata") or {}
    profile_name = (
        metadata.get("profile")
        or metadata.get("stateful_profile")
        or metadata.get("react_profile")
        or metadata.get("swarm_profile")
    )
    llm, profile, model_profile = _resolve_llm_and_profile(
        str(profile_name) if profile_name else None,
        profile_overrides=metadata.get("profile_overrides"),
        profile_inline=metadata.get("profile_inline"),
    )
    agent_cfg = (profile or {}).get("agent", {})

    max_turns = int(agent_cfg.get("main_max_turns", agent_cfg.get("max_turns", REACT_MAX_TURNS)))
    tool_timeout = float(agent_cfg.get("tool_timeout_s", REACT_TOOL_TIMEOUT_S))
    llm_timeout = float(agent_cfg.get("llm_timeout_s", REACT_LLM_TIMEOUT_S))
    first_chunk_s = agent_cfg.get("first_chunk_s")
    (
        reasoning_only_timeout_s,
        reasoning_only_max_tokens,
        logical_call_timeout_s,
    ) = _resolve_runaway_guardrails(agent_cfg)
    history_policy = resolve_history_policy(agent_cfg)
    keep_last_k = int(agent_cfg.get("keep_last_k", REACT_KEEP_LAST_K))
    compact_after_turns = int(agent_cfg.get("compact_after_turns", REACT_COMPACT_AFTER_TURNS))
    context_token_limit = int(agent_cfg.get("context_token_limit", REACT_CONTEXT_TOKEN_LIMIT))
    max_input_tokens = int(agent_cfg.get("max_input_tokens", 0) or 0)
    # Tiered context compaction (opt-in via profile ``context_compaction``):
    #   "off" (default) → legacy keep_last_k path.
    #   "tiered"        → compact ONLY when the REAL input tokens cross
    #     ``max_len`` * 0.8; Tier1 keeps the last ``tier1_keep_tool_result`` tool
    #     results (drops older), Tier2 LLM-summarises the middle only if Tier1
    #     left the estimate above max_len*0.6. ``max_len`` = model context window.
    context_compaction = str(agent_cfg.get("context_compaction", "off")).lower()
    compaction_spill = _flag(agent_cfg.get("compaction_spill"), default=False)
    max_len = int(agent_cfg.get("max_len", 0) or 0)
    # See the sibling call in agent_team: the sglang doctor covers the compose
    # path only, and nothing checked the values the loop is actually built from.
    check_context_budget(
        max_len=max_len,
        max_input_tokens=max_input_tokens,
        max_tokens=int((profile or {}).get("llm", {}).get("max_tokens", 0) or 0) or None,
        reasoning_only_max_tokens=reasoning_only_max_tokens,
        label="stateful_react",
    )
    tier1_keep_tool_result = int(agent_cfg.get("tier1_keep_tool_result", 5))
    keep_recent_turns = int(agent_cfg.get("keep_recent_turns", 5))
    fs_mode = bool(agent_cfg.get("fs_mode", False)) or bool(metadata.get("fs_mode", False))
    # Opt-in task-board mode: append the board addendum so the agent uses the
    # add_task / update_task board tools as a status checklist (open →
    # in_progress → resolved / cancelled). Off by default; the board tools must
    # also be in the profile's ``agent_tools``.
    task_board = bool(agent_cfg.get("task_board", False))
    # Opt-in direct-inference mode: NO tools are bound, so the model
    # answers from its own knowledge in a single turn (the first tool-free reply
    # is the final answer). Uses the tool-free direct system prompt and skips the
    # task-board / sandbox-FS prompt scaffolding, which are meaningless without
    # tools. The sandbox is still constructed below (harmless — unused when no
    # tool can invoke it).
    direct = bool(agent_cfg.get("direct", False))
    # Salvage philosophy for abnormal/infra exits (llm_error / wall_deadline /
    # budget_exhausted / max_attempts): True (default) = make a clean-context
    # LLM rescue call; False = skip that extra call. Both paths still return a
    # deterministic non-empty best-effort status when no answer was generated.
    salvage_infra_errors = bool(agent_cfg.get("salvage_infra_errors", True))
    # Stop-loss (both opt-in, 0 = off):
    #   ``research_wall_time_s`` (legacy alias ``wall_deadline_s``) — research
    #     budget only. The loop exits cleanly with ``wall_deadline`` and the
    #     reporter takes over outside that budget.
    #   ``stuck_target_hint_after`` — confirmed failures for one network host
    #     within the last ``stuck_target_window`` network turns. A fetch success
    #     resets that host. The first threshold asks for a route change; the
    #     escalation threshold (HARD failures only) quarantines the host for the
    #     rest of the loop.
    finalization_reserve_turns = int(
        agent_cfg.get("finalization_reserve_turns", 8) or 8,
    )
    finalization_timeout_s = _resolve_finalization_timeout_s(
        agent_cfg,
        llm_timeout_s=llm_timeout,
    )
    stuck_hint_after = int(agent_cfg.get("stuck_target_hint_after", 0) or 0)
    stuck_escalate_after = int(
        agent_cfg.get("stuck_target_escalate_after", stuck_hint_after * 2) or 0,
    )
    stuck_window = int(agent_cfg.get("stuck_target_window", 20) or 20)
    # Opt-in lightweight reporter: when on, a single tool-free
    # streaming LLM call synthesises a structured, cited report over the whole
    # conversation at loop end (standard-mode reporter parity), and it
    # REPLACES the raw-answer ReporterStreamObserver. Off by default → direct
    # answer, current behaviour. Profile ``agent.reporter`` sets it; a metadata
    # ``reporter`` key overrides (per-request opt-in without a profile edit).
    reporter_enabled = _flag(agent_cfg.get("reporter"), default=False)
    if metadata.get("reporter") is not None:
        reporter_enabled = _flag(metadata.get("reporter"), default=False)
    reporter_timeout_s = _resolve_reporter_timeout_s(
        agent_cfg,
        llm_timeout_s=llm_timeout,
    )
    reporter_phase_timeout_s = _resolve_reporter_phase_timeout_s(
        agent_cfg,
        llm_timeout_s=llm_timeout,
    )
    configured_wall_reserve_s = _nonnegative_seconds(
        agent_cfg.get("wall_deadline_reserve_s"),
        default=180,
        label="stateful wall_deadline_reserve_s",
    )
    landing_budget_s = (
        reporter_phase_timeout_s
        if reporter_enabled
        else finalization_timeout_s
    )
    # A tool may start just before the turn-end observer checks the research
    # deadline. Reserve enough for that complete overrun plus the bounded
    # reporter/finalization phase. The overrun is the tool's OUTER wait, not
    # the configured timeout — budget-aware tools get a grace on top so they
    # can report their own timeout — so ask the loop rather than assuming.
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
    # wall the reserve above is NOT what actually survives. Hand the
    # finalization stage an absolute instant instead of a static budget: it
    # clamps itself to the time really left and fails open with a real answer,
    # rather than being killed mid-call by the platform ceiling (the live serve
    # ``RenewableWallTimeLease`` gives only a few seconds of grace).
    node_started_monotonic = time.monotonic()
    hard_deadline_monotonic = (
        node_started_monotonic + research_wall.hard_total_s
        if research_wall.hard_total_s > 0
        else None
    )
    check_wall_feasibility(
        hard_total_s=research_wall.hard_total_s,
        research_deadline_s=wall_deadline_s,
        tool_timeout_s=worst_case_tool_s,
        landing_budget_s=landing_budget_s,
        label_prefix="stateful",
    )
    reporter_context_default = (
        max(1_024, max_input_tokens - 4_096)
        if max_input_tokens > 0
        else 220_000
    )
    reporter_context_max_tokens = int(
        agent_cfg.get(
            "reporter_context_max_tokens",
            reporter_context_default,
        )
        or reporter_context_default,
    )
    language_probe_text = _language_probe(state, question)
    answer_language = _resolve_answer_language(state, language_probe_text)
    # When the caller left the language unpinned, upgrade the character
    # heuristic to a single-call LLM detector (arbitrary-language coverage —
    # Spanish/Vietnamese/… that the CJK heuristic collapses to English).
    # The shared detector falls back to that same heuristic on any LLM error,
    # so this never does worse than the line above. The helper also gates on
    # reporter mode and ``LANGUAGE_DETECT_ENABLED``.
    answer_language = await _resolve_answer_language_with_llm(
        state,
        language_probe_text,
        answer_language=answer_language,
        reporter_enabled=reporter_enabled,
        llm=llm,
        llm_timeout=llm_timeout,
        profile=profile,
        metadata=metadata,
    )

    resource_mgr = registry.get(ResourceManager)
    tools = _replace_tool_impls(
        _tools_for_stateful_react(resource_mgr, agent_cfg),
        agent_cfg,
    )
    # Direct-inference mode: drop all tools so the model answers in one turn.
    if direct:
        tools, tool_names = [], []
    else:
        tool_names = [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]

    if direct:
        system_prompt = get_direct_system_prompt()
    else:
        system_prompt = get_react_system_prompt(fs_mode=fs_mode)
    addendum = str(metadata.get("_sys_prompt_addendum") or "").strip()
    if addendum:
        system_prompt = f"{system_prompt}\n\n{addendum}"
    # task-board + sandbox-FS notes are tool-dependent — skip them in direct mode.
    if task_board and not direct:
        system_prompt = f"{system_prompt}{BOARD_PROMPT_ADDENDUM}"
    if reporter_enabled:
        # Keep the research agent's draft/salvage answer in the same language
        # as the reporter.  This also makes the fail-open path language-stable
        # if report synthesis fails.
        system_prompt = f"{system_prompt}{language_instruction(answer_language)}"

    # Sandbox mode: trusted deployment config must explicitly select container;
    # a profile may only tighten auto to bwrap, never attest container isolation.
    #   container — the surrounding docker container IS the isolation; tools
    #     operate directly on the mounted /workspace, /outputs, /inputs via a
    #     CurrentSandbox. No bwrap needed. This is the production model.
    #   bwrap / auto — local benchmark path: bwrap namespaces (unchanged).
    sandbox_mode = resolve_sandbox_mode(agent_cfg)
    sandbox_binds: tuple[tuple[str, str, bool], ...] = ()
    workspace_root_str = ""
    # Bound only on the container/native branch below; the one reader is guarded
    # by the same sandbox_mode test, so "" never reaches it.
    inputs_dir = ""
    if sandbox_mode in ("container", "native"):
        workspace_dir, outputs_dir_str, inputs_dir = resolve_mount_dirs()
        worktree_root = Path(workspace_dir)
        outputs_dir = Path(outputs_dir_str)
        workspace_root_str = str(worktree_root)
    else:
        worktree_root = _resolve_worktree_root(state, ctx.task_id)
        sandbox_binds, outputs_dir = _resolve_sandbox_binds(state, worktree_root)
    worktree_root.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # /inputs is an external bind-mount (Worker Shell syncs it from S3); the
    # harness never populates it. Log what actually landed there so a
    # missing/misplaced input file is diagnosable from the worker log.
    if sandbox_mode in ("container", "native"):
        _log_inputs_dir_contents(
            [("container /inputs", inputs_dir)],
            fallback_roots=[
                ("workspace", str(worktree_root)),
                ("root", "/"),
                ("cwd", str(Path.cwd())),
            ],
        )
    else:
        _log_inputs_dir_contents(
            [
                (f"bwrap bind {dst}", src)
                for (src, dst, _ro) in sandbox_binds
                if dst.startswith("/inputs")
            ]
        )

    # Filesystem tools are usable in container mode OR when bwrap is present;
    # add the /workspace, /outputs, /inputs convention note accordingly.
    fs_enabled = sandbox_mode in ("container", "native") or bwrap_available()
    if fs_enabled and not direct:
        # Charts are written through the same filesystem tools, so the
        # clipping rule rides along with the /workspace-vs-/outputs note.
        runtime_notes = render_system_prompt_notes(
            sandbox_mode=sandbox_mode,
            tool_names=tool_names,
        )
        system_prompt = f"{system_prompt}{runtime_notes}"
    elif not direct:
        # auto/bwrap mode without bwrap must fail closed. CurrentSandbox only
        # changes cwd; it does not isolate the host filesystem or network.
        # Container mode is safe only when trusted deployment configuration
        # selected it explicitly and the surrounding container is the boundary.
        raise SandboxUnavailableError(
            "stateful-react-agent requires bubblewrap for tool execution "
            "(or SANDBOX_BACKEND=container inside an isolated task container); "
            "refusing unisolated host fallback"
        )

    event_store = registry.get_optional(EventStore)
    model_name = extract_model_name(llm)
    observers: list[Any] = [
        LeakedToolCallRetryObserver(tool_names=tool_names),
        RichConsoleObserver(),
        TrajectoryFileObserver(
            _resolve_trajectory_dir(state, ctx.task_id),
            filename="react_agent",
            tools=tools,
            model_name=model_name,
            system_prompt=system_prompt,
            user_message=question,
        ),
        ReactStepTracker(),
    ]
    if not direct:
        # Repetition stop-loss. Both of these stay hint-only: this agent IS
        # the run, so a false positive must cost one message, never the answer.
        observers.append(RepetitionGuard())
        observers.append(TextRepetitionGuard())
        if DuplicateQueryRollbackObserver.DEFAULT_TOOL_NAMES.intersection(tool_names):
            # Pops the turn before the duplicate search runs. Matters most
            # here: main_max_turns reaches 600 in the TUI profile, and tiered
            # compaction discards the older search results that would
            # otherwise remind the model it already ran this query.
            observers.append(DuplicateQueryRollbackObserver())
    # Task-board mode: re-inject the board on a cooldown so it survives
    # KeepLastN compaction (parity with agent_team). No-op if the agent never
    # writes a board; meaningless in direct mode (no tools).
    if task_board and not direct:
        observers.append(build_task_board_observer())
    if not direct:
        observers.append(FinalizationReserveObserver(
            reserve_turns=finalization_reserve_turns,
            message=_STATEFUL_FINALIZATION_MESSAGE,
        ))
        # The reserved turns above are tool-enabled so artifacts can still be
        # completed.  Strip tools only for the actual landing turn.
        observers.append(LastTurnForcer(terminal_tool=""))
    if max_input_tokens > 0:
        observers.append(ContextSizeGuard(
            max_input_tokens=max_input_tokens,
            force_compaction_first=(context_compaction == "tiered" and max_len > 0),
        ))
    # Stop-loss, both no-ops unless the profile opts in (see above).
    if stuck_hint_after > 0 and not direct:
        observers.append(StuckTargetGuard(
            hint_after=stuck_hint_after,
            escalate_after=stuck_escalate_after or stuck_hint_after * 2,
            window=stuck_window,
        ))
    if wall_deadline_s > 0:
        observers.append(WallClockDeadlineObserver(
            deadline_s=wall_deadline_s,
            # ``_resolve_wall_deadline_s`` already converted any hard
            # operational wall into this research-only soft deadline.
            reserve_s=0,
        ))
    if event_store is not None:
        observers.append(
            SSEObserver(
                event_store=event_store,
                task_id=ctx.task_id,
                run_id=str(metadata.get("run_id") or ""),
                run_type=str(metadata.get("run_type") or ""),
            ),
        )
    # Reporter-disabled runs need a clean-context rescue on bounded/infra exits.
    # When the reporter is enabled it is already the authoritative clean-context
    # LLM synthesis chain, so a coordinator rescue here would duplicate the
    # largest call immediately before it.
    if not reporter_enabled:
        observers.append(FinalAnswerSalvageObserver(
            llm=llm,
            timeout=finalization_timeout_s,
            phase_deadline_monotonic=hard_deadline_monotonic,
            language=answer_language,
            task_description=question,
            thinking_format=(
                model_profile.thinking_format if model_profile is not None else "tag"
            ),
            salvage_infra_errors=salvage_infra_errors,
        ))

    # Reporter output stream (serve stdout): re-emit the final answer as
    # ``response.swarm.llm_delta`` frames with ``agent_id="reporter"`` at
    # on_loop_end — immediately before the protocol stream observer's
    # terminal ``final``. MUST be appended to the node's own observers before the serve
    # chain in ``sdk_extra_observers`` so its critical on_loop_end runs before
    # the terminal.
    #
    # Two mutually exclusive modes:
    #   reporter_enabled → ReportSynthesisObserver: ONE streaming LLM call
    #     synthesises a cited report over the whole conversation and rewrites
    #     ``final_answer`` (runs regardless of an emitter — it improves the
    #     returned/traced answer; the emitter only adds live streaming).
    #   otherwise        → ReporterStreamObserver: re-streams the raw resolved
    #     answer (no-op without an emitter, so wire it only when one is present).
    sdk_emitter = metadata.get("sdk_protocol_emitter")
    if reporter_enabled:
        thinking_fmt = (
            model_profile.thinking_format if model_profile is not None else "tag"
        )
        # Inline-thinking risk: tag-mode + ``enable_thinking`` means the endpoint
        # MAY return ``<think>…</think>`` inside ``content`` — and SGLang/Qwen
        # often drop the opening tag while always emitting the closing one. The
        # reporter's stream filter needs to know so it never ships reasoning as
        # report text (it self-releases the hold on a native reasoning delta).
        chat_template_kwargs = (
            ((profile or {}).get("llm") or {}).get("extra_body") or {}
        ).get("chat_template_kwargs") or {}
        observers.append(ReportSynthesisObserver(
            llm=llm,
            # Finite timeout per fallback leg; deliberately no whole-reporter
            # phase wall, so the chain can advance after research has stopped.
            timeout=reporter_timeout_s,
            task_description=question,
            emitter=sdk_emitter,
            usage_aggregator=metadata.get("sdk_protocol_usage_aggregator"),
            language=answer_language,
            thinking_format=thinking_fmt,
            inline_thinking=(
                thinking_fmt == "tag"
                and bool(chat_template_kwargs.get("enable_thinking"))
            ),
            extra_observers=metadata.get("sdk_extra_observers"),
            context_max_tokens=reporter_context_max_tokens,
            phase_timeout=reporter_phase_timeout_s,
            phase_deadline_monotonic=hard_deadline_monotonic,
        ))
    elif sdk_emitter is not None:
        observers.append(ReporterStreamObserver(sdk_emitter))

    extra_observers = metadata.get("sdk_extra_observers") or []
    if extra_observers:
        observers.extend(list(extra_observers))

    sandbox = None
    sb_token = None
    if not direct:
        if sandbox_mode in ("container", "native"):
            sandbox = make_current_sandbox(worktree_root)
        else:
            sandbox = BwrapSandbox(workspace=worktree_root, binds=sandbox_binds)
        sb_token = set_task_sandbox(sandbox)
    # Default this eval workflow to the bash command allowlist (deny anything
    # off the read/analyse/python allowlist). This is defense-in-depth on top
    # of the required bwrap isolation.
    # Overridable via BASH_ALLOWLIST_MODE (env wins in resolve_mode()).
    policy_token = set_policy_mode("enforce")
    # Assemble the compactor + trigger. Tiered reuses KeepLastN (Tier1) +
    # LLMSummaryCompactor (Tier2) behind a real-input-token threshold; else the
    # legacy keep-last-k path (unchanged when ``context_compaction`` is off).
    keep_recent_msgs = max(6, keep_recent_turns * 3)  # ~3 msgs/turn (AI + tool(s))
    compaction_policy: Any = None
    if context_compaction == "tiered" and max_len > 0:
        gauge = InputTokenGauge()
        observers.append(gauge)
        from plugins.tools._overflow import spill_compacted_body

        compactor: Any = TieredCompactor(
            keep_tool_result=tier1_keep_tool_result,
            summary_llm=llm,
            relief_target=int(max_len * 0.6),
            gauge=gauge,  # calibrate relief to real tokens (unit-match trigger)
            spill=spill_compacted_body if compaction_spill else None,
            summary_retry_timeout_s=llm_timeout,
        )
        compaction_policy = InputTokenThresholdPolicy(
            gauge, compaction_trigger_tokens(max_len),
        )
    else:
        compactor = KeepLastNToolResultsCompactor(keep_tool_result=keep_last_k)

    try:
        import sys
        loop_fn = getattr(sys.modules.get("apodex.session"), "run_agent_loop", run_agent_loop)
        result = await loop_fn(
            system_prompt=system_prompt,
            user_message=question,
            llm=llm,
            tools=tools,
            config=LoopConfig(
                max_turns=max_turns,
                task_id=ctx.task_id,
                llm_session_id=str(metadata.get("session_id") or ctx.task_id),
                role_id="stateful_react",
                tool_timeout=int(tool_timeout),
                llm_timeout=int(llm_timeout),
                first_chunk_timeout=(
                    float(first_chunk_s) if first_chunk_s is not None else None
                ),
                reasoning_only_timeout_s=reasoning_only_timeout_s,
                reasoning_only_max_tokens=reasoning_only_max_tokens,
                logical_call_timeout_s=logical_call_timeout_s,
                context_token_limit=context_token_limit,
                compact_after_turns=compact_after_turns,
                keep_recent=keep_recent_msgs,
                loop_policy=_loop_policy(),
                compactor=compactor,
                compaction_policy=compaction_policy,
                tool_result_post_processor=ReactToolResultPostProcessor(),
            ),
            model_profile=model_profile,
            history_policy=history_policy,
            observers=observers,
            pause_check=pause_check_from_state(state),
            scope_metadata={
                "root_task_id": ctx.task_id,
                "run_id": metadata.get("run_id"),
                "run_type": metadata.get("run_type"),
                "pipeline_id": "stateful-react-agent",
                # Live ``add`` intervention renews this shared lease before its
                # queued ack. WallClockDeadlineObserver binds to it so the
                # in-process soft deadline follows the same sliding window.
                # Binding creates a per-loop deadline view; never forward the
                # view itself to sibling/concurrent loops.
                WALL_TIME_LEASE_SCOPE_KEY: metadata.get(
                    WALL_TIME_LEASE_SCOPE_KEY
                ),
                # Container mode: authorize file tools' direct-local access to
                # the mounted /workspace (consumed by plugins.tools._path_auth).
                # Empty string in bwrap mode → no host-workspace authorization.
                "workspace_root": workspace_root_str,
            },
        )
    finally:
        reset_policy_mode(policy_token)
        if sb_token is not None:
            clear_task_sandbox(sb_token)
        # Drop this run's task board (no-op when task_board is off) so boards
        # don't leak across trials in a long-lived worker process.
        clear_board(ctx.task_id)
        kill = getattr(sandbox, "kill", None)
        if callable(kill):
            kill()

    # ``force_final_answer`` for salvage stops ran at loop end via
    # Finalization is complete here: reporter-enabled runs were synthesised by
    # ``ReportSynthesisObserver``; reporter-disabled bounded exits were handled
    # by ``FinalAnswerSalvageObserver``.
    final_text = result.metadata.get("final_answer") or result.final_content
    final_text = _strip_leaked_tool_calls(str(final_text or ""))
    if not final_text:
        final_text = _minimal_best_effort_answer(
            question, result.stopped_by, language=answer_language,
        )
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
    return {
        "final_answer": final_text,
        "final_content": final_text,
        "session_turn": build_session_turn(
            _language_probe(state, question),
            result.messages,
            final_text,
            steps=result.metadata.get("react_steps", []),
        ),
        "react_steps": result.metadata.get("react_steps", []),
        # Authoritative loop telemetry. ``react_steps`` counts tool results,
        # not LLM turns, so it cannot substitute for either counter.
        "turns_used": result.turns_used,
        "tool_calls_count": result.tool_calls_count,
        "language": answer_language,
        # Keep the user-facing answer non-empty while preserving the old
        # machine-readable infra/eval failure signal out of band.
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
        "stopped_by": result.stopped_by,
        # Preserve the provider failure separately from deterministic fallback
        # prose so terminal clients can render an actionable error rather than
        # mislabeling the fallback as a completed report.
        "llm_error": str(result.metadata.get("llm_error") or ""),
        "llm_error_reason": str(result.metadata.get("llm_error_reason") or ""),
    }
