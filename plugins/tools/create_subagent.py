"""Tool: create_subagent — register persistent sub-agent sessions."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.runtime.loop.message_trimmer import TaskBoundaryTrimmer
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id, resolve_root_task_id
from plugins.tools._coerce import coerce_json_list

logger = logging.getLogger(__name__)

# create_subagent only registers sessions; actual concurrent execution is
# bounded by SpawnGuard(max_parallel). Raise via FRONTIER_AGENT_MAX_SUBAGENTS_PER_DISPATCH.
MAX_SUBAGENTS_PER_DISPATCH: int = int(
    os.environ.get("FRONTIER_AGENT_MAX_SUBAGENTS_PER_DISPATCH", "20")
)

SUB_ROLE_ID = "swarm_sub"
SUB_MAX_TURNS_DEFAULT = 100


def _resolve_sub_role_id(runtime: Any | None) -> str:
    """Sub-agent role id, workflow-overridable.

    Defaults to the shared ``swarm_sub`` registration (the name is
    historical — see ``workflows/agent_team/README.md``); a workflow can set
    ``sub_role_id`` on its runtime (agent_team → ``agent_team_sub``) so its
    sub-agents resolve their own fail-closed tool pool instead of sharing it.
    """
    return getattr(runtime, "sub_role_id", None) or SUB_ROLE_ID


# Sentinel: caller did not specify task_types → ask the workflow for its tuple
# (the behaviour assign_task / scope-rewrite callers rely on). Distinct from an
# explicit ``None`` (lenient role-label path) that create_subagent passes for
# agent_team.
_TASK_TYPES_UNSET: Any = object()


def _normalize_agent_name(
    name: str, task_types: tuple[str, ...] | None = _TASK_TYPES_UNSET,
) -> str:
    """Auto-fix underscored topic to dashed topic (strict-naming convention).

    The convention is ``{topic}_{task_type}[_{N}]`` where the topic segment
    uses dashes for multi-word descriptors. LLMs often emit
    ``gpu_market_research_1`` instead of canonical ``gpu-market_research_1``;
    rather than reject, we rewrite the topic span when the suffix
    unambiguously identifies a ``task_type``. Names whose last segment isn't a
    known task_type are returned unchanged so :func:`_validate_agent_name`
    surfaces the real error.

    ``task_types is None`` (workflows like agent_team that use free-form
    role-label names) → return ``name`` unchanged: there is no
    ``{topic}_{task_type}`` structure to normalize.
    """
    if task_types is _TASK_TYPES_UNSET:
        task_types = _resolve_task_types(None)  # caller did not say → ask the workflow
    if not task_types:
        return name

    parts = name.split("_")
    if len(parts) < 2:
        return name

    last = parts[-1]
    if last.isdigit():
        if len(parts) < 3:
            return name
        task_type = parts[-2]
        topic_parts = parts[:-2]
        suffix = f"_{task_type}_{last}"
    else:
        task_type = last
        topic_parts = parts[:-1]
        suffix = f"_{task_type}"

    if task_type not in task_types:
        return name
    if len(topic_parts) <= 1:
        return name
    return "-".join(topic_parts) + suffix


def _validate_agent_name(
    name: str, task_types: tuple[str, ...] | None = _TASK_TYPES_UNSET,
) -> str | None:
    """Return an error message if ``name`` violates the naming convention.

    With ``task_types`` (strict mode): valid format is ``{topic}_{task_type}`` or
    ``{topic}_{task_type}_{N}`` where the topic uses dashes and ``task_type``
    is one of ``task_types``.

    With ``task_types is None`` (agent_team and any workflow using free-form
    ROLE-label sub-agent names, resolved by substring in its prompts): use a
    LENIENT check — accept any safe identifier. The previous behaviour
    hard-coded a fixed task-type tuple here, so it rejected EVERY agent_team role
    name (``final_verifier`` / ``lit_search`` / ``match_researcher`` / …),
    created zero sub-agents, and deadlocked the planning-mode finalize gate.

    Returns ``None`` when valid. Callers should run :func:`_normalize_agent_name`
    first (a no-op in the lenient case).
    """
    if task_types is _TASK_TYPES_UNSET:
        task_types = _resolve_task_types(None)  # caller did not say → ask the workflow
    if not task_types:
        # Lenient role-label path: a safe identifier is all that's required;
        # the workflow's get_subagent_system_prompt resolves the specialist by
        # substring, so any reasonable label is valid. Reject only empty names
        # or shell-metacharacter/space injection in the session name.
        if (
            not name
            or not name[0].isalnum()
            or not all(c.isalnum() or c in "_-" for c in name)
        ):
            return (
                f"Agent name {name!r} must be a non-empty identifier "
                f"(letters/digits/_/-, starting alphanumeric; no spaces or "
                f"shell metacharacters)."
            )
        return None

    SUBAGENT_TASK_TYPES = task_types
    parts = name.split("_")
    # Need at least topic + task_type → 2 segments minimum.
    if len(parts) < 2:
        return (
            f"Agent name {name!r} must follow {{topic}}_{{task_type}}[_{{N}}]. "
            f"Valid task_types: {', '.join(SUBAGENT_TASK_TYPES)}. "
            f"Use dashes inside the topic: e.g. 'gpu-market_research_1'."
        )

    last = parts[-1]
    if last.isdigit():
        # Format: topic_tasktype_N — need at least 3 segments.
        if len(parts) < 3:
            return (
                f"Agent name {name!r}: numeric suffix requires a task_type before it. "
                f"Format: {{topic}}_{{task_type}}_{{N}}. "
                f"Valid task_types: {', '.join(SUBAGENT_TASK_TYPES)}."
            )
        task_type = parts[-2]
        topic_parts = parts[:-2]
    else:
        task_type = last
        topic_parts = parts[:-1]

    if task_type not in SUBAGENT_TASK_TYPES:
        return (
            f"Agent name {name!r} has invalid task_type {task_type!r}. "
            f"Must be one of: {', '.join(SUBAGENT_TASK_TYPES)}. "
            f"Format: {{topic}}_{{task_type}}[_{{N}}] — use dashes inside the topic. "
            f"Examples: 'gpu-market_research_1', 'draft-answer_verify', "
            f"'conflicting-claims_lverify'."
        )

    # After _normalize_agent_name(), a well-typed topic is a single
    # dash-joined token. If we still see extra ``_`` segments here it
    # means normalization bailed (unknown task_type path) — surface a
    # clear error instead of silently mangling the name.
    if len(topic_parts) != 1:
        bad_topic = "_".join(topic_parts)
        good_topic = "-".join(topic_parts)
        return (
            f"Agent name {name!r}: topic {bad_topic!r} contains underscores. "
            f"Use dashes instead: '{good_topic}_{task_type}'. "
            f"Format: {{topic}}_{{task_type}}[_{{N}}] — topic uses dashes only."
        )
    return None


def _resolve_runtime(scope_metadata: dict[str, Any]) -> Any | None:
    """Pull the sub-agent runtime config from the active ExecutionScope.

    Returns ``None`` outside a multi-agent run so the tool can no-op
    cleanly (still registers the session, just without the workflow's
    observers). The scope key is shared verbatim by every workflow that
    spawns sub-agents, so this resolves whichever one stashed its runtime.
    """
    from plugins.tools._bus_scope import SWARM_SCOPE_KEY
    return scope_metadata.get(SWARM_SCOPE_KEY)


def _runtime_workflow_pkg(runtime: Any | None) -> str:
    """Resolve the workflow package that owns ``runtime``.

    ``create_subagent`` is shared across coordinator-style workflows that
    each ship their own ``subagent_runtime`` / ``prompts`` /
    ``stream_repetition`` modules. The runtime dataclass lives in its
    workflow's ``subagent_runtime`` module, so its ``__module__`` (e.g.
    ``workflows.agent_team.subagent_runtime``) names the package to
    dispatch to. ``None`` / anything not under a ``workflows.<pkg>``
    namespace falls back to ``workflows.agent_team``.
    """
    mod = type(runtime).__module__ if runtime is not None else ""
    if mod.startswith("workflows.") and mod.count(".") >= 2:
        return mod.rsplit(".", 1)[0]
    return "workflows.agent_team"


def _resolve_task_types(runtime: Any | None) -> tuple[str, ...] | None:
    """Return the active workflow's ``SUBAGENT_TASK_TYPES``, or ``None`` when the
    workflow does not use the strict ``{topic}_{task_type}`` naming convention.

    No workflow in this repository declares it, so the strict path is an
    extension seam rather than a live branch here.

    A workflow opts into strict ``{topic}_{task_type}[_N]`` names by
    defining ``SUBAGENT_TASK_TYPES`` in its prompts module. ``agent_team``
    does not: it uses free-form ROLE-label names (``final_verifier`` /
    ``lit_search`` / ``match_researcher`` / …) that its
    ``get_subagent_system_prompt`` resolves by substring. Returning
    ``None`` switches name handling to the lenient path below, instead of
    rejecting every role-label name — which created ZERO sub-agents and
    deadlocked the planning-mode finalize gate.
    """
    pkg = _runtime_workflow_pkg(runtime)
    try:
        mod = importlib.import_module(f"{pkg}.prompts")
        tt = getattr(mod, "SUBAGENT_TASK_TYPES", None)
        return tuple(tt) if tt else None
    except Exception:
        return None


def _build_runtime_spec(
    runtime: Any,
    *,
    session_name: str,
    task_id: str,
    task_id_for_sse: str | None = None,
    run_id: str = "",
    run_type: str = "",
) -> Any:
    mod = importlib.import_module(f"{_runtime_workflow_pkg(runtime)}.subagent_runtime")
    return mod.build_swarm_session_runtime_spec(
        runtime,
        session_name=session_name,
        task_id=task_id,
        task_id_for_sse=task_id_for_sse,
        run_id=run_id,
        run_type=run_type,
    )


def _resolve_specialist_prompt(
    name: str,
    role_hint: str,
    *,
    fs_mode: bool,
    enhancements: bool = False,
    mcp_tool_names: list[str] | None = None,
    mcp_tool_specs: list[dict[str, Any]] | None = None,
    runtime: Any | None = None,
) -> str:
    """Route a sub-agent to its workflow's specialist system prompt by name."""
    mod = importlib.import_module(f"{_runtime_workflow_pkg(runtime)}.prompts")
    fn = mod.get_subagent_system_prompt
    kwargs: dict[str, Any] = dict(
        name=name,
        role=role_hint,
        include_domain_guide=fs_mode,
        mcp_tool_names=mcp_tool_names or (),
        mcp_tool_specs=mcp_tool_specs or (),
    )
    params = inspect.signature(fn).parameters
    # Optional prompt-builder knobs are capability-detected so the shared tool
    # remains compatible with workflows that do not expose them.
    if "enhancements" in params:
        kwargs["enhancements"] = enhancements
    # Two blocks of runtime facts the static templates cannot know. Both are
    # task-level constants — identical for every sub-agent of one task — so
    # both belong in the shared KV-cache prefix, ahead of the per-agent role.
    #
    # ``sub_prompt_suffix``: for agent_team this is the filesystem-convention
    # note
    # (scratch → /workspace, final deliverable → /outputs) when a sandbox is
    # active — the main agent gets it inline, and sub-agents share the same
    # mounts, so they need the same convention or a sub could drop a final into
    # its /workspace cwd and it would never be collected. In strict mode it is the
    # task's /inputs listing, since read_file cannot list a directory.
    #
    # ``notice``: the research/verifier templates hard-code web_search /
    # web_fetch and a web-centric methodology. If the active tool policy
    # disabled those tools, an explicit override keeps the model from trying
    # tools that will not be in its list.
    runtime_block = (
        str(getattr(runtime, "sub_prompt_suffix", "") or "")
        + _disabled_web_tools_notice(runtime)
    )
    accepts_suffix = "runtime_suffix" in params
    if accepts_suffix:
        kwargs["runtime_suffix"] = runtime_block
    prompt = fn(**kwargs)
    # Capable builders place the block before the role; legacy builders can
    # only have it appended, so adding a new workflow cannot crash. Same block,
    # same internal order either way.
    return prompt if accepts_suffix else prompt + runtime_block


def _runtime_tool_names(runtime: Any | None) -> set[str] | None:
    names = getattr(runtime, "sub_agent_tool_names", None)
    if names is None:
        return None
    return {str(name) for name in names}


def _runtime_tools_override(runtime: Any | None) -> list[Any] | None:
    tools = getattr(runtime, "sub_agent_tools", None)
    if tools is not None:
        return list(tools)
    names = getattr(runtime, "sub_agent_tool_names", None)
    if names is None:
        return None
    from frontier_agent.core.runtime.resources.manager import ResourceManager

    rm = registry.get_optional(ResourceManager)
    if rm is None:
        return []
    all_tools = rm.all_tools
    return [all_tools[name] for name in names if name in all_tools]


def _disabled_web_tools_notice(runtime: Any | None = None) -> str:
    """Override block emitted when the tool policy disabled web access for
    sub-agents — keeps the web-centric research prompt coherent without it.

    ``check_permission`` runs through the ResourceManager's effective context
    (which layers the global allow/deny policy), so this returns non-empty
    exactly when ``web_search`` / ``web_fetch`` were switched off for this run.

    The wording is position-independent because this task-level constant sits
    in the shared prompt prefix, before the per-agent role.
    """
    from frontier_agent.core.runtime.resources.manager import ResourceManager

    rm = registry.get_optional(ResourceManager)
    runtime_names = _runtime_tool_names(runtime)
    if rm is None and runtime_names is None:
        return ""
    if runtime_names is not None:
        # A lambda rather than ``.__contains__``: the bound method accepts
        # object, which does not match the (name: str) -> bool signature the
        # else-branch defines.
        def has_tool(name: str) -> bool:
            return name in runtime_names
    else:
        assert rm is not None

        def has_tool(name: str) -> bool:
            return rm.check_permission(_resolve_sub_role_id(runtime), name)
    disabled = [
        name for name in ("web_search", "web_fetch")
        if not has_tool(name)
    ]
    if not disabled:
        return ""
    if has_tool("bash"):
        fallback = " Use `bash` for local computation or data work."
    elif has_tool("run_python_code"):
        fallback = " Use `run_python_code` for local computation or data work."
    else:
        fallback = ""
    tools_str = " and ".join(f"`{n}`" for n in disabled)
    verb = "are" if len(disabled) > 1 else "is"
    return (
        "\n\n# Tool Availability Override (READ FIRST)\n"
        f"{tools_str} {verb} DISABLED for this run and will NOT appear in your "
        "tool list. Ignore every other instruction in this prompt that tells "
        "you to search the web or fetch pages, wherever it appears — before or "
        "after this section, your role included. Do not attempt them, and do not "
        "use code to issue web requests as a workaround." + fallback +
        " Answer from your own knowledge and reasoning; when a fact cannot be "
        "verified without the disabled tools, state it as unverified rather "
        "than fabricating a source."
    )


def _bind_sub_agent_llm(runtime: Any | None) -> Any | None:
    """Pick the sub-agent LLM and bind ``max_tokens`` for full reports."""
    if runtime is not None and runtime.sub_agent_llm is not None:
        llm = runtime.sub_agent_llm
    else:
        from frontier_agent.core.runtime.resources.manager import ResourceManager
        resource_mgr = registry.get_optional(ResourceManager)
        if resource_mgr is None:
            return None
        try:
            llm = resource_mgr.get_llm(_resolve_sub_role_id(runtime))
        except Exception:
            return None
    # Native clients carry no langchain ``Runnable.bind``; bind the
    # per-call ``max_tokens`` knob via the kernel loop's ``_BoundLLM``
    # shim (mirrors ``llm_client._bind_reduced_max_tokens``) so the
    # sub-agent loop threads a big output budget into every request for
    # full-length reports.
    #
    # BUT never request MORE output than the model actually accepts: the
    # profile's declared ``llm.max_tokens`` (threaded as
    # ``runtime.llm_max_tokens``) is the model's real
    # ``max_completion_tokens`` ceiling. Over-requesting (e.g. the legacy
    # hardcoded 65536 against a 32768-cap model) makes the server 400 the
    # sub-agent on turn 1 (``stopped_by=llm_error``, empty report). Cap the
    # desired budget by the profile ceiling when known; fall back to the
    # legacy desired value when unknown (no profile).
    desired_max_tokens = 65536
    profile_cap = getattr(runtime, "llm_max_tokens", None)
    if isinstance(profile_cap, int) and profile_cap > 0:
        eff_max_tokens = min(desired_max_tokens, profile_cap)
        if eff_max_tokens < desired_max_tokens:
            logger.info(
                "sub-agent LLM: capping max_tokens %d→%d per profile "
                "llm.max_tokens (model output ceiling)",
                desired_max_tokens, eff_max_tokens,
            )
    else:
        # No profile ceiling known: fall back to the legacy desired value.
        # This is the path that historically 400'd sub-agents when the model's
        # real output cap was below 65536 — warn so it's greppable in logs.
        eff_max_tokens = desired_max_tokens
        logger.warning(
            "sub-agent LLM: no profile llm.max_tokens known; requesting "
            "max_tokens=%d unbounded — if the model rejects it the sub-agent "
            "dies on turn 1 with stopped_by=llm_error and an empty report",
            eff_max_tokens,
        )
    try:
        from dataclasses import replace

        from frontier_agent.core.runtime.loop.llm_client import _ensure_bound
        bound = replace(_ensure_bound(llm), max_tokens=eff_max_tokens)
    except Exception:
        bound = llm
    stream_cfg = getattr(runtime, "stream_repetition_config", None)
    if stream_cfg is None:
        return bound
    _sr = importlib.import_module(
        f"{_runtime_workflow_pkg(runtime)}.stream_repetition"
    )
    wrap_llm_for_stream_repetition = _sr.wrap_llm_for_stream_repetition

    wrapped, _observer = wrap_llm_for_stream_repetition(
        bound,
        config=stream_cfg,
        role_id=SUB_ROLE_ID,
        label="subagent",
    )
    return wrapped


@tool
async def create_subagent(agents: list[Any] | str = "") -> str:
    """Create one or more persistent sub-agents.

    Each sub-agent is a long-lived session that can accept multiple
    ``assign_task`` calls, accumulating history across tasks. The
    session's history is trimmed between tasks (system + each completed
    task's prompt + final report) so reused agents stay context-efficient.

    Args:
        agents: list of dicts with:
            - ``name`` (required): Unique sub-agent name following
              ``{topic}_{task_type}[_{N}]``, e.g. ``lit-review_research``.
            - ``system_prompt`` (required): Custom system prompt for this
              instance — describes its specialty/focus.

    Returns:
        Confirmation text listing the created sub-agents.
    """
    # ``agents`` defaults to ``""`` so models emitting empty ``{}`` args
    # land on the actionable error path instead of crashing pydantic
    # with ``Field required``. JSON-string serialisation is handled by
    # ``coerce_json_list``.
    agents = coerce_json_list(agents) or []
    if not agents:
        return "Error: create_subagent requires at least one agent spec."

    if len(agents) > MAX_SUBAGENTS_PER_DISPATCH:
        return (
            f"Error: create_subagent supports at most "
            f"{MAX_SUBAGENTS_PER_DISPATCH} agents per call; you passed "
            f"{len(agents)}. Split the list across multiple calls so no "
            f"agent is silently dropped."
        )

    scope = get_current_execution_scope()
    if scope is None:
        return (
            "Error: create_subagent can only be called inside an "
            "active ReAct execution."
        )

    runtime = _resolve_runtime(scope.metadata)
    sub_role_id = _resolve_sub_role_id(runtime)
    fs_mode = bool(getattr(runtime, "fs_mode", False))
    # Online prompt enhancements gate — same ``sdk_protocol_emitter`` signal
    # the main agent reads (set by serve.py / run.py, absent on benchmark
    # eval). On → online-tuned sub-agent templates; off → ``*_lean`` baseline.
    online_prompt = bool((scope.metadata or {}).get("sdk_protocol_emitter"))
    bus = registry.get(AgentBus)
    bus_task_id = resolve_bus_task_id(scope)
    sse_task_id = resolve_root_task_id(scope)
    # Heavy-mode tags carried in scope_metadata by main_agent_node
    # (S0 plumbing). Empty in a normal run — passed through to
    # SSEObserver so per-run sub-agent events are tagged with the
    # owning run_id.
    run_id = str((scope.metadata or {}).get("run_id") or "")
    run_type = str((scope.metadata or {}).get("run_type") or "")
    bound_llm = _bind_sub_agent_llm(runtime)

    created: list[str] = []
    errors: list[str] = []
    renamed: list[tuple[str, str]] = []

    # Workflow-aware name validation: strict mode enforces {topic}_{task_type};
    # agent_team (no SUBAGENT_TASK_TYPES) uses lenient role-label names.
    task_types = _resolve_task_types(runtime)

    for spec in agents:
        if not isinstance(spec, dict):
            errors.append(f"Skipping non-dict agent spec: {spec!r}")
            continue
        raw_name = str(spec.get("name", "")).strip()
        prompt = str(spec.get("system_prompt", "")).strip()
        if not raw_name:
            errors.append("Skipping agent with no name")
            continue
        name = _normalize_agent_name(raw_name, task_types)
        if name != raw_name:
            renamed.append((raw_name, name))
        name_error = _validate_agent_name(name, task_types)
        if name_error:
            errors.append(name_error)
            continue
        if not prompt:
            errors.append(f"Skipping agent {name!r} with no system_prompt")
            continue

        try:
            effective_prompt = _resolve_specialist_prompt(
                name,
                prompt,
                fs_mode=fs_mode,
                enhancements=online_prompt,
                mcp_tool_names=getattr(runtime, "mcp_tool_names", None),
                mcp_tool_specs=getattr(runtime, "mcp_tool_specs", None),
                runtime=runtime,
            )
            runtime_spec = (
                _build_runtime_spec(
                    runtime,
                    session_name=name,
                    task_id=bus_task_id,
                    task_id_for_sse=sse_task_id,
                    run_id=run_id,
                    run_type=run_type,
                )
                if runtime is not None
                else None
            )

            await bus.create_session(
                task_id=bus_task_id,
                name=name,
                role_id=sub_role_id,
                system_prompt=effective_prompt,
                tools_override=_runtime_tools_override(runtime),
                trimmer=TaskBoundaryTrimmer(),
                max_turns=(
                    int(runtime.sub_agent_max_turns)
                    if runtime is not None
                    and getattr(runtime, "sub_agent_max_turns", None)
                    else SUB_MAX_TURNS_DEFAULT
                ),
                llm_override=bound_llm,
                tool_result_max_chars=getattr(runtime, "tool_result_max_chars", 15_000),
                runtime_spec=runtime_spec,
            )
            created.append(name)
        except Exception as exc:
            logger.warning(
                "create_subagent: failed to create %s: %s", name, exc,
            )
            errors.append(f"Failed to create {name!r}: {exc}")

    if not created and errors:
        return "Error: " + "; ".join(errors)

    lines = [f"Created {len(created)} sub-agent(s):"]
    lines.extend(f"  - {n}" for n in created)
    if renamed:
        lines.append("")
        lines.append(
            "Note: normalized underscores in topic → dashes "
            "(canonical form is {topic}_{task_type}[_{N}] with dash-only topic):"
        )
        lines.extend(f"  - {raw} → {fixed}" for raw, fixed in renamed)
    if errors:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {e}" for e in errors)
    lines.append("")
    lines.append(
        "Call assign_task(tasks=[{agent:NAME, prompt:...}]) to give them work."
    )
    return "\n".join(lines)


__all__ = ["create_subagent"]
