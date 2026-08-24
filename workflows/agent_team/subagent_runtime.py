"""Sub-agent runtime spec for agent-team sessions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Reuses ``INCOMPLETE_STOP_REASONS`` because "needs a forced final
# answer" and "report is partial" are the same concept; keeping them
# tied at the import boundary prevents drift between this module and
# the fan-in formatter.
from frontier_agent.components.agent_bus.fan_in import (
    INCOMPLETE_STOP_REASONS as _FORCED_FINAL_STOP_REASONS,
)
from frontier_agent.components.agent_bus.models import (
    SubAgentResult,
    SubAgentRuntimeSpec,
    SubTask,
)
from frontier_agent.components.finalization import (
    COMMON_RECOVERY_NUDGE_PREFIXES,
    build_recovery_context,
    chat_with_fallback_budget,
    minimal_best_effort_answer,
)
from frontier_agent.components.observers.context_size_guard import ContextSizeGuard
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
    ToolResult,
)
from frontier_agent.core.messages import (
    Message,
    assistant_msg_with_reasoning,
    is_assistant_msg,
    is_tool_msg,
    text_of,
    user_msg,
)
from frontier_agent.core.runtime import registry
from frontier_agent.core.runtime.loop.compact import (
    URL_RE,
    KeepLastNToolResultsCompactor,
    MessageCompactor,
)
from frontier_agent.core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
    compaction_trigger_tokens,
)
from frontier_agent.core.runtime.loop.tool_exec import ToolResultPostProcessor
from frontier_agent.state.event_store.sqlite import EventStore
from plugins.tools.bash import BASH_STDERR_SEPARATOR
from workflows.agent_team.identity import MAIN_AGENT_ID, SUB_ROLE_ID, llm_session_id
from workflows.agent_team.stream_repetition import StreamRepetitionConfig

logger = logging.getLogger(__name__)

SWARM_SCOPE_KEY = "swarm_subagent_runtime"
"""ExecutionScope.metadata key carrying :class:`SwarmSubagentRuntime`."""


def render_sandbox_fs_note(
    *,
    sandbox_mode: str,
    inputs_available: bool,
    audience: str,
) -> str:
    """Render truthful filesystem context for the active sandbox topology."""
    shared = sandbox_mode in ("container", "native")
    location = "this native workspace" if sandbox_mode == "native" else "this container"
    workspace_path = (
        os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "the workspace")
        if sandbox_mode == "native" else "/workspace"
    )
    inputs_path = (
        os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "the inputs directory")
        if sandbox_mode == "native" else "/inputs"
    )
    outputs_path = (
        os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "the outputs directory")
        if sandbox_mode == "native" else "/outputs"
    )
    project_path = os.environ.get("FRONTIER_AGENT_PROJECT_DIR", "").strip()
    if audience == "sub":
        workspace = (
            f"{workspace_path} is shared by the coordinator and all sub-agents in "
            f"{location}; use distinctive subdirectories and never assume "
            "unlabelled files are yours."
            if shared
            else f"{workspace_path} is private to this sub-agent."
        )
    else:
        workspace = (
            f"{workspace_path} is shared by the coordinator and all sub-agents in "
            f"{location}."
            if shared
            else f"{workspace_path} contains the coordinator worktree and the "
            "sub-agents' private worktree directories."
        )
    inputs = (
        f"Task input files are available under {inputs_path}."
        if inputs_available
        else f"No task input files were provided; {inputs_path} is empty. Do not probe it."
    )
    # Positive read-context guidance: /workspace is not just where YOU write —
    # in container mode it is the shared read source for teammates' candidates.
    # In bwrap mode each sub-agent's /workspace is private, so cross-agent reuse
    # can only come through the coordinator's <attach> report text.
    collab_read = (
        " To build on a teammate's work, read their candidate directly from "
        f"{workspace_path} at the path named in that agent's report — {workspace_path} is "
        "the team's shared read source for intermediate products, alongside "
        f"{inputs_path} for external input files."
        if shared
        else " To reuse another agent's work, rely on the <attach> report text "
        "the coordinator passes in; each sub-agent's /workspace is private and "
        "not readable by other agents here."
    )
    optional_dependencies = (
        "Scientific and plotting packages are optional. Office writer packages "
        "used by create_file are already installed. Do not pre-install optional "
        "packages; install only a package required by the task with "
        "`python -m pip install <package>`. Native mode stores it in the "
        "workspace-local dependency overlay. "
        if sandbox_mode == "native" else ""
    )
    project = ""
    if project_path and os.path.realpath(project_path) != os.path.realpath(workspace_path):
        project = (
            f"The user's project/coding directory is {project_path}; read or "
            "edit repository files there, while keeping clones, downloads, "
            f"drafts, probes, and other intermediate work under {workspace_path}. "
        )
    manifest_namespace = ""
    if audience == "main" and outputs_path != "/outputs":
        manifest_namespace = (
            " When calling assign_task, however, output_paths is a virtual "
            "manifest namespace: always declare /outputs/<relative-file>, "
            f"never {outputs_path}/<relative-file>. The publisher runtime maps "
            "that declaration back to the physical root automatically."
        )
    return (
        "\n\nFILESYSTEM CONTEXT (runtime-resolved): "
        f"{workspace} {inputs}{collab_read} "
        f"{optional_dependencies}{project}"
        "All research notes, calculations, candidate artifacts, temporary "
        f"files, and verification material belong under {workspace_path}. {outputs_path} "
        "is the collected final-deliverable root and is read-only by default. "
        "Only one assignment, the one carrying an exact output_paths "
        "manifest, may write there; it must publish exactly that manifest and "
        "must not add README, confirmation, validation, duplicate, or alternate-"
        "format files unless declared. The one exception is the top-level "
        f"{outputs_path}/scratch/ directory: writable by every assignment, persisted "
        "across rounds for intermediate products worth reusing later, never "
        "shown to the user, with a 512MB quota (over-quota writes fail until "
        f"you delete files there). Only the literal {outputs_path}/scratch/ prefix "
        f"qualifies.{manifest_namespace}"
    )


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")
_AGED_TOOL_SENTINEL = "[aged-tool:"
_REPORT_BLOCK_RE = re.compile(r"<report\b[^>]*>[\s\S]*?</report>")
_STITCH_MAX_CHARS = 60_000
_SUBAGENT_NO_TOOL_NUDGE = (
    "Every turn must end with a tool call. When you have gathered enough "
    "evidence, call `submit_report` with your Scope/Finding/Evidence report. "
    "Otherwise call a data-gathering tool (web_search, web_fetch, read_file, "
    "bash). Do not reply in plain text."
)
_SUBAGENT_FINALIZATION_MESSAGE = (
    "Your assignment is entering its finalization reserve. Stop new "
    "exploration. Finish and validate the best current artifact in /workspace; "
    "if this assignment has publish permission, publish exactly its declared "
    "/outputs manifest. Then call `submit_report` with concrete findings and "
    "file paths. Submit a useful partial report even if some work remains."
)
_RECOVERY_NUDGE_PREFIXES = (
    # This workflow's own sub-agent control messages, on top of the framework set.
    _SUBAGENT_NO_TOOL_NUDGE,
    _SUBAGENT_FINALIZATION_MESSAGE,
    "[stop signal] The coordinator has asked you to STOP immediately.",
    *COMMON_RECOVERY_NUDGE_PREFIXES,
)
_RECOVERY_EMPTY_FALLBACK = (
    "No reliable intermediate text was recoverable. Give the most useful "
    "honest answer possible from the original task."
)
_RECOVERY_LABELS = {
    "system": "System guidance",
    "assistant": "Visible coordinator draft",
    "tool": "Collected observation or sub-agent report",
    "user": "User instruction",
}

# Some text-mode models under long-context fatigue may emit tool calls as raw
# text in `content` (e.g.
# ``<tool_call>{"name": ...}</tool_call>``) instead of going through the
# structured ``tool_calls`` field. The runtime's text-mode parser still
# dispatches these correctly, but the raw markup remains in the content
# string. When the loop terminates abnormally and `force_final_answer`
# nudges the model for a plain-text answer, the model often keeps
# emitting tool-call markup because it is stuck in that mode. Strip
# anything resembling tool-call XML before treating the response as the
# final answer.
_LEAKED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*tool_call[^>]*>[\s\S]*?<\s*/\s*tool_call\s*>", re.IGNORECASE,
)
_LEAKED_TOOL_RESPONSE_BLOCK_RE = re.compile(
    r"<\s*tool_response[^>]*>[\s\S]*?<\s*/\s*tool_response\s*>", re.IGNORECASE,
)
# Function-call style: ``<function=name><parameter=...>...</function>``.
_LEAKED_FUNCTION_BLOCK_RE = re.compile(
    r"<\s*function\s*=[\s\S]*?<\s*/\s*function\s*>", re.IGNORECASE,
)
# Stray opening tag with no matching close.
_LEAKED_TAG_FRAGMENT_RE = re.compile(
    r"<\s*(tool_call|tool_response|function)\b[^>]*>?", re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def _strip_leaked_tool_calls(text: str) -> str:
    """Remove text-mode tool-call markup that leaked into content.

    Returns the cleaned string. If everything was tool-call XML and the
    remainder is empty/whitespace, returns ``""`` so the caller can
    decide on a sentinel.
    """
    if not text:
        return text
    out = _LEAKED_TOOL_CALL_BLOCK_RE.sub("", text)
    out = _LEAKED_TOOL_RESPONSE_BLOCK_RE.sub("", out)
    out = _LEAKED_FUNCTION_BLOCK_RE.sub("", out)
    out = _LEAKED_TAG_FRAGMENT_RE.sub("", out)
    return out.strip()


def _finalization_prompt(
    task_description: str,
    *,
    structured_report: bool,
) -> str:
    if structured_report:
        return (
            "Write the final user-facing report now using only the clean "
            "recovery context below. Directly answer the original task, state "
            "what was completed, name any deliverables that were produced, "
            "and clearly qualify remaining gaps. Use structured Markdown when "
            "helpful. Do not call tools and do not return an error sentinel."
        )
    return (
        "Provide the best final answer now using only the clean recovery "
        "context below. Preserve useful partial conclusions and mention any "
        "deliverables already produced. Do not call tools and do not return an "
        "error sentinel."
    )


def _build_recovery_messages(
    messages: list[Message],
    *,
    task_description: str,
    structured_report: bool,
) -> list[Message]:
    """Flatten history so malformed/orphan tool protocol cannot poison rescue."""
    recovery_context = build_recovery_context(
        messages,
        strip_thinking=_strip_thinking,
        strip_leaked_tool_calls=_strip_leaked_tool_calls,
        nudge_prefixes=_RECOVERY_NUDGE_PREFIXES,
        empty_fallback=_RECOVERY_EMPTY_FALLBACK,
        labels=_RECOVERY_LABELS,
    )
    prompt = (
        f"{_finalization_prompt(task_description, structured_report=structured_report)}"
        "\n\nThe original tool-call fields were intentionally removed because "
        "they may be malformed. Treat all observations as partial progress.\n\n"
        f"Original task:\n{task_description or '(not repeated)'}\n\n"
        f"Clean recovery context:\n{recovery_context}"
    )
    return [user_msg(prompt)]


def _stitch_collected_reports(messages: list[Message]) -> str:
    """Deterministically preserve collected sub-agent reports without an LLM."""
    blocks: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or not is_tool_msg(message):
            continue
        content = text_of(message.get("content"))
        for match in _REPORT_BLOCK_RE.finditer(content):
            block = match.group(0)
            if block in seen:
                continue
            seen.add(block)
            blocks.append(block)
    if not blocks:
        return ""
    stitched = "\n\n".join(blocks)
    if len(stitched) > _STITCH_MAX_CHARS:
        stitched = stitched[:_STITCH_MAX_CHARS] + "\n[... truncated ...]"
    return (
        "## Best-effort collected results\n\n"
        "The synthesis step ended early, so the completed sub-agent reports "
        "below are delivered directly instead of returning no answer.\n\n"
        + stitched
    )


def _minimal_best_effort_answer(
    task_description: str,
    stopped_by: str,
    *,
    language: str = "",
) -> str:
    """Always provide a user-facing answer even when the rescue LLM is down."""
    return minimal_best_effort_answer(
        task_description, stopped_by, language=language,
    )


async def force_final_answer(
    result: AgentLoopResult,
    llm: Any,
    timeout: float,
    *,
    task_description: str = "",
    structured_report: bool = False,
    language: str = "",
) -> AgentLoopResult:
    """Run one tool-free LLM call to extract a plain-text final answer.

    No-op when the loop already produced one (``finalize_answer`` /
    ``submit_report`` latched ``metadata["final_answer"]``) or when the
    exit reason is outside :data:`_FORCED_FINAL_STOP_REASONS`. Mutates
    ``result`` in place. The rescue request is rebuilt as one clean user
    message, so a malformed tool call in the original history cannot make the
    finalization request fail the same way. Best-effort — failures fall back to
    collected reports and finally an explicit user-facing partial result.
    """
    if (
        result.metadata.get("final_answer")
        or result.stopped_by not in _FORCED_FINAL_STOP_REASONS
    ):
        return result

    recovery_messages = _build_recovery_messages(
        list(result.messages),
        task_description=task_description,
        structured_report=structured_report,
    )
    ask_timeout = max(float(timeout), 1.0)

    async def _ask() -> str:
        resp = await chat_with_fallback_budget(
            llm,
            recovery_messages,
            per_leg_timeout_s=ask_timeout,
        )
        content = getattr(resp, "content", "") or ""
        # This rescue assistant turn is appended directly to history. It
        # bypasses the kernel normalizer, so route reasoning through the
        # shared format-aware builder (tag default) — a bare
        # ``reasoning_content`` field would land off the served
        # checkpoint's expected wire shape.
        cleaned = _strip_leaked_tool_calls(_strip_thinking(text_of(content)))
        if cleaned:
            result.messages.extend(recovery_messages)
            result.messages.append(assistant_msg_with_reasoning(
                cleaned,
                getattr(resp, "reasoning_content", "") or "",
                thinking_format="tag",
            ))
        return cleaned

    rescue_mode = "deterministic_fallback"
    try:
        text = await _ask()
        if text:
            rescue_mode = "clean_context_llm"
    except Exception as exc:
        logger.warning("force_final_answer failed: %s", exc)
        text = ""

    # If the clean recovery call still returns only leaked markup, fall back to
    # the original ``final_content`` (also stripped), completed sub-agent
    # reports, or an explicit best-effort status rather than scoring raw
    # tool-call XML. This is the *terminal* answer, so the coordinator's own
    # visible draft outranks the raw reports — it is already a synthesis, and
    # nothing downstream will turn the reports into one.
    # ``prepare_report_handoff`` deliberately reverses these two: there the
    # reporter is still to come, and raw reports are the better input for it.
    if not text:
        text = _strip_leaked_tool_calls(_strip_thinking(result.final_content))
        if text:
            rescue_mode = "existing_partial"
    if not text:
        text = _stitch_collected_reports(result.messages)
        if text:
            rescue_mode = "collected_reports"
    if not text:
        text = _minimal_best_effort_answer(
            task_description, result.stopped_by, language=language,
        )
    result.metadata["final_answer"] = text
    result.metadata["final_answer_rescued"] = True
    result.metadata["final_answer_rescue_mode"] = rescue_mode
    result.metadata["final_answer_source"] = rescue_mode
    result.final_content = text
    logger.info(
        "force_final_answer injected after %s (%d chars)",
        result.stopped_by, len(text),
    )
    return result


def prepare_report_handoff(
    result: AgentLoopResult,
    *,
    task_description: str = "",
    language: str = "",
) -> AgentLoopResult:
    """Prepare a non-empty baseline before an optional reporter runs.

    Completed reports outrank a truncated coordinator draft because they carry
    more evidence and remain usable if the reporter fails open.
    """
    if result.stopped_by not in _FORCED_FINAL_STOP_REASONS:
        return result

    source = str(result.metadata.get("final_answer_source") or "")
    text = _strip_leaked_tool_calls(_strip_thinking(str(
        result.metadata.get("final_answer") or "",
    )))
    if text and not source:
        source = "agent"
    if not text:
        text = _stitch_collected_reports(result.messages)
        if text:
            source = "collected_reports"
    if not text:
        text = _strip_leaked_tool_calls(_strip_thinking(result.final_content))
        if text:
            source = "existing_partial"
    if not text:
        text = _minimal_best_effort_answer(
            task_description, result.stopped_by, language=language,
        )
        source = "deterministic_fallback"

    result.metadata["final_answer"] = text
    result.metadata["final_answer_source"] = source
    result.metadata["report_handoff"] = True
    result.metadata["report_handoff_reason"] = result.stopped_by
    result.final_content = text
    logger.info(
        "prepared reporter handoff after %s (%d chars)",
        result.stopped_by,
        len(text),
    )
    return result


@dataclass(frozen=True)
class SwarmSubagentRuntime:
    """Per-task config the main agent threads to its sub-agents.

    The main-agent node populates one of these per loop run and passes
    it through ``run_agent_loop(scope_metadata={SWARM_SCOPE_KEY: …})``.
    ``create_subagent`` reads it back to build a
    :class:`SubAgentRuntimeSpec` aligned with the main agent's profile.
    """

    original_question: str = ""
    trajectory_dir: Path | None = None
    sub_agent_llm: Any = None
    sub_agent_llm_timeout: float | None = None
    sub_agent_tool_timeout: float | None = None
    reasoning_only_timeout_s: float | None = None
    reasoning_only_max_tokens: int | None = None
    logical_call_timeout_s: float | None = None
    sub_agent_max_turns: int | None = None
    # Reserve several tool-enabled turns to finish assigned artifacts and
    # submit a report before LastTurnForcer strips tools on the landing turn.
    finalization_reserve_turns: int = 6
    sub_agent_thinking_in_history: bool = False
    sub_agent_thinking_history_max_tokens: int | None = None
    # Compaction (loaded from profile yaml `agent:` section). Defaults
    # suit long-context tool use; ``keep_tool_result=-1`` disables.
    sub_keep_tool_result: int = 10
    sub_compact_after_turns: int = 100
    sub_context_token_limit: int = 180_000
    # Tiered context compaction (mirrors stateful_react); "off" (default) keeps
    # the legacy keep-last-N path. "tiered" → compact at max_len*0.8 (REAL input
    # tokens): Tier1 keep-last-N tool results, Tier2 LLM summary if not enough.
    context_compaction: str = "off"
    compaction_spill: bool = False
    max_len: int = 0
    max_input_tokens: int | None = None
    tier1_keep_tool_result: int = 5
    keep_recent_turns: int = 5
    sub_agent_model_profile: Any = None
    fs_mode: bool = False
    # ``None`` (default) disables BudgetObserver — max_turns is the only bound.
    max_tokens: int | None = None
    # StuckTargetGuard (0 = off): confirmed failures for one host within the
    # last ``stuck_target_window`` network turns. Success resets the host; the
    # escalation threshold quarantines it. With no token budget and
    # ``sub_max_turns`` at 100, a sub-agent can otherwise spend its allowance on
    # one unreachable URL. Tighter than the stateful_react defaults (6/10/20)
    # because a sub-agent works one narrow sub-question.
    stuck_target_hint_after: int = 5
    stuck_target_escalate_after: int = 8
    stuck_target_window: int = 15
    # Profile's declared per-call OUTPUT cap (``llm.max_tokens``) — the model's
    # real ``max_completion_tokens`` ceiling. Sub-agents want a big output
    # budget for full reports, but must never REQUEST more than the model
    # accepts or the server 400s the very first turn. ``_bind_sub_agent_llm``
    # caps its desired budget by this. ``None`` → unknown (no profile) → keep
    # the legacy hardcoded desired value.
    llm_max_tokens: int | None = None
    tool_result_max_chars: int = 15_000
    # When True (default, debug-friendly): each task on a sticky session
    # writes a separate ``<session>.tNN.{json,jsonl}`` so reuse is directly
    # observable from the filesystem. When False:
    # all tasks on a session share a single ``<session>.{json,jsonl}``
    # that is atomic-replaced after every event, keeping a stable on-disk shape
    # across reused sessions.
    trajectory_per_task: bool = True
    # When set, sub-agents also write
    # ``<workflow_id>__sub_<session>.json`` next to the existing
    # TrajectoryFileObserver output. The two writers coexist — they
    # target different files.
    worker_trace_dir: Path | None = None
    workflow_id: str = ""
    # A second, streaming trace: when both are set, sub-agents also attach
    # the caller's protocol stream observer so their LLM calls / tool calls /
    # per-call ``usage`` events flow to stdout through the same emitter
    # (single line-buffered stdout, atomic writes) and feed the same usage
    # aggregator, so ``final.usage`` covers the main agent plus every
    # sub-agent instead of the main agent alone. Both objects come from the
    # optional external protocol layer — see ``workflows/_shared/sdk_shim``.
    protocol_emitter: Any = None
    protocol_usage_aggregator: Any = None
    mcp_tool_names: list[str] | None = None
    mcp_tool_specs: list[dict[str, Any]] | None = None
    stream_repetition_config: StreamRepetitionConfig | None = None
    # Per-run resolved sub-agent tools. ``None`` preserves the role default
    # for callers that do not set it; agent_team sets this from profile config
    # without mutating the process-global AgentRegistry.
    sub_agent_tool_names: list[str] | None = None
    # Concrete per-run tool objects. This lets profiles replace an implementation
    # while retaining the public tool name and without mutating the registry.
    sub_agent_tools: list[Any] | None = None
    # Fail-closed role id for this workflow's sub-agents. Distinct from swarm's
    # ``swarm_sub`` so the two workflows' AgentDefinitions don't overwrite each
    # other when both are dir-scanned into one registry (create_subagent reads
    # this via ``_resolve_sub_role_id``).
    sub_role_id: str = SUB_ROLE_ID
    # Per-task worktree (working dir, NOT the trajectory/log dir). When set
    # and bwrap is available, each sub-agent runs its tools inside a bwrap
    # sandbox whose ``/workspace`` is ``worktree_root/<session_name>`` (private,
    # siblings invisible). ``sub_binds`` are the extra mounts shared by all
    # sub-agents — ``(/inputs ro, /outputs rw)`` — resolved by the main node
    # (subs write their deliverable to /outputs; see _resolve_sandbox_binds).
    worktree_root: Path | None = None
    sub_binds: tuple[tuple[str, str, bool], ...] = ()
    # Sandbox isolation mode (set by the main node). ``container`` → every
    # sub-agent SHARES the one mounted /workspace (``shared_workspace_dir``) and
    # /outputs via a CurrentSandbox (the container is the isolation boundary,
    # no per-sub subdir, no bwrap). ``bwrap`` / ``auto`` → the legacy per-sub
    # bwrap sandbox path below.
    sandbox_mode: str = "auto"
    shared_workspace_dir: str = ""
    # Appended to every sub-agent's system prompt (e.g. the sandbox filesystem
    # convention). create_subagent reads it off the runtime; empty = no-op.
    sub_prompt_suffix: str = ""
    # Output publication is single-owner and manifest-bound. The dataclass is
    # frozen, so ``assign_task`` mutates this per-run state bag rather than
    # replacing fields; follow-ups must reuse the same agent and manifest.
    publication_state: dict[str, Any] = field(default_factory=dict, compare=False)
    publication_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        compare=False,
        repr=False,
    )


class NoOpCompactor(MessageCompactor):
    """Disable mid-loop compaction — max_turns is the only bound."""

    def compact(
        self, messages: list[Message], keep_recent: int,
    ) -> list[Message]:
        return messages


class ReasoningStripCompactor(MessageCompactor):
    """Age old reasoning **and** tool-message bodies. Conservative — never drops messages.

    Why: ``NoOpCompactor``'s "no compaction needed" assumption breaks for
    thinking models *and* tool-heavy ReAct loops:

      1. Some reasoning models return large reasoning blocks that the next
         request re-sends
         (``thinking_in_history=true`` + provider-specific
         ``reasoning_content`` round-trip).
      2. Tool messages can also be large enough to exhaust the context window.

    Stripping ``<think>`` alone may not help when tool messages dominate.

    What: keeps the message *structure* intact (every ``ToolMessage``
    keeps its ``tool_call_id`` so the matching AI ``tool_calls`` doesn't
    orphan — Azure rejects orphans with HTTP 400) and replaces the bulky
    parts of messages older than ``keep_recent``:

      * AI ``<think>…</think>`` blocks in ``content`` (tag-mode)
      * AI ``reasoning_content`` field (DeepSeek/GPT-5)
      * Tool ``content`` → one-line stub
        ``[aged-tool: <name> url=<first_url>] <preview>…`` so the LLM
        still knows what was fetched and can re-fetch the URL if needed.

    Recent ``keep_recent`` messages are untouched. Aging is idempotent:
    a tool stub is detected by its sentinel and left alone, so triggering
    every turn after ``compact_after_turns`` is cheap.
    """

    _MAX_TOOL_PREVIEW = 120  # chars of body kept in the aged stub

    def compact(
        self, messages: list[Message], keep_recent: int,
    ) -> list[Message]:
        if len(messages) <= keep_recent:
            return messages
        cutoff = len(messages) - keep_recent
        for i in range(cutoff):
            msg = messages[i]
            if is_assistant_msg(msg):
                content = msg.get("content")
                if isinstance(content, str) and "<think>" in content:
                    msg["content"] = _THINK_BLOCK_RE.sub("", content)
                if "reasoning_content" in msg:
                    msg.pop("reasoning_content", None)
                continue
            if is_tool_msg(msg):
                raw = msg.get("content")
                content = raw if isinstance(raw, str) else str(raw)
                if content.startswith(_AGED_TOOL_SENTINEL):
                    continue
                tool_name = msg.get("name") or "tool"
                m = URL_RE.search(content)
                first_url = m.group(0) if m else ""
                preview = content[: self._MAX_TOOL_PREVIEW].replace("\n", " ").strip()
                head = f"{_AGED_TOOL_SENTINEL} {tool_name}"
                if first_url:
                    head += f" url={first_url}"
                msg["content"] = (
                    f"{head}] {preview}…"
                    if preview
                    else f"{head}] (body aged out — re-fetch if needed)"
                )
        return messages


class SwarmToolResultPostProcessor(ToolResultPostProcessor):
    """Per-tool dispatch for fresh ``ToolMessage`` content.

    Fires inside the agent loop right after a tool returns and before the
    result enters the conversation history (``agent_loop.py`` step 8d).
    Replaces ``DefaultToolResultPostProcessor``'s blunt char cap with
    per-tool dispatch: content-retrieval tools (web_fetch, web_search,
    scholar_search) pass through unmodified — academic-paper results
    sections and long Wikipedia articles need their full body. Compute
    tools (bash, run_python_code) keep tight head/tail caps because
    their output is unbounded by design (``find /``, runaway ``print``).

    Web results pass through because relevant evidence may occur after an
    abstract or introduction. Cap upstream at ``TOOL_RESULT_MAX_CHARS``, let
    ``ReasoningStripCompactor`` age old bodies to URL stubs at 200K,
    and trust the Summary LLM (``info_to_extract`` on web_fetch) for
    in-call compression when the agent asks for it.

    Errors (``is_error=True``) bypass all caps — the model needs the
    full error text to recover. ``delegate_subtask`` and
    ``submit_report`` pass through unmodified: sub-agent reports are
    structured and expensive to regenerate, and the terminal-tool ack
    is just a stub.

    Adding a new compute-style tool: append one ``_BUDGETS`` entry and
    a dispatch in ``process`` if it needs custom reshape (like
    ``bash``'s stderr-aware split); otherwise everything falls through
    to ``_head_cap`` with the looked-up budget.

    The hard 150K ``TOOL_RESULT_MAX_CHARS`` cap inside ``execute_tools``
    runs first, so this processor sees at most 150K of input. Output is
    the string that lands in the ``ToolMessage`` — observers / evidence
    still see the underlying ``ToolResult`` unchanged.
    """

    # Pass-through for content-retrieval tools: information density is
    # high (paper results sections sit 30K+ chars in), upstream Summary
    # LLM (web_fetch's info_to_extract path) handles in-call compression
    # when the agent asks for it, and ReasoningStripCompactor ages old
    # bodies to URL stubs once total context crosses 200K. The hard
    # ceiling is now TOOL_RESULT_MAX_CHARS=150K in execute_tools.
    # bash / run_python_code stay capped — their output is unbounded by
    # design (find /, big stdout) and head/tail-with-stderr is the right
    # diet, not lifting it.
    _PASS_THROUGH = frozenset({
        "delegate_subtask",
        "submit_report",
        # collect_reports concatenates the (already-structured, expensive-to-
        # regenerate) sub-agent reports the main agent synthesizes from — never
        # head-cap it to the 6K default, or the main agent only sees the first
        # ~1-2 reports when many sub-agents fan in.
        "collect_reports",
        "web_fetch",
        "web_search",
        "scholar_search",
        # Paginates itself: the trailing "PARTIAL READ … read_file again with
        # offset=N" line IS how the caller continues, so head-capping it replaces
        # a usable offset with generic "re-fetch" advice. Measured on a live
        # agent-team run before this was added: of 5 read_file results over 6K,
        # the tail instruction was lost in every one. React whitelisted this from
        # the start; the omission here was never deliberate.
        "read_file",
        # Same contract, same reason as read_file above. Head-capping it would cut
        # the continuation pointer off the content the call exists to fetch back,
        # so recovery would silently return less than it says it did.
        "recover_result",
    })

    _BUDGETS: ClassVar[dict[str, int]] = {
        "bash": 4_000,
        "run_python_code": 5_000,
        # Gate ① overflows these to disk at meta.max_result_chars=8K and puts the
        # "saved to <path>, use read_file" pointer at the TAIL. The 6K default cut
        # below that and took the pointer with it, leaving a truncated result with
        # no way back to the rest. 10K leaves the 8K body and its footer room to
        # both survive — the value react has carried for this reason all along.
        "grep_search": 10_000,
        "glob_search": 10_000,
        # Belt-and-suspenders: collect_reports is whitelisted above (pass-
        # through wins), but if it ever leaves the whitelist this large budget
        # keeps it effectively uncapped (bounded only by the 150K upstream
        # TOOL_RESULT_MAX_CHARS ceiling) rather than falling back to 6K.
        "collect_reports": 150_000,
    }
    _BUDGET_DEFAULT = 6_000

    _BASH_STDERR_KEEP = 3_000
    _ELLIPSIS = (
        "\n\n[... body truncated for context budget — re-fetch URL "
        "or call again with a focused query if you need the rest ...]"
    )

    def process(self, tool_result: ToolResult) -> str:
        content = (
            tool_result.result
            if isinstance(tool_result.result, str)
            else str(tool_result.result)
        )
        if tool_result.is_error:
            return content

        name = tool_result.name or ""
        if name in self._PASS_THROUGH:
            return content

        budget = self._BUDGETS.get(name, self._BUDGET_DEFAULT)
        if name == "bash":
            return self._compact_bash(content, budget)
        return self._head_cap(content, budget)

    def _head_cap(self, content: str, budget: int) -> str:
        # Skip the cut when it wouldn't actually shrink — a budget+30
        # input becomes budget+len(ELLIPSIS) bytes otherwise.
        if len(content) <= budget + len(self._ELLIPSIS):
            return content
        return content[:budget] + self._ELLIPSIS

    def _compact_bash(self, content: str, budget: int) -> str:
        """Keep stderr in full + stdout head/tail with elision marker.

        Depends on the ``BASH_STDERR_SEPARATOR`` literal emitted by
        ``plugins/tools/bash.py`` and ``run_python_code.py``; if those
        producers stop joining stdout/stderr with this exact string,
        the split silently degrades to "treat the whole blob as stdout".
        """
        if len(content) <= budget:
            return content

        if BASH_STDERR_SEPARATOR in content:
            stdout, stderr = content.split(BASH_STDERR_SEPARATOR, 1)
        else:
            stdout, stderr = content, ""

        stderr_keep = (
            stderr[-self._BASH_STDERR_KEEP:]
            if len(stderr) > self._BASH_STDERR_KEEP
            else stderr
        )
        overhead = len(BASH_STDERR_SEPARATOR) + 80 if stderr_keep else 0
        stdout_budget = max(500, budget - len(stderr_keep) - overhead)
        if len(stdout) > stdout_budget:
            half = stdout_budget // 2
            stdout_keep = (
                stdout[:half]
                + f"\n[... {len(stdout) - stdout_budget} stdout chars elided ...]\n"
                + stdout[-half:]
            )
        else:
            stdout_keep = stdout

        if stderr_keep:
            return stdout_keep + BASH_STDERR_SEPARATOR + stderr_keep
        return stdout_keep


def _swarm_loop_policy() -> LoopPolicy:
    return LoopPolicy(
        terminal_tool_names=("submit_report",),
        no_tool_behavior="nudge",
        no_tool_nudge_message=_SUBAGENT_NO_TOOL_NUDGE,
    )


def _sub_tool_names(runtime: SwarmSubagentRuntime | None = None) -> list[str]:
    if runtime is not None and runtime.sub_agent_tool_names is not None:
        return list(runtime.sub_agent_tool_names)
    from frontier_agent.core.runtime.resources.manager import ResourceManager
    rm = registry.get_optional(ResourceManager)
    if rm is None:
        return []
    role_id = getattr(runtime, "sub_role_id", None) or SUB_ROLE_ID
    return list(rm.get_tool_names_for_role(role_id))


def _sub_tools(runtime: SwarmSubagentRuntime | None = None) -> list[Any]:
    """Resolve the actual tool objects bound to a sub-agent role.

    Used by ``_swarm_observers`` to embed the OpenAI tool schema in each
    sub-agent's saved trajectory. Returns ``[]`` when the resource manager
    isn't registered (e.g. unit tests).
    """
    if runtime is not None and runtime.sub_agent_tools is not None:
        return list(runtime.sub_agent_tools)
    from frontier_agent.core.runtime.resources.manager import ResourceManager
    rm = registry.get_optional(ResourceManager)
    if rm is None:
        return []
    if runtime is not None and runtime.sub_agent_tool_names is not None:
        all_tools = rm.all_tools
        return [all_tools[name] for name in runtime.sub_agent_tool_names if name in all_tools]
    role_id = getattr(runtime, "sub_role_id", None) or SUB_ROLE_ID
    return list(rm.get_tools_for_role(role_id))


def _swarm_observers(
    runtime: SwarmSubagentRuntime,
    *,
    task_id: str,
    session_name: str,
    task_index: int = 1,
    task_id_for_sse: str | None = None,
    run_id: str = "",
    run_type: str = "",
) -> list[Any]:
    """Build the observer stack a sub-agent runs with.

    ``task_id`` is the AgentBus / LoopConfig scope id (may be a
    heavy-mode synthetic id like ``f"{root}.heavy_run_3"``).
    ``task_id_for_sse`` defaults to ``task_id``; heavy mode passes the
    root id so SSE / event_store stay scoped to the user-visible task.
    """
    from frontier_agent.components.observers.budget_observer import BudgetObserver
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
    from frontier_agent.components.observers.stop_signal_observer import StopSignalObserver
    from frontier_agent.components.observers.text_repetition_guard import (
        TextRepetitionGuard,
    )
    from workflows._shared.research.observers.assertion_observer import AssertionObserver
    from workflows._shared.research.observers.evidence_observer import EvidenceObserver
    from workflows._shared.research.observers.finalize_answer_observer import (
        FinalizeAnswerObserver,
    )
    from workflows.agent_team.observers.console import SubAgentLoggingObserver

    sse_task_id = task_id_for_sse if task_id_for_sse is not None else task_id

    tool_names = _sub_tool_names(runtime)
    terminal = "submit_report" if "submit_report" in tool_names else None

    observers: list[Any] = [
        SubAgentLoggingObserver(session_name=session_name),
        LeakedToolCallRetryObserver(tool_names=tool_names),
        # Repetition stop-loss (the search-specific rollback is mounted
        # below). ``RepetitionGuard`` hints at three consecutive turns of
        # byte-identical tool calls and stops at six; it is the ONLY guard
        # that can end this loop, because the rollback's budget expires into
        # permanent let-through and ``TextRepetitionGuard`` needs visible
        # prose the pathology does not produce (the deliberation goes to
        # ``thinking`` under ``thinking_format: tag``). ``TextRepetitionGuard``
        # covers the prose loop and also stops here. Both stop reasons are in
        # ``fan_in.INCOMPLETE_STOP_REASONS``, so the partial report still
        # reaches fan-in through ``force_final_answer``. Stopping is only
        # affordable for a sub-agent — the coordinator IS the run and gets a
        # hint at most.
        RepetitionGuard(stop_after=6),
        TextRepetitionGuard(enable_stop=True),
        ReactStepTracker(),
        EvidenceObserver(),
        AssertionObserver(),
        # Cooperative stop: main agent's stop_subagent registers a stop for
        # this session_id; we inject the stop prompt at the next turn boundary.
        StopSignalObserver(session_id=f"{task_id}::{session_name}"),
    ]
    if DuplicateQueryRollbackObserver.DEFAULT_TOOL_NAMES.intersection(tool_names):
        # Only meaningful for an agent that can search. The rollback pops the
        # turn before the duplicate search runs, so it costs an LLM call and
        # never a tool call.
        observers.append(DuplicateQueryRollbackObserver())
    if runtime.max_tokens:
        observers.append(BudgetObserver(max_tokens=runtime.max_tokens))
    if runtime.stuck_target_hint_after > 0:
        # Nudge, then quarantine, a repeatedly failing host. Deliberately not
        # mounted on the coordinator: its tool pool carries no network tools.
        from frontier_agent.components.observers.stuck_target_guard import (
            StuckTargetGuard,
        )
        observers.append(StuckTargetGuard(
            hint_after=runtime.stuck_target_hint_after,
            escalate_after=runtime.stuck_target_escalate_after,
            window=runtime.stuck_target_window,
        ))
    if runtime.stream_repetition_config is not None:
        from workflows.agent_team.stream_repetition import StreamRepetitionDeltaSink
        observers.append(StreamRepetitionDeltaSink())
    if terminal:
        observers.append(FinalizeAnswerObserver(tool_names=[terminal]))
        observers.append(FinalizationReserveObserver(
            reserve_turns=runtime.finalization_reserve_turns,
            message=_SUBAGENT_FINALIZATION_MESSAGE,
        ))
        observers.append(LastTurnForcer(terminal_tool=terminal))

    event_store = registry.get_optional(EventStore)
    if event_store is not None:
        observers.append(SSEObserver(
            event_store=event_store, task_id=sse_task_id,
            run_id=run_id, run_type=run_type,
        ))

    if runtime.trajectory_dir is not None:
        from frontier_agent.components.observers.trajectory import (
            TrajectoryFileObserver,
        )
        if runtime.trajectory_per_task:
            traj_filename = f"{session_name}.t{task_index:02d}"
        else:
            traj_filename = session_name
        observers.append(TrajectoryFileObserver(
            runtime.trajectory_dir / "subagents",
            filename=traj_filename,
            tools=_sub_tools(runtime),
            format_env_vars=("SWARM_TRAJECTORY_FORMATS",),
            tool_schema_detail="minimal",
            include_start_tool_names=False,
        ))

    # A second, ``worker_trace_dir``-scoped trace writer — coexists with the
    # TrajectoryFileObserver above rather than replacing it (see the
    # ``worker_trace_dir`` field docstring).
    if runtime.worker_trace_dir is not None and runtime.workflow_id:
        from workflows._shared.sdk_shim import (
            make_subagent_trace_observer,
        )
        wt = make_subagent_trace_observer(
            workflow_id=runtime.workflow_id,
            parent_agent_id=MAIN_AGENT_ID,
            sub_agent_id=session_name,
            session_name=session_name,
            trace_dir=runtime.worker_trace_dir,
            mode="frontier_agent_swarm",
        )
        if wt is not None:
            observers.append(wt)

    # The optional SDK protocol stream is not part of the OSS runtime.
    return observers


# Cap on per-job denial logs held for the result adapter. Only jobs whose
# adapter never ran (cancelled, timed out) linger, and a session tops out at
# MAX_TASKS_PER_SESSION tasks, so this is a leak backstop rather than a limit
# anything real reaches.
_MAX_TRACKED_DENIAL_JOBS = 64


async def _adapt_swarm_result(
    agent_result: AgentLoopResult,
    *,
    job_id: str,
    item: SubTask,
    session_name: str,
    runtime: SwarmSubagentRuntime,
    denials: tuple[tuple[str, str], ...] = (),
) -> SubAgentResult:
    """Normalise a sub-agent run into a :class:`SubAgentResult`.

    ``force_final_answer`` rescue runs in the bus's ``force_finalizer``
    hook (BEFORE absorb), not here — see
    :func:`build_swarm_session_runtime_spec`. By the time we reach
    this adapter, the rescue (if any) has already mutated
    ``agent_result.messages`` and ``agent_result.final_content``,
    and the bus has absorbed the change into the session.
    """
    from workflows._shared.research.result_adapter import adapt_result

    result = adapt_result(agent_result)
    prefix = session_name or job_id.rsplit(".", 1)[-1]

    assertions: list[dict[str, Any]] = []
    for idx, assertion in enumerate(result.get("assertions", []), 1):
        item_dict = dict(assertion)
        item_dict["id"] = f"as-{prefix}-{idx:03d}"
        assertions.append(item_dict)

    metadata = dict(getattr(agent_result, "metadata", {}) or {})
    metadata["evidence_cards"] = list(result.get("evidence_cards", []))
    metadata["assertions"] = assertions
    metadata["stopped_by"] = getattr(agent_result, "stopped_by", "") or ""

    final_content = result.get("final_content", "")
    if denials:
        # The refusal itself only ever reached the sub-agent, which cannot
        # authorize itself; the coordinator can. Prepending rather than
        # appending keeps it above a long report, matching how
        # ``fan_in.format_report_block`` places its own [NOTE: ...] line.
        from plugins.tools._deliverable_policy import render_denial_escalation
        blocked_writes = [
            list(entry) for entry in denials if entry[0] != "unverifiable"
        ]
        unverifiable_accesses = [
            list(entry) for entry in denials if entry[0] == "unverifiable"
        ]
        if blocked_writes:
            metadata["blocked_output_writes"] = blocked_writes
        if unverifiable_accesses:
            metadata["unverifiable_output_accesses"] = unverifiable_accesses
        final_content = render_denial_escalation(denials) + final_content

    return SubAgentResult(
        question=item.question,
        role_id=item.role_id,
        final_content=final_content,
        success=True,
        job_id=job_id,
        metadata=metadata,
    )


def build_swarm_session_runtime_spec(
    runtime: SwarmSubagentRuntime,
    *,
    session_name: str,
    task_id: str,
    task_id_for_sse: str | None = None,
    run_id: str = "",
    run_type: str = "",
) -> SubAgentRuntimeSpec:
    """Build a runtime spec for a single sub-agent session.

    The main-agent node calls this once per ``create_subagent`` to
    inject agent-team loop config, observers, and result adapter.

    ``task_id`` is the AgentBus / LoopConfig scope id (heavy mode passes
    a synthetic per-run id). ``task_id_for_sse`` defaults to ``task_id``;
    heavy mode passes the root user-facing id so the SSE stream stays
    keyed by what the user sees.
    """

    # Per-job real-token gauge, shared between _config (the trigger policy) and
    # _observers (the observer that updates it) via setdefault, so both see the
    # SAME instance for a given job.
    _gauges: dict[str, InputTokenGauge] = {}
    # Blocked /outputs writes per job. ``_context_setup`` opens the log and
    # ``_adapter`` drains it: the bus runs the adapter AFTER the context manager
    # has exited (bus.py, around the ``result_adapter`` call), so the contextvar
    # is already reset by then and the list has to be handed over explicitly.
    _denials: dict[str, list[tuple[str, str]]] = {}
    _tiered = runtime.context_compaction == "tiered" and runtime.max_len > 0
    from plugins.tools._overflow import spill_compacted_body

    def _config(_job_id: str, item: SubTask, max_turns: int) -> LoopConfig:
        if _tiered:
            gauge = _gauges.setdefault(_job_id, InputTokenGauge())
            compactor: Any = TieredCompactor(
                keep_tool_result=runtime.tier1_keep_tool_result,
                summary_llm=runtime.sub_agent_llm,
                relief_target=int(runtime.max_len * 0.6),
                gauge=gauge,  # calibrate relief to real tokens (unit-match trigger)
                spill=spill_compacted_body if runtime.compaction_spill else None,
                summary_retry_timeout_s=(
                    runtime.sub_agent_llm_timeout or LoopConfig.llm_timeout
                ),
            )
            compaction_policy: Any = InputTokenThresholdPolicy(
                gauge, compaction_trigger_tokens(runtime.max_len),
            )
            keep_recent = max(6, runtime.keep_recent_turns * 3)
        else:
            compactor = KeepLastNToolResultsCompactor(
                keep_tool_result=runtime.sub_keep_tool_result,
            )
            compaction_policy = None
            keep_recent = LoopConfig.keep_recent
        return LoopConfig(
            max_turns=max_turns,
            task_id=task_id,
            llm_session_id=llm_session_id(task_id, session_name),
            role_id=item.role_id,
            tool_result_max_chars=runtime.tool_result_max_chars,
            llm_timeout=int(runtime.sub_agent_llm_timeout or LoopConfig.llm_timeout),
            tool_timeout=int(runtime.sub_agent_tool_timeout or LoopConfig.tool_timeout),
            reasoning_only_timeout_s=runtime.reasoning_only_timeout_s,
            reasoning_only_max_tokens=runtime.reasoning_only_max_tokens,
            logical_call_timeout_s=runtime.logical_call_timeout_s,
            max_completion_tokens=(
                runtime.llm_max_tokens or LoopConfig.max_completion_tokens
            ),
            context_token_limit=runtime.sub_context_token_limit,
            compact_after_turns=runtime.sub_compact_after_turns,
            keep_recent=keep_recent,
            loop_policy=_swarm_loop_policy(),
            compactor=compactor,
            compaction_policy=compaction_policy,
            tool_result_post_processor=SwarmToolResultPostProcessor(),
        )

    def _observers(_job_id: str, _item: SubTask, task_index: int) -> list[Any]:
        obs = _swarm_observers(
            runtime,
            task_id=task_id,
            session_name=session_name,
            task_index=task_index,
            task_id_for_sse=task_id_for_sse,
            run_id=run_id,
            run_type=run_type,
        )
        if _tiered:
            obs = [*list(obs), _gauges.setdefault(_job_id, InputTokenGauge())]
            if runtime.max_input_tokens:
                obs.append(ContextSizeGuard(
                    max_input_tokens=runtime.max_input_tokens,
                    force_compaction_first=True,
                ))
        return obs

    async def _force_finalizer(raw_result: Any, _item: SubTask) -> Any:
        # Mirror the old _adapt_swarm_result behaviour: skip when no
        # LLM is bound (unit tests + lightweight call sites).
        if runtime.sub_agent_llm is None:
            return raw_result
        timeout = (
            runtime.sub_agent_llm_timeout
            if runtime.sub_agent_llm_timeout is not None
            else LoopConfig.llm_timeout
        )
        return await force_final_answer(
            raw_result,
            runtime.sub_agent_llm,
            timeout,
            task_description=_item.question,
        )

    async def _adapter(
        agent_result: Any, job_id: str, item: SubTask,
    ) -> SubAgentResult:
        return await _adapt_swarm_result(
            agent_result, job_id=job_id, item=item,
            session_name=session_name, runtime=runtime,
            denials=tuple(_denials.pop(job_id, ())),
        )

    @contextlib.contextmanager
    def _context_setup(job_id: str, _item: SubTask) -> Iterator[None]:
        """Scope a per-sub-agent bwrap sandbox for the duration of the loop.

        Binds ``worktree_root/<session_name>`` (this sub-agent's private dir)
        plus the shared dir; sibling sub-agents are not bound, so they are
        invisible to one another. No-op when no worktree is configured or
        bwrap is unavailable (falls back to the inherited sandbox).
        """
        from frontier_agent.components.agent_bus.stop_signal import get_stop_registry
        from plugins.tools._bash_policy import reset_policy_mode, set_policy_mode
        from plugins.tools._deliverable_policy import (
            new_output_write_denial_log,
            reset_deliverable_write_paths,
            reset_output_write_denial_log,
            set_deliverable_write_paths,
        )
        from plugins.tools._sandbox import (
            BwrapSandbox,
            bwrap_available,
            clear_task_sandbox,
            make_current_sandbox,
            set_task_sandbox,
        )

        # Reusable session: drop any cooperative-stop flag left over from a
        # PRIOR task so it can't roll back turn 1 of this new one. A stop for
        # THIS job is only queued after it starts running (see stop_subagent),
        # i.e. strictly after this point, so it is preserved.
        get_stop_registry().clear_stale(f"{task_id}::{session_name}", job_id)

        # Default sub-agent bash to the command allowlist too. Set BEFORE the
        # no-worktree / no-bwrap fallback (which just yields) so that path is
        # covered — not only the bwrap branch. AgentBus builds the sub-agent's
        # ExecutionScope without our scope_metadata, so the contextvar is the
        # reliable carrier here. Reset in the outer finally (covers both paths).
        policy_token = set_policy_mode("enforce")
        can_publish = _item.metadata.get("can_publish") is True
        output_paths = (
            _item.metadata.get("output_paths", []) if can_publish else []
        )
        # Manifest entries superseded by a replace_manifest follow-up: the
        # publisher may remove them from /outputs so a format change does not
        # leave the old deliverable stranded there.
        retired_paths = (
            _item.metadata.get("retired_paths", []) if can_publish else []
        )
        deliverable_token = set_deliverable_write_paths(
            output_paths, retired_paths,
        )
        # A stale entry would attach the previous task's denial to this one, so
        # the store is keyed per job and overwritten, not accumulated. The
        # adapter pops its entry, but it never runs for a cancelled or
        # timed-out job, so drop the oldest keys rather than growing forever.
        while len(_denials) >= _MAX_TRACKED_DENIAL_JOBS:
            _denials.pop(next(iter(_denials)))
        denial_token, _denials[job_id] = new_output_write_denial_log()
        try:
            # Container mode: every sub-agent SHARES the one mounted /workspace
            # (and /outputs) via a CurrentSandbox. AgentBus builds the sub's
            # ExecutionScope without the main node's scope_metadata, so we set
            # the shared sandbox on the contextvar explicitly here. No kill on
            # exit — the surrounding container owns the mount's lifecycle.
            if runtime.sandbox_mode in ("container", "native") and runtime.shared_workspace_dir:
                token = set_task_sandbox(
                    make_current_sandbox(runtime.shared_workspace_dir)
                )
                try:
                    yield
                finally:
                    clear_task_sandbox(token)
                return
            if runtime.worktree_root is None or not bwrap_available():
                yield
                return
            sandbox = BwrapSandbox(
                workspace=str(runtime.worktree_root / session_name),
                binds=tuple(
                    (src, dst, ro or (not output_paths and dst == "/outputs"))
                    for src, dst, ro in runtime.sub_binds
                ),
            )
            token = set_task_sandbox(sandbox)
            try:
                yield
            finally:
                clear_task_sandbox(token)
                # Reclaim the sandbox's mkdtemp tmpdir — without this each
                # sub-agent leaks one private /tmp dir for the worker's
                # lifetime (the main_agent node path kills its sandbox too).
                kill = getattr(sandbox, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except Exception:
                        logger.warning("agent_team sub-agent sandbox kill failed", exc_info=True)
        finally:
            reset_output_write_denial_log(denial_token)
            reset_deliverable_write_paths(deliverable_token)
            reset_policy_mode(policy_token)

    from frontier_agent.core.runtime.loop.model_profile import HistoryPolicy

    return SubAgentRuntimeSpec(
        config_builder=_config,
        observers_builder=_observers,
        result_adapter=_adapter,
        force_finalizer=_force_finalizer,
        context_setup=_context_setup,
        model_profile=runtime.sub_agent_model_profile,
        history_policy=HistoryPolicy(
            thinking_in_history=runtime.sub_agent_thinking_in_history,
            thinking_history_max_tokens=(
                runtime.sub_agent_thinking_history_max_tokens
            ),
        ),
    )
