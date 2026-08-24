"""Tool: assign_task — non-blocking task dispatch to persistent sub-agents.

Wraps AgentBus.submit_task_to_session. Each call queues a task on a
previously-created session; the task runs in the background. Use
``collect_reports`` to fetch results (FIRST_COMPLETED semantics).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.execution_context import (
    get_current_execution_scope,
    get_current_tool_call_id,
)
from frontier_agent.core.runtime.registries import services as registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id
from plugins.tools._coerce import coerce_json_list
from plugins.tools._deliverable_policy import (
    normalise_output_paths,
    output_write_directives,
    render_publish_assignment,
    render_retirement_note,
    render_workspace_assignment,
)
from plugins.tools.create_subagent import (
    _normalize_agent_name,
    _resolve_runtime,
    _resolve_task_types,
)

logger = logging.getLogger(__name__)

# Actual concurrency is bounded by SpawnGuard(max_parallel).
# Raise via FRONTIER_AGENT_MAX_TASKS_PER_DISPATCH.
MAX_TASKS_PER_DISPATCH: int = int(
    os.environ.get("FRONTIER_AGENT_MAX_TASKS_PER_DISPATCH", "20")
)

# Hard cap on tasks per session. Past ~5 reuses the message history is
# dominated by stale tool stubs and earlier conclusions, and the agent
# starts anchoring on a prior (often wrong) answer instead of doing fresh
# work — heavily-reused sessions score far below fresh ones. The limit
# forces the main agent to spawn a new specialist instead of poisoning a
# saturated session.
MAX_TASKS_PER_SESSION = 5


def _session_at_task_cap(session: Any) -> bool:
    """Whether this session can no longer accept an assignment.

    Counts dispatched tasks AND queued-but-not-yet-dispatched ones:
    ``total_task_count`` only increments at dispatch time
    (``bus.py:_dispatch_session_task``), so tasks parked in ``pending_tasks``
    would otherwise slip past the cap. A missing session counts as capped —
    nothing can be assigned to a name the bus does not know.
    """
    if session is None:
        return True
    return (
        getattr(session, "total_task_count", 0)
        + len(getattr(session, "pending_tasks", ()) or ())
    ) >= MAX_TASKS_PER_SESSION


def _session_has_publish_work(bus: AgentBus, session: Any) -> bool:
    """Whether a session still has a running or queued publishing task.

    Publication authorization is copied into each dispatched task. Changing
    ``publication_state`` therefore cannot revoke an incumbent task that is
    already running or queued; transferring the role while such work exists
    would leave two agents authorized to write the same manifest.

    Metadata lookup failures are treated conservatively. A delayed transfer is
    recoverable on the next coordinator turn, while an unsafe transfer can
    corrupt the final deliverable.
    """
    if session is None:
        return False

    if getattr(session, "current_job_id", None) is not None:
        if not hasattr(bus, "current_job_metadata"):
            return True
        try:
            metadata = dict(bus.current_job_metadata(session.session_id) or {})
            if metadata.get("can_publish") is True:
                return True
            if "can_publish" not in metadata:
                return True
        except Exception:
            logger.warning(
                "assign_task: could not inspect current job metadata for %s",
                getattr(session, "session_id", "<unknown>"),
                exc_info=True,
            )
            return True

    for pending in getattr(session, "pending_tasks", ()) or ():
        metadata = getattr(pending, "task_metadata", None) or {}
        if metadata.get("can_publish") is True:
            return True
        if "can_publish" not in metadata:
            return True
    return False


# Cross-agent report attachment:
#   <attach agent="q1_lit"/>
# The main agent puts these tags inside a task prompt to feed another
# session's last report into this new task. We expand them here before
# dispatch so the downstream sub-agent sees the literal report text.
_ATTACH_RE = re.compile(r'<attach\s+agent="([^"]+)"\s*/>')


class AssignmentSpec(BaseModel):
    """The permissive assignment shape exposed by the shared registry tool."""

    # The registry-level tool has always ignored unrelated item keys, and
    # workflows other than agent_team may still bind it. Keep that runtime
    # leniency here without advertising agent-team-only publication controls
    # on the shared model-facing schema — agent_team installs its own
    # stricter Tool object in its loop instead.
    model_config = ConfigDict(extra="ignore")

    agent: StrictStr = Field(
        description=(
            "Name of a sub-agent created via create_subagent during this "
            "execution. Agent names mentioned in prior task history do not "
            "exist automatically."
        ),
        min_length=1,
    )
    prompt: StrictStr = Field(
        description="Concrete task prompt for that sub-agent.",
        min_length=1,
    )


# ``publish`` used to be the authorization field and ``output_paths`` its
# attachment. They were never independent: ``validate_publish_contract``
# rejected true-without-paths and paths-without-true alike, so the boolean
# carried nothing the manifest did not already carry. Requiring it only added a
# key the coordinator omitted on roughly a quarter of its assignments (60/236
# and 63/225 across the two arms of the 30-task APEX replay), where the default
# silently absorbed it as false -- and an omission next to a correct manifest
# was a hard rejection rather than a publisher. The manifest is now the grant;
# ``publish`` survives only so callers that still send it are not rejected.
#
# NOTE: this class's docstring is rendered verbatim into the model-facing tool
# schema, so keep it to what the coordinator needs to read.
class AgentTeamAssignmentSpec(AssignmentSpec):
    """One assignment. Carrying ``output_paths`` is what authorizes it to write
    those ``/outputs`` paths; without them the task runs workspace-only."""

    model_config = ConfigDict(extra="forbid")

    publish: StrictBool | None = Field(
        default=None,
        # Marked deprecated in the advertised schema only. Pydantic's own
        # ``deprecated=True`` warns on every attribute read, and
        # ``validate_publish_contract`` below reads it twice per assignment --
        # that is a DeprecationWarning per validated task in every run.
        json_schema_extra={"deprecated": True},
        description=(
            "Deprecated, omit it. output_paths alone grants publication. If "
            "sent it must agree with the manifest."
        ),
    )
    output_paths: list[StrictStr] = Field(
        default_factory=list,
        description=(
            "Exact absolute final file paths under /outputs. Supplying them IS "
            "the publication grant, so set them on exactly one final-publisher "
            "assignment and omit them on every other. Pass a JSON array even "
            "for one path."
        ),
    )
    replace_manifest: StrictBool = Field(
        default=False,
        description=(
            "Set true only on a follow-up to the existing publisher when the "
            "required final output formats genuinely changed."
        ),
    )

    @property
    def can_publish(self) -> bool:
        """Whether this assignment may write ``/outputs``.

        Derived from the manifest alone: a path list is the grant.
        """
        return bool(self.output_paths)

    @field_validator("publish", mode="before")
    @classmethod
    def normalise_publish_boolean(cls, value: Any) -> Any:
        """Accept common JSON boolean strings in the structured field."""
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in {"false", "true"}:
                return normalised == "true"
        return value

    @field_validator("output_paths", mode="before")
    @classmethod
    def normalise_null_output_paths(cls, value: Any) -> Any:
        """Treat an explicitly unused optional manifest like an omitted one."""
        return [] if value is None else value

    @field_validator("replace_manifest", mode="before")
    @classmethod
    def normalise_null_replace_manifest(cls, value: Any) -> Any:
        """Treat an explicitly unused optional replacement flag as false."""
        return False if value is None else value

    @model_validator(mode="after")
    def validate_publish_contract(self) -> AgentTeamAssignmentSpec:
        """Reject a compatibility flag that contradicts the manifest.

        A contradiction is raised rather than resolved in either direction:
        honouring the flag would drop a manifest the coordinator asked for
        (the shape that loses the deliverable), and honouring the manifest
        would widen authority on the strength of a call that says not to.
        """
        if not self.output_paths:
            if self.publish is True:
                raise ValueError(
                    "publish=true requires at least one exact absolute "
                    "output_paths entry"
                )
            if self.replace_manifest:
                raise ValueError("replace_manifest requires output_paths")
            return self

        if self.publish is False:
            raise ValueError(
                "publish=false contradicts output_paths; output_paths is the "
                "publication grant, so omit publish to authorize this manifest "
                "or drop output_paths for workspace-only work"
            )
        self.output_paths = list(normalise_output_paths(self.output_paths))
        return self


def _assignment_validation_error(index: int, exc: ValidationError) -> str:
    issues: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        error_location = error.get("loc", ())
        location = ".".join(str(part) for part in error_location) or "task"
        issues.append(f"{location}: {error.get('msg', 'invalid value')}")
    detail = "; ".join(issues) or "invalid assignment object"
    return f"task {index}: {detail}"


def _resolve_original_question(scope_metadata: dict[str, Any]) -> str:
    from plugins.tools._bus_scope import SWARM_SCOPE_KEY
    runtime = scope_metadata.get(SWARM_SCOPE_KEY)
    return getattr(runtime, "original_question", "").strip()


def _unknown_agent_validation_errors(
    raw_tasks: list[Any],
    *,
    bus: AgentBus,
    bus_task_id: str,
    task_types: tuple[str, ...],
) -> list[str]:
    """Surface lifecycle errors alongside structured metadata errors.

    Pydantic validation used to return before session lookup. A call that had
    both a bad manifest and a stale agent name therefore needed two retries:
    fixing the path merely uncovered ``Unknown agent`` on the next turn.
    """
    errors: list[str] = []
    for index, raw_spec in enumerate(raw_tasks, start=1):
        if not isinstance(raw_spec, dict):
            continue
        raw_name = raw_spec.get("agent")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        agent_name = _normalize_agent_name(raw_name.strip(), task_types)
        if agent_name and bus.get_session(f"{bus_task_id}::{agent_name}") is None:
            errors.append(
                f"task {index}: agent: Unknown agent {agent_name!r}; call "
                "create_subagent first. Sub-agents are scoped to the current "
                "execution and names from prior task history are not active"
            )
    return errors


def _expand_attach_tags(task_prompt: str, task_id: str, bus: AgentBus) -> str:
    """Replace ``<attach agent="NAME"/>`` with the named session's last report.

    If the referenced agent doesn't exist or has no report yet, leave a
    visible placeholder rather than silently dropping the tag — that
    gives the sub-agent a chance to say "referenced report missing" in
    its output instead of blindly proceeding on an incomplete task.
    """
    if "<attach" not in task_prompt:
        return task_prompt

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        session = bus.get_session(f"{task_id}::{name}")
        if session is None:
            return (
                f"[attach agent={name!r}: agent not found — "
                f"main must create it before attaching]"
            )
        report = (session.last_report or "").strip()
        if not report:
            return (
                f"[attach agent={name!r}: no report yet — "
                f"the agent has not completed a task]"
            )
        return (
            f"\n\n--- BEGIN REPORT FROM {name} ---\n"
            f"{report}\n"
            f"--- END REPORT FROM {name} ---\n\n"
        )

    return _ATTACH_RE.sub(_sub, task_prompt)


@tool
async def assign_task(tasks: list[AssignmentSpec] | str = "") -> str:
    """Assign tasks to previously-created sub-agents. Non-blocking.

    Each task is submitted to an existing session; the session runs its
    tasks strictly serially (one at a time). Submitting a second task
    while another is in flight queues it FIFO behind the running one —
    it starts automatically as soon as the predecessor finalises, and
    its report flows through ``collect_reports`` like any other.

    Args:
        tasks: list of dicts with:
            - ``agent`` (required): Name of a sub-agent previously
              created via ``create_subagent``.
            - ``prompt`` (required): The task prompt for this sub-agent.
            - ``output_paths`` (agent-team only): Exact absolute final file
              paths under ``/outputs``. Supplying them authorizes this
              assignment to write exactly those paths, so set them on the
              single final publishing assignment only.
            - ``publish`` (optional, agent-team only): Compatibility only.
              Authority comes from ``output_paths``; if sent it must agree
              with the manifest.
            - ``replace_manifest`` (optional): Set true on a follow-up to the
              existing publisher when the required final formats have changed.
              Dropped entries become removable so the publisher can clear the
              superseded files out of ``/outputs``.

    Returns:
        Summary of successful submissions and any errors.
    """
    # ``tasks`` defaults to ``""`` so models that emit an empty ``{}``
    # args object (qwen35-397B occasionally does, before it has decided
    # what to assign) hit our actionable error path instead of a raw
    # pydantic ``Field required`` ValidationError that costs a retry.
    # Some models also serialise the list as a JSON-encoded string —
    # ``coerce_json_list`` handles that.
    raw_tasks = coerce_json_list(tasks) or []
    if not raw_tasks:
        return "Error: assign_task requires at least one task."
    if not isinstance(raw_tasks, list):
        return (
            "Error: assign_task.tasks must be a JSON array of assignment "
            "objects (or a JSON-encoded array)."
        )

    if len(raw_tasks) > MAX_TASKS_PER_DISPATCH:
        return (
            f"Error: assign_task supports at most "
            f"{MAX_TASKS_PER_DISPATCH} tasks per call; you passed "
            f"{len(raw_tasks)}. Split the list across multiple calls so no "
            f"assignment is silently dropped."
        )

    scope = get_current_execution_scope()
    if scope is None:
        return (
            "Error: assign_task can only be called inside an "
            "active ReAct execution."
        )

    bus = registry.get(AgentBus)
    bus_task_id = resolve_bus_task_id(scope)
    original_question = _resolve_original_question(scope.metadata)
    # Resolve the active workflow's naming convention the SAME way
    # create_subagent does, so create and assign normalize names identically.
    # Without this, assign_task defaulted to swarm's ``{topic}_{task_type}``
    # convention and silently rewrote free-form agent_team role names
    # (e.g. ``taiwan_visa_verify`` → ``taiwan-visa_verify``,
    # ``taiwan__visa__verify`` → ``taiwan--visa-_verify``). create_subagent
    # (lenient for agent_team) stored the literal name, so the rewritten
    # lookup missed the just-created session or hit a saturated near-duplicate
    # — the main agent could never reliably reach its own sub-agents and
    # spiralled into an unbounded create/assign loop. A workflow that does
    # declare task types resolves to its own tuple → behaviour unchanged
    # for it.
    runtime = _resolve_runtime(scope.metadata)
    task_types = _resolve_task_types(runtime)
    is_agent_team = (
        runtime is not None
        and hasattr(runtime, "publication_state")
        and hasattr(runtime, "publication_lock")
    )
    spec_type = AgentTeamAssignmentSpec if is_agent_team else AssignmentSpec
    specs: list[AssignmentSpec] = []
    validation_errors: list[str] = []
    for index, raw_spec in enumerate(raw_tasks, start=1):
        try:
            # Backward-compatible runtime coercion for historical callers that
            # supplied one path string. The model-facing schema remains the
            # stronger ``array[string]`` shape so new tool calls learn the
            # canonical structure.
            if isinstance(raw_spec, dict) and "output_paths" in raw_spec:
                raw_output_paths = coerce_json_list(raw_spec["output_paths"])
                if isinstance(raw_output_paths, str):
                    raw_output_paths = [raw_output_paths]
                raw_spec = {**raw_spec, "output_paths": raw_output_paths}
            validation_input = (
                raw_spec.model_dump()
                if isinstance(raw_spec, BaseModel)
                and not isinstance(raw_spec, spec_type)
                else raw_spec
            )
            spec = (
                raw_spec
                if isinstance(raw_spec, spec_type)
                else spec_type.model_validate(validation_input)
            )
        except ValidationError as exc:
            validation_errors.append(_assignment_validation_error(index, exc))
            continue
        specs.append(spec)
    if is_agent_team and validation_errors:
        validation_errors.extend(_unknown_agent_validation_errors(
            raw_tasks,
            bus=bus,
            bus_task_id=bus_task_id,
            task_types=task_types or (),
        ))
        return (
            "Error: invalid agent-team assignment metadata: "
            + "; ".join(validation_errors)
            + ". No tasks were dispatched."
        )
    if not specs:
        return "Error: invalid assignment metadata: " + "; ".join(validation_errors)

    submitted: list[dict[str, str]] = []
    errors: list[str] = []
    notices: list[str] = []
    # Deliverable paths the ORIGINAL QUESTION names. It is prepended verbatim to
    # every dispatched prompt below, so every workspace-only sub-agent inherits
    # the user's own instruction to write them -- see the note on
    # ``output_write_directives``. Computed once: the text is the same for all
    # assignments in the run.
    question_directives = output_write_directives(original_question)
    # Keyed on the manifest, not on ``publish``: the manifest is the grant.
    # A plain ``AssignmentSpec`` has no such field, hence the getattr default.
    publish_specs = [spec for spec in specs if getattr(spec, "output_paths", ())]
    # Whether the coordinator DECLARED a publisher -- one in this dispatch, or
    # one the run already recorded. This is safe only as prompt context: a spec
    # can still fail to dispatch and its manifest may not cover the inherited
    # paths. Actual authorization is derived from publication_state after all
    # submissions finish below.
    recorded_publisher = ""
    if is_agent_team and runtime is not None:
        recorded_publisher = str(
            (getattr(runtime, "publication_state", None) or {}).get(
                "publisher_agent_name"
            )
            or ""
        )
    publisher_declared = bool(publish_specs) or bool(recorded_publisher)
    if is_agent_team and len(publish_specs) > 1:
        return (
            "Error: only one publishing assignment is allowed per dispatch. "
            "Choose one final integrator and one exact output manifest."
        )

    for spec in specs:
        agent_name = _normalize_agent_name(
            spec.agent.strip(), task_types,
        )
        task_prompt = spec.prompt.strip()
        if not agent_name:
            errors.append("Skipping task with no 'agent' field")
            continue
        if not task_prompt:
            errors.append(f"Skipping empty task for {agent_name!r}")
            continue
        session_id = f"{bus_task_id}::{agent_name}"
        session = bus.get_session(session_id)
        if session is None:
            errors.append(
                f"Unknown agent {agent_name!r} — call create_subagent first. "
                "Sub-agents are scoped to the current execution; a name from "
                "prior task history is not active automatically"
            )
            continue

        task_metadata: dict[str, Any] = {}
        publication_claim: tuple[str, ...] = ()
        publication_state: dict[str, Any] | None = None
        previous_publisher = ""
        previous_manifest: tuple[str, ...] = ()
        replace_manifest = False
        if (
            is_agent_team
            and runtime is not None
            and isinstance(spec, AgentTeamAssignmentSpec)
        ):
            if spec.can_publish:
                if "verifier" in agent_name.lower():
                    errors.append(
                        f"{agent_name}: verifier tasks cannot publish files; "
                        "return verification as report text"
                    )
                    continue
                output_paths = tuple(spec.output_paths)
                publication_state = runtime.publication_state
                replace_manifest = spec.replace_manifest
                publication_claim = output_paths
                task_metadata = {
                    "can_publish": True,
                    "output_paths": list(output_paths),
                }
                task_prompt += render_publish_assignment(output_paths)
            else:
                task_metadata = {"can_publish": False, "output_paths": []}
                # The coordinator's own wording can direct the same write the
                # question does. Both are neutralised the same way; only this
                # one is worth reporting back, because only this one is a
                # contradiction the coordinator authored and can fix.
                own_directives = output_write_directives(task_prompt)
                task_prompt += render_workspace_assignment(
                    inherited_paths=(*own_directives, *question_directives),
                    publisher_declared=publisher_declared,
                )
                if own_directives:
                    named = ", ".join(own_directives)
                    notices.append(
                        f"{agent_name}: dispatched workspace-only, but its "
                        f"prompt tells it to write {named}. That write is "
                        f"blocked for a non-publisher and the agent may report "
                        f"success after routing it elsewhere. If this agent is "
                        f"meant to produce the deliverable, re-assign it with "
                        f"output_paths={list(own_directives)!r}; "
                        f"otherwise expect its result under /workspace"
                    )

        if _session_at_task_cap(session):
            errors.append(
                f"{agent_name}: session has reached the {MAX_TASKS_PER_SESSION}"
                f"-task limit (dispatched={session.total_task_count}, "
                f"queued={len(session.pending_tasks)}). Long sticky sessions "
                f"anchor on prior conclusions and degrade accuracy. If you "
                f"still need NEW information, create a fresh sub-agent (e.g. "
                f"{agent_name}_v2 or a role-renamed variant) and assign the "
                f"task to it. But if this session already reported what you "
                f"need — or you were only trying to wrap up / gave it a "
                f"trivial task — do NOT spawn more agents: deliver your final "
                f"answer directly as plain text now."
            )
            continue

        # Expand any cross-agent <attach agent="..."/> tags before
        # dispatch so the sub-agent sees the actual report text.
        expanded_prompt = _expand_attach_tags(
            task_prompt, bus_task_id, bus,
        )
        if original_question and original_question[:100] not in expanded_prompt:
            expanded_prompt = (
                f"# Original Question\n{original_question}\n\n"
                f"# Your Task\n{expanded_prompt}"
            )

        # Build the spawn_context dict so the sub-agent's trace file is
        # stamped with the delegation lineage. ``parent_run_id`` is the
        # parent loop's run_id (set by
        # per-run state on fan-out paths; empty for
        # single-loop SDK callers). ``delegation_prompt`` is the verbatim
        # text the sub-agent sees, post attach-expansion.
        try:
            _allowed_tools = [
                getattr(t, "name", str(t))
                for t in (getattr(session, "tools", None) or [])
            ]
        except Exception:
            _allowed_tools = []
        scope_md = scope.metadata or {}
        spawn_context = {
            "parent_run_id": str(scope_md.get("run_id") or ""),
            "parent_agent_id": str(
                scope_md.get("agent_id") or scope_md.get("role_id") or ""
            ),
            "parent_turn": int(scope_md.get("current_turn") or 0),
            "spawned_by_llm_call_id": str(
                scope_md.get("last_llm_call_id") or ""
            ),
            "spawned_by_tool_call_id": str(get_current_tool_call_id() or ""),
            "delegation_prompt": expanded_prompt,
            "allowed_tools": _allowed_tools,
            "depth": 1,
            "budget": {
                "max_turns": int(getattr(session, "max_turns", 0) or 0),
            },
        }
        try:
            if (
                publication_claim
                and publication_state is not None
                and runtime is not None
            ):
                # Serialize the check, claim, and queue submission. The bus
                # await only enqueues work; it does not wait for the sub-agent.
                # Recording first closes the check-then-set race, while the
                # rollback preserves an earlier manifest if enqueueing fails.
                async with runtime.publication_lock:
                    previous_publisher = str(
                        publication_state.get("publisher_agent_name") or ""
                    )
                    previous_manifest = tuple(
                        publication_state.get("deliverable_manifest") or ()
                    )
                    previous_retired = tuple(
                        publication_state.get("retired_paths") or ()
                    )
                    if previous_publisher and previous_publisher != agent_name:
                        # The lock exists so two sub-agents cannot race on the
                        # same deliverable — NOT to make the role permanent. An
                        # incumbent that can no longer be dispatched used to
                        # deadlock the run outright: the task cap above told the
                        # coordinator to "create a fresh sub-agent" while this
                        # branch told it to "reuse" the capped one, so nothing
                        # could ever write /outputs again. Seen for real — a
                        # trial spent its last 6 turns alternating between the
                        # two errors and shipped no deliverable at all, despite
                        # having a finished answer in the workspace. Same trap
                        # ``finalize_answer._finalize_gate`` already documents
                        # for unassigned agents.
                        incumbent = bus.get_session(
                            f"{bus_task_id}::{previous_publisher}"
                        )
                        if not _session_at_task_cap(incumbent):
                            errors.append(
                                f"{agent_name}: publisher already assigned to "
                                f"{previous_publisher!r}, which can still take "
                                f"work — one publisher per run, so reuse that "
                                f"agent for the deliverable"
                            )
                            continue
                        if _session_has_publish_work(bus, incumbent):
                            errors.append(
                                f"{agent_name}: publisher {previous_publisher!r} "
                                "still has a publishing task running or queued; "
                                "wait for it with collect_reports before "
                                "transferring the publisher role"
                            )
                            continue
                        logger.warning(
                            "assign_task: transferring publisher role %r -> %r "
                            "(incumbent can no longer be dispatched)",
                            previous_publisher, agent_name,
                        )
                    if (
                        previous_manifest
                        and previous_manifest != publication_claim
                        and not replace_manifest
                    ):
                        errors.append(
                            f"{agent_name}: output manifest is already fixed as "
                            f"{list(previous_manifest)!r}; set "
                            "replace_manifest=true only if the required final "
                            "formats genuinely changed"
                        )
                        continue
                    # Entries dropped by a replacement would otherwise be
                    # stranded: every write path to them stays blocked, so the
                    # run would end with the old AND the new format present.
                    # Carrying them as ``retired_paths`` lets the publisher —
                    # and only the publisher — delete or move them out.
                    retired_paths = tuple(
                        path
                        for path in dict.fromkeys(
                            (*previous_retired, *previous_manifest)
                        )
                        if path not in publication_claim
                    )
                    publication_state["publisher_agent_name"] = agent_name
                    publication_state["deliverable_manifest"] = publication_claim
                    publication_state["retired_paths"] = retired_paths
                    task_metadata["retired_paths"] = list(retired_paths)
                    publish_prompt = expanded_prompt
                    if retired_paths:
                        publish_prompt += render_retirement_note(retired_paths)
                        spawn_context["delegation_prompt"] = publish_prompt
                    try:
                        job_id = await bus.submit_task_to_session(
                            session_id,
                            publish_prompt,
                            spawn_context=spawn_context,
                            task_metadata=task_metadata,
                        )
                    except Exception:
                        if previous_publisher:
                            publication_state["publisher_agent_name"] = (
                                previous_publisher
                            )
                        else:
                            publication_state.pop("publisher_agent_name", None)
                        if previous_manifest:
                            publication_state["deliverable_manifest"] = (
                                previous_manifest
                            )
                        else:
                            publication_state.pop("deliverable_manifest", None)
                        if previous_retired:
                            publication_state["retired_paths"] = previous_retired
                        else:
                            publication_state.pop("retired_paths", None)
                        raise
            else:
                job_id = await bus.submit_task_to_session(
                    session_id,
                    expanded_prompt,
                    spawn_context=spawn_context,
                    task_metadata=task_metadata,
                )
            submitted.append({"agent": agent_name, "job_id": job_id})
        except RuntimeError as exc:
            errors.append(f"{agent_name}: {exc}")
        except Exception as exc:
            logger.warning(
                "assign_task: failed for %s: %s", agent_name, exc,
            )
            errors.append(f"{agent_name}: {exc}")

    all_errors = [*validation_errors, *errors]
    if not submitted and all_errors:
        return "Error: " + "; ".join(all_errors)

    # Derive effective authority only after submission: merely carrying a
    # manifest spec does not establish it (the agent may be unknown or
    # capped, the manifest may conflict, or enqueueing may fail). The manifest
    # is also path-specific: a publisher for report.pdf cannot write answer.md.
    effective_manifest: tuple[str, ...] = ()
    if is_agent_team and runtime is not None:
        effective_manifest = tuple(
            (getattr(runtime, "publication_state", None) or {}).get(
                "deliverable_manifest"
            )
            or ()
        )
    uncovered_directives = tuple(
        path for path in question_directives if path not in effective_manifest
    )
    # A research-only round is a legitimate reason to have no authorized path,
    # so this is a notice, not an error. Left unsaid, it is exactly the shape
    # that loses the deliverable at finalization.
    if is_agent_team and submitted and uncovered_directives:
        named = ", ".join(uncovered_directives)
        if effective_manifest:
            replacement_manifest = list(dict.fromkeys(
                (*effective_manifest, *uncovered_directives),
            ))
            notices.append(
                f"No agent in this run can write {named}, which the question "
                "names as the deliverable. The recorded publisher manifest "
                f"covers {list(effective_manifest)!r}, not every required "
                "path. Reuse that publisher with replace_manifest=true and "
                f"output_paths={replacement_manifest!r}."
            )
        else:
            notices.append(
                f"No agent in this run can write {named}, which the question "
                "names as the deliverable. That is expected for a research or "
                "verification round; before the run ends, collect the "
                "workspace paths from these reports and assign one task "
                f"with output_paths={list(uncovered_directives)!r}."
            )

    lines = [
        f"Submitted {len(submitted)} task(s) in parallel "
        f"(agents run concurrently in the background):"
    ]
    for s in submitted:
        lines.append(f"  - {s['agent']}")
    if all_errors:
        lines.append("")
        lines.append("Warnings:")
        for e in all_errors:
            lines.append(f"  - {e}")
    if notices:
        lines.append("")
        lines.append("Publishing notices:")
        for n in notices:
            lines.append(f"  - {n}")
    lines.append("")
    lines.append(
        "Reports arrive automatically between turns. Call collect_reports() "
        "only if you have no useful local work and need a running agent's "
        "result before deciding."
    )
    return "\n".join(lines)


_AGENT_TEAM_ASSIGN_TASK_PARAMETERS = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": AgentTeamAssignmentSpec.model_json_schema(),
            "description": (
                "Assignment objects. Give output_paths to the one final "
                "publisher; omit it for workspace-only work."
            ),
        },
    },
    "required": ["tasks"],
}


def _top_level_grants_authority(
    publish: bool | str | None,
    output_paths: list[str] | str | None,
) -> bool:
    """Whether top-level metadata would hand out ``/outputs`` write authority.

    Read by both the fold and the guard in front of it. Keeping the test in one
    place is deliberate: a literal check on one side and a normalising check on
    the other is what previously let ``"True"`` slip past the guard while the
    fold refused to expand it, so the batch ran with the publish intent and its
    manifest silently dropped.

    ``output_paths`` counts on its own now that it is the grant -- without this
    a top-level manifest would be folded into every item of a research batch
    and authorize all of them. Before, each item's ``publish: false`` collided
    with the folded manifest and the contract rejected the call; with the
    boolean optional there is nothing left to collide with.
    """
    if AgentTeamAssignmentSpec.normalise_publish_boolean(publish) is True:
        return True
    if output_paths is None:
        return False
    coerced = coerce_json_list(output_paths)
    if isinstance(coerced, str):
        return bool(coerced.strip())
    return bool(coerced)


def _fold_top_level_publish_metadata(
    tasks: list[AgentTeamAssignmentSpec | dict[str, Any]] | str,
    *,
    publish: bool | str | None = None,
    output_paths: list[str] | str | None = None,
    replace_manifest: bool | str | None = None,
) -> list[AgentTeamAssignmentSpec | dict[str, Any]] | str:
    """Recover a common model formatting error without widening authority.

    The canonical schema keeps publication metadata inside each ``tasks[]``
    item.  Models occasionally emit the same metadata beside ``tasks``.  Metadata
    that grants nothing can safely apply to every item; anything that grants
    ``/outputs`` authority is accepted only for a single assignment, so it
    cannot accidentally grant publication rights to a batch of researchers.

    Only *missing* keys are filled in.  A call that already carries correct
    per-item metadata and merely echoes a summary value at the top level must
    come through unchanged: overwriting would demote the real publisher and
    then fail the contract on its now-contradictory ``output_paths``.
    """
    if publish is None and output_paths is None and replace_manifest is None:
        return tasks
    raw_tasks = coerce_json_list(tasks)
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return tasks
    # A non-mapping item is the model's error to hear about, not something to
    # crash on: leave the list alone so per-task validation reports it.
    if not all(isinstance(raw, (BaseModel, dict)) for raw in raw_tasks):
        return tasks
    publish_value = AgentTeamAssignmentSpec.normalise_publish_boolean(publish)
    if _top_level_grants_authority(publish, output_paths) and len(raw_tasks) != 1:
        return tasks
    folded: list[AgentTeamAssignmentSpec | dict[str, Any]] = []
    for raw in raw_tasks:
        item = raw.model_dump() if isinstance(raw, BaseModel) else dict(raw)
        if publish is not None:
            item.setdefault("publish", publish_value)
        if output_paths is not None:
            item.setdefault("output_paths", output_paths)
        if replace_manifest is not None:
            item.setdefault("replace_manifest", replace_manifest)
        folded.append(item)
    return folded


@tool(
    name="assign_task",
    description=assign_task.description,
    parameters=_AGENT_TEAM_ASSIGN_TASK_PARAMETERS,
)
async def agent_team_assign_task(
    tasks: list[AgentTeamAssignmentSpec | dict[str, Any]] | str = "",
    publish: bool | str | None = None,
    output_paths: list[str] | str | None = None,
    replace_manifest: bool | str | None = None,
) -> str:
    """Assign agent-team tasks, authorizing at most one of them to publish.

    Args:
        tasks: Assignment objects. The one final publisher carries its exact
            absolute ``output_paths`` manifest, which is what grants it write
            access to those paths; every other item omits ``output_paths`` and
            runs workspace-only.
    """
    tasks = _fold_top_level_publish_metadata(
        tasks,
        publish=publish,
        output_paths=output_paths,
        replace_manifest=replace_manifest,
    )
    if _top_level_grants_authority(publish, output_paths):
        # The fold silently declines to expand an authority-granting value
        # across a batch. Saying so is the difference between the coordinator
        # re-sending the manifest on the right item and a round that runs with
        # no publisher and no error to act on.
        parsed = coerce_json_list(tasks)
        if not isinstance(parsed, list) or len(parsed) != 1:
            return (
                "Error: top-level output_paths (or publish=true) is accepted "
                "only for one task; put the output_paths manifest inside the "
                "single tasks[] item that should publish."
            )
        raw_item = parsed[0]
        item = (
            raw_item.model_dump()
            if isinstance(raw_item, BaseModel)
            else raw_item
        )
        if isinstance(item, dict) and not _top_level_grants_authority(
            item.get("publish"), item.get("output_paths"),
        ):
            return (
                "Error: top-level output_paths (or publish=true) conflicts "
                "with the single tasks[] item, whose existing publish or "
                "output_paths value prevents that authority from being "
                "applied. Put the complete output_paths manifest inside that "
                "task and omit the top-level publication fields."
            )
    return await assign_task.func(tasks=tasks)


__all__ = [
    "AgentTeamAssignmentSpec",
    "AssignmentSpec",
    "_fold_top_level_publish_metadata",
    "_top_level_grants_authority",
    "agent_team_assign_task",
    "assign_task",
]
