"""task_board — a per-run task board for the coordinator (main agent)."""

from __future__ import annotations

import logging
from typing import Any

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.components.observers.task_board import TaskBoardObserver
from frontier_agent.components.task_board_types import (
    BOARD_TOOLS,
    RESOLUTION_MARKS,
    VALID_RESOLUTION,
    BoardCounts,
    count_resolutions,
)
from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.runtime import registry
from frontier_agent.core.tool import tool
from plugins.tools._bus_scope import resolve_bus_task_id
from plugins.tools._coerce import coerce_json_list, coerce_json_object

logger = logging.getLogger(__name__)

# task_id -> {"seq": int, "tasks": {id: {description, resolution, owners, group}}}
_BOARDS: dict[str, dict[str, Any]] = {}

# task_id -> list of pending board write-ops the stream observer has not yet
# drained. Each op is ``{"op": "add"|"update"|"finish_planning", "ids": [...],
# "phase": "planning"|"execution"}``. The board tools append here on every write;
# A task-board stream observer drains them and emits one
# ``response.swarm.task_board`` frame per op. Recording is unconditional + cheap (the
# eval / HTTP paths simply never drain), and ``clear_board`` empties it at run end
# so it can never leak across trials.
_PENDING_OPS: dict[str, list[dict[str, Any]]] = {}


def _record_op(task_id: str, op: str, ids: list[str] | None = None) -> None:
    """Append a board write-op for the stream observer to drain.

    The op stamps the phase AT WRITE TIME. In the agent-team two-loop profile the
    stream observer is not attached to the planning loop, so ops written there are
    drained late (once the execution loop runs) — by then the phase has flipped to
    ``execution``. Freezing the write-time phase keeps a planning-loop ``add_task``
    rendered as ``phase=planning`` rather than the phase current at drain time.
    """
    _PENDING_OPS.setdefault(task_id, []).append(
        {"op": op, "ids": list(ids or []), "phase": current_phase(task_id)}
    )


# ── State helpers (also used by the observer + finalize_answer gate) ────────

def _board(task_id: str) -> dict[str, Any]:
    return _BOARDS.setdefault(task_id, {"seq": 0, "tasks": {}})


def board_size(task_id: str) -> int:
    """Number of tasks on this task's board (0 if no board)."""
    b = _BOARDS.get(task_id)
    return len(b["tasks"]) if b else 0


def build_task_board_observer(*, cooldown_turns: int = 5) -> TaskBoardObserver:
    """Bind the shared observer to this plugin's task-board state."""
    return TaskBoardObserver(
        board_size=board_size,
        render_board=lambda task_id, bus_task_id: render_board(
            task_id, bus_task_id=bus_task_id,
        ),
        resolve_bus_task_id=resolve_bus_task_id,
        cooldown_turns=cooldown_turns,
    )


def clear_board(task_id: str) -> None:
    """Drop a task's board + phase — called at the end of the main-agent run."""
    _BOARDS.pop(task_id, None)
    _PHASE.pop(task_id, None)
    _PENDING_OPS.pop(task_id, None)


# ── Task-board stream projection — pure read helpers ─────────────────────────
# Consumed by the protocol layer's task-board stream observer
# to build the ``task_board.*`` wire frames. Kept here (next to the state they
# read) and side-effect-free so the observer stays a thin emitter.

def current_phase(task_id: str) -> str:
    """The board's coordinator phase — ``planning`` or ``execution``.

    Defaults to ``execution`` when the run never entered Planning Mode
    (``planning_mode:false`` profiles — apodex production — never call
    ``start_planning``, so ``_PHASE`` has no entry and the board is live in
    execution from the first ``add_task``)."""
    return _PHASE.get(task_id, "execution")


def serialize_tasks(task_id: str, ids: list[str]) -> list[dict[str, Any]]:
    """Render the given task ids to the wire shape, reading CURRENT board state.

    Reading the live ``resolution`` (rather than assuming ``open``) matters: an
    ``add_task`` that de-dups onto an already-``resolved`` id must carry that
    real resolution, or the frontend's whole-row replace would roll the task
    back to ``open`` and regress progress. Ids no longer on the
    board (raced clear) are skipped rather than emitted as ghosts."""
    b = _BOARDS.get(task_id)
    if not b:
        return []
    out: list[dict[str, Any]] = []
    for tid in ids:
        t = b["tasks"].get(tid)
        if t is None:
            continue
        out.append({
            "id": tid,
            "description": t.get("description", ""),
            "resolution": t.get("resolution", "open"),
            "owners": list(t.get("owners") or []),
            "group": t.get("group", ""),
        })
    return out


def snapshot_tasks(task_id: str) -> list[dict[str, Any]]:
    """Return the complete board in display order for UI projections.

    Unlike :func:`render_board`, this keeps task data structured so a client
    can render a real board instead of scraping the human-readable tool result.
    """
    b = _BOARDS.get(task_id)
    if not b:
        return []
    return serialize_tasks(task_id, list(b["tasks"]))


def drain_board_ops(task_id: str) -> list[dict[str, Any]]:
    """Pop and return all pending board write-ops for this task (FIFO)."""
    return _PENDING_OPS.pop(task_id, [])


# ── Planning Mode (two-phase state machine) ─────────────────────────────────
# task_id -> "planning" | "execution". Default (no entry) = execution, so ONLY
# the agent-team main agent (which calls start_planning) is ever gated; every
# other caller / workflow is unaffected.
_PHASE: dict[str, str] = {}
# In Planning Mode the agent is a PLANNER, not a solver: it may use only
# READ-ONLY tools (to understand the problem / look up a basic term) plus the
# board tools. EVERYTHING ELSE — team building, dispatch, code/file writes,
# finalize — is blocked until it calls finish_planning. This is an ALLOWLIST
# (the complement is blocked) so a newly-added write tool is denied by default.
_PLANNING_ALLOWED = (
    # read-only inspection / understanding
    "grep_search", "glob_search", "read_file", "read_text", "view_image",
    "web_search", "web_fetch",
    # board tools
    "add_task", "update_task", "finish_planning",
)


def start_planning(task_id: str) -> None:
    _PHASE[task_id] = "planning"


def force_finish_planning(task_id: str) -> None:
    """Flip a task out of Planning Mode WITHOUT the empty-board guard the
    ``finish_planning`` tool enforces. Used by the planning-turn-cap path
    (auto-finish at ``planning_max_turns``) and by the two-loop driver after a
    planning loop that ended on max_turns rather than an explicit finish."""
    if task_id in _PHASE:
        _PHASE[task_id] = "execution"


def is_planning_allowed(tool_name: str) -> bool:
    """True if ``tool_name`` may run during Planning Mode (read-only / board)."""
    return tool_name in _PLANNING_ALLOWED


def in_planning(task_id: str) -> bool:
    return _PHASE.get(task_id) == "planning"


def planning_enabled(task_id: str) -> bool:
    """True if this run enabled Planning Mode (start_planning was called) —
    stays True after finish_planning, so the no-solo / verify finalize gates
    keep applying through execution. ``False`` for non-planning runs."""
    return task_id in _PHASE


def planning_block_message(task_id: str, tool_name: str) -> str | None:
    """Error string to return when ``tool_name`` is called during Planning Mode;
    ``None`` when the call is allowed. No-op unless start_planning() was called
    for this task (i.e. only the agent-team main agent)."""
    if in_planning(task_id) and not is_planning_allowed(tool_name):
        return (
            f"Blocked: `{tool_name}` is unavailable in PLANNING MODE. You are a "
            "PLANNER here — use only READ-ONLY tools (grep_search / glob_search "
            "/ web_search to understand the problem) and the board tools "
            "(add_task / update_task). List EVERY sub-question via add_task, "
            "then call finish_planning() to unlock the team."
        )
    return None


def unresolved_task_ids(task_id: str) -> list[str]:
    """Ids not yet finished — open or in_progress. ``[]`` if no board.

    Cancelled tasks are retracted work (dropped on purpose / created in error),
    so they do NOT count as unresolved and never hold back the finalize gate."""
    b = _BOARDS.get(task_id)
    if not b:
        return []
    return [
        tid for tid, t in b["tasks"].items()
        if t.get("resolution") not in ("resolved", "cancelled")
    ]


def _exec_status_by_owner(task_id: str) -> dict[str, str]:
    """Map owner sub-agent name -> coarse execution status, derived from the bus.

    This is the system-owned half of the board: the model never writes it, so
    the model's ``resolution`` verdict can never clobber the live run state.
    """
    bus = registry.get_optional(AgentBus)
    if bus is None:
        return {}
    out: dict[str, str] = {}
    try:
        sessions = bus.list_sessions_for_task(task_id)
    except Exception:
        return {}
    for s in sessions:
        if getattr(s, "current_job_id", None) is not None:
            st = "running"
        elif getattr(s, "pending_tasks", None):
            st = "queued"
        elif getattr(s, "total_task_count", 0) == 0:
            st = "created"
        else:
            st = "reported"
        out[s.name] = st
    return out


def render_board(task_id: str, *, bus_task_id: str | None = None) -> str:
    """Render the board, joining model fields with live bus exec-status."""
    b = _BOARDS.get(task_id)
    if not b or not b["tasks"]:
        return "[task board] empty — call add_task to register sub-questions."
    tasks = b["tasks"]
    c = count_resolutions(t["resolution"] for t in tasks.values())
    lines = [
        f"[task board] resolved {c.resolved}/{c.active} · "
        f"in-progress {c.in_progress} · "
        f"open {c.open} · cancelled {c.cancelled}"
    ]
    # The owners/exec column only makes sense when there IS a team: in a solo
    # run (e.g. react) no task is ever assigned, so showing "agents=[unassigned]"
    # on every row is pure noise. Drop the whole column unless at least one task
    # has an owner — then skip the bus query too (nothing to join against).
    show_owners = any(t.get("owners") for t in tasks.values())
    exec_by_owner = _exec_status_by_owner(bus_task_id or task_id) if show_owners else {}
    for tid, t in tasks.items():
        row = (
            f"  {RESOLUTION_MARKS.get(t['resolution'], '○')} {tid} "
            f"{t['resolution']:<11} "
        )
        if show_owners:
            owners = t.get("owners") or []
            # one "name:exec_status" per owning agent, so the coordinator sees
            # exactly who is on this task and how far each has got.
            agents = (
                " · ".join(f"{o}:{exec_by_owner.get(o, '?')}" for o in owners)
                if owners else "unassigned"
            )
            row += f"agents=[{agents}]  "
        lines.append(f"{row}{t['description'][:80]}")
    return "\n".join(lines)


# ── Tools ───────────────────────────────────────────────────────────────────

def _as_owner_list(val: Any) -> list[str]:
    """Normalise an ``owner`` field (a name, a comma-string, or a list of names)
    into a clean list of agent names. One task can have MANY owners — several
    agents attacking the SAME sub-question from different angles for
    corroboration are all owners of that one task (not separate tasks)."""
    if val is None:
        return []
    items = val if isinstance(val, list) else str(val).split(",")
    out: list[str] = []
    for x in items:
        name = str(x).strip()
        if name and name not in out:
            out.append(name)
    return out


@tool
async def add_task(tasks: list[Any]) -> str:
    """Register the work items / sub-questions for this run on the task board.

    This is your plan and external memory. Call it up front — before doing any
    real work (fetching, running code, gathering evidence) — once you've broken
    the question into the concrete steps you'll work through, and again whenever
    a new sub-question emerges. (Reasoning, and a few clarifying searches to
    understand the problem, may come first.) Duplicate descriptions are
    de-duplicated (you get the existing id).

    Args:
        tasks: a list, each item ``{"description": str}``:
            - description (required): ONE concrete, checkable work item, e.g.
              "Verify the Markov condition Δω ≫ system rate holds" — not a vague
              area like "look into the math".
            - owner (OPTIONAL, multi-agent runs only): if a teammate sub-agent is
              already assigned to this item, name it here (a name, comma-string,
              or list — a task may have many). In a solo run, just OMIT it.

    Returns:
        The assigned ids plus the rendered board.
    """
    items = coerce_json_list(tasks) or []
    scope = get_current_execution_scope()
    if scope is None:
        return "Error: add_task can only be called inside an active run."
    if not items:
        return "Error: add_task requires at least one {description} item."
    b = _board(scope.task_id)
    existing = {t["description"].strip(): tid for tid, t in b["tasks"].items()}
    new_ids: list[str] = []
    skipped = 0  # items that weren't usable {description} objects
    for raw in items:
        it = coerce_json_object(raw)
        if it is None:
            skipped += 1
            continue
        desc = str(it.get("description", "")).strip()
        if not desc:
            skipped += 1
            continue
        if desc in existing:  # dedup re-decomposition
            new_ids.append(existing[desc])
            continue
        b["seq"] += 1
        tid = f"t{b['seq']}"
        b["tasks"][tid] = {
            "description": desc,
            "resolution": "open",
            "owners": _as_owner_list(it.get("owner") or it.get("owners")),
            "group": str(it.get("group", "")).strip(),
        }
        existing[desc] = tid
        new_ids.append(tid)
    logger.info(
        "add_task(task=%s): +%d skipped=%d (ids=%s)",
        scope.task_id, len(new_ids), skipped, new_ids,
    )
    # Nothing landed but items were passed → the shape was wrong. Return a
    # corrective error (not a silent "Added []") so the model re-sends the right
    # shape instead of burning turns repeating the mistake.
    if not new_ids:
        return (
            'Error: add_task expects a list of objects like '
            '[{"description": "..."}]; none of the items were usable, so nothing '
            'was added. Re-call with each task as its own '
            '{"description": "<one concrete work item>"}.'
        )
    _record_op(scope.task_id, "add", new_ids)
    msg = f"Added {new_ids}.\n{render_board(scope.task_id, bus_task_id=resolve_bus_task_id(scope))}"
    if skipped:
        msg += (
            f'\nNote: {skipped} item(s) were skipped (not a {{"description": ...}} '
            "object). Re-add them with that shape if still needed."
        )
    return msg


@tool
async def update_task(updates: list[Any]) -> str:
    """Mark progress on the board — call this the MOMENT a task finishes.

    Update each task as soon as it is done, before starting the next one; don't
    let finished tasks pile up. The arg is a LIST only to cover the case where
    two tasks finished in the SAME turn (resolve both at once) — it is NOT a
    reason to batch resolutions across turns. Returns the full updated board.

    Args:
        updates: a list of ``{"id", "resolution"?}`` items:
            - id (required): task id from add_task, e.g. "t3".
            - resolution: "in_progress" (set this the MOMENT you start working a
              task — it marks the one you are on now) | "resolved" (the work item
              is answered AND corroborated — your judgment, not merely "I glanced
              at it") | "cancelled" (retract a task you no longer need or created
              in error — it stops counting toward unresolved work; the id stays
              on the board for the trail) | "open" (the default; not started).
            - owner (OPTIONAL, multi-agent runs only): teammate sub-agent
              name(s) now working it — ADDED to the task's owner list (a name,
              comma-string, or list). Set "replace_owners": true to overwrite
              instead of add. Omit entirely in a solo run.

    Returns:
        The full updated task board (+ any per-item errors).
    """
    items = coerce_json_list(updates) or []
    scope = get_current_execution_scope()
    if scope is None:
        return "Error: update_task can only be called inside an active run."
    b = _BOARDS.get(scope.task_id)
    if not b:
        return "Error: no task board yet — call add_task first."
    changed: list[str] = []
    errors: list[str] = []
    for raw in items:
        u = coerce_json_object(raw)
        if u is None:
            continue
        tid = str(u.get("id", ""))
        if tid not in b["tasks"]:
            errors.append(f"{tid or '?'}: no such task")
            continue
        res = str(u.get("resolution", "")).strip()
        if res and res not in VALID_RESOLUTION:
            errors.append(f"{tid}: bad resolution {res!r} (use {VALID_RESOLUTION})")
            continue
        t = b["tasks"][tid]
        t.setdefault("owners", [])
        if res:
            t["resolution"] = res
        # owners ACCUMULATE — assigning another agent to the same task adds it,
        # it does not replace the existing owner(s). ``replace_owners: true``
        # overwrites (e.g. to drop an agent that was reassigned elsewhere).
        new_owners = _as_owner_list(u.get("owner") or u.get("owners"))
        if new_owners:
            if u.get("replace_owners"):
                t["owners"] = new_owners
            else:
                for o in new_owners:
                    if o not in t["owners"]:
                        t["owners"].append(o)
        changed.append(tid)
    if not changed and not errors:
        return ("Error: update_task needs a list of "
                "{id, resolution?, owner?} items.")
    # Emit ONLY the actually-changed ids — rejected ids (bad id /
    # bad resolution) stay off the wire so the frontend never builds ghost rows.
    if changed:
        _record_op(scope.task_id, "update", changed)
    msg = f"Updated {changed}.\n{render_board(scope.task_id, bus_task_id=resolve_bus_task_id(scope))}"
    if errors:
        msg += "\nerrors: " + "; ".join(errors)
    return msg


@tool
async def finish_planning() -> str:
    """Leave Planning Mode and start building the team.

    While planning, only add_task / update_task are available. Call this once
    your task board lists EVERY sub-question; afterwards create_subagent /
    assign_task / collect_reports become available (you can still add_task /
    update_task to refine the plan as the investigation unfolds).
    """
    scope = get_current_execution_scope()
    if scope is None:
        return "Error: finish_planning can only be called inside an active run."
    if board_size(scope.task_id) == 0:
        return (
            "Error: the task board is empty — call add_task to list the "
            "sub-questions before finishing planning."
        )
    _PHASE[scope.task_id] = "execution"
    _record_op(scope.task_id, "finish_planning")
    return (
        "Planning complete — now in EXECUTION mode. You may create_subagent / "
        "assign_task to build and dispatch the team.\n"
        + render_board(scope.task_id, bus_task_id=resolve_bus_task_id(scope))
    )


__all__ = [
    "BOARD_TOOLS",
    "RESOLUTION_MARKS",
    "VALID_RESOLUTION",
    "BoardCounts",
    "add_task",
    "board_size",
    "build_task_board_observer",
    "clear_board",
    "count_resolutions",
    "current_phase",
    "drain_board_ops",
    "finish_planning",
    "force_finish_planning",
    "in_planning",
    "is_planning_allowed",
    "planning_block_message",
    "planning_enabled",
    "render_board",
    "serialize_tasks",
    "snapshot_tasks",
    "start_planning",
    "unresolved_task_ids",
    "update_task",
]
