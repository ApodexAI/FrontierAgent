"""recover_result — fetch back the part of a tool result the model never saw.

A tool result is cut in three places before it reaches the model, and the third
one throws its remainder away. ``ToolResultPostProcessor``
(``frontier_agent/core/runtime/loop/tool_exec.py``, applied at
``agent_loop.py:762``) caps the message text at a much smaller bound than the
150 K sitting in front of it — 15 000 chars for sub-agents
(``workflows/agent_team/subagent_runtime.py:532``), 4 000 for bash — and unlike
the two earlier cuts it persists nothing and leaves no pointer, only a
``[... truncated N chars past M-char cap]`` marker.

The content is not actually gone. That cut applies only to the string handed to
``tool_msg(...)``; the ``ToolResult`` keeps the full body, and
``notify_tool_result`` has already recorded it, so the JSONL trajectory holds
what the model cannot see. This tool reads it back.

Two properties follow from *where* it runs, and both matter:

* **In-process.** It deliberately does not import ``plugins.tools._sandbox``.
  That import is the only thing that makes a tool sandboxed, and the trajectory
  is not mounted into any sandbox. Reading it on the harness's own filesystem
  needs no mount, no ``_path_auth`` prefix, and no write guards.
* **The model never names a path.** It passes an opaque ``(turn, call_id)``
  handle, so there is no path-traversal surface to defend — unlike the spill
  store, whose agent-visible paths cost eight guard sites.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from frontier_agent.core.tool import tool

logger = logging.getLogger(__name__)

#: Cap on what one call returns, so a recovered 140 K body cannot simply
#: reintroduce the overflow the truncation was there to prevent. 8 K matches the
#: convention the rest of the pipeline already uses for self-paginating readers
#: (``read_file``'s upstream page size, and the post-processors' 6-10 K budgets),
#: which is what lets this tool sit in their ``_PASS_THROUGH`` sets without being
#: the largest thing in history. Paginate with ``offset`` for more.
_MAX_SLICE_CHARS = 8_000
#: Where the observer advertises the file. Kept in sync with
#: ``TrajectoryFileObserver.SCOPE_KEY``.
_SCOPE_KEY = "trajectory_jsonl"


def _trajectory_path() -> Path | None:
    """The JSONL file for the agent making this call, if one is being written.

    Read from ``ExecutionScope.metadata`` rather than a contextvar: the observer
    publishes from a hook that ``notify_observers`` dispatches as a separate
    task, where a contextvar write would be invisible. The scope object is shared
    by reference, so its dict is not.
    """
    try:
        from frontier_agent.core.execution_context import (
            get_current_execution_scope,
        )
        scope = get_current_execution_scope()
    except Exception:  # pragma: no cover - defensive
        return None
    if scope is None:
        return None
    raw = str(scope.metadata.get(_SCOPE_KEY) or "").strip()
    return Path(raw) if raw else None


def _find_record(path: Path, turn: int, call_id: str) -> dict[str, Any] | None:
    """Last ``t:"result"`` record matching ``(turn, call_id)`` in this run.

    **Last**, not first, and scoped to the final ``t:"start"``. The file is opened
    in append mode under a deterministic stem, so re-running the same task
    accumulates both runs in one file; and within a run ``agent_loop`` decrements
    the turn counter on ``continue_to_next_turn``, so one turn number can be
    attempted repeatedly. Synthetic call ids are deterministic
    (``f"call_{turn}_{idx}"``), so they collide exactly in both cases. Taking the
    last match after the last ``start`` reads the current attempt of the current
    run.

    Lines that do not parse are skipped rather than fatal: bodies reach 150 K
    against an 8 KB buffer, so a large record is flushed to disk in pieces, and
    ``_close_jsonl`` does not run on a hard kill.
    """
    hit: dict[str, Any] | None = None
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                kind = record.get("t")
                if kind == "start":
                    hit = None          # a new run begins; discard earlier matches
                    continue
                if kind != "result":
                    continue
                if record.get("turn") != turn:
                    continue
                # An empty id must never match. Pre-8a trajectories have no
                # ``tool_call_id`` field at all, and treating the absent field as
                # "" made a scan match the last result of that turn and return
                # the WRONG body while reporting success — measured on a real
                # trajectory. Silence beats a confident wrong answer.
                recorded = str(record.get("tool_call_id") or "")
                if recorded and recorded == call_id:
                    hit = record
    except OSError as exc:
        logger.debug("recover_result could not read %s: %s", path, exc)
        return None
    return hit


@tool
async def recover_result(
    turn: int, call_id: str, offset: int = 0, limit: int = 0,
) -> str:
    """Fetch back the part of an earlier tool result that was cut before you saw it.

    Only call this when a tool result you received said it was truncated AND you
    actually need the part that was cut. Use the exact ``turn`` and ``call_id``
    printed in that result's truncation notice — they are not guessable.

    Args:
        turn: The turn number from the truncation notice.
        call_id: The tool-call id from the truncation notice.
        offset: Character offset into the recovered body. Use it to page through
            content longer than one call returns. Default 0.
        limit: Maximum characters to return, capped at 20000. 0 means "as much as
            fits". Default 0.

    Returns:
        The requested slice of the original result, with a note on what remains,
        or a short explanation when the content cannot be recovered.
    """
    path = _trajectory_path()
    if path is None:
        return (
            "recover_result unavailable: this run keeps no recoverable "
            "transcript. Work from the truncated result you already have."
        )

    try:
        want_turn = int(turn)
    except (TypeError, ValueError):
        return f"recover_result: turn must be an integer, got {turn!r}."
    wanted_id = str(call_id or "").strip()
    if not wanted_id:
        return (
            "recover_result: call_id is required. Copy it from the truncation "
            "notice on the result you want back."
        )

    record = _find_record(path, want_turn, wanted_id)
    if record is None:
        return (
            f"recover_result found no stored result for turn={want_turn} "
            f"call_id={wanted_id!r}. Check both values against the truncation "
            "notice; only results that were actually truncated are recoverable."
        )

    body = str(record.get("result") or "")
    total = len(body)
    try:
        start = max(0, int(offset))
    except (TypeError, ValueError):
        start = 0
    if start >= total:
        return (
            f"recover_result: offset {start} is past the end of this "
            f"{total:,}-char result."
        )
    try:
        want = int(limit)
    except (TypeError, ValueError):
        want = 0
    width = _MAX_SLICE_CHARS if want <= 0 else min(want, _MAX_SLICE_CHARS)
    chunk = body[start:start + width]
    end = start + len(chunk)

    header = (
        f"[recovered {record.get('name') or 'tool'} result, turn {want_turn}, "
        f"chars {start:,}-{end:,} of {total:,}]\n"
    )
    if end < total:
        footer = (
            f"\n\n[... {total - end:,} chars remain; continue with "
            f"recover_result(turn={want_turn}, call_id=\"{wanted_id}\", "
            f"offset={end})]"
        )
    else:
        footer = "\n\n[end of result]"
    return header + chunk + footer


__all__ = ["recover_result"]
