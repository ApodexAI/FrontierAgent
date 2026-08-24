"""TodoWrite tool + a process-local task store (apodex parity).

The agent calls ``todo_write`` to maintain a visible plan/checklist for
multi-step work; the renderer shows the current list (with ✓ / ▶ / ○
status) so the user can follow progress. Single-session, single-process,
so a module-level store is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from apodex.tui.themes import GLYPHS
from frontier_agent.core.tool import tool
from plugins.tools._coerce import coerce_json_list

_VALID_STATUS = ("pending", "in_progress", "completed")
# Shared with every renderer so a plan looks the same in the sidebar, the line
# UI and the trace — and so the glyphs stay monochrome and take the theme colour.
_STATUS_GLYPH = {
    "pending": GLYPHS["pending"],
    "in_progress": GLYPHS["in_progress"],
    "completed": GLYPHS["ok"],
}


@dataclass
class TodoItem:
    content: str
    status: str = "pending"

    @property
    def glyph(self) -> str:
        return _STATUS_GLYPH.get(self.status, "○")


# Process-local store (one interactive session per process).
_TODOS: list[TodoItem] = []


def get_todos() -> list[TodoItem]:
    return list(_TODOS)


def set_todos(items: list[TodoItem]) -> None:
    _TODOS[:] = items


def clear_todos() -> None:
    _TODOS.clear()


def _parse(raw: object) -> list[TodoItem]:
    items: list[TodoItem] = []
    for entry in coerce_json_list(raw) or []:
        if isinstance(entry, dict):
            content = str(entry.get("content") or entry.get("task") or "").strip()
            status = str(entry.get("status") or "pending").strip().lower()
        elif isinstance(entry, str):
            content, status = entry.strip(), "pending"
        else:
            continue
        if not content:
            continue
        if status not in _VALID_STATUS:
            status = "pending"
        items.append(TodoItem(content=content, status=status))
    # Enforce "exactly one in_progress": keep the first, demote the rest to
    # pending so the plan always shows a single active step (Claude-Code rule).
    seen_active = False
    for it in items:
        if it.status == "in_progress":
            if seen_active:
                it.status = "pending"
            seen_active = True
    return items


@tool
async def todo_write(todos: list) -> str:
    """Record or update the task plan as a checklist (overwrites the list).

    Use this to plan multi-step work and to update progress as you go: call
    it once up front with your plan, then again to flip items to
    ``in_progress`` / ``completed``. Keep EXACTLY ONE item ``in_progress`` at a
    time, and mark a step ``completed`` as soon as it's done (don't batch).

    Args:
        todos: list of items, each ``{"content": str, "status":
            "pending"|"in_progress"|"completed"}``.
    """
    items = _parse(todos)
    if not items:
        return "Error: todo_write needs a non-empty list of {content, status} items."
    set_todos(items)
    done = sum(1 for i in items if i.status == "completed")
    lines = [f"Updated plan ({done}/{len(items)} done):"]
    lines.extend(f"  {i.glyph} {i.content}" for i in items)
    return "\n".join(lines)


__all__ = ["TodoItem", "clear_todos", "get_todos", "set_todos", "todo_write"]
