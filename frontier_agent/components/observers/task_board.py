"""Task-board reminders (shared ``TaskBoardObserver`` with product policy)."""

from agent_core.components.observers.task_board import TaskBoardObserver as _TaskBoardObserver


class TaskBoardObserver(_TaskBoardObserver):
    """Critical, so the board reminder is collected into the next turn.

    AgentCore defaults this observer to non-critical, where return values are
    dropped and the reminder would never reach the model.
    """

    critical: bool = True


__all__ = ["TaskBoardObserver"]
