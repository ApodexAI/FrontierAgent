"""Dependency-light shared vocabulary for task-board producers and views."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

#: Tools whose successful completion can change the task rows shown by a board
#: projection. Planning-phase transitions do not change row state and therefore
#: intentionally stay outside this set.
BOARD_TOOLS = frozenset({"add_task", "update_task"})

#: The single source of truth for valid resolutions and their display glyphs.
#: Keep insertion order stable because ``VALID_RESOLUTION`` is also rendered in
#: validation errors.
RESOLUTION_MARKS: dict[str, str] = {
    "open": "○",
    "in_progress": "▶",
    "resolved": "✓",
    "cancelled": "⊘",
}
VALID_RESOLUTION = tuple(RESOLUTION_MARKS)


class BoardCounts(NamedTuple):
    """Board tallies with the "cancelled is retracted" rule applied once."""

    total: int
    resolved: int
    in_progress: int
    open: int
    cancelled: int

    @property
    def active(self) -> int:
        """Denominator for the resolved ratio.

        Cancelled tasks are retracted, so they leave the denominator—a dropped
        task must not sit forever in the unresolved count.
        """
        return self.total - self.cancelled


def count_resolutions(resolutions: Iterable[str]) -> BoardCounts:
    """Tally resolutions, folding anything unrecognised into ``open``."""
    counts = Counter(
        resolution if resolution in RESOLUTION_MARKS else "open"
        for resolution in resolutions
    )
    return BoardCounts(
        total=sum(counts.values()),
        resolved=counts["resolved"],
        in_progress=counts["in_progress"],
        open=counts["open"],
        cancelled=counts["cancelled"],
    )


__all__ = [
    "BOARD_TOOLS",
    "RESOLUTION_MARKS",
    "VALID_RESOLUTION",
    "BoardCounts",
    "count_resolutions",
]
