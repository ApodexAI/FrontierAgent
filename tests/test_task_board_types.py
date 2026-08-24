from frontier_agent.components.task_board_types import (
    RESOLUTION_MARKS,
    VALID_RESOLUTION,
    count_resolutions,
)


def test_valid_resolutions_are_derived_from_the_rendering_map() -> None:
    assert tuple(RESOLUTION_MARKS) == VALID_RESOLUTION


def test_counts_exclude_cancelled_tasks_from_the_active_denominator() -> None:
    counts = count_resolutions([
        "resolved", "in_progress", "open", "cancelled", "cancelled",
    ])

    assert counts.total == 5
    assert counts.active == 3
    assert counts.resolved == 1
    assert counts.in_progress == 1
    assert counts.open == 1
    assert counts.cancelled == 2


def test_unknown_resolutions_are_counted_as_open() -> None:
    counts = count_resolutions(["future-status"])

    assert counts.total == 1
    assert counts.open == 1
    assert counts.active == 1
