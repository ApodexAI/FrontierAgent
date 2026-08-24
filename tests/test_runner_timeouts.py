"""The per-question ceiling and the no-progress watchdog."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

import benchmarks.public.runner.run_subprocess as R


def test_question_ceiling_stays_above_the_sub_agent_budget():
    """An outer kill below the inner budget truncates legitimate work.

    ``SUB_AGENT_TIMEOUT_S`` defaults to 5400s in .env.example; the runner's
    ceiling must leave room for that plus the coordinator's own turns.
    """
    assert R._QUESTION_TIMEOUT > 5400
    assert R._NO_PROGRESS_TIMEOUT < R._QUESTION_TIMEOUT


@pytest.mark.asyncio
async def test_watchdog_kills_a_silent_process(monkeypatch):
    monkeypatch.setattr(R, "_NO_PROGRESS_TIMEOUT", 3)
    monkeypatch.setattr(R, "_QUESTION_TIMEOUT", 600)
    log = Path(tempfile.mkdtemp()) / "agent.log"
    log.write_text("start\n")
    proc = await asyncio.create_subprocess_exec("sleep", "60")
    try:
        with pytest.raises(asyncio.TimeoutError, match="no_progress"):
            await R._wait_with_progress(proc, log, "hung")
    finally:
        proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_watchdog_spares_a_process_that_is_still_writing(monkeypatch):
    """Progress is measured by log writes, which every turn and tool touches."""
    monkeypatch.setattr(R, "_NO_PROGRESS_TIMEOUT", 3)
    monkeypatch.setattr(R, "_QUESTION_TIMEOUT", 600)
    log = Path(tempfile.mkdtemp()) / "agent.log"
    log.write_text("start\n")
    proc = await asyncio.create_subprocess_exec("sleep", "7")

    async def writer():
        for i in range(7):
            await asyncio.sleep(1)
            log.write_text("x" * (i + 2))

    task = asyncio.create_task(writer())
    try:
        assert await R._wait_with_progress(proc, log, "busy") == 0
    finally:
        task.cancel()
