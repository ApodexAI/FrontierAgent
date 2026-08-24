"""Type-ahead steering — let the user redirect the agent while it works.

While a task runs we watch stdin with ``loop.add_reader`` (TTY only) and
collect whole lines the user types. The observer drains them at a turn
boundary and injects them as the next user message (live steering), and the
session re-runs any leftover after the task — so a typed instruction is never
lost. Borrowed from Claude Code / kimi's "queue while running" pattern, with a
key safety detail: the reader is **paused during approval prompts** so it never
races the single-key approver for the same file descriptor.

Cooked-mode caveat: the terminal echoes typed characters inline with the
agent's output (we don't take over raw mode). Functional, slightly noisy — a
full bottom-anchored input widget is a later refinement.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from typing import Any


class SteerInbox:
    """Collects whole lines typed during a run; drained at turn boundaries."""

    def __init__(self, renderer: Any) -> None:
        self.r = renderer
        self.queue: list[str] = []
        # Fan-in tools such as ``collect_reports`` can park the coordinator for
        # many minutes.  This event lets the agent loop wake that idle wait as
        # soon as an intervention is queued instead of polling the list (or
        # waiting for the sub-agent timeout to expire).
        self._available = asyncio.Event()
        self._buf = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fd: int | None = None
        self._attached = False  # reader currently registered on the loop

    # ── lifecycle ────────────────────────────────────────────────────────
    def attach(self) -> bool:
        """Start watching stdin. No-op (returns False) without a TTY / loop."""
        if not sys.stdin.isatty():
            return False
        try:
            self._loop = asyncio.get_running_loop()
            fd = sys.stdin.fileno()
            self._fd = fd
            self._loop.add_reader(fd, self._on_readable)
            self._attached = True
        except Exception:
            self._attached = False
        return self._attached

    def detach(self) -> None:
        self.pause()
        self._loop = None

    def pause(self) -> None:
        """Stop reading (e.g. while an approval prompt owns stdin)."""
        if self._attached and self._loop is not None and self._fd is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._fd)
        self._attached = False

    def resume(self) -> None:
        """Resume reading after an approval prompt releases stdin."""
        if not self._attached and self._loop is not None and self._fd is not None:
            try:
                self._loop.add_reader(self._fd, self._on_readable)
                self._attached = True
            except Exception:
                self._attached = False

    # ── reading ──────────────────────────────────────────────────────────
    def _on_readable(self) -> None:
        try:
            data = os.read(self._fd, 4096) if self._fd is not None else b""
        except Exception:
            return
        if data:
            self._feed(data.decode("utf-8", "replace"))

    def _feed(self, text: str) -> None:
        """Buffer text and enqueue each completed line (split on newline).

        Separated from ``_on_readable`` so it's unit-testable without a TTY."""
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self.enqueue(line)

    def enqueue(self, text: str) -> None:
        """Queue one intervention and wake a parked coordinator wait."""
        text = text.strip()
        if not text:
            return
        self.queue.append(text)
        self._available.set()
        with contextlib.suppress(Exception):
            self.r.queued(text)

    async def wait_for_input(self) -> bool:
        """Wait until at least one intervention is available.

        The loop calls this only while an aggregation tool is blocking.  The
        pre/post-clear checks make the signal race-free even when input lands
        immediately before this coroutine starts waiting.
        """
        while not self.queue:
            self._available.clear()
            if self.queue:
                break
            await self._available.wait()
        return True

    def drain(self) -> list[str]:
        items, self.queue = self.queue, []
        if not self.queue:
            self._available.clear()
        return items


__all__ = ["SteerInbox"]
