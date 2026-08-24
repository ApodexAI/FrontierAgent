"""Non-blocking text-stream wrapper — console output must never wedge the."""

from __future__ import annotations

import atexit
import queue
import sys
import threading
from typing import TextIO

__all__ = ["NonBlockingStream", "nonblocking_stderr"]

_SENTINEL: object = object()


class NonBlockingStream:
    """Write-only text stream whose ``write()`` never blocks the caller.

    Implements the subset of the file protocol Rich's ``Console`` (and
    plain ``print(file=...)``) touch: ``write`` / ``flush`` / ``isatty``
    / ``encoding`` / ``fileno`` / ``closed``.
    """

    def __init__(self, target: TextIO, *, max_queue: int = 2000) -> None:
        self._target = target
        self._q: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self._dropped_total = 0     # cumulative drops
        self._dropped_reported = 0  # drops already covered by a notice
        self._dropped_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain,
            name="nonblocking-stream-writer",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self._shutdown)

    # ── file protocol ─────────────────────────────────────────────────

    def write(self, s: str) -> int:
        try:
            self._q.put_nowait(s)
        except queue.Full:
            # Pipe stalled long enough to back up the whole queue —
            # drop instead of blocking. Counted and reported on recovery.
            with self._dropped_lock:
                self._dropped_total += 1
        return len(s)

    def flush(self) -> None:
        """No-op — flushing is the writer thread's job. Never blocks."""

    def isatty(self) -> bool:
        try:
            return self._target.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._target.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", "utf-8")

    @property
    def closed(self) -> bool:
        return getattr(self._target, "closed", False)

    @property
    def dropped(self) -> int:
        """Cumulative writes dropped due to backpressure (tests/metrics)."""
        with self._dropped_lock:
            return self._dropped_total

    # ── writer thread ─────────────────────────────────────────────────

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            # Batch what's already queued so one syscall+flush covers a
            # burst (9-run fan-out turns produce write storms). Bounded:
            # an unbounded race against producers would keep the queue
            # forever empty (no backpressure → no drops → unbounded
            # memory growth while the pipe is stalled).
            # The queue is Queue[object] because it carries the sentinel
            # alongside the payload strings, so each item is narrowed on the way
            # out. Only str and _SENTINEL are ever enqueued (see ``write``), and
            # _SENTINEL is handled above, so the isinstance guards never reject
            # a real payload.
            parts: list[str] = [item] if isinstance(item, str) else []
            try:
                while len(parts) < 256:
                    nxt = self._q.get_nowait()
                    if nxt is _SENTINEL:
                        self._write_parts(parts)
                        return
                    if isinstance(nxt, str):
                        parts.append(nxt)
            except queue.Empty:
                pass
            self._write_parts(parts)

    def _write_parts(self, parts: list[str]) -> None:
        notice = ""
        with self._dropped_lock:
            unreported = self._dropped_total - self._dropped_reported
            if unreported:
                notice = (
                    f"\n[nonblocking-stream] pipe backpressure: "
                    f"dropped {unreported} writes\n"
                )
                self._dropped_reported = self._dropped_total
        try:
            if notice:
                self._target.write(notice)
            self._target.write("".join(parts))
            self._target.flush()
        except Exception:
            # Broken pipe / closed target: keep consuming the queue so
            # producers never back up — output is best-effort.
            pass

    def _shutdown(self) -> None:
        """atexit: best-effort drain so tail output isn't lost on clean exit."""
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            return
        self._thread.join(timeout=1.0)


_stderr_wrapper: NonBlockingStream | None = None
_stderr_wrapper_lock = threading.Lock()


def nonblocking_stderr() -> NonBlockingStream:
    """Process-wide non-blocking wrapper around ``sys.stderr`` (singleton)."""
    global _stderr_wrapper
    if _stderr_wrapper is None:
        with _stderr_wrapper_lock:
            if _stderr_wrapper is None:
                _stderr_wrapper = NonBlockingStream(sys.stderr)
    return _stderr_wrapper
