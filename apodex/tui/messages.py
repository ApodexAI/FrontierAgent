"""The one message the sink uses to talk to the app.

The agent run, its observers, and the sink all execute on the *same* asyncio
loop as the Textual app (the run is a Textual worker), so the sink could mutate
widgets directly. We funnel everything through a single ``Render`` message
instead, for one reason: ordering. ``post_message`` is processed in order on the
app's message pump, so streamed tokens, tool blocks, and notes always land in
the sequence they were emitted — even though the sink methods are synchronous
and widget mounts are async.

``Render.fn`` is a callable taking the app; it may return ``None`` (sync work)
or an awaitable (e.g. mounting a widget). The handler awaits whatever it gets.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.message import Message


class Render(Message):
    """Carry a UI mutation to run on the app's message pump, in order."""

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        super().__init__()
        self.fn = fn


__all__ = ["Render"]
