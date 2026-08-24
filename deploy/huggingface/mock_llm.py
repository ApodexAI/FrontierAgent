"""A fake OpenAI-compatible endpoint, for development and CI.

Deliberately a real HTTP server rather than a stubbed ``LLMClient``: it
exercises the code path a deployed Space actually uses — ``OPENAI_BASE_URL`` →
``OpenAIClient`` → SSE streaming → tool-call assembly — so a wire-format or
configuration regression is caught locally, without a token and without a GPU.

Run standalone::

    python -m deploy.huggingface.mock_llm --port 8018

Or drive it from a test::

    with MockLLMServer(script=[tool_call_turn("web_search", {"query": "x"}),
                              text_turn("done")]) as server:
        os.environ["OPENAI_BASE_URL"] = server.base_url
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_MODEL = "mock-frontier-model"


@dataclass
class Turn:
    """One scripted assistant turn."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    #: Exact SSE chunk boundaries for ``content``. Real chunk boundaries are
    #: arbitrary, so a test that needs a value split at a specific point (a
    #: secret straddling two frames, say) has to dictate them rather than hope
    #: the default word-splitter lands there.
    content_chunks: list[str] = field(default_factory=list)
    #: Set to return an HTTP error for this turn instead of a completion.
    status: int = 200
    error_body: str = ""
    #: Stall before responding — lets a test exercise timeout handling without
    #: waiting on a real slow endpoint.
    delay_s: float = 0.0

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"

    @property
    def full_content(self) -> str:
        """The assembled visible text, however it is chunked on the wire."""
        return "".join(self.content_chunks) if self.content_chunks else self.content


def text_turn(content: str, *, reasoning: str = "") -> Turn:
    """A plain-text turn. In the react loop this ends the run (no tool call)."""
    return Turn(content=content, reasoning=reasoning)


def tool_call_turn(
    name: str, arguments: dict[str, Any] | str, *, content: str = "",
) -> Turn:
    """A turn that calls one tool."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return Turn(
        content=content,
        tool_calls=[{
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {"name": name, "arguments": raw},
        }],
    )


def error_turn(status: int, body: str = "") -> Turn:
    """A turn that fails with an HTTP status, for error-mapping tests."""
    return Turn(status=status, error_body=body or f'{{"error":{{"message":"mock {status}"}}}}')


class _Script:
    """Thread-safe cursor over the scripted turns.

    The last turn repeats, so a script never runs out mid-run: an agent loop
    that takes one more turn than expected still terminates instead of 500ing.
    """

    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = list(turns) or [text_turn("mock answer")]
        self._index = 0
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []

    def next_turn(self, payload: dict[str, Any]) -> Turn:
        with self._lock:
            self.requests.append(payload)
            turn = self._turns[min(self._index, len(self._turns) - 1)]
            self._index += 1
            return turn

    @property
    def calls(self) -> int:
        with self._lock:
            return self._index

    def reset(self) -> None:
        with self._lock:
            self._index = 0
            self.requests.clear()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockOpenAI/1.0"

    # -- plumbing --------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:
        # Parameter names match BaseHTTPRequestHandler.log_message exactly.
        if getattr(self.server, "verbose", False):  # pragma: no cover
            super().log_message(format, *args)

    @property
    def _script(self) -> _Script:
        return self.server.script  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.server.model,  # type: ignore[attr-defined]
                    "object": "model",
                    "owned_by": "mock",
                }],
            })
            return
        self._send_json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        # The script cursor is per-server and advances across requests, so a
        # second smoke run would otherwise start past the scripted tool call.
        if path.endswith("/reset"):
            self._script.reset()
            self._send_json(200, {"ok": True})
            return
        if not path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return

        if self.server.require_auth and not self._authorized():  # type: ignore[attr-defined]
            self._send_json(401, {"error": {"message": "missing bearer token"}})
            return

        turn = self._script.next_turn(payload)
        if turn.delay_s:
            time.sleep(turn.delay_s)
        if turn.status != 200:
            self._send(turn.status, turn.error_body.encode(), "application/json")
            return

        if payload.get("stream"):
            self._stream(turn, payload)
        else:
            self._send_json(200, _completion_body(
                turn, model=str(payload.get("model") or self.server.model),  # type: ignore[attr-defined]
            ))

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization") or ""
        return header.startswith("Bearer ") and len(header) > len("Bearer ") + 1

    def _stream(self, turn: Turn, payload: dict[str, Any]) -> None:
        model = str(payload.get("model") or self.server.model)  # type: ignore[attr-defined]
        delay = float(getattr(self.server, "chunk_delay_s", 0.0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(chunk: dict[str, Any]) -> None:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            if delay:
                time.sleep(delay)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"

        def frame(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        emit(frame({"role": "assistant"}))
        for piece in _split_for_stream(turn.reasoning):
            emit(frame({"reasoning_content": piece}))
        for piece in (turn.content_chunks or _split_for_stream(turn.content)):
            emit(frame({"content": piece}))
        for index, call in enumerate(turn.tool_calls):
            emit(frame({"tool_calls": [{
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }]}))
            for piece in _split_for_stream(call["function"]["arguments"], size=24):
                emit(frame({"tool_calls": [{
                    "index": index,
                    "function": {"arguments": piece},
                }]}))
        emit(frame({}, turn.finish_reason))
        emit({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": _usage(turn),
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _split_for_stream(text: str, size: int = 12) -> Iterable[str]:
    """Chunk text so streaming is exercised, splitting on word boundaries."""
    if not text:
        return []
    pieces = re.findall(r"\S+\s*", text)
    out: list[str] = []
    buffer = ""
    for piece in pieces:
        buffer += piece
        if len(buffer) >= size:
            out.append(buffer)
            buffer = ""
    if buffer:
        out.append(buffer)
    return out


def _usage(turn: Turn) -> dict[str, int]:
    completion = max(1, (len(turn.full_content) + len(turn.reasoning)) // 4)
    return {
        "prompt_tokens": 128,
        "completion_tokens": completion,
        "total_tokens": 128 + completion,
    }


def _completion_body(turn: Turn, *, model: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.full_content or None}
    if turn.reasoning:
        message["reasoning_content"] = turn.reasoning
    if turn.tool_calls:
        message["tool_calls"] = turn.tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0, "message": message, "finish_reason": turn.finish_reason,
        }],
        "usage": _usage(turn),
    }


class MockLLMServer:
    """A threaded mock endpoint usable as a context manager."""

    def __init__(
        self,
        *,
        script: Sequence[Turn] | None = None,
        model: str = DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 0,
        require_auth: bool = True,
        chunk_delay_s: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self._script = _Script(script or [text_turn("mock answer")])
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.script = self._script          # type: ignore[attr-defined]
        self._server.model = model                  # type: ignore[attr-defined]
        self._server.require_auth = require_auth    # type: ignore[attr-defined]
        self._server.chunk_delay_s = chunk_delay_s  # type: ignore[attr-defined]
        self._server.verbose = verbose              # type: ignore[attr-defined]
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
            name="mock-openai",
        )
        self.model = model

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[0], self.port
        return f"http://{host}:{port}/v1"

    @property
    def calls(self) -> int:
        return self._script.calls

    @property
    def requests(self) -> list[dict[str, Any]]:
        return list(self._script.requests)

    def start(self) -> MockLLMServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def __enter__(self) -> MockLLMServer:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def _main() -> int:  # pragma: no cover - manual tool
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--answer",
        default="This is a mock answer from the FrontierAgent demo endpoint.",
    )
    parser.add_argument(
        "--no-auth", action="store_true", help="accept requests without a bearer token",
    )
    parser.add_argument("--chunk-delay", type=float, default=0.02)
    parser.add_argument(
        "--tool-demo", action="store_true",
        help=(
            "script one write_file call to outputs/answer.md before answering, "
            "so a deployment smoke test also covers tools and artifact download"
        ),
    )
    args = parser.parse_args()

    script = [text_turn(args.answer)]
    if args.tool_demo:
        # A *relative* path so the script works for any session: the runtime
        # resolves it against the session's own authorised workspace root.
        script = [
            tool_call_turn("write_file", {
                "path": "outputs/answer.md",
                "content": f"# Answer\n\n{args.answer}\n",
            }),
            *script,
        ]

    server = MockLLMServer(
        script=script,
        model=args.model,
        host=args.host,
        port=args.port,
        require_auth=not args.no_auth,
        chunk_delay_s=args.chunk_delay,
        verbose=True,
    )
    with server:
        print(f"mock OpenAI-compatible endpoint: {server.base_url}")
        print("OPENAI_BASE_URL=" + server.base_url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopping")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_MODEL",
    "MockLLMServer",
    "Turn",
    "error_turn",
    "text_turn",
    "tool_call_turn",
]
