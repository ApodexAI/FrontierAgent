"""Runtime proof: prompt → FrontierAgent ``react`` → tool → final answer.

No browser, no terminal UI, no real model endpoint. Run it to confirm the
adapter drives the *real* runtime and that structured events arrive in order::

    SANDBOX_BACKEND=container .venv/bin/python -m deploy.huggingface.poc

``--real`` skips the mock server and uses whatever ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` / ``OPENAI_MODEL`` are already in the environment, which is
the smoke test to run once a production endpoint exists.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from deploy.huggingface.adapter import FrontierAgentAdapter
from deploy.huggingface.config import load_config, preflight
from deploy.huggingface.events import (
    ACTIVITY_FINISHED,
    ACTIVITY_STARTED,
    ARTIFACT_CREATED,
    ASSISTANT_DELTA,
    RUN_COMPLETED,
    DemoEvent,
)
from deploy.huggingface.mock_llm import MockLLMServer, text_turn, tool_call_turn
from deploy.huggingface.sessions import DemoSession, SessionStore

DEFAULT_PROMPT = (
    "Write a two-line summary of what FrontierAgent is to summary.md in the "
    "outputs directory, then tell me what you wrote."
)

_ARTIFACT_TEXT = (
    "# FrontierAgent\n\n"
    "FrontierAgent builds and evaluates LLM agent workflows.\n"
    "This file was produced by the react workflow in a demo run.\n"
)


def _script(session: DemoSession) -> list[Any]:
    """One real tool call, then a plain-text turn (which ends the react loop).

    The scripted path is absolute because that is what the model would read out
    of the react prompt in native mode — the run proves the tool really writes
    into *this* session's outputs directory.
    """
    return [
        tool_call_turn("write_file", {
            "path": str(session.outputs / "summary.md"),
            "content": _ARTIFACT_TEXT,
        }),
        text_turn(
            "I wrote summary.md with a two-line summary: FrontierAgent builds "
            "and evaluates LLM agent workflows, and this file came from a react "
            "demo run.",
        ),
    ]


async def run_poc(
    prompt: str, *, use_mock: bool, verbose: bool,
) -> tuple[bool, list[DemoEvent]]:
    stack = contextlib.ExitStack()
    with stack:
        runtime_root = Path(stack.enter_context(
            tempfile.TemporaryDirectory(prefix="frontier-demo-poc-"),
        ))
        os.environ.setdefault("SANDBOX_BACKEND", "native")
        os.environ["DEMO_RUNTIME_ROOT"] = str(runtime_root)
        os.environ.setdefault("DEMO_TASK_TIMEOUT_SECONDS", "180")
        os.environ.setdefault("DEMO_MAX_TURNS", "6")

        store = SessionStore(runtime_root, ttl_s=3600)
        session = store.create()

        if use_mock:
            server = stack.enter_context(MockLLMServer(
                script=_script(session), chunk_delay_s=0.005,
            ))
            os.environ["OPENAI_BASE_URL"] = server.base_url
            os.environ["OPENAI_API_KEY"] = "mock-key-for-local-poc"
            os.environ["OPENAI_MODEL"] = server.model
            print(f"mock endpoint: {server.base_url}")

        config = load_config()
        checks = preflight(config)
        if checks.warnings or checks.errors:
            print(checks.format())
        if not checks.ok:
            return False, []

        adapter = FrontierAgentAdapter(config)

        print(f"session:  {session.session_id}")
        print(f"prompt:   {prompt}\n")

        collected: list[DemoEvent] = []
        answer_chunks: list[str] = []
        async for item in adapter.run(session=session, prompt=prompt):
            collected.append(item)
            if item.type == ASSISTANT_DELTA:
                answer_chunks.append(str(item.data.get("text", "")))
                if verbose:
                    sys.stdout.write(str(item.data.get("text", "")))
                    sys.stdout.flush()
                continue
            _print_event(item)

        answer = "".join(answer_chunks)
        print(f"\nstreamed {len(answer)} characters of assistant text")
        files = sorted(p.name for p in session.outputs.rglob("*") if p.is_file())
        print(f"session outputs: {files or '(none)'}")
        return _verify(collected, files), collected


def _print_event(item: DemoEvent) -> None:
    data = dict(item.data)
    detail = str(data.pop("detail", ""))
    rendered = json.dumps(data, ensure_ascii=False, default=str)
    if len(rendered) > 300:
        rendered = rendered[:299] + "…"
    print(f"[{item.type}] {rendered}")
    if detail:
        print(f"    detail: {detail[:200]}")


def _verify(events: list[DemoEvent], files: list[str]) -> bool:
    """Assert the Milestone-2 claims rather than eyeballing the log."""
    types = [item.type for item in events]
    checks = {
        "a terminal event closed the run": bool(events and events[-1].is_terminal),
        "the run completed": types[-1:] == [RUN_COMPLETED],
        "a tool actually started": ACTIVITY_STARTED in types,
        "the tool produced a result": ACTIVITY_FINISHED in types,
        "assistant text streamed": ASSISTANT_DELTA in types,
        "an artifact was announced": ARTIFACT_CREATED in types,
        "the artifact reached the session": bool(files),
        "a non-empty final answer": bool(
            events and str(events[-1].data.get("answer", "")).strip(),
        ),
    }
    print()
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    if events and events[-1].is_terminal:
        final = str(events[-1].data.get("answer", "")).strip()
        print(f"\nfinal answer:\n{final[:600]}")
    return all(checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--real", action="store_true",
        help="use the OPENAI_* endpoint already in the environment",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="do not echo streamed text",
    )
    args = parser.parse_args(argv)
    ok, _events = asyncio.run(run_poc(
        args.prompt, use_mock=not args.real, verbose=not args.quiet,
    ))
    print("\nPOC:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
