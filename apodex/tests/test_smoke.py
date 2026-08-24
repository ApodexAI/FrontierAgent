"""Mock-LLM smoke + end-to-end tests for apodex.

No network. A small scripted LLM client (:class:`_ScriptedLLM`) drives the
real ``run_agent_loop`` engine, the local file tools, and our observer /
renderer against a temp working directory.

The scripted client implements ``stream`` with ``StreamDelta.tool_call_deltas``
(not just ``chat``) because ``TerminalObserver.wants_llm_delta = True`` makes
the loop take the streaming path — so the test exercises the same
token-streaming + tool-call-accumulation path a real model uses, and would
catch a regression that disables streaming. Each scripted turn is a native
OpenAI-wire assistant dict; ``stream`` emits its content and re-fragments any
``tool_calls`` into per-index ``StreamDelta`` deltas (id + JSON-encoded
arguments), which the loop stitches back into wire tool_calls by ``index``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from apodex.agent_tools import (
    RISK_CONFIRM,
    RISK_DENY,
    RISK_SAFE,
    assess_tool_risk,
    coding_tools,
)
from apodex.config import ModelConfig
from apodex.render import Renderer
from apodex.session import TerminalSession
from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.messages import ToolCall, assistant_msg, text_of


class _ScriptedLLM:
    """Native ``LLMClient`` that yields scripted assistant turns.

    ``chat`` pops the next scripted assistant dict and adapts it to an
    ``LLMResponse``; ``stream`` re-fragments the same turn into per-index
    ``StreamDelta``s carrying ``tool_call_deltas`` so the loop's streaming
    accumulator reconstructs the tool call (the path a real model uses).
    Accepts and ignores the per-call kwargs the loop binds (``tools``,
    ``temperature``, ``extra_headers``, ``max_tokens``, ``timeout``).
    """

    def __init__(self, model: str = "scripted-test", *, script: list | None = None) -> None:
        self.model = model
        self.script: list = list(script or [])

    def _pop(self) -> dict:
        return self.script.pop(0)

    async def chat(self, messages, **kw) -> LLMResponse:
        msg = self._pop()
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=list(msg.get("tool_calls") or []),
        )

    async def stream(self, messages, **kw):
        msg = self._pop()
        # Re-fragment the turn's wire tool_calls into per-index stream deltas:
        # id set once, name + JSON-encoded arguments forwarded as a fragment.
        deltas = [
            {
                "index": i,
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        yield StreamDelta(content=msg.get("content") or "", tool_call_deltas=deltas)


def _tc(name: str, args: dict, i: int) -> ToolCall:
    return {
        "id": f"c{i}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _session(ws: str, *, auto_approve: bool, interactive: bool, script: list) -> TerminalSession:
    s = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None),
        cwd=ws,
        renderer=Renderer(color=False),
        auto_approve=auto_approve,
        max_turns=8,
        interactive=interactive,
        mode="coding",  # legacy local-loop fixture; product defaults to react
    )
    s.llm = _ScriptedLLM(script=list(script))
    return s


def _history_text(session: TerminalSession) -> str:
    return " ".join(text_of(m.get("content")) for m in session.history)


def test_coding_tools_names():
    names = {t.name for t in coding_tools()}
    assert {"bash", "read_file", "write_file", "grep_search", "glob_search"} <= names
    assert {"file_editor_view", "file_editor_create", "file_editor_str_replace"} <= names


def test_risk_classification(tmp_path):
    cwd = str(tmp_path)
    assert assess_tool_risk("read_file", {"path": "a.py"}, cwd).level == RISK_SAFE
    assert assess_tool_risk("write_file", {"path": "a.py"}, cwd).level == RISK_CONFIRM
    assert assess_tool_risk("write_file", {"path": "/etc/passwd"}, cwd).level == RISK_DENY
    assert assess_tool_risk("bash", {"command": "rm -rf /"}, cwd).level == RISK_DENY
    assert assess_tool_risk("bash", {"command": "ls -la"}, cwd).level == RISK_SAFE  # read-only → auto
    assert assess_tool_risk("bash", {"command": "rm file.txt"}, cwd).level == RISK_CONFIRM  # mutating → confirm


def test_end_to_end_read_create_edit(tmp_path):
    """The agent reads a file, creates one, and edits it — on the real FS."""
    ws = str(tmp_path)
    (tmp_path / "hello.txt").write_text("MAGIC_42\n")

    script = [
        assistant_msg("read it", tool_calls=[_tc("read_file", {"path": "hello.txt"}, 1)]),
        assistant_msg("create", tool_calls=[_tc("file_editor_create", {"path": "out.py", "content": "x = 1\n"}, 2)]),
        assistant_msg("edit", tool_calls=[_tc("file_editor_str_replace", {"path": "out.py", "old_str": "x = 1", "new_str": "x = 2"}, 3)]),
        assistant_msg("Done: read MAGIC_42, created out.py, set x=2."),
    ]
    session = _session(ws, auto_approve=True, interactive=False, script=script)
    asyncio.run(session.run_task("read hello, create out.py, set x=2"))

    out = tmp_path / "out.py"
    assert out.exists(), "agent should have created out.py on the real filesystem"
    assert out.read_text().strip() == "x = 2", "str_replace edit should have applied"

    joined = _history_text(session)
    assert "MAGIC_42" in joined, "read_file output should reach the conversation"
    assert "Access denied" not in joined, "cwd must be an authorized workspace"


def test_rejected_tool_is_skipped(tmp_path):
    """A rejected write must not touch the filesystem and must inform the model."""
    ws = str(tmp_path)
    script = [
        assistant_msg("write", tool_calls=[_tc("write_file", {"path": "danger.txt", "content": "nope"}, 1)]),
        assistant_msg("OK, I won't write that."),
    ]
    # interactive=False + auto_approve=False → Approver auto-rejects writes.
    session = _session(ws, auto_approve=False, interactive=False, script=script)
    asyncio.run(session.run_task("write danger.txt"))

    assert not (tmp_path / "danger.txt").exists(), "rejected write must not hit disk"
    assert "rejected" in _history_text(session).lower(), "model should be told it was rejected"


def test_llm_error_surfaced_clearly(tmp_path, monkeypatch, capsys):
    """On stopped_by='llm_error' the renderer shows the provider's own message
    (plus the short reason) and does NOT fire the doomed _force_final rescue
    (which would hit the same failure)."""
    import apodex.session as sess_mod
    from frontier_agent.core.loop_types import AgentLoopResult

    async def fake_loop(**kwargs):
        return AgentLoopResult(
            messages=[],
            final_content="",
            stopped_by="llm_error",
            metadata={
                "llm_error": "Error code: 401 - {'type': 'shell_api_error'}",
                "llm_error_reason": "non_transient",
            },
        )

    monkeypatch.setattr(sess_mod, "run_agent_loop", fake_loop)
    session = _session(str(tmp_path), auto_approve=True, interactive=False, script=[])
    asyncio.run(session.run_task("do something"))  # must not raise

    out = capsys.readouterr().out
    assert "LLM configuration error" in out, out
    assert "LLM call failed" not in out, out
    assert "non_transient" in out          # short reason tag
    assert "Error code: 401" in out        # provider's own message, verbatim
    assert "no answer produced" not in out  # _force_final was skipped
    assert "Final report" not in out
    assert "Result (" not in out


def test_response_truncated_partial_text_is_forced_to_a_final_answer(tmp_path):
    """A cut-off fragment is non-empty but is not a usable final answer."""
    from frontier_agent.core.loop_types import AgentLoopResult

    session = _session(
        str(tmp_path),
        auto_approve=True,
        interactive=False,
        script=[assistant_msg("Recovered complete answer.")],
    )
    result = AgentLoopResult(
        messages=[assistant_msg("The shell ate my `")],
        final_content="The shell ate my `",
        stopped_by="response_truncated",
    )

    final = asyncio.run(session._force_final(result))

    assert final == "Recovered complete answer."
    assert session.llm.script == [], "the rescue LLM call must not be skipped"


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 503 - upstream endpoint temporarily overloaded",
        "Error code: 429 - rate limit exceeded (request id req_8fdns2kx)",
        "Error code: 500 - internal server error, authored by upstream",
        "Read timed out after 180s",
    ],
)
def test_transient_llm_failures_are_not_blamed_on_configuration(message):
    """A retry-me failure must not send the user off editing a working profile."""
    from apodex.task_runner import _LLM_CONFIGURATION_ERROR_RE

    assert not _LLM_CONFIGURATION_ERROR_RE.search(message), message


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 401 - {'type': 'shell_api_error'}",
        "Error code: 403 - Forbidden",
        "invalid_api_key: the key is not valid",
        "model_not_found: no such model 'gpt-x'",
        "Connection refused to 127.0.0.1:30000",
        "base_url must include /v1",
        "Unauthorized",
        "authentication failed",
    ],
)
def test_configuration_llm_failures_are_recognized(message):
    from apodex.task_runner import _LLM_CONFIGURATION_ERROR_RE

    assert _LLM_CONFIGURATION_ERROR_RE.search(message), message


def test_llm_error_keeps_the_text_the_model_had_already_produced(
    tmp_path, monkeypatch, capsys,
):
    """A partial answer is real work: show it inside the error, not as a report."""
    import apodex.session as sess_mod
    from frontier_agent.core.loop_types import AgentLoopResult

    async def fake_loop(**kwargs):
        return AgentLoopResult(
            messages=[],
            final_content="I found the config bug in line 42 before dying.",
            stopped_by="llm_error",
            metadata={"llm_error": "Error code: 429 rate limit", "llm_error_reason": "transient"},
        )

    monkeypatch.setattr(sess_mod, "run_agent_loop", fake_loop)
    session = _session(str(tmp_path), auto_approve=True, interactive=False, script=[])
    asyncio.run(session.run_task("do something"))

    out = capsys.readouterr().out
    assert "LLM call failed" in out, out            # transient → not a config error
    assert "I found the config bug in line 42" in out
    assert "Partial output produced before the failure" in out
    assert "Final report" not in out                # but never framed as delivery
    assert "Result (" not in out


def test_generic_wall_deadline_is_not_reported_as_delivery(
    tmp_path, monkeypatch, capsys,
):
    import apodex.session as sess_mod
    from frontier_agent.core.loop_types import AgentLoopResult

    async def fake_loop(**kwargs):
        return AgentLoopResult(
            messages=[],
            final_content="Still validating the generated artifact.",
            stopped_by="wall_deadline",
            turns_used=12,
        )

    monkeypatch.setattr(sess_mod, "run_agent_loop", fake_loop)
    session = _session(
        str(tmp_path), auto_approve=True, interactive=False, script=[],
    )
    asyncio.run(session.run_task("build and validate the artifact"))

    out = capsys.readouterr().out
    assert "Incomplete output" in out
    assert "wall_deadline" in out
    assert "Still validating" in out
    assert "partial output was not saved as a final report" in out
    assert "Result (" not in out


def test_llm_error_exception_surfaced(tmp_path, monkeypatch, capsys):
    """An LLMError that escapes the loop is caught by type and shows the wrapped
    provider message — not mislabeled a generic 'agent loop failed'."""
    import apodex.session as sess_mod
    from frontier_agent.core.errors import LLMCallExhausted

    async def boom(**kwargs):
        raise LLMCallExhausted(ValueError("Error code: 429 rate limit"), "exhausted")

    monkeypatch.setattr(sess_mod, "run_agent_loop", boom)
    session = _session(str(tmp_path), auto_approve=True, interactive=False, script=[])
    asyncio.run(session.run_task("do something"))  # must not raise

    out = capsys.readouterr().out
    assert "LLM call failed" in out, out
    assert "exhausted" in out              # reason
    assert "Error code: 429" in out        # wrapped last_exc
    assert "agent loop failed" not in out


def test_non_llm_exception_not_mislabeled(tmp_path, monkeypatch, capsys):
    """A non-LLM exception raised by the loop must NOT be labeled an LLM failure
    (only LLMError hits the LLM branch; everything else stays generic)."""
    import apodex.session as sess_mod

    async def boom(**kwargs):
        raise ValueError("some tool blew up")

    monkeypatch.setattr(sess_mod, "run_agent_loop", boom)
    session = _session(str(tmp_path), auto_approve=True, interactive=False, script=[])
    asyncio.run(session.run_task("do something"))  # must not raise

    out = capsys.readouterr().out
    assert "agent loop failed" in out, out
    assert "some tool blew up" in out
    assert "LLM call failed" not in out


def test_smoke_image_vision_read_file(tmp_path, monkeypatch):
    """Smoke test: Agent reads an image ~/Downloads/dragon.png via read_file Vision VLM."""
    ws = str(tmp_path)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    dragon_png = downloads / "dragon.png"
    # Write a valid tiny PNG
    dragon_png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    # Mock _vision_read returning VLM transcription
    monkeypatch.setattr(
        "plugins.tools._reader_core._vision_read",
        lambda data, mime="image/png", question=None: "A low-poly orange Charmander (小火龙) sprite standing facing front-right."
    )

    script = [
        assistant_msg(
            "I'll inspect the dragon.png image using read_file.",
            tool_calls=[_tc("read_file", {"path": str(dragon_png)}, 1)],
        ),
        assistant_msg("The image ~/Downloads/dragon.png was read via vision: it shows a low-poly Charmander (小火龙)."),
    ]

    session = _session(ws, auto_approve=True, interactive=False, script=script)
    asyncio.run(session.run_task(f"Inspect image at {dragon_png}"))

    history_text = _history_text(session)
    assert "read via vision" in history_text
    assert "Charmander (小火龙)" in history_text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
