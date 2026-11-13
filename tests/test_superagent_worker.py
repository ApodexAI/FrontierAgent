from __future__ import annotations

import asyncio
import importlib
import json
import uuid

import pytest

from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.loop_types import LoopConfig, LoopPolicy, TurnContext
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.runtime.observer_scope import execution_observers, observe_execution
from frontier_agent.core.tool import tool
from frontier_agent.superagent import Budget, Store
from frontier_agent.superagent.models import Fenced
from frontier_agent.superagent.worker import Bridge, ExecutionObserver, Jobs, run


class LocalBridge:
    def __init__(self, store, assignment):
        self.store = store
        self.assignment = assignment
        self.seen = []

    async def call(self, kind, payload, *, key=None):
        self.seen.append((kind, payload))
        try:
            return self.store.exchange(
                self.assignment.id, self.assignment.owner, key or uuid.uuid4().hex, kind, payload
            )
        except Fenced as exc:
            raise asyncio.CancelledError(str(exc)) from exc


class ScriptedLLM:
    model = "fixture"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append(list(messages))
        if not self.script:
            raise asyncio.CancelledError("Unexpected extra model call in scripted test")
        return self.script.pop(0)

    async def stream(self, messages, **kwargs):
        response = await self.chat(messages, **kwargs)
        deltas = [
            {
                "index": index,
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for index, call in enumerate(response.tool_calls)
        ]
        yield StreamDelta(content=response.content, tool_call_deltas=deltas)


def tool_response(name, args):
    return LLMResponse(
        tool_calls=[
            {
                "id": "tool-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    )


async def test_real_loop_persists_tool_intent_before_call_and_charges_requests(tmp_path):
    with Store(tmp_path) as store:
        ticket = store.submit("test", request_key="r")
        assignment = store.claim("owner")
        bridge = LocalBridge(store, assignment)
        observer = ExecutionObserver(bridge, Jobs(bridge), 1)

        @tool
        async def checked_write() -> str:
            """Perform a test action only after its intent is committed."""
            assert len(store.unresolved(ticket.id)) == 1
            return "written"

        llm = ScriptedLLM([tool_response("checked_write", {}), LLMResponse(content="done")])
        with observe_execution([observer]):
            result = await run_agent_loop(
                system_prompt="test",
                user_message="go",
                llm=llm,
                tools=[checked_write],
                config=LoopConfig(
                    task_id="root",
                    role_id="stateful_react",
                    max_turns=3,
                    loop_policy=LoopPolicy(no_tool_behavior="stop"),
                ),
            )
        assert result.final_content == "done"
        assert store.get(ticket.id).calls_used == 2
        assert not store.unresolved(ticket.id)
        assert store.context(ticket.id)["checkpoints"]
        assert execution_observers() == []


async def test_request_budget_fails_closed_before_second_model_call(tmp_path):
    with Store(tmp_path) as store:
        ticket = store.submit("test", request_key="r", budget=Budget(model_calls=1))
        bridge = LocalBridge(store, store.claim("owner"))
        observer = ExecutionObserver(bridge, Jobs(bridge), 1)

        @tool
        async def ping() -> str:
            """Return test data."""
            return "pong"

        llm = ScriptedLLM([tool_response("ping", {}), LLMResponse(content="must not call")])
        with observe_execution([observer]), pytest.raises(asyncio.CancelledError):
            await run_agent_loop(
                system_prompt="test",
                user_message="go",
                llm=llm,
                tools=[ping],
                config=LoopConfig(task_id="root", role_id="stateful_react", max_turns=3),
            )
        assert len(llm.requests) == 1
        assert store.get(ticket.id).calls_used == 1


async def test_bridge_rejection_is_cancellation_not_swallowed_observer_error():
    import io

    reader = asyncio.StreamReader()
    reader.feed_data(b'{"key":"test","ok":false,"error":"revoked"}\n')
    bridge = Bridge(reader, io.StringIO())
    with pytest.raises(asyncio.CancelledError):
        await bridge.call("tool_start", {}, key="test")


async def test_broken_output_pipe_also_fails_closed():
    class BrokenOutput:
        def write(self, text):
            raise BrokenPipeError("controller died")

    bridge = Bridge(asyncio.StreamReader(), BrokenOutput())
    with pytest.raises(asyncio.CancelledError):
        await bridge.call("tool_start", {})


async def test_team_child_loops_inherit_durable_observer(tmp_path):
    from agent_core.runtime.loop.agent_loop import run_agent_loop as shared_loop

    from workflows.agent_team.subagent_runtime import SwarmSubagentRuntime, _swarm_observers

    with Store(tmp_path) as store:
        ticket = store.submit("test", request_key="r")
        bridge = LocalBridge(store, store.claim("owner"))
        observer = ExecutionObserver(bridge, Jobs(bridge), 1)
        llm = ScriptedLLM([LLMResponse(content="child finished")])
        with observe_execution([observer]):
            observers = _swarm_observers(
                SwarmSubagentRuntime(), task_id="child", session_name="researcher"
            )
            await shared_loop(
                system_prompt="child",
                user_message="work",
                llm=llm,
                tools=[],
                observers=observers,
                config=LoopConfig(
                    task_id="child",
                    role_id="swarm_sub",
                    loop_policy=LoopPolicy(no_tool_behavior="stop"),
                ),
            )
        assert store.get(ticket.id).calls_used == 1


async def test_background_job_returns_before_completion_and_persists_result(tmp_path, monkeypatch):
    bash_module = importlib.import_module("plugins.tools.bash")
    release = asyncio.Event()

    async def delayed(command):
        await release.wait()
        return "command completed"

    monkeypatch.setattr(bash_module.bash, "func", delayed)
    with Store(tmp_path) as store:
        ticket = store.submit("test", request_key="r")
        bridge = LocalBridge(store, store.claim("owner"))
        jobs = Jobs(bridge)
        tools = jobs.tools()
        started = await tools["bash"].ainvoke({"command": "test"})
        assert started.startswith("Started job")
        assert jobs.pending
        assert len(store.unresolved(ticket.id)) == 1
        release.set()
        await asyncio.gather(*jobs.tasks.values())
        assert not store.unresolved(ticket.id)
        assert "command completed" in jobs.notices[0]


async def test_parallel_job_starts_reserve_capacity_before_persistence(tmp_path, monkeypatch):
    gate = asyncio.Event()
    admitted = asyncio.Event()

    class SlowBridge(LocalBridge):
        async def call(self, kind, payload, **kwargs):
            if kind == "job_start":
                admitted.set()
                await gate.wait()
            return await super().call(kind, payload, **kwargs)

    async def command(command):
        return "done"

    bash_module = importlib.import_module("plugins.tools.bash")
    monkeypatch.setattr(bash_module.bash, "func", command)
    with Store(tmp_path) as store:
        store.submit("test", request_key="r")
        jobs = Jobs(SlowBridge(store, store.claim("owner")), limit=1)
        bash = jobs.tools()["bash"]
        first = asyncio.create_task(bash.ainvoke({"command": "first"}))
        await admitted.wait()
        assert "capacity reached" in await bash.ainvoke({"command": "second"})
        gate.set()
        await first
        await asyncio.gather(*jobs.tasks.values())
        assert len(jobs.tasks) == 1


async def test_follow_up_consumed_only_after_a_model_response(tmp_path):
    with Store(tmp_path) as store:
        ticket = store.submit("test", request_key="r")
        store.add_input(ticket.id, "new constraint", request_key="followup")
        bridge = LocalBridge(store, store.claim("owner"))
        observer = ExecutionObserver(bridge, Jobs(bridge), 1)
        context = TurnContext(
            turn=1,
            max_turns=5,
            task_id="root",
            role_id="stateful_react",
            ai_text="",
            thinking="",
            tool_calls=[],
            messages=[],
            usage={},
            metadata={},
        )
        injection = await observer.on_turn_end(context)
        assert injection.inject_messages == ["new constraint"]
        assert store.context(ticket.id)["inputs"][0]["consumed_by"] is None
        await observer.on_llm_response(context)
        assert store.context(ticket.id)["inputs"][0]["consumed_by"] is not None


@pytest.mark.parametrize("workflow", ["stateful-react-agent", "agent_team"])
async def test_actual_workflow_runs_through_product_runtime(tmp_path, monkeypatch, workflow):
    """Exercise worker -> WorkflowSession -> DAG -> real loop, with only the LLM mocked."""
    from frontier_agent.infra import llm_adapter

    profiles = importlib.import_module(
        "workflows.stateful_react_agent.profile"
        if workflow == "stateful-react-agent"
        else "workflows.agent_team.profile"
    )
    prefix = "react" if workflow == "stateful-react-agent" else "swarm"

    for name, value in {
        "OPENAI_API_KEY": "test",
        "OPENAI_MODEL": "fixture",
        "SANDBOX_BACKEND": "container",
        "REACT_NO_WEB": "1",
        "LANGUAGE_DETECT_ENABLED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    for name, folder in (
        ("FRONTIER_AGENT_WORKSPACE_DIR", "work"),
        ("FRONTIER_AGENT_OUTPUTS_DIR", "outputs"),
        ("FRONTIER_AGENT_INPUTS_DIR", "inputs"),
    ):
        (tmp_path / folder).mkdir()
        monkeypatch.setenv(name, str(tmp_path / folder))
    llm = ScriptedLLM([LLMResponse(content="Verified workflow output")])
    monkeypatch.setattr(llm_adapter, "create_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr(profiles, f"create_{prefix}_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr(profiles, f"build_{prefix}_model_profile", lambda *args, **kwargs: None)
    original_load = getattr(profiles, f"load_{prefix}_profile")

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        data["agent"].update(
            direct=True,
            reasoning_only_timeout_s=0,
            reasoning_only_max_tokens=0,
            context_compaction="off",
        )
        return data

    monkeypatch.setattr(profiles, f"load_{prefix}_profile", load)
    with Store(tmp_path / "state") as store:
        ticket = store.submit("Answer the task", request_key="r", workflow=workflow)
        assignment = store.claim("owner")
        bridge = LocalBridge(store, assignment)
        await run(store.context(ticket.id), bridge)
        reports = [payload for kind, payload in bridge.seen if kind == "report"]
        assert reports and reports[-1]["complete"]
        assert reports[-1]["result"] == "Verified workflow output"
        assert store.get(ticket.id).calls_used == 1


async def test_waiting_background_job_is_interrupted_by_new_user_input(tmp_path, monkeypatch):
    from frontier_agent import runtime as runtime_module

    bash_module = importlib.import_module("plugins.tools.bash")
    started = asyncio.Event()
    received = asyncio.Event()

    async def long_command(command):
        started.set()
        await asyncio.Event().wait()

    class Runtime:
        def __init__(self, *, extra_tools, **kwargs):
            self.tools = extra_tools
            self.count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, instruction, **kwargs):
            self.count += 1
            if self.count == 1:
                text = await self.tools["bash"].ainvoke({"command": "long job"})
                self.job_id = text.split()[2].rstrip(".")
            elif self.count == 2:
                assert "stop this command" in instruction
                received.set()
                await self.tools["job_cancel"].ainvoke({"job_id": self.job_id})
            return {"final_answer": "done", "stopped_by": "completed"}

    monkeypatch.setattr(bash_module.bash, "func", long_command)
    monkeypatch.setattr(runtime_module, "WorkflowSession", Runtime)
    with Store(tmp_path) as store:
        ticket = store.submit("background work", request_key="r")
        bridge = LocalBridge(store, store.claim("owner"))
        worker = asyncio.create_task(run(store.context(ticket.id), bridge))
        try:
            async with asyncio.timeout(3):
                await started.wait()
                store.add_input(ticket.id, "stop this command", request_key="stop")
                await received.wait()
                await worker
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert not store.unresolved(ticket.id)


async def test_controller_loss_terminates_idle_workflow(tmp_path, monkeypatch):
    from frontier_agent import runtime as runtime_module

    class Runtime:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, *args, **kwargs):
            await asyncio.Event().wait()

    class Disconnected(LocalBridge):
        async def call(self, kind, payload, **kwargs):
            raise asyncio.CancelledError("control channel lost")

    monkeypatch.setattr(runtime_module, "WorkflowSession", Runtime)
    with Store(tmp_path) as store:
        ticket = store.submit("work", request_key="r")
        bridge = Disconnected(store, store.claim("owner"))
        async with asyncio.timeout(4):
            worker = asyncio.create_task(run(store.context(ticket.id), bridge))
            with pytest.raises(asyncio.CancelledError):
                await worker
