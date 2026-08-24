from __future__ import annotations

import asyncio
import json

from frontier_agent.components.agent_bus import SubAgentResult
from frontier_agent.components.observers.task_board import TaskBoardObserver
from frontier_agent.components.observers.trajectory import TrajectoryFileObserver
from frontier_agent.core.loop_types import LoopConfig, TurnContext
from plugins.tools.task_board import build_task_board_observer
from frontier_agent.components.agent_bus import fan_in


class _NativeTool:
    name = "search"
    description = "Search"
    parameters = {"type": "object", "properties": {"q": {"type": "string"}}}

    def __init__(self) -> None:
        self.schema_calls = 0

    def to_openai_schema(self) -> dict:
        self.schema_calls += 1
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def test_agent_team_trajectory_options_preserve_minimal_schema_and_env(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("FRONTIER_AGENT_TRAJECTORY_FORMATS", "json")
    monkeypatch.setenv("SWARM_TRAJECTORY_FORMATS", "jsonl")
    tool = _NativeTool()
    observer = TrajectoryFileObserver(
        tmp_path,
        filename="agent",
        tools=[tool],
        format_env_vars=("SWARM_TRAJECTORY_FORMATS",),
        tool_schema_detail="minimal",
        include_start_tool_names=False,
    )

    assert observer._formats == {"jsonl"}
    assert observer._tools_schema == [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search",
            "parameters": {},
        },
    }]
    assert tool.schema_calls == 0

    asyncio.run(observer.on_loop_start(LoopConfig(
        task_id="task", role_id="role", max_turns=2,
    )))
    observer._jsonl_handle.close()
    record = json.loads((tmp_path / "agent.jsonl").read_text().splitlines()[0])
    assert "tool_names" not in record
    assert record["t"] == "start"


def test_shared_report_formatting() -> None:
    failure = SubAgentResult(
        question="q",
        role_id="researcher",
        final_content="partial",
        success=False,
        error="boom",
        error_class="ProviderError",
    )
    block = fan_in.format_report_block("worker", failure)
    assert "agent failed mid-task (ProviderError: boom)" in block

    repeated = SubAgentResult(
        question="q",
        role_id="researcher",
        final_content="partial",
        success=True,
        metadata={"stopped_by": "cross_turn_repetition"},
    )
    assert "stopped after repeating itself" in fan_in.format_report_block(
        "worker", repeated,
    )
    assert "cross_turn_repetition" in fan_in.INCOMPLETE_STOP_REASONS


def test_shared_task_board_observer_preserves_reminder_cooldown() -> None:
    board_count = 0
    rendered: list[tuple[str, str | None]] = []
    observer = TaskBoardObserver(
        board_size=lambda _task_id: board_count,
        render_board=lambda task_id, bus_task_id: (
            rendered.append((task_id, bus_task_id)) or "board"
        ),
        resolve_bus_task_id=lambda _scope: "bus-task",
    )

    def context(turn: int) -> TurnContext:
        return TurnContext(
            turn=turn,
            max_turns=10,
            task_id="task",
            role_id="role",
            ai_text="",
            thinking="",
            tool_calls=[],
            messages=[],
            usage=None,
            metadata={},
        )

    assert asyncio.run(observer.on_turn_end(context(1))) is None
    board_count = 1
    first = asyncio.run(observer.on_turn_end(context(1)))
    assert first is not None
    assert first.inject_messages == [
        "Current task board (keep it current via add_task / update_task; "
        "finalize only once every task is resolved):\nboard"
    ]
    assert asyncio.run(observer.on_turn_end(context(5))) is None
    assert asyncio.run(observer.on_turn_end(context(6))) is not None
    assert rendered == [("task", None), ("task", None)]
    assert isinstance(build_task_board_observer(), TaskBoardObserver)
