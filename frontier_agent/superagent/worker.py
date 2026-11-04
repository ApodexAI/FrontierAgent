"""Container worker. Acknowledged stdio RPC keeps state outside the execution sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from typing import TextIO

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    Intervention,
    LLMAttemptContext,
    ToolCallIntervention,
    ToolResult,
    TurnContext,
)
from frontier_agent.core.tool import Tool, tool
from frontier_agent.superagent.models import Json


class Bridge:
    """One request at a time; losing the controller cancels execution, never fails open."""

    def __init__(self, reader: asyncio.StreamReader, output: TextIO) -> None:
        self.reader = reader
        self.output = output
        self.lock = asyncio.Lock()

    async def call(self, kind: str, payload: Json, *, key: str | None = None) -> Json:
        async with self.lock:
            key = key or uuid.uuid4().hex
            try:
                self.output.write(json.dumps({"key": key, "kind": kind, "payload": payload}) + "\n")
                self.output.flush()
                raw = await asyncio.wait_for(self.reader.readline(), timeout=30)
                response = json.loads(raw)
                if response.get("key") != key or not response.get("ok"):
                    raise ValueError(response.get("error", "Invalid control response"))
                return response.get("payload", {})
            except Exception as exc:
                # Observer exceptions derived from Exception are intentionally isolated by
                # AgentCore. Cancellation must propagate across that boundary to fail closed.
                raise asyncio.CancelledError(f"Controller unavailable: {exc}") from exc


class Jobs:
    """Background shell commands use the existing sandbox/policy tool implementation."""

    def __init__(self, bridge: Bridge, *, limit: int = 4) -> None:
        self.bridge = bridge
        self.limit = limit
        self.starting = 0
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.results: dict[str, str] = {}
        self.notices: list[str] = []

    @property
    def pending(self) -> bool:
        return any(not task.done() for task in self.tasks.values())

    def tools(self) -> dict[str, Tool]:
        @tool
        async def bash(command: str) -> str:
            """Start a sandboxed shell job and return its ID immediately. Use job_status or job_cancel.

            Args:
                command: Shell command to execute in this task's sandbox.
            """
            if self.starting + sum(not task.done() for task in self.tasks.values()) >= self.limit:
                return "Job capacity reached. Check or cancel an existing job."
            job_id = uuid.uuid4().hex
            self.starting += 1
            try:
                await self.bridge.call("job_start", {"operation_id": job_id, "command": command})
            finally:
                self.starting -= 1

            async def run() -> None:
                from plugins.tools.bash import bash as sandbox_bash

                result = "Interrupted before a confirmed result; inspect existing work."
                try:
                    result = str(await sandbox_bash.ainvoke({"command": command}))
                except asyncio.CancelledError:
                    result = "Cancelled; inspect files before retrying any partial operation."
                    raise
                except Exception as exc:
                    result = f"Job failed: {type(exc).__name__}: {exc}"
                finally:
                    # Retain a bounded preview; full file artifacts live in /outputs.
                    self.results[job_id] = result[-16000:]
                    await self.bridge.call(
                        "job_end", {"operation_id": job_id, "result": self.results[job_id]}
                    )
                    self.notices.append(f"Job {job_id} ended: {self.results[job_id]}")

            self.tasks[job_id] = asyncio.create_task(run())
            return f"Started job {job_id}. Continue other work; completion will arrive as an event."

        @tool
        async def job_status(job_id: str) -> str:
            """Read a background job's state and bounded result without blocking."""
            if job_id in self.results:
                return self.results[job_id]
            return "Running" if job_id in self.tasks else "Unknown job"

        @tool
        async def job_cancel(job_id: str) -> str:
            """Cancel an active background command and its subprocess tree."""
            task = self.tasks.get(job_id)
            if task is None:
                return "Unknown job"
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return self.results.get(job_id, "Cancelled")

        return {item.name: item for item in (bash, job_status, job_cancel)}

    async def close(self) -> None:
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)


class ExecutionObserver(BaseObserver):
    """Fence tools and persist recovery context for main and child loops."""

    critical = True

    def __init__(self, bridge: Bridge, jobs: Jobs, revision: int) -> None:
        self.bridge = bridge
        self.jobs = jobs
        self.revision = revision
        self.calls: dict[tuple[str, str], str] = {}
        self.input_seqs: list[int] = []

    async def on_llm_attempt(self, ctx: LLMAttemptContext) -> Intervention | None:
        if ctx.phase == "started":
            await self.bridge.call(
                "model_call",
                {"actor": ctx.task_id, "attempt_id": ctx.attempt_id},
                key=f"model:{ctx.task_id}:{ctx.role_id}:{ctx.attempt_id}",
            )
        return None

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        usage = ctx.usage or {}
        await self.bridge.call(
            "usage",
            {
                "tokens": int(usage.get("prompt_tokens", 0) or 0)
                + int(usage.get("completion_tokens", 0) or 0)
            },
        )
        if ctx.role_id in {"stateful_react", "swarm_main"} and self.input_seqs:
            await self.bridge.call(
                "checkpoint",
                {
                    "actor": f"{ctx.task_id}:{ctx.role_id}",
                    "turn": ctx.turn,
                    "messages": ctx.messages[-20:],
                },
            )
            await self.bridge.call("consume", {"seqs": self.input_seqs})
            self.input_seqs = []
        return None

    async def on_tool_call(self, ctx: TurnContext, tool_call: Json) -> ToolCallIntervention | None:
        op = uuid.uuid4().hex
        self.calls[(ctx.task_id, str(tool_call.get("id", "")))] = op
        await self.bridge.call(
            "tool_start", {"operation_id": op, "actor": ctx.task_id, "call": tool_call}
        )
        return None

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> ToolResult | None:
        op = self.calls.pop((ctx.task_id, result.tool_call_id), None)
        if op:
            await self.bridge.call(
                "tool_end", {"operation_id": op, "result": str(result.result)[-16000:]}
            )
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        # This is recovery material, not a wire-protocol replay checkpoint. Keep
        # the recent window bounded; authoritative user inputs remain in the DB.
        await self.bridge.call(
            "checkpoint",
            {
                "actor": f"{ctx.task_id}:{ctx.role_id}",
                "turn": ctx.turn,
                "messages": ctx.messages[-20:],
            },
        )
        if ctx.role_id not in {"stateful_react", "swarm_main"}:
            return None
        incoming = await self.bridge.call("inputs", {})
        messages = [str(item["text"]) for item in incoming["inputs"]]
        self.input_seqs = [int(item["seq"]) for item in incoming["inputs"]]
        self.revision = int(incoming["revision"])
        messages.extend(self.jobs.notices)
        self.jobs.notices.clear()
        return Intervention(inject_messages=messages) if messages else None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        # Per-turn actor checkpoints above remain the recovery source. The final
        # task report is emitted only after the workflow and all jobs settle.
        return None


async def run(context: Json, bridge: Bridge) -> None:
    from frontier_agent.core.runtime.observer_scope import observe_execution
    from frontier_agent.runtime import WorkflowSession

    ticket = context["ticket"]
    budget = ticket["budget"]
    jobs = Jobs(bridge, limit=int(budget["parallel_agents"]))
    observer = ExecutionObserver(bridge, jobs, int(ticket["revision"]))
    observer.input_seqs = [
        int(item["seq"]) for item in context["inputs"] if item["consumed_by"] is None
    ]
    workflow = ticket["workflow"]
    if workflow == "stateful-react-agent":
        from workflows.stateful_react_agent.profile import load_react_profile

        profile = load_react_profile("tui")
        tool_keys = ("agent_tools",)
    else:
        from workflows.agent_team.profile import load_swarm_profile

        profile = load_swarm_profile("tui")
        tool_keys = ("main_agent_tools", "sub_agent_tools")
    agent = dict(profile["agent"])
    for key in tool_keys:
        agent[key] = list(dict.fromkeys([*agent.get(key, []), "job_status", "job_cancel"]))
    # Recovery is a new workflow run informed by persisted context and artifacts,
    # not a blind re-execution of previous tool calls or a replay of child sessions.
    instruction = (
        "Complete the durable task below in this isolated workspace. Inspect existing work "
        "and recorded operation outcomes before doing anything again. Do not repeat an "
        "already successful external action. bash starts a background job; use job_status "
        "or job_cancel, and wait for completion events before claiming success. "
        "Write final deliverables under /outputs. Treat recovered context as historical "
        "evidence; follow the goal and all current user constraints.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    remaining = max(1, int(budget["model_calls"]) - int(ticket["calls_used"]))
    agent.update(
        main_max_turns=remaining,
        sub_max_turns=remaining,
        reporter=False,
        planning_mode=False,
        salvage_infra_errors=False,
    )
    metadata = {
        "profile": "tui",
        "session_id": ticket["id"],
        "fs_mode": True,
        "profile_overrides": {"agent": agent},
        "reporter": False,
    }
    parent = asyncio.current_task()
    closing = False

    async def monitor_controller() -> None:
        try:
            while True:
                await asyncio.sleep(2)
                await bridge.call("heartbeat", {})
        except asyncio.CancelledError:
            if not closing and parent is not None and not parent.cancelling():
                parent.cancel()
            raise

    monitor = asyncio.create_task(monitor_controller())
    try:
        with observe_execution([observer]):
            async with WorkflowSession(
                extra_tools=jobs.tools(), max_parallel=int(budget["parallel_agents"])
            ) as runtime:
                while True:
                    state = await runtime.run(instruction, meta=metadata, pipeline_id=workflow)
                    if jobs.pending or jobs.notices:
                        # A workflow may finish its turn while a shell job is still
                        # running. Human input must be able to wake it before that
                        # command completes, and every completion must be reviewed.
                        while jobs.pending and not jobs.notices:
                            await asyncio.wait(
                                [task for task in jobs.tasks.values() if not task.done()],
                                timeout=0.5,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            incoming = await bridge.call("inputs", {})
                            if incoming["inputs"]:
                                observer.input_seqs = [
                                    int(item["seq"]) for item in incoming["inputs"]
                                ]
                                observer.revision = int(incoming["revision"])
                                instruction += "\nNew user input:\n" + "\n".join(
                                    str(item["text"]) for item in incoming["inputs"]
                                )
                                break
                        instruction += "\nBackground work update:\n" + "\n".join(jobs.notices)
                        instruction += "\nActive jobs: " + ", ".join(
                            key for key, task in jobs.tasks.items() if not task.done()
                        )
                        jobs.notices.clear()
                        instruction += "\nPrevious workflow response:\n" + str(
                            state.get("final_answer", "")
                        )
                        continue
                    final = str(state.get("final_answer") or state.get("final_content") or "")
                    stopped = str(state.get("stopped_by") or "workflow_complete")
                    status = str(state.get("answer_status") or "")
                    complete = (
                        bool(final.strip())
                        and status not in {"best_effort", "not_found", "incomplete", "partial"}
                        and (
                            stopped
                            in {"completed", "final_answer", "submit_report", "workflow_complete"}
                            or (stopped == "no_tool" and status == "complete")
                        )
                    )
                    await bridge.call(
                        "report",
                        {
                            "result": final,
                            "complete": complete,
                            "reason": stopped,
                            "revision": observer.revision,
                        },
                    )
                    break
    finally:
        # Mark normal monitor shutdown explicitly so it does not cancel its parent.
        closing = True
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor
        await jobs.close()


async def _main(output: TextIO) -> None:
    reader = asyncio.StreamReader(limit=8 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    context = json.loads(await reader.readline())
    await run(context, Bridge(reader, output))


def main() -> None:
    """Reserve stdout for controller RPC; workflow diagnostics go to stderr."""
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        asyncio.run(_main(output))
    except asyncio.CancelledError:
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
