import asyncio
import os
import re
from typing import TYPE_CHECKING, Any

from apodex.observers import TerminalObserver
from apodex.profiles import get_profile
from frontier_agent.core.errors import LLMError
from frontier_agent.core.loop_types import LoopConfig, LoopPolicy
from frontier_agent.core.messages import Message, assistant_msg, text_of, user_msg
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
from frontier_agent.core.runtime.session_history import (
    SessionCompactionConfig,
    SessionHistoryCompactor,
    SessionTurn,
    build_session_turn,
    coerce_session_turn,
    render_session_history,
)

if TYPE_CHECKING:
    # Host-session types, for the declaration block on TaskRunnerMixin below.
    # Import-only under TYPE_CHECKING: apodex.session imports this module, so
    # pulling these in at runtime would close a cycle.
    from apodex.changes import WorkspaceJournal
    from apodex.config import ModelConfig, RuntimeConfigStatus
    from apodex.observers import Approver
    from apodex.permissions import PermissionStore
    from apodex.plan import PlanState
    from apodex.render import Renderer
    from apodex.trace import TraceObserver
    from apodex.usage import Usage
    from frontier_agent.core.llm import LLMClient


def _flatten(content: Any) -> str:
    """Flatten LLM message content (str or [{type,text},…]) to a string."""
    return text_of(content)


# Only the wording that actually implicates the *configuration* belongs here:
# this flag decides whether the user is told to go fix a key/model/endpoint, so
# a transient 5xx matching it sends them editing settings that were fine. Every
# alternative is anchored on word boundaries — unanchored `auth` matched
# "authored", and unanchored `dns` matched provider request ids like
# "req_8fdns2kx". "endpoint" is deliberately absent: it appears in plenty of
# overload messages, and base_url/connection-refused already cover the real
# endpoint misconfigurations.
_LLM_CONFIGURATION_ERROR_RE = re.compile(
    r"(?:\b(?:400|401|403)\b|\b(?:un)?authenticat(?:ion|ed)\b|"
    r"\b(?:un)?authori[sz](?:ation|ed)\b|\bforbidden\b|"
    r"invalid[_ -]?(?:api[_ -]?)?key|api[_ -]?key|shell_api_error|"
    r"model[_ -]?(?:not[_ -]?found|invalid)|unknown model|base[_ -]?url|"
    r"connection refused|name or service not known|\bdns\b)",
    re.IGNORECASE,
)
_PARTIAL_OUTPUT_LIMIT = 2000

# A top-level run is deliverable only when it reached a real terminal. Rescue
# calls after resource/observer stops can preserve useful prose, but they do
# not retroactively complete the work that the run was still performing.
_COMPLETE_TOP_LEVEL_STOPS = frozenset({
    "", "completed", "final_answer", "submit_report", "workflow_complete",
})
# ``no_tool`` means opposite things on the two paths that reach here, so it is
# never in the shared set above. The generic coding loop runs with
# ``no_tool_behavior="stop"``: a turn without tool calls IS how it answers.
# Workflows run with ``no_tool_behavior="nudge"``, so the same reason means the
# nudge budget ran out mid-task — which ``agent_bus.fan_in`` also classifies as
# INCOMPLETE ("agent stopped without producing a final answer").
_NO_TOOL_STOP = "no_tool"

# These stops can leave a non-empty ``final_content`` that is known to be
# incomplete. Do not let that fragment short-circuit the tool-free final-answer
# rescue in :meth:`TaskRunnerMixin._force_final`.
_FORCE_FINAL_WITH_PARTIAL_STOP_REASONS = frozenset({"response_truncated"})


def _is_complete_run(
    stopped_by: str,
    *,
    answer_status: str = "",
    answer_source: str = "",
    no_tool_is_complete: bool = False,
) -> bool:
    """Fail closed when a workflow says its user-facing answer is partial."""
    stopped_by = stopped_by.strip()
    answer_status = answer_status.strip()
    answer_source = answer_source.strip()
    if answer_status in {"best_effort", "not_found", "incomplete", "partial"}:
        return False
    # A downstream reporter really can finish synthesis after the research
    # phase hit its soft deadline. Its explicit complete status is authoritative.
    if answer_status == "complete" and answer_source == "reporter_llm":
        return True
    # Main-agent workflows may finish with a plain-text assistant turn, which
    # the loop records as ``no_tool``. Once the workflow has explicitly marked
    # that agent-produced answer complete, treat that terminal as successful.
    # Keep this narrower than accepting workflow ``no_tool`` in general: an
    # exhausted no-tool nudge budget can also use the same stop reason.
    if (
        stopped_by == _NO_TOOL_STOP
        and answer_status == "complete"
        and answer_source == "agent"
    ):
        return True
    if no_tool_is_complete and stopped_by == _NO_TOOL_STOP:
        return True
    return stopped_by in _COMPLETE_TOP_LEVEL_STOPS


def _clip(text: str, limit: int = _PARTIAL_OUTPUT_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n… (truncated; the full text is in the log)"


class TaskRunnerMixin:
    # This mixin is designed to be combined with TerminalSession.
    # It assumes the presence of standard TerminalSession attributes.

    if TYPE_CHECKING:
        # Declaration-only mirror of the TerminalSession surface this mixin
        # reaches through ``self``. A mixin has no way to state what its host
        # provides, so a type checker sees every ``self.r`` / ``self.cfg`` here
        # as an unknown attribute (78 errors before this block existed).
        #
        # These are annotations under TYPE_CHECKING, so they create no class
        # attributes and no runtime import cycle — apodex.session imports this
        # module, never the reverse. Keep them in sync with
        # TerminalSession.__init__ in apodex/session.py.
        cfg: ModelConfig
        cwd: str
        r: Renderer
        max_turns: int
        mode: str
        tui_mode: bool
        approver: Approver
        plan_state: PlanState
        rules: PermissionStore
        usage: Usage
        journal: WorkspaceJournal
        tracer: TraceObserver
        session_id: str
        llm: LLMClient
        history: list[Message]
        display_history: list[Message]
        workflow_turns: list[SessionTurn]
        _env_section: str

        def runtime_config_status(self) -> RuntimeConfigStatus: ...
        def _enrich_task(self, task: str) -> str: ...
        def _persist(self) -> None: ...
        async def _on_turn(
            self, turn: int, messages: list[Any], metadata: dict[str, Any],
        ) -> None: ...
        @staticmethod
        def _workflow_display_messages(
            task: str, steps: list[dict[str, Any]], final: str,
        ) -> list[Message]: ...

    def _show_llm_failure(
        self, detail: str = "", reason: str = "", partial: str = "",
    ) -> None:
        """Render an LLM failure as an error, never as a completed report.

        *partial* is text the run genuinely produced before failing. It is shown
        inside the error, so the work is not lost, without the framing that
        would let it pass for a delivered report.
        """
        detail = detail.strip()
        reason = reason.strip()
        partial = partial.strip()
        status = self.runtime_config_status()
        is_configuration_error = bool(
            status.errors or _LLM_CONFIGURATION_ERROR_RE.search(detail)
        )
        kind = "LLM configuration error" if is_configuration_error else "LLM call failed"
        target = f"{status.provider}/{status.model or 'missing model'}"
        lines = [f"Provider/model: {target}"]
        if reason:
            lines.append(f"Reason: {reason}")
        if detail:
            lines.append(f"Provider response: {detail}")
        if is_configuration_error:
            lines.append(
                "Check /config, then correct the API key, model, or endpoint "
                "in the active profile and retry."
            )
        else:
            lines.append("Check /config and the provider connection, then retry.")
        lines.append(
            f"Full log: {os.path.join(os.environ.get('APODEX_RUN_DIR', ''), 'engine.log')}"
        )
        if partial:
            lines.extend(("", "Partial output produced before the failure:", _clip(partial)))
        renderer = getattr(self.r, "llm_failure", None)
        if callable(renderer):
            renderer("\n".join(lines), configuration_error=is_configuration_error)
        else:
            # Compatibility for custom renderers implementing the older sink API.
            self.r.error(f"{kind}\n" + "\n".join(lines))

    def _show_incomplete_run(
        self,
        text: str,
        *,
        stopped_by: str,
        turns: int = 0,
        tool_calls: int = 0,
    ) -> None:
        """Keep partial work visible without promoting it to a deliverable."""
        renderer = getattr(self.r, "incomplete", None)
        if callable(renderer):
            renderer(
                text,
                turns=turns,
                tool_calls=tool_calls,
                stopped_by=stopped_by,
            )
            return
        self.r.error(
            f"run incomplete ({stopped_by or 'unknown stop'}); "
            "partial output was not saved as a final report:\n"
            + _clip(text)
        )

    async def _render_changed_files(self) -> None:
        """Render the changed-files summary for the task that just finished.

        Revertable paths only: the panel is titled "``/revert`` to undo", and
        the journal also carries every path a tree scan merely observed — one
        ``make`` or ``cargo build`` puts thousands of build outputs there that
        ``/revert`` deliberately leaves alone.

        Off the event loop, like the TUI's own polling: this re-reads every
        journalled file and builds the whole unified diff to keep the stats off
        the front of it.
        """
        stats = await asyncio.to_thread(self.journal.revertable_diffstat)
        self.r.changes(stats)

    async def run_task(self, task: str) -> None:
        profile = get_profile(self.mode)
        if profile.workflow:
            await self._run_native_workflow(task, profile)
            return

        # Fresh plan per task — the todo store is process-global, so clear any
        # leftover checklist from a previous task (it overwrites on todo_write,
        # but an un-planned task should not show the prior task's plan).
        from apodex.todo import clear_todos
        clear_todos()
        if task.strip():  # a genuinely new task re-arms the compact-and-resume guard
            self._compact_retried = False

        # The active profile supplies the prompt / tools / robustness observers.
        # The session always owns the UI+approval observer (TerminalObserver).
        tools = profile.tools()
        # Skills load full SKILL.md via read_text; make sure it's callable even
        # if the profile's tools list didn't include it explicitly.
        if profile.skills and not any(getattr(t, "name", "") == "read_text" for t in tools):
            from apodex.agent_tools import terminal_tool_registry
            rt = terminal_tool_registry().get("read_text")
            if rt is not None:
                tools = [*tools, rt]
        tool_names = [getattr(t, "name", "") for t in tools]
        # Type-ahead steering: collect lines the user types mid-run; the
        # observer injects them at a turn boundary, and any leftover runs as a
        # follow-up below. Only active on an interactive TTY.
        from apodex.steer import SteerInbox
        from apodex.usage import UsageObserver
        from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
        inbox = SteerInbox(self.r)
        observer = TerminalObserver(
            self.r, self.approver, self.cwd,
            journal=self.journal, plan_state=self.plan_state, steer_inbox=inbox,
            rules=self.rules,
        )
        observers = [observer, UsageObserver(self.usage, tools=tools), self.tracer,
                     *profile.extra_observers(tool_names)]
        config = LoopConfig(
            max_turns=self.max_turns,
            role_id=f"{self.mode}_agent",
            tool_timeout=180,
            llm_timeout=180,
            tool_result_max_chars=16_000,
            # Context management (reuses the engine's pieces): summarise old
            # turns with the LLM instead of the crude string-slice default, and
            # turn on the hard-overflow guard so a big tool result yields a clean
            # ``context_limit_reached`` stop (→ compact-and-resume below) rather
            # than an ``llm_error`` cliff.
            compactor=LLMSummaryCompactor(summary_llm=self.llm),
            context_overflow_guard=True,
            max_context_length=self.cfg.context_window,
            context_token_limit=int(self.cfg.context_window * 0.8),
            max_completion_tokens=self.cfg.max_tokens,
            # A plain-text turn (no tool call) is the agent's final answer.
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        )
        # Multi-turn: run_agent_loop IGNORES ``system_prompt`` when
        # ``initial_messages`` is given (it does ``messages = list(initial_messages)``
        # then appends the new user turn — see agent_loop.py). So we must pass
        # the FULL prior history, which already starts with the SystemMessage
        # from turn 1's result.messages — do NOT strip it, or the model loses
        # its system prompt on every follow-up turn.
        system_prompt = profile.system_prompt(self.cwd) + "\n\n# Environment\n" + self._env_section
        if self.plan_state.active:
            from apodex.plan import PLAN_MODE_PROMPT
            system_prompt = system_prompt + "\n\n" + PLAN_MODE_PROMPT

        # Line mode watches stdin via add_reader; the TUI owns the terminal and
        # feeds ``inbox`` from its own input box, so skip attach there.
        self._inbox = inbox
        if not self.tui_mode:
            inbox.attach()       # start watching stdin (no-op without a TTY)
        self.approver.inbox = inbox
        result = None
        status = "ok"
        try:
            import sys
            loop_fn = getattr(sys.modules.get("apodex.session"), "run_agent_loop", run_agent_loop)
            result = await loop_fn(
                system_prompt=system_prompt,
                user_message=self._enrich_task(task),
                llm=self.llm,
                tools=tools,
                config=config,
                observers=observers,
                initial_messages=self.history or None,
                # Persist after every turn so a mid-task Ctrl-C can still be
                # resumed from the last completed turn (not just last task).
                on_turn_complete=self._on_turn,
                # Scope the file-tool workspace to the cwd for this run too
                # (belt-and-braces with the CODING_WORKSPACE_ROOT env var).
                scope_metadata={
                    "coding_workspace_root": self.cwd,
                    "workspace_root": self.cwd,
                },
            )
        except KeyboardInterrupt:
            status = "interrupted"  # partial output stays in the scrollback
        except LLMError as exc:  # a genuine LLM/provider failure escaped the loop
            status = "error"
            # Type-checked: an LLMError IS an LLM failure (no string-sniffing to
            # decide). Show the provider's own message (last_exc) — it's already
            # actionable — tagged with the short reason.
            reason = getattr(exc, "reason", "") or ""
            last_exc = getattr(exc, "last_exc", None)
            detail = str(last_exc if last_exc is not None else exc)
            self.r.error(f"✗ LLM call failed ({reason}):\n  {detail}" if reason
                         else f"✗ LLM call failed:\n  {detail}")
        except Exception as exc:  # never let one task kill the REPL
            status = "error"
            # Not an LLMError → a tool bug / loop bug. Stay generic so it isn't
            # mislabeled an LLM failure.
            self.r.error(f"agent loop failed: {exc}")
        finally:
            inbox.detach()
            self.approver.inbox = None
            self._inbox = None

        # Lines the user typed during the run that weren't injected live (the
        # finishing turn, or after it) — run them next so nothing is lost.
        leftover = inbox.drain()

        if status == "ok" and result is not None:
            self.history = list(result.messages)
            self.display_history = list(self.history)
            if (result.stopped_by == "context_limit_reached"
                    and not self._compact_retried):
                # Context filled up → summarise old turns and resume the task,
                # instead of ending it. (One retry per task; guarded above.)
                from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
                self._compact_retried = True
                self.r.note("↻ context was full — compacting earlier turns and resuming")
                self.history = await LLMSummaryCompactor(summary_llm=self.llm).compact(
                    self.history, keep_recent=16,
                )
                self.usage.compactions += 1
                await self.run_task("")  # resume on the compacted history
                return
            if result.stopped_by == "user_rejected":
                # User declined an action → stop cleanly and hand control back
                # (don't force a summary answer; that would just keep talking).
                self.r.note(
                    "■ stopped — you declined that action. "
                    "Tell me how to proceed differently, or rephrase the task."
                )
            elif result.stopped_by == "llm_error":
                # The LLM was unreachable/rejected. Surface a clear, actionable
                # error instead of a cryptic footer — and DON'T call _force_final
                # (its rescue LLM call would just hit the same failure).
                detail = str(result.metadata.get("llm_error") or "").strip()
                reason = str(result.metadata.get("llm_error_reason") or "").strip()
                # Unlike the workflows, this path has no deterministic fallback
                # prose: a non-empty final_content here is text the model really
                # produced before the endpoint died, so it is worth keeping.
                self._show_llm_failure(
                    detail, reason, partial=str(result.final_content or ""),
                )
            else:
                final = await self._force_final(result)
                # ``no_tool_behavior="stop"`` above: a plain-text turn is this
                # loop's normal finish, not a truncation.
                complete = _is_complete_run(
                    result.stopped_by, no_tool_is_complete=True,
                )
                render = self.r.final if complete else None
                if render is not None:
                    render(
                        final,
                        turns=result.turns_used,
                        tool_calls=result.tool_calls_count,
                        stopped_by=result.stopped_by,
                    )
                else:
                    self._show_incomplete_run(
                        final,
                        turns=result.turns_used,
                        tool_calls=result.tool_calls_count,
                        stopped_by=result.stopped_by,
                    )
            # Deterministic changed-files summary (from the journal, not the
            # model) — satisfies "final output includes changed files". ``/revert``
            # undoes them; trace is at self.trace_path.
            await self._render_changed_files()
            self._persist()
        elif status == "interrupted":
            self.r.note("■ interrupted" + (
                " — running what you typed" if leftover else ""))

        # Follow-up steering: run anything still queued (unless the task errored).
        if leftover and status != "error":
            await self.run_task("\n".join(leftover))

    async def _run_native_workflow(self, task: str, profile: Any) -> None:
        """Run one of the shipped workflow DAGs from the terminal.

        ``BenchmarkSession`` is the project's small, scoped runtime bootstrap:
        it registers the AgentBus, workflow roles, and the full tool registry,
        then restores the caller's registry after the task.  This is important
        for Agent Team: merely placing its tools in the generic terminal loop
        would make it *look* like a team while never creating sub-agents.

        Steering works here exactly as it does in the generic loop: the
        workflow's main agent appends ``sdk_extra_observers`` to its own
        ``run_agent_loop`` observers, so our ``TerminalObserver`` sees
        ``on_turn_end`` and can inject queued lines as the coordinator's next
        user message. Sub-agents already dispatched are NOT interrupted — a
        steer reaches them only through the coordinator's next delegation.
        """
        from apodex.steer import SteerInbox
        from apodex.todo import clear_todos
        from benchmarks.public.core.kernel_adapter import BenchmarkSession

        clear_todos()
        workflow_profile = profile.workflow_profile
        if not workflow_profile:
            self.r.error(f"mode {self.mode!r} is missing its workflow profile")
            return

        inbox = SteerInbox(self.r)
        observer = TerminalObserver(
            self.r, self.approver, self.cwd,
            journal=self.journal, plan_state=self.plan_state,
            rules=self.rules, steer_inbox=inbox,
        )
        from apodex.usage import UsageObserver
        usage_observer = UsageObserver(self.usage)
        metadata = {
            "profile": workflow_profile,
            "coding_workspace_root": self.cwd,
            "sdk_extra_observers": [observer, usage_observer, self.tracer],
            # Stable across workflow executions; ``turn_index`` advances
            # within it. Workflows use this for upstream LLM session affinity.
            "session_id": self.session_id,
            "turn_index": len(self.workflow_turns) + 1,
        }
        self.r.note(
            f"workflow → {profile.workflow} · profile → {workflow_profile}"
        )
        # Line mode watches stdin via add_reader; the TUI owns the terminal and
        # feeds ``inbox`` from its own input box, so skip attach there.
        self._inbox = inbox
        if not self.tui_mode:
            inbox.attach()       # start watching stdin (no-op without a TTY)
        self.approver.inbox = inbox
        current_query = self._enrich_task(task)
        compaction = await SessionHistoryCompactor(
            summary_llm=self.llm,
            config=SessionCompactionConfig(
                context_window=self.cfg.context_window,
                max_completion_tokens=self.cfg.max_tokens,
            ),
        ).compact(self.workflow_turns, current_query)
        if compaction.changed:
            self.workflow_turns = compaction.turns
            self._persist()
            actions = []
            if compaction.tool_results_removed:
                actions.append("tool results removed")
            if compaction.summarized:
                actions.append("older turns summarized")
            self.r.note(
                "↻ session context auto-compacted "
                f"({compaction.before_tokens:,} → "
                f"{compaction.after_tokens:,} tokens; "
                f"{', '.join(actions) or 'oldest turns trimmed'})"
            )
        workflow_input = render_session_history(
            compaction.turns, current_query,
        )
        state: dict | None = None
        status = "ok"
        try:
            async with BenchmarkSession() as runtime:
                state = await runtime.run(
                    workflow_input,
                    meta=metadata,
                    pipeline_id=profile.workflow,
                    extra_input={"current_query": current_query},
                )
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "error"
            self.r.error(f"{profile.workflow} workflow failed: {exc}")
        finally:
            inbox.detach()
            self.approver.inbox = None
            self._inbox = None

        # Lines typed during the run that weren't injected live (the finishing
        # turn, or after it) — run them next so nothing is lost.
        leftover = inbox.drain()
        if status == "interrupted":
            self.r.note("■ interrupted" + (
                " — running what you typed" if leftover else ""))
        if status != "ok":
            # As in the generic loop: a genuine failure discards the queue
            # (re-running into a broken workflow just repeats the error).
            if leftover and status != "error":
                await self.run_task("\n".join(leftover))
            return

        state = state or {}
        final = str(state.get("final_answer") or state.get("final_content") or "").strip()
        if not final:
            final = "(the workflow finished without a final answer)"
        stopped_by = str(state.get("stopped_by") or "workflow_complete")
        if stopped_by == "llm_error":
            # ``answer_status`` (see stateful_react_agent/nodes/main_agent.py)
            # separates the workflow's deterministic placeholder prose
            # (``not_found``) from an answer its salvage call really recovered
            # (``best_effort``). Only the latter is content worth keeping, and
            # stopped_by alone cannot tell them apart.
            salvaged = (
                final if str(state.get("answer_status") or "") == "best_effort" else ""
            )
            self._show_llm_failure(
                str(state.get("llm_error") or ""),
                str(state.get("llm_error_reason") or ""),
                partial=salvaged,
            )
            await self._render_changed_files()
            self._persist()
            if leftover:
                # As elsewhere, a genuine failure discards the queue — but say
                # so, or the lines the user typed vanish without a trace.
                self.r.note("■ queued input discarded — fix the LLM error and retype it")
            return
        complete = _is_complete_run(
            stopped_by,
            answer_status=str(state.get("answer_status") or ""),
            answer_source=str(state.get("final_answer_source") or ""),
        )
        turns_used = (
            int(state.get("turns_used") or 0)
            if "turns_used" in state
            else len(state.get("react_steps") or [])
        )
        tool_calls_count = (
            int(state.get("tool_calls_count") or 0)
            if "tool_calls_count" in state
            else 0
        )
        if complete:
            self.r.final(
                final,
                turns=turns_used,
                tool_calls=tool_calls_count,
                stopped_by=stopped_by,
            )
        else:
            self._show_incomplete_run(
                final,
                turns=turns_used,
                tool_calls=tool_calls_count,
                stopped_by=stopped_by,
            )
        await self._render_changed_files()
        # Native workflow internals are task-scoped, but the terminal session
        # is not: retain a compact user/final-answer pair so /resume has useful
        # transcript history instead of an opaque 0-message checkpoint.
        self.history.extend((user_msg(task), assistant_msg(final)))
        session_turn = coerce_session_turn(state.get("session_turn"))
        if session_turn is None:
            # Compatibility with older/out-of-tree workflows that have not yet
            # adopted the framework's normalized session-turn output.
            session_turn = build_session_turn(
                current_query,
                state.get("session_messages") or [],
                final,
                steps=state.get("react_steps") or [],
            )
        self.workflow_turns.append(session_turn)
        self.display_history.extend(self._workflow_display_messages(
            task, state.get("react_steps") or [], final,
        ))
        self._persist()

        # Follow-up steering: anything still queued runs as the next task.
        if leftover:
            await self.run_task("\n".join(leftover))

    async def _force_final(self, result: Any) -> str:
        """If the loop ended without a usable answer, do one tool-free LLM call.

        Most non-empty final content is already usable, but a response truncated
        at the output cap is explicitly known to be an unfinished fragment and
        must still go through the rescue call.
        """
        final = (result.final_content or "").strip()
        stopped_by = result.stopped_by or ""
        # Clean exits and resumable pauses keep their content as-is.
        if stopped_by in ("", "no_tool", "paused", "user_rejected"):
            return result.final_content
        if final and stopped_by not in _FORCE_FINAL_WITH_PARTIAL_STOP_REASONS:
            return result.final_content
        try:
            msgs = list(result.messages)
            # Drop a dangling assistant turn that ended in unanswered tool calls,
            # so the plain-text nudge doesn't follow an assistant tool_call with
            # no tool results (which some providers reject).
            while msgs and msgs[-1].get("tool_calls"):
                msgs.pop()
            msgs.append(user_msg(
                "Provide your best final answer now based on everything "
                "gathered, as plain text. Do not call any tools.",
            ))
            resp = await asyncio.wait_for(self.llm.chat(msgs), timeout=120)
            text = _flatten(getattr(resp, "content", "")).strip()
            return text or result.final_content or "(no answer produced)"
        except Exception:
            return result.final_content or "(no answer produced)"
