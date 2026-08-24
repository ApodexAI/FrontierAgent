"""Observers bridging ``run_agent_loop`` to the terminal UI.

A single :class:`TerminalObserver` (critical → awaited in order) handles
everything the TUI needs:

* ``on_llm_delta``    → stream assistant content + thinking token-by-token
* ``on_llm_response`` → fallback render if the provider didn't stream
* ``on_tool_call``    → render the proposed call + **human-approval gate**
  (returns ``ToolCallIntervention(skip_with_result=...)`` to veto)
* ``on_tool_result``  → render tool output / diff
* ``on_loop_end``     → nothing (final answer rendered by the session)

The line-mode approval gate reads stdin synchronously. The agent is already
paused at this point and the full-screen TUI uses its own modal approver, so a
background input thread adds no useful concurrency and can prevent Python
from exiting when a terminal disappears mid-prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import sys
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from apodex import fsguard
from apodex.agent_tools import (
    MUTATING_TOOLS,
    RISK_DENY,
    RISK_SAFE,
    assess_with_rules,
    is_mutating_tool,
    localize_path_args,
)
from apodex.diff_preview import change_stats, unified_diff
from apodex.render import Renderer
from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    Intervention,
    LLMDeltaContext,
    LoopConfig,
    ToolCallIntervention,
    ToolResult,
    TurnContext,
)

# Tools whose proposed change we preview as a diff before approval.
_DIFF_TOOLS = frozenset({"write_file", "file_editor_create", "file_editor_str_replace"})
#: Longest target rendered inline in an approval prompt before eliding.
_MAX_TARGET_CHARS = 72


def _target_suffix(name: str, target: str) -> str:
    """``" <target>"`` for the approval prompt, or ``""`` when it adds nothing.

    The gate computes a destination for every write, but neither prompt used to
    show it — so the question named the tool and not the file. ``bash`` is the
    exception: its target is the command, which the call line and the preview
    already carry verbatim.
    """
    if name == "bash" or not target:
        return ""
    shown = target if len(target) <= _MAX_TARGET_CHARS else f"…{target[-_MAX_TARGET_CHARS:]}"
    return f" {shown}"
# Tools whose success means the file is now "seen" (read-before-edit): reads,
# and writes — creating/editing a file means you know its content, so a
# follow-up edit shouldn't require a separate read.
_SEEN_TOOLS = frozenset({
    "read_file", "file_editor_view",
    "write_file", "file_editor_create", "file_editor_str_replace",
})
# How many assistant turns of no TodoWrite before we re-inject the plan reminder.
_TODO_REMINDER_TURNS = 10

# Synthetic tool-results we inject for blocked/rejected/redirected calls. The
# approval note already surfaces these inline, so ``on_tool_result`` skips
# re-rendering them (a ``✓ tool`` panel would misleadingly imply success).
_SYNTHETIC_RESULT_MARKERS = (
    "[user rejected this ",
    "[The user declined",
    "[blocked by safety policy",
)


@dataclass
class Decision:
    """Outcome of an approval prompt. ``feedback`` carries a free-text
    instruction the user typed instead of a yes/no (human-in-the-loop)."""

    approved: bool
    feedback: str = ""
    remember: bool = False  # 'always allow this command' → persist an allow rule


def _read_single_key() -> str | None:
    """Read one keypress without Enter (POSIX raw mode). ``None`` if no TTY /
    not supported (Windows, piped stdin) so the caller can fall back to input().
    """
    if not sys.stdin.isatty():
        return None
    try:
        import termios
        import tty
    except Exception:
        return None
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        # Restore cooked mode even if read raised; guard the restore itself so
        # a tcsetattr failure can't leave the terminal stuck in raw mode.
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    # Ctrl-C / Ctrl-D in raw mode arrive as bytes, not exceptions → treat as cancel.
    if ch in ("\x03", "\x04"):
        return "n"
    return ch


class Approver:
    """Single-key y/n/a approval prompt (graceful in non-interactive runs).

    The proposed tool call + its diff are already shown by the renderer; this
    just collects the decision. One keypress (no Enter) when a TTY is
    available; falls back to a line read otherwise, and fail-closed (reject)
    when there is no way to ask.
    """

    def __init__(
        self, *, auto_approve: bool = False, auto_for_me: bool = False, interactive: bool = True,
    ) -> None:
        self.auto_approve = auto_approve
        self.auto_for_me = auto_for_me
        self.interactive = interactive
        self.inbox: Any = None  # SteerInbox | None — paused while we own stdin

    async def confirm(
        self, name: str, target: str, reason: str, *, dangerous: str = "",
        preview: str = "", preview_kind: str = "",
    ) -> Decision:
        """Ask the human to approve a tool call.

        Returns a :class:`Decision`. Besides yes/no/all, the user may pick
        ``[e]`` (or just type a sentence) to **redirect** — that free text is
        fed back to the agent as the (declined) call's result so it can adjust.
        ``a`` flips ``auto_approve`` for the rest of the session.

        When ``dangerous`` is set (delete / dep-install / destructive shell),
        a single keypress is NOT enough — the user must type the full word
        ``yes`` (a deliberate second confirmation).

        ``preview`` / ``preview_kind`` (shared with :class:`TuiApprover`) are
        accepted for signature parity but unused here — line mode already prints
        the diff/command to stdout before this prompt.
        """
        if self.auto_approve:
            return Decision(True)
        if not self.interactive:
            return Decision(False)  # no TTY to ask → fail safe
        # Pause the type-ahead steer reader so it doesn't race us for stdin.
        if self.inbox is not None:
            self.inbox.pause()
        try:
            return await self._ask(name, target, reason, dangerous)
        finally:
            if self.inbox is not None:
                self.inbox.resume()

    async def _ask(self, name: str, target: str, reason: str, dangerous: str) -> Decision:
        if dangerous:
            sys.stdout.write(
                f"  ▲ DANGEROUS — {dangerous}. Type 'yes' to confirm · "
                "[n]o · or type an instruction to redirect › "
            )
            sys.stdout.flush()
            try:
                line = input("").strip()
            except (EOFError, KeyboardInterrupt):
                return Decision(False)
            low = line.lower()
            if low == "yes":
                return Decision(True)
            if low in ("", "y", "n", "no"):  # 'y' alone is intentionally NOT enough
                return Decision(False)
            return Decision(False, feedback=line)  # any other text = redirect

        sys.stdout.write(
            f"  Approve {name}{_target_suffix(name, target)} ({reason})?  "
            "[y]es · [n]o · [a]ll · [A]lways this cmd · [e] redirect › "
        )
        sys.stdout.flush()
        ch = _read_single_key()
        if ch is None:  # no raw TTY → line input fallback (typed text = redirect)
            try:
                line = input("").strip()
            except (EOFError, KeyboardInterrupt):
                return Decision(False)
            if line == "A":  # capital A = always allow this command (persisted)
                return Decision(True, remember=True)
            low = line.lower()
            if low in ("y", "yes"):
                return Decision(True)
            if low in ("a", "all"):
                self.auto_approve = True
                return Decision(True)
            if low in ("", "n", "no"):
                return Decision(False)
            return Decision(False, feedback=line)
        sys.stdout.write((ch if ch.isprintable() else "") + "\n")
        sys.stdout.flush()
        if ch == "A":  # capital A = always allow this command (persisted)
            return Decision(True, remember=True)
        c = (ch or "").lower()
        if c == "a":
            self.auto_approve = True
            return Decision(True)
        if c == "y":
            return Decision(True)
        if c == "e":
            try:
                fb = input("  ↳ tell the agent what to do instead › ").strip()
            except (EOFError, KeyboardInterrupt):
                return Decision(False)
            return Decision(False, feedback=fb)
        return Decision(False)


class TerminalObserver(BaseObserver):
    """Render the agent loop + gate risky tool calls. Critical (awaited)."""

    critical = True
    # The loop only streams ``on_llm_delta`` when an observer opts in via this
    # probe (agent_loop.py: ``stream_llm_tokens = any(wants_llm_delta)``).
    # Without it the TUI would only render whole turns via the
    # ``on_llm_response`` fallback — no token-by-token streaming.
    wants_llm_delta = True

    def __init__(
        self, renderer: Renderer, approver: Approver, cwd: str, journal: Any = None,
        plan_state: Any = None, steer_inbox: Any = None, rules: Any = None,
    ) -> None:
        self.r = renderer
        self.approver = approver
        self.cwd = cwd
        self.journal = journal  # WorkspaceJournal | None — snapshots mutations
        self.plan_state = plan_state  # PlanState | None — gates edits while planning
        self.steer_inbox = steer_inbox  # SteerInbox | None — type-ahead steering
        self.rules = rules  # PermissionStore | None — persistent allow/deny rules
        self._turn_streamed = False
        # Set when the user plainly rejects a call → end the task at on_turn_end
        # (rather than let the model keep retrying the declined action).
        self._abort_reason: str | None = None
        # Turn counters for the periodic TodoWrite reminder.
        self._turns_since_todo = 0
        self._turns_since_reminder = 0
        self._activity_ids: deque[str] = deque()
        self._activity_synthetic: deque[bool] = deque()
        self._activity_sequence = 0
        # ``(roots, ephemeral baseline)`` for mutating tools such as bash /
        # create_file, whose host write targets are not structured ``path``
        # arguments. Only actual changes are retained by WorkspaceJournal.
        #
        # ONE baseline per tool phase, not one per call. A tool phase runs its
        # calls through ``asyncio.gather``, so per-call baselines were both
        # unpairable (a synthetic call id never comes back on the result, and
        # the positional fallback matches whatever finished first) and racy
        # (a baseline captured while a sibling call is already writing has
        # that sibling's edits folded into it, hiding them). Sharing the
        # earliest baseline of the phase is correct under every interleaving:
        # it is taken before any mutating call runs, and ``setdefault`` in
        # ``finish_tree_scan`` keeps the oldest content per file anyway.
        self._journal_scan: tuple[list[str], Any] | None = None
        # Serialises the check-then-capture: without it two calls entering the
        # hook together both see ``None`` and both walk the tree.
        self._journal_scan_lock = asyncio.Lock()

    def _journal_scan_roots(self) -> list[str]:
        roots = [self.cwd]
        for env_name in ("FRONTIER_AGENT_OUTPUTS_DIR", "APODEX_HOST_OUTPUTS_DIR"):
            value = os.environ.get(env_name, "").strip()
            if value:
                roots.append(value)
        return roots

    async def _begin_journal_scan(self) -> None:
        """Ensure this tool phase has a baseline, capturing one if it doesn't."""
        async with self._journal_scan_lock:
            if self._journal_scan is not None or self.journal is None:
                return
            roots = self._journal_scan_roots()
            # Walking and reading the tree is seconds of blocking I/O on a
            # large checkout; on the event loop it freezes the TUI before
            # every mutating shell command.
            before = await asyncio.to_thread(self.journal.begin_tree_scan, roots)
            self._journal_scan = (roots, before)

    async def _finish_journal_scan(self) -> None:
        """Fold the phase's changes into the journal and drop the baseline.

        Safe to call when there is nothing pending. A baseline holds the text
        of the whole scanned tree, so leaving one alive costs real memory —
        which is why ``on_turn_end`` settles anything the result phase missed
        instead of only discarding it.
        """
        async with self._journal_scan_lock:
            scan, self._journal_scan = self._journal_scan, None
        if scan is None or self.journal is None:
            return
        roots, before = scan
        await asyncio.to_thread(self.journal.finish_tree_scan, roots, before)

    @staticmethod
    def _accepts_keyword(method: Any, keyword: str) -> bool:
        """Keep third-party renderers with the pre-activity signature working."""
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == keyword
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _render_tool_call(
        self, name: str, args: dict, *, call_id: str,
        risk_reason: str = "", danger: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {"risk_reason": risk_reason, "danger": danger}
        if self._accepts_keyword(self.r.tool_call, "call_id"):
            kwargs["call_id"] = call_id
        self.r.tool_call(name, args, **kwargs)

    def _render_activity_call(self, name: str, args: dict, *, call_id: str) -> None:
        method = getattr(self.r, "activity_call", None)
        if method is None:
            return
        kwargs = {"call_id": call_id} if self._accepts_keyword(method, "call_id") else {}
        method(name, args, **kwargs)

    def _render_tool_result(
        self, result: ToolResult, *, call_id: str,
    ) -> None:
        kwargs: dict[str, Any] = {
            "is_error": result.is_error,
            "ms": result.duration_ms,
        }
        if self._accepts_keyword(self.r.tool_result, "call_id"):
            kwargs["call_id"] = call_id
        self.r.tool_result(result.name, result.result, **kwargs)

    def _complete_activity(
        self, result: ToolResult, *, call_id: str, outcome: str = "",
    ) -> None:
        method = getattr(self.r, "activity_result", None)
        if method is None:
            return
        kwargs: dict[str, Any] = {
            "is_error": result.is_error,
            "ms": result.duration_ms,
            "outcome": outcome,
        }
        if self._accepts_keyword(method, "call_id"):
            kwargs["call_id"] = call_id
        method(result.name, **kwargs)

    def on_subagent_status(
        self, snapshots: list[dict[str, Any]], *, done: bool = False,
        timeout_s: int = 0,
    ) -> None:
        """Forward AgentBus wait progress to renderers that support it."""
        method = getattr(self.r, "subagent_status", None)
        if method is not None:
            method(snapshots, done=done, timeout_s=timeout_s)

    def _skip_tool(self, result: str) -> ToolCallIntervention:
        self._activity_synthetic[-1] = True
        return ToolCallIntervention(skip_with_result=result)

    async def on_loop_start(self, config: LoopConfig) -> None:
        # The model is about to generate turn 1 (can take a couple of seconds
        # on the first request) — show a "working…" indicator so it isn't
        # mistaken for a freeze.
        self.r.working_on()

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None:
        self._turn_streamed = True
        if getattr(ctx, "thinking_delta", ""):
            self.r.thinking_delta(ctx.thinking_delta)
        if ctx.delta:
            self.r.content_delta(ctx.delta)
        return None

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        # Fallback: if the provider never streamed deltas this turn, show
        # the full assistant text now so nothing is silently dropped.
        if not self._turn_streamed:
            self.r.turn_text_fallback(ctx.ai_text or "", ctx.thinking or "")
        else:
            # One assistant message = one rendered block. The turn boundary is
            # the only reliable signal for it: a task whose tools all present
            # themselves in the timeline mounts nothing in between.
            end_turn = getattr(self.r, "end_turn_text", None)
            if end_turn is not None:
                end_turn()
        self._turn_streamed = False
        return None

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        self._activity_sequence += 1
        call_id = str(
            tool_call.get("id") or f"apodex-{ctx.turn}-{self._activity_sequence}"
        )
        self._activity_ids.append(call_id)
        self._activity_synthetic.append(False)
        # Rewrite an absolute cwd-internal path to relative so file tools take
        # the fast local path (abs paths hit a ~50s sandbox bootstrap). The
        # rewritten args are what actually executes (via rewrite_args).
        rewritten = localize_path_args(name, args, self.cwd)
        eff_args = rewritten if rewritten is not None else args

        # Plan mode: ``exit_plan_mode`` is a human-approval gate that unlocks
        # edits; every other mutating call is blocked until then.
        if name == "exit_plan_mode":
            self._render_activity_call(name, eff_args, call_id=call_id)
            intervention = await self._review_plan(
                str(eff_args.get("plan", "")).strip(),
            )
            if intervention.skip_with_result is not None:
                self._activity_synthetic[-1] = True
            return intervention
        if (self.plan_state is not None and self.plan_state.active
                and is_mutating_tool(name, eff_args)):
            self._render_tool_call(
                name, eff_args, call_id=call_id, risk_reason="plan mode",
            )
            self.r.note(f"✗ plan mode: '{name}' is disabled until the plan is approved")
            return self._skip_tool(
                f"[Plan mode is active — '{name}' is disabled. Investigate with "
                "read/search tools, then call exit_plan_mode(plan=...) to get the "
                "user's approval before editing.]"
            )

        risk = assess_with_rules(
            name, eff_args, self.cwd, self.rules,
            auto_for_me=getattr(self.approver, "auto_for_me", False),
        )
        self._render_tool_call(
            name, eff_args, call_id=call_id,
            risk_reason=(risk.danger or ("" if risk.level == RISK_SAFE else risk.reason)),
            danger=bool(risk.danger),
        )

        # Preview of the proposed action, shown *inside* the approval modal so
        # the user can see exactly what they're approving (diff / command).
        preview = ""
        preview_kind = ""
        # Read-before-edit: don't let the model overwrite/edit an existing file
        # it hasn't read this session, or one changed on disk since it read it.
        if name in _DIFF_TOOLS:
            guard = fsguard.check_can_edit(
                str(eff_args.get("path") or eff_args.get("file_path") or ""), self.cwd,
            )
            if guard:
                self.r.note(f"✗ {name} blocked: {guard}")
                return self._skip_tool(f"[{guard}]")
            # Show exactly what will change on disk before asking for approval.
            diff = unified_diff(name, eff_args, self.cwd)
            if diff:
                self.r.diff_preview(diff, stats=change_stats(name, eff_args, self.cwd))
                preview, preview_kind = diff, "diff"
        elif name == "bash":
            cmd = str(eff_args.get("command", "")).strip()
            if cmd:
                preview, preview_kind = cmd, "command"

        if risk.level == RISK_DENY:
            self.r.note(f"✗ blocked: {risk.reason}")
            return self._skip_tool(
                f"[blocked by safety policy: {risk.reason}]",
            )
        if risk.level != RISK_SAFE:  # confirm: ask the human
            decision = await self.approver.confirm(
                name, risk.target, risk.reason, dangerous=risk.danger,
                preview=preview, preview_kind=preview_kind,
            )
            if not decision.approved:
                # Human-in-the-loop: a typed instruction redirects the agent;
                # it's fed back as the (declined) call's result so the model
                # adapts on the next turn instead of just retrying blindly.
                if decision.feedback:
                    self.r.note(f"↳ redirecting {name}: {decision.feedback}")
                    return self._skip_tool(
                        f"[The user declined to run this {name} call. "
                        f"Follow their instruction instead: {decision.feedback}]"
                    )
                # Plain reject → stop the task and hand control back (Claude /
                # kimi behaviour). Re-trying a rejected action just makes the
                # model ramble; ``[e] redirect`` is how the user keeps it going.
                self._abort_reason = "user_rejected"
                self.r.note(f"✗ rejected {name} — stopping task (use [e] to redirect instead)")
                return self._skip_tool(
                    f"[user rejected this {name} call — task stopped]",
                )
            # 'Always allow this command' → persist a rule so identical calls
            # auto-approve next time (danger/deny still override it).
            if decision.remember and self.rules is not None:
                self.r.note(f"✓ always allowing: {self.rules.add_allow(name, eff_args)}")

        # Allowed (safe or approved) → snapshot a mutating op before it runs so
        # the change is diffable + revertable, then let it execute.
        if self.journal is not None and name in MUTATING_TOOLS:
            p = eff_args.get("path") or eff_args.get("file_path")
            if isinstance(p, str) and p:
                self.journal.record_before(p)
        elif self.journal is not None and is_mutating_tool(name, eff_args):
            # Awaited before the call executes, so the baseline is always in
            # place first — including for a sibling that reuses it.
            await self._begin_journal_scan()
        return ToolCallIntervention(rewrite_args=rewritten) if rewritten is not None else None

    async def _review_plan(self, plan: str) -> ToolCallIntervention:
        """Human-approval gate for ``exit_plan_mode``: show the plan, ask the
        user, and unlock edits only on approval (otherwise revise, stay planning)."""
        self.r.plan_review(plan)
        if self.plan_state is None or not self.plan_state.active:
            return ToolCallIntervention(
                skip_with_result="[Not in plan mode — continue normally.]"
            )
        decision = await self.approver.confirm(
            "exit_plan_mode", "", "approve this plan and start editing?",
        )
        if decision.approved:
            self.plan_state.active = False
            self.r.note("✓ plan approved — edits unlocked")
            return ToolCallIntervention(skip_with_result=(
                "[The user APPROVED your plan. Plan mode is now OFF — you may "
                "edit files and run commands. Implement the plan now.]"
            ))
        fb = decision.feedback
        self.r.note("↺ plan needs revision" + (f": {fb}" if fb else ""))
        return ToolCallIntervention(skip_with_result=(
            "[The user did NOT approve the plan. Stay in plan mode and revise it"
            + (f". Their feedback: {fb}" if fb else "")
            + ". Then call exit_plan_mode again with the updated plan.]"
        ))

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        queued_id = self._activity_ids.popleft() if self._activity_ids else ""
        call_id = queued_id or str(getattr(result, "tool_call_id", "") or "")
        # Settle the phase's baseline once no call is still outstanding —
        # ``_activity_ids`` drains exactly one entry per result, so an empty
        # deque is the "every call has reported" signal, and it needs no id
        # pairing. A mutating call that starts afterwards takes a fresh
        # baseline in its own hook, before it runs.
        if not self._activity_ids:
            await self._finish_journal_scan()
        is_synthetic = (
            self._activity_synthetic.popleft()
            if self._activity_synthetic else False
        )
        body = result.result if isinstance(result.result, str) else str(result.result)
        # A skipped call never ran. Complete its activity row without showing a
        # success panel or marking a file as seen by read-before-edit.
        if is_synthetic or body.startswith(_SYNTHETIC_RESULT_MARKERS):
            self._complete_activity(result, call_id=call_id, outcome="skipped")
            self.r.working_on()
            return None

        # Record a successful read/write so read-before-edit knows this file is
        # seen (you can edit a file you just read OR just wrote).
        if result.name in _SEEN_TOOLS and not result.is_error:
            args = result.args if isinstance(result.args, dict) else {}
            path = str(args.get("path") or args.get("file_path") or "")
            if path:
                fsguard.record_read(path, self.cwd)

        # The plan tool renders as a checklist panel instead of a raw result.
        if result.name == "todo_write":
            self._turns_since_todo = 0  # reset the reminder clock
            self._complete_activity(result, call_id=call_id)
            if result.is_error:
                self.r.error(result.result)
            else:
                from apodex.todo import get_todos
                self.r.todos(get_todos())
            self.r.working_on()
            return None

        # Empty-output guard: a silent success ("") can make some models stall
        # or emit a stop sequence. Replace it with an explicit marker — for
        # both the display AND the model-facing result.
        out: ToolResult | None = None
        if not result.is_error and not body.strip():
            result = replace(result, result=f"({result.name} completed with no output)")
            out = result

        self._render_tool_result(result, call_id=call_id)
        # The model now generates the next turn — show the working indicator
        # so the (often silent) gap before the next tool call isn't dead air.
        self.r.working_on()
        return out

    async def wait_for_tool_interrupt(
        self, ctx: TurnContext, tool_call: dict,
    ) -> bool:
        """Wake an idle aggregation-tool wait when the user intervenes."""
        if self.steer_inbox is None:
            return False
        wait_for_input: Callable[[], Awaitable[bool]] | None = getattr(
            self.steer_inbox, "wait_for_input", None,
        )
        if not callable(wait_for_input):
            return False
        return bool(await wait_for_input())

    async def on_tool_wait_interrupted(
        self, ctx: TurnContext,
    ) -> Intervention | None:
        """Claim queued input for the coordinator's immediate next turn."""
        if self.steer_inbox is None:
            return None
        steers = self.steer_inbox.drain()
        if not steers:
            return None
        self.r.note("⤷ steering: " + " / ".join(steers))
        return Intervention(inject_messages=steers)

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        # A call that never reported (interrupted, or a result the loop
        # dropped) leaves the phase's baseline outstanding. Settle it here
        # rather than discard it: the files it changed are real, and the
        # snapshot would otherwise stay in memory for the whole session.
        await self._finish_journal_scan()
        # A plain user rejection ends the task here — after this turn's tool
        # phase — so the model doesn't spin on the declined action. A typed
        # redirect does NOT set this, so feedback still continues the task.
        if self._abort_reason:
            reason, self._abort_reason = self._abort_reason, None
            return Intervention(stop_reason=reason)
        self._turns_since_todo += 1
        self._turns_since_reminder += 1
        # Live steering: if the user typed while the agent worked AND this turn
        # made tool calls (so the loop will continue), inject those lines as the
        # next user message. When the turn had no tool calls the model is
        # finishing — leave the queue for the session to run as a follow-up
        # (avoids a dangling user message on a turn that's about to stop).
        if self.steer_inbox is not None and getattr(ctx, "tool_calls", None):
            steers = self.steer_inbox.drain()
            if steers:
                self.r.note("⤷ steering: " + " / ".join(steers))
                return Intervention(inject_messages=steers)
        # Periodic plan reminder: if a todo list exists but hasn't been touched
        # for a while, re-inject it so a long run doesn't drift off-plan.
        if (self._turns_since_todo >= _TODO_REMINDER_TURNS
                and self._turns_since_reminder >= _TODO_REMINDER_TURNS):
            from apodex.todo import get_todos
            todos = get_todos()
            if todos:
                self._turns_since_reminder = 0
                listing = "\n".join(f"{i.glyph} {i.content} ({i.status})" for i in todos)
                return Intervention(inject_messages=[
                    "<system-reminder>You haven't updated your todo list recently. "
                    f"Current plan:\n{listing}\nKeep it current with todo_write; "
                    "exactly one item should be in_progress. Don't mention this "
                    "reminder to the user.</system-reminder>"
                ])
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        # Final answer is rendered by the session (it has turn/tool counts).
        return None
