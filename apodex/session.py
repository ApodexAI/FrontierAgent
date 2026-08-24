"""Interactive session: the REPL + a single agent run.

Wires the LLM, the local coding tools, and the :class:`TerminalObserver`
into :func:`run_agent_loop`. Conversation context is preserved across
tasks (follow-ups see prior turns + the now-modified repo), matching
apodex's "continue refining" behaviour.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from apodex.config import (
    ModelConfig,
    RuntimeConfigStatus,
    format_preflight_errors,
    format_runtime_config_status,
)
from apodex.llm import build_llm
from apodex.middleware import (
    _SKILLS_DIR as _SKILLS_DIR,
)
from apodex.middleware import (
    _wrap_skills_llm,
)
from apodex.observers import Approver
from apodex.profiles import get_profile, profile_names, terminal_mode_names
from apodex.render import Renderer
from apodex.session_state import (
    _session_state_path,
    list_saved_sessions,
    load_session_state,
    new_session_id,
)
from apodex.task_runner import TaskRunnerMixin
from frontier_agent.core.messages import Message, assistant_msg, user_msg
from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop as run_agent_loop
from frontier_agent.core.runtime.session_history import (
    SessionTurn,
    messages_to_session_turns,
)

_PROMPT = "\napodex› "

_HELP = """\
Commands:
  /help            show this help
  /mode <name>     switch mode/workflow: {modes} (resets context)
  /workflow <name> alias for /mode; native options: react | agent_team
  /model [<name|n>] list the profile's models / switch to one (rebuilds client)
  /cwd [<path>]    show or change the working directory (resets context)
  /new             save this session and start a fresh one
  /fork            save and branch the current context into a new session
  /sessions        list recent saved sessions
  /rename <name>   give the current session a readable name
  /clear           clear the conversation context + plan (repo state is kept)
  /plan            toggle plan mode: investigate + propose a plan; edits stay
                   locked until you approve the agent's exit_plan_mode plan
  /revert          undo this session's edits to files in the working directory
                   (not shell-driven edits or sandbox deliverables)
  /compact         compress tool results, then summarize the whole session
  /cost, /context  show token usage + context window remaining
  /config          show safe local provider/model configuration (never the key)
  /init            analyze the repo and write an AGENTS.md project guide
  /resume          pick a saved session to continue
  /attach <path>   attach a file/directory; relative paths start at session cwd
  /attachments     list files attached to this session
  /detach <name>   remove an attachment copy (the source is unchanged)
  /paste           attach Finder files or an image from the macOS clipboard
  /log             show the trace file path (all tool calls + results)
  /auto            toggle auto-approve (skip confirmation prompts)
  /verbose         toggle thinking stream: full (default) ↔ collapsed indicator
  /theme <name>    switch colour theme (use /theme with no name to list)
  /exit, /quit     leave
Anything else is sent to the current-mode agent as a task.
While the agent is working you can TYPE to steer it (applied at the next step)
or press Ctrl-C to interrupt the current task.
At an approval prompt: [A] always-allows that command (persisted)."""


_INIT_PROMPT = (
    "Analyze this repository and create a concise AGENTS.md at its root (a "
    "project guide for AI coding agents). Investigate first with read/search "
    "tools — do not guess. Cover: what the project is and its entry points; the "
    "build / test / run / lint commands (find them in package.json, pyproject, "
    "Makefile, CI configs); the high-level code layout; and any conventions a "
    "newcomer must follow. Keep it tight and factual — link to real files. If an "
    "AGENTS.md already exists, read it and improve it rather than duplicating."
)




class TerminalSession(TaskRunnerMixin):
    def __init__(
        self,
        *,
        cfg: ModelConfig,
        cwd: str,
        renderer: Renderer,
        auto_approve: bool,
        max_turns: int,
        interactive: bool = True,
        mode: str = "react",
        session_id: str | None = None,
        plan_mode: bool = False,
    ) -> None:
        from apodex.changes import WorkspaceJournal
        from apodex.env import environment_section
        from apodex.permissions import PermissionStore
        from apodex.plan import PlanState
        from apodex.trace import TraceObserver, default_trace_path
        from apodex.usage import Usage

        self.cfg = cfg
        self.cwd = cwd
        self.r = renderer
        self.max_turns = max_turns
        self.interactive = interactive
        # The CLI only exposes the two native workflows, while retaining this
        # broader constructor compatibility for embedders and old checkpoints.
        self.mode = mode if mode in profile_names() else "react"
        # TUI seam (line mode leaves these untouched): the full-screen TUI owns
        # the terminal, so it sets ``tui_mode`` to skip the stdin steer-reader
        # (which would race Textual for the same fd) and reads ``_inbox`` to feed
        # steered lines from its input box. See apodex/tui/.
        self.tui_mode = False
        self._inbox: Any = None  # current run's SteerInbox, exposed for the TUI
        from apodex.config import UserSettings
        self.user_settings = UserSettings.load()
        eff_auto_approve = auto_approve or self.user_settings.auto_approve
        self.approver = Approver(
            auto_approve=eff_auto_approve,
            auto_for_me=self.user_settings.auto_for_me,
            interactive=interactive,
        )
        self.verbose = bool(getattr(renderer, "_verbose", False))
        self.plan_state = PlanState(active=plan_mode or self.user_settings.plan_mode)
        # Persistent allow/deny rules + running token usage (surfaced in the
        # spinner / footer / /cost). The env block is cache-stable per cwd.
        self.rules = PermissionStore.load()
        self.usage = Usage()
        self.r.set_usage(self.usage, cfg.context_window)
        self._env_section = environment_section(cwd, cfg.model)
        self._compact_retried = False  # one compact-and-resume per task
        self._build_llm()
        # Models the active profile offers for /model selection (default first).
        try:
            self.models = list(get_profile(self.mode).models)
        except Exception:
            self.models = [self.cfg.model]
        self.history: list[Message] = []  # native dict messages, across tasks
        # Kept separately from compacted model context so /resume can restore
        # the complete visible conversation and tool protocol.
        self.display_history: list[Message] = []
        # Native workflow DAGs are task-scoped.  Keep their richer per-turn
        # histories separately so one terminal session can replay every prior
        # turn, including intermediate query/tool results.
        self.workflow_turns: list[SessionTurn] = []
        # Presentation-only metadata used by the full-screen TUI.  The model
        # never receives this data; it lets /resume restore Agent Team worker
        # hierarchy that cannot be inferred from OpenAI wire messages.
        self.tui_state: dict[str, Any] = {}
        self.session_id = (
            session_id
            or os.environ.get("APODEX_SESSION_ID", "").strip()
            or new_session_id(self.mode)
        )
        from apodex.run_layout import activate_run, new_run_timestamp
        activate_run(self.session_id, cwd)
        _stamp, self.created_at, self.local_timezone = new_run_timestamp()
        self.session_name = ""
        self._activate_session_workspace(self.session_id, cwd)
        self._activate_session_outputs(self.session_id, cwd)
        from apodex.attachments import AttachmentManager
        self.attachments = AttachmentManager(self.cwd, self.session_id)
        os.environ["FRONTIER_AGENT_INPUTS_DIR"] = str(self.attachments.agent_dir)
        # Journal (revert + changed-files) and trace (local log) — both are
        # plain observers/objects threaded into each run; no engine changes.
        self.journal = WorkspaceJournal(cwd)
        self.trace_path = default_trace_path(self.session_id)
        self.tracer = TraceObserver(self.trace_path, mode=self.mode, cwd=cwd)
        # Authorize the working directory for the file tools (read_file /
        # write_file / file_editor all gate on CODING_WORKSPACE_ROOT via
        # plugins.tools._path_auth._authorized_local_path). Without this they
        # only allow a few default dirs and deny the user's repo.
        self._authorize_workspace(cwd)

    @staticmethod
    def _active_spill_workspace() -> Path | None:
        """Resolve the physical workspace before a session link is retargeted."""
        # A deployment-provided FRONTIER_AGENT_WORKSPACE_DIR can be shared by
        # several sessions. Broad cleanup is safe only when TerminalSession
        # itself activated a workspace below one of its private session roots.
        roots = [
            value
            for value in (
                os.environ.get("APODEX_RUNS_ROOT", "").strip(),
                os.environ.get("APODEX_SESSION_WORKSPACES_ROOT", "").strip(),
            )
            if value
        ]
        if not roots:
            return None
        raw = os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "").strip()
        if not raw:
            return None
        try:
            workspace = Path(raw).resolve()
        except OSError:
            return None
        # Verify the containment the comment above relies on. Having a session
        # root configured does not prove this workspace is inside one:
        # FRONTIER_AGENT_WORKSPACE_DIR may still point at a shared or
        # user-owned directory, and recursive cleanup would then delete
        # ``<that dir>/.spill`` on every /clear, /mode and /cwd. Resolving first
        # is what makes the check hold for the ``APODEX_WORKSPACE_LINK`` symlink
        # that ``_activate_session_workspace`` may install here.
        for root in roots:
            try:
                workspace.relative_to(Path(root).expanduser().resolve())
            except (OSError, ValueError):
                continue
            return workspace
        return None

    @staticmethod
    def _cleanup_discarded_spill(workspace: Path | None) -> int:
        """Drop recovery files only when their owning context is discarded.

        ``workspace`` is now only the signal that a session-private context was
        identified; the store itself lives outside the workspace, so what gets
        removed is every store THIS process created — which is exactly the set
        the cleared history could have referenced, and cannot include another
        session's files.
        """
        if workspace is None:
            return 0
        from plugins.tools._overflow import cleanup_overflow_process

        return cleanup_overflow_process()

    def start_new_session(self, *, fork: bool = False) -> tuple[str, str]:
        """Save the current checkpoint and switch to a fresh session identity.

        A fork retains model-visible conversation history. A plain new session
        starts with an empty context. Neither operation changes workspace files;
        the new journal begins at the workspace's current state.
        """
        from apodex import fsguard
        from apodex.attachments import AttachmentManager
        from apodex.changes import WorkspaceJournal
        from apodex.todo import clear_todos
        from apodex.trace import TraceObserver, default_trace_path
        from apodex.usage import Usage

        self._persist()
        previous = self.session_id
        kept_history = list(self.history) if fork else []
        kept_display_history = list(self.display_history) if fork else []
        kept_turns = list(self.workflow_turns) if fork else []
        self.session_id = new_session_id(self.mode)
        from apodex.run_layout import new_run_timestamp
        _stamp, self.created_at, self.local_timezone = new_run_timestamp()
        self._activate_session_workspace(self.session_id, self.cwd)
        self._activate_session_outputs(self.session_id, self.cwd)
        self.session_name = ""
        self.history = kept_history
        self.display_history = kept_display_history
        self.workflow_turns = kept_turns
        self._compact_retried = False
        clear_todos()
        fsguard.clear()
        self.usage = Usage()
        self.r.set_usage(self.usage, self.cfg.context_window)
        self.journal = WorkspaceJournal(self.cwd)
        self.attachments = AttachmentManager(self.cwd, self.session_id)
        os.environ["FRONTIER_AGENT_INPUTS_DIR"] = str(self.attachments.agent_dir)
        self.trace_path = default_trace_path(self.session_id)
        self.tracer = TraceObserver(self.trace_path, mode=self.mode, cwd=self.cwd)
        self._persist()
        return previous, self.session_id

    def rename_session(self, name: str) -> str:
        """Persist a readable label without changing the stable session id."""
        clean = " ".join(name.split()).strip()
        if not clean:
            raise ValueError("usage: /rename <name>")
        if len(clean) > 80:
            raise ValueError("session names are limited to 80 characters")
        self.session_name = clean
        self._persist()
        return clean

    def _build_llm(self) -> None:
        """Build ``self.llm`` from ``self.cfg``, skill-wrapped per the active
        profile. Called on construction and whenever the model or mode changes."""
        llm = build_llm(self.cfg)
        try:
            skills = get_profile(self.mode).skills
        except Exception:
            skills = []
        self.llm = _wrap_skills_llm(llm, skills) if skills else llm

    @staticmethod
    def _activate_session_outputs(
        session_id: str, workspace: str | None = None,
    ) -> None:
        """Point the stable agent path at the active session's output dir.

        ``workspace`` moves the whole run record with the session, so an
        in-app ``/resume`` into another project stops writing that project's
        trace, log, and deliverables into the previous one."""
        from apodex.run_layout import activate_run, pinned_mounts

        if pinned_mounts():
            # The launcher bound a real ``/outputs``; leave the alias on it.
            # ``Session.__init__`` calls ``activate_run`` before us, and that
            # sets ``APODEX_RUNS_ROOT`` unconditionally, so the guard below is
            # always open inside a jail — without this the deliverable root
            # would silently become ``<runs>/<session>/outputs``, off whatever
            # the grader collects from.
            return
        runs_root_value = os.environ.get("APODEX_RUNS_ROOT", "").strip()
        root_value = os.environ.get("APODEX_SESSION_OUTPUTS_ROOT", "").strip()
        if not runs_root_value and not root_value:
            return
        active = activate_run(session_id, workspace) if runs_root_value else None
        # Re-read: activate_run may have repointed the root at ``workspace``.
        root = Path(
            os.environ.get("APODEX_RUNS_ROOT", "").strip() or root_value
        ).expanduser().resolve()
        target = active / "outputs" if active is not None else (root / session_id).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"invalid output session id: {session_id!r}") from exc
        target.mkdir(parents=True, exist_ok=True)

        link_value = os.environ.get("APODEX_OUTPUTS_LINK", "").strip()
        if link_value:
            link = Path(link_value)
            if link.is_symlink():
                link.unlink()
            elif link.is_dir() and not any(link.iterdir()):
                link.rmdir()
            elif link.exists():
                raise ValueError(f"session output link is not a symlink: {link}")
            link.symlink_to(target, target_is_directory=True)
            os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] = str(link)
        else:
            os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] = str(target)

        host_root_value = os.environ.get("APODEX_HOST_OUTPUTS_ROOT", "").strip()
        host_runs_root = os.environ.get("APODEX_HOST_RUNS_ROOT", "").strip()
        if host_runs_root:
            os.environ["APODEX_HOST_OUTPUTS_DIR"] = str(
                Path(host_runs_root).expanduser() / session_id / "outputs"
            )
        elif host_root_value:
            os.environ["APODEX_HOST_OUTPUTS_DIR"] = str(
                Path(host_root_value).expanduser() / session_id
            )

    @staticmethod
    def _activate_session_workspace(
        session_id: str, project: str | None = None,
    ) -> None:
        """Point the stable scratch path at this run's private workspace.

        The project named by ``--cwd`` remains the coding/journal root. Shell
        commands start in this session-scoped scratch tree so downloads,
        clones, drafts, and verification scripts cannot spill into the project
        or collide with another run.
        """
        from apodex.run_layout import activate_run, pinned_mounts

        if pinned_mounts():
            # A jail already isolates this run, and its ``/workspace`` mount is
            # the scratch tree. Repointing the alias at ``<runs>/<session>/
            # workspace`` would put the model's relative paths somewhere the
            # jail's own contract never mentions.
            #
            # Session switches still need activate_run's identity side effects
            # (APODEX_SESSION_ID / APODEX_RUN_DIR). Initial construction calls
            # it directly, but start_new_session and switch_session reach it
            # only through this helper.
            activate_run(session_id, project)
            return
        runs_root_value = os.environ.get("APODEX_RUNS_ROOT", "").strip()
        root_value = os.environ.get("APODEX_SESSION_WORKSPACES_ROOT", "").strip()
        if not runs_root_value and not root_value:
            return
        active = activate_run(session_id, project) if runs_root_value else None
        root = Path(
            os.environ.get("APODEX_RUNS_ROOT", "").strip() or root_value
        ).expanduser().resolve()
        target = active / "workspace" if active is not None else (root / session_id).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"invalid workspace session id: {session_id!r}") from exc
        target.mkdir(parents=True, exist_ok=True)

        link_value = os.environ.get("APODEX_WORKSPACE_LINK", "").strip()
        if link_value:
            link = Path(link_value)
            if link.is_symlink():
                link.unlink()
            elif link.is_dir() and not any(link.iterdir()):
                link.rmdir()
            elif link.exists():
                raise ValueError(f"session workspace link is not a symlink: {link}")
            link.symlink_to(target, target_is_directory=True)
            os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] = str(link)
        else:
            os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] = str(target)

        host_root_value = os.environ.get("APODEX_HOST_WORKSPACE_ROOT", "").strip()
        host_runs_root = os.environ.get("APODEX_HOST_RUNS_ROOT", "").strip()
        if host_runs_root:
            os.environ["APODEX_HOST_WORKSPACE_DIR"] = str(
                Path(host_runs_root).expanduser() / session_id / "workspace"
            )
        elif host_root_value:
            os.environ["APODEX_HOST_WORKSPACE_DIR"] = str(
                Path(host_root_value).expanduser() / session_id
            )

    def runtime_config_status(self) -> RuntimeConfigStatus:
        """Return the active profile's secret-free configuration status."""
        profile = get_profile(self.mode)
        return profile.runtime_config(self.cfg, mode=self.mode)

    def demo_welcome(self) -> str:
        """Compact startup context for the local harness demo."""
        status = self.runtime_config_status()
        endpoint = status.endpoint_host or "provider default"
        try:
            from apodex.sandbox import active_strategy
            sandbox = active_strategy().name
        except Exception:
            sandbox = "not resolved"
        return (
            "Ready for a local BYOK demo\n"
            f"{status.mode} · {status.provider}/{status.model} · {endpoint} · "
            f"sandbox {sandbox}\n"
            f"workspace: {self.cwd}\n"
            "Try: explain this repository · find and fix a failing test · "
            "review the current diff\n"
            "/config settings · F1 help · Ctrl-P commands · Ctrl-C interrupt"
        )

    @staticmethod
    def _authorize_workspace(cwd: str) -> None:
        # Authorize the cwd for the local file tools …
        os.environ["CODING_WORKSPACE_ROOT"] = cwd
        # Keep the user's project explicit now that model-authored shell
        # commands start in a separate, run-private scratch directory.
        os.environ["FRONTIER_AGENT_PROJECT_DIR"] = cwd
        # … and force a fully-LOCAL toolchain. With E2B_API_KEY set, the
        # shared file tools spin up an E2B cloud sandbox (slow + writes land
        # in the sandbox, not the local repo). A local coding agent must stay
        # on the host filesystem, consistent with our local-cwd bash. The
        # only E2B-only tool (run_python_code) is not in the coding tool set.
        os.environ.pop("E2B_API_KEY", None)



    async def _on_turn(self, turn: int, messages: list, metadata: dict) -> None:
        """Per-turn checkpoint (interrupt-safe resume). Fired by run_agent_loop
        after each completed turn — keep history current and persist."""
        self.history = list(messages)
        self.display_history = list(messages)
        self._persist()

    # ── persistence (interrupt-safe resume) ───────────────────────────────
    def _enrich_task(self, task: str) -> str:
        profile = get_profile(self.mode)
        enriched = self.attachments.enrich_task(
            task,
            # Agent Team deliberately keeps full file readers on sub-agents;
            # its coordinator only has locate/peek tools.
            delegate_file_reading=profile.workflow == "agent_team",
        )
        artifacts = self._deliverable_context()
        return f"{enriched}\n\n{artifacts}" if artifacts else enriched

    def _deliverable_context(self) -> str:
        """Give resumed follow-ups exact agent and user-visible file paths."""
        agent_value = os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "").strip()
        if not agent_value:
            return ""
        agent_root = Path(agent_value)
        if not agent_root.is_dir():
            return ""
        host_value = os.environ.get("APODEX_HOST_OUTPUTS_DIR", "").strip()
        host_root = Path(host_value) if host_value else None
        try:
            files = sorted(path for path in agent_root.rglob("*") if path.is_file())
        except OSError:
            return ""
        if not files:
            return ""
        lines = [
            "Prior deliverables from this same session are listed below. "
            "Use these exact paths before searching the workspace:"
        ]
        if os.environ.get("SANDBOX_BACKEND", "").strip().lower() == "native":
            lines.append(
                "Native-path note: the paths below are physical paths for "
                "reading existing files. In agent_team assign_task calls, "
                "output_paths must still use /outputs/<relative-file>; do not "
                "copy the physical path into the manifest."
            )
        for path in files[:200]:
            relative = path.relative_to(agent_root)
            line = f"- agent path: {path.absolute()}"
            if host_root is not None:
                line += f"; user-visible host path: {host_root / relative}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _workflow_display_messages(
        task: str, steps: list[dict[str, Any]], final: str,
    ) -> list[Message]:
        messages: list[Message] = [user_msg(task)]
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            name = str(step.get("tool_name") or "tool")
            args = step.get("tool_args") or {}
            arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            call_id = f"workflow-tool-{index}"
            messages.append(assistant_msg("", tool_calls=[{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]))
            messages.append({
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": str(step.get("tool_result") or ""),
                "duration_ms": int(step.get("duration_ms") or 0),
                "is_error": bool(step.get("is_error", False)),
            })
        messages.append(assistant_msg(final))
        return messages

    def replay_history(self) -> list[Message]:
        """Return full UI history, with a legacy workflow fallback."""
        if self.display_history:
            return list(self.display_history)
        if self.workflow_turns:
            replay: list[Message] = []
            known_calls: set[str] = set()
            legacy_index = 0
            for turn in self.workflow_turns:
                for raw in turn.get("messages") or []:
                    if not isinstance(raw, dict):
                        continue
                    # ``.copy()`` not ``dict(...)``: same shallow copy, but it
                    # keeps the Message TypedDict instead of widening to
                    # dict[str, object].
                    message = raw.copy()
                    for call in message.get("tool_calls") or []:
                        if call.get("id"):
                            known_calls.add(str(call["id"]))
                    if message.get("role") == "tool":
                        call_id = str(message.get("tool_call_id") or "")
                        if not call_id or call_id not in known_calls:
                            legacy_index += 1
                            call_id = call_id or f"legacy-workflow-tool-{legacy_index}"
                            name = str(message.get("name") or "tool")
                            replay.append(assistant_msg("", tool_calls=[{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": "{}"},
                            }]))
                            message["tool_call_id"] = call_id
                            known_calls.add(call_id)
                    replay.append(message)
            return replay
        return list(self.history)

    def _persist(self) -> None:
        """Checkpoint session state so ``--resume <id>`` can continue it.
        Best-effort; a failed write never disrupts the session."""
        try:
            import json

            from apodex.todo import get_todos

            snapshot = getattr(self.r, "snapshot_state", None)
            if callable(snapshot):
                raw_tui_state = snapshot()
                self.tui_state = raw_tui_state if isinstance(raw_tui_state, dict) else {}

            path = _session_state_path(self.session_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": self.session_id,
                    "created_at": self.created_at,
                    "local_timezone": self.local_timezone,
                    "name": self.session_name,
                    "mode": self.mode,
                    "cwd": self.cwd,
                    "model": self.cfg.model,
                    # Native messages are plain OpenAI-wire dicts — already
                    # JSON-serializable, so they round-trip verbatim (no
                    # langchain messages_to_dict / messages_from_dict needed).
                    "history": list(self.history),
                    "display_history": list(self.display_history),
                    "workflow_turns": list(self.workflow_turns),
                    "usage": self.usage.to_dict(),
                    "tui": dict(self.tui_state),
                    "outputs": {
                        "agent_root": os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", ""),
                        "host_root": os.environ.get("APODEX_HOST_OUTPUTS_DIR", ""),
                    },
                    "journal": self.journal.to_dict(),
                    "journal_observed": self.journal.observed_paths(),
                    "journal_revert_base": self.journal.revert_bases(),
                    "plan_active": bool(self.plan_state.active),
                    "todos": [
                        {"content": item.content, "status": item.status}
                        for item in get_todos()
                    ],
                }, f, ensure_ascii=False)
        except Exception:
            pass

    def restore(self, state: dict) -> None:
        """Reload history + journal from a persisted state dict."""
        from apodex.changes import WorkspaceJournal
        from apodex.todo import TodoItem, clear_todos, set_todos
        self.session_name = str(state.get("name") or "")
        self.created_at = str(state.get("created_at") or self.created_at)
        self.local_timezone = str(state.get("local_timezone") or self.local_timezone)
        saved_outputs = state.get("outputs") or {}
        if isinstance(saved_outputs, dict):
            for env_name, key in (
                ("FRONTIER_AGENT_OUTPUTS_DIR", "agent_root"),
                ("APODEX_HOST_OUTPUTS_DIR", "host_root"),
            ):
                value = str(saved_outputs.get(key) or "").strip()
                if value and not os.environ.get(env_name, "").strip():
                    os.environ[env_name] = value
        try:
            # History is stored as native dict messages (see ``_persist``);
            # they load back verbatim.
            self.history = list(state.get("history") or [])
        except Exception:
            self.history = []
        try:
            self.display_history = list(state.get("display_history") or [])
        except Exception:
            self.display_history = []
        try:
            self.workflow_turns = list(state.get("workflow_turns") or [])
        except Exception:
            self.workflow_turns = []
        self.usage.restore(state.get("usage"))
        raw_tui_state = state.get("tui") or {}
        self.tui_state = dict(raw_tui_state) if isinstance(raw_tui_state, dict) else {}
        if not self.workflow_turns:
            try:
                is_workflow_mode = bool(get_profile(self.mode).workflow)
            except Exception:
                is_workflow_mode = False
            if is_workflow_mode:
                # Checkpoints written before workflow_turns existed still have
                # the visible user/assistant transcript. Upgrade them in memory
                # so the first follow-up after /resume is genuinely multi-turn.
                self.workflow_turns = messages_to_session_turns(self.history)
        self.journal = WorkspaceJournal.from_dict(
            self.cwd, state.get("journal") or {},
            state.get("journal_observed") or [],
            state.get("journal_revert_base") or {},
        )
        self.plan_state.active = bool(state.get("plan_active", False))
        raw_todos = state.get("todos") or []
        restored_todos = [
            TodoItem(
                str(item.get("content", "")),
                str(item.get("status", "pending"))
                if str(item.get("status", "pending")) in ("pending", "in_progress", "completed")
                else "pending",
            )
            for item in raw_todos if isinstance(item, dict) and item.get("content")
        ]
        if restored_todos:
            set_todos(restored_todos)
        else:
            clear_todos()

    def switch_session(self, state: dict, *, fallback_id: str = "") -> None:
        """Switch a running UI to a persisted session checkpoint."""
        import dataclasses

        from apodex.env import environment_section
        from apodex.trace import TraceObserver, default_trace_path
        from apodex.usage import Usage

        target_mode = str(state.get("mode") or self.mode)
        if target_mode not in terminal_mode_names():
            raise ValueError(f"saved mode {target_mode!r} is no longer available")
        target_cwd = os.path.abspath(str(state.get("cwd") or self.cwd))
        if not os.path.isdir(target_cwd):
            raise ValueError(f"saved working directory does not exist: {target_cwd}")

        profile = get_profile(target_mode)
        cfg = dataclasses.replace(profile.model_config)
        saved_model = str(state.get("model") or "").strip()
        if saved_model:
            cfg.model = saved_model
        status = profile.runtime_config(cfg, mode=target_mode)
        if not status.ok:
            raise ValueError(format_preflight_errors(status))

        # Build fallible derived state before mutation so an invalid checkpoint
        # or model configuration leaves the current session usable.
        env_section = environment_section(target_cwd, cfg.model)
        llm = build_llm(cfg)
        if profile.skills:
            llm = _wrap_skills_llm(llm, profile.skills)
        usage = Usage()
        session_id = str(state.get("session_id") or fallback_id or self.session_id)
        from apodex.attachments import AttachmentManager
        attachments = AttachmentManager(target_cwd, session_id)
        self._activate_session_workspace(session_id, target_cwd)
        self._activate_session_outputs(session_id, target_cwd)
        trace_path = default_trace_path(session_id)
        tracer = TraceObserver(trace_path, mode=target_mode, cwd=target_cwd)

        os.chdir(target_cwd)
        self.cwd = target_cwd
        self.mode = target_mode
        self.cfg = cfg
        self.models = list(profile.models)
        self.session_id = session_id
        self._authorize_workspace(self.cwd)
        self._env_section = env_section
        self.llm = llm
        self.usage = usage
        self.r.set_usage(self.usage, self.cfg.context_window)
        self.trace_path = trace_path
        self.tracer = tracer
        self.attachments = attachments
        os.environ["FRONTIER_AGENT_INPUTS_DIR"] = str(self.attachments.agent_dir)
        self.restore(state)
        for warning in status.warnings:
            self.r.note(f"warning: {warning.message}")

    # ── REPL ──────────────────────────────────────────────────────────────
    async def repl(self, *, skip_banner: bool = False) -> None:
        if not skip_banner:
            self.r.banner(
                model=self.cfg.model, cwd=self.cwd,
                auto_approve=self.approver.auto_approve, mode=self.mode,
            )
            self.r.note(f"session {self.session_id}  ·  /log trace · /revert undo · /resume to switch")
            from apodex.env import dirty_summary
            dirty = dirty_summary(self.cwd)
            if dirty:
                self.r.note(f"git: {dirty}")
            if self.plan_state.active:
                self.r.note("▤ plan mode ON — edits locked until you approve a plan (/plan to toggle)")
        while True:
            try:
                # Line mode has no UI work to advance while waiting. Keeping
                # input on this thread also avoids a non-daemon executor thread
                # holding the process open after a terminal disconnect.
                line = input(_PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                self.r.note("bye")
                return
            if not line:
                continue
            if line.startswith("/"):
                if await self._slash(line):
                    return
                continue
            self.r.rule()
            await self.run_task(line)

    async def _slash(self, line: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("/exit", "/quit"):
            self.r.note("bye")
            return True
        if cmd == "/help":
            self.r.note(_HELP.format(modes=" | ".join(terminal_mode_names())))
        elif cmd in ("/mode", "/workflow"):
            if not arg:
                self.r.note(f"mode = {self.mode}  (available: {', '.join(terminal_mode_names())})")
            elif arg not in terminal_mode_names():
                self.r.error(f"unknown mode {arg!r}; available: {', '.join(terminal_mode_names())}")
            else:
                import dataclasses

                from apodex.env import environment_section
                from apodex.todo import clear_todos
                # The profile carries its own model list — adopt its default
                # (copy so /model edits don't mutate the cached profile), refresh
                # the selectable models, and rebuild the client.
                new_profile = get_profile(arg)
                new_cfg = dataclasses.replace(new_profile.model_config)
                status = new_profile.runtime_config(new_cfg, mode=arg)
                if not status.ok:
                    self.r.error(format_preflight_errors(status))
                    return False
                new_env_section = environment_section(self.cwd, new_cfg.model)
                new_llm = build_llm(new_cfg)
                if new_profile.skills:
                    new_llm = _wrap_skills_llm(new_llm, new_profile.skills)

                # Commit only after all fallible derived state is ready, so a
                # client/profile error leaves the current session untouched.
                discarded_workspace = self._active_spill_workspace()
                self.mode = arg
                self.history = []
                self.workflow_turns = []
                self.usage.clear_context()
                self._cleanup_discarded_spill(discarded_workspace)
                clear_todos()
                self.cfg = new_cfg
                self.models = list(new_profile.models)
                self.r.set_usage(self.usage, self.cfg.context_window)
                self._env_section = new_env_section
                self.llm = new_llm
                self.r.note(f"mode → {arg} (model {self.cfg.model}, context reset)")
                for warning in status.warnings:
                    self.r.note(f"warning: {warning.message}")
        elif cmd == "/clear":
            discarded_workspace = self._active_spill_workspace()
            self.history = []
            self.workflow_turns = []
            self.usage.clear_context()
            self._cleanup_discarded_spill(discarded_workspace)
            from apodex import fsguard
            from apodex.todo import clear_todos
            clear_todos()
            fsguard.clear()
            self.r.note("context + plan cleared (repo state unchanged)")
        elif cmd == "/new":
            previous, current = self.start_new_session()
            self.r.note(f"saved {previous}\nstarted new session {current}")
        elif cmd == "/fork":
            previous, current = self.start_new_session(fork=True)
            self.r.note(f"saved {previous}\nforked context into {current}")
        elif cmd == "/rename":
            try:
                name = self.rename_session(arg)
                self.r.note(f"session renamed → {name}")
            except ValueError as exc:
                self.r.error(str(exc))
        elif cmd == "/sessions":
            sessions = list_saved_sessions()[:20]
            if not sessions:
                self.r.note("no saved sessions yet")
            else:
                rows = ["Recent sessions:"]
                for item in sessions:
                    label = f" · {item['name']}" if item.get("name") else ""
                    current = "  (current)" if item["session_id"] == self.session_id else ""
                    rows.append(
                        f"  {item['session_id']}{label}{current}\n"
                        f"    {item['modified_at']} · {item['mode']} · "
                        f"{item['message_count']} msgs · {item['cwd']}"
                    )
                self.r.note("\n".join(rows))
        elif cmd == "/revert":
            # Read before reverting: revert_all clears the journal.
            observed = self.journal.observed_only()
            reverted = self.journal.revert_all()
            if reverted:
                self.r.note("reverted " + ", ".join(reverted))
            elif observed:
                # Not "nothing changed" — the very next line lists changes that
                # were found and deliberately left alone.
                self.r.note("nothing to revert (no attributed edits)")
            else:
                self.r.note(
                    "nothing to revert (no journaled edits — sandbox writes "
                    "outside the session directory are not tracked)"
                )
            if observed:
                # The Diff tab shows these, so silence here would read as
                # "everything is back" when it is not.
                self.r.note(
                    "left alone (found by scanning, not attributed to a file "
                    "tool — undo them yourself): " + ", ".join(observed)
                )
        elif cmd == "/log":
            self.r.note(f"trace: {self.trace_path}")
        elif cmd in {"/attach", "/attachments", "/detach", "/paste"}:
            import shlex

            from apodex.attachments import AttachmentError, format_size
            from apodex.clipboard import ClipboardError

            try:
                attachment_args = shlex.split(line)[1:]
                if cmd == "/attach":
                    if not attachment_args:
                        self.r.error("usage: /attach <path> [path ...]")
                    else:
                        added = self.attachments.attach_many(attachment_args)
                        self.r.note("attached: " + ", ".join(
                            item.relative_path for item in added
                        ))
                elif cmd == "/detach":
                    if len(attachment_args) != 1:
                        self.r.error("usage: /detach <attachment>")
                    else:
                        removed = self.attachments.detach(attachment_args[0])
                        self.r.note(f"detached {attachment_args[0]} ({removed} file(s))")
                elif cmd == "/attachments":
                    items = self.attachments.list()
                    self.r.note("no files attached" if not items else "attached files:\n" + "\n".join(
                        f"  {item.relative_path} · {format_size(item.size)} · {item.agent_path}"
                        for item in items
                    ))
                else:
                    from apodex.clipboard import paste_from_clipboard

                    result = await asyncio.to_thread(
                        paste_from_clipboard, self.attachments,
                    )
                    if result.kind == "attachments":
                        self.r.note("pasted attachments: " + ", ".join(result.attachments))
                    elif result.kind == "text":
                        self.r.note("clipboard contains text; paste it at the prompt")
                    else:
                        self.r.note(result.message or "clipboard is empty or unsupported")
            except (AttachmentError, ClipboardError, ValueError) as exc:
                self.r.error(str(exc))
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            self.r.set_verbose(self.verbose)
            self.r.note(f"verbose thinking = {self.verbose}")
        elif cmd == "/plan":
            self.plan_state.active = not self.plan_state.active
            if self.plan_state.active:
                self.r.note("▤ plan mode ON — edits locked until you approve a plan")
            else:
                self.r.note("plan mode OFF — edits allowed (with approval)")
        elif cmd == "/theme":
            from apodex.tui.themes import CLI_THEME_NAMES
            if not arg:
                self.r.note("usage: /theme " + "|".join(CLI_THEME_NAMES))
            elif arg not in CLI_THEME_NAMES:
                self.r.note("unknown theme: " + arg)
            else:
                from apodex.render import Renderer
                self.r = Renderer(theme=arg, verbose=self.verbose)
                self.user_settings.theme = arg
                self.user_settings.save()
                self.r.note(f"theme → {arg}")
        elif cmd in ("/auto", "/bypass"):
            self.approver.auto_approve = not self.approver.auto_approve
            self.user_settings.auto_approve = self.approver.auto_approve
            self.user_settings.save()
            self.r.note(f"bypass permission (auto-approve) = {self.approver.auto_approve}")
        elif cmd in ("/autome", "/auto-for-me"):
            self.approver.auto_for_me = not self.approver.auto_for_me
            self.user_settings.auto_for_me = self.approver.auto_for_me
            self.user_settings.save()
            self.r.note(f"auto for me (docker/trusted env mode) = {self.approver.auto_for_me}")
        elif cmd == "/cwd":
            if arg:
                try:
                    discarded_workspace = self._active_spill_workspace()
                    os.chdir(arg)
                    self.cwd = os.getcwd()
                    manager = getattr(self, "attachments", None)
                    update_root = getattr(manager, "set_source_root", None)
                    if callable(update_root):
                        update_root(self.cwd)
                    self._activate_session_workspace(self.session_id, self.cwd)
                    self._activate_session_outputs(self.session_id, self.cwd)
                    self._authorize_workspace(self.cwd)
                    self.history = []
                    self.workflow_turns = []
                    self.usage.clear_context()
                    self._cleanup_discarded_spill(discarded_workspace)
                    from apodex import fsguard
                    from apodex.changes import WorkspaceJournal
                    from apodex.env import environment_section
                    self.journal = WorkspaceJournal(self.cwd)
                    fsguard.clear()
                    self._env_section = environment_section(self.cwd, self.cfg.model)
                    self.r.note(f"cwd → {self.cwd} (context reset)")
                except Exception as exc:
                    self.r.error(f"cd failed: {exc}")
            else:
                self.r.note(f"cwd = {self.cwd}")
        elif cmd == "/model":
            models = self.models
            if not arg:
                if models:
                    lines = "\n".join(
                        f"  {i}. {m}" + ("  (current)" if m == self.cfg.model else "")
                        for i, m in enumerate(models, 1)
                    )
                    self.r.note(
                        f"model = {self.cfg.model}\n"
                        f"available (profile '{self.mode}'):\n{lines}\n"
                        f"switch with /model <name|number>"
                    )
                else:
                    self.r.note(f"model = {self.cfg.model}")
            else:
                # Accept a 1-based index into the profile list, a listed name, or
                # any free-form model id (with a note that it's off-list).
                target = arg
                if arg.isdigit() and 1 <= int(arg) <= len(models):
                    target = models[int(arg) - 1]
                try:
                    from apodex.env import environment_section
                    self.cfg.model = target
                    self._build_llm()
                    self._env_section = environment_section(self.cwd, self.cfg.model)
                    note = f"model → {target}"
                    if models and target not in models:
                        note += " (not in the profile's list)"
                    self.r.note(note)
                except Exception as exc:
                    self.r.error(f"model switch failed: {exc}")
        elif cmd == "/compact":
            if not self.history:
                self.r.note("nothing to compact (no conversation yet)")
            else:
                from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
                before = len(self.history)
                tool_results = sum(
                    message.get("role") == "tool" for message in self.history
                )
                self.history = await LLMSummaryCompactor(summary_llm=self.llm).compact(
                    self.history,
                    # An explicit compact is a checkpoint: compress every tool
                    # result first, then synthesize the complete conversation.
                    # No raw tail is retained, avoiding an old large result
                    # immediately filling the next request's context.
                    keep_recent=0,
                    compress_all_tool_results=True,
                )
                self.usage.compactions += 1
                self.r.note(
                    f"compacted {tool_results} tool results and summarized history: "
                    f"{before} → {len(self.history)} messages"
                )
        elif cmd in ("/cost", "/context"):
            self.r.note(self.usage.context_report(
                self.cfg.context_window,
                output_reserve=int(getattr(self.cfg, "max_tokens", 0) or 0),
            ))
        elif cmd == "/config":
            self.r.note(format_runtime_config_status(self.runtime_config_status()))
        elif cmd == "/init":
            self.r.rule()
            await self.run_task(_INIT_PROMPT)
        elif cmd == "/resume":
            await self._resume_picker()
        else:
            self.r.error(f"unknown command {cmd!r} (try /help)")
        return False

    async def _resume_picker(self) -> None:
        """Interactive list of saved sessions → load the chosen one."""
        sessions = list_saved_sessions()[:20]
        if not sessions:
            self.r.note("no saved sessions yet")
            return
        rows, choices = ["Recent sessions:"], []
        for i, saved in enumerate(sessions, 1):
            sid = saved["session_id"]
            name = f"{saved['name']} · " if saved.get("name") else ""
            meta = f"{name}{saved['mode']}, {saved['message_count']} msgs"
            rows.append(f"  {i}. {sid}  ({meta})")
            choices.append(sid)
        self.r.note("\n".join(rows))
        try:
            sel = input("  pick # (Enter to cancel) › ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not (sel.isdigit() and 1 <= int(sel) <= len(choices)):
            self.r.note("cancelled")
            return
        state = load_session_state(choices[int(sel) - 1])
        if state is None:
            self.r.error("could not load that session")
            return
        try:
            self.switch_session(state, fallback_id=choices[int(sel) - 1])
        except Exception as exc:
            self.r.error(f"could not resume session: {exc}")
            return
        self.r.note(f"resumed {self.session_id} ({len(self.history)} prior messages)")
