"""Filesystem containment for the demo, enforced at the tool-call boundary.

The runtime's own path guard (``plugins/tools/_path_auth``) is fail-closed, but
several tools treat a denial as a cue to retry through the *sandbox* writer
rather than refusing — ``plugins/tools/write_file.py`` does exactly that. In the
demo's configuration the sandbox is a ``CurrentSandbox``: a plain working
directory with no namespace. So a relative path carrying enough ``..`` segments
writes outside the visitor's own session.

Rather than patch shared tool code, this closes the gap at a boundary the
runtime already offers observers: ``on_tool_call`` may return a
:class:`ToolCallIntervention` whose ``skip_with_result`` replaces the call with
a message. The agent sees an ordinary tool error telling it where it may write,
and carries on.

Relative paths are resolved against the session **workspace**, because that is
what both mechanisms the tools use do: ``_path_auth._candidate_paths`` resolves
against the configured workspace root, and the sandbox's working directory is
that same root.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deploy.huggingface.security import Redactor, redact_deep
from frontier_agent.core.loop_types import (
    BaseObserver,
    ToolCallIntervention,
    TurnContext,
)

logger = logging.getLogger(__name__)

#: Tool arguments that name a filesystem location. The demo's tools take paths
#: only through these names, so checking them covers every path they accept.
PATH_ARGUMENTS: frozenset[str] = frozenset({
    "path", "file_path", "save_to", "out", "output_path", "dir", "directory",
    "root", "ops_path", "image_path", "program_path",
})

#: Arguments that always name a *destination*, whatever tool they appear on —
#: ``read_file(save_to=…)`` writes a file despite being a read tool.
WRITE_ARGUMENTS: frozenset[str] = frozenset({
    "save_to", "out", "output_path",
})

#: Tools whose primary path argument is a destination. An unknown tool with a
#: path argument is treated as a writer too: guessing "read" for something we do
#: not recognise would fail open.
WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "create_file", "download_file",
    "file_editor_create", "file_editor_str_replace",
})

#: Tools known to only ever read through their path argument.
READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "read_text", "glob_search", "grep_search", "file_editor_view",
    "view_image",
})


class PathContainmentObserver(BaseObserver):
    """Refuses any tool call naming a path outside the session's own tree.

    Reads and writes get different allowances. ``inputs/`` is readable but must
    never be written — task inputs are supplied *to* the run, and letting the
    agent rewrite them would contradict the documented boundary — while
    ``state/`` (traces, bookkeeping) is neither. So a single allow-list shared
    by both access modes is not enough: the mode has to be decided per argument.

    ``critical = True`` is required, not cosmetic: ``notify_observers`` awaits
    critical observers inline and fans the rest out as background tasks, so a
    permission check registered as non-critical would be advisory only.
    (``on_tool_call`` is always awaited, but the flag keeps the guarantee
    explicit and survives a future dispatch change.)
    """

    critical = True

    def __init__(
        self,
        *,
        workspace: Path,
        read_roots: Iterable[Path] = (),
        session_label: str = "",
    ) -> None:
        self._workspace = _absolute(workspace)
        #: Writes may only land in the workspace (which contains ``outputs/``).
        self._write_roots = (self._workspace,)
        #: Reads may additionally reach the read-only roots.
        self._read_roots = (
            self._workspace, *(_absolute(r) for r in read_roots),
        )
        self._label = session_label
        self.denied: list[tuple[str, str, str]] = []

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        name, arguments = _call_arguments(tool_call)
        if not isinstance(arguments, dict):
            return None
        for key, value in arguments.items():
            if key not in PATH_ARGUMENTS or not isinstance(value, str) or not value:
                continue
            writes = _is_write(name, key)
            reason = self._reject_reason(value, writes=writes)
            if reason is None:
                continue
            self.denied.append((name, key, value))
            logger.info(
                "demo containment: %s denied %s=%r (%s) for session %s (%s)",
                name, key, value, "write" if writes else "read",
                self._label or "?", reason,
            )
            return ToolCallIntervention(
                skip_with_result=self._message(key, value, reason, writes=writes),
            )
        return None

    # -- policy ----------------------------------------------------------

    def _reject_reason(self, raw: str, *, writes: bool) -> str | None:
        """``None`` when ``raw`` is acceptable, else why it is not."""
        candidate = Path(raw.strip())
        # ``..`` is refused even when it would resolve back inside: a task never
        # legitimately needs to climb out and back, and permitting it would make
        # the verdict depend on which directory happens to be current. Checked
        # on the *raw* argument, before alias mapping: ``resolve_runtime_path``
        # normalises as it rewrites, so ``/outputs/../x`` would otherwise reach
        # the containment test with the traversal already collapsed away.
        if ".." in candidate.parts:
            return "parent-directory traversal"
        resolved = _absolute(_resolved_alias(candidate), base=self._workspace)
        roots = self._write_roots if writes else self._read_roots
        for root in roots:
            if resolved == root or root in resolved.parents:
                return None
        if writes and any(
            resolved == r or r in resolved.parents for r in self._read_roots
        ):
            return "that directory is read-only"
        return "outside the session directory"

    def _message(self, key: str, value: str, reason: str, *, writes: bool) -> str:
        allowed = (
            f"write under {self._workspace}" if writes
            else "read under " + ", ".join(str(r) for r in self._read_roots)
        )
        return (
            f"Error: {key}={value!r} was refused ({reason}). This run may only "
            f"{allowed}. Write deliverables to {self._workspace / 'outputs'}; "
            "use a relative path such as 'outputs/report.md' or an absolute "
            "path inside that directory."
        )


def _is_write(tool_name: str, argument: str) -> bool:
    """Whether ``argument`` on ``tool_name`` names a destination.

    Fails closed: an argument that always writes counts as a write whatever the
    tool, and an unrecognised tool with a path argument is assumed to write.
    """
    if argument in WRITE_ARGUMENTS:
        return True
    name = (tool_name or "").lower()
    if name in WRITE_TOOLS:
        return True
    return name not in READ_TOOLS


class SecretArgumentObserver(BaseObserver):
    """Strips configured secrets out of tool arguments before the tool runs.

    Events and answers were already redacted, but a tool receives the model's
    arguments verbatim — so an endpoint (which holds the API key, being the
    party that authenticates with it) could have the agent write that key into a
    deliverable and hand it to every visitor who downloads the file.

    Rewriting rather than refusing keeps the run usable: the agent gets its file
    written, with ``***REDACTED***`` where the credential was.
    """

    critical = True

    def __init__(self, redactor: Redactor, *, session_label: str = "") -> None:
        self._redactor = redactor
        self._label = session_label
        self.redacted: list[str] = []

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        if not self._redactor.literals:
            return None
        name, arguments = _call_arguments(tool_call)
        if not isinstance(arguments, dict):
            return None
        cleaned = redact_deep(arguments, self._redactor)
        if cleaned == arguments:
            return None
        self.redacted.append(name)
        logger.warning(
            "demo secret guard: redacted a configured secret out of %s "
            "arguments for session %s", name, self._label or "?",
        )
        return ToolCallIntervention(rewrite_args=cleaned)


def _resolved_alias(path: Path) -> Path:
    """Map the canonical sandbox aliases onto this session's real directories.

    Every file tool's own description teaches the model ``/workspace``,
    ``/outputs`` and ``/inputs``, and since ``plugins/tools/_sandbox.
    resolve_runtime_path`` those aliases are honoured in native mode too — the
    tool rewrites them to whatever ``FRONTIER_AGENT_*_DIR`` names, which for
    this demo is the visitor's own session tree. Checking the literal alias
    instead would refuse a path the runtime was about to resolve *correctly*,
    costing a turn on a call that was never an escape attempt.

    Falls back to the argument unchanged, which is the stricter verdict: the
    alias then fails the containment test as it did before this mapping existed.
    """
    try:
        from plugins.tools._sandbox import resolve_runtime_path

        return Path(resolve_runtime_path(str(path)))
    except Exception:  # pragma: no cover - defensive: never fail open
        logger.debug("demo containment: alias resolution unavailable", exc_info=True)
        return path


def _absolute(path: Path | str, *, base: Path | None = None) -> Path:
    """Normalise to an absolute path without requiring it to exist.

    ``os.path.normpath`` collapses ``.`` and ``..`` textually; ``resolve`` then
    follows any symlinks that do exist, so a link cannot smuggle the target out
    of the allowed tree.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or Path.cwd()) / candidate
    return Path(os.path.normpath(str(candidate))).resolve()


def _call_arguments(tool_call: Mapping[str, Any]) -> tuple[str, Any]:
    """``(tool_name, parsed_arguments)`` for either tool-call shape.

    Observers see the loop's *parsed* form — ``{"name", "args", "id"}`` (see
    ``agent_loop._execute_tool_calls``) — while the provider wire form is
    ``{"function": {"name", "arguments"}}``. Both are accepted so the check
    cannot be silently bypassed by whichever shape reaches it.
    """
    function = tool_call.get("function") or {}
    name = str(function.get("name") or tool_call.get("name") or "")
    for raw in (
        tool_call.get("args"),
        function.get("arguments"),
        tool_call.get("arguments"),
    ):
        if isinstance(raw, dict):
            return name, raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return name, parsed
    return name, None


__all__ = [
    "PATH_ARGUMENTS",
    "READ_TOOLS",
    "WRITE_ARGUMENTS",
    "WRITE_TOOLS",
    "PathContainmentObserver",
    "SecretArgumentObserver",
]
