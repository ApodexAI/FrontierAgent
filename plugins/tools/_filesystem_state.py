"""The one trusted description of this run's filesystem.

Every path the agent can be told about or allowed to touch — the workspace it
works in, the directory its deliverables go to, the read-only inputs, the
scratch root, the shell variables naming them — is derived HERE, once, and read
from here by everyone else:

* ``_path_auth`` builds its allowed prefixes from it;
* ``write_file`` / ``create_file`` / ``file_editor`` refuse a read-only root
  through :func:`runtime_file_write_error`;
* ``_sandbox._build_tool_env`` projects :meth:`RuntimeFilesystemState.shell_env`
  into model-authored commands;
* ``_deliverable_policy`` and ``create_file`` take their roots from it;
* the system prompts name exactly the directories it reports.

Before this module those facts were re-derived in six places from a mix of
environment variables, scope metadata and backend guesses, so they could
disagree: a sub-agent could write a directory its main agent could not, native
mode's own outputs directory was under no allowed prefix, and the prompt could
name ``/outputs`` while the file tools used a relocated path.

Two properties are load-bearing:

**Only trusted code installs a state.** A workflow node or the CLI builds one
from its own configuration and installs it for the duration of the run.
``ExecutionScope`` metadata and request fields are workload INPUT; they may
describe a run, never widen it. When nothing is installed, the state is derived
from the process environment by the same builder, so there is still exactly one
derivation rather than a second, looser one.

**``local_filesystem`` says whose filesystem these paths name.** It is true only
when model commands run against this host's filesystem under these same
spellings (``native``, ``container``). For a remote backend the paths exist only
in the sandbox, so resolving them here — or granting them on this host — would
describe the wrong machine.

Deliberately stdlib-only and importing nothing from ``plugins`` or
``frontier_agent``: ``_path_auth``, ``_sandbox`` and ``_deliverable_policy`` all
import it, and it must not import them back.
"""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeFilesystemState",
    "current_filesystem_state",
    "install_filesystem_state",
    "reset_filesystem_state",
    "runtime_file_write_error",
    "state_for_sandbox_mode",
    "state_from_environment",
]

#: Canonical mount points. A deployment that cannot create top-level mounts
#: (macOS under ``native``) overrides them, which is why nothing may hardcode
#: these outside this module.
DEFAULT_WORKSPACE_DIR = "/workspace"
DEFAULT_OUTPUTS_DIR = "/outputs"
DEFAULT_INPUTS_DIR = "/inputs"

#: Backends whose commands run on THIS host's filesystem, under the same path
#: spellings the harness sees. ``bwrap`` is absent: its canonical
#: ``/workspace`` is a mount of a host directory with a different name, so the
#: two namespaces must not be conflated. Remote backends are absent for the
#: stronger reason that the paths do not exist here at all.
_LOCAL_FILESYSTEM_BACKENDS = frozenset({"native", "container"})

_MAX_PATH_CHARS = 4096


def _absolute(path: str | None, *, field: str) -> str:
    """Normalize one configured directory, or "" when it is unusable.

    A relative or malformed value is dropped rather than resolved against
    whatever cwd this process happens to have: the agent's cwd is not the
    harness's, so guessing produces a path that names a different directory in
    each namespace.
    """
    raw = str(path or "").strip()
    if not raw:
        return ""
    if "\x00" in raw or len(raw) > _MAX_PATH_CHARS:
        logger.warning("Ignoring malformed %s directory", field)
        return ""
    if not os.path.isabs(raw):
        logger.warning("Ignoring relative %s directory %r", field, raw)
        return ""
    return os.path.normpath(raw)


def _contains(root: str, candidate: str) -> bool:
    """Whether ``candidate`` is ``root`` or sits under it, component-aware.

    A string prefix test also accepts the sibling ``/outputs-old`` and
    ``/outputs/../etc``; ``..`` is collapsed before comparing.
    """
    if not root:
        return False
    root = os.path.normpath(root)
    candidate = os.path.normpath(candidate)
    return candidate == root or candidate.startswith(root.rstrip("/") + os.sep)


@dataclass(frozen=True)
class RuntimeFilesystemState:
    """This run's directories, and whether they name the local filesystem."""

    workspace: str = DEFAULT_WORKSPACE_DIR
    outputs: str = DEFAULT_OUTPUTS_DIR
    inputs: str = DEFAULT_INPUTS_DIR
    tmpdir: str = "/tmp"
    home: str = ""
    local_filesystem: bool = False

    def __post_init__(self) -> None:
        # Normalize through ``object.__setattr__``: the dataclass is frozen so
        # that a consumer cannot quietly retarget an installed state.
        for field in ("workspace", "outputs", "inputs", "tmpdir", "home"):
            object.__setattr__(
                self, field, _absolute(getattr(self, field), field=field),
            )

    def dirs(self) -> tuple[str, str, str]:
        """``(workspace, outputs, inputs)`` — the triple every consumer wants."""
        return self.workspace, self.outputs, self.inputs

    def write_roots(self) -> tuple[str, ...]:
        """Directories that BELONG to this run and may be written.

        ``inputs`` is absent by construction: it is a read-only mount, and that
        absence is the whole enforcement rather than a check each writer has to
        remember. ``tmpdir`` is absent too — see :meth:`scratch_roots`.
        """
        return tuple(dict.fromkeys(d for d in (self.workspace, self.outputs) if d))

    def scratch_roots(self) -> tuple[str, ...]:
        """:meth:`write_roots` plus the scratch root.

        Separate because the two answer different questions. A shell command
        writing ``$TMPDIR/x`` is ordinary, so the bash policy must not refuse
        it; but ``tmpdir`` is frequently the shared ``/tmp``, and authorizing
        that for the in-process file tools would hand them every path under it,
        which is far wider than the run's own directories.
        """
        return tuple(dict.fromkeys((*self.write_roots(), self.tmpdir)))

    def read_only_roots(self) -> tuple[str, ...]:
        """Directories the agent may read but never write."""
        return tuple(d for d in (self.inputs,) if d)

    def shell_env(self) -> dict[str, str]:
        """Variables naming these directories for model-authored commands.

        Without them a command has no way to find the directory the prompt told
        it to write to except by hardcoding a path that is wrong in some
        deployments. Empty values are omitted so a shell test on them works.
        """
        env = {
            "WORKSPACE_DIR": self.workspace,
            "OUTPUT_DIR": self.outputs,
            "INPUT_DIR": self.inputs,
            "TMPDIR": self.tmpdir,
        }
        return {key: value for key, value in env.items() if value}

    # ── path questions ───────────────────────────────────────────────────

    def _spellings(self, path: str) -> tuple[str, ...]:
        """``path`` as written, plus its resolved form on a local filesystem.

        Resolving is what makes protection follow identity rather than
        spelling, so a symlink into the read-only inputs is caught. It is done
        only for a local state and only for an absolute path: the shell's cwd is
        not this process's, and a remote path must never be interpreted by this
        host (``os.path.realpath`` of a path that exists only in the sandbox
        answers about the wrong machine).
        """
        written = os.path.normpath(path)
        if not self.local_filesystem or not os.path.isabs(path):
            return (written,)
        try:
            resolved = os.path.realpath(path)
        except OSError:
            # An unresolvable path keeps the textual answer: a failed lookup
            # must never drop a check.
            return (written,)
        return (written,) if resolved == written else (written, resolved)

    def write_error(self, path: str) -> str | None:
        """Why this run must not write ``path``, or ``None``.

        Only a read-only root is refused here. Whether a path is otherwise
        reachable is ``_path_auth``'s question, which has the workspace root,
        the service checkout and the spill store to consider as well.
        """
        raw = str(path or "").strip()
        if not raw:
            return "file path is required"
        if not os.path.isabs(raw) and self.workspace:
            raw = os.path.join(self.workspace, raw)
        for root in self.read_only_roots():
            roots = self._spellings(root) if self.local_filesystem else (root,)
            if any(
                _contains(root_spelling, candidate)
                for root_spelling in roots
                for candidate in self._spellings(raw)
            ):
                return (
                    f"{path!r} is inside the read-only input directory "
                    f"{root}. Write to {self.outputs or self.workspace} instead."
                )
        return None

    def with_backend(self, sandbox: object) -> RuntimeFilesystemState:
        """This state refreshed from facts the live backend exposes.

        Only what the backend states about itself is taken; this never probes
        the host. A backend that says nothing leaves the state unchanged.
        """
        dirs = getattr(sandbox, "filesystem_dirs", None)
        updates: dict[str, object] = {}
        if isinstance(dirs, tuple) and len(dirs) == 3:
            updates.update(workspace=dirs[0], outputs=dirs[1], inputs=dirs[2])
        tmpdir = getattr(sandbox, "tool_tmpdir", "")
        if tmpdir:
            updates["tmpdir"] = tmpdir
        return replace(self, **updates) if updates else self  # type: ignore[arg-type]


def state_from_environment(
    *, backend: str = "", local_filesystem: bool | None = None,
) -> RuntimeFilesystemState:
    """Build a state from this process's configuration.

    THE derivation: both the explicit install path and the fallback for code
    running outside one go through here, so there is no second, looser answer
    to "where does this run write".
    """
    resolved_backend = (backend or os.environ.get("SANDBOX_BACKEND", "")).strip().lower()
    if os.environ.get("APODEX_IN_NATIVE", "").strip() == "1":
        resolved_backend = resolved_backend or "native"
    if local_filesystem is None:
        local_filesystem = resolved_backend in _LOCAL_FILESYSTEM_BACKENDS
    return RuntimeFilesystemState(
        workspace=os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "") or DEFAULT_WORKSPACE_DIR,
        outputs=os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "") or DEFAULT_OUTPUTS_DIR,
        inputs=os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "") or DEFAULT_INPUTS_DIR,
        tmpdir=os.environ.get("TMPDIR", "") or "/tmp",
        home=os.environ.get("HOME", "") if local_filesystem else "",
        local_filesystem=local_filesystem,
    )


def state_for_sandbox_mode(
    sandbox_mode: str,
    *,
    workspace: str = "",
    outputs: str = "",
    inputs: str = "",
    tmpdir: str = "",
) -> RuntimeFilesystemState:
    """The state for a workflow that has just resolved its own directories.

    ``container`` / ``native`` run commands against the host directories the
    caller passes, so those are the paths the model is told about, the paths the
    file tools authorize, and the paths worth resolving for symlink identity.

    Every other mode presents the CANONICAL mount points to the model (bwrap
    binds the host directories at ``/workspace`` and friends; a remote backend
    has its own filesystem), so the host spellings deliberately do not appear in
    the state — naming them would authorize host writes on the strength of a
    path that describes a different namespace. The physical directories stay the
    caller's business: it is the one that mounts them.
    """
    mode = (sandbox_mode or "").strip().lower()
    if mode in _LOCAL_FILESYSTEM_BACKENDS:
        return RuntimeFilesystemState(
            workspace=workspace or DEFAULT_WORKSPACE_DIR,
            outputs=outputs or DEFAULT_OUTPUTS_DIR,
            inputs=inputs or DEFAULT_INPUTS_DIR,
            tmpdir=tmpdir or os.environ.get("TMPDIR", "") or "/tmp",
            home=os.environ.get("HOME", ""),
            local_filesystem=True,
        )
    return RuntimeFilesystemState(
        workspace=DEFAULT_WORKSPACE_DIR,
        outputs=DEFAULT_OUTPUTS_DIR,
        inputs=DEFAULT_INPUTS_DIR,
        tmpdir=tmpdir or "/tmp",
        local_filesystem=False,
    )


# ── installation ─────────────────────────────────────────────────────────

_state: contextvars.ContextVar[RuntimeFilesystemState | None] = contextvars.ContextVar(
    "frontier_agent_runtime_filesystem", default=None,
)


def install_filesystem_state(state: RuntimeFilesystemState) -> contextvars.Token:
    """Install ``state`` for this context. Only trusted code may call this.

    Returns a token for :func:`reset_filesystem_state`. A contextvar, so a
    sub-agent running as a child task inherits its parent's state until it
    installs its own, and resetting cannot leak into a sibling.
    """
    if not state.workspace:
        raise ValueError("a runtime filesystem state needs an absolute workspace")
    return _state.set(state)


def reset_filesystem_state(token: contextvars.Token) -> None:
    """Restore the previously installed state."""
    _state.reset(token)


def current_filesystem_state() -> RuntimeFilesystemState:
    """The installed state, else one derived from the environment."""
    state = _state.get()
    return state if state is not None else state_from_environment()


def installed_filesystem_state() -> RuntimeFilesystemState | None:
    """The installed state, or ``None`` — for code that must tell them apart."""
    return _state.get()


def runtime_file_write_error(path: str | Path) -> str | None:
    """Why this run must not write ``path``, or ``None``.

    The one call every file-writing tool makes, so the read-only contract is
    enforced in a single place instead of per writer.
    """
    return current_filesystem_state().write_error(str(path))
