"""Where apodex's ``bash`` actually runs.

Four strategies, resolved once at startup:

``native``
    The default for Linux host installs. Commands run as the current host user,
    while mutable runtime state and package-manager caches live below
    ``<workspace>/.apodex/runtime/native``. This is a convenience boundary, not an OS
    security boundary.

``bwrap``
    An explicit Linux isolation option. The command runs inside a bubblewrap
    jail: the working directory is bound read-write **at its own path**, the
    system is read-only, and the rest of ``$HOME`` is not in the mount
    namespace at all. Path identity matters — the model writes
    ``/home/me/repo/src/x.py`` and that is the same file inside and outside,
    so its paths, tracebacks and diffs all line up.

``host``
    No namespace: the command runs as you, in your working directory. This is
    what the interactive approval gate was designed around, but a mistake the
    gate approves has your whole filesystem in reach. Requires an explicit
    opt-in, because silently degrading a sandbox is how you end up believing
    in a boundary that is not there.

``container``
    We are already inside a container the CLI launched (macOS path, see
    :mod:`apodex.docker`). The container *is* the boundary; nesting bwrap
    inside it buys nothing and most container runtimes forbid it anyway.

Order of resolution: explicit argument → configured backend (``APODEX_SANDBOX``
or ``SANDBOX_BACKEND``) → in-container marker → native marker → Linux native
default. macOS is normally handled by the CLI's Docker-or-native selection
before this resolver is called.

``SANDBOX_BACKEND`` is read here as well as by :mod:`plugins.tools._sandbox`
precisely so the two layers cannot disagree: a process told "bubblewrap jail"
in the banner must not execute through the native backend, and our own compose
files configure the boundary with that variable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BWRAP = "bwrap"
HOST = "host"
CONTAINER = "container"
NATIVE = "native"

_IN_CONTAINER_ENV = "APODEX_IN_CONTAINER"
_STRATEGY_ENV = "APODEX_SANDBOX"
_BACKEND_ENV = "SANDBOX_BACKEND"

# ``SANDBOX_BACKEND`` has a wider vocabulary than this module's strategies:
# ``local`` is the tool layer's older spelling of bwrap, and ``e2b`` selects a
# cloud executor on an orthogonal axis, so it names no local strategy and is
# deliberately absent here.
_STRATEGY_BY_BACKEND = {
    BWRAP: BWRAP,
    "local": BWRAP,
    HOST: HOST,
    CONTAINER: CONTAINER,
    NATIVE: NATIVE,
}


class SandboxUnavailable(RuntimeError):
    """No usable isolation, and the user has not opted into running without."""


@dataclass(frozen=True)
class Strategy:
    name: str
    reason: str

    @property
    def isolated(self) -> bool:
        return self.name in (BWRAP, CONTAINER)

    def describe(self) -> str:
        if self.name == BWRAP:
            return "bubblewrap jail (working directory writable, system read-only)"
        if self.name == CONTAINER:
            return "container (the whole CLI runs inside it)"
        if self.name == NATIVE:
            return "native workspace runtime (not an OS sandbox)"
        return "NO SANDBOX — commands run as you, on your filesystem"


def _bwrap_usable() -> tuple[bool, str]:
    """Whether the shared bubblewrap backend reports itself usable.

    Delegates to :func:`plugins.tools._sandbox.bwrap_available`, which probes
    with the real argument list rather than checking for the binary — a host
    can ship ``bwrap`` and still refuse to mount a fresh procfs, and the probe
    is the only thing that catches that.
    """
    try:
        from plugins.tools._sandbox import bwrap_available
    except Exception as exc:
        return False, f"sandbox backend unavailable ({exc})"
    if bwrap_available():
        return True, "bubblewrap available"
    return False, (
        "bubblewrap is not usable here (no bwrap binary, or the host forbids "
        "user namespaces / mounting a fresh /proc — common inside an "
        "unprivileged container)"
    )


def configured_backend() -> tuple[str, str]:
    """The backend named by configuration, and the setting that named it.

    Returns ``("", "")`` when nothing names one — an unset variable and an
    explicit ``auto`` both mean "you decide". ``APODEX_SANDBOX`` wins over
    ``SANDBOX_BACKEND`` because it is the CLI's own switch.

    Callers use this to answer two different questions: which strategy to run
    (via :data:`_STRATEGY_BY_BACKEND`) and, in the CLI, whether *any* explicit
    choice exists — a default must never quietly replace a configured backend,
    which is how a user ends up trusting a boundary that is not there.
    """
    for env in (_STRATEGY_ENV, _BACKEND_ENV):
        value = (os.environ.get(env) or "").strip().lower()
        if value and value != "auto":
            return value, env
    return "", ""


def resolve_strategy(requested: str | None = None) -> Strategy:
    """Pick the execution strategy for this process.

    ``requested`` is used by explicit CLI switches such as ``--bwrap`` and
    takes precedence over the configured backend.
    """
    if requested is not None:
        forced = requested.strip().lower()
        source = f"--{forced}"
    else:
        backend, env = configured_backend()
        forced = _STRATEGY_BY_BACKEND.get(backend, "")
        source = f"{env}={backend}"

    if forced in (BWRAP, HOST, CONTAINER, NATIVE):
        if forced == BWRAP:
            ok, why = _bwrap_usable()
            if not ok:
                raise SandboxUnavailable(
                    f"{source} selects bubblewrap, but {why}."
                )
        return Strategy(forced, source)

    if os.environ.get(_IN_CONTAINER_ENV, "").strip() == "1":
        return Strategy(CONTAINER, "running inside the CLI's own container")

    if os.environ.get("APODEX_IN_NATIVE", "").strip() == "1":
        return Strategy(NATIVE, "workspace-local native runtime")

    if sys.platform == "darwin":
        raise SandboxUnavailable(
            "macOS has no bubblewrap. Run the CLI in Docker instead — that is "
            "the supported macOS path and `apodex` does it for you:\n"
            "    apodex --docker [args]\n"
            "To use the workspace-local host runtime instead, pass --native."
        )

    if sys.platform.startswith("linux"):
        return Strategy(NATIVE, "default Linux host runtime")

    # Only Linux gets the native default, because only Linux reaches
    # ``prepare_native_runtime`` in the CLI. Announcing a workspace runtime
    # that was never prepared would point the user at a boundary that does not
    # exist — commands would run against their real $HOME and caches.
    raise SandboxUnavailable(
        f"no execution strategy is configured for platform {sys.platform!r}. "
        f"Set {_BACKEND_ENV} to one of bwrap, container, or native (or pass "
        "--no-sandbox to accept unsandboxed execution)."
    )


_active: Strategy | None = None


def active_strategy() -> Strategy:
    """The strategy for this process, resolved on first use and then cached.

    ``set_active_strategy`` is what the CLI calls after it has printed the
    banner; tools call this and get the same answer for the whole session, so
    a command can never silently run under different isolation than the one
    the user was told about.
    """
    global _active
    if _active is None:
        _active = resolve_strategy()
    return _active


def set_active_strategy(strategy: Strategy) -> None:
    global _active
    _active = strategy


# ── execution ────────────────────────────────────────────────────────────

_bwrap_sandbox = None  # one jail per process; commands are cheap, setup is not


def _get_bwrap_sandbox(cwd: str) -> Any:
    """A ``BwrapSandbox`` whose jail exposes *cwd* at its own absolute path.

    ``workspace=`` binds a directory at ``/workspace`` and runs there, but a
    local repository has to keep its real path for the model's paths to mean
    anything, so *cwd* is bound a second time at itself and each command cds
    into it.
    """
    global _bwrap_sandbox
    if _bwrap_sandbox is None:
        from plugins.tools._sandbox import BwrapSandbox
        real = str(Path(cwd).expanduser().resolve())
        _bwrap_sandbox = BwrapSandbox(workspace=real, binds=((real, real, False),))
    return _bwrap_sandbox


async def run_shell(
    command: str, cwd: str, timeout: int, strategy: Strategy,
) -> tuple[int, str, str]:
    """Run *command*, returning ``(exit_code, stdout, stderr)``."""
    if strategy.name == BWRAP:
        sandbox = _get_bwrap_sandbox(cwd)
        real = str(Path(cwd).expanduser().resolve())
        wrapped = f"cd {shlex.quote(real)} && {command}"
        # ``allow_net=True``: the agent legitimately installs packages, runs
        # tests that hit localhost, and uses git over the network. The jail is
        # a filesystem boundary here, not a network one.
        result = await asyncio.to_thread(
            sandbox.commands.run, wrapped, timeout=timeout, allow_net=True,
        )
        return (
            int(getattr(result, "exit_code", 0) or 0),
            getattr(result, "stdout", "") or "",
            getattr(result, "stderr", "") or "",
        )

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )
