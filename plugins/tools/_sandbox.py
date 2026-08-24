"""Sandbox management — fail-closed E2B or bubblewrap isolation."""

from __future__ import annotations

import codecs
import collections
import contextlib
import contextvars
import functools
import io
import logging
import math
import os
import select
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from frontier_agent.infra.usage_meter import (
    close_meter_span,
    open_meter_span,
    record_api_request,
    set_meter_gauge,
)
from plugins.tools._exec_cgroup import (
    MEM_MAX_ENV as _EXEC_CG_MEM_MAX_ENV,
)
from plugins.tools._exec_cgroup import (
    ExecCgroup,
    create_exec_cgroup,
)
from plugins.tools._path_auth import _path_within

logger = logging.getLogger(__name__)

_sandbox = None  # E2B Sandbox | BwrapSandbox | task override | None
# Serializes shared-singleton creation. Without it, a burst of concurrent
# sub-agents all see ``_sandbox is None``, each call ``Sandbox.create()``, and
# overwrite the global — spinning up (and leaking) N sandboxes instead of one.
_sandbox_lock = threading.Lock()

# ── Per-task sandbox override ──────────────────────────────────────────
# Used by SWE benchmark for per-task E2B isolation. When set, get_sandbox()
# returns this instead of the shared singleton.

_task_sandbox: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_task_sandbox", default=None,
)


def set_task_sandbox(sandbox: Any) -> contextvars.Token:
    """Set a per-task sandbox for the current async context.

    Returns a token that must be passed to clear_task_sandbox().
    """
    return _task_sandbox.set(sandbox)


def clear_task_sandbox(token: contextvars.Token) -> None:
    """Restore the previous sandbox context."""
    _task_sandbox.reset(token)


# ── Shared result/error types ──────────────────────────────────────────


@dataclass
class _CommandResult:
    """Result of a sandbox command execution."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class SandboxError(RuntimeError):
    """Base class for sandbox configuration and availability failures."""


class SandboxConfigurationError(SandboxError):
    """The configured isolation backend is invalid or incomplete."""


class SandboxUnavailableError(SandboxError):
    """The configured isolation backend cannot be created."""


class SandboxCapacityError(SandboxUnavailableError):
    """The E2B pool cannot provide an isolated sandbox in time."""


# Thread-pool caps for native math libs. Without these, every concurrent
# local bwrap exec spins up one BLAS/OpenMP thread *per core* — with dozens
# of sub-agents running code at once that oversubscribes CPU and
# inflates memory. Pinning to 1 keeps aggregate footprint predictable; heavy
# parallelism comes from running many agents, not many threads
# per exec.
_THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# ── Minimal environment for model-authored subprocesses ───────────────
# A denylist is not a security boundary: credentials also arrive under names
# such as DATABASE_URL / *_DSN, while Kubernetes injects service topology,
# pod identity, and internal hostnames under entirely non-secret-looking keys.
# Build the child environment from zero instead. The harness process keeps its
# full environment for LLM/search clients; only model-authored bash/python sees
# this allowlisted projection.
_TOOL_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    # Runtime/toolchain configuration that contains paths, never credentials.
    # Some vendor images install LibreOffice beside private shared libraries,
    # so its launcher needs the image-provided dynamic-library search path.
    "LD_LIBRARY_PATH",
    # Set by the images to a baked cache. Without it tiktoken re-downloads its
    # BPE files on every in-sandbox call — the old substring denylist ate this
    # name too (it contains "TOKEN"), so it has been silently missing for a while.
    "TIKTOKEN_CACHE_DIR",
    # Global document-generation modules baked into the serving images.
    "NODE_PATH",
    "MATPLOTLIBRC",
    "MPLCONFIGDIR",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # Non-secret runtime root for the AmberTools CLI copied into the unified
    # react/stateful image. antechamber uses it to locate its data files.
    "AMBERHOME",
    # Egress proxy. Without these, every outbound call a model command makes on
    # a proxied deployment fails — including the ``pip install`` that PIP_TARGET
    # above exists to keep working. Both cases are listed because most tools
    # read only one of them.
    #
    # Caveat for deployments: a proxy URL of the form
    # ``http://user:pass@proxy`` puts those credentials into the model's
    # environment. Stripping the userinfo here would just break authenticated
    # proxies, so prefer a credential-free proxy URL (or a transparent/PAC
    # proxy) wherever model commands run.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
})
_TOOL_ENV_EXTRA_ALLOWLIST = "FRONTIER_AGENT_TOOL_ENV_ALLOWLIST"

# ── Unprivileged identity for model-authored subprocesses ─────────────
# Clearing the child's own environment is necessary but not sufficient: in
# container mode the model's bash/python shares the harness's PID namespace, so
# ``cat /proc/<harness-pid>/environ`` reads the real provider credentials
# straight out of the parent. That file is ``-r--------`` and additionally
# gated by ptrace_may_access, so the boundary is simply *not being the same
# uid*. The task container runs as root, which means dropping to a dedicated
# unprivileged uid needs no capability, no namespace, and no runtime policy
# change — unlike bubblewrap, which the production runtime does not permit.
# ── Sandbox profile: who are we protecting from whom ──────────────────────
#
# ``local`` (default): one user, their own machine, their own keys. The jail is
# a *filesystem* boundary — the working directory read-write, the system
# read-only, and $HOME absent, so a mistaken command cannot reach ~/.ssh or
# delete an unrelated project.
#
# ``service``: a shared worker running someone else's task. Now the harness
# process itself holds credentials that the task must not read, so the jail
# additionally needs a PID namespace with its own procfs. That costs privileges
# many runtimes will not grant, which is the right trade for a service and the
# wrong one for a laptop.
#
# Set ``SANDBOX_PROFILE=service`` when deploying this as a multi-tenant backend.
_SANDBOX_PROFILE_ENV = "SANDBOX_PROFILE"
_PROFILE_LOCAL = "local"
_PROFILE_SERVICE = "service"


def _sandbox_profile() -> str:
    """``local`` (default) or ``service`` — see the note above."""
    raw = (os.environ.get(_SANDBOX_PROFILE_ENV) or "").strip().lower()
    return _PROFILE_SERVICE if raw == _PROFILE_SERVICE else _PROFILE_LOCAL


_TOOL_USER_ENV = "FRONTIER_AGENT_TOOL_USER"
_DEFAULT_TOOL_USER = "agent-tool"
_REQUIRE_TOOL_USER_ENV = "FRONTIER_AGENT_REQUIRE_TOOL_USER"
# Model-installed packages: /opt/venv is root-owned and must stay that way (a
# writable venv lets the model plant code the harness itself later imports), and
# pip refuses ``--user`` inside a virtualenv.
#
# The images therefore bake an OVERLAY VENV owned by the tool user
# (``_DEFAULT_TOOL_VENV``): its site-packages is writable, and a ``.pth`` there
# puts the baked venv's site-packages on ``sys.path``. With its ``bin`` first on
# PATH the model gets a ``python3``/``pip`` pair that sees every baked package as
# INSTALLED, so ``pip install matplotlib`` answers "already satisfied" and only
# genuinely-missing packages are fetched — into the overlay, where they import.
#
# The previous approach (``PIP_TARGET`` + ``PYTHONPATH`` into the tool user's
# HOME) kept installs working but re-downloaded what the image already had:
# ``PIP_TARGET`` is ``pip install --target``, which resolves against the target
# directory alone and ignores distributions installed elsewhere on sys.path —
# and that directory is empty in every fresh task container. On an image whose
# venv held both, ``pip install matplotlib python-pptx`` measured 45 downloads /
# 181 MB with 0 "already satisfied", and the fetched copies then SHADOWED the
# baked, self-tested versions via PYTHONPATH. It remains the fallback below for
# images built before the overlay existed — degraded, never fatal, matching how
# the tool-user drop itself rolls out.
#
# What the tool user may write there is the WHOLE overlay, ``bin/`` included:
# pip has to land console scripts, so narrowing it would break the installs this
# exists for. The overlay is therefore container-lifetime state the model can
# shape (replace ``bin/pip``, drop a ``sitecustomize.py``) where the PIP_TARGET
# directory was per-task HOME. No privilege gain — same uid, and the harness
# imports only /opt/venv — but if a container ever serves more than one
# request, that state carries across them.
_TOOL_SITE_SUBDIR = ".local/site-packages"
_TOOL_VENV_ENV = "FRONTIER_AGENT_TOOL_VENV"
_DEFAULT_TOOL_VENV = "/opt/tool-venv"
# Read-only runtime toolchains outside the standard /usr tree. The unified
# react/stateful image places AmberTools under /opt/chem; container mode sees
# it directly, while bwrap must bind it explicitly or `antechamber` vanishes.
# Deployments can add more colon-separated roots without changing the sandbox
# code. Only existing absolute directories are accepted.
_EXTRA_TOOL_ROOTS_ENV = "FRONTIER_AGENT_EXTRA_TOOL_ROOTS"
_DEFAULT_EXTRA_TOOL_ROOTS = "/opt/chem"


class _ToolIdentity(NamedTuple):
    name: str
    uid: int
    gid: int


@functools.cache
def _tool_identity_for(name: str, euid: int) -> _ToolIdentity | None:
    """Resolve *name* to a uid/gid usable for dropping privileges.

    Returns ``None`` (with a warning) when privilege separation cannot apply:
    a non-root harness cannot ``setuid``, and an image without the account
    cannot host it. Callers must treat that as degraded, never as fatal —
    failing closed here would take down every task on a runtime that simply
    has not rolled out the new base image yet.
    """
    if not name or name.lower() in {"off", "none", "disabled"}:
        return None
    if os.name != "posix":
        return None
    if euid != 0:
        logger.warning(
            "Tool-user isolation inactive: harness euid=%s cannot setuid to %r; "
            "model commands share the harness uid and can read "
            "/proc/<pid>/environ", euid, name,
        )
        return None
    try:
        import pwd

        entry = pwd.getpwnam(name)
    except KeyError:
        logger.error(
            "Tool-user isolation inactive: account %r missing from the image; "
            "model commands run as root and can read the harness credentials "
            "from /proc/<pid>/environ. Add it (useradd -r %s) or set %s=off "
            "to acknowledge.", name, name, _TOOL_USER_ENV,
        )
        return None
    except Exception as exc:  # pragma: no cover - platform specific
        logger.warning("Tool-user lookup failed for %r: %s", name, exc)
        return None
    if entry.pw_uid == 0:
        logger.error(
            "Tool-user %r resolves to uid 0; refusing to treat it as a "
            "privilege boundary", name,
        )
        return None
    return _ToolIdentity(name, entry.pw_uid, entry.pw_gid)


def tool_identity_or_none() -> _ToolIdentity | None:
    """The unprivileged identity model commands run as, or ``None``.

    Never raises — for best-effort callers (permission prep) that must not turn
    a missing tool account into a failed task.
    """
    name = (os.environ.get(_TOOL_USER_ENV) or _DEFAULT_TOOL_USER).strip()
    return _tool_identity_for(name, os.geteuid() if os.name == "posix" else -1)


def tool_identity() -> _ToolIdentity | None:
    """As :func:`tool_identity_or_none`, honouring the opt-in strict switch."""
    identity = tool_identity_or_none()
    if identity is None and _require_tool_user():
        name = (os.environ.get(_TOOL_USER_ENV) or _DEFAULT_TOOL_USER).strip()
        raise SandboxUnavailableError(
            f"{_REQUIRE_TOOL_USER_ENV} is set but the unprivileged tool user "
            f"{name!r} is unavailable; model commands would run with the "
            "harness's own uid and could read its credentials from "
            "/proc/<pid>/environ"
        )
    return identity


def _require_tool_user() -> bool:
    return (
        os.environ.get(_REQUIRE_TOOL_USER_ENV, "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _tool_overlay_venv(env: Mapping[str, str]) -> str | None:
    """Path of the tool user's writable overlay venv, or ``None`` if absent.

    ``FRONTIER_AGENT_TOOL_VENV=off`` forces the ``PIP_TARGET`` fallback; any other
    value names the venv to use. Probed by interpreter presence, not directory
    existence, so a half-built path can never shadow a working PATH.
    """
    raw = (env.get(_TOOL_VENV_ENV) or _DEFAULT_TOOL_VENV).strip()
    if not raw or raw.lower() in {"off", "none", "disabled"}:
        return None
    return raw if Path(raw, "bin", "python3").exists() else None


def _tool_env_allowlist(env: dict[str, str]) -> frozenset[str]:
    """Return the built-in allowlist plus trusted, exact-name extensions.

    The extension is deployment configuration, not a glob or substring match.
    Its own control variable is never copied into the child environment.
    """
    extra = env.get(_TOOL_ENV_EXTRA_ALLOWLIST, "")
    names = {
        item.strip()
        for item in extra.split(",")
        if item.strip() and item.strip() != _TOOL_ENV_EXTRA_ALLOWLIST
    }
    return _TOOL_ENV_ALLOWLIST | frozenset(names)


def _build_tool_env(
    env: dict[str, str],
    *,
    home: str,
    env_allow: tuple[str, ...] = (),
    overlay_writable: bool = True,
    tmpdir: str = "/tmp",
) -> dict[str, str]:
    """Construct a minimal environment for model-authored subprocesses.

    This function deliberately does not preserve unknown keys. Tools requiring
    an additional non-secret variable must opt in by exact name through
    ``FRONTIER_AGENT_TOOL_ENV_ALLOWLIST`` or pass it explicitly for that call.

    ``env_allow`` is the per-call counterpart: trusted harness callers may
    expose exact name prefixes without granting them to every model-authored
    command sharing the sandbox. Never pass model-controlled values.

    ``overlay_writable=False`` says the caller mounts the overlay venv read-only
    (bwrap does — see ``_bwrap_base_args``). The overlay still leads PATH, so
    ``python3`` is the same interpreter in both backends, but pip is pointed at
    the workspace HOME instead of a filesystem it cannot write.
    """
    allowed = _tool_env_allowlist(env)
    opt_in = tuple(prefix.upper() for prefix in env_allow)
    projected = {
        key: value
        for key, value in env.items()
        if key in allowed or (opt_in and key.upper().startswith(opt_in))
    }
    projected["HOME"] = home
    # Do not inherit the harness's TMPDIR: it may name a root-only directory
    # that the unprivileged tool user cannot access.
    #
    # Callers that own a private scratch root pass it here. That redirects
    # everything which ASKS the platform for a temp path — ``tempfile``,
    # ``mktemp``, most libraries' spill files — into per-agent storage. It does
    # NOT redirect a literal ``/tmp/foo`` written by model code: without a
    # mount namespace nothing can, because ``/tmp`` is then genuinely one
    # directory (see ``_container_inner_bwrap_enabled`` — production container
    # mode has no inner namespace). Measured on 4 concurrent sub-agents writing
    # ``/tmp/scratch.csv``: all four read back the last writer's content.
    projected["TMPDIR"] = tmpdir
    # Keep pip/matplotlib/caches inside a HOME the tool user owns. Without
    # these they target root-owned paths and fail with bare permission errors
    # the model cannot act on.
    projected["PIP_CACHE_DIR"] = f"{home}/.cache/pip"
    overlay = _tool_overlay_venv(env)
    site = f"{home}/{_TOOL_SITE_SUBDIR}"
    if overlay:
        # Overlay leads PATH so `python3` is the same interpreter under both
        # backends, and its python3/pip see the baked stack through the .pth.
        projected["VIRTUAL_ENV"] = overlay
        projected["PATH"] = f"{overlay}/bin:{home}/.local/bin:" + projected.get("PATH", "")
        if not overlay_writable:
            # Read-only mount (bwrap): pip must not try to write the overlay, or
            # a genuinely-missing package fails with `[Errno 30] Read-only file
            # system` — an "environment looks broken" error of exactly the kind
            # that sends a model thrashing. Fall back to the workspace HOME,
            # which bwrap binds read-write. Accepted cost, eval/local only:
            # --target resolves against that directory alone, so a baked package
            # is re-fetched there, and PYTHONPATH lets the copy shadow it.
            projected["PIP_TARGET"] = site
            projected["PYTHONPATH"] = site
    else:
        projected["PIP_TARGET"] = site
        projected["PYTHONPATH"] = site
        projected["PATH"] = f"{site}/bin:{home}/.local/bin:" + projected.get("PATH", "")
    # setdefault, not assignment: a trusted caller passing an ``env_allow``
    # prefix that covers these (e.g. ``MPL``) means to supply its own.
    projected.setdefault("MPLCONFIGDIR", f"{home}/.cache/matplotlib")
    # Same reason as MPLCONFIGDIR, for the two other libraries that write a
    # cache on import. Numba is the load-bearing one: it caches compiled
    # kernels NEXT TO the module by default, and the tool user cannot write
    # /opt/venv, so ``import scanpy`` dies with "cannot cache function
    # 'agg_sum_csr-parallel': no locator available" — a failure a model reads as
    # a broken package and answers with a reinstall. Caught by the agent-team
    # build gate, which runs as the tool user with no writable HOME.
    projected.setdefault("NUMBA_CACHE_DIR", f"{home}/.cache/numba")
    projected.setdefault("XDG_CACHE_HOME", f"{home}/.cache")
    projected.update(_THREAD_CAP_ENV)
    return projected


def _mem_limit_mb(field: str, env: str, default: int) -> int:
    """Per-exec virtual-memory cap (MB), read from a ``get_config()`` field with
    an env-var then literal-default fallback. 0 = unlimited. Shared by the
    local / E2B / bwrap caps so the config-with-fallback logic lives in one place.
    """
    try:
        from frontier_agent.infra.config import get_config
        return int(getattr(get_config(), field))
    except Exception:
        try:
            return int(os.environ.get(env, str(default)))
        except ValueError:
            return default


def _ulimit_cap(mem_mb: int) -> str:
    """Shell prefix capping virtual memory via ``ulimit -v`` (a shell builtin,
    thread-safe unlike ``preexec_fn``) so a runaway allocation aborts with
    MemoryError instead of OOM-killing the worker. Empty string when the cap is
    disabled (``mem_mb <= 0``)."""
    if mem_mb <= 0:
        return ""
    return f"ulimit -v {mem_mb * 1024} 2>/dev/null; "


def _data_ulimit_cap(mem_mb: int) -> str:
    """Shell prefix capping the DATA segment via ``ulimit -d`` (RLIMIT_DATA).

    Sibling of :func:`_ulimit_cap`, and the right tool when the sandbox IS the
    worker container (``CurrentSandbox``), where ``ulimit -v`` is too blunt:

    * ``-v`` caps virtual address space, and this image ships runtimes whose
      VSZ:RSS ratio is large by design. A JVM (LibreOffice needs one for some
      conversions) reserves its whole max heap plus metaspace up front and
      cannot even initialise under a 2GB ``-v``; measured in this image, ``-d``
      at the same number leaves it working.
    * Since Linux 4.7 RLIMIT_DATA covers brk *and* private anonymous mmap, so
      it tracks what a runaway actually consumes — an unbounded Python
      container growing without limit — while ignoring file mappings.

    Enforced by the kernel at allocation time, so unlike a polling watchdog it
    cannot lose a race against a single huge allocation. It is per-PROCESS, not
    per-process-tree: a command that forks N hungry children can still exceed
    the container's cgroup limit. Empty string when disabled (``mem_mb <= 0``).
    """
    if mem_mb <= 0:
        return ""
    return f"ulimit -d {mem_mb * 1024} 2>/dev/null; "


def _e2b_mem_limit_mb() -> int:
    """Per-exec virtual-memory cap (MB) for remote (E2B/Docker) sandboxes.
    0 = disabled. See ``sandbox_e2b_mem_mb`` in config for the sizing story."""
    return _mem_limit_mb("sandbox_e2b_mem_mb", "SANDBOX_E2B_MEM_MB", 896)


_DATA_CAP_EFFECTIVE: bool | None = None
_DATA_CAP_LOCK = threading.Lock()

# Cap used by the data_cap_effective() probe. Deliberately roomy: see the
# comment in that function for why a tight probe is actively wrong.
_PROBE_MB = 1024


# ── live exec registry (memory watchdog disposal) ──────────────────────
#
# Every in-flight model-authored exec, keyed by process-group id. The memory
# watchdog needs a way to shed load before the kernel does, and "walk /proc and
# kill the biggest thing" is not it: that also finds MCP stdio servers and the
# harness's own children. Only what this backend started is eligible.
_LIVE_EXECS: dict[int, dict[str, Any]] = {}
_LIVE_EXECS_LOCK = threading.Lock()


def _register_exec(
    pgid: int, command: str, cgroup: ExecCgroup | None = None,
) -> None:
    with _LIVE_EXECS_LOCK:
        _LIVE_EXECS[pgid] = {
            "command": command,
            "killed_by_guard": False,
            "cgroup": cgroup,
        }


def _killed_by_guard(pgid: int) -> bool:
    """Whether the watchdog killed *pgid*. Read-only; unregistration is the
    caller's ``finally``, so this stays correct on every exit path."""
    with _LIVE_EXECS_LOCK:
        entry = _LIVE_EXECS.get(pgid)
    return bool(entry and entry["killed_by_guard"])


def _unregister_exec(pgid: int) -> None:
    with _LIVE_EXECS_LOCK:
        _LIVE_EXECS.pop(pgid, None)


def _exec_mem_kb(pgid: int, meta: dict[str, Any]) -> int:
    """Memory charged to one live exec, in KiB. 0 when unknown.

    Prefers the exec cgroup's ``memory.current`` when the exec has one: a
    single read that covers the WHOLE tree (including ``setsid`` escapees a
    pgid walk misses) instead of a full /proc scan per candidate — and the
    watchdog calls this every 3 seconds.
    """
    cgroup: ExecCgroup | None = meta.get("cgroup")
    if cgroup is not None:
        try:
            with open(cgroup.current_path) as fh:
                return int(fh.read()) // 1024
        except (OSError, ValueError):
            pass  # already rmdir'd or unreadable — fall back to /proc
    return _pgid_rss_kb(pgid)


def _pgid_rss_kb(pgid: int) -> int:
    """Summed RSS of every live process in *pgid*, from /proc. 0 when unknown."""
    total = 0
    try:
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            try:
                with open(f"/proc/{pid_dir}/stat", "rb") as fh:
                    fields = fh.read().split()
                # After the (comm) field: state ppid pgrp ...  comm may contain
                # spaces and parens, so index from the LAST ')' instead of
                # splitting blindly — the classic /proc/stat parsing trap.
                raw = b" ".join(fields)
                tail = raw[raw.rindex(b")") + 2:].split()
                if int(tail[2]) != pgid:      # tail[2] = pgrp
                    continue
                with open(f"/proc/{pid_dir}/statm", "rb") as fh:
                    total += int(fh.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        return 0
    return total


_COMMAND_DIGEST_CHARS = 240

# Grace given to the output pumps AFTER the process has exited. Independent of
# the command timeout — see _stream_capped.
_POST_EXIT_DRAIN_S = 5.0


def _command_digest(command: str) -> str:
    """Shorten a command for a log line, keeping the END rather than the start.

    Callers prepend boilerplate before we ever see the string — bash.py alone
    contributes the two-branch ``ulimit -f`` prefix and a net-guard ``export`` —
    so the head is ~200 characters of scaffolding and the actual command is last.
    Truncating from the front produced kill lines whose entire payload was the
    prefix, i.e. it failed at the one thing recording the command is for. Found
    by reading a real kill line in an end-to-end run, not by unit tests.
    """
    command = command.strip()
    if len(command) <= _COMMAND_DIGEST_CHARS:
        return command
    return "…" + command[-_COMMAND_DIGEST_CHARS:]


def kill_heaviest_exec() -> dict[str, Any] | None:
    """SIGKILL the largest in-flight exec process group. Best effort.

    The memory watchdog's disposal step. Returns what was killed (for the trace
    event and the log line) or None when there is nothing eligible — in which
    case the pressure is the harness's own and killing an exec would not help.

    Deliberately best effort, NOT a guarantee: sampling loses to a single huge
    allocation, and RLIMIT_DATA (which is a kernel-enforced guarantee) already
    covers the single-runaway-process case. This exists for the case RLIMIT_DATA
    structurally cannot cover — several concurrent execs each individually under
    their per-process cap, together over the container's.
    """
    with _LIVE_EXECS_LOCK:
        candidates = list(_LIVE_EXECS.items())
    if not candidates:
        return None
    ranked = sorted(
        ((pgid, meta, _exec_mem_kb(pgid, meta)) for pgid, meta in candidates),
        key=lambda t: t[2], reverse=True,
    )
    pgid, meta, rss_kb = ranked[0]
    if rss_kb <= 0:
        return None
    with _LIVE_EXECS_LOCK:
        entry = _LIVE_EXECS.get(pgid)
        if entry is None:                 # finished while we were ranking
            return None
        entry["killed_by_guard"] = True
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)
    # Also kill through the exec cgroup when there is one: killpg cannot reach
    # a descendant that left the process group, cgroup.kill can. Taken from
    # the re-fetched ENTRY, not the ranking snapshot: if the pgid was reused
    # between the two, the snapshot's cgroup belongs to the finished exec.
    cgroup: ExecCgroup | None = entry.get("cgroup")
    if cgroup is not None:
        cgroup.kill()
    return {
        "pgid": pgid,
        "rss_mb": rss_kb // 1024,
        "command": _command_digest(meta["command"]),
        "live_execs": len(candidates),
    }


def _output_cap_bytes() -> int:
    """Per-stream retention cap for captured command output. 0 = unbounded."""
    return _mem_limit_mb("sandbox_output_cap_kb", "SANDBOX_OUTPUT_CAP_KB", 8192) * 1024


class _CappedSink:
    """Retains at most ``cap`` bytes of a stream as head + tail, counting the rest.

    Two properties matter and both are load bearing:

    * **Bounded memory.** ``Popen.communicate`` buffers the WHOLE stream before
      anyone can truncate it, so a command emitting gigabytes (``cat`` of a big
      file, ``yes |``) grows the harness heap until the kernel intervenes. The
      per-tool caps that exist downstream (``maybe_overflow``, and the loop's
      150k-char ceiling) act on a string that has *already* been materialised —
      the truncation is on the wrong side of the pipe.
    * **Head AND tail.** Errors surface at the start (a traceback, ``command not
      found``) and results at the end. Keeping only the head — the obvious
      implementation — throws away exactly the part several tools parse, e.g. a
      success marker printed last.

    The caller must keep READING after the cap is reached; see the drain note in
    ``_CurrentCommands.run``. This class only stops *retaining*.
    """

    __slots__ = ("_cap", "_head", "_head_len", "_tail", "_tail_len", "total")

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._head: list[str] = []
        self._head_len = 0
        # Tail is a bounded deque of recent chunks, trimmed from the left.
        self._tail: collections.deque[str] = collections.deque()
        self._tail_len = 0
        self.total = 0

    def feed(self, chunk: str) -> None:
        self.total += len(chunk)
        if self._cap <= 0:
            self._head.append(chunk)
            return
        half = max(1, self._cap // 2)
        if self._head_len < half:
            room = half - self._head_len
            self._head.append(chunk[:room])
            self._head_len += min(room, len(chunk))
            chunk = chunk[room:]
            if not chunk:
                return
        self._tail.append(chunk)
        self._tail_len += len(chunk)
        while self._tail_len > half and self._tail:
            oldest = self._tail[0]
            if self._tail_len - len(oldest) >= half:
                self._tail.popleft()
                self._tail_len -= len(oldest)
            else:
                cut = self._tail_len - half
                self._tail[0] = oldest[cut:]
                self._tail_len -= cut
                break

    def value(self) -> str:
        head, tail = "".join(self._head), "".join(self._tail)
        elided = self.total - len(head) - len(tail)
        if elided <= 0:
            return head + tail
        return f"{head}\n... [{elided} characters elided by the output cap] ...\n{tail}"


def data_cap_effective() -> bool:
    """Whether ``ulimit -d`` actually takes effect here. Probed once, cached.

    The ``2>/dev/null`` in the cap prefix is deliberate — a shell complaining
    about a limit it will not lower must not pour stderr into the model's tool
    output — but it also means a platform that ignores ``ulimit -d`` leaves us
    with a safety control that silently does nothing. macOS is exactly that
    platform: the limit reads back as RLIM_INFINITY. Production workers are
    Linux, so this is a local-development trap rather than a production one, and
    one loud line at first exec is cheaper than discovering it during an
    incident. Mirrors :func:`bwrap_available`'s probe-once-and-warn shape.
    """
    global _DATA_CAP_EFFECTIVE
    if _DATA_CAP_EFFECTIVE is not None:
        return _DATA_CAP_EFFECTIVE
    # Locked, result published LAST — same race shape as exec_cgroup_root():
    # pre-setting the cache before a probe that releases the GIL (here for up
    # to 15s of subprocess) hands every concurrent caller a hardcoded False.
    # Today's only production caller discards the return value, but the
    # contract is a bool for decisions, and a future caller gating on it
    # would inherit the hole.
    with _DATA_CAP_LOCK:
        if _DATA_CAP_EFFECTIVE is not None:
            return _DATA_CAP_EFFECTIVE
        effective = False
        try:
            # Probe with a GENEROUS cap. The question is only "does setting
            # RLIMIT_DATA take effect", and a tight probe answers a different,
            # harmful question: it can kill the probe process itself and report a
            # false negative. Observed with a 64MB probe under Rosetta x86-64
            # emulation, where the translation layer's own mmap failed
            # ("rosetta error: mmap_anonymous_rw mmap failed") before python even
            # started. Same trap on any platform whose interpreter startup is
            # heavier than the probe value.
            probe = (
                "python3 -c 'import resource;print(resource.getrlimit(resource.RLIMIT_DATA)[0])'"
            )
            out = subprocess.run(
                ["/bin/sh", "-c", f"{_data_ulimit_cap(_PROBE_MB)}{probe}"],
                capture_output=True, text=True, timeout=15,
            )
            effective = out.stdout.strip() == str(_PROBE_MB * 1024 * 1024)
        except Exception:
            pass
        if not effective:
            logger.warning(
                "ulimit -d has no effect on this platform (%s): model-authored commands "
                "run WITHOUT a per-process memory cap, so a runaway allocation can take "
                "the whole container down. Expected on macOS; on a Linux worker this is "
                "a misconfiguration worth investigating.",
                sys.platform,
            )
        _DATA_CAP_EFFECTIVE = effective
        return effective


def _container_mem_limit_mb() -> int:
    """Per-exec RLIMIT_DATA cap (MB) for the ``container`` backend, i.e. model
    code running directly in the worker container. 0 = disabled.

    This backend had NO memory cap at all, which is how a sub-agent's own
    runaway tokenizer took 8GB in two minutes and got the whole pod OOM-killed:
    the kernel reaped the pod's cgroup, so the harness and the worker shell died
    with it, the task surfaced as an unexplained heartbeat timeout, and the
    trace upload (which runs after the agent exits) never happened. With the cap
    the same code dies alone with a MemoryError the model can read and react to.

    Sizing: the cap must be > the heaviest legitimate tool and small enough that
    K concurrent execs cannot collectively reach the pod's cgroup limit.
    Measured peaks for this image's tool set are well under 1GB (LibreOffice
    headless conversion being the largest), and with agent_bus_max_parallel=8 on
    an 8Gi worker 8 x 1024MB is exactly the container limit — hence 1024 rather
    than something roomier. It is a cap per PROCESS, not a budget, so a
    deployment with a different fan-out or memory_limit should set this to
    roughly ``pod_memory_limit / max_parallel``. Keep this literal in sync with
    ``sandbox_container_mem_mb`` in config.py — this is only the env fallback.
    """
    return _mem_limit_mb("sandbox_container_mem_mb", "SANDBOX_CONTAINER_MEM_MB", 1024)


def _local_mem_limit_mb() -> int:
    """Per-exec cap for worker-local model code, including bwrap fallback."""
    return _mem_limit_mb("sandbox_local_mem_mb", "SANDBOX_LOCAL_MEM_MB", 640)


def remote_exec_prefix() -> str:
    """Shell prefix for execs inside REMOTE sandboxes (E2B / Docker): a
    ``ulimit -v`` memory cap plus single-threaded math-lib env.

    Without this guard, on the 512MB base template a buffered
    big-file parse OOM-killed the whole VM → opaque "exit code -1" tool
    errors. With the cap, the python process trips RLIMIT_AS first and dies
    with a clean MemoryError traceback the model can act on, and the VM
    survives. Remote sandboxes are always Linux.
    """
    parts = []
    mem_mb = _e2b_mem_limit_mb()
    if mem_mb > 0:
        parts.append(f"ulimit -v {mem_mb * 1024} 2>/dev/null;")
    caps = " ".join(f"{k}={v}" for k, v in _THREAD_CAP_ENV.items())
    parts.append(caps)
    return " ".join(parts) + " "


# ── Bubblewrap Sandbox (per-task filesystem isolation) ─────────────────

_BWRAP_PATH = shutil.which("bwrap")
_BWRAP_USABLE: bool | None = None


def bwrap_available() -> bool:
    """True on Linux when ``bwrap`` is present and usable."""
    global _BWRAP_USABLE
    if not sys.platform.startswith("linux"):
        return False
    if _BWRAP_PATH is None:
        return False
    if _BWRAP_USABLE is not None:
        return _BWRAP_USABLE
    try:
        result = subprocess.run(
            [_BWRAP_PATH, *_bwrap_base_args(), "--", "true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _BWRAP_USABLE = result.returncode == 0
        if not _BWRAP_USABLE:
            logger.warning("bubblewrap probe failed: %s", result.stderr.strip())
    except Exception as exc:
        logger.warning("bubblewrap probe failed: %s", exc)
        _BWRAP_USABLE = False
    return _BWRAP_USABLE


# System paths bound read-only into every bwrap sandbox. ``/lib64`` is absent
# on some distros, so each is bound only when it exists. The base interpreter of
# the harness venv lives under ``/usr`` (``/usr/local/...`` on the musl images),
# so binding ``/usr`` also exposes it; the venv itself is added by
# ``_venv_bind_args``. ``/proc`` is deliberately NOT here — see ``--proc`` below.
_BWRAP_SYSTEM_PATHS = ("/usr", "/bin", "/lib", "/lib64", "/etc")

# Recovery store below the workspace. Kept in sync with ``_overflow``'s
# ``_WORKSPACE_SUBDIR`` by name rather than by import: ``_overflow`` imports this
# module (lazily, from inside its functions), so a module-level import back the
# other way would be a cycle.
_SPILL_SUBDIR = ".spill"


def _interpreter_bind_args() -> list[str]:
    """Expose the running interpreter read-only at its real path(s).

    The runtime images bake the ``sandbox`` extra (numpy/pandas/scipy/sympy)
    into ``/opt/venv`` and put ``/opt/venv/bin`` on ``PATH``. Without mounting
    that venv into the sandbox the directory is absent, so ``python3`` falls
    through to a bare system interpreter with none of the scientific stack.

    Bind both ``sys.prefix`` (the venv) and ``sys.base_prefix`` (the base
    interpreter a venv's ``bin/python`` symlinks into, whose stdlib the venv
    resolves through). On the images base_prefix sits under ``/usr`` and is
    already exposed, but on pyenv/conda/Homebrew deployments it lives elsewhere
    and the venv is unusable without it. Anything already covered by a
    system-path bind — or by an earlier entry here — is skipped, so a non-venv
    interpreter (``sys.prefix == sys.base_prefix`` under ``/usr``) emits nothing.

    NB: fed into every BwrapSandbox via ``_bwrap_base_args``, so a future
    bwrap-backed per-task (SWE) sandbox would also see the harness interpreter
    mounted read-only at its real path, alongside the task's own env.
    """
    args: list[str] = []
    emitted: list[str] = []
    for raw in (sys.prefix, sys.base_prefix):
        prefix = os.path.realpath(raw)
        if any(_path_within(prefix, p) for p in (*_BWRAP_SYSTEM_PATHS, *emitted)):
            continue
        if not Path(prefix).exists():
            continue
        emitted.append(prefix)
        args.extend(["--ro-bind", prefix, prefix])
    return args


def _extra_tool_root_bind_args() -> list[str]:
    """Expose configured non-system CLI roots read-only inside bwrap."""
    raw = os.environ.get(_EXTRA_TOOL_ROOTS_ENV, _DEFAULT_EXTRA_TOOL_ROOTS)
    args: list[str] = []
    emitted: list[str] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item or not os.path.isabs(item):
            continue
        root = os.path.realpath(item)
        if any(_path_within(root, parent) for parent in (*_BWRAP_SYSTEM_PATHS, *emitted)):
            continue
        if not Path(root).is_dir():
            continue
        emitted.append(root)
        args.extend(["--ro-bind", root, root])
    return args


def _bwrap_base_args() -> list[str]:
    """Build conservative bubblewrap args for the current host."""
    args: list[str] = [
        "--unshare-user",
        "--unshare-ipc",
        "--die-with-parent",
    ]
    # A PID namespace is only usable together with a fresh procfs (see the
    # /proc note at the end of this function), so the two move together.
    if _sandbox_profile() == _PROFILE_SERVICE:
        args.insert(1, "--unshare-pid")
    # Hard network isolation for sandboxed bash/python (web_search/web_fetch run
    # in-harness, not here, so the agent's search tools are unaffected). This is
    # the real network boundary; ``_net_guard.py``'s byte cap is best-effort
    # defense-in-depth for backends without it. Gated by env (default on) so an
    # outer container that rejects net-namespace creation can fall back —
    # ``_bwrap_base_args`` feeds ``bwrap_available()``'s probe, so a rejected
    # ``--unshare-net`` would otherwise disable bwrap entirely.
    if os.environ.get("SANDBOX_UNSHARE_NET", "1").strip().lower() not in ("0", "false", "no"):
        args.append("--unshare-net")
    for path in _BWRAP_SYSTEM_PATHS:
        if Path(path).exists():
            args.extend(["--ro-bind", path, path])
    args.extend(_interpreter_bind_args())
    args.extend(_extra_tool_root_bind_args())
    # The tool user's overlay venv sits outside the system paths above, so bind
    # it too — otherwise PATH names a directory absent from the jail and
    # ``python3`` silently resolves to a different interpreter than in container
    # mode. Read-only, because the overlay lives for the whole container while a
    # bwrap jail is per-command: a writable bind would let one command persist
    # code there for every later command (and for container-mode execs) to
    # import. Model bash DOES have network here — ``bash.py`` passes
    # ``allow_net=True``, which strips ``--unshare-net`` above — so installs are
    # possible and must land somewhere writable; ``_build_tool_env``'s
    # ``overlay_writable=False`` points pip at the workspace HOME instead.
    overlay = _tool_overlay_venv(os.environ)
    if overlay:
        args.extend(["--ro-bind", overlay, overlay])
    # ── /proc, and why it differs by profile ─────────────────────────────
    #
    # ``service``: mount a FRESH procfs for the unshared PID namespace. A bind
    # of the outer /proc keeps the host process view (defeating --unshare-pid)
    # and exposes /proc/<pid>/{environ,root} — on a multi-tenant worker that
    # means one task's model code can read the *service's* provider
    # credentials out of the harness process. There, the fresh procfs is the
    # boundary and there is no acceptable fallback: if a runtime forbids it,
    # ``bwrap_available()``'s probe (which runs these same args) fails and the
    # caller fails closed rather than binding the outer instance.
    #
    # ``local`` (default): a single-user install has no second tenant to
    # protect from. What the jail is for here is the *filesystem* — the agent
    # gets the working directory read-write and a read-only system, and never
    # sees ~/.ssh, ~/.aws or the rest of $HOME. The credentials reachable
    # through a bound /proc are the user's own, and are already the keys the
    # agent is calling the model with, so the fresh procfs buys almost nothing
    # while costing a great deal: mounting one needs privileges that neither a
    # stock Docker seccomp profile nor many unprivileged container runtimes
    # grant, and without it the whole sandbox is unavailable.
    #
    # So local mode binds /proc read-only and drops --unshare-pid, keeping the
    # two consistent (a PID namespace whose /proc shows host processes is worse
    # than no namespace: `ps` and anything reading /proc/self lie).
    #
    # The cost is real and worth stating: without a PID namespace a command
    # that backgrounds a process can leave it running after the command's
    # timeout, because killing the sandbox no longer tears down a namespace.
    if _sandbox_profile() == _PROFILE_SERVICE:
        args.extend(["--proc", "/proc"])
    else:
        args.extend(["--ro-bind", "/proc", "/proc"])
    args.extend(["--dev", "/dev"])
    return args


def _bwrap_mem_limit_mb() -> int:
    """Per-command virtual-memory cap (MB) for BwrapSandbox. 0 = unlimited.

    A ceiling, not a reservation — nothing is pre-allocated; an allocation only
    fails once the command's total virtual memory (VSZ, not RSS) would exceed
    it, protecting the pod from a single runaway bash call. Kept generous
    (default 12 GB) because runtimes reserve large virtual regions they never
    touch. Applied via a ``ulimit -v`` shell prefix (thread-safe, the codebase
    idiom — see _ulimit_cap) rather than a preexec_fn."""
    return _mem_limit_mb("sandbox_bwrap_mem_mb", "SANDBOX_BWRAP_MEM_MB", 12 * 1024)


class _BwrapCommands:
    """Command executor that wraps each command in a bubblewrap sandbox."""

    def __init__(
        self,
        bind_args: list[str],
        chdir: str,
        *,
        mem_limit_mb: int | None = None,
    ) -> None:
        self._bind_args = bind_args
        self._chdir = chdir
        self._mem_limit_mb = mem_limit_mb

    def run(self, command: str, timeout: int = 60,
            input: str | None = None, allow_net: bool = False,
            env_allow: tuple[str, ...] = ()) -> _CommandResult:
        if _BWRAP_PATH is None:
            return _CommandResult(stderr="bubblewrap (bwrap) is not installed", exit_code=127)
        mem_limit_mb = (
            _bwrap_mem_limit_mb()
            if self._mem_limit_mb is None
            else self._mem_limit_mb
        )
        capped = _ulimit_cap(mem_limit_mb) + command
        # Most commands retain the isolated network namespace. Callers that
        # intentionally provide a network-backed capability (read_file OCR,
        # controlled downloads, or bash research) opt in explicitly.
        base_args = _bwrap_base_args()
        if allow_net:
            base_args = [a for a in base_args if a != "--unshare-net"]
        argv = [
            _BWRAP_PATH,
            *base_args,
            *self._bind_args,
            "--chdir",
            self._chdir,
            "--",
            "bash",
            # ``-c`` (not ``-lc``): a *login* shell sources host /etc/profile.d/*
            # which prints a multi-KB MOTD banner onto stdout, corrupting every
            # parsed tool result (JSON / ls / cat). PATH is provided via env, so
            # binaries are still found without the login-shell profile.
            "-c",
            capped,
        ]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_build_tool_env(
                    os.environ.copy(),
                    home="/workspace",
                    env_allow=env_allow,
                    # The jail mounts the overlay venv read-only (see
                    # ``_bwrap_base_args``), so pip goes to the workspace HOME.
                    overlay_writable=False,
                ),
                input=input,
            )
            return _CommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command timed out after {timeout}s") from exc
        except Exception as e:
            return _CommandResult(stderr=str(e), exit_code=1)


class _BwrapFiles:
    """Minimal E2B-like file API for code paths that call ``files.write``."""

    def __init__(self, commands: _BwrapCommands) -> None:
        self._commands = commands

    def write(self, path: str, content: str) -> None:
        import base64

        b64c = base64.b64encode(content.encode()).decode()
        b64p = base64.b64encode(path.encode()).decode()
        cmd = (
            "python3 -c \"import base64, os; "
            f"p = base64.b64decode('{b64p}').decode(); "
            "os.makedirs(os.path.dirname(p) or '.', exist_ok=True); "
            f"open(p, 'wb').write(base64.b64decode('{b64c}'))\""
        )
        result = self._commands.run(cmd, timeout=30)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr.strip() or "bwrap file write failed")


class BwrapSandbox:
    """Per-task bubblewrap sandbox with an E2B-compatible ``commands.run`` API.

    Args:
        workspace: host directory bound read-write at ``/workspace``. Commands
            run with ``/workspace`` as their current directory.
        binds: extra mounts as ``(host_src, sandbox_dst, read_only)`` tuples.
            Read-only missing paths are skipped; read-write paths are created.
        mem_limit_mb: optional per-command virtual-memory ceiling. The generic
            model-code fallback passes the tighter worker-local limit; callers
            that omit it retain the larger file-processing limit.
    """

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        binds: tuple[tuple[str, str, bool], ...] = (),
        mem_limit_mb: int | None = None,
    ) -> None:
        if not bwrap_available():
            raise SandboxUnavailableError(
                "bubblewrap fallback requires Linux, an installed bwrap binary, "
                "and usable user namespaces"
            )

        self._owns_workspace = workspace is None
        if workspace is None:
            workspace_path = Path(
                tempfile.mkdtemp(prefix="frontier_agent-bwrap-workspace-")
            ).resolve()
        else:
            workspace_path = Path(workspace).expanduser().resolve()
            workspace_path.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(prefix="frontier_agent-bwrap-tmp-")).resolve()

        dir_args = [
            "--dir",
            "/workspace",
            "--dir",
            "/inputs",
            "--dir",
            "/outputs",
            "--dir",
            "/tmp",
        ]
        bind_args: list[str] = [
            *dir_args,
            "--bind",
            str(workspace_path),
            "/workspace",
            "--bind",
            str(tmp_path),
            "/tmp",
        ]
        for src, dst, read_only in binds:
            src_path = Path(src).expanduser().resolve()
            if not src_path.exists():
                if read_only:
                    logger.warning("BwrapSandbox: read-only input missing, skipped: %s", src)
                    continue
                src_path.mkdir(parents=True, exist_ok=True)
            parent_dst = str(Path(dst).parent)
            if parent_dst not in {".", "/"}:
                bind_args.extend(["--dir", parent_dst])
            bind_args.extend(
                (["--ro-bind"] if read_only else ["--bind"])
                + [str(src_path), str(dst)]
            )

        # Mount the spill store read-only at its own top-level path.
        #
        # Without a mount the read-only store is only a lexical promise:
        # ``_deliverable_policy`` refuses every natural way to write there —
        # including bash, which IS token-scanned — but shell expansion can hide a
        # path from any scanner (a glob, a brace, a ``$VAR`` assembled in pieces,
        # a ``$(…)`` substitution), and the files are ordinary 0644 files. File
        # modes cannot close it either — model commands run as uid 0 inside the
        # user namespace, so DAC is not consulted. A mount is, which is why this
        # is the layer that actually holds.
        #
        # This no longer has to be the LAST bind to win: the source now sits
        # outside the workspace, so nothing above it overlaps and the ordering
        # constraint that used to be load-bearing is gone.
        #
        # ``--ro-bind-try``, not ``--ro-bind``: the store is created lazily on the
        # first spill, and bwrap aborts the whole jail when a ``--ro-bind`` source
        # is missing. Args are rebuilt per command, so the mount appears as soon
        # as the directory does — and until then there is nothing to protect.
        #
        # The harness writes spill through the host filesystem, not through the
        # jail, so this constrains model commands only.
        bind_args.extend([
            "--ro-bind-try", str(spill_root()), _DEFAULT_SPILL_DIR,
        ])

        self._workdir = str(workspace_path)
        self._tmpdir = str(tmp_path)
        self.commands = _BwrapCommands(
            bind_args, "/workspace", mem_limit_mb=mem_limit_mb,
        )
        self.files = _BwrapFiles(self.commands)
        self.sandbox_id = f"bwrap-{workspace_path.name}"
        logger.info("BwrapSandbox created: id=%s binds=%s", self.sandbox_id, bind_args)

    def kill(self) -> None:
        """Clean up sandbox-owned temporary directories."""
        with contextlib.suppress(Exception):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        if self._owns_workspace:
            with contextlib.suppress(Exception):
                shutil.rmtree(self._workdir, ignore_errors=True)


# ── Current Container Sandbox (for Harbor / pre-provisioned workspaces) ──


# Largest write guaranteed not to block on a pipe the kernel reported writable.
# Same constant, same reason, as subprocess._PIPE_BUF.
_PIPE_BUF = getattr(select, "PIPE_BUF", 512)

# How long past the deadline a command that has ALREADY EXITED may keep
# streaming its tail before we give up. See the grace note in _stream_capped.
_POST_EXIT_GRACE_STEP_S = 0.25


def _incremental_text_decoder(stream: Any) -> io.IncrementalNewlineDecoder:
    """Reproduce what ``text=True`` would have done to *stream*, incrementally.

    We read raw bytes off the fd and cannot decode at the end the way
    ``communicate()`` does, because we do not keep the bytes. Two things then
    only an incremental decoder gets right, and both are real: a multi-byte
    character split across an ``os.read`` boundary, and a ``\\r\\n`` split
    across one (``translate=True`` is the newline half of ``text=True``).

    ``errors="replace"``, deliberately not the strict default. Under strict, one
    stray non-UTF-8 byte — ``cat`` of a binary, ``grep`` in a JPEG, a tool that
    prints latin-1 — raises mid-capture. In the previous thread-per-stream
    design that killed the pump, so nobody drained that pipe again and the child
    blocked in ``write()`` until the timeout with all of its output discarded.
    A tool result is for a model to read; U+FFFD is the right answer.
    """
    enc = getattr(stream, "encoding", None) or "utf-8"
    return io.IncrementalNewlineDecoder(codecs.getincrementaldecoder(enc)("replace"), True)


def _release_pipe(sel: selectors.BaseSelector, fileobj: Any) -> None:
    """Unregister *fileobj* from *sel* and close it.

    Cannot block: this thread is the only reader/writer of these objects, so
    nobody else holds their buffer lock. That is the invariant the whole
    single-threaded design exists to provide — see _stream_capped.
    """
    with contextlib.suppress(KeyError, ValueError):
        sel.unregister(fileobj)
    with contextlib.suppress(Exception):
        fileobj.close()


def _stream_capped(
    proc: subprocess.Popen, *, input: str | None, timeout: float,
) -> tuple[str, str]:
    """``Popen.communicate`` with a retention cap instead of unbounded buffering.

    Drop-in for communicate(): feeds stdin, reads stdout and stderr, waits for
    exit, and raises ``subprocess.TimeoutExpired`` past the deadline so the
    caller's existing handling is unchanged. Structured like CPython's POSIX
    ``Popen._communicate``: one selector over all three raw fds, driven from the
    CALLING thread.

    **Why not a thread per stream.** That was the first implementation and it
    could wedge the caller forever. A pump parked in ``read()`` holds its
    ``BufferedReader`` lock, and the ``close()`` in ``_CurrentCommands.run``'s
    ``finally`` needs that same lock — so cleanup blocked, and ``run()`` neither
    returned nor raised. It needed only a grandchild outside the process group
    (``setsid``, or ``Popen(..., start_new_session=True)`` — a very ordinary way
    for model code to start a background server): ``killpg`` cannot reach it, so
    the pipe never EOFs. Measured on Linux 5.10 / CPython 3.14, **nothing**
    rescues that pump — not ``os.close(fd)``, not ``dup2`` over it, not
    ``pthread_kill``; a blocked ``read(2)`` on a pipe is uninterruptible, and
    ``close()``/``detach()`` from another thread are themselves the deadlock.
    So the fix is not to park: one thread, one selector, every ``close()``
    lock-free by construction.

    Reading both pipes in one selector is also what makes the naive-loop
    objection moot — neither "the child filled the pipe you are not reading" nor
    "the child will not read stdin until it has written output" can arise when
    all three fds are polled together.

    Retention is capped but reads never stop: see ``_CappedSink``. Stopping the
    reads would block the child in ``write()``, which for a child that never
    exits on its own is the very hang this backend's ``killpg`` contract exists
    to prevent.
    """
    if os.name != "posix":  # pragma: no cover - workers are Linux
        # selectors cannot poll pipes on Windows; CPython falls back to threads
        # there too. Uncapped, but this backend does not run on Windows.
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
        return stdout or "", stderr or ""

    cap = _output_cap_bytes()
    sinks = {"stdout": _CappedSink(cap), "stderr": _CappedSink(cap)}
    names: dict[Any, str] = {}
    decoders: dict[Any, io.IncrementalNewlineDecoder] = {}
    pending: memoryview | None = None
    offset = 0

    # ONE deadline covering both process exit and pipe EOF, because that is
    # communicate()'s contract: it returns when the pipes EOF, not when the
    # shell exits. A command that exits 0 but leaves a daemon holding the pipe
    # must still surface as a timeout (see the "Edge:" note in
    # _CurrentCommands.run) rather than returning early with partial output.
    deadline = time.monotonic() + timeout
    hard_deadline = deadline + _POST_EXIT_DRAIN_S
    progressed = False

    with selectors.DefaultSelector() as sel:
        if proc.stdin is not None:
            raw = (input or "").encode(
                getattr(proc.stdin, "encoding", None) or "utf-8", "replace",
            )
            if raw:
                pending = memoryview(raw)
                sel.register(proc.stdin, selectors.EVENT_WRITE)
            else:
                _release_pipe(sel, proc.stdin)  # EOF at once, as communicate() does
        for name in ("stdout", "stderr"):
            stream = getattr(proc, name)
            if stream is not None:
                names[stream] = name
                decoders[stream] = _incremental_text_decoder(stream)
                sel.register(stream, selectors.EVENT_READ)

        while sel.get_map():
            now = time.monotonic()
            if now >= deadline:
                # Post-exit drain grace. A command that exits at 299.9s of a
                # 300s timeout would otherwise be reported as a timeout WITH ITS
                # OUTPUT DISCARDED even though it succeeded — communicate()
                # drains without a timeout once the process is gone. Granting
                # the grace unconditionally is wrong the other way: it turned
                # every `foo &` timeout into timeout + 5s. So extend in small
                # steps, only while the process has exited AND the last select
                # actually delivered bytes, hard-capped at +_POST_EXIT_DRAIN_S.
                if not progressed or proc.poll() is None or deadline >= hard_deadline:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                deadline = min(now + _POST_EXIT_GRACE_STEP_S, hard_deadline)
            progressed = False
            for key, _events in sel.select(max(0.0, deadline - time.monotonic())):
                if key.fileobj is proc.stdin:
                    # At most _PIPE_BUF to a pipe the kernel called writable, so
                    # this cannot block. Closing on EPIPE is the child's choice.
                    assert pending is not None
                    try:
                        offset += os.write(key.fd, pending[offset:offset + _PIPE_BUF])
                    except OSError:
                        _release_pipe(sel, key.fileobj)
                    else:
                        if offset >= len(pending):
                            _release_pipe(sel, key.fileobj)
                    continue
                # Readable means data or EOF is pending, so this cannot block
                # either — which is why the fds stay in blocking mode.
                chunk = os.read(key.fd, 65536)
                sink = sinks[names[key.fileobj]]
                decoder = decoders[key.fileobj]
                if chunk:
                    progressed = True
                    sink.feed(decoder.decode(chunk))
                else:
                    sink.feed(decoder.decode(b"", True))  # flush a split character
                    _release_pipe(sel, key.fileobj)

        # Every pipe hit EOF, so every process that held one is gone; what is
        # left is the reap, under the same deadline — a shell that detached its
        # own stdio (`exec >/dev/null 2>&1; sleep 30`) EOFs immediately and must
        # still time out rather than being waited on forever.
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    return sinks["stdout"].value(), sinks["stderr"].value()


_SETPRIV_PATH = shutil.which("setpriv")
_SETPRIV_MISSING_WARNED = False
_ULIMIT_CGROUP_ORDER_CHECKED = False


def _check_ulimit_below_cgroup(mem_mb: int) -> None:
    """Warn once if ``ulimit -d`` >= the per-exec ``memory.max``.

    The two caps have OPPOSITE failure semantics and their order is what keeps
    the model able to self-correct: RLIMIT_DATA fails the allocation
    synchronously (MemoryError + traceback, actionable), the cgroup group-kills
    the whole tree (bare exit 137). The ulimit must therefore trip FIRST for
    the single-runaway-process case, leaving the cgroup to catch only what
    RLIMIT_DATA structurally cannot (fan-out, MAP_SHARED). Inverting the order
    silently downgrades every plain runaway to the unreadable signal.
    """
    global _ULIMIT_CGROUP_ORDER_CHECKED
    if _ULIMIT_CGROUP_ORDER_CHECKED:
        return
    _ULIMIT_CGROUP_ORDER_CHECKED = True
    raw = (os.environ.get(_EXEC_CG_MEM_MAX_ENV) or "").strip()
    try:
        cg_max = int(raw)
    except ValueError:
        return
    if cg_max <= 0:
        return
    if mem_mb <= 0:
        logger.warning(
            "ulimit -d is disabled while the per-exec cgroup is armed: EVERY "
            "runaway — including plain single-process ones — will be group-"
            "killed to exit 137 instead of raising a readable MemoryError. "
            "Set SANDBOX_CONTAINER_MEM_MB below WORKER_EXEC_MEM_MAX_BYTES.",
        )
    elif mem_mb * 1024 * 1024 >= cg_max:
        logger.warning(
            "ulimit -d (%d MB) >= per-exec memory.max (%d MB): the cgroup will "
            "group-kill before RLIMIT_DATA can raise MemoryError, so models get "
            "exit 137 instead of a readable traceback. Lower "
            "SANDBOX_CONTAINER_MEM_MB or raise WORKER_EXEC_MEM_MAX_BYTES.",
            mem_mb, cg_max // (1024 * 1024),
        )


class _CurrentCommands:
    """Command executor for an existing checkout in the current process.

    Model-authored commands are dropped to an unprivileged uid when one is
    available (see :func:`tool_identity`) — that, not the filesystem layout, is
    what keeps them out of ``/proc/<harness-pid>/environ``.
    """

    def __init__(self, workdir: str, *, private_tmp: bool = False) -> None:
        self._workdir = workdir
        self._identity = tool_identity()
        native = os.environ.get("APODEX_IN_NATIVE", "").strip() == "1"
        self._runtime_home = (
            os.environ.get("HOME", "").strip() or workdir
            if native else workdir
        )
        # Per-agent scratch root. Only meaningful when this executor owns a
        # private workdir (a sub-agent); the task-wide sandbox keeps the
        # shared /tmp so the documented ``read_file(save_to="/tmp/...")`` and
        # skill conventions still resolve to the same place for everyone.
        self._tmpdir = (
            os.environ.get("TMPDIR", "").strip() or "/tmp"
            if native else "/tmp"
        )
        if private_tmp:
            candidate = Path(workdir) / ".tmp"
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                if self._identity is not None:
                    candidate.chmod(0o777)
                self._tmpdir = str(candidate)
            except OSError as exc:
                logger.warning(
                    "private TMPDIR under %s unavailable (%s); falling back to "
                    "the shared /tmp", workdir, exc,
                )

    def _privilege_kwargs(self) -> dict[str, Any]:
        """Popen kwargs that drop the child to the unprivileged tool user."""
        if self._identity is None:
            return {}
        return {
            "user": self._identity.uid,
            "group": self._identity.gid,
            # Explicit: without it the child keeps root's supplementary groups
            # and the uid drop would be cosmetic for group-granted access.
            "extra_groups": [],
        }

    def _cgroup_wrapped(
        self, command: str, mem_mb: int,
    ) -> tuple[str, ExecCgroup] | None:
        """Per-exec cgroup form of the command, or ``None`` to run as today.

        Shape (root shell, then drop):

            echo $$ > <cg>/cgroup.procs; ulimit -d ...; \\
            exec setpriv --reuid=U --regid=G --clear-groups /bin/sh -c <command>

        The self-migration MUST happen before the privilege drop and cannot be
        done by the tool uid itself: cgroup v2 delegation containment also
        requires write access to the source/destination COMMON ANCESTOR's
        ``cgroup.procs`` (the container root, root:root), verified on a real
        worker node — an unprivileged writer fails even with the leaf files
        chmod'd open. That containment is the lock keeping model code from
        migrating itself OUT of its limit; do not loosen it. Hence root writes
        ``$$`` first, then ``exec setpriv`` replaces Popen's user/group/
        extra_groups kwargs (CPython applies those BEFORE any hook we own).
        Rlimits and cgroup membership both survive exec.

        Fail-open on missing pieces (no cgroup root, mkdir failed), fail-closed
        on the privilege drop itself: if ``setpriv`` is absent from PATH at
        exec time the shell exits 127 and the command never runs as root.
        """
        if self._identity is not None and _SETPRIV_PATH is None:
            # No way to both join the cgroup as root and drop privileges:
            # keep the (security-critical) uid drop, forgo the cgroup.
            # Checked BEFORE creating anything — this is a property of the
            # image, so creating (and immediately destroying) a cgroup per
            # exec would be pure churn. Warn once, like the other probes.
            global _SETPRIV_MISSING_WARNED
            if not _SETPRIV_MISSING_WARNED:
                _SETPRIV_MISSING_WARNED = True
                logger.warning(
                    "setpriv not found; per-exec cgroup isolation disabled to "
                    "keep the tool-user privilege drop",
                )
            return None
        cg = create_exec_cgroup()
        if cg is None:
            return None
        _check_ulimit_below_cgroup(mem_mb)
        prefix = (
            f"echo $$ > {shlex.quote(cg.procs_path)} 2>/dev/null; "
            + _data_ulimit_cap(mem_mb)
        )
        if self._identity is None:
            return prefix + command, cg
        wrapped = (
            prefix
            + f"exec {shlex.quote(_SETPRIV_PATH or '')}"
            + f" --reuid={self._identity.uid} --regid={self._identity.gid}"
            + f" --clear-groups /bin/sh -c {shlex.quote(command)}"
        )
        return wrapped, cg

    def run(self, command: str, timeout: int = 60,
            input: str | None = None,
            env_allow: tuple[str, ...] = ()) -> _CommandResult:
        """Execute a command in the current container or host environment.

        Runs the shell in its OWN session/process group (start_new_session)
        and SIGKILLs that whole group in the ``finally``. This closes a
        subprocess capture-pipe hang: a command that backgrounds a daemon
        (``foo &``) leaves a grandchild holding the stdout/stderr pipe, and
        ``communicate()`` returns only on pipe EOF — not on the shell exiting
        — so it would block forever (even after a ``TimeoutExpired``, since
        CPython then re-drains the pipe with no timeout). ``killpg`` reaps the
        whole group so the pipes always EOF. The bwrap/E2B/Docker backends get
        this for free from their PID namespace / container; CurrentSandbox runs
        on the bare host, so it must reap the group itself.

        Edge: a command that finishes 0 but leaves a surviving daemon blocks
        until ``timeout`` and is then reported as a timeout (the daemon holds
        the pipe past the shell's exit). That is a strict improvement over the
        previous unbounded hang, and cross-call daemon persistence is not part
        of the one-shot sandbox-exec contract anyway.

        Edge: a grandchild that LEFT the process group (``setsid``,
        ``Popen(start_new_session=True)``) is out of ``killpg``'s reach. When
        the worker provides a per-exec cgroup (see ``_exec_cgroup``) the
        ``finally`` additionally kills through ``cgroup.kill``, which such a
        grandchild cannot escape; without one (old pod specs, local dev) it
        survives this call as before. Either way we time out, release our fds
        and return — see the ``finally`` and ``_stream_capped`` for why that is
        not merely likely but structural.
        """
        # Memory cap, applied here rather than in each tool so every caller of
        # this backend is covered — bash, the read_file/create_file bundles and
        # download_file all previously ran with no memory limit whatsoever.
        # Mirrors _BwrapCommands.run, which caps in the backend for the same
        # reason. See _data_ulimit_cap / _container_mem_limit_mb.
        mem_mb = _container_mem_limit_mb()
        if mem_mb > 0:
            data_cap_effective()  # probe once; warns loudly if it is a no-op here
        # Per-exec cgroup on top of (never instead of) the RLIMIT_DATA cap:
        # the ulimit keeps single-process runaways dying with a readable
        # MemoryError, the cgroup catches what a per-process limit cannot —
        # N compliant children summing past the budget, and MAP_SHARED mmaps.
        cg: ExecCgroup | None = None
        wrapped = self._cgroup_wrapped(command, mem_mb)
        if wrapped is not None:
            capped, cg = wrapped
            privilege_kwargs: dict[str, Any] = {}  # setpriv drops instead
        else:
            capped = _data_ulimit_cap(mem_mb) + command
            privilege_kwargs = self._privilege_kwargs()
        try:
            proc = subprocess.Popen(
                capped,
                shell=True,
                stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._workdir,
                # Two independent measures, because either alone is bypassable:
                # a minimal env (the child cannot read what it was not given)
                # and an unprivileged uid (the child cannot read the parent's
                # /proc/<pid>/environ either).
                env=_build_tool_env(
                    os.environ.copy(),
                    home=self._runtime_home,
                    env_allow=env_allow,
                    tmpdir=self._tmpdir,
                ),
                start_new_session=True,
                **privilege_kwargs,
            )
        except Exception as e:
            if cg is not None:
                cg.close()
            return _CommandResult(stderr=str(e), exit_code=1)

        # New session leader → its PGID equals its PID. Capture it now: once
        # the process is reaped, os.getpgid() would raise ProcessLookupError.
        pgid = proc.pid
        try:
            if cg is not None:
                # Idempotent backstop for the shell's own `echo $$ >` (whose
                # failure is silenced): the harness is root, so containment does
                # not apply to it. Children forked before this write would stay
                # behind, which is why the prefix remains the primary path.
                with contextlib.suppress(OSError), open(cg.procs_path, "w") as fh:
                    fh.write(str(proc.pid))
            _register_exec(pgid, command, cgroup=cg)
            stdout, stderr = _stream_capped(proc, input=input, timeout=timeout)
            if cg is not None and proc.returncode != 0:
                # A group kill SIGKILLs the shell too, so the model would see a
                # bare exit 137 — no MemoryError, no traceback, nothing to act
                # on. The cgroup still holds memory.peak / memory.events until
                # the finally rmdirs it; read them NOW and say what happened in
                # the channel the model reads errors from.
                #
                # Gated on a non-zero exit: with a degraded cgroup (the
                # memory.oom.group write failed) the kernel kills only the
                # biggest child, and a shell that recovered (`hog || fallback`)
                # can still exit 0 with oom_kill > 0 — a note would then tell
                # the model its SUCCESSFUL command was killed.
                note = cg.oom_note()
                if note:
                    stderr = (stderr + "\n" if stderr else "") + note
            if _killed_by_guard(pgid) and proc.returncode != 0:
                # Tell the model WHY, in the same channel it reads errors from.
                # Without this the kill is indistinguishable from the command
                # crashing on its own, and "retry verbatim" looks reasonable.
                #
                # Same non-zero-exit gate as the oom_note above: the watchdog
                # can rank a command as the disposal victim in the gap after it
                # exited 0 but before this check (its cgroup still holds page
                # cache), and the killpg then hits nothing — telling the model
                # its SUCCESSFUL command was killed would make it "fix" it.
                stderr = (stderr + "\n" if stderr else "") + (
                    "[memory guard] this command was killed: the container was "
                    "close to its memory limit and this was the largest running "
                    "command. Reduce the working set (process in chunks, stream "
                    "instead of loading whole files) or run fewer commands at once."
                )
            return _CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            # A kill and a timeout can coincide: the kill lands but a setsid
            # orphan keeps the pipes open past the deadline. The memory story
            # is the more actionable half — carry it into the timeout message
            # (the cgroup is still intact here; the finally below removes it).
            notes = []
            if cg is not None:
                note = cg.oom_note()
                if note:
                    notes.append(note)
            if _killed_by_guard(pgid):
                notes.append(
                    "[memory guard] this command was killed mid-run: the "
                    "container was close to its memory limit and this was the "
                    "largest running command."
                )
            raise TimeoutError(
                f"Command timed out after {timeout}s"
                + "".join(f"\n{n}" for n in notes)
            ) from exc
        except Exception as e:
            return _CommandResult(stderr=str(e), exit_code=1)
        finally:
            _unregister_exec(pgid)
            # Reap any surviving group member (a backgrounded grandchild still
            # holding the pipe, or the shell itself on the timeout path).
            # Harmless when the leader already exited (ProcessLookupError).
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pgid, signal.SIGKILL)
            if cg is not None:
                # cgroup.kill + rmdir. The kill reaches setsid escapees killpg
                # cannot; the rmdir is load-bearing — a leaked exec cgroup pins
                # node-level slab that NO container limit accounts for (see
                # ExecCgroup.close). Before pipe close on purpose: it reaps
                # writers, so the drain below sees EOF instead of a live fd.
                cg.close()
            # Lock-free BY CONSTRUCTION: _stream_capped reads and writes these
            # pipes from THIS thread, so no other thread can be parked inside a
            # BufferedReader/Writer holding its lock. That invariant is the whole
            # reason the capture loop is a selector and not a thread per stream —
            # with a parked pump this loop blocked forever on the first close()
            # and run() neither returned nor raised. killpg above cannot reach a
            # grandchild that left the process group (setsid /
            # Popen(start_new_session=True)), and nothing else can interrupt a
            # read(2) on a pipe. Do not add os.close(fd)/dup2 "unblock" attempts:
            # measured, they neither wake the reader nor are they safe (they free
            # an fd number another thread still intends to use).
            #   * read ends: their wrapper buffers are empty (we never read
            #     through them), so close() is just a close(2).
            #   * write end: every stdin byte went out via os.write, so there is
            #     nothing buffered to flush and close() cannot block on a full
            #     pipe either.
            for stream in (proc.stdout, proc.stderr, proc.stdin):
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)


# Inner command sandboxes, keyed by workspace dir. In container mode the main
# agent and every sub-agent attach their own ``CurrentSandbox`` to the SAME
# mounted /workspace, and none of them owns the mount's lifecycle — so none of
# them calls ``kill()``. Building a fresh BwrapSandbox per attach therefore
# leaked one mkdtemp'd /tmp per sub-agent for the life of the container, and
# gave each agent a private /tmp that the others could not see. One inner
# sandbox per workspace restores the shared-mount semantics and the leak.
_INNER_SANDBOXES: dict[str, BwrapSandbox] = {}
_INNER_SANDBOX_LOCK = threading.Lock()


def _shared_inner_sandbox(workdir: str) -> BwrapSandbox:
    """Return the process-wide inner bwrap sandbox for *workdir*."""
    with _INNER_SANDBOX_LOCK:
        existing = _INNER_SANDBOXES.get(workdir)
        if existing is not None:
            return existing
        _, outputs_dir, inputs_dir = resolve_mount_dirs()
        # /outputs is bound unconditionally. Skipping a missing one (as a
        # conditional bind would) is the worst outcome: the model's writes land
        # in the sandbox's own tmpfs and vanish silently — no error, no
        # file_delta, no deliverable. BwrapSandbox creates missing read-write
        # bind sources, and raises loudly if it cannot.
        binds: list[tuple[str, str, bool]] = [(outputs_dir, "/outputs", False)]
        if Path(inputs_dir).is_dir():
            binds.append((inputs_dir, "/inputs", True))
        # mem_limit_mb=None → the generous BwrapSandbox default. The tighter
        # worker-local cap is for opportunistic model-code fallbacks; container
        # mode IS the production execution path, where a data-analysis step
        # legitimately reserves more virtual memory than that cap allows.
        sandbox = BwrapSandbox(workspace=workdir, binds=tuple(binds))
        _INNER_SANDBOXES[workdir] = sandbox
        return sandbox


def reset_inner_sandboxes() -> None:
    """Tear down cached inner sandboxes (process shutdown / tests)."""
    with _INNER_SANDBOX_LOCK:
        cached = list(_INNER_SANDBOXES.values())
        _INNER_SANDBOXES.clear()
    for sandbox in cached:
        with contextlib.suppress(Exception):
            sandbox.kill()


def _container_inner_bwrap_enabled() -> bool:
    """Whether container mode should additionally wrap commands in bwrap.

    Off by default and deliberately so: production runs ``container`` WITHOUT
    bubblewrap (the runtime does not permit its namespace operations), and a
    dev box that silently added an inner namespace would exercise a different
    network model, /tmp, memory ceiling, and write path than production —
    which is exactly how the missing-boundary gap went unnoticed.
    """
    return (
        os.environ.get("FRONTIER_AGENT_CONTAINER_INNER_BWRAP", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def container_uses_inner_bwrap() -> bool:
    """Whether container-mode commands actually enter the inner bwrap jail."""
    return _container_inner_bwrap_enabled() and bwrap_available()


def _grant_tool_access(path: Path) -> None:
    """Best-effort: let the unprivileged tool user own/write *path*.

    The harness writes deliverables as root (``sandbox_write_file``); without
    this the model could not later append to or replace its own file.
    """
    identity = tool_identity_or_none()
    if identity is None:
        return
    with contextlib.suppress(OSError):
        os.chown(path, identity.uid, identity.gid)


def _prepare_tool_writable(*dirs: str) -> None:
    """Make the shared mounts writable by the unprivileged tool user."""
    identity = tool_identity_or_none()
    if identity is None:
        return
    host_identity = (
        os.environ.get("APODEX_TOOL_HOST_IDENTITY", "").strip() == "1"
    )
    for raw in dirs:
        target = Path(raw)
        with contextlib.suppress(OSError):
            target.mkdir(parents=True, exist_ok=True)
        owned = False
        if host_identity:
            # The Compose launcher remapped agent-tool to the invoking host
            # user, so handing the mount to that identity grants write access
            # without opening it to every other local account. Ownership is
            # best effort on its own: NFS with root_squash, uid=-mounted
            # exfat/vfat volumes and some virtiofs shares reject chown even for
            # root, and there the fallback below is the only thing that keeps
            # the mount writable at all.
            try:
                os.chown(target, identity.uid, identity.gid)
                owned = True
            except OSError as exc:
                logger.warning(
                    "Cannot give %s to the tool user (%s); widening its mode "
                    "instead so model commands can still write there.",
                    raw, exc,
                )
        try:
            # These are bind mounts of host directories — for the Compose agent
            # service /workspace is the user's checkout itself — so only ever
            # add bits. Replacing the mode would outlive the container and can
            # lock other host accounts out of their own repository.
            current = target.stat().st_mode & 0o7777
            if owned:
                # The tool user owns it now; ensure only that the owner bits do
                # not deny access. Usually already true, so usually a no-op.
                if current & 0o700 != 0o700:
                    target.chmod(current | 0o700)
            else:
                # Benchmark/platform containers may not have a corresponding
                # host identity. Retain their established single-task behavior.
                target.chmod(current | 0o777)
        except OSError as exc:
            logger.error(
                "Cannot make %s writable by the tool user (%s); model commands "
                "will fail to write there. Mount it read-write, or set "
                "%s=off to run them with the harness's own uid.",
                raw, exc, _TOOL_USER_ENV,
            )


# Both readability probes below run AS the tool user, so they must not carry the
# harness environment. A child running under that uid owns its own
# /proc/<pid>/environ — meaning a concurrent model command, running as the same
# uid, could read the provider credentials straight out of a probe that inherited
# them. That is precisely the boundary dropping the uid exists to hold. Neither
# probe needs anything from the harness env: one runs `ls` through /bin/sh, the
# other an absolute-path interpreter with a stdlib-only script.
_PROBE_ENV = {"PATH": "/usr/bin:/bin"}


def _probe_inputs_ls(inputs_dir: str) -> tuple[Any, Any] | None:
    """List ``inputs_dir`` AS the tool user; ``(identity, completed)`` or None.

    None means the question does not apply or could not be answered: no tool
    account (model commands run as the harness itself), no ``/inputs`` for this
    task, or the probe could not be spawned.

    This opens the directory as the unprivileged uid rather than inspecting the
    mode bits. Object-storage mounts (ossfs / s3fs and other FUSE drivers) are
    the common case here and they defeat a mode-bit check outright: without
    ``-o allow_other`` the kernel refuses every uid except the one that mounted
    the filesystem, while ``stat`` still reports a perfectly ordinary ``0755``.
    """
    identity = tool_identity_or_none()
    if identity is None:
        return None
    target = Path(inputs_dir)
    if not target.is_dir():
        return None  # no /inputs for this task
    try:
        probe = subprocess.run(
            ["/bin/sh", "-c", f"ls -1 {shell_quote(str(target))} >/dev/null"],
            capture_output=True, text=True, timeout=15,
            user=identity.uid, group=identity.gid, extra_groups=[],
            env=_PROBE_ENV,
        )
    except Exception as exc:  # pragma: no cover - platform specific
        logger.warning("Could not probe %s readability: %s", inputs_dir, exc)
        return None
    return identity, probe


def probe_inputs_readable(inputs_dir: str) -> bool | None:
    """Can the tool user actually read ``inputs_dir``? None = cannot tell.

    Callers that describe ``/inputs`` to a model must consult this first: the
    harness runs as root and can list files that the unprivileged tool uid then
    fails to open, so an ungated listing promises files that every subsequent
    read reports as ``exit=1 … Permission denied``.
    """
    probed = _probe_inputs_ls(inputs_dir)
    if probed is None:
        return None
    _, probe = probed
    return probe.returncode == 0


# Opens each candidate as the tool user and echoes back the ones that worked.
# Paths cross on stdin NUL-separated, never through a shell word or an argv
# slot, so a filename containing a quote, a space or a metacharacter cannot
# change what runs. Reads one byte rather than calling access(2): on a FUSE
# mount the mode bits are not the authority (see :func:`_probe_inputs_ls`).
_READABLE_FILTER_SRC = """
import sys
out = []
for raw in sys.stdin.buffer.read().split(b"\\0"):
    if not raw:
        continue
    try:
        with open(raw, "rb") as fh:
            fh.read(1)
    except OSError:
        continue
    out.append(raw)
sys.stdout.buffer.write(b"\\0".join(out))
"""


def filter_readable_by_tool_user(paths: Sequence[str]) -> list[str] | None:
    """Keep only the paths the tool user can actually open. ``None`` = can't tell.

    :func:`probe_inputs_readable` answers this for the mount root, which is not
    the same question: ``ls -1 /inputs`` succeeds while a nested directory or a
    ``0600`` file inside it stays unreadable. The walk that produced *paths* ran
    as the harness (root), so without this filter a listing can still promise
    individual files whose every ``read_file`` returns ``Permission denied``.

    ``None`` means the question does not apply or could not be answered — no
    tool account (model commands run as the harness itself, so whatever the
    harness enumerated is readable), or the probe could not be spawned. Callers
    treat that as "keep everything", matching :func:`probe_inputs_readable`.
    """
    if not paths:
        return []
    identity = tool_identity_or_none()
    if identity is None:
        return None
    payload = b"\0".join(os.fsencode(p) for p in paths)
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _READABLE_FILTER_SRC],
            input=payload, capture_output=True, timeout=30,
            user=identity.uid, group=identity.gid, extra_groups=[],
            env=_PROBE_ENV,
        )
    except Exception as exc:
        logger.warning("Could not probe per-file readability: %s", exc)
        return None
    if probe.returncode != 0:
        logger.warning(
            "Per-file readability probe failed (exit=%s): %s",
            probe.returncode, (probe.stderr or b"").decode("utf-8", "replace").strip(),
        )
        return None
    return [
        os.fsdecode(raw) for raw in (probe.stdout or b"").split(b"\0") if raw
    ]


def _warn_if_inputs_unreadable(inputs_dir: str) -> None:
    """Warn when the task's input files are invisible to the tool user.

    ``/inputs`` is a read-only mount, so this cannot be repaired from here —
    and the failure is otherwise silent: the model simply reports that it
    cannot open the files it was asked to work on.
    """
    probed = _probe_inputs_ls(inputs_dir)
    if probed is None:
        return
    identity, probe = probed
    if probe.returncode == 0:
        return
    logger.error(
        "%s is not readable by the tool user %s(uid=%s): %s. Model commands "
        "will not be able to open the task's input files. For a FUSE/object-"
        "storage mount add `-o allow_other` (plus a non-restrictive "
        "mp_umask/file_mode); for an ordinary mount `chmod a+rX`. Or set "
        "%s=off to run model commands with the harness's own uid.",
        inputs_dir, identity.name, identity.uid,
        probe.stderr.strip() or f"exit {probe.returncode}", _TOOL_USER_ENV,
    )


class CurrentSandbox:
    """Sandbox facade for a workspace provisioned by the outer task container.

    Production runs one task per isolated container and does NOT run bubblewrap
    — its namespace operations are not permitted by the runtime. The container
    is the boundary against the host; within it, model-authored commands are
    separated from the harness's credentials by two measures:

    - a minimal, allowlisted child environment (:func:`_build_tool_env`), and
    - an unprivileged uid (:func:`tool_identity`), which is what actually stops
      ``cat /proc/<harness-pid>/environ`` — that file is owner-readable only.

    Isolation degrades to env-only (with a loud warning) when the image has no
    tool account or the harness is not root; set
    ``FRONTIER_AGENT_REQUIRE_TOOL_USER=1`` to make that fatal instead. An inner
    bubblewrap namespace is available via ``FRONTIER_AGENT_CONTAINER_INNER_BWRAP``
    for hosts that do allow it, but is off by default so dev matches prod.
    """

    def __init__(self, workdir: str | Path, *, private_tmp: bool = False) -> None:
        self._workdir = str(Path(workdir).resolve())
        self._inner: BwrapSandbox | None = None
        if container_uses_inner_bwrap():
            self._inner = _shared_inner_sandbox(self._workdir)
            self.commands = self._inner.commands
            self.files = self._inner.files
        else:
            identity = tool_identity()
            if identity is None:
                logger.warning(
                    "CurrentSandbox running model commands with the harness's "
                    "own uid: the child environment is scrubbed, but "
                    "/proc/<harness-pid>/environ remains readable"
                )
            else:
                _, outputs_dir, inputs_dir = resolve_mount_dirs()
                _prepare_tool_writable(self._workdir, outputs_dir)
                _warn_if_inputs_unreadable(inputs_dir)
                logger.info(
                    "CurrentSandbox model commands drop to %s(uid=%s)",
                    identity.name, identity.uid,
                )
            self.commands = _CurrentCommands(self._workdir, private_tmp=private_tmp)
        self.sandbox_id = f"current-{os.getpid()}"
        logger.info("CurrentSandbox attached: workdir=%s", self._workdir)

    def kill(self) -> None:
        """No-op: the outer container owns the mounts, and the inner command
        sandbox is shared with every other facade on the same workspace. Use
        :func:`reset_inner_sandboxes` for process-wide teardown."""
        return None


def make_current_sandbox(
    workdir: str | Path, *, private_tmp: bool = False
) -> CurrentSandbox:
    """Create a sandbox facade for an already-provisioned workspace.

    ``private_tmp`` points ``TMPDIR`` at a scratch root inside *workdir*. Set
    it for a per-agent workspace; leave it off for the task-wide sandbox, whose
    ``/tmp`` several tools and skills name explicitly.
    """
    path = Path(workdir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Current sandbox workdir does not exist: {path}")
    return CurrentSandbox(path, private_tmp=private_tmp)


# ── Docker Sandbox (container-based, for pre-built SWE-bench images) ───


class _DockerCommands:
    """Command executor for DockerSandbox using ``docker exec``."""

    # Sentinel emitted as stderr when the container is gone. A consumer
    # watching for container death can match this string to terminate
    # cleanly instead of burning turns against a dead container.
    DEAD_MSG = "Error: sandbox container has died"

    def __init__(self, container_id: str, *, login_shell: bool = True) -> None:
        self._container_id = container_id
        self._login_shell = login_shell

    def run(self, command: str, timeout: int = 60,
            input: str | None = None) -> _CommandResult:
        """Execute *command* inside the container via ``docker exec``.

        With ``login_shell=True`` (default) uses ``bash -l -c`` so the
        container's login profile (e.g. conda activation in swebench
        images) is sourced automatically. terminal-bench parity uses
        ``login_shell=False`` (plain ``bash -c``) to match harbor's exec.
        ``-i`` is added when stdin input is supplied.
        """
        bash_args = ["bash", "-l", "-c"] if self._login_shell else ["bash", "-c"]
        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    *(["-i"] if input is not None else []),
                    self._container_id,
                    *bash_args, command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input,
            )
            if result.returncode != 0 and "No such container" in result.stderr:
                return _CommandResult(stderr=self.DEAD_MSG, exit_code=1)
            return _CommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Command timed out after {timeout}s"
            ) from exc
        except Exception as e:
            return _CommandResult(stderr=str(e), exit_code=1)


class DockerSandbox:
    """Docker container sandbox for SWE-bench evaluation.

    Starts a detached container from a pre-built image; commands are
    executed via ``docker exec``.  The container is auto-removed on
    ``kill()`` thanks to ``--rm``.
    """

    # docker run can stretch into minutes under heavy concurrent load.
    _RUN_TIMEOUT = 300

    def __init__(
        self,
        image: str,
        *,
        timeout: int = 3600,
        envs: dict[str, str] | None = None,
        login_shell: bool = True,
    ) -> None:
        cmd: list[str] = ["docker", "run", "-d", "--rm"]
        for k, v in (envs or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [image, "sleep", str(timeout + 60)]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self._RUN_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed: {result.stderr.strip()}"
            )
        self._container_id = result.stdout.strip()
        self.sandbox_id = f"docker-{self._container_id[:12]}"
        self.commands = _DockerCommands(self._container_id, login_shell=login_shell)
        logger.info(
            "DockerSandbox created: container=%s image=%s",
            self._container_id[:12], image,
        )

    def copy_in(self, src: str, dst: str, *, timeout: int = 120) -> None:
        """Copy a host path into the container via ``docker cp``.

        ``src`` ending in ``/.`` copies directory *contents* into ``dst``
        (Docker semantics), matching harbor's ``upload_dir`` behaviour.
        """
        result = subprocess.run(
            ["docker", "cp", src, f"{self._container_id}:{dst}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker cp failed: {result.stderr.strip()}")

    def copy_out(self, src: str, dst: str, *, timeout: int = 120) -> None:
        """Copy a container path out to the host via ``docker cp``.

        Mirror of :meth:`copy_in`. ``src`` ending in ``/.`` copies directory
        *contents* into ``dst`` (matches harbor's ``download_dir``).
        """
        result = subprocess.run(
            ["docker", "cp", f"{self._container_id}:{src}", dst],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker cp (out) failed: {result.stderr.strip()}")

    def kill(self) -> None:
        """Stop and remove the Docker container."""
        try:
            subprocess.run(
                ["docker", "kill", self._container_id],
                capture_output=True,
                timeout=15,
            )
            logger.info(
                "DockerSandbox killed: %s",
                self._container_id[:12],
            )
        except Exception as e:
            logger.warning("Error killing DockerSandbox: %s", e)


# ── E2B Config ──────────────────────────────────────────────────────────


def _get_e2b_config() -> tuple[str, str, int]:
    """Get E2B config — reads from FrontierAgentConfig (which loads .env) first."""
    try:
        from frontier_agent.infra.config import get_config
        cfg = get_config()
        if cfg.e2b_api_key:
            return cfg.e2b_api_key, cfg.e2b_template, cfg.e2b_timeout
    except Exception:
        pass
    api_key = os.environ.get("E2B_API_KEY", "")
    template = os.environ.get("E2B_TEMPLATE", "base")
    timeout = int(os.environ.get("E2B_TIMEOUT", "1800"))
    return api_key, template, timeout


def _get_sandbox_backend() -> str:
    """Resolve the configured execution backend, failing closed on typos.

    ``local`` is retained as a compatibility alias for ``bwrap``; it no longer
    means an in-process host subprocess. ``container`` reuses the production
    task container's mounted ``/inputs`` (ro), ``/outputs`` (rw), and
    ``/workspace`` (rw) through :class:`CurrentSandbox`, which runs
    model-authored commands under an unprivileged uid rather than a namespace
    (production does not permit bwrap). Invalid values are errors rather than
    implicit backend downgrades.
    """
    # The terminal CLI resolves its execution boundary after startup and then
    # exports SANDBOX_BACKEND (notably, macOS falls back from an unavailable
    # Docker daemon to ``native``).  ``get_config()`` is process-cached and may
    # still contain the earlier ``auto`` value, so the live environment must
    # win or workflow tools can disagree with the CLI and incorrectly demand
    # Linux bubblewrap on macOS.
    backend = os.environ.get("SANDBOX_BACKEND", "").strip().lower()
    if not backend:
        try:
            from frontier_agent.infra.config import get_config
            backend = (get_config().sandbox_backend or "").strip().lower()
        except Exception:
            pass
    backend = backend or "auto"
    if backend not in ("auto", "local", "bwrap", "e2b", "container", "native"):
        raise SandboxConfigurationError(
            f"invalid sandbox_backend={backend!r}; "
            "expected e2b, bwrap, local, container, native, or auto"
        )
    return backend


# ── Container-mode filesystem (production: one task = one isolated container) ──
#
# In ``container`` mode the surrounding task container owns three real dirs the
# agents (main + every sub) share. CurrentSandbox runs model commands under an
# unprivileged uid so they cannot read the harness's /proc/<pid>/environ:
#   /workspace — rw scratch / intermediate files (NOT delivered)
#   /outputs   — rw final deliverables (what the frontend collects)
#   /inputs    — ro task input files
# These are fixed absolute mount points, overridable via env for local runs.

_DEFAULT_WORKSPACE_DIR = "/workspace"
_DEFAULT_OUTPUTS_DIR = "/outputs"
_DEFAULT_INPUTS_DIR = "/inputs"
#: Canonical mount point of the spill store. A sibling of the three above, NOT a
#: subdirectory of ``/workspace``: the store has to sit outside every write root
#: for the read-only mount to be a containment boundary rather than a remount
#: that must out-order an overlapping rw bind. Deliberately parallel to
#: ``/inputs`` — mounted read-only, authorized for READ only, gated on existing.
_DEFAULT_SPILL_DIR = "/spill"
#: Explicit override for the physical store, for a deployment that wants it on a
#: specific volume.
_SPILL_DIR_ENV = "APODEX_SPILL_DIR"


def resolve_sandbox_mode(agent_cfg: dict[str, Any] | None = None) -> str:
    """Return the filesystem-isolation mode: ``container`` | ``bwrap`` | ``auto``.

    Precedence:
      1. ``SANDBOX_BACKEND`` env / config — ``container`` → container mode;
         ``bwrap``/``local`` → bwrap mode; ``e2b`` → ``auto`` (E2B is a separate
         axis, unrelated to the container/bwrap FS split).
      2. profile ``agent.sandbox_mode=bwrap`` may tighten ``auto`` to bwrap.
         A profile can never select ``container``: profiles (including
         request-supplied ``profile_inline``) are workload input, not trusted
         attestation that the worker is already inside a task container.
      3. ``auto`` — the caller decides (bwrap if available, else fail closed).

    CurrentSandbox is therefore only reachable from trusted deployment
    configuration explicitly selecting the ``container`` backend; a
    model-authored profile cannot select it. Within container mode the
    credential boundary is the unprivileged tool uid, not a namespace — see
    :class:`CurrentSandbox`.
    """
    backend = _get_sandbox_backend()
    if backend in ("container", "native"):
        return backend
    if backend in ("bwrap", "local"):
        return "bwrap"
    if backend == "e2b":
        return "auto"
    # backend == "auto": a workload profile may demand bwrap, but it may not
    # attest that the process itself is already inside a trusted container.
    mode = ""
    if agent_cfg:
        mode = str(agent_cfg.get("sandbox_mode") or "").strip().lower()
    if mode == "bwrap":
        return "bwrap"
    if mode == "container":
        logger.warning(
            "Ignoring untrusted profile sandbox_mode=container while "
            "SANDBOX_BACKEND is auto; configure SANDBOX_BACKEND=container "
            "at deployment time to attest the task-container boundary"
        )
    return "auto"


def resolve_mount_dirs() -> tuple[str, str, str]:
    """Return ``(workspace_dir, outputs_dir, inputs_dir)`` for container mode.

    Defaults to the production mount points ``/workspace``, ``/outputs``,
    ``/inputs``; each is overridable via ``FRONTIER_AGENT_WORKSPACE_DIR`` /
    ``FRONTIER_AGENT_OUTPUTS_DIR`` / ``FRONTIER_AGENT_INPUTS_DIR`` so a local run
    (no root) can point them at repo-relative dirs.
    """
    ws = os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "").strip() or _DEFAULT_WORKSPACE_DIR
    out = os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "").strip() or _DEFAULT_OUTPUTS_DIR
    inp = os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "").strip() or _DEFAULT_INPUTS_DIR
    return ws, out, inp


def spill_root() -> Path:
    """Physical root of the spill store, outside every root the agent can write.

    Lives here rather than in ``_overflow`` because two other modules need it and
    ``_overflow`` imports this one: the sandbox mounts it, and ``_path_auth``
    authorizes it for reads. Putting it there would make the dependency circular.

    Order: an explicit ``APODEX_SPILL_DIR``, then a run-scoped directory when the
    harness has one, then the OS temp dir — which is where codex puts the same
    thing (``<temp>/hook_outputs/<thread_id>``). Never the workspace: under
    ``native`` that is frequently the user's own repository.
    """
    explicit = os.environ.get(_SPILL_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit)
    run_dir = os.environ.get("APODEX_RUN_DIR", "").strip()
    if run_dir:
        return Path(run_dir) / "spill"
    # The uid is in the NAME so two accounts on one host do not share a store.
    # It is not a permission boundary and cannot be: under ``container`` the
    # model runs as a DIFFERENT uid than the harness and has to read these
    # files, so the directory must stay traversable by others (0755). What keeps
    # one conversation out of another's recovery files is ``_path_auth``, which
    # authorizes only the current scope plus the stores this process created —
    # a local user with their own shell is outside that model either way.
    return Path(tempfile.gettempdir()) / f"apodex-spill-{os.getuid()}"


def is_spill_path(path: str) -> bool:
    """Whether ``path`` names the spill store, lexically or once resolved.

    The single rule the store's guards share, replacing a hardcoded ``.spill``
    path component tested in four places. It keys off the real root, so a
    deployment that redirects the store with ``APODEX_SPILL_DIR`` — or a run
    directory that happens to sit inside the repository — is still covered,
    which a fixed directory name was not.

    Both the canonical mount path the model sees and the physical path are
    matched: a tool can be handed either.
    """
    raw = str(path or "").strip()
    if not raw:
        return False
    roots = [_DEFAULT_SPILL_DIR, str(spill_root())]
    candidates = [os.path.normpath(raw)]
    with contextlib.suppress(OSError, RuntimeError):
        candidates.append(str(Path(raw).expanduser().resolve()))
    for candidate in candidates:
        for root in roots:
            normalized_root = os.path.normpath(root)
            if candidate == normalized_root or candidate.startswith(
                normalized_root + os.sep,
            ):
                return True
    return False


def spill_path_matcher() -> Callable[[str | Path], bool]:
    """Resolved-once form of :func:`is_spill_path`, for per-file loops.

    ``is_spill_path`` re-reads the environment for the store root and resolves the
    candidate on every call: 35.6us per path against 2.9us here. Be honest about
    what that buys end to end, though — measured in-process on the same tree it is
    1.01x for ``grep_search`` and 1.04x for ``glob_search``, because reading and
    scanning a file, or stat-ing and authorizing it, costs far more than deciding
    whether to skip it. The reason to prefer this form is that the callers already
    resolve ``task_input_matcher`` once for exactly this shape of lookup, and a
    per-file variant sitting under a comment that warns against per-file variants
    is a trap for the next reader.

    The resolve is narrowed, not dropped: only a symlink can put a path inside the
    store without saying so lexically, so that is the one case worth a syscall.
    ``test_a_symlink_into_the_recovery_store_stays_hidden`` pins it — the store
    leaks through a link without this branch.
    """
    roots = [
        os.path.normpath(_DEFAULT_SPILL_DIR),
        os.path.normpath(str(spill_root())),
    ]

    def _within(path: str | Path) -> bool:
        raw = str(path or "").strip()
        if not raw:
            return False
        # ``expanduser`` is pure string work, but only a leading ``~`` needs it.
        candidate = os.path.normpath(
            str(Path(raw).expanduser()) if raw.startswith("~") else raw,
        )
        for root in roots:
            if candidate == root or candidate.startswith(root + os.sep):
                return True
        try:
            if not Path(raw).is_symlink():
                return False
        except OSError:
            return False
        return is_spill_path(raw)

    return _within


def resolve_runtime_path(path: str) -> str:
    """Map a canonical sandbox path onto the current runtime namespace.

    Only the mount prefix is rewritten; the rest of the path is handed back as
    given. Nothing here canonicalises the *result* — collapsing ``..`` against
    a symlinked directory would name a different file than the caller asked for.

    Docker, container and bwrap modes expose real ``/workspace``, ``/outputs``
    and ``/inputs`` mounts, so their configured roots equal the aliases and
    this is a no-op. Native mode uses physical run-local directories because
    hosts such as macOS do not allow creating writable mount points below ``/``.

    Only an exact alias or a component beneath it is rewritten. Relative paths
    and sibling names such as ``/outputs-old`` are intentionally unchanged.
    """
    if not path or not os.path.isabs(path):
        return path
    normalized = os.path.normpath(path)
    workspace, outputs, inputs = resolve_mount_dirs()
    for alias, root in (
        (_DEFAULT_WORKSPACE_DIR, workspace),
        (_DEFAULT_OUTPUTS_DIR, outputs),
        (_DEFAULT_INPUTS_DIR, inputs),
        # ``/spill`` is an alias for exactly the same reason as the three above:
        # a tool running in THIS process is handed the canonical path the model
        # saw inside the sandbox, and has to reach the real directory.
        (_DEFAULT_SPILL_DIR, str(spill_root())),
    ):
        # ``resolve_mount_dirs`` documents repo-relative overrides. Anchoring
        # them keeps the substitution absolute: an absolute alias that came
        # back as ``outputs/report.docx`` would be resolved against whatever
        # cwd the sandbox process happens to have, which is not this one.
        normalized_root = os.path.normpath(os.path.abspath(root))
        if normalized_root == alias:
            continue
        if normalized == alias:
            return normalized_root
        prefix = alias + os.sep
        if normalized.startswith(prefix):
            return os.path.join(normalized_root, normalized[len(prefix):])
    return path


# ── CJK Font Provisioning ──────────────────────────────────────────────
#
# matplotlib does not consult fontconfig: it resolves families through its own
# font_manager, whose packaged default is ``font.sans-serif: DejaVu Sans`` —
# zero CJK coverage. Installing Noto CJK is therefore necessary but NOT
# sufficient; without an rc, 中文/日本語/한국어 labels still render as tofu
# boxes (□). So we install the fonts AND write a matplotlibrc.
#
# The CJK family must come FIRST in each chain: matplotlib's per-glyph
# fallback down font.sans-serif does not engage for these families (verified
# in-image — "DejaVu Sans, Noto Sans CJK SC" still warns about missing glyphs,
# the reverse order renders clean). Noto CJK carries Latin/Kana/Hangul too, so
# leading with it costs no coverage.
#
# Keep the font chains in sync with ``docker/matplotlibrc``, which is the same
# config baked into the agent images (where it is installed system-wide via
# MATPLOTLIBRC and so covers container mode as well). This copy carries extra
# WenQuanYi / mplfonts names for third-party E2B templates where Noto is not
# guaranteed; tests/test_chart_visualization_skill.py asserts the two agree.

_CJK_MATPLOTLIBRC = (
    "font.family: sans-serif\n"
    "font.sans-serif: Noto Sans CJK SC, Noto Sans CJK JP, Noto Sans, "
    "WenQuanYi Zen Hei, WenQuanYi Micro Hei, Source Han Sans CN, "
    "PingFang SC, Heiti SC, Arial Unicode MS, "
    "Liberation Sans, DejaVu Sans, Arial\n"
    "font.serif: Noto Serif CJK SC, Noto Serif CJK JP, Noto Serif, "
    "Source Han Serif CN, Liberation Serif, Caladea, DejaVu Serif, "
    "Times New Roman\n"
    "font.monospace: Noto Sans Mono CJK SC, Liberation Mono, "
    "DejaVu Sans Mono, Courier New\n"
    "axes.unicode_minus: False\n"
)

_CJK_REMOTE_PROVISION_SCRIPT = r"""
# Best-effort CJK font install (apt-get first, mplfonts fallback).
if command -v apt-get >/dev/null 2>&1; then
    (sudo -n apt-get update -qq 2>/dev/null || apt-get update -qq 2>/dev/null || true)
    (sudo -n apt-get install -y -qq --no-install-recommends fonts-noto-cjk fonts-noto-cjk-extra fonts-liberation fonts-crosextra-caladea fonts-wqy-zenhei 2>/dev/null \
     || apt-get install -y -qq --no-install-recommends fonts-noto-cjk fonts-noto-cjk-extra fonts-liberation fonts-crosextra-caladea fonts-wqy-zenhei 2>/dev/null || true)
    fc-cache -f >/dev/null 2>&1 || true
fi

if ! (fc-list :lang=zh 2>/dev/null | head -n 1 | grep -q .); then
    python -c "from mplfonts.bin.cli import init; init()" >/dev/null 2>&1 || true
fi

mkdir -p "$HOME/.config/matplotlib"
cat > "$HOME/.config/matplotlib/matplotlibrc" <<'__MIRORC__'
__RC_PLACEHOLDER__
__MIRORC__

# Invalidate matplotlib's font cache so new TTFs are picked up on next import.
rm -rf "$HOME/.cache/matplotlib" "$HOME/.matplotlib" 2>/dev/null || true
true
""".replace("__RC_PLACEHOLDER__", _CJK_MATPLOTLIBRC.rstrip("\n"))


_PDF_CLI_PROVISION_SCRIPT = r"""
# Best-effort: install poppler-utils so `pdftotext` CLI works for PDF extraction.
if ! command -v pdftotext >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        (sudo -n apt-get update -qq 2>/dev/null || apt-get update -qq 2>/dev/null || true)
        (sudo -n apt-get install -y -qq --no-install-recommends poppler-utils 2>/dev/null \
         || apt-get install -y -qq --no-install-recommends poppler-utils 2>/dev/null || true)
    fi
fi
true
"""


def _provision_pdf_cli(sandbox: Any) -> None:
    """Best-effort install of poppler-utils for `pdftotext` CLI inside *sandbox*.

    Only runs on remote (E2B / Docker) sandboxes where apt-get is available and
    sudo may succeed. BwrapSandbox / CurrentSandbox use pre-installed tools.
    """
    try:
        if isinstance(sandbox, (BwrapSandbox, CurrentSandbox)):
            return
        sandbox.commands.run(_PDF_CLI_PROVISION_SCRIPT, timeout=120)
    except Exception as exc:
        logger.warning("pdftotext provisioning skipped: %s", exc)


def _provision_cjk_fonts(sandbox: Any) -> None:
    """Install CJK fonts and set matplotlib defaults inside *sandbox*.

    Best-effort: failures are logged, never raised. On remote sandboxes
    (E2B/Docker) tries ``apt-get install fonts-noto-cjk`` then falls back
    to the ``mplfonts`` pip package. On BwrapSandbox we only write a
    matplotlibrc inside its workspace; CurrentSandbox is left untouched.

    NB container mode never reaches this function at all — ``get_sandbox()``
    returns a :class:`CurrentSandbox` directly without going through
    ``_create_provisioned_sandbox``. That path is covered at the image layer
    instead: the agent images ship ``docker/matplotlibrc`` at
    ``/etc/matplotlib/matplotlibrc`` with ``MATPLOTLIBRC`` pointing at it, so
    the defaults apply to every interpreter regardless of ``$HOME``.
    """
    try:
        if isinstance(sandbox, BwrapSandbox):
            # Bwrap commands override HOME to /workspace, so writing
            # workdir/.config/matplotlib/matplotlibrc is isolated from the
            # user's real host config and will be picked up by matplotlib.
            rc_dir = Path(sandbox._workdir) / ".config" / "matplotlib"
            rc_dir.mkdir(parents=True, exist_ok=True)
            (rc_dir / "matplotlibrc").write_text(
                _CJK_MATPLOTLIBRC, encoding="utf-8",
            )
            return
        if isinstance(sandbox, CurrentSandbox):
            # Container mode: the image already ships /etc/matplotlib/matplotlibrc
            # with MATPLOTLIBRC pointing at it, and MATPLOTLIBRC is on the tool
            # env allowlist, so the inner bwrap children pick it up unchanged.
            return
        sandbox.commands.run(_CJK_REMOTE_PROVISION_SCRIPT, timeout=180)
    except Exception as exc:
        logger.warning("CJK font provisioning skipped: %s", exc)


# ── Sandbox Accessor ────────────────────────────────────────────────────


#: Absolute (monotonic) instant each remote sandbox's TTL has been extended to.
#: ``set_timeout`` is RELATIVE to now, so without this a routine refresh would
#: silently shorten a deadline a long exec had already raised — see
#: :func:`_extend_sandbox_ttl`. Keyed weakly so a discarded sandbox is not
#: pinned; guarded by a lock because ``_ensure_ttl_outlives_exec`` calls in via
#: ``asyncio.to_thread`` while the singleton path calls in from the event loop.
_TTL_DEADLINES: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()
_TTL_LOCK = threading.Lock()


def _extend_sandbox_ttl(sandbox: Any, min_seconds: float = 0.0) -> None:
    """Reset an E2B sandbox's time-to-live on each reuse.

    ``Sandbox.create(timeout=...)`` only sets the lifetime at birth — a long
    multi-agent run that exceeds it gets the sandbox auto-killed mid-flight,
    forcing a cold recreate. Calling ``set_timeout`` on every access keeps the
    shared singleton alive for the whole run. No-op for BwrapSandbox (no remote
    lifecycle, so no ``set_timeout`` attribute).

    ``min_seconds`` raises the floor for a single long exec. The configured
    ``e2b_timeout`` (1800s) is the same order as the per-command deadlines
    profiles now grant bash, so a command allowed to run for the full budget
    would otherwise race the TTL set at its own start and die as a sandbox
    failure instead of a clean timeout.

    The TTL is only ever moved LATER. ``set_timeout`` is relative to the moment
    it is called, so on the shared singleton a concurrent short tool call —
    which refreshes with no minimum — would otherwise reset a long command's
    extension back to the configured ``e2b_timeout``, killing the sandbox mid
    exec. That is not hypothetical whenever ``tool_timeout_s`` exceeds
    ``e2b_timeout``: the long call asks for 3660s, an ordinary call 200ms later
    asks for 1800s, and the command dies at half its own deadline. Tracking the
    absolute instant instead of the relative window makes a refresh unable to
    lower it.
    """
    set_timeout = getattr(sandbox, "set_timeout", None)
    if set_timeout is None:
        return
    try:
        _, _, configured = _get_e2b_config()
    except Exception as exc:  # best-effort; reuse still works without it
        logger.debug("Sandbox TTL extension skipped: %s", exc)
        return
    now = time.monotonic()
    # The window is a promise of "at least this long", so it is passed through
    # untouched on the common path rather than recovered as ``deadline - now``,
    # and rounded UP (never truncated) when the clamp below rewrites it. CI
    # once observed a 1799 out of a configured 1800 here, which the previous
    # ``int(deadline - now)`` form could produce but this one cannot.
    window = max(float(configured), float(min_seconds))
    deadline = now + window
    with _TTL_LOCK:
        try:
            existing = _TTL_DEADLINES.get(sandbox)
            if existing is not None and existing > deadline:
                deadline = existing
                window = float(math.ceil(deadline - now))
            _TTL_DEADLINES[sandbox] = deadline
        except TypeError:
            # Not weak-referenceable; fall back to the un-tracked behaviour
            # rather than losing the refresh entirely.
            pass
    try:
        set_timeout(max(math.ceil(window), 1))
    except Exception as exc:  # best-effort; reuse still works without it
        logger.debug("Sandbox TTL extension skipped: %s", exc)


# Pre-installed scientific stack for fresh E2B sandboxes. Kept as a module
# constant so the single-shared-singleton path (``get_sandbox``) and the
# per-sub-agent ``SandboxPool`` install the exact same set.
_E2B_PIP_PACKAGES = (
    "pip install matplotlib numpy pandas seaborn scipy "
    "Pillow markdown sympy plotly tabulate mplfonts "
    "beautifulsoup4 requests openpyxl "
    "pypdf PyMuPDF pdfplumber -q --quiet"
)


def _create_provisioned_sandbox(
    *,
    use_e2b: bool,
    api_key: str = "",
    template: str = "base",
    timeout: int = 1800,
) -> Any:
    """Create and provision a fresh isolated sandbox (E2B or bubblewrap).

    Shared by the single-shared-singleton path (:func:`get_sandbox`) and the
    per-sub-agent :class:`SandboxPool` so both get an identical environment:
    pre-installed scientific stack (E2B only), CJK fonts + matplotlibrc, and
    the canonical ``/tmp/agent-outputs`` dir.

    Blocking (E2B create + pip install can take ~minutes); callers in async
    contexts must wrap in ``asyncio.to_thread``.
    """
    if use_e2b:
        from e2b_code_interpreter import Sandbox
        logger.info(
            "Creating E2B sandbox (template=%s, timeout=%ds)", template, timeout,
        )
        sandbox = Sandbox.create(
            template=template, timeout=timeout, api_key=api_key,
        )
        try:
            logger.info("E2B sandbox created: %s", sandbox.sandbox_id)
            # E2B bills by sandbox lifetime, not per execution.
            # Open a wall-clock span keyed by sandbox_id; the kill paths
            # close it, and still-open spans contribute elapsed-so-far at
            # snapshot time (covers wall_deadline kills mid-flight). The TTL
            # gauge lets consumers bound the unobservable post-exit tail:
            # ``true_bill ≤ sandbox_seconds + spans_open × sandbox_ttl_seconds``.
            record_api_request("e2b", requests=0, sandboxes_created=1)
            open_meter_span("e2b", str(sandbox.sandbox_id))
            set_meter_gauge("e2b", "sandbox_ttl_seconds", timeout)
            sandbox.commands.run(_E2B_PIP_PACKAGES, timeout=180)
            logger.info("E2B sandbox pre-packages installed")
            _provision_pdf_cli(sandbox)
        except Exception:
            # A sandbox may exist (and bill) even though provisioning failed.
            # Reap it before auto mode creates the bwrap fallback.
            _safe_kill(sandbox)
            raise
    else:
        logger.info("Creating local BwrapSandbox")
        # This is generic model code, not the large-document file sandbox.
        # Keep the tighter local limit so an E2B outage cannot become a worker
        # OOM incident during fallback.
        sandbox = BwrapSandbox(mem_limit_mb=_local_mem_limit_mb())
        probe = sandbox.commands.run("true", timeout=10)
        if probe.exit_code != 0:
            sandbox.kill()
            raise SandboxUnavailableError(
                "bubblewrap sandbox failed its execution probe: "
                f"{probe.stderr.strip() or f'exit code {probe.exit_code}'}"
            )

    # Pre-provision CJK fonts + matplotlibrc so plots render Chinese/Japanese
    # labels instead of tofu boxes. Best-effort; failures are logged.
    _provision_cjk_fonts(sandbox)

    # Pre-create the canonical outputs dir so the LLM's first `ls` of it
    # succeeds and matches what the system prompt promises. Cheap & idempotent.
    try:
        sandbox.commands.run(
            "mkdir -p /tmp/agent-outputs && chmod 755 /tmp/agent-outputs",
            timeout=10,
        )
    except Exception as e:
        logger.warning("Failed to pre-create outputs dir in sandbox: %s", e)

    return sandbox


def _resolve_use_e2b() -> tuple[bool, str, str, int]:
    """Decide whether to use E2B and return ``(use_e2b, api_key, template,
    timeout)``. Centralises the backend-precedence logic so the singleton
    path and the pool agree on when E2B is in play."""
    backend = _get_sandbox_backend()
    api_key, template, timeout = _get_e2b_config()
    if backend in ("local", "bwrap", "container", "native"):
        # The outer one-task worker container is the selected sandbox. An E2B
        # credential may still be present for unrelated services, but it must
        # not silently move sub-agents off-box: those VMs have none of this
        # task's /inputs or /outputs mounts.
        use_e2b = False
    elif backend == "e2b":
        if not api_key:
            raise SandboxConfigurationError(
                "sandbox_backend=e2b requires E2B_API_KEY; refusing to execute "
                "model-authored code without an isolated backend"
            )
        use_e2b = True
    else:  # auto
        # ``auto`` chooses only isolated backends: E2B when configured,
        # otherwise bubblewrap. Bwrap creation still fails closed if unusable.
        #
        # But a key sitting in ``.env`` is not consent to ship the user's files
        # to a third party. Under the ``local`` profile ``auto`` therefore stays
        # on this machine, and reaching E2B takes an explicit
        # ``SANDBOX_BACKEND=e2b``.
        #
        # This was not hypothetical: with a key present, agent-team sub-agents'
        # ``create_file`` calls went to a cloud sandbox on what was supposed to
        # be a local benchmark run, and two of five questions then sat silent
        # for two hours after an E2B keepalive — invisible until the 8.5h
        # per-question timeout. The CLI had already worked around it by deleting
        # E2B_API_KEY from its own environment; fixing it here means every
        # caller gets the same guarantee instead of each remembering to.
        use_e2b = bool(api_key) and _sandbox_profile() == _PROFILE_SERVICE
        if api_key and not use_e2b:
            logger.info(
                "E2B key present but ignored: SANDBOX_PROFILE=%s keeps execution "
                "on this machine. Set SANDBOX_BACKEND=e2b to use the cloud "
                "sandbox deliberately.",
                _sandbox_profile(),
            )
    return use_e2b, api_key, template, timeout


def _live_shared_sandbox() -> Any:
    """Return the shared singleton if it's alive (and refresh its TTL), else
    clear it and return None. Caller must hold or not need the creation lock."""
    global _sandbox
    if _sandbox is None:
        return None
    try:
        result = _sandbox.commands.run("echo ok", timeout=5)
        if getattr(result, "exit_code", 0) != 0:
            raise SandboxUnavailableError(
                f"sandbox health check failed with exit code {result.exit_code}"
            )
        _extend_sandbox_ttl(_sandbox)
        return _sandbox
    except Exception:
        logger.warning("Sandbox connection lost, creating new one")
        _sandbox = None
        return None


def get_sandbox() -> Any:
    """Get a sandbox instance for the current context.

    Priority:
    1. Per-task sandbox (set via set_task_sandbox) — for SWE benchmark isolation
    2. Shared singleton (E2B or BwrapSandbox) — for research tasks
    """
    # Check per-task override first
    task_sb = _task_sandbox.get(None)
    if task_sb is not None:
        return task_sb

    global _sandbox

    # Fast path: reuse the live shared singleton without taking the lock.
    live = _live_shared_sandbox()
    if live is not None:
        return live

    # Slow path: serialize creation so a burst of concurrent sub-agents shares
    # ONE sandbox instead of each spinning up (and leaking) its own.
    with _sandbox_lock:
        # Double-check — another thread may have created it while we waited.
        live = _live_shared_sandbox()
        if live is not None:
            return live

        backend = _get_sandbox_backend()
        # Container filesystem mode: attach the shared mounts; CurrentSandbox
        # drops model commands to the unprivileged tool uid.
        # The node normally sets this explicitly per task.
        if backend in ("container", "native"):
            workspace_dir = resolve_mount_dirs()[0]
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            _sandbox = CurrentSandbox(workspace_dir)
            return _sandbox
        use_e2b, api_key, template, timeout = _resolve_use_e2b()
        try:
            _sandbox = _create_provisioned_sandbox(
                use_e2b=use_e2b,
                api_key=api_key,
                template=template,
                timeout=timeout,
            )
        except Exception as exc:
            # ``auto`` is the availability-oriented policy: E2B first, then a
            # real Linux bubblewrap sandbox. Explicit ``e2b`` remains strict so
            # operators can require off-host execution where policy demands it.
            if backend == "auto" and use_e2b:
                logger.warning(
                    "E2B sandbox creation failed; trying isolated bubblewrap "
                    "fallback: %s", exc,
                )
                try:
                    _sandbox = _create_provisioned_sandbox(use_e2b=False)
                except Exception as bwrap_exc:
                    raise SandboxUnavailableError(
                        "E2B sandbox creation failed and the Linux bubblewrap "
                        f"fallback is unavailable: E2B={exc}; bwrap={bwrap_exc}"
                    ) from bwrap_exc
            else:
                backend_name = "E2B" if use_e2b else "bubblewrap"
                raise SandboxUnavailableError(
                    f"{backend_name} sandbox creation failed; no unisolated "
                    f"host fallback is permitted: {exc}"
                ) from exc
        return _sandbox


def current_local_workspace() -> str:
    """Return the host workspace mounted at ``/workspace`` for this task.

    Unlike :func:`resolve_mount_dirs`, this respects a per-agent BwrapSandbox,
    whose private host worktree can differ from the process-wide mount config.
    Remote sandboxes intentionally return an empty string.
    """
    sandbox = _task_sandbox.get(None)
    if isinstance(sandbox, (BwrapSandbox, CurrentSandbox)):
        return sandbox._workdir
    return ""


def close_sandbox() -> None:
    """Close the shared sandbox. Called on application shutdown."""
    global _sandbox
    if _sandbox is not None:
        _close_e2b_meter_span(_sandbox)
        try:
            _sandbox.kill()
            logger.info("Sandbox closed")
        except Exception as e:
            logger.warning("Error closing sandbox: %s", e)
        _sandbox = None


def get_existing_sandbox() -> Any:
    """Return the current sandbox if one exists, without creating a new one."""
    task_sb = _task_sandbox.get(None)
    if task_sb is not None:
        return task_sb
    return _sandbox


def sandbox_available() -> bool:
    """Return whether the configured isolated backend is currently usable."""
    try:
        backend = _get_sandbox_backend()
        if backend in ("container", "native"):
            # The mounted workspace is always usable; the credential boundary
            # is the tool uid, not a namespace, and its absence degrades
            # (loudly) rather than making the sandbox unavailable.
            return True
        api_key, _, _ = _get_e2b_config()
        if backend == "e2b":
            return bool(api_key)
        if backend in ("local", "bwrap"):
            return bwrap_available()
        return bool(api_key) or bwrap_available()
    except SandboxError:
        return False


def is_e2b_available() -> bool:
    """Check if E2B cloud sandbox would be used (API key set and not forced local)."""
    if _get_sandbox_backend() in ("local", "bwrap"):
        return False
    api_key, _, _ = _get_e2b_config()
    return bool(api_key)


def shell_quote(s: str) -> str:
    """Simple POSIX shell quoting for sandbox commands."""
    return "'" + s.replace("'", "'\\''") + "'"


def sandbox_write_file(
    sandbox: Any, path: str, content: str, *, mode: str = "w",
) -> tuple[bool, str]:
    """Write *content* to *path* inside *sandbox*. Returns (success, error_message).

    Args:
        mode: "w" (overwrite, default) or "a" (append).

    Dispatch order:
    1. CurrentSandbox — direct pathlib write onto the container's real mounts
    2. E2B — native files.write() API (append not supported, falls through)
    3. DockerSandbox — docker cp (overwrite only; append via fallback)
    4. Fallback — base64 via python3 inside the sandbox
    """
    # ── Current container mounts ─────────────────────────────────────────
    # FIRST, ahead of the generic ``files`` branch: CurrentSandbox exposes the
    # inner bwrap file API, whose write marshals content as base64 inside a
    # single ``python3 -c`` argv. Linux caps one argv entry at MAX_ARG_STRLEN
    # (128 KB), so routing deliverables through it fails with E2BIG somewhere
    # around 96 KB of content — and container mode has no /tmp fallback, so the
    # write becomes a hard error. The harness process already has the mount
    # bound read-write and the path has passed ``_path_auth``; writing it
    # directly is both unlimited and cheaper (no bwrap + python3 spawn).
    if isinstance(sandbox, CurrentSandbox):
        try:
            p = Path(path)
            new_parent = not p.parent.exists()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, mode, encoding="utf-8") as fh:
                fh.write(content)
            # Written as root; hand it to the tool user so the model's own
            # bash can still append to, replace, or delete its deliverable.
            _grant_tool_access(p)
            if new_parent:
                _prepare_tool_writable(str(p.parent))
            return True, ""
        except Exception as e:
            return False, str(e)

    # ── E2B ──────────────────────────────────────────────────────────────
    if mode == "w" and hasattr(sandbox, "files") and hasattr(sandbox.files, "write"):
        try:
            sandbox.files.write(path, content)
            return True, ""
        except Exception as e:
            return False, str(e)

    # ── Docker ───────────────────────────────────────────────────────────
    if mode == "w" and isinstance(sandbox, DockerSandbox):
        parent = os.path.dirname(path)
        if parent:
            sandbox.commands.run(f"mkdir -p {parent}", timeout=10)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".tmp",
        ) as f:
            f.write(content)
            tmp = f.name
        try:
            r = subprocess.run(
                ["docker", "cp", tmp, f"{sandbox._container_id}:{path}"],
                capture_output=True, text=True, timeout=30,
            )
            return (r.returncode == 0, "" if r.returncode == 0 else r.stderr.strip())
        except Exception as e:
            return False, str(e)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    # ── Fallback: base64 via python3 inside the sandbox ──────────────────
    import base64
    b64c = base64.b64encode(content.encode()).decode()
    b64p = base64.b64encode(path.encode()).decode()
    open_mode = "ab" if mode == "a" else "wb"
    cmd = (
        f"python3 -c \"import base64, os; p = base64.b64decode('{b64p}').decode(); "
        f"os.makedirs(os.path.dirname(p) or '.', exist_ok=True); "
        f"open(p, '{open_mode}').write(base64.b64decode('{b64c}'))\""
    )
    r = sandbox.commands.run(cmd, timeout=30)
    return (r.exit_code == 0, "" if r.exit_code == 0 else r.stderr.strip() or "write failed")


# ── Async wrappers ──────────────────────────────────────────────────────
# The E2B SDK is synchronous (blocking httpx); BwrapSandbox / DockerSandbox
# use ``subprocess.run``. Calling these directly inside an ``async def``
# tool freezes the event loop — these helpers offload the blocking call
# to a worker thread so a stalled sandbox cannot starve sibling tools or
# parallel sub-agents in the agent-team runtime.


async def aget_sandbox() -> Any:
    """Async ``get_sandbox`` — never blocks the event loop."""
    import asyncio
    return await asyncio.to_thread(get_sandbox)


# Bounds how many local bwrap execs run at once across the whole process.
# Each local exec consumes real worker RAM (unlike E2B, which offloads
# to the cloud), so without a gate ~dozens of concurrent sub-agents
# running code simultaneously could exhaust the container. Created lazily so
# it binds to the running loop. E2B/Docker execs are not gated (they run
# off-box). 0 = unlimited.
_local_exec_sem: Any = None


def _local_concurrency() -> int:
    try:
        from frontier_agent.infra.config import get_config
        return int(get_config().sandbox_local_max_concurrency)
    except Exception:
        try:
            return int(os.environ.get("SANDBOX_LOCAL_MAX_CONCURRENCY", "2"))
        except ValueError:
            return 2


def _get_local_exec_sem(asyncio_mod: Any) -> Any:
    global _local_exec_sem
    if _local_exec_sem is None:
        n = _local_concurrency()
        _local_exec_sem = asyncio_mod.Semaphore(n if n > 0 else 1_000_000)
    return _local_exec_sem


async def _ensure_ttl_outlives_exec(sandbox: Any, timeout: int) -> None:
    """Make a remote sandbox's TTL outlast a single long exec.

    ``aget_sandbox`` already refreshes the TTL to ``e2b_timeout`` on every
    access, which used to dominate every per-command deadline. It no longer
    does: profiles grant bash up to ``tool_timeout_s`` (1800s in the TUI
    profile), the same order as the default TTL, so a command permitted to use
    its whole budget would race the TTL set at its own start and come back as a
    sandbox death rather than a clean timeout.

    Only reached when the exec needs MORE than the configured TTL, so ordinary
    short commands pay no extra round-trip, and never for local backends (no
    ``set_timeout``). Runs off the event loop because ``set_timeout`` is a
    blocking API call.
    """
    if getattr(sandbox, "set_timeout", None) is None:
        return
    needed = timeout + 60  # covers the post-timeout kill and output drain
    try:
        _, _, configured = _get_e2b_config()
    except Exception:  # pragma: no cover - config failure is handled downstream
        return
    if needed <= configured:
        return
    import asyncio
    await asyncio.to_thread(_extend_sandbox_ttl, sandbox, needed)


async def arun_sandbox_cmd(
    sandbox: Any, command: str, *, timeout: int, input: str | None = None,
    allow_net: bool = False, env_allow: tuple[str, ...] = (),
) -> Any:
    """Async ``sandbox.commands.run`` — never blocks the event loop.

    ``input`` (when set) is fed to the command's stdin — used by read_file to
    pipe the ~95KB reader bundle in (echoing it into argv overflows execve's
    128KB single-arg limit). Only passed through when set, so callers that
    don't need stdin (bash/grep) keep the original signature.

    ``allow_net=True``: used by explicitly network-backed capabilities
    (read_file OCR/VISION, controlled downloads, and bash research). It is
    forwarded only when the backend supports the parameter (BwrapCommands);
    E2B/current-container backends already use their surrounding network.

    ``env_allow`` exposes trusted, per-call environment-name prefixes only to
    backends that support the parameter. Never pass model-controlled values.

    BwrapSandbox execs are throttled by a process-wide semaphore
    so dozens of concurrent sub-agents can't collectively OOM the worker.
    Cloud/Docker sandboxes run off-box and are not gated.

    E2B's SDK *raises* on non-zero exits (``CommandExitException``, a
    ``CommandResult`` subclass carrying stdout/stderr/exit_code) instead of
    returning a result. Without normalisation every failing remote exec
    surfaced to callers as ``Error: CommandExitException: ...`` — dropping
    stdout and dead-coding their exit-code handling (the 124/137 timeout
    message in ``run_python_code``, the ``[Exit code N]`` prefix in
    ``bash``). Convert it back to a ``_CommandResult``. E2B's
    ``TimeoutException`` likewise isn't a ``TimeoutError`` subclass — re-raise
    as one so callers' timeout branches fire. Both are matched by class name
    to keep ``e2b`` an optional import here.
    """
    import asyncio
    await _ensure_ttl_outlives_exec(sandbox, timeout)
    # Only forward ``input`` when set — E2B's commands.run has no stdin param,
    # so unconditional passthrough would break the remote path.
    # Annotated: inferred from the first entry this became dict[str, int],
    # which then rejected the str / bool / tuple capabilities added below.
    kw: dict[str, Any] = {"timeout": timeout}
    if input is not None:
        kw["input"] = input
    # Forward explicit network/environment capabilities only to backends that
    # support them.
    optional: dict[str, Any] = {}
    if allow_net:
        optional["allow_net"] = True
    if env_allow:
        optional["env_allow"] = env_allow
    if optional:
        try:
            import inspect
            supported = inspect.signature(sandbox.commands.run).parameters
            kw.update({key: value for key, value in optional.items() if key in supported})
        except (TypeError, ValueError):
            pass
    if isinstance(sandbox, BwrapSandbox):
        async with _get_local_exec_sem(asyncio):
            return await asyncio.to_thread(
                sandbox.commands.run, command, **kw,
            )
    try:
        return await asyncio.to_thread(
            sandbox.commands.run, command, **kw,
        )
    except Exception as exc:
        exit_code = getattr(exc, "exit_code", None)
        if type(exc).__name__ == "CommandExitException" and isinstance(exit_code, int):
            return _CommandResult(
                stdout=getattr(exc, "stdout", "") or "",
                stderr=getattr(exc, "stderr", "") or "",
                exit_code=exit_code,
            )
        if type(exc).__name__ == "TimeoutException":
            raise TimeoutError(str(exc)) from exc
        raise


# ── Per-sub-agent E2B sandbox pool ─────────────────────────────────────
# Each sub-agent leases ONE sandbox for its whole loop lifetime (so its
# tools share a consistent filesystem). Different sub-agents get different
# E2B VMs. In auto mode, pool saturation and creation failures use bounded
# bwrap isolation; they never downgrade to host execution.


def _sandbox_is_healthy(sandbox: Any) -> bool:
    try:
        result = sandbox.commands.run("echo ok", timeout=5)
        return getattr(result, "exit_code", 0) == 0
    except Exception:
        return False


def _reset_e2b_workspace(sandbox: Any) -> bool:
    """Wipe a reused E2B sandbox's transient working files so the next lessee
    starts clean. Keeps the sandbox + its installed packages for reuse.

    Targets exactly what the tools write: ``run_python_code`` scripts
    (``/tmp/exec_*.py``) and the canonical outputs dir. Best-effort."""
    try:
        sandbox.commands.run(
            "rm -rf /tmp/exec_*.py /tmp/agent-outputs/* 2>/dev/null; "
            "mkdir -p /tmp/agent-outputs",
            timeout=15,
        )
        return True
    except Exception as exc:
        logger.warning("E2B workspace reset failed: %s", exc)
        return False


def _safe_kill(sandbox: Any) -> None:
    _close_e2b_meter_span(sandbox)
    try:
        sandbox.kill()
    except Exception as exc:
        logger.debug("sandbox kill failed: %s", exc)


def is_e2b_sandbox(sandbox: Any) -> bool:
    """True only for real E2B SDK sandboxes (the ones that bill).

    Every facade here carries a ``sandbox_id`` (``local-*`` /
    ``current-*`` / ``docker-*``), so attribute presence can't
    discriminate — go by the implementing module instead.
    """
    try:
        return type(sandbox).__module__.startswith("e2b")
    except Exception:
        return False


def _close_e2b_meter_span(sandbox: Any) -> None:
    """Fold an E2B sandbox's lifetime into ``external_apis.e2b.sandbox_seconds``.

    Bwrap/Current/Docker facades don't bill anyone → no-op. Idempotent:
    the meter pops the span on close.
    """
    if not is_e2b_sandbox(sandbox):
        return
    sandbox_id = getattr(sandbox, "sandbox_id", None)
    if sandbox_id:
        close_meter_span("e2b", str(sandbox_id))


class SandboxPool:
    """Bounded E2B pool with an optional isolated bubblewrap fallback.

    ``lease`` returns ``(sandbox, is_e2b)`` and never waits for E2B past
    ``lease_timeout_s``. In ``auto`` mode, pool exhaustion and E2B creation
    failures fall back to a fresh Linux BwrapSandbox. Bwrap creation and tool
    execution share the process-wide local concurrency gate. If bubblewrap is
    unavailable too, leasing fails closed; model code never runs on the host.

    Invariant: live E2B sandboxes (idle + leased) never exceed ``size`` — a new
    one is only created while holding a semaphore slot and only when no idle
    one can be reused.
    """

    def __init__(
        self,
        *,
        size: int,
        lease_timeout_s: float,
        api_key: str,
        template: str,
        e2b_timeout: int,
        allow_bwrap_fallback: bool = True,
    ) -> None:
        self._size = max(1, size)
        self._lease_timeout_s = lease_timeout_s
        self._api_key = api_key
        self._template = template
        self._e2b_timeout = e2b_timeout
        self._allow_bwrap_fallback = allow_bwrap_fallback
        self._sem: Any = None  # asyncio.Semaphore, lazily bound to the loop
        self._lock: Any = None  # asyncio.Lock guarding _idle / _created
        self._idle: list[Any] = []
        self._created = 0  # live E2B count (idle + leased)

    def _ensure_primitives(self) -> None:
        import asyncio
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._size)
            self._lock = asyncio.Lock()

    async def lease(self, label: str = "") -> tuple[Any, bool]:
        """Lease E2B, or isolated bwrap when ``auto`` fallback is enabled."""
        import asyncio
        self._ensure_primitives()
        try:
            await asyncio.wait_for(
                self._sem.acquire(), timeout=self._lease_timeout_s,
            )
        except TimeoutError as exc:
            capacity_exc = SandboxCapacityError(
                f"E2B sandbox pool is full (size={self._size}); "
                f"lease timed out for {label or 'sub-agent'} after "
                f"{self._lease_timeout_s}s"
            )
            if not self._allow_bwrap_fallback:
                raise capacity_exc from exc
            return await self._lease_bwrap(label, capacity_exc)

        try:
            sandbox = await self._reuse_or_create(label)
        except Exception as exc:
            self._sem.release()
            if not self._allow_bwrap_fallback:
                raise
            return await self._lease_bwrap(label, exc)
        return sandbox, True

    async def _lease_bwrap(
        self, label: str, e2b_exc: Exception,
    ) -> tuple[Any, bool]:
        """Create a bounded local fallback, preserving both failure causes."""
        import asyncio
        logger.warning(
            "E2B unavailable for %s; trying isolated bubblewrap fallback: %s",
            label or "sub-agent", e2b_exc,
        )
        try:
            # Provisioning itself executes a few commands. Put it behind the
            # same gate as subsequent local tool calls so an E2B outage cannot
            # cause an unbounded burst of worker-local processes.
            async with _get_local_exec_sem(asyncio):
                sandbox = await asyncio.to_thread(
                    _create_provisioned_sandbox, use_e2b=False,
                )
        except Exception as bwrap_exc:
            raise SandboxUnavailableError(
                f"E2B unavailable for {label or 'sub-agent'} and the Linux "
                "bubblewrap fallback also failed: "
                f"E2B={e2b_exc}; bwrap={bwrap_exc}"
            ) from bwrap_exc
        return sandbox, False

    async def _reuse_or_create(self, label: str) -> Any:
        import asyncio
        # Reuse an idle healthy sandbox if one exists.
        async with self._lock:
            while self._idle:
                cand = self._idle.pop()
                if await asyncio.to_thread(_sandbox_is_healthy, cand):
                    return cand
                # dead → drop and keep looking
                await asyncio.to_thread(_safe_kill, cand)
                self._created -= 1
            self._created += 1  # reserve the create slot under the lock
        try:
            return await asyncio.to_thread(
                _create_provisioned_sandbox,
                use_e2b=True,
                api_key=self._api_key,
                template=self._template,
                timeout=self._e2b_timeout,
            )
        except Exception as exc:
            async with self._lock:
                self._created -= 1
            raise SandboxUnavailableError(
                f"E2B sandbox creation failed for {label or 'sub-agent'}: {exc}"
            ) from exc

    async def release(self, sandbox: Any, is_e2b: bool) -> None:
        import asyncio
        if not is_e2b:
            # Local fallbacks are intentionally not pooled: their private
            # workspace is deleted at the end of this one sub-agent loop.
            await asyncio.to_thread(_safe_kill, sandbox)
            return
        if await asyncio.to_thread(_reset_e2b_workspace, sandbox):
            _extend_sandbox_ttl(sandbox)
            async with self._lock:
                self._idle.append(sandbox)
        else:
            await asyncio.to_thread(_safe_kill, sandbox)
            async with self._lock:
                self._created -= 1
        self._sem.release()

    async def close(self) -> None:
        """Kill all idle sandboxes (e.g. on task teardown)."""
        import asyncio
        if self._lock is None:
            # Async primitives are lazily bound on first ``lease()`` —
            # a pool that was constructed but never leased has nothing
            # idle to kill (and ``async with None`` would raise).
            return
        async with self._lock:
            idle, self._idle = self._idle, []
            self._created -= len(idle)
        for sandbox in idle:
            await asyncio.to_thread(_safe_kill, sandbox)


_sandbox_pool: SandboxPool | None = None
_sandbox_pool_lock = threading.Lock()


def _pool_settings() -> tuple[int, float]:
    """``(size, lease_timeout_s)`` for the sub-agent pool."""
    try:
        from frontier_agent.infra.config import get_config
        cfg = get_config()
        return (
            int(getattr(cfg, "e2b_pool_size", 20)),
            float(getattr(cfg, "e2b_pool_lease_timeout_s", 5.0)),
        )
    except Exception:
        return 20, 5.0


def subagent_pool_enabled() -> bool:
    """Will each sub-agent lease its OWN sandbox rather than share this process's?

    Answers a question callers need *before* any sandbox exists: a leased
    sandbox is a separate machine, so it carries none of this container's
    mounts. Anything that describes ``/inputs`` to a sub-agent has to check
    this first, or it promises files that sub-agent cannot open.

    Kept next to :func:`get_sandbox_pool`, which now shares this predicate, so
    the two cannot drift — note that ``e2b_pool_size <= 0`` disables pooling
    just as effectively as having no E2B key.
    """
    if not _resolve_use_e2b()[0]:
        return False
    return _pool_settings()[0] > 0


def get_sandbox_pool() -> SandboxPool | None:
    """Return the process-wide per-sub-agent pool, or ``None`` when pooling is
    disabled.

    Disabled when E2B isn't in play (backend=container/local/bwrap, or ``auto``
    without a key) or ``e2b_pool_size <= 0``. Those callers use the configured
    isolated singleton; no host-process fallback exists.
    """
    global _sandbox_pool
    backend = _get_sandbox_backend()
    use_e2b, api_key, template, e2b_timeout = _resolve_use_e2b()
    if not use_e2b:
        return None
    size, lease_timeout = _pool_settings()
    if size <= 0:
        return None
    # Check the effective backend before returning the cached singleton. This
    # matters when a long-lived process changes deployment configuration (and
    # in tests): a pool created while E2B was enabled must never leak back into
    # container/local mode, where /inputs and /outputs are worker mounts.
    if _sandbox_pool is not None:
        return _sandbox_pool
    with _sandbox_pool_lock:
        if _sandbox_pool is None:
            _sandbox_pool = SandboxPool(
                size=size,
                lease_timeout_s=lease_timeout,
                api_key=api_key,
                template=template,
                e2b_timeout=e2b_timeout,
                allow_bwrap_fallback=backend == "auto",
            )
    return _sandbox_pool


async def asandbox_write_file(
    sandbox: Any, path: str, content: str, *, mode: str = "w",
) -> tuple[bool, str]:
    """Async ``sandbox_write_file`` — never blocks the event loop."""
    import asyncio
    return await asyncio.to_thread(
        sandbox_write_file, sandbox, path, content, mode=mode,
    )
