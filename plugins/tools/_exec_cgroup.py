"""Per-exec cgroup v2 isolation — consumer side of the worker-shell contract."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT_ENV = "WORKER_EXEC_CGROUP_ROOT"
MEM_MAX_ENV = "WORKER_EXEC_MEM_MAX_BYTES"
_PIDS_MAX_ENV = "WORKER_EXEC_PIDS_MAX"

# Probe-once cache, mirroring bwrap_available() / data_cap_effective():
# None = not probed yet; "" = probed and unavailable.
_ROOT_CACHE: str | None = None
_ROOT_PROBED = False
_ROOT_PROBE_LOCK = threading.Lock()
_MKDIR_FAIL_WARNED = False


def exec_cgroup_root() -> str | None:
    """The delegated ``execs/`` directory, or ``None`` when the feature is dark.

    Probed once and cached: env unset is the normal fail-open state on every
    deployment that predates the worker-shell contract and stays silent; env
    set but unusable is a platform-side misconfiguration and warns once.

    The probe is locked and ``_ROOT_PROBED`` is published LAST. Setting it
    before the isdir/access syscalls (which release the GIL) let a concurrent
    cold-start exec read PROBED=True with the cache still empty and silently
    run without its cgroup — one uncontained command is exactly the hole this
    module exists to close. The unlocked fast path is safe under the GIL
    because the cache is complete by the time the flag is visible.
    """
    global _ROOT_CACHE, _ROOT_PROBED
    if _ROOT_PROBED:
        return _ROOT_CACHE or None
    with _ROOT_PROBE_LOCK:
        if _ROOT_PROBED:
            return _ROOT_CACHE or None
        cache = ""
        raw = (os.environ.get(_ROOT_ENV) or "").strip()
        if raw:
            if os.path.isdir(raw) and os.access(raw, os.W_OK):
                cache = raw
                logger.info("per-exec cgroup isolation armed at %s", raw)
            else:
                logger.warning(
                    "%s=%r is set but not a writable directory; per-exec cgroup "
                    "isolation stays OFF and commands run with only the "
                    "per-process ulimit cap", _ROOT_ENV, raw,
                )
        _ROOT_CACHE = cache
        _ROOT_PROBED = True
        return cache or None


def _env_int(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class ExecCgroup:
    """One created ``exec-<id>/`` directory. Kill and remove via :meth:`close`."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def procs_path(self) -> str:
        """The ``cgroup.procs`` file the exec's shell writes ``$$`` into."""
        return str(self.path / "cgroup.procs")

    @property
    def current_path(self) -> str:
        """``memory.current`` — instantaneous usage of the whole exec tree."""
        return str(self.path / "memory.current")

    @property
    def kill_path(self) -> str:
        return str(self.path / "cgroup.kill")

    def _read_int(self, name: str) -> int | None:
        try:
            return int((self.path / name).read_text().strip())
        except (OSError, ValueError):
            return None

    def oom_kills(self) -> int:
        """Kernel OOM kills charged to THIS exec's cgroup.

        Reads the exec cgroup's own ``memory.events`` — never the container's,
        whose ``oom_kill`` counter is RECURSIVE and accumulates every per-exec
        kill below it (a healthy container reads as OOMKilled after two of
        them; measured on a real worker node).
        """
        try:
            body = (self.path / "memory.events").read_text()
        except OSError:
            return 0
        total = 0
        for line in body.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] in ("oom_kill", "oom_group_kill"):
                with contextlib.suppress(ValueError):
                    total = max(total, int(fields[1]))
        return total

    def oom_note(self) -> str | None:
        """Human-readable account of a group kill, or ``None`` if none happened.

        The whole point of synthesising this: ``memory.oom.group=1`` SIGKILLs
        the shell along with the runaway, so the model sees a bare exit 137
        with no MemoryError and no traceback — nothing it could self-correct
        from. The cgroup still holds the numbers until we rmdir it.
        """
        if self.oom_kills() <= 0:
            return None
        # memory.peak saturates AT the limit (the allocation that would pass it
        # fails), so peak is a floor of what the tree wanted, not an overshoot.
        peak = self._read_int("memory.peak")
        limit = self._read_int("memory.max")
        peak_s = f" (peak usage {peak // (1024 * 1024)}MB)" if peak else ""
        limit_s = f"its {limit // (1024 * 1024)}MB" if limit else "its"
        return (
            f"[memory limit] this command's process tree hit {limit_s} "
            f"per-command memory limit{peak_s} and was killed as a group. "
            "The limit counts ALL its processes together: lower the "
            "parallelism (each concurrent worker holds its own copy), process "
            "the data in chunks, and avoid large anonymous mmap regions."
        )

    def kill(self) -> None:
        """Atomically SIGKILL every process in the exec tree (``cgroup.kill``).

        Unlike ``killpg`` this also reaches descendants that left the process
        group via ``setsid`` / ``start_new_session`` — cgroup membership is
        inherited across fork/exec and cannot be shed from inside.
        """
        with contextlib.suppress(OSError), open(self.kill_path, "w") as fh:
            fh.write("1")

    def close(self) -> None:
        """Kill the tree and remove the directory. Best effort, never raises.

        The rmdir is NOT optional hygiene: an exec cgroup pins node-level slab
        memory that no container limit accounts for, and a worker leaking one
        per exec eventually takes the NODE NotReady (kubernetes KEP-5474
        measured ~42k leaked groups exhausting 14GB). The worker shell sweeps
        leftovers at task end as a backstop for SIGKILL'd harnesses, but the
        normal path is this method, on the exec's own ``finally``.
        """
        self.kill()
        # rmdir succeeds only once every member process is gone; SIGKILL'd
        # processes can take a moment to be reaped, hence the short retry.
        for _ in range(10):
            try:
                self.path.rmdir()
                return
            except OSError:
                time.sleep(0.05)
        logger.warning(
            "exec cgroup %s not removable after kill; leaving it to the "
            "worker-shell sweep", self.path,
        )


def create_exec_cgroup() -> ExecCgroup | None:
    """Create and configure one ``exec-<id>/`` cgroup, or ``None`` (fail-open).

    Limits come from the platform-provided env verbatim; a missing variable
    skips its file rather than inventing a value. Each write is suppressed
    individually — one controller not being enabled must not cost the exec the
    others.

    ``memory.high`` is deliberately NOT set even though the platform suggests
    a value: it throttles instead of killing, which turns an over-memory
    ``soffice`` into a silent tool timeout — a harder signal for the model
    than a group kill with a synthesized explanation. Revisit once
    ``memory.events`` data from production says otherwise.
    """
    global _MKDIR_FAIL_WARNED
    root = exec_cgroup_root()
    if root is None:
        return None
    path = Path(root) / f"exec-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        # exist_ok=False: names must never collide across threads — landing two
        # execs in one cgroup would kill both when either overruns.
        path.mkdir(exist_ok=False)
    except OSError as exc:
        # Warn once per failure streak: a persistent cause (cgroupfs turned
        # read-only, execs/ swept away) would otherwise log every exec.
        if not _MKDIR_FAIL_WARNED:
            _MKDIR_FAIL_WARNED = True
            logger.warning("per-exec cgroup mkdir failed (%s); running uncontained", exc)
        return None
    _MKDIR_FAIL_WARNED = False
    cg = ExecCgroup(path)
    mem_max = _env_int(MEM_MAX_ENV)
    settings: list[tuple[str, str]] = []
    if mem_max:
        settings.append(("memory.max", str(mem_max)))
        # Load-bearing, not hygiene: with swap available the kernel swaps
        # anonymous pages out instead of ever reaching memory.max, so nothing
        # is killed AND MAP_SHARED writes keep landing (reproduced under
        # Docker Desktop). EKS nodes have no swap today; do not rely on that.
        settings.append(("memory.swap.max", "0"))
        # Make THIS cgroup the oom_domain: the kernel kills the whole exec
        # tree here and never walks up to the container's oom.group=1.
        settings.append(("memory.oom.group", "1"))
    pids_max = _env_int(_PIDS_MAX_ENV)
    if pids_max:
        settings.append(("pids.max", str(pids_max)))
    for name, value in settings:
        try:
            (path / name).write_text(value)
        except OSError as exc:
            logger.warning("per-exec cgroup: writing %s=%s failed: %s", name, value, exc)
    return cg
