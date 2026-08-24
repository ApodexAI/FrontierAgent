"""Per-browser-session isolation: one unguessable id, one directory tree.

    <runtime-root>/sessions/<session-id>/
    ├── inputs/            task inputs — agent may read, never write
    ├── state/             run bookkeeping and traces — agent may neither
    └── workspace/         the agent's writable root
        └── outputs/       deliverables — the ONLY downloadable directory

Read and write allowances differ, and both are enforced at the tool-call
boundary by ``containment.PathContainmentObserver`` as well as by the runtime's
own path guard (the authorised write root is ``workspace/``).

Session ids are cryptographically random, so one visitor cannot address
another's directory even though every session lives under a shared root. The
tree is created outside the service checkout on purpose: the runtime's own path
guard (``plugins/tools/_path_auth``) refuses host writes to a workspace inside
the checkout.

**Why ``outputs/`` is nested inside ``workspace/``**, rather than a sibling:
the react node hands ``run_agent_loop`` a single authorised write root — the
value of ``FRONTIER_AGENT_WORKSPACE_DIR``, forwarded as the execution scope's
``workspace_root`` (``workflows/stateful_react_agent/nodes/main_agent.py``).
``_path_auth`` authorises exactly that subtree, so a sibling ``outputs/`` would
be *unwritable* and every deliverable would be refused. Nesting keeps one
authorised subtree and leaves ``state/`` and ``inputs/`` outside it, which is
also why the agent cannot tamper with its own trace files.
"""

from __future__ import annotations

import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Accepts exactly what :func:`new_session_id` produces. Anything else coming
#: from the browser is rejected rather than sanitised — a "cleaned" id would
#: silently resolve to somebody else's session.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_SUBDIRS = ("inputs", "state", "workspace", "workspace/outputs")


class InvalidSessionId(ValueError):
    """The supplied session id is not one this store could have issued."""


def new_session_id() -> str:
    """A fresh, unpredictable, filesystem-safe session id."""
    return secrets.token_urlsafe(24)


def validate_session_id(session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not _SESSION_ID_RE.match(sid):
        raise InvalidSessionId("malformed session id")
    return sid


@dataclass(frozen=True)
class DemoSession:
    """One isolated session directory tree."""

    session_id: str
    root: Path
    created_at: float = field(default_factory=time.time)

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def outputs(self) -> Path:
        """Deliverables. Nested in ``workspace/`` — see the module docstring."""
        return self.root / "workspace" / "outputs"

    @property
    def state(self) -> Path:
        return self.root / "state"

    def ensure_dirs(self) -> DemoSession:
        for name in _SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def outputs_hint(self) -> str:
        """How the agent is told to name the deliverable directory."""
        return str(self.outputs)

    @property
    def short_id(self) -> str:
        """A UI-friendly prefix. Never use this to address a session."""
        return self.session_id[:8]


class SessionStore:
    """Creates, finds, clears and expires session directories."""

    def __init__(self, runtime_root: Path, *, ttl_s: float = 3600.0) -> None:
        self._root = Path(runtime_root).expanduser()
        self._sessions_root = self._root / "sessions"
        self._ttl_s = float(ttl_s)
        self._sessions: dict[str, DemoSession] = {}
        self._sessions_root.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_root(self) -> Path:
        return self._sessions_root

    def create(self) -> DemoSession:
        session_id = new_session_id()
        session = DemoSession(
            session_id=session_id, root=self._sessions_root / session_id,
        ).ensure_dirs()
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DemoSession | None:
        """Return a known session, or ``None``. Raises on a malformed id."""
        sid = validate_session_id(session_id)
        session = self._sessions.get(sid)
        if session is not None:
            return session
        # Survive a process restart when the runtime root is persistent.
        path = self._sessions_root / sid
        if path.is_dir():
            recovered = DemoSession(
                session_id=sid, root=path, created_at=path.stat().st_mtime,
            ).ensure_dirs()
            self._sessions[sid] = recovered
            return recovered
        return None

    def get_or_create(self, session_id: str | None) -> DemoSession:
        """Resolve ``session_id``, minting a new session when it is unusable."""
        if session_id:
            try:
                existing = self.get(session_id)
            except InvalidSessionId:
                existing = None
            if existing is not None:
                return existing
        return self.create()

    def clear(self, session_id: str) -> DemoSession:
        """Wipe one session's workspace (and so its outputs), keeping its id."""
        session = self.get(validate_session_id(session_id))
        if session is None:
            return self.create()
        shutil.rmtree(session.workspace, ignore_errors=True)
        return session.ensure_dirs()

    def delete(self, session_id: str) -> None:
        sid = validate_session_id(session_id)
        self._sessions.pop(sid, None)
        path = self._sessions_root / sid
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    def sweep(self, *, now: float | None = None, keep: set[str] | None = None) -> int:
        """Delete sessions older than the TTL. Returns how many were removed.

        ``keep`` protects sessions with a run in flight.
        """
        if self._ttl_s <= 0:
            return 0
        now = time.time() if now is None else now
        protected = keep or set()
        removed = 0
        for path in sorted(self._sessions_root.iterdir()):
            if not path.is_dir() or path.name in protected:
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age > self._ttl_s:
                shutil.rmtree(path, ignore_errors=True)
                self._sessions.pop(path.name, None)
                removed += 1
        return removed


__all__ = [
    "DemoSession",
    "InvalidSessionId",
    "SessionStore",
    "new_session_id",
    "validate_session_id",
]
