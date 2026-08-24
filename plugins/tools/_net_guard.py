"""Sandbox network download cap — socket-level, injected via sitecustomize."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_CAP_KB = 64 * 1024

GUARD_DIR = "/tmp/mh_net_guard"
GUARD_FILE = f"{GUARD_DIR}/sitecustomize.py"

# The in-sandbox guard. Idempotent (``_mh_net_guarded`` flag) so the
# kernel-preamble variant can re-run it safely in a persistent Jupyter
# kernel. Pure stdlib; every counter operation is fail-open — telemetry
# must never break working agent code.
SITECUSTOMIZE_SOURCE = '''\
# frontier_agent sandbox network guard (auto-injected; do not edit)
import os as _mh_os

def _mh_install_net_guard():
    try:
        _cap = int(_mh_os.environ.get("FRONTIER_AGENT_NET_DOWNLOAD_CAP_BYTES", "0"))
    except ValueError:
        _cap = 0
    if _cap <= 0:
        return
    import socket as _socket
    import ssl as _ssl
    import weakref as _weakref
    if getattr(_socket, "_mh_net_guarded", False):
        return
    _socket._mh_net_guarded = True
    _rx = _weakref.WeakKeyDictionary()
    _len = len
    _msg = (
        "[frontier_agent] per-connection download cap exceeded (%d MB). "
        "Downloading large datasets/archives/packages in the sandbox is "
        "not allowed. Use aggregate/count API endpoints, sample a few "
        "records, or stream + filter and keep only the small subset you "
        "need." % max(1, _cap // (1024 * 1024))
    )

    def _bump(sock, n):
        try:
            total = _rx.get(sock, 0) + n
            _rx[sock] = total
        except TypeError:
            return
        if total > _cap:
            raise OSError(_msg)

    _orig_recv = _socket.socket.recv
    _orig_recv_into = _socket.socket.recv_into

    def _recv(self, *args, **kwargs):
        data = _orig_recv(self, *args, **kwargs)
        _bump(self, _len(data))
        return data

    def _recv_into(self, *args, **kwargs):
        n = _orig_recv_into(self, *args, **kwargs)
        _bump(self, int(n or 0))
        return n

    _socket.socket.recv = _recv
    _socket.socket.recv_into = _recv_into

    _orig_ssl_read = _ssl.SSLSocket.read

    def _ssl_read(self, *args, **kwargs):
        result = _orig_ssl_read(self, *args, **kwargs)
        buffer = kwargs.get("buffer") if "buffer" in kwargs else (
            args[1] if _len(args) > 1 else None
        )
        if buffer is not None:
            _bump(self, int(result or 0))
        else:
            _bump(self, _len(result))
        return result

    _ssl.SSLSocket.read = _ssl_read

try:
    _mh_install_net_guard()
except Exception:
    pass
'''


def cap_bytes() -> int:
    """Per-connection cap in bytes from host env; 0 = guard disabled."""
    raw = (os.getenv("SANDBOX_NET_DOWNLOAD_CAP_KB") or "").strip()
    if raw:
        try:
            kb = int(raw)
            return max(0, kb) * 1024
        except ValueError:
            pass
    return DEFAULT_CAP_KB * 1024


def guard_env_prefix() -> str:
    """Shell env prefix arming the guard for a ``python3 file`` exec.

    Empty string when disabled. The ``VAR=x VAR2=y command`` form scopes
    the env to the exec'd process tree without mutating the sandbox
    shell, matching the ``_OFFLINE_DOWNLOAD_ENV`` convention in
    ``run_python_code.py``.
    """
    cap = cap_bytes()
    if cap <= 0:
        return ""
    return (
        f"FRONTIER_AGENT_NET_DOWNLOAD_CAP_BYTES={cap} "
        f'PYTHONPATH="{GUARD_DIR}:${{PYTHONPATH:-}}" '
    )


def kernel_preamble() -> str:
    """Guard source for persistent-kernel execs (``Sandbox.run_code``).

    There is no per-process startup to hook in a long-lived Jupyter
    kernel, so the guard is prepended to the submitted code instead.
    Idempotent across calls via the ``_mh_net_guarded`` flag. Empty when
    disabled.
    """
    cap = cap_bytes()
    if cap <= 0:
        return ""
    return (
        f'import os as _mh_os\n'
        f'_mh_os.environ.setdefault('
        f'"FRONTIER_AGENT_NET_DOWNLOAD_CAP_BYTES", "{cap}")\n'
        f"{SITECUSTOMIZE_SOURCE}\n"
    )


_guard_installed_sandboxes: set[int] = set()


async def ensure_guard_file(sandbox: Any) -> None:
    """Write the sitecustomize into the sandbox once per sandbox object.

    Remote sandboxes (E2B/Docker) pay one file-write round-trip on first
    exec only; local sandboxes write to host ``/tmp``. Failures are
    swallowed — the guard is defence-in-depth, never a reason to fail an
    exec (and ``guard_env_prefix`` pointing at a missing file is a
    harmless no-op: python ignores absent PYTHONPATH entries).
    """
    import asyncio
    from pathlib import Path

    if cap_bytes() <= 0:
        return
    key = id(sandbox)
    if key in _guard_installed_sandboxes:
        return
    try:
        if hasattr(sandbox, "files"):
            await asyncio.to_thread(
                sandbox.files.write, GUARD_FILE, SITECUSTOMIZE_SOURCE,
            )
        else:
            def _write_local() -> None:
                Path(GUARD_DIR).mkdir(parents=True, exist_ok=True)
                Path(GUARD_FILE).write_text(SITECUSTOMIZE_SOURCE)
            await asyncio.to_thread(_write_local)
        _guard_installed_sandboxes.add(key)
    except Exception:
        pass


__all__ = [
    "DEFAULT_CAP_KB",
    "GUARD_DIR",
    "GUARD_FILE",
    "SITECUSTOMIZE_SOURCE",
    "cap_bytes",
    "ensure_guard_file",
    "guard_env_prefix",
    "kernel_preamble",
]
