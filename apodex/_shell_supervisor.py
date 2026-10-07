"""Linux subprocess supervisor for one local shell command.

The supervisor is a child subreaper. Once the shell's process group is killed,
descendants that created another session are adopted here and can be killed
and reaped before the caller reports a timeout. It stays alive until the
caller has seen both output pipes close, so an exited shell cannot strand a
background writer outside the cleanup path.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import select
import signal
import sys
import time

_PR_SET_CHILD_SUBREAPER = 36
_CLEANUP_SECONDS = 5.0


def _children() -> list[int]:
    path = f"/proc/self/task/{os.getpid()}/children"
    with open(path, encoding="ascii") as stream:
        return [int(pid) for pid in stream.read().split()]


def _reap() -> dict[int, int]:
    exited: dict[int, int] = {}
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        exited[pid] = os.waitstatus_to_exitcode(status)
    return exited


def _kill_descendants(shell_pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(shell_pid, signal.SIGKILL)
    deadline = time.monotonic() + _CLEANUP_SECONDS
    while time.monotonic() < deadline:
        _reap()
        children = _children()
        if not children:
            return
        for pid in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.01)
    _reap()


def main(control_fd: int, command: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
    # A missing procfs would make escaped descendants invisible. Do not run
    # an unsupervised command when the promised cleanup is unavailable.
    _children()

    shell_pid = os.fork()
    if shell_pid == 0:
        os.close(control_fd)
        os.setsid()
        os.execl("/bin/sh", "sh", "-c", command)

    # Only the shell and its descendants keep the output pipes open. The
    # caller can therefore detect EOF even while this supervisor waits for a
    # success or kill instruction on its separate control pipe.
    os.close(1)
    os.close(2)
    success_requested = False
    shell_status: int | None = None
    try:
        while True:
            exited = _reap()
            if shell_pid in exited:
                shell_status = exited[shell_pid]
            if success_requested and shell_status is not None:
                return shell_status
            ready, _, _ = select.select([control_fd], [], [], 0.02)
            if ready:
                instruction = os.read(control_fd, 1)
                if instruction == b"S":
                    success_requested = True
                else:  # K, EOF, or an invalid instruction all stop the tree.
                    _kill_descendants(shell_pid)
                    return 124
    finally:
        os.close(control_fd)


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
