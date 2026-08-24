"""Bash execution tool — fail-closed E2B, container, or bubblewrap sandbox."""

from __future__ import annotations

import logging
import os

from frontier_agent.core.execution_context import get_current_tool_budget
from frontier_agent.core.tool import tool

# Re-exported for backward compatibility: callers/tests import these from
# ``plugins.tools.bash``. The implementation now lives in ``_bash_policy``.
from plugins.tools._bash_policy import (
    BashCommandAssessment,
    assess_bash_command,
)
from plugins.tools._deliverable_policy import bash_output_write_error
from plugins.tools._net_guard import ensure_guard_file, guard_env_prefix
from plugins.tools._sandbox import aget_sandbox, arun_sandbox_cmd

logger = logging.getLogger(__name__)

# (requested, budget) pairs already warned about — see _warn_clamped_override.
_CLAMP_WARNED: set[tuple[int, int]] = set()

_MAX_OUTPUT_CHARS = 10_000
# Fallback deadline for a bash call made OUTSIDE the agent loop (a script, a
# test, a direct ``bash.ainvoke``). Inside the loop the configured
# ``tool_timeout_s`` wins — see _resolve_timeout.
_DEFAULT_BASH_TIMEOUT = 300
_DEFAULT_FILE_MAX_BYTES = 256 * 1024 * 1024

# Public so consumers (e.g. the report post-processor, sibling
# ``run_python_code.py``) can split stdout/stderr blocks deterministically
# without re-typing the literal — silent drift here would degrade their
# splitting back to "treat the whole blob as stdout".
BASH_STDERR_SEPARATOR = "\n--- stderr ---\n"

__all__ = ["BASH_STDERR_SEPARATOR", "BashCommandAssessment", "assess_bash_command", "bash"]


# 137 = 128+SIGKILL (the kernel OOM killer, or the watchdog's disposal step);
# 133 = 128+SIGTRAP, which is how a memory failure surfaces under x86-64
# emulation. MemoryError is the clean in-process case the memory cap produces.
# "[memory limit]" is the note _sandbox synthesizes when a per-exec cgroup
# group-killed the command (the kill itself leaves no output at all).
_OOM_MARKERS = ("MemoryError", "Cannot allocate memory", "std::bad_alloc",
                "[memory guard]", "[memory limit]", "Killed")
_OOM_EXIT_CODES = (137, 133, -9)


def _looks_out_of_memory(stderr: str, exit_code: int | None) -> bool:
    """Whether this FAILURE was about memory rather than logic.

    Takes the RAW stderr, not the rendered tool output. Recovering stderr by
    splitting the rendered text on ``BASH_STDERR_SEPARATOR`` misses the most
    common case there is: the separator is only inserted when stdout is
    non-empty, and a Python ``MemoryError`` prints nothing to stdout — so the
    incident's own tool result came back with no hint attached at all.

    Two guards against telling a successful command it ran out of memory, which
    would be worse than saying nothing — the model would "fix" working code:

    * A zero exit is never a memory failure, whatever the text says. ``grep
      MemoryError app.log`` succeeds and prints the marker; so does ``cat`` of a
      traceback someone committed.
    * Markers are matched against stderr only. A memory failure reports itself
      there; stdout is data the command chose to print. Exit codes are still
      authoritative on their own, since a SIGKILLed process prints nothing.
    """
    if exit_code in _OOM_EXIT_CODES:
        return True
    if not exit_code:                      # 0 or None → not a failure at all
        return False
    return any(marker in stderr for marker in _OOM_MARKERS)


def _file_size_limit_prefix() -> str:
    """POSIX-shell file-size rlimit for direct curl/wget and other children.

    ``ulimit -f`` takes a block count whose size depends on the shell: bash
    scales it by 1024, while POSIX shells (dash/ash/zsh — what ``shell=True``
    gives us on the ``CurrentSandbox`` container path) scale it by 512. Picking
    one unit would silently halve or double the intended cap depending on the
    backend, so branch on ``$BASH_VERSION`` and emit the matching block count.
    ``2>/dev/null`` matches ``_ulimit_cap``: a shell that refuses to lower the
    limit must not spray stderr into the model's tool output.
    """
    raw = (os.environ.get("BASH_FILE_MAX_BYTES") or "").strip()
    try:
        max_bytes = int(raw) if raw else _DEFAULT_FILE_MAX_BYTES
    except ValueError:
        max_bytes = _DEFAULT_FILE_MAX_BYTES
    if max_bytes <= 0:
        return ""
    kib_blocks = (max_bytes + 1023) // 1024
    posix_blocks = (max_bytes + 511) // 512
    return (
        f'if [ -n "$BASH_VERSION" ]; then ulimit -f {kib_blocks} 2>/dev/null; '
        f"else ulimit -f {posix_blocks} 2>/dev/null; fi; "
    )


def _resolve_timeout() -> int:
    """Seconds this command may run.

    Precedence, highest first:

    1. ``BASH_TIMEOUT`` in the environment — an explicit operator override,
       read per call rather than at import so exporting it from a launcher
       still takes effect and tests can set it.
    2. The agent loop's configured budget for this tool call, i.e. the
       profile's ``tool_timeout_s``.
    3. :data:`_DEFAULT_BASH_TIMEOUT`, for calls made outside the loop.

    The loop budget is a CEILING on (1): ``execute_tools`` cancels the call at
    the budget plus its own grace, so a larger override would only replace the
    structured timeout message below with a bare "tool timed out" — the model
    would lose the recovery hint and still lose the command. It is clamped, and
    the clamp is logged once so a misconfiguration is visible without spamming
    a long run.

    This function is why ``tool_timeout_s`` reaches bash at all: the deadline
    used to be a module constant frozen at import, so a profile asking for 1800s
    still got 300s and every long compile/simulation died at 5 minutes (#53).
    """
    budget = get_current_tool_budget()
    override = os.environ.get("BASH_TIMEOUT", "").strip()
    if override:
        try:
            requested = int(float(override))
        except ValueError:
            requested = 0
        if requested > 0:
            if budget is not None and requested > int(budget):
                _warn_clamped_override(requested, int(budget))
                return int(budget)
            return requested
    if budget is not None:
        return max(int(budget), 1)
    return _DEFAULT_BASH_TIMEOUT


def _warn_clamped_override(requested: int, budget: int) -> None:
    """Log a clamped ``BASH_TIMEOUT`` once per (requested, budget) pair."""
    key = (requested, budget)
    if key in _CLAMP_WARNED:
        return
    _CLAMP_WARNED.add(key)
    logger.warning(
        "BASH_TIMEOUT=%ss exceeds the loop's tool_timeout_s budget of %ss and was "
        "clamped; raise tool_timeout_s in the active profile to actually grant "
        "longer commands.",
        requested, budget,
    )


# ── Tool ────────────────────────────────────────────────────────────────


@tool
async def bash(command: str, description: str = "") -> str:
    """Execute a bash command in an isolated E2B or bubblewrap sandbox.

    Use this for: Python code execution (with any packages like matplotlib,
    numpy, pandas), shell commands, file operations, and computation.

    To run Python, pipe a script to ``python3`` via a heredoc — this is the
    preferred way and avoids the quoting/escaping pain of ``python3 -c`` while
    supporting full multi-line scripts:

        python3 <<'PY'
        import pandas as pd
        df = pd.read_csv("/inputs/data.csv")
        print(df.describe())
        PY

    Reserve ``python3 -c "..."`` for trivial one-liners.

    Args:
        command: The bash command to execute. For Python, prefer a
            ``python3 <<'PY' ... PY`` heredoc (multi-line) over ``python3 -c``.
        description: Optional description of what this command does (for logging).

    Each command runs with a per-process memory limit (1 GB by default; see
    SANDBOX_CONTAINER_MEM_MB). Processing a large dataset by loading it whole
    will hit it — read in chunks or stream instead. Hitting the limit raises
    MemoryError in the command, not an infrastructure failure.

    Genuinely long jobs (compiles, simulations, training runs) are fine to run
    in one call — the per-command deadline is the session's configured tool
    timeout, not a few minutes. Do not pre-emptively split work into chunks to
    stay under a guessed limit. If a job may exceed the deadline, launch it with
    nohup, redirect its output to a file, and poll that file on later calls; the
    timeout message says so too if you hit it.

    Returns:
        Command output (stdout + stderr), or error message.
    """
    if not command or not command.strip():
        return "Error: empty command."

    deliverable_error = bash_output_write_error(command)
    if deliverable_error:
        return f"Error: command denied. {deliverable_error}"

    assessment = assess_bash_command(command)
    if assessment.level == "deny":
        return f"Error: command denied. {assessment.reason}"
    if assessment.level == "confirm":
        # In SWE benchmark mode (per-task sandbox), downgrade to audit
        from plugins.tools._sandbox import _task_sandbox
        if _task_sandbox.get(None) is not None:
            assessment = BashCommandAssessment(level="audit", reason=assessment.reason)
        else:
            return f"Error: command requires confirmation. {assessment.reason}"

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        return f"Error: {e}"

    # Arm the socket-level download cap for any python the command spawns
    # (``python3 -c``, pip, scripts) — env exports propagate to children.
    # ``export`` (not the ``VAR=x cmd`` prefix form) so compound commands
    # (``cd x && python3 ...``) are covered too. See _net_guard.py.
    await ensure_guard_file(sandbox)
    guard_env = guard_env_prefix()
    guarded_command = (
        f"export {guard_env.rstrip()}; {command}" if guard_env else command
    )
    exec_command = f"{_file_size_limit_prefix()}{guarded_command}"

    # Resolved per call, not at import: the deadline belongs to the running
    # loop's profile, and a module constant made ``tool_timeout_s`` a no-op.
    timeout_s = _resolve_timeout()

    try:
        result = await arun_sandbox_cmd(
            sandbox,
            exec_command,
            timeout=timeout_s,
            # Current task workspaces intentionally support network-backed
            # research and document retrieval. Resource-safe document
            # downloads should use download_file; bash networking remains
            # available for APIs and existing skills.
            allow_net=True,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += BASH_STDERR_SEPARATOR
            output += result.stderr

        if not output:
            output = "(no output)"

        if result.exit_code == -1:
            # E2B's "died without a normal exit status" code (OOM kill /
            # sandbox-side failure) — usually paired with empty output.
            output = (
                "[Exit code -1 — sandbox process died unexpectedly, likely "
                f"out-of-memory or a sandbox-side failure]\n{output}"
            )
        elif result.exit_code != 0:
            output = f"[Exit code {result.exit_code}]\n{output}"

        # Turn a memory failure into an actionable instruction. The model reads
        # this text and nothing else, so a bare MemoryError traceback leaves
        # "retry the same thing" looking reasonable. See WORKER_OOM_HARDENING
        # (P0-2).
        if _looks_out_of_memory(result.stderr or "", result.exit_code):
            output += (
                "\n\n[hint] This failed on MEMORY, not on logic — retrying the same "
                "command will fail the same way. Reduce peak memory instead: read the "
                "input in chunks or line by line rather than loading it whole, write "
                "intermediate results to a file under /workspace instead of keeping "
                "them in a list, and process one item at a time."
            )

        if assessment.level == "audit":
            output = f"[Audit] {assessment.reason}\n{output}"

        # Mask host filesystem paths in output
        from plugins.tools._paths import mask_paths_in_output
        output = mask_paths_in_output(output)

        # Apply overflow handling (truncate + save full to disk if needed)
        from plugins.tools._overflow import maybe_overflow
        return maybe_overflow("bash", output)

    except TimeoutError:
        return (
            f"Error: Command timed out after {timeout_s} seconds.\n\n"
            "[hint] The command or script took too long and was interrupted. "
            "Do NOT retry the exact same long-running script. "
            "If the work genuinely needs more than this limit, start it in the "
            "background instead of shortening it — redirect its output to a file "
            "under /workspace, launch it with nohup, and poll that file on later "
            "calls. Otherwise switch to a faster method (alternative API, smaller "
            "data fetch, or pre-calculated data)."
        )
    except Exception as e:
        logger.warning("bash tool error: %s", e)
        return f"Error: {type(e).__name__}: {e}"
