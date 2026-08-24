"""The ``# Environment`` system-prompt block (Claude-Code parity).

Gives the model ground truth it otherwise guesses: the working directory,
whether it's a git repo (+ current branch), platform / OS, today's date, and
which model it is. Only static, cache-stable facts go here — NOT ``git status``
(which changes every turn and would bust prompt caching); a one-line dirty-state
note is surfaced separately at session start instead.
"""

from __future__ import annotations

import datetime
import platform
import subprocess


def _git(cwd: str, *args: str, timeout: float = 2.0) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def is_git_repo(cwd: str) -> bool:
    return _git(cwd, "rev-parse", "--is-inside-work-tree") == "true"


def dirty_summary(cwd: str) -> str:
    """Short, human-facing note about uncommitted changes (NOT for the prompt)."""
    status = _git(cwd, "status", "--porcelain")
    if not status:
        return ""
    n = len([ln for ln in status.splitlines() if ln.strip()])
    return f"{n} uncommitted change(s) in the working tree"


def environment_section(cwd: str, model: str) -> str:
    """The cache-stable env facts, formatted for a system-prompt section."""
    git = is_git_repo(cwd)
    branch = _git(cwd, "branch", "--show-current") if git else ""
    return (
        f"Working directory: {cwd}\n"
        f"Is a git repository: {'yes' if git else 'no'}"
        + (f" (branch: {branch})" if branch else "")
        + "\n"
        f"Platform: {platform.system().lower()}  ·  OS: {platform.platform()}\n"
        f"Today's date (UTC): {datetime.datetime.now(datetime.UTC).date().isoformat()}\n"
        "System time zone: UTC\n"
        f"Model: you are powered by {model}. Use this model's actual "
        "capabilities; do not assume features of other models or invent tools "
        "you were not given."
    )


__all__ = ["dirty_summary", "environment_section", "is_git_repo"]
