"""Canonical user-visible layout for one interactive FrontierAgent run."""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

_OFFSET_RE = re.compile(r"^([+-])(\d{2})(\d{2})$")


def local_timezone() -> dt.tzinfo:
    raw = os.environ.get("APODEX_LOCAL_UTC_OFFSET", "").strip()
    match = _OFFSET_RE.fullmatch(raw)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        delta = dt.timedelta(
            hours=int(match.group(2)), minutes=int(match.group(3)),
        )
        return dt.timezone(sign * delta)
    return dt.datetime.now().astimezone().tzinfo or dt.UTC


def new_run_timestamp() -> tuple[str, str, str]:
    """Return filesystem timestamp, canonical UTC ISO, and local zone label."""
    local = dt.datetime.now(local_timezone())
    utc = local.astimezone(dt.UTC)
    return (
        local.strftime("%Y%m%d-%H%M%S%z"),
        utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
        local.tzname() or local.strftime("%z"),
    )


def runs_root(workspace: str | Path | None = None) -> Path:
    configured = os.environ.get("APODEX_RUNS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(workspace or os.getcwd()).expanduser().resolve()
    return base / ".apodex" / "runs"


def run_dir(
    session_id: str,
    workspace: str | Path | None = None,
    *,
    create: bool = True,
) -> Path:
    root = runs_root(workspace)
    target = (root / session_id).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"invalid run session id: {session_id!r}") from exc
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def pinned_mounts() -> bool:
    """Whether the caller already owns ``/workspace``, ``/outputs`` and ``/inputs``.

    A launcher that binds the three canonical mount points itself — a container,
    a bubblewrap jail, any harness that hands the process a prepared filesystem —
    has already put the real directories where the tools' default aliases look.
    Repointing the aliases at run-local paths there moves the whole tool
    namespace off the mounts the caller prepared, and it does so silently: reads
    find an empty directory rather than an error, and deliverables land where
    nothing collects them.

    ``apodex/docker.py`` is the in-tree launcher that already avoids this, but
    only because it controls the image: it passes ``APODEX_WORKSPACE_LINK`` /
    ``APODEX_OUTPUTS_LINK``, which keep the aliases at ``/workspace`` and
    ``/outputs`` by *replacing those paths with symlinks* into the run directory.
    A launcher that hands the process real bind mounts cannot do that — the
    activators would have to unlink or ``rmdir`` a live mount point — and there is
    no other environment value for it to win with, because ``Session`` derives
    both paths from a runs root it sets itself. This flag is that missing lever.

    Same boundary rule as ``APODEX_RUNS_ROOT_PINNED``: only a launcher that
    controls the mounts may set this, never a session inside one.
    """
    return os.environ.get("APODEX_PINNED_MOUNTS", "").strip() == "1"


def activate_run(session_id: str, workspace: str | Path | None = None) -> Path:
    # A pinned root is a boundary mapping (container mounts) that only the
    # launcher may define; anything else follows the workspace it was given.
    repointed = (
        workspace is not None
        and os.environ.get("APODEX_RUNS_ROOT_PINNED") != "1"
    )
    if workspace is not None and repointed:
        os.environ["APODEX_RUNS_ROOT"] = str(
            Path(workspace).expanduser().resolve() / ".apodex" / "runs"
        )
    target = run_dir(session_id, workspace)
    os.environ["APODEX_SESSION_ID"] = session_id
    os.environ["APODEX_RUNS_ROOT"] = str(target.parent)
    os.environ["APODEX_RUN_DIR"] = str(target)
    if repointed and os.environ.get("APODEX_HOST_RUNS_ROOT", "").strip():
        # Without a pinned mapping the host path *is* the local path, so a
        # workspace switch must not leave the previous host root behind.
        os.environ["APODEX_HOST_RUNS_ROOT"] = str(target.parent)
    host_root = os.environ.get("APODEX_HOST_RUNS_ROOT", "").strip()
    if host_root:
        os.environ["APODEX_HOST_RUN_DIR"] = str(Path(host_root) / session_id)
    return target


def local_time_from_timestamp(timestamp: float) -> str:
    value = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC)
    local = value.astimezone(local_timezone())
    return local.strftime("%Y-%m-%d %H:%M %z")


__all__ = [
    "activate_run", "local_time_from_timestamp", "local_timezone",
    "new_run_timestamp", "pinned_mounts", "run_dir", "runs_root",
]
