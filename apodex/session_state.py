import os
import uuid
from pathlib import Path
from typing import Any


def new_session_id(mode: str) -> str:
    """Return a readable local-time run id with an explicit UTC offset."""
    from apodex.run_layout import new_run_timestamp

    timestamp, _utc, _zone = new_run_timestamp()
    return f"{timestamp}-{mode}-{uuid.uuid4().hex[:4]}"


def _session_state_path(session_id: str) -> str:
    from apodex.run_layout import run_dir

    return str(run_dir(session_id, create=False) / "session.json")


def _legacy_session_roots() -> list[Path]:
    roots = [Path(os.path.expanduser("~/.apodex/sessions"))]
    configured = os.environ.get("APODEX_LEGACY_SESSION_ROOTS", "")
    roots.extend(Path(value) for value in configured.split(os.pathsep) if value)
    return roots


def load_session_state(session_id: str) -> dict | None:
    """Load a persisted session checkpoint by id (for ``--resume``), or None."""
    import json
    try:
        candidates = [_session_state_path(session_id)]
        candidates.extend(str(root / f"{session_id}.json") for root in _legacy_session_roots())
        for candidate in candidates:
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
            except OSError:
                continue
    except Exception:
        return None


def list_saved_sessions(
    extra_roots: list[str] | None = None,
    workspace: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return saved-session metadata, newest first, for ``--resume`` listings.

    ``workspace`` selects the run tree to read when the process has not entered
    it yet (``--resume`` is answered before the ``--cwd`` chdir).

    Checkpoints are user-local and may be interrupted or manually edited, so a
    malformed entry is skipped rather than making every other session hidden.
    """
    import json

    from apodex.run_layout import local_time_from_timestamp, runs_root

    legacy_roots = _legacy_session_roots()
    legacy_roots.extend(Path(root) for root in (extra_roots or []))
    try:
        paths = {
            path.resolve() for root in legacy_roots for path in root.glob("*.json")
        }
        paths.update(
            path.resolve() for path in runs_root(workspace).glob("*/session.json")
        )
        paths = sorted(
            paths,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []

    sessions: list[dict[str, Any]] = []
    # A resumed legacy checkpoint is rewritten into the run tree without the
    # old file being removed, so the same id can be found twice. Newest first
    # means the first hit is the live one.
    seen: set[str] = set()
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                continue
            # In the run layout the id is the directory, not the file name.
            fallback = path.parent.name if path.name == "session.json" else path.stem
            session_id = str(state.get("session_id") or fallback)
            if session_id in seen:
                continue
            seen.add(session_id)
            sessions.append({
                "session_id": session_id,
                "name": str(state.get("name") or ""),
                "mode": str(state.get("mode") or "unknown"),
                "cwd": str(state.get("cwd") or "unknown directory"),
                "message_count": len(state.get("history") or []),
                "modified_at": local_time_from_timestamp(path.stat().st_mtime),
            })
        except (OSError, ValueError, TypeError):
            continue
    return sessions
