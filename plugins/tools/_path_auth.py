"""Local path authorization — the fail-closed gate for host filesystem access."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from frontier_agent.core.execution_context import get_current_execution_scope

logger = logging.getLogger(__name__)

_SERVICE_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]

# Allowed directory prefixes (relative paths for project dirs, absolute for output)
_ALLOWED_RELATIVE_PREFIXES = [
    "plugins/skills/",
    "data/",
]

#: The operator-curated skills tree. Its symlinks are trusted (see
#: :func:`_candidate_paths`), so it is named once rather than spelled inline.
_SKILLS_DIR = str(_SERVICE_CHECKOUT_ROOT / "plugins" / "skills")

_ALLOWED_ABSOLUTE_PREFIXES = [
    "/tmp/agent-outputs/",
    _SKILLS_DIR,
]

# Blocked file names (security). Matched on the BASENAME at word level rather
# than as a substring of the whole path: a substring test also refused every
# path under a directory that merely contains one of these words, plus names
# like ``tokenizer_config.json``, ``secretary_notes.md`` and ``deck.keynote``.
_BLOCKED_NAME_WORDS = frozenset({
    "credential", "credentials", "secret", "secrets",
    "password", "passwords", "token", "tokens",
})
_BLOCKED_SUFFIXES = (".key", ".pem", ".cert")
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def _configured_workspace_root() -> Path | None:
    """Return the explicit workspace root, if one was configured for this task."""
    scope = get_current_execution_scope()
    metadata = scope.metadata if scope else {}
    raw_root = (
        str(metadata.get("coding_workspace_root") or metadata.get("workspace_root") or "").strip()
        or os.getenv("CODING_WORKSPACE_ROOT", "").strip()
    )
    if not raw_root:
        return None

    workspace_root = Path(raw_root).expanduser().resolve()
    if not workspace_root.is_dir():
        logger.warning("Ignoring invalid CODING_WORKSPACE_ROOT '%s'", raw_root)
        return None
    return workspace_root


def _is_isolated_workspace_root(workspace_root: Path) -> bool:
    """Host writes are only allowed for workspaces outside the service checkout."""
    service_root = _SERVICE_CHECKOUT_ROOT.resolve()
    try:
        workspace_root.relative_to(service_root)
        return False
    except ValueError:
        pass

    try:
        service_root.relative_to(workspace_root)
        return False
    except ValueError:
        return True


def _candidate_paths(
    file_path: str, workspace_root: Path | None, *, write_access: bool = False,
) -> list[Path]:
    """Resolve a path against the explicit workspace root before falling back locally."""
    raw_path = Path(file_path)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        if resolved == raw_path:
            return [resolved]
        # A symlink may only WIDEN access from inside the operator-curated
        # plugins/skills/ tree, whose links deliberately point at SKILL.md
        # bodies outside the project — and then only for READS, since nothing
        # in that tree is a write target. Everywhere else (and for every write)
        # the resolved target is the only candidate: the task workspace is
        # model-writable, so also accepting the unresolved path there let the
        # model ``ln -s ~/.ssh`` into the workspace and read the target
        # straight back through this gate. That reaches past bubblewrap too —
        # it jails ``bash``, while the file tools do in-process host IO, so
        # this gate is their only boundary.
        #
        # The resolved form comes FIRST so the blocked-name check in
        # :func:`_authorized_local_path` sees the real target: ordered the other
        # way, a curated ``SKILL.md -> .env.prod`` link would be judged by the
        # link's own harmless name.
        if _is_skill_path(raw_path) and not write_access:
            return [resolved, raw_path]
        return [resolved]

    if workspace_root is not None:
        candidates.append((workspace_root / raw_path).resolve())
    candidates.append(raw_path.resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _is_skill_path(path: Path) -> bool:
    """True for a path inside the operator-curated ``plugins/skills/`` tree."""
    return _path_within(str(path), _resolve_prefix(_SKILLS_DIR))


def _blocked_name(name: str) -> str:
    """The blocked pattern *name* trips, else ``""``.

    Word-level so ``token`` refuses ``api_token.txt`` but not
    ``tokenizer_config.json``, and suffix-level so ``.key`` refuses
    ``server.key`` but not ``deck.keynote``.
    """
    lower = name.lower()
    if lower == ".env" or lower.startswith(".env."):
        return ".env"
    for suffix in _BLOCKED_SUFFIXES:
        if lower.endswith(suffix):
            return suffix
    for word in _WORD_SPLIT.split(lower):
        if word in _BLOCKED_NAME_WORDS:
            return word
    return ""


def _authorized_local_path(file_path: str, *, write_access: bool = False) -> tuple[Path | None, str]:
    """Resolve a local path and verify it stays inside approved prefixes."""
    normalized = os.path.normpath(file_path)

    # Block path traversal
    if ".." in normalized:
        return None, "Path traversal (..) is not allowed"

    workspace_root = _configured_workspace_root()
    all_prefixes = _allowed_local_prefixes(
        write_access=write_access,
        workspace_root=workspace_root,
    )
    # The blocked-name test must see the REAL target, so it runs against the
    # fully-resolved name as well as the candidate's own. A curated
    # ``plugins/skills/x/SKILL.md -> .env.prod`` link is authorized through the
    # unresolved candidate (that is the point of the exception), and judging
    # only that candidate would let the link's harmless name stand in for the
    # secret it points at.
    target_name = Path(normalized).resolve().name
    for candidate in _candidate_paths(
        normalized, workspace_root, write_access=write_access,
    ):
        resolved = str(candidate)
        for prefix in all_prefixes:
            prefix_resolved = _resolve_prefix(prefix)
            if _path_within(resolved, prefix_resolved):
                blocked = _blocked_name(candidate.name) or _blocked_name(target_name)
                if blocked:
                    return None, (
                        f"Access to files matching '{blocked}' is blocked for security"
                    )
                return candidate, ""

    return None, f"Access restricted. Allowed directories: {', '.join(all_prefixes)}"


def _is_path_allowed(file_path: str, *, write_access: bool = False) -> tuple[bool, str]:
    """Check if a local path is allowed for the requested access mode."""
    resolved_path, reason = _authorized_local_path(file_path, write_access=write_access)
    return resolved_path is not None, reason


def _resolve_prefix(prefix: str) -> str:
    """Resolve an allowed prefix to an absolute, symlink-resolved path.

    Uses Path.resolve() for all paths so that symlink targets match
    (e.g., on macOS /tmp → /private/tmp).
    """
    return str(Path(prefix).resolve())


def _path_within(path: str, prefix: str) -> bool:
    """Check whether a resolved path is inside a prefix."""
    norm_path = os.path.normpath(path)
    norm_prefix = os.path.normpath(prefix)
    return norm_path == norm_prefix or norm_path.startswith(norm_prefix + os.sep)


def _resolve_inputs_dir() -> Path | None:
    """Return the mounted read-only ``/inputs`` dir, but only when it exists.

    Task input files are bind-mounted read-only at ``/inputs`` (container/serve
    mode; overridable via ``FRONTIER_AGENT_INPUTS_DIR``). Gating on the dir actually
    existing means non-container runs (local bwrap, tests — no ``/inputs``) get
    no new prefix and are unaffected. Imported lazily to avoid an import cycle
    with ``_sandbox``.
    """
    try:
        from plugins.tools._sandbox import resolve_mount_dirs

        inputs_dir = Path(resolve_mount_dirs()[2]).expanduser().resolve()
    except Exception:
        return None
    return inputs_dir if inputs_dir.is_dir() else None


def task_input_matcher() -> Callable[[str | Path], bool]:
    """Resolve the read-only input root once and return a per-path predicate.

    Task inputs may intentionally live below a repository-ignored runtime
    directory (for example ``.apodex/``). Search tools use this signal to avoid
    applying repository ignore rules to the separately-authorized input mount;
    normal path authorization and per-result symlink checks still apply.

    The root lookup imports ``_sandbox``, reads the environment and stats the
    mount, so it must not run once per candidate file: a search over a large
    checkout would spend more time re-deriving a constant than reading files.
    Runs with no input mount get a predicate that costs nothing at all.
    """
    inputs_dir = _resolve_inputs_dir()
    if inputs_dir is None:
        return lambda _file_path: False
    root = str(inputs_dir)

    def _within(file_path: str | Path) -> bool:
        try:
            candidate = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        return _path_within(str(candidate), root)

    return _within


def _resolve_spill_dirs() -> list[Path]:
    """Return the spill directories this conversation may read.

    Authorized for READ so ``read_file`` / ``grep_search`` can recover a body
    compaction dropped, and never for write — the same shape as ``/inputs``. The
    canonical ``/spill`` path a model sees is rewritten to this by
    ``resolve_runtime_path`` before it reaches here. Gating on existence means a
    run that never spilled adds no prefix. Imported lazily to avoid an import
    cycle with ``_sandbox``.
    """
    try:
        from plugins.tools._overflow import _created_stores, _current_task_id
        from plugins.tools._overflow import _scope_component as scope_of
        from plugins.tools._sandbox import spill_root

        root = spill_root()
    except Exception:
        return []
    if not root.is_dir():
        return []

    # Narrower than the root on purpose. The root is shared — a temp directory,
    # or a run directory — so authorizing it would let one conversation read
    # another's spilled tool results, which the old in-workspace layout made
    # impossible. Two things are authorized instead:
    #
    #   * this conversation's own scope, which is what its recovery index names;
    #   * every store THIS process created, because in-process sub-agents spill
    #     under their own scope and a fan-in report can carry one of those paths
    #     back to the parent.
    #
    # A different session in a different process matches neither.
    allowed: list[Path] = []
    scope = scope_of(_current_task_id())
    if scope and (root / scope).is_dir():
        allowed.append(root / scope)
    allowed.extend(store for store in _created_stores if store.is_dir())
    return allowed


def _allowed_local_prefixes(
    *,
    write_access: bool = False,
    workspace_root: Path | None = None,
) -> list[str]:
    """Return the local path prefixes allowed in the current execution context."""
    prefixes = list(_ALLOWED_RELATIVE_PREFIXES) + list(_ALLOWED_ABSOLUTE_PREFIXES)
    resolved_workspace_root = workspace_root or _configured_workspace_root()
    if resolved_workspace_root is not None:
        if write_access and not _is_isolated_workspace_root(resolved_workspace_root):
            logger.warning(
                "Refusing local write access to non-isolated workspace root '%s'",
                resolved_workspace_root,
            )
        else:
            prefixes.append(str(resolved_workspace_root))
    # Uploaded task inputs live under a read-only ``/inputs`` mount. Authorize
    # them for READ so grep_search / glob_search / read_text can list and search
    # them; never for write (the mount is read-only).
    if not write_access:
        inputs_dir = _resolve_inputs_dir()
        if inputs_dir is not None:
            prefixes.append(str(inputs_dir))
        # Recovery reads of spilled tool results. READ ONLY, and deliberately
        # absent from the write branch: that omission is what makes the store
        # read-only to every file tool, replacing a special case each writer had
        # to remember.
        prefixes.extend(str(path) for path in _resolve_spill_dirs())
    return prefixes
