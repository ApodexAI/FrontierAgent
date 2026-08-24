"""Session-scoped user attachments for the terminal client.

The harness writes copies through ``staging_dir`` while agents only receive
``agent_dir``.  Docker mounts the same host directory at those two locations:
read-write for the trusted TUI and read-only at ``/inputs`` for tools.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Attachment:
    relative_path: str
    agent_path: str
    size: int


class AttachmentError(ValueError):
    """A user-facing attachment validation or copy error."""


class AttachmentManager:
    def __init__(self, cwd: str, session_id: str) -> None:
        # Relative /attach paths are user-facing workspace paths.  Resolve them
        # against the session's explicit --cwd instead of the process cwd,
        # which can differ in Docker, the TUI, and after an in-app /resume.
        self.set_source_root(cwd)
        # Keep copies outside the writable workspace. Otherwise Docker would
        # expose the same inode read-only at /inputs and read-write through
        # /workspace/.apodex/inputs, defeating the attachment boundary.
        configured_staging = os.environ.get("APODEX_INPUT_STAGING_DIR", "").strip()
        staging_root = os.environ.get("APODEX_INPUT_STAGING_ROOT", "").strip()
        configured_agent = os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "").strip()
        agent_root = os.environ.get("FRONTIER_AGENT_INPUTS_ROOT", "").strip()

        from apodex.run_layout import pinned_mounts

        mounts_are_pinned = pinned_mounts()
        if mounts_are_pinned:
            # A jail bound the task's files at ``/inputs`` before this process
            # started, so the inherited alias IS the truth. Checked ahead of
            # every branch below because those all derive a session-scoped
            # directory instead, and ``Session`` then copies whatever we pick
            # back into ``FRONTIER_AGENT_INPUTS_DIR`` — pointing the read tools
            # at an empty directory while the real corpus sits on the mount.
            #
            # Both paths are kept exactly as the launcher declared them, not
            # ``resolve()``d: the mount point is the name every tool works in
            # (``resolve_runtime_path`` rewrites prefixes, it does not follow
            # links), so canonicalising it here would rename the namespace.
            self.agent_dir = Path(configured_agent or "/inputs").expanduser()
            self.staging_dir = Path(
                configured_staging or self.agent_dir
            ).expanduser()
        elif configured_staging:
            # The native Docker launcher mounts one already session-scoped host
            # directory at both paths, so its exact directory overrides win.
            self.staging_dir = Path(configured_staging).expanduser().resolve()
            self.agent_dir = Path(configured_agent or self.staging_dir)
        elif staging_root:
            # Compose cannot know the generated session id before the process
            # starts. Give it mount roots and scope both aliases here instead.
            self.staging_dir = (
                Path(staging_root).expanduser().resolve() / session_id
            )
            self.agent_dir = Path(agent_root or staging_root) / session_id
        else:
            # A local terminal session owns its private copies. Do not inherit
            # FRONTIER_AGENT_INPUTS_DIR: TerminalSession updates that variable
            # as sessions change in the same process.
            self.staging_dir = (
                Path.home() / ".apodex-inputs" / session_id
            ).expanduser().resolve()
            self.agent_dir = self.staging_dir
        if mounts_are_pinned:
            # Pinned paths are launcher-owned mount points, not directories for
            # the session to provision. Creating a missing path here masks a
            # broken mount with an empty local directory. Validate both aliases
            # because staging and agent paths may be configured separately.
            for label, path in (
                ("staging", self.staging_dir),
                ("agent", self.agent_dir),
            ):
                if not path.is_dir():
                    raise AttachmentError(
                        f"pinned input {label} directory does not exist: {path}"
                    )
        else:
            # Preserve the old fail-fast behavior for session-owned paths. Any
            # mkdir error here matters even when a later is_dir() probe happens
            # to succeed (for example an I/O or permission failure).
            self.staging_dir.mkdir(parents=True, exist_ok=True)

    def set_source_root(self, cwd: str) -> None:
        """Update the workspace used to resolve relative source paths."""
        self.source_root = Path(cwd).expanduser().resolve()

    def list(self) -> list[Attachment]:
        items: list[Attachment] = []
        try:
            paths = sorted(self.staging_dir.rglob("*"))
        except OSError as exc:
            raise AttachmentError(f"cannot list attachments: {exc}") from exc
        for path in paths:
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(self.staging_dir).as_posix()
                items.append(Attachment(
                    relative_path=relative,
                    agent_path=(self.agent_dir / relative).as_posix(),
                    size=path.stat().st_size,
                ))
            except OSError:
                continue
        return items

    def attach_many(self, paths: Iterable[str]) -> list[Attachment]:
        added: list[Attachment] = []
        for raw in paths:
            added.extend(self.attach(raw))
        return added

    def attach(self, raw_path: str) -> list[Attachment]:
        source = Path(raw_path).expanduser()
        relative_source = not source.is_absolute()
        if relative_source:
            source = self.source_root / source
        if source.is_symlink():
            raise AttachmentError(f"symbolic links cannot be attached: {raw_path}")
        try:
            source = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            location = (
                f" (looked under {self.source_root})" if relative_source else ""
            )
            raise AttachmentError(f"file not found: {raw_path}{location}") from exc
        if relative_source and source != self.source_root and self.source_root not in source.parents:
            raise AttachmentError(
                f"relative attachment path escapes workspace {self.source_root}: {raw_path}"
            )
        if source == self.staging_dir or self.staging_dir in source.parents:
            return self.list()
        if source.is_file():
            existing = self.staging_dir / source.name
            try:
                if existing.is_file() and filecmp.cmp(source, existing, shallow=False):
                    return [self._attachment_for(existing)]
            except OSError:
                pass
            target = self._available_target(source.name)
            self._copy_file(source, target)
            return [self._attachment_for(target)]
        if source.is_dir():
            try:
                linked = next((path for path in source.rglob("*") if path.is_symlink()), None)
            except OSError as exc:
                raise AttachmentError(f"cannot inspect directory {raw_path}: {exc}") from exc
            if linked is not None:
                raise AttachmentError(
                    f"directories containing symbolic links cannot be attached: {linked}"
                )
            target = self._available_target(source.name)
            try:
                shutil.copytree(source, target, symlinks=False)
                self._normalize_tree_permissions(target)
            except OSError as exc:
                raise AttachmentError(f"cannot attach directory {raw_path}: {exc}") from exc
            return [item for item in self.list() if item.relative_path.startswith(target.name + "/")]
        raise AttachmentError(f"unsupported attachment path: {raw_path}")

    def detach(self, name: str) -> int:
        candidate = (self.staging_dir / name).resolve()
        if candidate == self.staging_dir or self.staging_dir not in candidate.parents:
            raise AttachmentError(f"invalid attachment name: {name}")
        if not candidate.exists():
            matches = [
                self.staging_dir / item.relative_path
                for item in self.list()
                if Path(item.relative_path).name == name
            ]
            if len(matches) != 1:
                raise AttachmentError(f"attachment not found: {name}")
            candidate = matches[0]
        try:
            if candidate.is_dir():
                count = sum(1 for path in candidate.rglob("*") if path.is_file())
                shutil.rmtree(candidate)
                return count
            candidate.unlink()
            return 1
        except OSError as exc:
            raise AttachmentError(f"cannot detach {name}: {exc}") from exc

    def enrich_task(
        self, task: str, *, delegate_file_reading: bool = False,
    ) -> str:
        attachments = self.list()
        if not attachments:
            return task
        inspection_instruction = (
            "The user attached these read-only files. Delegate full inspection "
            "to a sub-agent with file-reading capability. Do not call or invent "
            "a file-reading tool yourself. "
            "Use `glob_search` / `grep_search` only for discovery or brief "
            "inspection, and obtain the file contents through the sub-agent's "
            "report."
            if delegate_file_reading
            else
            "The user attached these read-only files. Inspect relevant files "
            "or images (charts, photos, diagrams) with read_file before answering."
        )
        lines = [
            task,
            "",
            "<attached_files>",
            inspection_instruction,
        ]
        lines.extend(
            f"- path={json.dumps(item.agent_path)} size={item.size} bytes"
            for item in attachments
        )
        lines.append("</attached_files>")
        return "\n".join(lines)

    def _available_target(self, basename: str) -> Path:
        safe = Path(basename).name or "attachment"
        if any(ord(char) < 32 or ord(char) == 127 for char in safe):
            raise AttachmentError("attachment names cannot contain control characters")
        candidate = self.staging_dir / safe
        stem, suffix = Path(safe).stem, Path(safe).suffix
        index = 2
        while candidate.exists():
            candidate = self.staging_dir / f"{stem}-{index}{suffix}"
            index += 1
        return candidate

    @staticmethod
    def _copy_file(source: Path, target: Path) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
            target.chmod(0o644)
        except OSError as exc:
            raise AttachmentError(f"cannot attach {source}: {exc}") from exc

    def _attachment_for(self, path: Path) -> Attachment:
        relative = path.relative_to(self.staging_dir).as_posix()
        return Attachment(relative, (self.agent_dir / relative).as_posix(), path.stat().st_size)

    @staticmethod
    def _normalize_tree_permissions(root: Path) -> None:
        """Keep Docker's unprivileged tool user from mutating the RW staging alias."""
        root.chmod(0o755)
        for path in root.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
