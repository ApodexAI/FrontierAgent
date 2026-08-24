"""WorkspaceJournal — track file mutations so they are diffable and revertable.

A single session-lived object that snapshots a file's content the **first**
time a mutating tool touches it. From those originals it can, at any point,
report what changed (changed-files + diffstat), show a per-file diff, and
revert every change back to the session's starting state.

This is the one place that knows "what did the agent change", so it powers
three acceptance items at once: revert, the deterministic changed-files
summary, and treating ``delete_file`` as a first-class (revertable) op.
"""

from __future__ import annotations

import difflib
import os
import stat as stat_module
from dataclasses import dataclass, field

_SCAN_MAX_BYTES = 5 * 1024 * 1024
#: Ceiling on the text a single ``begin_tree_scan`` keeps in memory. The
#: baseline has to be captured *before* the command runs — once bash has
#: written, the old bytes are gone — so it cannot be made lazy. Past the
#: budget a file is snapshotted as opaque, which under-reports rather than
#: mis-attributing, and keeps a big monorepo from pinning gigabytes per call.
_SCAN_MAX_TOTAL_BYTES = 64 * 1024 * 1024
#: Ceiling on the baseline text a session state file carries. The scan budget
#: above is per tool phase and phases accumulate, so without this the state
#: file grows without limit and every save rewrites all of it.
_PERSIST_MAX_TOTAL_BYTES = 8 * 1024 * 1024
_SCAN_EXCLUDED_DIRS = frozenset({
    ".apodex", ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "__pycache__", "node_modules",
})
type _Fingerprint = tuple[int, int]
#: Stand-in fingerprint for "this path does not exist right now". Real files
#: always have a non-negative size, so it can never collide with one.
_ABSENT: _Fingerprint = (-1, -1)
#: ``path -> (fingerprint, text)``. The text is ``None`` when the file exists
#: but has no usable baseline (binary, oversized, unreadable, over budget).
#: Such files are still *listed*, so that "absent from the baseline" keeps its
#: one meaning: the path did not exist before the call.
type _TreeSnapshot = dict[str, tuple[_Fingerprint, str | None]]


def _read_or_none(path: str) -> str | None:
    """Current text content of ``path``, or ``None`` if it doesn't exist /
    can't be read as text (treated as 'absent' for snapshot purposes)."""
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as fb:
            if b"\x00" in fb.read(8192):
                return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _read_small_text(path: str) -> str | None:
    """Read a scan candidate without retaining large or binary files."""
    try:
        if os.path.islink(path) or os.path.getsize(path) > _SCAN_MAX_BYTES:
            return None
        with open(path, "rb") as f:
            raw = f.read(_SCAN_MAX_BYTES + 1)
        if len(raw) > _SCAN_MAX_BYTES or b"\x00" in raw:
            return None
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


def _fingerprint(path: str) -> _Fingerprint:
    """``(size, mtime_ns)`` for ``path``, or :data:`_ABSENT` if it is gone.

    Cheap enough to run over the whole journal on every UI tick, which is what
    lets :meth:`WorkspaceJournal.report` skip re-reading untouched files.
    """
    try:
        st = os.stat(path)
    except OSError:
        return _ABSENT
    return (st.st_size, st.st_mtime_ns)


def _walk_files(roots: list[str]) -> dict[str, _Fingerprint]:
    """Return every regular file below ``roots``, without following symlinks.

    Deliberately unfiltered by size: the *only* thing a missing entry may mean
    is "this path is gone". Dropping oversized files here would make a file
    that bash grew past the cap look deleted, and ``/revert`` would then
    truncate it back to the pre-command text.

    Paths are canonical without a per-file ``realpath``: each root is resolved
    once and symlinked directories are never descended into, so nothing below
    a root can reach it by a second name.
    """
    files: dict[str, _Fingerprint] = {}
    seen_roots: set[str] = set()
    for raw_root in roots:
        if not raw_root:
            continue
        root = os.path.realpath(raw_root)
        if root in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(root)
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [
                name for name in dirs
                if name not in _SCAN_EXCLUDED_DIRS
                and not os.path.islink(os.path.join(directory, name))
            ]
            for name in names:
                path = os.path.join(directory, name)
                try:
                    st = os.stat(path, follow_symlinks=False)
                except OSError:
                    continue
                # One lstat decides everything: symlinks, fifos and sockets are
                # not regular files and never carry a diffable baseline.
                if not stat_module.S_ISREG(st.st_mode):
                    continue
                files.setdefault(path, (st.st_size, st.st_mtime_ns))
    return files


@dataclass
class WorkspaceJournal:
    """Records pre-change snapshots of files the agent mutates under ``cwd``."""

    cwd: str
    # abspath -> content before the FIRST change this session (None = absent).
    _original: dict[str, str | None] = field(default_factory=dict)
    # Paths whose baseline came from a tree scan rather than from a tool that
    # named them. Shown in the diff, never written back by ``revert_all``:
    # a before/after scan cannot tell the shell's writes apart from anything
    # else that touched the tree in the same window — the user's own editor,
    # a watcher, a dev server — and reverting those would destroy work the
    # session never did.
    _observed: set[str] = field(default_factory=set)
    # abspath -> the content ``revert_all`` writes back, when that differs from
    # the diff baseline. Only scan-discovered paths that a later tool named get
    # an entry: the diff still starts from the scan baseline, but the revert
    # rewinds no further than the moment attribution began.
    _revert_base: dict[str, str | None] = field(default_factory=dict)
    # abspath -> (fingerprint, diffstat-or-None, chunk). Memoises ``report()``
    # so a 1 Hz poll re-reads only the files that actually moved.
    _diff_cache: dict[
        str, tuple[_Fingerprint, tuple[str, int, int] | None, str]
    ] = field(default_factory=dict, repr=False, compare=False)

    def _abs(self, path: str) -> str:
        p = path if os.path.isabs(path) else os.path.join(self.cwd, path)
        return os.path.realpath(p)

    def _rel(self, abspath: str) -> str:
        try:
            return os.path.relpath(abspath, os.path.realpath(self.cwd))
        except Exception:
            return abspath

    def record_before(self, path: str) -> None:
        """Snapshot ``path``'s current content the first time it's touched."""
        ap = self._abs(path)
        current = _read_or_none(ap)
        if current is None and os.path.lexists(ap):
            # Something is there, but it leaves no text baseline: a binary, an
            # unreadable file, a directory, a broken symlink. ``None`` means
            # "absent" everywhere else here, so journaling it would render a
            # pre-existing file as a create and have ``/revert`` delete it.
            # :meth:`finish_tree_scan` skips the same case for the same reason.
            return
        if ap not in self._original:
            self._original[ap] = current
            return
        if ap in self._observed:
            # A tool naming this path supplies the attribution ``_observed``
            # lacks — but only from *now* on. The scan baseline still describes
            # a window that may contain the user's own editor save, so it stays
            # the diff's starting point while the revert target becomes the
            # content as it stands at the moment attribution begins.
            self._revert_base[ap] = current
            self._observed.discard(ap)

    def begin_tree_scan(self, roots: list[str]) -> _TreeSnapshot:
        """Capture an ephemeral baseline for a tool with unknown write targets.

        Shell commands can modify several paths and expose no structured
        ``path`` argument.  The returned snapshot is intentionally not stored
        on the journal or persisted: only files that actually change are
        promoted into ``_original`` by :meth:`finish_tree_scan`.

        This blocks on the whole tree, so callers must run it off the event
        loop — ``TerminalObserver.on_tool_call`` hands it to a thread.
        """
        snapshot: _TreeSnapshot = {}
        budget = _SCAN_MAX_TOTAL_BYTES
        for path, fingerprint in _walk_files(roots).items():
            content = None
            if fingerprint[0] <= min(_SCAN_MAX_BYTES, budget):
                content = _read_small_text(path)
                if content is not None:
                    budget -= fingerprint[0]
            snapshot[path] = (fingerprint, content)
        return snapshot

    def finish_tree_scan(self, roots: list[str], before: _TreeSnapshot) -> None:
        """Record text files changed since ``before`` as observed-only changes."""
        after = _walk_files(roots)

        def observe(path: str, baseline: str | None) -> None:
            if path in self._original:
                return          # an earlier, better-attributed baseline wins
            self._original[path] = baseline
            self._observed.add(path)

        for path, (fingerprint, old_content) in before.items():
            current_fingerprint = after.pop(path, None)
            if old_content is None:
                # The file predates the call but has no recoverable baseline.
                # Recording it either way is worse than ignoring it: as a
                # create ``/revert`` would delete a file the session never
                # made, and as an edit the diff would invent content.
                continue
            if current_fingerprint is None:
                observe(path, old_content)
                continue
            if current_fingerprint[0] != fingerprint[0]:
                # A different size is a different file; no read needed.
                observe(path, old_content)
                continue
            # Same size: the fingerprint CANNOT prove this file is unchanged, so
            # the content decides. ``mtime_ns`` looks precise and is not — its
            # real resolution is the filesystem's, and on an overlayfs container
            # or a coarse network mount two writes microseconds apart share a
            # timestamp exactly (measurably: st_mtime_ns delta 0). Skipping the
            # read on an equal fingerprint therefore missed same-size edits made
            # inside one tick, which is most of them: the file stayed
            # unattributed, so ``/revert`` could not undo it and the sidebar
            # never showed it.
            #
            # Cost: one bounded re-read per same-size baselined file per scan,
            # on top of the read ``begin_tree_scan`` already did. Both are capped
            # by _SCAN_MAX_BYTES / _SCAN_MAX_TOTAL_BYTES and run off the event
            # loop. If it ever shows up in a profile, the sound optimisation is
            # to keep the fast path only for files whose mtime predates the scan
            # by more than one filesystem tick — not to trust equal fingerprints.
            if _read_small_text(path) != old_content:
                observe(path, old_content)

        # Everything still in ``after`` is absent from the baseline, and the
        # baseline lists every file that existed — so these paths are new.
        for path in after:
            if _read_small_text(path) is not None:
                observe(path, None)

    # ── reporting ─────────────────────────────────────────────────────────
    def report(self) -> tuple[list[tuple[str, int, int]], str]:
        """Return ``(diffstat, unified_diff)`` from ONE pass over the journal.

        The two halves are what the sidebar shows together — a header counting
        files and lines above the hunks those counts describe. Computing them
        separately reads every file twice and takes the two snapshots at
        different instants, so a file rewritten in between makes the header
        disagree with the body underneath it.

        The TUI polls this once a second, so an unchanged file must cost one
        ``stat`` and nothing more: results are memoised per path against the
        file's ``(size, mtime_ns)``.

        KNOWN LIMIT: that key inherits the filesystem's mtime resolution, which
        on an overlayfs container or a coarse network mount cannot separate two
        writes in the same tick. A same-size edit landing inside one tick keeps
        the cached hunks until the file's mtime next moves, so the sidebar can
        show a stale diff. Not fixed here on purpose — the per-second budget in
        this docstring is the constraint, and re-reading every journaled file
        each tick to close it is the wrong trade. ``finish_tree_scan`` does not
        share the limit: attribution and ``/revert`` correctness depend on it, so
        it compares content whenever the size is unchanged.
        """
        stats: list[tuple[str, int, int]] = []
        chunks: list[tuple[str, str]] = []
        fresh: dict[str, tuple[_Fingerprint, tuple[str, int, int] | None, str]] = {}
        # A TUI reader may take this snapshot in a worker thread while the
        # observer records another file on the event-loop thread.
        for ap, orig in list(self._original.items()):
            # Fingerprint first: a write landing between here and the read
            # caches new content under the old key, which the next tick's
            # differing fingerprint discards. The reverse order would cache
            # stale content under the new key and never re-read it.
            fingerprint = _fingerprint(ap)
            cached = self._diff_cache.get(ap)
            if cached is not None and cached[0] == fingerprint:
                fresh[ap] = cached
                if cached[1] is not None:
                    stats.append(cached[1])
                    chunks.append((cached[1][0], cached[2]))
                continue
            # A scan-observed path is read under the scan's own limits. It was
            # baselined that way, it can never be reverted, and an unbounded
            # read here is reachable from a plain ``bash`` append: a 100 KB
            # ``build.log`` grown to 200 MB would materialise 200 MB of text
            # plus a two-million-line diff, and then cache both.
            if ap in self._observed:
                cur = _read_small_text(ap)
                if cur is None and os.path.exists(ap):
                    # Still on disk, just no longer diffable. Letting ``None``
                    # through would render the file as deleted by the session.
                    fresh[ap] = (fingerprint, None, "")
                    continue
            else:
                cur = _read_or_none(ap)
            if cur == orig:
                fresh[ap] = (fingerprint, None, "")
                continue
            relative = self._rel(ap)
            lines = list(difflib.unified_diff(
                (orig or "").splitlines(keepends=True),
                (cur or "").splitlines(keepends=True),
                fromfile="/dev/null" if orig is None else f"a/{relative}",
                tofile="/dev/null" if cur is None else f"b/{relative}",
            ))
            if not lines:
                # ``unified_diff`` of two empty sequences is empty — headers
                # included. An empty create (``touch``, an empty ``write_file``)
                # would become a ``(path, 0, 0)`` stat carrying nothing to
                # render: the tab opens, the status bar counts a file, and the
                # pane names none. Synthesise the headers so the path shows.
                lines = [
                    f"--- {'/dev/null' if orig is None else f'a/{relative}'}\n",
                    f"+++ {'/dev/null' if cur is None else f'b/{relative}'}\n",
                ]
            # Count off the diff that is already being built. ``ndiff`` answers
            # the same question by recursive intra-line matching, which is
            # super-quadratic — a wholly rewritten 800-line file exhausts the
            # recursion limit outright. Skip the two file headers rather than
            # filtering on ``---``/``+++``, which a removed line reading ``--``
            # would be mistaken for.
            added = sum(1 for line in lines[2:] if line.startswith("+"))
            removed = sum(1 for line in lines[2:] if line.startswith("-"))
            stat = (relative, added, removed)
            stats.append(stat)
            # ``difflib`` preserves a missing final newline on content lines;
            # joining those verbatim would fuse ``-old`` and ``+new`` into one
            # visually misclassified line in the pane.
            chunk = "".join(
                line if line.endswith("\n") else line + "\n" for line in lines
            )
            chunks.append((relative, chunk))
            fresh[ap] = (fingerprint, stat, chunk)
        # Rebuilt rather than pruned, so paths dropped by ``revert_all`` (or by
        # a replaced journal) cannot keep an entry alive.
        self._diff_cache = fresh
        chunks.sort(key=lambda item: item[0])
        return sorted(stats), "".join(chunk for _, chunk in chunks)

    def diffstat(self) -> list[tuple[str, int, int]]:
        """``[(relpath, added_lines, removed_lines)]`` for each changed file."""
        return self.report()[0]

    def revertable_diffstat(self) -> list[tuple[str, int, int]]:
        """:meth:`diffstat` less the paths :meth:`revert_all` will not touch.

        What the sidebar shows and what ``/revert`` acts on are deliberately
        different sets, so a summary titled "``/revert`` to undo" has to be the
        second one. The difference is not marginal: a task that shells out to a
        build lists every file under ``dist/`` or ``target/`` as observed.
        """
        skip = {self._rel(ap) for ap in self._observed}
        return [stat for stat in self.diffstat() if stat[0] not in skip]

    def unified_diff(self) -> str:
        """Return the current session changes as one Git-style unified diff.

        The journal is the source of truth rather than ``git diff`` itself: it
        excludes changes that already existed before this session and works in
        non-Git directories, while retaining the familiar ``a/`` / ``b/``
        headers and ``/dev/null`` markers for creates and deletes.
        """
        return self.report()[1]

    def observed_only(self) -> list[str]:
        """Changed paths the diff shows but :meth:`revert_all` will not undo.

        Read under the scan's limits, like the rest of an observed path's
        lifetime: an oversized or binary current state reads as ``None``, which
        still differs from the text baseline, so the path stays listed without
        the file being pulled into memory to say so.
        """
        return sorted(
            self._rel(ap) for ap in self._observed
            if ap in self._original and _read_small_text(ap) != self._original[ap]
        )

    # ── revert ──────────────────────────────────────────────────────────────
    def revert_all(self) -> list[str]:
        """Restore every *attributed* change to its session-start state.

        Returns the relative paths reverted. Paths known only from a tree scan
        are skipped — see ``_observed`` — and reported by :meth:`observed_only`
        so the gap is stated rather than silent. Clears the journal afterwards.
        """
        reverted = []
        # A path whose baseline came from a scan is rewound only to where its
        # attribution began, so what is left on disk afterwards is once again
        # a change this journal cannot claim. It goes back to being observed.
        re_observed: set[str] = set()
        for ap, orig in list(self._original.items()):
            if ap in self._observed:
                continue
            target = self._revert_base.get(ap, orig)
            if ap in self._revert_base:
                re_observed.add(ap)
            cur = _read_or_none(ap)
            if cur == target:
                continue
            try:
                if target is None:
                    if os.path.exists(ap):
                        os.remove(ap)
                else:
                    os.makedirs(os.path.dirname(ap) or ".", exist_ok=True)
                    with open(ap, "w", encoding="utf-8") as f:
                        f.write(target)
                reverted.append(self._rel(ap))
            except Exception:
                pass
        # Keep the observed entries: they are still changed on disk, so the
        # diff has to keep showing them. Clearing them would make the pane
        # claim the revert put everything back.
        self._observed |= re_observed
        self._original = {
            ap: orig for ap, orig in self._original.items() if ap in self._observed
        }
        self._observed &= set(self._original)
        self._revert_base = {}
        self._diff_cache = {}
        return sorted(reverted)

    # ── persistence (for --resume) ───────────────────────────────────────────
    def to_dict(self) -> dict[str, str | None]:
        """Baselines to persist, revertable-first within a byte budget.

        Every tool phase can promote a whole changed tree into ``_original``
        (a single ``npm run build`` is thousands of files), and the state file
        is rewritten on each save. Unbounded, that turns ``--resume`` into a
        multi-hundred-megabyte read. Revertable baselines are kept first —
        losing one silently disarms ``/revert`` — and scan-only entries, which
        only feed the diff, are what a tight budget drops.
        """
        budget = _PERSIST_MAX_TOTAL_BYTES
        kept: dict[str, str | None] = {}
        ordered = sorted(
            self._original.items(),
            key=lambda item: (item[0] in self._observed, len(item[1] or "")),
        )
        for path, text in ordered:
            cost = len(text or "")
            if cost > budget:
                continue
            budget -= cost
            kept[path] = text
        return kept

    def observed_paths(self) -> list[str]:
        """The non-revertable subset, persisted alongside :meth:`to_dict`."""
        return sorted(self._observed)

    def revert_bases(self) -> dict[str, str | None]:
        """Revert targets that differ from the diff baseline, for persistence.

        Dropping these on resume would silently widen ``/revert`` back to the
        scan baseline, which is the boundary ``_observed`` exists to hold.
        """
        return dict(self._revert_base)

    @classmethod
    def from_dict(
        cls, cwd: str, data: dict[str, str | None],
        observed: list[str] | None = None,
        revert_base: dict[str, str | None] | None = None,
    ) -> WorkspaceJournal:
        j = cls(cwd=cwd)
        j._original = dict(data or {})
        # A state file written before scan-discovered changes existed has no
        # list; everything in it came from a tool that named its path, so
        # every entry stays revertable.
        j._observed = {p for p in (observed or []) if p in j._original}
        j._revert_base = {
            p: text for p, text in (revert_base or {}).items()
            if p in j._original and p not in j._observed
        }
        return j
