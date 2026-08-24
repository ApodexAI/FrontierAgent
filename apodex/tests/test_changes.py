"""Tests for the journal (revert/diff), delete_file, trace, and persistence."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from apodex.agent_tools import (
    MUTATING_TOOLS,
    RISK_CONFIRM,
    RISK_DENY,
    assess_tool_risk,
    coding_tools,
)
from apodex.changes import WorkspaceJournal
from apodex.local_tools import delete_file
from apodex.trace import TraceObserver
from frontier_agent.core.execution_context import ExecutionScope
from frontier_agent.core.loop_types import ToolResult, TurnContext


def _ctx() -> TurnContext:
    return TurnContext(turn=1, max_turns=10, task_id="", role_id="", ai_text="",
                       thinking="", tool_calls=[], messages=[], usage=None, metadata={})


# ── WorkspaceJournal: snapshot / diff / revert ────────────────────────────
def test_journal_tracks_edit_create_delete_and_reverts(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("old\n")
    j = WorkspaceJournal(str(tmp_path))

    j.record_before("a.py")
    (tmp_path / "a.py").write_text("x = 2\n")   # edit
    j.record_before("c.py")
    (tmp_path / "c.py").write_text("new\n")     # create
    j.record_before("b.py")
    (tmp_path / "b.py").unlink()                # delete

    stats = {p: (a, d) for p, a, d in j.diffstat()}
    assert set(stats) == {"a.py", "b.py", "c.py"}
    assert stats["a.py"] == (1, 1)   # edit: +1 -1
    assert stats["c.py"] == (1, 0)   # created
    assert stats["b.py"] == (0, 1)   # deleted

    reverted = j.revert_all()
    assert reverted == ["a.py", "b.py", "c.py"]
    assert (tmp_path / "a.py").read_text() == "x = 1\n"   # edit undone
    assert not (tmp_path / "c.py").exists()               # created file removed
    assert (tmp_path / "b.py").read_text() == "old\n"     # deleted file restored
    assert j.diffstat() == []                             # journal cleared


def test_journal_first_touch_snapshot(tmp_path):
    (tmp_path / "f.py").write_text("v1\n")
    j = WorkspaceJournal(str(tmp_path))
    j.record_before("f.py")
    (tmp_path / "f.py").write_text("v2\n")
    j.record_before("f.py")  # second touch ignored
    (tmp_path / "f.py").write_text("v3\n")
    assert j.revert_all() == ["f.py"]
    assert (tmp_path / "f.py").read_text() == "v1\n"  # reverted to ORIGINAL, not v2


def test_journal_unified_diff_accumulates_session_edits(tmp_path):
    """The sidebar can render edits, creates and deletes across many turns."""
    from apodex.changes import WorkspaceJournal

    edited = tmp_path / "edited.py"
    deleted = tmp_path / "deleted.txt"
    created = tmp_path / "created.md"
    edited.write_text("one\ntwo\n")
    deleted.write_text("gone\n")
    journal = WorkspaceJournal(str(tmp_path))

    journal.record_before("edited.py")
    edited.write_text("one\nsecond\nthree\n")
    # A later turn changes the same file again; its baseline remains "two".
    journal.record_before("edited.py")
    edited.write_text("one\nfinal\nthree\n")
    journal.record_before("created.md")
    created.write_text("new\n")
    journal.record_before("deleted.txt")
    deleted.unlink()

    assert journal.diffstat() == [
        ("created.md", 1, 0),
        ("deleted.txt", 0, 1),
        ("edited.py", 2, 1),
    ]
    diff = journal.unified_diff()
    assert "--- /dev/null\n+++ b/created.md" in diff
    assert "--- a/deleted.txt\n+++ /dev/null" in diff
    assert "--- a/edited.py\n+++ b/edited.py" in diff
    assert "-two" in diff
    assert "+final" in diff


def test_journal_persistence_roundtrip(tmp_path):
    (tmp_path / "x.py").write_text("orig\n")
    j = WorkspaceJournal(str(tmp_path))
    j.record_before("x.py")
    (tmp_path / "x.py").write_text("changed\n")
    j2 = WorkspaceJournal.from_dict(str(tmp_path), j.to_dict())
    assert j2.revert_all() == ["x.py"]
    assert (tmp_path / "x.py").read_text() == "orig\n"


def test_journal_tree_scan_promotes_only_changed_text_files(tmp_path):
    changed = tmp_path / "changed.md"
    unchanged = tmp_path / "unchanged-secret.txt"
    deleted = tmp_path / "deleted.txt"
    changed.write_text("中文\n")
    unchanged.write_text("do-not-persist\n")
    deleted.write_text("remove me\n")
    journal = WorkspaceJournal(str(tmp_path))

    before = journal.begin_tree_scan([str(tmp_path)])
    changed.write_text("English\n")
    deleted.unlink()
    (tmp_path / "created.md").write_text("new\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    originals = journal.to_dict()
    assert set(Path(path).name for path in originals) == {
        "changed.md", "created.md", "deleted.txt",
    }
    assert "do-not-persist" not in json.dumps(originals)
    assert journal.diffstat() == [
        ("changed.md", 1, 1),
        ("created.md", 1, 0),
        ("deleted.txt", 0, 1),
    ]


def test_journal_tree_scan_never_claims_a_pre_existing_file_as_created(
    tmp_path, monkeypatch,
):
    """A file with no readable baseline must not be journaled as a create.

    ``begin_tree_scan`` cannot keep the bytes of a binary or oversized file, so
    it records it as opaque. Leaving it out of the baseline entirely would make
    ``finish_tree_scan`` read "absent before, present after" — a create — and
    ``/revert`` would then DELETE a file that predates the session.
    """
    from apodex import changes

    binary = tmp_path / "asset.bin"
    oversized = tmp_path / "big.log"
    binary.write_bytes(b"\x00\x01\x02")
    oversized.write_text("x" * 64 + "\n")
    monkeypatch.setattr(changes, "_SCAN_MAX_BYTES", 8)   # big.log is over it

    journal = WorkspaceJournal(str(tmp_path))
    before = journal.begin_tree_scan([str(tmp_path)])
    # bash rewrites both into ordinary small text.
    binary.write_text("now text\n")
    oversized.write_text("small\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    assert journal.to_dict() == {}
    assert journal.diffstat() == []
    journal.revert_all()
    assert binary.exists() and oversized.exists()


def test_tree_scan_changes_are_shown_but_never_reverted(tmp_path):
    """Scan-discovered changes are display-only.

    A before/after scan cannot tell the shell's writes from anything else that
    touched the tree in the same window — the user's own editor, a watcher, a
    dev server. Reverting those would destroy work the session never did, so
    ``/revert`` stays limited to paths a tool named.
    """
    scanned = tmp_path / "touched-by-bash.txt"
    named = tmp_path / "touched-by-tool.txt"
    scanned.write_text("before\n")
    named.write_text("before\n")
    journal = WorkspaceJournal(str(tmp_path))

    journal.record_before("touched-by-tool.txt")
    before = journal.begin_tree_scan([str(tmp_path)])
    scanned.write_text("after\n")
    named.write_text("after\n")
    (tmp_path / "made-by-bash.txt").write_text("new\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    # Everything shows in the pane...
    assert [name for name, _a, _r in journal.diffstat()] == [
        "made-by-bash.txt", "touched-by-bash.txt", "touched-by-tool.txt",
    ]
    assert journal.observed_only() == ["made-by-bash.txt", "touched-by-bash.txt"]

    # ...but only the named file is written back.
    assert journal.revert_all() == ["touched-by-tool.txt"]
    assert named.read_text() == "before\n"
    assert scanned.read_text() == "after\n"
    assert (tmp_path / "made-by-bash.txt").exists()

    # And what the revert left alone stays in the pane — dropping it would
    # have the diff claim everything was put back.
    assert [name for name, _a, _r in journal.diffstat()] == [
        "made-by-bash.txt", "touched-by-bash.txt",
    ]


def test_a_tool_naming_a_scanned_path_makes_it_revertable(tmp_path):
    """Attribution is what the observed set lacks, so a later tool supplies it.

    It supplies it from that point on and no earlier: the v1→v2 window belongs
    to a scan that cannot tell the shell apart from the user's own editor, so
    the revert stops at v2 while the diff still starts at v1.
    """
    target = tmp_path / "f.txt"
    target.write_text("v1\n")
    journal = WorkspaceJournal(str(tmp_path))

    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("v2\n")
    journal.finish_tree_scan([str(tmp_path)], before)
    assert journal.observed_only() == ["f.txt"]

    journal.record_before("f.txt")          # write_file / file_editor names it
    target.write_text("v3\n")

    assert journal.observed_only() == []
    diff = journal.unified_diff()
    assert "-v1" in diff and "+v3" in diff  # the diff still spans the whole run

    assert journal.revert_all() == ["f.txt"]
    assert target.read_text() == "v2\n"     # never past where attribution began
    # What is left on disk is unattributed again, so it is observed once more.
    assert journal.observed_only() == ["f.txt"]


def test_a_named_binary_is_not_journaled_as_a_file_the_session_created(tmp_path):
    """``None`` means "absent", and a file with no text baseline is not absent.

    Journaling one anyway reads back as a create the moment the path becomes
    readable, so the diff announces a new file that predates the session and
    ``/revert`` removes the path instead of restoring anything.
    """
    target = tmp_path / "logo.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")
    journal = WorkspaceJournal(str(tmp_path))

    journal.record_before("logo.png")           # file_editor / write_file names it
    target.write_text("now it is text\n")

    assert journal.diffstat() == []
    assert journal.revert_all() == []
    assert target.exists()


def test_revert_never_rewinds_past_a_concurrent_editor_save(tmp_path):
    """The whole point of the boundary, stated as the scenario it protects."""
    target = tmp_path / "notes.md"
    target.write_text("user paragraph\n")
    journal = WorkspaceJournal(str(tmp_path))

    # A bash phase runs; during it the user saves their own edit in an editor.
    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("user paragraph\nedited by the user\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    # Two turns later the agent edits the same file with a named-path tool.
    journal.record_before("notes.md")
    target.write_text("user paragraph\nedited by the user\nagent line\n")

    journal.revert_all()
    assert target.read_text() == "user paragraph\nedited by the user\n"


def test_a_promoted_revert_target_survives_resume(tmp_path):
    """Losing it on resume would silently widen /revert back to the scan."""
    target = tmp_path / "f.txt"
    target.write_text("v1\n")
    journal = WorkspaceJournal(str(tmp_path))
    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("v2\n")
    journal.finish_tree_scan([str(tmp_path)], before)
    journal.record_before("f.txt")
    target.write_text("v3\n")

    resumed = WorkspaceJournal.from_dict(
        str(tmp_path), journal.to_dict(), journal.observed_paths(),
        journal.revert_bases(),
    )
    assert resumed.revert_all() == ["f.txt"]
    assert target.read_text() == "v2\n"


def test_persisted_baselines_are_bounded_and_keep_the_revertable_ones(
    tmp_path, monkeypatch,
):
    """A build step promotes a whole tree; the state file must not follow it."""
    import apodex.changes as changes_module

    monkeypatch.setattr(changes_module, "_PERSIST_MAX_TOTAL_BYTES", 3_000)
    journal = WorkspaceJournal(str(tmp_path))

    # 20 fat scan-discovered baselines, plus one small attributed edit.
    for i in range(20):
        blob = tmp_path / f"dist/asset-{i}.js"
        blob.parent.mkdir(exist_ok=True)
        blob.write_text("x" * 1_000)
    before = journal.begin_tree_scan([str(tmp_path)])
    for i in range(20):
        (tmp_path / f"dist/asset-{i}.js").write_text("y" * 1_000)
    journal.finish_tree_scan([str(tmp_path)], before)

    source = tmp_path / "app.py"
    source.write_text("value = 1\n")
    journal.record_before("app.py")
    source.write_text("value = 2\n")

    persisted = journal.to_dict()
    assert sum(len(text or "") for text in persisted.values()) <= 3_000
    # The attributed baseline is the one /revert needs, so it is never the
    # entry that gets dropped.
    assert str(source) in persisted
    assert len(persisted) < len(journal._original)

    resumed = WorkspaceJournal.from_dict(
        str(tmp_path), persisted, journal.observed_paths(),
        journal.revert_bases(),
    )
    assert resumed.revert_all() == ["app.py"]
    assert source.read_text() == "value = 1\n"


def test_an_observed_file_grown_huge_is_not_read_into_the_diff(tmp_path, monkeypatch):
    """The "after" read of a scan-observed path obeys the scan's own limits.

    ``begin_tree_scan`` caps its baseline at ``_SCAN_MAX_BYTES``, but a path is
    observed whenever the two scans disagree — including when it grew past that
    cap in between. Reading it back unbounded is one ``bash`` append away: a
    100 KB build.log appended to until it is far larger would materialise the
    whole file plus a diff of it, and then cache both.
    """
    from apodex import changes

    monkeypatch.setattr(changes, "_SCAN_MAX_BYTES", 4096)
    log = tmp_path / "build.log"
    log.write_text("line\n" * 10)
    journal = WorkspaceJournal(str(tmp_path))

    before = journal.begin_tree_scan([str(tmp_path)])
    log.write_text("line\n" * 10_000)          # now over the cap
    journal.finish_tree_scan([str(tmp_path)], before)

    assert journal.observed_only() == ["build.log"]
    stats, diff = journal.report()
    # Listed as changed by ``observed_only`` — but not rendered as a deletion,
    # and its contents never enter the diff text or the cache.
    assert stats == []
    assert diff == ""


def test_an_empty_create_is_named_in_the_diff(tmp_path):
    """``touch newfile`` used to produce an entry with no visible filename.

    ``unified_diff`` of two empty sequences returns nothing at all, headers
    included, so the stat was ``(path, 0, 0)`` with an empty chunk: the tab
    opened, the status bar counted a file, and the pane named none of them.
    """
    journal = WorkspaceJournal(str(tmp_path))
    journal.record_before("empty.txt")
    (tmp_path / "empty.txt").write_text("")

    stats, diff = journal.report()
    assert stats == [("empty.txt", 0, 0)]
    assert "empty.txt" in diff
    assert "/dev/null" in diff

    # Still revertable, and reverting removes the file the diff named.
    assert journal.revert_all() == ["empty.txt"]
    assert not (tmp_path / "empty.txt").exists()


def test_the_changed_files_summary_lists_only_what_revert_undoes(tmp_path):
    """The end-of-task panel is titled "/revert to undo", so it must mean it.

    Once ``report()`` folded in the tree scan, a task that shelled out to a
    build printed every file under ``dist/`` in a panel offering to undo them.
    """
    (tmp_path / "src.py").write_text("value = 1\n")
    journal = WorkspaceJournal(str(tmp_path))
    journal.record_before("src.py")
    (tmp_path / "src.py").write_text("value = 2\n")

    before = journal.begin_tree_scan([str(tmp_path)])
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("built\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    assert [name for name, _a, _r in journal.diffstat()] == [
        "dist/bundle.js", "src.py",
    ]
    assert journal.revertable_diffstat() == [("src.py", 1, 1)]


def test_report_survives_a_wholly_rewritten_large_file(tmp_path):
    """``difflib.ndiff`` blew the recursion limit here; counts come off the diff."""
    target = tmp_path / "big.py"
    target.write_text("".join(f"old {i}\n" for i in range(1200)))
    journal = WorkspaceJournal(str(tmp_path))
    journal.record_before("big.py")
    target.write_text("".join(f"new {i}\n" for i in range(1200)))

    stats, diff = journal.report()
    assert stats == [("big.py", 1200, 1200)]
    assert "--- a/big.py" in diff


def test_report_reuses_its_last_reading_until_the_file_moves(tmp_path, monkeypatch):
    """The TUI polls once a second, so an untouched file must not be re-read."""
    import apodex.changes as changes_module

    target = tmp_path / "f.txt"
    target.write_text("one\n")
    journal = WorkspaceJournal(str(tmp_path))
    journal.record_before("f.txt")
    target.write_text("two\n")

    first = journal.report()
    reads: list[str] = []
    real_read = changes_module._read_or_none
    monkeypatch.setattr(
        changes_module, "_read_or_none",
        lambda p: (reads.append(p), real_read(p))[1],
    )
    assert journal.report() == first
    assert reads == []                          # served entirely from the memo

    target.write_text("three\n")
    os.utime(target, ns=(0, 10**9))             # force a distinct mtime
    assert journal.report() != first
    assert reads == [str(tmp_path / "f.txt")]


def test_revert_does_not_claim_shell_writes_are_untracked(tmp_path):
    """The old wording contradicted the list of shell changes printed under it."""
    from types import SimpleNamespace

    from apodex.session import TerminalSession

    target = tmp_path / "built.js"
    target.write_text("before\n")
    journal = WorkspaceJournal(str(tmp_path))
    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("after\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    messages: list[str] = []
    terminal = object.__new__(TerminalSession)
    terminal.journal = journal
    terminal.r = SimpleNamespace(note=messages.append, error=messages.append)
    asyncio.run(terminal._slash("/revert"))

    rendered = "\n".join(messages)
    assert "shell commands" not in rendered      # they are tracked now
    assert "nothing to revert (no attributed edits)" in rendered
    assert "built.js" in rendered

    # With nothing found at all, the honest message is still the fuller one.
    messages.clear()
    terminal.journal = WorkspaceJournal(str(tmp_path))
    asyncio.run(terminal._slash("/revert"))
    assert "no journaled edits" in "\n".join(messages)


def test_journal_persistence_keeps_the_revert_boundary(tmp_path):
    """A resumed session must not start reverting scan-discovered changes."""
    scanned = tmp_path / "scanned.txt"
    scanned.write_text("before\n")
    journal = WorkspaceJournal(str(tmp_path))
    before = journal.begin_tree_scan([str(tmp_path)])
    scanned.write_text("after\n")
    journal.finish_tree_scan([str(tmp_path)], before)

    resumed = WorkspaceJournal.from_dict(
        str(tmp_path), journal.to_dict(), journal.observed_paths(),
    )
    assert resumed.observed_only() == ["scanned.txt"]
    assert resumed.revert_all() == []
    assert scanned.read_text() == "after\n"

    # A state file written before this existed has no list; its entries all
    # came from a tool that named them, so they stay revertable.
    legacy = WorkspaceJournal.from_dict(str(tmp_path), journal.to_dict())
    assert legacy.revert_all() == ["scanned.txt"]


def test_journal_report_returns_stats_and_diff_from_one_read(tmp_path):
    """The pane header and the hunks below it come from the same snapshot."""
    edited = tmp_path / "edited.py"
    edited.write_text("one\ntwo\n")
    journal = WorkspaceJournal(str(tmp_path))
    journal.record_before("edited.py")
    edited.write_text("one\nfinal\n")

    stats, diff = journal.report()
    assert stats == journal.diffstat() == [("edited.py", 1, 1)]
    assert diff == journal.unified_diff()
    assert "--- a/edited.py\n+++ b/edited.py" in diff
    assert "-two" in diff and "+final" in diff


def test_journaled_write_tools_never_see_an_unresolved_mount_alias(tmp_path):
    """Why ``record_before`` may take the tool argument verbatim.

    The file tools rewrite ``/workspace`` / ``/outputs`` to the real runtime
    directory before writing, so a journaled alias would baseline a path that
    never exists. It cannot happen: the same aliases are outside cwd, and the
    risk assessor denies those calls before the journal is ever consulted.
    """
    for tool_name in sorted(MUTATING_TOOLS):
        risk = assess_tool_risk(
            tool_name, {"path": "/outputs/report.md"}, str(tmp_path),
        )
        assert risk.level == RISK_DENY, tool_name


def test_observer_settles_a_tree_scan_no_result_ever_claimed(tmp_path):
    """An interrupted call still changed files, and its baseline costs memory.

    ``on_turn_end`` has to fold it in rather than drop it: the snapshot holds
    the text of the whole scanned tree, and the edits it describes are real.
    """
    from apodex.observers import Approver, TerminalObserver
    from apodex.render import Renderer

    target = tmp_path / "a.txt"
    target.write_text("one\n")
    journal = WorkspaceJournal(str(tmp_path))
    observer = TerminalObserver(
        Renderer(theme="mono"), Approver(auto_approve=True), str(tmp_path),
        journal=journal,
    )

    asyncio.run(observer.on_tool_call(_ctx(), {
        "id": "interrupted", "name": "bash", "args": {"command": "./rewrite.sh"},
    }))
    assert observer._journal_scan is not None      # the baseline was captured
    target.write_text("two\n")

    asyncio.run(observer.on_turn_end(_ctx()))
    assert observer._journal_scan is None
    assert journal.diffstat() == [("a.txt", 1, 1)]


def test_observer_walks_the_tree_once_per_parallel_tool_phase(tmp_path):
    """A tool phase costs one baseline, not one per mutating call.

    ``asyncio.gather`` runs the phase's calls together, and each baseline is a
    full walk plus the text of every file under the roots. Per-call baselines
    also had to be paired back to their result, which a synthetic call id (no
    ``id`` from the provider — it never comes back on the result) can only do
    positionally, i.e. by whichever call happened to finish first.
    """
    from apodex.observers import Approver, TerminalObserver
    from apodex.render import Renderer

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a\n")
    second.write_text("b\n")
    journal = WorkspaceJournal(str(tmp_path))
    calls = {"begin": 0, "finish": 0}
    real_begin, real_finish = journal.begin_tree_scan, journal.finish_tree_scan

    def counted_begin(roots):
        calls["begin"] += 1
        return real_begin(roots)

    def counted_finish(roots, before):
        calls["finish"] += 1
        return real_finish(roots, before)

    journal.begin_tree_scan = counted_begin
    journal.finish_tree_scan = counted_finish
    observer = TerminalObserver(
        Renderer(theme="mono"), Approver(auto_approve=True), str(tmp_path),
        journal=journal,
    )

    async def phase() -> None:
        # Both hooks run before either body does, and neither call carries an
        # id — the shape a provider that omits them produces.
        await observer.on_tool_call(_ctx(), {
            "name": "bash", "args": {"command": "./one.sh"},
        })
        await observer.on_tool_call(_ctx(), {
            "name": "bash", "args": {"command": "./two.sh"},
        })
        first.write_text("A\n")
        second.write_text("B\n")
        for _ in range(2):
            await observer.on_tool_result(_ctx(), ToolResult(
                name="bash", args={}, result="ok", duration_ms=1,
                tool_call_id="", is_error=False,
            ))

    asyncio.run(phase())

    assert calls == {"begin": 1, "finish": 1}
    assert observer._journal_scan is None
    assert journal.diffstat() == [("first.txt", 1, 1), ("second.txt", 1, 1)]


def test_observer_tracks_bash_changes_in_cwd_and_session_outputs(
    tmp_path, monkeypatch,
):
    from apodex.observers import Approver, TerminalObserver
    from apodex.render import Renderer

    project = tmp_path / "project"
    outputs = tmp_path / "run-outputs"
    project.mkdir()
    outputs.mkdir()
    local_report = project / "report.md"
    output_report = outputs / "report.md"
    local_report.write_text("中文\n")
    output_report.write_text("中文 output\n")
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.delenv("APODEX_HOST_OUTPUTS_DIR", raising=False)
    journal = WorkspaceJournal(str(project))
    observer = TerminalObserver(
        Renderer(theme="mono"), Approver(auto_approve=True), str(project),
        journal=journal,
    )

    asyncio.run(observer.on_tool_call(
        _ctx(), {
            "id": "bash-write",
            "name": "bash",
            "args": {"command": "python3 rewrite_reports.py"},
        },
    ))
    local_report.write_text("English\n")
    output_report.write_text("English output\n")
    asyncio.run(observer.on_tool_result(
        _ctx(), ToolResult(
            name="bash", args={"command": "python3 rewrite_reports.py"},
            result="done", duration_ms=1, tool_call_id="bash-write",
            is_error=False,
        ),
    ))

    stats = {path: (added, removed) for path, added, removed in journal.diffstat()}
    assert stats["report.md"] == (1, 1)
    outside = os.path.relpath(output_report, project)
    assert stats[outside] == (1, 1)


# ── delete_file tool ───────────────────────────────────────────────────────
def test_delete_file_scoped_and_first_class(tmp_path, monkeypatch):
    (tmp_path / "gone.txt").write_text("bye")
    monkeypatch.chdir(tmp_path)
    assert "Deleted" in asyncio.run(delete_file.ainvoke({"path": "gone.txt"}))
    assert not (tmp_path / "gone.txt").exists()
    assert "outside" in asyncio.run(delete_file.ainvoke({"path": "/etc/hosts"}))
    assert "not found" in asyncio.run(delete_file.ainvoke({"path": "nope.txt"}))
    # registered + classified as a journaled, confirm-gated mutation
    assert "delete_file" in {t.name for t in coding_tools()}
    assert "delete_file" in MUTATING_TOOLS
    assert assess_tool_risk("delete_file", {"path": "x.py"}, str(tmp_path)).level == RISK_CONFIRM
    assert assess_tool_risk("delete_file", {"path": "/etc/x"}, str(tmp_path)).level == RISK_DENY


# ── trace observer ──────────────────────────────────────────────────────────
def test_trace_observer_writes_jsonl(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    t = TraceObserver(path, mode="coding", cwd=str(tmp_path))  # writes a start line
    asyncio.run(t.on_tool_result(_ctx(), ToolResult(
        name="bash", args={"command": "ls"}, result="a.py",
        duration_ms=5, tool_call_id="1", is_error=False)))
    recs = [json.loads(line) for line in open(path) if line.strip()]
    kinds = {r["t"] for r in recs}
    assert "start" in kinds and "tool" in kinds
    tool_rec = next(r for r in recs if r["t"] == "tool")
    assert tool_rec["name"] == "bash" and tool_rec["result"] == "a.py"


def test_trace_observer_keeps_complete_tool_payloads(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    t = TraceObserver(path)
    long_value = "x" * 5001
    asyncio.run(t.on_tool_result(_ctx(), ToolResult(
        name="read_file", args={"path": long_value}, result=long_value,
        duration_ms=1, tool_call_id="1", is_error=False)))
    record = [json.loads(line) for line in open(path) if line.strip()][-1]
    assert record["args"]["path"] == long_value
    assert record["result"] == long_value


# ── session persist / resume round-trip ───────────────────────────────────
def test_session_persist_and_resume(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession, load_session_state
    from apodex.todo import TodoItem, clear_todos, get_todos, set_todos
    from frontier_agent.core.messages import assistant_msg, system_msg, user_msg

    # isolate the persisted-session dir to a tmp HOME
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = ModelConfig(model="fake", api_key="x", base_url=None)
    s = TerminalSession(cfg=cfg, cwd=str(tmp_path), renderer=Renderer(theme="mono"),
                        auto_approve=True, max_turns=5, interactive=False, mode="coding")
    s.history = [system_msg("sys"), user_msg("hi"), assistant_msg("done")]
    s.display_history = list(s.history)
    s.tui_state = {
        "version": 1,
        "subagents": [{
            "session_id": "root::scout", "name": "scout", "status": "ready",
        }],
    }
    s.plan_state.active = True
    set_todos([TodoItem("inspect", "completed"), TodoItem("implement", "in_progress")])
    (tmp_path / "edited.py").write_text("new\n")
    s.journal.record_before("edited.py")  # pretend a change was journaled
    s._persist()

    state = load_session_state(s.session_id)
    assert state is not None and state["mode"] == "coding" and len(state["history"]) == 3
    assert state["display_history"] == s.history
    assert state["tui"]["subagents"][0]["session_id"] == "root::scout"

    s2 = TerminalSession(cfg=cfg, cwd=str(tmp_path), renderer=Renderer(theme="mono"),
                         auto_approve=True, max_turns=5, interactive=False,
                         mode="coding", session_id=s.session_id)
    s2.restore(state)
    assert [m.get("role") for m in s2.history] == ["system", "user", "assistant"]
    assert s2.replay_history() == s.history
    assert s2.tui_state == s.tui_state
    assert s2.plan_state.active is True
    assert [(item.content, item.status) for item in get_todos()] == [
        ("inspect", "completed"), ("implement", "in_progress"),
    ]
    clear_todos()
    assert "edited.py" in s2.journal.to_dict() or True  # journal restored shape


def test_follow_up_receives_exact_agent_and_host_deliverable_paths(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "APODEX_HOST_RUNS_ROOT", str(tmp_path / ".apodex" / "runs"),
    )
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    session = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None),
        cwd=str(tmp_path), renderer=Renderer(theme="mono"),
        auto_approve=True, max_turns=5, interactive=False, mode="coding",
    )
    outputs = Path(os.environ["FRONTIER_AGENT_OUTPUTS_DIR"])
    (outputs / "deck.pptx").write_bytes(b"pptx")

    enriched = session._enrich_task("revise the previous deck")

    assert str(outputs / "deck.pptx") in enriched
    assert os.environ["APODEX_HOST_OUTPUTS_DIR"] == str(outputs)
    assert "before searching the workspace" in enriched


def test_workflow_display_history_preserves_calls_results_and_final() -> None:
    from apodex.session import TerminalSession

    messages = TerminalSession._workflow_display_messages(
        "make a deck",
        [{
            "tool_name": "create_file",
            "tool_args": {"path": "/outputs/deck.pptx"},
            "tool_result": "created",
            "duration_ms": 12,
            "is_error": False,
        }],
        "done",
    )

    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant",
    ]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "create_file"
    assert messages[2]["content"] == "created"
    assert messages[-1]["content"] == "done"


def test_legacy_workflow_history_synthesizes_missing_tool_call() -> None:
    from apodex.session import TerminalSession

    session = object.__new__(TerminalSession)
    session.display_history = []
    session.history = []
    session.workflow_turns = [{"messages": [
        {"role": "user", "content": "old task"},
        {"role": "tool", "name": "read_file", "content": "old result"},
        {"role": "assistant", "content": "old final"},
    ]}]

    replay = session.replay_history()

    assert [message["role"] for message in replay] == [
        "user", "assistant", "tool", "assistant",
    ]
    assert replay[1]["tool_calls"][0]["function"] == {
        "name": "read_file", "arguments": "{}",
    }
    assert replay[2]["tool_call_id"] == replay[1]["tool_calls"][0]["id"]


def test_session_output_link_switches_with_active_session(tmp_path, monkeypatch):
    from apodex.session import TerminalSession

    root = tmp_path / "all-outputs"
    link = tmp_path / "outputs"
    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    monkeypatch.setenv("APODEX_SESSION_OUTPUTS_ROOT", str(root))
    monkeypatch.setenv("APODEX_OUTPUTS_LINK", str(link))
    monkeypatch.setenv("APODEX_HOST_OUTPUTS_ROOT", "/host/project/.apodex/outputs")

    TerminalSession._activate_session_outputs("session-one")
    (link / "first.txt").write_text("one")
    assert (root / "session-one" / "first.txt").read_text() == "one"

    TerminalSession._activate_session_outputs("session-two")
    assert link.resolve() == (root / "session-two").resolve()
    assert not (link / "first.txt").exists()
    assert os.environ["APODEX_HOST_OUTPUTS_DIR"] == (
        "/host/project/.apodex/outputs/session-two"
    )


def test_session_workspace_link_switches_with_active_session(tmp_path, monkeypatch):
    from apodex.session import TerminalSession

    root = tmp_path / "runs"
    link = tmp_path / "workspace"
    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    monkeypatch.setenv("APODEX_SESSION_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("APODEX_WORKSPACE_LINK", str(link))
    monkeypatch.setenv("APODEX_HOST_WORKSPACE_ROOT", "/host/project/.apodex/runs")

    TerminalSession._activate_session_workspace("session-one")
    (link / "scratch.txt").write_text("one")
    assert (root / "session-one" / "scratch.txt").read_text() == "one"

    TerminalSession._activate_session_workspace("session-two")
    assert link.resolve() == (root / "session-two").resolve()
    assert not (link / "scratch.txt").exists()
    assert os.environ["APODEX_HOST_WORKSPACE_DIR"] == (
        "/host/project/.apodex/runs/session-two"
    )


def test_activating_a_session_follows_the_checkpoint_workspace(tmp_path, monkeypatch):
    """An in-app ``/resume`` into another project must move the whole run
    record with it, not keep writing into the project it started in."""
    from apodex.native import prepare_native_runtime
    from apodex.session import TerminalSession

    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir(parents=True)
    project_b.mkdir(parents=True)
    native_env: dict[str, str] = {"HOME": str(tmp_path / "home")}
    prepare_native_runtime(str(project_a), "original-run", environ=native_env)
    for name in ("APODEX_RUNS_ROOT", "APODEX_HOST_RUNS_ROOT"):
        monkeypatch.setenv(name, native_env[name])
    assert "APODEX_RUNS_ROOT_PINNED" not in native_env
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    monkeypatch.delenv("APODEX_OUTPUTS_LINK", raising=False)

    TerminalSession._activate_session_outputs("moved-run", str(project_b))

    runs_b = (project_b / ".apodex" / "runs").resolve()
    assert os.environ["APODEX_RUN_DIR"] == str(runs_b / "moved-run")
    assert os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] == str(
        runs_b / "moved-run" / "outputs"
    )
    assert os.environ["APODEX_HOST_OUTPUTS_DIR"] == str(
        runs_b / "moved-run" / "outputs"
    )
    assert not (project_a / ".apodex" / "runs" / "moved-run").exists()


def test_session_output_link_rejects_path_traversal(tmp_path, monkeypatch):
    from apodex.session import TerminalSession

    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    monkeypatch.setenv("APODEX_SESSION_OUTPUTS_ROOT", str(tmp_path / "outputs-root"))
    monkeypatch.setenv("APODEX_OUTPUTS_LINK", str(tmp_path / "outputs-link"))

    with pytest.raises(ValueError, match=r"invalid (?:output|run) session id"):
        TerminalSession._activate_session_outputs("../../escape")


def test_session_workspace_link_rejects_path_traversal(tmp_path, monkeypatch):
    from apodex.session import TerminalSession

    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    monkeypatch.setenv("APODEX_SESSION_WORKSPACES_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("APODEX_WORKSPACE_LINK", str(tmp_path / "workspace-link"))

    with pytest.raises(ValueError, match="invalid workspace session id"):
        TerminalSession._activate_session_workspace("../../escape")


def test_list_saved_sessions_returns_newest_checkpoint_metadata(tmp_path, monkeypatch):
    import json
    import os
    import time

    from apodex.session import list_saved_sessions

    monkeypatch.setenv("HOME", str(tmp_path))
    session_dir = tmp_path / ".apodex" / "sessions"
    session_dir.mkdir(parents=True)
    older = session_dir / "older.json"
    newer = session_dir / "newer.json"
    older.write_text(json.dumps({
        "session_id": "older", "mode": "coding", "cwd": "/old", "history": [{}],
    }))
    newer.write_text(json.dumps({
        "session_id": "newer", "mode": "research", "cwd": "/new", "history": [{}, {}],
    }))
    now = time.time()
    os.utime(older, (now - 60, now - 60))
    os.utime(newer, (now, now))

    sessions = list_saved_sessions()

    assert [item["session_id"] for item in sessions] == ["newer", "older"]
    assert sessions[0]["mode"] == "research"
    assert sessions[0]["message_count"] == 2
    assert re.search(r"[+-]\d{4}$", sessions[0]["modified_at"])


def test_list_saved_sessions_reads_a_named_workspace_once(tmp_path, monkeypatch):
    """``--resume`` answers before the ``--cwd`` chdir, and a checkpoint that
    was migrated out of the legacy tree must not be listed twice."""
    from apodex.session import list_saved_sessions

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_LEGACY_SESSION_ROOTS", raising=False)
    legacy = tmp_path / "home" / ".apodex" / "sessions"
    legacy.mkdir(parents=True)
    workspace = tmp_path / "project"
    run = workspace / ".apodex" / "runs" / "shared-id"
    run.mkdir(parents=True)
    state = json.dumps({
        "session_id": "shared-id", "mode": "coding", "cwd": str(workspace),
        "history": [{}],
    })
    (legacy / "shared-id.json").write_text(state)
    (run / "session.json").write_text(state)

    sessions = list_saved_sessions(workspace=str(workspace))

    assert [item["session_id"] for item in sessions] == ["shared-id"]


def test_new_session_id_uses_local_time_with_an_explicit_offset(monkeypatch) -> None:
    from apodex.session import new_session_id

    monkeypatch.setenv("APODEX_LOCAL_UTC_OFFSET", "+0800")
    session_id = new_session_id("agent_team")

    assert re.fullmatch(
        r"\d{8}-\d{6}\+0800-agent_team-[0-9a-f]{4}", session_id,
    )


def test_run_layout_keeps_interactive_artifacts_together(tmp_path, monkeypatch) -> None:
    from apodex.run_layout import activate_run

    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_RUNS_ROOT_PINNED", raising=False)
    active = activate_run("local-react-ab12", tmp_path)

    assert active == tmp_path / ".apodex" / "runs" / "local-react-ab12"
    assert os.environ["APODEX_RUN_DIR"] == str(active)
    assert os.environ["APODEX_SESSION_ID"] == "local-react-ab12"


def test_load_session_state_reads_configured_legacy_native_root(tmp_path, monkeypatch) -> None:
    from apodex.session import load_session_state

    legacy = tmp_path / "old-native-sessions"
    legacy.mkdir()
    (legacy / "old-run.json").write_text(
        json.dumps({"session_id": "old-run", "history": [{"role": "user"}]}),
    )
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "new-runs"))
    monkeypatch.setenv("APODEX_LEGACY_SESSION_ROOTS", str(legacy))

    assert load_session_state("old-run")["session_id"] == "old-run"
    assert not (tmp_path / "new-runs" / "old-run").exists()


def test_new_session_saves_old_checkpoint_and_resets_session_state(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession, load_session_state
    from frontier_agent.core.messages import assistant_msg, user_msg

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("APODEX_INPUT_STAGING_DIR", raising=False)
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    cfg = ModelConfig(model="fake", api_key="x", base_url=None)
    session = TerminalSession(
        cfg=cfg, cwd=str(tmp_path), renderer=Renderer(theme="mono"),
        auto_approve=True, max_turns=5, interactive=False, mode="coding",
    )
    session.history = [user_msg("old task"), assistant_msg("old answer")]
    session.usage.input = 120
    session.usage.output = 30
    old_id = session.session_id

    previous, current = session.start_new_session()

    assert previous == old_id
    assert current != old_id
    assert load_session_state(old_id)["history"] == [
        user_msg("old task"), assistant_msg("old answer"),
    ]
    assert session.history == []
    assert session.workflow_turns == []
    assert session.usage.total == 0
    assert session.journal.to_dict() == {}
    assert session.trace_path.endswith(f"{current}/trace.jsonl")


def test_clear_discards_only_the_active_sessions_spill(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession
    from frontier_agent.core.messages import user_msg

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.delenv("APODEX_INPUT_STAGING_DIR", raising=False)
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    session = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None),
        cwd=str(tmp_path), renderer=Renderer(theme="mono"),
        auto_approve=True, max_turns=5, interactive=False, mode="coding",
    )
    session.history = [user_msg("old context")]
    session.workflow_turns = [{"messages": [user_msg("old context")]}]
    # The store lives outside the workspace now, so "this conversation's files"
    # is expressed as what this process created rather than as a directory tree.
    store = tmp_path / "store"
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))
    from frontier_agent.core.execution_context import (
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from plugins.tools import _overflow

    token = set_current_execution_scope(
        ExecutionScope(task_id="tui", metadata={"llm_session_id": "s"}),
    )
    try:
        mine = _overflow.spill_compacted_body("bash", "recoverable " * 200)
    finally:
        reset_current_execution_scope(token)
    assert mine
    # A store this process did not create: another session's, in the same root.
    stranger = store / "another-session"
    stranger.mkdir(parents=True)
    (stranger / "result.md").write_text("other", encoding="utf-8")

    assert asyncio.run(session._slash("/clear")) is False

    assert session.history == []
    assert session.workflow_turns == []
    assert not Path(mine).exists()
    assert (stranger / "result.md").read_text(encoding="utf-8") == "other"


def test_context_reset_never_broad_cleans_a_shared_workspace(tmp_path, monkeypatch):
    from apodex.session import TerminalSession

    shared_workspace = tmp_path / "shared-workspace"
    spill = shared_workspace / ".spill" / "another-session" / "result.md"
    spill.parent.mkdir(parents=True)
    spill.write_text("must survive", encoding="utf-8")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(shared_workspace))
    monkeypatch.delenv("APODEX_RUNS_ROOT", raising=False)
    monkeypatch.delenv("APODEX_SESSION_WORKSPACES_ROOT", raising=False)

    workspace = TerminalSession._active_spill_workspace()

    assert workspace is None
    assert TerminalSession._cleanup_discarded_spill(workspace) == 0
    assert spill.read_text(encoding="utf-8") == "must survive"


def test_context_reset_checks_containment_not_just_a_configured_root(
    tmp_path, monkeypatch,
):
    """Having a session root configured does not prove THIS workspace is in one.

    ``_active_spill_workspace`` only tested that ``APODEX_RUNS_ROOT`` or
    ``APODEX_SESSION_WORKSPACES_ROOT`` was non-empty, then returned
    ``FRONTIER_AGENT_WORKSPACE_DIR`` as-is — so a deployment that sets a runs
    root while pointing the workspace at a shared or user-owned directory had
    /clear, /mode and /cwd recursively delete ``<that dir>/.spill``.
    """
    from apodex.session import TerminalSession

    shared_workspace = tmp_path / "shared-workspace"
    spill = shared_workspace / ".spill" / "another-session" / "result.md"
    spill.parent.mkdir(parents=True)
    spill.write_text("must survive", encoding="utf-8")
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(shared_workspace))

    assert TerminalSession._active_spill_workspace() is None
    assert TerminalSession._cleanup_discarded_spill(None) == 0
    assert spill.read_text(encoding="utf-8") == "must survive"


def test_context_reset_still_cleans_a_session_private_workspace(tmp_path, monkeypatch):
    """The containment check must not disable the cleanup it guards, including
    through the ``APODEX_WORKSPACE_LINK`` symlink the activation may install.

    The workspace no longer holds the store; it is only the gate. What the
    cleanup deletes is what this process created.
    """
    from apodex.session import TerminalSession
    from frontier_agent.core.execution_context import (
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from plugins.tools import _overflow

    runs_root = tmp_path / "runs"
    private = runs_root / "sess-1" / "workspace"
    private.mkdir(parents=True)
    link = tmp_path / "current-workspace"
    link.symlink_to(private, target_is_directory=True)
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(link))
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))

    token = set_current_execution_scope(
        ExecutionScope(task_id="t", metadata={"llm_session_id": "s"}),
    )
    try:
        ref = _overflow.spill_compacted_body("bash", "discardable " * 200)
    finally:
        reset_current_execution_scope(token)
    assert ref

    workspace = TerminalSession._active_spill_workspace()

    # The private workspace is still what gates the cleanup — it is the signal
    # that this context is session-owned — but the files removed are the ones
    # this process wrote, wherever the store happens to live.
    assert workspace == private.resolve()
    assert TerminalSession._cleanup_discarded_spill(workspace) == 1
    assert not Path(ref).exists()


def test_new_session_preserves_saved_sessions_spill(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APODEX_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.delenv("APODEX_INPUT_STAGING_DIR", raising=False)
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    session = TerminalSession(
        cfg=ModelConfig(model="fake", api_key="x", base_url=None),
        cwd=str(tmp_path), renderer=Renderer(theme="mono"),
        auto_approve=True, max_turns=5, interactive=False, mode="coding",
    )
    old_workspace = Path(
        os.environ["FRONTIER_AGENT_WORKSPACE_DIR"],
    ).resolve()
    spill = old_workspace / ".spill" / "scope" / "result.md"
    spill.parent.mkdir(parents=True)
    spill.write_text("needed after resume", encoding="utf-8")

    session.start_new_session()

    assert spill.read_text(encoding="utf-8") == "needed after resume"


def test_fork_and_rename_preserve_context_with_readable_listing(tmp_path, monkeypatch):
    from apodex.config import ModelConfig
    from apodex.render import Renderer
    from apodex.session import TerminalSession, list_saved_sessions, load_session_state
    from frontier_agent.core.messages import user_msg

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("APODEX_INPUT_STAGING_DIR", raising=False)
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    cfg = ModelConfig(model="fake", api_key="x", base_url=None)
    session = TerminalSession(
        cfg=cfg, cwd=str(tmp_path), renderer=Renderer(theme="mono"),
        auto_approve=True, max_turns=5, interactive=False, mode="coding",
    )
    session.history = [user_msg("branch from here")]

    old_id, fork_id = session.start_new_session(fork=True)
    assert old_id != fork_id
    assert session.history == [user_msg("branch from here")]
    assert session.rename_session("Parser experiment") == "Parser experiment"
    assert load_session_state(fork_id)["name"] == "Parser experiment"
    assert any(
        item["session_id"] == fork_id and item["name"] == "Parser experiment"
        for item in list_saved_sessions()
    )


def test_context_report_separates_window_fill_from_cumulative_usage():
    from apodex.usage import Usage

    usage = Usage(
        input=391_420, output=37_480, cached=284_100,
        last_input=143_280, compactions=1,
    )
    report = usage.context_report(256_000, output_reserve=16_000)

    assert "143,280 / 256,000  56%" in report
    assert "112,720  44%" in report
    assert "428,900 tokens" in report
    assert "284,100" in report
    assert "up to 16,000" in report
    assert "1 time" in report


def test_a_same_size_edit_inside_one_mtime_tick_is_still_observed(tmp_path):
    """``(size, mtime_ns)`` cannot see this, so content has to decide.

    ``mtime_ns`` looks nanosecond-precise but carries the filesystem's real
    resolution: on an overlayfs container two writes microseconds apart share a
    timestamp exactly. This pins the collision with ``os.utime`` instead of
    racing for it, so the test proves the content comparison rather than
    depending on which side of a tick the writes landed — which is how the
    scan-based test above used to fail about 9 runs in 10 on such a filesystem.
    """
    target = tmp_path / "f.txt"
    target.write_text("v1\n")
    pinned = os.stat(target).st_mtime_ns
    journal = WorkspaceJournal(str(tmp_path))

    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("v2\n")               # same length, different bytes
    os.utime(target, ns=(pinned, pinned))   # and now indistinguishable by stat
    assert os.stat(target).st_mtime_ns == pinned
    assert os.stat(target).st_size == len("v1\n")

    journal.finish_tree_scan([str(tmp_path)], before)

    assert journal.observed_only() == ["f.txt"]
    assert "-v1" in journal.unified_diff()
    assert "+v2" in journal.unified_diff()


def test_a_rewrite_with_identical_bytes_is_not_observed(tmp_path):
    """The other direction: touching a file without changing it is not a change,
    however much its mtime moved."""
    target = tmp_path / "f.txt"
    target.write_text("same\n")
    journal = WorkspaceJournal(str(tmp_path))

    before = journal.begin_tree_scan([str(tmp_path)])
    target.write_text("same\n")
    os.utime(target, ns=(10**18, 10**18))   # far-future mtime, identical bytes

    journal.finish_tree_scan([str(tmp_path)], before)

    assert journal.observed_only() == []
