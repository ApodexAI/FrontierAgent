"""The fail-closed host-path gate in ``plugins.tools._path_auth``.

This gate is the whole boundary for the in-process file tools (read_text,
grep_search, glob_search, write_file, file_editor): bubblewrap jails ``bash``,
not them. So a hole here reaches the host even in an isolated run.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.tools._path_auth import _authorized_local_path, _blocked_name


@pytest.fixture
def workspace(monkeypatch, tmp_path: Path) -> Path:
    """A task workspace authorized the way the terminal authorizes ``--cwd``."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("CODING_WORKSPACE_ROOT", str(root))
    return root


def test_workspace_file_is_authorized(workspace: Path) -> None:
    resolved, reason = _authorized_local_path(str(workspace / "app.py"))

    assert resolved == workspace / "app.py", reason


def test_user_owned_spill_directory_remains_searchable(workspace: Path) -> None:
    """Only the resolved recovery store is reserved; ``spill`` is an ordinary
    and common project directory name."""
    from plugins.tools.glob_search import _local_glob
    from plugins.tools.grep_search import _local_grep

    spill_dir = workspace / "spill"
    spill_dir.mkdir()
    note = spill_dir / "notes.md"
    note.write_text("user-owned canary", encoding="utf-8")

    globbed = _local_glob("**/*.md", str(workspace), 100)
    grepped = _local_grep("user-owned canary", str(workspace), None, 0, 50)

    assert "spill/notes.md" in globbed
    assert "user-owned canary" in grepped


@pytest.mark.parametrize("write_access", [False, True])
def test_symlink_out_of_the_workspace_is_refused(workspace: Path, tmp_path: Path,
                                                 write_access: bool) -> None:
    """The model can create symlinks in its own workspace, so honouring the
    unresolved path let it read (and write) anything on the host."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "id_rsa").write_text("PRIVATE KEY")
    (workspace / "notes").symlink_to(outside)

    resolved, reason = _authorized_local_path(
        str(workspace / "notes" / "id_rsa"), write_access=write_access,
    )

    assert resolved is None
    assert "Access restricted" in reason


def test_symlink_cannot_rename_a_blocked_file(workspace: Path) -> None:
    """A benign-looking name must not smuggle a blocked file past the check —
    the pattern test runs on the resolved name."""
    (workspace / ".env.prod").write_text("OPENAI_API_KEY=sk-x")
    (workspace / "notes.md").symlink_to(workspace / ".env.prod")

    resolved, reason = _authorized_local_path(str(workspace / "notes.md"))

    assert resolved is None
    assert ".env" in reason


def test_skill_symlinks_still_widen_access() -> None:
    """plugins/skills/ is operator-curated, and its links deliberately point at
    SKILL.md bodies outside the project — that stays allowed."""
    skills = Path(__file__).resolve().parents[1] / "plugins" / "skills"

    resolved, reason = _authorized_local_path(str(skills / "any" / "SKILL.md"))

    assert resolved is not None, reason


@pytest.mark.parametrize("name", [
    ".env", ".env.production", "credentials.json", "aws_credentials",
    "api_token.txt", "server.key", "client.pem", "ca.cert", "my-secret.yaml",
])
def test_sensitive_names_stay_blocked(name: str) -> None:
    assert _blocked_name(name)


@pytest.mark.parametrize("name", [
    "tokenizer_config.json",   # 'token'
    "secretary_notes.md",      # 'secret'
    "deck.keynote",            # '.key'
    "env.example",
    "passwordless_login.md",   # 'passwordless' is not 'password'
    "README.md",
])
def test_ordinary_names_are_not_false_positives(name: str) -> None:
    """The old substring test refused all of these — ``tokenizer_config.json``
    in particular makes any ML checkout unreadable."""
    assert not _blocked_name(name)


@pytest.fixture
def skills_tree(tmp_path: Path):
    """A real ``plugins/skills/<id>`` symlink pointing at an external body.

    The OSS checkout ships no ``plugins/skills/``, so the curated tree is
    created for the duration of the test and removed afterwards.
    """
    skills = Path(__file__).resolve().parents[1] / "plugins" / "skills"
    created_root = not skills.exists()
    skills.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external-skill"
    external.mkdir()
    (external / "SKILL.md").write_text("EXTERNAL SKILL BODY")
    (external / ".env.prod").write_text("OPENAI_API_KEY=sk-x")
    # An innocuous name, so the ONLY thing that can block it is the target.
    (external / "notes.md").symlink_to(external / ".env.prod")
    link = skills / "_pytest_probe"
    link.symlink_to(external)
    try:
        yield link
    finally:
        link.unlink()
        if created_root:
            skills.rmdir()


def test_curated_skill_symlinks_stay_readable(skills_tree: Path) -> None:
    """The exception exists so an operator-installed skill body outside the
    checkout can be read — that must keep working."""
    resolved, reason = _authorized_local_path(str(skills_tree / "SKILL.md"))

    assert resolved is not None, reason


def test_curated_skill_symlinks_do_not_widen_write_access(skills_tree: Path) -> None:
    """Nothing in the skills tree is a write target, so the read-only exception
    must not authorize a write through the same link."""
    resolved, _reason = _authorized_local_path(
        str(skills_tree / "overwrite.txt"), write_access=True,
    )

    assert resolved is None


def test_a_curated_symlink_cannot_rename_a_blocked_target(skills_tree: Path) -> None:
    """The exception authorizes the unresolved path, so the blocked-name test
    has to judge the real target rather than the link's own harmless name."""
    resolved, reason = _authorized_local_path(str(skills_tree / "notes.md"))

    assert resolved is None
    assert ".env" in reason


async def test_recursive_search_does_not_read_through_workspace_symlinks(
    workspace: Path, tmp_path: Path,
) -> None:
    """Authorizing only the search root is not enough: rglob/glob walk into
    symlinked descendants, including links that are an ANCESTOR of the match."""
    from plugins.tools.glob_search import glob_search
    from plugins.tools.grep_search import grep_search

    external = tmp_path / "outside"
    external.mkdir()
    (external / "host-data.txt").write_text("EXTERNAL MARKER 4242\n")
    (workspace / "own.txt").write_text("EXTERNAL MARKER own\n")
    (workspace / "innocent.txt").symlink_to(external / "host-data.txt")
    (workspace / "linkdir").symlink_to(external)

    grepped = await grep_search.ainvoke(
        {"pattern": "EXTERNAL MARKER", "path": str(workspace)},
    )
    globbed = await glob_search.ainvoke({"pattern": "*.txt", "path": str(workspace)})

    assert "4242" not in grepped          # external content
    assert "own.txt" in grepped           # the workspace's own file still matches
    assert "host-data" not in globbed     # external names are not disclosed even as names


async def test_search_reads_explicit_inputs_below_ignored_runtime_dir(
    workspace: Path, monkeypatch,
) -> None:
    """The input mount is authoritative even when its host path is gitignored."""
    from plugins.tools.glob_search import glob_search
    from plugins.tools.grep_search import grep_search

    (workspace / ".gitignore").write_text(".apodex/\n")
    inputs = workspace / ".apodex" / "runtime" / "inputs" / "run-id"
    inputs.mkdir(parents=True)
    (inputs / "brief.md").write_text("ATTACHED INPUT MARKER\n")
    unrelated = workspace / ".apodex" / "unrelated.txt"
    unrelated.write_text("MUST STAY IGNORED\n")
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))

    globbed = await glob_search.ainvoke({"pattern": "*", "path": str(inputs)})
    grepped = await grep_search.ainvoke(
        {"pattern": "ATTACHED INPUT", "path": str(inputs)},
    )
    grepped_file = await grep_search.ainvoke(
        {"pattern": "ATTACHED INPUT", "path": str(inputs / "brief.md")},
    )
    workspace_glob = await glob_search.ainvoke(
        {"pattern": "**/*.txt", "path": str(workspace)},
    )

    assert "brief.md" in globbed
    assert "ATTACHED INPUT MARKER" in grepped
    assert "ATTACHED INPUT MARKER" in grepped_file
    assert "unrelated.txt" not in workspace_glob


async def test_grep_reads_explicit_spill_directory(workspace: Path) -> None:
    """A manifest path opts into the hidden store without exposing it globally."""
    from plugins.tools.grep_search import grep_search

    spill = workspace / ".spill" / "session-id"
    spill.mkdir(parents=True)
    (spill / "evidence.md").write_text("RECOVERY MARKER 7391\n")

    explicit = await grep_search.ainvoke({
        "pattern": "RECOVERY MARKER",
        "path": str(spill),
    })
    ordinary = await grep_search.ainvoke({
        "pattern": "RECOVERY MARKER",
        "path": str(workspace),
    })

    assert "RECOVERY MARKER 7391" in explicit
    assert "RECOVERY MARKER 7391" not in ordinary


async def test_glob_hides_the_spill_store_but_honours_an_explicit_path(
    workspace: Path,
) -> None:
    """Recovery goes through the paths the manifest names, not discovery.

    ``_overflow`` writes spill INTO the workspace, so without an ignore rule a
    post-compaction ``glob_search("**/*.md")`` lists the agent's own spilled
    bodies and reading one re-injects exactly what compaction removed. Nested
    stores count: a sub-agent's workspace sits below the main worktree.
    """
    from plugins.tools.glob_search import glob_search

    (workspace / ".spill" / "session").mkdir(parents=True)
    (workspace / ".spill" / "session" / "top.md").write_text("TOP SPILL\n")
    (workspace / "sub" / ".spill" / "s2").mkdir(parents=True)
    (workspace / "sub" / ".spill" / "s2" / "nested.md").write_text("NESTED SPILL\n")
    (workspace / "report.md").write_text("ordinary deliverable\n")

    ordinary = await glob_search.ainvoke({"pattern": "**/*.md", "path": str(workspace)})
    explicit = await glob_search.ainvoke({
        "pattern": "*.md", "path": str(workspace / ".spill" / "session"),
    })

    assert "report.md" in ordinary
    assert ".spill" not in ordinary
    assert "top.md" in explicit


async def test_grep_hides_nested_spill_stores_from_an_ordinary_search(
    workspace: Path,
) -> None:
    """``.spill`` must be skipped at any depth, not only at the search root."""
    from plugins.tools.grep_search import grep_search

    nested = workspace / "sub" / ".spill" / "s2"
    nested.mkdir(parents=True)
    (nested / "evidence.md").write_text("NESTED RECOVERY 5512\n")
    (workspace / "report.md").write_text("NESTED RECOVERY is discussed here\n")

    ordinary = await grep_search.ainvoke({
        "pattern": "NESTED RECOVERY", "path": str(workspace),
    })
    explicit = await grep_search.ainvoke({
        "pattern": "NESTED RECOVERY", "path": str(nested),
    })

    assert "5512" not in ordinary
    assert "report.md" in ordinary  # ordinary files are still searched
    assert "5512" in explicit


async def test_a_symlink_into_the_recovery_store_stays_hidden(
    monkeypatch, workspace: Path,
) -> None:
    """A lexical test cannot see through a link, so the resolve has to stay.

    ``spill_path_matcher`` only spends a ``resolve`` on paths that ARE symlinks —
    that narrowing is what took the per-file cost from ~36us to ~3us — so this
    pins the case the narrowing kept. The store is placed inside the workspace so
    the link's target is authorized and the spill filter is what does the hiding;
    an unauthorized target would make this pass for the wrong reason.
    """
    from plugins.tools.grep_search import grep_search

    store = workspace / "runs" / "spill"
    store.mkdir(parents=True)
    (store / "evidence.md").write_text("LINKED RECOVERY 8823\n")
    (workspace / "shortcut.md").symlink_to(store / "evidence.md")
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))

    ordinary = await grep_search.ainvoke({
        "pattern": "LINKED RECOVERY", "path": str(workspace),
    })
    explicit = await grep_search.ainvoke({
        "pattern": "LINKED RECOVERY", "path": str(store),
    })

    assert "8823" not in ordinary
    assert "shortcut.md" not in ordinary
    assert "8823" in explicit


@pytest.mark.parametrize(
    ("search_path", "expect_excluded"),
    [("/workspace", True), ("/workspace/.spill/session", False)],
)
async def test_sandbox_search_branch_prunes_spill_unless_targeted(
    monkeypatch, search_path: str, expect_excluded: bool,
) -> None:
    """The sandbox branch, not ``_local_grep``, is what real workspace paths
    take — it shells out to grep/find directly, so the ignore rules never ran
    there. Without a prune the store is only hidden when no sandbox exists.
    """
    from plugins.tools import _sandbox
    from plugins.tools.glob_search import glob_search
    from plugins.tools.grep_search import grep_search

    commands: list[str] = []

    class _Commands:
        def run(self, command: str, timeout: int = 30):
            commands.append(command)
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

    monkeypatch.setattr(_sandbox, "sandbox_available", lambda: True)
    monkeypatch.setattr(
        _sandbox, "get_sandbox", lambda: SimpleNamespace(commands=_Commands()),
    )

    await grep_search.ainvoke({"pattern": "MARKER", "path": search_path})
    await glob_search.ainvoke({"pattern": "**/*.md", "path": search_path})

    assert len(commands) == 2, commands
    grep_cmd, find_cmd = commands
    assert ("--exclude-dir=.spill" in grep_cmd) is expect_excluded, grep_cmd
    assert ("-name .spill -prune" in find_cmd) is expect_excluded, find_cmd


# ── the spill store's containment ────────────────────────────────────────


def test_the_store_is_read_authorized_and_never_write_authorized(
    tmp_path, monkeypatch,
) -> None:
    """One rule replaces the per-writer special cases: the store is not inside
    any write root, so write authorization simply never covers it."""
    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from plugins.tools import _overflow
    from plugins.tools._path_auth import _is_path_allowed
    from plugins.tools._sandbox import resolve_runtime_path

    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    token = set_current_execution_scope(
        ExecutionScope(task_id="t", metadata={"llm_session_id": "s"}),
    )
    try:
        ref = _overflow.spill_compacted_body("bash", "recovered " * 400)
        assert ref
        # The tools resolve the canonical path before authorizing; so must this.
        target = resolve_runtime_path(ref)

        assert _is_path_allowed(target)[0]
        assert not _is_path_allowed(target, write_access=True)[0]
    finally:
        reset_current_execution_scope(token)


def test_an_absent_store_adds_no_prefix(tmp_path, monkeypatch) -> None:
    """Gated on existence, like /inputs: a run that never spilled must not widen
    the read surface."""
    from plugins.tools._path_auth import _allowed_local_prefixes

    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "never-created"))

    assert str(tmp_path / "never-created") not in _allowed_local_prefixes()


def test_another_conversations_store_is_not_readable(tmp_path, monkeypatch) -> None:
    """The root is shared — a temp dir, or a run dir — so authorizing it would let
    one conversation read another's spilled bodies, which the old in-workspace
    layout made impossible. Only this scope, and stores this process created, are
    allowed."""
    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from plugins.tools import _overflow
    from plugins.tools._path_auth import _is_path_allowed

    root = tmp_path / "store"
    monkeypatch.setenv("APODEX_SPILL_DIR", str(root))
    saved = set(_overflow._created_stores)
    _overflow._created_stores.clear()

    # Another session, another process: present in the root, never created here.
    stranger = root / ("0" * 16)
    stranger.mkdir(parents=True)
    theirs = stranger / "body.md"
    theirs.write_text("theirs", encoding="utf-8")

    token = set_current_execution_scope(
        ExecutionScope(task_id="mine", metadata={"llm_session_id": "s"}),
    )
    try:
        from plugins.tools._sandbox import resolve_runtime_path

        mine = _overflow.spill_compacted_body("bash", "mine " * 400)
        assert mine
        assert _is_path_allowed(resolve_runtime_path(mine))[0], (
            "own recovery must stay readable"
        )
        assert not _is_path_allowed(str(theirs))[0], "another session's must not"
    finally:
        reset_current_execution_scope(token)
        _overflow._created_stores.clear()
        _overflow._created_stores.update(saved)


def test_an_in_process_subagents_store_stays_readable(tmp_path, monkeypatch) -> None:
    """A sub-agent spills under its OWN scope, and a fan-in report can carry that
    path back to the parent — so scope alone is too narrow. Stores this process
    created are authorized too."""
    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from plugins.tools import _overflow
    from plugins.tools._path_auth import _is_path_allowed

    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    saved = set(_overflow._created_stores)
    _overflow._created_stores.clear()
    try:
        token = set_current_execution_scope(
            ExecutionScope(task_id="sub", metadata={"llm_session_id": "sub-s"}),
        )
        try:
            sub_ref = _overflow.spill_compacted_body("collect_reports", "sub " * 400)
        finally:
            reset_current_execution_scope(token)
        assert sub_ref

        # Back in the parent's scope, the sub-agent's path is still readable.
        token = set_current_execution_scope(
            ExecutionScope(task_id="parent", metadata={"llm_session_id": "p-s"}),
        )
        try:
            from plugins.tools._sandbox import resolve_runtime_path

            assert _is_path_allowed(resolve_runtime_path(sub_ref))[0]
        finally:
            reset_current_execution_scope(token)
    finally:
        _overflow._created_stores.clear()
        _overflow._created_stores.update(saved)


def test_every_write_guard_refuses_the_store(tmp_path, monkeypatch) -> None:
    """The guards stay — ``native`` has no isolation to enforce anything with —
    but they now share one root-keyed rule instead of matching a directory name."""
    from plugins.tools._deliverable_policy import spill_write_error
    from plugins.tools._sandbox import is_spill_path
    from plugins.tools._writer_core import _write_root_error

    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setenv("APODEX_SPILL_DIR", str(store))
    target = str(store / "scope" / "body.md")

    assert is_spill_path(target)
    assert spill_write_error(target)
    assert _write_root_error(target, [str(tmp_path)])
    # The canonical path the model sees is covered by the same rule.
    assert is_spill_path("/spill/scope/body.md")
    assert spill_write_error("/spill/scope/body.md")
    # A sibling name is not the store.
    assert not is_spill_path(str(tmp_path / "store-old" / "x.md"))


def test_bwrap_mounts_the_store_read_only_outside_the_workspace(monkeypatch) -> None:
    """bwrap is the only backend that can mount; container relies on ownership
    and native on nothing, which is why the store had to leave the workspace."""
    from plugins.tools import _sandbox

    monkeypatch.setenv("APODEX_SPILL_DIR", "/tmp/spill-mount-test")

    assert _sandbox._DEFAULT_SPILL_DIR == "/spill"
    assert not _sandbox._DEFAULT_SPILL_DIR.startswith("/workspace")
    assert str(_sandbox.spill_root()) == "/tmp/spill-mount-test"


def test_the_store_is_not_made_tool_writable(monkeypatch) -> None:
    """The whole container guarantee, in one assertion.

    ``container`` cannot mount anything — it reuses the task container's existing
    mounts — so what stops a model command writing the store there is ownership:
    commands are dropped to an unprivileged uid, and only the workspace and
    outputs dirs are handed to that uid by ``_prepare_tool_writable``. The store
    being outside both is therefore the enforcement, which is exactly what moving
    it out of the workspace bought. If a future change adds the store root to that
    call, container silently loses its only protection — hence this test.
    """
    import inspect

    from plugins.tools import _sandbox

    source = inspect.getsource(_sandbox)
    calls = [
        line.strip() for line in source.splitlines()
        if "_prepare_tool_writable(" in line and "def " not in line
    ]
    assert calls, "the call sites moved; re-check what is made tool-writable"
    for call in calls:
        assert "spill" not in call.lower(), call
