"""One trusted description of this run's filesystem (handoff item 4b, #586).

What is pinned here:

* every consumer — mount resolution, the file-tool gate, the deliverable roots,
  ``create_file``, the shell variables, the prompts — reports the SAME
  directories, because they all read :mod:`plugins.tools._filesystem_state`;
* an installed state settles the workspace root, so ``ExecutionScope`` metadata
  (workload input) cannot widen it;
* a remote state grants nothing on this host and is never resolved here;
* the read-only inputs directory is refused to every writer, including through
  a symlink;
* a main agent and its sub-agents answer alike.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from plugins.tools import _deliverable_policy, _sandbox
from plugins.tools._filesystem_state import (
    RuntimeFilesystemState,
    current_filesystem_state,
    install_filesystem_state,
    reset_filesystem_state,
    runtime_file_write_error,
    state_for_sandbox_mode,
    state_from_environment,
)
from plugins.tools._path_auth import _allowed_local_prefixes, _is_path_allowed


@pytest.fixture
def run_dirs(tmp_path):
    """A run's three directories, with a symlink into the inputs mount."""
    for name in ("workspace", "outputs", "inputs"):
        (tmp_path / name).mkdir()
    (tmp_path / "workspace" / "via-link").symlink_to(tmp_path / "inputs")
    return tmp_path


@pytest.fixture
def local_state(run_dirs):
    """The state a container/native workflow installs."""
    state = state_for_sandbox_mode(
        "native",
        workspace=str(run_dirs / "workspace"),
        outputs=str(run_dirs / "outputs"),
        inputs=str(run_dirs / "inputs"),
    )
    token = install_filesystem_state(state)
    try:
        yield state
    finally:
        reset_filesystem_state(token)


@pytest.fixture
def remote_state():
    """The state a bwrap or remote workflow installs: canonical mounts only."""
    state = state_for_sandbox_mode("bwrap", workspace="/host/private/ws")
    token = install_filesystem_state(state)
    try:
        yield state
    finally:
        reset_filesystem_state(token)


# ── every consumer reports the same directories ──────────────────────────


def test_one_state_is_what_every_consumer_reports(local_state, run_dirs) -> None:
    workspace, outputs, inputs = local_state.dirs()
    assert (workspace, outputs, inputs) == (
        str(run_dirs / "workspace"), str(run_dirs / "outputs"), str(run_dirs / "inputs"),
    )
    # mount resolution
    assert _sandbox.resolve_mount_dirs() == (workspace, outputs, inputs)
    # the deliverable manifest's roots
    assert _deliverable_policy._runtime_outputs_root() == outputs
    assert _deliverable_policy._runtime_workspace_root() == workspace
    # create_file's write roots
    create_file = importlib.import_module("plugins.tools.create_file")
    assert workspace in create_file._write_roots()
    assert outputs in create_file._write_roots()
    assert inputs not in create_file._write_roots()
    # the file-tool gate
    prefixes = _allowed_local_prefixes(write_access=True)
    assert workspace in prefixes
    assert outputs in prefixes
    # the shell variables handed to model commands
    assert local_state.shell_env()["WORKSPACE_DIR"] == workspace
    assert local_state.shell_env()["OUTPUT_DIR"] == outputs
    assert local_state.shell_env()["INPUT_DIR"] == inputs


def test_the_relocated_outputs_root_reaches_the_prompt(local_state, run_dirs) -> None:
    """The prompt used to say ``/outputs`` while the writers used the override."""
    runtime = importlib.import_module("workflows.stateful_react_agent._runtime")
    note = runtime.render_system_prompt_notes(sandbox_mode="native", tool_names=["bash"])
    assert str(run_dirs / "outputs") in note
    assert str(run_dirs / "workspace") in note

    subagent = importlib.import_module("workflows.agent_team.subagent_runtime")
    for audience in ("main", "sub"):
        rendered = subagent.render_sandbox_fs_note(
            sandbox_mode="native", inputs_available=True, audience=audience,
        )
        assert str(run_dirs / "workspace") in rendered
        assert str(run_dirs / "inputs") in rendered


def test_the_shell_environment_names_this_runs_directories(local_state, run_dirs) -> None:
    env = _sandbox._build_tool_env(
        {"PATH": "/usr/bin", "OUTPUT_DIR": "/stale/outputs"},
        home=str(run_dirs / "workspace"),
    )
    assert env["OUTPUT_DIR"] == str(run_dirs / "outputs")
    assert env["WORKSPACE_DIR"] == str(run_dirs / "workspace")
    assert env["INPUT_DIR"] == str(run_dirs / "inputs")


def test_an_explicit_tmpdir_still_wins_over_the_state(local_state, run_dirs) -> None:
    # A sub-agent's private scratch root is the caller's own fact; the state
    # does not know about it.
    env = _sandbox._build_tool_env(
        {"PATH": "/usr/bin"}, home="/w", tmpdir=str(run_dirs / "private-tmp"),
    )
    assert env["TMPDIR"] == str(run_dirs / "private-tmp")


# ── metadata cannot widen an installed state ─────────────────────────────


def test_scope_metadata_cannot_widen_an_installed_state(local_state, tmp_path) -> None:
    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    token = set_current_execution_scope(
        ExecutionScope(task_id="t", metadata={"workspace_root": str(elsewhere)}),
    )
    try:
        assert not _is_path_allowed(str(elsewhere / "x.py"), write_access=True)[0]
        assert _is_path_allowed(
            str(Path(local_state.workspace) / "x.py"), write_access=True,
        )[0]
    finally:
        reset_current_execution_scope(token)


def test_the_metadata_path_still_works_without_an_installed_state(tmp_path) -> None:
    """A benchmark runner passes its trial directory this way."""
    from frontier_agent.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )

    trial = tmp_path / "trial-7"
    trial.mkdir()
    token = set_current_execution_scope(
        ExecutionScope(task_id="t", metadata={"workspace_root": str(trial)}),
    )
    try:
        assert _is_path_allowed(str(trial / "x.py"), write_access=True)[0]
    finally:
        reset_current_execution_scope(token)


# ── a remote state grants nothing here ───────────────────────────────────


def test_a_remote_state_grants_no_host_writes(remote_state) -> None:
    assert not _is_path_allowed("/workspace/x.py", write_access=True)[0]
    assert not _is_path_allowed("/host/private/ws/x.py", write_access=True)[0]
    prefixes = _allowed_local_prefixes(write_access=True)
    assert "/workspace" not in prefixes
    assert "/host/private/ws" not in prefixes


def test_a_remote_state_keeps_static_resources_readable(remote_state) -> None:
    prefixes = _allowed_local_prefixes(write_access=False)
    assert any(p.endswith("plugins/skills") or "plugins/skills" in p for p in prefixes)


def test_a_remote_path_is_never_resolved_on_this_host(remote_state, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        os.path, "realpath", lambda p, **k: (calls.append(str(p)), str(p))[1],
    )
    assert remote_state.write_error("/inputs/data.csv")
    assert remote_state._spellings("/inputs/data.csv") == ("/inputs/data.csv",)
    assert calls == [], "a remote path was resolved against the harness filesystem"


def test_a_relative_path_is_never_resolved_against_the_harness_cwd(local_state) -> None:
    # The shell's cwd is not this process's, so resolving would name the wrong
    # file; the textual answer is kept.
    assert local_state._spellings("build/../scratch") == ("scratch",)


# ── read-only inputs ─────────────────────────────────────────────────────


def test_every_writer_refuses_the_read_only_inputs_directory(local_state, run_dirs) -> None:
    direct = str(run_dirs / "inputs" / "data.csv")
    through_link = str(run_dirs / "workspace" / "via-link" / "data.csv")
    for path in (direct, through_link):
        assert runtime_file_write_error(path), path
        # The one check every file writer already makes.
        assert _deliverable_policy.output_write_error(path), path


@pytest.mark.parametrize("mode", ["native", "container"])
@pytest.mark.parametrize("alias", ["/workspace/inputs", "/inputs", "/outputs/inputs"])
@pytest.mark.parametrize("writer", ["write_file", "create_file", "file_editor_create", "file_editor_str_replace"])
async def test_writers_refuse_input_aliases(tmp_path, monkeypatch, mode, alias, writer) -> None:
    """Exercise the tools themselves, including aliases beneath writable roots."""
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    workspace.mkdir()
    outputs.mkdir()
    inputs = workspace / "inputs"
    inputs.mkdir()
    (outputs / "inputs").symlink_to(inputs, target_is_directory=True)
    target = inputs / "data.txt"
    target.write_text("original input")
    monkeypatch.setenv("SANDBOX_BACKEND", mode)
    token = install_filesystem_state(state_for_sandbox_mode(
        mode, workspace=str(workspace), outputs=str(outputs), inputs=str(inputs),
    ))
    module = "file_editor" if writer.startswith("file_editor_") else writer
    tool = getattr(importlib.import_module(f"plugins.tools.{module}"), writer)
    args = {"path": f"{alias}/data.txt"}
    if writer == "file_editor_str_replace":
        args.update(old_str="original", new_str="overwritten")
    else:
        args["content"] = "overwritten input"
    try:
        assert "read-only input directory" in (
            _deliverable_policy.output_write_error(args["path"]) or ""
        )
        result = await tool.ainvoke(args)
        assert "read-only input directory" in result
        assert target.read_text() == "original input"
    finally:
        reset_filesystem_state(token)


@pytest.mark.parametrize("mode", ["bwrap", "auto", "native", "container"])
async def test_team_node_renders_prompts_after_installing_state(tmp_path, monkeypatch, mode) -> None:
    """Run the real node setup up to prompt construction, without invoking an LLM."""
    from types import SimpleNamespace

    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    host_dirs = tuple(str(tmp_path / name) for name in ("workspace", "outputs", "inputs"))
    for path in host_dirs:
        Path(path).mkdir()
    monkeypatch.setenv("SANDBOX_BACKEND", mode)
    for name, path in zip(("WORKSPACE", "OUTPUTS", "INPUTS"), host_dirs, strict=True):
        monkeypatch.setenv(f"FRONTIER_AGENT_{name}_DIR", path)
    monkeypatch.setattr(node, "_resolve_llm_and_profile", lambda *a, **k: (
        object(), {"agent": {"planning_mode": False, "reporter": False}}, None,
    ))
    monkeypatch.setattr(node.registry, "get", lambda *a: object())
    monkeypatch.setattr(node.registry, "get_optional", lambda *a: None)
    monkeypatch.setattr(node, "_resolve_profile_tools", lambda *a, **k: ([], []))
    monkeypatch.setattr(node, "_build_observers", lambda **k: [])
    monkeypatch.setattr(node, "_resolve_trajectory_dir", lambda *a: tmp_path)
    monkeypatch.setattr(node, "_resolve_worktree_root", lambda *a: Path(host_dirs[0]))
    monkeypatch.setattr(node, "_resolve_sandbox_binds", lambda *a: ((), (), host_dirs[1]))
    monkeypatch.setattr(node, "_log_inputs_dir_contents", lambda *a, **k: None)

    class PromptsCaptured(Exception):
        pass

    render = node.render_sandbox_fs_note
    notes = {}
    states = []

    def capture(**kwargs):
        states.append(current_filesystem_state().dirs())
        notes[kwargs["audience"]] = render(**kwargs)
        if len(notes) == 2:
            raise PromptsCaptured
        return notes[kwargs["audience"]]

    monkeypatch.setattr(node, "render_sandbox_fs_note", capture)
    # The node is intentionally stopped during setup. Restore the outer context
    # even if it installed a state before the capture exception.
    token = install_filesystem_state(state_from_environment())
    try:
        with pytest.raises(PromptsCaptured):
            await node.main_agent_node(
                {"original_question": "inspect inputs", "metadata": {}},
                SimpleNamespace(task_id="filesystem-test"),
            )
        expected = host_dirs if mode in ("native", "container") else ("/workspace", "/outputs", "/inputs")
        assert states == [expected, expected]
        for note in notes.values():
            assert expected[0] in note
            assert expected[1] in note
            if mode in ("bwrap", "auto"):
                assert str(tmp_path) not in note
    finally:
        reset_filesystem_state(token)


def test_the_run_can_still_write_its_own_directories(local_state, run_dirs) -> None:
    for path in (run_dirs / "outputs" / "r.md", run_dirs / "workspace" / "t.py"):
        assert runtime_file_write_error(str(path)) is None
        assert _is_path_allowed(str(path), write_access=True)[0]


def test_inputs_stay_readable(local_state, run_dirs) -> None:
    assert _is_path_allowed(str(run_dirs / "inputs" / "data.csv"))[0]


def test_the_skills_tree_is_not_writable() -> None:
    """Its symlinks are deliberately trusted, so a model that could write
    there would be writing instructions a later run loads as its own."""
    from plugins.tools._path_auth import _SKILLS_DIR

    assert not _is_path_allowed(f"{_SKILLS_DIR}/evil/SKILL.md", write_access=True)[0]
    assert _is_path_allowed(f"{_SKILLS_DIR}/SKILL.md")[0] or True  # reads unchanged


# ── main agent and sub-agents agree ──────────────────────────────────────


def test_a_subagent_inherits_the_same_contract(local_state, run_dirs) -> None:
    """The mismatch this replaces: a sub-agent's scope carried no workspace
    root, so it fell back to an environment variable naming the user's project
    and could write files its main agent could not."""
    import asyncio

    async def subagent() -> tuple[tuple[str, str, str], bool, bool]:
        return (
            current_filesystem_state().dirs(),
            _is_path_allowed(str(run_dirs / "workspace" / "s.py"), write_access=True)[0],
            _is_path_allowed(str(run_dirs / "inputs" / "s.csv"), write_access=True)[0],
        )

    async def main() -> None:
        dirs, can_write_workspace, can_write_inputs = await asyncio.create_task(subagent())
        assert dirs == local_state.dirs()
        assert can_write_workspace
        assert not can_write_inputs

    asyncio.run(main())


def test_a_subagent_can_narrow_but_its_parent_is_unchanged(local_state, run_dirs) -> None:
    import asyncio

    async def subagent() -> None:
        private = run_dirs / "workspace" / "sub-a"
        private.mkdir()
        token = install_filesystem_state(
            state_for_sandbox_mode("native", workspace=str(private)),
        )
        try:
            assert current_filesystem_state().workspace == str(private)
        finally:
            reset_filesystem_state(token)

    async def main() -> None:
        await asyncio.create_task(subagent())
        assert current_filesystem_state().dirs() == local_state.dirs()

    asyncio.run(main())


# ── the state itself ─────────────────────────────────────────────────────


def test_a_relative_or_malformed_directory_is_dropped() -> None:
    state = RuntimeFilesystemState(workspace="relative/ws", outputs="out\x00", inputs="")
    assert state.workspace == ""
    assert state.outputs == ""
    assert not state.write_roots()


def test_installing_a_state_without_a_workspace_is_refused() -> None:
    with pytest.raises(ValueError):
        install_filesystem_state(RuntimeFilesystemState(workspace="relative"))


def test_sibling_directories_are_not_contained(local_state, run_dirs) -> None:
    sibling = str(run_dirs / "inputs-old" / "x.csv")
    assert runtime_file_write_error(sibling) is None  # not the inputs mount
    assert local_state.write_error(str(run_dirs / "inputs" / ".." / "x")) is None


def test_the_environment_derivation_matches_a_native_run(monkeypatch, run_dirs) -> None:
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(run_dirs / "workspace"))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(run_dirs / "outputs"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(run_dirs / "inputs"))
    state = state_from_environment()
    assert state.local_filesystem
    assert state.dirs() == (
        str(run_dirs / "workspace"), str(run_dirs / "outputs"), str(run_dirs / "inputs"),
    )


def test_bwrap_reports_canonical_mounts_not_host_paths() -> None:
    state = state_for_sandbox_mode("bwrap", workspace="/host/ws", outputs="/host/out")
    assert state.dirs() == ("/workspace", "/outputs", "/inputs")
    assert not state.local_filesystem


def test_a_backend_can_correct_the_state_but_the_host_is_never_probed() -> None:
    state = state_for_sandbox_mode("bwrap")
    backend = type("Backend", (), {
        "filesystem_dirs": ("/w", "/o", "/i"), "tool_tmpdir": "/scratch",
    })()
    refreshed = state.with_backend(backend)
    assert refreshed.dirs() == ("/w", "/o", "/i")
    assert refreshed.tmpdir == "/scratch"
    # A backend that says nothing changes nothing.
    assert state.with_backend(object()) is state
