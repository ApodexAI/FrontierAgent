from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier_agent.core.runtime.registries.agents import AgentRegistry
from frontier_agent.core.runtime.registries.workflows import WorkflowContext
from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
from plugins.tools._deliverable_policy import (
    bash_output_write_error,
    output_write_error,
    render_publish_assignment,
    reset_deliverable_write_paths,
    set_deliverable_write_paths,
    spill_write_error,
)
from plugins.tools.assign_task import _unknown_agent_validation_errors
from workflows.agent_team import register
from workflows.agent_team.fast_reporter_v1_evidence import parse_review_reply
from workflows.agent_team.nodes.main_agent import _replace_profile_tool_impls
from workflows.agent_team.nodes.reporter import agent_team_reporter
from workflows.agent_team.profile import (
    load_swarm_profile,
    resolve_reporter_backend,
    resolve_reporter_enabled,
)
from workflows.agent_team.prompts import _render_tools_main
from workflows.agent_team.subagent_runtime import render_sandbox_fs_note


def test_agent_team_registers_canonical_and_legacy_pipelines() -> None:
    pipelines = PipelineRegistry()
    agents = AgentRegistry()
    register(WorkflowContext(pipelines, agents))

    assert {spec.pipeline_id for spec in pipelines.list_all()} == {
        "agent_team",
        "agent_team_report",
        "agent-team",
        "agent-team-report",
    }
    assert not pipelines.get("agent_team").hidden
    assert pipelines.get("agent-team").hidden
    assert agents.has("agent_team_main")
    assert agents.has("agent_team_sub")


def test_reporter_defaults_are_fast_and_reporter_off_is_supported() -> None:
    assert resolve_reporter_backend(None) == "fast"
    assert not resolve_reporter_enabled(
        {"agent": {"reporter": False}},
        pipeline_id="agent_team_report",
        profile_name="agent_team_report",
    )


@pytest.mark.asyncio
async def test_heavy_reporter_fails_with_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="not included in the OSS port"):
        await agent_team_reporter(
            {"reporter_backend": "heavy", "metadata": {}},
            None,  # type: ignore[arg-type]
        )


def test_fast_reporter_review_cannot_invent_a_url() -> None:
    candidates = [{
        "url": "https://example.com/source",
        "title": "Source",
        "snippet": "Verbatim evidence",
        "source_type": "search",
    }]
    nodes, reason = parse_review_reply(
        '{"nodes":[{"i":1,"title":"Reviewed","quality":"high",'
        '"url":"https://invented.invalid"}]}',
        candidates,
    )
    assert reason == ""
    assert len(nodes) == 1
    assert nodes[0].url == "https://example.com/source"


def test_agent_team_ships_consistent_profiles() -> None:
    from pathlib import Path

    profile_dir = Path(__file__).parents[1] / "workflows" / "agent_team" / "profiles"
    assert {path.name for path in profile_dir.glob("*.yaml")} == {
        "simple.yaml",
        "benchmark.yaml",
        "tui.yaml",
    }


def test_main_prompt_requires_read_file_delegation() -> None:
    prompt = _render_tools_main(
        False,
        False,
        sub_agent_tools=["read_file", "bash", "submit_report"],
    )

    assert "may call ONLY the tools listed" in prompt
    assert "never invent another tool name" in prompt
    assert "pass the exact path in `assign_task`" in prompt
    assert "obtain the contents through `collect_reports`" in prompt
    assert "read_file" not in prompt
    assert "read_file_stub" not in prompt
    assert "read_file_wrapper" not in prompt
    assert "`bash`" not in prompt


def test_default_profile_aliases_benchmark(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    benchmark = load_swarm_profile("benchmark")
    tui = load_swarm_profile("tui")
    assert load_swarm_profile("default") == benchmark
    assert benchmark["agent"]["web_fetch_impl"] == "aligned"
    assert tui["agent"]["web_fetch_impl"] == "original"


def test_profile_selects_web_fetch_implementation() -> None:
    from types import SimpleNamespace

    from plugins.tools.create_subagent import _runtime_tools_override
    from plugins.tools.web_fetch import web_fetch
    from plugins.tools.web_fetch_aligned import web_fetch_aligned

    assert _replace_profile_tool_impls(
        [web_fetch], {"web_fetch_impl": "original"},
    ) == [web_fetch]
    assert _replace_profile_tool_impls(
        [web_fetch], {"web_fetch_impl": "aligned"},
    ) == [web_fetch_aligned]
    runtime = SimpleNamespace(sub_agent_tools=[web_fetch_aligned])
    assert _runtime_tools_override(runtime) == [web_fetch_aligned]


def test_native_publish_contract_maps_virtual_manifest_to_physical_root(
    tmp_path, monkeypatch,
) -> None:
    outputs = tmp_path / "run" / "outputs"
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))

    rendered = render_publish_assignment(["/outputs/report.md"])

    assert "`/outputs/report.md`" in rendered
    assert f"`{outputs / 'report.md'}`" in rendered
    assert "not a usable filesystem mount" in rendered


def test_native_physical_output_paths_still_obey_virtual_manifest_policy(
    tmp_path, monkeypatch,
) -> None:
    outputs = tmp_path / "run" / "outputs"
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    token = set_deliverable_write_paths(["/outputs/report.md"])
    try:
        assert output_write_error(str(outputs / "report.md")) is None
        assert "Undeclared deliverable" in (
            output_write_error(str(outputs / "extra.md")) or ""
        )
        assert bash_output_write_error(
            f"cp /workspace/report.md {outputs / 'report.md'}"
        ) is None
        assert "Undeclared deliverable" in (
            bash_output_write_error(
                f"cp /workspace/extra.md {outputs / 'extra.md'}"
            ) or ""
        )
        # Component-aware mapping must not protect or authorise a sibling root.
        assert output_write_error(str(outputs) + "-old/extra.md") is None
    finally:
        reset_deliverable_write_paths(token)


def test_spill_store_is_read_only_for_structured_and_symlinked_paths(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    spill = tmp_path / "store" / "session"
    spill.mkdir(parents=True)
    link = tmp_path / "workspace" / "recovery-link"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(spill, target_is_directory=True)

    for path in (
        "/spill/session/result.md",
        str(spill / "result.md"),
        # A symlink into the store is still the store.
        str(link / "result.md"),
    ):
        assert "read-only recovery store" in (spill_write_error(path) or "")
        assert "read-only recovery store" in (output_write_error(path) or "")

    assert output_write_error("/workspace/notes/result.md") is None


def test_bash_allows_spill_reads_but_rejects_mutations() -> None:
    spill = "/spill/session/result.md"
    allowed = (
        f"grep -n TARGET {spill}",
        f"grep -n TARGET {spill} 2>/dev/null",
        f"cat {spill} > /tmp/recovered-copy.md",
        f"cp {spill} /workspace/recovered-copy.md",
        f"python3 -c \"print(open('{spill}').read())\"",
    )
    denied = (
        f"echo changed > {spill}",
        f"printf changed >{spill}",
        f"> {spill} cat /workspace/input.md",
        f">{spill} cat /workspace/input.md",
        f"&> {spill} cat /workspace/input.md",
        f">| {spill} cat /workspace/input.md",
        f"<> {spill} cat /workspace/input.md",
        f"rm {spill}",
        f"mv {spill} /workspace/old.md",
        f"chmod 777 {spill}",
        f"cp /workspace/new.md {spill}",
        f"python3 -c \"open('{spill}', 'w').write('changed')\"",
        "cd /spill && echo changed > result.md",
        "P=/spill/session/result.md; echo changed > \"$P\"",
        "P=/spill/session/result.md echo changed > \"$P\"",
    )

    for command in allowed:
        assert bash_output_write_error(command) is None, command
    for command in denied:
        assert "read-only recovery store" in (
            bash_output_write_error(command) or ""
        ), command


@pytest.mark.asyncio
async def test_write_tools_reject_spill_before_acquiring_a_sandbox() -> None:
    from plugins.tools.create_file import create_file
    from plugins.tools.file_editor import file_editor_create, file_editor_str_replace
    from plugins.tools.read_file import read_file
    from plugins.tools.write_file import write_file

    spill = "/spill/session/result.md"
    results = [
        await write_file.ainvoke({"path": spill, "content": "changed"}),
        await file_editor_create.ainvoke({"path": spill, "content": "changed"}),
        await file_editor_str_replace.ainvoke({
            "path": spill, "old_str": "old", "new_str": "changed",
        }),
        await create_file.ainvoke({
            "path": spill,
            "ops": [{"create": {"content": "changed"}}],
        }),
        await create_file.ainvoke({
            "path": "/workspace/report.docx",
            "ops": [{"export_pdf": {"out": spill}}],
        }),
        await read_file.ainvoke({
            "path": "/workspace/source.txt", "save_to": spill,
        }),
    ]

    assert all("read-only recovery store" in result for result in results)


def test_native_main_prompt_distinguishes_manifest_and_filesystem_paths(
    tmp_path, monkeypatch,
) -> None:
    outputs = tmp_path / "run" / "outputs"
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))

    note = render_sandbox_fs_note(
        sandbox_mode="native", inputs_available=False, audience="main",
    )

    assert "always declare /outputs/<relative-file>" in note
    assert f"never {outputs}/<relative-file>" in note


def test_assignment_validation_reports_stale_agent_in_same_failure() -> None:
    bus = SimpleNamespace(get_session=lambda _session_id: None)

    errors = _unknown_agent_validation_errors(
        [{"agent": "report_publisher", "output_paths": ["bad"]}],
        bus=bus,
        bus_task_id="current-run",
        task_types=(),
    )

    assert len(errors) == 1
    assert "Unknown agent 'report_publisher'" in errors[0]
    assert "scoped to the current execution" in errors[0]


def test_bash_allows_spill_reads_that_publish_elsewhere() -> None:
    """A redirect OUT of the store is not a write INTO it.

    ``_definite_write_signal`` fires on any redirect naming ``/outputs``, so a
    recovery read that saved its hits refused with a reason that was false
    ("redirect output into it") — the target was ``/outputs/scratch``. Whether
    that write is allowed belongs to ``output_write_error``, which runs after.
    """
    spill = "/spill/session/result.md"
    allowed = (
        f"grep -n TARGET {spill} > /outputs/scratch/hits.txt",
        f"cat {spill} > /outputs/scratch/copy.md",
        f"grep -c TARGET {spill} >> /outputs/scratch/hits.txt",
        f"head -20 {spill} > /outputs/scratch/head.txt",
    )
    for command in allowed:
        assert bash_output_write_error(command) is None, command


def test_spill_gate_still_refuses_mutations_that_also_redirect_to_outputs() -> None:
    """The ``/outputs`` exemption must not become a laundering channel: a
    genuine mutation of the store stays refused however it redirects."""
    spill = "/spill/session/result.md"
    denied = (
        f"sed -i s/a/b/ {spill} > /outputs/scratch/log.txt",
        f"rm {spill} > /outputs/scratch/log.txt",
        f"tee {spill} < /etc/hostname > /outputs/scratch/log.txt",
        "find /spill -name '*.md' -delete > /outputs/scratch/log.txt",
        f"echo x > {spill} 2> /outputs/scratch/log.txt",
    )
    for command in denied:
        assert "read-only recovery store" in (
            bash_output_write_error(command) or ""
        ), command


def test_spill_cd_refusal_explains_the_real_reason() -> None:
    """``cd`` stays refused — segments split on ``&&``, so a later relative
    write never mentions the store and would escape the gate — but the generic
    "never redirect output into it" text described something the model did not
    do. It needs the absolute-path instruction instead."""
    error = bash_output_write_error("cd /spill/session && grep -c X result.md")

    assert error is not None
    assert "absolute path" in error
    assert "read-only recovery store" in error


def test_bwrap_mounts_the_spill_store_read_only(tmp_path, monkeypatch) -> None:
    """The lexical gate cannot be the only layer protecting recovery files.

    The bash token scan refuses every parseable way to write into the store, but
    shell expansion can hide a path from any scanner (a glob, a brace, a ``$VAR``
    assembled in pieces, a ``$(…)`` substitution), and the files are ordinary 0644
    files. File modes cannot close it either — model commands run as uid 0 inside
    the user namespace, so DAC is never consulted. A mount is.

    The mount is now a sibling of ``/workspace`` rather than a remount inside it,
    which is what let the ordering requirement go: the source no longer overlaps
    any writable bind, so it does not have to come last to win.
    """
    from plugins.tools import _sandbox

    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: True)
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    workspace = tmp_path / "ws"

    sandbox = _sandbox.BwrapSandbox(workspace=workspace)
    args = sandbox.commands._bind_args

    spill_at = args.index("--ro-bind-try")
    assert args[spill_at + 1] == str(tmp_path / "store")
    assert args[spill_at + 2] == "/spill"
    # Outside the workspace, so nothing above it overlaps.
    assert not str(tmp_path / "store").startswith(str(workspace))
    # ``--ro-bind-try``, not ``--ro-bind``: the store is created lazily on the
    # first spill and bwrap aborts the jail when a --ro-bind source is missing.
    assert "--ro-bind" not in args[spill_at:spill_at + 1]
    assert not (workspace / ".spill").exists()


def test_the_shipped_writer_agrees_with_the_authoritative_spill_rule(
    tmp_path, monkeypatch,
) -> None:
    """``_writer_core`` is concatenated into a standalone script and run where the
    ``plugins`` package does not exist, so it cannot call the real predicate and
    carries its own copy. This is what keeps the copy honest."""
    from plugins.tools import _writer_core
    from plugins.tools._sandbox import is_spill_path

    for env in ({"APODEX_SPILL_DIR": str(tmp_path / "store")},
                {"APODEX_RUN_DIR": str(tmp_path / "run")},
                {}):
        monkeypatch.delenv("APODEX_SPILL_DIR", raising=False)
        monkeypatch.delenv("APODEX_RUN_DIR", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from plugins.tools._sandbox import spill_root

        inside = str(spill_root() / "scope" / "body.md")
        for path in ("/spill/scope/body.md", inside, "/workspace/notes.md",
                     str(tmp_path / "store-old" / "x.md")):
            assert _writer_core._is_spill_path(path) == is_spill_path(path), (env, path)


def test_spill_gate_refuses_verbs_that_can_name_their_own_write_target() -> None:
    """``_READ_ONLY_VERBS`` membership is not proof a segment only reads.

    These verbs write only when asked to — but they can be asked to in shapes a
    token scan cannot distinguish from an input: inside their own program text,
    behind an output flag, or as a trailing operand. They reached
    ``_segment_reads_only`` and were exempted.

    It matters where no filesystem layer backs the promise: the bwrap jail
    re-mounts the store read-only and container mode leaves it owned by the
    harness, but with no tool account (or a non-root harness) ``CurrentSandbox``
    warns and runs model commands with the harness's OWN uid — and then this gate
    is all that protects the recovery files.
    """
    spill = "/spill/session/result.md"
    denied = (
        f"""awk '{{print > "{spill}"}}' /workspace/in.txt""",
        f"""awk '{{printf "x" >> "{spill}"}}' /workspace/in.txt""",
        f"sort -o {spill} /workspace/in.txt",
        f"sort --output={spill} /workspace/in.txt",
        f"uniq /workspace/in.txt {spill}",
        f"xxd /workspace/in.txt {spill}",
        f"yq -i '.a=1' {spill}",
    )
    for command in denied:
        error = bash_output_write_error(command)
        assert error is not None, command
        # The refusal must name a usable alternative, not just say no.
        assert "read_file" in error and "grep_search" in error, command

    # The documented recovery forms are unaffected...
    for command in (
        f"cat {spill}",
        f"grep -n TARGET {spill}",
        f"head -20 {spill}",
        f"wc -l {spill}",
    ):
        assert bash_output_write_error(command) is None, command

    # ...and the restriction is scoped to the store, not to the verbs.
    for command in (
        """awk '{print > "/workspace/out.txt"}' /workspace/in.txt""",
        "sort -o /workspace/out.txt /workspace/in.txt",
    ):
        assert bash_output_write_error(command) is None, command
