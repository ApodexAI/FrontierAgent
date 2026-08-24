from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from benchmarks.public.apex_world import populate
from benchmarks.public.file_render import render_dir
from benchmarks.public.sandbox_profiles import apply_sandbox_profile


def test_gdpval_profile_disambiguates_duplicate_basenames() -> None:
    meta = {
        "benchmark": "gdpval",
        "question_id": "g1",
        "reference_files": "2023/report.xlsx|2024/report.xlsx",
    }

    apply_sandbox_profile(meta)

    mounts = meta["_sandbox_mounts"]
    assert mounts[0]["dst"] == "/inputs/report.xlsx"
    assert mounts[1]["dst"] == "/inputs/2024__report.xlsx"
    assert meta["_collect_outputs"] is True
    assert meta["bench_task_id"] == "g1"


def test_apex_populate_layers_task_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "world_files_zipped").mkdir(parents=True)
    with zipfile.ZipFile(root / "world_files_zipped" / "world.zip", "w") as archive:
        archive.writestr("filesystem/base.txt", "base")
    overlay = root / "task_files" / "task" / "filesystem"
    overlay.mkdir(parents=True)
    (overlay / "overlay.txt").write_text("overlay", encoding="utf-8")

    filesystem, apps_data = populate("world", "task", root, tmp_path / "work")

    assert (Path(filesystem) / "base.txt").read_text() == "base"
    assert (Path(filesystem) / "overlay.txt").read_text() == "overlay"
    assert Path(apps_data).is_dir()


def test_apex_populate_rejects_zip_traversal(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "world_files_zipped").mkdir(parents=True)
    with zipfile.ZipFile(root / "world_files_zipped" / "bad.zip", "w") as archive:
        archive.writestr("../escape.txt", "no")

    with pytest.raises(ValueError, match="unsafe zip member"):
        populate("bad", "task", root, tmp_path / "work")


def test_render_dir_labels_nested_outputs(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "answer.txt"
    output.parent.mkdir(parents=True)
    output.write_text("forty two", encoding="utf-8")

    rendered = render_dir(tmp_path)

    assert "===== /outputs/nested/answer.txt =====" in rendered
    assert "forty two" in rendered


# ── Book policy ───────────────────────────────────────────────────────────


def test_corpus_benchmarks_default_to_closed_book():
    """A benchmark answerable only from its own corpus must not get web tools.

    Leaving them bound does not just add capability — it changes what the score
    means, and makes it incomparable with anyone reporting the same benchmark
    closed-book. APEX's own prompt already says "use only provided files", so a
    bound web tool there contradicts the instructions the agent is given.
    """
    from benchmarks.public.sandbox_profiles import resolve_closed_book

    for bench in ("officeqa", "officeqa_full", "apex"):
        assert resolve_closed_book(bench) is True, bench
    # Web-research benchmarks stay open-book.
    for bench in ("browsecomp", "onemillion_bench", "gdpval"):
        assert resolve_closed_book(bench) is False, bench
    # An unregistered benchmark defaults to open rather than silently muting it.
    assert resolve_closed_book("not_a_benchmark") is False


def test_book_policy_override_wins_both_ways():
    from benchmarks.public.sandbox_profiles import resolve_closed_book

    assert resolve_closed_book("officeqa", False) is False   # --web
    assert resolve_closed_book("browsecomp", True) is True    # --no-web


class _FakeResourceManager:
    """Minimal stand-in exposing only what the tool resolvers touch."""

    global_tool_policy = None

    def __init__(self, names):
        self.all_tools = {n: type("T", (), {"name": n})() for n in names}

    def get_tools_for_role(self, _role):  # pragma: no cover - override path only
        return []


_ALL = [
    "web_search", "web_fetch", "download_file",
    "bash", "grep_search", "glob_search", "submit_report", "read_file",
]


def test_closed_book_unbinds_web_tools_on_the_path_that_actually_binds(monkeypatch):
    """Assert the *bound* list, not the AgentDefinition's permission pool.

    This test previously checked ``REACT_AGENT_DEF.allowed_tools`` and passed
    while the feature was completely inert: a profile's ``agent_tools`` list
    overrides the role pool, every shipped profile names the web tools, and
    that list is what reaches the model. Two full benchmark runs were labelled
    closed-book while both had web tools bound.
    """
    from workflows.stateful_react_agent.nodes.main_agent import (
        _tools_for_stateful_react,
    )

    rm = _FakeResourceManager(_ALL)
    cfg = {"agent_tools": ["web_search", "web_fetch", "bash",
                           "download_file", "grep_search", "glob_search"]}

    monkeypatch.setenv("REACT_NO_WEB", "1")
    got = [t.name for t in _tools_for_stateful_react(rm, cfg)]
    assert got == ["bash", "grep_search", "glob_search"], got

    monkeypatch.setenv("REACT_NO_WEB", "0")
    got = [t.name for t in _tools_for_stateful_react(rm, cfg)]
    assert "web_search" in got and "web_fetch" in got and "download_file" in got


def test_closed_book_unbinds_agent_team_sub_agent_web_tools(monkeypatch):
    """Same defect existed in agent-team's profile-override path."""
    from workflows.agent_team.nodes.main_agent import _resolve_profile_tools

    rm = _FakeResourceManager(_ALL)
    override = ["web_search", "web_fetch", "submit_report", "bash",
                "download_file", "grep_search", "glob_search", "read_file"]

    monkeypatch.setenv("SWARM_NO_WEB", "1")
    _, names = _resolve_profile_tools(
        rm, role_id="sub", override=override, label="sub_agent_tools",
    )
    assert not ({"web_search", "web_fetch", "download_file"} & set(names)), names
    assert "bash" in names

    monkeypatch.setenv("SWARM_NO_WEB", "0")
    _, names = _resolve_profile_tools(
        rm, role_id="sub", override=override, label="sub_agent_tools",
    )
    assert "web_search" in names


def test_local_profile_never_auto_selects_a_cloud_sandbox(monkeypatch):
    """An API key in .env is not consent to ship the user's files off-box.

    Regression: with E2B_API_KEY present and SANDBOX_BACKEND unset, agent-team
    sub-agents' create_file calls went to a cloud sandbox during what was meant
    to be a local run, and two of five questions then hung silently for two
    hours. Reaching the cloud must take an explicit backend selection.
    """
    import importlib

    import plugins.tools._sandbox as sb
    from frontier_agent.infra.config import get_config

    monkeypatch.setenv("E2B_API_KEY", "test-key-not-used")
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    # CI deliberately supplies SANDBOX_BACKEND=bwrap for the test process.
    # Config is a process singleton, so deleting only os.environ leaves that
    # cached deployment setting active and never exercises the auto branch.
    monkeypatch.setattr(get_config(), "sandbox_backend", "")

    monkeypatch.setenv("SANDBOX_PROFILE", "local")
    mod = importlib.reload(sb)
    assert mod._resolve_use_e2b()[0] is False

    # A service deployment keeps the old behaviour.
    monkeypatch.setenv("SANDBOX_PROFILE", "service")
    mod = importlib.reload(sb)
    assert mod._resolve_use_e2b()[0] is True
