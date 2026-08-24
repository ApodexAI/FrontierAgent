from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apodex.attachments import AttachmentError, AttachmentManager


def _manager(monkeypatch, tmp_path: Path) -> AttachmentManager:
    staging = tmp_path / "staging"
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(staging))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    return AttachmentManager(str(tmp_path), "session")


def test_attach_copies_and_enriches_task(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "claim.pdf"
    source.write_bytes(b"%PDF-example")
    manager = _manager(monkeypatch, tmp_path)

    added = manager.attach(str(source))

    assert [item.relative_path for item in added] == ["claim.pdf"]
    assert added[0].agent_path == "/inputs/claim.pdf"
    assert (manager.staging_dir / "claim.pdf").read_bytes() == b"%PDF-example"
    assert manager.attach(str(source))[0].relative_path == "claim.pdf"
    assert [item.relative_path for item in manager.list()] == ["claim.pdf"]
    enriched = manager.enrich_task("review this claim")
    assert enriched.startswith("review this claim\n\n<attached_files>")
    assert "with read_file before answering" in enriched
    assert 'path="/inputs/claim.pdf" size=12 bytes' in enriched


def test_agent_team_attachment_prompt_delegates_file_reading(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "claim.txt"
    source.write_text("evidence")
    manager = _manager(monkeypatch, tmp_path)
    manager.attach(str(source))

    enriched = manager.enrich_task(
        "review this claim", delegate_file_reading=True,
    )

    assert "Delegate full inspection to a sub-agent" in enriched
    assert "Do not call or invent a file-reading tool yourself" in enriched
    assert "with read_file before answering" not in enriched


def test_session_selects_delegated_attachment_contract_for_agent_team(
    monkeypatch,
) -> None:
    from apodex.session import TerminalSession

    captured: dict[str, object] = {}

    class Attachments:
        def enrich_task(self, task: str, **kwargs) -> str:
            captured.update(kwargs)
            return task

    session = SimpleNamespace(
        mode="agent_team",
        attachments=Attachments(),
        _deliverable_context=lambda: "",
    )
    monkeypatch.setattr(
        "apodex.session.get_profile",
        lambda _mode: SimpleNamespace(workflow="agent_team"),
    )

    assert TerminalSession._enrich_task(session, "inspect") == "inspect"
    assert captured == {"delegate_file_reading": True}


def test_relative_file_and_directory_paths_resolve_under_workspace(
    monkeypatch, tmp_path: Path,
) -> None:
    workspace = tmp_path / "mounted-workspace"
    nested = workspace / "inputs" / "quarterly"
    nested.mkdir(parents=True)
    (nested / "brief.txt").write_text("brief")
    (nested / "chart.csv").write_text("x,y\n1,2\n")
    elsewhere = tmp_path / "process-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manager = _manager(monkeypatch, workspace)

    added_file = manager.attach("inputs/quarterly/brief.txt")
    added_directory = manager.attach("inputs/quarterly")

    assert [item.relative_path for item in added_file] == ["brief.txt"]
    assert {item.relative_path for item in added_directory} == {
        "quarterly/brief.txt", "quarterly/chart.csv",
    }


def test_relative_attachment_path_cannot_escape_workspace(
    monkeypatch, tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    manager = _manager(monkeypatch, workspace)

    with pytest.raises(AttachmentError, match="escapes workspace"):
        manager.attach("../outside.txt")


def test_mount_roots_are_scoped_per_session(monkeypatch, tmp_path: Path) -> None:
    staging_root = tmp_path / "staging-root"
    monkeypatch.delenv("APODEX_INPUT_STAGING_DIR", raising=False)
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    monkeypatch.setenv("APODEX_INPUT_STAGING_ROOT", str(staging_root))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_ROOT", "/inputs")

    first = AttachmentManager(str(tmp_path), "session-one")
    # Simulate TerminalSession publishing the first manager's exact agent path.
    # The root contract must still win when an in-process /new changes session.
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(first.agent_dir))
    second = AttachmentManager(str(tmp_path), "session-two")

    assert first.staging_dir == staging_root / "session-one"
    assert first.agent_dir == Path("/inputs/session-one")
    assert second.staging_dir == staging_root / "session-two"
    assert second.agent_dir == Path("/inputs/session-two")


def test_duplicate_names_are_preserved_and_detach_is_scoped(
    monkeypatch, tmp_path: Path,
) -> None:
    first = tmp_path / "one" / "photo.png"
    second = tmp_path / "two" / "photo.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manager = _manager(monkeypatch, tmp_path)

    manager.attach_many([str(first), str(second)])
    assert [item.relative_path for item in manager.list()] == [
        "photo-2.png", "photo.png",
    ]
    assert manager.detach("photo.png") == 1
    assert [item.relative_path for item in manager.list()] == ["photo-2.png"]
    with pytest.raises(AttachmentError, match="invalid attachment name"):
        manager.detach("../outside")


def test_directory_with_symlink_is_rejected(monkeypatch, tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    (folder / "link.txt").symlink_to(target)
    manager = _manager(monkeypatch, tmp_path)

    with pytest.raises(AttachmentError, match="symbolic links"):
        manager.attach(str(folder))
