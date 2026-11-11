"""Regression tests for the persistent permission store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apodex import permissions
from apodex.agent_tools import RISK_DENY, assess_with_rules
from apodex.permissions import PermissionStore


def test_missing_file_is_an_empty_policy(tmp_path: Path) -> None:
    store = PermissionStore.load(str(tmp_path / "permissions.json"))

    assert store.load_error is None
    assert not store.denies("bash", {"command": "git push"})


def test_unparseable_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    path.write_text('{"allow": [], "deny": ["Bash(git push)"]', encoding="utf-8")

    store = PermissionStore.load(str(path))

    assert store.load_error is not None
    assert store.denies("bash", {"command": "git push"})
    assert not store.allows("bash", {"command": "git push"})


def test_wrong_shape_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    path.write_text('{"allow": "Bash(git push)", "deny": []}', encoding="utf-8")

    store = PermissionStore.load(str(path))

    assert store.load_error is not None
    assert store.denies("bash", {"command": "git push"})


def test_unreadable_path_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    path.mkdir()

    store = PermissionStore.load(str(path))

    assert store.load_error is not None
    assert store.denies("bash", {"command": "git push"})


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require privileges on Windows")
def test_broken_symlink_fails_closed(tmp_path: Path) -> None:
    link = tmp_path / "permissions.json"
    link.symlink_to(tmp_path / "missing.json")

    store = PermissionStore.load(str(link))

    assert store.load_error is not None
    assert store.denies("bash", {"command": "git push"})


def test_load_reads_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    path.write_text(
        json.dumps({"allow": ["Bash(npm test)"], "deny": ["Bash(git push)"]}),
        encoding="utf-8",
    )

    store = PermissionStore.load(str(path))

    assert store.load_error is None
    assert store.denies("bash", {"command": "git push"})
    assert store.allows("bash", {"command": "npm test"})


def test_gate_blocks_when_rules_could_not_be_loaded(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    path.write_text("{", encoding="utf-8")
    store = PermissionStore.load(str(path))

    risk = assess_with_rules(
        "write_file",
        {"path": str(tmp_path / "created.txt")},
        str(tmp_path),
        store,
    )

    assert risk.level == RISK_DENY
    assert "could not be read" in risk.reason


def test_failed_save_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    path = policy_dir / "permissions.json"
    store = PermissionStore({"Bash(npm test)"}, {"Bash(git push)"}, str(path))
    store.save()
    before = path.read_text(encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(permissions.json, "dump", fail)
    store.save()

    assert path.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in policy_dir.iterdir()) == ["permissions.json"]


def test_save_replaces_the_file_atomically(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    path = policy_dir / "permissions.json"
    store = PermissionStore({"Bash(npm test)"}, {"Bash(git push)"}, str(path))

    store.save()
    store.deny.add("write_file")
    store.save()

    reloaded = PermissionStore.load(str(path))
    assert reloaded.load_error is None
    assert reloaded.denies("write_file", {})
    assert reloaded.allows("bash", {"command": "npm test"})
    assert sorted(p.name for p in policy_dir.iterdir()) == ["permissions.json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_save_keeps_restrictive_permissions(tmp_path: Path) -> None:
    path = tmp_path / "permissions.json"
    store = PermissionStore({"Bash(npm test)"}, {"Bash(git push)"}, str(path))
    store.save()
    os.chmod(path, 0o600)

    store.deny.add("write_file")
    store.save()

    assert path.stat().st_mode & 0o777 == 0o600
