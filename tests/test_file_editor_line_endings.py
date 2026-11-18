"""file_editor_str_replace keeps a local file's line endings."""

from __future__ import annotations

import pytest

from plugins.tools._filesystem_state import (
    install_filesystem_state,
    reset_filesystem_state,
    state_for_sandbox_mode,
)
from plugins.tools.file_editor import file_editor_str_replace


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    dirs = {name: tmp_path / name for name in ("workspace", "outputs", "inputs")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    token = install_filesystem_state(
        state_for_sandbox_mode(
            "native",
            **{name: str(path) for name, path in dirs.items()},
        )
    )
    try:
        yield dirs["workspace"]
    finally:
        reset_filesystem_state(token)


@pytest.mark.parametrize("eol", ["\r\n", "\n"])
async def test_str_replace_keeps_the_file_line_endings(workspace, eol) -> None:
    target = workspace / "app.py"
    target.write_bytes(eol.join(["a = 1", "b = 2", "c = 3", ""]).encode())

    # The model writes "\n" whatever the file uses; the match is unchanged.
    result = await file_editor_str_replace.ainvoke(
        {"path": str(target), "old_str": "b = 2\nc = 3", "new_str": "b = 20\nc = 30"}
    )

    assert result.startswith("Replaced in")
    assert target.read_bytes() == eol.join(["a = 1", "b = 20", "c = 30", ""]).encode()
