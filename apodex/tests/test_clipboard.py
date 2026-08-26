from __future__ import annotations

from pathlib import Path

import pytest

from apodex.attachments import AttachmentManager
from apodex.clipboard import (
    _BROKER_TOKEN_ENV,
    _BROKER_URL_ENV,
    ClipboardBroker,
    ClipboardPaste,
    _path_text,
    capture_macos_clipboard,
    paste_from_clipboard,
)


def _manager(monkeypatch, tmp_path: Path) -> AttachmentManager:
    monkeypatch.setenv("APODEX_INPUT_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", "/inputs")
    return AttachmentManager(str(tmp_path), "session")


def test_path_text_requires_explicit_local_file_urls(tmp_path: Path) -> None:
    source = tmp_path / "policy wording.pdf"
    source.write_text("policy")
    second = tmp_path / "claim photo.png"
    second.write_bytes(b"png")

    assert _path_text(source.as_uri()) == [str(source.resolve())]
    assert _path_text(f"{source.as_uri()}\n{second.as_uri()}") == [
        str(source.resolve()), str(second.resolve()),
    ]
    assert _path_text(f'"{source}"') is None
    assert _path_text(f"{source}\n{second}") is None
    assert _path_text("file://evil.test/etc/passwd") is None
    assert _path_text("ordinary clipboard text") is None


def test_capture_path_text_attaches_instead_of_inserting(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "claim.pdf"
    source.write_bytes(b"claim")
    manager = _manager(monkeypatch, tmp_path)

    result = capture_macos_clipboard(manager, pasted_text=source.as_uri())

    assert result == ClipboardPaste("attachments", ("claim.pdf",))
    assert (manager.staging_dir / "claim.pdf").read_bytes() == b"claim"


def test_capture_image_uses_attachment_manager(monkeypatch, tmp_path: Path) -> None:
    manager = _manager(monkeypatch, tmp_path)

    def fake_read(temp_dir: str):
        image = Path(temp_dir) / "clipboard-123.png"
        image.write_bytes(b"png")
        return {"kind": "image", "path": str(image)}

    monkeypatch.setattr("apodex.clipboard._read_macos_pasteboard", fake_read)
    result = capture_macos_clipboard(manager)

    assert result.attachments == ("clipboard-123.png",)
    assert (manager.staging_dir / "clipboard-123.png").read_bytes() == b"png"
    assert capture_macos_clipboard(manager).attachments == ("clipboard-123.png",)
    assert len(manager.list()) == 1


def test_broker_round_trip_supports_clipboard_and_pasted_text(
    monkeypatch, tmp_path: Path,
) -> None:
    manager = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "apodex.clipboard.capture_macos_clipboard",
        lambda _manager, pasted_text=None: ClipboardPaste(
            "text", text=pasted_text or "clipboard text",
        ),
    )
    try:
        broker = ClipboardBroker(manager)
    except PermissionError:
        pytest.skip("test sandbox does not allow loopback listeners")
    broker.start()
    monkeypatch.setenv(_BROKER_URL_ENV, f"http://127.0.0.1:{broker.port}")
    monkeypatch.setenv(_BROKER_TOKEN_ENV, broker.token)
    try:
        assert paste_from_clipboard(manager).text == "clipboard text"
        assert paste_from_clipboard(manager, pasted_text="pasted").text == "pasted"
    finally:
        broker.close()


def test_broker_never_resolves_request_text_as_a_host_path(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "host-secret.txt"
    source.write_text("host-only")
    manager = _manager(monkeypatch, tmp_path)
    try:
        broker = ClipboardBroker(manager)
    except PermissionError:
        pytest.skip("test sandbox does not allow loopback listeners")
    broker.start()
    monkeypatch.setenv(_BROKER_URL_ENV, f"http://127.0.0.1:{broker.port}")
    monkeypatch.setenv(_BROKER_TOKEN_ENV, broker.token)
    try:
        result = paste_from_clipboard(manager, pasted_text=str(source))
    finally:
        broker.close()

    assert result == ClipboardPaste("text", text=str(source))
    assert manager.list() == []
