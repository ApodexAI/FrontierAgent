"""macOS clipboard capture and the Docker host bridge.

Terminal paste protocols carry text only.  On macOS the outer launcher reads
NSPasteboard on behalf of the containerized TUI, then feeds the existing
session attachment manager.  The HTTP bridge accepts no commands or output
paths and is protected by a per-process bearer token.
"""

from __future__ import annotations

import filecmp
import json
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from apodex.attachments import AttachmentManager

_BROKER_URL_ENV = "APODEX_CLIPBOARD_BROKER_URL"
_BROKER_TOKEN_ENV = "APODEX_CLIPBOARD_BROKER_TOKEN"
_MAX_REQUEST_BYTES = 256_000


class ClipboardError(RuntimeError):
    """A user-facing clipboard or bridge failure."""


@dataclass(frozen=True)
class ClipboardPaste:
    kind: str
    attachments: tuple[str, ...] = ()
    text: str = ""
    message: str = ""


_JXA_READ_PASTEBOARD = r"""
ObjC.import('AppKit');
ObjC.import('Foundation');

function unwrap(value) {
    if (!value) return '';
    return ObjC.unwrap(value);
}

function run(argv) {
    const pb = $.NSPasteboard.generalPasteboard;
    const items = pb.pasteboardItems;
    const paths = [];
    if (items) {
        for (let i = 0; i < items.count; i++) {
            const raw = items.objectAtIndex(i).stringForType('public.file-url');
            if (!raw) continue;
            const url = $.NSURL.URLWithString(raw);
            if (url && url.isFileURL) paths.push(unwrap(url.path));
        }
    }
    if (paths.length) return JSON.stringify({kind: 'paths', paths: paths});

    const imageTypes = [
        ['public.png', 'png'],
        ['public.jpeg', 'jpg'],
        ['public.tiff', 'tiff']
    ];
    for (const pair of imageTypes) {
        const data = pb.dataForType(pair[0]);
        if (!data || data.length === 0) continue;
        const name = 'clipboard-' + Date.now() + '.' + pair[1];
        const path = $(argv[0]).stringByAppendingPathComponent(name);
        const written = $.NSFileManager.defaultManager
            .createFileAtPathContentsAttributes(path, data, $.NSDictionary.dictionary);
        if (!written) {
            return JSON.stringify({kind: 'error', message: 'could not save clipboard image'});
        }
        return JSON.stringify({kind: 'image', path: unwrap(path)});
    }

    let text = unwrap(pb.stringForType('public.utf8-plain-text'));
    if (!text) text = unwrap(pb.stringForType('public.plain-text'));
    return JSON.stringify(text ? {kind: 'text', text: text} : {kind: 'empty'});
}
"""


def _read_macos_pasteboard(temp_dir: str) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise ClipboardError("clipboard attachments are currently supported on macOS only")
    try:
        result = subprocess.run(
            [
                "/usr/bin/osascript", "-l", "JavaScript",
                "-e", _JXA_READ_PASTEBOARD, temp_dir,
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClipboardError(f"could not read the macOS clipboard: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "osascript failed").strip()
        raise ClipboardError(f"could not read the macOS clipboard: {detail}")
    try:
        payload = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise ClipboardError("macOS clipboard returned an invalid response") from exc
    if not isinstance(payload, dict):
        raise ClipboardError("macOS clipboard returned an invalid response")
    return payload


def _path_text(text: str) -> list[str] | None:
    """Return absolute existing paths represented by clipboard text."""
    raw = text.strip()
    if not raw:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and all(line.startswith("file://") for line in lines):
        candidates = []
        for line in lines:
            parsed = urlparse(line)
            if parsed.scheme == "file":
                candidates.append(unquote(parsed.path))
    elif len(lines) > 1 and all(
        Path(line.strip("'\"")).expanduser().is_absolute()
        and Path(line.strip("'\"")).expanduser().exists()
        for line in lines
    ):
        candidates = [line.strip("'\"") for line in lines]
    else:
        try:
            candidates = shlex.split(raw)
        except ValueError:
            candidates = [line.strip() for line in raw.splitlines() if line.strip()]
        # A single unescaped path may contain spaces. Prefer it when it exists.
        if Path(raw).expanduser().is_absolute() and Path(raw).expanduser().exists():
            candidates = [raw]
    if not candidates:
        return None
    resolved: list[str] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute() or not path.exists():
            return None
        resolved.append(str(path.resolve()))
    return resolved


def capture_macos_clipboard(
    manager: AttachmentManager, *, pasted_text: str | None = None,
) -> ClipboardPaste:
    """Capture Finder files, an image, a path string, or ordinary text."""
    if pasted_text is not None:
        paths = _path_text(pasted_text)
        if paths is None:
            return ClipboardPaste("text", text=pasted_text)
        added = manager.attach_many(paths)
        return ClipboardPaste("attachments", tuple(item.relative_path for item in added))

    with tempfile.TemporaryDirectory(prefix="apodex-clipboard-") as temp_dir:
        payload = _read_macos_pasteboard(temp_dir)
        kind = str(payload.get("kind") or "")
        if kind == "paths":
            raw_paths = payload.get("paths")
            if not isinstance(raw_paths, list) or not all(isinstance(p, str) for p in raw_paths):
                raise ClipboardError("macOS clipboard returned invalid file paths")
            added = manager.attach_many(raw_paths)
            return ClipboardPaste("attachments", tuple(item.relative_path for item in added))
        if kind == "image":
            raw_path = str(payload.get("path") or "")
            image = Path(raw_path).resolve()
            temp_root = Path(temp_dir).resolve()
            if temp_root not in image.parents or not image.is_file():
                raise ClipboardError("macOS clipboard returned an invalid image")
            for item in manager.list():
                name = Path(item.relative_path).name
                staged = manager.staging_dir / item.relative_path
                try:
                    if name.startswith("clipboard-") and filecmp.cmp(
                        image, staged, shallow=False,
                    ):
                        return ClipboardPaste("attachments", (item.relative_path,))
                except OSError:
                    continue
            added = manager.attach(str(image))
            return ClipboardPaste("attachments", tuple(item.relative_path for item in added))
        if kind == "text":
            text = str(payload.get("text") or "")
            paths = _path_text(text)
            if paths is not None:
                added = manager.attach_many(paths)
                return ClipboardPaste("attachments", tuple(item.relative_path for item in added))
            return ClipboardPaste("text", text=text)
        if kind == "empty":
            return ClipboardPaste("empty", message="clipboard is empty or unsupported")
        raise ClipboardError(str(payload.get("message") or "unsupported clipboard content"))


class ClipboardBroker:
    """Loopback-only host service used by the macOS Docker TUI."""

    def __init__(self, manager: AttachmentManager) -> None:
        self.manager = manager
        self.token = secrets.token_urlsafe(32)
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/paste" or self.headers.get("Authorization") != f"Bearer {broker.token}":
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400)
                    return
                if length < 0 or length > _MAX_REQUEST_BYTES:
                    self.send_error(413)
                    return
                try:
                    request = json.loads(self.rfile.read(length) or b"{}")
                    pasted_text = request.get("text") if isinstance(request, dict) else None
                    if pasted_text is not None and not isinstance(pasted_text, str):
                        raise ClipboardError("invalid pasted text")
                    response = capture_macos_clipboard(
                        broker.manager, pasted_text=pasted_text,
                    )
                    body = json.dumps(asdict(response)).encode()
                    self.send_response(200)
                except Exception as exc:  # keep the broker alive after one bad paste
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                # Silences the default stderr access log. Parameter names match
                # BaseHTTPRequestHandler.log_message exactly — renaming them
                # breaks the override for keyword callers.
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="apodex-clipboard", daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def paste_from_clipboard(
    manager: AttachmentManager, *, pasted_text: str | None = None,
) -> ClipboardPaste:
    """Use the Docker host broker when configured, else read macOS directly."""
    url = os.environ.get(_BROKER_URL_ENV, "").strip()
    token = os.environ.get(_BROKER_TOKEN_ENV, "").strip()
    if not url:
        return capture_macos_clipboard(manager, pasted_text=pasted_text)
    body = json.dumps({} if pasted_text is None else {"text": pasted_text}).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/paste", data=body, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ClipboardError(f"clipboard broker is unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClipboardError("clipboard broker returned an invalid response")
    if payload.get("error"):
        raise ClipboardError(str(payload["error"]))
    return ClipboardPaste(
        kind=str(payload.get("kind") or "empty"),
        attachments=tuple(str(item) for item in payload.get("attachments") or ()),
        text=str(payload.get("text") or ""),
        message=str(payload.get("message") or ""),
    )
