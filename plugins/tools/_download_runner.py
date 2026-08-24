"""Sandbox-side implementation for the controlled ``download_file`` tool."""

from __future__ import annotations

import argparse
import contextlib
import email.message
import fcntl
import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

_LOCK_NAME = ".frontier_agent-download.lock"
# Disk head start demanded before a response with no Content-Length starts
# streaming, and how often free space is re-checked while it grows.
_UNKNOWN_LENGTH_UPFRONT = 32 * 1024 * 1024
_DISK_RECHECK_INTERVAL = 32 * 1024 * 1024
_MAX_FILENAME = 180
# ``<basename>.part-<pid>x<random>-<reserved bytes>``. The reservation is
# encoded in the name so a concurrent download can charge in-flight bytes
# against the task quota without holding the lock for the whole transfer.
_PART_RE = re.compile(r"\.part-\d+x[0-9a-f]{8}-(\d+)$")

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_EXTENSIONS = frozenset({
    ".csv", ".doc", ".docx", ".epub", ".gif", ".gz", ".htm", ".html",
    ".jpeg", ".jpg", ".json", ".md", ".odp", ".ods", ".odt", ".pdf",
    ".png", ".ppt", ".pptx", ".rtf", ".svg", ".tar", ".text", ".tif",
    ".tiff", ".tsv", ".txt", ".webp", ".xls", ".xlsx", ".xml", ".zip",
})
_ALLOWED_CONTENT_TYPES = (
    "application/json",
    "application/msword",
    "application/pdf",
    "application/rtf",
    "application/vnd.ms-",
    "application/vnd.oasis.opendocument.",
    "application/vnd.openxmlformats-officedocument.",
    "application/x-gzip",
    "application/xml",
    "application/zip",
    "image/",
    "text/",
)
_CONTENT_DISPOSITION_NAME_RE = re.compile(
    r"""filename\*?=(?:UTF-8''|["']?)([^;"']+)""",
    re.IGNORECASE,
)


class DownloadError(RuntimeError):
    """Expected, user-actionable download failure."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: email.message.Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def _pinned_create_connection(ip: str) -> Callable[..., socket.socket]:
    """``HTTPConnection._create_connection`` that ignores the resolved name.

    Validating a hostname and then letting the client resolve it again is a
    TOCTOU: a name the attacker controls can answer with a public address for
    the check and ``169.254.169.254`` for the connection (DNS rebinding).
    Dialling the address we already vetted removes the second lookup. The
    hostname still drives the ``Host`` header and TLS SNI/cert validation
    because only the socket-level address is substituted.
    """
    def _create(
        address: tuple[str, int],
        timeout: float | None = socket._GLOBAL_DEFAULT_TIMEOUT,  # pyright: ignore[reportAttributeAccessIssue]
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        _host, port = address
        return socket.create_connection((ip, port), timeout, source_address)
    return _create


def _opener(ip: str) -> urllib.request.OpenerDirector:
    """Build an opener pinned to one already-validated IP.

    Proxy variables are ignored: they can route an otherwise-public URL
    through an internal endpoint and make SSRF validation meaningless (and
    would also defeat the pinning).
    """
    class _PinnedHTTP(http.client.HTTPConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = _pinned_create_connection(ip)

    class _PinnedHTTPS(http.client.HTTPSConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = _pinned_create_connection(ip)

    class _PinnedHTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req: urllib.request.Request) -> Any:
            return self.do_open(_PinnedHTTP, req)

    class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req: urllib.request.Request) -> Any:
            context = self._context  # pyright: ignore[reportAttributeAccessIssue]
            return self.do_open(_PinnedHTTPS, req, context=context)

    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedHTTPHandler(),
        _PinnedHTTPSHandler(),
        _NoRedirect(),
    )


def _validate_public_url(url: str) -> tuple[str, ...]:
    """Vet *url* and return every public IP the request may be sent to.

    Every resolved address must be global — retaining only the usable-looking
    answers would let a split-horizon answer through. IPv4 is attempted first
    because task containers commonly have no IPv6 route; all other vetted
    addresses remain available for connection failover.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError("only http:// and https:// URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise DownloadError("URL must have a public hostname and must not contain credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise DownloadError(f"invalid URL port: {exc}") from exc
    try:
        resolved = socket.getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise DownloadError(f"cannot resolve {parsed.hostname}: {exc}") from exc
    addresses = tuple(dict.fromkeys(str(item[4][0]) for item in resolved))
    if not addresses:
        raise DownloadError(f"cannot resolve {parsed.hostname}")
    validated: list[tuple[int, str]] = []
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise DownloadError(
                f"refusing non-public address for {parsed.hostname}: {ip}",
            )
        validated.append((ip.version, str(ip)))
    return tuple(address for _version, address in sorted(validated, key=lambda item: item[0]))


def _open_once(
    url: str,
    *,
    method: str,
    timeout: float,
) -> Any:
    addresses = _validate_public_url(url)
    # Compatibility for callers/tests that patched the former single-address
    # helper contract while this sandbox-side module rolls forward.
    if isinstance(addresses, str):
        addresses = (addresses,)
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "User-Agent": "FrontierAgent-download/1.0",
        },
    )
    last_error: Exception | None = None
    for address in addresses:
        opener = _opener(address)
        try:
            return opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # With redirects disabled urllib represents 3xx as HTTPError, but
            # it still carries the headers needed to validate the next hop.
            if exc.code in _REDIRECT_CODES:
                return exc
            raise DownloadError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Connection failures are address-specific. Keep the DNS answer
            # pinned, but try the next already-vetted address like urllib/curl.
            last_error = exc
    raise DownloadError(f"request failed for {url}: {last_error}") from last_error


def _open_following_redirects(
    url: str,
    *,
    method: str,
    timeout: float,
    max_redirects: int,
) -> Any:
    current = url
    for redirect_count in range(max_redirects + 1):
        # Each hop is re-validated *and* re-pinned; a redirect is just another
        # attacker-controlled URL.
        response = _open_once(current, method=method, timeout=timeout)
        if response.getcode() not in _REDIRECT_CODES:
            return response, current, redirect_count
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise DownloadError("redirect response is missing a Location header")
        if redirect_count >= max_redirects:
            raise DownloadError(f"too many redirects (maximum {max_redirects})")
        current = urllib.parse.urljoin(current, location)
    raise DownloadError(f"too many redirects (maximum {max_redirects})")


def _declared_size(headers: email.message.Message) -> int | None:
    raw = (headers.get("Content-Length") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _preflight(
    url: str,
    *,
    timeout: float,
    max_redirects: int,
) -> int | None:
    """Advisory Content-Length probe. Never authoritative — see ``download``."""
    try:
        response, _final_url, _ = _open_following_redirects(
            url,
            method="HEAD",
            timeout=timeout,
            max_redirects=max_redirects,
        )
    except DownloadError:
        # HEAD is advisory. Many object stores reject it even though GET works;
        # the streaming byte counter remains the authoritative hard boundary.
        return None
    try:
        return _declared_size(response.headers)
    finally:
        response.close()


def _header_filename(headers: email.message.Message) -> str:
    disposition = headers.get("Content-Disposition") or ""
    match = _CONTENT_DISPOSITION_NAME_RE.search(disposition)
    if not match:
        return ""
    return urllib.parse.unquote(match.group(1)).strip()


def _safe_filename(candidate: str) -> str:
    name = Path(candidate.replace("\\", "/")).name.strip().strip(".")
    name = re.sub(r"[^A-Za-z0-9._() -]+", "_", name)
    if not name:
        return "download"
    if len(name) <= _MAX_FILENAME:
        return name
    # Truncate the stem, never the extension: the suffix drives both the
    # allowed-type check and the magic-byte check, so losing it to a long
    # URL filename would silently skip signature validation.
    suffix = Path(name).suffix
    if len(suffix) > 16:  # not a real extension, just a dotted filename
        suffix = ""
    stem = name[: len(name) - len(suffix)]
    return (stem[: _MAX_FILENAME - len(suffix)] or "download") + suffix


def _download_root(dir_arg: str) -> Path:
    """Resolve the controlled download directory.

    The caller passes the task workspace explicitly (``--dir``, derived from
    ``resolve_mount_dirs()`` so a relocated ``FRONTIER_AGENT_WORKSPACE_DIR`` is
    honoured). The env var and the literal default only matter when the runner
    is executed directly, e.g. in tests.
    """
    raw = (
        dir_arg.strip()
        or os.environ.get("FRONTIER_AGENT_DOWNLOAD_DIR", "").strip()
        or "/workspace/downloads"
    )
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve_basename(
    path_arg: str, final_url: str, headers: email.message.Message,
) -> str:
    """Collapse the requested destination to a safe basename.

    Downloads always land in the controlled directory, never at a caller-chosen
    path: they are task inputs, not ``/outputs`` deliverables.
    """
    if path_arg:
        candidate = Path(path_arg.replace("\\", "/")).name
        if candidate in {"", ".", ".."} or path_arg.endswith("/"):
            candidate = "download"
    else:
        from_url = Path(urllib.parse.unquote(urllib.parse.urlsplit(final_url).path)).name
        candidate = _header_filename(headers) or from_url or "download"
    return _safe_filename(candidate)


def _unique_target(root: Path, basename: str) -> Path:
    """Pick a free destination name. Must be called under ``_download_lock``:
    the answer is only valid until another download publishes.
    """
    target = root / basename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(1, 10_000):
        candidate = root / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise DownloadError("could not allocate a unique destination filename")


def _validate_file_type(basename: str, headers: email.message.Message) -> str:
    content_type = (headers.get_content_type() or "").lower()
    suffix = Path(basename).suffix.lower()
    if suffix in _ALLOWED_EXTENSIONS:
        return content_type
    if any(content_type == prefix or content_type.startswith(prefix)
           for prefix in _ALLOWED_CONTENT_TYPES):
        return content_type
    raise DownloadError(
        f"unsupported download type: extension={suffix or '(none)'}, "
        f"content-type={content_type or '(missing)'}",
    )


def _validate_magic(path: Path, basename: str) -> None:
    """Reject common disguised payloads after the bounded body is on disk."""
    with open(path, "rb") as source:
        head = source.read(512)
    suffix = Path(basename).suffix.lower()
    signatures: dict[str, tuple[bytes, ...]] = {
        ".pdf": (b"%PDF-",),
        ".zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
        ".docx": (b"PK\x03\x04",),
        ".xlsx": (b"PK\x03\x04",),
        ".pptx": (b"PK\x03\x04",),
        ".odt": (b"PK\x03\x04",),
        ".ods": (b"PK\x03\x04",),
        ".odp": (b"PK\x03\x04",),
        ".epub": (b"PK\x03\x04",),
        ".gz": (b"\x1f\x8b",),
        ".doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
        ".xls": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
        ".ppt": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".gif": (b"GIF87a", b"GIF89a"),
    }
    expected = signatures.get(suffix)
    if expected and not any(head.startswith(signature) for signature in expected):
        raise DownloadError(
            f"downloaded bytes do not match the expected {suffix} file signature",
        )
    # RIFF alone is also AVI/WAV; the format tag at offset 8 is what makes it
    # a WebP.
    if suffix == ".webp" and not (
        head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    ):
        raise DownloadError("downloaded bytes do not match the expected .webp file signature")
    # A tar's magic lives at offset 257, so anything shorter cannot be one —
    # requiring the full header instead of skipping the check on short reads.
    if suffix == ".tar" and head[257:262] != b"ustar":
        raise DownloadError("downloaded bytes do not match the expected .tar file signature")


def _accounted_bytes(root: Path, *, stale_after: float) -> int:
    """Bytes charged against the task quota: published files plus the amount
    reserved by downloads still streaming in other processes.

    Charging the *reservation* rather than the current on-disk size is what
    lets the transfer itself run outside the lock: two concurrent downloads
    cannot jointly overrun the task quota even though neither has finished.
    A ``.part`` whose mtime has stopped advancing past ``stale_after`` belonged
    to a killed process; it is ignored so an abandoned transfer cannot pin the
    quota for the rest of the task. It is deliberately not unlinked — a slow
    but live download would then fail its publish step with a confusing error.
    """
    total = 0
    now = time.time()
    for item in root.iterdir():
        if item.name == _LOCK_NAME:
            continue
        try:
            if not item.is_file():
                continue
            stat = item.stat()
            match = _PART_RE.search(item.name)
            if match is None:
                total += stat.st_size
                continue
            if now - stat.st_mtime > stale_after:
                continue
            total += max(int(match.group(1)), stat.st_size)
        except OSError:
            continue
    return total


@contextlib.contextmanager
def _download_lock(root: Path) -> Iterator[None]:
    """Serialize quota accounting and publication within one task workspace.

    Held only for the two short critical sections (reserve, then publish) —
    never across the transfer, so parallel sub-agent downloads overlap instead
    of queueing into the caller's wall-clock timeout.
    """
    lock_path = root / _LOCK_NAME
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _grow_reservation(
    root: Path,
    part: Path,
    *,
    current: int,
    requested: int,
    task_max_bytes: int,
    stale_after: float,
) -> tuple[Path, int]:
    """Atomically increase an in-flight unknown-length reservation."""
    if requested <= current:
        return part, current
    with _download_lock(root):
        used = _accounted_bytes(root, stale_after=stale_after)
        try:
            own_size = part.stat().st_size
        except OSError as exc:
            raise DownloadError(f"download reservation disappeared: {part.name}") from exc
        used_without_self = max(0, used - max(current, own_size))
        available = task_max_bytes - used_without_self
        if requested > available:
            raise DownloadError(
                f"download exceeded the remaining task download quota "
                f"of {available} bytes",
            )
        match = _PART_RE.search(part.name)
        if match is None:
            raise DownloadError(f"invalid download reservation: {part.name}")
        renamed = part.with_name(
            f"{part.name[:match.start(1)]}{requested}{part.name[match.end(1):]}",
        )
        os.replace(part, renamed)
        return renamed, requested


def _provenance_url(url: str) -> str:
    """Keep stable provenance without leaking signed query parameters."""
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        "",
        "",
    ))


def download(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    declared = _preflight(
        args.url,
        timeout=args.connect_timeout,
        max_redirects=args.max_redirects,
    )
    if declared is not None and declared > args.max_bytes:
        raise DownloadError(
            f"declared size {declared} bytes exceeds the per-file limit "
            f"of {args.max_bytes} bytes",
        )

    response, final_url, redirect_count = _open_following_redirects(
        args.url,
        method="GET",
        timeout=args.read_timeout,
        max_redirects=args.max_redirects,
    )
    target: Path | None = None
    part: Path | None = None
    try:
        declared = _declared_size(response.headers)
        if declared is not None and declared > args.max_bytes:
            raise DownloadError(
                f"declared size {declared} bytes exceeds the per-file limit "
                f"of {args.max_bytes} bytes",
            )
        root = _download_root(args.dir)
        basename = _resolve_basename(args.path, final_url, response.headers)
        content_type = _validate_file_type(basename, response.headers)

        # ── critical section 1: charge the quota and reserve a .part slot ──
        with _download_lock(root):
            used = _accounted_bytes(root, stale_after=args.total_timeout + 60)
            if used >= args.task_max_bytes:
                raise DownloadError(
                    f"task download quota is exhausted ({used}/{args.task_max_bytes} bytes)",
                )
            remaining_task = args.task_max_bytes - used
            effective_limit = min(args.max_bytes, remaining_task)
            if declared is not None and declared > effective_limit:
                raise DownloadError(
                    f"declared size {declared} bytes exceeds the remaining task "
                    f"download quota of {effective_limit} bytes",
                )

            reservation = declared if declared is not None else min(
                effective_limit,
                _UNKNOWN_LENGTH_UPFRONT,
            )
            # Up-front disk check. With a declared length we can size it
            # exactly; without one (chunked responses are common) demanding
            # the whole per-file limit up front would reject a 200KB PDF on a
            # small workspace, so only a modest head start is required and the
            # transfer re-checks free space as it grows.
            upfront = reservation
            free = shutil.disk_usage(root).free
            required_free = upfront + args.reserve_bytes
            if free < required_free:
                raise DownloadError(
                    f"insufficient workspace disk: {free} bytes free, "
                    f"{required_free} bytes required including reserve",
                )

            token = f"{os.getpid()}x{uuid.uuid4().hex[:8]}"
            reserved = root / f"{basename}.part-{token}-{reservation}"
            # ``x`` mode, and ``part`` is only adopted once the create wins, so
            # a collision can never make the cleanup path unlink someone else's
            # in-flight transfer.
            output = open(reserved, "xb", buffering=args.chunk_bytes)  # noqa: SIM115
            part = reserved

        # ── transfer: outside the lock, so downloads run in parallel ──
        # Memory is bounded by ``chunk_bytes`` (the write buffer), not by any
        # checkpoint logic: each chunk is handed to the OS as it arrives.
        digest = hashlib.sha256()
        received = 0
        since_disk_check = 0
        stream_limit = min(effective_limit, declared) if declared is not None else effective_limit
        with output:
            while True:
                if time.monotonic() - started > args.total_timeout:
                    raise DownloadError(
                        f"download exceeded total timeout of {args.total_timeout:g}s",
                    )
                if declared is None and received + args.chunk_bytes > reservation:
                    requested = min(
                        effective_limit,
                        max(
                            received + args.chunk_bytes,
                            reservation + _UNKNOWN_LENGTH_UPFRONT,
                        ),
                    )
                    part, reservation = _grow_reservation(
                        root,
                        part,
                        current=reservation,
                        requested=requested,
                        task_max_bytes=args.task_max_bytes,
                        stale_after=args.total_timeout + 60,
                    )
                chunk = response.read(args.chunk_bytes)
                if not chunk:
                    break
                received += len(chunk)
                since_disk_check += len(chunk)
                if received > stream_limit:
                    raise DownloadError(
                        f"download exceeded the effective byte limit "
                        f"of {stream_limit} bytes",
                    )
                output.write(chunk)
                digest.update(chunk)
                if since_disk_check >= _DISK_RECHECK_INTERVAL:
                    since_disk_check = 0
                    output.flush()
                    free = shutil.disk_usage(root).free
                    if free < args.reserve_bytes:
                        raise DownloadError(
                            f"workspace disk reserve exhausted after {received} "
                            f"bytes: {free} bytes free, {args.reserve_bytes} reserved",
                        )
            output.flush()
            os.fsync(output.fileno())

        # ── critical section 2: resolve a free name and publish atomically ──
        # The name must be re-resolved here: any name picked before the
        # transfer could have been taken by a download that finished meanwhile,
        # and os.replace would silently clobber it.
        with _download_lock(root):
            target = _unique_target(root, basename)
            _validate_magic(part, basename)
            os.replace(part, target)
            part = None
        return {
            "status": "downloaded",
            "path": str(target),
            "size_bytes": received,
            "sha256": digest.hexdigest(),
            "content_type": content_type,
            "source_url": _provenance_url(args.url),
            "final_url": _provenance_url(final_url),
            "redirect_count": redirect_count,
        }
    finally:
        response.close()
        if part is not None:
            with contextlib.suppress(OSError):
                part.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--path", default="")
    parser.add_argument("--dir", default="")
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--task-max-bytes", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--reserve-bytes", type=int, required=True)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=60)
    parser.add_argument("--total-timeout", type=float, default=600)
    parser.add_argument("--max-redirects", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = download(args)
    except DownloadError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }))
        return 3
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
