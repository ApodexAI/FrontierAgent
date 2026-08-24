"""Bounded HTTP body reads for the web_fetch tool family."""

from __future__ import annotations

import asyncio
import os

import httpx

# Byte ceiling for one response body. Env-tunable for callers that
# legitimately need bigger text payloads.
DEFAULT_MAX_FETCH_BYTES = 5 * 1024 * 1024


def max_fetch_bytes() -> int:
    raw = (os.getenv("WEB_FETCH_MAX_BYTES") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_MAX_FETCH_BYTES


# Content-types that are data blobs, never extractable page text. PDF is
# deliberately absent: Jina converts PDFs to text, so blocking it here
# would regress the primary scrape path. Prefix match on the bare type
# (parameters like ``; charset=`` stripped by the caller helper).
_BINARY_TYPE_PREFIXES = ("image/", "audio/", "video/", "font/")
_BINARY_TYPES = frozenset({
    "application/zip",
    "application/octet-stream",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/x-hdf",
    "application/x-hdf5",
    "application/vnd.ms-cab-compressed",
    "application/x-matlab-data",
    "application/x-msdownload",
    "application/wasm",
})


# URL path extensions that denote dataset / archive / binary downloads.
# Screened BEFORE any request is issued (no Jina call, no bytes on the
# wire). Text-ish data formats (.csv/.json/.txt/.xml) are deliberately
# absent: small ones are legitimate fetch targets and the byte cap
# bounds the large ones. PDF stays allowed (Jina converts it).
_BLOCKED_URL_EXTENSIONS = (
    ".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz",
    ".gz", ".bz2", ".xz", ".7z", ".rar", ".zst",
    ".mat", ".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle",
    ".pt", ".pth", ".onnx", ".safetensors", ".parquet", ".feather",
    ".whl", ".deb", ".rpm", ".dmg", ".iso", ".exe", ".msi", ".apk",
)


def blocked_download_url(url: str) -> str | None:
    """The matched extension when ``url``'s path names a dataset/archive
    download, else ``None``. Match is on the URL *path* (query/fragment
    stripped), so ``?format=zip`` params don't false-positive."""
    try:
        from urllib.parse import unquote, urlsplit

        path = unquote(urlsplit(url).path).strip().lower()
    except Exception:
        return None
    for ext in _BLOCKED_URL_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return None


def binary_content_type(content_type: str | None) -> str | None:
    """The normalized content-type when it denotes a non-text blob, else
    ``None``. Callers use the returned value in the error message."""
    if not content_type:
        return None
    bare = content_type.split(";", 1)[0].strip().lower()
    if bare in _BINARY_TYPES or bare.startswith(_BINARY_TYPE_PREFIXES):
        return bare
    return None


async def read_bounded(
    response: httpx.Response,
    max_bytes: int | None = None,
) -> tuple[bytes, bool]:
    """Read a streaming response up to ``max_bytes``; ``(body, truncated)``.

    Must be called inside the ``client.stream(...)`` context. Stopping
    early closes the connection, so a 2GB download costs at most
    ``max_bytes`` of transfer and memory.
    """
    cap = max_bytes if max_bytes is not None else max_fetch_bytes()
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= cap:
            return b"".join(chunks)[:cap], True
    return b"".join(chunks), False


def decode_body(response: httpx.Response, body: bytes) -> str:
    """Decode a bounded body with the response's declared charset.

    ``errors="replace"`` because a truncated multi-byte sequence at the
    cap boundary must not raise. No chardet sniffing on the (possibly
    huge) body — absent/unknown charset falls back to UTF-8, matching
    the dominant real-world default.
    """
    encoding = response.charset_encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


async def non_public_url_error(url: str) -> str:
    """Reason *url* must not be fetched, else ``""``.

    ``web_fetch`` is auto-approved in the terminal's risk gate, so a URL that
    arrives from page content — the prompt-injection path for a research agent
    — is requested without a human ever seeing it. Unguarded, the target may be
    a cloud metadata endpoint or any service on the deployment's private
    network, and with ``JINA_API_KEY`` set the internal URL is handed to the
    scrape provider before the fetch is even attempted.

    Reuses ``download_file``'s vetting so both tools share one definition of
    "public": http(s) only, no credentials in the URL, and EVERY resolved
    address global, so a split-horizon DNS answer cannot slip a private address
    through. Fail-closed — a name that cannot be resolved cannot be vetted.
    Resolution is off-loaded because it blocks.

    Set ``FRONTIER_AGENT_ALLOW_PRIVATE_FETCH=1`` to fetch a localhost or
    intranet service deliberately.
    """
    refusal, _addresses = await vet_public_url(url)
    return refusal


async def vet_public_url(url: str) -> tuple[str, tuple[str, ...]]:
    """``(refusal, validated_addresses)`` for *url*.

    Returning the addresses is what makes DNS-rebinding defence possible:
    validating a name and then letting the client resolve it a second time is a
    TOCTOU — an attacker-controlled resolver can answer with a public address
    for the check and a private one for the connection. Callers hand these
    addresses to :func:`pin_to_address` so the socket goes where the check
    looked. ``download_file`` has always pinned for this reason; the scrape
    paths now do too.
    """
    if (os.getenv("FRONTIER_AGENT_ALLOW_PRIVATE_FETCH") or "").strip() == "1":
        return "", ()
    from plugins.tools._download_runner import DownloadError, _validate_public_url
    try:
        addresses = await asyncio.to_thread(_validate_public_url, url)
    except DownloadError as exc:
        return str(exc), ()
    if isinstance(addresses, str):   # older single-address contract
        addresses = (addresses,)
    return "", tuple(addresses)


#: Headers that authenticate the caller and must not follow a redirect to a
#: different origin. httpx strips these itself when it follows redirects; a
#: hand-rolled hop loop has to do it explicitly or it leaks the credential to
#: whatever host the first origin names.
_CREDENTIAL_HEADERS = frozenset({
    "authorization", "cookie", "proxy-authorization", "www-authenticate",
})


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = httpx.URL(url)
    return (parts.scheme, parts.host, parts.port)


def strip_cross_origin_credentials(
    headers: dict[str, str], from_url: str, to_url: str,
) -> dict[str, str]:
    """Drop caller credentials when a redirect hop changes origin.

    Same-origin hops keep them, so an authenticated fetch that redirects within
    one host still works.
    """
    try:
        if _origin(from_url) == _origin(to_url):
            return headers
    except Exception:
        pass
    return {
        name: value for name, value in headers.items()
        if name.lower() not in _CREDENTIAL_HEADERS
    }


def pin_to_address(
    url: str, addresses: tuple[str, ...], headers: dict[str, str],
) -> tuple[str, dict[str, str], dict[str, object]]:
    """Rewrite a request to dial an already-validated address.

    Returns ``(url, headers, extensions)``. The address replaces the URL host so
    no second DNS lookup can happen, while the original hostname is preserved
    twice over: in the ``Host`` header (virtual-host routing) and in the
    ``sni_hostname`` extension, which drives TLS SNI *and* the certificate
    hostname check — so a pinned HTTPS request still fails closed on a
    mismatched certificate.

    A no-op when there is nothing to pin (the private-fetch opt-in returns no
    addresses) or the URL is already literal-IP.
    """
    if not addresses:
        return url, headers, {}
    parsed = httpx.URL(url)
    hostname = parsed.host
    if not hostname or hostname == addresses[0]:
        return url, headers, {}
    pinned = str(parsed.copy_with(host=addresses[0]))
    return (
        pinned,
        {**headers, "Host": parsed.netloc.decode("ascii")},
        {"sni_hostname": hostname},
    )


#: Redirect hops a scrape may follow. Matches httpx's own default ceiling.
MAX_REDIRECT_HOPS = 20


class RedirectRefused(Exception):
    """A redirect hop pointed somewhere ``non_public_url_error`` refuses."""


async def next_hop(response: httpx.Response, current_url: str) -> str | None:
    """The vetted URL a 30x response redirects to, or ``None`` if it is final.

    Automatic redirect following defeats the URL guard: only the FIRST URL is
    vetted, so a public attacker-controlled page can answer 302 → localhost or
    a cloud metadata endpoint and the client follows it. Callers therefore
    disable ``follow_redirects`` and walk the chain through this, which vets
    every hop with the same rule the initial URL passed.

    Raises :class:`RedirectRefused` rather than returning the reason, so a
    refused hop cannot be mistaken for "no more redirects" and silently treated
    as a successful fetch.
    """
    if not response.is_redirect:
        return None
    location = response.headers.get("location", "").strip()
    if not location:
        return None
    target = str(httpx.URL(current_url).join(location))
    refusal = await non_public_url_error(target)
    if refusal:
        raise RedirectRefused(refusal)
    return target


__all__ = [
    "DEFAULT_MAX_FETCH_BYTES",
    "MAX_REDIRECT_HOPS",
    "RedirectRefused",
    "binary_content_type",
    "blocked_download_url",
    "decode_body",
    "max_fetch_bytes",
    "next_hop",
    "non_public_url_error",
    "pin_to_address",
    "read_bounded",
    "strip_cross_origin_credentials",
    "vet_public_url",
]
