from __future__ import annotations

import importlib
import json
import pathlib
from types import SimpleNamespace

import pytest

from plugins.tools.web_search import raw_web_search

download_module = importlib.import_module("plugins.tools.download_file")


def test_raw_web_search_remains_available_to_resource_manager() -> None:
    assert callable(raw_web_search)


@pytest.mark.asyncio
async def test_download_file_executes_bundled_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    sandbox = object()

    async def fake_get_sandbox() -> object:
        return sandbox

    async def fake_run(current: object, command: str, **kwargs: object) -> SimpleNamespace:
        captured.update(current=current, command=command, **kwargs)
        return SimpleNamespace(
            stdout=json.dumps({"status": "ok", "path": "/workspace/downloads/sample.txt"}),
            stderr="",
        )

    monkeypatch.setattr(download_module, "aget_sandbox", fake_get_sandbox)
    monkeypatch.setattr(download_module, "arun_sandbox_cmd", fake_run)

    result = json.loads(
        await download_module.download_file.ainvoke(
            {"url": "https://example.com/sample.txt", "path": "sample.txt"}
        )
    )

    assert result["status"] == "ok"
    assert captured["current"] is sandbox
    assert "https://example.com/sample.txt" in str(captured["command"])
    runner = captured["input"]
    assert isinstance(runner, str)
    assert "def main(" in runner
    assert "--max-bytes" in str(captured["command"])


# Literal IPs so the checks resolve locally — these tests never touch DNS.
_NON_PUBLIC_URLS = [
    "http://127.0.0.1:8080/admin",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://10.0.0.5/internal",
    "http://[::1]/",
    "http://user:pw@127.0.0.1/",
]


@pytest.mark.parametrize("url", _NON_PUBLIC_URLS)
async def test_web_fetch_refuses_non_public_targets(
    url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_fetch is auto-approved by the risk gate, so a URL taken from page
    content is requested with no human in the loop. Nothing may leave the
    process — not even the scrape provider's request."""
    web_fetch_module = importlib.import_module("plugins.tools.web_fetch")

    def no_transport(*args: object, **kwargs: object) -> None:
        raise AssertionError("a request was issued for a non-public URL")

    monkeypatch.setattr(web_fetch_module.httpx, "AsyncClient", no_transport)

    result = await web_fetch_module._fetch_one(url, "anything")

    assert result.startswith("[BLOCKED]")


async def test_private_fetch_opt_in_restores_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetching a localhost dev server stays possible, but only deliberately."""
    from plugins.tools._bounded_fetch import non_public_url_error

    assert await non_public_url_error("http://127.0.0.1:8080/") != ""

    monkeypatch.setenv("FRONTIER_AGENT_ALLOW_PRIVATE_FETCH", "1")

    assert await non_public_url_error("http://127.0.0.1:8080/") == ""


@pytest.mark.parametrize("url", _NON_PUBLIC_URLS)
async def test_aligned_web_fetch_refuses_non_public_targets(
    url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``web_fetch_impl: aligned`` in the shipped react profile means THIS is
    the implementation the tool named ``web_fetch`` actually runs, so guarding
    only ``plugins/tools/web_fetch.py`` left the shipped path open."""
    aligned = importlib.import_module("plugins.tools.web_fetch_aligned")

    def no_transport(*args: object, **kwargs: object) -> None:
        raise AssertionError("a request was issued for a non-public URL")

    monkeypatch.setattr(aligned.httpx, "AsyncClient", no_transport)

    result = await aligned._fetch_single(url, "anything")

    assert result.startswith("Blocked:")


def test_terminal_modes_only_select_guarded_web_fetch_implementations() -> None:
    """Pins the reachability this PR originally got wrong.

    ``--mode react`` resolves to a workflow profile that sets
    ``web_fetch_impl: aligned``, so the tool named ``web_fetch`` runs
    ``web_fetch_aligned``. Guarding only ``plugins/tools/web_fetch.py`` left that
    path open. Resolved through the profile loader rather than a hardcoded
    filename, so a profile rename cannot silently drop the coverage.
    """
    import yaml

    from apodex.profiles import get_profile, terminal_mode_names

    repo = pathlib.Path(__file__).resolve().parents[1]
    guarded = {"aligned", "original"}   # both implementations now vet the URL
    seen = set()
    for mode in terminal_mode_names():
        profile = get_profile(mode)
        workflow_dir = profile.workflow.replace("-", "_")
        path = repo / "workflows" / workflow_dir / "profiles" / f"{profile.workflow_profile}.yaml"
        agent_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["agent"]
        impl = agent_cfg.get("web_fetch_impl") or "original"
        assert impl in guarded, f"{mode} selects an unguarded web_fetch: {impl}"
        seen.add(impl)

    # If this stops holding, the aligned guard is no longer on a shipped path
    # and the test above has quietly stopped protecting anything.
    assert "aligned" in seen


async def test_redirect_hops_are_validated_not_followed_blindly() -> None:
    """A vetted public URL can still answer 302 → private, so each hop is
    re-vetted instead of letting httpx follow the chain."""
    import httpx

    from plugins.tools._bounded_fetch import RedirectRefused, next_hop

    private = httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data/"},
    )
    with pytest.raises(RedirectRefused):
        await next_hop(private, "https://example.com/start")

    # A public hop is returned for the caller to follow, and a final response
    # reports no hop at all.
    public = httpx.Response(302, headers={"location": "/moved"})
    assert await next_hop(public, "https://www.iana.org/start") == (
        "https://www.iana.org/moved"
    )
    assert await next_hop(httpx.Response(200), "https://www.iana.org/start") is None


async def test_cross_origin_redirect_drops_caller_credentials() -> None:
    """Hand-rolling the hop loop lost what httpx does for free: it strips
    authentication headers when a redirect changes origin. Without this the
    manual loop hands the caller's token to whatever host the first origin
    names."""
    from plugins.tools._bounded_fetch import strip_cross_origin_credentials

    creds = {
        "Authorization": "Bearer SECRET",
        "Cookie": "session=SECRET",
        "User-Agent": "keep-me",
    }

    crossed = strip_cross_origin_credentials(
        creds, "https://a.example.com/1", "https://b.example.com/2",
    )
    same = strip_cross_origin_credentials(
        creds, "https://a.example.com/1", "https://a.example.com/2",
    )

    assert crossed == {"User-Agent": "keep-me"}
    assert same == creds                      # same-origin hops stay authenticated
    # A port or scheme change is also a different origin.
    assert "Authorization" not in strip_cross_origin_credentials(
        creds, "https://a.example.com/1", "https://a.example.com:8443/2",
    )
    assert "Authorization" not in strip_cross_origin_credentials(
        creds, "https://a.example.com/1", "http://a.example.com/2",
    )


async def test_request_is_pinned_to_the_validated_address() -> None:
    """Validating a name and letting the client resolve it again is a TOCTOU:
    an attacker-controlled resolver answers public for the check and private for
    the connection. The socket must go where the check looked."""
    from plugins.tools._bounded_fetch import pin_to_address

    url, headers, extensions = pin_to_address(
        "https://example.com/a?b=1", ("93.184.216.34",), {"User-Agent": "x"},
    )

    assert url == "https://93.184.216.34/a?b=1"     # no second DNS lookup possible
    assert headers["Host"] == "example.com"          # virtual-host routing preserved
    # Drives TLS SNI *and* the certificate hostname check, so a pinned HTTPS
    # request still fails closed on a mismatched certificate.
    assert extensions == {"sni_hostname": "example.com"}
    assert headers["User-Agent"] == "x"

    # Nothing to pin: the private-fetch opt-in returns no addresses.
    assert pin_to_address("https://example.com/", (), {}) == (
        "https://example.com/", {}, {},
    )


async def test_dns_rebinding_cannot_reach_a_private_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a resolver that answers public once, then private."""
    import socket

    from plugins.tools._bounded_fetch import pin_to_address, vet_public_url

    real = socket.getaddrinfo
    calls = {"n": 0}

    def rebinding(host: str, port: object, *args: object, **kwargs: object) -> object:
        if host != "rebind.test":
            return real(host, port, *args, **kwargs)
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)

    refusal, addresses = await vet_public_url("http://rebind.test/secret")
    dial_url, headers, _ext = pin_to_address(
        "http://rebind.test/secret", addresses, {},
    )

    assert refusal == ""
    assert addresses == ("93.184.216.34",)
    assert dial_url == "http://93.184.216.34/secret"   # not the rebound private IP
    assert headers["Host"] == "rebind.test"
    assert calls["n"] == 1                             # resolved once, then pinned
