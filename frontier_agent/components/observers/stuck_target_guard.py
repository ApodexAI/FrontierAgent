"""Detect and quarantine repeatedly failing network targets.

The guard tracks outcomes by host rather than exact tool-call text, while a
successful fetch clears the host's failure history.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, deque
from typing import Any

from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    ToolCallIntervention,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://([^/\s\"'>)\\]+)", re.IGNORECASE)
_BARE_HOST_RE = re.compile(
    r"(?<![@a-z0-9.-])((?:[a-z0-9-]+\.)+[a-z]{2,63})(?::\d+)?",
    re.IGNORECASE,
)
_HOST_TAIL_RE = re.compile(r"[^a-z0-9.\-].*\Z")
_FAILURE_LINE_RE = re.compile(
    r"(?:\A|\n)\s*(?:Info:\s*)?(?:"
    r"\[ERROR\]|\[NOT RENDERED\]|\[POSSIBLY NOT RENDERED\]|"
    r"\[ACCESS BLOCKED\]|\[BLOCKED\]|\[STUCK TARGET BLOCKED\]|"
    r"\[Exit code\s+-?\d+\]|Error:|Blocked:|URL skipped:|"
    r"Could not extract content from|Scraping failed:)",
    re.IGNORECASE,
)
_NO_RESULT_RE = re.compile(
    r"(?:\A|\n)\s*(?:Info:\s*)?(?:No results found|No results \(error or empty\))",
    re.IGNORECASE,
)
# ``[POSSIBLY NOT RENDERED]`` prefixes content the fetch tool decided to hand
# over anyway ("use it if it answers the question"). Treating that as grounds to
# cut a host off would quarantine any host whose pages are legitimately short —
# a small JSON API fetched a dozen times, for instance. It nudges; it never
# quarantines. ``[NOT RENDERED]`` (no content at all) stays a hard failure.
_SOFT_FAILURE_LINE_RE = re.compile(
    r"(?:\A|\n)\s*(?:Info:\s*)?\[POSSIBLY NOT RENDERED\]",
    re.IGNORECASE,
)
# ``bash`` is a shell: a curl/urllib attempt that was refused by the origin
# still exits 0 with the refusal printed on stdout, so tool-level success says
# nothing about whether the target was reached. Match common HTTP and transport
# failures in output even when the shell command exits successfully.
_HTTP_FAILURE_RE = re.compile(
    r"\bHTTP(?:/[\d.]+)?\s*(?:Error\s*)?(?:4\d\d|5\d\d)\b"      # HTTP 403 / HTTP/2 500
    r"|^\s*(?:4\d\d|5\d\d)\s*--"                                # "403 --  /styles.css"
    r"|\bHTTPError\b"                                           # urllib
    r"|\berror code:\s*\d+"                                     # Cloudflare 1010 …
    r"|\bcurl:\s*\(\d+\)"                                       # curl transport error
    r"|\b(?:403\s+)?Forbidden\b|\bUnauthorized\b|\bToo Many Requests\b"
    r"|\bAccess (?:denied|Denied)\b|\bService Unavailable\b",
    re.IGNORECASE | re.MULTILINE,
)
# A raw markup/JS dump in shell output means the agent is scraping by hand. We
# cannot tell from here whether that HTML carried the content it wanted, so the
# attempt is NEUTRAL: it must not count as a failure, but it must not clear the
# host's failure history either (that is what made the guard inert).
_MARKUP_DUMP_RE = re.compile(r"<!doctype\s+html|<html\b|<script\b", re.IGNORECASE)
_NUMBERED_FETCH_BLOCK_RE = re.compile(
    r"^\[\d+\]\s+URL:\s*(?P<url>[^\n]+)\n"
    r"\s*Info:\s*(?P<info>.*?)"
    r"(?=^\[\d+\]\s+URL:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

_FAILED = "failed"          # hard failure: counts toward hint AND quarantine
_SOFT_FAILED = "soft"       # counts toward the hint only — see _result_verdict
_SUCCEEDED = "succeeded"
_NEUTRAL = "neutral"

# Only a fetch-class tool can vouch for a host. A search engine answering with a
# snippet about ``example.com`` says nothing about whether that page can be read,
# and a shell command's exit status says nothing at all. Only ``web_fetch`` /
# ``download_file`` return content that has passed the content-level render
# check, so only they may clear a failure history.
#
# This leans on those tools labelling their own weak results
# (``[NOT RENDERED]`` / ``[POSSIBLY NOT RENDERED]`` — see
# ``plugins/tools/_render_check``); a fetch tool that silently handed back an app
# shell would still read as content here.
_VOUCHING_TOOL_HINTS = ("web_fetch", "download")

# ``.pdf`` / ``.json`` / … in a search query look exactly like a bare hostname.
# Without this, "read report.pdf" registers a host called ``report.pdf``.
_NON_TLD_SUFFIXES = frozenset({
    "json", "pdf", "md", "txt", "csv", "tsv", "xml", "html", "htm", "js",
    "css", "png", "jpg", "jpeg", "gif", "svg", "zip", "gz", "tar", "yaml",
    "yml", "toml", "ini", "log", "sh", "sql", "py", "ipynb", "xlsx", "docx",
    "pptx", "parquet", "db", "sqlite", "env", "lock", "cfg", "conf",
})
_NETWORK_TOOL_NAMES = frozenset({
    "web_fetch", "web_search", "scholar_search", "download_file", "bash",
})
# Hosts that are infrastructure for *reaching* a target rather than the target
# itself — counting them would blame the proxy instead of the page, and they
# legitimately recur across unrelated subtasks.
_TRANSPARENT_HOSTS = (
    "r.jina.ai", "google.serper.dev", "webcache.googleusercontent.com",
    "translate.goog", "corsproxy.io", "api.allorigins.win", "web.archive.org",
    "archive.org", "archive.ph", "archive.today", "index.commoncrawl.org",
)


def _is_transparent(host: str) -> bool:
    return any(host == item or host.endswith(f".{item}") for item in _TRANSPARENT_HOSTS)


def _hosts_in(value: Any, *, include_bare: bool = False) -> set[str]:
    """Every URL host mentioned anywhere in a tool-call argument structure.

    A reader/archive wrapper (``r.jina.ai/https://target/x``) yields BOTH hosts;
    the transparent one is dropped, so proxying an attempt still counts against
    the page it was aimed at.
    """
    hosts: set[str] = set()
    if isinstance(value, str):
        for raw in _URL_RE.findall(value):
            host = raw.split("@")[-1].split(":")[0].lower().removeprefix("www.")
            # Shell/format leftovers glued to the host (``target.com$p``,
            # ``target.com{path}``) must not read as a distinct host.
            host = _HOST_TAIL_RE.sub("", host)
            if host and not _is_transparent(host):
                hosts.add(host)
        if include_bare:
            for raw in _BARE_HOST_RE.findall(value):
                host = raw.lower().removeprefix("www.")
                if host.rsplit(".", 1)[-1] in _NON_TLD_SUFFIXES:
                    continue  # a filename, not a host
                if host and not _is_transparent(host):
                    hosts.add(host)
    elif isinstance(value, dict):
        for item in value.values():
            hosts |= _hosts_in(item, include_bare=include_bare)
    elif isinstance(value, (list, tuple)):
        for item in value:
            hosts |= _hosts_in(item, include_bare=include_bare)
    return hosts


def _is_network_tool(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return (
        lowered in _NETWORK_TOOL_NAMES
        or "web_fetch" in lowered
        or "web_search" in lowered
        or "download" in lowered
    )


def _hosts_for_tool(name: str, args: Any) -> set[str]:
    if not _is_network_tool(name):
        return set()
    lowered = (name or "").lower()
    if "web_fetch" in lowered and isinstance(args, dict):
        # ``info_to_extract`` is a prompt, not a network destination. Scanning
        # every argument can make a URL mentioned in the prompt look like a
        # second fetched host, corrupting both attribution and quarantine.
        targets = [
            value
            for key, value in args.items()
            if str(key).strip().lower() in {"url", "urls"}
        ]
        return _hosts_in(targets)
    return _hosts_in(args, include_bare="search" in lowered)


def _is_shell_tool(name: str) -> bool:
    return (name or "").strip().lower() in {"bash", "shell", "run_command"}


def _may_vouch(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return any(hint in lowered for hint in _VOUCHING_TOOL_HINTS)


def _is_search_tool(name: str) -> bool:
    return "search" in (name or "").strip().lower()


def _is_fetch_tool(name: str) -> bool:
    return "web_fetch" in (name or "").strip().lower()


def _result_verdict(result: ToolResult) -> str:
    """How did this call fare against its target?

    ``failed``     — hard: the target refused or returned nothing usable.
    ``soft``       — a failure worth a nudge but too weak to justify cutting the
                     host off: a search engine with no hits (the index is not
                     the site), or a fetch that returned a suspiciously short
                     page the tool itself flagged as *possibly* un-rendered and
                     told the agent it might still be useful.
    ``succeeded``  — a fetch-class tool returned real content (see
                     :data:`_VOUCHING_TOOL_HINTS`).
    ``neutral``    — unknowable, so it neither accuses nor vouches.

    The distinction that matters is NOT whether the tool call errored — a shell
    command that prints ``403 Forbidden`` exits 0 — but whether the target
    yielded something.
    """
    if result.is_error:
        # A search-provider failure says nothing about whether the target site
        # itself is readable. It may justify a route-change hint, but it must
        # never quarantine that site before a fetch has even been attempted.
        return _SOFT_FAILED if _is_search_tool(result.name) else _FAILED
    text = (result.result or "").strip()
    if not text or text == "(no output)":
        return _SOFT_FAILED if _is_search_tool(result.name) else _FAILED
    if _SOFT_FAILURE_LINE_RE.search(text):
        return _SOFT_FAILED
    if _NO_RESULT_RE.search(text):
        # "no results" from a search engine is soft; from a fetch it is hard.
        return _SOFT_FAILED if _is_search_tool(result.name) else _FAILED
    if _FAILURE_LINE_RE.search(text):
        return _SOFT_FAILED if _is_search_tool(result.name) else _FAILED
    if _is_shell_tool(result.name):
        if _HTTP_FAILURE_RE.search(text):
            return _FAILED
        if _MARKUP_DUMP_RE.search(text):
            # A hand-rolled scrape dumped markup. Whether it held the wanted
            # content is not decidable here, so it must not vouch either.
            return _NEUTRAL
    return _SUCCEEDED if _may_vouch(result.name) else _NEUTRAL


def _merge_verdict(current: str | None, new: str) -> str:
    """Combine outcomes for multiple URLs on the same host.

    One useful fetched page proves that the host is reachable, so success wins.
    Otherwise retain the strongest failure signal.
    """
    if current is None or new == _SUCCEEDED:
        return new
    if current == _SUCCEEDED:
        return current
    rank = {_NEUTRAL: 0, _SOFT_FAILED: 1, _FAILED: 2}
    return new if rank[new] > rank[current] else current


def _numbered_fetch_verdicts(result: ToolResult) -> dict[str, str]:
    """Read ``[N] URL: ... / Info: ...`` batches one URL at a time."""
    verdicts: dict[str, str] = {}
    for match in _NUMBERED_FETCH_BLOCK_RE.finditer(result.result or ""):
        block_hosts = _hosts_in(match.group("url"), include_bare=True)
        if not block_hosts:
            continue
        block_result = ToolResult(
            name=result.name,
            args={},
            result=match.group("info"),
            duration_ms=result.duration_ms,
            tool_call_id=result.tool_call_id,
            is_error=False,
            interrupted=result.interrupted,
        )
        verdict = _result_verdict(block_result)
        for host in block_hosts:
            verdicts[host] = _merge_verdict(verdicts.get(host), verdict)
    return verdicts


def _host_verdicts(result: ToolResult, hosts: set[str]) -> dict[str, str]:
    """Attribute a tool result without blaming unrelated hosts in one call."""
    if len(hosts) <= 1:
        verdict = _result_verdict(result)
        return {host: verdict for host in hosts}

    if _is_fetch_tool(result.name):
        # Both web_fetch implementations expose an explicit URL boundary for
        # batched calls. Hosts missing from a parsed block remain neutral rather
        # than inheriting another URL's failure.
        parsed = _numbered_fetch_verdicts(result)
        return {host: parsed[host] for host in hosts if host in parsed}

    verdict = _result_verdict(result)
    if verdict == _FAILED:
        # A combined bash/download result does not say which target failed.
        # Preserve the nudge signal, but do not quarantine every host in it.
        verdict = _SOFT_FAILED
    return {host: verdict for host in hosts}


class StuckTargetGuard(BaseObserver):
    """Nudge, then quarantine, a repeatedly failing network target.

    critical = True → awaited; the returned ``Intervention`` is collected.

    Args:
        hint_after: failures of one host within the window before the first
            nudge. Both hard and soft failures count.
        escalate_after: HARD failures before the host is quarantined. Soft
            failures (an empty search index, a short-but-delivered page) never
            reach this, so being unable to *find* a page cannot cut off the
            ability to *fetch* it.
        window: how many network turns the count looks back over. Local-only
            turns do not enter the window.

    The default threshold stays below the configured window so quarantine can
    engage during one sustained failure burst.
    """

    critical = True

    def __init__(
        self,
        *,
        hint_after: int = 6,
        escalate_after: int = 10,
        window: int = 20,
    ) -> None:
        self.hint_after = max(2, int(hint_after))
        self.escalate_after = max(self.hint_after + 1, int(escalate_after))
        self.window = max(self.escalate_after, int(window))
        # Per network turn: (hard failures, soft failures) by host.
        self._recent: deque[tuple[frozenset[str], frozenset[str]]] = deque(
            maxlen=self.window,
        )
        self._fired: set[tuple[str, str]] = set()   # (host, "hint"|"block")
        self._blocked_hosts: set[str] = set()
        self._network_turns: set[int] = set()
        self._failed_by_turn: dict[int, set[str]] = {}
        self._soft_failed_by_turn: dict[int, set[str]] = {}
        self._succeeded_by_turn: dict[int, set[str]] = {}

    async def on_loop_start(self, config: Any) -> None:
        self._recent.clear()
        self._fired.clear()
        self._blocked_hosts.clear()
        self._network_turns.clear()
        self._failed_by_turn.clear()
        self._soft_failed_by_turn.clear()
        self._succeeded_by_turn.clear()

    async def on_tool_call(
        self,
        ctx: TurnContext,
        tool_call: dict,
    ) -> ToolCallIntervention | None:
        hosts = _hosts_for_tool(
            str(tool_call.get("name") or ""),
            tool_call.get("args") or {},
        )
        blocked = sorted(hosts & self._blocked_hosts)
        if not blocked:
            return None
        joined = ", ".join(f"`{host}`" for host in blocked)
        return ToolCallIntervention(skip_with_result=(
            f"[STUCK TARGET BLOCKED] Further calls targeting {joined} are "
            "disabled for this run after repeated confirmed failures. Use a "
            "different source or finish with an explicit limitation."
        ))

    async def on_tool_result(
        self,
        ctx: TurnContext,
        result: ToolResult,
    ) -> ToolResult | None:
        if not _is_network_tool(result.name):
            return None
        hosts = _hosts_for_tool(result.name, result.args)
        # ``bash`` is a mixed local/network tool. A local pytest/file/python
        # command must not age this network-only window.
        if result.name.strip().lower() == "bash" and not hosts:
            return None
        self._network_turns.add(ctx.turn)
        if not hosts:
            return None
        for host, verdict in _host_verdicts(result, hosts).items():
            if verdict == _NEUTRAL:
                # Counted as a network turn (it ages the window) but attributed
                # to neither bucket: an unreadable outcome must not vouch.
                continue
            target = {
                _FAILED: self._failed_by_turn,
                _SOFT_FAILED: self._soft_failed_by_turn,
                _SUCCEEDED: self._succeeded_by_turn,
            }[verdict]
            target.setdefault(ctx.turn, set()).add(host)
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        is_network_turn = ctx.turn in self._network_turns
        self._network_turns.discard(ctx.turn)
        hard = self._failed_by_turn.pop(ctx.turn, set())
        soft = self._soft_failed_by_turn.pop(ctx.turn, set())
        succeeded = self._succeeded_by_turn.pop(ctx.turn, set())
        if not is_network_turn:
            return None

        # A successful response is concrete progress. Forget this host's prior
        # failures and allow a future independent failure sequence to start
        # fresh. If one turn both failed and succeeded, success wins.
        for host in succeeded:
            self._reset_host(host)
        hard -= succeeded
        soft -= succeeded | hard

        # Every network turn ages the rolling window. A successful request to
        # another host contributes an empty slot; local work contributes none.
        self._recent.append((frozenset(hard), frozenset(soft)))
        hard_counts: Counter[str] = Counter()
        all_counts: Counter[str] = Counter()
        for turn_hard, turn_soft in self._recent:
            hard_counts.update(turn_hard)
            all_counts.update(turn_hard | turn_soft)
        # A host whose old burst aged below a threshold may earn a fresh hint
        # if a later, independent failure burst starts. Quarantined hosts stay
        # latched for the rest of the loop.
        self._fired = {
            (host, kind)
            for host, kind in self._fired
            if host in self._blocked_hosts or self._count_for(
                kind, host, hard_counts, all_counts,
            ) >= self._threshold_for(kind)
        }

        # Quarantine outranks a nudge, and only hard failures can reach it.
        for host, seen in hard_counts.most_common():
            if seen >= self.escalate_after and (host, "block") not in self._fired:
                self._fired.add((host, "block"))
                self._blocked_hosts.add(host)
                self._log(seen, host, ctx.turn, "quarantining")
                return Intervention(
                    inject_messages=[self._message(host, seen, "block")],
                )
        for host, seen in all_counts.most_common():
            if seen >= self.hint_after and (host, "hint") not in self._fired:
                self._fired.add((host, "hint"))
                self._log(seen, host, ctx.turn, "nudging")
                return Intervention(
                    inject_messages=[self._message(host, seen, "hint")],
                )
        return None

    def _threshold_for(self, kind: str) -> int:
        return self.escalate_after if kind == "block" else self.hint_after

    @staticmethod
    def _count_for(
        kind: str,
        host: str,
        hard_counts: Counter[str],
        all_counts: Counter[str],
    ) -> int:
        counts = hard_counts if kind == "block" else all_counts
        return counts.get(host, 0)

    def _log(self, seen: int, host: str, turn: int, action: str) -> None:
        logger.info(
            "StuckTargetGuard: %d failed attempts in the last %d network turns "
            "targeted %s (turn %d) — %s",
            seen, len(self._recent), host, turn, action,
        )

    def _reset_host(self, host: str) -> None:
        self._recent = deque(
            (
                (
                    frozenset(item for item in turn_hard if item != host),
                    frozenset(item for item in turn_soft if item != host),
                )
                for turn_hard, turn_soft in self._recent
            ),
            maxlen=self.window,
        )
        self._fired = {item for item in self._fired if item[0] != host}
        self._blocked_hosts.discard(host)

    def _message(self, host: str, seen: int, kind: str) -> str:
        span = len(self._recent)
        if kind == "block":
            return (
                f"[guard] {seen} of your last {span} network steps targeting "
                f"`{host}` produced confirmed failures. Further calls to this "
                f"host are now blocked for this run. Use another source, or "
                f"answer with what you have and state plainly which material "
                f"could not be retrieved."
            )
        return (
            f"[guard] {seen} of your last {span} network steps targeting "
            f"`{host}` failed. Change route now: use a different source, search "
            f"for the page title, find an official PDF/API/archive, or accept "
            f"that the material is unavailable and state the limitation. Do "
            f"not keep retrying `{host}` with cosmetic variations."
        )


__all__ = ["StuckTargetGuard"]
