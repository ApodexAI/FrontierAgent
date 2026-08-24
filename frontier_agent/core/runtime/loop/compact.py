"""Message-history compaction for the agent loop."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from frontier_agent.core.messages import (
    Message,
    is_assistant_msg,
    is_tool_msg,
    text_of,
    tool_msg,
    user_msg,
)

__all__ = [
    "COMPACTION_SEQ_KEY",
    "FORCE_COMPACTION_KEY",
    "INPUT_ESTIMATE_KEY",
    "OMITTED_TOOL_RESULT_PLACEHOLDER",
    "SPILL_MANIFEST_HEADER",
    "URL_RE",
    "CompactionPolicy",
    "DefaultCompactionPolicy",
    "DefaultMessageCompactor",
    "KeepLastNToolResultsCompactor",
    "MessageCompactor",
    "StringSliceCompactor",
    "compact_messages",
    "compress_tool_results",
    "estimate_tokens",
]

# Observer-to-loop handshake: request one pass, then record only completed passes.
FORCE_COMPACTION_KEY = "_force_compaction"
COMPACTION_SEQ_KEY = "_compaction_seq"
# Loop-to-observer handshake: the token estimate of the message list actually
# handed to the provider this turn. An observer that samples the history itself
# cannot reproduce it — by turn end the list has grown by this turn's completion
# and tool results, and the per-call system addendum was never in it at all.
INPUT_ESTIMATE_KEY = "_input_token_estimate"

# The model reads this in place of a result it already consumed, with its own
# tool call still visible above it. "Omitted to save tokens" invites the
# obvious repair — call the same tool again — which is how a compacted
# research agent ends up re-issuing queries it already ran. Say plainly that
# re-calling cannot bring the result back.
OMITTED_TOOL_RESULT_PLACEHOLDER = (
    "Tool result dropped to save tokens. You already read it; re-running the "
    "same call will not restore it. Rely on your notes and later messages."
)

URL_RE = re.compile(r'https?://[^\s\)>"\'<]+')
_TOOL_RESULT_COMPACT_MAX_CHARS = 1_200

# Header of the spill recovery index. This is presentation only — the text the
# MODEL reads above the paths — since the index is identified by
# ``Message.spill_refs``. The two remaining substring checks against it
# (``compact_messages`` here, and the summarizer input filter) are the legacy path
# for a history checkpointed before that field existed, and can go once no such
# checkpoint can still be resumed.
#
# Lives here, not in ``tiered_compact``, because ``tiered_compact`` imports this
# module, so the dependency cannot go the other way.
SPILL_MANIFEST_HEADER = (
    "[Read-only recovery index; use only for missing older detail. "
    "Never write here.]"
)


def _tool_names_by_call_id(messages: list[Message]) -> dict[str, str]:
    """Map ``tool_call_id`` → tool name from AIMessage ``tool_calls``.

    A ``ToolMessage`` carries no tool name, so any name-keyed policy has to
    resolve it through the requesting assistant message.
    """
    out: dict[str, str] = {}
    for msg in messages:
        if not is_assistant_msg(msg):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name")
            tid = tc.get("id") or (fn.get("id") if isinstance(fn, dict) else None)
            if tid and name:
                out[tid] = name
    return out


def _condense(content: str, max_chars: int) -> str:
    """Head + tail + URLs of *content*, never longer than the original."""
    prefix = f"[Compressed tool result: {len(content):,} characters]\n"
    marker = "\n… [middle omitted] …\n"
    url_lines: list[str] = []
    url_budget = max_chars // 2
    for url in dict.fromkeys(URL_RE.findall(content)):
        candidate = "\n[Source URLs]\n" + "\n".join([*url_lines, url])
        if len(candidate) > url_budget:
            break
        url_lines.append(url)
    url_section = (
        "\n[Source URLs]\n" + "\n".join(url_lines) if url_lines else ""
    )
    remaining = max_chars - len(prefix) - len(marker) - len(url_section)
    head_size = max(0, int(remaining * 0.7))
    tail_size = max(0, remaining - head_size)
    summary = (
        f"{prefix}{content[:head_size]}{marker}"
        f"{content[-tail_size:] if tail_size else ''}"
        f"{url_section}"
    )
    # A compactor must never enlarge a result, and max_chars is an actual cap.
    return summary if len(summary) < len(content) else content


def compress_tool_results(
    messages: list[Message],
    *,
    max_chars: int = _TOOL_RESULT_COMPACT_MAX_CHARS,
    protect_tool_names: frozenset[str] = frozenset(),
    protect_max_chars: int | None = None,
    preserve_tool_result_ids: frozenset[str] = frozenset(),
) -> list[Message]:
    """Return a copy of *messages* with every large tool result condensed.

    The tool-call protocol fields stay untouched, so the result is safe to
    summarize or send back to a provider.  Keep both ends of a result and its
    URLs: command output often ends with the meaningful status, while research
    output needs source links for a later re-fetch.

    ``protect_tool_names`` (agent-team fan-in: collect_reports / submit_report /
    …) are left intact, or bounded by the wider ``protect_max_chars`` when one
    is given. A caller whose output goes straight back to the provider MUST pass
    the protect set: those same results are pinned out of
    ``KeepLastNToolResultsCompactor``'s spill path, so a sub-agent report cut to
    a few hundred characters here is gone for good. A caller that only feeds a
    summarizer can pass ``protect_max_chars`` instead, keeping the summary
    request affordable while the report still arrives as more than a stub.

    ``preserve_tool_result_ids`` keeps specific results byte-for-byte. Tiered
    compaction uses it when no spill store is available so the latest tool-call
    turn — whose results have not reached the model yet — cannot be shortened.
    """
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if protect_max_chars is not None and protect_max_chars < 200:
        raise ValueError("protect_max_chars must be at least 200")

    id_to_name = _tool_names_by_call_id(messages) if protect_tool_names else {}
    compacted: list[Message] = []
    for message in messages:
        clone = message.copy()
        if not is_tool_msg(message):
            compacted.append(clone)
            continue
        if str(message.get("tool_call_id") or "") in preserve_tool_result_ids:
            compacted.append(clone)
            continue
        budget = max_chars
        if id_to_name.get(message.get("tool_call_id", "")) in protect_tool_names:
            if protect_max_chars is None:
                compacted.append(clone)
                continue
            budget = protect_max_chars
        content = text_of(message.get("content"))
        if len(content) <= budget:
            compacted.append(clone)
            continue
        clone["content"] = _condense(content, budget)
        compacted.append(clone)
    return compacted


@runtime_checkable
class MessageCompactor(Protocol):
    """Pluggable strategy for shrinking a message history.

    Workflows can provide their own compactor via ``LoopConfig.compactor``
    when the default middle-squash policy does not fit. Implementations
    must honour two invariants:

    - Any ``SystemMessage`` that sits at the head of the list stays at the
      head of the returned list (agent-loop assumes this).
    - The message right after any dropped ``AIMessage(tool_calls=[...])``
      cannot be a bare ``ToolMessage`` — an orphan ``tool_call_id`` is a
      hard HTTP 400 on Azure and other providers.

    Optionally, an implementation may expose a ``last_event``
    :class:`~frontier_agent.core.loop_types.CompactionEvent` describing what the
    most recent ``compact`` call did. The agent loop stamps the turn and
    compaction sequence onto it — which a compactor has no way to know — and
    broadcasts it to ``on_compaction`` observers, which is what puts the
    summary into the durable trajectory. Compactors that expose nothing are
    read with ``getattr`` and simply go unreported.
    """

    def compact(
        self,
        messages: list[Message],
        keep_recent: int,
    ) -> list[Message]:
        ...


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate combined token count with a small per-message overhead."""
    from frontier_agent.core.runtime.loop.context_budget import estimate_tokens as _est
    total = 0
    for msg in messages:
        total += _est(text_of(msg.get("content"))) + 4  # +4 per message overhead
    return total


def compact_messages(
    messages: list[Message],
    keep_recent: int,
) -> list[Message]:
    """Compact a message history by summarising the middle.

    Keeps system messages at the start, keeps the last ``keep_recent``
    messages verbatim, and replaces the middle with a single summary
    ``HumanMessage`` containing short snippets of user / agent / tool
    turns so the loop still has some context of what happened earlier.
    """
    system_msgs: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.get("role") == "system" and not rest:
            system_msgs.append(msg)
        else:
            rest.append(msg)

    if len(rest) <= keep_recent:
        return messages  # nothing to compact

    # Find a clean split point: the recent window must NOT start on a
    # ToolMessage — its matching AIMessage(tool_calls=[...]) would be
    # split into the middle and Azure would reject the orphan
    # tool_call_id with HTTP 400.
    split_idx = len(rest) - keep_recent
    while split_idx < len(rest) - 1 and is_tool_msg(rest[split_idx]):
        split_idx += 1

    middle = rest[:split_idx]
    recent = rest[split_idx:]

    # Keep short snippets of user / agent / tool output so the loop can
    # still reason about what happened earlier. For tool calls we preserve
    # name + args preview on the AIMessage side and the first URL + a
    # longer result preview on the ToolMessage side, so the LLM can
    # re-fetch a source whose full text was truncated away.
    parts: list[str] = []
    for msg in middle:
        content = text_of(msg.get("content")).strip()
        if content.startswith("[Compacted"):
            continue
        # A manifest must be dropped, never summarized: the ``content[:400]``
        # cut below lands mid-path on its last entry, and Tier 2 re-attaches the
        # real index from the refs it collected anyway. Recognised by its field;
        # the header check is the legacy path for a history checkpointed before
        # the field existed.
        if msg.get("spill_refs") or SPILL_MANIFEST_HEADER in content:
            continue
        if msg.get("role") == "user":
            if content:
                parts.append(f"[User: {content[:400]}]")
        elif is_assistant_msg(msg):
            tcs = msg.get("tool_calls") or []
            if tcs:
                tc_parts: list[str] = []
                for tc in tcs[:3]:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict):
                        name = fn.get("name", "?")
                        args = fn.get("arguments", {})
                    elif isinstance(tc, dict):
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                    else:
                        name = getattr(tc, "name", "?")
                        args = getattr(tc, "args", {})
                    tc_parts.append(f"{name}({str(args)[:120]})")
                tc_line = "; ".join(tc_parts)
                if content:
                    parts.append(
                        f"[Agent: {content[:200]} | called: {tc_line}]"
                    )
                else:
                    parts.append(f"[Agent called: {tc_line}]")
            elif content:
                parts.append(f"[Agent: {content[:300]}]")
        elif is_tool_msg(msg):
            tool_name = msg.get("name") or "tool"
            urls = URL_RE.findall(content)
            url_tag = f" url={urls[0]}" if urls else ""
            if content:
                parts.append(
                    f"[Tool {tool_name}{url_tag}: {content[:300]}]"
                )
    summary_body = "\n".join(parts[-20:]) if parts else ""
    summary_text = f"[Compacted {len(middle)} earlier messages]"
    if summary_body:
        summary_text = f"{summary_text}\n{summary_body}"

    compact_summary = user_msg(summary_text)

    return [*system_msgs, compact_summary, *recent]


class StringSliceCompactor:
    """Thin adapter that wraps :func:`compact_messages` as a
    :class:`MessageCompactor`.

    Sync, deterministic, no LLM call. Squashes the middle of the
    conversation into a single ``HumanMessage`` with short snippets per
    dropped turn (see :func:`compact_messages`). The agent loop's
    historical default; the SDK now defaults to
    :class:`frontier_agent.core.runtime.loop.compact_llm.LLMSummaryCompactor`
    instead because the structured-summary prompt preserves entities,
    ruled-out candidates, and source URLs more faithfully on long runs.

    Use this compactor explicitly when:
    - You don't have a summarizer LLM available (offline / restricted).
    - You need fully deterministic compaction for golden-file tests.
    - Latency-sensitive runs where the extra LLM call is not affordable.
    """

    def compact(
        self,
        messages: list[Message],
        keep_recent: int,
    ) -> list[Message]:
        return compact_messages(messages, keep_recent)


# Compatibility alias used by the agent-loop fallback.
DefaultMessageCompactor = StringSliceCompactor


@runtime_checkable
class CompactionPolicy(Protocol):
    """Decides *when* the agent loop should shrink its message history.

    Paired with :class:`MessageCompactor` which decides *how*. The loop
    calls ``should_compact(...)`` every turn (pre-LLM) and only invokes
    the compactor on ``True``. Implementations must be fast — this runs
    on every turn — and deterministic given the same inputs.

    Workflows override this when turn-count + token-limit heuristics
    don't fit: e.g. "only compact when the next message would exceed
    90% of the provider's window" or "never compact, I wrote my own
    retention inside the compactor".
    """

    def should_compact(
        self,
        turn: int,
        messages: list[Message],
        estimated_tokens: int,
    ) -> bool:
        ...


class DefaultCompactionPolicy:
    """Default policy: compact when the turn or token threshold trips.

    Mirrors the inline check that lived in ``agent_loop.py`` before this
    Protocol was extracted — ``turn > compact_after_turns`` OR
    ``estimated_tokens > context_token_limit``.
    """

    def __init__(
        self,
        compact_after_turns: int,
        context_token_limit: int,
    ) -> None:
        self._compact_after_turns = compact_after_turns
        self._context_token_limit = context_token_limit

    def should_compact(
        self,
        turn: int,
        messages: list[Message],
        estimated_tokens: int,
    ) -> bool:
        return (
            turn > self._compact_after_turns
            or estimated_tokens > self._context_token_limit
        )


class KeepLastNToolResultsCompactor:
    """Replace older ``ToolMessage`` bodies with a short placeholder.

    Keeps the last ``keep_tool_result`` tool results verbatim and replaces
    the content of every earlier one with :data:`OMITTED_TOOL_RESULT_PLACEHOLDER`.
    ``SystemMessage``, ``HumanMessage``, and every ``AIMessage`` (including
    its thinking trace) are left intact, so the model retains its full
    chain of reasoning and tool-call metadata while dropping the bulk of
    old tool-result bodies (the dominant context cost in long ReAct runs).

    Idempotent: already-placeheld messages are detected by content match
    and left alone, so this is safe to invoke on every turn.

    ``keep_tool_result == -1`` disables filtering entirely.

    Caveat: only ``ToolMessage`` content is redacted. Workflows that
    inject large content as ``HumanMessage`` (e.g. an observer that
    splices fan-in reports between turns) bypass this compactor; pair
    with a different strategy or route the content through a tool so
    it lands as ``ToolMessage``.
    """

    def __init__(
        self,
        keep_tool_result: int,
        protect_tool_names: frozenset[str] = frozenset(),
        spill: Callable[[str, str], str | None] | None = None,
    ) -> None:
        if keep_tool_result < -1:
            raise ValueError(
                f"keep_tool_result must be >= -1 (got {keep_tool_result})"
            )
        self._keep = keep_tool_result
        # Tool names whose results are NEVER blanked regardless of age (e.g.
        # agent-team fan-in: collect_reports / assign_task / submit_report).
        # Resolved by tool_call_id → the requesting AIMessage's tool_calls,
        # because a ``ToolMessage`` carries no name. Empty (default) = blank by
        # age only, so existing callers are unaffected.
        self._protect = frozenset(protect_tool_names)
        self._spill = spill

    def compact(
        self,
        messages: list[Message],
        keep_recent: int,
    ) -> list[Message]:
        if self._keep == -1:
            return messages

        tool_indices = [
            i for i, m in enumerate(messages) if is_tool_msg(m)
        ]
        if not tool_indices:
            return messages

        keep_count = min(self._keep, len(tool_indices))
        keep_set = (
            set(tool_indices[-keep_count:]) if keep_count > 0 else set()
        )
        if len(keep_set) == len(tool_indices):
            return messages

        id_to_name = (
            _tool_names_by_call_id(messages)
            if self._protect or self._spill is not None
            else {}
        )

        out: list[Message] = []
        for idx, msg in enumerate(messages):
            if not is_tool_msg(msg) or idx in keep_set:
                out.append(msg)
                continue
            if self._protect and id_to_name.get(msg.get("tool_call_id", "")) in self._protect:
                out.append(msg)  # protected fan-in result — never blank
                continue
            content = text_of(msg.get("content"))
            if content.startswith(OMITTED_TOOL_RESULT_PLACEHOLDER):
                out.append(msg)
                continue
            placeholder = OMITTED_TOOL_RESULT_PLACEHOLDER
            spill_path: str | None = None
            if self._spill is not None:
                tool_name = id_to_name.get(msg.get("tool_call_id", ""), "tool")
                try:
                    spill_path = self._spill(tool_name, content)
                except Exception:
                    spill_path = None
                if spill_path:
                    placeholder += f"\n[Full text] {spill_path}"
            replacement = tool_msg(placeholder, msg.get("tool_call_id", ""))
            if spill_path:
                # The text is for the model; this is for us. ``TieredCompactor``
                # collects refs from the field, so nothing has to recognise a
                # path by its shape.
                replacement["spill_refs"] = [spill_path]
            out.append(replacement)
        return out
