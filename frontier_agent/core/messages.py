"""OpenAI-compatible message types — provider-agnostic chat representation."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(TypedDict):
    """OpenAI-style tool_call payload — ``function.arguments`` is JSON-encoded.

    Wire key order is fixed ``{type, id, function}`` to match the LangChain
    serializer the served checkpoints were aligned against; do not reorder.
    """

    id: str
    type: Literal["function"]
    function: dict[str, str]  # {"name": str, "arguments": str}


class Message(TypedDict, total=False):
    """One chat message in OpenAI Chat Completions format.

    The last three keys are in-process only and must never reach the wire; see
    the note on each. Everything above them is OpenAI-compatible.
    """

    role: Role
    content: Any            # str | list[dict] (Anthropic blocks) | None
    name: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    reasoning_content: str
    # Presentation metadata the full-screen TUI reads back off a replayed tool
    # message to redraw a tool call the way it originally rendered (duration in
    # the header, error styling). Written by
    # ``TerminalSession._workflow_display_messages`` and consumed by
    # apodex/tui/{app,widgets}.py.
    #
    # Unlike ``reasoning_content`` these need no stripping step: they are only
    # ever put on messages in ``display_history``, which feeds the TUI and
    # ``/resume`` and is never the list sent to a model. Do not set them on a
    # message that goes into the model-facing history.
    duration_ms: int
    is_error: bool
    # Spill-store paths this message carries, as data rather than as prose for a
    # parser to recover. Two producers: a Tier 1 tool placeholder naming the file
    # its discarded body went to, and the compaction recovery index. The rendered
    # text stays — the MODEL reads the paths and acts on them — but nothing reads
    # the text back, which is what makes an index distinguishable from a summary
    # that happens to quote one. Filtered out by ``for_wire``.
    spill_refs: list[str]


# ── Wire boundary ────────────────────────────────────────────────────────

# Message-level keys the OpenAI Chat Completions wire accepts. Anything else on a
# ``Message`` is in-process bookkeeping.
#
# ``reasoning_content`` is deliberately INSIDE this set. It is not
# unconditionally in-process: DeepSeek-V4 / o-series proxies require it on prior
# assistant turns, which is why the decision belongs to
# :func:`assistant_msg_with_reasoning` at construction time — the format the
# model wants is known there and not here. Stripping it at the boundary would
# silently break those providers.
#
# ``cache_control`` is absent on purpose: Anthropic prompt caching attaches it
# INSIDE a content block (``{"type": "text", "text": …, "cache_control": …}``),
# never to the message, so a message-level filter leaves it alone.
WIRE_MESSAGE_KEYS = frozenset({
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "reasoning_content",
})


def for_wire(messages: list[Message]) -> list[Message]:
    """Drop in-process-only keys before a message list is handed to a provider.

    Today this is a backstop rather than a fix: the keys it can remove
    (``duration_ms``, ``is_error``) are only ever set on ``display_history``,
    which is persisted and restored separately from the model-facing ``history``
    and is never what reaches a client. But that invariant currently lives in a
    comment, and ``OpenAIClient`` passes each message dict to the SDK verbatim —
    so a single stray key anywhere in the loop, a compactor or a workflow lands
    on the wire and, on a served checkpoint, off its training distribution. This
    makes the invariant enforced at the one place it matters.

    Key ORDER is preserved by iterating each message rather than rebuilding in a
    fixed order: some served checkpoints are byte-shape sensitive (see the note
    above the builders). A list with nothing to strip is returned unchanged, so
    the common path is not merely equal but identical.

    Only the OpenAI-compatible client needs this. The Anthropic and Responses
    clients project messages through ``_to_anthropic_msg`` /
    ``_to_responses_input``, which read named keys and therefore cannot carry an
    unexpected one onto the wire.
    """
    if all(key in WIRE_MESSAGE_KEYS for message in messages for key in message):
        return messages
    return [
        message
        if all(key in WIRE_MESSAGE_KEYS for key in message)
        else {
            key: value
            for key, value in message.items()
            if key in WIRE_MESSAGE_KEYS
        }  # type: ignore[misc]
        for message in messages
    ]


# ── Builders ─────────────────────────────────────────────────────────────


# Key insertion order: ``content`` first, then ``role``. Some served
# checkpoints are sensitive to this byte shape (wire byte-equality with
# LangChain's ``_convert_message_to_dict`` — see migration gotcha #2). Do
# not reorder these dict literals.


def system_msg(content: str) -> Message:
    return {"content": content, "role": "system"}


def user_msg(content: str) -> Message:
    return {"content": content, "role": "user"}


def assistant_msg(
    content: Any = "",
    *,
    tool_calls: list[ToolCall] | None = None,
    reasoning: str = "",
) -> Message:
    # ``reasoning_content`` is carried here ONLY for in-process bookkeeping;
    # it must NOT be serialised onto the wire assistant message (it is not in
    # the served checkpoint's training distribution and silently degrades
    # multi-turn benchmarks — migration gotcha #1). The history normaliser is
    # responsible for stripping/inlining it before send.
    m: Message = {"content": content, "role": "assistant"}
    if tool_calls:
        m["tool_calls"] = tool_calls
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def tool_msg(content: str, tool_call_id: str) -> Message:
    return {"content": content, "role": "tool", "tool_call_id": tool_call_id}


def assistant_msg_with_reasoning(
    visible: Any,
    reasoning: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    thinking_format: str = "tag",
) -> Message:
    """Build a wire assistant message, handling reasoning per ``thinking_format``.

    Single source of truth for the **outbound** ``reasoning_content``
    contract (the PR #209 leak guard). The kernel's
    ``NativeMessageNormalizer.to_history`` delegates here; self-contained
    agent loops that bypass the kernel normalizer (workflows running their
    own loop, force-final rescue paths) MUST route reasoning through this
    helper instead of passing ``reasoning=`` to :func:`assistant_msg`
    directly — otherwise a bare ``reasoning_content`` field leaks onto the
    wire and lands the request off the served checkpoint's distribution.

    - ``tag`` (SGLang / Qwen): inline reasoning into ``content`` as
      ``<think>…</think>`` (nested ``</think>`` escaped) so the chat
      template reconstructs it next turn; NO bare wire field.
    - ``reasoning_content`` (DeepSeek V4 / o-series proxies): keep
      ``reasoning_content`` on the wire — those proxies require it on
      prior assistant turns.
    - ``none`` / ``content_block`` / empty reasoning: drop reasoning.

    LIMITATION (``content_block`` / Anthropic extended thinking): the visible
    text is kept but the prior ``thinking`` block and its ``signature`` are
    NOT round-tripped. Anthropic *extended thinking* with tool use expects the
    signed thinking block echoed back across turns; that continuation is not
    yet supported. No effect on ``tag`` (served qwen/SGLang checkpoints) or
    plain Anthropic calls. Tracked as a follow-up.
    """
    if reasoning and thinking_format == "tag":
        safe = reasoning.replace("</think>", "</ think>")
        wrapped = f"<think>{safe}</think>"
        visible = f"{wrapped}\n{visible}" if visible else wrapped
        return assistant_msg(visible, tool_calls=tool_calls)
    if reasoning and thinking_format == "reasoning_content":
        return assistant_msg(visible, tool_calls=tool_calls, reasoning=reasoning)
    return assistant_msg(visible, tool_calls=tool_calls)


# ── Helpers ──────────────────────────────────────────────────────────────


def text_of(content: Any) -> str:
    """Flatten an OpenAI/Anthropic message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                val = block.get("text") or block.get("content") or ""
                if isinstance(val, str):
                    parts.append(val)
        return "\n".join(parts)
    return str(content)


def is_tool_msg(m: Message) -> bool:
    return m.get("role") == "tool"


def is_assistant_msg(m: Message) -> bool:
    return m.get("role") == "assistant"


__all__ = [
    "WIRE_MESSAGE_KEYS",
    "Message",
    "Role",
    "ToolCall",
    "assistant_msg",
    "assistant_msg_with_reasoning",
    "for_wire",
    "is_assistant_msg",
    "is_tool_msg",
    "system_msg",
    "text_of",
    "tool_msg",
    "user_msg",
]
