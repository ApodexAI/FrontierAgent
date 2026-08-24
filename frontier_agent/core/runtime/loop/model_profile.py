"""Model-agnostic abstractions for thinking extraction and history management."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, TypeGuard, get_args

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.messages import (
    Message,
    ToolCall,
    assistant_msg,
    assistant_msg_with_reasoning,
)

logger = logging.getLogger(__name__)

# ── Regex for <think> tag extraction ─────────────────────────────────────────

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>\s*", re.DOTALL)


# ── Thinking-format inference ────────────────────────────────────────────────
#
# The pattern table itself lives in ``frontier_agent/model_registry.yaml``
# (bundled inside the package so it ships with the wheel — putting it at
# the repo root would make ``pip install frontier_agent`` lose access to
# it). It changes every time we onboard a new endpoint, so keeping the
# data out of core Python lets ops edit one YAML file without touching
# code. Loader is cached for the process lifetime; restart to pick up
# edits.

ThinkingFormat = Literal["tag", "content_block", "reasoning_content", "none"]

_VALID_FORMATS: frozenset[str] = frozenset(get_args(ThinkingFormat))

# Wire protocol the client speaks. Named alias so config readers can declare it
# instead of returning a bare str that every ModelProfile call site then rejects.
WireProtocol = Literal["chat_completions", "anthropic", "responses", "bedrock"]
_VALID_PROTOCOLS: frozenset[str] = frozenset(get_args(WireProtocol))


def is_thinking_format(value: object) -> TypeGuard[ThinkingFormat]:
    """Narrow an unvalidated value (YAML, model registry) to a ThinkingFormat.

    The isinstance check comes first because these frozensets are consulted with
    whatever the YAML parser produced: a bare ``value in _VALID_FORMATS`` raises
    ``TypeError: unhashable type`` for a list or dict, which would abort profile
    loading instead of warning and falling back.
    """
    return isinstance(value, str) and value in _VALID_FORMATS


def is_wire_protocol(value: object) -> TypeGuard[WireProtocol]:
    """Narrow an unvalidated value (YAML ``llm.protocol``) to a WireProtocol.

    See :func:`is_thinking_format` for why the isinstance check is required.
    """
    return isinstance(value, str) and value in _VALID_PROTOCOLS

def _resolve_registry_path() -> Path:
    try:
        from importlib.resources import files
        res = Path(str(files("frontier_agent").joinpath("model_registry.yaml")))
        if res.is_file():
            return res
    except Exception:
        pass
    return Path(__file__).resolve().parents[3] / "model_registry.yaml"


_REGISTRY_PATH = _resolve_registry_path()


@lru_cache(maxsize=1)
def _load_thinking_format_patterns() -> (
    tuple[tuple[re.Pattern[str], ThinkingFormat], ...]
):
    """Read ``thinking_formats`` from ``model_registry.yaml``.

    Returns an empty tuple when the file is missing or malformed —
    callers then get only the user-supplied default. Bad individual
    entries are skipped with a warning rather than failing the whole
    load (so a typo in one row never breaks production inference).
    """
    if not _REGISTRY_PATH.is_file():
        logger.debug(
            "model_registry.yaml not found at %s — inference returns default",
            _REGISTRY_PATH,
        )
        return ()
    try:
        import yaml
    except ImportError:  # pragma: no cover — PyYAML is a project dep
        return ()
    try:
        raw = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse model_registry.yaml: %s", exc)
        return ()

    table = raw.get("thinking_formats") or []
    if not isinstance(table, list):
        logger.warning(
            "model_registry.yaml: 'thinking_formats' must be a list (got %s)",
            type(table).__name__,
        )
        return ()

    out: list[tuple[re.Pattern[str], ThinkingFormat]] = []
    for entry in table:
        if not isinstance(entry, dict):
            continue
        pattern_str = entry.get("pattern")
        fmt = entry.get("format")
        if not pattern_str or fmt not in _VALID_FORMATS:
            logger.warning(
                "model_registry.yaml: skipped invalid entry %r", entry,
            )
            continue
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
        except re.error as exc:
            logger.warning(
                "model_registry.yaml: bad regex %r — %s",
                pattern_str, exc,
            )
            continue
        out.append((compiled, fmt))
    return tuple(out)


def reset_thinking_format_cache() -> None:
    """Clear the cached pattern table — used by tests that rewrite the YAML."""
    _load_thinking_format_patterns.cache_clear()


def infer_thinking_format(
    model_id: str | None,
    *,
    default: ThinkingFormat = "tag",
) -> ThinkingFormat:
    """Pick a thinking format for a model id when none is set explicitly.

    The pattern table is loaded from ``model_registry.yaml`` (see that
    file's header for editing rules + endpoint-dependency caveat).
    Matching is case-insensitive on the bare model id (no provider
    prefix needed). Returns ``default`` for empty or unknown ids.
    """
    if not model_id:
        return default
    needle = model_id.lower()
    for pattern, fmt in _load_thinking_format_patterns():
        if pattern.search(needle):
            return fmt
    return default


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Static facts about a model's capabilities.

    These are properties of the *model*, not the agent using it.
    """

    model_id: str
    provider: str
    context_window: int = 128_000
    supports_native_fc: bool = True
    supports_streaming: bool = True
    supports_images: bool = False
    thinking_format: ThinkingFormat = "none"
    tool_call_format: Literal["native_fc", "text", "both"] = "native_fc"
    # Wire protocol the client speaks: ``chat_completions`` (default),
    # ``anthropic`` (native Messages API + extended thinking), or ``responses``
    # (OpenAI Responses API + encrypted reasoning). Native Anthropic + Responses
    # both return content as a typed block list → ``thinking_format`` is
    # ``content_block`` so the parser keeps the verbatim blocks (signatures /
    # encrypted_content) for faithful multi-turn replay + trajectory.
    protocol: WireProtocol = "chat_completions"


@dataclass
class HistoryPolicy:
    """Agent design choices about how thinking output is used.

    These are properties of the *agent role*, not the model.
    """

    thinking_in_history: bool = False
    thinking_in_memory: bool = True
    thinking_in_sse: bool = True
    tool_result_max_chars: int = 15_000
    compress_thinking_after_turns: int = 0
    max_images_in_history: int = 5
    # PER-ASSISTANT-TURN cap, not a budget over the whole history: ``None``
    # means one turn's reasoning is kept in full, a positive value keeps only
    # the most recent reasoning of that turn that fits this many tokens. Total
    # reasoning in history still grows with turn count — bounding *that* is
    # compaction's job (see ``compress_thinking_after_turns`` / tiered_compact).
    # Disabled history always wins over this cap. Kept last for positional
    # compatibility with the original HistoryPolicy constructor.
    thinking_history_max_tokens: int | None = None


@dataclass
class ThinkingResult:
    """Parsed output from a single LLM response."""

    thinking: str
    visible_content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Native Anthropic thinking / OpenAI Responses reasoning: the VERBATIM
    # content-block list exactly as returned — [{type:"thinking", thinking,
    # signature}, {type:"redacted_thinking", data}, {type:"reasoning",
    # encrypted_content, summary}, {type:"text", ...}]. Preserved so multi-turn
    # replay re-sends the blocks (incl. encrypted signatures / encrypted_content)
    # UNMODIFIED — the provider validates them server-side — and the trajectory
    # can persist them. ``None`` for every other thinking_format → all downstream
    # signature/encrypted-reasoning logic is a no-op there.
    raw_content_blocks: list[Any] | None = None


_CAP_ERROR = (
    "thinking_history_max_tokens must be a non-negative integer or null"
)
_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})


def _coerce_optional_bool(value: Any, key: str) -> bool | None:
    """Read a tri-state flag: absent (``None``), on, or off.

    A quoted YAML ``"false"`` must mean off. Plain ``bool(value)`` would read
    that non-empty string as ON — the exact inversion this policy cannot
    afford, since the flag decides whether reasoning is replayed at all.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise ValueError(f"{key} must be a boolean or null, got {value!r}")


def _coerce_cap(value: Any) -> int:
    """Read the cap as an exact non-negative integer; ``0``/absent = no cap.

    Rejects rather than coerces the two values ``int()`` would quietly
    mangle: ``True`` (→ a 1-token cap, i.e. reasoning effectively off while
    the config reads as enabled) and a fractional float (silently floored).
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{_CAP_ERROR}, not a boolean")
    if isinstance(value, int):
        cap = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{_CAP_ERROR}; got the fractional {value!r}")
        cap = int(value)
    elif isinstance(value, str):
        try:
            cap = int(value.strip())
        except ValueError as exc:
            raise ValueError(_CAP_ERROR) from exc
    else:
        raise ValueError(_CAP_ERROR)
    if cap < 0:
        raise ValueError(_CAP_ERROR)
    return cap


def resolve_history_policy(config: Mapping[str, Any]) -> HistoryPolicy:
    """Resolve the two history settings without losing explicit ``false``.

    ``thinking_in_history`` is the master switch. An explicit ``false`` always
    disables reasoning history. Otherwise a positive
    ``thinking_history_max_tokens`` implicitly enables capped history, while
    ``true`` with a missing/zero cap keeps the full reasoning. With neither
    setting present the legacy disabled default is preserved.

    Both values are validated here — at profile-load time, where a bad value
    is a loud startup failure — rather than silently coerced per turn.
    """
    enabled = _coerce_optional_bool(
        config.get("thinking_in_history"), "thinking_in_history",
    )
    cap = _coerce_cap(config.get("thinking_history_max_tokens"))

    if enabled is False:
        return HistoryPolicy(thinking_in_history=False)
    if cap > 0:
        return HistoryPolicy(
            thinking_in_history=True,
            thinking_history_max_tokens=cap,
        )
    return HistoryPolicy(thinking_in_history=bool(enabled))


# ── Protocols ─────────────────────────────────────────────────────────────────


class ThinkingParser(Protocol):
    """Extract thinking and visible content from an LLM response."""

    def extract(self, response: Any, profile: ModelProfile) -> ThinkingResult:
        """Parse *response* according to the model's ``thinking_format``."""
        ...


class MessageNormalizer(Protocol):
    """Convert an LLM response to the message stored in conversation history."""

    def to_history(
        self,
        response: Any,
        thinking_result: ThinkingResult,
        policy: HistoryPolicy,
    ) -> Message:
        """Return the message form appropriate for the conversation history."""
        ...


# ── Default implementations ───────────────────────────────────────────────────


class DefaultThinkingParser:
    """Handles four thinking formats.

    - ``"none"`` (classic OpenAI / non-reasoning models): no thinking
      extraction; full text is visible.
    - ``"tag"`` (Qwen / DeepSeek-via-SGLang): extract ``<think>…</think>``
      via regex from ``content``.
    - ``"content_block"`` (Anthropic): ``content`` is a list of typed blocks;
      separate ``{type: "thinking"}`` from ``{type: "text"}``.
    - ``"reasoning_content"`` (DeepSeek-V4 API, gpt-5/o-series via standard
      OpenAI-compatible proxies): thinking is a separate top-level
      ``reasoning_content`` field on the response message. The native
      :class:`~frontier_agent.core.llm.LLMResponse` surfaces it directly via
      its ``reasoning_content`` attribute; this parser reads it from there.
    """

    def extract(self, response: Any, profile: ModelProfile) -> ThinkingResult:
        tool_calls: list[dict[str, Any]] = list(response.tool_calls or [])
        fmt = profile.thinking_format

        if fmt == "none":
            content = response.content or ""
            return ThinkingResult(
                thinking="",
                visible_content=content if isinstance(content, str) else "",
                tool_calls=tool_calls,
            )

        if fmt == "tag":
            raw = response.content or ""
            if not isinstance(raw, str):
                raw = ""
            # Prefer the typed reasoning channel when the endpoint actually
            # separates it (SGLang ``--reasoning-parser qwen3`` populates
            # ``additional_kwargs.reasoning_content`` via our ChatOpenAI
            # subclass). Falls back to the ``<think>…</think>`` regex when
            # the channel is empty — that is the stock-SGLang / inline-tag
            # case this format originally handled.
            typed_rc = _extract_reasoning(response)
            if typed_rc:
                return ThinkingResult(
                    thinking=typed_rc,
                    visible_content=raw,
                    tool_calls=tool_calls,
                )
            # Every block, not just the first: a turn can reopen <think>
            # between tool calls, and ``to_history`` rebuilds the message from
            # this result, so a dropped block is reasoning lost from history.
            # Substituting a newline (rather than deleting) keeps the text that
            # surrounded a block from being concatenated —
            # "a<think>x</think>\nb" must not become "ab".
            matches = _THINK_RE.findall(raw)
            if matches:
                thinking = "\n".join(matches)
                visible = _THINK_RE.sub("\n", raw).strip()
            else:
                thinking = ""
                visible = raw
            return ThinkingResult(thinking=thinking, visible_content=visible, tool_calls=tool_calls)

        if fmt == "content_block":
            blocks = response.content or []
            # A content_block turn can still come back as a PLAIN STRING: the
            # native Anthropic client returns a bare string whenever a turn
            # carries no thinking block (adaptive thinking omitted, or the
            # post-tool-call final answer turn), and any provider may degrade to
            # a string. Iterating a string here would walk it CHARACTER by
            # character, match no dict blocks, and silently drop the whole
            # visible answer. Treat the string as visible text (reasoning read
            # off the separate channel, if any).
            if isinstance(blocks, str):
                return ThinkingResult(
                    thinking=_extract_reasoning(response),
                    visible_content=blocks,
                    tool_calls=tool_calls,
                )
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                # ``thinking`` text may be "" when display=omitted, but the block
                # still carries a ``signature`` preserved via raw_content_blocks.
                if block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
                elif block.get("type") == "reasoning":
                    # OpenAI Responses reasoning item: the opaque
                    # ``encrypted_content`` is preserved in raw_content_blocks
                    # below; its human-readable ``summary`` (often empty unless
                    # opted-in) becomes the thinking text. Anthropic emits no
                    # "reasoning" blocks, so this never fires there.
                    for s in block.get("summary") or []:
                        if isinstance(s, dict):
                            thinking_parts.append(s.get("text", ""))
                        elif isinstance(s, str):
                            thinking_parts.append(s)
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return ThinkingResult(
                thinking="\n".join(thinking_parts),
                visible_content="\n".join(text_parts),
                tool_calls=tool_calls,
                # Keep the verbatim block list (thinking/redacted_thinking incl.
                # signatures / reasoning incl. encrypted_content) for faithful
                # replay + trajectory. Only when content really is a non-empty list.
                raw_content_blocks=(
                    list(blocks) if isinstance(blocks, list) and blocks else None
                ),
            )

        if fmt == "reasoning_content":
            # Reasoning lives on a separate channel — the native
            # ``LLMResponse.reasoning_content`` field. ``_extract_reasoning``
            # reads it. Content stays untouched and becomes the visible reply.
            content = response.content or ""
            return ThinkingResult(
                thinking=_extract_reasoning(response),
                visible_content=content if isinstance(content, str) else "",
                tool_calls=tool_calls,
            )

        # Unknown format — treat as no thinking
        content = response.content or ""
        return ThinkingResult(
            thinking="",
            visible_content=content if isinstance(content, str) else "",
            tool_calls=tool_calls,
        )


# ── Native (langchain-free) layer ─────────────────────────────────────────
#
# ``_extract_reasoning`` reads the model's reasoning channel from the native
# ``LLMResponse.reasoning_content`` field so ``DefaultThinkingParser`` can
# surface thinking regardless of the model's ``thinking_format``.
#
# ``NativeMessageNormalizer`` returns an OpenAI-wire ``Message`` dict and,
# critically, does NOT carry ``reasoning_content`` onto the wire except for the
# one format that requires it. For
# ``tag`` models the reasoning is inlined into ``content`` as
# ``<think>{rc}</think>\n{visible}`` instead.


def _extract_reasoning(response: Any) -> str:
    """Reasoning text from a native ``LLMResponse``.

    Reads the native ``reasoning_content`` attribute. As a defensive
    fallback (no langchain dependency) it also accepts a response object
    exposing ``additional_kwargs['reasoning_content']`` — harmless on a
    native ``LLMResponse``, which has no such attribute.
    """
    rc = getattr(response, "reasoning_content", "") or ""
    if isinstance(rc, str) and rc:
        return rc
    kwargs = getattr(response, "additional_kwargs", None) or {}
    if isinstance(kwargs, dict):
        v = kwargs.get("reasoning_content")
        if isinstance(v, str):
            return v
    return ""


def _to_openai_tool_calls(parsed: list[dict[str, Any]]) -> list[ToolCall]:
    """Echo parsed tool calls into OpenAI wire format.

    Key order ``{type, id, function: {name, arguments}}`` — served checkpoints
    are sensitive to this byte shape (migration gotcha #2).

    Native streaming responses can end with an empty or truncated
    ``function.arguments`` string.  The executor already treats that as
    ``{}``, but echoing the malformed string into history makes the *next* LLM
    request invalid on stricter OpenAI-compatible gateways.  Re-serialise
    malformed arguments as ``{}`` so an unknown/bad tool call remains a normal
    recoverable tool error instead of poisoning every later request, including
    the force-final/report call.
    """
    def _arguments_json(raw: Any) -> str:
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False)
        if isinstance(raw, str) and raw.strip():
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError):
                return "{}"
            return raw if isinstance(decoded, dict) else "{}"
        return "{}"

    out: list[ToolCall] = []
    for tc in parsed:
        args = tc.get("args") if "args" in tc else None
        if args is None and "function" in tc:
            raw_args = tc["function"].get("arguments", "{}")
            args_str = _arguments_json(raw_args)
            name = tc["function"].get("name", "")
        else:
            args_str = _arguments_json(args)
            name = tc.get("name", "")
        out.append({
            "type": "function",
            "id": tc.get("id") or "",
            "function": {"name": name, "arguments": args_str},
        })
    return out


class NativeMessageNormalizer:
    """Convert an LLM response to the OpenAI-wire ``Message`` stored in history.

    Returns a plain OpenAI-wire ``Message`` dict. ``reasoning_content`` is
    deliberately omitted from the wire message except for the
    ``reasoning_content`` format — see the leak-guard note above.
    """

    def to_history(
        self,
        response: LLMResponse,
        thinking_result: ThinkingResult,
        policy: HistoryPolicy,
        thinking_format: str = "none",
    ) -> Message:
        # Native Anthropic thinking / OpenAI Responses reasoning: when the
        # response carried a verbatim content-block list (thinking+signature /
        # redacted_thinking / reasoning+encrypted_content / text), keep it INTACT
        # as the history message content. It is re-sent UNMODIFIED next turn — the
        # provider validates signatures / encrypted_content server-side — so it
        # must survive independently of the agent's thinking_in_history flag
        # (verbatim replay is transport correctness, not an agent choice). No
        # effect on other formats, where raw_content_blocks is None.
        if thinking_result.raw_content_blocks is not None:
            tool_calls = _to_openai_tool_calls(thinking_result.tool_calls)
            return assistant_msg(
                thinking_result.raw_content_blocks, tool_calls=tool_calls,
            )

        # The parser is the source of truth for visible text. Reusing raw
        # ``response.content`` when it contains inline <think> tags makes the
        # builder below prepend the same reasoning a second time.
        visible = thinking_result.visible_content
        tool_calls = _to_openai_tool_calls(thinking_result.tool_calls)
        reasoning = ""
        if policy.thinking_in_history:
            reasoning = (
                thinking_result.thinking
                or getattr(response, "reasoning_content", "")
                or ""
            )
            if reasoning and policy.thinking_history_max_tokens:
                reasoning = _cap_reasoning_tail(
                    reasoning,
                    policy.thinking_history_max_tokens,
                )

        # Wire round-trip of reasoning, by format — this is where the kernel
        # owns the outbound reasoning shape (previously scattered across
        # per-workflow ChatOpenAI reasoning subclasses). ``reasoning_content``
        # is NEVER emitted as a bare wire field except in the one format that
        # requires it. The per-format rules live in
        # ``assistant_msg_with_reasoning`` (core.messages) so self-contained
        # workflow loops that bypass this normalizer share one implementation.
        return assistant_msg_with_reasoning(
            visible, reasoning,
            tool_calls=tool_calls, thinking_format=thinking_format,
        )


def _cap_reasoning_tail(reasoning: str, max_tokens: int) -> str:
    """Keep the most recent reasoning of ONE turn within ``max_tokens``.

    Conclusions and tool-use intent tend to occur at the end of a reasoning
    trace, so the cap discards the oldest prefix. Token estimation uses the
    loop's non-blocking tokenizer with its CJK-aware fallback; ``estimator`` is
    passed explicitly rather than relying on the default so tests can swap the
    module-level estimator.
    """
    from .context_budget import estimate_tokens, truncate_text_to_tokens

    if max_tokens <= 0:
        return reasoning
    capped = truncate_text_to_tokens(
        reasoning,
        max_tokens,
        marker="[... earlier reasoning truncated by the per-turn history token cap ...]\n",
        estimator=estimate_tokens,
        keep="tail",
    )
    if capped != reasoning:
        logger.debug(
            "thinking history: capped this turn's reasoning to %d tokens "
            "(%d chars kept of %d)",
            max_tokens, len(capped), len(reasoning),
        )
    return capped
