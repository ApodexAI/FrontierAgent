from __future__ import annotations

import logging
import re
from typing import Any

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.messages import Message

logger = logging.getLogger(__name__)

# Inlined ``<think>…</think>`` blocks may be carried through history (so
# the model sees its prior reasoning on the next turn) but must never
# surface as a final answer. Stripped at the answer-extraction site.
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")
_DANGLING_THINK_RE = re.compile(r"<think>[\s\S]*\Z")


def _strip_thinking_blocks(text: str) -> str:
    """Strip inlined ``<think>`` from a model-facing answer.

    Handles closed pairs, unclosed openers, and the SGLang
    ``preserve_thinking`` quirk where the closing tag is emitted without
    an opener — everything before the last ``</think>`` is the thinking
    trace and must be stripped, not just the tag character.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _DANGLING_THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _flatten_message_text(content: Any) -> str:
    """Collapse an ``AIMessage.content`` (str | list of str/text blocks |
    other) into plain text. Thinking blocks are NOT stripped here."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def _visible_response_text(response: Any) -> str:
    """The model-facing answer text of one response, thinking stripped."""
    return _strip_thinking_blocks(_flatten_message_text(getattr(response, "content", "")))

def extract_final_content(messages: list[Message]) -> str:
    """Find the most recent assistant message with non-empty visible text.

    Walks backward past empty AIMessages so we surface the last
    meaningful answer instead of an empty shell (common when the final
    turn was a tool-only call or a safety-driven empty reply). Inlined
    ``<think>…</think>`` blocks are stripped — they're history-only
    reasoning and must never reach a downstream judge as the answer.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        cleaned = _strip_thinking_blocks(_flatten_message_text(msg.get("content", "")))
        if cleaned:
            return cleaned
    return ""


def extract_leaked_reasoning(response: Any) -> str:
    """Pull reasoning recovered from leaked think/reasoning tags, if any.

    ``MultiFormatToolCallParser`` stashes the salvaged inner text on
    ``response.additional_kwargs[LEAKED_REASONING_KEY]`` when it strips
    out leaked ``<think_never_used_…>`` / ``<model:thinking>`` /
    ``<think>`` blocks during ``parse()``. Mirroring it onto
    ``TurnContext.leaked_reasoning`` lets observers surface it without
    grovelling through the raw response.
    """
    from frontier_agent.core.runtime.loop.tool_call_parser import LEAKED_REASONING_KEY
    meta = getattr(response, "response_metadata", None) or {}
    value = meta.get(LEAKED_REASONING_KEY, "")
    return value if isinstance(value, str) else ""

def _pick_int(*candidates: Any) -> int:
    """Return the first non-zero int-coercible candidate, else 0.

    ``None`` and unparseable values are skipped (so the next candidate
    is tried), matching the ``... or ... or 0`` chains this replaces
    while tolerating gateway-side ``null`` (seen on aggregating gateways
    for ``cached_tokens``: the key is present but the value is JSON null,
    which ``dict.get`` returns as ``None``).

    ``0`` is treated as "no signal" so a missing field can fall through
    to a real source — same semantics as the chained ``or``. This is
    safe because the downstream billing rollup sums per-call ints; the
    only effect of picking 0 from candidate A over 0 from candidate B
    is which provider name lands in the audit log, not the total.
    """
    for v in candidates:
        if v is None:
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n:
            return n
    return 0


def extract_usage(response: Any) -> dict | None:
    """Extract token usage from an LLM response, normalized to OpenAI shape.

    Returns a dict with keys ``provider`` / ``model`` / ``prompt_tokens`` /
    ``completion_tokens`` / ``cache_read_tokens`` / ``cache_write_tokens`` /
    ``cached_tokens`` / ``cache_creation_tokens`` / ``reasoning_tokens``
    (the shape consumed by the protocol stream and worker-trace observers),
    or ``None`` when the response carries
    no usage info. Handles both LangChain ``usage_metadata`` (canonical
    ``input_tokens`` / ``output_tokens``, no ``model`` key — that lives in
    ``response_metadata.model_name``) and the OpenAI raw
    ``response_metadata.token_usage`` shape.

    ``provider`` is sourced from ``response_metadata.provider_actually_used``
    (stamped by ``LLMFallbackChain`` per attempt). Empty string when the
    construction path didn't stamp it (e.g. a plain ``ChatOpenAI`` with no
    fallback chain wrapper). Downstream billing should treat ``""`` as
    "vendor unknown — fall back to whatever the model id implies".

    Cache token fields:

    - ``cache_read_tokens`` — cache READ (a.k.a. cache hit). Bills at
      ~0.1× base input on Anthropic, free on OpenAI.
    - ``cache_write_tokens`` — cache WRITE (cache creation), summing
      Anthropic's 5m-TTL and 1h-TTL counts (5m ~1.25×, 1h ~2×). Only
      Anthropic exposes write; OpenAI returns 0 here.
    - ``cached_tokens`` — backward-compat alias = ``read + write``.
      Pre-split this name was cache-read-only; cost boards using it
      with a single rate would have under-attributed Anthropic write
      spend, which is what motivated this split.
    - ``cache_creation_tokens`` — backward-compat alias of
      ``cache_write_tokens`` (deprecated; prefer the new name).

    ``reasoning_tokens`` captures OpenAI o-series / Gemini thinking
    output, billed as completion tokens but worth surfacing separately.
    """
    # Native path: ``LLMResponse.usage`` is already normalised by the client
    # adapter (prompt/completion/total/cached_tokens), so read it directly.
    # The langchain ``usage_metadata`` / ``token_usage`` parsing below is
    # retained as a fallback for compatible legacy response objects.
    if isinstance(response, LLMResponse):
        usage = response.usage or {}
        inp = int(usage.get("prompt_tokens", 0) or 0)
        out = int(usage.get("completion_tokens", 0) or 0)
        if not inp and not out:
            return None
        # Vendor label stamped by ``LLMFallbackChain`` (``_stamp_metadata`` on
        # the non-streaming path; ``_stream_llm_response`` folds the streamed
        # ``StreamDelta.provider`` here). Empty when the client was built
        # without a provider-stamp wrapper — downstream billing treats ""
        # as "vendor unknown", same as the langchain branch below.
        rmd = response.response_metadata or {}
        provider = str(rmd.get("provider_actually_used") or "") if isinstance(
            rmd, dict,
        ) else ""
        if "cache_read_tokens" in usage or "cache_write_tokens" in usage:
            cache_read = int(usage.get("cache_read_tokens", 0) or 0)
            cache_write = int(usage.get("cache_write_tokens", 0) or 0)
        else:
            # Backward compatibility for native adapters that still expose
            # the pre-split cache fields.
            cache_read = int(usage.get("cached_tokens", 0) or 0)
            cache_write = int(usage.get("cache_creation_tokens", 0) or 0)
        out_dict = {
            "provider": provider,
            "model": response.model or "",
            "prompt_tokens": inp,
            "completion_tokens": out,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "cached_tokens": cache_read + cache_write,
            "cache_creation_tokens": cache_write,
        }
        # Reasoning/thinking tokens (Anthropic extended thinking / OpenAI
        # reasoning models). They are part of completion_tokens but surfaced
        # separately for cost / analysis; the client's usage dict carries them.
        # Omitted (backward-compat) when absent/zero.
        reasoning = int(usage.get("reasoning_tokens", 0) or 0)
        if reasoning:
            out_dict["reasoning_tokens"] = reasoning
        return out_dict

    rmd = getattr(response, "response_metadata", None) or {}
    if not isinstance(rmd, dict):
        rmd = {}
    # ``model_actually_used`` is stamped by ``LLMFallbackChain`` and is
    # the only model identifier present on streaming chunks (the
    # provider's own ``model_name`` lands on ``ainvoke`` responses but
    # not on ``astream`` usage chunks). Falling through to it keeps
    # streaming usage attribution alive.
    model = (
        rmd.get("model_name")
        or rmd.get("model")
        or rmd.get("model_actually_used")
        or ""
    )
    provider = str(rmd.get("provider_actually_used") or "")

    def _build(inp: int, out: int, cached: int,
               cache_create: int, reasoning: int) -> dict:
        # ``cached`` carries cache READ; ``cache_create`` carries cache
        # WRITE. The legacy ``cached_tokens`` / ``cache_creation_tokens``
        # keys are kept as a derived sum and an alias respectively so
        # existing consumers don't break — see module docstring on the
        # New consumers should read the explicit
        # ``cache_read_tokens`` / ``cache_write_tokens`` keys.
        cache_read = int(cached or 0)
        cache_write = int(cache_create or 0)
        return {
            "provider": provider,
            "model": model,
            "prompt_tokens": int(inp or 0),
            "completion_tokens": int(out or 0),
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            # Backward-compat: sum is the intuitive read of "cached"
            # for cost boards using a single field.
            "cached_tokens": cache_read + cache_write,
            # Backward-compat alias for legacy callers; identical to
            # ``cache_write_tokens`` (deprecated, will be removed in
            # a future cleanup once all consumers migrate).
            "cache_creation_tokens": cache_write,
            "reasoning_tokens": int(reasoning or 0),
        }

    # LangChain canonical shape (input_tokens / output_tokens).
    um = getattr(response, "usage_metadata", None)
    if um is not None and not isinstance(um, dict):
        try:
            um = dict(um)
        except (TypeError, ValueError):
            um = None
    if isinstance(um, dict):
        idetails = um.get("input_token_details") or {}
        odetails = um.get("output_token_details") or {}
        if not isinstance(idetails, dict):
            idetails = {}
        if not isinstance(odetails, dict):
            odetails = {}
        inp = _pick_int(um.get("input_tokens"), um.get("prompt_tokens"))
        out = _pick_int(um.get("output_tokens"), um.get("completion_tokens"))
        cached = _pick_int(idetails.get("cache_read"))
        cache_create = _pick_int(idetails.get("cache_creation"))
        reasoning = _pick_int(odetails.get("reasoning"))
        if inp or out:
            # Apodex (and some other OpenAI-compatible gateways) on
            # non-streaming ``ainvoke`` populate ``input_tokens`` /
            # ``output_tokens`` on the canonical map but leave
            # ``input_token_details`` empty — cache hits only show up
            # on the raw ``prompt_tokens_details.cached_tokens`` field.
            # Cross-check the raw shape when the canonical pass came
            # back zero so streaming-vs-ainvoke don't silently disagree
            # on cached token attribution. Same fix lives in
            # ``infra/usage.py`` for the SDK aux path.
            if not cached or not cache_create or not reasoning:
                tu_raw = rmd.get("token_usage") or rmd.get("usage")
                if isinstance(tu_raw, dict):
                    ptd_raw = tu_raw.get("prompt_tokens_details") or {}
                    ctd_raw = tu_raw.get("completion_tokens_details") or {}
                    if not isinstance(ptd_raw, dict):
                        ptd_raw = {}
                    if not isinstance(ctd_raw, dict):
                        ctd_raw = {}
                    if not cached:
                        cached = _pick_int(
                            ptd_raw.get("cached_tokens"),
                            tu_raw.get("cache_read_input_tokens"),
                            # Symmetric with the write path below: some
                            # gateways nest the Anthropic READ key
                            # under prompt_tokens_details, not at root.
                            ptd_raw.get("cache_read_input_tokens"),
                            ptd_raw.get("cache_read_tokens"),
                            tu_raw.get("cache_read_tokens"),
                        )
                    if not cache_create:
                        cache_create = _pick_int(
                            tu_raw.get("cache_creation_input_tokens"),
                            # Apodex/qwen nest the Anthropic write key
                            # under prompt_tokens_details — see comments
                            # in the raw-shape branch below for the
                            # full alias list.
                            ptd_raw.get("cache_creation_input_tokens"),
                            ptd_raw.get("cache_creation_tokens"),
                            ptd_raw.get("cache_write_tokens"),
                            tu_raw.get("cache_write_tokens"),
                        )
                    if not reasoning:
                        reasoning = _pick_int(
                            ctd_raw.get("reasoning_tokens"),
                            tu_raw.get("reasoning_tokens"),
                        )
            return _build(inp, out, cached, cache_create, reasoning)

    # OpenAI raw shape (response_metadata.token_usage / usage).
    tu = rmd.get("token_usage") or rmd.get("usage")
    if isinstance(tu, dict):
        ptd = tu.get("prompt_tokens_details") or {}
        ctd = tu.get("completion_tokens_details") or {}
        if not isinstance(ptd, dict):
            ptd = {}
        if not isinstance(ctd, dict):
            ctd = {}
        inp = _pick_int(tu.get("prompt_tokens"), tu.get("input_tokens"))
        out = _pick_int(tu.get("completion_tokens"), tu.get("output_tokens"))
        # Cache READ (cache-hit tokens). Field name varies by provider:
        # - OpenAI / OpenAI-compatible:  ptd.cached_tokens
        # - Anthropic direct (Messages API): tu.cache_read_input_tokens
        #   at the usage root, *not* nested under prompt_tokens_details
        # - Apodex / bedrock via OpenAIClient gateway nest the Anthropic
        #   READ key UNDER prompt_tokens_details — mirror of the write
        #   path's ptd.cache_creation_input_tokens candidate below. Without
        #   ptd.cache_read_input_tokens the read count silently dropped to
        #   0 on every bedrock-via-gateway call while write was captured
        #   so reads are not silently lost when writes are present.
        # - Some custom gateways flatten: ptd.cache_read_tokens or
        #   tu.cache_read_tokens
        cached = _pick_int(
            ptd.get("cached_tokens"),
            tu.get("cache_read_input_tokens"),
            ptd.get("cache_read_input_tokens"),  # bedrock/apodex nested shape
            ptd.get("cache_read_tokens"),
            tu.get("cache_read_tokens"),
        )
        # Cache WRITE (cache-creation tokens). Field name varies:
        # - Anthropic direct: tu.cache_creation_input_tokens at root
        # - OpenAI-compatible nested (no _input_ infix):
        #   ptd.cache_creation_tokens
        # - Apodex / qwen3.5 / some custom gateways nest the Anthropic
        #   name UNDER prompt_tokens_details — same key, different
        #   parent. Observed shape (2026-05):
        #     usage: { prompt_tokens_details: {
        #       cached_tokens: ..., cache_creation_input_tokens: ... }
        #     }
        #   Without this candidate the write count was silently dropped
        #   on every apodex non-streaming call (DAG analyzer / synth
        #   / decision_llm), under-attributing write spend on Claude
        #   served through the apodex gateway.
        # - OpenRouter passthrough alias: ptd.cache_write_tokens
        #   (some wrappers drop Anthropic's standard names for this alias)
        # - Some custom gateways flatten: tu.cache_write_tokens
        cache_create = _pick_int(
            tu.get("cache_creation_input_tokens"),
            ptd.get("cache_creation_input_tokens"),  # apodex/qwen shape
            ptd.get("cache_creation_tokens"),
            ptd.get("cache_write_tokens"),
            tu.get("cache_write_tokens"),
        )
        # Anthropic 1h-TTL extension (extended prompt-cache, ~2× base
        # rate vs 5m's ~1.25×) surfaces under a nested ``cache_creation``
        # dict alongside the 5m count. Both bill as write, just at
        # different rates — sum them so the schema field captures the
        # full write footprint. Cost boards needing per-TTL breakdown
        # should consume the raw provider response directly.
        #
        # **Provider scope**: the 1h-TTL extension is Anthropic-direct
        # only as of 2026-05; Bedrock supports only the 5m TTL and
        # omits the nested ``cache_creation`` dict entirely, so this
        # branch is a no-op there (gracefully degrades to just the
        # 5m count read above).
        cc_nested = tu.get("cache_creation")
        if isinstance(cc_nested, dict):
            cache_create += _pick_int(
                cc_nested.get("ephemeral_1h_input_tokens"),
            )
            # If the root ``cache_creation_input_tokens`` was absent but
            # the 5m count is nested here, pick it up. Guard against
            # double-counting when both root + nested are populated.
            if not tu.get("cache_creation_input_tokens"):
                cache_create += _pick_int(
                    cc_nested.get("ephemeral_5m_input_tokens"),
                )
        # Reasoning tokens (o-series / Gemini thinking / qwen thinking
        # via aliyun gateway). Standard location is nested under
        # completion_tokens_details, but some gateways flatten to root.
        reasoning = _pick_int(
            ctd.get("reasoning_tokens"),
            tu.get("reasoning_tokens"),
        )
        if inp or out:
            return _build(inp, out, cached, cache_create, reasoning)

    return None


def extract_model_name(
    llm: Any, profile: dict[str, Any] | None = None,
) -> str:
    """Best-effort model id from a YAML profile or LLM attribute.

    Resolution order:

    1. ``profile["llm"]["model"]`` if a profile dict was passed (workflow
       YAML profiles are the authoritative source — they're what the
       benchmark run was configured with).
    2. Common LangChain attributes on the bound LLM
       (``model_name`` / ``model`` / ``model_id``) — covers OpenAI,
       Anthropic, Qwen alike.

    Returns ``""`` when nothing identifies the model — observers treat
    empty as "omit the field" rather than recording an empty string.
    """
    if profile:
        name = (profile.get("llm") or {}).get("model")
        if isinstance(name, str) and name:
            return name
    for attr in ("model_name", "model", "model_id"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""
