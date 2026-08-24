"""Generic tool call parser: native function-calling + JSON text fallback."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Compiled once at import time for performance.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_MCP_USE_TOOL_RE = re.compile(
    r"<use_mcp_tool>\s*(.*?)\s*</use_mcp_tool>",
    re.DOTALL | re.IGNORECASE,
)
# Dangling ``<tool_call>{...`` — closing tag (or trailing brace) lost to
# a ``max_tokens`` truncation. Captures the JSON body from the first
# unmatched opening; brace-balancing happens in
# ``_parse_dangling_json_tool_call``.
_DANGLING_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{[\s\S]*)\Z", re.DOTALL,
)

# XML fallback patterns for models that leak tool calls as text.
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_QWEN_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
_SEED_FUNCTION_RE = re.compile(r'<function\s+name="(\w+)">(.*?)</function>', re.DOTALL)
_SEED_PARAM_RE = re.compile(
    r'<parameter\s+name="(\w+)"[^>]*>(.*?)</parameter>', re.DOTALL
)
_SEED_TAG_HINT_RE = re.compile(r"<seed[:\w]*tool")

# Seed reasoning-mode FunctionCall wrapper. Body is a JSON array of
# ``{"name": str, "parameters": dict}`` objects. Emitted by the model
# verbatim when an upstream proxy doesn't repack it into native
# ``response.tool_calls``.
_FC_WRAPPED_RE = re.compile(
    r"<\|FunctionCallBegin\|>\s*(\[.*?\])\s*<\|FunctionCallEnd\|>", re.DOTALL,
)

# Leaked inner-monologue / reasoning tags that some models emit into
# ``content`` even though they're supposed to be private. These fragments
# confuse downstream parsers and pollute the visible conversation
# history — we strip them defensively before any parser sees the text.
_LEAKED_TAG_RE = re.compile(
    r"<(?:think|thinking|reasoning|seed:think|seed:reasoning)>"
    r"(.*?)"
    r"</(?:think|thinking|reasoning|seed:think|seed:reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
# Open tag with no matching close → trim from the open tag onward.
_DANGLING_LEAKED_TAG_RE = re.compile(
    r"<(?:think|thinking|reasoning|seed:think|seed:reasoning)>(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)
# Thinking-tag variants Seed/GPT-OSS emit despite the token never making it
# into the tokenizer vocabulary. Covered separately so the inner content can
# be captured as reasoning rather than silently discarded.
_NEVER_USED_TAG_RE = re.compile(
    r"<think_never_used[^>]*>(.*?)</think_never_used[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_NEVER_USED_DANGLING_RE = re.compile(
    r"<think_never_used[^>]*>(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)
_MODEL_THINKING_RE = re.compile(
    r"<model:\s*thinking>(.*?)</model:\s*thinking>",
    re.DOTALL | re.IGNORECASE,
)

# Key used to stash salvaged reasoning on AIMessage.additional_kwargs so
# downstream consumers (trace logger, SSE observer, UI) can render it.
LEAKED_REASONING_KEY = "leaked_reasoning"


@runtime_checkable
class ToolCallParser(Protocol):
    """Protocol for tool call parsers."""

    def parse(self, response: Any, tool_names: set[str]) -> list[dict]:
        """Extract tool calls from an LLM response.

        Args:
            response: An AIMessage-like object with optional ``.tool_calls``
                      and ``.content`` attributes.
            tool_names: Set of known/allowed tool names. Calls whose name is
                        not in this set are filtered out.

        Returns:
            List of dicts with at least ``"name"`` and ``"args"`` keys,
            matching the LangChain tool-call dict structure.
        """
        ...


def _normalize_native_tool_call(tc: Any) -> dict | None:
    """Normalise one native tool_call into the executor's ``{name, args, id}``.

    Accepts the OpenAI wire shape ``{id, type, function:{name, arguments}}``
    (a native ``LLMResponse`` carries this — ``arguments`` is a JSON string)
    and the already-parsed langchain shape ``{name, args(dict), id}``. Returns
    ``None`` for anything unrecognised.
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function")
    if isinstance(fn, dict):
        raw_args = fn.get("arguments", "")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except (ValueError, TypeError):
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return {
            "name": fn.get("name", "") or "",
            "args": args if isinstance(args, dict) else {},
            "id": tc.get("id", "") or "",
        }
    # Already-parsed shape (legacy langchain AIMessage.tool_calls).
    return {
        "name": tc.get("name", "") or "",
        "args": tc.get("args", {}) or {},
        "id": tc.get("id", "") or "",
    }


class DefaultToolCallParser:
    """Two-strategy tool call parser (native FC first, JSON fallback second)."""

    def parse(self, response: Any, tool_names: set[str]) -> list[dict]:
        # ── Strategy 1: Native function calling ──────────────────────────────
        # ``response.tool_calls`` is OpenAI wire shape on a native
        # ``LLMResponse`` (``{id, type, function:{name, arguments}}``) and the
        # parsed ``{name, args, id}`` shape on a legacy langchain message;
        # ``_normalize_native_tool_call`` accepts either and emits the parsed
        # shape the executor consumes.
        native_raw = list(getattr(response, "tool_calls", None) or [])
        native = [
            n for n in (_normalize_native_tool_call(tc) for tc in native_raw) if n
        ]
        if native:
            known_calls = [tc for tc in native if tc.get("name") in tool_names]
            logger.debug(
                "native FC: %d total, %d known", len(native), len(known_calls),
            )
            if known_calls:
                # When the model emitted at least one executable action, drop
                # unknown companions instead of turning an otherwise useful
                # parallel batch into a visible failure. If every call is
                # unknown, keep them so the executor can return an explicit
                # correction; returning an empty list there could be mistaken
                # for a completed no-tool turn.
                #
                # A dropped call still occupies a ``tool_call_id`` in the
                # assistant history message, which the agent loop wrote from
                # the raw response before calling us. ``_answer_dropped_tool_calls``
                # there answers those ids — without it an orphan id is a hard
                # HTTP 400 on Azure and other providers.
                unknown = [
                    tc.get("name", "")
                    for tc in native
                    if tc.get("name") not in tool_names
                ]
                if unknown:
                    logger.warning(
                        "native FC: dropped unknown companion tool calls: %s",
                        unknown,
                    )
                return known_calls
            return native

        # ── Strategy 2: JSON text fallback ───────────────────────────────────
        raw_content = getattr(response, "content", "") or ""
        text = self._extract_text(raw_content)
        parsed = self._parse_json_tool_calls(text, tool_names)
        if parsed:
            return parsed
        return self._parse_mcp_tool_calls(text, tool_names)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalise content to a plain string.

        Handles:
        - ``str`` — returned as-is.
        - ``list`` of blocks (Anthropic / LangChain format) — text blocks
          are concatenated.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content) if content else ""

    @staticmethod
    def _parse_json_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
        """Find all <tool_call>…</tool_call> blocks and parse each as JSON."""
        results: list[dict] = []
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Skipping malformed JSON tool_call block: %r", raw[:120])
                continue

            if not isinstance(payload, dict):
                logger.debug("Skipping non-object tool_call payload: %r", payload)
                continue

            name = str(payload.get("tool", "") or "").strip()
            if not name or name not in tool_names:
                logger.debug("Skipping unknown/missing tool name: %r", name)
                continue

            args = payload.get("args", {})
            if not isinstance(args, dict):
                args = {}

            results.append({"name": name, "args": args})

        logger.debug("JSON fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _parse_mcp_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
        """Parse Agent-Protocol style <use_mcp_tool> blocks."""
        if "use_mcp_tool" not in tool_names:
            return []

        results: list[dict] = []
        for idx, match in enumerate(_MCP_USE_TOOL_RE.finditer(text)):
            body = match.group(1)
            server_name = _extract_xml_value(body, "server_name")
            tool_name = _extract_xml_value(body, "tool_name")
            if not server_name or not tool_name:
                logger.debug("Skipping malformed use_mcp_tool block: %r", body[:120])
                continue

            raw_args = _extract_xml_value(body, "arguments")
            arguments: dict[str, Any] = {}
            if raw_args:
                try:
                    parsed = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    logger.debug(
                        "use_mcp_tool arguments were not JSON: %r",
                        raw_args[:120],
                    )
                    parsed = {}
                if isinstance(parsed, dict):
                    arguments = parsed

            results.append({
                "name": "use_mcp_tool",
                "args": {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                "id": f"mcp_tc_{idx}",
            })

        logger.debug("MCP XML fallback: found %d tool calls", len(results))
        return results


class MultiFormatToolCallParser(DefaultToolCallParser):
    """Default parser + Qwen/Seed XML + Seed FunctionCall wrapper fallbacks.

    Priority order:
    1. Native ``response.tool_calls`` (via parent)
    2. JSON ``<tool_call>{"tool": …}</tool_call>`` (via parent)
    3. Seed FunctionCall wrapper ``<|FunctionCallBegin|>[…]<|FunctionCallEnd|>``
    4. Qwen XML ``<tool_call><function=name>…</function></tool_call>``
    5. Seed XML ``<function name="name">…</function>``

    Strategies 3-5 only fire when there is no native ``tool_calls`` field —
    an empty native list (all filtered out) is treated as a deliberate signal
    and short-circuits the fallbacks, matching the base parser's semantics.
    The FunctionCall wrapper is the exception: an unambiguous Seed-specific
    marker overrides truthy-but-malformed native ``tool_calls`` from upstream
    proxy repack failures.

    Leaked ``<think>…</think>`` / ``<reasoning>…</reasoning>`` tags are
    stripped from ``response.content`` before parsing.

    Use :meth:`parse_text` to recover leaks from arbitrary text that isn't
    on ``response.content`` (e.g. a model's native ``<think>`` block when it
    wrote tool calls inside its private reasoning instead of using the
    visible content channel).
    """

    def parse(self, response: Any, tool_names: set[str]) -> list[dict]:
        # Response-level cleaning: strip leaked <think>/<reasoning> tags
        # in place before any parsing. Safe — these tags are private
        # inner-monologue that should never have been emitted.
        self._clean_leaked_content(response)

        base = super().parse(response, tool_names)
        if base:
            return base

        text = self._extract_text(getattr(response, "content", "") or "")

        # FunctionCall wrapper has higher priority than the native short-
        # circuit: this marker is unambiguous (no false-positive matches in
        # real prose) and we trust it over a possibly-empty native list
        # produced by an upstream proxy that failed to repack it. Side-
        # effect strip keeps history clean — that's why this branch lives
        # here in ``parse()`` rather than the side-effect-free
        # :meth:`parse_text`.
        if text and "<|FunctionCallBegin|>" in text:
            fc = _parse_fc_wrapped(text, tool_names)
            if fc:
                self._strip_fc_wrappers(response)
                return fc

        # Native API was used (even with unknown names) — don't second-guess
        # with XML regexes on residual content.
        if getattr(response, "tool_calls", None):
            return base

        return self.parse_text(text, tool_names)

    def parse_text(self, text: str, tool_names: set[str]) -> list[dict]:
        """Recover tool calls from a raw text string (pure, no side effects).

        Used both as the content fallback in :meth:`parse` and to recover
        leaks from a model's thinking block — Qwen 3.5 35B writes Hermes XML
        inside ``<think>`` instead of using native ``tool_calls``. Returns
        ``[]`` when no recognisable leak markers are present.
        """
        if not text:
            return []
        if "<use_mcp_tool>" in text and (
            calls := self._parse_mcp_tool_calls(text, tool_names)
        ):
            return calls
        if (
            "<tool_call>" in text
            and "<function=" in text
            and (calls := self._parse_qwen_xml(text, tool_names))
        ):
            return calls
        if "<|FunctionCallBegin|>" in text and (
            calls := _parse_fc_wrapped(text, tool_names)
        ):
            return calls
        # Seed XML returns directly even when empty — once we recognise the
        # format we don't fall through to other parsers.
        if '<function name="' in text and (
            '<parameter name="' in text or _SEED_TAG_HINT_RE.search(text)
        ):
            return self._parse_seed_xml(text, tool_names)
        # Last-resort: ``<tool_call>{...`` truncated mid-stream by
        # ``max_tokens`` (closing ``</tool_call>`` lost). Try brace-
        # balanced JSON extraction. Only fires when nothing else matched
        # — guarded by an explicit check so we don't pay the cost on the
        # common path.
        if "<tool_call>" in text and "</tool_call>" not in text:
            recovered = _parse_dangling_json_tool_call(text, tool_names)
            if recovered:
                return recovered
        return []

    @staticmethod
    def _clean_leaked_content(response: Any) -> None:
        """Strip leaked ``<think>``/``<reasoning>`` tags from response.content
        while preserving the extracted inner content as reasoning.

        Mutates the response in place: visible content loses the tag blocks,
        and any recovered thinking/reasoning text is concatenated onto
        ``response.additional_kwargs[LEAKED_REASONING_KEY]`` so downstream
        consumers (trace logger, SSE UI, evidence observer) can display it.
        """
        content = getattr(response, "content", None)
        if content is None:
            return

        recovered_parts: list[str] = []

        if isinstance(content, str):
            cleaned, reasoning = extract_leaked_reasoning(content)
            if reasoning:
                recovered_parts.append(reasoning)
            if cleaned != content:
                # A frozen/immutable response shouldn't happen here, but must
                # not crash the parse if it does.
                with contextlib.suppress(Exception):
                    response.content = cleaned
        elif isinstance(content, list):
            # LangChain content-blocks list: each block may be a str or dict.
            changed = False
            new_blocks: list[Any] = []
            for block in content:
                if isinstance(block, str):
                    clean, reasoning = extract_leaked_reasoning(block)
                    if reasoning:
                        recovered_parts.append(reasoning)
                    changed = changed or clean != block
                    new_blocks.append(clean)
                elif isinstance(block, dict) and isinstance(
                    block.get("text"), str,
                ):
                    clean, reasoning = extract_leaked_reasoning(block["text"])
                    if reasoning:
                        recovered_parts.append(reasoning)
                    b2 = dict(block)
                    b2["text"] = clean
                    changed = changed or b2["text"] != block["text"]
                    new_blocks.append(b2)
                else:
                    new_blocks.append(block)
            if changed:
                with contextlib.suppress(Exception):
                    response.content = new_blocks

        if recovered_parts:
            _attach_leaked_reasoning(response, "\n\n".join(recovered_parts))

    @staticmethod
    def _parse_qwen_xml(text: str, tool_names: set[str]) -> list[dict]:
        results: list[dict] = []
        for i, match in enumerate(_QWEN_TOOL_CALL_RE.finditer(text)):
            name = match.group(1)
            if name not in tool_names:
                logger.debug("Qwen XML: skipping unknown tool %r", name)
                continue
            body = match.group(2)
            args: dict = {}
            for pm in _QWEN_PARAM_RE.finditer(body):
                key = pm.group(1)
                args[key] = _coerce_param_value(pm.group(2).strip())
            results.append({"name": name, "args": args, "id": f"qwen_tc_{i}"})
        logger.debug("Qwen XML fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _parse_seed_xml(text: str, tool_names: set[str]) -> list[dict]:
        results: list[dict] = []
        for i, match in enumerate(_SEED_FUNCTION_RE.finditer(text)):
            name = match.group(1)
            if name not in tool_names:
                logger.debug("Seed XML: skipping unknown tool %r", name)
                continue
            body = match.group(2)
            args: dict = {}
            for pm in _SEED_PARAM_RE.finditer(body):
                key = pm.group(1)
                args[key] = _coerce_param_value(pm.group(2).strip())
            results.append({"name": name, "args": args, "id": f"seed_tc_{i}"})
        logger.debug("Seed XML fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _strip_fc_wrappers(response: Any) -> None:
        """Strip ``<|FunctionCallBegin|>…<|FunctionCallEnd|>`` from
        ``response.content`` after we've parsed the call out, so the
        message history doesn't carry the duplicate wire format."""
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return
        cleaned = _FC_WRAPPED_RE.sub("", content).strip()
        if cleaned != content:
            with contextlib.suppress(Exception):
                response.content = cleaned


def _balance_json_object(text: str, start: int) -> int | None:
    """Return the index one past the matching ``}`` for ``text[start] == '{'``.

    Brace-counts depth while respecting JSON string semantics (``"..."``
    with ``\\`` escapes) so braces inside strings don't bump the depth.
    Returns ``None`` when the object never closes — i.e. truncation
    happened inside the JSON.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _parse_dangling_json_tool_call(
    text: str, tool_names: set[str],
) -> list[dict]:
    """Recover a single ``<tool_call>{...}`` whose closing tag was
    truncated by ``max_tokens``.

    Fires only when ``<tool_call>`` is present and ``</tool_call>`` is
    not. Brace-balances the JSON body from the first ``{`` after the
    opening tag; bails out when the body is genuinely incomplete
    (truncation inside a string). At most one call is recovered — the
    truncation point is by definition the end of useful content, so any
    later calls in the same response don't exist.
    """
    match = _DANGLING_TOOL_CALL_RE.search(text)
    if not match:
        return []
    body_start = match.start(1)
    end = _balance_json_object(text, body_start)
    if end is None:
        logger.debug(
            "Dangling <tool_call>: JSON body truncated mid-value, "
            "cannot recover. preview=%r", text[body_start:body_start + 120],
        )
        return []
    raw = text[body_start:end]
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug(
            "Dangling <tool_call>: balanced body did not parse "
            "(%s). preview=%r", exc, raw[:120],
        )
        return []
    if not isinstance(payload, dict):
        return []
    name = str(payload.get("tool", "") or "").strip()
    if not name or name not in tool_names:
        logger.debug(
            "Dangling <tool_call>: unknown or missing tool %r", name,
        )
        return []
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}
    logger.info(
        "Recovered dangling <tool_call> for %r (lost </tool_call>; "
        "%d-byte body)", name, end - body_start,
    )
    return [{"name": name, "args": args, "id": "dangling_tc_0"}]


def _parse_fc_wrapped(text: str, tool_names: set[str]) -> list[dict]:
    """Parse Seed reasoning-mode ``<|FunctionCallBegin|>[…]<|FunctionCallEnd|>``.

    Body is a JSON array of ``{"name": str, "parameters": dict}`` objects.
    """
    results: list[dict] = []
    idx = 0
    for match in _FC_WRAPPED_RE.finditer(text):
        try:
            calls = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "FunctionCall wrapper: bad JSON (%s) body_preview=%r",
                exc, match.group(1)[:200],
            )
            continue
        if not isinstance(calls, list):
            logger.warning(
                "FunctionCall wrapper: body not a list, got %s",
                type(calls).__name__,
            )
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            if name not in tool_names:
                logger.warning(
                    "FunctionCall wrapper: skipping unknown tool %r "
                    "(allowed=%s)", name, sorted(tool_names),
                )
                continue
            args = call.get("parameters") or call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            results.append({"name": name, "args": args, "id": f"fc_tc_{idx}"})
            idx += 1
    logger.debug("FunctionCall wrapper: found %d tool calls", len(results))
    return results


def _extract_xml_value(body: str, tag: str) -> str:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _coerce_param_value(raw: str) -> Any:
    """Try to decode a parameter value as JSON; fall back to the raw string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _strip_leaked_tags(text: str) -> str:
    """Remove leaked inner-monologue / reasoning tags from model output."""
    cleaned, _ = extract_leaked_reasoning(text)
    return cleaned


def _attach_leaked_reasoning(response: Any, reasoning: str) -> None:
    """Append recovered reasoning to ``response.response_metadata``.

    Native ``LLMResponse`` carries salvaged reasoning on ``response_metadata``
    (where ``llm_client.extract_leaked_reasoning`` reads it); a legacy
    langchain message's ``additional_kwargs`` is honoured as a fallback.
    Accumulates across repeated parser invocations so we don't clobber
    reasoning extracted on a prior pass (e.g. when the same message is
    re-parsed as history). Silent no-op if the response is frozen.
    """
    if not reasoning:
        return
    meta = getattr(response, "response_metadata", None)
    if not isinstance(meta, dict):
        meta = getattr(response, "additional_kwargs", None)
    if not isinstance(meta, dict):
        with contextlib.suppress(Exception):
            response.response_metadata = {LEAKED_REASONING_KEY: reasoning}
        return
    prior = meta.get(LEAKED_REASONING_KEY, "")
    if prior:
        if reasoning in prior:
            return
        meta[LEAKED_REASONING_KEY] = f"{prior}\n\n{reasoning}"
    else:
        meta[LEAKED_REASONING_KEY] = reasoning


def extract_leaked_reasoning(text: str) -> tuple[str, str]:
    """Strip leaked thinking/reasoning tags and return (cleaned, reasoning).

    Recovers inner content from every leak pattern we know about:
      * ``<think>…</think>`` / ``<thinking>…`` / ``<reasoning>…`` (balanced)
      * ``<think_never_used_…>…</think_never_used_…>`` (Seed / GPT-OSS)
      * ``<model:thinking>…</model:thinking>``
      * dangling opens truncated mid-response — the tail is taken as
        reasoning, not silently dropped.

    The reasoning string is the concatenation of every captured block, joined
    by blank lines. Empty captures are skipped. The cleaned text has all
    matched blocks removed, matching the previous ``_strip_leaked_tags``
    contract.
    """
    if not text:
        return text, ""

    reasoning_parts: list[str] = []

    def _collect(pattern: re.Pattern, src: str) -> str:
        out = src
        for match in pattern.finditer(src):
            inner = match.group(1).strip() if match.groups() else ""
            if inner:
                reasoning_parts.append(inner)
        out = pattern.sub("", out)
        return out

    cleaned = text
    # Specific "never_used" variants first — they have a different closing
    # shape from the generic <think>/</think> pair and must not be left
    # dangling after the generic strip.
    cleaned = _collect(_NEVER_USED_TAG_RE, cleaned)
    cleaned = _collect(_MODEL_THINKING_RE, cleaned)
    cleaned = _collect(_LEAKED_TAG_RE, cleaned)

    # Dangling tails — capture the remainder as reasoning (truncated inner
    # monologue), then drop it from the cleaned output.
    for dangling_re in (_NEVER_USED_DANGLING_RE, _DANGLING_LEAKED_TAG_RE):
        match = dangling_re.search(cleaned)
        if match:
            inner = match.group(1).strip()
            if inner:
                reasoning_parts.append(inner)
            cleaned = dangling_re.sub("", cleaned)

    reasoning = "\n\n".join(p for p in reasoning_parts if p)
    return cleaned, reasoning
