from __future__ import annotations

import json
import logging
from typing import Any

from frontier_agent.core.llm import LLMResponse

logger = logging.getLogger(__name__)


def _required_tool_arguments(llm: Any) -> dict[str, set[str]]:
    """Map bound tool name → its non-empty set of required argument names.

    Tools with no required property are omitted entirely: for them ``{}`` is
    a legitimate call and must never trigger a second model request.
    """
    required_by_name: dict[str, set[str]] = {}
    for tool_schema in getattr(llm, "tools", None) or []:
        if not isinstance(tool_schema, dict):
            continue
        function = tool_schema.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        required = parameters.get("required")
        if isinstance(required, list) and required:
            fields = {str(f) for f in required if isinstance(f, str)}
            if fields:
                required_by_name[name] = fields
    return required_by_name


def _lost_required_tool_arguments(
    raw_arguments: Any,
    required_fields: set[str],
) -> bool:
    """Whether a streamed ``arguments`` payload lost its required fields.

    Two shapes count as lost, both observed from Qwen/SGLang when generation
    stops before the closing parameter marker:

    - ``None`` / ``""`` / whitespace — the parser emitted a bare tool-call
      shell carrying no payload at all.
    - a well-formed JSON object missing at least one required field, ``{}``
      being the common case.

    A payload that does not parse also counts as lost. The native tool-call
    normalizer cannot preserve or repair that raw fragment: it degrades a
    ``json.loads`` failure to ``args={}``, which is indistinguishable from the
    empty-argument failure by the time tool validation runs.
    """
    if raw_arguments is None:
        return True
    if not isinstance(raw_arguments, str):
        return False
    if not raw_arguments.strip():
        return True
    try:
        parsed = json.loads(raw_arguments)
    except (ValueError, TypeError):
        return True
    return isinstance(parsed, dict) and bool(required_fields - set(parsed))


def _stream_tool_calls_missing_required_arguments(
    response: LLMResponse,
    llm: Any,
) -> list[str]:
    """Return required-argument tool names whose streamed args came back empty.

    Some OpenAI-compatible serving parsers emit a native tool-call shell with
    a valid function name but ``arguments=""`` when generation stops before
    the closing parameter marker. The non-streaming parser can often recover
    the same truncated payload, so a single replay is worth one extra request.
    """
    required_by_name = _required_tool_arguments(llm)
    if not required_by_name:
        return []

    missing: list[str] = []
    for tool_call in response.tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        required_fields = required_by_name.get(name)
        if required_fields and _lost_required_tool_arguments(
            function.get("arguments"), required_fields,
        ):
            missing.append(name)
    return missing
