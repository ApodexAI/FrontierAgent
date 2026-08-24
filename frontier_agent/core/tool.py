"""Tool definitions — function + OpenAI function-schema."""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, overload

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    """A callable tool with an OpenAI-compatible JSON-schema signature."""

    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema (OpenAI ``function.parameters``)
    func: ToolFn
    metadata: dict[str, Any] = field(default_factory=dict)

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        """Run the tool with keyword args. ``func`` must be async."""
        return await self.func(**args)

    def to_openai_schema(self) -> dict[str, Any]:
        """One entry of OpenAI's ``tools=`` array.

        A workflow that must reproduce an exact served byte-shape (e.g. the
        FastMCP-aligned profiles) pins the final dict in
        ``metadata["openai_schema"]``; this method returns it verbatim then.
        """
        if (spec := self.metadata.get("openai_schema")) is not None:
            return spec
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_PRIMITIVE_TYPES: dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    type(None): "null",
}


def _schema_for_type(t: Any) -> dict[str, Any]:
    """Build a JSON-schema fragment for a Python type hint."""
    if t is Any:
        return {}
    if t in _PRIMITIVE_TYPES:
        return {"type": _PRIMITIVE_TYPES[t]}

    # Pydantic models are the preferred way to describe structured tool
    # arguments.  Keep the dependency duck-typed here: ``core.tool`` does not
    # need to import Pydantic, while any BaseModel subclass still contributes
    # its full object schema (properties, required fields, descriptions, and
    # ``additionalProperties`` policy) to the surrounding tool schema.
    model_json_schema = getattr(t, "model_json_schema", None)
    if callable(model_json_schema):
        model_schema = model_json_schema()
        if isinstance(model_schema, dict):
            return model_schema

    origin = get_origin(t)
    if origin is None:
        # Unannotated / unknown — accept anything.
        return {"type": "string"}

    if origin in (list, set, tuple):
        args = get_args(t)
        if args:
            return {"type": "array", "items": _schema_for_type(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin in (Union, typing.Union):  # type: ignore[attr-defined]
        non_none = [a for a in get_args(t) if a is not type(None)]
        nullable = len(non_none) != len(get_args(t))
        schema: dict[str, Any]
        if len(non_none) == 1:
            schema = _schema_for_type(non_none[0])
        else:
            schema = {"anyOf": [_schema_for_type(a) for a in non_none]}
        if nullable:
            # ``default: null`` is FastMCP's idiom; many proxies accept it.
            schema.setdefault("default", None)
        return schema

    # Python 3.10+ ``X | Y`` resolves to ``types.UnionType``; treated the
    # same as ``typing.Union`` above when get_origin returns it.
    if origin is type(int | str):  # types.UnionType
        # A runtime subscript of typing.Union (not an annotation), so PEP 604
        # `X | Y` syntax does not apply here.
        return _schema_for_type(Union[get_args(t)])  # noqa: UP007

    return {"type": "string"}


def _hoist_schema_defs(
    schema: Any,
    definitions: dict[str, Any],
) -> None:
    """Move nested JSON Schema definitions to the parameters document root.

    Pydantic emits references such as ``#/$defs/Inner``. Those references are
    document-root-relative, so leaving ``$defs`` embedded in an argument or
    array-item schema produces a dangling reference. Collect every definition
    while the inferred parameters document is assembled.
    """
    if isinstance(schema, list):
        for item in schema:
            _hoist_schema_defs(item, definitions)
        return
    if not isinstance(schema, dict):
        return

    local_defs = schema.pop("$defs", {})
    if isinstance(local_defs, dict):
        for name, definition in local_defs.items():
            existing = definitions.get(name)
            if existing is not None and existing != definition:
                raise ValueError(
                    "Conflicting inferred JSON Schema definition "
                    f"{name!r}; provide an explicit tool parameters schema."
                )
            definitions[name] = definition
        for definition in local_defs.values():
            _hoist_schema_defs(definition, definitions)

    for value in schema.values():
        _hoist_schema_defs(value, definitions)


_ARG_HEADER_RE = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$", re.MULTILINE)
_RETURNS_HEADER_RE = re.compile(
    r"^\s*(Returns|Yields|Raises|Examples):\s*$", re.MULTILINE,
)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {arg_name: description})`` from a Google-style docstring."""
    if not doc:
        return "", {}
    # ``inspect.cleandoc`` normalises indentation (strips the common leading
    # whitespace shared by all lines after the first), so parsing is
    # independent of how deeply the function — and therefore its docstring —
    # is indented in source: a nested / method-level ``Args:`` block now
    # parses the same as a module-level one. The previous fixed-width
    # ``line.startswith(" " * 12)`` heuristic silently dropped EVERY arg
    # description for tools whose docstring sat at >=12-space indent.
    text = inspect.cleandoc(doc)
    arg_match = _ARG_HEADER_RE.search(text)
    if not arg_match:
        return text, {}
    summary = text[:arg_match.start()].rstrip()
    body = text[arg_match.end():]
    end = _RETURNS_HEADER_RE.search(body)
    if end:
        body = body[:end.start()]

    # An arg line is one at the Args-block base indent; more-indented lines
    # fold in as continuations of the current arg.
    body_lines = [ln for ln in body.splitlines() if ln.strip()]
    base = min((len(ln) - len(ln.lstrip()) for ln in body_lines), default=0)
    descriptions: dict[str, str] = {}
    current = ""  # name of the arg whose description is being accumulated
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        m = re.match(r"^([A-Za-z_][\w]*)(?:\s*\([^)]+\))?\s*:\s*(.*)", stripped)
        if m and indent <= base:
            # Both groups are mandatory subpatterns, so they always
            # participate; the ``or ""`` fallbacks are unreachable at runtime.
            current = m.group(1) or ""
            descriptions[current] = m.group(2) or ""
        elif current:
            descriptions[current] = f"{descriptions[current]} {stripped}".strip()
    return summary, descriptions


def _infer_parameters(
    fn: Callable[..., Any],
    arg_docs: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON schema from a function's annotations + docstring."""
    hints = typing.get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    definitions: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in {"self", "cls"} or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        schema = _schema_for_type(hints.get(name, str))
        _hoist_schema_defs(schema, definitions)
        if desc := arg_docs.get(name):
            schema["description"] = desc
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif "default" not in schema:
            schema["default"] = param.default
        properties[name] = schema
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    if definitions:
        out["$defs"] = definitions
    return out


@overload
def tool(fn: ToolFn, /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    parameters: dict[str, Any] | None = ...,
) -> Callable[[ToolFn], Tool]: ...


def tool(
    fn: ToolFn | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Tool | Callable[[ToolFn], Tool]:
    """Decorator that wraps an async function as a :class:`Tool`.

    The two overloads above distinguish the bare ``@tool`` form (which produces
    a :class:`Tool`) from the ``@tool(...)`` factory form (which produces the
    decorator). Without them every decorated symbol in the codebase infers as
    the union of both, so any attribute access on one — ``.func``,
    ``.description``, ``.name`` — is an error at each of the ~30 use sites.

    Usage::

        @tool
        async def web_search(query: str) -> str:
            ...

        @tool(name="run_python", description="Run code in a sandbox.")
        async def _runner(code: str) -> str:
            ...

    When ``parameters`` is omitted, a JSON schema is inferred from the
    function's type hints + Google-style docstring.
    """

    def _wrap(f: ToolFn) -> Tool:
        summary, arg_docs = _parse_docstring(f.__doc__ or "")
        return Tool(
            name=name or f.__name__,
            description=description or summary or "",
            parameters=parameters or _infer_parameters(f, arg_docs),
            func=f,
        )

    return _wrap(fn) if fn is not None else _wrap


__all__ = ["Tool", "ToolFn", "tool"]
