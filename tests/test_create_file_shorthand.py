"""``create_file(content=...)`` must work, because the model will write it.

Measured across the two compaction A/B runs: 8 calls lost to ``TypeError:
create_file() got an unexpected keyword argument 'content'``, from five
different sub-agents in both runs, a turn each. ``content`` is the obvious name
for a file's body, and the tool guidance named it in a sentence about calling
the tool when it is really a ``create`` op param.

These exercise the desugaring rather than the write: ``create_file`` writes
through a sandbox that is not available under test, and the desugaring is the
part that changed.
"""

import inspect

import pytest

from plugins.tools.create_file import create_file, desugar_text_shorthand


def _fold(**kwargs):
    base = {"ops": None, "content": None, "rows": None, "data": None}
    return desugar_text_shorthand(**{**base, **kwargs})


# ── the shorthand exists and means exactly one create op ─────────────────────


def test_content_becomes_a_create_op() -> None:
    ops, error = _fold(content="# Title\nbody")

    assert error == ""
    assert ops == [{"create": {"content": "# Title\nbody"}}]


@pytest.mark.parametrize(("key", "value"), [
    ("content", "literal body"),
    ("rows", [["a", "b"], ["1", "2"]]),
    ("data", {"k": [1, 2]}),
])
def test_every_advertised_spelling_folds(key, value) -> None:
    """The docstring offers three; all three must reach the op."""
    ops, error = _fold(**{key: value})

    assert error == ""
    assert ops == [{"create": {key: value}}]


def test_the_tool_actually_accepts_the_shorthand() -> None:
    """The regression, stated where it happened: these were the kwargs the tool
    rejected. A helper that folds them is no use if the signature still refuses."""
    params = inspect.signature(create_file.func).parameters

    for name in ("content", "rows", "data", "overwrite"):
        assert name in params, f"create_file({name}=...) would still raise TypeError"


def test_the_tool_schema_accepts_the_advertised_json_shapes() -> None:
    """Schema-aware providers must be able to emit nested rows and JSON data."""
    properties = create_file.parameters["properties"]
    rows_array = next(
        option for option in properties["rows"]["anyOf"]
        if option.get("type") == "array"
    )
    data_array = next(
        option for option in properties["data"]["anyOf"]
        if option.get("type") == "array"
    )

    assert rows_array["items"] == {}
    assert data_array["items"] == {}


def test_empty_string_content_still_folds() -> None:
    """An empty deliverable is a legitimate ask, and ``""`` is falsy — a
    truthiness check here would drop it and write nothing."""
    ops, error = _fold(content="")

    assert error == ""
    assert ops == [{"create": {"content": ""}}]


@pytest.mark.parametrize("kwargs", [
    {"content": "body", "rows": ["a"]},
    {"content": "body", "data": {"a": 1}},
    {"rows": [["a"]], "data": {"a": 1}},
])
def test_several_spellings_at_once_are_an_explicit_error(kwargs) -> None:
    """The writer prioritises content > data > rows, so accepting more than one
    would silently discard caller-provided data."""
    ops, error = _fold(**kwargs)

    assert ops is None
    assert error.startswith("Error: pass exactly ONE")


# ── what the shorthand must not change ───────────────────────────────────────


def test_no_shorthand_leaves_ops_untouched() -> None:
    explicit = [{"create": {"content": "x"}}, {"append": {"content": "y"}}]

    assert _fold(ops=explicit) == (explicit, "")
    assert _fold(ops="@/workspace/program.json") == (
        "@/workspace/program.json", ""
    )
    assert _fold() == (None, "")


def test_overwrite_only_travels_with_the_shorthand() -> None:
    """It is a shorthand convenience. On the ops path the caller already writes
    ``{"create": {"overwrite": true}}`` and must not be second-guessed."""
    ops, error = _fold(content="new", overwrite=True)
    assert error == ""
    assert ops == [{"create": {"content": "new", "overwrite": True}}]

    explicit = [{"create": {"content": "x"}}]
    assert _fold(ops=explicit, overwrite=True) == (explicit, "")


def test_ops_and_shorthand_together_is_an_explicit_error() -> None:
    """Preferring one silently would write a file the caller did not describe."""
    _ops, error = _fold(ops=[{"create": {"content": "from ops"}}], content="other")

    assert "EITHER ops" in error
    assert "content" in error
    # The error is the whole reply, so ops is never acted on.
    assert error.startswith("Error:")


@pytest.mark.parametrize("ops", [[], ""])
def test_empty_ops_still_conflicts_with_shorthand(ops) -> None:
    """Supplying ops and shorthand is always ambiguous, even when ops is empty."""
    _ops, error = _fold(ops=ops, content="other")

    assert error.startswith("Error: pass EITHER ops")
