"""Contract tests for the shared tool-argument coercion helpers.

Both the task_board tools and the TUI pane that projects them depend on these,
so the contract is pinned here rather than only through either caller.
"""

from __future__ import annotations

import pytest

from plugins.tools._coerce import coerce_json_list, coerce_json_object


def test_a_dict_is_returned_unchanged() -> None:
    item = {"description": "explain money creation"}
    assert coerce_json_object(item) is item


@pytest.mark.parametrize("raw, expected", [
    ('{"id":"t3","resolution":"resolved"}', {"id": "t3", "resolution": "resolved"}),
    ('[{"description":"one item"}]', {"description": "one item"}),
    ('  {"id":"t1"}  ', {"id": "t1"}),
])
def test_double_encoded_objects_are_parsed(raw: str, expected: dict) -> None:
    assert coerce_json_object(raw) == expected


@pytest.mark.parametrize("raw", [
    '[{"id":"t1"},{"id":"t2"}]',  # ambiguous: which of the two is the item?
    "[]",
    "resolved",                    # a bare word, not JSON
    '{"id":"t1"',                  # malformed JSON
    '"t1"',                        # JSON, but a string not an object
    "[1, 2]",
    42,
    None,
    ["already", "a", "list"],
])
def test_unusable_items_return_none(raw: object) -> None:
    assert coerce_json_object(raw) is None


def test_only_json_shaped_strings_are_parsed_at_the_list_level() -> None:
    already = [{"id": "t1"}]
    assert coerce_json_list(already) is already
    assert coerce_json_list('[{"id":"t1"}]') == [{"id": "t1"}]
    # A lone object is wrapped so callers always see a uniform list.
    assert coerce_json_list('{"id":"t1"}') == [{"id": "t1"}]
    assert coerce_json_list("not json") == "not json"
