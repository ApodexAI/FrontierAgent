"""Shared JSONL field schema used by every standard family DatasetConfig."""

from __future__ import annotations

from typing import TypedDict


class StdSchema(TypedDict):
    """The JSONL field names shared by every standard family.

    A TypedDict rather than a plain dict so ``DatasetConfig(**STD_SCHEMA)``
    type-checks per key. As ``dict(...)`` this inferred as ``dict[str, str]``,
    and unpacking it made every DatasetConfig parameter look like it was being
    handed a ``str`` — including ``extra_metadata_fields: tuple[str, ...]``,
    which reported an error in all nine family modules at once.
    """

    id_field: str
    question_field: str
    answer_field: str
    file_field: str
    file_name_field: str


STD_SCHEMA: StdSchema = {
    "id_field": "task_id",
    "question_field": "task_question",
    "answer_field": "ground_truth",
    "file_field": "file_name",
    "file_name_field": "file_name",
}
