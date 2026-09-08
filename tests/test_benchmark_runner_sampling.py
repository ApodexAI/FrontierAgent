from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from benchmarks.public.core.question import BenchmarkQuestion


class _SelectionCaptured(Exception):
    """Stop a benchmark run after its selected questions are observable."""


def _questions(count: int = 20) -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion(
            id=f"q{index:02d}",
            question=f"Question {index}",
            ground_truth=f"Answer {index}",
            answer_type="exactMatch",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("seed", "no_shuffle", "expected_ids"),
    [
        (42, False, ["q19", "q05", "q14", "q04", "q09"]),
        (1234, False, ["q19", "q13", "q04", "q09", "q16"]),
        (42, True, ["q00", "q01", "q02", "q03", "q04"]),
    ],
    ids=["seed-42", "seed-1234", "no-shuffle"],
)
async def test_runner_selects_questions_after_optional_shuffle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    no_shuffle: bool,
    expected_ids: list[str],
) -> None:
    from benchmarks.public import sandbox_profiles
    from benchmarks.public.core import harbor_task_generator, registry
    from benchmarks.public.runner import run_subprocess

    source = _questions()
    selected: list[str] = []

    monkeypatch.setattr(
        registry,
        "get_config",
        lambda _benchmark: SimpleNamespace(
            default_pipeline="stateful-react-agent",
            scoring_mode="external",
            name="Sample",
        ),
    )

    def load_questions(
        _benchmark: str, *, limit: int | None = None, **_kwargs: object
    ) -> list[BenchmarkQuestion]:
        return source[:limit] if limit else source.copy()

    monkeypatch.setattr(registry, "load_questions", load_questions)
    monkeypatch.setattr(
        sandbox_profiles,
        "resolve_closed_book",
        lambda _benchmark, _override=None: False,
    )

    def capture_selection(question_dicts, _tasks_dir, *, pipeline_id: str) -> None:
        assert pipeline_id == "stateful-react-agent"
        selected.extend(question["id"] for question in question_dicts)
        raise _SelectionCaptured

    monkeypatch.setattr(
        harbor_task_generator,
        "generate_task_dirs",
        capture_selection,
    )

    args = argparse.Namespace(
        benchmark="sample",
        pipeline=None,
        web=None,
        limit=5,
        offset=2,
        answer_type=None,
        category=None,
        no_shuffle=no_shuffle,
        profile="default",
        fs_mode=False,
    )

    with pytest.raises(_SelectionCaptured):
        await run_subprocess.run_eval(args, out_dir=tmp_path, seed=seed)

    assert selected == expected_ids
