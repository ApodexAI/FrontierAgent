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
async def test_runner_applies_limit_after_seeded_shuffle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.public import sandbox_profiles
    from benchmarks.public.core import harbor_task_generator, registry
    from benchmarks.public.runner import run_subprocess

    source = _questions()
    selected_runs: list[list[str]] = []
    load_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        registry,
        "get_config",
        lambda _benchmark: SimpleNamespace(
            default_pipeline="stateful-react-agent",
            scoring_mode="external",
            name="Sample",
        ),
    )

    def load_questions(_benchmark: str, **kwargs: object) -> list[BenchmarkQuestion]:
        load_calls.append(kwargs)
        return source.copy()

    monkeypatch.setattr(registry, "load_questions", load_questions)
    monkeypatch.setattr(
        sandbox_profiles,
        "resolve_closed_book",
        lambda _benchmark, _override=None: False,
    )

    def capture_selection(question_dicts, _tasks_dir, *, pipeline_id: str) -> None:
        assert pipeline_id == "stateful-react-agent"
        selected_runs.append([question["id"] for question in question_dicts])
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
        no_shuffle=False,
        profile="default",
        fs_mode=False,
    )

    for seed in (42, 1234):
        with pytest.raises(_SelectionCaptured):
            await run_subprocess.run_eval(
                args,
                out_dir=tmp_path / str(seed),
                seed=seed,
            )

    assert all("limit" not in call for call in load_calls)
    assert len(selected_runs[0]) == len(selected_runs[1]) == args.limit
    assert set(selected_runs[0]) != set(selected_runs[1])


@pytest.mark.asyncio
async def test_runner_limit_preserves_order_when_shuffle_is_disabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        registry,
        "load_questions",
        lambda _benchmark, **_kwargs: source.copy(),
    )
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
        no_shuffle=True,
        profile="default",
        fs_mode=False,
    )

    with pytest.raises(_SelectionCaptured):
        await run_subprocess.run_eval(args, out_dir=tmp_path, seed=42)

    assert selected == [question.id for question in source[: args.limit]]
