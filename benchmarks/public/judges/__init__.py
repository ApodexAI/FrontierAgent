"""LLM-as-judge dispatch for benchmark answer grading."""

from __future__ import annotations

from benchmarks.public.judges._common import JudgeFn, ScoreFn, Verdict
from benchmarks.public.judges.apex import score_apex
from benchmarks.public.judges.browsecomp import (
    JUDGE_PROMPT_BC_ZH,
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    verify_browsecomp,
    verify_browsecomp_zh,
)
from benchmarks.public.judges.deepsearchqa import (
    JUDGE_PROMPT_DEEPSEARCHQA,
    score_deepsearchqa,
    verify_deepsearchqa,
)
from benchmarks.public.judges.frontier_science import (
    JUDGE_PROMPT_FS,
    JUDGE_PROMPT_FS_OLYMPIAD,
    score_frontier_science,
    verify_frontier_science,
    verify_frontier_science_olympiad,
)
from benchmarks.public.judges.hle import (
    JUDGE_PROMPT_HLE,
    verify_hle,
)
from benchmarks.public.judges.officeqa import score_officeqa
from benchmarks.public.judges.onemillion import score_onemillion
from benchmarks.public.judges.xbench import verify_xbench

# WideSearch's judge is the only one needing pandas (structural table F1).
# Importing it eagerly made a missing optional dep break *every* benchmark:
# run_one_task.py imports this package unconditionally, so a tree installed
# without the dep crashed all 20 workers before any question ran. Degrade to a
# judge that names the fix instead.
try:
    from benchmarks.public.judges.widesearch import score_widesearch
except ModuleNotFoundError as exc:
    _WIDESEARCH_MISSING = str(exc)

    async def score_widesearch(  # type: ignore[misc]
        question: str, target: str, predicted: str,
    ) -> tuple[Verdict, float | None]:
        raise RuntimeError(
            f"the widesearch judge needs an optional dependency "
            f"({_WIDESEARCH_MISSING}); install it with: uv sync --extra eval"
        )

JUDGE_REGISTRY: dict[str, JudgeFn] = {
    "browsecomp":                  verify_browsecomp,
    "browsecomp_zh":               verify_browsecomp_zh,
    "xbench_dr_202510":            verify_xbench,
    "frontier_science_research":   verify_frontier_science,
    "frontier_science_olympiad":   verify_frontier_science_olympiad,
    "hle_text":                    verify_hle,
    "deepsearchqa":                verify_deepsearchqa,
    # SUPERChem-Text is single-letter MCQ (A-J); HLE's extract-final-answer
    # + correct: yes|no schema handles it cleanly — judge pulls the chosen
    # letter from the model's natural-language reply and matches it against
    # the gold letter in ground_truth.
    "superchem_text":              verify_hle,
    # WideSearch uses score_widesearch (structural row/item F1 + judge LLM
    # for column alignment). It returns (verdict, f1_by_item) so it is
    # dispatched via a separate score_* path in the runner, NOT through
    # JUDGE_REGISTRY's standard Verdict-returning dispatch.
}


def has_judge(benchmark: str) -> bool:
    """True if ``benchmark`` has an LLM judge configured (rubric or binary)."""
    return benchmark in JUDGE_REGISTRY or benchmark in SCORE_REGISTRY


async def verify_answer(
    benchmark: str, question: str, target: str, predicted: str,
) -> Verdict:
    """Dispatch to the registered judge. Raises ``KeyError`` if unregistered."""
    fn = JUDGE_REGISTRY.get(benchmark)
    if fn is None:
        raise KeyError(
            f"No LLM judge registered for benchmark {benchmark!r}. "
            f"Registered: {sorted(JUDGE_REGISTRY)}"
        )
    return await fn(question, target, predicted)


# Benchmarks whose judge surfaces a raw rubric value (0-10 for FS,
# F1 0-1 for WideSearch). Stored in result.json's ``rubric_score`` field.
# WideSearch has no binary verify_* sibling — it only exists here, so
# SCORE_REGISTRY is also the dispatch root for it.
SCORE_REGISTRY: dict[str, ScoreFn] = {
    "apex":                     score_apex,
    "deepsearchqa":              score_deepsearchqa,
    "frontier_science_research": score_frontier_science,
    "officeqa":                  score_officeqa,
    "officeqa_full":             score_officeqa,
    "onemillion_bench":          score_onemillion,
    "widesearch":                score_widesearch,
}


async def score_answer(
    benchmark: str, question: str, target: str, predicted: str,
) -> tuple[Verdict, float | None]:
    """Verdict plus optional raw rubric score (``None`` for binary judges)."""
    if (fn := SCORE_REGISTRY.get(benchmark)) is not None:
        return await fn(question, target, predicted)
    return await verify_answer(benchmark, question, target, predicted), None


__all__ = [
    "JUDGE_PROMPT_BC_ZH",
    "JUDGE_PROMPT_BROWSECOMP_OFFICIAL",
    "JUDGE_PROMPT_DEEPSEARCHQA",
    "JUDGE_PROMPT_FS",
    "JUDGE_PROMPT_FS_OLYMPIAD",
    "JUDGE_PROMPT_HLE",
    "JUDGE_REGISTRY",
    "SCORE_REGISTRY",
    "JudgeFn",
    "ScoreFn",
    "Verdict",
    "has_judge",
    "preflight_judge",
    "score_answer",
    "score_apex",
    "score_deepsearchqa",
    "score_frontier_science",
    "score_officeqa",
    "score_onemillion",
    "score_widesearch",
    "verify_answer",
    "verify_browsecomp",
    "verify_browsecomp_zh",
    "verify_deepsearchqa",
    "verify_frontier_science",
    "verify_frontier_science_olympiad",
    "verify_hle",
    "verify_xbench",
]

# Benchmark -> the judge model that benchmark's grader will actually call, so a
# preflight can test reachability without depending on judge semantics.
def _judge_models() -> dict[str, str]:
    from benchmarks.public.judges._common import _DEFAULT_JUDGE_MODEL, _resolve_judge_model
    from benchmarks.public.judges.browsecomp import _BC_JUDGE_MODEL, _BC_ZH_JUDGE_MODEL
    from benchmarks.public.judges.deepsearchqa import _DSQA_DEFAULT_JUDGE_MODEL
    from benchmarks.public.judges.frontier_science import _FS_DEFAULT_JUDGE_MODEL
    from benchmarks.public.judges.hle import _HLE_JUDGE_MODEL
    from benchmarks.public.judges.xbench import _XBENCH_JUDGE_MODEL
    pinned = {
        "browsecomp": _BC_JUDGE_MODEL,
        "browsecomp_zh": _BC_ZH_JUDGE_MODEL,
        "xbench_dr_202510": _XBENCH_JUDGE_MODEL,
        "deepsearchqa": _DSQA_DEFAULT_JUDGE_MODEL,
        "frontier_science_research": _FS_DEFAULT_JUDGE_MODEL,
        "frontier_science_olympiad": _FS_DEFAULT_JUDGE_MODEL,
        "hle_text": _HLE_JUDGE_MODEL,
        "superchem_text": _HLE_JUDGE_MODEL,
        # These grade through score_* rubric paths rather than
        # JUDGE_REGISTRY, and pass no model to ``_build_judge_kwargs`` — so
        # they land on the global default. Listed here so they are preflighted
        # too; the deterministic validators (officeqa, gdpval) have no model
        # and are correctly absent.
        "apex": _DEFAULT_JUDGE_MODEL,
        "onemillion_bench": _DEFAULT_JUDGE_MODEL,
    }
    try:
        from benchmarks.public.judges.widesearch import _WIDESEARCH_DEFAULT_JUDGE_MODEL
        pinned["widesearch"] = _WIDESEARCH_DEFAULT_JUDGE_MODEL
    except Exception:
        pass
    return {b: _resolve_judge_model(None, m or _DEFAULT_JUDGE_MODEL)
            for b, m in pinned.items()}


async def preflight_judge(benchmark: str) -> tuple[bool, str]:
    """Prove the judge's model is reachable before a run starts.

    An unreachable judge model does not announce itself: the judge swallows the
    transport error, returns NOT_ATTEMPTED for every question, and the run
    completes reporting 0% accuracy — indistinguishable from a model that got
    everything wrong, after however many hours the run took. Several benchmarks
    pin gateway-specific names (``google/gemini-2.5-flash``, ``openai/gpt-5``)
    that a differently-routed gateway answers with 503.

    This probes the *model*, not the grader's semantics. Grading a sample pair
    instead would misfire on the judges whose prompt is Chinese or whose target
    is a rubric — for those a trivially-correct English pair is legitimately
    NOT_ATTEMPTED, which says nothing about reachability.
    """
    model = _judge_models().get(benchmark)
    if model is None:
        # The remaining benchmarks (officeqa, gdpval) validate deterministically
        # and call no model at all, so there is nothing to probe.
        return True, f"{benchmark}: deterministic scorer, no judge model"
    from benchmarks.public.judges._common import _make_client
    try:
        client = _make_client()
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            max_completion_tokens=4,
        )
    except Exception as exc:
        return False, (
            f"judge model {model!r} is not usable: "
            f"{type(exc).__name__}: {str(exc)[:180]}"
        )
    return True, f"judge model {model!r} reachable"
