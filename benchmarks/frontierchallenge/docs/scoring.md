# Scoring

FrontierChallenge reports two numbers over a fixed denominator of 97 tasks:

- **Pass Rate:** completed evaluations with `task_score == 1.0`, divided by 97.
- **Score:** the mean of `task_score` across all 97, usually reported times 100.

Unrun tasks and harness failures count as zero in the fixed denominator. The
summarizer marks an incomplete run as partial while retaining the denominator
97. It never drops missing or failed tasks from the headline metrics.

Equality is exact: `0.999` and `0.999999` do not pass. No rounding, epsilon,
per-task pass threshold, or native `passed` decision enters this calculation.
`evaluation_complete == 1` is required for a valid score. Invalid scores
(non-numeric, non-finite, or outside `[0, 1]`) contribute zero and are flagged.

## Authoritative fields

Each trial writes `verifier/reward.json`:

| Field | Meaning |
|---|---|
| `passed` | 1 only when evaluation completed and `task_score == 1.0`; otherwise 0 |
| `task_score` | score from 0 to 1 |
| `evaluation_complete` | whether verification completed |

There is only one pass field, `passed`, with the same meaning in
`verifier/reward.json`, `summary.csv`, and `summary.json`. A completed score
of 1 passes; a score of 0.8 does not. No alternate pass field is emitted.

After authenticating and unsealing the reference, the runtime applies the
full-score rule to the staged reward adapter before Harbor runs it. The
encrypted reference archives and partial-credit rubrics remain unchanged.
The summarizer also derives `passed` from score and completion when processing
older results, discarding their old pass decision rather than copying it.

This policy is identified by `metric_definition: exact-full-score` in the
summary. Recompute historical results from raw rewards before comparing them;
results computed with native thresholds or `>= 0.999` are not interchangeable.

Summarize a Harbor job directory with:

```bash
python3 scripts/summarize_results.py results/harbor/<job>
```

Use one predeclared attempt per task. Duplicate task trials are rejected rather
than silently counted twice or selected by their score. For a deliberately
separate subset report (for example, the 81-task open track), specify
`--expected-total 81` and report that denominator explicitly; it is not the
97-task headline metric. `--expected-total` must not be reduced to the number
of tasks that happened to succeed. Missing tasks still contribute zero.

For repeated trials, use `run_eval.sh --n-attempts N --no-summary` and summarize
each predeclared attempt separately. Use this runtime's summary for the
fixed-denominator headline metric; Harbor may aggregate only attempted trials.

## Verifiers and judges

Each task has a frozen verifier. Deterministic checks validate submitted files,
values, units, schemas, and tolerances. Seventy-seven tasks also use an LLM
judge for the written report. The declared judge uses three repetitions.

Changing the judge model changes the result. `run_eval.sh` uses `JUDGE_MODEL`
from `.env` as an override when it is set and announces the substitution. Pass
`--no-judge-override` to use every task's frozen judge declaration. Always name
the judge and repeat count when reporting results.

## Public solve-side and encrypted verifiers

The solve dataset contains only agent-visible material:

- `instruction.md`;
- `task.toml` and `task.json`;
- `environment/`, including inputs and the runtime definition.

The separate reference dataset contains one authenticated `verifier.fcref` per
task. Each archive holds `tests/`, including graders, rubrics, fixtures, and
reference outputs. The public password is `frontier-challenge-reference`.
Encryption is an anti-indexing measure, not access control.

During a run, the controller copies the solve task to an evaluator-owned stage,
copies in the matching encrypted verifier, and authenticates and decrypts it.
Harbor exposes the instruction and task environment to the agent, while
`tests/` is used only by the verifier after the agent phase. HF and judge
credentials remain on the controller side.

`registry.json` binds the GitHub runtime and both datasets: every task has a
solve-side hash and an encrypted-verifier hash. Setup refuses mixed releases.

## Reporting checklist

Report:

- denominator 97, with missing tasks counted as zero;
- Pass Rate from completed `task_score == 1.0`;
- mean `task_score` times 100;
- agent, model, judge model, and judge repetitions;
- pinned Docker image identity and ORCA version for full-track runs;
- any changed timeout, tool, network, or judge setting.
