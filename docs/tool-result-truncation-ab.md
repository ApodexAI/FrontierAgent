# Tool-result truncation: what changed and how to check it

Four changes to how an oversized tool result enters the context, and the two
experiments that decide whether they were right.

## What changed

| | before | after |
|---|---|---|
| inline shape | head only | head **and** tail, gap marked `… N chars elided …` |
| recovery pointer | appended *after* the cap | measured first, charged to the cap |
| spill filename | `uuid4()` | `sha256(tool_name, body)[:16]`, write skipped if present |
| per-turn total | `check_aggregate_budget` had no caller | applied in `execute_tools` after the gather, spilling before it cuts |

The shape is selected at runtime by `TOOL_RESULT_TRUNCATION`: `auto` (the
default — per tool, from `ToolMeta.result_is_ranked`), or `middle` / `head` to
force one shape on every tool, which is what the A/B arms pin. Config field
`FrontierAgentConfig.tool_result_truncation`. Nothing else in the pipeline reads
it, so the two arms differ in exactly one place.

Why the tail matters: a pytest run states its verdict in the last three lines, a
webpack build in the last error, a script in its exit status. A head-only cut
removes precisely the lines the tool was called for, and the model then spends a
turn reading the spill file to get them back. codex cuts the middle for the same
reason (`codex-rs/utils/output-truncation`, `truncate_middle_with_token_budget`),
and budgets its recovery footer before truncating
(`codex-rs/hooks/src/output_spill.rs`).

## Experiment 1 — offline, deterministic, no model

```
uv run python scripts/truncation_ab.py
uv run python scripts/truncation_ab.py --caps 2000,8000 --json /tmp/ab.json
```

Eight tool outputs (pytest, webpack, docker build, traceback, grep, a
single-line JSON body, a fetched page, `find`), each with its load-bearing lines
labelled `head` / `middle` / `tail`. Both arms run through the real
`maybe_overflow`, and the script reports what survived.

Current result — 21 labelled lines, caps 2K/8K/32K:

```
    cap  arm        head  middle    tail  over cap  pointer
   2000  head       100%      0%      0%         0        8
   2000  middle     100%      0%    100%         0        8
```

Read it as three claims: the tail becomes visible, the middle is what both arms
give up (recoverable only through the pointer), and no result exceeds the cap it
advertises. The script exits non-zero if `middle` fails to beat `head` on tail
recall, so it also works as a regression gate.

What it does **not** show: whether the model actually uses what it now sees.
That needs experiment 2.

## Experiment 2 — live A/B on a deliberately small context

```
scripts/run-truncation-ab.sh
BENCH=widesearch LIMIT=12 CONCURRENCY=6 scripts/run-truncation-ab.sh
```

Same questions, same seed, same profile in both arms; only
`TOOL_RESULT_TRUNCATION` differs. Two knobs make it cheap rather than changing
what is being tested:

- `OPENAI_CONTEXT_WINDOW=32768` — the tiered policy keeps its 80% trigger and
  60% relief geometry, it just reaches them roughly 8× sooner.
- `TOOL_EXEC_RESULT_MAX_CHARS=2000` — nearly every bash/grep result overflows,
  putting the preview shape on the critical path of every turn.
- `TOOL_RESULT_MAX_CHARS=8000` — the global cap, and the **only** one that
  applies to `web_fetch` / `web_search` / `read_file`, which set
  `max_result_chars=0`. Without lowering it, a search benchmark would leave the
  changed code path almost unexercised (most pages are well under the 150K
  default) and the two arms would come out identical.
- `COMPACTION_SPILL=true` — recovery reads are the metric; they need a store.

A prerequisite: `benchmarks/datasets/` is untracked, so the dataset has to be in
place first — see [the evaluation guide](eval.md#datasets).

Metrics come from artifacts the runner already writes:

```
uv run python scripts/truncation_metrics.py \
  results/<stamp>_trunc-ab/head results/<stamp>_trunc-ab/middle --labels head,middle
```

| metric | what it tells you |
|---|---|
| `recovery reads` | tool calls naming a `.spill` path — the round-trips the head-only shape pays for. **Expected to fall.** |
| `truncated results` | how often truncation fired. Should be ~equal; if not, the arms did not see comparable work and nothing else compares. |
| `score` | guards against a shape that saves tokens by losing the answer. |
| `turns`, `tool calls`, `peak prompt tok`, `wall clock` | the run's shape. |
| `compactions` | counted from the `TieredCompactor selected=` lines in `agent.log`, which also name the tier that won; falls back to prompt-token drops when no log is present. |

### Results: two runs, and what did not replicate

Both runs used the same 30/100 questions, seed 42, `benchmark` profile, 32K
window, 8K global cap, 2K exec cap, arms in the order head then middle.

Two independent reasons the absolute scores below are not DeepSearchQA numbers:
the throttle degrades both arms, and `JUDGE_MODEL` in `.env` overrode the
benchmark's pinned grader (`gpt-4.1-2025-04-14`), which the runner warns about.
Both arms of a run share the judge, so the within-run comparison holds — but do
not put these figures next to a published score.

| metric | n=30: head | n=30: middle | n=100: head | n=100: middle |
|---|---|---|---|---|
| score | 7/30 (.233) | 12/30 (.400) | 27/100 (.270) | 21/100 (.210) |
| recovery reads | 47 | 9 | 102 | 151 |
| truncated results | 305 | 286 | 916 | 965 |
| turns (mean) | 52.9 | 55.7 | 51.3 | 56.9 |
| wall clock s (mean) | 820 | 988 | 913 | 1000 |
| compactions | 114 | 121 | 309 | 335 |

Paired tests at n=100 (exact McNemar for score, sign test for the counts):

```
score           head 27/100  middle 21/100   p = 0.362   favours head
recovery reads  head 102     middle 151      p = 0.184   favours head
turns           middle higher on 50, lower on 45         p = 0.682
```

**Two claims from the n=30 run are retracted.**

*"The tail-inline shape removes pathological re-reading."* It does not. That
finding rested on a single trial doing 28 recovery reads. At n=100 the
distributions are broad and similar, and the direction reverses:

```
head   : 26/100 trials, nonzero values ... 5, 6, 10, 11, 13, 13
middle : 32/100 trials, nonzero values ... 7, 8, 10, 12, 12, 14, 14, 16
```

*"Score moved in the right direction."* Two runs, opposite signs, neither
significant. This is what a null effect looks like; the n=30 direction was noise.

**A mechanism worth considering, opposite to the one assumed.** The middle shape
prints `… N chars elided …` and its footer says to read the path *if the elided
middle is required*. The head shape only says part of the result is shown. Naming
the gap may invite fetching it — which is consistent with middle showing MORE
recovery reads at n=100, not fewer.

**Turns is the only metric with a consistent direction** (middle worse in both
runs, +5% then +11%) and it is not significant either. Nothing here supports the
middle shape on this benchmark.

### What survives

Independent of preview shape, and not dependent on any model behaviour:

- the recovery pointer is inside the cap it advertises (it was over on every
  overflowing call);
- the same body spills to one file however many times it is spilled;
- the per-turn aggregate total is actually enforced (it was dead code);
- the footer names a tool the running profile actually binds — a real bug this
  experiment found, unrelated to the shape.

Offline, the tail is visible where it was not before (0% → 100% recall). That is
a fact about the code. It is not evidence that the model benefits from it, and
the live runs did not find such evidence either way.

### What this says about the default

The live evidence gives no support for `middle` and mildly (never significantly)
favours `head` on every metric. But note what dominated these runs: DeepSearchQA
is search-heavy and `web_search` sets `max_result_chars=0`, so the 8K global cap
made *ranked search results* the most-truncated output in the run. The runs mostly
measured which shape suits a ranked list — and ranked lists are exactly where
`head` should win.

That is the reasoning behind `TOOL_RESULT_TRUNCATION=auto` (head for
`ToolMeta.result_is_ranked`, middle otherwise): it takes `head` where this run
gave weak evidence for it, and keeps `middle` only for the sequential output
where the a-priori argument applies and this benchmark says nothing. It is also
the only setting consistent with both references — codex cuts the middle of exec
output, kimi's head-only preview sits behind a 50K threshold.

`auto` is now the default. It is a reasoned choice, not a validated one:
validating it needs an exec-heavy benchmark (`apex`, `gdpval`), whose data is a
separate download. Its production blast radius is small — only `web_search` and
`scholar_search` are marked ranked, and both are capped solely by the 150K global
cap, so they are rarely truncated at production settings at all.

### Operational note

Do not edit `run-truncation-ab.sh` while it is running. bash re-reads a script
from a byte offset as it executes, so an in-place edit mid-run makes it execute a
fragment — that is how the n=100 run ended with
`line 53: reads: command not found` after both arms had finished, losing only the
final metrics step. Re-running `truncation_metrics.py` by hand recovered it.
Likewise, an edit to the truncation code itself lands in every question that
starts afterwards, because each question is a fresh subprocess.

## The cap experiment, and what all three runs add up to

Run 2026-08-22, DeepSearchQA, 40 questions per arm, **production window**
(262K/229K), global cap 150K, `auto` shape in both arms, arms
`TOOL_EXEC_RESULT_MAX_CHARS` = 8000 then 50000.

| metric | 8K | 50K |
|---|---|---|
| score | 12/40 (.300) | 11/40 (.275) |
| truncated results | 51 (bash 49, grep 2) | 3 (bash 3) |
| recovery reads | 21 | 2 |
| bash calls | 1070 | 855 |
| turns (mean) | 39.2 | 38.2 |
| peak prompt tokens (mean) | 57,845 | 62,766 |
| wall clock s (mean) | 672 | 817 |
| compactions (40 questions) | 4 | 1 |

Paired: score McNemar **p = 1.000** (8 discordant each way); wall clock sign test
p = 0.636, median +62s for 50K.

**The instrument was sound.** bash dominated the run and 49 of 51 truncations
were bash results, so the arms really did vary what they claimed to. This is not
the tautology it would have been on a purely search-driven run.

**Raising the cap 6.25× eliminated 94% of truncation and changed nothing
measurable.** Not score (p=1.000, as close to identical as 40 questions can
show), not turns, not significantly wall clock. So `8000` is not "too tight
versus kimi's 50000": the 6× gap has no observable consequence for this workload.

**And that explains the shape results.** If removing almost all truncation events
changes nothing, the shape of the few that remain can hardly matter — which is
exactly what two shape A/Bs found in two directions. The unifying conclusion
across all three runs is that **truncation is not on the critical path for this
workload**. The agent routes around it: the 8K arm made ~25% MORE bash calls
(1070 vs 855) and 10× the recovery reads, and still finished faster on average,
so narrowing a command is cheaper than shipping a 50K-char result through the
model.

**Also worth noting: at the production window the compaction machinery is nearly
idle** — 4 compactions across 40 questions, versus 300+ across 100 questions at
the 32K stress window. For this workload, neither truncation nor compaction is
where the runs are won or lost.

### What to do with that

- Keep the deterministic fixes. They are correctness, they are cheap, and they
  do not depend on any of the above being true.
- Stop tuning caps and preview shapes for search workloads. Three runs and
  ~6 hours of API budget say the expected value is near zero.
- The remaining context-offloading work worth doing is in
  [the follow-ups](context-offloading-followups.md), and the one most likely to
  matter is the compaction prompt (item 7) — tier 2 fired 180–192 times per
  100 questions under stress, and it is the only component that rewrites content
  the model then has to rely on.
- Evaluating any of it needs an exec/coding-heavy benchmark (`apex`, `gdpval`).
  DeepSearchQA has now been shown to be a weak instrument for this whole area.

## The compaction-prompt A/B (apex, exec-heavy)

The follow-up the three truncation runs pointed at. Run 2026-08-23, apex, 20
questions per arm, `--profile benchmark`, 65K window, production caps
(150K global / 8K exec), `COMPACTION_SPILL=true`, arms
`COMPACTION_PROMPT_STYLE` = research then handoff.

Two configuration traps, both worth knowing before repeating this:

- apex's sandbox profile defaults the workflow profile to `default`, which
  aliases `simple`, which sets **`context_compaction: "off"`**. Run apex that way
  and Tier 2 never fires — a compaction-prompt A/B measures nothing.
  `--profile benchmark` (tiered) is required, and `apply_sandbox_profile` uses
  `setdefault`, so an explicit `--profile` wins.
- The compacted summary never reaches any artifact. The trajectory observer
  records the message *stream*; compaction rewrites the loop's history, so the
  `[Compacted summary …]` message is not in the JSON or the JSONL. **Summary
  quality cannot be audited after the fact** — only behaviour can. Worth fixing
  by having the observer record the compaction event.

| metric | research | handoff | paired |
|---|---|---|---|
| score | 5/20 | 5/20 | McNemar **p = 1.000** (2 discordant each way) |
| peak prompt tokens (mean) | 47,518 | 43,741 | sign test **p = 0.012** (handoff lower on 16/20) |
| tool calls | 951 | 849 | p = 0.115 (handoff lower on 14/20) |
| turns (mean) | 44.6 | 39.0 | p = 0.503 |
| wall clock s (mean) | 1031 | 770 | **p = 1.000** (10 up, 10 down) |
| redone calls | 3 | 1 | p = 1.000 (18 ties) |
| compactions | 28 (t1 13, t2 15) | 25 (t1 9, t2 15, tc300 1) | — |

**The wall-clock gap is an artifact.** Mean 1031 → 770 looks decisive and is not:
the paired median delta is −5s and the split is exactly 10/10. One research task
ran 6733s longer than its handoff counterpart and carries the whole mean. The
interim reading of this run — "handoff is much faster" — was that single task.

**The one real signal: smaller prompts at identical score.** handoff's peak prompt
is lower on 16 of 20 questions, mean −8%. It is not explained by compacting more
often — handoff compacted slightly *less* (25 vs 28) with Tier 2 firing equally
often (15 vs 15). The mechanism is plausible: the research prompt asks for
`## Candidates`, `## Sources consulted`, `## Queries already run`, which on a
coding task are empty or padded, while the handoff shape spends its budget on
commands, paths and error text.

**Caveats, stated rather than buried.** Six metrics were tested, so at α=0.05 the
chance of one spurious hit is ~26%, and p=0.012 does **not** survive Bonferroni
for six comparisons (0.0083). It is the most credible signal in this whole
investigation — consistent direction, mechanistic explanation, unchanged score —
but at n=20 it is not established. Score is flat, so nothing here claims an
accuracy benefit.

**`redone calls` did not earn its keep here.** Built for this experiment, it fired
on 2 of 20 questions (18 ties) because apex saw only ~1.4 Tier 2 compactions per
question. It needs either many more questions or a tighter window to say anything.

### What was done with that

Not a global flip to handoff: the research shape was presumably tuned for the
benchmarks this framework mostly runs (browsecomp, deepsearchqa, hle), where
preserving candidates, ruled-out options and verbatim queries is the point, and
applying the handoff shape there risks a regression this run cannot see.

`COMPACTION_PROMPT_STYLE=auto` — now the default — dispatches at compaction time
on the conversation's own tool mix, via `ToolMeta.category`: a majority of
compute / file / search / finance calls picks handoff, a majority of web calls
picks research. This makes the default **strictly conservative for research
workloads**: a web-dominated conversation gets exactly the prompt it had before
`auto` existed, so the flip cannot regress the benchmarks the A/B did not cover.

Ties, empty conversations and a missing message list all go to research, on the
asymmetry of the two errors: routing a research run to handoff loses the
candidate and query preservation that is the whole point of that prompt, while
routing a coding run to research merely keeps the summary it has always had.
`meta` and `orchestration` calls cast no vote. `research` / `handoff` still force
one shape on everything, which is what an A/B arm needs.

## Appendix: how to run the cap A/B

Taking the two reference implementations seriously points at a different
variable. kimi truncates every tool result at **50,000 chars**
(`toolResultTruncationService.ts`); our exec cap is **8,000**
(`tool_exec_result_max_chars`). But a cap only means something relative to the
window it competes with:

| regime | per-result budget | share of window |
|---|---|---|
| kimi (50K cap, ~256K window) | 12,500 tok | **4.8%** |
| us, production (8K bash cap, 262K window) | 2,000 tok | **0.8%** |
| us, in the stress run above (8K cap, 32K window) | 2,000 tok | 6.1% |
| naive transplant (50K cap, 32K window) | 12,500 tok | 38.1% |

Two things follow.

**Our production cap is ~6× tighter than kimi's, relative to the window.** That
is the real finding behind "should we just copy them", and it is a question about
the cap, not about which half of the body survives it.

**The stress run above was already at kimi-like relative generosity** (6.1% vs
4.8%), which is a point in favour of its shape result transferring: the arms were
not competing in some artificially cramped regime.

**Do not transplant 50K onto a shrunk window.** A single result would take 38% of
a 32K window, so the large arm would spend the run in compaction and lose for a
reason that has nothing to do with the cap.

So the cap A/B has to run at a production-like window, where the cap fires (bash
output over 8K is routine) even though compaction rarely does — and compaction is
not what is being measured:

```
ARM_VAR=TOOL_EXEC_RESULT_MAX_CHARS ARMS="8000 50000" \
  OPENAI_CONTEXT_WINDOW=262144 OPENAI_MAX_INPUT_TOKENS=229376 \
  TOOL_RESULT_MAX_CHARS=150000 LIMIT=40 \
  scripts/run-truncation-ab.sh
```

The driver refuses to start if a per-tool arm exceeds `TOOL_RESULT_MAX_CHARS`,
since the loop's global cap would clamp the large arm and it would measure
nothing. That check exists because the default in this script is 8K.

### Reading the result

The change is worth keeping if, at equal or better score, `recovery reads` falls
and `turns` does not rise. Kept as written rather than moved to fit the result:
the n=30 run met the first half and not the second, and the n=100 run met
neither. Failure modes to watch for:

- **Score falls in `middle`.** ← **This is what happened.** The prescription
  written here before any data was run — "keep `middle` for exec/compute tools
  and `head` for ranked lists" — is what `auto` implements. Worth stating plainly
  because it means `auto` is a pre-registered prediction that came true, not a
  rationalisation built after seeing the numbers. The reason it applies:
  DeepSearchQA is relevance-ordered search, and `web_search` was the
  most-truncated tool in both runs.
- **A structured single-line payload comes back unparseable.** It did under the
  old shape too — a cut JSON body is invalid either way — but the middle shape
  now also keeps the closing keys (`next_cursor`, `total`), which is usually the
  half that decides the next call. If a tool needs a *valid* structured result
  rather than a readable one, it should page its own output, not lean on this.
- **`recovery reads` stays flat.** The model was not reading spill files in the
  first place, so the tail it now sees is not what was blocking it. The change
  is still correct on cap accounting, but do not claim a token win for it.

One caveat on `score` at this throttle: an 8K cap on fetched pages and a 32K
window degrade both arms, so the absolute number is not a production score and a
low one is not evidence against either shape. It is there to catch a *difference*
between the arms — and at a few dozen questions even that is only sensitive to a
large one.

`LIMIT=8` with `CONCURRENCY=4` is a signal check, not a benchmark result — small
enough to run in one sitting, and its score difference is noise unless it is
large. Raise `LIMIT` before quoting a score.
