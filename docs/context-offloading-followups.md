# Context offloading: the deferred half

Handoff for items 5–8 of the PR64 spill review. Items 1–4 shipped in that PR
(preview shape, footer budgeting, content-hash spill names, per-turn aggregate
budget) and are documented in
[Tool-result truncation A/B](tool-result-truncation-ab.md). The four below were
split out because each one changes something structural — a sandbox mount, the
shape of a `Message`, the summarizer contract, or where the durable record of a
run lives — and none of them should ride along with a truncation fix.

Everything here is written against two reference implementations that were read
in full for the review: **openai/codex** (`codex-rs/`) and
**MoonshotAI/kimi-code** (`packages/agent-core-v2/`). Their file paths are cited
inline so the next person can check the claim rather than trust it.

## What the two references establish

Spilling oversized tool output to a file and handing the model a path is not a
local invention — both do it. The disagreements are in the details, and the
details are what items 5–8 are about.

| | codex | kimi-code | us (after PR64) |
|---|---|---|---|
| tool output over cap | middle-truncated inline, no file (`codex-rs/utils/output-truncation/src/lib.rs`) | spilled at 50K chars, 2K head preview (`agent/toolResultTruncation/toolResultTruncationService.ts`) | spilled, head+tail preview |
| spill location | `$TMPDIR/hook_outputs/<thread_id>/` — hook output only (`codex-rs/hooks/src/output_spill.rs`) | `<homeDir>/<agent scope>/tool-results/` | **inside the agent's workspace** (`.spill/`) |
| compaction state | typed `ResponseItemEnvelope` / `CompactedItem`, projected for the wire (`codex-rs/app-server-protocol/src/protocol/thread_history_projection.rs`) | folded event log, `ContextApplyCompaction` event (`agent/contextMemory/contextOps.ts`) | **prose in a user message, re-parsed with regex** |
| compaction prompt | 9 lines, "handoff summary for another LLM" (`codex-rs/prompts/templates/compact/prompt.md`) | 75-line first-person handoff note (`agent/fullCompaction/compaction-instruction.md`) | research/QA-shaped structured summary |
| durable record | full rollout on disk; truncation only affects the projection | replayable event log + `blobref:` media offload (`agent/blob/agentBlobService.ts`) | trajectory JSONL (observer-owned) |
| no-summarizer path | token-budget compaction installs a fresh window (`codex-rs/core/src/compact_token_budget.rs`) | — | — |

Two things to carry as principles rather than as tasks:

- **The record and the projection are different objects.** codex keeps the full
  rollout and sends a projection of it; truncation is a property of what goes on
  the wire, not of what is retained. Every item below gets simpler under that
  split.
- **Compaction state should be typed, not written and re-read as prose.** Both
  references treat "this history was compacted, these are the recovery handles"
  as structured data. We render it into a markdown bullet list and then parse it
  back out with heuristics.

---

## 5. Move the spill store out of the agent-writable workspace — **done (2026-08-23)**

The store now has one root outside every write root — `APODEX_SPILL_DIR`, else
`<run>/spill`, else `<temp>/apodex-spill-<uid>` — reached through a canonical
`/spill` that is a **sibling** of `/workspace`, `/outputs` and `/inputs` rather
than a child of the first. `_overflow_dir`'s four-branch resolution collapsed to
one root plus three visibility rules (physical under `native`, canonical through
the mount otherwise, nothing for a remote backend).

**Per-backend protection, which is the part the original plan got wrong.** Only
bwrap can mount; the item's premise that six lexical guards existed only to
re-create one mount was false, and the guards stay:

| backend | what protects the store |
|---|---|
| `bwrap` | `--ro-bind-try <root> /spill`. No longer has to be the LAST bind — the source is outside the workspace, so nothing overlaps it and the ordering requirement that used to be load-bearing is gone. |
| `container` | **ownership, which moving the store is what bought.** It cannot mount (it reuses the task container's mounts), but model commands drop to an unprivileged uid and only the workspace and outputs dirs are handed to that uid by `_prepare_tool_writable`. Being outside both is the enforcement. A test asserts the store is never added to that call, because if it were, container silently loses its only protection. |
| `native` | nothing structural — no isolation exists. The lexical guards are the whole surface, which is why they could not be deleted. |
| `e2b` | no path advertised. |

**What did get deleted or consolidated**

- One predicate, `_sandbox.is_spill_path`, keyed off the real root, replacing a
  hardcoded `.spill` path component tested in four places. A fixed name did not
  cover `APODEX_SPILL_DIR` or a run directory inside the repository; this does.
- Write refusal for file tools now falls out of path authorization: the store is
  in no write root, so `write_access=True` never covers it. Reads are authorized
  explicitly, the same shape as `/inputs`.
- `cleanup_overflow_workspace`'s recursive walk went, with its symlink and
  filesystem-root defences. Those existed because it walked a tree inside the
  agent's own workspace; cleanup now removes only directories this process
  created. **Precondition:** one conversation per process — true for the terminal
  app and the benchmark's subprocess-per-question, false for a server
  multiplexing sessions, which must use `cleanup_overflow(scope)` instead.
- The legacy `data/tool-results` fallback and `_get_overflow_dir` are gone.

**Guard sites the item had miscounted: there were eight, not six.** The two extra
are a bash token scan in `_deliverable_policy` (`_SPILL_RAW_RE`,
`_SPILL_ASSIGNMENT_RE`, `_redirects_to_spill`, `_copy_only_reads_spill`) and the
`--exclude-dir` / `-prune` flags `grep_search` and `glob_search` add to their
sandbox-side commands. An earlier version of this document, and a comment in
`_sandbox.py`, claimed bash was not scanned for spill paths at all. It is.

**A constraint discovered by breaking it.** `_writer_core.py` is concatenated into
a standalone script and run where the `plugins` package does not exist, so it
cannot call the shared predicate and carries its own copy — which is why that
check was a literal in the first place. Delegating from there made `create_file`
exit 1 with `ModuleNotFoundError`. The copy is now cross-checked against the
authority by a test, under three different root configurations.

**Isolation, deliberately.** The root is shared (a temp or run directory), so
authorizing it wholesale would let one conversation read another's spilled bodies
— something the old in-workspace layout made impossible. Path auth therefore
authorizes only this conversation's scope plus the stores this process created;
the second half is needed because an in-process sub-agent spills under its own
scope and a fan-in report can carry that path back to its parent. The uid in the
root name keeps two accounts from sharing a store, but it is NOT a permission
boundary and cannot be: under `container` the model runs as a different uid and
must read these files, so the directory stays traversable by others.

**Not verified end to end.** Only `native` could be exercised on the development
host: bwrap cannot create a user namespace in a nested container
(`Operation not permitted`), and there is no container or e2b environment. The
bwrap and container guarantees rest on unit assertions over the mount arguments
and on the ownership test above. Anyone with access to those environments should
confirm a model command can read `/spill` and cannot write it.

## 6. Give the spill manifest a typed carrier — **done (2026-08-23)**

**What it was.** The recovery index was a markdown block inside a user message,
written by `_with_spill_manifest` and read back by `_spill_refs` through two
heuristics: `_is_spill_ref` (is this bullet a spill path?) and
`_manifest_block_start` (is this block an index, or a summary quoting one?).

Neither was incidental complexity. Each clause was a bug that had happened: a
path had to be absolute and carry no whitespace after the store marker, because
summarizer prose reached the same function; the header had to be on its own line
followed only by `- ` bullets, because `format_conversation_for_summary` renders
the previous index verbatim and a summary that quoted the header would otherwise
be truncated at the quote — dropping every finding after it — and have its own
bullets harvested as dead refs that evict real ones from the bounded index.

**What it is now.** `Message.spill_refs: list[str]` carries the paths as data.
Both producers set it — the Tier 1 tool placeholder and the index itself — and
`_spill_refs` reads the field. The rendered text stays, because the model is what
actually acts on a path; nothing reads the text back.

The index also became its own message rather than being appended into the summary.
That is what removes the bug class at the root: updating the index no longer means
locating where it begins inside prose a model wrote, so a summary is never edited
and an echoed header is just words.

`-95/+45` in `tiered_compact.py`, both heuristics deleted, and the tests that
pinned their edge cases were rewritten to assert the new invariant while recording
which bug class each one used to guard.

**Two legacy reads kept, deliberately.** A history checkpointed before the field
existed still resumes: a `[Full text] <path>` line in a Tier 1 placeholder is read
by fixed prefix (a format, not a shape guess), and a prose index is left in place
so its paths stay visible to the model even though they are not harvested. The two
remaining `SPILL_MANIFEST_HEADER` substring checks exist only for that case and
can go once no such checkpoint can be resumed.

**Enabled by 6a.** `spill_refs` is outside `WIRE_MESSAGE_KEYS`, so `for_wire`
strips it at the provider boundary with no per-path audit.

## 7. Replace the compaction prompt for long-run coding work

**Now.** `COMPACTION_PROMPT` (`frontier_agent/infra/llm/summary_prompt.py:13`)
is shaped for research QA: its `PRESERVE EXACTLY` list is entity names, ruled-out
candidates, source URLs and verbatim search queries, and its output sections are
`## Candidates`, `## Sources consulted`, `## Queries already run`. For a
tool-heavy coding run it never asks for the exact commands that were run, the
files that were touched, the error text that came back, or which claims are
still unverified.

**What the references do.** kimi's `compaction-instruction.md` asks the model for
a **first-person** note to itself — present tense, its own train of thought —
and names what to keep: the exact commands, the exact paths, whether each
succeeded, the concrete values returned, decisions already settled kept separate
from questions still open, and an explicit forward plan. It ends with an
instruction to flag anything an earlier step *claimed* was done but never
verified, and to treat it as unverified rather than fact. It also keeps the most
recent user messages verbatim (size-capped) rather than summarizing them. codex
goes the other way — nine lines — but still asks for "critical data, examples, or
references needed to continue".

**Approach.** Either select a prompt by task shape, or write one prompt whose
preserve-list is task-neutral (exact commands and paths, returned values and
error text, settled-vs-open decisions, forward plan, honest unverified flags) and
let the research-specific sections go. Add an explicit instruction to preserve
spill paths that appear in the history, so the prose half backs up the typed
manifest from item 6 instead of depending on it.

**Verification.** The same harness as PR64: `scripts/run-truncation-ab.sh` with
the prompt as the arm variable instead of the truncation mode. It wants a coding
benchmark rather than DeepSearchQA — `apex` or `gdpval`, which need the separate
per-dataset download in [the evaluation guide](eval.md#file-benchmarks).

---

> **Status update, 2026-08-23.** Item 7 is done: `HANDOFF_COMPACTION_PROMPT`
> plus `COMPACTION_PROMPT_STYLE=auto`, which dispatches on the conversation's
> tool mix. Measured on apex — identical score, prompts 8% smaller
> (p=0.012 unadjusted, not surviving Bonferroni across six metrics). See
> [the A/B write-up](tool-result-truncation-ab.md).
>
> Two things that experiment surfaced, both belonging to item 8:
>
> - **The compacted summary reaches no artifact.** `TrajectoryFileObserver`
>   records the message stream; compaction rewrites the loop's history, so the
>   `[Compacted summary …]` message appears in neither the JSON nor the JSONL.
>   Summary quality therefore cannot be audited after a run — only behaviour can.
>   Anything that makes the transcript the source of truth should record the
>   compaction event itself, with the summary it installed and the messages it
>   replaced.
> - **`redone_calls`** (identical tool calls reissued after a compaction, in
>   `scripts/truncation_metrics.py`) is the right instrument for judging whether a
>   summary preserved what was already done, but it needs far more compactions per
>   run than apex produced (~1.4) to say anything. A tighter window or a
>   longer-horizon benchmark would make it usable.

## 8. Make the durable transcript the source of truth — **partly done (2026-08-23)**

**The observation this item was built on was wrong — corrected 2026-08-23.**

The original claim: *"`TrajectoryFileObserver` spools one JSONL line per event with
every tool-result body untruncated, so the content `_write_spill` copies is already
on disk a second time."* The `_BODY_MAX_CHARS` half is true — that 16K bound applies
only to the JSON envelope, not the JSONL stream. The conclusion is not, and it is
worth being precise about why, because it inverts the design.

A tool result is cut in **three** places, and only the third leaves the trajectory
holding anything the model cannot already see:

| | where | cap | persists the original? | in the recorded `ToolResult`? |
|---|---|---|---|---|
| site 1 | `tool_exec.py:244`, inside `execute_tools` | 150 000 | yes — `_write_spill` | **yes, already cut** |
| site 2 | `_apply_aggregate_budget`, `tool_exec.py:299` | per-turn aggregate | yes | **yes, already cut** |
| site 3 | `ToolResultPostProcessor`, applied `agent_loop.py:762` | **15 000** for sub-agents (`subagent_runtime.py:532`); bash 4 000 | **no — nothing at all** | no |

Sites 1 and 2 cut `result_str` **before** the `ToolResult` is constructed
(`tool_exec.py:243-247`), so the observer faithfully records a body that was already
truncated. Measured: a 60 019-char tool return with the cap at 3 000 lands in the
JSONL as **3 000 chars, footer included**. A recovery tool reading the trajectory for
those sites would hand the model back the very preview it is looking at.

Site 3 is the opposite, and by luck of ordering: `agent_loop.py:761-762` calls
`notify_tool_result` **first**, and the post-processor's return value flows only into
`tool_msg(...)`. So the `ToolResult` — and therefore the trajectory — keeps the
pre-site-3 body, while the message the model sees is cut at a far smaller cap with no
spill and no footer. That discarded content exists **only** in the trajectory, and the
trajectory is not sandbox-visible, so an in-process tool is the only way to reach it.

So the trajectory is not a second copy of what spill already holds. It is the sole
copy of what site 3 throws away — which is a smaller claim, and a more useful one.

**Where the references are.** codex separates the rollout from the projection
sent to the model (`thread_history_projection.rs`, `ResponseItemEnvelope`,
`CompactedItem`), which is what lets resume and fork read full fidelity while the
wire sees a truncated view. kimi's context memory is a replayable event log whose
blob layer dehydrates media parts to `blobref:` on write and rehydrates on read
(`agent/contextMemory/contextOps.ts`, `agent/blob/agentBlobService.ts`).

**Approach, as revised.** The original one — *"a recovery tool reading the
trajectory would replace spill copies entirely, and the cleanup machinery in
`_overflow.py` goes away with it"* — cannot work, for the reason above: for the sites
spill covers, the trajectory holds the same truncated preview. Spill stays exactly
where it is.

What the trajectory can do is cover **site 3**, which nothing covers today. So the
recovery tool reads the trajectory, the footer at site 3 names a
`(turn, tool_call_id)` handle where today there is no pointer at all, and sites 1
and 2 are left alone. Nothing is deleted, and the model still never receives a
filesystem path — the security half of the original idea survives intact, because a
handle is opaque whatever store backs it.

**Why the tool must be in-process.** Every existing recovery route asks the model to
read a path with a file tool, which is why the spill store needs a bwrap mount, a
`_path_auth` prefix, and eight write guards. The trajectory has no mount and should
not get one. A tool that never imports `plugins.tools._sandbox` runs in the harness
and reads the file directly — no mount, no authorization, and no traversal surface,
since the model names a handle rather than a path.

**Why the four blocking properties no longer block.** They were blockers for
*replacing* spill. Scoped to site 3 they become fallbacks: when jsonl is off, or the
observer is unbound, or the handle cannot be resolved, site 3 degrades to exactly
today's behaviour — a plain char-count marker. Nothing becomes load-bearing, so no
deployment guarantee is needed.

### What landed for site 3 — **done (2026-08-23)**

`plugins/tools/recover_result.py`, an in-process tool taking `(turn, call_id)`. The
footer is minted in `agent_loop.py` rather than in any post-processor: `ctx.turn`, the
pre-processor body and the post-processor body are all in hand there, so the Protocol
and its three implementations stay untouched. The predicate is
`len(post) < len(pre)` — not an approximation of "was it truncated" but the exact
condition under which recovery helps, since it says the trajectory holds something the
model cannot see.

**The footer is gated on the tool being bound for that agent**, not on a config flag.
Profiles carry their own tool lists — the stateful_react benchmark profile binds no
reader at all — and `_spill_footer` already carries a comment about what happens when
a footer names a tool the agent cannot call.

**Two things the review caught that would have made it a no-op.**

`recover_result` was not in either post-processor's `_PASS_THROUGH`, so its own output
would have been head-capped at the 6 000-char default — cutting the continuation
pointer off the content the call exists to fetch. It is whitelisted in both now, for
the reason `read_file` already is: it paginates itself and its trailing offset is part
of the recovery contract. Its slice cap is 8 000 to match that convention rather than
being the largest thing in history.

And an empty `call_id` must never match. Pre-8a trajectories have no `tool_call_id`
field, and treating the absent field as `""` made a scan match the last result of that
turn and return **the wrong body while reporting success** — measured on a real
trajectory file. Both the tool and the footer refuse empty ids.

The scan takes the **last** match after the final `t:"start"`. The file is opened in
append mode under a deterministic stem, so re-running a task accumulates runs in one
file, and `agent_loop.py` decrements the turn counter on `continue_to_next_turn`, so a
turn can be attempted twice with colliding synthetic ids.

What site 3 told the agent before this: *"re-fetch or rerun with a focused query if
you need the rest"* — redo the work. It can now fetch instead.

Not measured yet: whether the agent uses it, and whether recovering beats re-running.
The A/B for that is `ARM_VAR=TOOL_RESULT_RECOVERY`, and the metric it needs was dead
until this branch fixed it (see the retraction above).

**Adjacent, smaller, and independently useful:** codex's token-budget compaction
(`compact_token_budget.rs`) skips the summarizer entirely and installs a fresh
window, reinjecting initial context per `InitialContextInjection`. That is a
cleaner last resort than our `tool_compression_300` candidate in
`TieredCompactor.compact`, and it does not depend on any of the above.

Measured, not assumed: the PR64 A/B run (29 DeepSearchQA questions, 32K window)
selected `tier1` 26 times, `tier2` 50 times and **`tool_compression_300` 18
times** — so the crudest fallback, which blanks tool bodies to 300 characters, is
on roughly a fifth of all compactions rather than being the unreachable
last resort it reads as. A fresh-window path would replace those 18. kimi's
`blobref:` pattern is the right answer for images if media ever enters our
history.

### What landed, and what the code turned out to be

Two prerequisites were missing rather than merely unbuilt, and both are now in.

**The JSONL record was not addressable.** It carried `turn`, `name`, `result`,
`error` and `ms` — no `tool_call_id`. `parallel_tool_calls` is enabled, so one
turn routinely holds several results from the same tool and `(turn, name)`
identifies none of them; the handle this item proposes could not have been
built. `ToolResult.tool_call_id` is a required field, so recording it was a
one-line change. The JSON branch's id-synthesis fallback is deliberately not
shared: it advances `_tool_results_seen`, so driving it from the JSONL path
would double-count, and a synthesised id matches nothing outside that snapshot.

Recording the id was right; the reason given for it was not. Commit `33ca9ae`'s
message says the JSONL *"is the only record that keeps tool-result bodies
untruncated … which makes it the copy a recovery tool should read instead of a second
copy under the spill store."* The second half is the error corrected at the top of
this item — it holds for site 3 only, not for the sites spill covers. The commit is
pushed and its message cannot be amended, so the correction lives here.

**The compaction event existed and was wired to nothing.** `compact_llm.py`
carries a full `emit_event` mechanism — `_emit`, `_PENDING_EMITS`, an
async fire-and-forget path — and **no caller anywhere passed it**, so
`self._emit_event` was always `None`. Its success payload also omitted the
summary text, so even once wired it would have reported that a compaction
happened and how much it freed, but nothing about what survived. The blind spot
was unwired infrastructure, not absent infrastructure.

The fix keeps the summariser out of the broadcast business: `emit_event` is now
the *internal* channel by which `LLMSummaryCompactor` hands its summary up to
`TieredCompactor.last_event`, and the agent loop — which is the only party that
knows the turn number and the compaction sequence — stamps those and calls a new
passive `on_compaction` observer hook. `TrajectoryFileObserver` writes a
`t: "compaction"` JSONL record carrying the selected tier, the token pair,
`relief_met`, the spill-ref count, and the summary whole.

Three traps worth keeping in the tests. Tier 2 can **run and still lose** to a
cheaper candidate, so the event keys its summary off the selected label rather
than off the stash being populated. The stash is cleared per compaction, so an
earlier turn's summary can never be reported as this turn's. And a summariser
that **fails** rolls back to a deterministic slice which can itself be the
smallest candidate and win under the `tier2` label — measured, not assumed: with
ten tool-calling turns of history the slice beats Tier 1 every time, because
Tier 1 blanks tool bodies but keeps every message including the assistant
reasoning, while the slice drops the whole middle. Recording only an empty
summary there would read as "nothing was summarised" when in fact the summariser
ran and broke, so `rollback_reason` carries `llm_error` / `empty_summary` and
the three outcomes stay distinguishable.

Note for anyone writing a loop-level compaction test: compaction runs at turn
end, and a turn that answers with `finish_reason="stop"` returns before getting
there. The stub has to call a tool.

### What is still open — and why it is not a small change

Replacing spill copies with `(turn, tool_call_id)` handles now has its handle,
but four properties still do not hold:

1. The observer is **optional** and `critical = False`. Nothing guarantees it is
   bound, and a recovery tool that reads a file nobody wrote fails silently.
2. A **JSON-only** configuration keeps only the 16K `_BODY_MAX_CHARS` clip, so
   recovery would return truncated content while looking like it succeeded —
   worse than today's honest "not readable from this backend".
3. **Sub-agents each own their own file**, so a handle needs an agent identity
   as well as a turn and a call id.
4. Promoting the trajectory from diagnostics to a load-bearing store means it
   can no longer be disabled for cost, which is a deployment decision, not a
   refactor.

Item 8's remaining half should be planned against those four, not started from
the handle.

---

## Suggested order

| # | Item | Blast radius | Prerequisite |
|---|---|---|---|
| ~~6a~~ | `for_wire()` helper in `core/messages.py` | **done** | — |
| ~~6b~~ | Typed manifest carrier | **done** | — |
| ~~5~~ | Spill store out of the workspace | **done** | — |
| 7 | Compaction prompt | prompt + an A/B run | coding-benchmark data |
| 8a | `tool_call_id` in the record + compaction observability | **done** | — |
| 8b | Site-3 truncation made recoverable from the trajectory | **done** | — |
| ~~8c~~ | Spill copies replaced by trajectory handles | **dropped** — the trajectory holds the same preview spill does at those sites |

5 and 6 are independent and can go in parallel. 7 is gated on data, not code. 8
should wait until the store and the manifest have stopped moving.
