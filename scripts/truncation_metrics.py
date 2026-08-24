"""Per-arm metrics for a live truncation A/B, read off benchmark run artifacts.

Consumes what a ``benchmarks.runner.run_subprocess`` run already writes —
``trials/<qid>/result.json`` for the score and
``trials/<qid>/agent/trajectories/*.jsonl`` for the turn-by-turn record — and
reports the numbers that separate the two preview shapes:

* ``recovery_reads``  — tool calls that go back for content a truncation cut:
  either one whose arguments name the spill store, or a ``recover_result`` call.
  This is the cost the head-only shape pays: an extra round-trip to fetch the tail
  it cut. It should FALL in the ``middle`` arm.
* ``handle_reads``    — the ``recover_result`` subset of the above, reported on its
  own line. The two mechanisms answer different questions — "did the agent read a
  spill file" and "did the agent use the trajectory handle" — and the merged total
  cannot support either claim by itself.

  Matching both ``.spill`` and ``/spill`` is not redundant. Item 5 moved the store
  out of the workspace, so pre-item-5 runs name ``…/.spill/<hash>.md`` and later
  ones name ``/spill/<hash>.md``. Matching only the dotted form — as this did —
  reads **zero** for every run after item 5, which silently turns "the agent never
  needed to recover" and "the metric stopped working" into the same number.
* ``truncations``     — tool results that came back cut. Should be ~equal across
  arms; if it is not, the arms did not see comparable work.

  Counts site-1/2 markers only, and structurally cannot do better: those cuts
  happen before the ``ToolResult`` is built, so the trajectory records an already
  short body. The site-3 post-processor runs AFTER ``notify_tool_result``, so its
  cuts leave no trace in any recorded body — a run where this reads 7 can still
  have had 43 results shortened on the way to the model. To count those, replay
  the recorded bodies through the workflow's own post-processor.
* ``peak_prompt_tokens`` / ``turns`` / ``tool_calls`` — the run's shape.
* ``compactions``     — counted from the ``TieredCompactor selected=`` lines in
  ``agent.log``, which also say WHICH tier won. Falls back to prompt-token drops
  between consecutive LLM calls when no log is present.
* ``redone calls``    — identical tool calls (same name and arguments) issued
  again AFTER a compaction has happened in that run. This is the instrument for a
  compaction-prompt A/B: a summary that drops what was already done is paid for
  by the agent repeating it, and unlike ``score`` it is mechanical and has decent
  power at a few dozen questions.
* ``score``           — from result.json, so a shape that saves tokens by losing
  the answer cannot look like a win.

    uv run python scripts/truncation_metrics.py results/ab-head results/ab-middle
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Iterator
from pathlib import Path

# History is append-only within a turn, so any real shrink between consecutive
# LLM calls means it was replaced. Both bounds together: a relative one to ignore
# rounding noise in reported usage, and an absolute one so a tiny early prompt
# cannot register a compaction on a few tokens of jitter.
_COMPACTION_DROP_RATIO = 0.95
_COMPACTION_DROP_TOKENS = 1_000
_SELECTED_RE = re.compile(r"TieredCompactor selected=(\S+)")
#: Tool names that ARE a recovery, whatever their arguments say.
_RECOVERY_TOOLS = frozenset({"recover_result"})
#: Both spellings of the store. See the module docstring on why the dotted form
#: alone is not enough.
_SPILL_MARKERS = ("/spill/", ".spill/", "/spill", ".spill")


def _names_recovery(tool_name: str, args: dict) -> bool:
    """True when this call is fetching back content a truncation cut."""
    if tool_name in _RECOVERY_TOOLS:
        return True
    blob = json.dumps(args, ensure_ascii=False)
    return any(marker in blob for marker in _SPILL_MARKERS)
# A ``tier2`` selection does NOT prove a summary was produced: a failed
# summariser rolls back to a deterministic slice that can still win under that
# label. Counting rollbacks separates "summarised" from "tried and broke",
# which every earlier A/B in this series reported as the same thing.
_ROLLBACK_RE = re.compile(r"rollback_reason=(?!-)(\S+)")


def _iter_events(trial: Path) -> Iterator[dict]:
    for jsonl in sorted((trial / "agent" / "trajectories").rglob("*.jsonl")):
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _trial_metrics(trial: Path) -> dict | None:
    result_path = trial / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    turns = tool_calls = recovery = truncations = compactions = 0
    peak = previous = 0
    # A call is "redone" when the same (name, args) was already issued earlier in
    # the run AND at least one compaction has happened since this run began. The
    # compaction gate matters: an agent legitimately re-runs a build or a test
    # after editing, and counting those would drown the signal.
    seen_calls: set[str] = set()
    redone = 0
    handle_reads = 0
    for event in _iter_events(trial):
        if event.get("t") == "llm":
            turns = max(turns, int(event.get("turn") or 0))
            calls = event.get("tool_calls") or []
            tool_calls += len(calls)
            for call in calls:
                args = call.get("args") or {}
                name = call.get("name") or ""
                if name in _RECOVERY_TOOLS:
                    # Counted separately from a spill read, because "the agent
                    # used the trajectory handle" and "the agent read a spill
                    # file" are different claims and the merged number cannot
                    # support either one on its own.
                    handle_reads += 1
                if _names_recovery(name, args):
                    recovery += 1
                signature = json.dumps(
                    [call.get("name"), args], sort_keys=True, ensure_ascii=False,
                )
                if signature in seen_calls and compactions:
                    redone += 1
                seen_calls.add(signature)
            prompt = int((event.get("usage") or {}).get("prompt_tokens") or 0)
            if prompt:
                if previous and (
                    prompt < previous * _COMPACTION_DROP_RATIO
                    and previous - prompt >= _COMPACTION_DROP_TOKENS
                ):
                    compactions += 1
                peak = max(peak, prompt)
                previous = prompt
        elif event.get("t") == "result":
            body = event.get("result")
            if isinstance(body, str) and (
                "chars elided" in body or "only part of this" in body
            ):
                truncations += 1
    # The log is authoritative when present: it names the tier that won, which a
    # token drop cannot distinguish.
    tiers: dict[str, int] = {}
    rollbacks: dict[str, int] = {}
    log = trial / "agent.log"
    if log.is_file():
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
            for match in _SELECTED_RE.finditer(text):
                tiers[match.group(1)] = tiers.get(match.group(1), 0) + 1
            for match in _ROLLBACK_RE.finditer(text):
                reason = match.group(1)
                rollbacks[reason] = rollbacks.get(reason, 0) + 1
        except OSError:
            pass
    return {
        "qid": trial.name,
        "compaction_tiers": tiers,
        "summary_rollbacks": rollbacks,
        "reward": result.get("reward"),
        "duration_s": result.get("duration_seconds"),
        "turns": turns,
        "tool_calls": tool_calls,
        "truncations": truncations,
        "recovery_reads": recovery,
        "handle_reads": handle_reads,
        "redone_calls": redone,
        "compactions": sum(tiers.values()) if tiers else compactions,
        "peak_prompt_tokens": peak,
    }


def _arm(run_dir: Path, keep: set[str] | None = None) -> dict:
    trials = [
        m
        for trial in sorted(run_dir.rglob("trials/*"))
        if trial.is_dir()
        and (keep is None or trial.name in keep)
        and (m := _trial_metrics(trial)) is not None
    ]
    rewards = [t["reward"] for t in trials if isinstance(t["reward"], (int, float))]

    def total(key: str) -> int:
        return sum(int(t[key] or 0) for t in trials)

    def mean(key: str) -> float:
        values = [float(t[key] or 0) for t in trials]
        return statistics.fmean(values) if values else 0.0

    tiers: dict[str, int] = {}
    rollbacks: dict[str, int] = {}
    for t in trials:
        for tier, count in (t.get("compaction_tiers") or {}).items():
            tiers[tier] = tiers.get(tier, 0) + count
        for reason, count in (t.get("summary_rollbacks") or {}).items():
            rollbacks[reason] = rollbacks.get(reason, 0) + count
    return {
        "run_dir": str(run_dir),
        "compaction_tiers": tiers,
        "summary_rollbacks": rollbacks,
        "trials": len(trials),
        "score": statistics.fmean(rewards) if rewards else None,
        "turns_mean": mean("turns"),
        "tool_calls_total": total("tool_calls"),
        "truncations_total": total("truncations"),
        "recovery_reads_total": total("recovery_reads"),
        "handle_reads_total": total("handle_reads"),
        "redone_calls_total": total("redone_calls"),
        "compactions_total": total("compactions"),
        "peak_prompt_tokens_mean": mean("peak_prompt_tokens"),
        "duration_s_mean": mean("duration_s"),
        "per_trial": trials,
    }


_ROWS = (
    ("trials", "trials", "{:.0f}"),
    ("score", "score", "{:.3f}"),
    ("turns (mean)", "turns_mean", "{:.1f}"),
    ("tool calls", "tool_calls_total", "{:.0f}"),
    ("truncated results", "truncations_total", "{:.0f}"),
    ("recovery reads (any)", "recovery_reads_total", "{:.0f}"),
    ("  of which recover_result", "handle_reads_total", "{:.0f}"),
    ("redone calls", "redone_calls_total", "{:.0f}"),
    ("compactions", "compactions_total", "{:.0f}"),
    ("peak prompt tok (mean)", "peak_prompt_tokens_mean", "{:.0f}"),
    ("wall clock s (mean)", "duration_s_mean", "{:.1f}"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="One run directory per arm")
    ap.add_argument("--labels", default="", help="Comma-separated arm labels")
    ap.add_argument("--json", default="", help="Also write the full report here")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if len(labels) != len(args.run_dirs):
        labels = [d.name for d in args.run_dirs]
    # Compare on the questions every arm actually scored. Per-arm totals
    # (recovery reads, tool calls) are only comparable over the same question
    # set, and the runner synthesizes a result rather than hanging when a
    # question hits its ceiling — so an arm losing one is normal, not an error.
    full = {label: _arm(directory) for label, directory in zip(labels, args.run_dirs, strict=True)}
    common: set[str] | None = None
    for arm in full.values():
        ids = {t["qid"] for t in arm["per_trial"]}
        common = ids if common is None else (common & ids)
    dropped = {
        label: sorted({t["qid"] for t in arm["per_trial"]} - (common or set()))
        for label, arm in full.items()
    }
    if len(labels) > 1 and any(dropped.values()):
        arms = {
            label: _arm(directory, keep=common)
            for label, directory in zip(labels, args.run_dirs, strict=True)
        }
    else:
        arms = full

    # Wide enough for the widest cell, not just the widest label: short arm
    # names like "8K" would otherwise run the numbers together.
    cells_by_label = {
        label: [
            "n/a" if arms[label][key] is None else fmt.format(arms[label][key])
            for _, key, fmt in _ROWS
        ]
        for label in labels
    }
    width = max(
        max((len(c) for cells in cells_by_label.values() for c in cells), default=0),
        max(len(label) for label in labels),
    ) + 2
    print(f"{'metric':<24}" + "".join(f"{label:>{width}}" for label in labels))
    print("-" * (24 + width * len(labels)))
    for idx, (title, _key, _fmt) in enumerate(_ROWS):
        print(
            f"{title:<24}"
            + "".join(f"{cells_by_label[label][idx]:>{width}}" for label in labels),
        )

    if len(labels) > 1 and any(dropped.values()):
        print(
            f"\ncompared on the {len(common or set())} questions scored in every arm; "
            "excluded per arm: "
            + ", ".join(f"{label}={len(ids)}" for label, ids in dropped.items()),
        )

    if any(arms[label]["compaction_tiers"] for label in labels):
        print("\ncompaction tiers selected:")
        for label in labels:
            tiers = arms[label]["compaction_tiers"]
            detail = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())) or "none"
            print(f"  {label}: {detail}")

    if any(arms[label].get("summary_rollbacks") for label in labels):
        # Printed right below the tiers on purpose: a ``tier2`` count includes
        # compactions where the summariser broke and its deterministic slice won,
        # so the two numbers have to be read together.
        print("\nsummariser rollbacks (tier2 selections that summarised nothing):")
        for label in labels:
            rb = arms[label].get("summary_rollbacks") or {}
            detail = ", ".join(f"{k}={v}" for k, v in sorted(rb.items())) or "none"
            print(f"  {label}: {detail}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(arms, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\nfull report -> {args.json}")
    if any(arm["trials"] == 0 for arm in arms.values()):
        print("\nWARNING: an arm has no readable trials", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
