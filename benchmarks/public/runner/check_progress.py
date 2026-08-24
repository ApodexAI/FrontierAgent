"""Real-time progress monitor for benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Terminal colours ──────────────────────────────────────────────────────────

_R = "\033[0m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _colour(pct: float) -> str:
    if pct >= 65:
        return _GREEN
    if pct >= 45:
        return _YELLOW
    return _RED


def _bar(pct: float, width: int = 26) -> str:
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"{_colour(pct)}[{bar}] {pct:5.1f}%{_R}"


def _dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ── Per-run collection ────────────────────────────────────────────────────────


@dataclass
class RunStats:
    name: str
    n_done: int = 0
    n_correct: int = 0
    n_wrong: int = 0
    n_unscored: int = 0
    n_running: int = 0
    n_pending: int = 0
    n_errors: int = 0
    durations: list[float] = field(default_factory=list)
    grading: Counter = field(default_factory=Counter)
    wrong_samples: list[dict] = field(default_factory=list)
    earliest_done_ts: float = 0.0
    latest_done_ts: float = 0.0


def _scan_trials_dir(trials_root: Path, run_name: str) -> RunStats:
    s = RunStats(name=run_name)
    if not trials_root.exists():
        return s
    for td in sorted(trials_root.iterdir()):
        if not td.is_dir():
            continue
        result_path = td / "result.json"
        if result_path.exists():
            r = _load_json(result_path)
            correct = r.get("is_correct", False)
            err = r.get("error") or ""
            dur = r.get("duration_seconds") or 0
            s.n_done += 1
            if correct is None:
                s.n_unscored += 1
            elif bool(correct):
                s.n_correct += 1
            else:
                s.n_wrong += 1
                if len(s.wrong_samples) < 30:
                    s.wrong_samples.append({
                        "id": td.name,
                        "duration": dur,
                        "error": err,
                        "predicted": str(r.get("predicted_answer") or "")[:60],
                        "ground_truth": str(r.get("ground_truth") or "")[:40],
                    })
            if err:
                s.n_errors += 1
            if dur > 0:
                s.durations.append(dur)
            s.grading[r.get("judge_method") or r.get("grading_method", "N/A")] += 1
            mtime = result_path.stat().st_mtime
            s.earliest_done_ts = (
                min(s.earliest_done_ts, mtime) if s.earliest_done_ts else mtime
            )
            s.latest_done_ts = max(s.latest_done_ts, mtime)
        elif (td / "agent").exists():
            s.n_running += 1
        else:
            s.n_pending += 1
    return s


# ── Layout detection ──────────────────────────────────────────────────────────


def _detect_runs(run_dir: Path) -> list[tuple[str, Path]]:
    """Return ``[(run_name, trials_dir), ...]``.

    Multi-run layout: ``<run_dir>/run_N/trials/`` for N=1..R.
    Single-run layout: ``<run_dir>/trials/``.
    """
    multi = sorted(p for p in run_dir.glob("run_*") if (p / "trials").is_dir())
    if multi:
        return [(p.name, p / "trials") for p in multi]
    if (run_dir / "trials").is_dir():
        return [(run_dir.name, run_dir / "trials")]
    return []


def _expected_total(run_dir: Path, cfg: dict) -> int | None:
    """Best-effort total trial count.

    Reads ``launcher.log`` for ``Loaded N questions × R run(s)``; if that
    fails, derive from ``config.json`` (``limit`` or counting ``tasks/``).
    """
    launcher_log = run_dir / "launcher.log"
    if launcher_log.exists():
        try:
            for line in launcher_log.open():
                m = re.search(r"Loaded\s+(\d+)\s+\S+\s+questions\s+×\s+(\d+)\s+run", line)
                if m:
                    return int(m.group(1)) * int(m.group(2))
        except Exception:
            pass
    limit = cfg.get("limit")
    runs = cfg.get("runs") or 1
    if limit:
        return int(limit) * int(runs)
    tasks_dir = run_dir / "tasks"
    if tasks_dir.is_dir():
        return len(list(tasks_dir.iterdir())) * int(runs)
    return None


# ── Reporting ─────────────────────────────────────────────────────────────────


def analyse(run_dir: Path, verbose: bool = False) -> None:
    cfg = _load_json(run_dir / "config.json")
    benchmark = cfg.get("benchmark", "unknown")
    runs_cfg = cfg.get("runs") or 1
    concurrency = cfg.get("concurrency", "?")

    layout = _detect_runs(run_dir)
    if not layout:
        print(f"{_RED}No trials/ or run_*/trials/ found under {run_dir}{_R}")
        return

    stats = [_scan_trials_dir(trials, name) for name, trials in layout]
    is_multi = len(stats) > 1 or (len(layout) == 1 and layout[0][0].startswith("run_"))

    n_done = sum(s.n_done for s in stats)
    n_correct = sum(s.n_correct for s in stats)
    n_wrong = sum(s.n_wrong for s in stats)
    n_unscored = sum(s.n_unscored for s in stats)
    n_running = sum(s.n_running for s in stats)
    n_pending = sum(s.n_pending for s in stats)
    n_errors = sum(s.n_errors for s in stats)

    expected_total = _expected_total(run_dir, cfg) or (n_done + n_running + n_pending)
    completion_pct = (n_done / expected_total * 100) if expected_total else 0
    n_scored = n_correct + n_wrong
    accuracy_pct = (n_correct / n_scored * 100) if n_scored else 0

    # Throughput / ETA: trials/sec from earliest-done to now (or to latest-done
    # if process exited)
    all_done_ts = [s.earliest_done_ts for s in stats if s.earliest_done_ts] + \
                  [s.latest_done_ts for s in stats if s.latest_done_ts]
    rate = 0.0
    if all_done_ts:
        elapsed = max(all_done_ts) - min(all_done_ts)
        if elapsed > 0 and n_done > 1:
            rate = n_done / elapsed
    remaining = max(0, expected_total - n_done) if expected_total else 0
    eta_str = "–"
    if rate > 0 and remaining > 0:
        eta_str = f"~{_dur(remaining / rate)}"
    elif remaining == 0 and n_done > 0:
        eta_str = "done"

    avg_dur = sum(d for s in stats for d in s.durations) / max(
        1, sum(len(s.durations) for s in stats)
    )

    # ── Header ───────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("═" * 72)
    print(f"  {_BOLD}FrontierAgent Benchmark Monitor{_R}  {_DIM}{now}{_R}")
    print(f"  Run:        {_CYAN}{run_dir.name}{_R}")
    print(f"  Benchmark:  {_CYAN}{benchmark}{_R}  "
          f"({runs_cfg} run(s), concurrency={concurrency})")
    print("═" * 72)

    # ── Aggregate progress ───────────────────────────────────────────────────
    print()
    print(f"  Completion  {_bar(completion_pct)}   {n_done}/{expected_total}")
    if n_scored > 0:
        print(f"  Accuracy    {_bar(accuracy_pct)}   {n_correct}/{n_scored}")
    elif n_unscored > 0:
        print(f"  Answers     {_CYAN}{n_unscored}{_R} collected; official scoring pending")

    print()
    status_parts = [
        f"Done: {_BOLD}{n_done}{_R}",
        f"Running: {_YELLOW}{n_running}{_R}",
        f"Pending: {_DIM}{n_pending}{_R}",
    ]
    if n_errors:
        status_parts.append(f"Errors: {_RED}{n_errors}{_R}")
    print(f"  {'  │  '.join(status_parts)}  │  ETA: {eta_str}")
    if rate > 0:
        print(f"  Throughput: {rate * 60:.1f} trials/min  │  "
              f"Avg trial: {_dur(avg_dur)}")

    # ── Per-run breakdown (multi-run only) ───────────────────────────────────
    if is_multi:
        print()
        print(f"  {_BOLD}Per-run{_R}")
        for s in stats:
            scored = s.n_correct + s.n_wrong
            if scored:
                acc = s.n_correct / scored * 100
                score_text = f"{_colour(acc)}{s.n_correct}/{scored} ({acc:5.1f}%){_R}"
            else:
                score_text = f"{_CYAN}{s.n_unscored} collected{_R}"
            print(f"    {_CYAN}{s.name:<10}{_R}  {score_text}"
                  f"  running={_YELLOW}{s.n_running}{_R}"
                  f"  pending={_DIM}{s.n_pending}{_R}"
                  f"  errors={_RED}{s.n_errors}{_R}")

    # ── Grading methods ──────────────────────────────────────────────────────
    if n_done > 0:
        all_grading: Counter = Counter()
        for s in stats:
            all_grading.update(s.grading)
        methods = "  ".join(f"{m}: {c}" for m, c in sorted(all_grading.items()))
        print(f"\n  {_DIM}Grading: {methods}{_R}")

    # ── Verbose: show some wrong answers ─────────────────────────────────────
    if verbose and n_wrong > 0:
        print(f"\n  {_RED}✗ Sample wrong answers (first 10){_R}")
        all_wrong = [w for s in stats for w in s.wrong_samples][:10]
        for d in all_wrong:
            err_tag = f"  {_RED}[error: {d['error'][:30]}]{_R}" if d["error"] else ""
            print(f"    {_DIM}#{d['id']:<8}{_R}  truth={d['ground_truth']!r:<32}  "
                  f"pred={d['predicted']!r:<40}  ({_dur(d['duration'])}){err_tag}")

    print()
    print("═" * 72)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor an FrontierAgent benchmark run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir",
                        help="Run output directory (single-run trials/ or multi-run run_*/trials/)")
    parser.add_argument("--watch", "-w", type=int, metavar="SEC", default=0,
                        help="Auto-refresh interval in seconds (default: one-shot)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show sample wrong answers")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return

    if args.watch > 0:
        try:
            while True:
                os.system("clear")
                analyse(run_dir, verbose=args.verbose)
                print(f"  {_DIM}Refreshing every {args.watch}s — Ctrl+C to stop{_R}\n")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        analyse(run_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
