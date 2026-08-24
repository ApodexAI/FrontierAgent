#!/usr/bin/env python3
"""Compare a post-refresh benchmark run against the frozen golden baseline.

    python tools/compare_golden.py results/golden_react_base_2026-08-03 results/after

Aggregate accuracy matching is necessary but not sufficient — the same score can
hide a set of questions flipping both ways. This reports per-question flips,
which is what actually tells you whether the kernel refresh changed behaviour.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load(run_dir: Path) -> dict[str, dict]:
    results = run_dir / "results.json"
    if results.exists():
        data = json.loads(results.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("results", [])
        return {str(r["question_id"]): r for r in rows}
    # Fall back to per-trial files (a run killed mid-flight still has these).
    out = {}
    for p in sorted(run_dir.glob("trials/*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[str(r["question_id"])] = r
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    gold_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    gold, new = load(gold_dir), load(new_dir)
    if not gold:
        print(f"no results found in {gold_dir}")
        return 2

    if (meta := gold_dir / "GOLDEN.txt").exists():
        print(meta.read_text(encoding="utf-8").rstrip())
        print()

    common = sorted(set(gold) & set(new))
    only_gold, only_new = sorted(set(gold) - set(new)), sorted(set(new) - set(gold))

    def acc(d: dict[str, dict[str, Any]], keys: Sequence[str]) -> float:
        return sum(1 for k in keys if d[k].get("reward") == 1) / len(keys) if keys else 0.0

    print(f"golden {len(gold)} q   new {len(new)} q   comparable {len(common)}")
    if only_gold:
        print(f"  missing from new run: {', '.join(only_gold[:10])}")
    if only_new:
        print(f"  extra in new run:     {', '.join(only_new[:10])}")
    print(f"\naccuracy  golden {acc(gold, common):.1%}   new {acc(new, common):.1%}")

    flips = [
        (q, gold[q].get("reward"), new[q].get("reward"))
        for q in common
        if gold[q].get("reward") != new[q].get("reward")
    ]
    regressions = [f for f in flips if f[1] == 1]
    print(f"\nflipped: {len(flips)}  ({len(regressions)} correct->wrong, "
          f"{len(flips) - len(regressions)} wrong->correct)")
    for q, g, n in flips:
        arrow = "REGRESSION" if g == 1 else "improved  "
        print(f"  {arrow}  {q}  reward {g} -> {n}")
        print(f"      gold: {str(gold[q].get('predicted_answer'))[:100]!r}")
        print(f"      new:  {str(new[q].get('predicted_answer'))[:100]!r}")

    errs = [q for q in common if new[q].get("error") and not gold[q].get("error")]
    if errs:
        print(f"\nnew errors on {len(errs)} question(s) that were clean before:")
        for q in errs:
            print(f"  {q}: {new[q]['error']}")

    # Non-zero exit only for behaviour that got worse — usable as a CI gate.
    return 1 if (regressions or errs) else 0


if __name__ == "__main__":
    sys.exit(main())
