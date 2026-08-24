#!/usr/bin/env python3
"""Lint scores.json schemas — catch silent schema drift in per-query scorers.

run_all.py 通过 ``extract_total_rates`` 读每题的 ``scorers/query_NN/auto_scores/
scores.json``。如果某 scorer 写出来的 schema 跟 reader 假设的不一致，那一题在
最终 ranking 里会**静默消失**——scorer 自身跑通、模型也答了，但聚合矩阵就是
读不出 rate。这就是过去 Q03/Q28/Q39 出现过的问题。

本脚本做两件事：

1. Synthetic reader tests — 用拼装数据覆盖所有 reader 接受的 schema 变体
   （dict-form / list-form / total_score / total / total_rate / score_rate）。
2. Real-output structural test — 扫每个真实存在的 scores.json，断言
   extract_total_rates 能读出非空 ``{model: rate}``。

Exit 0 全 PASS，否则 1。建议在 ``run_all.py`` 跑完后做一次，或作为 pre-merge
检查跑一次，保证新增 scorer 没把 schema 写歪。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from run_all import extract_total_rates  # noqa: E402


def _close(
    actual: dict[str, float], expected: dict[str, float], label: str
) -> tuple[bool, str]:
    if set(actual) != set(expected):
        return False, (
            f"FAIL: {label} — keys differ. "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    for k in expected:
        if abs(actual[k] - expected[k]) > 1e-9:
            return False, (
                f"FAIL: {label} — {k}={actual[k]:.4f}, expected {expected[k]:.4f}"
            )
    return True, f"PASS: {label}"


SYNTHETIC_CASES: list[tuple[str, dict, dict[str, float]]] = [
    (
        "dict-form + total_score + per-entry max_score",
        {"results": {"m": {"total_score": 2, "max_score": 4}}},
        {"m": 0.5},
    ),
    (
        "dict-form + total_score + top-level max_score",
        {"max_score": 4, "results": {"m": {"total_score": 2}}},
        {"m": 0.5},
    ),
    (
        "dict-form + total_rate overrides total_score+max_score",
        {"max_score": 4, "results": {"m": {"total_score": 999, "total_rate": 0.5}}},
        {"m": 0.5},
    ),
    (
        "dict-form + score_rate alias",
        {"max_score": 4, "results": {"m": {"score_rate": 0.75}}},
        {"m": 0.75},
    ),
    (
        "dict-form + total alias (legacy Q03 shape pre-fix)",
        {"max_score": 4, "results": {"m": {"total": 1}}},
        {"m": 0.25},
    ),
    (
        "list-form + total_score (legacy Q28/Q39 shape pre-fix)",
        {"max_score": 3, "results": [{"model": "m", "total_score": 3}]},
        {"m": 1.0},
    ),
    (
        "list-form + total_rate",
        {
            "max_score": 3,
            "results": [{"model": "m", "total_score": 1, "total_rate": 0.9}],
        },
        {"m": 0.9},
    ),
    (
        "multi-model dict-form",
        {
            "max_score": 4,
            "results": {
                "a": {"total_score": 4},
                "b": {"total_score": 2},
                "c": {"total_score": 0},
            },
        },
        {"a": 1.0, "b": 0.5, "c": 0.0},
    ),
    (
        "non-dict / non-list results → empty",
        {"max_score": 4, "results": "garbage"},
        {},
    ),
    (
        "missing results → empty",
        {"max_score": 4},
        {},
    ),
]


def run_synthetic_tests() -> tuple[int, int]:
    print("─" * 64)
    print("[1/2] Synthetic reader tests")
    print("─" * 64)
    passed, failed = 0, 0
    for label, scores_in, expected in SYNTHETIC_CASES:
        rates, _ = extract_total_rates(scores_in)
        ok, msg = _close(rates, expected, label)
        print(("  ✓ " if ok else "  ✗ ") + msg)
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"  → {passed}/{passed + failed} passed")
    return passed, failed


def run_real_output_tests() -> tuple[int, int, int]:
    print()
    print("─" * 64)
    print("[2/2] Real-output structural tests")
    print("─" * 64)
    scorers_root = THIS / "scorers"
    paths = sorted(scorers_root.glob("query_*/auto_scores/scores.json"))
    if not paths:
        print("  (no scorers/query_*/auto_scores/scores.json found; skipping)")
        print("  Hint: run `python run_all.py --models ...` first to populate.")
        return 0, 0, 0

    passed, failed = 0, 0
    for p in paths:
        qid = p.parent.parent.name  # "query_NN"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  ✗ {qid}: JSONDecodeError — {e}")
            failed += 1
            continue
        rates, _ = extract_total_rates(data)
        if not rates:
            print(
                f"  ✗ {qid}: extract_total_rates returned empty dict — "
                "schema drift. Writer should emit `results` as "
                "dict[model: entry] with `total_score` + `max_score` "
                "(per-entry `total_rate` preferred)."
            )
            failed += 1
        else:
            print(f"  ✓ {qid}: {len(rates)} model rate(s) extracted")
            passed += 1
    print(f"  → {passed}/{passed + failed} passed ({len(paths)} file(s) scanned)")
    return passed, failed, len(paths)


def main() -> int:
    syn_pass, syn_fail = run_synthetic_tests()
    real_pass, real_fail, real_n = run_real_output_tests()
    total_pass = syn_pass + real_pass
    total_fail = syn_fail + real_fail
    print()
    print("=" * 64)
    if total_fail:
        print(
            f"FAIL: {total_fail} test(s) failed, {total_pass} passed "
            f"(synthetic: {syn_pass}/{len(SYNTHETIC_CASES)}, "
            f"real outputs: {real_pass}/{real_n})."
        )
        return 1
    print(
        f"PASS: all {total_pass} test(s) passed "
        f"(synthetic: {syn_pass}/{len(SYNTHETIC_CASES)}, "
        f"real outputs: {real_pass}/{real_n})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
