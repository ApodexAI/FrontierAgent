"""Query 34 — Peninsula hotels with To Summer (观夏) within 1km — auto-scorer.

Reference answer (locked 2026-05-08):
  Only 2 of the 12 worldwide Peninsula hotels have a To Summer store
  within 1km:
    - **上海半岛 (The Peninsula Shanghai, 中山东一路 32 号)**
      ↔ 观夏外滩源店 (圆明园路 149 号), ≈ 150-250 m
    - **北京半岛 / 王府半岛 (The Peninsula Beijing, 金鱼胡同 8 号)**
      ↔ 观夏书阁 WF Central 店, ≈ 380 m
  All 10 other Peninsulas (Hong Kong / Tokyo / Manila / Bangkok /
  Beverly Hills / Chicago / Paris / New York / Istanbul / London) are
  not within 1km of any To Summer store.

Scoring rule (max 2, no theoretical lower bound):
  +1 per distinct whitelist hotel correctly identified (Shanghai / Beijing)
  -1 per extra non-whitelist hotel the model lists as qualifying
  Duplicates (same whitelist hotel listed twice) are de-duped and NOT
  re-credited; they also do not get a penalty.

Matching is fully deterministic — alias-based city + Peninsula detection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from extract import (  # noqa: E402
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark / whitelist
# ═══════════════════════════════════════════════════════════════════════════

CORRECT_HOTEL_SCORE = 1
WRONG_HOTEL_PENALTY = -1

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-05-08",
    "whitelist": {
        "shanghai": {
            "canonical_name": "The Peninsula Shanghai (上海半岛酒店)",
            "city_keys": ["上海", "shanghai", "沪"],
            "hotel_name_keys": [
                "上海半岛",
                "上海半岛酒店",
                "the peninsula shanghai",
                "peninsula shanghai",
                "shanghai peninsula",
            ],
            "address_keys": [
                "中山东一路32号",
                "中山东一路 32号",
                "中山东一路32",
                "zhongshan east 1",
                "the bund 32",
            ],
            "expected_to_summer_store": "观夏外滩源店 (圆明园路 149 号)",
            "expected_distance": "约 150-250 米",
        },
        "beijing": {
            "canonical_name": "The Peninsula Beijing (北京王府半岛酒店)",
            "city_keys": ["北京", "beijing", "京"],
            "hotel_name_keys": [
                "北京半岛",
                "北京半岛酒店",
                "王府半岛",
                "北京王府半岛",
                "the peninsula beijing",
                "peninsula beijing",
                "beijing peninsula",
            ],
            "address_keys": [
                "金鱼胡同8号",
                "金鱼胡同 8号",
                "金鱼胡同8",
                "goldfish lane 8",
                "8 goldfish lane",
                "8 jinyu hutong",
            ],
            "expected_to_summer_store": "观夏书阁 WF Central 店",
            "expected_distance": "约 380 米",
        },
    },
    # Other 10 Peninsula hotels — recognised as "Peninsula but not in
    # whitelist" so we can give an informative penalty reason.
    "other_peninsulas": {
        "hong_kong": ["香港", "hong kong", "hongkong", "尖沙咀"],
        "tokyo": ["东京", "tokyo", "东京半岛", "the peninsula tokyo"],
        "manila": ["马尼拉", "manila", "the peninsula manila"],
        "bangkok": ["曼谷", "bangkok", "the peninsula bangkok"],
        "beverly_hills": [
            "比佛利",
            "beverly hills",
            "beverly",
            "the peninsula beverly hills",
        ],
        "chicago": ["芝加哥", "chicago", "the peninsula chicago"],
        "paris": ["巴黎", "paris", "the peninsula paris"],
        "new_york": ["纽约", "new york", "newyork", "the peninsula new york"],
        "istanbul": ["伊斯坦布尔", "istanbul", "the peninsula istanbul"],
        "london": ["伦敦", "london", "the peninsula london"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════


def _norm(value) -> str:
    if value is None:
        return ""
    v = unicodedata.normalize("NFKC", str(value)).lower().strip()
    v = v.replace("–", "-").replace("—", "-").replace("‑", "-")
    v = re.sub(r"\s+", " ", v)
    return v


def _has_any(text: str, keys: list[str]) -> str | None:
    for k in keys:
        kn = _norm(k)
        if kn and kn in text:
            return k
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Per-item classification
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ItemVerdict:
    label: str  # "correct" | "wrong_other_peninsula" | "wrong_unknown"
    matched_key: str | None  # whitelist key (shanghai/beijing) if correct
    other_peninsula_id: str | None  # which non-whitelist Peninsula was matched
    reason: str


def _classify_item(item: dict) -> ItemVerdict:
    city_raw = item.get("city") or ""
    hotel_raw = item.get("hotel_name") or ""
    addr_raw = item.get("hotel_address") or ""
    blob = _norm(" ".join([city_raw, hotel_raw, addr_raw]))

    if not blob.strip():
        return ItemVerdict(
            "wrong_unknown",
            None,
            None,
            "item has no city / hotel_name / hotel_address",
        )

    # Stage 1 — whitelist match.
    for key, info in BENCHMARK["whitelist"].items():
        city_hit = _has_any(blob, info["city_keys"])
        name_hit = _has_any(blob, info["hotel_name_keys"])
        addr_hit = _has_any(blob, info["address_keys"])
        # 任一显式 hotel name 或 address 命中 → 必然是该项；
        # 仅城市命中也算（题目语境下"上海半岛"是唯一项）。
        if name_hit or addr_hit or city_hit:
            hit = name_hit or addr_hit or city_hit
            return ItemVerdict(
                "correct",
                key,
                None,
                f"whitelist hit '{hit}' → {info['canonical_name']}",
            )

    # Stage 2 — other Peninsula city detection.
    for op_id, aliases in BENCHMARK["other_peninsulas"].items():
        hit = _has_any(blob, aliases)
        if hit:
            return ItemVerdict(
                "wrong_other_peninsula",
                None,
                op_id,
                f"non-whitelist Peninsula '{hit}' (id={op_id})",
            )

    return ItemVerdict(
        "wrong_unknown",
        None,
        None,
        f"could not classify ('{blob[:60]}…' — not in any whitelist / known Peninsula)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ItemResult:
    item_index: int
    raw_item: dict
    verdict: ItemVerdict
    counted_score: int  # +1, -1, or 0 (duplicate)


@dataclass
class ModelScore:
    model: str
    results: list[ItemResult] = field(default_factory=list)
    matched_keys: list[str] = field(default_factory=list)
    duplicate_matches: list[ItemResult] = field(default_factory=list)
    wrong_other_peninsulas: list[ItemResult] = field(default_factory=list)
    wrong_unknown: list[ItemResult] = field(default_factory=list)
    excluded_hotels: list[dict] = field(default_factory=list)
    positive_score: int = 0
    penalty_score: int = 0
    total_score: int = 0
    resolution: str = "unknown"


def _score_one_model(model_name: str, payload: dict) -> ModelScore:
    entities = payload.get("entities", [])
    if not entities:
        return ModelScore(model=model_name, resolution="empty")
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value") or {}
    if not isinstance(canonical, dict):
        canonical = {}

    items = canonical.get("items")
    if not isinstance(items, list):
        items = []
    excluded = canonical.get("excluded_hotels") or []
    if not isinstance(excluded, list):
        excluded = []

    matched_keys_seen: set[str] = set()
    results: list[ItemResult] = []
    duplicates: list[ItemResult] = []
    wrong_others: list[ItemResult] = []
    wrong_unknowns: list[ItemResult] = []

    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        verdict = _classify_item(raw)
        counted = 0

        if verdict.label == "correct":
            if verdict.matched_key in matched_keys_seen:
                counted = 0  # duplicate — neither credit nor penalty
                ir = ItemResult(idx, raw, verdict, counted)
                duplicates.append(ir)
            else:
                matched_keys_seen.add(verdict.matched_key)  # type: ignore[arg-type]
                counted = CORRECT_HOTEL_SCORE
                ir = ItemResult(idx, raw, verdict, counted)
        elif verdict.label == "wrong_other_peninsula":
            counted = WRONG_HOTEL_PENALTY
            ir = ItemResult(idx, raw, verdict, counted)
            wrong_others.append(ir)
        else:  # wrong_unknown
            counted = WRONG_HOTEL_PENALTY
            ir = ItemResult(idx, raw, verdict, counted)
            wrong_unknowns.append(ir)

        results.append(ir)

    positive = sum(r.counted_score for r in results if r.counted_score > 0)
    penalty = sum(r.counted_score for r in results if r.counted_score < 0)
    total = positive + penalty

    return ModelScore(
        model=model_name,
        results=results,
        matched_keys=sorted(matched_keys_seen),
        duplicate_matches=duplicates,
        wrong_other_peninsulas=wrong_others,
        wrong_unknown=wrong_unknowns,
        excluded_hotels=[e for e in excluded if isinstance(e, dict)],
        positive_score=positive,
        penalty_score=penalty,
        total_score=total,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _item_to_dict(ir: ItemResult) -> dict:
    return {
        "item_index": ir.item_index,
        "raw_item": ir.raw_item,
        "verdict": {
            "label": ir.verdict.label,
            "matched_key": ir.verdict.matched_key,
            "other_peninsula_id": ir.verdict.other_peninsula_id,
            "reason": ir.verdict.reason,
        },
        "counted_score": ir.counted_score,
    }


def _write_model_score(out_dir: Path, r: ModelScore) -> None:
    path = out_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # max_score = 2 (size of whitelist) — used for cross-task normalization;
    # clip total_rate to [0, 1].
    max_for_norm = len(BENCHMARK["whitelist"])
    total_rate = max(0.0, min(r.total_score / max_for_norm, 1.0))
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.total_score,
                "max_score": max_for_norm,
                "total_rate": total_rate,
                "positive_score": r.positive_score,
                "penalty_score": r.penalty_score,
                "scoring_rule": {
                    "correct_hotel_score": CORRECT_HOTEL_SCORE,
                    "wrong_hotel_penalty": WRONG_HOTEL_PENALTY,
                    "matching_rule": (
                        "命中 whitelist (上海半岛 / 北京半岛) 计 +1； "
                        "非 whitelist 半岛酒店 / 无法识别的酒店 计 -1； "
                        "同一 whitelist 重复列出 → 不再加分也不扣分。"
                    ),
                },
                "matched_whitelist_keys": r.matched_keys,
                "items": [_item_to_dict(ir) for ir in r.results],
                "duplicate_matches": [_item_to_dict(ir) for ir in r.duplicate_matches],
                "wrong_other_peninsulas": [
                    _item_to_dict(ir) for ir in r.wrong_other_peninsulas
                ],
                "wrong_unknown": [_item_to_dict(ir) for ir in r.wrong_unknown],
                "excluded_hotels_from_extractor": r.excluded_hotels,
                "benchmark_version": BENCHMARK["version"],
                "extraction_resolution": r.resolution,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_scores_json(out_dir: Path, results: list[ModelScore]) -> None:
    ranked = sorted(results, key=lambda r: (-r.total_score, r.model))
    max_for_norm = len(BENCHMARK["whitelist"])
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": max_for_norm,
                "min_score": None,
                "scoring_rule": (
                    f"whitelist hit → +{CORRECT_HOTEL_SCORE}; "
                    f"non-whitelist hotel → {WRONG_HOTEL_PENALTY}; "
                    "duplicate whitelist → 0."
                ),
                "benchmark_version": BENCHMARK["version"],
                "whitelist": list(BENCHMARK["whitelist"].keys()),
                "results": {
                    r.model: {
                        "total_score": r.total_score,
                        "positive_score": r.positive_score,
                        "penalty_score": r.penalty_score,
                        "total_rate": max(
                            0.0, min(r.total_score / max_for_norm, 1.0)
                        ),
                        "matched_whitelist_keys": r.matched_keys,
                        "wrong_other_peninsulas": len(r.wrong_other_peninsulas),
                        "wrong_unknown": len(r.wrong_unknown),
                        "extraction_resolution": r.resolution,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.total_score,
                        "matched": r.matched_keys,
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(out_dir: Path, results: list[ModelScore]) -> None:
    ranked = sorted(results, key=lambda r: (-r.total_score, r.model))
    lines: list[str] = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        f"> Benchmark v{BENCHMARK['version']} (locked {BENCHMARK['locked_at']})  ",
        f"> Scoring: whitelist hit → +{CORRECT_HOTEL_SCORE}; non-whitelist hotel → "
        f"{WRONG_HOTEL_PENALTY}; duplicate whitelist → 0.  ",
        "> Whitelist (2 hotels): **上海半岛 (中山东一路 32 号)** · "
        "**北京/王府半岛 (金鱼胡同 8 号)**.",
        "",
        "| Rank | Model | Total | Shanghai | Beijing | Wrong (other PN) | Wrong (unknown) |",
        "|---:|---|---:|:---:|:---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        sh = "✓" if "shanghai" in r.matched_keys else "✗"
        bj = "✓" if "beijing" in r.matched_keys else "✗"
        lines.append(
            f"| {i} | {r.model} | {r.total_score:+d} | {sh} | {bj} "
            f"| {len(r.wrong_other_peninsulas)} | {len(r.wrong_unknown)} |"
        )
    lines.append("")
    lines.append("## 每模型逐项明细")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model} (total={r.total_score:+d})")
        if not r.results:
            lines.append("- （未列出任何半岛酒店）")
        for ir in r.results:
            raw = ir.raw_item or {}
            mark = (
                "✅"
                if ir.verdict.label == "correct" and ir.counted_score > 0
                else ("🔁" if ir.verdict.label == "correct" else "❌")
            )
            lines.append(
                f"- {mark} `{raw.get('hotel_name') or raw.get('city') or '?'}` "
                f"({ir.counted_score:+d}) — {ir.verdict.reason}"
            )
        if r.excluded_hotels:
            ex = ", ".join(
                str(e.get("hotel_name") or e.get("city") or "?")[:30]
                for e in r.excluded_hotels[:6]
            )
            lines.append(f"- (extractor 记录主动排除的酒店：{ex})")
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        all_results: dict[str, dict] = {}
        for spec in args.models:
            name = spec.split("=", 1)[0].strip()
            ep = out_dir / name / "extraction.json"
            if not ep.exists():
                sys.exit(f"[ERROR] missing extraction: {ep}")
            all_results[name] = json.loads(ep.read_text(encoding="utf-8"))
    else:
        all_results = run_pipeline(
            query_id=QUERY_ID,
            query_text=QUERY_TEXT,
            entities=ENTITIES,
            prompt_hints=PROMPT_HINTS,
            schema=VALUE_SCHEMA,
            models_input=args.models,
            output_dir=out_dir,
            primary=args.primary_model,
            secondary=args.secondary_model,
            parallel=args.parallel_models,
            analyzer=args.analyzer_model,
            concurrency=args.concurrency,
        )

    results: list[ModelScore] = []
    for model_name, payload in all_results.items():
        r = _score_one_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    _write_scores_json(out_dir, results)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total_score, x.model)):
        mark = "✅" if r.total_score == 2 else ("🟡" if r.total_score > 0 else "❌")
        print(
            f"  {mark} {r.model:28s} total={r.total_score:+d}  "
            f"matched={r.matched_keys}  "
            f"wrong_PN={len(r.wrong_other_peninsulas)}  "
            f"wrong_other={len(r.wrong_unknown)}"
        )


if __name__ == "__main__":
    main()
