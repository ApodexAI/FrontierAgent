"""
Query 35 — Auto scorer v2

Per-country composite score 0/1/2/3; 18 countries × 3 = 54 max.

  3 = store pair matches GT + distance within 5%
  2 = store pair matches GT + distance error 5-15%  OR
      pair is GT's known secondary pair
  1 = pair wrong but distance order-of-magnitude ok (error ≤30%)
  0 = pair wrong + distance off >30% / not answered / country-only

Cambodia (D18) special rule:
  3 if model says "only 1 store"
  0 if model gives a distance
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from extract import (  # noqa: E402
    COUNTRIES,
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth — per-country store-pair + distance
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: id → {pair_keywords: list[tuple], second_pair: optional, km, only_one, country}
# pair_keywords: a list of 2-tuples of keyword lists; matching means model's
# store_a/b together contain a keyword from each tuple.
GT: dict[str, dict] = {
    "D1": {
        "country": "中国大陆",
        "pairs": [[["喀什", "kashgar", "kashi"], ["延吉", "yanji"]]],
        "secondary": [
            [["牡丹江", "mudanjiang"], ["喀什", "kashgar"]],
            [["乌鲁木齐", "urumqi", "urumchi"], ["三亚", "sanya"]],
        ],
        "km": 4418,
    },
    "D2": {
        "country": "香港",
        "pairs": [[["荃湾", "tsuen wan", "tsuen-wan"], ["将军澳", "tseung kwan"]]],
        "secondary": [],
        "km": 16.6,
    },
    "D3": {
        "country": "澳门",
        "pairs": [[["威尼斯人", "venetian"], ["伦敦人", "londoner"]]],
        "secondary": [],
        "km": 0.9,
    },
    "D4": {
        "country": "台湾",
        "pairs": [[["台北", "taipei"], ["高雄", "kaohsiung"]]],
        "secondary": [],
        "km": 297,
    },
    "D5": {
        "country": "美国",
        "pairs": [[["flushing", "法拉盛"], ["cupertino", "库比蒂诺"]]],
        "secondary": [[["irvine"], ["new york", "flushing"]]],
        "km": 4126,
    },
    "D6": {
        "country": "加拿大",
        "pairs": [
            [["vancouver", "温哥华"], ["montreal", "montréal", "蒙特利尔", "蒙特婁"]]
        ],
        "secondary": [],
        "km": 3690,
    },
    "D7": {
        "country": "英国",
        "pairs": [[["london", "伦敦"], ["birmingham", "伯明翰"]]],
        "secondary": [[["london"], ["manchester", "曼彻斯特"]]],
        "km": 169,
    },
    "D8": {
        "country": "澳大利亚",
        "pairs": [[["melbourne", "墨尔本"], ["brisbane", "布里斯班"]]],
        "secondary": [[["melbourne"], ["sydney", "悉尼"]]],
        "km": 1370,
    },
    "D9": {
        "country": "日本",
        "pairs": [[["千叶", "chiba", "makuhari"], ["大阪", "osaka", "namba"]]],
        "secondary": [
            [["tokyo", "东京"], ["osaka", "大阪"]],
            [["fukuoka", "福冈"], ["tokyo", "东京"]],
        ],  # fukuoka outdated
        "km": 426,
    },
    "D10": {
        "country": "韩国",
        "pairs": [[["首尔", "seoul"], ["济州", "jeju"]]],
        "secondary": [[["seoul"], ["busan", "釜山"]]],
        "km": 454,
    },
    "D11": {
        "country": "新加坡",
        "pairs": [
            [["jurong", "裕廊"], ["century square", "tampines", "勿洛", "century"]]
        ],
        "secondary": [[["jurong"], ["downtown east", "pasir ris"]]],
        "km": 26.3,
    },
    "D12": {
        "country": "马来西亚",
        "pairs": [[["penang", "槟城", "gurney"], ["kuching", "古晋", "vivacity"]]],
        "secondary": [[["kuala lumpur", "吉隆坡"], ["johor", "新山"]]],
        "km": 1195,
    },
    "D13": {
        "country": "泰国",
        "pairs": [[["chiang mai", "清迈"], ["phuket", "普吉"]]],
        "secondary": [[["chiang mai"], ["bangkok", "曼谷"]]],
        "km": 1215,
    },
    "D14": {
        "country": "越南",
        "pairs": [[["hanoi", "河内"], ["ho chi minh", "胡志明", "saigon"]]],
        "secondary": [],
        "km": 1144,
    },
    "D15": {
        "country": "印度尼西亚",
        "pairs": [[["medan", "棉兰", "delipark"], ["surabaya", "泗水", "pakuwon"]]],
        "secondary": [[["jakarta", "雅加达"], ["surabaya", "泗水"]]],
        "km": 1973,
    },
    "D16": {
        "country": "菲律宾",
        "pairs": [[["moa", "mall of asia", "sm moa"], ["robinsons", "galleria"]]],
        "secondary": [],
        "km": 10,
    },
    "D17": {
        "country": "阿联酋",
        "pairs": [[["dubai mall"], ["festival city", "dubai festival"]]],
        "secondary": [],
        "km": 8,
    },
    "D18": {
        "country": "柬埔寨",
        "pairs": [],
        "secondary": [],
        "km": None,
        "only_one": True,
    },
}

BENCHMARK = {
    "version": "2.0",
    "locked_at": "2026-04-23",
    "gt": GT,
}


# ═══════════════════════════════════════════════════════════════════════════
# Matching helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s一-鿿]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _contains_any(hay: str, needles: list[str]) -> bool:
    return any(_norm(n) in hay for n in needles)


def _pair_matches(
    store_a: str | None, store_b: str | None, pair_kw: list[list[str]]
) -> bool:
    """A GT pair is [kw_list_1, kw_list_2]. Model matches if {a,b} covers both sides."""
    if len(pair_kw) != 2:
        return False
    side1, side2 = pair_kw
    a = _norm(store_a)
    b = _norm(store_b)
    # try both assignments a→side1, b→side2, and vice versa
    if (_contains_any(a, side1) and _contains_any(b, side2)) or (
        _contains_any(a, side2) and _contains_any(b, side1)
    ):
        return True
    # also accept a single joined string on either side (e.g. "荃湾->将军澳")
    joined = f"{a} {b}".strip()
    return bool(_contains_any(joined, side1) and _contains_any(joined, side2))


def _any_pair_match(
    store_a: str | None, store_b: str | None, pair_list: list[list[list[str]]]
) -> bool:
    return any(_pair_matches(store_a, store_b, p) for p in pair_list)


# ═══════════════════════════════════════════════════════════════════════════
# Distance scoring
# ═══════════════════════════════════════════════════════════════════════════


def _distance_error(claimed: float | None, gt_km: float | None) -> float | None:
    if claimed is None or gt_km is None or gt_km == 0:
        return None
    return abs(claimed - gt_km) / gt_km


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


def _score_country(cid: str, canonical: dict | None) -> tuple[int, dict]:
    gt = GT[cid]
    canonical = canonical if isinstance(canonical, dict) else {}
    store_a = canonical.get("store_a")
    store_b = canonical.get("store_b")
    distance_km = canonical.get("distance_km")
    only_one = canonical.get("only_one_store")

    # numeric distance
    try:
        distance_km = float(distance_km) if distance_km is not None else None
    except (TypeError, ValueError):
        m = re.search(r"\d+(?:\.\d+)?", str(distance_km or ""))
        distance_km = float(m.group()) if m else None

    # Cambodia special rule
    if gt.get("only_one"):
        if only_one is True or (
            store_b is None
            and distance_km is None
            and (
                store_a is not None
                or "金边" in _norm(canonical.get("supporting_span", ""))
            )
        ):
            return 3, {"rule": "only_one_ok"}
        if distance_km is not None:
            return 0, {"rule": "cambodia_gave_distance"}
        if store_b is None and store_a is None and only_one is None:
            return 0, {"rule": "cambodia_not_answered"}
        return 0, {"rule": "cambodia_unclear"}

    # Not answered
    if store_a is None and store_b is None and distance_km is None:
        return 0, {"rule": "not_answered"}

    pair_ok = _any_pair_match(store_a, store_b, gt["pairs"])
    secondary_ok = _any_pair_match(store_a, store_b, gt.get("secondary", []))
    err = _distance_error(distance_km, gt["km"])

    if pair_ok:
        if err is not None and err <= 0.05:
            return 3, {"rule": "pair_ok_tight", "error": err}
        if err is not None and err <= 0.15:
            return 2, {"rule": "pair_ok_loose", "error": err}
        if err is None:
            return 2, {"rule": "pair_ok_no_distance"}
        # pair ok but large error
        return 1, {"rule": "pair_ok_big_error", "error": err}
    if secondary_ok:
        if err is not None and err <= 0.15:
            return 2, {"rule": "secondary_pair_close"}
        return 1, {"rule": "secondary_pair"}
    # pair wrong — check distance magnitude
    if err is not None and err <= 0.30:
        return 1, {"rule": "magnitude_only", "error": err}
    return 0, {"rule": "wrong_or_way_off", "error": err}


@dataclass
class ScoreResult:
    model: str
    country_scores: dict[str, int]
    country_detail: dict[str, dict]
    total: int
    max_total: int


def _score_model(model_name: str, payload: dict) -> ScoreResult:
    entities = payload.get("entities", [])
    by_id = {e["id"]: e for e in entities}
    c_scores: dict[str, int] = {}
    c_detail: dict[str, dict] = {}
    for cid, _ in COUNTRIES:
        ent = by_id.get(cid, {})
        canonical = (ent.get("canonical") or {}).get("value")
        s, d = _score_country(cid, canonical)
        c_scores[cid] = s
        d["claimed"] = canonical
        c_detail[cid] = d
    return ScoreResult(
        model=model_name,
        country_scores=c_scores,
        country_detail=c_detail,
        total=sum(c_scores.values()),
        max_total=len(COUNTRIES) * 3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def _write_model_score(output_dir: Path, r: ScoreResult) -> None:
    path = output_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.total,
                "max_score": r.max_total,
                "country_scores": r.country_scores,
                "country_detail": r.country_detail,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 35 Ranking Report (v2)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append("")
    header = (
        "| Rank | Model | Total | " + " | ".join(cid for cid, _ in COUNTRIES) + " |"
    )
    sep = "|---:|---|---:|" + "---:|" * len(COUNTRIES)
    lines.append(header)
    lines.append(sep)
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    for i, r in enumerate(ranked, start=1):
        cells = [str(r.country_scores[cid]) for cid, _ in COUNTRIES]
        lines.append(
            f"| {i} | {r.model} | {r.total}/{r.max_total} | " + " | ".join(cells) + " |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 35 auto-scorer v2")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
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

    results = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    print("\n" + "─" * 60)
    print("Query 35 scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        print(f"  {r.model}: {r.total}/{r.max_total}")


if __name__ == "__main__":
    main()
