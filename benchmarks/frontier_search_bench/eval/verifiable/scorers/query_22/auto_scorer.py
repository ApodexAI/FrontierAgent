"""Query 22 — Drug fast-track first approval auto-scorer.

Reference answer (locked 2026-04-29):
  Gazyva (obinutuzumab) by Genentech, Inc.
  BLA 125486, approved by FDA on 2013-11-01 — first drug approved through
  the FDA Breakthrough Therapy Designation program.

Scoring rule (4 independent dimensions, ±1 each, max 4, min -4):
  A drug identity   : brand "gazyva" or generic "obinutuzumab" / "奥妥珠单抗"
  B company         : "genentech" / "基因泰克"  → +1
                      "roche" / "罗氏" (without genentech) → -1
                      anything else / empty → -1
  C approval number : digit string contains "125486" → +1, else -1
  D approval date   : ISO-equivalent 2013-11-01 → +1
                      year=2013 + month=11 only → 0 (partial credit, no penalty)
                      else / empty → -1

Pipeline (Stage 1 extract → Stage 2 align →
Stage 3 null_review/resolutions → Stage 4 score).
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
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.alignment import (  # noqa: E402
    align_claims,
    apply_null_resolutions,
    export_null_claims_for_review,
)
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "1.0",
    "locked_at": "2026-04-29",
    "brand_keys": ["gazyva"],
    "generic_keys": ["obinutuzumab", "奥妥珠单抗"],
    "company_correct_keys": ["genentech", "基因泰克"],
    "company_wrong_keys": ["roche", "罗氏"],
    "approval_number_digits": "125486",
    "approval_year": 2013,
    "approval_month": 11,
    "approval_day": 1,
    "fast_track_program": "Breakthrough Therapy Designation (BTD)",
    "country_agency": "United States / FDA",
}

# Single baseline entry consumed by alignment.align_claims().
BASELINES = [
    {
        "id": "GAZYVA",
        "description": (
            "Gazyva (obinutuzumab) — Genentech, Inc. — BLA 125486 — "
            "FDA approval 2013-11-01 — first drug approved under the FDA "
            "Breakthrough Therapy Designation (BTD) program. [基准: ✅]"
        ),
        "match_fields": {
            "brand_name": "Gazyva",
            "generic_name": "obinutuzumab",
            "company": "Genentech",
            "approval_number": "125486",
            "approval_date": "2013-11-01",
        },
        "kw": [
            "gazyva",
            "obinutuzumab",
            "奥妥珠单抗",
            "125486",
            "genentech",
            "基因泰克",
            "breakthrough therapy",
            "突破性疗法",
        ],
        "judgment": "✅",
        "score": 1.0,
    }
]


def build_baselines() -> list[dict]:
    return BASELINES


# ═══════════════════════════════════════════════════════════════════════════
# Field normalization helpers
# ═══════════════════════════════════════════════════════════════════════════


def _norm(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _digits(s) -> str:
    """Extract all digits as a single string."""
    return re.sub(r"[^0-9]", "", str(s) if s is not None else "")


_DATE_PATTERNS = [
    # ISO-like: 2013-11-01, 2013/11/01, 2013.11.01
    re.compile(r"\b(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})\b"),
    # CN: 2013年11月1日 or 2013年11月01日
    re.compile(r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"),
    # CN year+month only: 2013年11月
    re.compile(r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月(?!\s*\d)"),
    # US: November 1, 2013 / Nov 1 2013 / Nov. 1, 2013
    re.compile(
        r"\b(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
    # US year-month: November 2013 / Nov 2013
    re.compile(
        r"\b(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\.?\s+(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
    # Numeric MDY: 11/01/2013, 11-01-2013
    re.compile(r"\b(?P<m>\d{1,2})[/.\-](?P<d>\d{1,2})[/.\-](?P<y>\d{4})\b"),
    # Year + month numeric only "2013-11" or "2013/11"
    re.compile(r"\b(?P<y>\d{4})[-/](?P<m>\d{1,2})\b(?!\s*[-/]\s*\d)"),
]

_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


_HYPHEN_LIKE = "‐‑‒–—―﹘﹣－"


def parse_date(s) -> tuple[int | None, int | None, int | None]:
    """Return (year, month, day) — any may be None if not parsed."""
    if not s:
        return (None, None, None)
    text = str(s).strip()
    # Normalize unicode hyphens (e.g. U+2011 non-breaking hyphen used by some
    # models) to ASCII '-' so the regex patterns match.
    for ch in _HYPHEN_LIKE:
        text = text.replace(ch, "-")
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        gd = m.groupdict()
        y = int(gd["y"]) if gd.get("y") else None
        if "mon" in gd and gd.get("mon"):
            mo = _MONTH_MAP.get(gd["mon"].lower().rstrip("."))
        elif gd.get("m"):
            mo = int(gd["m"])
        else:
            mo = None
        day = int(gd["d"]) if gd.get("d") else None
        return (y, mo, day)
    # Fallback: standalone 4-digit year, no month
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return (int(m.group()), None, None)
    return (None, None, None)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════


def score_A_drug(claim: dict, canonical_id: str | None) -> tuple[int, str]:
    """A — drug identity. +1 if claim's brand or generic matches Gazyva /
    obinutuzumab / 奥妥珠单抗 (also accept canonical_id == GAZYVA from align)."""
    brand = _norm(claim.get("brand_name"))
    generic = _norm(claim.get("generic_name"))
    if any(k in brand for k in BENCHMARK["brand_keys"]):
        return (1, f"brand_name='{brand}' contains 'gazyva'")
    if any(k in generic for k in BENCHMARK["generic_keys"]):
        return (
            1,
            f"generic_name='{generic}' contains '{next(k for k in BENCHMARK['generic_keys'] if k in generic)}'",
        )
    # canonical_id from align is a strong signal
    if canonical_id == "GAZYVA":
        return (1, "alignment canonical_id=GAZYVA (drug identity confirmed)")
    return (-1, f"drug not Gazyva (brand='{brand}', generic='{generic}')")


def score_B_company(claim: dict) -> tuple[int, str]:
    """B — company. +1 for Genentech (with or without Roche); -1 for Roche
    only (no Genentech), or anything else / empty."""
    c = _norm(claim.get("company"))
    if not c:
        return (-1, "company empty")
    has_correct = any(k in c for k in BENCHMARK["company_correct_keys"])
    has_wrong = any(k in c for k in BENCHMARK["company_wrong_keys"])
    if has_correct:
        return (1, f"company='{c}' contains genentech/基因泰克")
    if has_wrong:
        return (
            -1,
            f"company='{c}' is Roche/罗氏 (parent group, not the BLA applicant)",
        )
    return (-1, f"company='{c}' not Genentech")


def score_C_approval_number(claim: dict) -> tuple[int, str]:
    """C — approval number. +1 if digit string contains '125486'."""
    n = claim.get("approval_number") or ""
    digits = _digits(n)
    if BENCHMARK["approval_number_digits"] in digits:
        return (1, f"approval_number='{n}' digits='{digits}' contains 125486")
    return (-1, f"approval_number='{n}' digits='{digits}' missing 125486")


def score_D_date(claim: dict) -> tuple[int, str]:
    """D — approval date.
    +1 if year=2013, month=11, day=1
     0 if year=2013 and month=11 but no day (partial)
    -1 otherwise (year wrong, month wrong, missing year/month, etc.)
    """
    raw = claim.get("approval_date")
    y, mo, day = parse_date(raw)
    target_y = BENCHMARK["approval_year"]
    target_m = BENCHMARK["approval_month"]
    target_d = BENCHMARK["approval_day"]
    if y == target_y and mo == target_m and day == target_d:
        return (1, f"date='{raw}' parsed=({y},{mo},{day}) matches 2013-11-01")
    if y == target_y and mo == target_m and day is None:
        return (0, f"date='{raw}' parsed year+month only (2013-11), partial credit 0")
    return (-1, f"date='{raw}' parsed=({y},{mo},{day}) does not match 2013-11-01")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DimScore:
    score: int
    reason: str


@dataclass
class ModelScore:
    model: str
    A: DimScore
    B: DimScore
    C: DimScore
    D: DimScore
    total: int
    canonical_id: str | None
    alignment_confidence: str
    raw_claim: dict
    notes: str = ""


def _score_one_model(model_name: str, aligned_claims: list[dict]) -> ModelScore:
    """Each model produces a single composite claim. Pick the (only) aligned
    claim; if absent (extraction empty / not_mentioned), all dims are -1
    except D which is -1 (no date given)."""
    if not aligned_claims:
        empty: dict = {}
        a = score_A_drug(empty, None)
        b = score_B_company(empty)
        c = score_C_approval_number(empty)
        d = score_D_date(empty)
        return ModelScore(
            model=model_name,
            A=DimScore(*a),
            B=DimScore(*b),
            C=DimScore(*c),
            D=DimScore(*d),
            total=a[0] + b[0] + c[0] + d[0],
            canonical_id=None,
            alignment_confidence="empty",
            raw_claim={},
            notes="no aligned claim (extraction empty / not_mentioned)",
        )

    ac = aligned_claims[0]
    raw = ac.get("raw", {}) or {}
    cid = ac.get("canonical_id")
    conf = ac.get("alignment_confidence", "medium")

    # If the model lists alternative_candidates and the primary is wrong but
    # one candidate is Gazyva, we still score the primary (per the prompt:
    # the model's chosen answer is what counts).
    a = score_A_drug(raw, cid)
    b = score_B_company(raw)
    c = score_C_approval_number(raw)
    d = score_D_date(raw)
    return ModelScore(
        model=model_name,
        A=DimScore(*a),
        B=DimScore(*b),
        C=DimScore(*c),
        D=DimScore(*d),
        total=a[0] + b[0] + c[0] + d[0],
        canonical_id=cid,
        alignment_confidence=conf,
        raw_claim=raw,
    )


def score_all(aligned_by_model: dict[str, list[dict]]) -> list[ModelScore]:
    return [_score_one_model(m, aligned_by_model.get(m, [])) for m in aligned_by_model]


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(out_dir: Path) -> dict[str, list[dict]]:
    """Read each model's extraction.json and return a list-of-claims dict.
    Q22 uses a single composite value, so we wrap it in a 1-element list."""
    out: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        ext = d / "extraction.json"
        if not ext.exists():
            continue
        payload = json.loads(ext.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = []
            continue
        ent = entities[0]
        canonical_value = (ent.get("canonical") or {}).get("value")
        # canonical_value is a dict (composite). If empty/None or all fields
        # are null, treat as empty list.
        if canonical_value is None or canonical_value == {}:
            out[d.name] = []
            continue
        if isinstance(canonical_value, list):
            # Defensive: if some extractor returned a list anyway, take first.
            out[d.name] = canonical_value[:1] if canonical_value else []
            continue
        if not isinstance(canonical_value, dict):
            out[d.name] = []
            continue
        # Skip entirely empty composite (all primary fields null + no
        # alternatives). Otherwise keep it as a single claim for alignment.
        primary_keys = (
            "brand_name",
            "generic_name",
            "company",
            "approval_number",
            "approval_date",
        )
        if all(
            canonical_value.get(k) in (None, "", "null") for k in primary_keys
        ) and not canonical_value.get("alternative_candidates"):
            out[d.name] = []
            continue
        out[d.name] = [canonical_value]
    return out


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        aligned[d.name] = json.loads(af.read_text(encoding="utf-8")).get(
            "aligned_claims", []
        )
    return aligned


def _dim_to_dict(ds: DimScore) -> dict:
    return {"score": ds.score, "reason": ds.reason}


def write_outputs(out_dir: Path, results: list[ModelScore]) -> None:
    # Per-model score.json
    for r in results:
        (out_dir / r.model).mkdir(parents=True, exist_ok=True)
        (out_dir / r.model / "score.json").write_text(
            json.dumps(
                {
                    "query_id": QUERY_ID,
                    "model": r.model,
                    "total_score": r.total,
                    "max_score": 4,
                    "min_score": -4,
                    "dimensions": {
                        "A_drug": _dim_to_dict(r.A),
                        "B_company": _dim_to_dict(r.B),
                        "C_approval_number": _dim_to_dict(r.C),
                        "D_approval_date": _dim_to_dict(r.D),
                    },
                    "alignment_canonical_id": r.canonical_id,
                    "alignment_confidence": r.alignment_confidence,
                    "raw_claim": r.raw_claim,
                    "notes": r.notes,
                    "benchmark_version": BENCHMARK["version"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Top-level scores.json
    ranked = sorted(results, key=lambda r: (-r.total, r.model))
    (out_dir / "scores.json").write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "max_score": 4,
                "min_score": -4,
                "benchmark_version": BENCHMARK["version"],
                "scoring_rule": (
                    "A drug ±1, B company ±1, C approval# ±1, "
                    "D date {-1, 0, +1} — independent dimensions"
                ),
                "results": {
                    r.model: {
                        "total_score": r.total,
                        "A_drug": r.A.score,
                        "B_company": r.B.score,
                        "C_approval_number": r.C.score,
                        "D_approval_date": r.D.score,
                        "alignment_canonical_id": r.canonical_id,
                        "alignment_confidence": r.alignment_confidence,
                        "claim": r.raw_claim,
                    }
                    for r in results
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": r.model,
                        "score": r.total,
                        "A": r.A.score,
                        "B": r.B.score,
                        "C": r.C.score,
                        "D": r.D.score,
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ranking_report.md
    lines = [
        f"# Query {QUERY_ID} Ranking Report",
        "",
        f"> Benchmark v{BENCHMARK['version']} (locked {BENCHMARK['locked_at']})  ",
        "> Scoring: A drug ±1 · B company ±1 · C approval# ±1 · D date {-1, 0, +1}  ",
        "> Reference answer: **Gazyva (obinutuzumab) / Genentech / BLA 125486 / 2013-11-01**",
        "",
        "## 排名",
        "",
        "| Rank | Model | Total | A drug | B company | C BLA# | D date | Alignment |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.model} | {r.total:+d}/4 "
            f"| {r.A.score:+d} | {r.B.score:+d} | {r.C.score:+d} | {r.D.score:+d} "
            f"| {r.canonical_id or '—'} ({r.alignment_confidence}) |"
        )
    lines.append("")
    lines.append("## 各模型主张")
    lines.append("")
    lines.append("| Model | Brand | Generic | Company | Approval# | Date |")
    lines.append("|---|---|---|---|---|---|")
    for r in ranked:
        rc = r.raw_claim or {}

        def _g(k, _rc=rc):
            v = _rc.get(k)
            return str(v)[:40] if v else "—"

        lines.append(
            f"| {r.model} | {_g('brand_name')} | {_g('generic_name')} "
            f"| {_g('company')} | {_g('approval_number')} | {_g('approval_date')} |"
        )
    lines.append("")
    lines.append("## 维度判定理由")
    lines.append("")
    for r in ranked:
        lines.append(f"### {r.model}  (total={r.total:+d})")
        lines.append(f"- A (drug): **{r.A.score:+d}** — {r.A.reason}")
        lines.append(f"- B (company): **{r.B.score:+d}** — {r.B.reason}")
        lines.append(f"- C (approval#): **{r.C.score:+d}** — {r.C.reason}")
        lines.append(f"- D (date): **{r.D.score:+d}** — {r.D.reason}")
        lines.append("")
    (out_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 — extraction
    if not args.skip_extract:
        run_pipeline(
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

    # Stage 2 — alignment (single Gazyva baseline)
    baselines = build_baselines()
    if args.skip_align:
        aligned = _load_alignment(out_dir)
    else:
        client = get_client()
        overrides: dict = {}
        if args.aligner_models:
            overrides["aligner_models"] = args.aligner_models
        if args.judge_model:
            overrides["judge_model"] = args.judge_model
        if args.concurrency:
            overrides["concurrency"] = args.concurrency
        raw_claims = load_raw_claims(out_dir)
        aligned = align_claims(
            client,
            claims_by_model=raw_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # Stage 3a — export null claims for verification
    null_items = export_null_claims_for_review(
        aligned,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(
        f"\n[*] Exported {len(null_items)} null/needs_review claims → null_review.json"
    )

    # Stage 3b — apply null_resolutions if a verifier produced one.
    # NOTE: For Q22 we DO NOT use baseline_add to expand — the answer is
    # closed (1 drug). resolutions can still mark hallucination for unaligned
    # nulls, but per A/B/C/D independent scoring, the score is computed from
    # the raw claim regardless of canonical_id; canonical_id mainly helps
    # confirm A.
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        apply_null_resolutions(aligned, resolutions_path, dims_ref=None)
    else:
        print(
            "[*] No null_resolutions.json (will rely on independent A/B/C/D scoring)."
        )

    # Stage 4 — score
    results = score_all(aligned)
    write_outputs(out_dir, results)

    print("\n" + "─" * 64)
    print(f"Query {QUERY_ID} scoring done.")
    for r in sorted(results, key=lambda x: (-x.total, x.model)):
        mark = "✅" if r.total >= 3 else ("🟡" if r.total > 0 else "❌")
        print(
            f"  {mark} {r.model:28s} {r.total:+d}/4  "
            f"A{r.A.score:+d} B{r.B.score:+d} C{r.C.score:+d} D{r.D.score:+d}"
        )


if __name__ == "__main__":
    main()
