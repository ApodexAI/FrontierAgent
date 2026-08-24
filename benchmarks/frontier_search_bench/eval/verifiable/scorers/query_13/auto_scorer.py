"""
Query 13 — Tech earnings ~5% drop → upstream/downstream max single-day drop
auto-scorer.

Cold-start mode: baseline starts EMPTY. Stage 2
alignment trivially produces null for every claim. Stage C web-verification
agent runs the 6-condition checklist on each claim and writes
null_resolutions.json. Resolutions feed back into DIMS so subsequent runs
benefit from the GT inferred this round.

Scoring rule (binary, per user spec 2026-04-30):
  ✅ baseline_add (Stage C 验证 6 条件全过) →  +1.0
  ❌ hallucination (Stage C 验证任一条件失败 / 事实编造) →  -1.0
  ⚪ null + unresolved (Stage C 搜索无果) →   0.0  (人工二次判别)
  alignment_confidence == "needs_review" →    0.0
  null + 未走 Stage C (即 null_resolutions 还没产出) →  0.0 (gracefully)

总分可正可负。无 ⚠️ 中间档（不同于 voting-style 评分模式）。

6 项核查条件（Stage C agent 必须逐项判定）：
  C1 财报日期 ∈ [2016-01-01, 2026-01-01]
  C2 母公司是科技股（GICS IT 或 META/GOOGL/AMZN/NFLX/TSLA 放宽口径）
  C3 母公司次一交易日复权跌幅 ∈ [-6%, -4%]（±0.3% 容差）
  C4 答案股票与母公司有公开披露的上下游/产业链关系
  C5 答案股票在 answer_drop_date 实际复权跌幅与模型声称一致（±0.3%）
  C6 answer_drop_date ∈ [earnings_date, earnings_date + 2 交易日]

Pipeline:
  1. Extraction — v2 pipeline (claude-sonnet-4 + gpt-5 + analyzer)
  2. Alignment — empty baseline → all null (trivial)
  3. Stage C — Web-search agent applies 6-condition checklist
  4. Scoring — binary +1/-1/0 lookup from canonical_id

Output:
  - auto_scores/{model}/extraction.json + alignment.json
  - auto_scores/null_review.json (Stage C 输入)
  - auto_scores/null_resolutions.json (Stage C 输出)
  - auto_scores/scores.json
  - auto_scores/ranking_report.md
  - auto_scores/unresolved_review.md (人工二次判别用，含 Stage C 完整搜索证据)
"""

from __future__ import annotations

import argparse
import json
import sys
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
    persist_new_baseline_entries,
)
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark — cold-start: empty DIMS. Stage C resolutions populate this list
# via apply_null_resolutions + persist_new_baseline_entries.
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-04-30"
SCORING_MODE = "Q13-binary-cold-start"  # ✅+1 / ❌-1 / unresolved 0

DIMS: list = [
    # Empty at cold start. Stage C agent's null_resolutions.json with
    # resolution=baseline_add appends entries here via
    # persist_new_baseline_entries().
    #
    # Tuple shape (consistent with open-set DIM convention):
    #   (id, description, judgment, score, [kw, ...])
    # judgment ∈ {"✅", "❌"}; score ∈ {+1.0, -1.0}
    # ⚠️ 中间档在 Q13 不使用——题目硬条件清晰，要么全过要么不过
]

DIM_MAP: dict = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
MAX_SCORE = sum(d[3] for d in DIMS if d[3] > 0) or 1.0  # 至少 1.0 防 div-zero


# ═══════════════════════════════════════════════════════════════════════════
# DIMS → alignment baseline spec (cold start: empty)
# ═══════════════════════════════════════════════════════════════════════════


def build_baselines() -> list[dict]:
    """Convert DIMS into baseline spec for alignment. Empty at cold start.

    After Stage C runs, persist_new_baseline_entries() appends new tuples
    here. On subsequent --skip-extract --skip-align runs, those entries
    feed back into alignment via apply_null_resolutions (which sets
    canonical_id directly on aligned claims, bypassing re-alignment)."""
    return [
        {
            "id": d_id,
            "description": f"{desc} [基准: {judgment}]",
            "match_fields": {},  # 冷启动；Stage C 直接给 canonical_id 不靠 cross_check
            "kw": kw,
            "judgment": judgment,
            "score": score,
        }
        for d_id, desc, judgment, score, kw in DIMS
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Load raw claims from v2 extraction output
# ═══════════════════════════════════════════════════════════════════════════


def load_raw_claims(extraction_dir: Path) -> dict[str, list[dict]]:
    """Read each model's `{dir}/{model}/extraction.json` and return the raw
    `entities[0].canonical.value` list."""
    out: dict[str, list[dict]] = {}
    for d in Path(extraction_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        if not entities:
            out[d.name] = []
            continue
        canonical_value = (entities[0].get("canonical") or {}).get("value") or []
        if not isinstance(canonical_value, list):
            canonical_value = []
        out[d.name] = canonical_value
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scoring (binary: +1 / -1 / 0)
# ═══════════════════════════════════════════════════════════════════════════


def score_aligned(aligned_by_model: dict[str, list[dict]]) -> dict[str, dict]:
    """Q13 binary scoring rule:

      - canonical_id ∈ DIM_MAP (Stage C verified ✅) → DIM_MAP[cid].score (+1.0)
      - canonical_id == "__HALLUCINATION__"          →  -1.0
      - canonical_id is None  (Stage C unresolved 或未走) →   0.0
      - alignment_confidence == "needs_review"       →   0.0
      - duplicates dropped (only first occurrence per cid counts)

    Totals can be negative.
    """
    all_scores: dict[str, dict] = {}
    for model_name, claims in aligned_by_model.items():
        seen_dims: set = set()
        scored: list[dict] = []
        unverified: list[dict] = []  # null + unresolved (人工二次判别用)
        total = 0.0

        for c in claims:
            raw = c.get("raw", {}) or {}
            cid = c.get("canonical_id")
            conf = c.get("alignment_confidence", "medium")
            reason = c.get("alignment_reasoning", "")
            judge_invoked = c.get("judge_invoked", False)
            event_label = (
                f"{raw.get('parent_company', '?')} {raw.get('earnings_date', '?')} "
                f"→ {raw.get('answer_ticker', '?')}"
            )

            if conf == "needs_review":
                unverified.append(
                    {
                        "event": event_label,
                        "raw_claim": raw,
                        "canonical_id_tentative": cid,
                        "reason": f"needs_review: {reason}",
                        "score": 0.0,
                    }
                )
                continue

            if cid == "__HALLUCINATION__":
                scored.append(
                    {
                        "id": "__HALLUCINATION__",
                        "name": "(Stage C 验证 6 条件失败)",
                        "event": event_label,
                        "score": -1.0,
                        "judgment": "❌",
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason,
                    }
                )
                total += -1.0
                continue

            if cid and cid in DIM_MAP:
                if cid in seen_dims:
                    continue  # dedup
                seen_dims.add(cid)
                d = DIM_MAP[cid]
                scored.append(
                    {
                        "id": cid,
                        "name": d["name"],
                        "event": event_label,
                        "score": d["score"],
                        "judgment": d["judgment"],
                        "confidence": conf,
                        "judge_invoked": judge_invoked,
                        "reason": reason or f"对齐到 {cid}（基准 {d['judgment']}）",
                    }
                )
                total += d["score"]
            else:
                # canonical_id is None → Stage C unresolved (or not yet run)
                unverified.append(
                    {
                        "event": event_label,
                        "raw_claim": raw,
                        "canonical_id_tentative": cid,
                        "reason": reason or "Stage C 搜索无果 / 未走 Stage C",
                        "score": 0.0,
                    }
                )

        n_answered = len(scored)
        all_scores[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / n_answered, 4) if n_answered else 0.0,
            "total_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
            "events_scored": n_answered,
            "events_unresolved": len(unverified),
            "per_event": sorted(scored, key=lambda x: x.get("id", "")),
            "unverified_claims": unverified,
        }
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: dict[str, dict]) -> dict:
    ranking = sorted(all_scores.items(), key=lambda x: -x[1]["total_score"])
    unverified_all = []
    for m, s in all_scores.items():
        for uv in s.get("unverified_claims", []):
            unverified_all.append({"model": m, **uv, "action_needed": "需人工核实"})
    return {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "extraction_pipeline": "v2 (primary=claude-sonnet-4, secondary=gpt-5, analyzer=claude-opus-4.6)",
        "scoring_rule": (
            "✅ Stage C 6 条件全过 → +1.0；"
            "❌ Stage C 任一条件失败 → -1.0；"
            "⚪ Stage C 搜索无果（unresolved）→ 0.0（待人工二次判定）"
        ),
        "results": all_scores,
        "ranking": [
            {
                "rank": i + 1,
                "model": m,
                "score": s["total_score"],
                "scored": s["events_scored"],
                "unresolved": s["events_unresolved"],
            }
            for i, (m, s) in enumerate(ranking)
        ],
        "unverified_items": unverified_all,
    }


def build_ranking_md(all_scores: dict[str, dict]) -> str:
    ranked = sorted(
        all_scores.items(),
        key=lambda x: (-x[1]["total_score"], -x[1].get("score_rate", 0)),
    )
    lines = [
        "# Query 13 排名报告（财报跌 ~5% 科技股 → 上下游单日跌幅最大）",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}  Mode: {SCORING_MODE}",
        f"> 基准条目数：{len(DIMS)}",
        "> 评分：✅+1 / ❌-1 / ⚪unresolved 0",
        "",
        "| Rank | Model | Score | Scored(✅+❌) | Unresolved |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, (m, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m} | {s['total_score']:+.1f} "
            f"| {s['events_scored']} "
            f"| {s['events_unresolved']} |"
        )
    lines.append("")
    lines.append("> ⚪Unresolved 的事件请看 `unresolved_review.md` 做人工二次判定。")
    return "\n".join(lines) + "\n"


def build_unresolved_review_md(
    all_scores: dict[str, dict],
    null_resolutions_path: Path,
) -> str:
    """Generate a human-readable review file listing every unresolved claim
    along with the Stage C agent's full search evidence. The user reads this
    file, decides each case manually, and updates null_resolutions.json
    accordingly. Re-running the scorer then promotes those claims to
    +1 / -1.

    If null_resolutions.json doesn't exist yet, this report only lists
    claims with their raw fields (no Stage C evidence available)."""

    # Load Stage C output for evidence dump
    resolutions_lookup: dict = {}  # (model, claim_idx) → resolution_dict
    if null_resolutions_path.exists():
        try:
            payload = json.loads(null_resolutions_path.read_text(encoding="utf-8"))
            items = (
                payload.get("items")
                or payload.get("resolutions")
                or (payload if isinstance(payload, list) else [])
            )
            for it in items:
                if not isinstance(it, dict):
                    continue
                key = (it.get("model"), it.get("claim_idx"))
                resolutions_lookup[key] = it
        except Exception as e:
            resolutions_lookup = {"_load_error": str(e)}

    lines = [
        "# Query 13 — Unresolved Claims 人工二次判定单",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}",
        "> 本文件列出所有 Stage C agent 标为 `unresolved` 的 claim，附完整搜索证据。",
        "> 你的任务：逐条判定后，把 `null_resolutions.json` 里对应 item 的"
        "`resolution` 字段从 `unresolved` 改为 `baseline_add`（+1）或 `hallucination`（-1），"
        "然后重跑 `--skip-extract --skip-align` 即可更新分数。",
        "",
        "---",
        "",
    ]

    total_unresolved = 0
    for m, s in sorted(all_scores.items()):
        unverified = s.get("unverified_claims", [])
        if not unverified:
            continue
        lines.append(f"## Model: `{m}`  (unresolved 数: {len(unverified)})")
        lines.append("")
        for i, uv in enumerate(unverified, 1):
            total_unresolved += 1
            raw = uv.get("raw_claim", {})
            event_label = uv.get("event", "?")
            lines.append(f"### {i}. {event_label}")
            lines.append("")
            lines.append("**原始 claim 字段：**")
            lines.append("```json")
            lines.append(json.dumps(raw, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append(f"**Alignment reason**: {uv.get('reason', '(none)')}")
            lines.append("")

            # Locate this claim's resolution by matching event/raw fields
            # (claim_idx may not be reliable across re-runs)
            evidence_block = None
            for (mdl, _idx), res in resolutions_lookup.items():
                if mdl != m:
                    continue
                # Match by raw fields
                res_raw = res.get("raw_claim") or res.get("claim") or {}
                if (
                    isinstance(res_raw, dict)
                    and res_raw.get("parent_company") == raw.get("parent_company")
                    and res_raw.get("answer_ticker") == raw.get("answer_ticker")
                ):
                    evidence_block = res
                    break

            if evidence_block:
                lines.append("**Stage C 搜索证据：**")
                lines.append("")
                cc = evidence_block.get("condition_check", {})
                if cc:
                    lines.append("| 条件 | 验证结果 | 证据 |")
                    lines.append("|---|---|---|")
                    for cid in ["C1", "C2", "C3", "C4", "C5", "C6"]:
                        if cid in cc:
                            v = cc[cid]
                            verdict = v.get("verdict", "?")
                            ev = (v.get("evidence", "") or "").replace("\n", " ")
                            if len(ev) > 200:
                                ev = ev[:200] + "..."
                            lines.append(f"| {cid} | {verdict} | {ev} |")
                    lines.append("")
                notes = evidence_block.get("verification_notes", "")
                if notes:
                    lines.append(f"**Stage C notes**: {notes}")
                    lines.append("")
                urls = evidence_block.get("evidence_urls", [])
                if urls:
                    lines.append("**Evidence URLs**:")
                    for u in urls:
                        lines.append(f"- {u}")
                    lines.append("")
            else:
                lines.append(
                    "> ⚠️ Stage C 未对此 claim 产出 resolution，"
                    "或 null_resolutions.json 不存在。请人工搜索后判定。"
                )
                lines.append("")

            lines.append("**建议判定（请改 null_resolutions.json）：**")
            lines.append("- [ ] `baseline_add`（6 条件人工判全过 → +1）")
            lines.append("- [ ] `hallucination`（人工判任一条件失败 → -1）")
            lines.append("- [ ] 保持 `unresolved`（确实查不到 → 0）")
            lines.append("")
            lines.append("---")
            lines.append("")

    if total_unresolved == 0:
        lines.append("> ✅ 没有 unresolved 的 claim。所有 claim 都已判定。")

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    """Reload alignment.json from disk when --skip-align is set."""
    aligned: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        payload = json.loads(af.read_text(encoding="utf-8"))
        aligned[d.name] = payload.get("aligned_claims", [])
    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 13 auto-scorer (cold-start binary)")
    ap.add_argument(
        "--models", nargs="+", help="name=path list of model answer JSON files."
    )
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; reuse existing extraction.json.",
    )
    ap.add_argument(
        "--skip-align",
        action="store_true",
        help="Skip alignment; reuse existing alignment.json.",
    )
    args = ap.parse_args()

    if not args.models:
        sys.exit("[ERROR] --models is required.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extraction
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

    # Stage 2: alignment (cold-start: empty baseline → all null)
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

    # Stage 3a: export null claims for Stage C web verification agent
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

    # Stage 3b: apply null_resolutions.json if present
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baseline_entries = apply_null_resolutions(
            aligned, resolutions_path, dims_ref=DIMS
        )
        if new_baseline_entries:
            print(
                f"[*] {len(new_baseline_entries)} new baseline entries from resolutions"
            )
            for e in new_baseline_entries:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "❌"),
                    "score": e.get("score", -1.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baseline_entries, __file__)
    else:
        print(
            f"[*] No null_resolutions.json yet."
            f"\n    → Stage C agent (read {resolutions_path.parent}/null_review.json)"
            f"\n      逐条 web 验证 6 条件清单 → 产出 {resolutions_path.name}"
            f"\n      然后重跑：python3 {Path(__file__).name} "
            f"--skip-extract --skip-align --models …"
        )

    # Stage 4: scoring
    scores = score_aligned(aligned)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(scores),
        encoding="utf-8",
    )
    (out_dir / "unresolved_review.md").write_text(
        build_unresolved_review_md(scores, resolutions_path),
        encoding="utf-8",
    )

    print("\n" + "─" * 62)
    print("Query 13 scoring done.")
    for i, (m, s) in enumerate(
        sorted(scores.items(), key=lambda x: -x[1]["total_score"]), 1
    ):
        print(
            f"  {i}. {m:28s} {s['total_score']:+5.1f}"
            f"  scored={s['events_scored']}"
            f"  unresolved={s['events_unresolved']}"
        )


if __name__ == "__main__":
    main()
