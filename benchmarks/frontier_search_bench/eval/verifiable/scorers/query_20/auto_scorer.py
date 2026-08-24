"""
Query 20 — Auto scorer v2.2 (tiered matching + LLM judge fallback)

Scoring rule (see scoring_framework.md):
  +1 per candidate song that matches the GT whitelist. No upper cap.

Matching layers (v2.2 hardening):
  0. Unicode normalize (curly quotes → straight, NFKC, case-fold).
  1. Tier 1 — exact title match after normalize + artist match → auto YES.
  2. Tier 2 — title substring (≥ 0.8 overlap ratio OR full containment for
              long titles ≥ 15 chars) + artist match → auto YES.
  3. Tier 3 — title matches whitelist entry but artist is missing / unknown
              → defer to LLM judge.
  4. Tier 4 — title is a non-English alias (present in aliases list for a
              whitelist entry) OR title very short / ambiguous
              → defer to LLM judge.
  Otherwise → auto NO.

Every candidate carries a `tier` tag; every Tier 3/4 decision is logged
in `score.json.llm_judge_log` for audit.
"""

from __future__ import annotations

import argparse
import contextlib
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
from pipeline.extraction_pipeline import (  # noqa: E402
    _extract_json,
    call_llm,
    get_client,
    run_pipeline,
)

# ═══════════════════════════════════════════════════════════════════════════
# Ground truth whitelist (v2.2 — with aliases + richer metadata)
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK: dict = {
    "version": "2.2",
    "locked_at": "2026-04-24",
    "notes": (
        "Tiered matching + LLM-judge fallback; alias lists for known translations. "
        "See changelog in scoring_framework.md."
    ),
    "whitelist": {
        "demons": {
            "canonical_title": "Demons",
            "artist": "imagine dragons",
            "aliases": ["恶魔"],
        },
        "someone you loved": {
            "canonical_title": "Someone You Loved",
            "artist": "lewis capaldi",
            "aliases": ["你曾爱过的人"],
        },
        "you're beautiful": {
            "canonical_title": "You're Beautiful",
            "artist": "james blunt",
            "aliases": ["youre beautiful", "you are beautiful", "你如此美丽"],
        },
        "i must belong somewhere": {
            "canonical_title": "I Must Belong Somewhere",
            "artist": "bright eyes",
            "aliases": [],
        },
    },
}

# LLM judge configuration
LLM_JUDGE_MODEL = "anthropic/claude-sonnet-4"
LLM_JUDGE_SYSTEM = "你是严格的歌曲匹配判别器。只输出 JSON，无多余文字。"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0 — Unicode / punctuation normalization
# ═══════════════════════════════════════════════════════════════════════════

# Map curly/typographic quotes to straight equivalents.
_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "´": "'",
        "`": "'",
        "′": "'",
        "″": '"',
    }
)


def _normalize(s: str | None) -> str:
    """NFKC + lower + curly-quote fold + strip punct + collapse ws."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTE_MAP).lower()
    # strip punctuation but keep letters (ASCII + CJK) and whitespace
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _overlap_ratio(a: str, b: str) -> float:
    """Symmetric token-set Jaccard for quick fuzzy title comparison."""
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# ═══════════════════════════════════════════════════════════════════════════
# Tiered matching
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MatchResult:
    verdict: str  # "yes" | "no"
    tier: str  # "tier1" | "tier2" | "tier3" | "tier4" | "tier_none"
    matched_key: str | None = None  # whitelist key the candidate was matched to
    reason: str = ""
    llm_used: bool = False
    llm_log: dict | None = None


def _artist_matches(model_artist: str, wl_artist: str) -> bool:
    """Soft artist match: substring either way OR shared token."""
    na, wa = _normalize(model_artist), _normalize(wl_artist)
    if not na or not wa:
        return False
    if wa in na or na in wa:
        return True
    wa_tokens = wa.split()
    return any(tok in na.split() for tok in wa_tokens)


def _rule_match(nt: str, na: str) -> MatchResult:
    """Apply deterministic Tier 1/2/None rules. Return None-verdict if
    caller should escalate to LLM (Tier 3/4)."""
    wl = BENCHMARK["whitelist"]

    # ─── Tier 1 — exact title + artist match ─────────────────────────────
    for wl_key, info in wl.items():
        if nt == wl_key or nt == _normalize(info["canonical_title"]):
            if _artist_matches(na, info["artist"]):
                return MatchResult(
                    "yes",
                    "tier1",
                    wl_key,
                    reason=f"exact title + artist match to '{wl_key}'",
                )
            if not na:
                # exact title, no artist → Tier 3 (needs LLM)
                return MatchResult(
                    "defer",
                    "tier3",
                    wl_key,
                    reason="exact title match but artist missing",
                )

    # ─── Tier 2 — substring / overlap + artist match ──────────────────────
    for wl_key, info in wl.items():
        if not nt:
            continue
        bidirectional = (wl_key in nt) or (nt in wl_key)
        overlap = _overlap_ratio(nt, wl_key)
        # Guard against short-string false positives
        title_is_long = len(wl_key) >= 15
        # Need: long title (substring OK) OR overlap ≥ 0.8
        if (title_is_long and bidirectional) or overlap >= 0.8:
            if _artist_matches(na, info["artist"]):
                return MatchResult(
                    "yes",
                    "tier2",
                    wl_key,
                    reason=f"fuzzy title match + artist to '{wl_key}' "
                    f"(overlap={overlap:.2f})",
                )
            if not na:
                return MatchResult(
                    "defer",
                    "tier3",
                    wl_key,
                    reason=f"fuzzy title match but artist missing "
                    f"(overlap={overlap:.2f})",
                )

    # ─── Tier 4 — alias list (e.g. Chinese translation) ──────────────────
    for wl_key, info in wl.items():
        for alias in info.get("aliases", []):
            na_alias = _normalize(alias)
            if na_alias and (nt == na_alias or na_alias in nt or nt in na_alias):
                if _artist_matches(na, info["artist"]):
                    return MatchResult(
                        "yes",
                        "tier2",
                        wl_key,
                        reason=f"alias match '{alias}' + artist to '{wl_key}'",
                    )
                # alias match but artist uncertain → LLM judge
                return MatchResult(
                    "defer",
                    "tier4",
                    wl_key,
                    reason=f"alias match '{alias}'; needs LLM to confirm",
                )

    # ─── No rule matched, but short ambiguous titles could still be LLM material
    #     (e.g. single-word title that isn't in whitelist either) → reject.
    return MatchResult("no", "tier_none", None, reason="no rule match")


def _llm_judge(client, candidate: dict, suggested_key: str | None) -> dict:
    """Ask Claude Sonnet 4 whether the candidate refers to any whitelist song."""
    wl_list = "\n".join(
        f"- [{k}] {info['canonical_title']} — {info['artist']}"
        f"{' (aliases: ' + ', '.join(info['aliases']) + ')' if info['aliases'] else ''}"
        for k, info in BENCHMARK["whitelist"].items()
    )
    prompt = f"""题目背景：用户在找一首歌，线索：capo 3、和弦 C-G-Am-F（主歌/前奏重复）、
桥段高八度 arpeggio、情绪忧郁反思、男歌手 30–40 岁。

候选歌曲：
- title: {candidate.get("title")!r}
- artist: {candidate.get("artist")!r}

白名单歌曲（GT 认可的 ≥4/6 线索匹配歌曲，每首 1 分）：
{wl_list}

请判断这个候选是否**实质上**指白名单里的某一首歌（考虑翻译、昵称、简写、分隔符差异等）。

严格要求：
- 若 artist 缺失但 title 与白名单某首歌常见指代一致（如 "Demons" 一般就是 Imagine Dragons 这首）且 title 在流行语境下没有其他著名同名歌能混淆，可判 match。
- 若 title 过于普通（如 "Someone"、"Lonely"）且没有 artist 佐证，判 none。
- 若 title 是白名单歌曲的外语/中文翻译，判 match 到对应白名单。

输出 JSON：
{{
  "match_to": "<whitelist key 如 'demons' / 'someone you loved' / 'youre beautiful' / 'i must belong somewhere' 中的一个；不匹配则写 'none'>",
  "confidence": "<high|medium|low>",
  "rationale": "<一句话≤50 字>"
}}
"""
    raw = call_llm(
        client,
        LLM_JUDGE_MODEL,
        LLM_JUDGE_SYSTEM,
        prompt,
        max_tokens=512,
        temperature=0,
    )
    parsed = _extract_json(raw) or {}
    return {
        "prompt": prompt,
        "raw_output": raw,
        "match_to": parsed.get("match_to", "none"),
        "confidence": parsed.get("confidence", "low"),
        "rationale": parsed.get("rationale", ""),
    }


def candidate_match(
    candidate: dict,
    client=None,
) -> MatchResult:
    """Full pipeline: normalize → rule match → (LLM judge if deferred)."""
    title = candidate.get("title", "") or ""
    artist = candidate.get("artist", "") or ""
    nt, na = _normalize(title), _normalize(artist)

    result = _rule_match(nt, na)

    if result.verdict != "defer":
        return result

    # Need LLM judge. Escalate.
    if client is None:
        client = get_client()
    judge = _llm_judge(client, candidate, result.matched_key)
    mt = judge["match_to"]
    if mt and mt != "none" and mt in BENCHMARK["whitelist"]:
        verdict = "yes" if judge["confidence"] in {"high", "medium"} else "no"
    else:
        verdict = "no"

    return MatchResult(
        verdict=verdict,
        tier=result.tier,
        matched_key=mt if verdict == "yes" else None,
        reason=f"LLM judge ({judge['confidence']}): {judge['rationale']}",
        llm_used=True,
        llm_log={
            "candidate": candidate,
            "tier": result.tier,
            "rule_reason": result.reason,
            "judge_match_to": judge["match_to"],
            "judge_confidence": judge["confidence"],
            "judge_rationale": judge["rationale"],
            "final_verdict": verdict,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    model: str
    score: int
    matched_candidates: list[dict]
    unmatched_candidates: list[dict]
    llm_judge_log: list[dict] = field(default_factory=list)
    resolution: str = "unknown"


def _score_model(
    model_name: str,
    payload: dict,
    client,
) -> ScoreResult:
    entities = payload.get("entities", [])
    if not entities:
        return ScoreResult(model_name, 0, [], [], resolution="empty")
    ent = entities[0]
    canonical = (ent.get("canonical") or {}).get("value")
    canonical = canonical if isinstance(canonical, dict) else {}
    candidates = canonical.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    matched: list[dict] = []
    unmatched: list[dict] = []
    llm_log: list[dict] = []
    matched_keys: set[str] = set()

    for c in candidates:
        if not isinstance(c, dict):
            continue
        entry = {"title": c.get("title"), "artist": c.get("artist")}
        r = candidate_match(c, client=client)
        if r.llm_used and r.llm_log:
            llm_log.append(r.llm_log)
        if r.verdict == "yes" and r.matched_key and r.matched_key not in matched_keys:
            matched_keys.add(r.matched_key)
            matched.append({**entry, "tier": r.tier, "matched_key": r.matched_key})
        elif r.verdict == "yes":
            # duplicate match — skip silently (already credited)
            continue
        else:
            unmatched.append({**entry, "tier": r.tier, "reason": r.reason})

    return ScoreResult(
        model=model_name,
        score=len(matched),
        matched_candidates=matched,
        unmatched_candidates=unmatched,
        llm_judge_log=llm_log,
        resolution=ent.get("resolution", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


# `max_score` 不是 scoring 真实上限——本题 scoring_framework v2.2 明确"无上限，
# 每首独立计数"。这里写的是**归一化参考基准** = 白名单大小，仅供 orchestrator
# 跨题汇总时把 score 软 clip 到 [0, 1] 区间，方便和其他题平均；不改变本题原始
# score 语义。
NORMALIZATION_REFERENCE_MAX = len(BENCHMARK["whitelist"])


def _write_model_score(output_dir: Path, r: ScoreResult) -> None:
    path = output_dir / r.model / "score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rate = min(r.score / NORMALIZATION_REFERENCE_MAX, 1.0)
    path.write_text(
        json.dumps(
            {
                "query_id": QUERY_ID,
                "model": r.model,
                "score": r.score,
                "max_score": NORMALIZATION_REFERENCE_MAX,
                "total_rate": total_rate,
                "matched_candidates": r.matched_candidates,
                "unmatched_candidates": r.unmatched_candidates,
                "llm_judge_log": r.llm_judge_log,
                "extraction_resolution": r.resolution,
                "benchmark_version": BENCHMARK["version"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_ranking_report(output_dir: Path, results: list[ScoreResult]) -> None:
    lines: list[str] = []
    lines.append("# Query 20 Ranking Report (v2.2)")
    lines.append("")
    lines.append(
        f"> Benchmark version: {BENCHMARK['version']} (locked {BENCHMARK['locked_at']})"
    )
    lines.append("> Scoring: +1 per candidate matching the whitelist.")
    lines.append("> Matching: deterministic rule tiers + LLM-judge fallback for")
    lines.append("> title-without-artist / alias cases.")
    lines.append("")
    lines.append(
        "| Rank | Model | Score | Matched | Other Candidates | LLM judge calls |"
    )
    lines.append("|---:|---|---:|---|---|---:|")
    ranked = sorted(results, key=lambda r: (-r.score, r.model))
    for i, r in enumerate(ranked, start=1):
        matched = (
            "; ".join(f"{m.get('title', '?')}" for m in r.matched_candidates) or "—"
        )
        unmatched = (
            "; ".join(f"{m.get('title', '?')}" for m in r.unmatched_candidates) or "—"
        )
        lines.append(
            f"| {i} | {r.model} | {r.score} | {matched[:60]} | "
            f"{unmatched[:60]} | {len(r.llm_judge_log)} |"
        )
    (output_dir / "ranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    global LLM_JUDGE_MODEL
    ap = argparse.ArgumentParser(description="Query 20 auto-scorer v2.2")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument(
        "--judge-model",
        default=LLM_JUDGE_MODEL,
        help="LLM used to resolve Tier 3/4 cases.",
    )
    args = ap.parse_args()
    LLM_JUDGE_MODEL = args.judge_model

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

    # Lazy-initialize the LLM client only if any model might need judge
    client = None
    with contextlib.suppress(SystemExit):
        client = get_client()

    results = []
    for model_name, payload in all_results.items():
        r = _score_model(model_name, payload, client=client)
        _write_model_score(out_dir, r)
        results.append(r)

    _write_ranking_report(out_dir, results)
    total_llm = sum(len(r.llm_judge_log) for r in results)
    print("\n" + "─" * 60)
    print(f"Query 20 scoring done. LLM judge calls: {total_llm}")
    for r in sorted(results, key=lambda x: (-x.score, x.model)):
        matches = [m.get("title") for m in r.matched_candidates]
        print(
            f"  {r.model}: {r.score}  matches={matches}  "
            f"llm_judge_used={len(r.llm_judge_log)}"
        )


if __name__ == "__main__":
    main()
