"""
Orchestrator — 一次跑遍 eval/verifiable/scorers/query_*/auto_scorer.py，输出跨题汇总。

流程：
  1. Pre-flight：用 queries/verifiable.json 当 canonical 真相，
     校验每个模型 unified JSON 的 (id, query) 一致性，算覆盖度。
  2. 逐题 subprocess 调用 scorers/query_NN/auto_scorer.py，跳过 scores.json 里
     已含全部目标模型的题（除非 --force-rerun）。
  3. Aggregate：读每题的 scores.json，组装 41 行 × N 列矩阵。
  4. 输出 matrix.csv / ranking.md / coverage.json + 每题一份 logs/query_NN.log，
     ranking.md 顶部带每个 model 文件 + canonical query 集的 sha256 指纹。

CLI:
  python run_all.py \\
      --models claude=/path/claude.json gpt=/path/gpt.json \\
      [--out all_results/] \\
      [--only 22,33] \\
      [--force]                # 覆盖已有 all_results 目录
      [--force-rerun]          # 重跑已经打过分的题
      [--allow-query-mismatch] # 容忍 id↔query text mismatch（默认 abort）
      [--dry-run]              # 只打印计划，不真跑

输入约定：每个模型一份 unified JSON，形如
  [{"id": 1, "query": "...", "response": "..."}, ...]
id 与 queries/verifiable.json 对齐。

注意：每题的 scorers/query_NN/auto_scores/scores.json 是被对应 scorer 整体覆盖
写入的——若某次 run 只传了部分模型，scores.json 里其他模型的旧数据会被擦掉。
所以建议每次都把所有关心的模型一起传进来，让 orchestrator 一次到位。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent  # eval/verifiable/
REPO_ROOT = THIS.parent.parent  # frontier-search-bench/
CANONICAL_QUERIES = REPO_ROOT / "queries" / "verifiable.json"

# 历史上用来登记尚未实现自动评分的题号；41 题已全部覆盖，留空集保留下方一致性自检。
UNIMPLEMENTED_IDS: set[int] = set()


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelInput:
    name: str  # CLI 上传入的别名，例如 "claude"
    path: Path  # unified JSON 路径
    sha256: str  # 文件内容指纹
    by_id: dict[int, dict]  # {id: entry} 解析后
    present_ids: set[int]  # 文件里含哪些 id

    def has(self, qid: int) -> bool:
        return qid in self.present_ids


@dataclass
class QueryRun:
    qid: int
    scorer_path: Path  # scorers/query_NN/auto_scorer.py
    out_dir: Path  # scorers/query_NN/auto_scores/
    target_models: list[
        str
    ]  # 这一题要打分的模型名（仅含那些 unified JSON 里有该 id 的）
    skipped: bool = False  # 已有 scores.json 且覆盖到位 → 跳过
    skip_reason: str = ""
    success: bool = False  # subprocess 跑完且产生 scores.json
    duration_s: float = 0.0
    stderr_tail: str = ""
    # 解析 scores.json 后的：{model_name: total_rate}
    rates: dict[str, float] = field(default_factory=dict)
    max_score: float | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight
# ═══════════════════════════════════════════════════════════════════════════


_ZERO_WIDTH = ("​", "‌", "‍", "﻿", "‎", "‏")


def _norm_query(s: str) -> str:
    """归一化 query 文本以做比对：strip 零宽 + NFKC + strip + 折叠所有 whitespace 为单空格。
    NFKC 不消除 zero-width / BOM / RLM 等不可见字符，所以先手动剔。"""
    if s is None:
        return ""
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_models_arg(items: list[str]) -> list[tuple[str, Path]]:
    """解析 ['claude=/a/b.json', 'gpt=/c/d.json'] → [(name, Path)]。仅在第一个 '=' 切。"""
    out: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for raw in items:
        if "=" not in raw:
            sys.exit(f"[ERROR] --models 项必须是 name=path 形式，收到: {raw!r}")
        name, _, path = raw.partition("=")
        name = name.strip()
        path = path.strip()
        if not name or not path:
            sys.exit(f"[ERROR] --models 项 name 或 path 为空: {raw!r}")
        if name in seen_names:
            sys.exit(f"[ERROR] --models 中模型名 {name!r} 重复出现")
        seen_names.add(name)
        p = Path(path).expanduser()
        if not p.is_file():
            sys.exit(f"[ERROR] 模型 {name} 的文件不存在: {p}")
        out.append((name, p))
    return out


def load_canonical_queries() -> dict[int, str]:
    """读 queries/verifiable.json，返回 {id: normalized_query}。"""
    if not CANONICAL_QUERIES.is_file():
        sys.exit(f"[ERROR] 找不到 canonical query 文件: {CANONICAL_QUERIES}")
    data = json.loads(CANONICAL_QUERIES.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit(f"[ERROR] {CANONICAL_QUERIES} 顶层不是 list")
    out: dict[int, str] = {}
    for i, ent in enumerate(data):
        if not isinstance(ent, dict) or "id" not in ent or "query" not in ent:
            sys.exit(f"[ERROR] {CANONICAL_QUERIES} 第 {i} 条缺 id/query 字段")
        qid = ent["id"]
        if not isinstance(qid, int):
            sys.exit(f"[ERROR] {CANONICAL_QUERIES} 第 {i} 条 id 非 int: {qid!r}")
        if qid in out:
            sys.exit(f"[ERROR] {CANONICAL_QUERIES} 中 id={qid} 重复")
        out[qid] = _norm_query(ent["query"])
    return out


def load_model_input(
    name: str,
    path: Path,
    canonical: dict[int, str],
    allow_mismatch: bool,
) -> tuple[ModelInput, list[str], list[str]]:
    """读单个模型的 unified JSON，返回 (ModelInput, mismatches, warnings)。
    mismatches 在 allow_mismatch=False 时会让上层 abort。"""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] {name} 文件不是合法 JSON: {path}: {e}")
    if not isinstance(data, list):
        sys.exit(f"[ERROR] {name} 文件顶层应为 list: {path}")

    by_id: dict[int, dict] = {}
    mismatches: list[str] = []
    warnings: list[str] = []
    for i, ent in enumerate(data):
        if not isinstance(ent, dict):
            warnings.append(f"  - 第 {i} 条不是 dict，跳过")
            continue
        qid = ent.get("id")
        if not isinstance(qid, int):
            warnings.append(f"  - 第 {i} 条 id 非 int 或缺失：{qid!r}，跳过")
            continue
        if qid in by_id:
            warnings.append(f"  - id={qid} 在文件中重复出现，保留第一个")
            continue
        if qid not in canonical:
            warnings.append(f"  - id={qid} 不在 canonical 41 题里（多余条目），忽略")
            continue
        # query 文本一致性
        their_q = _norm_query(ent.get("query", ""))
        if their_q and their_q != canonical[qid]:
            mismatches.append(
                f"  - id={qid}: query text 与 canonical 不一致\n"
                f"      canonical: {canonical[qid][:80]}…\n"
                f"      this file: {their_q[:80]}…"
            )
        # 没填 query 字段也警告（数据契约要求）
        if not their_q:
            warnings.append(
                f"  - id={qid}: 没有 query 字段（按 id 对齐打分，可能有风险）"
            )
        by_id[qid] = ent

    sha = _sha256_file(path)
    mi = ModelInput(
        name=name,
        path=path,
        sha256=sha,
        by_id=by_id,
        present_ids=set(by_id.keys()),
    )
    return mi, mismatches, warnings


# ═══════════════════════════════════════════════════════════════════════════
# Query discovery + scores.json reading
# ═══════════════════════════════════════════════════════════════════════════


SCORERS_DIR = THIS / "scorers"


def discover_implemented_queries() -> list[tuple[int, Path]]:
    """扫 scorers/query_*/auto_scorer.py，返回 [(qid, scorer_path)] 按 qid 升序。"""
    out: list[tuple[int, Path]] = []
    for d in sorted(SCORERS_DIR.glob("query_*")):
        if not d.is_dir():
            continue
        m = re.match(r"query_(\d+)$", d.name)
        if not m:
            continue
        qid = int(m.group(1))
        scorer = d / "auto_scorer.py"
        if not scorer.is_file():
            continue
        out.append((qid, scorer))
    return out


def read_existing_scores(scores_json: Path) -> dict | None:
    """安全地读 scorers/query_NN/auto_scores/scores.json。损坏返回 None。"""
    if not scores_json.is_file():
        return None
    try:
        return json.loads(scores_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _synthesize_scores_from_per_model(out_dir: Path) -> dict | None:
    """部分 scorer 不写顶层 scores.json，只写 per-model <out_dir>/<model>/score.json。
    扫这些文件并合成一个等价 scores.json dict，让 orchestrator 用统一接口。

    字段名兼容：
      score 字段：total_score | score（不同 scorer 历史命名不一）
      上限：max_score
      若 score.json 直接含 total_rate 也接受。
    每个 per-model 文件必须能算出 total_rate（直接给、或 score+max_score）才算数；
    否则该模型在该题被静默丢弃（最终矩阵中显示 failed）。

    返回 None 表示一个能用的 per-model 文件都没找到。"""
    if not out_dir.is_dir():
        return None
    results: dict[str, dict] = {}
    max_seen: float | None = None
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        sf = sub / "score.json"
        if not sf.is_file():
            continue
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        # score: 接受 total_score 或 score
        score: float | None = None
        for key in ("total_score", "score"):
            v = data.get(key)
            if isinstance(v, (int, float)):
                score = float(v)
                break
        # max_score: 必须是正数
        ms_raw = data.get("max_score")
        ms = float(ms_raw) if isinstance(ms_raw, (int, float)) and ms_raw > 0 else None
        # total_rate: 直接给的优先
        tr_raw = data.get("total_rate")
        tr = float(tr_raw) if isinstance(tr_raw, (int, float)) else None

        if tr is not None:
            entry: dict = {"total_rate": tr}
            if score is not None:
                entry["total_score"] = score
            if ms is not None:
                entry["max_score"] = ms
                if max_seen is None or ms > max_seen:
                    max_seen = ms
            results[sub.name] = entry
        elif score is not None and ms is not None:
            results[sub.name] = {
                "total_score": score,
                "max_score": ms,
                "total_rate": score / ms,
            }
            if max_seen is None or ms > max_seen:
                max_seen = ms
        # 既无 total_rate 也无 (score, max_score)：跳过（如 Q20 没有 max_score 概念）

    if not results:
        return None
    return {
        "results": results,
        "max_score": max_seen,
        "_synthesized_from": "per_model_score_json",
    }


def models_already_scored(scores: dict | None) -> set[str]:
    """已"成功打分"的模型名集合 = scores.json 里能抽出 numeric total_rate 的模型。
    只是出现在 results 但 score 是 null/string/garbage 的情况算"未打分"，会触发重跑。"""
    rates, _ = extract_total_rates(scores)
    return set(rates.keys())


def extract_total_rates(scores: dict | None) -> tuple[dict[str, float], float | None]:
    """从 scores.json 提取 {model: total_rate}，以及题的 max_score（若有）。
    total_rate = total_score / max_score，是 0-1 区间的归一化分。"""
    rates: dict[str, float] = {}
    max_score: float | None = None
    if not isinstance(scores, dict):
        return rates, max_score
    if isinstance(scores.get("max_score"), (int, float)):
        max_score = float(scores["max_score"])
    results = scores.get("results")
    # 兼容两种 schema：{model: entry} dict 形式，或 [{"model": name, ...}] list 形式。
    # 旧 scorer（Q28/Q39 等）历史上写 list，新 scorer 全部走 dict（Q22 reference）。
    items: list[tuple[str, dict]] = []
    if isinstance(results, dict):
        items = [(m, r) for m, r in results.items() if isinstance(r, dict)]
    elif isinstance(results, list):
        for r in results:
            if isinstance(r, dict) and isinstance(r.get("model"), str):
                items.append((r["model"], r))
    for m, r in items:
        # 优先 total_rate / score_rate（per-entry 已归一化），其次 total_score / total 配 max_score 计算。
        tr = r.get("total_rate") if "total_rate" in r else r.get("score_rate")
        if isinstance(tr, (int, float)):
            rates[m] = float(tr)
            continue
        ts = r.get("total_score") if "total_score" in r else r.get("total")
        ms = (
            r.get("max_score")
            if isinstance(r.get("max_score"), (int, float))
            else max_score
        )
        if isinstance(ts, (int, float)) and ms:
            rates[m] = float(ts) / float(ms)
    return rates, max_score


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess runner
# ═══════════════════════════════════════════════════════════════════════════


def _read_tail(p: Path, max_bytes: int) -> str:
    """读文件末尾 max_bytes 字节（避免对大日志一次性 read_text）。"""
    try:
        size = p.stat().st_size
        offset = max(0, size - max_bytes)
        with p.open("rb") as f:
            f.seek(offset)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def invoke_scorer(
    qrun: QueryRun,
    model_inputs: dict[str, ModelInput],
    log_dir: Path,
    timeout_s: int | None,
) -> None:
    """跑一题。结果直接写回 qrun（success / duration_s / stderr_tail / rates）。"""
    cmd = [
        sys.executable,
        str(qrun.scorer_path),
        "--output-dir",
        str(qrun.out_dir),
        "--models",
    ] + [f"{m}={model_inputs[m].path}" for m in qrun.target_models]
    log_path = log_dir / f"query_{qrun.qid:02d}.log"
    t0 = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(
                    THIS
                ),  # 让 scorer 内部的相对 import / sys.path 行为与人手跑一致
                stdout=logf,
                stderr=subprocess.STDOUT,  # stderr 合并到同一个 log
                timeout=timeout_s,
                check=False,
            )
        qrun.duration_s = time.monotonic() - t0
        if proc.returncode != 0:
            qrun.success = False
            qrun.stderr_tail = _read_tail(log_path, max_bytes=2000)
            return
    except subprocess.TimeoutExpired:
        qrun.duration_s = time.monotonic() - t0
        qrun.success = False
        qrun.stderr_tail = f"[TIMEOUT after {timeout_s}s]"
        return
    except Exception as e:
        qrun.duration_s = time.monotonic() - t0
        qrun.success = False
        qrun.stderr_tail = f"[ORCHESTRATOR EXCEPTION] {type(e).__name__}: {e}"
        return

    # 解析产物：优先读顶层 scores.json，没有就从 per-model score.json 合成
    scores = read_existing_scores(qrun.out_dir / "scores.json")
    if scores is None:
        scores = _synthesize_scores_from_per_model(qrun.out_dir)
    if scores is None:
        qrun.success = False
        qrun.stderr_tail = "[NO scores.json or per-model score.json produced]"
        return
    rates, max_score = extract_total_rates(scores)
    qrun.rates = rates
    qrun.max_score = max_score
    # 检查目标模型是否都被打到了
    missing = [m for m in qrun.target_models if m not in rates]
    if missing:
        qrun.success = False
        qrun.stderr_tail = f"[scores.json 缺模型: {missing}]"
        return
    qrun.success = True


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation + reporting
# ═══════════════════════════════════════════════════════════════════════════


def write_matrix_csv(
    out_path: Path,
    canonical_ids: list[int],
    model_names: list[str],
    runs_by_qid: dict[int, QueryRun],
) -> None:
    """41 行 × (1+N) 列。每格：total_rate（4 位小数）/ N/A / unimpl / failed。"""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", *model_names])
        for qid in canonical_ids:
            row: list[object] = [qid]
            run = runs_by_qid.get(qid)
            for m in model_names:
                row.append(_cell_value(qid, m, run))
            w.writerow(row)


def _cell_value(qid: int, model: str, run: QueryRun | None) -> str:
    if qid in UNIMPLEMENTED_IDS:
        return "unimpl"
    if run is None:
        return "unimpl"  # 题在 UNIMPLEMENTED 里，或目录没找到（理论上前者已 cover）
    if model not in run.target_models:
        return "N/A"  # 模型 unified JSON 里没这个 id
    if not run.success:
        return "failed"
    rate = run.rates.get(model)
    if rate is None:
        return "failed"
    return f"{rate:.4f}"


def build_ranking(
    canonical_ids: list[int],
    model_names: list[str],
    runs_by_qid: dict[int, QueryRun],
) -> list[dict]:
    """按 mean(total_rate over implemented & answered & success) 排序。
    每个模型再附 coverage = 实际打到分的题数 / len(impl_ids)。"""
    impl_ids = [q for q in canonical_ids if q not in UNIMPLEMENTED_IDS]
    n_impl = len(impl_ids)
    rows = []
    for m in model_names:
        scored: list[float] = []
        for qid in impl_ids:
            run = runs_by_qid.get(qid)
            if run is None or not run.success:
                continue
            if m not in run.target_models:
                continue
            r = run.rates.get(m)
            if isinstance(r, float):
                scored.append(r)
        avg = sum(scored) / len(scored) if scored else None
        rows.append(
            {
                "model": m,
                "avg_rate": avg,
                "scored_on": len(scored),
                "implemented": n_impl,
            }
        )
    # avg=None（一题没打到）排到最后
    rows.sort(key=lambda r: -1 if r["avg_rate"] is None else -r["avg_rate"])
    return rows


def write_ranking_md(
    out_path: Path,
    canonical_ids: list[int],
    model_inputs: dict[str, ModelInput],
    canonical_text_hash: str,
    runs_by_qid: dict[int, QueryRun],
    ranking_rows: list[dict],
    started_at: str,
    finished_at: str,
) -> None:
    impl_ids = [q for q in canonical_ids if q not in UNIMPLEMENTED_IDS]
    model_names = list(model_inputs.keys())
    lines: list[str] = []
    lines.append("# Verifiable Eval — 跨题汇总")
    lines.append("")
    lines.append(f"> Started: {started_at}  Finished: {finished_at}")
    if UNIMPLEMENTED_IDS:
        lines.append(
            f"> Implemented: {len(impl_ids)}/41  Unimplemented (placeholder): {sorted(UNIMPLEMENTED_IDS)}"
        )
    else:
        lines.append(f"> Implemented: {len(impl_ids)}/41 (full coverage)")
    lines.append("")
    lines.append("## 输入指纹")
    lines.append("")
    lines.append(
        f"- canonical 41 queries hash (NFKC-normalized): `{canonical_text_hash[:16]}…`"
    )
    for m in model_names:
        mi = model_inputs[m]
        lines.append(f"- model `{m}`: `{mi.sha256[:16]}…`  ({mi.path})")
    lines.append("")
    lines.append("## 模型总排名")
    lines.append("")
    lines.append("> 排名依据：mean(total_rate) over (已实现题 ∩ 模型答了 ∩ 评分成功)；")
    lines.append(f"> coverage = 模型实际拿到分的题数 / {len(impl_ids)}（已实现题）。")
    lines.append("")
    lines.append("| Rank | Model | Avg Score | Coverage |")
    lines.append("|---:|---|---:|---:|")
    for i, row in enumerate(ranking_rows, 1):
        avg = row["avg_rate"]
        avg_s = f"{avg * 100:.2f}%" if avg is not None else "—"
        lines.append(
            f"| {i} | {row['model']} | {avg_s} | {row['scored_on']}/{row['implemented']} |"
        )
    lines.append("")
    lines.append("## 每题胜出模型")
    lines.append("")
    lines.append("| QID | 状态 | Best | Score | Models scored |")
    lines.append("|---:|---|---|---:|---:|")
    for qid in canonical_ids:
        if qid in UNIMPLEMENTED_IDS:
            lines.append(f"| Q{qid:02d} | unimpl | — | — | — |")
            continue
        run = runs_by_qid.get(qid)
        if run is None or not run.success:
            reason = "skipped" if (run and run.skipped) else "failed"
            lines.append(f"| Q{qid:02d} | {reason} | — | — | — |")
            continue
        if not run.rates:
            lines.append(f"| Q{qid:02d} | empty | — | — | 0 |")
            continue
        best_m, best_r = max(run.rates.items(), key=lambda kv: kv[1])
        lines.append(
            f"| Q{qid:02d} | ✓ | {best_m} | {best_r * 100:.1f}% | {len(run.rates)} |"
        )
    lines.append("")
    lines.append("## 完整矩阵")
    lines.append("")
    lines.append(
        "> 见同目录 `matrix.csv`。每格：total_rate（0-1）/ `N/A`（模型 JSON 缺该 id）/ `unimpl`（题未实现）/ `failed`（scorer 出错）。"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_only(s: str | None, canonical_ids: set[int]) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip().lstrip("Qq")
        if not tok:
            continue
        try:
            qid = int(tok)
        except ValueError:
            sys.exit(f"[ERROR] --only 解析失败: {tok!r}")
        if qid not in canonical_ids:
            sys.exit(f"[ERROR] --only 包含未知 query id: {qid}")
        out.add(qid)
    if not out:
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="跑遍 verifiable 评测并汇总。")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="形如 name=path 的模型 unified JSON 列表（>=1 个）。",
    )
    ap.add_argument(
        "--out",
        default=str(THIS / "all_results"),
        help="输出目录（默认 eval/verifiable/all_results）。",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="只跑这些 query id（逗号分隔，如 22,33 或 Q22,Q33）。",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已存在且非空的 --out 目录（默认 abort）。",
    )
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="即使 scorers/query_NN/auto_scores/scores.json 已含全部模型，也重跑。",
    )
    ap.add_argument(
        "--allow-query-mismatch",
        action="store_true",
        help="容忍模型 JSON 里 query 文本与 canonical 不一致（默认 abort）。",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="只 pre-flight + 打印计划，不调 scorer。"
    )
    ap.add_argument(
        "--per-query-timeout",
        type=int,
        default=3600,
        help="单题 subprocess 超时秒数（默认 3600 = 1h）。0 表示不限。",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists() and not out_dir.is_dir():
        sys.exit(f"[ERROR] --out 指向的不是目录：{out_dir}")
    # ─── Fail-safe #3: 拒绝覆盖已有非空目录 ───
    if not args.dry_run and out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            sys.exit(
                f"[ERROR] 输出目录已存在且非空: {out_dir}\n"
                f"    加 --force 才会覆盖（会清掉里面的 matrix.csv / ranking.md / coverage.json / logs/）。"
            )
        # 只清我们自己产出的文件，避免误删用户放在那里的别的东西
        for name in ("matrix.csv", "ranking.md", "coverage.json"):
            p = out_dir / name
            if p.exists():
                p.unlink()
        logs_dir = out_dir / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)

    # ─── 读 canonical ───
    canonical = load_canonical_queries()
    canonical_ids = sorted(canonical.keys())
    canonical_text = json.dumps(
        [canonical[i] for i in canonical_ids], ensure_ascii=False
    )
    canonical_hash = _sha256_text(canonical_text)
    print(
        f"[*] Canonical queries: {len(canonical_ids)} 题  hash={canonical_hash[:12]}…"
    )

    # ─── 解析 --models ───
    pairs = parse_models_arg(args.models)
    model_inputs: dict[str, ModelInput] = {}
    all_mismatches: dict[str, list[str]] = {}
    all_warnings: dict[str, list[str]] = {}
    for name, path in pairs:
        mi, mm, ww = load_model_input(name, path, canonical, args.allow_query_mismatch)
        model_inputs[name] = mi
        if mm:
            all_mismatches[name] = mm
        if ww:
            all_warnings[name] = ww

    # ─── 打印 coverage 表 ───
    print("\n[*] 模型 unified JSON 覆盖度（vs canonical 41 题）：")
    header = "         " + "  ".join(f"{n[:10]:>10s}" for n in model_inputs)
    print(header)
    for qid in canonical_ids:
        cells = []
        for _n, mi in model_inputs.items():
            cells.append(f"{'✓' if mi.has(qid) else '✗':>10s}")
        print(f"  Q{qid:02d}   " + "  ".join(cells))
    summary = "  total"
    for _n, mi in model_inputs.items():
        summary += f"  {len(mi.present_ids):>4d}/41    "
    print(summary)

    if all_warnings:
        print("\n[!] 警告：")
        for n, ws in all_warnings.items():
            print(f"  model {n}:")
            for w in ws:
                print(w)

    # ─── Mismatch fail-safe ───
    if all_mismatches:
        print("\n[!] 检测到 query 文本与 canonical 不一致：")
        for n, mm in all_mismatches.items():
            print(f"  model {n}:")
            for line in mm:
                print(line)
        if not args.allow_query_mismatch:
            sys.exit("[ERROR] mismatch 默认致命，加 --allow-query-mismatch 才继续。")
        print("[!] --allow-query-mismatch 开启，按 id 强行打分。")

    # ─── 发现已实现的题 ───
    impl_pairs = discover_implemented_queries()
    impl_ids_present = {qid for qid, _ in impl_pairs}
    # roadmap 一致性自检：UNIMPLEMENTED ∪ implemented == canonical
    uncovered = set(canonical_ids) - impl_ids_present - UNIMPLEMENTED_IDS
    spurious = impl_ids_present & UNIMPLEMENTED_IDS
    if uncovered:
        print(
            f"\n[!] 警告：以下 id 既不在 UNIMPLEMENTED_IDS，也没找到 scorers/query_NN/auto_scorer.py：{sorted(uncovered)}"
        )
    if spurious:
        print(
            f"\n[!] 警告：UNIMPLEMENTED_IDS 里的题居然有 auto_scorer.py：{sorted(spurious)}（请更新 UNIMPLEMENTED_IDS）"
        )

    only_set = parse_only(args.only, set(canonical_ids))
    if only_set is not None:
        unimpl_in_only = sorted(only_set & UNIMPLEMENTED_IDS)
        if unimpl_in_only:
            print(
                f"\n[!] --only 包含未实现自动评分的 id：{unimpl_in_only}（这些题会被跳过，"
                f"在 ranking/matrix 中显示 unimpl）。"
            )

    # ─── 组装 QueryRun 列表 ───
    runs_by_qid: dict[int, QueryRun] = {}
    plan: list[QueryRun] = []
    for qid, scorer in impl_pairs:
        if only_set is not None and qid not in only_set:
            continue
        out_q = scorer.parent / "auto_scores"
        targets = [n for n, mi in model_inputs.items() if mi.has(qid)]
        qr = QueryRun(
            qid=qid,
            scorer_path=scorer,
            out_dir=out_q,
            target_models=targets,
        )
        if not targets:
            qr.skipped = True
            qr.skip_reason = "no model has this id"
        else:
            existing = read_existing_scores(out_q / "scores.json")
            if existing is None:
                existing = _synthesize_scores_from_per_model(out_q)
            already = models_already_scored(existing)
            if not args.force_rerun and set(targets).issubset(already):
                qr.skipped = True
                qr.skip_reason = "already scored"
                # models_already_scored 保证了所有 target 都有可抽取的 rate，
                # 所以 qr.rates 必然完整、success 必然为 True。
                rates, mx = extract_total_rates(existing)
                qr.rates = {m: rates[m] for m in targets}
                qr.max_score = mx
                qr.success = True
        runs_by_qid[qid] = qr
        plan.append(qr)

    print(f"\n[*] 计划：{len(plan)} 题")
    todo = [r for r in plan if not r.skipped]
    skipped = [r for r in plan if r.skipped]
    print(
        f"    待跑 {len(todo)} 题，跳过 {len(skipped)} 题（已有结果或目标模型 0 个）。"
    )
    if args.dry_run:
        for r in plan:
            tag = "RUN" if not r.skipped else f"SKIP({r.skip_reason})"
            print(f"    Q{r.qid:02d} [{tag}] models={r.target_models}")
        print("\n[*] dry-run 结束。")
        return

    # ─── 真跑 ───
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    timeout = args.per_query_timeout if args.per_query_timeout > 0 else None
    for i, qr in enumerate(todo, 1):
        print(
            f"\n[{i}/{len(todo)}] Q{qr.qid:02d} → {qr.scorer_path.relative_to(THIS)} (models={qr.target_models})"
        )
        invoke_scorer(qr, model_inputs, out_dir / "logs", timeout)
        tag = "OK" if qr.success else "FAIL"
        print(f"    [{tag}] {qr.duration_s:.1f}s")
        if not qr.success:
            print(f"    log: {out_dir / 'logs' / f'query_{qr.qid:02d}.log'}")
            tail_lines = (qr.stderr_tail or "").splitlines()
            if tail_lines:
                print(f"    {tail_lines[-1]}")
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S %z")

    # ─── 写产物 ───
    coverage_obj = {
        "canonical_hash": canonical_hash,
        "models": {
            n: {
                "path": str(mi.path),
                "sha256": mi.sha256,
                "present_ids": sorted(mi.present_ids),
                "missing_ids": sorted(set(canonical_ids) - mi.present_ids),
            }
            for n, mi in model_inputs.items()
        },
        "runs": {
            f"Q{q:02d}": {
                "skipped": runs_by_qid[q].skipped,
                "skip_reason": runs_by_qid[q].skip_reason,
                "success": runs_by_qid[q].success,
                "duration_s": round(runs_by_qid[q].duration_s, 2),
                "target_models": runs_by_qid[q].target_models,
                "rates": runs_by_qid[q].rates,
                "max_score": runs_by_qid[q].max_score,
            }
            for q in sorted(runs_by_qid)
        },
    }
    (out_dir / "coverage.json").write_text(
        json.dumps(coverage_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_matrix_csv(
        out_dir / "matrix.csv",
        canonical_ids,
        list(model_inputs.keys()),
        runs_by_qid,
    )
    ranking_rows = build_ranking(
        canonical_ids,
        list(model_inputs.keys()),
        runs_by_qid,
    )
    write_ranking_md(
        out_dir / "ranking.md",
        canonical_ids,
        model_inputs,
        canonical_hash,
        runs_by_qid,
        ranking_rows,
        started_at,
        finished_at,
    )

    # ─── 汇总打屏 ───
    print("\n" + "═" * 64)
    print("跑完。产物：")
    print(f"  - {out_dir / 'matrix.csv'}")
    print(f"  - {out_dir / 'ranking.md'}")
    print(f"  - {out_dir / 'coverage.json'}")
    print(f"  - {out_dir / 'logs/'}")
    print("\n模型总排名：")
    print(f"  {'Rank':>4} {'Model':<24} {'Avg':>8} {'Coverage':>10}")
    for i, row in enumerate(ranking_rows, 1):
        avg = row["avg_rate"]
        avg_s = f"{avg * 100:.2f}%" if avg is not None else "    —"
        cov = f"{row['scored_on']}/{row['implemented']}"
        print(f"  {i:>4} {row['model']:<24} {avg_s:>8} {cov:>10}")


if __name__ == "__main__":
    main()
