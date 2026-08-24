"""Standardize OfficeQA / GDPval / OneMillion-Bench / APEX into the harness JSONL."""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env before resolving the datasets root: that is where
# FRONTIER_AGENT_DATASETS_DIR normally lives, and reading os.environ alone would
# standardize into the in-repo directory while the runner loads from the
# override. registry.py gets the same side effect via frontier_agent.infra.config.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

# Must resolve exactly like benchmarks.public.core.registry._DATA_ROOT, or the
# standardized rows land somewhere the runner does not look.
_DATASETS = Path(
    os.environ.get("FRONTIER_AGENT_DATASETS_DIR", "").strip()
    or _REPO_ROOT / "benchmarks" / "public" / "datasets"
)


def _write(key: str, rows: list[dict], jsonl: str = "standardized_data.jsonl") -> None:
    out = _DATASETS / key / jsonl
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {out}")


def _officeqa_rows(csv_name: str) -> list[dict]:
    """Read an OfficeQA CSV into standardized rows.

    The whole Treasury corpus is mounted at /inputs regardless of question;
    ``source_files`` is kept for reference. ``answer_type='officeqa'`` routes
    to the numeric-tolerant scorer (same for Pro and Full).
    """
    src = _DATASETS / "OfficeQA" / csv_name
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "task_id": row["uid"],
                "task_question": row["question"],
                "ground_truth": row["answer"],
                "answer_type": "officeqa",
                "source_files": row.get("source_files", ""),
                "difficulty": row.get("difficulty", ""),
            })
    return rows


def officeqa() -> None:
    """OfficeQA Pro (133): the harder curated subset."""
    _write("OfficeQA", _officeqa_rows("officeqa_pro.csv"))


def officeqa_full() -> None:
    """OfficeQA Full (246): the full question set, SAME corpus/prompt/scorer.

    Shares the ``OfficeQA/`` directory (so the Treasury corpus is stored once);
    only the question JSONL differs (``standardized_full.jsonl``).
    """
    _write("OfficeQA", _officeqa_rows("officeqa_full.csv"), "standardized_full.jsonl")


def gdpval() -> None:
    """GDPval (220): file deliverable with deterministic artifact validation.

    ``ground_truth`` = JSON ``{"rubric": [...], "reference_files": [<GDPval-root-
    relative paths to expected deliverable_files>]}``. ``reference_files``
    (pipe-joined) are the input attachments mounted read-only at /inputs.
    """
    import pandas as pd

    ds = _DATASETS / "GDPval"
    pq = glob.glob(str(ds / "data" / "*.parquet"))[0]
    df = pd.read_parquet(pq)
    rows: list[dict] = []
    for _, r in df.iterrows():
        rf = r["reference_files"]
        refs = [str(x) for x in rf] if rf is not None else []
        df_files = r["deliverable_files"]
        delivs = [str(x) for x in df_files] if df_files is not None else []
        raw_rubric = r["rubric_json"]
        rubric = (
            json.loads(raw_rubric) if isinstance(raw_rubric, str)
            else list(raw_rubric)
        )
        ground_truth = json.dumps(
            {"rubric": rubric, "reference_files": delivs}, ensure_ascii=False,
        )
        rows.append({
            "task_id": str(r["task_id"]),
            "task_question": str(r["prompt"]),
            "ground_truth": ground_truth,
            "answer_type": "gdpval",
            "category": str(r.get("occupation", "")),
            "reference_files": "|".join(refs),
        })
    _write("GDPval", rows)


def onemillion() -> None:
    """OneMillion-Bench (400): long-form text answer, weighted-rubric judge.

    ``ground_truth`` carries the rubrics JSON (the judge's target). No files.
    """
    rows: list[dict] = []
    for fp in sorted((_DATASETS / "OneMillion-Bench").glob("*/test.json")):
        with fp.open(encoding="utf-8") as fh:
            entries = json.load(fh)
        for e in entries:
            rubrics_json = json.dumps(e.get("rubrics", []), ensure_ascii=False)
            topics = (e.get("tags", {}) or {}).get("topics", []) or []
            rows.append({
                "task_id": e["id"],
                "task_question": e["question"],
                "ground_truth": rubrics_json,
                "answer_type": "onemillion",
                "category": " > ".join(str(t) for t in topics),
                "rubrics": rubrics_json,
                "language": e.get("language", ""),
                "economic_value": str(e.get("economic_value", "")),
            })
    _write("OneMillion-Bench", rows)


def apex() -> None:
    """APEX (480): stateful multi-app tasks, rubric-judged (``score_apex``).

    Source is ``APEX/tasks_and_rubrics.json`` (from HF ``mercor/apex-agents``).
    Each task ships a per-world snapshot (``world_files_zipped/{world_id}.zip``)
    plus a per-task overlay (``task_files/{task_id}/``); the apex sandbox
    profile's ``prepare`` hook stages them into the agent's worktree at run
    time. ``ground_truth`` = JSON ``{gold_response, rubric}`` (the rubric judge's
    target); ``world_id`` / ``domain`` are threaded so the profile can locate
    and stage the right world. ``answer_type='apex_rubric'``.
    """
    with (_DATASETS / "APEX" / "tasks_and_rubrics.json").open(encoding="utf-8") as fh:
        tasks = json.load(fh)
    rows: list[dict] = []
    for t in tasks:
        ground_truth = json.dumps(
            {"gold_response": t.get("gold_response", ""), "rubric": t.get("rubric", [])},
            ensure_ascii=False,
        )
        rows.append({
            "task_id": t["task_id"],
            "task_question": t["prompt"],
            "ground_truth": ground_truth,
            "answer_type": "apex_rubric",
            "world_id": t.get("world_id", ""),
            "domain": t.get("domain", ""),
            "file_name": "",
        })
    _write("APEX", rows)


_FNS = {
    "officeqa": officeqa,
    "officeqa_full": officeqa_full,
    "gdpval": gdpval,
    "onemillion": onemillion,
    "apex": apex,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(_FNS) if which == "all" else [which]
    for t in targets:
        print(f"standardizing {t} ...")
        _FNS[t]()
    print("done.")
