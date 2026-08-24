"""WideSearch judge — port of ByteDance-Seed/WideSearch official scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from io import StringIO
from urllib.parse import urlparse

import pandas as pd

from benchmarks.public.judges._common import Verdict

logger = logging.getLogger(__name__)

_WIDESEARCH_DEFAULT_JUDGE_MODEL = "openai/gpt-4.1"  # full row+item structural match


# ── Part 1: Structural F1 scorer (was scorer/widesearch.py) ─────────────

# ---------------------------------------------------------------------------
# Sync judge LLM client
#   Reads JUDGE_* env vars (with OPENAI_* fallback) so it shares the same
#   endpoint as benchmarks.public.judges. Kept sync intentionally — wrapped via
#   asyncio.to_thread() in the async caller.
# ---------------------------------------------------------------------------
_JUDGE_BASE_URL: str | None = None
_JUDGE_API_KEY: str | None = None
_JUDGE_MODEL: str | None = None


def _init_judge_llm(model_name: str = _WIDESEARCH_DEFAULT_JUDGE_MODEL) -> None:
    """Init judge LLM endpoint.

    Reads ``JUDGE_BASE_URL`` / ``JUDGE_API_KEY`` env (with ``OPENAI_*``
    fallback). The judge model is pinned to ``model_name``.
    """
    global _JUDGE_BASE_URL, _JUDGE_API_KEY, _JUDGE_MODEL
    base = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    _JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    _JUDGE_MODEL = model_name
    if base and not base.endswith("/chat/completions"):
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        base += "/chat/completions"
    _JUDGE_BASE_URL = base


def _llm_completion(prompt: str, max_retries: int = 3) -> str | None:
    if _JUDGE_BASE_URL is None:
        _init_judge_llm()
    # Bind the module global to a local: a checker cannot see that
    # _init_judge_llm() populated it, and the judge cannot run without a URL.
    base_url = _JUDGE_BASE_URL
    if base_url is None:
        return None
    import time

    import requests
    headers = {"Content-Type": "application/json"}
    if _JUDGE_API_KEY:
        headers["Authorization"] = f"Bearer {_JUDGE_API_KEY}"
    payload = {
        "model": _JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return None
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def norm_column(col: str) -> str:
    return col.strip().lower().replace(" ", "")


def parse_markdown_json(completion: str) -> dict | None:
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", completion, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Preprocess (lower/strip/extract-number/parse-date)
# ---------------------------------------------------------------------------
def preprocess_norm_str(content) -> str:
    return str(content).lower().strip().replace(" ", "").replace("*", "")


def preprocess_extract_number(content) -> str:
    nums = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", str(content).replace(",", ""))
    return nums[0] if nums else "NULL"


def preprocess_norm_date(content) -> str:
    try:
        import dateparser
        d = dateparser.parse(str(content), settings={"PREFER_DAY_OF_MONTH": "first"})
        return d.strftime("%Y-%m-%d") if d else str(content)
    except Exception:
        return str(content)


PREPROCESS_REGISTRY = {
    "norm_str": preprocess_norm_str,
    "extract_number": preprocess_extract_number,
    "norm_date": preprocess_norm_date,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_exact_match(response, target, criterion=None):
    return (1.0, "match") if response.lower() == target.lower() else (0.0, "no match")


def metric_url_match(response, target, criterion=None):
    pat = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    rd = {urlparse(u).netloc for u in pat.findall(response)}
    td = {urlparse(u).netloc for u in pat.findall(target)}
    return (1.0, "match") if rd == td else (0.0, "no match")


def metric_in_match(response, target, criterion=None):
    return (1.0, "match") if response in target else (0.0, "no match")


def metric_number_near(response, target, criterion=0.05):
    def _parse(s):
        if "%" in s:
            try:
                return float(s.replace("%", "")) / 100.0
            except (ValueError, TypeError):
                return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
    r, t = _parse(response), _parse(target)
    if r is None or t is None:
        if r is None and t is None and response == target:
            return (1.0, "equal")
        return (0.0, "not convertible")
    if abs(r - t) <= abs(t) * criterion:
        return (1.0, f"near within {criterion*100}%")
    return (0.0, f"not near: {r} vs {t}")


def metric_date_near(response, target, criterion=None):
    try:
        import dateparser
        rd = dateparser.parse(response, settings={"PREFER_DAY_OF_MONTH": "first"})
        td = dateparser.parse(target, settings={"PREFER_DAY_OF_MONTH": "first"})
        if rd is None or td is None:
            if rd is None and td is None:
                return (1.0, "both unparseable")
            return (0.0, "parse error")
        if abs((rd - td).days) <= 31:
            return (1.0, "near")
        return (0.0, f"too far: {abs((rd - td).days)} days")
    except Exception:
        return (0.0, "parse error")


METRIC_REGISTRY = {
    "exact_match": metric_exact_match,
    "url_match": metric_url_match,
    "in_match": metric_in_match,
    "number_near": metric_number_near,
    "date_near": metric_date_near,
    "llm_judge": None,  # dispatched separately in evaluate_single
}


# ---------------------------------------------------------------------------
# LLM-based column / primary-key alignment
# ---------------------------------------------------------------------------
PRIMARY_KEY_PREPROCESS_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""


def primary_key_preprocess(response_list: list, reference_list: list) -> dict:
    prompt = PRIMARY_KEY_PREPROCESS_PROMPT.format(response=response_list, reference=reference_list)
    result = _llm_completion(prompt)
    return (parse_markdown_json(result) or {}) if result else {}


def llm_judge_column(response_list: list[str], target_list: list[str], criterion: str) -> tuple:
    response_dict = {f"idx_{i}": {"response": r, "target": t}
                     for i, (r, t) in enumerate(zip(response_list, target_list, strict=False))}
    prompt = EVAL_COLUMN_PROMPT.format(criterion=criterion, response=response_dict)
    result = _llm_completion(prompt)
    if not result:
        return [0] * len(response_list), ["judge failed"] * len(response_list)
    score_dict = parse_markdown_json(result)
    if not score_dict:
        return [0] * len(response_list), ["parse error"] * len(response_list)
    scores = [score_dict.get(f"idx_{i}", 0) for i in range(len(response_list))]
    if len(scores) != len(response_list):
        return [0] * len(response_list), ["length mismatch"] * len(response_list)
    return scores, [result] * len(response_list)


# ---------------------------------------------------------------------------
# Markdown table → DataFrame
# ---------------------------------------------------------------------------
def extract_dataframe(response_text: str) -> pd.DataFrame | None:
    md = re.findall(r"```markdown(.*?)```", response_text, re.DOTALL)
    if not md:
        pipe_pos = [m.start() for m in re.finditer(r"\|", response_text)]
        if len(pipe_pos) >= 4:
            start = response_text.rfind("\n", 0, pipe_pos[0])
            start = 0 if start == -1 else start
            end = response_text.find("\n", pipe_pos[-1])
            end = len(response_text) if end == -1 else end
            md = re.findall(r"((?:\|.*\n?)+)", response_text[start:end])
    if not md:
        return None
    s = md[0].strip()
    lines = s.split("\n")
    lines[0] = lines[0].replace(" ", "").lower()
    lines = [ln.strip() for ln in lines]
    new_lines = []
    for ln in lines:
        if set(ln.strip()).issubset(set("|- :")) or "|" not in ln:
            continue
        new_lines.append("|".join([_l.strip() for _l in ln.split("|")]))
    s = "\n".join(new_lines)
    try:
        df = pd.read_csv(StringIO(s), sep="|")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gold answers loader — HuggingFace download.
# Only used to pre-bake gold tables into the JSONL. Runtime scoring reads
# the embedded ``gold_table`` directly from each row.
# ---------------------------------------------------------------------------
_gold_cache: dict = {}


def load_gold_answers() -> dict:
    global _gold_cache
    if _gold_cache:
        return _gold_cache
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        repo_id = "ByteDance-Seed/WideSearch"
        snapshot_download(repo_id=repo_id, repo_type="dataset")
        ds = load_dataset(repo_id)["full"]
        # The datasets stubs type a row as a broad union (Mapping / list /
        # dict), so string keys otherwise read as slice indices.
        for raw_item in ds:
            if not isinstance(raw_item, dict):
                continue
            item = raw_item
            instance_id = item["instance_id"]
            ev = item["evaluation"]
            ev = json.loads(ev) if isinstance(ev, str) else ev
            required = ev["required"]
            cache_path = try_to_load_from_cache(
                repo_id=repo_id,
                filename=f"widesearch_gold/{instance_id}.csv",
                repo_type="dataset",
            )
            if cache_path is None:
                continue
            try:
                df = pd.read_csv(cache_path)
                df.columns = [norm_column(c.strip()) for c in df.columns]
                if any(c not in df.columns for c in required):
                    continue
                _gold_cache[instance_id] = df[required]
            except Exception:
                continue
    except Exception as e:
        print(f"[WideSearch] load_gold_answers failed: {e}")
    return _gold_cache


# ---------------------------------------------------------------------------
# Result + scoring routine
# ---------------------------------------------------------------------------
@dataclass
class WideSearchResult:
    task_id: str = ""
    instance_id: str = ""
    score: float = 0.0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    msg: str = ""


def evaluate_single(
    task_id: str, instance_id: str, response_text: str,
    eval_spec: dict, answer_df: pd.DataFrame,
) -> WideSearchResult:
    """Single-task end-to-end scoring. Verbatim from source (modulo style)."""
    result = WideSearchResult(task_id=task_id, instance_id=instance_id)
    try:
        required_columns = eval_spec["required"]
        unique_columns = eval_spec["unique_columns"]

        response_df = extract_dataframe(response_text)
        if response_df is None:
            result.msg = "Failed to extract table from response"
            return result

        response_df.columns = [norm_column(c) for c in response_df.columns]
        gold_df = answer_df.copy()
        gold_df.columns = [norm_column(c) for c in gold_df.columns]

        # LLM column alignment
        if set(required_columns) != set(response_df.columns):
            col_map = primary_key_preprocess(response_df.columns.tolist(), required_columns)
            if col_map:
                response_df.rename(columns=col_map, inplace=True)
        if set(required_columns) != set(response_df.columns):
            result.msg = f"Column mismatch: expected {required_columns}, got {response_df.columns.tolist()}"
            return result

        # type alignment
        for col in required_columns:
            try:
                at, rt = gold_df[col].dtype, response_df[col].dtype
                # pandas dtype intentionally compares with `==` to Python types;
                # ruff E721 flags this but it's the idiomatic pandas pattern.
                if (rt == float and at == int) or (rt == int and at == float):  # noqa: E721
                    if rt == int:  # noqa: E721
                        response_df[col] = response_df[col].astype(float)
                    else:
                        gold_df[col] = gold_df[col].astype(float)
            except Exception:
                pass
            gold_df[col] = gold_df[col].astype(str)
            response_df[col] = response_df[col].astype(str)

        response_df.drop_duplicates(subset=unique_columns, inplace=True)
        gold_df.drop_duplicates(subset=unique_columns, inplace=True)

        # primary-key LLM alignment for unique columns
        for col in unique_columns:
            item = eval_spec["eval_pipeline"].get(col)
            if item is None:
                continue
            metrics = item.get("metric", [])
            if "llm_judge" in metrics or "exact_match" in metrics:
                pk_map = primary_key_preprocess(response_df[col].tolist(), gold_df[col].tolist())
                if pk_map:
                    response_df[col] = response_df[col].apply(
                        lambda x, _pk=pk_map: _pk.get(x, x)
                    )

        # preprocess each column
        for col, item in eval_spec["eval_pipeline"].items():
            for pp_name in item.get("preprocess", []):
                pp_func = PREPROCESS_REGISTRY.get(pp_name)
                if pp_func:
                    response_df[col] = response_df[col].apply(pp_func)
                    gold_df[col] = gold_df[col].apply(pp_func)

        # quick exact check
        temp_score = 0.0
        if gold_df.shape == response_df.shape:
            gt_sorted = gold_df.sort_values(by=required_columns).reset_index(drop=True)
            pred_sorted = response_df.sort_values(by=required_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                temp_score = 1.0
        result.score = temp_score

        # inner-join + per-column scoring
        df_inner = pd.merge(gold_df, response_df, on=unique_columns, how="inner",
                            suffixes=("_query", "_response"))
        score_df = pd.DataFrame(index=df_inner.index)

        for col in required_columns:
            if col in unique_columns:
                score_df[col] = 1.0
                continue
            item = eval_spec["eval_pipeline"][col]
            metric_names = item.get("metric", [])
            criterion = item.get("criterion")
            for metric_name in metric_names:
                if metric_name == "llm_judge":
                    scores, _ = llm_judge_column(
                        df_inner[col + "_response"].tolist(),
                        df_inner[col + "_query"].tolist(),
                        criterion,
                    )
                    score_df[f"{col}_{metric_name}"] = scores
                else:
                    metric_func = METRIC_REGISTRY.get(metric_name)
                    if metric_func is None:
                        continue
                    score_df[f"{col}_{metric_name}"] = df_inner.apply(
                        lambda x, _mf=metric_func, _c=criterion, _col=col: _mf(
                            x[_col + "_response"], x[_col + "_query"], _c
                        )[0],
                        axis=1,
                    )

        col_scores = pd.DataFrame(index=df_inner.index)
        for col in required_columns:
            if col in unique_columns:
                col_scores[col] = 1.0
            else:
                metric_cols = [c for c in score_df.columns if c.startswith(f"{col}_")]
                col_scores[col] = score_df[metric_cols].max(axis=1) if metric_cols else 0.0

        row_scores = col_scores.min(axis=1)
        tp_by_row = row_scores.sum()
        tp_by_item = col_scores.sum().sum()

        n_pred_rows = len(response_df)
        n_gt_rows = len(gold_df)
        n_pred_items = n_pred_rows * len(required_columns)
        n_gt_items = n_gt_rows * len(required_columns)

        result.precision_by_row = tp_by_row / n_pred_rows if n_pred_rows else 0.0
        result.recall_by_row = tp_by_row / n_gt_rows if n_gt_rows else 0.0
        result.precision_by_item = tp_by_item / n_pred_items if n_pred_items else 0.0
        result.recall_by_item = tp_by_item / n_gt_items if n_gt_items else 0.0

        def _f1(p, r):
            return (2 * p * r / (p + r)) if (p + r) > 1e-9 else 0.0
        result.f1_by_row = _f1(result.precision_by_row, result.recall_by_row)
        result.f1_by_item = _f1(result.precision_by_item, result.recall_by_item)

        if (result.precision_by_item == result.recall_by_item == result.f1_by_item == 1.0
                and result.precision_by_row == result.recall_by_row == result.f1_by_row == 1.0):
            result.score = 1.0

        result.msg = f"pred_rows={n_pred_rows}, gt_rows={n_gt_rows}, inner={len(df_inner)}"
    except Exception:
        import traceback
        result.msg = f"Error: {traceback.format_exc()}"
    return result


def evaluate_from_eval_spec(
    response_text: str, eval_spec: dict,
) -> WideSearchResult:
    """Entry point for benchmarks.public.judges.score_widesearch.

    Reads ``eval_spec["gold_table"]`` (list[dict]) embedded in the JSONL.
    Scoring is offline at runtime — if ``gold_table`` is missing or empty,
    returns a ``WideSearchResult`` with a diagnostic ``.msg``.
    """
    iid = eval_spec.get("_instance_id", "") if isinstance(eval_spec, dict) else ""
    gold_table = eval_spec.get("gold_table") if isinstance(eval_spec, dict) else None
    if not gold_table:
        result = WideSearchResult(instance_id=iid)
        result.msg = f"No gold_table embedded for instance_id={iid!r}"
        return result
    answer_df = pd.DataFrame(gold_table)
    if answer_df.empty:
        result = WideSearchResult(instance_id=iid)
        result.msg = f"Empty gold_table for instance_id={iid!r}"
        return result
    return evaluate_single(
        task_id="", instance_id=iid,
        response_text=response_text, eval_spec=eval_spec, answer_df=answer_df,
    )


# ── Part 2: Async judge wrapper ──────

async def score_widesearch(
    question: str, target: str, predicted: str,
) -> tuple[Verdict, float | None]:
    """Score a WideSearch trial using the official ByteDance-Seed scoring.

    The scorer is sync (pandas + requests-based judge calls), so we
    offload it to a thread via ``asyncio.to_thread``.

    Args:
        question: not used by widesearch scoring; kept for signature parity.
        target: JSON string of the ``eval_spec`` (with
            ``unique_columns`` / ``required`` / ``eval_pipeline`` plus
            ``_instance_id`` and ``gold_table``).
        predicted: model's raw response (expected to contain a markdown table).

    Returns:
        ``(verdict, f1_by_item)``:
        * ``verdict`` is ``CORRECT`` iff the official ``score`` is 1.0
          (full row+item match — the strict "fully correct" definition).
        * ``f1_by_item ∈ [0, 1]`` is the soft metric for partial credit and
          best-of-N analysis (stored as ``rubric_score`` in ``result.json``).
    """
    if not predicted or not predicted.strip():
        return "INCORRECT", 0.0
    # ground_truth schema errors are dataset bugs, not model mistakes — return
    # NOT_ATTEMPTED so they don't quietly inflate the incorrect-answer count.
    try:
        eval_spec = json.loads(target) if isinstance(target, str) else target
    except (json.JSONDecodeError, TypeError):
        logger.warning("WideSearch judge: ground_truth is not valid JSON eval_spec")
        return "NOT_ATTEMPTED", None
    if not isinstance(eval_spec, dict) or "eval_pipeline" not in eval_spec:
        logger.warning("WideSearch judge: eval_spec missing required keys")
        return "NOT_ATTEMPTED", None

    # Run sync widesearch scorer on a worker thread so this coroutine doesn't
    # block the eval event loop on pandas + sync HTTP judge calls.
    result = await asyncio.to_thread(evaluate_from_eval_spec, predicted, eval_spec)
    logger.info(
        "WideSearch judge: instance=%s f1_row=%.3f f1_item=%.3f score=%.1f msg=%s",
        result.instance_id, result.f1_by_row, result.f1_by_item, result.score,
        (result.msg[:80] if result.msg else ""),
    )
    verdict: Verdict = "CORRECT" if result.score == 1.0 else "INCORRECT"
    return verdict, float(result.f1_by_item)


__all__ = [
    "EVAL_COLUMN_PROMPT",
    "METRIC_REGISTRY",
    "PREPROCESS_REGISTRY",
    "PRIMARY_KEY_PREPROCESS_PROMPT",
    "WideSearchResult",
    "evaluate_from_eval_spec",
    "evaluate_single",
    "extract_dataframe",
    "llm_judge_column",
    "load_gold_answers",
    "metric_date_near",
    "metric_exact_match",
    "metric_in_match",
    "metric_number_near",
    "metric_url_match",
    "norm_column",
    "parse_markdown_json",
    "preprocess_extract_number",
    "preprocess_norm_date",
    "preprocess_norm_str",
    "primary_key_preprocess",
    "score_widesearch",
]
