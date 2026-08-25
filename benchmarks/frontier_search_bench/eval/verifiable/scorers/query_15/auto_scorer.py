#!/usr/bin/env python3
"""
Auto-scorer — Query 15 (涅瓦色西森 WCF Best Cat 猫舍)

评分模式: Mode C（二值评分制），仅 1 个维度：猫舍名称是否回答正确。
标准答案: LucianAurora

用法:
  # 从模型回答文件评分
  python auto_scorer.py --query-id 15 \
    --models claude=path/to/claude.json glm=path/to/chatglm.json \
    --output-dir query_15/auto_scores/

  # 从已有 result.json 评分
  python auto_scorer.py --query-id 15 \
    --result-json path/to/result.json \
    --output-dir query_15/auto_scores/
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════
# 硬编码配置
# ═══════════════════════════════════════════════════════════════════════════

QUERY_ID = 15
QUERY_TEXT = "截至2026年1月1日，哪个猫舍繁育的涅瓦色西森获得WCF Best Cat次数最多？"
SCORING_MODE = "C"
MAX_SCORE = 1.0
CORRECT_ANSWER = "LucianAurora"

# ═══════════════════════════════════════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_MAX_TOKENS = 4000


def load_env():
    config = {}
    for p in [SCRIPT_DIR / ".env", SCRIPT_DIR.parent / ".env", SCRIPT_DIR.parent.parent / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip().strip("'\"")
            if config:
                return config
    return config


def get_client():
    env = load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY") or env.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL") or env.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    if not api_key:
        sys.exit("[ERROR] No API key. Set OPENROUTER_API_KEY in .env or environment.")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/ApodexAI/FrontierAgent",
            "X-Title": "frontier-search-bench",
        },
    )


def call_llm(client, model, system_prompt, user_message, max_tokens):
    print(f"  Calling {model} ...")
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    tok_in = usage.prompt_tokens if usage else "?"
    tok_out = usage.completion_tokens if usage else "?"
    print(f"  Done in {time.time() - start:.1f}s | in={tok_in} out={tok_out}")
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 输入加载
# ═══════════════════════════════════════════════════════════════════════════


def load_model_answer(filepath, query_id, model_name):
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(raw)
    if isinstance(data, list):
        for item in data:
            if item.get("id") == query_id:
                return item.get("report_content") or item.get("response") or ""
        sys.exit(f"[ERROR] {model_name}: id={query_id} not found in {filepath}")
    if isinstance(data, dict):
        if data.get("id") == query_id or "id" not in data:
            return data.get("report_content") or data.get("response") or ""
        sys.exit(
            f"[ERROR] {model_name}: single-object id={data.get('id')} != {query_id}"
        )
    sys.exit(f"[ERROR] Unsupported format in {filepath}")


def parse_model_args(model_args, query_id):
    models = {}
    for arg in model_args:
        if "=" not in arg:
            sys.exit(f"[ERROR] Invalid: '{arg}'. Expected: name=path")
        name, path = arg.split("=", 1)
        name, path = name.strip(), path.strip()
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Not found: {path}")
        print(f"  Loading {name} ...")
        resp = load_model_answer(path, query_id, name)
        models[name] = resp
        print(f"    {len(resp)} chars")
    return models


def extract_json(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    m2 = re.search(r"[\[{]", candidate)
    if m2:
        try:
            return json.loads(candidate[m2.start() :])
        except json.JSONDecodeError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: 提取（仅提取猫舍名称）
# ═══════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = r"""你是一个事实提取专家。

查询是："{query}"

请从每个模型的回答中提取其给出的**最终答案猫舍名称**。

规则：
- 如果模型明确给出了一个猫舍名称作为答案 → 提取该名称
- 如果模型使用代码/别名（如"C1猫舍"），请在回答中找到对应的实际猫舍名称（通常在映射表中）
- 如果模型说"无法确定"、"数据不足"等，即使提到了某个猫舍名称但未作为最终答案 → 填 null
- 只关心模型的最终结论，不关心中间讨论

## 模型回答

{model_answers}

## 输出

```json
{{
  "models": {{
    "model_name": {{
      "core_answer": "猫舍名称 或 null"
    }}
  }}
}}
```

只输出 JSON。"""


def stage_extract(client, models, llm_model, output_path, max_tokens):
    print("\n" + "=" * 60)
    print("STAGE 1: Extract core answer")
    print("=" * 60)
    model_block = {}
    for name, text in models.items():
        model_block[name] = text  # No truncation — full text needed to resolve aliases
    input_str = json.dumps(model_block, ensure_ascii=False, indent=2)
    user_msg = EXTRACT_PROMPT.replace("{query}", QUERY_TEXT).replace(
        "{model_answers}", input_str
    )
    raw = call_llm(
        client, llm_model, "你是一个事实提取专家。只输出 JSON。", user_msg, max_tokens
    )
    result = extract_json(raw)
    if result is None:
        debug = output_path.with_suffix(".extract_raw.txt")
        debug.write_text(raw, encoding="utf-8")
        sys.exit(f"[ERROR] JSON parse failed. Raw saved to {debug}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: 评分（纯规则，无需 LLM）
# ═══════════════════════════════════════════════════════════════════════════


def score_answer(claimed):
    """Score a single model's answer. Returns (score, reason)."""
    if claimed is None:
        return 0.0, "❌ 未给出确定答案"
    claimed_lower = claimed.lower().strip()
    if "lucianaurora" in claimed_lower:
        return 1.0, "✅ 正确"
    return 0.0, f"❌ 错误答案: {claimed}"


def stage_score(result_data, output_dir):
    print("\n" + "=" * 60)
    print("STAGE 3: Score (rule-based, no LLM needed)")
    print("=" * 60)
    models_data = result_data.get("models", {})
    results = {}
    ranking = []

    for model_name, mdata in models_data.items():
        claimed = mdata.get("core_answer")
        score, reason = score_answer(claimed)
        results[model_name] = {
            "total_score": score,
            "dimensions_answered": 1 if claimed is not None else 0,
            "per_dimension": [
                {
                    "id": "D1",
                    "name": "核心答案：猫舍名称",
                    "score": score,
                    "claimed_value": claimed,
                    "benchmark_value": CORRECT_ANSWER,
                    "reason": reason,
                }
            ],
        }
        ranking.append(
            {
                "model": model_name,
                "score": score,
                "rate": f"{score / MAX_SCORE * 100:.0f}%",
            }
        )
        print(f"  {model_name}: {claimed!r} → {reason}")

    ranking.sort(key=lambda x: (-x["score"], x["model"]))
    for i, r in enumerate(ranking):
        r["rank"] = i + 1
    # Adjust tied ranks
    for i in range(1, len(ranking)):
        if ranking[i]["score"] == ranking[i - 1]["score"]:
            ranking[i]["rank"] = ranking[i - 1]["rank"]

    final = {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "max_score": MAX_SCORE,
        "total_dimensions": 1,
        "correct_answer": CORRECT_ANSWER,
        "results": results,
        "ranking": ranking,
    }

    # Save JSON
    json_path = output_dir / "scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {json_path}")

    # Save Markdown
    md_path = output_dir / "ranking_report.md"
    md_path.write_text(generate_markdown(final), encoding="utf-8")
    print(f"  Saved: {md_path}")
    return final


def generate_markdown(data):
    lines = [
        "# 模型排名报告：涅瓦色西森 WCF Best Cat 猫舍评估（自动评分）\n",
        f"> 评分模式: Mode {data['scoring_mode']}（二值评分制）",
        "> 评分维度: 猫舍名称是否正确（1 个维度）",
        f"> 标准答案: **{data['correct_answer']}**",
        f"> 理论满分: {data['max_score']}\n",
        "---\n",
        "## 评分结果\n",
        "| 排名 | 模型 | 模型回答 | 得分 | 判定 |",
        "|:---:|------|---------|:---:|------|",
    ]
    for r in data["ranking"]:
        m = data["results"][r["model"]]
        d = m["per_dimension"][0]
        cv = d["claimed_value"] if d["claimed_value"] else "（未给出确定答案）"
        lines.append(
            f"| {r['rank']} | {r['model']} | {cv} | {d['score']:.0f} | {d['reason']} |"
        )

    correct = sum(1 for r in data["ranking"] if r["score"] == 1.0)
    total = len(data["ranking"])
    lines.append(f"\n**{correct}/{total} 个模型回答正确**\n")
    lines.append("---\n")
    lines.append("## 注意事项\n")
    lines.append("1. 评分模式: Mode C，1 个维度，满分 1.0")
    lines.append(
        f"2. 标准答案 {data['correct_answer']} 基于 wcf-bestcat.de 2019-2025 Adult Top10 逐年核验"
    )
    lines.append("3. 此排名仅评估模型在本特定查询上的表现")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Auto Scorer — Query 15")
    parser.add_argument("--query-id", type=int, default=QUERY_ID)
    parser.add_argument("--models", nargs="*", help="name=path pairs")
    parser.add_argument("--bench", help="Benchmark JSON path (unused)")
    parser.add_argument("--result-json", help="Pre-extracted result.json path")
    parser.add_argument("--output-dir", default="auto_scores/")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_id = args.query_id

    if args.result_json:
        print(f"Loading pre-extracted result: {args.result_json}")
        with open(args.result_json, encoding="utf-8") as f:
            result_data = json.load(f)
        stage_score(result_data, output_dir)
    elif args.models:
        print(f"Query {query_id}: {QUERY_TEXT}")
        models = parse_model_args(args.models, query_id)
        client = get_client()
        result_path = output_dir / "result.json"
        result_data = stage_extract(
            client, models, args.llm_model, result_path, args.max_tokens
        )
        stage_score(result_data, output_dir)
    else:
        parser.print_help()
        sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
