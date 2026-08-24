#!/usr/bin/env python3
"""
auto_scorer.py — Query 05 自动评分脚本
985高校河南省2020-2024计算机相关专业录取线 (Mode C, 二值评分, 161个评分点)

评分规则:
  - 单值: |claimed - benchmark| <= 1 → 1分
  - 区间 [low, high]: low <= benchmark <= high → 1分
  - 仅评CS专业线, 排除院校投档线和2025年数据

用法:
  # 从已有 result.json 直接评分 (跳过 Stage 1)
  python auto_scorer.py --result-json query_05_result.json --output-dir auto_scores/

  # 从模型原始回答提取+评分
  python auto_scorer.py --models claude=path/to/claude.json glm=path/to/glm.json --output-dir auto_scores/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 硬编码配置（GT inline below）
# ═══════════════════════════════════════════════════════════════

QUERY_ID = 5
QUERY_TEXT = (
    "帮我查询985高校在河南省2020-2025年计算机相关专业的理科投档线，并汇总成一个表格。"
)
SCORING_MODE = "C"
MAX_SCORE = 161
TOLERANCE = 1  # 单值容差 ±1 分

# 161 个已验证 CS 专业线基准值: BENCHMARKS[year][school] = score
BENCHMARKS = {
    2020: {  # 30 个
        "清华大学": 706,
        "北京大学": 703,
        "复旦大学": 696,
        "南京大学": 689,
        "同济大学": 683,
        "北京理工大学": 677,
        "哈尔滨工业大学": 674,
        "武汉大学": 673,
        "东南大学": 671,
        "华中科技大学": 669,
        "华南理工大学": 666,
        "华东师范大学": 660,
        "重庆大学": 658,
        "中山大学": 657,
        "湖南大学": 656,
        "山东大学": 655,
        "吉林大学": 655,
        "四川大学": 665,
        "电子科技大学": 665,
        "国防科技大学": 664,
        "厦门大学": 664,
        "西北工业大学": 663,
        "中国农业大学": 654,
        "大连理工大学": 653,
        "兰州大学": 648,
        "东北大学": 645,
        "中国海洋大学": 644,
        "中央民族大学": 644,
        "西北农林科技大学": 622,
        "中南大学": 611,
    },
    2021: {  # 14 个
        "北京大学": 697,
        "复旦大学": 691,
        "中国人民大学": 678,
        "哈尔滨工业大学": 662,
        "华东师范大学": 653,
        "东南大学": 655,
        "厦门大学": 651,
        "中山大学": 647,
        "山东大学": 641,
        "北京师范大学": 636,
        "东北大学": 635,
        "中央民族大学": 627,
        "中国海洋大学": 631,
        "西北农林科技大学": 601,
    },
    2022: {  # 39 个
        "清华大学": 686,
        "北京大学": 686,
        "南京大学": 666,
        "复旦大学": 668,
        "浙江大学": 669,
        "北京航空航天大学": 668,
        "上海交通大学": 668,
        "中国科学技术大学": 663,
        "北京理工大学": 650,
        "同济大学": 650,
        "西安交通大学": 650,
        "武汉大学": 646,
        "国防科技大学": 646,
        "天津大学": 639,
        "东南大学": 639,
        "华中科技大学": 638,
        "西北工业大学": 636,
        "厦门大学": 633,
        "北京师范大学": 633,
        "中山大学": 632,
        "华东师范大学": 631,
        "电子科技大学": 629,
        "山东大学": 627,
        "四川大学": 627,
        "中南大学": 626,
        "大连理工大学": 625,
        "重庆大学": 622,
        "湖南大学": 622,
        "吉林大学": 621,
        "哈尔滨工业大学": None,  # 无法核实: 原 620 实为威海校区数据; 本部计算机类 2022 河南理科 专业级数据在公开源不可得
        "中国农业大学": 616,
        "兰州大学": 616,
        "中国海洋大学": 613,
        "中央民族大学": 606,
        "西北农林科技大学": 594,
        "南开大学": 587,
        "中国人民大学": 653,
        "华南理工大学": 634,
        "东北大学": 619,
    },
    2023: {  # 39 个
        "北京大学": 701,
        "清华大学": 699,
        "上海交通大学": 691,
        "复旦大学": 688,
        "南京大学": 681,
        "浙江大学": 679,
        "中国科学技术大学": 676,
        "中国人民大学": 672,
        "北京航空航天大学": 667,
        "华中科技大学": 665,
        "北京理工大学": 664,
        "哈尔滨工业大学": 661,
        "西安交通大学": 660,
        "武汉大学": 659,
        "电子科技大学": 655,
        "东南大学": 653,
        "同济大学": 652,
        "中山大学": 650,
        "西北工业大学": 650,
        "北京师范大学": 649,
        "南开大学": 649,
        "华南理工大学": 645,
        "天津大学": 645,
        "大连理工大学": 643,
        "厦门大学": 642,
        "四川大学": 642,
        "山东大学": 641,
        "中南大学": 639,
        "湖南大学": 635,
        "重庆大学": 635,
        "吉林大学": 632,
        "中国农业大学": 629,
        "华东师范大学": 629,
        "兰州大学": 625,
        "中国海洋大学": 625,
        "中央民族大学": 615,
        "西北农林科技大学": 604,
        "东北大学": 632,
        "国防科技大学": 663,  # 更正: 623 实为其他批次/指挥类, 计算机类(河南理科2023)确认为 663, 位次1942
    },
    2024: {  # 39 个
        "清华大学": 708,
        "北京大学": 697,
        "上海交通大学": 693,
        "复旦大学": 684,
        "浙江大学": 681,
        "南京大学": 680,
        "北京航空航天大学": 678,
        "中国科学技术大学": 678,
        "华中科技大学": 678,  # 更正: 676 为专业级 计算机科学与技术, 778 为大类(计算机类·计算机学院)投档线 — 与 query '计算机相关专业' 更匹配
        "武汉大学": 675,
        "中国人民大学": 672,
        "哈尔滨工业大学": 671,
        "北京理工大学": 668,
        "国防科技大学": 666,
        "西安交通大学": 664,
        "东南大学": 661,
        "西北工业大学": 655,
        "电子科技大学": 655,
        "同济大学": 653,
        "中山大学": 652,
        "南开大学": 651,
        "北京师范大学": 650,
        "大连理工大学": 649,
        "华南理工大学": 647,
        "天津大学": 645,
        "厦门大学": 645,
        "四川大学": 644,
        "重庆大学": 644,
        "山东大学": 643,
        "中南大学": 642,
        "湖南大学": 639,
        "华东师范大学": 639,
        "吉林大学": 636,
        "东北大学": 635,
        "中国农业大学": 633,
        "兰州大学": 630,
        "中国海洋大学": 628,
        "中央民族大学": 619,
        "西北农林科技大学": 611,
    },
}

_total = sum(len(v) for v in BENCHMARKS.values())
assert _total == 161, f"Benchmark count mismatch: {_total} != 161"

# 学校名归一化: 别名 → 全称
SCHOOL_ALIASES = {
    "清华": "清华大学",
    "北大": "北京大学",
    "复旦": "复旦大学",
    "南大": "南京大学",
    "浙大": "浙江大学",
    "上交": "上海交通大学",
    "上海交大": "上海交通大学",
    "中科大": "中国科学技术大学",
    "中国科技大学": "中国科学技术大学",
    "北航": "北京航空航天大学",
    "人大": "中国人民大学",
    "北理工": "北京理工大学",
    "北京理工": "北京理工大学",
    "同济": "同济大学",
    "哈工大": "哈尔滨工业大学",
    "武大": "武汉大学",
    "东南": "东南大学",
    "华科": "华中科技大学",
    "华中科大": "华中科技大学",
    "华南理工": "华南理工大学",
    "电子科大": "电子科技大学",
    "川大": "四川大学",
    "国防科大": "国防科技大学",
    "厦大": "厦门大学",
    "西工大": "西北工业大学",
    "华东师大": "华东师范大学",
    "重大": "重庆大学",
    "中山": "中山大学",
    "湖大": "湖南大学",
    "山大": "山东大学",
    "吉大": "吉林大学",
    "中国农大": "中国农业大学",
    "农大": "中国农业大学",
    "大连理工": "大连理工大学",
    "大工": "大连理工大学",
    "兰大": "兰州大学",
    "东北": "东北大学",
    "东北大": "东北大学",
    "海大": "中国海洋大学",
    "中国海大": "中国海洋大学",
    "民大": "中央民族大学",
    "中央民大": "中央民族大学",
    "西农": "西北农林科技大学",
    "西北农林": "西北农林科技大学",
    "中南": "中南大学",
    "南开": "南开大学",
    "北师大": "北京师范大学",
    "天大": "天津大学",
    "西交": "西安交通大学",
    "西安交大": "西安交通大学",
}

ALL_SCHOOLS = set()
for d in BENCHMARKS.values():
    ALL_SCHOOLS.update(d.keys())

CS_MAJOR_KEYWORDS = [
    "计算机",
    "软件",
    "信息",
    "智能",
    "CS",
    "工科试验班",
    "理科试验班",
    "试验班",
    "计科",
    "数据",
    "电气信息",
    "电子信息",
    "电信",
    "AI",
    "计相关",
]


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════


def normalize_school(name: str) -> str:
    """学校名归一化 → 标准全称。"""
    name = re.sub(r"[（(].*?[）)]", "", name).strip()
    if name in ALL_SCHOOLS:
        return name
    if name in SCHOOL_ALIASES:
        return SCHOOL_ALIASES[name]
    for full in ALL_SCHOOLS:
        if name in full or full in name:
            return full
    for alias, full in SCHOOL_ALIASES.items():
        if alias in name:
            return full
    return name


def parse_score(s: str) -> list[int]:
    """解析 claimed_score 字符串 → 整数列表。
    "621" → [621], "~621" → [621], "620-625" → [620,625], "675/674" → [675,674]
    """
    if not s or s.strip() in ("—", "-", ""):
        return []
    s = s.strip()
    s = re.sub(r"^[~约]", "", s)  # 去前缀
    s = s.rstrip("分").strip()  # 去后缀
    if "/" in s:
        return [int(p) for p in s.split("/") if p.strip().lstrip("-").isdigit()]
    m = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", s)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    try:
        return [int(s)]
    except ValueError:
        return []


def is_correct(claimed_str: str, benchmark: int) -> bool:
    if benchmark is None:
        return False  # 基准未核实, 统一判为未通过
    """判定声称值是否正确。单值: ±1容差; 区间: 基准值在区间内。"""
    values = parse_score(claimed_str)
    if not values:
        return False
    if len(values) == 1:
        return abs(values[0] - benchmark) <= TOLERANCE
    # 检测是否为区间 (含 - – —)
    cleaned = re.sub(r"^[~约]", "", claimed_str.strip()).rstrip("分").strip()
    if re.search(r"\d\s*[-–—]\s*\d", cleaned):
        low, high = min(values), max(values)
        return low <= benchmark <= high
    # 斜杠并列: 任一值 ±1
    return any(abs(v - benchmark) <= TOLERANCE for v in values)


def is_cs_major(major: str) -> bool:
    """判断专业是否为CS相关。"""
    return any(kw in major for kw in CS_MAJOR_KEYWORDS)


def load_model_answer(filepath: str, query_id: int, model_name: str) -> str:
    """从模型 JSON 中提取指定 query_id 的回答。"""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if item.get("id") == query_id:
                return item.get("response") or item.get("report_content") or ""
        sys.exit(f"[ERROR] {model_name}: id={query_id} not found in {filepath}")
    if isinstance(data, dict):
        return data.get("report_content") or data.get("response") or ""
    sys.exit(f"[ERROR] {model_name}: unrecognized format in {filepath}")


# ═══════════════════════════════════════════════════════════════
# Stage 1: LLM 提取
# ═══════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """请从以下模型回答中提取所有关于985高校在河南省录取线/投档线的数据声明。

## 提取规则
1. 每条记录包含: school(学校全称), year(年份整数), major(专业名), claimed_score(分数), line_type(类型)
2. line_type 判定:
   - "专业线": 明确标注为计算机/软件/信息类等CS相关专业的录取线
   - "院校投档线": 学校整体最低投档线 (major 填 "院校整体")
3. claimed_score 格式: 纯数字如 "621"; 区间写 "620-625"; 约数写 "~621"
4. 排除2025年数据
5. 学校名用全称 (如 "清华大学" 而非 "清华")

## 输出格式 (纯JSON数组, 无其他文字)
[{{"school":"清华大学","year":2020,"major":"计算机类","claimed_score":"706","line_type":"专业线"}}]

## 模型回答
{answer}"""


def get_llm_client():
    """从 .env 读取配置, 创建 OpenAI 客户端。

    依次尝试 scorers/query_05/.env、scorers/.env、eval/verifiable/.env，
    与 eval/verifiable/README "三处都会被自动加载" 的约定一致（Q15/Q31/Q38
    同款多路径查找）。原版只看脚本同目录 .env，在 .env 实际放在
    eval/verifiable/ 时会以 `OPENROUTER_API_KEY not set` 报错。
    """
    here = Path(__file__).parent  # scorers/query_05/
    for env_path in [here / ".env", here.parent / ".env", here.parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        sys.exit("[ERROR] OPENROUTER_API_KEY not set. Create .env or set env var.")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def extract_claims(model_name: str, answer_text: str, client) -> list:
    """调用 LLM 从模型回答中提取结构化声明。"""
    print(f"  [Stage 1] Extracting: {model_name}...")
    prompt = EXTRACT_PROMPT.format(answer=answer_text[:15000])
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4",
        messages=[
            {"role": "system", "content": "你是数据提取助手。输出纯JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=8000,
    )
    content = resp.choices[0].message.content
    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        print(f"  [WARN] {model_name}: extraction parse failed")
        return []
    try:
        claims = json.loads(m.group(0))
    except json.JSONDecodeError:
        print(f"  [WARN] {model_name}: JSON decode error")
        return []
    print(f"  [Stage 1] {model_name}: {len(claims)} claims extracted")
    return claims


# ═══════════════════════════════════════════════════════════════
# Stage 3: 规则评分 (纯 Python, 无需 LLM)
# ═══════════════════════════════════════════════════════════════


def score_model(model_name: str, claims: list) -> dict:
    """对单个模型的声明列表进行评分, 返回完整结果。"""
    # 过滤: 仅保留 CS 专业线, 排除院校投档线和 2025
    cs_claims = []
    for c in claims:
        lt = c.get("line_type", "")
        if "专业线" not in lt:
            continue
        year = int(c.get("year", 0))
        if year < 2020 or year > 2024:
            continue
        major = c.get("major", "")
        if major and not is_cs_major(major):
            continue
        cs_claims.append(c)

    # 构建 (school, year) → [claims] 查找表
    lookup = {}
    for c in cs_claims:
        school = normalize_school(c.get("school", ""))
        year = int(c.get("year", 0))
        lookup.setdefault((school, year), []).append(c)

    per_dimension = []
    total_score = 0
    answered = 0
    year_scores = {y: 0 for y in BENCHMARKS}
    dim_id = 0

    for year in sorted(BENCHMARKS):
        for school, benchmark in BENCHMARKS[year].items():
            dim_id += 1
            key = (school, year)
            if key in lookup:
                answered += 1
                # 同校同年多条声明 → 取最接近基准值的
                best_str, best_diff = None, float("inf")
                for c in lookup[key]:
                    s = c.get("claimed_score", "")
                    vals = parse_score(s)
                    if vals and benchmark is not None:
                        diff = min(abs(v - benchmark) for v in vals)
                        if diff < best_diff:
                            best_diff, best_str = diff, s
                    elif vals and benchmark is None:
                        # benchmark 未核实 → 记录最后一个声明值，is_correct 会判 False
                        best_str = s
                correct = best_str is not None and is_correct(best_str, benchmark)
                score = 1 if correct else 0
                total_score += score
                year_scores[year] += score
                reason = (
                    f"匹配 (声称{best_str}, 基准{benchmark})"
                    if correct
                    else f"偏差 (声称{best_str}, 基准{benchmark}, 差{best_diff}分)"
                )
                per_dimension.append(
                    {
                        "id": f"D{dim_id}",
                        "school": school,
                        "year": year,
                        "score": score,
                        "claimed_value": best_str,
                        "benchmark_value": benchmark,
                        "reason": reason,
                    }
                )
            else:
                per_dimension.append(
                    {
                        "id": f"D{dim_id}",
                        "school": school,
                        "year": year,
                        "score": 0,
                        "claimed_value": None,
                        "benchmark_value": benchmark,
                        "reason": "未作答",
                    }
                )

    acc = round(total_score / answered * 100, 1) if answered else 0.0
    return {
        "model": model_name,
        "total_score": total_score,
        "max_score": MAX_SCORE,
        "score_rate": round(total_score / MAX_SCORE, 4),
        "dimensions_answered": answered,
        "accuracy": acc,
        "year_scores": year_scores,
        "year_max": {y: len(s) for y, s in BENCHMARKS.items()},
        "per_dimension": per_dimension,
        "cs_claims_count": len(cs_claims),
        "total_claims_count": len(claims),
    }


# ═══════════════════════════════════════════════════════════════
# 输出生成
# ═══════════════════════════════════════════════════════════════


def generate_ranking_report(results: list, output_dir: str):
    ranked = sorted(results, key=lambda r: (-r["total_score"], r["model"]))
    L = []
    L.append("# 自动评分排名报告: 985高校河南省计算机相关专业录取线\n")
    L.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"> 评分模式: Mode {SCORING_MODE} (二值评分制)")
    L.append(f"> 理论满分: {MAX_SCORE} 分 (161个已验证CS专业线基准值)")
    L.append(f"> 判定标准: 单值 ±{TOLERANCE}分=正确; 区间含基准值=正确\n")

    # 排名表
    L.append("## 最终排名\n")
    L.append("| 排名 | 模型 | 总分 | 得分率 | CS线作答数 | 准确率 |")
    L.append("|:---:|------|:---:|:-----:|:---------:|:-----:|")
    rank, prev = 0, None
    for i, r in enumerate(ranked):
        if r["total_score"] != prev:
            rank, prev = i + 1, r["total_score"]
        acc = f"{r['accuracy']}%" if r["dimensions_answered"] else "—"
        L.append(
            f"| {rank} | {r['model']} | {r['total_score']}/{MAX_SCORE} | "
            f"{r['score_rate'] * 100:.1f}% | {r['dimensions_answered']}条 | {acc} |"
        )

    # 分年得分
    L.append("\n## 各模型分年得分\n")
    years = sorted(BENCHMARKS)
    hdr = (
        "| 模型 |"
        + "".join(f" {y}(/{len(BENCHMARKS[y])}) |" for y in years)
        + " 合计 |"
    )
    sep = "|------|" + ":---------:|" * len(years) + ":----------:|"
    L += [hdr, sep]
    for r in ranked:
        row = f"| {r['model']} |"
        for y in years:
            row += f" {r['year_scores'][y]} |"
        row += f" **{r['total_score']}** |"
        L.append(row)

    # 各模型逐维度详情
    for r in ranked:
        L.append(f"\n### {r['model']} ({r['total_score']}/{MAX_SCORE})\n")
        for year in years:
            dims = [
                d
                for d in r["per_dimension"]
                if d["year"] == year and d["claimed_value"] is not None
            ]
            if not dims:
                continue
            L.append(f"**{year}年:**\n")
            L.append("| 学校 | 基准 | 声称 | 判定 |")
            L.append("|------|:---:|:---:|:---:|")
            for d in dims:
                mark = "\u2705" if d["score"] else "\u274c"
                L.append(
                    f"| {d['school']} | {d['benchmark_value']} | {d['claimed_value']} | {mark} |"
                )
            L.append("")

    path = os.path.join(output_dir, "ranking_report.md")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] Ranking report: {path}")


def generate_scores_json(results: list, output_dir: str):
    ranked = sorted(results, key=lambda r: (-r["total_score"], r["model"]))
    # 处理并列排名
    ranking_list = []
    rank, prev = 0, None
    for i, r in enumerate(ranked):
        if r["total_score"] != prev:
            rank, prev = i + 1, r["total_score"]
        ranking_list.append(
            {
                "rank": rank,
                "model": r["model"],
                "score": r["total_score"],
                "rate": f"{r['score_rate'] * 100:.1f}%",
            }
        )

    output = {
        "query_id": QUERY_ID,
        "scoring_mode": SCORING_MODE,
        "max_score": MAX_SCORE,
        "tolerance": TOLERANCE,
        "generated_at": datetime.now().isoformat(),
        "benchmarks": {
            f"{school}_{year}": score
            for year in sorted(BENCHMARKS)
            for school, score in BENCHMARKS[year].items()
        },
        "results": {
            r["model"]: {
                "total_score": r["total_score"],
                "max_score": MAX_SCORE,
                "score_rate": r["score_rate"],
                "dimensions_answered": r["dimensions_answered"],
                "accuracy": r["accuracy"],
                "year_scores": {str(k): v for k, v in r["year_scores"].items()},
                "per_dimension": r["per_dimension"],
            }
            for r in ranked
        },
        "ranking": ranking_list,
    }
    path = os.path.join(output_dir, "scores.json")
    Path(path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Scores JSON: {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Query 05 自动评分: 985高校河南省CS专业线 (Mode C, 161评分点)"
    )
    parser.add_argument("--query-id", type=int, default=QUERY_ID)
    parser.add_argument(
        "--models", nargs="+", metavar="name=path", help="模型名=JSON文件路径, 可传多个"
    )
    parser.add_argument("--result-json", help="已有 result.json 路径 (跳过 Stage 1)")
    parser.add_argument("--output-dir", default="auto_scores", help="输出目录")
    args = parser.parse_args()

    if not args.models and not args.result_json:
        parser.error("Must provide --models or --result-json")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    if args.result_json:
        print(f"[*] Loading: {args.result_json}")
        with open(args.result_json, encoding="utf-8") as f:
            data = json.load(f)
        for model_name, info in data.get("models", {}).items():
            claims = info.get("data", [])
            print(f"  [Stage 3] Scoring {model_name} ({len(claims)} claims)...")
            r = score_model(model_name, claims)
            all_results.append(r)
            print(
                f"  → {model_name}: {r['total_score']}/{MAX_SCORE} "
                f"({r['dimensions_answered']} answered, {r['accuracy']}% acc)"
            )
    else:
        client = get_llm_client()
        for arg in args.models:
            if "=" not in arg:
                sys.exit(f"[ERROR] Invalid: {arg}. Expected name=path")
            name, path = arg.split("=", 1)
            answer = load_model_answer(path, args.query_id, name)
            claims = extract_claims(name, answer, client)
            r = score_model(name, claims)
            all_results.append(r)
            print(
                f"  → {name}: {r['total_score']}/{MAX_SCORE} "
                f"({r['dimensions_answered']} answered, {r['accuracy']}% acc)"
            )

    print("\n[*] Generating reports...")
    generate_ranking_report(all_results, args.output_dir)
    generate_scores_json(all_results, args.output_dir)

    print(f"\n{'=' * 50}")
    print(f"评分完成! 共评估 {len(all_results)} 个模型")
    for i, r in enumerate(sorted(all_results, key=lambda x: -x["total_score"]), 1):
        print(
            f"  #{i} {r['model']}: {r['total_score']}/{MAX_SCORE} ({r['score_rate'] * 100:.1f}%)"
        )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
