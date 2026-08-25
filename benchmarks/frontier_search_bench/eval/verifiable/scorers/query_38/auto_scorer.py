#!/usr/bin/env python3
"""
Auto-scorer — Query 38 (旅行商最短路径 TSP)

评分维度、基准值和规则全部硬编码在本脚本中。
Ground-truth verification summary 嵌入 prompt 作为评分参考。

用法：
  python auto_scorer.py \
    --models claude=path/to/Claude.json glm=path/to/ChatGlm.json \
    --output-dir query_38/auto_scores/

  # 跳过提取，直接用已有 result.json 评分
  python auto_scorer.py \
    --result-json path/to/result.json \
    --output-dir query_38/auto_scores/
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
# 硬编码配置（ground truth inline below）
# ═══════════════════════════════════════════════════════════════════════════

QUERY_ID = 38
QUERY_TEXT = (
    "一个旅行商要从北京出发，依次访问上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨这7个城市后返回北京。"
    "利用各城市间的实际飞行距离，求出最短总路径及其距离是多少公里？"
)
DOMAIN = "运筹学/组合优化"

SCORING_MODE = (
    "A2+B+C"  # Cov-1(Mode A2) + Sub-1(Mode C) + Num-1/Num-2(Mode B 自定义误差带)
)
MAX_SCORE = 9  # D1(2) + D2(1) + D3(3) + D4(3) = 9

# 基准最优 TSP 路径（由 Haversine 穷举 5040 条路径确认）
OPTIMAL_ROUTE = [
    "北京",
    "西安",
    "乌鲁木齐",
    "拉萨",
    "成都",
    "广州",
    "上海",
    "哈尔滨",
    "北京",
]
OPTIMAL_DISTANCE_KM = 11033.7  # 大圆距离（机场坐标，R=6371km）

SCORING_RULES = """评分模式: Mode A2 变体（Cov-1）+ Mode C（Sub-1）+ Mode B 自定义误差带（Num-1, Num-2）组合。
共 4 个评分维度（D1-D4），理论满分 = 2 + 1 + 3 + 3 = 9 分。

### D1 — Cov-1 城市覆盖完整性（满分 2 分）

| 得分 | 判定标准 |
|:---:|---------|
| 2 分 | 路径包含全部7个目标城市（上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨），起终点均为北京，每城市恰访问一次 |
| 1 分 | 覆盖 5-6 个目标城市，或起终点不完整（如单程未返回） |
| 0 分 | 覆盖 <5 个目标城市、未构建路径、重复访问或遗漏多城，或为空回答 |

### D2 — Sub-1 距离类型正确性（满分 1 分）

| 得分 | 判定标准 |
|:---:|---------|
| 1 分 | 明确使用"飞行距离"或"大圆距离/Haversine 距离"（框架定义为同一正确主体） |
| 0.5 分 | 使用替代距离（公路/铁路/直线欧氏）但明确说明原因（如"因无飞行数据"） |
| 0 分 | 使用替代距离且无说明，或未声明距离类型 |

> **注意**：大圆距离 × 统一系数（如 ×1.2）的基础距离类型仍视为大圆，D2 可以拿 1 分；
> 但系数若无权威依据且导致总距离严重偏离，将在 D4（Num-2）中体现扣分。

### D3 — Num-1 路径最优性（满分 3 分）

以模型所声称路径的"真实大圆距离"（按基准距离矩阵重算）与最优解 11033.7km 的差距评分：

| 得分 | 判定标准 |
|:---:|---------|
| 3 分 | 路径确为最优解（或其等价逆序） — 真实距离 = 11033.7km |
| 2 分 | 真实距离与最优解差距 ≤ 500km |
| 1 分 | 差距 500–1500km |
| 0 分 | 差距 > 1500km 或未给出路径 |

### D4 — Num-2 总距离声明准确性（满分 3 分）

以模型声称的总距离与"模型所声称路径的真实大圆距离"的偏差评分（考察算术一致性）：

| 得分 | 判定标准 |
|:---:|---------|
| 3 分 | 偏差 ≤ 200km（若声称的是区间且基准值落在区间内，也视为精确档） |
| 2 分 | 偏差 200–500km |
| 1 分 | 偏差 500–1500km |
| 0 分 | 偏差 > 1500km 或未声称总距离 |

### 特殊规则

1. **若模型未给出有效路径**：D1-D4 全为 0。
2. **路径偏差但答案自洽**：模型用自定义距离矩阵得出次优路径，但其声称总距离与该路径真实大圆距离吻合 → D3 按偏差计分，D4 仍可拿满分。
3. **总距离含缩放系数**：模型给出大圆 × 系数（如 ×1.2）后的值作为最终答案 → D4 按声称值与最优值偏差扣分。
4. **D4 声称值选取**：当模型给出多个总距离值时（如大圆 11058 + ×1.2 后 13270），取模型最终答案/结论中呈现的数值评分。
5. **区间声明**：如 "11000–11065km"，若基准 11034 落在区间内，视为精确档（D4 = 3）。"""

# 评分维度（4 个）及基准值
DIMENSIONS = [
    {
        "id": "D1",
        "name": "城市覆盖完整性 (Cov-1)",
        "benchmark": "7 个目标城市（上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨）+ 北京起终点，共 8 节点",
        "max_score": 2,
        "source": "穷举确认（路径结构规范）",
        "verification_ref": "覆盖度统计表",
        "extract_field": "route, covered_city_count",
    },
    {
        "id": "D2",
        "name": "距离类型正确性 (Sub-1)",
        "benchmark": "飞行距离（航线距离 / 大圆距离 / Haversine 均为框架定义的正确主体）",
        "max_score": 1,
        "source": "评分框架定义",
        "verification_ref": "距离类型判定表",
        "extract_field": "distance_type",
    },
    {
        "id": "D3",
        "name": "路径最优性 (Num-1)",
        "benchmark": "北京→西安→乌鲁木齐→拉萨→成都→广州→上海→哈尔滨→北京，总距离 11033.7km（或等价逆序）",
        "max_score": 3,
        "source": "Haversine 穷举 5040 条路径",
        "verification_ref": "#1 (基准路径) / #2 (Gemini 次优路径)",
        "extract_field": "route, segment_distances",
    },
    {
        "id": "D4",
        "name": "总距离声明准确性 (Num-2)",
        "benchmark": "模型路径的大圆真实距离（最优路径 = 11033.7km；Gemini 路径 = 11134.1km）",
        "max_score": 3,
        "source": "Haversine 计算",
        "verification_ref": "#4–#12 (各模型声称总距离)",
        "extract_field": "claimed_total_distance",
    },
]

# Ground-truth verification summary
VERIFICATION_SUMMARY = r"""
## 基准距离矩阵（km，大圆距离；机场坐标 PEK/PVG/CAN/CTU/XIY/HRB/URC/LXA）

|  | 北京 | 上海 | 广州 | 成都 | 西安 | 哈尔滨 | 乌鲁木齐 | 拉萨 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 北京     | —      | 1099.5 | 1881.0 | 1557.7 | 933.5  | 999.2  | 2430.0 | 2622.4 |
| 上海     | 1099.5 | —      | 1203.1 | 1704.2 | 1273.1 | 1655.4 | 3312.3 | 2965.9 |
| 广州     | 1881.0 | 1203.1 | —      | 1221.2 | 1306.0 | 2733.9 | 3277.8 | 2321.1 |
| 成都     | 1557.7 | 1704.2 | 1221.2 | —      | 624.4  | 2556.3 | 2073.2 | 1263.2 |
| 西安     | 933.5  | 1273.1 | 1306.0 | 624.4  | —      | 1932.5 | 2105.2 | 1776.3 |
| 哈尔滨   | 999.2  | 1655.4 | 2733.9 | 2556.3 | 1932.5 | —      | 3037.3 | 3567.2 |
| 乌鲁木齐 | 2430.0 | 3312.3 | 3277.8 | 2073.2 | 2105.2 | 3037.3 | —      | 1652.8 |
| 拉萨     | 2622.4 | 2965.9 | 2321.1 | 1263.2 | 1776.3 | 3567.2 | 1652.8 | —      |

## 基准最优路径

**北京 → 西安 → 乌鲁木齐 → 拉萨 → 成都 → 广州 → 上海 → 哈尔滨 → 北京**

| 航段 | 基准距离(km) |
|------|:---:|
| 北京 → 西安       | 933.5 |
| 西安 → 乌鲁木齐   | 2105.2 |
| 乌鲁木齐 → 拉萨   | 1652.8 |
| 拉萨 → 成都       | 1263.2 |
| 成都 → 广州       | 1221.2 |
| 广州 → 上海       | 1203.1 |
| 上海 → 哈尔滨     | 1655.4 |
| 哈尔滨 → 北京     | 999.2 |
| **合计**          | **11033.7** |

逆序路径（北京→哈尔滨→上海→广州→成都→拉萨→乌鲁木齐→西安→北京）距离相同，为等价对称解。

## 次优路径示例

| 路径 | 总距离(km) | 与最优差距 |
|------|:---:|:---:|
| BJ→XA→UR→LS→CD→GZ→SH→HRB→BJ（含逆序） — 最优 | 11033.7 | 0 |
| BJ→HRB→SH→GZ→XA→CD→LS→UR→BJ（含逆序） | 11134.1 | +100.4 |
| BJ→SH→GZ→XA→CD→LS→UR→HRB→BJ（含逆序） | 11185.5 | +151.9 |
"""

# ═══════════════════════════════════════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_MAX_TOKENS = 16000


def load_env():
    """从 .env 读取配置。"""
    config = {}
    for p in [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
        SCRIPT_DIR.parent.parent / ".env",
    ]:
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
    """创建 OpenAI 客户端。"""
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
    """调用 LLM。"""
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
    print(
        f"  Done in {time.time() - start:.1f}s | "
        f"in={usage.prompt_tokens if usage else '?'} "
        f"out={usage.completion_tokens if usage else '?'}"
    )
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 输入加载
# ═══════════════════════════════════════════════════════════════════════════


def load_model_answer(filepath, query_id, model_name):
    """从 JSON 中提取指定 query 的回答。"""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if item.get("id") == query_id:
                return item.get("response") or item.get("report_content") or ""
        sys.exit(f"[ERROR] {model_name}: id={query_id} not found in {filepath}")
    if isinstance(data, dict):
        return data.get("report_content") or data.get("response") or ""
    sys.exit(f"[ERROR] Unsupported format in {filepath}")


def parse_model_args(model_args, query_id):
    """解析 name=path 参数。"""
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: 提取
# ═══════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = r"""你是一个事实提取专家。请从以下模型回答中，针对 TSP 最短路径任务提取各模型的结构化声明。

## 任务背景

Query: 一个旅行商要从北京出发，依次访问上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨这7个城市后返回北京。利用各城市间的实际飞行距离，求出最短总路径及其距离是多少公里？

## 提取字段（严格对齐下列 5 个字段，对应评分维度 D1-D4）

对每个模型，按以下字段提取：

1. **route**（对应 D1/D3）：完整路径的城市访问顺序数组，含北京起终点。例如 ["北京","西安","乌鲁木齐","拉萨","成都","广州","上海","哈尔滨","北京"]。若模型未给出路径则为 null。
2. **claimed_total_distance**（对应 D4）：模型声称的总距离（字符串，含单位；可以是数值或区间，如 "11112" 或 "11000–11065"）。若未声称则为 null。
3. **segment_distances**（参考 D3/D4）：各航段距离数组，每项含 `from`/`to`/`distance_km`。若模型未列出则为 null。
4. **distance_type**（对应 D2）：枚举值之一：`flight` / `great_circle` / `road` / `rail` / `straight_line` / `flight_with_coefficient`（大圆×系数） / `unspecified`。若模型声称"实际飞行距离"但实际可能为大圆，以其**自述**为准（实际判定由评分阶段处理）。
5. **covered_city_count**（对应 D1）：路径中包含的目标城市数（满分=7，不含北京起终点）。若未构建路径则为 0。

## 额外规则

- 若模型回答为空或完全未提及任务，所有字段为 null（covered_city_count = 0），note 记录"空回答"或"未求解 TSP"。
- 若模型给出多个总距离（如大圆值 + ×1.2 后的值），`claimed_total_distance` 取其**最终答案**中呈现的数值，并在 note 中注明其他候选值。
- 保留原始精度和措辞，不要修正或四舍五入。

## 输入

{model_answers}

## 输出

只输出 JSON，格式如下：
```json
{{
  "query": "一个旅行商要从北京出发，依次访问上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨这7个城市后返回北京。利用各城市间的实际飞行距离，求出最短总路径及其距离是多少公里？",
  "domain": "运筹学/组合优化",
  "extraction_mode": "answer_factual",
  "extraction_fields": ["route","claimed_total_distance","segment_distances","distance_type","covered_city_count"],
  "models": {{
    "model_name": [
      {{"field": "route", "value": ["北京","西安","..."], "note": "..."}},
      {{"field": "claimed_total_distance", "value": "11112", "unit": "km", "note": null}},
      {{"field": "segment_distances", "value": [{{"from":"北京","to":"西安","distance_km":933.5}}, ...], "note": null}},
      {{"field": "distance_type", "value": "flight", "note": "..."}},
      {{"field": "covered_city_count", "value": 7}}
    ]
  }}
}}
```

只输出 JSON，不要其他文字。"""


def stage_extract(
    client, models, model_name_str, output_path, max_tokens=DEFAULT_MAX_TOKENS
):
    """Stage 1: 提取结构化声明。"""
    print("\n" + "=" * 60)
    print("STAGE 1: Extract")
    print("=" * 60)

    input_block = json.dumps(
        {"query": QUERY_TEXT, "domain": DOMAIN, **models},
        ensure_ascii=False,
        indent=2,
    )

    user_msg = EXTRACT_PROMPT.replace("{model_answers}", input_block)
    system = "你是一个事实提取专家。请严格按照指示只输出 JSON。"

    raw = call_llm(client, model_name_str, system, user_msg, max_tokens)

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
# Stage 3: 评分
# ═══════════════════════════════════════════════════════════════════════════


def build_benchmark_table():
    """构建基准值表的 Markdown 文本。"""
    lines = [
        "| 维度ID | 维度名称 | 基准值 | 满分 | 基准来源 | 对应验证声明# |",
        "|:---:|---------|---------|:---:|---------|:---:|",
    ]
    for d in DIMENSIONS:
        lines.append(
            f"| {d['id']} | {d['name']} | {d['benchmark']} | {d['max_score']} | {d['source']} | {d['verification_ref']} |"
        )
    return "\n".join(lines)


SCORE_PROMPT_TEMPLATE = r"""你是一个评分机器。请严格按照以下规则对每个模型的 4 个维度（D1-D4）打分，**不要修改任何维度或基准值**。

## 评分规则

{scoring_rules}

## 评分维度与基准值（4 个维度，满分 9 分）

{benchmark_table}

## 验证报告摘要（作为评分参考 — 基准距离矩阵与各模型声明判定）

{verification_summary}

## 待评分的模型声明

{result_json}

## 你的任务

对每个模型，逐一打 D1-D4 四个维度的分，并计算总分与得分率。

### D1（Cov-1，满分 2）
- 路径覆盖 7 个目标城市（上海、广州、成都、西安、哈尔滨、乌鲁木齐、拉萨）+ 北京起终点 → 2 分
- 覆盖 5-6 个 → 1 分
- <5 个 / 未构建路径 / 空回答 → 0 分

### D2（Sub-1，满分 1）
- 使用飞行距离或大圆距离/Haversine（含 `flight`, `great_circle`, `flight_with_coefficient`） → 1 分
- 使用替代距离（公路/铁路/直线）并明确说明原因 → 0.5 分
- 使用替代距离无说明 / 未声明 / 未作答 → 0 分

### D3（Num-1，满分 3）
判断模型所声称的路径是否为最优：
- 路径 = BJ→XA→UR→LS→CD→GZ→SH→HRB→BJ（或等价逆序）→ 3 分（真实距离 11033.7km）
- Gemini 路径 BJ→HRB→SH→GZ→XA→CD→LS→UR→BJ → 2 分（真实距离 11134.1km，偏差 +100.4km ≤500km）
- 其他路径：按该路径在基准距离矩阵下的真实总距离与 11033.7km 的差距分档（≤500=2, 500-1500=1, >1500=0）
- 未给出路径 → 0 分

### D4（Num-2，满分 3）
**D4 评分基准 = 该模型声称路径的真实大圆距离**（非最优解 11033.7）：
- 若路径为最优（11033.7），模型声称与 11033.7 的偏差
- 若为 Gemini 路径（11134.1），与 11134.1 的偏差
- 若为其他路径，与对应真实距离的偏差
- 偏差 ≤ 200km（或声称区间覆盖基准）→ 3 分
- 偏差 200-500km → 2 分
- 偏差 500-1500km → 1 分
- 偏差 > 1500km / 未声称总距离 → 0 分

**关键判例提示**（与手动评分一致）：
- 声称 ~11034km（偏差 ≤200km）: D4 = 3
- 声称 11034 ± 500km: D4 = 2
- 声称 11034 ± 500-1500km: D4 = 1
- 声称偏差 > 1500km 或未给出总距离: D4 = 0
- 模型未给出有效路径（未作答）: D1-D4 全 0

## 输出格式

先输出 Markdown 排名报告，格式如下：

```
# 模型排名报告：旅行商最短路径评估（自动评分版）

> 评分标准: Mode A2 + C + B 组合，4 维度，满分 9 分（D1=2, D2=1, D3=3, D4=3）

## 排名方法
（简述评分规则与基准来源）

## 一、全局基准
（4 维度基准表）

## 二、各模型详情

### 1. 模型名
| 维度ID | 声明内容 | 模型声称值 | 基准值 | 得分 | 说明 |
|:---:|---------|-----------|:---:|:---:|------|
| D1 | ... | ... | 7/7 + 北京起终点 | X | ... |
| D2 | ... | ... | 飞行/大圆距离 | X | ... |
| D3 | ... | ... | 最优路径 11033.7km | X | ... |
| D4 | ... | ... | 该路径大圆值 | X | ... |

- 作答维度: X/4
- 完全正确: X 个
- **总分: X/9 (X.X%)**

## 三、最终模型排名
| 排名 | 模型 | D1(2) | D2(1) | D3(3) | D4(3) | 总分 | 得分率 | 作答率 |

### 排名解读
（每个模型1-2句点评，特别说明并列情况与精度差异）
```

然后另起一行输出 `===SCORES_JSON===`，再输出：
```json
{{
  "query_id": 44,
  "scoring_mode": "A2+B+C",
  "max_score": 9,
  "dimensions": ["D1_城市覆盖","D2_距离类型","D3_路径最优性","D4_总距离准确性"],
  "benchmarks": {{
    "D1": {{"value": "7城市+北京起终点", "max_score": 2}},
    "D2": {{"value": "飞行/大圆距离", "max_score": 1}},
    "D3": {{"value": "北京→西安→乌鲁木齐→拉萨→成都→广州→上海→哈尔滨→北京 (11033.7km)", "max_score": 3}},
    "D4": {{"value": "该模型路径的大圆真实距离", "max_score": 3}}
  }},
  "results": {{
    "model_name": {{
      "total_score": <int>,
      "max_score": 9,
      "score_rate": <float>,
      "dimensions_answered": <int>,
      "per_dimension": [
        {{
          "id": "D1",
          "name": "城市覆盖完整性",
          "score": <0-2>,
          "claimed_value": "...",
          "benchmark_value": "7城市+北京起终点",
          "reason": "..."
        }},
        {{
          "id": "D2",
          "name": "距离类型正确性",
          "score": <0/0.5/1>,
          "claimed_value": "...",
          "benchmark_value": "飞行/大圆距离",
          "reason": "..."
        }},
        {{
          "id": "D3",
          "name": "路径最优性",
          "score": <0-3>,
          "claimed_value": "BJ→...",
          "benchmark_value": "最优路径 11033.7km",
          "path_real_distance_km": <float>,
          "reason": "..."
        }},
        {{
          "id": "D4",
          "name": "总距离声明准确性",
          "score": <0-3>,
          "claimed_value": "...",
          "benchmark_value": <float>,
          "deviation_km": <float>,
          "reason": "..."
        }}
      ]
    }}
  }},
  "ranking": [
    {{"rank": 1, "model": "...", "score": <int>, "rate": "XX.X%"}}
  ]
}}
```"""


def stage_score(
    client, result_json, model_name_str, output_dir, max_tokens=DEFAULT_MAX_TOKENS
):
    """Stage 3: 评分。"""
    print("\n" + "=" * 60)
    print("STAGE 3: Score")
    print("=" * 60)

    user_msg = (
        SCORE_PROMPT_TEMPLATE.replace("{scoring_rules}", SCORING_RULES)
        .replace("{benchmark_table}", build_benchmark_table())
        .replace("{verification_summary}", VERIFICATION_SUMMARY)
        .replace("{result_json}", json.dumps(result_json, ensure_ascii=False, indent=2))
    )

    system = (
        "你是一个评分机器。请严格按照给定的维度、基准值和评分规则打分，"
        "不要修改任何维度或基准值。在 Markdown 报告后以 ===SCORES_JSON=== 分隔输出 JSON。"
    )

    raw = call_llm(client, model_name_str, system, user_msg, max_tokens)

    ranking_md, scores = split_score_output(raw)

    ranking_path = output_dir / "ranking_report.md"
    ranking_path.write_text(ranking_md, encoding="utf-8")
    print(f"  Saved: {ranking_path}")

    scores_path = output_dir / "scores.json"
    if scores:
        scores["query_id"] = QUERY_ID
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {scores_path}")
    else:
        fallback = scores_path.with_suffix(".raw.txt")
        fallback.write_text(raw, encoding="utf-8")
        print(f"  [WARN] JSON parse failed. Raw saved to {fallback}")

    return scores or {}


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════


def _try_repair(text):
    """用 json_repair 兜底，处理 LLM 偶发的未转义内引号 / 悬挂逗号等。
    返回解析结果或 None；导入失败时也返回 None（保持原有行为）。
    """
    try:
        import json_repair
    except ImportError:
        return None
    try:
        return json_repair.loads(text)
    except Exception:
        return None


def extract_json(text):
    """从文本中提取 JSON。

    依次尝试：直接 json.loads → markdown fence 内层 → 首 `{` 到末 `}` 子串。
    每个候选解析失败时，再用 json_repair 兜底——LLM 偶尔会在字符串内写未
    转义的双引号（如 \"称为\"方案一\"的逆时针路径\"），标准 json 解析直接
    抛 JSONDecodeError；json_repair 能把这类问题修好。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            repaired = _try_repair(m)
            if repaired is not None:
                return repaired
            continue
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            repaired = _try_repair(text[first : last + 1])
            if repaired is not None:
                return repaired
    return None


def split_score_output(text):
    """分离 Markdown 和 JSON。"""
    sep = "===SCORES_JSON==="
    if sep in text:
        md, js = text.split(sep, 1)
        return md.strip(), extract_json(js.strip())
    idx = text.rfind("```json")
    if idx != -1:
        scores = extract_json(text[idx:])
        if scores:
            return text[:idx].strip(), scores
    return text.strip(), None


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Auto Scorer — Query 38 (旅行商最短路径 TSP)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="name=path pairs (e.g. claude=Claude.json glm=ChatGlm.json)",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Skip extraction, use existing result.json",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="Benchmark JSON (not needed, query is hardcoded)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "auto_scores"),
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=QUERY_ID,
        help=f"Query ID for model answer lookup (default: {QUERY_ID})",
    )
    parser.add_argument("--llm-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Auto Scorer — Query 38 (TSP Shortest Path)")
    print(f"  Output: {output_dir}")
    print(f"  Model:  {args.llm_model}")

    client = get_client()
    result_path = output_dir / "result.json"

    # Stage 1 or load
    if args.result_json:
        print(f"\n[SKIP] Stage 1: Using {args.result_json}")
        with open(args.result_json, encoding="utf-8") as f:
            result_json = json.load(f)
    elif args.models:
        models = parse_model_args(args.models, args.query_id)
        result_json = stage_extract(
            client, models, args.llm_model, result_path, args.max_tokens
        )
    else:
        sys.exit("[ERROR] Provide --models or --result-json")

    # Stage 3
    scores = stage_score(
        client, result_json, args.llm_model, output_dir, args.max_tokens
    )

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for f in sorted(output_dir.iterdir()):
        if not f.name.startswith("."):
            print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

    if scores and "ranking" in scores:
        print("\n  Ranking:")
        for e in scores["ranking"]:
            print(
                f"    #{e['rank']} {e['model']}: {e.get('score', '?')} ({e.get('rate', '?')})"
            )


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════════
# 评估模型（从原始 JSON 文件中自动提取并评分）：
#
#   python query_38/auto_scorer.py \
#     --models \
#       <model-a>=/path/to/<model-a>.json \
#       <model-b>=/path/to/<model-b>.json \
#     --output-dir query_38/auto_scores/
#
# 跳过提取，直接从已有 result.json 评分：
#
#   python query_38/auto_scorer.py \
#     --result-json /path/to/result.json \
#     --output-dir query_38/auto_scores/
