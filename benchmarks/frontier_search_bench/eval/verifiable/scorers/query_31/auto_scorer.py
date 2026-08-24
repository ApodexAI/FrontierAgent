#!/usr/bin/env python3
"""
auto_scorer.py — Query 31 自动评分脚本
正方体蜡块铅芯几何形状与物理运动评估
评分模式: Mode B + Mode C 组合
  D1 (Q1 形状): Mode B 精度分，0/2/4/6 四档，满分 6
  D2 (Q2 初态): Mode C 二值分，0/2
  D3 (Q3 滑动): Mode C 三值分，0/1/2
  D4 (Q4 旋转): Mode C 二值分，0/2
  理论满分 = 12

用法:
  # 从模型原始 JSON 提取并评分
  python auto_scorer.py \\
    --query-id 31 \\
    --models claude=path/to/claude.json glm=path/to/glm.json \\
    --output-dir query_31/auto_scores/

  # 跳过提取，直接从已有 result.json 评分
  python auto_scorer.py \\
    --result-json path/to/result.json \\
    --output-dir query_31/auto_scores/
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 硬编码评分配置（ground truth inline below）
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_ID = 31
QUERY_TEXT = (
    "请你在脑海中构建一个边长为10厘米的完美正方体纯蜡块，将其平放在水平桌面上。"
    "第一步：用一个直径为2厘米的钻头，从正方体的'左前上'顶点，笔直地贯穿钻向"
    "'右后下'顶点，打通一个圆柱形的斜向隧道。第二步：保持蜡块在水平桌面上不动，"
    "向这个贯穿的隧道里注入液态铅。由于底部有开口，你需要先用一层薄膜封住底部的洞口。"
    "然后注入液态铅，直到铅液的体积恰好占整个隧道总容积的50%。等待铅液完全冷却凝固。"
    "第三步：将包含铅芯的蜡块整体转移到一个倾斜角为30度的绝对光滑斜坡上。"
    "第四步：瞬间将环境温度升高到蜡的熔点以上，正方体蜡块在一瞬间完全气化消失，"
    "只剩下那块凝固的铅块由于重力作用直接落在30度的斜坡上。"
    "请问：这块剩下的铅块，其精确的三维几何形状是什么样子的？"
    "当蜡块瞬间消失，铅块直接接触30度斜坡的那个瞬间，它会发生怎样的物理运动？"
)
SCORING_MODE = "B+C 组合"
MAX_SCORE = 12
SNAPSHOT_DATE = "2026-04-21"

# 4个评分维度
DIMENSIONS = [
    {
        "id": "D1",
        "framework_code": "Q1",
        "name": "铅块三维几何形状",
        "benchmark_value": (
            "斜截圆柱（顶面椭圆半轴 √3/1 cm；底端三棱锥尖端于 (10,10,0) 顶点；"
            "体积 5π√3−√6 ≈ 24.76 cm³；液面 z=5）"
        ),
        "source": "解析推导 + 数值积分 + 蒙特卡洛复核",
        "verification_claim": "#4, #6, #9, #11, #13, #14",
        "extract_field": "Q1_shape",
        "max_score": 6,
        "rule": (
            "6分=完整斜截圆柱+椭圆顶面+尺寸合理范围内（≤10%相对误差）; "
            "4分=识别斜截圆柱+椭圆顶面，但尺寸/底端细节有明显误差; "
            "2分=描述为某种圆柱体，但未识别顶面为椭圆（如标准圆柱两端圆形）; "
            "0分=形状完全错误（如半圆柱体水平切）"
        ),
    },
    {
        "id": "D2",
        "framework_code": "Q2",
        "name": "铅块初始接触状态",
        "benchmark_value": (
            "底端（三棱锥尖端）位于 (10,10,0) 顶点紧贴斜面，整体倾斜悬空"
        ),
        "source": "数值验证（铅块 z 范围 [0,5]，底端在 (10,10,0) 附近）",
        "verification_claim": "#15",
        "extract_field": "Q2_initial_state",
        "max_score": 2,
        "rule": (
            "2分=识别底端贴斜面+整体倾斜悬空; "
            "0分=未识别底端贴斜面，或错误描述为自由落体、弧面朝下等"
        ),
    },
    {
        "id": "D3",
        "framework_code": "Q3",
        "name": "斜面滑动",
        "benchmark_value": "无摩擦 → 必然滑动，a = g sin30° = g/2 ≈ 4.9 m/s²",
        "source": "牛顿第二定律",
        "verification_claim": "#17",
        "extract_field": "Q3_sliding",
        "max_score": 2,
        "rule": (
            "2分=明确指出无摩擦必然滑动+给出加速度 g/2 或等价定量表述; "
            '1分=仅定性说"会滑动"，无定量分析; '
            "0分=错误或未提及"
        ),
    },
    {
        "id": "D4",
        "framework_code": "Q4",
        "name": "旋转/翻倒",
        "benchmark_value": (
            "形状不对称，质心 (7.24,7.24,2.70) 不在接触点法向上方，"
            "重力产生力矩 → 滑动同时发生旋转/翻倒"
        ),
        "source": "力矩分析 + 数值质心",
        "verification_claim": "#18",
        "extract_field": "Q4_rotation",
        "max_score": 2,
        "rule": (
            "2分=正确分析旋转/翻倒及原因（形状不对称/质心偏离接触点/重力力矩）; "
            '0分=判定不旋转（常见错误：混淆"滚动需摩擦"与"翻倒不需摩擦"），或未提及'
        ),
    },
]

# Ground-truth verification summary
VERIFICATION_SUMMARY = """\
## Ground-truth verification (snapshot 2026-04-21)

### 几何参数基准（解析推导 + 数值积分 + 蒙特卡洛复核）
- 空间对角线长度: 10√3 ≈ 17.32 cm
- 隧道轴与铅直方向夹角: arccos(1/√3) ≈ 54.74°
- 液面高度（50%体积）: **z = 5 cm**（对称性证明 + 数值积分）
- 顶面椭圆半长轴: **√3 ≈ 1.732 cm**
- 顶面椭圆半短轴: **1 cm**
- 隧道精确体积: 10π√3 − 2√6 ≈ 49.52 cm³（顶点裁剪 2√6）
- 铅块精确体积: 5π√3 − √6 ≈ 24.76 cm³
- 铅块简化体积 5π√3 ≈ 27.21 cm³（偏差 +9.9%，合理范围内）
- 铅块质心: (7.24, 7.24, 2.70)

### 关键形状声明
- ✅ 正确：斜截圆柱（oblique truncated cylinder），顶面椭圆，底端三棱锥尖端
- ❌ 错误：半圆柱体（水平切通过轴线，将 50% 体积误解为 50% 截面积）
- ❌ 错误：标准圆柱（两端圆形，未识别顶面椭圆）
- ❌ 常见错误：液面高度 z = 5√3 ≈ 8.66 cm（源于水平截面积误算为 π 而非 π√3）
- ❌ 常见错误：两端都是椭圆的对称斜截圆柱（忽略了底端三棱锥裁切）

### 关键物理声明
- ✅ 底端贴斜面：铅块 z 范围 [0,5]，底端位于 (10,10,0) 顶点附近
- ❌ 错误：铅块从蜡块内部自由落体撞击斜面（实际底端已在斜面上）
- ✅ 加速度 a = g sin30° = g/2 ≈ 4.9 m/s²
- ✅ 旋转/翻倒：质心 (7.24, 7.24, 2.70) 投影远在支撑域外，重力对接触点产生翻倒力矩
- ❌ 错误：绝对光滑 → 无摩擦力矩 → 不旋转（混淆"滚动需摩擦"与"翻倒不需摩擦"）

### 全部 20 条声明判定统计
- 完全正确: 12 条 (60%)
- 错误: 4 条 (20%): z=5√3 液面高度 / 半圆柱体 / 标准圆柱 / 纯平动不旋转
- 部分正确/不够精确: 4 条 (20%): 简化体积 10π√3 和 5π√3（合理范围内）+ 自由落体描述
"""

# 评分规则文本
SCORING_RULES_TEXT = """\
## 评分规则（Mode B + Mode C 组合）

### 维度 D1（Q1 铅块三维几何形状，Mode B，6 分档 6/4/2/0）
- **6 分**：正确描述为斜截圆柱 + 顶面椭圆 + 尺寸参数在合理范围内（≤10%相对误差）
  - 范例：给出椭圆半轴 √3/1 cm 或等价 1.73/1 cm；体积 24.76 或 27.21 cm³（简化值亦可）；
    底端描述为三棱锥/角锥/脚印/三平面截切（精确体积 10π√3−2√6 ≈ 49.52 是满分范例；
    用简化体积 5√3π ≈ 27.21（相对误差 9.9%）但底端/椭圆参数正确亦给 6 分）
- **4 分**：正确识别斜截圆柱 + 椭圆顶面，但尺寸或底端细节有明显误差
  - 范例：尺寸大错（轴长 15cm 而非 ≈8.66cm）; 液面高度 z=5√3 而非 z=5;
    认为两端都是椭圆（对称斜截圆柱，忽略底端三棱锥）; 仅结论提及椭圆而未深入
- **2 分**：描述为某种圆柱体，但未识别顶面为椭圆
  - 范例：标准圆柱（两端圆形）; 完整圆柱斜向放置两端圆形
- **0 分**：形状完全错误
  - 范例：半圆柱体（水平切通过轴线，把 50% 体积误解为 50% 截面积）

### 维度 D2（Q2 初态，Mode C 二值 0/2）
- **2 分**：识别铅块底端贴斜面 + 整体倾斜悬空
  - 范例：trihedral beak tip 点接触 / 脚印区域接触 / 角锥状尖端接触 / 椭圆面接触（底端形状判定错但已在 D1 扣分）
- **0 分**：未识别或错误描述
  - 范例：铅块自由落体撞击斜面; 半圆柱弧面朝下落向斜坡; 笼统"初速度为零受力作用"

### 维度 D3（Q3 滑动，Mode C 三值 0/1/2）
- **2 分**：明确指出无摩擦必然滑动 + 给出加速度 g/2 或 g sin30° 或 4.9 m/s² 的定量表述
- **1 分**：仅定性说"会滑动"，未给定量加速度
- **0 分**：错误或未提及

### 维度 D4（Q4 旋转，Mode C 二值 0/2）
- **2 分**：正确分析旋转/翻倒 + 原因（形状不对称/质心偏离接触点/重力对接触点产生力矩/支撑多边形判据）
- **0 分**：判定不旋转或未提及
  - 常见错误：认为"绝对光滑无摩擦 → 无力矩 → 不旋转"。此为混淆"滚动需摩擦"与
    "翻倒不需摩擦"。正确区分：摩擦只对滚动和静止约束有关；重力对非对称接触点产生的
    翻倒力矩与摩擦无关，必然存在

### 总分计算
- 总分 = D1 + D2 + D3 + D4（四维度独立，直接加和，不加权）
- 满分 = 6 + 2 + 2 + 2 = 12

### 语义等价判定指引（Mode C 8.6）
- 判断时关注**事实本质**而非具体措辞
- "a = g/2" / "g sin30°" / "4.9 m/s²" / "4.905 m/s²" 均视为等价
- "翻倒" / "翻滚" / "旋转" / "倾覆" / "tipping" / "toppling" 在讨论非对称物体时视为等价
- "三棱锥" / "trihedral beak" / "角锥状" / "脚印" / "三面截切" 均视为正确的底端描述
- "斜截圆柱" / "oblique truncated cylinder" / "cylindrical wedge" 视为等价

### 额外信息处理（Mode C 8.7）
- 模型给出的超出 4 个维度的信息，记录在 scores.json 的 extra_claims 字段，不计分
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """\
你是一个事实提取专家。从模型回答中提取结构化信息，用于后续评分。

## 原始查询
{query}

## 需要提取的字段（严格对齐评分维度 D1-D4）

| 字段名 | 对应维度 | 说明 |
|--------|---------|------|
| Q1_shape | D1/Q1 | 嵌套对象，包含: shape_name(形状名称), liquid_surface_horizontal(液面是否水平 bool), liquid_surface_description(液面描述), top_surface_shape(顶面形状), top_surface_is_ellipse(顶面是否椭圆 bool), ellipse_semi_major(椭圆半长轴 cm), ellipse_semi_minor(椭圆半短轴 cm), bottom_surface_shape(底端形状), overall_description(整体描述), volume(铅块体积), liquid_level_z(液面高度 z), note(特殊说明) |
| Q2_initial_state | D2/Q2 | 嵌套对象，包含: description(描述), recognizes_bottom_on_slope(是否识别底端贴斜面 bool), mentions_falling(是否提到自由落体 bool) |
| Q3_sliding | D3/Q3 | 嵌套对象，包含: sliding(是否滑动 bool), acceleration(加速度表达式), quantitative(是否给出定量值 bool) |
| Q4_rotation | D4/Q4 | 嵌套对象，包含: rotation(是否旋转 bool), reason(原因), description(运动描述) |
| extra_claims | 超出维度 | 模型给出的其他信息（不在 D1-D4 中），列表形式 |

## 模型回答
{response}

## 输出要求
请严格输出一个合法 JSON 对象，字段名与上表完全一致。嵌套对象中未提供的子字段值设为 null。extra_claims 为列表。不包含任何 JSON 以外的文字。
注意：JSON 字符串值中不要使用未转义的 ASCII 双引号。如需引用名称请用《》或「」代替。

```json
"""

SCORE_PROMPT = """\
你是一个评分机器。请严格按照以下规则评分，不要修改任何维度或基准值。

{rules}

## 评分维度与基准值

| 维度ID | 框架代号 | 维度名称 | 基准值 | 满分 | 评分规则 |
|:------:|:---:|---------|:---:|:---:|---------|
| D1 | Q1 | 铅块三维几何形状 | 斜截圆柱+椭圆顶面(半轴√3/1)+三棱锥底端+体积24.76 cm³+液面z=5 | 6 | 6/4/2/0 见上 |
| D2 | Q2 | 铅块初始接触状态 | 底端贴斜面于(10,10,0)，整体倾斜悬空 | 2 | 2=识别 0=未识别 |
| D3 | Q3 | 斜面滑动 | 无摩擦滑动 a=g/2≈4.9 m/s² | 2 | 2=定量 1=定性 0=错/无 |
| D4 | Q4 | 旋转/翻倒 | 质心偏离接触点，重力力矩→翻倒+滑动 | 2 | 2=正确分析 0=未/错 |

## 验证报告（作为参考）
{verification}

## 待评分的模型声明
模型名: {model_name}
```json
{extracted_json}
```

## 输出要求
输出一个合法 JSON 数组，包含 D1-D4 共 4 个对象，每个对象格式如下。不包含任何 JSON 以外的文字。
注意：JSON 字符串值中不要使用未转义的 ASCII 双引号。如需引用名称请用《》或「」代替。
注意：score 必须严格取自对应维度的允许值集合（D1: {{0,2,4,6}}, D2: {{0,2}}, D3: {{0,1,2}}, D4: {{0,2}}）。

```json
[
  {{"id": "D1", "name": "铅块三维几何形状", "claimed_value": "...", "benchmark_value": "...", "score": 6, "reason": "..."}},
  {{"id": "D2", "name": "铅块初始接触状态", "claimed_value": "...", "benchmark_value": "...", "score": 2, "reason": "..."}},
  {{"id": "D3", "name": "斜面滑动", "claimed_value": "...", "benchmark_value": "...", "score": 2, "reason": "..."}},
  {{"id": "D4", "name": "旋转/翻倒", "claimed_value": "...", "benchmark_value": "...", "score": 2, "reason": "..."}}
]
```
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════


def load_env():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def call_llm(
    prompt: str, tag: str = "", model: str = "anthropic/claude-sonnet-4"
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[ERROR] 请先安装 openai: pip install openai")

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        sys.exit("[ERROR] OPENROUTER_API_KEY 未设置。请在 .env 或环境变量中配置。")

    if tag:
        print(f"  [{tag}] 调用 LLM ({model}) ...")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    # 去除 markdown 代码块标记
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def repair_json_text(text: str) -> str:
    """修复 LLM 返回的 JSON 中常见的未转义引号问题。
    典型场景：中文语境中的 ASCII 双引号被误嵌在 JSON 字符串内。"""
    import re

    cjk = r"[一-鿿　-〿＀-￯—…·]"
    text = re.sub(rf'(?<={cjk})"(?={cjk})', r'\\"', text)
    return text


def safe_json(text: str, label: str = ""):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json_text(text)
        result = json.loads(repaired)
        print(f"  [INFO] {label}: JSON 修复成功（转义了中文语境内的引号）")
        return result
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON 解析失败 ({label}): {e}")
        print(f"  原始文本（前500字）: {text[:500]}")
        return None


def load_model_answer(filepath: str, query_id: int, model_name: str) -> str:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if item.get("id") == query_id:
                return item.get("response") or item.get("report_content") or ""
        sys.exit(f"[ERROR] {model_name}: id={query_id} 在 {filepath} 中未找到")
    if isinstance(data, dict):
        return data.get("report_content") or data.get("response") or ""
    sys.exit(f"[ERROR] {model_name}: {filepath} 格式无法识别")


def parse_model_args(model_args: list, query_id: int) -> dict:
    models = {}
    for arg in model_args:
        if "=" not in arg:
            sys.exit(f"[ERROR] --models 格式错误: '{arg}'，期望 name=path")
        name, path = arg.split("=", 1)
        print(f"  加载 {name} 从 {path} ...")
        models[name] = load_model_answer(path, query_id, name)
        print(f"  {name}: {len(models[name])} 字符")
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: 提取
# ═══════════════════════════════════════════════════════════════════════════════


def extract_stage(models_responses: dict, llm_model: str) -> dict:
    results = {}
    for model_name, response in models_responses.items():
        if not response or not response.strip():
            print(f"  [{model_name}] 空回答，全部字段记 null")
            results[model_name] = {d["extract_field"]: None for d in DIMENSIONS}
            results[model_name]["extra_claims"] = []
            continue
        prompt = EXTRACT_PROMPT.format(query=QUERY_TEXT, response=response[:20000])
        raw = call_llm(prompt, tag=f"Extract-{model_name}", model=llm_model)
        parsed = safe_json(raw, label=f"extract-{model_name}")
        if parsed is None:
            results[model_name] = {d["extract_field"]: None for d in DIMENSIONS}
            results[model_name]["extra_claims"] = []
        else:
            results[model_name] = parsed
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: 评分
# ═══════════════════════════════════════════════════════════════════════════════

ALLOWED_SCORES = {"D1": {0, 2, 4, 6}, "D2": {0, 2}, "D3": {0, 1, 2}, "D4": {0, 2}}


def clamp_score(dim_id: str, score) -> float:
    """将非法分数吸附到最近的允许值。"""
    allowed = ALLOWED_SCORES.get(dim_id, {0})
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    return float(min(allowed, key=lambda x: abs(x - s)))


def score_stage(extracted_results: dict, llm_model: str) -> tuple[dict, list]:
    """逐模型评分，返回 (all_scores, all_extra_claims)。"""
    all_scores = {}
    all_extra_claims = []

    for model_name, extracted in extracted_results.items():
        extra = extracted.get("extra_claims") or []
        for ec in extra:
            all_extra_claims.append(
                {
                    "model": model_name,
                    "content": ec
                    if isinstance(ec, str)
                    else json.dumps(ec, ensure_ascii=False),
                    "note": "不在4个评分维度中，不计分",
                }
            )

        extracted_json = json.dumps(
            {k: v for k, v in extracted.items() if k != "extra_claims"},
            ensure_ascii=False,
            indent=2,
        )
        prompt = SCORE_PROMPT.format(
            rules=SCORING_RULES_TEXT,
            verification=VERIFICATION_SUMMARY,
            model_name=model_name,
            extracted_json=extracted_json,
        )
        raw = call_llm(prompt, tag=f"Score-{model_name}", model=llm_model)
        parsed = safe_json(raw, label=f"score-{model_name}")

        if parsed is None or not isinstance(parsed, list):
            print(f"  [WARN] {model_name} 评分失败，全部记0")
            parsed = [
                {
                    "id": d["id"],
                    "name": d["name"],
                    "claimed_value": None,
                    "benchmark_value": d["benchmark_value"],
                    "score": 0,
                    "reason": "LLM 评分失败",
                }
                for d in DIMENSIONS
            ]

        # 合法性校验：分数只能取对应维度的允许值
        for item in parsed:
            dim_id = item.get("id")
            s_raw = item.get("score", 0)
            s_fixed = clamp_score(dim_id, s_raw)
            if s_fixed != s_raw:
                print(
                    f"  [WARN] {model_name}/{dim_id}: 非法分数 {s_raw} → 吸附为 {s_fixed}"
                )
                item["score"] = s_fixed

        # 补充 benchmark_value（防止 LLM 遗漏）
        dim_map = {d["id"]: d for d in DIMENSIONS}
        for item in parsed:
            if not item.get("benchmark_value"):
                item["benchmark_value"] = dim_map.get(item["id"], {}).get(
                    "benchmark_value", ""
                )

        # 按 D1-D4 顺序排序
        order = {d["id"]: i for i, d in enumerate(DIMENSIONS)}
        parsed.sort(key=lambda x: order.get(x.get("id"), 999))

        all_scores[model_name] = parsed

    return all_scores, all_extra_claims


# ═══════════════════════════════════════════════════════════════════════════════
# 输出生成
# ═══════════════════════════════════════════════════════════════════════════════


def compute_results(all_scores: dict) -> dict:
    results = {}
    for model_name, dim_scores in all_scores.items():
        total = sum(d.get("score", 0) for d in dim_scores)
        answered = sum(1 for d in dim_scores if d.get("claimed_value") is not None)
        results[model_name] = {
            "total_score": total,
            "max_score": MAX_SCORE,
            "score_rate": round(total / MAX_SCORE, 4),
            "dimensions_answered": answered,
            "per_dimension": dim_scores,
        }
    return results


def build_ranking(results: dict) -> list:
    items = sorted(results.items(), key=lambda x: -x[1]["total_score"])
    ranking = []
    for i, (name, data) in enumerate(items):
        if i > 0 and data["total_score"] == items[i - 1][1]["total_score"]:
            rank = ranking[-1]["rank"]
        else:
            rank = i + 1
        ranking.append(
            {
                "rank": rank,
                "model": name,
                "score": data["total_score"],
                "rate": f"{data['score_rate'] * 100:.1f}%",
            }
        )
    return ranking


def write_scores_json(
    results: dict, ranking: list, extra_claims: list, output_dir: str
) -> dict:
    output = {
        "query_id": QUERY_ID,
        "query": QUERY_TEXT,
        "scoring_mode": SCORING_MODE,
        "max_score": MAX_SCORE,
        "snapshot_date": SNAPSHOT_DATE,
        "dimensions": [d["name"] for d in DIMENSIONS],
        "benchmarks": {
            d["id"]: {
                "name": d["name"],
                "value": d["benchmark_value"],
                "source": d["source"],
                "max_score": d["max_score"],
            }
            for d in DIMENSIONS
        },
        "results": results,
        "ranking": ranking,
        "extra_claims": extra_claims,
    }
    path = os.path.join(output_dir, "scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  scores.json → {path}")
    return output


def write_ranking_md(scores_data: dict, output_dir: str):
    L = []
    L.append("# 模型排名报告（自动生成）：正方体蜡块铅芯几何形状与物理运动评估")
    L.append(f"query: {QUERY_TEXT[:100]}……")
    L.append("")
    L.append(f"> 自动评分日期: {date.today().isoformat()}")
    L.append(
        "> 数据来源: Stage 0 预设评分框架 + Stage 2 验证报告（解析推导+数值积分+蒙特卡洛复核）"
    )
    L.append(f"> **评分模式: Mode {SCORING_MODE}**，理论满分 {MAX_SCORE} 分")
    L.append(f"> 基准快照日期: {SNAPSHOT_DATE}")
    L.append("")
    L.append("## 排名方法")
    L.append("")
    L.append("1. **框架来源**: 评分维度和规则硬编码于 auto_scorer.py，本次评分未调整")
    L.append("2. **基准值来源**: 由解析推导+数值积分+蒙特卡洛复核的事实填入")
    L.append(
        "3. **积分规则**: Mode B（D1 形状 6/4/2/0）+ Mode C（D2/D4 各 2/0；D3 2/1/0）"
    )
    L.append("4. **总分**: 四维度独立计分直接加和，不加权")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 一、全局基准（维度→基准值→来源→声明映射）")
    L.append("")
    L.append(
        "| 维度ID | 框架代号 | 维度名称 | 基准值 | 满分 | 基准来源 | 对应验证声明# | 对应提取字段 |"
    )
    L.append("|:------:|:---:|---------|:---:|:---:|---------|:---:|---------|")
    for d in DIMENSIONS:
        L.append(
            f"| {d['id']} | {d['framework_code']} | {d['name']} | {d['benchmark_value']} "
            f"| {d['max_score']} | {d['source']} | {d['verification_claim']} | {d['extract_field']} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 二、各模型详情")
    L.append("")

    results = scores_data["results"]
    dim_code = {d["id"]: d["framework_code"] for d in DIMENSIONS}
    dim_max = {d["id"]: d["max_score"] for d in DIMENSIONS}

    for item in scores_data["ranking"]:
        model_name = item["model"]
        data = results[model_name]
        L.append(f"### {item['rank']}. {model_name}")
        L.append("")
        L.append("| 维度ID | 框架代号 | 模型声称值 | 基准值 | 得分 | 说明 |")
        L.append("|:------:|:---:|-----------|:---:|:---:|------|")
        for dim in data["per_dimension"]:
            cv = str(dim.get("claimed_value") or "null")[:80]
            bv = str(dim.get("benchmark_value", ""))[:50]
            max_s = dim_max.get(dim["id"], "?")
            L.append(
                f"| {dim['id']} | {dim_code.get(dim['id'], '')} | {cv} "
                f"| {bv} | {dim['score']}/{max_s} | {dim.get('reason', '')} |"
            )
        L.append("")
        L.append(f"- 作答维度: {data['dimensions_answered']}/4")
        total = data["total_score"]
        rate = data["score_rate"] * 100
        L.append(f"- **总分: {total}/{MAX_SCORE} ({rate:.1f}%)**")
        L.append("")
        L.append("---")
        L.append("")

    L.append("## 三、最终模型排名")
    L.append("")
    L.append("| 排名 | 模型 | 总分 | 得分率 | 作答维度 |")
    L.append("|:---:|------|:---:|:-----:|:------:|")
    for item in scores_data["ranking"]:
        data = results[item["model"]]
        L.append(
            f"| {item['rank']} | {item['model']} | {item['score']}/{MAX_SCORE} "
            f"| {item['rate']} | {data['dimensions_answered']}/4 |"
        )
    L.append("")

    extra = scores_data.get("extra_claims", [])
    if extra:
        L.append("## 四、额外发现（不计分）")
        L.append("")
        L.append("以下信息由模型给出但不在 4 个预设评分维度中，供参考，不计入总分：")
        L.append("")
        for e in extra:
            L.append(
                f"- **{e.get('model', '')}**: {e.get('content', '')}（{e.get('note', '不计分')}）"
            )
        L.append("")

    L.append("### 注意事项")
    L.append("")
    L.append(
        f"1. **[必填] 评分模式与理论满分**: Mode {SCORING_MODE}，理论满分 = {MAX_SCORE} (6+2+2+2)"
    )
    L.append(
        "2. **[Mode B] 尺寸合理范围**: D1 的 6 分档采用 ≤10% 相对误差标准；体积 5√3π vs 5π√3−√6 偏差 9.9% 给 6 分；>10% 或主要尺寸错（如 15cm 轴长）降为 4 分"
    )
    L.append(
        "3. **[Mode C 语义等价]**: D3 加速度的不同表述（g/2、g sin30°、4.9 m/s²）等价；D4 翻倒/翻滚/旋转/倾覆等价"
    )
    L.append(
        "4. **[组合模式] 子部分权重**: 四维度独立计分直接加和，不加权；各评分点不重叠"
    )
    L.append("5. **[必填] 此排名仅评估模型在本特定查询上的表现**，不代表综合能力排名")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本报告由 auto_scorer.py 自动生成*")

    path = os.path.join(output_dir, "ranking_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  ranking_report.md → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Query 31 自动评分脚本（Mode B+C 组合）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query-id", type=int, default=QUERY_ID)
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="name=path",
        help="模型回答文件，格式: claude=path/to/claude.json（可多个）",
    )
    parser.add_argument("--bench", help="题库 JSON（预留参数，暂不使用）")
    parser.add_argument("--result-json", help="已有 result.json 路径（跳过 Stage 1）")
    parser.add_argument("--output-dir", default="query_31/auto_scores/")
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4",
        help="LLM 模型（默认: anthropic/claude-sonnet-4）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Stage 1: 提取 ──────────────────────────────────────────────────────────
    if args.result_json:
        print(f"\n[Stage 1] 读取已有 result.json: {args.result_json}")
        with open(args.result_json, encoding="utf-8") as f:
            data = json.load(f)
        skip = {
            "query_id",
            "query",
            "domain",
            "scoring_framework",
            "fields",
            "extraction_mode",
            "scoring_dimensions",
        }
        extracted = data.get("models") or {
            k: v for k, v in data.items() if k not in skip
        }
    elif args.models:
        print(f"\n[Stage 1] 从 {len(args.models)} 个模型文件提取...")
        models_responses = parse_model_args(args.models, args.query_id)
        extracted = extract_stage(models_responses, args.model)
        result_path = os.path.join(args.output_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {"query_id": args.query_id, "query": QUERY_TEXT, "models": extracted},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  result.json → {result_path}")
    else:
        sys.exit("[ERROR] 请提供 --models 或 --result-json")

    # ── Stage 3: 评分 ──────────────────────────────────────────────────────────
    print(f"\n[Stage 3] 评分 {len(extracted)} 个模型...")
    all_scores, extra_claims = score_stage(extracted, args.model)

    # ── 输出 ───────────────────────────────────────────────────────────────────
    print(f"\n[输出] 生成报告到 {args.output_dir} ...")
    results = compute_results(all_scores)
    ranking = build_ranking(results)
    scores_data = write_scores_json(results, ranking, extra_claims, args.output_dir)
    write_ranking_md(scores_data, args.output_dir)

    # ── 摘要 ───────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Query {args.query_id} 自动评分完成")
    print(f"  模式: {SCORING_MODE} | 满分: {MAX_SCORE} | 模型数: {len(extracted)}")
    print(f"{'=' * 60}")
    for item in scores_data["ranking"]:
        print(
            f"  #{item['rank']}  {item['model']:<25s}  {item['score']}/{MAX_SCORE}  ({item['rate']})"
        )

    if extra_claims:
        print(
            f"\n[!] 发现 {len(extra_claims)} 条额外声明（不在4个维度中，已记录至 scores.json extra_claims）"
        )
    print()


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════════════
# 1. 完整运行（Stage 1 提取 + Stage 3 评分）:
#    python auto_scorer.py \
#      --query-id 31 \
#      --models claude=../<crawl-output>/claude.json \
#               openai=../<crawl-output>/openai.json \
#               gemini=../<crawl-output>/gemini.json \
#      --output-dir query_31/auto_scores/
#
# 2. 跳过提取，直接从已有 result.json 评分:
#    python auto_scorer.py \
#      --result-json query_31/result.json \
#      --output-dir query_31/auto_scores/
#
# 3. 使用自定义 LLM 模型:
#    python auto_scorer.py --models ... --model "openai/gpt-4o" --output-dir ...
#
# 环境变量（.env 或系统环境）:
#   OPENROUTER_API_KEY=sk-...
#   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  (default)
