# Query 03 评分框架 v3

> **Version**: 3.0
> **Locked at**: 2026-04-29
> **Cutoff date**: 2026-01-01
> **Query ID**: 3
> **Query**: 截至2026年1月1日，SpaceX 星舰一共完成了几次飞行测试？逐次列出每次的关键进展和失败点。

---

## 一、评分规则（满分 +23，最低 -23）

23 个独立维度，每个维度独立判定 ±1（或 0）：

| 维度 | 判定 |
|---|---|
| **A. 飞行测试总次数** | claim ∈ **{10, 11}** → +1；其他 → -1；未提 → -1 |
| **IFT-N 进展子维度**（N=1..11，共 11 个）| 见下方"IFT 子维度规则" |
| **IFT-N 失败点子维度**（N=1..11，共 11 个）| 同上 |

### IFT 子维度规则（β2 严格）

对每个 IFT × {progress / failures} 子维度：

1. 若该 IFT entity 的 `canonical.value == None` 或 `not_mentioned == true`（**模型完全未提及该 IFT**）→ 该 IFT 的 progress 和 failures 都 = **-1**
2. 若 entity 存在但该子类的 list 为空（**提到 IFT 但未列任何 progress / failures**）→ **-1**
3. 若 list 非空：
   - 任一条 claim 匹配 `confirmed_wrong` kw → **-1**（覆盖所有正确命中，严格扣分）
   - 否则任一条 claim 匹配 `progress` / `failures` GT kw → **+1**
   - 否则（都没匹配）→ **0**（提了内容但没对上 GT 也没明显错）

**总分** = A_score + Σ(progress_score) + Σ(failures_score)

满分 +23 的获取条件：A 答对 + 11 个 IFT 的 progress 和 failures 都至少 1 条命中 GT 且无错答。

最低 -23 的获取条件：A 答错（或没答）+ 11 个 IFT 完全未列。

---

## 二、Ground Truth（v3，2026-04-29 锁定）

### A. 飞行测试次数

- **基准值**: **11 次**已完成（IFT-1 至 IFT-11，截至 2026-01-01）
- **接受集合**: {10, 11}
  - 11 = 主流标准答案
  - 10 = 给"不算 IFT-1（爆炸未达分级）"的紧扣字眼派留容差
- **拒绝**: 12（IFT-12 计划 2026 春发射，截止日期前未完成）/ 9 / 8 / 其他
- **来源**: [Wikipedia: List of Starship launches](https://en.wikipedia.org/wiki/List_of_Starship_launches)

### B. 逐次飞行 GT 概览

每个 IFT 的 GT 包含 3 类条目，每条都有 `canonical`（事实描述）和 `kw` 列表（鲁棒匹配的关键词）：

| IFT | Date | Block | progress 条数 | failures 条数 | confirmed_wrong 条数 |
|---|---|---|---:|---:|---:|
| IFT-1 | 2023-04-20 | Block 1 | 2 | 4 | 2 |
| IFT-2 | 2023-11-18 | Block 1 | 3 | 2 | 0 |
| IFT-3 | 2024-03-14 | Block 1 | 3 | 2 | 0 |
| IFT-4 | 2024-06-06 | Block 1 | 1 | 3 | 0 |
| IFT-5 | 2024-10-13 | Block 1 | 2 | 1 | 0 |
| IFT-6 | 2024-11-19 | Block 1 | 4 | 1 | 1 |
| IFT-7 | 2025-01-16 | Block 2 / V2 (首飞) | 3 | 2 | 1 |
| IFT-8 | 2025-03-06 | Block 2 / V2 | 1 | 2 | 1 |
| IFT-9 | 2025-05-27 | Block 2 / V2 | 3 | 2 | 1 |
| IFT-10 | 2025-08-26 | Block 2 / V2 | 3 | 2 | 2 |
| IFT-11 | 2025-10-13 | Block 2 / V2 (最后一飞) | 5 | 1 | 0 |
| **TOTAL** | | | **30** | **22** | **8** |

**GT 来源**：60 条结构化事实，每个 IFT 含 progress / failures / confirmed_wrong 三类（progress 共 30 / failures 22 / confirmed_wrong 8），逐条硬编码在 `auto_scorer.py::PER_IFT_GT`。

### C. confirmed_wrong（已知错答）

8 条已记录的模型错答（放在 `auto_scorer.py::PER_IFT_GT[ift]["confirmed_wrong"]`）：

1. **IFT-1**: AFSS 在 T+4:01 触发（实际 T+3:20，T+4:01 是解体时间）
2. **IFT-1**: 促使加装导流系统（实际 IFT-1 前已开建）
3. **IFT-6**: 助推器未满足安全判定（实际是塔通信故障）
4. **IFT-7**: 11 项纠正（数目无确认）
5. **IFT-8**: 第二次 Block 2 因振动起火（实际是发动机硬件故障）
6. **IFT-9**: 极限应力测试解体（实际解体非预期）
7. **IFT-10**: 首次在轨重点火（IFT-6 才是首次）
8. **IFT-10**: 报告无失败/全部成功（遗漏发动机故障和尾舱爆炸）

---

## 三、抽取设计（12 个 entity）

```python
ENTITIES = [
    {"id": "A_flight_count", ...},   # 单整数 (or null)
    {"id": "IFT_1_facts",   ...},    # composite {progress: [str], failures: [str]}
    {"id": "IFT_2_facts",   ...},
    ...
    {"id": "IFT_11_facts",  ...},
]
```

每个 entity 走 5-LLM 投票。每个 IFT_N_facts 抽出形如：

```json
{
  "progress": ["33 台 Raptor 全部点火", "首次热分离成功", ...],
  "failures": ["LOX 滤网堵塞导致助推器爆炸", "排气泄漏起火", ...]
}
```

模型未提及该 IFT → `value: null` 或 `not_mentioned: true`，触发整 IFT -2 分。

---

## 四、与标准 4-stage pipeline 的差异

| Stage | 标准 4-stage flow | Q03 实际 | 一致？ |
|---|---|---|---|
| Stage 1 抽取 | 多 LLM 投票 → `{model}/extraction.json` | 12 entity 各跑 5-LLM 投票 | ✓ |
| Stage 2 对齐 | `align_claims()` 映射 canonical_id | **跳过**（无需 baseline 对齐） | 偏离（kw 直接匹配模式）|
| Stage 3 Null 验证 | `null_review.json` + Stage C agent | **跳过**（无 null 概念） | 偏离 |
| Stage 4 打分 | 确定性 lookup → `scores.json` + `ranking_report.md` | per-IFT kw 匹配 + β2 严格规则 | ✓ |

**为什么跳过 Stage 2/3**：
- 每条 claim 已自带 IFT 编号（"IFT-2: ..."），不需要 align_claims 做实体归属
- 事实级匹配用 kw 直接做，没有"同实体不同事件"的歧义需要 judge LLM 仲裁
- T1 复合事实查找模式（与 Q21、Q22 共用同一抽取骨架）

---

## 五、关键设计决策

### 5.1 为什么 missing IFT 给 -1（不是 0）

题目"逐次列出每次"明确要求 enumerate 完整。漏列等同于不完整作答，应扣分。

副作用：模型只覆盖 5 个 IFT 时会被扣 12 分。这是有意为之——奖励完整 enumerate 的模型。

### 5.2 为什么混合规则用严格扣分（任一错答 → -1）

用户决策：模型如果某 IFT 列了 3 条事实里有 1 条错的，就判该子维度 -1。

理由：题目要求"准确列出"，混杂错误等同信任度污染，应重罚而非"中和"。

替代方案被否决：
- 净分（正-错）：实现稍复杂，且容易让"3 对 1 错"也得 +2，偏宽
- 中和（任一错 → 0）：太宽松，模型敷衍写一长串夹杂错误反而占便宜

### 5.3 kw 匹配规范化

`_normalize()` 统一处理：
- lowercase + NFKC（全角转半角）
- 删除所有空白字符
- 统一连字符变体（U+2010 ~ U+2212）→ ASCII '-'

避免 "33 台 Raptor 全部点火"（含空格）vs "33台全部点火"（kw 不含空格）的子串匹配失败。

### 5.4 GT 自检：每个 progress / failures fact 必须 self-match 自己的 kw

写 GT 时容易出现 canonical 描述里有专有名词（Raptor / Mechazilla / Starlink）但 kw 列表用了去专有名词的简化短语，导致 model 用 canonical 风格的描述时反而匹配不上。

CI/sanity 自检脚本会检查每条 fact 的 canonical 能否被自己的 kw 命中（normalize 后的子串匹配）。confirmed_wrong 不强制 self-match（它是描述模型错答的元信息，不一定字面包含 wrong kw）。

---

## 六、Changelog

| Version | Date | Change |
|---|---|---|
| **3.0** | 2026-04-29 | 大改写：query 简化为 "几次 IFT + 逐次进展/失败" 后，砍掉 Mars 外推 (C) / Musk 兑现倍数 (E) / 迭代速度 (B) 三维；新增 11 IFT × 2 子维度的结构化 GT；A 维度 accept 收紧到 {10, 11}（旧版含 12）；引入 β2 严格扣分规则 |
| 2.0 | 2026-04-23 | 从 94 条事实声明精简为 5 项二值（A/B/C/D/E）；接受合理区间，不做事实细节扣分 |
| 1.0 | 2026-03 | 初版：94 条事实声明分 5 组，含每次 IFT 技术细节 |

---

## 七、运行命令

### 全量跑（Stage 1 + Stage 4）

```bash
python3 query_03/auto_scorer.py --models \
  ChatGlm=<crawl-output>/chatglm.json \
  claude=<crawl-output>/claude.json \
  gemini=<crawl-output>/gemini.json \
  ...
```

### 重打分（不重抽取）

```bash
python3 query_03/auto_scorer.py --skip-extract --models \
  ChatGlm=_skip claude=_skip ...
```

### 输出

- `auto_scores/scores.json` — 完整结果 + 排名
- `auto_scores/ranking_report.md` — 人类可读排名表 + 每模型 IFT 明细
- `auto_scores/{model}/score.json` — 每模型独立分数（含每 IFT 子维度 reason）
- `auto_scores/{model}/extraction.json` — 5-LLM 投票后的 canonical 抽取

---
