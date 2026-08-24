# Query 16 评分框架 v2.1

> **Version**: 2.1
> **Locked at**: 2026-04-23
> **v2.1 变更**：基于 2000-2025 诺奖原始表（202 行 × 3 学科）重新校准了统计量区间 + 极端案例池 + 趋势规则。Medicine 实际 slope +0.058（持平），不再要求必须答"lengthening"；Physics/Chemistry 仍以 lengthening 为主但接受 "stable" 作为合理替代。
> **Query ID**: 16
> **Query**: 请查找 2000–2025 年间所有诺贝尔物理学奖、化学奖和生理学或医学奖的获奖者，对于每位获奖者，找到其获奖所依据的核心论文（或核心发现的最早公开发表时间），计算从发表到获奖的间隔年数。按三个学科分别统计中位数、均值和趋势（是否在缩短），并列出等待时间最长和最短的各 3 个案例。

---

## 一、评分规则（满分 15）

**A. 三学科统计量（9 分）** —— 每项 0/1

| # | 项 | 接受区间 | 实际值（2000–2025 原始表）|
|---|---|---|---|
| 1 | 物理学 - 中位数 | **25–35** 年 | 31 |
| 2 | 物理学 - 均值 | **22–32** 年 | 28.6 |
| 3 | 物理学 - 趋势方向 | `lengthening` 系列关键词 | slope +0.414 变长 |
| 4 | 化学 - 中位数 | **18–28** 年 | 24 |
| 5 | 化学 - 均值 | **18–28** 年 | 23.7 |
| 6 | 化学 - 趋势方向 | `lengthening` 或 `stable`（持平）均接受 | slope +0.283 轻微变长 |
| 7 | 生理医学 - 中位数 | **20–30** 年 | 25 |
| 8 | 生理医学 - 均值 | **20–30** 年 | 24.8 |
| 9 | 生理医学 - 趋势方向 | `stable`（持平）**或** `lengthening` 均接受；`shortening` 不接受 | slope +0.058 基本持平 |

**B. 极端案例（6 分）**

| # | 项 | 判定 |
|---|---|---|
| 10–12 | 最长等待 3 例 | 每命中一例 = 1 分。**接受池**（来自原始表 cross-discipline top 9 + 行业已知长等待）：Penrose（55 年）/ Peebles（54）/ Manabe（54）/ Ginzburg（53）/ Clauser（50）/ Gurdon（50）/ Englert（49）/ Higgs（49）/ Nambu（47）/ Shimomura（46，chem 2008）/ Alter（45，med 2020）/ Whittingham（43）/ Carlsson（43）/ Yekimov（42）/ Goodenough（39）/ Gross/Wilczek/Politzer（31）|
| 13–15 | 最短等待 3 例 | **接受池**（来自原始表 cross-discipline top 9）：Weiss / Thorne / Barish（LIGO 物理 2017，1 年）/ Hassabis / Jumper（AlphaFold 化学 2024，3 年）/ MacKinnon（化学 2003，5 年）/ Kornberg（化学 2006，5 年）/ Kobilka（化学 2012，5 年）/ Cornell / Wieman（物理 2001，6 年）/ Yamanaka（医学 2012，6 年）/ Fire / Mello（医学 2006 RNAi，8 年）/ Doudna / Charpentier（化学 2020 CRISPR，8 年）/ Geim / Novoselov（物理 2010 graphene，6 年） |

> 命中判定：模型列出的人名中，只要有 3 个能和上述接受池匹配即可。姓氏匹配足够（如 "Higgs"、"Goodenough"、"Doudna"）。

**容差与合并规则**
- 中位数/均值：具体数字（含小数）落在区间内 = 1；范围跨区间（如"15–25 年"）且主体在接受区间内 = 1
- 趋势：关键词匹配（去大小写/空格）；"增长/lengthening/getting longer" 均算匹配
- 极端案例：任意 3 个对即可（无序）。若只列 1-2 例则按比例计分（2 例 → 2/3, 1 例 → 1/3，四舍五入到整数）

---

## 二、Ground Truth 依据

### 主要数据源

- Nobel Prize 官方获奖人列表（2000–2025）
- 各得主的"核心论文"发表年份（官方 nobelprize.org 介绍 / Wikipedia / Google Scholar）
- 多项学术研究（如 Santo Fortunato 等对诺贝尔等待时间的统计）

### 典型案例（用于极端 3 例判定）

**最长等待** (≥35 年)：
- Peter W. Higgs — Physics 2013 for 1964 paper — 49 years
- Frank Wilczek, David Gross, H. D. Politzer — Physics 2004 for 1973 — 31 years
- John B. Goodenough — Chemistry 2019 for 1980 — 39 years
- Andre Geim, Konstantin Novoselov — Physics 2010 for 2004 graphene — 6 years（这个是**短**的）
- Roger Penrose — Physics 2020 for 1965 paper — 55 years
- James Peebles — Physics 2019 for 1960s cosmology — ~50 years

**最短等待** (≤15 年)：
- Jennifer Doudna & Emmanuelle Charpentier — Chemistry 2020 for 2012 CRISPR — 8 years
- Andre Geim & Konstantin Novoselov — Physics 2010 for 2004 graphene — 6 years
- Pierre Agostini, Ferenc Krausz, Anne L'Huillier — Physics 2023 for attosecond pulses — ~10-20 years
- Katalin Karikó & Drew Weissman — Medicine 2023 for 2005 mRNA — 18 years
- Yoshinori Ohsumi — Medicine 2016 for early 1990s — ~23 years（偏中等）

### 趋势（2000-2025 分析结果）

- 物理学：趋势明显**变长**（early 20th century ~10 年 → 近年 ~25 年）
- 化学：**变长**（~10 年 → ~20 年）
- 生理医学：**变长**（~15 年 → ~25 年）
- 共识来源：Fortunato et al. 2014；Inefuku & Glänzel 多篇更新分析

---

## 三、GT Changelog

| Version | Date | Change |
|---|---|---|
| 2.0 | 2026-04-23 | 从 131 分（Cov/Sub/Stat/Num/Ext 5 组）精简为 15 分（9 统计量 + 6 极端）|
| 1.0 | 2026-03 | 初版 |
