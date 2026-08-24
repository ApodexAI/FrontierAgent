# Query 21 评分框架 v2

> **Version**: 2.0
> **Locked at**: 2026-04-23
> **Query ID**: 21
> **Query**: 有一个人代表获奖者人领了一个很重要的世界级奖项，这个奖项的颁奖时间是哪一年？我记得这个人的名字中有 do 这两个字母。被领奖人的名字中有 in 这两个字母。

---

## 一、评分规则

**核心答案**：颁奖年份。

**满分 = 1**。

| 项 | 分值 | 判定 |
|---|---|---|
| **年份正确** | 1 / 0 | 模型给出的年份 ∈ `{1922, 2025}` → 1；否则 0 |

**参考项（不计分，仅审计用）**：
- 奖项名称
- 代领人姓名（应含连续字母 "do"）
- 被领奖人姓名（应含连续字母 "in"）
- 缺席原因

---

## 二、Ground Truth

### 主候选（1922 — 置信度最高）

| 字段 | 值 | 来源 |
|---|---|---|
| 年份 | **1922** | [NobelPrize.org](https://www.nobelprize.org/prizes/physics/1921/summary/) |
| 奖项 | 1921 年诺贝尔物理学奖（1922 年追授 / 颁奖） | 同上 |
| 代领人 | **Rudolf Nadolny**（德国驻瑞典公使；含 "do"）| [Wikipedia: Rudolf Nadolny](https://en.wikipedia.org/wiki/Rudolf_Nadolny) |
| 被领奖人 | **Albert Einstein**（含 "in"，出现两次）| [NobelPrize.org Einstein Facts](https://www.nobelprize.org/prizes/physics/1921/einstein/facts/) |
| 缺席原因 | 在赴日本讲学途中（1922 秋） | [arXiv:2112.13519](https://arxiv.org/pdf/2112.13519) |

### 次候选（2025 — 有效但带条件）

| 字段 | 值 | 备注 |
|---|---|---|
| 年份 | **2025** | [NobelPrize.org 2025 Peace](https://www.nobelprize.org/prizes/peace/2025/) |
| 奖项 | 2025 年诺贝尔和平奖 | |
| 代领人 | **必须含"Machado"才接受**。短名 "Ana Corina Sosa" 不含 "do" → 不满足题目字面约束 | [Al Jazeera 2025-12-10](https://www.aljazeera.com/news/2025/12/10/machado-in-oslo-but-will-not-attend-nobel-ceremony-to-receive-award) 用全名；NPR 用短名 |
| 被领奖人 | María Corina Machado（含 "in"）| |
| 缺席原因 | 因政治迫害 / 行动受限 | [NPR](https://www.npr.org/2025/12/10/nx-s1-5638521/) |

> **判定规则**：
> - 答 **1922** → 1 分（代领人 Rudolf Nadolny 无条件含 "do"）
> - 答 **2025** 且 proxy_name 含 "Machado" 或 "do" → 1 分
> - 答 **2025** 但 proxy_name 只写 "Ana Corina Sosa" → **0 分**（不满足字面约束）
> - 其他年份 → 0 分

---

## 三、GT Changelog

| Version | Date | Change |
|---|---|---|
| 2.0 | 2026-04-23 | 从旧版 5 维度（各 1 分）简化为 1 维二值；D2–D5 降为参考项。接受 2025 作为第二候选 |
| 1.0 | 2026-03-29 | 初版，5 维度各 1 分 |
