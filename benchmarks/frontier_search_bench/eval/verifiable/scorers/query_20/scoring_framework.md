# Query 20 评分框架 v2.2

> **Version**: 2.2
> **Locked at**: 2026-04-24
> **Query ID**: 20
> **Query**: I've had an earworm for a few days, but I can't remember the exact name of the song. I can replay it on the guitar with a capo on the 3rd fret, using a repeating chord progression of C–G–Am–F in the intro and verses. The tune shifts to a slightly higher arpeggio pattern in the bridge. The overall tone is very melancholic and reflective. I also remember the singer is a man in his 30s or 40s (can't be sure). Can you help me identify (or narrow down) the song?

---

## 一、评分规则

**核心答案**：模型给出的候选歌曲里有多少个命中 GT 白名单。

| 项 | 分值规则 |
|---|---|
| **每个命中 GT 白名单的候选歌曲** | +1 分（每首独立计数，无上限）|
| 候选不在白名单 | 0 分（不扣分）|

**音乐理论分析（原 Sub-1 / Sub-2）**：v2 已删除，不再评分。

**关于 `max_score` 字段（仅供跨题归一化）**：score.json 里写的 `max_score` 等于白名单大小（4），并据此填 `total_rate = min(score / 4, 1.0)`。这个上限**不是** scoring 真实约束——上面"每首独立计数，无上限"的规则照旧；`max_score` 只是为 `run_all.py` 跨题汇总提供归一化基准，让 Q20 能进入跨题平均。score > 4 的情况（一个模型把同一首歌列多种别名都对）`total_rate` 被 clip 到 1.0，本题原始 `score` 字段保留无上限信息可单独审视。

---

## 二、Ground Truth

### 白名单（≥4/6 条线索匹配）— 每首 1 分

| 歌曲 | 歌手 | 匹配线索（6 条中）| 验证来源 |
|---|---|---|---|
| **Demons** | Imagine Dragons | 4.5/6（和弦 + capo3 + 重复 + 忧郁 + 歌手半匹配）| Ultimate Guitar, Songsterr |
| **Someone You Loved** | Lewis Capaldi | 4/6（和弦 + 重复 + 桥段 + 忧郁）| Ultimate Guitar |
| **You're Beautiful** | James Blunt | 4/6（和弦 + 重复 + 忧郁 + 歌手）| Ultimate Guitar |
| **I Must Belong Somewhere** | Bright Eyes | 4.5/6（和弦 + capo3 + 重复 + 忧郁 + 歌手半匹配）| Spy Tunes |

> **核查后移除**：`traitor (Olivia Rodrigo)` 被移出白名单。Rodrigo 为女歌手（2003 年生，发布 2021 时 18 岁），违反题目硬约束"singer is a man in his 30s or 40s"。来源：[Wikipedia: traitor (song)](https://en.wikipedia.org/wiki/Traitor_(song))。

### 候选匹配机制（v2.2）

白名单匹配不再是简单子串，而是**分层**进行，兼顾确定性与语义理解：

| Tier | 触发条件 | 处理 |
|---|---|---|
| **Tier 1** | 归一化后 title 完全相同 + artist 匹配 | 自动命中 |
| **Tier 2** | title 为长歌名的子串（或 Jaccard ≥ 0.8）+ artist 匹配；**或** title 命中已知翻译别名（如"恶魔"→Demons）+ artist 匹配 | 自动命中 |
| **Tier 3** | title 能对上某白名单歌曲，但 **artist 缺失** | 走 **LLM judge**（Claude Sonnet 4，temperature=0）|
| **Tier 4** | title 命中翻译别名但 artist 不确定 | 走 **LLM judge** |
| 其他 | 完全不匹配 | 自动 0 分 |

**归一化**：Unicode NFKC + 弯引号（`' " '`）→ 直引号 + 全半角统一 + 去标点 + 小写 + 折叠空白。解决 "You're" vs "You're" 的误判。

**LLM judge** 只在 Tier 3/4 触发，且**每次调用都带 `rationale` + `confidence` + `raw_output`** 写入 `score.json.llm_judge_log`，可审计复核。规则见 `auto_scorer.py` 的 `_llm_judge()` + `candidate_match()`。

**别名表**（含翻译）：
- Demons: 恶魔
- Someone You Loved: 你曾爱过的人
- You're Beautiful: youre beautiful, you are beautiful, 你如此美丽
- I Must Belong Somewhere: —

### 6 条线索定义（来自原框架，保留作参考）

1. 和弦进行 C-G-Am-F（I-V-vi-IV）
2. Capo 位于第 3 品
3. 在 intro + verses 中重复出现
4. 桥段有琶音或音域上升
5. 整体忧郁/反思基调
6. 男歌手 30-49 岁

### 典型**不合格候选**（≤3/6）— 0 分

- With or Without You (U2)
- Let It Be (Beatles)
- Say You Won't Let Go (James Arthur)
- Fix You (Coldplay)
- Before You Go (Lewis Capaldi)
- Chasing Cars (Snow Patrol)

---

## 三、GT Changelog

| Version | Date | Change |
|---|---|---|
| 2.2 | 2026-04-24 | 匹配升级为 4-tier + LLM judge fallback；Unicode/curly-quote 归一化；白名单加入翻译别名 |
| 2.1 | 2026-04-23 | Phase 4 核查移除 `traitor (Olivia Rodrigo)`（违反"male 30s-40s"硬约束）|
| 2.0 | 2026-04-23 | 去掉维度组 A（音乐理论推导，3 分），只按候选歌曲白名单 1 分/首计算（无上限）|
| 1.0 | 2026-03-31 | 初版：A 组（3 分）+ B 组（每首 0/1/2）|
