# Query 35 评分框架 v2

> **Version**: 2.0
> **Locked at**: 2026-04-23
> **Query ID**: 35
> **Query**: 帮我调研，海底捞在每个国家相距最远的两家店之间有多远，每个国家是哪两家店？

---

## 一、评分规则

每国一个综合分（0/1/2/3），18 国 × 3 = **满分 54**。

| 得分 | 判定 |
|---|---|
| **3** | 正确识别最远门店对（两家店城市 / 商场名任一匹配 GT）+ 距离精度 ±5% |
| **2** | 正确识别最远门店对 + 距离误差 5–15% **或** 门店对错但识别的两家店是 GT 的次远组合 |
| **1** | 门店对错（未识别最远）但距离量级合理（误差 ≤30%）|
| **0** | 门店对错 + 距离明显错误（误差 >30%）/ 未作答 / 只答国名 |

**柬埔寨（仅 1 家店）** 特殊规则：
- 正确回答"仅 1 家"/"无最远对" = 3 分
- 强行给一个距离 = 0 分
- 未作答 = 0 分

---

## 二、Ground Truth（18 国 × 门店对 + 距离）

| 维度 | 国家/地区 | 门店对 | GT 距离 (km) | 来源 |
|---|---|---|---|---|
| D1 | 中国大陆 | 喀什东城万达 ↔ 延吉百利城 | 4418 | haidilao.com + Haversine |
| D2 | 香港 | 荃湾海之戀商場 ↔ 将军澳中心 | 16.6 | 多模型交叉 |
| D3 | 澳门 | 威尼斯人 ↔ 伦敦人（共 2 家）| 0.9 | 多模型交叉 |
| D4 | 台湾 | 台北 ↔ 高雄大遠百 | 297 | 维基 + Haversine |
| D5 | 美国 | Flushing NY ↔ Cupertino CA | 4126 | Yelp + Haversine |
| D6 | 加拿大 | Vancouver ↔ Montréal | 3690 | haidilaoca.org + Haversine |
| D7 | 英国 | London O2 ↔ Birmingham Bullring | 169 | 官方 API + Haversine |
| D8 | 澳大利亚 | Melbourne ↔ Brisbane Q+A | 1370 | 多模型交叉 |
| D9 | 日本 | 千叶海浜幕张 ↔ 大阪难波（福冈 2024-03 已关）| 426 | jp.haidilao-inc.com |
| D10 | 韩国 | 首尔 ↔ 济州 Bolton Hotel 5F | 454 | NamuWiki + Google |
| D11 | 新加坡 | Jurong Point ↔ Century Square | 26.3 | 官方门店页 |
| D12 | 马来西亚 | 槟城 Gurney Paragon ↔ 古晋 Vivacity | 1195 | Miri + Vivacity 官网 |
| D13 | 泰国 | 清迈 Central Festival ↔ 普吉 Central Phuket | 1215 | haidilaothailand.com |
| D14 | 越南 | 河内 ↔ 胡志明市 | 1144 | 多模型交叉 |
| D15 | 印度尼西亚 | 棉兰 DeliPark ↔ 泗水 Pakuwon Trade Center | 1973 | Google Maps + Instagram |
| D16 | 菲律宾 | SM MOA ↔ Robinsons Galleria（共 2 家）| 10 | robinsonsmalls.com |
| D17 | 阿联酋 | Dubai Mall ↔ Dubai Festival City（共 2 家）| 8 | dubaifestivalcitymall.com |
| D18 | 柬埔寨 | **仅 1 家门店（金边 U Mall）** | N/A | 多模型一致 |

**店名模糊匹配**：对店名/城市名做 lowercase + 去空格 / 去标点的子串匹配（例如"荃湾" vs "Tsuen Wan"，"Flushing" vs "法拉盛"都算匹配）。若模型用中文或英文任一表达对上 GT 都算正确。

---

## 三、GT Changelog

| Version | Date | Change |
|---|---|---|
| 2.0 | 2026-04-23 | 每国从（识别 2 + 距离精度 3）改为单一综合分 0/1/2/3；满分 90→54 |
| 1.0 | 2026-03 | 初版 18 × (A+B) 拆分 |
