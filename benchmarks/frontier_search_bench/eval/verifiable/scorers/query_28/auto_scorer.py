"""Query 28 auto-scorer.

Composite three-part scoring:
  Part A — total exoplanet count       (max +1)
  Part B — JWST water-signal planets   (max +10, open-set list scoring)
  Part C — latest such planet          (max +4 = name(2) + date(1) + dist(1))

Total max = 15. Total score can be negative.

Pipeline:
  1. Extraction — two entities (jwst_water_planets list + summary_facts dict).
  2. Alignment — only E1 (jwst_water_planets) goes through align_claims.
                E2 (summary_facts) is read directly from extraction.json.
  3. Null verification — Stage C agent produces:
       - null_resolutions.json (for E1 nulls, standard schema)
       - summary_facts_verified.json (for E2 latest_planet, custom schema)
  4. Scoring — three independent score functions; aggregate per model.
"""

from __future__ import annotations

import argparse
import json
import re as _re
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from extract import (  # noqa: E402
    ENTITIES,
    PROMPT_HINTS,
    QUERY_ID,
    QUERY_TEXT,
    VALUE_SCHEMA,
)
from pipeline.alignment import (  # noqa: E402
    align_claims,
    apply_null_resolutions,
    export_null_claims_for_review,
    persist_new_baseline_entries,
)
from pipeline.extraction_pipeline import get_client, run_pipeline  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Part B — JWST water-signal planets baseline (DIMS)
#
# Cross-verified 2026-05-02 against:
#   D1-D8: NASA/STScI press releases + Nature/Nature Astronomy/MNRAS papers
#   D9    HAT-P-30 b ≡ WASP-51 b  : MNRAS BOWIE-ALIGN 2026-01 (3.3σ H₂O)
#   D10   WASP-166 b              : arXiv 2501.00609 (NIRISS+NIRSpec)
#   D11   K2-18 b   (⚠️ 0)        : JWST data does NOT robustly detect H₂O;
#                                   main signal is CH₄/CO₂; historic Hubble
#                                   "water" narrative is not JWST result
#   D12   L 98-59 d (❌ -1)       : Nature Astronomy 2026-03-16 paper
#                                   conclusively rules out water world,
#                                   identifies it as sulfur-rich magma world
#   D13   GJ 486 b  (❌ -1)       : 2024 follow-up (Mansfield et al.)
#                                   ruled out water signal as stellar spots
#   D14   HD 209458 b (❌ -1)     : Hubble-era detection, NOT JWST
# ═══════════════════════════════════════════════════════════════════════════

SNAPSHOT_DATE = "2026-05-02"

DIMS = [
    # ─── ✅ confirmed JWST H₂O detections (10 entries, +1 each) ───
    (
        "D1",
        "WASP-96 b / 1,150 ly / 2022-07 / JWST first transmission spectrum (ERS)，"
        "明确水蒸气信号",
        "✅",
        1.0,
        ["wasp-96", "wasp 96", "wasp96", "wasp-96 b", "wasp 96 b"],
    ),
    (
        "D2",
        "WASP-39 b / 700 ly / 2022-11 / JWST ERS Nature 2023, 水 + CO₂ + SO₂ 等多分子",
        "✅",
        1.0,
        ["wasp-39", "wasp 39", "wasp39", "wasp-39 b", "wasp 39 b"],
    ),
    (
        "D3",
        "WASP-18 b / 400 ly / 2023-05 (Coulombe et al. Nature) / "
        "2025-12 follow-up (3D thermal map); ultra-hot Jupiter, 水痕迹",
        "✅",
        1.0,
        ["wasp-18", "wasp 18", "wasp18", "wasp-18 b", "wasp 18 b"],
    ),
    (
        "D4",
        "WASP-107 b / 211 ly / 2023 (Dyrek et al. Nature, 12σ H₂O + SO₂) / "
        "2025-12-01 Nature Astronomy follow-up，H₂O 强检测",
        "✅",
        1.0,
        ["wasp-107", "wasp 107", "wasp107", "wasp-107 b", "wasp 107 b"],
    ),
    (
        "D5",
        "WASP-80 b / 163 ly / 2024 / 水 + 甲烷",
        "✅",
        1.0,
        ["wasp-80", "wasp 80", "wasp80", "wasp-80 b", "wasp 80 b"],
    ),
    (
        "D6",
        "GJ 9827 d / 98 ly / 2024-09 ApJL (Piaulet-Ghorayeb et al.) / "
        "首个被确认的 'steam world / 蒸汽世界'，大气以 H₂O 主导",
        "✅",
        1.0,
        ["gj 9827", "gj9827", "gj 9827 d", "gj-9827", "蒸汽世界", "steam world"],
    ),
    (
        "D7",
        "HAT-P-18 b / 532 ly / 2024 MNRAS (Fournier-Tondreau et al.) / "
        "JWST NIRISS/SOSS, H₂O 12.5σ 强检测 + CO₂ 7.3σ",
        "✅",
        1.0,
        ["hat-p-18", "hat p 18", "hatp18", "hat-p-18 b", "hat p 18 b"],
    ),
    (
        "D8",
        "WASP-121 b / 880 ly / 2025 Nature Astronomy + 2025 arXiv 2506.02199 / "
        "JWST NIRSpec G395H, 日侧 H₂O + CO + SiO，热解离效应",
        "✅",
        1.0,
        ["wasp-121", "wasp 121", "wasp121", "wasp-121 b", "wasp 121 b"],
    ),
    (
        "D9",
        "HAT-P-30 b ≡ WASP-51 b / 679 ly / 2026-01-21 MNRAS BOWIE-ALIGN / "
        "JWST NIRSpec G395H, H₂O 3.3σ + CO₂ 2.1σ (弱探测但已发表)",
        "✅",
        1.0,
        [
            "hat-p-30",
            "hat p 30",
            "hatp30",
            "hat-p-30 b",
            "wasp-51",
            "wasp 51",
            "wasp51",
            "wasp-51 b",
        ],
    ),
    (
        "D10",
        "WASP-166 b / 368 ly / 2025-01 arXiv 2501.00609 / "
        "JWST NIRISS SOSS + NIRSpec G395M, H₂O + CO₂ 主导谱线",
        "✅",
        1.0,
        ["wasp-166", "wasp 166", "wasp166", "wasp-166 b", "wasp 166 b"],
    ),
    # ─── ⚠️ 边界 (0 分): 学界叙事中常被算含水但 JWST 数据未 robust 检测 ───
    (
        "D11",
        "K2-18 b / 124 ly / Madhusudhan 2023 ApJL / "
        "JWST 主分子是 CH₄ + CO₂；H₂O 上限 < 0.1%（反而压低水丰度）；"
        "Hubble 时代'含水'叙事但 JWST 数据未 robust 检水",
        "⚠️",
        0.0,
        ["k2-18", "k2 18", "k218", "k2-18 b"],
    ),
    # ─── ❌ 错误 (-1 分): 学界已定论非水 / 非 JWST ───
    (
        "D12",
        "L 98-59 d / 35 ly / 2026-03-16 Nature Astronomy Bello-Arufe et al. / "
        "已被定论为富硫岩浆世界（H₂S/SO₂ 主导大气），明确**排除**水世界假说；"
        "模型若将其列为'JWST 含水行星'属事实错答",
        "❌",
        -1.0,
        [
            "l 98-59",
            "l98-59",
            "l98 59",
            "l 98-59 d",
            "l98-59 d",
            "l-98-59",
            "l-98-59 d",
        ],
    ),
    (
        "D13",
        "GJ 486 b / 26 ly / Moran 2023 ApJL 报告水信号 / "
        "Mansfield 2024 follow-up 已定论为恒星黑子伪信号，行星大概率无大气；"
        "模型若仍列为 JWST 含水属过时/错误",
        "❌",
        -1.0,
        ["gj 486", "gj486", "gj 486 b", "gj-486"],
    ),
    (
        "D14",
        "HD 209458 b / 159 ly / Hubble 时代水信号发现 / "
        "**非 JWST**；模型若把它当 JWST 含水行星属归错望远镜",
        "❌",
        -1.0,
        ["hd 209458", "hd209458", "hd 209458 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D15",
        "TOI-270 d / ~73 ly / 2024-03 / Benneke et al. arXiv:2403.03325 reports JWST NIRISS+NIRSpec transmission spectrum with CH4 (9.4σ) and CO2 (4.8σ) but H2O only at ~2.5σ (tentative). Holmberg & Madhusudhan (arXiv:2403.03244) finds H2O at 1.6–4.4σ. Hycean candidate but H2O not robustly detected at JWST significance threshold; treat as borderline like K2-18 b.",
        "⚠️",
        0.0,
        ["toi-270 d", "toi 270 d", "toi-270d", "toi270d", "TOI-270 d"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D16",
        "LTT 9779 b (Cuancoá) / ~260 ly / 2025-02 / Coulombe et al. 2025 Nature Astronomy report JWST/NIRISS detection of water vapor alongside silicate (MgSiO3, Mg2SiO4) clouds and asymmetric dayside atmosphere on this ultra-hot Neptune. Peer-reviewed.",
        "✅",
        1.0,
        ["ltt 9779 b", "ltt9779 b", "ltt-9779 b", "cuancoá", "cuancoa", "LTT 9779 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D17",
        "14 Herculis c / ~58.4 ly / 2025-06 / Bardalez Gagliuffi et al. arXiv:2506.09201 (AAS). JWST/NIRCam coronagraphic direct imaging shows photometry consistent with carbon disequilibrium chemistry and water-ice clouds, BUT this is indirect evidence from photometric color, not gaseous H2O detection in transmission/emission spectroscopy. Not in the same category as JWST atmospheric H2O spectral detections.",
        "❌",
        -1.0,
        ["14 herculis c", "14 her c", "14herculis c", "14-herculis c", "14 Herculis c"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D18",
        "WASP-17 b / ~1300 ly / 2023-10 (quartz cloud paper, Grant et al. ApJL 956 L29) and 2024-12 / 2025 (Louie et al. AJ 169 86, precise H2O abundance from NIRISS SOSS, log H2O ≈ -2.96, super-solar). H2O detection robust at 6σ+ and well-constrained.",
        "✅",
        1.0,
        ["wasp-17 b", "wasp 17 b", "wasp17 b", "wasp-17b", "WASP-17 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D19",
        "TOI-421 b / ~244 ly / 2025-05-05 / Davenport et al. 2025 ApJL 984 L44 (arXiv:2501.01498). JWST NIRISS+NIRSpec 0.83–5 μm transmission spectrum reveals haze-free, H2-dominated atmosphere with water (H2O) detected at low mean molecular weight, plus hints of SO2 and CO. First JWST sub-Neptune around a Sun-like star.",
        "✅",
        1.0,
        ["toi-421 b", "toi 421 b", "toi-421b", "toi421b", "TOI-421 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D20",
        "WASP-52 b / ~570 ly / 2025-03-26 / Fournier-Tondreau et al. 2025 MNRAS 539, 422 (arXiv:2412.17072). JWST/NIRISS SOSS transmission spectrum detects H2O at 10.8σ and He at 7.3σ, alongside prominent star-spot crossings. Peer-reviewed.",
        "✅",
        1.0,
        ["wasp-52 b", "wasp 52 b", "wasp-52b", "WASP-52 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D21",
        "HAT-P-12 b / ~465 ly / 2025-07 (arXiv) / 2025-11 (A&A 703 A264) / Crouzet et al. 2025 A&A. JWST NIRSpec G395M + HST/WFC3 transmission spectrum detects CO2 at 12.2σ, CO at 4.1σ, H2O at 6.0σ. Peer-reviewed.",
        "✅",
        1.0,
        ["hat-p-12 b", "hat p 12 b", "hat-p-12b", "HAT-P-12 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D22",
        "HAT-P-26 b / ~437 ly / 2025-09-19 (arXiv) / 2025 AJ 170, 292 / Gressier et al. JWST-TST DREAMS. JWST NIRSpec G395H transmission spectrum detects H2O, CO2 (with high ln B), and SO2 (ln B 13.5). Peer-reviewed.",
        "✅",
        1.0,
        ["hat-p-26 b", "hat p 26 b", "hat-p-26b", "HAT-P-26 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D23",
        "HD 189733 b / ~64.5 ly / 2024-08 / Fu et al. 2024 Nature 631 (DOI 10.1038/s41586-024-07760-y). JWST/NIRSpec transmission spectrum 2.4–5.0 μm detects H2O at 13.4σ, CO2 at 11.2σ, CO at 5σ, H2S at 4.5σ. Peer-reviewed.",
        "✅",
        1.0,
        ["hd 189733 b", "hd189733 b", "hd-189733 b", "HD 189733 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D24",
        "GJ 3470 b / ~96 ly / 2024-07 / Beatty et al. 2024 ApJL 970 L10 (arXiv:2406.04450). JWST NIRCam + HST/WFC3 + Spitzer combined transmission spectrum detects H2O, CH4, SO2, and CO2 each at >3σ. Peer-reviewed.",
        "✅",
        1.0,
        ["gj 3470 b", "gj-3470 b", "gj3470 b", "GJ 3470 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D25",
        "GJ 1214 b / ~48 ly / 2023-05-10 / Kempton et al. 2023 Nature 620, 67 (arXiv:2305.06240). JWST/MIRI phase curve dayside+nightside emission spectra each show >3σ absorption features with H2O as most likely cause; transmission spectrum is featureless due to thick aerosols. Tentative H2O.",
        "⚠️",
        0.0,
        ["gj 1214 b", "gj-1214 b", "gj1214 b", "GJ 1214 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D26",
        "TRAPPIST-1 system planets / ~40 ly / 2023–2025 / multiple JWST observations. As of 2026-03, JWST has obtained transmission/emission spectra for TRAPPIST-1 b, c, e but has NOT robustly detected H2O in any TRAPPIST-1 planet's atmosphere. TRAPPIST-1 b/c are likely airless; TRAPPIST-1 e shows ambiguous secondary atmosphere with no H2O signal reported.",
        "❌",
        -1.0,
        ["trappist-1", "trappist 1", "trappist-1 b", "trappist-1 e", "TRAPPIST-1"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D27",
        "LHS 1140 b / ~48.8 ly / 2024-06 / Cadieux et al. 2024 ApJL 970 L2 (arXiv:2406.15136). JWST/NIRISS transmission spectrum rules out H2-rich atmosphere; finds tentative N2-dominated atmosphere with possible water clouds below transit photosphere. NO direct atmospheric H2O vapor detection in JWST data. Density-based evidence supports interior water, not atmospheric H2O.",
        "❌",
        -1.0,
        ["lhs 1140 b", "lhs-1140 b", "lhs1140 b", "LHS 1140 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D28",
        "WASP-43 b / ~284 ly / 2024-04 / Bell et al. 2024 Nature Astronomy (DOI 10.1038/s41550-024-02230-x); Yang et al. 2024 MNRAS 532 460 (arXiv:2406.03490). JWST/MIRI/LRS phase curve emission shows H2O at 6.5σ alongside NH3 at 4σ (first NH3 detection in hot Jupiter). Peer-reviewed.",
        "✅",
        1.0,
        ["wasp-43 b", "wasp 43 b", "wasp-43b", "WASP-43 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D29",
        "WASP-33 b / ~380 ly / 2024-07 (Yang et al. ApJL 971 L8) and 2026-03 (A&A thermal emission). H2O is detected, but observations are GROUND-BASED (CARMENES, GIANO-B, CFHT/WIRCam) plus archival HST/WFC3 and Spitzer — NOT JWST. The 2026-03 PMO claim cited by the model is not a JWST water detection.",
        "❌",
        -1.0,
        ["wasp-33 b", "wasp 33 b", "wasp-33b", "WASP-33 b"],
    ),
    # web-search-verified 2026-05-02 — added from null_resolutions.json
    (
        "D30",
        "Epsilon Indi Ab / ~12 ly / 2026-04-22 / Matthews et al. 2026 ApJL. JWST/MIRI direct imaging at 11.3 μm vs 10.6 μm shows excess attributed to thick water-ice clouds. BUT (a) this is photometric color inference (indirect), not a spectroscopic atmospheric H2O vapor detection, and (b) the peer-reviewed publication is 2026-04, AFTER the 2026-03 cutoff. Model claim of arXiv:2603.08780 published 2026-03-09 is unverifiable.",
        "❌",
        -1.0,
        ["epsilon indi ab", "eps indi ab", "ε indi ab", "Epsilon Indi Ab"],
    ),
]

DIM_MAP = {
    d[0]: {"id": d[0], "name": d[1], "judgment": d[2], "score": d[3], "kw": d[4]}
    for d in DIMS
}
PART_B_MAX = sum(d[3] for d in DIMS if d[3] > 0)  # = 10.0


# ═══════════════════════════════════════════════════════════════════════════
# Part C — Latest planet baseline
#
# Roles:
#   "main"        — 双主答案：WASP-107 b (2025-12, 211 ly) + HAT-P-30 b/WASP-51 b
#                                                             (2026-01, 679 ly)
#                   命中 → C1 = +2
#   "alt"         — 其他 ✅ baseline 行星（D1-D10）
#                   命中 → C1 = +1
#   "wrong_zero"  — 时间最新但已被排除的（L 98-59 d）或学界争议（K2-18 b）
#                   命中 → C1 = 0
#   "wrong_minus" — 已定论非含水 / 非 JWST（GJ 486 b, HD 209458 b）
#                   命中 → C1 = -1
# 没匹配上 → 交 Stage C summary_facts_verified.json 判定
# ═══════════════════════════════════════════════════════════════════════════

LATEST_BASELINES = {
    # ─── main 主答案 (C1=+2) ───
    "WASP-107 b": {
        "role": "main",
        "announce_date": "2025-12",
        "distance_ly": 211.0,
        "kw": ["wasp-107", "wasp 107", "wasp107"],
    },
    "HAT-P-30 b": {
        "role": "main",
        "announce_date": "2026-01",
        "distance_ly": 679.0,
        "kw": ["hat-p-30", "hat p 30", "hatp30", "wasp-51", "wasp 51", "wasp51"],
    },
    # ─── alt 替代 (C1=+1)，下面映射到 D1-D8 + D10 ───
    "WASP-96 b": {
        "role": "alt",
        "announce_date": "2022-07",
        "distance_ly": 1150.0,
        "kw": ["wasp-96", "wasp 96", "wasp96"],
    },
    "WASP-39 b": {
        "role": "alt",
        "announce_date": "2022-11",
        "distance_ly": 700.0,
        "kw": ["wasp-39", "wasp 39", "wasp39"],
    },
    "WASP-18 b": {
        "role": "alt",
        "announce_date": "2023-05",
        "distance_ly": 400.0,
        "kw": ["wasp-18", "wasp 18", "wasp18"],
    },
    "WASP-80 b": {
        "role": "alt",
        "announce_date": "2024",
        "distance_ly": 163.0,
        "kw": ["wasp-80", "wasp 80", "wasp80"],
    },
    "GJ 9827 d": {
        "role": "alt",
        "announce_date": "2024-09",
        "distance_ly": 98.0,
        "kw": ["gj 9827", "gj9827", "gj-9827"],
    },
    "HAT-P-18 b": {
        "role": "alt",
        "announce_date": "2024",
        "distance_ly": 532.0,
        "kw": ["hat-p-18", "hat p 18", "hatp18"],
    },
    "WASP-121 b": {
        "role": "alt",
        "announce_date": "2025",
        "distance_ly": 880.0,
        "kw": ["wasp-121", "wasp 121", "wasp121"],
    },
    "WASP-166 b": {
        "role": "alt",
        "announce_date": "2025-01",
        "distance_ly": 368.0,
        "kw": ["wasp-166", "wasp 166", "wasp166"],
    },
    # ─── wrong_zero 时间相关但内容错 (C1=0) ───
    "L 98-59 d": {
        "role": "wrong_zero",
        "announce_date": "2026-03",
        "distance_ly": 35.0,
        "kw": ["l 98-59", "l98-59", "l98 59", "l-98-59"],
    },
    "K2-18 b": {
        "role": "wrong_zero",
        "announce_date": "2023",
        "distance_ly": 124.0,
        "kw": ["k2-18", "k2 18", "k218"],
    },
    # ─── wrong_minus 已定论非含水 / 非 JWST (C1=-1) ───
    "GJ 486 b": {
        "role": "wrong_minus",
        "announce_date": "2023",
        "distance_ly": 26.0,
        "kw": ["gj 486", "gj486", "gj-486"],
    },
    "HD 209458 b": {
        "role": "wrong_minus",
        "announce_date": "2001",
        "distance_ly": 159.0,
        "kw": ["hd 209458", "hd209458"],
    },
}

PART_A_MAX = 1.0
PART_C_MAX = 4.0
MAX_SCORE = PART_A_MAX + PART_B_MAX + PART_C_MAX  # = 15.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — DIMS → alignment baseline spec
# ═══════════════════════════════════════════════════════════════════════════


def _derive_match_fields(description: str) -> dict:
    """Parse distance and announce year out of DIMS description for cross-check.

    Description format: "<planet> / <distance> ly / <date> / <notes>"
    """
    fields: dict = {}
    m_dist = _re.search(r"/\s*([\d,]+(?:\.\d+)?)\s*ly\b", description)
    if m_dist:
        fields["distance_ly"] = m_dist.group(1).replace(",", "")
    m_date = _re.search(r"/\s*(\d{4}(?:-\d{2})?)\b", description)
    if m_date:
        fields["announce_date"] = m_date.group(1)
    return fields


def build_baselines() -> list[dict]:
    """Convert DIMS into the baseline spec consumed by pipeline.alignment."""
    out = []
    for d_id, desc, judgment, score, kw in DIMS:
        out.append(
            {
                "id": d_id,
                "description": f"{desc} [基准: {judgment}]",
                "match_fields": _derive_match_fields(desc),
                "kw": kw,
                "judgment": judgment,
                "score": score,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Load extraction results — separate E1 (water_planets) and E2 (summary_facts)
# ═══════════════════════════════════════════════════════════════════════════


def _load_entity_value(out_dir: Path, entity_id: str) -> dict[str, list[dict]]:
    """For each model subdir under out_dir, read extraction.json and return
    the canonical.value list for the named entity."""
    out: dict[str, list[dict]] = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        ext_file = d / "extraction.json"
        if not ext_file.exists():
            continue
        payload = json.loads(ext_file.read_text(encoding="utf-8"))
        entities = payload.get("entities", [])
        match = next((e for e in entities if e.get("id") == entity_id), None)
        if match is None:
            out[d.name] = []
            continue
        cv = (match.get("canonical") or {}).get("value") or []
        if not isinstance(cv, list):
            cv = []
        out[d.name] = cv
    return out


def load_e1_claims(out_dir: Path) -> dict[str, list[dict]]:
    """Load jwst_water_planets list for each model."""
    return _load_entity_value(out_dir, "jwst_water_planets")


def load_e2_facts(out_dir: Path) -> dict[str, dict]:
    """Load summary_facts single record for each model. Returns {} if model
    didn't produce a record."""
    raw = _load_entity_value(out_dir, "summary_facts")
    out: dict[str, dict] = {}
    for model, items in raw.items():
        if items and isinstance(items[0], dict):
            out[model] = items[0]
        else:
            out[model] = {}
    return out


def _load_alignment(out_dir: Path) -> dict[str, list[dict]]:
    """Load existing alignment.json for each model (used with --skip-align)."""
    aligned = {}
    for d in Path(out_dir).iterdir():
        if not d.is_dir():
            continue
        af = d / "alignment.json"
        if not af.exists():
            continue
        aligned[d.name] = json.loads(af.read_text(encoding="utf-8")).get(
            "aligned_claims", []
        )
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# Part A — total exoplanet count scoring
# ═══════════════════════════════════════════════════════════════════════════


def _parse_count(s) -> int | None:
    """Parse a count like '6,150' / '6150' / '~6,150' to int. None on failure."""
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("，", "")
    m = _re.search(r"(\d{4,5})", t)
    if not m:
        return None
    return int(m.group(1))


def score_count(total_count_raw) -> tuple[float, dict]:
    """Part A scoring.
    6050-6200 → +1
    5800-6049 / 6201-6400 → 0
    其他 / 无 → -1
    """
    n = _parse_count(total_count_raw)
    if n is None:
        return -1.0, {
            "score": -1.0,
            "raw": total_count_raw,
            "parsed": None,
            "verdict": "missing_or_unparsable",
        }
    if 6050 <= n <= 6200:
        v = "exact_match"
        s = 1.0
    elif 5800 <= n < 6050 or 6200 < n <= 6400:
        v = "off_by_a_bit"
        s = 0.0
    else:
        v = "wrong_magnitude"
        s = -1.0
    return s, {"score": s, "raw": total_count_raw, "parsed": n, "verdict": v}


# ═══════════════════════════════════════════════════════════════════════════
# Part B — JWST water-signal planet list scoring (open-set with dedup)
# ═══════════════════════════════════════════════════════════════════════════


def score_water_list(aligned_claims: list[dict]) -> tuple[float, dict]:
    """Open-set list scoring: sum DIM scores; first-occurrence-only dedup;
    __HALLUCINATION__ → -1; null → 0 (open set, conservative)."""
    seen: set = set()
    scored: list[dict] = []
    unverified: list[dict] = []
    total = 0.0
    for c in aligned_claims:
        cid = c.get("canonical_id")
        conf = c.get("alignment_confidence", "medium")
        raw = c.get("raw") or {}
        name = raw.get("name", "")
        if conf == "needs_review":
            unverified.append({"name": name, "reason": "needs_review"})
            continue
        if cid == "__HALLUCINATION__":
            scored.append({"id": cid, "name": name, "score": -1.0, "judgment": "❌"})
            total += -1.0
            continue
        if cid and cid in DIM_MAP and cid not in seen:
            seen.add(cid)
            d = DIM_MAP[cid]
            scored.append(
                {
                    "id": cid,
                    "name": name,
                    "matched_baseline": d["name"][:60],
                    "score": d["score"],
                    "judgment": d["judgment"],
                }
            )
            total += d["score"]
        elif cid in seen:
            continue  # dedup
        else:
            unverified.append({"name": name, "canonical_id": cid})
    return total, {
        "score": total,
        "max": PART_B_MAX,
        "hits": scored,
        "unverified_or_null": unverified,
        "n_listed": len(aligned_claims),
        "n_dedup": len(seen),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Part C — Latest planet scoring (composite C1+C2+C3)
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_planet_name(s: str) -> str:
    return _re.sub(r"\s+", " ", str(s or "").strip().lower())


def _match_latest_baseline(planet_name: str) -> tuple[str | None, dict | None]:
    """kw-match planet_name against LATEST_BASELINES; return (key, entry) or
    (None, None)."""
    name = _normalize_planet_name(planet_name)
    if not name:
        return None, None
    # Prefer longest kw match to avoid 'wasp-1' matching 'wasp-18'
    candidates = []
    for key, entry in LATEST_BASELINES.items():
        for kw in entry["kw"] + [_normalize_planet_name(key)]:
            if kw and kw in name:
                candidates.append((len(kw), key, entry))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, k, e = candidates[0]
    return k, e


_MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _parse_announce_date(s) -> tuple[int | None, int | None]:
    """Parse a date string to (year, month_or_None). Handles digits like
    '2025-12' / '2025年12月' and English month names like 'October 2025'."""
    if s is None:
        return None, None
    t = str(s).strip()
    t_low = t.lower()
    # Numeric YYYY-MM or YYYY年MM月 patterns first (most reliable)
    m = _re.search(r"(20\d{2})\D{1,5}(\d{1,2})", t)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            return y, mo
    # English month name + year (or year + month name)
    for name, num in _MONTH_NAMES.items():
        # use word-boundary match
        if _re.search(rf"\b{name}\b", t_low):
            ym = _re.search(r"(20\d{2})", t)
            if ym:
                return int(ym.group(1)), num
    # Fall back to year-only
    m = _re.search(r"(20\d{2})", t)
    if m:
        return int(m.group(1)), None
    return None, None


def _date_within_pm1(model_date, baseline_date) -> bool:
    """True if the two dates are within ±1 month (treating year-only as ±6mo
    centered on July of that year)."""
    my, mm = _parse_announce_date(model_date)
    by, bm = _parse_announce_date(baseline_date)
    if my is None or by is None:
        return False
    if mm is None:
        mm = 6
    if bm is None:
        bm = 6
    diff_months = (my - by) * 12 + (mm - bm)
    return abs(diff_months) <= 1


def _parse_distance(s) -> float | None:
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    m = _re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    return float(m.group(1))


def _distance_within_pct(model_dist, baseline_dist, pct=0.20) -> bool:
    md = _parse_distance(model_dist)
    if md is None or baseline_dist is None:
        return False
    return abs(md - baseline_dist) / baseline_dist <= pct


def score_latest(
    e2_record: dict,
    verified_info: dict | None = None,
) -> tuple[float, dict]:
    """Part C composite scoring.

    e2_record: {latest_planet_name, latest_announce_date, latest_distance_ly}
    verified_info: optional Stage C verdict for the model's latest_planet_name
                   when name doesn't kw-match LATEST_BASELINES, schema:
                   {"verdict": "hallucination|baseline_alt|wrong_zero|wrong_minus",
                    "match_id": <key in LATEST_BASELINES or null>,
                    "actual_announce_date": "...", "actual_distance_ly": ..., "evidence": "..."}

    Returns (total_part_c_score, detail_dict).
    """
    name_raw = e2_record.get("latest_planet_name") if e2_record else None
    date_raw = e2_record.get("latest_announce_date") if e2_record else None
    dist_raw = e2_record.get("latest_distance_ly") if e2_record else None

    if not name_raw:
        return -1.0, {
            "score": -1.0,
            "C1": -1.0,
            "C2": 0.0,
            "C3": 0.0,
            "raw_name": name_raw,
            "raw_date": date_raw,
            "raw_distance": dist_raw,
            "verdict": "no_latest_claim",
            "matched_baseline": None,
        }

    matched_key, matched_entry = _match_latest_baseline(name_raw)

    # If kw match failed, defer to Stage C verified info
    if matched_key is None and verified_info:
        verdict = verified_info.get("verdict")
        match_id = verified_info.get("match_id")
        if match_id and match_id in LATEST_BASELINES:
            matched_key = match_id
            matched_entry = LATEST_BASELINES[match_id]
        elif verdict == "hallucination":
            return -1.0, {
                "score": -1.0,
                "C1": -1.0,
                "C2": 0.0,
                "C3": 0.0,
                "raw_name": name_raw,
                "raw_date": date_raw,
                "raw_distance": dist_raw,
                "verdict": "hallucination",
                "matched_baseline": None,
                "verified": verified_info,
            }
        elif verdict == "wrong_zero":
            # Real planet but no JWST water signal → topic-related misread,
            # similar to L 98-59 d / K2-18 b in LATEST_BASELINES → C1=0, no
            # date/distance check.
            return 0.0, {
                "score": 0.0,
                "C1": 0.0,
                "C2": 0.0,
                "C3": 0.0,
                "raw_name": name_raw,
                "raw_date": date_raw,
                "raw_distance": dist_raw,
                "verdict": "wrong_zero_verified",
                "matched_baseline": None,
                "verified": verified_info,
            }
        elif verdict == "wrong_minus":
            # Real planet but outright wrong category (e.g., GJ 486 b /
            # HD 209458 b style) → C1=-1.
            return -1.0, {
                "score": -1.0,
                "C1": -1.0,
                "C2": 0.0,
                "C3": 0.0,
                "raw_name": name_raw,
                "raw_date": date_raw,
                "raw_distance": dist_raw,
                "verdict": "wrong_minus_verified",
                "matched_baseline": None,
                "verified": verified_info,
            }
        elif verdict == "alt_real_but_not_baseline":
            # Real planet WITH real JWST water paper, just not in
            # LATEST_BASELINES. C1=+1; check C2/C3 against verified
            # actual_announce_date and actual_distance_ly.
            actual_date = verified_info.get("actual_announce_date")
            actual_dist = verified_info.get("actual_distance_ly")
            try:
                actual_dist_f = float(actual_dist) if actual_dist is not None else None
            except (TypeError, ValueError):
                actual_dist_f = _parse_distance(actual_dist)
            c1 = 1.0
            c2 = 1.0 if _date_within_pm1(date_raw, actual_date) else -1.0
            c3 = (
                1.0
                if (actual_dist_f and _distance_within_pct(dist_raw, actual_dist_f))
                else -1.0
            )
            total = c1 + c2 + c3
            return total, {
                "score": total,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "raw_name": name_raw,
                "raw_date": date_raw,
                "raw_distance": dist_raw,
                "verdict": "alt_real_verified",
                "matched_baseline": None,
                "verified_announce_date": actual_date,
                "verified_distance_ly": actual_dist_f,
                "verified": verified_info,
            }

    if matched_key is None:
        # No match, no Stage C verdict — treat as hallucination
        return -1.0, {
            "score": -1.0,
            "C1": -1.0,
            "C2": 0.0,
            "C3": 0.0,
            "raw_name": name_raw,
            "raw_date": date_raw,
            "raw_distance": dist_raw,
            "verdict": "no_match_no_verification",
            "matched_baseline": None,
        }

    role = matched_entry["role"]
    if role == "main":
        c1 = 2.0
    elif role == "alt":
        c1 = 1.0
    elif role == "wrong_zero":
        c1 = 0.0
    elif role == "wrong_minus":
        c1 = -1.0
    else:
        c1 = 0.0

    # C2/C3 only computed if C1 ≥ 1
    if c1 >= 1.0:
        c2 = 1.0 if _date_within_pm1(date_raw, matched_entry["announce_date"]) else -1.0
        c3 = (
            1.0
            if _distance_within_pct(dist_raw, matched_entry["distance_ly"])
            else -1.0
        )
    else:
        c2 = 0.0
        c3 = 0.0

    total = c1 + c2 + c3
    return total, {
        "score": total,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "raw_name": name_raw,
        "raw_date": date_raw,
        "raw_distance": dist_raw,
        "matched_baseline": matched_key,
        "matched_role": role,
        "matched_announce_date": matched_entry["announce_date"],
        "matched_distance_ly": matched_entry["distance_ly"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate per-model score
# ═══════════════════════════════════════════════════════════════════════════


def score_model(
    model_name: str,
    aligned_e1: list[dict],
    e2_record: dict,
    summary_verified: dict | None = None,
) -> dict:
    """Combine Part A + B + C for one model."""
    a_score, a_detail = score_count(e2_record.get("total_count") if e2_record else None)
    b_score, b_detail = score_water_list(aligned_e1)
    verified_info = get_verified_for_model(summary_verified, model_name)
    c_score, c_detail = score_latest(e2_record or {}, verified_info=verified_info)
    total = a_score + b_score + c_score
    return {
        "model": model_name,
        "total_score": total,
        "max_score": MAX_SCORE,
        "score_rate": round(total / MAX_SCORE, 4) if MAX_SCORE else 0.0,
        "part_A_count": a_detail,
        "part_B_water_list": b_detail,
        "part_C_latest": c_detail,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════


def build_scores_json(all_scores: list[dict]) -> dict:
    ranked = sorted(all_scores, key=lambda x: -x["total_score"])
    return {
        "query_id": QUERY_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "max_score": MAX_SCORE,
        "part_max": {
            "A_count": PART_A_MAX,
            "B_water_list": PART_B_MAX,
            "C_latest": PART_C_MAX,
        },
        # dict-form: keyed by model name, matches Q22 reference schema so
        # run_all.extract_total_rates picks it up via the canonical path.
        "results": {s["model"]: s for s in all_scores},
        "ranking": [
            {"rank": i + 1, "model": r["model"], "score": r["total_score"]}
            for i, r in enumerate(ranked)
        ],
    }


def build_ranking_md(all_scores: list[dict]) -> str:
    ranked = sorted(all_scores, key=lambda x: -x["total_score"])
    lines = [
        f"# Query {QUERY_ID} Ranking",
        "",
        f"> Snapshot: {SNAPSHOT_DATE}; MAX_SCORE = {MAX_SCORE}"
        f" (A={PART_A_MAX} + B={PART_B_MAX} + C={PART_C_MAX})",
        "",
        "| Rank | Model | Total | A (count) | B (list) | C (latest) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        a = r["part_A_count"]["score"]
        b = r["part_B_water_list"]["score"]
        c = r["part_C_latest"]["score"]
        lines.append(
            f"| {i} | {r['model']} | {r['total_score']:.1f}/{MAX_SCORE:.0f} "
            f"| {a:+.1f} | {b:+.1f} | {c:+.1f} |"
        )
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Stage C input writers — extra summary_facts_review.json
# ═══════════════════════════════════════════════════════════════════════════


def export_summary_facts_review(e2_facts: dict[str, dict], path: Path) -> int:
    """Dump per-model E2 records into summary_facts_review.json so a Stage C
    agent can WebSearch-verify each model's latest_planet claim. Only emits
    items where latest_planet_name is non-null AND doesn't kw-match an
    existing LATEST_BASELINES entry (those are decided locally in scorer)."""
    items = []
    for model, rec in sorted(e2_facts.items()):
        name = (rec or {}).get("latest_planet_name")
        if not name:
            continue
        matched_key, _ = _match_latest_baseline(name)
        if matched_key is not None:
            continue  # decided locally; no Stage C needed
        items.append(
            {
                "model": model,
                "latest_planet_name": name,
                "latest_announce_date": rec.get("latest_announce_date"),
                "latest_distance_ly": rec.get("latest_distance_ly"),
                "note": rec.get("note"),
            }
        )
    payload = {
        "query_id": QUERY_ID,
        "query_text": QUERY_TEXT,
        "total_items": len(items),
        "items": items,
        "instructions": (
            "对每条 item 用 WebSearch + WebFetch 核实 latest_planet_name 是否真存在，"
            "且是否真有 JWST 探测到水信号的 paper（截至 2026-03）。"
            "对每条输出 verdict ∈ {hallucination | wrong_zero | wrong_minus | "
            "alt_real_but_not_baseline | matches_existing_baseline}，"
            "若是 matches_existing_baseline 则给 match_id（LATEST_BASELINES 的 key），"
            "若是真实新行星但不在 baseline 则建议 baseline_add 信息。"
            "公认日期参考：WASP-107 b=2025-12, HAT-P-30 b=2026-01, "
            "WASP-166 b=2025-01；L 98-59 d 是 2026-03 但属反驳水世界 paper。"
        ),
        "schema_for_resolutions": (
            "summary_facts_verified.json 形如 "
            '{"<model>": {"latest_planet": {"verdict": "...", "match_id": "..."|null, '
            '"actual_announce_date": "...", "actual_distance_ly": ..., "evidence": "..."}}}'
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items)


def load_summary_facts_verified(path: Path) -> dict | None:
    """Load Stage C output for E2. Schema: {<model>: {"latest_planet": {...}}}.
    For convenience also accepts top-level {"items": [...]} form (will be
    re-keyed by model)."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        out = {}
        for it in payload["items"]:
            m = it.get("model")
            if not m:
                continue
            out[m] = {"latest_planet": {k: v for k, v in it.items() if k != "model"}}
        return out
    if isinstance(payload, dict):
        # already in {<model>: {...}} form — but flatten if "latest_planet" key
        # is missing (some agents may put fields directly under model)
        out = {}
        for k, v in payload.items():
            if not isinstance(v, dict):
                continue
            if "latest_planet" in v:
                out[k] = v
            else:
                out[k] = {"latest_planet": v}
        return out
    return None


def get_verified_for_model(summary_verified: dict | None, model: str) -> dict | None:
    if not summary_verified:
        return None
    entry = summary_verified.get(model)
    if not entry:
        return None
    return entry.get("latest_planet")


# ═══════════════════════════════════════════════════════════════════════════
# main()
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Query {QUERY_ID} auto-scorer")
    ap.add_argument("--models", nargs="+", required=True, help="name=path list")
    ap.add_argument("--output-dir", default=str(THIS / "auto_scores"))
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--aligner-models", nargs="+", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Extract (both entities) ─────────────────────────────────
    if not args.skip_extract:
        run_pipeline(
            query_id=QUERY_ID,
            query_text=QUERY_TEXT,
            entities=ENTITIES,
            prompt_hints=PROMPT_HINTS,
            schema=VALUE_SCHEMA,
            models_input=args.models,
            output_dir=out_dir,
            concurrency=args.concurrency,
        )

    # ── Stage 2: Align (only E1) ─────────────────────────────────────────
    baselines = build_baselines()
    if args.skip_align:
        aligned_e1 = _load_alignment(out_dir)
    else:
        client = get_client()
        overrides = {}
        if args.aligner_models:
            overrides["aligner_models"] = args.aligner_models
        if args.judge_model:
            overrides["judge_model"] = args.judge_model
        if args.concurrency:
            overrides["concurrency"] = args.concurrency
        e1_claims = load_e1_claims(out_dir)
        aligned_e1 = align_claims(
            client,
            claims_by_model=e1_claims,
            baselines=baselines,
            query_text=QUERY_TEXT,
            output_dir=out_dir,
            overrides=overrides,
        )

    # ── Stage 3a: Export null_review for E1 ──────────────────────────────
    null_items = export_null_claims_for_review(
        aligned_e1,
        out_dir / "null_review.json",
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        models_input=args.models,
    )
    print(f"[*] Exported {len(null_items)} E1 null/needs_review → null_review.json")

    # ── Stage 3b: Apply E1 null_resolutions if present ───────────────────
    resolutions_path = out_dir / "null_resolutions.json"
    if resolutions_path.exists():
        print("[*] Found null_resolutions.json, applying …")
        new_baselines = apply_null_resolutions(
            aligned_e1, resolutions_path, dims_ref=DIMS
        )
        if new_baselines:
            print(f"[*] {len(new_baselines)} new E1 baseline entries")
            for e in new_baselines:
                DIM_MAP[e["id"]] = {
                    "id": e["id"],
                    "name": e.get("description", e["id"]),
                    "judgment": e.get("judgment", "⚠️"),
                    "score": e.get("score", 0.0),
                    "kw": e.get("kw", []),
                }
            persist_new_baseline_entries(new_baselines, __file__)
    else:
        print("[*] No null_resolutions.json (run Stage C agent for E1).")

    # ── Stage 3c: Load E2 + export summary_facts_review for Stage C ──────
    e2_facts = load_e2_facts(out_dir)
    n_e2_pending = export_summary_facts_review(
        e2_facts, out_dir / "summary_facts_review.json"
    )
    print(
        f"[*] {len(e2_facts)} models have E2; "
        f"{n_e2_pending} need Stage C latest_planet verification "
        f"→ summary_facts_review.json"
    )

    # ── Stage 3d: Load E2 verified ───────────────────────────────────────
    summary_verified = load_summary_facts_verified(
        out_dir / "summary_facts_verified.json"
    )
    if summary_verified:
        print(
            f"[*] Loaded summary_facts_verified.json ({len(summary_verified)} models)"
        )
    else:
        print("[*] No summary_facts_verified.json (Stage C may need to run)")

    # ── Stage 4: Score all parts ─────────────────────────────────────────
    all_scores = []
    all_models = sorted(set(list(aligned_e1.keys()) + list(e2_facts.keys())))
    for model in all_models:
        s = score_model(
            model,
            aligned_e1.get(model, []),
            e2_facts.get(model, {}),
            summary_verified=summary_verified,
        )
        all_scores.append(s)

    (out_dir / "scores.json").write_text(
        json.dumps(build_scores_json(all_scores), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ranking_report.md").write_text(
        build_ranking_md(all_scores), encoding="utf-8"
    )

    print(f"\nQuery {QUERY_ID} scoring done.")
    for i, r in enumerate(sorted(all_scores, key=lambda x: -x["total_score"]), 1):
        a = r["part_A_count"]["score"]
        b = r["part_B_water_list"]["score"]
        c = r["part_C_latest"]["score"]
        print(
            f"  {i}. {r['model']:30s} {r['total_score']:>5.1f}/{MAX_SCORE:.0f}"
            f" (A={a:+.0f} B={b:+.1f} C={c:+.0f})"
        )


if __name__ == "__main__":
    main()
