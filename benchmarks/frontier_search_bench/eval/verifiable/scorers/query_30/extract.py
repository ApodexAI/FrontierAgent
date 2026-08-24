"""Query 30 — Alan Yuille academic genealogy traced back to the 18th century.

Question (Chinese): build the academic genealogy with Alan Yuille as the leaf
node, traced back to the 18th century.

Reference: Mathematics Genealogy Project (mathgenealogy.org).
  Core front: Yuille → Hawking → Sciama → Dirac → Fowler.
  Two valid branches above Fowler:
    - Rutherford branch: Fowler ← Rutherford ← Thomson ← Rayleigh ← Routh ← …
                         ← T. Jones ← Postlethwaite ← Whisson ← W. Taylor ←
                         R. Smith ← Cotes ← Newton ← Barrow.
                         (common error: Thomson → Routh, skipping Rayleigh)
    - Hill branch: Fowler ← A.V. Hill ← Fletcher ← Langley ← Foster ← Huxley
                   ← T.W. Jones ← Mackenzie ← Beer ← Barth ← von Störck
                   ← van Swieten ← Boerhaave.

Either branch reaching the 18th century is acceptable.

T1 granularity — one composite value (chain links + 18th-century names +
Fowler-upstream branch flag).
"""

from __future__ import annotations

QUERY_ID = 30

QUERY_TEXT = "建立以alan yuille为叶子结点的学术谱系，向上追溯到18世纪"

ENTITIES = [
    {
        "id": "yuille_genealogy",
        "name": (
            "模型给出的 Alan Yuille 学术谱系（导师链 + 18 世纪人物列表 + "
            "Fowler 上游分支选择）"
        ),
    }
]

PROMPT_HINTS = {
    "yuille_genealogy": (
        "请从模型回答中抽取关于 Alan Yuille 学术谱系的核心信息。\n\n"
        "**抽取规则：**\n"
        "- **advisor_chain**：从 Alan Yuille 起按'学生→导师'顺序的人名列表"
        "（例如 ['Alan Yuille', 'Stephen Hawking', 'Dennis Sciama', "
        "'Paul Dirac', 'Ralph Fowler', 'Ernest Rutherford', ...]）。\n"
        "  顺序与模型给出的一致；若模型给的是树形结构，按主要分支拉直。\n"
        "  若模型完全没给出可识别的导师链，留空数组。\n"
        "- **century_18_names**：模型明确列出的 18 世纪（约 1700-1799 年生"
        "或活跃于 18 世纪）的人物名字。常见的有 Gerard van Swieten / "
        "Anton von Störck / Joseph Barth / Roger Cotes / Robert Smith / "
        "Walter Taylor / Stephen Whisson / Thomas Postlethwaite / Thomas "
        "Jones / Herman Boerhaave / Georg Joseph Beer 等。\n"
        "- **fowler_branch**：Fowler 上游模型走的分支：\n"
        "    'rutherford'  — Fowler ← Ernest Rutherford ← J.J. Thomson 路线\n"
        "    'hill'        — Fowler ← A.V. Hill ← Fletcher 路线\n"
        "    'both'        — 模型同时讨论了两条分支\n"
        "    'unclear' / null — 模型没给出明确的 Fowler 上游\n"
        "- **reaches_18th_century**：链条是否实际追溯到了 18 世纪人物。\n"
        "- 模型在反例 / 排除语境提到的人物不抽。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "advisor_chain": ["<名字字符串>", ...],
    "century_18_names": ["<名字字符串>", ...],
    "fowler_branch": "<rutherford | hill | both | unclear | null>",
    "reaches_18th_century": <true | false>
  },
  "not_mentioned": <true 仅当模型完全没建立任何谱系；否则 false>,
  "supporting_span": "<原文片段 30-200 字，证明上面字段确实来自模型回答>",
  "confidence": "<high|medium|low>"
}"""
