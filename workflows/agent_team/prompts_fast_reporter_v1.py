"""Prompts for the fast reporter's evidence review and cited writing calls.

The reviewer selects candidates by index, making invented source URLs
unrepresentable, and repairs weak URL-derived titles when needed.
"""

from __future__ import annotations

REVIEW_SYSTEM_PROMPT = """You review candidate sources for a research \
report's reference table. You do not write or edit the report — you only \
judge which sources belong in the table, what each should be called, and \
how much weight it deserves.

For each candidate you keep, emit three things:

* `"i"` — the candidate's 1-based number, exactly as shown in the list. \
Never emit a URL; the system already has it.
* `"title"` — how this source should be labelled in the reference list. \
Copy the candidate's own title when it is a real page title. Replace it \
only when it is a bare hostname, a URL slug, or otherwise says nothing \
about the page. Keep it under 120 characters, and never end it with a \
period.
* `"quality"` — one of `high`, `medium`, `low`, from the tiering below.

# Quality tiers

* `high` — official and primary sources (company filings, official sites, \
regulator documents, original papers), plus major media and industry \
research (e.g. Reuters, TrendForce, Counterpoint, Gartner).
* `medium` — retail pages and aggregators.
* `low` — forums and social (e.g. Reddit, Threads, 知乎专栏, CSDN, 今日头条, \
搜狐, 百家号, 简书), and forecast or quant-score sites (e.g. \
WalletInvestor, CoinCodex, SimplyWallSt, TradersUnion).

# What to keep

Relevance first: a source that is topically adjacent but never states one \
of the question's or draft's factual claims earns no slot, whatever its \
tier. Then credibility: prefer a smaller, harder table over a full one \
padded with weak sources. The limit you are given is a ceiling, not a \
quota.

Drop a candidate by simply leaving it out of `"nodes"`. Keep a `low` \
source only when nothing stronger in the list covers the same claim, and \
never let it crowd out a stronger source that covers that claim.

Respond with ONLY a JSON object, nothing else: no prose, no markdown \
fence, no explanation, and no keys besides `"nodes"`.
"""


def build_review_prompt(
    *,
    question: str,
    native_draft: str,
    candidates_block: str,
    cap: int,
) -> str:
    """Assemble the user turn for call 1.

    Pairs with :data:`REVIEW_SYSTEM_PROMPT`, which holds the fixed tiering
    and philosophy. This supplies the per-call content and restates the
    exact output shape last, immediately before the model replies.
    """
    return (
        f"# Question\n\n{question}\n\n"
        f"# Draft answer\n\n{native_draft}\n\n"
        f"# Candidate sources\n\n{candidates_block}\n\n"
        "# Your task\n\nReview the numbered candidates above and keep the "
        "ones worth citing, by relevance to the question and draft plus "
        f"source credibility. Keep at most {cap} — fewer is better; do not "
        f"pad the list to reach {cap}.\n\n"
        'Respond with ONLY this JSON object and nothing else:\n'
        '{"nodes": [{"i": 1, "title": "...", "quality": "high"}, ...]}\n\n'
        '"i" is the 1-based candidate number from the list above. "quality" '
        'is exactly one of "high", "medium", "low". No other text, no '
        "markdown fence, no explanation."
    )


WRITER_ADDENDUM = """

# Citation density — one marker per source per block

Within a single bounded block — one bullet or numbered list, one table, one
paragraph — cite each source **once**, at its first occurrence, and leave the
later mentions unmarked. The reader carries the attribution forward; a
repeated `[N]` down a list adds no information and crowds the block.

- List:  `- AAA [1]` / `- BBB [1]` / `- CCC [1]`
      →  `- AAA [1]` / `- BBB`     / `- CCC`
- Table: if a whole row or column comes from one source, mark it once in the
  row label or column header and leave the other cells clean.
- Paragraph: mark the first sentence the source supports, not every sentence.

This still applies when every bullet states a **different number or fact**
pulled from the same source — a different number is not a different source,
and does not earn its own marker:

    Bad:                              Good:
    - Total citations: 1,541 [1]      - Total citations: 1,541 [1]
    - h-index: 22 [1]                 - h-index: 22
    - i10-index: 39 [1]               - i10-index: 39

A new block starts the count over: if the next list or paragraph also draws
on `[1]`, mark its first occurrence there again.

This never changes WHICH sources you cite — only how often you repeat a
marker already placed in the same block. If two different sources support
different items in one list, each still gets its own first-occurrence marker.
"""


__all__ = [
    "REVIEW_SYSTEM_PROMPT",
    "WRITER_ADDENDUM",
    "build_review_prompt",
]
