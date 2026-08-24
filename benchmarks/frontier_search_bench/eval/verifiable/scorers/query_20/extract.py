"""
Query 20 — Song identification from musical clues.
T1 granularity: one entity holding the candidate list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from pipeline.extraction_pipeline import run_pipeline  # noqa: E402

QUERY_ID = 20
QUERY_TEXT = (
    "I've had an earworm for a few days, but I can't remember the exact name of "
    "the song. I can replay it on the guitar with a capo on the 3rd fret, using "
    "a repeating chord progression of C–G–Am–F in the intro and verses. The tune "
    "shifts to a slightly higher arpeggio pattern in the bridge. The overall tone "
    "is very melancholic and reflective. I also remember the singer is a man in "
    "his 30s or 40s (can't be sure). Can you help me identify (or narrow down) "
    "the song?"
)

ENTITIES = [
    {
        "id": "song_candidates",
        "name": "候选歌曲列表（每个候选含歌曲名和歌手）",
    },
]

PROMPT_HINTS = {
    "song_candidates": (
        "提取模型推荐的所有候选歌曲。每首歌给出：title（歌曲名）+ artist（歌手）。"
        "若模型只给了一个答案也列为长度=1 的数组。若模型只讨论了音乐理论却未给出任何歌曲，"
        "返回空数组并置 not_mentioned=false（视为模型未答）。"
    ),
}

VALUE_SCHEMA = """{
  "value": {
    "candidates": [
      {"title": "<歌曲名>", "artist": "<歌手>"},
      ...
    ]
  },
  "not_mentioned": <true|false>,
  "supporting_span": "<原文片段 30-200 字>",
  "confidence": "<high|medium|low>"
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--secondary-model", default=None)
    ap.add_argument("--parallel-models", nargs="+", default=None)
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    run_pipeline(
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        entities=ENTITIES,
        prompt_hints=PROMPT_HINTS,
        schema=VALUE_SCHEMA,
        models_input=args.models,
        output_dir=Path(args.output_dir),
        primary=args.primary_model,
        secondary=args.secondary_model,
        parallel=args.parallel_models,
        analyzer=args.analyzer_model,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
