"""Detect near-verbatim repetition across recent assistant turns.

Word-bigram similarity targets repeated wording, not semantic paraphrases;
stopping is opt-in so false positives default to a corrective hint only.
"""

from __future__ import annotations

import re
from collections import deque

from frontier_agent.core.loop_types import Intervention, LoopConfig, TurnContext
from frontier_agent.utils.language import detect_language

_WHITESPACE_RE = re.compile(r"\s+")
# Strip ASCII punctuation so "pending." and "pending" hash to the same token.
_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~]+")


def _normalise(text: str) -> str:
    """Lower-case + collapse whitespace. We compare semantic text, not
    formatting jitter."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _shingles(text: str, n: int) -> set[str]:
    """Return token-level shingles of ``text``.

    Uses whitespace-tokenised n-grams (sliding windows of ``n`` tokens)
    so similarity is robust to small wording edits like
    "in this exact format" → "in the format below" — char-trigrams
    over-react to such local edits because every trigram crossing the
    edit boundary changes. For CJK text the whitespace split degrades
    to one "token" per chunk which is still adequate as the run is
    long enough that punctuation-segmented chunks differ across turns.
    """
    cleaned = _PUNCT_RE.sub(" ", text)
    tokens = cleaned.split()
    if not tokens:
        return set()
    if n <= 1 or len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


# Hint templates keyed by detect_language() output. English is the
# default — non-English models still understand it, but matching the
# agent's working language lowers the risk that the LLM interprets the
# hint as user-supplied content in a foreign tongue.
_HINT_TEMPLATES: dict[str, str] = {
    "Simplified Chinese": (
        "你最近的几次回复与前几轮高度雷同(连续 {n} 轮 ≥ {thr:.0%} 词汇重合)。"
        "你似乎卡在等待永远不会到达的输入。请使用你已经掌握的信息推进:"
        "撰写部分报告、换一个工具、或直接调用终止/最终化动作。"
    ),
    "Japanese": (
        "直近の応答が前のターンとほぼ重複しています({n} 回連続で ≥ {thr:.0%} の語彙重複)。"
        "届かない入力を待ち続けているように見えます。今ある情報で進めてください:"
        "部分的なレポートを書く、別のツールを使う、または最終化アクションを呼び出す。"
    ),
    "Korean": (
        "최근 응답이 이전 턴과 거의 동일합니다({n}회 연속 ≥ {thr:.0%} 어휘 중복)."
        " 도착하지 않을 입력을 기다리고 있는 것 같습니다. 이미 가진 정보로"
        " 진행하세요: 부분 보고서를 작성하거나, 다른 도구를 선택하거나,"
        " 종료/최종화 액션을 호출하세요."
    ),
}

_DEFAULT_HINT_TEMPLATE = (
    "Your last several responses are near-duplicates of earlier turns "
    "(≥{thr:.0%} word similarity for {n} turns in a row). You appear to "
    "be stuck waiting for inputs that will not arrive. Make progress "
    "with the information you already have: write a partial report, "
    "choose a different tool, or call your terminal/finalize action."
)


def _localised_hint(ai_text: str, *, threshold: float, streak: int) -> str:
    """Pick the hint template matching the agent's working language.

    Falls back to the English template when ``detect_language`` returns a
    label we don't have a template for.
    """
    label = detect_language(ai_text)
    template = _HINT_TEMPLATES.get(label, _DEFAULT_HINT_TEMPLATE)
    return template.format(thr=threshold, n=streak)


class TextRepetitionGuard:
    """Detect near-identical AI text across consecutive turns.

    Critical observer: intervention return values are awaited and
    collected by the agent loop.
    """

    critical = True

    def __init__(
        self,
        *,
        window_size: int = 4,
        similarity_threshold: float = 0.85,
        min_chars: int = 60,
        shingle_size: int = 2,
        inject_after: int = 4,
        stop_after: int = 6,
        enable_stop: bool = False,
        hint_message: str = "",
    ) -> None:
        if inject_after < 2:
            raise ValueError("inject_after must be ≥ 2")
        if stop_after < inject_after:
            raise ValueError("stop_after must be ≥ inject_after")
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.min_chars = min_chars
        self.shingle_size = shingle_size
        self.inject_after = inject_after
        self.stop_after = stop_after
        self.enable_stop = enable_stop
        # Verbatim replacement for the built-in hint. The default advice ends
        # with "call your terminal/finalize action", which is wrong for an
        # agent whose repetition IS a correct wait state — a coordinator
        # polling running sub-agents would be told to finalize early. Callers
        # in that position pass a hint that fits their wait instead.
        self.hint_message = hint_message.strip()

        self._history: deque[set[str]] = deque(maxlen=window_size)
        self._consecutive_matches = 0
        self._hint_injected = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._history.clear()
        self._consecutive_matches = 0
        self._hint_injected = False

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        text = _normalise(ctx.ai_text or "")
        if len(text) < self.min_chars:
            # Too short to be meaningful repetition — keep window pristine.
            return None

        current = _shingles(text, self.shingle_size)
        matched = any(
            _jaccard(current, prior) >= self.similarity_threshold
            for prior in self._history
        )
        self._history.append(current)

        if not matched:
            # Reset to 1: the current turn is itself the seed of a
            # potential new run-of-duplicates, so the next matching
            # turn brings the streak to 2 (not 1).
            self._consecutive_matches = 1
            self._hint_injected = False
            return None

        self._consecutive_matches += 1

        if self.enable_stop and self._consecutive_matches >= self.stop_after:
            return Intervention(
                stop_reason="cross_turn_repetition",
            )

        if self._consecutive_matches >= self.inject_after and not self._hint_injected:
            self._hint_injected = True
            hint = self.hint_message or _localised_hint(
                ctx.ai_text or "",
                threshold=self.similarity_threshold,
                streak=self._consecutive_matches,
            )
            return Intervention(inject_messages=[hint])

        return None
