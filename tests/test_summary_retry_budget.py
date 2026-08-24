"""A failed summary must not be retried blindly, nor be unbounded in time.

`_generate_summary` used to be a single call with no retry: any failure returned
the fallback layout, which on a long research run is the whole point of the
compaction being thrown away. The naive repair — retry twice — is worse than it
looks, because one summariser call can occupy the full LLM timeout (600s in the
shipped profiles). So the two properties that matter are inseparable: retry only
what can succeed on a second attempt, and bound the sequence in wall time.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from frontier_agent.core.runtime.loop import compact_llm
from frontier_agent.core.runtime.loop.compact_llm import (
    LLMSummaryCompactor,
    is_transient_summary_error,
)
from frontier_agent.core.runtime.loop.tiered_compact import TieredCompactor


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the real backoff. Every case here asserts on attempt counts or on
    a ceiling being respected; none of them is about how long 2s is."""
    monkeypatch.setattr(compact_llm, "_RETRY_BACKOFF_BASE_S", 0.01)
    monkeypatch.setattr(compact_llm, "_RETRY_BACKOFF_CAP_S", 0.02)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _LLM:
    """Fails `fail_times` times with `exc`, then succeeds."""

    def __init__(self, exc: Exception, fail_times: int = 99) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    async def chat(self, _messages: object, **_kw: object) -> _Resp:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return _Resp("a real summary of the earlier turns")


def _history(n: int = 12) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 400})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 400})
    return msgs


# ── classification ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "Error code: 400 - context length exceeded",
    "maximum context length is 32768 tokens",
    "invalid_request_error: input too large",
    "413 Payload Too Large",
    "401 unauthorized",
])
def test_a_deterministic_failure_is_not_worth_retrying(message: str) -> None:
    """These fail again identically, and the second failure costs the same
    minutes as the first."""
    assert is_transient_summary_error(RuntimeError(message)) is False


@pytest.mark.parametrize("message", [
    "429 Too Many Requests",
    "503 Service Unavailable",
    "upstream connection reset",
    "the request timed out",
    "model is overloaded",
])
def test_a_transient_failure_is_worth_one_more_attempt(message: str) -> None:
    assert is_transient_summary_error(RuntimeError(message)) is True


@pytest.mark.parametrize("message", [
    "Error code: 400 - code=bad_response_status_code, upstream 503",
    "Error code: 400 - type=new_api_error, upstream timeout",
])
def test_a_proxy_wrapped_transient_400_is_retried(message: str) -> None:
    """A gateway's outer 400 must not hide the recoverable upstream error."""
    assert is_transient_summary_error(RuntimeError(message)) is True


def test_a_proxy_wrapped_transient_in_a_structured_body_is_retried() -> None:
    class _ProxyError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("Bad request")
            self.status_code = 400
            self.body = {
                "error": {
                    "code": "bad_response_status_code",
                    "message": "upstream 503",
                },
            }

    assert is_transient_summary_error(_ProxyError()) is True


def test_an_unrecognised_failure_is_treated_as_transient() -> None:
    """The bias is deliberate: one extra attempt is cheaper than losing the
    research history. `retry_total_timeout_s` is what keeps the bias bounded."""
    assert is_transient_summary_error(RuntimeError("something new")) is True


def test_cancellation_is_never_retried() -> None:
    """Retrying a cancellation fights the caller that asked us to stop."""
    assert is_transient_summary_error(asyncio.CancelledError()) is False


# ── retry behaviour ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_and_can_succeed() -> None:
    llm = _LLM(RuntimeError("503 Service Unavailable"), fail_times=1)
    compactor = LLMSummaryCompactor(
        summary_llm=llm, max_transient_retries=2, retry_total_timeout_s=30,
    )

    out = await compactor.compact(_history(), keep_recent=4)

    assert llm.calls == 2
    assert any("a real summary" in str(m.get("content")) for m in out), (
        "the retry succeeded, so its summary must be what lands in the history"
    )


@pytest.mark.asyncio
async def test_a_deterministic_failure_costs_exactly_one_attempt() -> None:
    """The saving this classification exists for: two extra calls against an
    endpoint that cannot answer, each able to run for the full LLM timeout."""
    llm = _LLM(RuntimeError("400 context length exceeded"))
    events: list[dict] = []
    compactor = LLMSummaryCompactor(
        summary_llm=llm,
        max_transient_retries=2,
        retry_total_timeout_s=30,
        emit_event=events.append,
    )

    await compactor.compact(_history(), keep_recent=4)

    assert llm.calls == 1
    assert events[-1]["rollback_reason"] == "llm_error_permanent", (
        "a permanent failure must be distinguishable in the record from having "
        "run out of attempts — only the former means the target is unreachable"
    )
    assert events[-1]["attempts"] == 1


@pytest.mark.asyncio
async def test_a_transient_failure_stops_at_the_retry_count() -> None:
    llm = _LLM(RuntimeError("429 Too Many Requests"))
    events: list[dict] = []
    compactor = LLMSummaryCompactor(
        summary_llm=llm,
        max_transient_retries=2,
        retry_total_timeout_s=30,
        emit_event=events.append,
    )

    await compactor.compact(_history(), keep_recent=4)

    assert llm.calls == 3, "one initial attempt plus two retries"
    assert events[-1]["rollback_reason"] == "llm_error"
    assert events[-1]["attempts"] == 3


# ── the time ceiling is a ceiling ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_ceiling_bounds_a_single_hanging_attempt() -> None:
    """The property a between-attempts check cannot give you.

    A summariser call carries its own transport timeout, which is far larger
    than the retry budget. Checking the deadline only *after* an attempt returns
    means one hanging call blows through the ceiling entirely, so each attempt
    is wrapped in `asyncio.wait_for`.
    """

    class _Hangs:
        calls = 0

        async def chat(self, _messages: object, **_kw: object) -> _Resp:
            _Hangs.calls += 1
            await asyncio.sleep(60)
            return _Resp("never reached")

    compactor = LLMSummaryCompactor(
        summary_llm=_Hangs(), max_transient_retries=2, retry_total_timeout_s=0.2,
    )

    started = time.monotonic()
    out = await compactor.compact(_history(), keep_recent=4)
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"ceiling was 0.2s, sequence took {elapsed:.1f}s"
    assert out, "a blown ceiling must still return a usable history"


@pytest.mark.asyncio
async def test_a_retry_whose_backoff_alone_exceeds_the_budget_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sleeping out the remaining budget and *then* returning the same failure
    is strictly worse than returning now — the ceiling is meant to be strict,
    not approximate."""
    monkeypatch.setattr(compact_llm, "_RETRY_BACKOFF_BASE_S", 2.0)
    monkeypatch.setattr(compact_llm, "_RETRY_BACKOFF_CAP_S", 5.0)
    llm = _LLM(RuntimeError("503 Service Unavailable"))
    compactor = LLMSummaryCompactor(
        summary_llm=llm, max_transient_retries=5, retry_total_timeout_s=0.05,
    )

    started = time.monotonic()
    await compactor.compact(_history(), keep_recent=4)
    elapsed = time.monotonic() - started

    assert elapsed < 2, f"backoff is 2s+, so it must not have been slept: {elapsed:.1f}s"
    assert llm.calls < 6


# ── the default must not change for existing callers ─────────────────────────


@pytest.mark.asyncio
async def test_a_bare_caller_still_gets_exactly_one_attempt() -> None:
    """`apodex.session`, `apodex.task_runner` and `session_history` construct
    this class without a time ceiling. A retry default of anything but zero
    would hand them retries that nothing bounds in time.
    """
    llm = _LLM(RuntimeError("503 Service Unavailable"))
    compactor = LLMSummaryCompactor(summary_llm=llm)

    await compactor.compact(_history(), keep_recent=4)

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_tiered_without_a_ceiling_does_not_retry_either() -> None:
    """The two knobs are not independent. A retry count with no time bound is
    the same defect one layer up, so the half-specified pair resolves to the
    conservative reading rather than to an unbounded one.
    """
    llm = _LLM(RuntimeError("503 Service Unavailable"))
    compactor = TieredCompactor(
        keep_tool_result=2,
        summary_llm=llm,
        relief_target=1,
        summary_retries=2,
        summary_retry_timeout_s=None,
    )

    await compactor.compact(_history(40), keep_recent=4)

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_tiered_with_a_ceiling_does_retry() -> None:
    llm = _LLM(RuntimeError("503 Service Unavailable"), fail_times=1)
    compactor = TieredCompactor(
        keep_tool_result=2,
        summary_llm=llm,
        relief_target=1,
        summary_retries=2,
        summary_retry_timeout_s=30,
    )

    await compactor.compact(_history(40), keep_recent=4)

    assert llm.calls == 2
    assert compactor.last_event is not None
    assert compactor.last_event.attempts == 2
