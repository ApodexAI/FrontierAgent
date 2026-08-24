from __future__ import annotations

import runpy
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module() -> dict:
    return runpy.run_path(str(ROOT / "docker/smoke_sglang.py"))


def test_smoke_accepts_health_models_and_structured_tool_call(capsys) -> None:
    module = _module()
    responses: Iterator[object] = iter(
        (
            {},
            {"data": [{"id": "local-model"}]},
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":"123 + 456"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "OK"}}]},
        )
    )

    requests: list[tuple[str, dict | None]] = []

    def fake_request(_path: str, _payload: dict | None = None) -> object:
        requests.append((_path, _payload))
        return next(responses)

    assert module["main"](fake_request) == 0
    rendered = capsys.readouterr().out
    assert "PASS  SGLang health endpoint" in rendered
    assert "PASS  structured qwen3_coder tool call" in rendered
    assert "PASS  non-greedy sampling kernel" in rendered
    assert "not general agent correctness" in rendered
    sampling_payload = requests[-1][1]
    assert sampling_payload is not None
    assert sampling_payload["temperature"] > 0
    assert sampling_payload["seed"] == 0


def test_smoke_rejects_plain_text_pseudo_tool_call() -> None:
    module = _module()
    responses: Iterator[object] = iter(
        (
            {},
            {"data": [{"id": "local-model"}]},
            {"choices": [{"message": {"content": "calculator(123 + 456)"}}]},
        )
    )

    def fake_request(_path: str, _payload: dict | None = None) -> object:
        return next(responses)

    with pytest.raises(RuntimeError, match="structured tool call"):
        module["main"](fake_request)
