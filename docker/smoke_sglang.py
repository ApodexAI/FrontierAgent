#!/usr/bin/env python3
"""Secret-free infrastructure and structured-tool-call smoke for SGLang."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable


BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
MODEL = os.environ.get("SGLANG_SERVED_MODEL_NAME", "local-model")


def request(path: str, payload: dict | None = None) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


Request = Callable[[str, dict | None], object]


def main(http_request: Request = request) -> int:
    http_request("/health", None)
    print("PASS  SGLang health endpoint")

    models = http_request("/v1/models", None)
    if not isinstance(models, dict) or not models.get("data"):
        raise RuntimeError("/v1/models returned no served models")
    print("PASS  OpenAI-compatible model listing")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Call the calculator tool to compute 123 + 456. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate an arithmetic expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    result = http_request("/v1/chat/completions", payload)
    try:
        calls = result["choices"][0]["message"]["tool_calls"]  # type: ignore[index]
        call = calls[0]
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("model did not return a structured tool call") from exc
    if name != "calculator" or "expression" not in arguments:
        raise RuntimeError(f"unexpected tool call: {name}")
    print("PASS  structured qwen3_coder tool call")

    # Keep the structured tool-call probe deterministic, then make a separate
    # non-greedy request. Some SGLang/FlashInfer sampling kernels are compiled
    # lazily and need CUDA development headers such as curand.h; a
    # temperature-zero smoke can stay green while the first real agent request
    # crashes the scheduler during that JIT build.
    sampling_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 16,
    }
    sampled = http_request("/v1/chat/completions", sampling_payload)
    try:
        sampled_content = sampled["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("non-greedy sampling returned no message content") from exc
    if not isinstance(sampled_content, str) or not sampled_content.strip():
        raise RuntimeError("non-greedy sampling returned empty message content")
    print("PASS  non-greedy sampling kernel")
    print("NOTE  This verifies plumbing and parsing, not general agent correctness.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
