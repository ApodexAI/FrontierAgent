#!/usr/bin/env python3
"""Check the LLM config a benchmark run will use, with one cheap call.

A wrong provider or an unsupported sampling parameter fails identically on
every question, so a 20-question run spends 20 workers to learn one fact.
This makes a single 1-token call through the same code path first.

    python tools/preflight.py                       # stateful-react-agent / keep5
    python tools/preflight.py --pipeline agent_team --profile benchmark

Exits non-zero with the fix, not just the error. Secrets are never printed —
only whether each key is set.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

AH = Path(__file__).resolve().parents[1]


def report_env() -> None:
    """Report the resolved config, not raw os.environ.

    config._ENV_PATH is Path(".env") — relative to cwd — and the loader
    tolerates malformed lines, so os.environ is not a reliable mirror of
    what the run will actually use.
    """
    from frontier_agent.infra.config import get_config
    c = get_config()
    print(f"  {'llm_provider':22} = {c.llm_provider or '<unset>'}")
    print(f"  {'openai_model':22} = {c.openai_model or '<unset>'}")
    # host only: enough to tell endpoints apart, never a token in a query string
    url = c.openai_base_url or ""
    print(f"  {'openai_base_url':22} = {url.split('//')[-1].split('/')[0] or '<unset>'}")
    for name, val in (
        ("openai_api_key", c.openai_api_key),
        ("serper_api_key", c.serper_api_key),
        ("jina_api_key", c.jina_api_key),
        # judges read these straight from the environment, not through config
        ("JUDGE_API_KEY", os.environ.get("JUDGE_API_KEY")),
        ("JUDGE_BASE_URL", os.environ.get("JUDGE_BASE_URL")),
    ):
        print(f"  {name:22} = {'set' if val else '<unset>'}")


async def check_kernel_llm() -> str | None:
    """The LLM BenchmarkSession._bootstrap() builds from LLM_PROVIDER."""
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra.llm_adapter import create_llm
    try:
        create_llm(get_config())
    except ValueError as e:
        if "Unknown LLM provider" in str(e):
            return (
                f"{e}\n"
                f"      BenchmarkSession._bootstrap() builds a default LLM from\n"
                f"      LLM_PROVIDER, and this build of infra/llm_adapter.py knows\n"
                f"      only openai / qwen / anthropic. Set LLM_PROVIDER=openai and\n"
                f"      point OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL at the\n"
                f"      endpoint you want (any OpenAI-compatible /v1 works)."
            )
        return str(e)
    return None


async def check_workflow_llm(pipeline: str, profile: str) -> str | None:
    """The LLM the workflow actually runs on, with the profile's sampling args."""
    mod = f"workflows.{pipeline}.profile"
    try:
        p = __import__(mod, fromlist=["load_profile", "create_llm"])
    except ImportError as e:
        return f"cannot import {mod}: {e}"
    try:
        # create_llm takes the whole profile dict and reads profile["llm"] itself
        llm = p.create_llm(p.load_profile(profile))
    except Exception as e:
        return f"building the profile LLM failed: {type(e).__name__}: {e}"

    try:
        # LLMClient.chat is the kernel's non-streaming entry point; max_tokens=1
        # keeps this to one token so the check is nearly free.
        await llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "top_p" in msg or "repetition_penalty" in msg:
            hint = (
                "\n      The profile sends top_p / repetition_penalty, which "
                "profile.py\n      injects into extra_body unconditionally — "
                "removing them from the\n      YAML does not help. GPT-5-family "
                "endpoints reject them. Use a\n      vLLM/SGLang-style endpoint, "
                "or change profile.py first (which\n      means the golden "
                "baseline has to be frozen after that change)."
            )
        return f"{type(e).__name__}: {msg[:300]}{hint}"
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="stateful-react-agent")
    ap.add_argument("--profile", default="keep5")
    args = ap.parse_args()

    print("config:")
    report_env()

    print("\nkernel LLM (LLM_PROVIDER -> infra/llm_adapter.create_llm):")
    if err := await check_kernel_llm():
        print(f"  FAIL  {err}")
        return 1
    print("  ok")

    print(f"\nworkflow LLM ({args.pipeline} / {args.profile}), 1 real call:")
    if err := await check_workflow_llm(args.pipeline, args.profile):
        print(f"  FAIL  {err}")
        return 1
    print("  ok")

    print("\njudge:")
    if os.environ.get("JUDGE_API_KEY") and os.environ.get("JUDGE_BASE_URL"):
        print("  ok — pinned to JUDGE_BASE_URL")
    else:
        # judges/_common.py falls back to OPENAI_API_KEY / OPENAI_BASE_URL, so an
        # unset judge silently becomes the agent's own model grading itself, and
        # it moves whenever OPENAI_BASE_URL moves — which destroys a baseline.
        print("  WARN  JUDGE_API_KEY / JUDGE_BASE_URL unset — the judge falls back")
        print("        to OPENAI_*, i.e. the agent model grades its own answers,")
        print("        and the grader moves whenever the agent endpoint moves.")
        print("        Set both explicitly before freezing a baseline.")

    print("\npreflight OK — safe to launch the run")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(AH))
    sys.exit(asyncio.run(main()))
