"""Hugging Face Space demo for the FrontierAgent ``react`` workflow.

This package is a *thin adapter* over the existing FrontierAgent runtime:

    Browser → Gradio (app.py)
            → FrontierAgentAdapter (adapter.py)
            → BenchmarkSession / stateful-react-agent pipeline
            → OpenAI-compatible endpoint (OPENAI_BASE_URL)

Nothing here re-implements the agent loop, and nothing here belongs in
``frontier_agent/`` — the Space consumes the runtime through two seams the
runtime already exposes: ``metadata['sdk_extra_observers']`` (structured
events) and ``metadata['pause_check']`` (cooperative cancellation).
"""
