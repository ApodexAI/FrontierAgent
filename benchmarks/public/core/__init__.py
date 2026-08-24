"""Benchmark-agnostic infrastructure.

Lightweight re-exports only. ``kernel_adapter`` pulls in the LLM
runtime, so import it from its submodule path directly.
"""

from benchmarks.public.core.question import BenchmarkQuestion
from benchmarks.public.core.registry import (
    REGISTRY,
    DatasetConfig,
    get_config,
    load_questions,
)

__all__ = [
    "REGISTRY",
    "BenchmarkQuestion",
    "DatasetConfig",
    "get_config",
    "load_questions",
]
