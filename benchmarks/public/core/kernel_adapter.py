"""Compatibility import for the product-owned workflow runtime."""

from frontier_agent.runtime import WorkflowSession
from frontier_agent.runtime import _discover_pipeline_specs as _discover_pipeline_specs

BenchmarkSession = WorkflowSession

__all__ = ["BenchmarkSession"]
