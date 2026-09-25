# pyright: reportWildcardImportFromLibrary=false
"""PipelineRegistry — stores and retrieves PipelineSpec objects (implemented by ``agent_core.scheduling.pipeline_registry``)."""

import sys

import agent_core.scheduling.pipeline_registry as _implementation
from agent_core.scheduling.pipeline_registry import *  # noqa: F403

sys.modules[__name__] = _implementation
