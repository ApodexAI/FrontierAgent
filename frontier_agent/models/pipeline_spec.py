# pyright: reportWildcardImportFromLibrary=false
"""Declarative pipeline specification models (implemented by ``agent_core.models.pipeline_spec``)."""

import sys

import agent_core.models.pipeline_spec as _implementation
from agent_core.models.pipeline_spec import *  # noqa: F403

sys.modules[__name__] = _implementation
