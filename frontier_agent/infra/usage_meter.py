# pyright: reportWildcardImportFromLibrary=false
"""Process-wide external API and tool-call meter (implemented by ``agent_core.runtime.usage_meter``)."""

import sys

import agent_core.runtime.usage_meter as _implementation
from agent_core.runtime.usage_meter import *  # noqa: F403

sys.modules[__name__] = _implementation
