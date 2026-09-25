# pyright: reportWildcardImportFromLibrary=false
"""Renewable wall-time lease shared by live intervention and loop guards (implemented by ``agent_core.runtime.wall_time_lease``)."""

import sys

import agent_core.runtime.wall_time_lease as _implementation
from agent_core.runtime.wall_time_lease import *  # noqa: F403

sys.modules[__name__] = _implementation
