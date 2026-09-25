# pyright: reportWildcardImportFromLibrary=false
"""Filesystem-backed implementation of the ``SkillLoader`` Protocol (implemented by ``agent_core.components.skills.file_system_loader``)."""

import sys

import agent_core.components.skills.file_system_loader as _implementation
from agent_core.components.skills.file_system_loader import *  # noqa: F403

sys.modules[__name__] = _implementation
