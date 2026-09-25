# pyright: reportWildcardImportFromLibrary=false
"""Allowlist filter wrapper for the ``SkillLoader`` Protocol (implemented by ``agent_core.components.skills.allowlist_loader``)."""

import sys

import agent_core.components.skills.allowlist_loader as _implementation
from agent_core.components.skills.allowlist_loader import *  # noqa: F403

sys.modules[__name__] = _implementation
