# pyright: reportWildcardImportFromLibrary=false
"""Service-registry scopes (implemented by ``agent_core.runtime.registries.scope``)."""

import sys

import agent_core.runtime.registries.scope as _implementation
from agent_core.runtime.registries.scope import *  # noqa: F403

sys.modules[__name__] = _implementation
