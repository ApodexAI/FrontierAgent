# pyright: reportWildcardImportFromLibrary=false
"""WorkflowContext — registration interface for external workflow plugins (implemented by ``agent_core.runtime.registries.workflows``)."""

import sys

import agent_core.runtime.registries.workflows as _implementation
from agent_core.runtime.registries.workflows import *  # noqa: F403

sys.modules[__name__] = _implementation
