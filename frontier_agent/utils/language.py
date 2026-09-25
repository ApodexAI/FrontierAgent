# pyright: reportWildcardImportFromLibrary=false
"""Generic language detection and instruction utilities (implemented by ``agent_core.utils.language``)."""

import sys

import agent_core.utils.language as _implementation
from agent_core.utils.language import *  # noqa: F403

sys.modules[__name__] = _implementation
