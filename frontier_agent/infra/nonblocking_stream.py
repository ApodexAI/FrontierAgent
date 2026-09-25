# pyright: reportWildcardImportFromLibrary=false
"""Non-blocking text-stream wrapper — console output must never wedge the (implemented by ``agent_core.providers.nonblocking_stream``)."""

import sys

import agent_core.providers.nonblocking_stream as _implementation
from agent_core.providers.nonblocking_stream import *  # noqa: F403

sys.modules[__name__] = _implementation
