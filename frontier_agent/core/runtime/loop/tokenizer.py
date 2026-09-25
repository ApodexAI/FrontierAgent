# pyright: reportWildcardImportFromLibrary=false
"""Provide non-blocking access to tiktoken encoders (implemented by ``agent_core.runtime.loop.tokenizer``)."""

import sys

import agent_core.runtime.loop.tokenizer as _implementation
from agent_core.runtime.loop.tokenizer import *  # noqa: F403

sys.modules[__name__] = _implementation
