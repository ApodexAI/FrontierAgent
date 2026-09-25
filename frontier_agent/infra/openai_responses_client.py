# pyright: reportWildcardImportFromLibrary=false
"""OpenAI **Responses API** LLMClient — wraps :class:`openai.AsyncOpenAI` (implemented by ``agent_core.providers.openai_responses``)."""

import sys

import agent_core.providers.openai_responses as _implementation
from agent_core.providers.openai_responses import *  # noqa: F403

sys.modules[__name__] = _implementation
