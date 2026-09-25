# pyright: reportWildcardImportFromLibrary=false
"""FrontierAgent composition of AgentCore's OpenAI Chat Completions client.

Importing this module wires FrontierAgent's session-affinity policy into the
shared provider. Every ``OpenAIClient`` in the product must be constructed
through this module; building ``agent_core.providers.openai_chat.OpenAIClient``
directly before this import silently drops the session id.
"""

import sys

import agent_core.providers.openai_chat as _implementation
from agent_core.providers.openai_chat import *  # noqa: F403

from frontier_agent.infra.session_context import get_task_session_id, mirror_session_query

_implementation.configure_session_query_resolver(mirror_session_query)
_implementation.configure_session_scope_resolver(get_task_session_id)

sys.modules[__name__] = _implementation
