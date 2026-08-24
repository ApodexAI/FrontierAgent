"""Middleware.

LLM-call middleware lives in :mod:`frontier_agent.components.middleware.llm`
(proxy, summarization, stream-repetition, skill injection).

The phase-level middleware *contract* — ``ExecutionMiddleware``,
``PhaseContext``, ``PhaseMiddlewareChain`` — lives in
:mod:`frontier_agent.core.protocols`, which is where extensions should import it
from. A concrete duplicate of that contract plus four unused built-in phase
middlewares used to live here; nothing outside this package referenced them, so
they were removed rather than shipped as a second, diverging definition.
"""
