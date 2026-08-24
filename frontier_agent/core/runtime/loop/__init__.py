"""Generic ReAct agent loop — LLM call + tool execution + runtime primitives.

Import from the submodule you need, e.g.::

    from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
    from frontier_agent.core.runtime.loop.compact import compact_messages

This package deliberately re-exports nothing. It used to aggregate every
submodule's public names, which meant importing one piece eagerly loaded all
eight — paid again in every worker subprocess — and nothing imported from the
package root anyway. An aggregate list is also what drifts: the same habit left
this tree with seven lazily re-exported middlewares whose modules did not exist.
"""
