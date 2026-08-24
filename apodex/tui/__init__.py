"""Full-screen Textual TUI for ``apodex``.

The agent engine (:func:`run_agent_loop`) and its observers are unchanged: the
TUI plugs in exactly where line mode does, through the ``TerminalObserver``
hooks, but swaps the *sink* — instead of writing stdout it posts updates to
Textual widgets, and instead of a single-key stdin prompt it shows a modal.

See :mod:`apodex.tui.app` for the entry point (:class:`FrontierAgentApp`).
"""
