"""apodex — a Claude-Code-style local coding-agent TUI.

A terminal-native coding agent that runs against your **local working
directory**, reusing FrontierAgent's generic ReAct engine
(:func:`frontier_agent.core.runtime.loop.agent_loop.run_agent_loop`), its
local file/shell tools (``plugins.tools``), and its observer streaming
contract (``frontier_agent.core.loop_types``).

Design note — why not reuse ``workflows/swe`` directly: that workflow is
built for *SWE-bench* — it spins up a Docker/E2B sandbox, exposes only
``bash`` + ``submit_solution``, and emits a git-diff patch for scoring.
A local interactive coding experience (à la apodex_terminal) needs the
full Claude-Code tool surface (Bash/Read/Write/Edit/Grep/Glob) operating
on the user's real repo, with live streaming + per-edit approval. So we
reuse the *engine* (run_agent_loop + tools + observers) — the genuinely
reusable core — rather than the SWE-bench sandbox pipeline.

Run it from the FrontierAgent repo root so ``frontier_agent`` / ``plugins`` /
``workflows`` are importable::

    python -m apodex                 # interactive TUI in $PWD
    python -m apodex --cwd /path/to/repo
    python -m apodex --print "explain src/foo.py"   # one-shot
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
