# pyright: reportWildcardImportFromLibrary=false
"""FrontierAgent facade for shared model-profile behaviour.

Importing this module points AgentCore at the packaged ``model_registry.yaml``.
"""

import sys
from importlib.resources import files
from pathlib import Path

import agent_core.runtime.loop.model_profile as _implementation
from agent_core.runtime.loop.model_profile import *  # noqa: F403


def _resolve_registry_path() -> Path:
    try:
        res = Path(str(files("frontier_agent").joinpath("model_registry.yaml")))
        if res.is_file():
            return res
    except Exception:
        pass
    return Path(__file__).resolve().parents[3] / "model_registry.yaml"


_implementation.configure_model_registry(_resolve_registry_path())

sys.modules[__name__] = _implementation
