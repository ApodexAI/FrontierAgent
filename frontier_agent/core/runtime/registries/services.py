# pyright: reportWildcardImportFromLibrary=false
"""Service Registry — simplified dependency injection container (implemented by ``agent_core.runtime.registries.services``)."""

import sys

import agent_core.runtime.registries.services as _implementation
from agent_core.runtime.registries.services import *  # noqa: F403
from agent_core.runtime.registries.services import (  # not in __all__; named for static checkers
    _services as _services,
)

sys.modules[__name__] = _implementation
