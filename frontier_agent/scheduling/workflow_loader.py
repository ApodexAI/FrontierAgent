"""Discover and load external workflow packages from ``workflows/``."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_plugin_dir(
    plugin_dir_name: str,
    ctx: Any,
    *,
    skip_packages: set[str] | None = None,
) -> int:
    """Discover and load external plugin packages from a root-level directory.

    Returns the number of packages whose ``register(ctx)`` ran successfully.
    """
    import importlib
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    plugin_dir = project_root / plugin_dir_name

    if not plugin_dir.is_dir():
        return 0

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    loaded = 0
    skipped = skip_packages or set()

    for child in sorted(plugin_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skipped:
            continue
        if not (child / "__init__.py").exists():
            continue

        module_name = f"{plugin_dir_name}.{child.name}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            logger.warning("Failed to import plugin %s", module_name, exc_info=True)
            continue

        register_fn = getattr(mod, "register", None)
        if not callable(register_fn):
            logger.debug("Plugin %s has no register() function, skipping", module_name)
            continue

        try:
            register_fn(ctx)
            loaded += 1
        except ValueError as exc:
            logger.error("Plugin %s register() rejected: %s", module_name, exc)
        except Exception:
            logger.warning("Plugin %s register() failed", module_name, exc_info=True)

    return loaded


def load_workflow_plugins(
    pipeline_registry: Any,
    agent_registry: Any,
    *,
    skip_packages: set[str] | None = None,
) -> None:
    """Discover and load external workflow packages from ``workflows/``."""
    from frontier_agent.core.runtime.registries.workflows import WorkflowContext

    ctx = WorkflowContext(pipeline_registry, agent_registry)
    loaded = _load_plugin_dir("workflows", ctx, skip_packages=skip_packages)
    if loaded:
        logger.info("Loaded %d external workflow plugin(s)", loaded)
