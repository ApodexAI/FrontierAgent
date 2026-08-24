"""Per-family DatasetConfig registry."""

from __future__ import annotations

import importlib
import pkgutil

from benchmarks.public.core.registry import DatasetConfig

REGISTRY: dict[str, DatasetConfig] = {}

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    for _key, _cfg in getattr(_mod, "CONFIGS", []):
        if _key in REGISTRY:
            raise RuntimeError(
                f"Duplicate benchmark key {_key!r} in families/"
            )
        REGISTRY[_key] = _cfg

__all__ = ["REGISTRY"]
