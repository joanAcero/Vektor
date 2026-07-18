"""
registry.py
-----------
Explicit strategy registry.

Strategies still live one-per-file in strategies/, but discovery is NOT
automatic. The strategies package's __init__.py imports each strategy module it
wants to expose; importing a module runs its @register decorator, which adds the
class to this registry. To enable a strategy you add one import line to
strategies/__init__.py; to disable one you remove (or comment) that line.

Why explicit over auto-discovery:
  * No import side-effects from files you didn't choose (helpers, WIP, drafts).
  * A broken/unfinished strategy file cannot affect the others -- it's simply
    not listed until you add it.
  * The set of active strategies is a readable list under version control, not
    an emergent property of what happens to be on disk.

@register still validates the class and rejects duplicate keys, so mistakes
surface at import time rather than at run time.
"""

from __future__ import annotations

import importlib
import logging
from typing import Type

from src.strategy import Strategy

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Class decorator: add a Strategy subclass to the registry."""
    if not isinstance(cls, type) or not issubclass(cls, Strategy):
        raise TypeError(f"@register expects a Strategy subclass, got {cls!r}")
    meta = getattr(cls, "meta", None)
    if meta is None:
        raise TypeError(f"{cls.__name__} must define a class-level `meta`")
    existing = _REGISTRY.get(meta.key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Duplicate strategy key {meta.key!r}: "
            f"{existing.__name__} vs {cls.__name__}"
        )
    _REGISTRY[meta.key] = cls
    return cls


def load_strategies(package: str = "strategies") -> dict[str, Type[Strategy]]:
    """
    Import the strategies package so its __init__.py runs the explicit imports
    that register each strategy. Idempotent. Returns the registry.
    """
    importlib.import_module(package)
    if not _REGISTRY:
        log.warning(
            "No strategies registered. Did you add imports to %s/__init__.py?",
            package,
        )
    return get_registry()


def get_registry() -> dict[str, Type[Strategy]]:
    return dict(_REGISTRY)


def get_strategy(key: str) -> Type[Strategy]:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown strategy {key!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
