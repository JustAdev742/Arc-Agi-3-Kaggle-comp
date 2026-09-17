"""Agents. Register new ones in ``REGISTRY`` so ``scripts/eval.py --agent`` finds them."""
from __future__ import annotations

from collections.abc import Callable

from .base import Agent, AgentContext

REGISTRY: dict[str, Callable[..., Agent]] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get(name: str):
    if name not in REGISTRY:
        # Lazy imports so optional deps (e.g. requests for the REPL agent) stay optional.
        from . import explorer, random_agent, rules_agent  # noqa: F401  (registration side effect)
        try:
            from . import council  # noqa: F401
        except Exception as e:  # noqa: BLE001  (optional dependency: any import error means 'not available')
            import logging
            logging.getLogger(__name__).warning("council agent unavailable: %s", e)
        try:
            from . import repl_agent  # noqa: F401
        except Exception as e:  # noqa: BLE001  (optional dependency: any import error means 'not available')
            import logging
            logging.getLogger(__name__).warning("repl agent unavailable: %s", e)
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["REGISTRY", "Agent", "AgentContext", "get", "register"]
