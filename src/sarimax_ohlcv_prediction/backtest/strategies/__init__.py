"""Backtesting strategies package."""

from .base import BaseStrategy
from .exit_after_n import ExitAfterNBarsStrategy
from .exit_on_signal import ExitOnSignalStrategy

STRATEGY_REGISTRY = {
    "exit_after_n": ExitAfterNBarsStrategy,
    "exit_on_signal": ExitOnSignalStrategy,
}

_UNKNOWN_STRATEGY_MSG = "Unknown strategy: {name}. Available: {available}"


def get_strategy(name: str) -> BaseStrategy:
    """Get strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            _UNKNOWN_STRATEGY_MSG.format(name=name, available=list(STRATEGY_REGISTRY.keys()))
        )
    return STRATEGY_REGISTRY[name]()


__all__ = [
    "BaseStrategy",
    "ExitAfterNBarsStrategy",
    "ExitOnSignalStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
