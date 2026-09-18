"""Backtesting package."""

from .engine import BacktestResult, print_backtest_result, run_backtest
from .strategies import (
    STRATEGY_REGISTRY,
    BaseStrategy,
    ExitAfterNBarsStrategy,
    ExitOnSignalStrategy,
    get_strategy,
)
from .walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "BacktestResult",
    "run_backtest",
    "print_backtest_result",
    "BaseStrategy",
    "ExitAfterNBarsStrategy",
    "ExitOnSignalStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "WalkForwardResult",
    "run_walk_forward",
]
