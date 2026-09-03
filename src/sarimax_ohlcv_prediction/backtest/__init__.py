"""Backtesting package."""

from .engine import BacktestResult, print_backtest_result, run_backtest
from .strategies import (
    STRATEGY_REGISTRY,
    BaseStrategy,
    ExitAfterNBarsStrategy,
    ExitOnSignalStrategy,
    get_strategy,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "print_backtest_result",
    "BaseStrategy",
    "ExitAfterNBarsStrategy",
    "ExitOnSignalStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
