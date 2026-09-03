"""Backtesting strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for a backtesting strategy."""

    name: str
    params: dict


class BaseStrategy(ABC):
    """Abstract base strategy."""

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        **params,
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals.

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        pass


class ExitAfterNBarsStrategy(BaseStrategy):
    """Simple strategy: enter on prediction direction, exit after N bars."""

    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        exit_bars: int = 5,
        **params,
    ) -> tuple[pd.Series, pd.Series]:
        """Generate signals based on predicted price direction.

        Args:
            data: Historical OHLCV data
            predictions: Predicted OHLCV (same length as data for backtest)
            exit_bars: Number of bars to hold position
        """
        # Entry: long if predicted close > current close
        predicted_direction = predictions["close"] > data["close"]
        entries = predicted_direction

        # Exit: after N bars
        exits = entries.shift(exit_bars).fillna(False)

        return entries, exits


class ExitOnSignalStrategy(BaseStrategy):
    """Exit when prediction direction changes."""

    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        **params,
    ) -> tuple[pd.Series, pd.Series]:
        """Enter on direction, exit when direction reverses."""
        predicted_direction = predictions["close"] > data["close"]

        # Enter long when prediction says up
        entries = predicted_direction & ~predicted_direction.shift(1).fillna(False)

        # Exit when prediction says down (for long positions)
        exits = (~predicted_direction) & predicted_direction.shift(1).fillna(False)

        return entries, exits


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
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