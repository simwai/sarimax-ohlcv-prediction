"""Exit after N bars strategy."""

from typing import Any

import pandas as pd

from .base import BaseStrategy

_EXIT_BARS_MSG = "exit_bars must be positive, got {exit_bars}"


class ExitAfterNBarsStrategy(BaseStrategy):
    """Simple strategy: enter on prediction direction, exit after N bars."""

    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        exit_bars: int = 5,
        **params: Any,  # pyrefly: ignore -- open strategy params
    ) -> tuple[pd.Series, pd.Series]:
        """Generate signals based on predicted price direction.

        Args:
            data: Historical OHLCV data
            predictions: Predicted OHLCV (same length as data for backtest)
            exit_bars: Number of bars to hold position
        """
        if exit_bars <= 0:
            raise ValueError(_EXIT_BARS_MSG.format(exit_bars=exit_bars))  # noqa: TRY003
        # Entry: long if predicted close > current close
        predicted_direction = predictions["close"] > data["close"]
        entries = predicted_direction

        # Exit: after N bars
        exits = entries.shift(exit_bars).fillna(False)

        return entries, exits
