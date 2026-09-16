"""Exit on signal change strategy."""

from typing import Any

import pandas as pd

from .base import BaseStrategy


class ExitOnSignalStrategy(BaseStrategy):
    """Exit when prediction direction changes."""

    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        **params: Any,  # pyrefly: ignore -- open strategy params
    ) -> tuple[pd.Series, pd.Series]:
        """Enter on direction, exit when direction reverses."""
        predicted_direction = predictions["close"] > data["close"]

        # Enter long when prediction says up
        entries = predicted_direction & ~(predicted_direction.shift(1).fillna(False))

        # Exit when prediction says down (for long positions)
        exits = (~predicted_direction) & (predicted_direction.shift(1).fillna(False))

        return entries, exits
