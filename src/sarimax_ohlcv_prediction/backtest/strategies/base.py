"""Base strategy protocol."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base strategy."""

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame,
        **params: Any,  # pyrefly: ignore -- open strategy params
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals.

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        pass
