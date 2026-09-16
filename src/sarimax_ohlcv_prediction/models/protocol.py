"""Protocol for all forecasting models."""

from typing import Protocol

import pandas as pd


class BaseModel(Protocol):
    """Protocol for all forecasting models."""

    def fit(self, data: pd.DataFrame, **kwargs) -> "BaseModel":
        """Train the model on data."""
        ...

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict next `periods` steps.

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        ...

    def predict_with_context(self, recent_data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Predict with recent data context (required for stateful models like LSTM)."""
        ...

    def save(self, path: str) -> None:
        """Save model to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        ...

    @property
    def data_hash(self) -> str:
        """Hash of training data for cache invalidation."""
        ...
