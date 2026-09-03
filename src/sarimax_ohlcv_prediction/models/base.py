"""Base model protocol and common utilities."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
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


class ModelBase(ABC):
    """Abstract base class with common functionality."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.columns = ["open", "high", "low", "close", "volume"]

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> "ModelBase":
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        pass

    def predict_with_context(self, recent_data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Default context prediction delegates to predict for stateless models."""
        _ = recent_data
        return self.predict(periods)

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ModelBase":
        pass
