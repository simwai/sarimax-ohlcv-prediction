"""Base model protocol and common utilities."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd

from ..cache import compute_data_hash


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

    @property
    def data_hash(self) -> str:
        """Hash of training data for cache invalidation."""
        ...


class ModelBase(ABC):
    """Abstract base class with common functionality."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.columns = ["open", "high", "low", "close", "volume"]
        self._data_hash: str | None = None
        self._training_data: pd.DataFrame | None = None

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

    def _store_training_data(self, data: pd.DataFrame) -> None:
        """Store training data and compute hash for caching."""
        self._training_data = data.copy()
        self._data_hash = compute_data_hash(data)

    @property
    def data_hash(self) -> str:
        """Hash of training data for cache invalidation."""
        if self._data_hash is None:
            return "unknown"
        return self._data_hash
