"""Models package - exports all model implementations."""

import logging
from typing import Any

import pandas as pd

from ..cache import (
    cache_get,
    cache_set,
    compute_data_hash,
    get_model_key_from_params,
)
from ..config import SETTINGS
from .base import ModelBase
from .lstm import LSTMModel
from .prophet import ProphetModel
from .protocol import BaseModel
from .sarimax import SARIMAXModel

logger = logging.getLogger(__name__)

__all__ = [
    "BaseModel",
    "ModelBase",
    "SARIMAXModel",
    "ProphetModel",
    "LSTMModel",
]

# Model registry for CLI/REPL
MODEL_REGISTRY = {
    "SARIMAX": SARIMAXModel,
    "Prophet": ProphetModel,
    "LSTM": LSTMModel,
}

_UNKNOWN_MODEL_MSG = "Unknown model: {name}. Available: {available}"


def get_model_class(name: str) -> type[ModelBase]:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            _UNKNOWN_MODEL_MSG.format(name=name, available=list(MODEL_REGISTRY.keys()))
        )
    return MODEL_REGISTRY[name]


def create_model(name: str, **kwargs: Any) -> BaseModel:  # pyrefly: ignore -- open model params
    """Create model instance by name.

    Args:
        name: Model name (SARIMAX, Prophet, LSTM)
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    return get_model_class(name)(**kwargs)


def _get_or_train_model(
    name: str,
    data: pd.DataFrame,
    lookback: int,
    iterations: int,
    use_cache: bool = True,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs: Any,  # pyrefly: ignore -- open model params
) -> BaseModel:
    """Get cached model or train new one.

    Args:
        name: Model name
        data: Training data
        lookback: Lookback days
        iterations: Training iterations
        use_cache: Whether to use cache
        exchange_id: Optional exchange id for cache namespacing
        symbol: Optional symbol override
        timeframe: Optional timeframe override
        **kwargs: Additional model arguments

    Returns:
        Trained model instance
    """
    if not use_cache:
        model = create_model(name, **kwargs)
        model.fit(data, iterations=iterations)
        return model

    # Compute data hash for cache key
    data_hash = compute_data_hash(data)
    cache_key = get_model_key_from_params(
        name,
        lookback,
        iterations,
        data_hash,
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
    )

    # Try to load from cache
    cached_model = cache_get(cache_key)
    if cached_model is not None:
        logger.debug("Cache HIT for model: %s", cache_key)
        return cached_model

    # Cache miss - train new model
    logger.debug("Cache MISS for model: %s", cache_key)
    model = create_model(name, **kwargs)
    model.fit(data, iterations=iterations)

    # Save to cache
    cache_set(cache_key, model, SETTINGS.cache_ttl_models)
    return model


def get_cached_model(
    name: str,
    data: pd.DataFrame,
    lookback: int | None = None,
    iterations: int | None = None,
    use_cache: bool = True,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs: Any,  # pyrefly: ignore -- open model params
) -> BaseModel:
    """Get model from cache or train new one.

    Args:
        name: Model name
        data: Training data
        lookback: Lookback days (default from SETTINGS)
        iterations: Training iterations (default from SETTINGS)
        use_cache: Whether to use cache
        exchange_id: Optional exchange id for cache namespacing
        symbol: Optional symbol override
        timeframe: Optional timeframe override
        **kwargs: Additional model arguments

    Returns:
        Trained model instance
    """
    lookback = lookback or SETTINGS.default_lookback_days
    iterations = iterations or SETTINGS.default_iterations

    return _get_or_train_model(
        name,
        data,
        lookback,
        iterations,
        use_cache,
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        **kwargs,
    )
