"""Cache module - persistent caching with joblib.Memory and TTL."""

import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import pandas as pd

from ..config import SETTINGS
from .backend import CacheManager, get_cache_manager, reset_cache_manager
from .keys import (
    _data_hash,
    get_model_key_from_params,
    get_ohlcv_key_from_params,
    get_prediction_key_from_params,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Module-level cache manager instance
_cache_manager: CacheManager | None = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = get_cache_manager()
    return _cache_manager


def _cached_with_ttl(
    ttl: int,
    key_builder: Callable[..., str],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Create a decorator that caches function results with TTL.

    Args:
        ttl: Time to live in seconds
        key_builder: Function to build cache key from function args

    Returns:
        Decorator function
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache = get_cache()
            if not cache.enabled:
                return func(*args, **kwargs)

            # Build cache key
            key = key_builder(*args, **kwargs)

            # Try to get from cache
            cached = cache.get(key)
            if cached is not None:
                logger.debug("Cache HIT: %s", key)
                return cached

            # Cache miss - execute function
            logger.debug("Cache MISS: %s", key)
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


def cached_ohlcv(mode: str, lookback: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for OHLCV data fetching with mode-based TTL.

    Args:
        mode: "current" or "historical"
        lookback: Lookback days

    Returns:
        Decorator function
    """
    ttl = (
        SETTINGS.cache_ttl_ohlcv_current
        if mode == "current"
        else SETTINGS.cache_ttl_ohlcv_historical
    )

    def key_builder(*args: P.args, **kwargs: P.kwargs) -> str:
        return get_ohlcv_key_from_params(mode, lookback)

    return _cached_with_ttl(ttl, key_builder)


def cached_model(
    model_name: str,
    lookback: int,
    iterations: int,
    data_hash: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for model training with 7-day TTL.

    Args:
        model_name: Model class name
        lookback: Training data lookback days
        iterations: Training iterations/epochs
        data_hash: Hash of training data

    Returns:
        Decorator function
    """

    def key_builder(*args: P.args, **kwargs: P.kwargs) -> str:
        return get_model_key_from_params(model_name, lookback, iterations, data_hash)

    return _cached_with_ttl(SETTINGS.cache_ttl_models, key_builder)


def cached_prediction(
    model_name: str,
    periods: int,
    data_hash: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for model predictions with 30-min TTL.

    Args:
        model_name: Model class name
        periods: Number of prediction periods
        data_hash: Hash of context data

    Returns:
        Decorator function
    """

    def key_builder(*args: P.args, **kwargs: P.kwargs) -> str:
        return get_prediction_key_from_params(model_name, periods, data_hash)

    return _cached_with_ttl(SETTINGS.cache_ttl_predictions, key_builder)


# Convenience functions for direct cache manipulation
def cache_get(key: str) -> Any | None:
    """Get value from cache."""
    return get_cache().get(key)


def cache_set(key: str, value: Any, ttl: int) -> None:
    """Set value in cache with TTL."""
    get_cache().set(key, value, ttl)


def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    return get_cache().delete(key)


def cache_clear() -> int:
    """Clear all cache entries. Returns count cleared."""
    return get_cache().clear()


def cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    return get_cache().stats()


def cache_inspect(limit: int = 50) -> list[dict[str, Any]]:
    """Inspect cache entries."""
    return get_cache().inspect(limit)


def compute_data_hash(df: pd.DataFrame, n: int = 100) -> str:
    """Compute hash of DataFrame for cache key.

    Args:
        df: DataFrame to hash
        n: Number of last rows to include

    Returns:
        Hash string
    """
    return _data_hash(df, n)


__all__ = [
    "CacheManager",
    "get_cache",
    "reset_cache_manager",
    "cached_ohlcv",
    "cached_model",
    "cached_prediction",
    "cache_get",
    "cache_set",
    "cache_delete",
    "cache_clear",
    "cache_stats",
    "cache_inspect",
    "compute_data_hash",
    "get_ohlcv_key_from_params",
    "get_model_key_from_params",
    "get_prediction_key_from_params",
]