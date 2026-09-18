"""Cache module - persistent caching with joblib.Memory and TTL."""

import logging
from typing import Any

import pandas as pd

from .backend import CacheManager, get_cache_manager, reset_cache_manager
from .keys import (
    _data_hash,
    get_model_key_from_params,
    get_ohlcv_key_from_params,
    get_prediction_key_from_params,
)

logger = logging.getLogger(__name__)


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return get_cache_manager()


# NOTE: per-function TTL decorators were removed; callers use cache_get/cache_set directly.


# Convenience functions for direct cache manipulation
def cache_get(key: str) -> Any | None:  # pyrefly: ignore -- cache stores arbitrary payloads
    """Get value from cache."""
    return get_cache().get(key)


def cache_set(
    key: str, value: Any, ttl: int
) -> None:  # pyrefly: ignore -- cache stores arbitrary payloads
    """Set value in cache with TTL."""
    get_cache().set(key, value, ttl)


def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    return get_cache().delete(key)


def cache_clear() -> int:
    """Clear all cache entries. Returns count cleared."""
    return get_cache().clear()


def cache_stats() -> dict[str, Any]:  # pyrefly: ignore -- cache stores arbitrary payloads
    """Get cache statistics."""
    return get_cache().stats()


def cache_inspect(
    limit: int = 50,
) -> list[dict[str, Any]]:  # pyrefly: ignore -- cache stores arbitrary payloads
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
