"""Data fetching from Binance via CCXT."""

import logging
import time
from typing import Literal

import ccxt
import pandas as pd

from ..cache import cache_get, cache_set, get_ohlcv_key_from_params
from ..config import SETTINGS

logger = logging.getLogger(__name__)

binance = ccxt.binance()

_INVALID_MODE_MSG = "Invalid mode: {mode}"


def _raise_invalid_mode(mode: str) -> None:
    raise ValueError(_INVALID_MODE_MSG.format(mode=mode))


def fetch_data(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    lookback = SETTINGS.default_lookback_days if lookback_days is None else lookback_days

    try:
        if mode == "current":
            since = binance.milliseconds() - (24 * 60 * 60 * 1000)
            ohlcv = binance.fetch_ohlcv(SETTINGS.symbol, SETTINGS.timeframe, since=since)
        elif mode == "historical":
            since = binance.milliseconds() - (lookback * 24 * 60 * 60 * 1000)
            ohlcv = binance.fetch_ohlcv(SETTINGS.symbol, SETTINGS.timeframe, since=since)
        else:
            _raise_invalid_mode(mode)
            ohlcv = []  # unreachable; satisfies type checker for unbound name

        data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        return data
    except Exception:
        logger.exception("Failed to fetch data")
        return pd.DataFrame()


def _fetch_with_retry_uncached(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch data with retry logic (uncached version)."""
    for attempt in range(max_retries):
        data = fetch_data(mode, lookback_days)
        if not data.empty:
            return data
        if attempt < max_retries - 1:
            logger.warning("Retry %d/%d after empty data", attempt + 1, max_retries)
            time.sleep(SETTINGS.error_sleep_seconds)
    return pd.DataFrame()


def fetch_with_retry(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch data with retry logic and optional caching.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)
        max_retries: Maximum retry attempts
        use_cache: Whether to use cache (default True)

    Returns:
        DataFrame with OHLCV data
    """
    if use_cache:
        return fetch_cached(mode, lookback_days, max_retries)
    return _fetch_with_retry_uncached(mode, lookback_days, max_retries)


def fetch_cached(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch data with retry logic using the persistent cache.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)
        max_retries: Maximum retry attempts

    Returns:
        DataFrame with OHLCV data
    """
    lookback = SETTINGS.default_lookback_days if lookback_days is None else lookback_days

    # Check cache first
    cache_key = get_ohlcv_key_from_params(mode, lookback)
    cached_data = cache_get(cache_key)
    if cached_data is not None:
        logger.debug("Cache HIT for OHLCV: %s", cache_key)
        return cached_data

    # Cache miss - fetch data
    logger.debug("Cache MISS for OHLCV: %s", cache_key)
    data = _fetch_with_retry_uncached(mode, lookback_days, max_retries)

    # Store in cache if we got data
    if not data.empty:
        if mode == "current":
            ttl = SETTINGS.cache_ttl_ohlcv_current
        else:
            ttl = SETTINGS.cache_ttl_ohlcv_historical
        cache_set(cache_key, data, ttl)

    return data
