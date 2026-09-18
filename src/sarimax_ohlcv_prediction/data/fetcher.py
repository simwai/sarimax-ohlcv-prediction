"""Data fetching from exchanges via CCXT."""

import logging
import time
from typing import Literal

import ccxt
import pandas as pd

from ..cache import cache_get, cache_set, get_ohlcv_key_from_params
from ..config import SETTINGS

logger = logging.getLogger(__name__)

_INVALID_MODE_MSG = "Invalid mode: {mode}"
_UNKNOWN_EXCHANGE_MSG = "Unknown exchange: {exchange_id}"
_INVALID_SYMBOL_MSG = "Symbol not available on {exchange_id}: {symbol}"
_INVALID_TIMEFRAME_MSG = "Timeframe not supported by {exchange_id}: {timeframe}"
_LOAD_MARKETS_FAILED_MSG = "Failed to load markets for {exchange_id}: {error}"


def _raise_invalid_mode(mode: str) -> None:
    raise ValueError(_INVALID_MODE_MSG.format(mode=mode))


def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    """Create a CCXT exchange instance by id."""
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(_UNKNOWN_EXCHANGE_MSG.format(exchange_id=exchange_id))
    return exchange_cls()


def get_default_exchange() -> ccxt.Exchange:
    """Return the default Binance exchange instance for backward compatibility."""
    return _build_exchange(SETTINGS.default_exchange_id)


def _validate_market(
    exchange_id: str,
    symbol: str,
    timeframe: str,
) -> tuple[ccxt.Exchange, str, str]:
    """Validate exchange/symbol/timeframe and return a prepared exchange instance."""
    exchange = _build_exchange(exchange_id)
    try:
        markets = exchange.load_markets()
    except Exception as exc:
        raise RuntimeError(
            _LOAD_MARKETS_FAILED_MSG.format(exchange_id=exchange_id, error=exc)
        ) from exc

    if symbol not in markets:
        raise ValueError(_INVALID_SYMBOL_MSG.format(exchange_id=exchange_id, symbol=symbol))

    if timeframe not in exchange.timeframes:
        raise ValueError(
            _INVALID_TIMEFRAME_MSG.format(exchange_id=exchange_id, timeframe=timeframe)
        )

    return exchange, symbol, timeframe


def fetch_data(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from an exchange via CCXT.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)
        exchange_id: CCXT exchange id, defaults to SETTINGS.default_exchange_id
        symbol: Trading symbol, defaults to SETTINGS.symbol
        timeframe: Candle timeframe, defaults to SETTINGS.timeframe

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    lookback = SETTINGS.default_lookback_days if lookback_days is None else lookback_days
    resolved_exchange_id = exchange_id or SETTINGS.default_exchange_id
    resolved_symbol = symbol or SETTINGS.symbol
    resolved_timeframe = timeframe or SETTINGS.timeframe

    exchange, validated_symbol, validated_timeframe = _validate_market(
        resolved_exchange_id, resolved_symbol, resolved_timeframe
    )

    try:
        if mode == "current":
            since = exchange.milliseconds() - (24 * 60 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(validated_symbol, validated_timeframe, since=since)
        elif mode == "historical":
            since = exchange.milliseconds() - (lookback * 24 * 60 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(validated_symbol, validated_timeframe, since=since)
        else:
            _raise_invalid_mode(mode)
            ohlcv = []  # unreachable; satisfies type checker for unbound name

        data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        return data
    except Exception:
        logger.exception("Failed to fetch data from %s", resolved_exchange_id)
        return pd.DataFrame()


def _fetch_with_retry_uncached(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Fetch data with retry logic (uncached version)."""
    for attempt in range(max_retries):
        data = fetch_data(mode, lookback_days, exchange_id, symbol, timeframe)
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
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Fetch data with retry logic and optional caching.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)
        max_retries: Maximum retry attempts
        use_cache: Whether to use cache (default True)
        exchange_id: CCXT exchange id, defaults to SETTINGS.default_exchange_id
        symbol: Trading symbol, defaults to SETTINGS.symbol
        timeframe: Candle timeframe, defaults to SETTINGS.timeframe

    Returns:
        DataFrame with OHLCV data
    """
    if use_cache:
        return fetch_cached(mode, lookback_days, max_retries, exchange_id, symbol, timeframe)
    return _fetch_with_retry_uncached(
        mode, lookback_days, max_retries, exchange_id, symbol, timeframe
    )


def fetch_cached(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Fetch data with retry logic using the persistent cache.

    Args:
        mode: "current" for last 24h, "historical" for lookback_days
        lookback_days: Number of days to look back (for historical mode)
        max_retries: Maximum retry attempts
        exchange_id: CCXT exchange id, defaults to SETTINGS.default_exchange_id
        symbol: Trading symbol, defaults to SETTINGS.symbol
        timeframe: Candle timeframe, defaults to SETTINGS.timeframe

    Returns:
        DataFrame with OHLCV data
    """
    lookback = SETTINGS.default_lookback_days if lookback_days is None else lookback_days
    resolved_exchange_id = exchange_id or SETTINGS.default_exchange_id
    resolved_symbol = symbol or SETTINGS.symbol
    resolved_timeframe = timeframe or SETTINGS.timeframe

    # Check cache first
    cache_key = get_ohlcv_key_from_params(
        mode,
        lookback,
        exchange_id=resolved_exchange_id,
        symbol=resolved_symbol,
        timeframe=resolved_timeframe,
    )
    cached_data = cache_get(cache_key)
    if cached_data is not None:
        logger.debug("Cache HIT for OHLCV: %s", cache_key)
        return cached_data

    # Cache miss - fetch data
    logger.debug("Cache MISS for OHLCV: %s", cache_key)
    data = _fetch_with_retry_uncached(
        mode, lookback_days, max_retries, exchange_id, symbol, timeframe
    )

    # Store in cache if we got data
    if not data.empty:
        if mode == "current":
            ttl = SETTINGS.cache_ttl_ohlcv_current
        else:
            ttl = SETTINGS.cache_ttl_ohlcv_historical
        cache_set(cache_key, data, ttl)

    return data
