"""Data fetching from Binance via CCXT."""

import logging
import time
from typing import Literal

import ccxt
import pandas as pd

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
    lookback = lookback_days or SETTINGS.default_lookback_days

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

        data = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        return data
    except Exception:
        logger.exception("Failed to fetch data")
        return pd.DataFrame()


def fetch_with_retry(
    mode: Literal["current", "historical"],
    lookback_days: int | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch data with retry logic."""
    for attempt in range(max_retries):
        data = fetch_data(mode, lookback_days)
        if not data.empty:
            return data
        if attempt < max_retries - 1:
            logger.warning("Retry %d/%d after empty data", attempt + 1, max_retries)
            time.sleep(SETTINGS.error_sleep_seconds)
    return pd.DataFrame()