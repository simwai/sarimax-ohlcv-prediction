"""Cache key builders."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from ..config import SETTINGS


@dataclass(frozen=True)
class ModelKeyParams:
    """Parameters for building a model cache key."""

    model_name: str
    symbol: str
    timeframe: str
    lookback: int
    iterations: int
    data_hash: str
    exchange_id: str | None = None


def _sanitize(s: str) -> str:
    """Sanitize string for use in cache key."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_").replace("\\", "_")


def _data_hash(df: pd.DataFrame, n: int = 100) -> str:
    """Generate hash from last N rows of DataFrame.

    Args:
        df: DataFrame to hash
        n: Number of last rows to include in hash

    Returns:
        Short hash string (16 chars)
    """
    if df.empty:
        return "empty"
    tail = df.tail(n)
    # Use pandas built-in hashing for consistent results
    hash_val = pd.util.hash_pandas_object(tail, index=True).to_numpy()
    # Create MD5 hash of the hash values
    m = hashlib.md5()
    m.update(hash_val.tobytes())
    return m.hexdigest()[:16]


def ohlcv_key(
    symbol: str,
    timeframe: str,
    mode: str,
    lookback: int,
    exchange_id: str | None = None,
) -> str:
    """Build cache key for OHLCV data fetch.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "5m")
        mode: "current" or "historical"
        lookback: Lookback days
        exchange_id: Optional exchange id for multi-exchange cache namespacing

    Returns:
        Cache key string
    """
    exchange = _sanitize(exchange_id) if exchange_id else "binance"
    return f"ohlcv:{exchange}:{_sanitize(symbol)}:{_sanitize(timeframe)}:{mode}:{lookback}"


def model_key(params: ModelKeyParams, exchange_id: str | None = None) -> str:
    """Build cache key for trained model.

    Args:
        params: Bundled model identity and training parameters.
        exchange_id: Optional exchange id for multi-exchange cache namespacing.

    Returns:
        Cache key string
    """
    exchange = _sanitize(exchange_id) if exchange_id else "binance"
    return (
        f"model:{params.model_name}:{exchange}:{_sanitize(params.symbol)}:{_sanitize(params.timeframe)}:"
        f"{params.lookback}:{params.iterations}:{params.data_hash}"
    )


def prediction_key(  # noqa: PLR0913
    model_name: str,
    symbol: str,
    timeframe: str,
    periods: int,
    data_hash: str,
    exchange_id: str | None = None,
) -> str:
    """Build cache key for model predictions.

    Args:
        model_name: Model class name
        symbol: Trading symbol
        timeframe: Timeframe
        periods: Number of prediction periods
        data_hash: Hash of context data (last N rows)
        exchange_id: Optional exchange id for multi-exchange cache namespacing

    Returns:
        Cache key string
    """
    exchange = _sanitize(exchange_id) if exchange_id else "binance"
    return (
        f"pred:{model_name}:{exchange}:{_sanitize(symbol)}:{_sanitize(timeframe)}:"
        f"{periods}:{data_hash}"
    )


def get_ohlcv_key_from_params(
    mode: str,
    lookback: int,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> str:
    """Build OHLCV key using SETTINGS for symbol/timeframe."""
    return ohlcv_key(
        symbol or SETTINGS.symbol,
        timeframe or SETTINGS.timeframe,
        mode,
        lookback,
        exchange_id=exchange_id or SETTINGS.default_exchange_id,
    )


def get_model_key_from_params(
    model_name: str,
    lookback: int,
    iterations: int,
    data_hash: str,
    exchange_id: str | None = None,
) -> str:
    """Build model key using SETTINGS for symbol/timeframe."""
    return model_key(
        ModelKeyParams(
            model_name=model_name,
            symbol=SETTINGS.symbol,
            timeframe=SETTINGS.timeframe,
            lookback=lookback,
            iterations=iterations,
            data_hash=data_hash,
        ),
        exchange_id=exchange_id or SETTINGS.default_exchange_id,
    )


def get_prediction_key_from_params(
    model_name: str,
    periods: int,
    data_hash: str,
    exchange_id: str | None = None,
) -> str:
    """Build prediction key using SETTINGS for symbol/timeframe."""
    return prediction_key(
        model_name,
        SETTINGS.symbol,
        SETTINGS.timeframe,
        periods,
        data_hash,
        exchange_id=exchange_id or SETTINGS.default_exchange_id,
    )
