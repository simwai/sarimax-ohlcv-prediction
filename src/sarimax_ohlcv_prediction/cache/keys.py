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


def ohlcv_key(symbol: str, timeframe: str, mode: str, lookback: int) -> str:
    """Build cache key for OHLCV data fetch.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "5m")
        mode: "current" or "historical"
        lookback: Lookback days

    Returns:
        Cache key string
    """
    return f"ohlcv:{_sanitize(symbol)}:{_sanitize(timeframe)}:{mode}:{lookback}"


def model_key(params: ModelKeyParams) -> str:
    """Build cache key for trained model.

    Args:
        params: Bundled model identity and training parameters.

    Returns:
        Cache key string
    """
    return (
        f"model:{params.model_name}:{_sanitize(params.symbol)}:{_sanitize(params.timeframe)}:"
        f"{params.lookback}:{params.iterations}:{params.data_hash}"
    )


def prediction_key(
    model_name: str,
    symbol: str,
    timeframe: str,
    periods: int,
    data_hash: str,
) -> str:
    """Build cache key for model predictions.

    Args:
        model_name: Model class name
        symbol: Trading symbol
        timeframe: Timeframe
        periods: Number of prediction periods
        data_hash: Hash of context data (last N rows)

    Returns:
        Cache key string
    """
    return (
        f"pred:{model_name}:{_sanitize(symbol)}:{_sanitize(timeframe)}:"
        f"{periods}:{data_hash}"
    )


def get_ohlcv_key_from_params(mode: str, lookback: int) -> str:
    """Build OHLCV key using SETTINGS for symbol/timeframe."""
    return ohlcv_key(SETTINGS.symbol, SETTINGS.timeframe, mode, lookback)


def get_model_key_from_params(
    model_name: str,
    lookback: int,
    iterations: int,
    data_hash: str,
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
        )
    )


def get_prediction_key_from_params(
    model_name: str,
    periods: int,
    data_hash: str,
) -> str:
    """Build prediction key using SETTINGS for symbol/timeframe."""
    return prediction_key(model_name, SETTINGS.symbol, SETTINGS.timeframe, periods, data_hash)
