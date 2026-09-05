"""Data processing utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import SETTINGS

_EMPTY_DATA_MSG = "LSTM training data is empty"
_MISSING_COLS_MSG = "Data missing columns: {missing}"
_SHORT_DATA_MSG = "Not enough rows ({got}) for sequence length {need}"
_PERIODS_MSG = "periods must be positive, got {periods}"


def prepare_lstm_data(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    sequence_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, MinMaxScaler]]:
    """Prepare data for LSTM training.

    Args:
        data: OHLCV DataFrame
        columns: Columns to use (default: OHLCV)
        sequence_length: Lookback window (default from settings)

    Returns:
        Tuple of (X_train, y_train, scalers_dict)
    """
    cols = columns or ["open", "high", "low", "close", "volume"]
    seq_len = SETTINGS.lstm_sequence_length if sequence_length is None else sequence_length
    if data.empty:
        raise ValueError(_EMPTY_DATA_MSG)  # noqa: TRY003
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(_MISSING_COLS_MSG.format(missing=missing))  # noqa: TRY003
    if len(data) <= seq_len:
        raise ValueError(_SHORT_DATA_MSG.format(got=len(data), need=seq_len))  # noqa: TRY003

    scalers = {}
    scaled_data = {}

    for col in cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data[[col]])
        scalers[col] = scaler
        scaled_data[col] = scaled.flatten()

    scaled_df = pd.DataFrame(scaled_data, index=data.index)

    X, y = [], []
    for i in range(seq_len, len(scaled_df)):
        X.append(scaled_df[cols].iloc[i - seq_len : i].values)
        y.append(scaled_df[cols].iloc[i].values)

    return (
        np.asarray(X),
        np.asarray(y),
        scalers,
    )


def inverse_transform_predictions(
    predictions: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Inverse transform scaled predictions back to original scale."""
    cols = columns or ["open", "high", "low", "close", "volume"]
    result = predictions.copy()

    for col in cols:
        if col in scalers and col in result.columns:
            result[col] = scalers[col].inverse_transform(result[[col]])

    return result


def create_future_timestamps(
    last_timestamp: pd.Timestamp,
    periods: int,
    freq: str = "5min",
) -> pd.DatetimeIndex:
    """Create future timestamps for predictions."""
    if periods <= 0:
        raise ValueError(_PERIODS_MSG.format(periods=periods))  # noqa: TRY003
    start = last_timestamp + pd.Timedelta(minutes=5)
    return pd.date_range(start=start, periods=periods, freq=freq)
