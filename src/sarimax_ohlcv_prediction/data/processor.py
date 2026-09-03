"""Data processing utilities."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import SETTINGS


def prepare_lstm_data(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    sequence_length: int | None = None,
) -> tuple:
    """Prepare data for LSTM training.

    Args:
        data: OHLCV DataFrame
        columns: Columns to use (default: OHLCV)
        sequence_length: Lookback window (default from settings)

    Returns:
        Tuple of (X_train, y_train, scalers_dict)
    """
    cols = columns or ["open", "high", "low", "close", "volume"]
    seq_len = sequence_length or SETTINGS.lstm_sequence_length

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
        pd.array(X),
        pd.array(y),
        scalers,
    )


def inverse_transform_predictions(
    predictions: pd.DataFrame,
    scalers: dict,
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
    freq: str = "5T",
) -> pd.DatetimeIndex:
    """Create future timestamps for predictions."""
    start = last_timestamp + pd.Timedelta(minutes=5)
    return pd.date_range(start=start, periods=periods, freq=freq)
