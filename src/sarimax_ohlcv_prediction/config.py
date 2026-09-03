"""Configuration and constants."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    default_lookback_days: int = 100
    default_prediction_periods: int = 12
    default_iterations: int = 100
    csv_filename: str = "predicted_prices.csv"
    csv_max_size_mb: int = 10
    sleep_interval_seconds: int = 15 * 60
    error_sleep_seconds: int = 60

    # LSTM settings
    lstm_sequence_length: int = 60
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_validation_split: float = 0.1

    # Backtest settings
    backtest_default_exit_bars: int = 5
    backtest_default_lookback: int = 500

    # Compare / benchmark settings
    compare_fast_m_range: range = range(7, 10)  # 3 values: 7,8,9 (fast) vs 7 for full
    compare_full_m_range: range = range(7, 14)  # 7 values (capped from 43 for time budget)
    compare_fast_iterations: int = 10
    compare_full_iterations: int = 25
    compare_timeout_seconds: int = 90
    compare_full_timeout_seconds: int = 300

    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_ohlcv_current: int = 3600      # 1 hour
    cache_ttl_ohlcv_historical: int = 86400  # 24 hours
    cache_ttl_predictions: int = 1800        # 30 min
    cache_ttl_models: int = 604800           # 7 days


SETTINGS = Settings()
