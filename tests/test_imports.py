"""Basic tests for package structure."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sarimax_ohlcv_prediction import backtest, cli, config, data, models, viz
from sarimax_ohlcv_prediction.config import SETTINGS
from sarimax_ohlcv_prediction.models import MODEL_REGISTRY, create_model

DEFAULT_LOOKBACK_DAYS = 100
LSTM_SEQUENCE_LENGTH = 60


def test_imports() -> None:
    """Test that all modules can be imported."""
    assert config.SETTINGS is not None
    assert hasattr(data, "fetcher")
    assert hasattr(data, "processor")
    assert hasattr(models, "SARIMAXModel")
    assert hasattr(models, "ProphetModel")
    assert hasattr(models, "LSTMModel")
    assert hasattr(backtest, "run_backtest")
    assert hasattr(viz, "plotly")
    assert hasattr(viz, "rich")
    assert hasattr(cli, "app")
    assert hasattr(cli, "repl")


def test_model_registry() -> None:
    """Test model registry."""
    assert "SARIMAX" in MODEL_REGISTRY
    assert "Prophet" in MODEL_REGISTRY
    assert "LSTM" in MODEL_REGISTRY

    # Test creation
    for name in MODEL_REGISTRY:
        model = create_model(name)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_config() -> None:
    """Test config values."""
    assert SETTINGS.symbol == "BTC/USDT"
    assert SETTINGS.timeframe == "5m"
    assert SETTINGS.default_lookback_days == DEFAULT_LOOKBACK_DAYS
    assert SETTINGS.lstm_sequence_length == LSTM_SEQUENCE_LENGTH


if __name__ == "__main__":
    test_imports()
    test_model_registry()
    test_config()
    print("All tests passed!")