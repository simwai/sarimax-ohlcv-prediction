"""Regression tests for REVIEW-accepted defects (PATCH W-items).

Each test fails against the pre-fix behavior and passes post-fix.
Fast and deterministic: no network, no model training.
"""

from typing import Any

import joblib
import pandas as pd
import pytest
import typer

from sarimax_ohlcv_prediction.backtest.strategies import ExitAfterNBarsStrategy
from sarimax_ohlcv_prediction.cli.main import _model_class
from sarimax_ohlcv_prediction.cli.repl_core import _parse_compare_args, _parse_int
from sarimax_ohlcv_prediction.data import modes
from sarimax_ohlcv_prediction.data.processor import create_future_timestamps, prepare_lstm_data
from sarimax_ohlcv_prediction.models.prophet import ProphetModel
from sarimax_ohlcv_prediction.models.sarimax import SARIMAXModel
from sarimax_ohlcv_prediction.streamlit_app import write_to_csv
from sarimax_ohlcv_prediction.viz.components import print_data_summary, print_predictions_table

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

_EXPECTED_ROWS = 70
_EXPECTED_SAMPLES = 10
_EXPECTED_SEQ = 60
_EXPECTED_FEATURES = 5
_EXPECTED_X_NDIM = 3
_EXPECTED_Y_NDIM = 2
_EXPECTED_PARSED = 12
_EXPECTED_PERIODS = 24
_EXPECTED_HEADER_LINES = 3
_EXPECTED_APPENDED_LINES = 5


def _ohlcv(rows=70):
    """OHLCV frame with distinct constant values per column."""
    values = {"open": 1.0, "high": 2.0, "low": 0.5, "close": 100.0, "volume": 10.0}
    data: dict[str, Any] = {"timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min")}
    for col in OHLCV_COLS:
        data[col] = [values[col]] * rows
    return pd.DataFrame(data)


def test_prepare_lstm_data_returns_3d_arrays():
    """Windows must stack to (samples, sequence, features)."""
    x, y, _scalers = prepare_lstm_data(_ohlcv(_EXPECTED_ROWS))
    assert x.ndim == _EXPECTED_X_NDIM
    assert y.ndim == _EXPECTED_Y_NDIM
    assert x.shape == (_EXPECTED_SAMPLES, _EXPECTED_SEQ, _EXPECTED_FEATURES)


def test_prepare_lstm_data_rejects_empty():
    """Empty input must raise ValueError, not KeyError."""
    with pytest.raises(ValueError):
        prepare_lstm_data(pd.DataFrame())


def test_future_timestamps_rejects_nonpositive():
    """Non-positive periods must raise instead of returning empties."""
    with pytest.raises(ValueError):
        create_future_timestamps(pd.Timestamp("2024-01-01"), 0)


def test_print_predictions_table_runs():
    """Predictions table must render without AttributeError."""
    print_predictions_table(_ohlcv(3)[OHLCV_COLS])


def test_print_data_summary_shows_each_column(capsys):
    """Summary means must reflect every column, not just the first."""
    print_data_summary(_ohlcv(10))
    out = capsys.readouterr().out
    assert "100.00" in out
    assert "1.00" in out


def test_sarimax_rejects_foreign_envelope(tmp_path):
    """Unrecognized SARIMAX payloads must raise ValueError."""
    path = tmp_path / "foreign.joblib"
    joblib.dump({"models": {}}, path)
    with pytest.raises(ValueError):
        SARIMAXModel.load(str(path))


def test_prophet_rejects_foreign_envelope(tmp_path):
    """Unrecognized Prophet payloads must raise ValueError."""
    path = tmp_path / "foreign.joblib"
    joblib.dump({"models": {}}, path)
    with pytest.raises(ValueError):
        ProphetModel.load(str(path))


def test_model_class_validation():
    """Unknown CLI model names must raise BadParameter, not KeyError."""
    assert _model_class("SARIMAX") is not None
    with pytest.raises(typer.BadParameter):
        _model_class("NOPE")


def test_as_mode_strict():
    """Shared mode validator must reject unknown modes."""
    assert modes.as_mode("current") == "current"
    with pytest.raises(ValueError):
        modes.as_mode("bogus")


def test_parse_int_guards():
    """REPL int parsing must return None on garbage, not raise."""
    assert _parse_int("12", "periods") == _EXPECTED_PARSED
    assert _parse_int("abc", "periods") is None


def test_compare_args_accept_custom_periods(capsys):
    """Explicit periods must not print a false Invalid periods error."""
    periods, _fast, _holdout, _timeout = _parse_compare_args(["24"])
    assert periods == _EXPECTED_PERIODS
    assert "Invalid periods" not in capsys.readouterr().out


def test_exit_bars_guard():
    """Non-positive exit bars must raise instead of silent nonsense."""
    data = _ohlcv(10)
    with pytest.raises(ValueError):
        ExitAfterNBarsStrategy().generate_signals(data, data, exit_bars=0)


def test_write_csv_roundtrip_and_append(tmp_path):
    """CSV writes must be atomic, headed, and preserve prior rows."""
    preds = pd.DataFrame({col: [1.0, 2.0] for col in OHLCV_COLS})
    target = tmp_path / "pred.csv"
    write_to_csv(preds, str(target), last_timestamp=pd.Timestamp("2024-01-01"))
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,open")
    assert len(lines) == _EXPECTED_HEADER_LINES
    write_to_csv(preds, str(target), last_timestamp=pd.Timestamp("2024-01-01"))
    assert len(target.read_text(encoding="utf-8").splitlines()) == _EXPECTED_APPENDED_LINES


def test_write_csv_rejects_missing_columns(tmp_path):
    """CSV writes must reject frames missing OHLCV columns."""
    with pytest.raises(ValueError):
        write_to_csv(pd.DataFrame({"open": [1.0]}), str(tmp_path / "x.csv"))
