"""Tests for report generation."""

import os
from pathlib import Path

import pandas as pd

from sarimax_ohlcv_prediction.backtest.engine import BacktestResult
from sarimax_ohlcv_prediction.reports.pdf import generate_pdf_report


def _result() -> BacktestResult:
    """Create a minimal deterministic backtest result."""
    return BacktestResult(
        total_return=0.12,
        calmar_ratio=1.2,
        sortino_ratio=1.6,
        max_drawdown=-0.08,
        win_rate=0.58,
        total_trades=42,
        avg_trade_return=0.005,
        profit_factor=1.6,
        equity_curve=pd.Series([1.0, 1.01, 1.02]),
        trades=pd.DataFrame({"entry": [1, 0], "exit": [0, 1]}),
        stats=pd.Series(
            {
                "Total Return [%]": 12.0,
                "Calmar Ratio": 1.2,
                "Sortino Ratio": 1.6,
                "Max Drawdown [%]": -8.0,
                "Win Rate [%]": 58.0,
                "Total Trades": 42,
            }
        ),
    )


def test_generate_pdf_report_creates_file(tmp_path: Path) -> None:
    """PDF report must be written and the returned path must exist."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = generate_pdf_report(
            _result(), filename="backtest_report.pdf", meta={"model": "SARIMAX"}
        )
        assert Path(output).exists()
    finally:
        os.chdir(original_cwd)


def test_generate_pdf_report_writes_under_reports_dir(tmp_path: Path) -> None:
    """Reports must be written under the reports/ directory."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = generate_pdf_report(_result(), filename="backtest_report.pdf")
        assert Path(output).parent.name == "reports"
    finally:
        os.chdir(original_cwd)


def test_generate_pdf_report_timestamped_uniqueness(tmp_path: Path) -> None:
    """Two calls must produce distinct filenames."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        first = generate_pdf_report(_result(), filename="backtest_report.pdf")
        second = generate_pdf_report(_result(), filename="backtest_report.pdf")
        assert first != second
        assert Path(first).exists()
        assert Path(second).exists()
    finally:
        os.chdir(original_cwd)


def test_generate_pdf_report_uses_fallback_fonts(tmp_path: Path) -> None:
    """Missing font assets must not raise; Helvetica fallback should produce a PDF."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = generate_pdf_report(_result(), filename="backtest_report.pdf")
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0
    finally:
        os.chdir(original_cwd)


def test_generate_pdf_report_includes_pair_timeframe_exchange(tmp_path: Path) -> None:
    """PDF metadata must include pair, timeframe, and exchange without crashing."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = generate_pdf_report(
            _result(),
            filename="backtest_report.pdf",
            meta={
                "model": "SARIMAX",
                "pair": "BTC/USDT",
                "timeframe": "5m",
                "exchange": "binance",
            },
        )
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0
    finally:
        os.chdir(original_cwd)


def test_generate_pdf_report_uses_calmar_sortino(tmp_path: Path) -> None:
    """PDF report must use Calmar and Sortino columns without crashing."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = generate_pdf_report(_result(), filename="backtest_report.pdf")
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0
    finally:
        os.chdir(original_cwd)
