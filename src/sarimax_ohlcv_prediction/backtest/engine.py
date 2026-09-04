"""Backtesting engine using vectorbt."""

import logging
from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from ..models.base import BaseModel, ModelBase
from ..viz.components import print_backtest_results
from .strategies import BaseStrategy, get_strategy

logger = logging.getLogger(__name__)

_DATA_EMPTY_MSG = "backtest data is empty"
_LOOKBACK_POSITIVE_MSG = "lookback must be positive, got {lookback}"
_LOOKBACK_EXCEEDS_MSG = "lookback {lookback} exceeds data length {data_len}"
_TEST_DATA_EMPTY_MSG = "test_data is empty after slicing"
_NOT_DATAFRAME_MSG = "model.predict returned {got}, expected DataFrame"
_PRED_EMPTY_MSG = "model predictions empty"
_PRED_MISSING_COLS_MSG = "predictions missing columns: {missing}"
_PRED_LEN_MISMATCH_MSG = "predictions length {got} != expected {expected}"
_PRED_NAN_MSG = "predictions contain NaN"


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float
    equity_curve: pd.Series
    trades: pd.DataFrame
    stats: pd.Series


def run_backtest(  # noqa: PLR0912
    model: BaseModel,
    data: pd.DataFrame,
    strategy_name: str = "exit_after_n",
    exit_bars: int = 5,
    lookback: int | None = None,
    **strategy_params,
) -> BacktestResult:
    """Run backtest on historical data using model predictions.

    Args:
        model: Fitted model instance
        data: Historical OHLCV data
        strategy_name: Name of strategy to use
        exit_bars: Number of bars to hold (for exit_after_n)
        lookback: Number of recent bars to test on (default: all)
        **strategy_params: Additional strategy parameters

    Returns:
        BacktestResult with performance metrics

    Raises:
        ValueError: if data is empty, lookback invalid, or predictions malformed
        TypeError: if model returns wrong type
        RuntimeError: if model not fitted
        NotImplementedError: if LSTM predict() called without context
    """
    if data.empty:
        raise ValueError(_DATA_EMPTY_MSG)  # noqa: TRY003
    if lookback is not None and lookback <= 0:
        raise ValueError(_LOOKBACK_POSITIVE_MSG.format(lookback=lookback))  # noqa: TRY003
    lookback = lookback or len(data)
    if lookback > len(data):
        raise ValueError(  # noqa: TRY003
            _LOOKBACK_EXCEEDS_MSG.format(lookback=lookback, data_len=len(data))
        )
    test_data = data.iloc[-lookback:].copy()

    if test_data.empty:
        raise ValueError(_TEST_DATA_EMPTY_MSG)  # noqa: TRY003

    logger.info("Running backtest on %d bars with strategy: %s", len(test_data), strategy_name)

    # Fail-fast prediction - no silent fallback to perfect predictions
    n_periods = len(test_data)
    is_context_model = type(model).predict_with_context is not ModelBase.predict_with_context
    if is_context_model:
        predictions = model.predict_with_context(data, n_periods)
    else:
        predictions = model.predict(n_periods)

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(  # noqa: TRY003
            _NOT_DATAFRAME_MSG.format(got=type(predictions).__name__)
        )
    if predictions.empty:
        raise ValueError(_PRED_EMPTY_MSG)  # noqa: TRY003
    expected_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in expected_cols if c not in predictions.columns]
    if missing:
        raise ValueError(_PRED_MISSING_COLS_MSG.format(missing=missing))  # noqa: TRY003
    if len(predictions) != n_periods:
        raise ValueError(  # noqa: TRY003
            _PRED_LEN_MISMATCH_MSG.format(got=len(predictions), expected=n_periods)
        )
    if predictions.isna().any().any():
        raise ValueError(_PRED_NAN_MSG)  # noqa: TRY003
    predictions.index = test_data.index

    # Get strategy
    strategy: BaseStrategy = get_strategy(strategy_name)

    # Generate signals
    if strategy_name == "exit_after_n":
        entries, exits = strategy.generate_signals(
            test_data, predictions, exit_bars=exit_bars, **strategy_params
        )
    else:
        entries, exits = strategy.generate_signals(test_data, predictions, **strategy_params)

    # Run vectorbt portfolio simulation
    close_prices = test_data["close"]

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        freq="5T",
        init_cash=10000,
        fees=0.001,  # 0.1% fee
        slippage=0.0005,  # 0.05% slippage
    )

    stats = _normalize_stats(portfolio.stats())

    trades_df = _read_trades(portfolio)
    equity_curve = _read_equity_curve(portfolio)

    return BacktestResult(
        total_return=_stat_float(stats, "Total Return [%]", scale=0.01),
        sharpe_ratio=_stat_float(stats, "Sharpe Ratio"),
        max_drawdown=_stat_float(stats, "Max Drawdown [%]", scale=0.01),
        win_rate=_stat_float(stats, "Win Rate [%]", scale=0.01),
        total_trades=_stat_int(stats, "Total Trades"),
        avg_trade_return=_stat_float(stats, "Avg Trade Return [%]", scale=0.01),
        profit_factor=_stat_float(stats, "Profit Factor"),
        equity_curve=equity_curve,
        trades=trades_df,
        stats=stats,
    )


def _normalize_stats(raw_stats: pd.Series | pd.DataFrame | None) -> pd.Series:
    if raw_stats is None:
        return pd.Series(dtype=float)
    if isinstance(raw_stats, pd.DataFrame):
        if raw_stats.shape[1] == 1:
            return raw_stats.iloc[:, 0]
        if len(raw_stats) > 0:
            return raw_stats.iloc[0]
        return pd.Series(dtype=float)
    return raw_stats


def _read_trades(portfolio: vbt.Portfolio) -> pd.DataFrame:
    records = getattr(portfolio.trades, "records_readable", None)
    if isinstance(records, pd.DataFrame):
        return records
    return pd.DataFrame()


def _read_equity_curve(portfolio: vbt.Portfolio) -> pd.Series:
    value = portfolio.value()
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            return value.iloc[:, 0]
        if len(value) > 0:
            return value.iloc[0]
        return pd.Series(dtype=float)
    return pd.Series(dtype=float)


def _stat_float(stats: pd.Series, key: str, scale: float = 1.0) -> float:
    value = stats.get(key, 0)
    if value is None:
        return 0.0
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return 0.0


def _stat_int(stats: pd.Series, key: str) -> int:
    value = stats.get(key, 0)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def print_backtest_result(result: BacktestResult) -> None:
    """Pretty print backtest results using themed components."""
    print_backtest_results(
        total_return=result.total_return,
        sharpe=result.sharpe_ratio,
        max_dd=result.max_drawdown,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        title="Backtest Results",
    )
