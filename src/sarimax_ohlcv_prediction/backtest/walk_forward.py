"""Rolling walk-forward backtesting."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .engine import BacktestResult, run_backtest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardResult:
    """Results from a rolling walk-forward backtest."""

    windows: list[BacktestResult]
    avg_total_return: float
    avg_calmar_ratio: float
    avg_sortino_ratio: float
    avg_max_drawdown: float
    avg_win_rate: float
    total_trades: int


def _aggregate(results: list[BacktestResult]) -> WalkForwardResult:
    total_return = sum(result.total_return for result in results) / len(results)
    calmar = sum(result.calmar_ratio for result in results) / len(results)
    sortino = sum(result.sortino_ratio for result in results) / len(results)
    max_dd = sum(result.max_drawdown for result in results) / len(results)
    win_rate = sum(result.win_rate for result in results) / len(results)
    total_trades = sum(result.total_trades for result in results)
    return WalkForwardResult(
        windows=results,
        avg_total_return=total_return,
        avg_calmar_ratio=calmar,
        avg_sortino_ratio=sortino,
        avg_max_drawdown=max_dd,
        avg_win_rate=win_rate,
        total_trades=total_trades,
    )


def run_walk_forward(  # noqa: PLR0912, PLR0913
    model_name: str,
    data: pd.DataFrame,
    strategy_name: str = "exit_after_n",
    train_window: int = 500,
    test_window: int = 120,
    step_size: int = 60,
    exit_bars: int = 5,
    exchange_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    no_cache: bool = False,
    **strategy_params: Any,  # pyrefly: ignore -- open strategy params
) -> WalkForwardResult:
    """Run rolling walk-forward backtest.

    Args:
        model_name: Model to use
        data: Full historical OHLCV dataset
        strategy_name: Strategy name
        train_window: Number of bars used for training
        test_window: Number of bars used for testing
        step_size: Step size between windows
        exit_bars: Exit bars for exit_after_n strategy
        exchange_id: Optional exchange id for cache namespacing
        symbol: Optional symbol override
        timeframe: Optional timeframe override
        no_cache: Whether to skip model caching
        **strategy_params: Additional strategy parameters

    Returns:
        WalkForwardResult with per-window and aggregated metrics
    """
    _DATA_EMPTY_MSG = "data is empty"
    _TRAIN_WINDOW_POSITIVE_MSG = "train_window must be positive, got {value}"
    _TEST_WINDOW_POSITIVE_MSG = "test_window must be positive, got {value}"
    _STEP_SIZE_POSITIVE_MSG = "step_size must be positive, got {value}"
    _WINDOW_EXCEEDS_MSG = "train_window + test_window ({value}) exceeds data length ({data_len})"
    _NO_WINDOWS_MSG = "No walk-forward windows were generated"

    if data.empty:
        raise ValueError(_DATA_EMPTY_MSG)
    if train_window <= 0:
        raise ValueError(_TRAIN_WINDOW_POSITIVE_MSG.format(value=train_window))
    if test_window <= 0:
        raise ValueError(_TEST_WINDOW_POSITIVE_MSG.format(value=test_window))
    if step_size <= 0:
        raise ValueError(_STEP_SIZE_POSITIVE_MSG.format(value=step_size))

    if train_window + test_window > len(data):
        raise ValueError(
            _WINDOW_EXCEEDS_MSG.format(value=train_window + test_window, data_len=len(data))
        )

    results: list[BacktestResult] = []
    start = 0

    while start + train_window + test_window <= len(data):
        train_data = data.iloc[start : start + train_window].copy()
        test_data = data.iloc[start + train_window : start + train_window + test_window].copy()

        if test_data.empty:
            break

        logger.info(
            "Walk-forward window %d-%d (%d train, %d test)",
            start,
            start + train_window + test_window,
            len(train_data),
            len(test_data),
        )

        model_cls = getattr(
            __import__("sarimax_ohlcv_prediction.models", fromlist=[model_name]),
            model_name,
        )
        model = model_cls()
        model.fit(train_data)

        window_result = run_backtest(
            model=model,
            data=test_data,
            strategy_name=strategy_name,
            exit_bars=exit_bars,
            lookback=len(test_data),
            **strategy_params,
        )
        results.append(window_result)
        start += step_size

    if not results:
        raise ValueError(_NO_WINDOWS_MSG)

    return _aggregate(results)
