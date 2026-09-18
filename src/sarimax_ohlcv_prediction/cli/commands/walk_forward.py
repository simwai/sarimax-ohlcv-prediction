"""Walk-forward backtest command module."""

from ...backtest.walk_forward import run_walk_forward
from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...viz.components import print_backtest_results
from ...viz.theme import get_console
from ..parsers import _as_mode, _model_class


def register(app) -> None:
    """Register walk-forward command on the provided Typer app."""

    @app.command()
    def walk_forward(
        model: str = "SARIMAX",
        strategy: str = "exit_after_n",
        train_window: int = 500,
        test_window: int = 120,
        step_size: int = 60,
        exit_bars: int = SETTINGS.backtest_default_exit_bars,
        mode: str = "historical",
        data_lookback: int = 2000,
        exchange: str = SETTINGS.default_exchange_id,
        symbol: str = SETTINGS.symbol,
        timeframe: str = SETTINGS.timeframe,
        no_cache: bool = False,
    ) -> None:
        """Run rolling walk-forward validation."""
        cli_console = get_console()
        _model_class(model)

        with cli_console.status("[status.running]Fetching data for walk-forward..."):
            data = fetch_with_retry(
                _as_mode(mode),
                data_lookback,
                use_cache=not no_cache,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )

        if data.empty:
            cli_console.print("[error]Failed to fetch data[/error]")
            raise SystemExit(1)

        with cli_console.status("[status.running]Running walk-forward backtest..."):
            result = run_walk_forward(
                model_name=model,
                data=data,
                strategy_name=strategy,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                exit_bars=exit_bars,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                no_cache=no_cache,
            )

        print_backtest_results(
            total_return=result.avg_total_return,
            calmar=result.avg_calmar_ratio,
            sortino=result.avg_sortino_ratio,
            max_dd=result.avg_max_drawdown,
            win_rate=result.avg_win_rate,
            total_trades=result.total_trades,
            title=f"Walk-Forward Results ({model} / {strategy})",
        )
