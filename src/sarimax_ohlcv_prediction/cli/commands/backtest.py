"""Backtest command module."""

from ...backtest import run_backtest
from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...models import get_cached_model
from ...viz.components import print_backtest_results
from ...viz.theme import get_console
from ..parsers import _as_mode, _model_class


def register(app) -> None:
    """Register backtest command on the provided Typer app."""

    @app.command()
    def backtest(
        model: str = "SARIMAX",
        model_path: str | None = None,
        strategy: str = "exit_after_n",
        lookback: int = SETTINGS.backtest_default_lookback,
        exit_bars: int = SETTINGS.backtest_default_exit_bars,
        mode: str = "historical",
        data_lookback: int = 500,
        no_cache: bool = False,
    ) -> None:
        """Run backtest on historical data."""
        cli_console = get_console()
        _model_class(model)

        if model_path:
            with cli_console.status(f"[status.running]Loading {model}..."):
                model_class = _model_class(model)
                model_instance = model_class.load(model_path)
        else:
            with cli_console.status("[status.running]Fetching data and training..."):
                data = fetch_with_retry(_as_mode(mode), data_lookback, use_cache=not no_cache)
                if data.empty:
                    cli_console.print("[error]Failed to fetch data[/error]")
                    raise SystemExit(1)
                model_instance = get_cached_model(
                    model, data, lookback=data_lookback, use_cache=not no_cache
                )

        with cli_console.status("[status.running]Fetching test data..."):
            test_data = fetch_with_retry(_as_mode(mode), lookback, use_cache=not no_cache)
            if test_data.empty:
                cli_console.print("[error]Failed to fetch test data[/error]")
                raise SystemExit(1)

        with cli_console.status("[status.running]Running backtest..."):
            result = run_backtest(
                model_instance,
                test_data,
                strategy_name=strategy,
                exit_bars=exit_bars,
                lookback=lookback,
            )

        print_backtest_results(
            total_return=result.total_return,
            calmar=result.calmar_ratio,
            sortino=result.sortino_ratio,
            max_dd=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            title=f"Backtest Results ({model} / {strategy})",
        )
