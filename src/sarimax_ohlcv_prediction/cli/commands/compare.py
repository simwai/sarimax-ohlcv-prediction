"""Compare command module."""

from ...cli.compare_core import _run_compare_core
from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...viz.rich import print_model_comparison
from ...viz.theme import get_console
from ..parsers import _as_mode


def register(app) -> None:
    """Register compare command on the provided Typer app."""

    @app.command()
    def compare(
        periods: int = SETTINGS.default_prediction_periods,
        mode: str = "current",
        lookback: int = SETTINGS.default_lookback_days,
        fast: bool = True,
        holdout: int | None = None,
        timeout: int | None = None,
        exchange: str = SETTINGS.default_exchange_id,
        symbol: str = SETTINGS.symbol,
        timeframe: str = SETTINGS.timeframe,
        no_cache: bool = False,
    ) -> None:
        """Compare all models with progress, scoring, timeout - non-REPL version."""
        cli_console = get_console()

        with cli_console.status(f"[status.running]Fetching {mode} data ({lookback}d)..."):
            data = fetch_with_retry(
                _as_mode(mode),
                lookback,
                use_cache=not no_cache,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )

        if data.empty:
            cli_console.print("[error]Failed to fetch data[/error]")
            raise SystemExit(1)

        from ..viz.rich import print_data_summary

        print_data_summary(
            data, f"Fetched Data ({exchange}, {symbol}, {timeframe}, {mode}, {lookback}d)"
        )

        if timeout is None:
            timeout = (
                SETTINGS.compare_timeout_seconds if fast else SETTINGS.compare_full_timeout_seconds
            )

        results = _run_compare_core(data, periods, fast, holdout, timeout)
        holdout_str = holdout if holdout is not None else periods
        title = f"Model Comparison (periods={periods} holdout={holdout_str} {'fast' if fast else 'full'})"
        print_model_comparison(results, title=title)
