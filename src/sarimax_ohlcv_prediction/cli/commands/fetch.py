"""Fetch command module."""

from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...viz.rich import print_data_summary
from ..parsers import _as_mode


def register(app) -> None:
    """Register fetch command on the provided Typer app."""

    @app.command()
    def fetch(
        mode: str = "current",
        lookback: int = SETTINGS.default_lookback_days,
        exchange: str = SETTINGS.default_exchange_id,
        symbol: str = SETTINGS.symbol,
        timeframe: str = SETTINGS.timeframe,
        output: str | None = None,
        no_cache: bool = False,
    ) -> None:
        """Fetch OHLCV data from an exchange."""
        from ..viz.theme import get_console

        cli_console = get_console()
        with cli_console.status(f"[status.running]Fetching {mode} data..."):
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

        print_data_summary(
            data, f"Fetched Data ({exchange}, {symbol}, {timeframe}, {mode}, {lookback}d)"
        )

        if output:
            data.to_csv(output, index=False)
            cli_console.print(f"[success]Saved to {output}[/success]")
