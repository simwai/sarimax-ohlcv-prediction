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
        output: str | None = None,
        no_cache: bool = False,
    ) -> None:
        """Fetch OHLCV data from Binance."""
        from ..viz.theme import get_console

        cli_console = get_console()
        with cli_console.status(f"[status.running]Fetching {mode} data..."):
            data = fetch_with_retry(_as_mode(mode), lookback, use_cache=not no_cache)

        if data.empty:
            cli_console.print("[error]Failed to fetch data[/error]")
            raise SystemExit(1)

        print_data_summary(data, f"Fetched Data ({mode}, {lookback}d)")

        if output:
            data.to_csv(output, index=False)
            cli_console.print(f"[success]Saved to {output}[/success]")
