"""Train command module."""

from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...models import get_cached_model
from ...viz.theme import get_console
from ..parsers import _as_mode, _model_class


def register(app) -> None:
    """Register train command on the provided Typer app."""

    @app.command()
    def train(
        model: str = "SARIMAX",
        mode: str = "historical",
        lookback: int = SETTINGS.default_lookback_days,
        iterations: int = SETTINGS.default_iterations,
        exchange: str = SETTINGS.default_exchange_id,
        symbol: str = SETTINGS.symbol,
        timeframe: str = SETTINGS.timeframe,
        save_path: str | None = None,
        no_cache: bool = False,
    ) -> None:
        """Train a model on historical data."""
        cli_console = get_console()
        _model_class(model)

        with cli_console.status("[status.running]Fetching training data..."):
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

        with cli_console.status(f"[status.running]Training {model}..."):
            model_instance = get_cached_model(
                model,
                data,
                lookback=lookback,
                iterations=iterations,
                use_cache=not no_cache,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )

        cli_console.print(f"[success]{model} trained successfully[/success]")

        if save_path:
            model_instance.save(save_path)
            cli_console.print(f"[success]Model saved to {save_path}[/success]")
