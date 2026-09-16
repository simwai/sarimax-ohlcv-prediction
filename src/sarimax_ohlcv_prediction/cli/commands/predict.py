"""Predict command module."""

from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...models import get_cached_model
from ...viz.rich import print_predictions_table
from ...viz.theme import get_console
from ..parsers import _as_mode, _model_class


def register(app) -> None:
    """Register predict command on the provided Typer app."""

    @app.command()
    def predict(
        model: str = "SARIMAX",
        model_path: str | None = None,
        periods: int = SETTINGS.default_prediction_periods,
        mode: str = "current",
        lookback: int = SETTINGS.default_lookback_days,
        save_csv: str | None = None,
        no_cache: bool = False,
    ) -> None:
        """Make predictions using a trained model."""
        cli_console = get_console()
        _model_class(model)

        if model_path:
            with cli_console.status(f"[status.running]Loading {model} from {model_path}..."):
                model_class = _model_class(model)
                model_instance = model_class.load(model_path)
        else:
            with cli_console.status("[status.running]Fetching data and training..."):
                data = fetch_with_retry(_as_mode(mode), lookback, use_cache=not no_cache)
                if data.empty:
                    cli_console.print("[error]Failed to fetch data[/error]")
                    raise SystemExit(1)
                model_instance = get_cached_model(
                    model, data, lookback=lookback, use_cache=not no_cache
                )

        with cli_console.status("[status.running]Generating predictions..."):
            if model == "LSTM":
                data = fetch_with_retry(_as_mode(mode), lookback, use_cache=not no_cache)
                predictions = model_instance.predict_with_context(data, periods)
            else:
                predictions = model_instance.predict(periods)

        print_predictions_table(predictions, title=f"{model} Predictions ({periods} periods)")

        if save_csv:
            predictions.to_csv(save_csv, index=False)
            cli_console.print(f"[success]Predictions saved to {save_csv}[/success]")
