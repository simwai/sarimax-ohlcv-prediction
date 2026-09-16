"""Main CLI application with Typer and Rich REPL."""

import logging

import typer

from ..viz.theme import get_console
from .commands.backtest import register as register_backtest
from .commands.cache import register as register_cache_commands
from .commands.compare import register as register_compare
from .commands.explore import register as register_explore
from .commands.fetch import register as register_fetch
from .commands.predict import register as register_predict
from .commands.repl import register as register_repl
from .commands.train import register as register_train
from .parsers import _as_mode, _model_class  # noqa: F401 -- re-exported for tests/back-compat

app = typer.Typer(
    name="sarimax-ohlcv",
    help="Bitcoin price prediction with SARIMAX, Prophet, and LSTM",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

cache_app = typer.Typer(name="cache", help="Cache management commands")
app.add_typer(cache_app, name="cache")

cli_console = get_console()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Bitcoin OHLCV Prediction CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


register_fetch(app)
register_train(app)
register_predict(app)
register_backtest(app)
register_explore(app)
register_compare(app)
register_repl(app)
register_cache_commands(cache_app)


if __name__ == "__main__":
    app()
