"""Explore command module."""

from ...config import SETTINGS
from ...data.fetcher import fetch_with_retry
from ...viz.rich import print_data_summary
from ...viz.theme import get_console
from ..parsers import _as_mode


def register(app) -> None:
    """Register explore command on the provided Typer app."""

    @app.command()
    def explore(
        mode: str = "historical",
        lookback: int = SETTINGS.default_lookback_days,
        no_cache: bool = False,
    ) -> None:
        """Explore data statistics and correlations."""
        from rich.table import Table

        cli_console = get_console()
        with cli_console.status("[status.running]Fetching data..."):
            data = fetch_with_retry(_as_mode(mode), lookback, use_cache=not no_cache)

        if data.empty:
            cli_console.print("[error]Failed to fetch data[/error]")
            raise SystemExit(1)

        print_data_summary(data, f"Data Exploration ({lookback}d)")

        corr_table = Table(
            title="Correlation Matrix",
            title_style="panel.title",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
            header_style="table.header",
            row_styles=["table.row_even", "table.row_odd"],
        )
        corr_table.add_column("", style="brand", no_wrap=True)
        for col in ["open", "high", "low", "close", "volume"]:
            from ..viz.components import DATA_STYLES

            corr_table.add_column(col, style=DATA_STYLES.get(col, "ui.text"), justify="right")

        corr = data[["open", "high", "low", "close", "volume"]].corr()
        for idx, row in corr.iterrows():
            corr_table.add_row(str(idx), *[f"{v:.3f}" for v in row])

        cli_console.print(corr_table)
