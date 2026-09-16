"""Cache management command module."""

from ...cache import cache_clear, cache_inspect, cache_stats
from ...viz.theme import get_console


def register(cache_app) -> None:
    """Register cache subcommands on the provided Typer sub-app."""

    cli_console = get_console()

    @cache_app.command("clear")
    def cache_clear_cmd() -> None:
        """Clear all cache entries."""
        count = cache_clear()
        cli_console.print(f"[success]Cleared {count} cache entries[/success]")

    @cache_app.command("stats")
    def cache_stats_cmd() -> None:
        """Show cache statistics."""
        from rich.table import Table

        stats = cache_stats()
        if not stats["enabled"]:
            cli_console.print("[warning]Cache is disabled[/warning]")
            return

        table = Table(
            title="Cache Statistics",
            title_style="panel.title",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
            header_style="table.header",
            row_styles=["table.row_even", "table.row_odd"],
        )
        table.add_column("Metric", style="brand")
        table.add_column("Value", style="ui.text")

        table.add_row("Enabled", str(stats["enabled"]))
        table.add_row("Total Entries", str(stats["entries"]))
        table.add_row("Active Entries", str(stats["active"]))
        table.add_row("Expired Entries", str(stats["expired"]))
        table.add_row("Size (bytes)", f"{stats['size_bytes']:,}")
        table.add_row("Cache Directory", stats["cache_dir"])

        cli_console.print(table)

    @cache_app.command("inspect")
    def cache_inspect_cmd(
        limit: int = 50,
    ) -> None:
        """Inspect cache entries."""
        from rich.table import Table

        entries = cache_inspect(limit)
        if not entries:
            cli_console.print("[warning]Cache is empty or disabled[/warning]")
            return

        table = Table(
            title=f"Cache Entries (showing {len(entries)})",
            title_style="panel.title",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
            header_style="table.header",
            row_styles=["table.row_even", "table.row_odd"],
        )
        table.add_column("Key", style="brand", max_width=60)
        table.add_column("TTL Remaining (s)", style="warning")
        table.add_column("Status", style="ui.text", justify="center")

        for entry in entries:
            status = "[error]EXPIRED[/error]" if entry["expired"] else "[success]ACTIVE[/success]"
            table.add_row(entry["key"], str(entry["ttl_remaining"]), status)

        cli_console.print(table)
