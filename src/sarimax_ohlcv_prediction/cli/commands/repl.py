"""REPL command module."""

from ...cli.repl_inquirer import run_repl


def register(app) -> None:
    """Register repl command on the provided Typer app."""

    @app.command()
    def repl() -> None:
        """Start interactive REPL (inquirer-style with autocomplete)."""
        run_repl()
