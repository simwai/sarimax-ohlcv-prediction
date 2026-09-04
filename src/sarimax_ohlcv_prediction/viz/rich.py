"""Rich console visualizations for CLI/REPL - using centralized theme."""

from .components import (
    console,
    print_backtest_results,
    print_data_summary,
    print_model_comparison,
    print_predictions_table,
    status_spinner,
)
from .theme import ICONS, PALETTE, REPL_THEME, get_console

__all__ = [
    "console",
    "print_data_summary",
    "print_model_comparison",
    "print_predictions_table",
    "print_backtest_results",
    "status_spinner",
    "ICONS",
    "PALETTE",
    "REPL_THEME",
    "get_console",
]