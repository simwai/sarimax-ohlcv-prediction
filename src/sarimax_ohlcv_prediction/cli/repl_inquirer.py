"""Inquirer-style REPL using prompt_toolkit with autocomplete, history, and fuzzy search."""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.patch_stdout import patch_stdout

from .repl import (
    _CMD_HANDLERS,
    parse_command,
    run_repl as run_rich_repl,
)
from ..viz.rich import console
from ..viz.theme import ICONS


COMMANDS = sorted(_CMD_HANDLERS.keys())

COMMAND_ARGS = {
    "fetch": ["current", "historical"],
    "train": ["SARIMAX", "Prophet", "LSTM"],
    "load": ["SARIMAX", "Prophet", "LSTM"],
    "predict": [],
    "compare": ["--fast", "--full", "--holdout", "--timeout"],
    "benchmark": ["--fast", "--full", "--holdout", "--timeout"],
    "backtest": ["exit_after_n", "exit_on_signal"],
    "models": [],
    "save": [],
    "explore": [],
    "show": [],
    "status": ["--verbose", "-v"],
    "help": [],
    "clear": [],
    "strategies": [],
    "exit": [],
    "quit": [],
}


class REPLCompleter(Completer):
    """Context-aware completer for REPL commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text:
            for cmd in COMMANDS:
                yield Completion(cmd, start_position=0, display=cmd, display_meta=self._get_meta(cmd))
            return

        parts = text.split()
        if len(parts) == 1:
            # Completing command name
            prefix = parts[0]
            for cmd in COMMANDS:
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix), display=cmd, display_meta=self._get_meta(cmd))
        else:
            # Completing arguments for a command
            cmd = parts[0]
            if cmd in COMMAND_ARGS:
                args = COMMAND_ARGS[cmd]
                current_arg = parts[-1] if not text.endswith(" ") else ""
                for arg in args:
                    if arg.startswith(current_arg):
                        yield Completion(arg, start_position=-len(current_arg), display=arg)

    def _get_meta(self, cmd: str) -> str:
        metas = {
            "fetch": "Fetch data (current/historical)",
            "train": "Train a model (SARIMAX/Prophet/LSTM)",
            "load": "Load model from file",
            "save": "Save current model",
            "predict": "Make predictions",
            "compare": "Compare all models",
            "benchmark": "Benchmark all models on held-out tail",
            "backtest": "Run backtest",
            "models": "List available models",
            "explore": "Show data statistics and correlations",
            "show": "Show current data summary",
            "status": "Show current state",
            "help": "Show this help",
            "clear": "Clear screen",
            "strategies": "List available strategies",
            "exit": "Exit REPL",
            "quit": "Exit REPL",
        }
        return metas.get(cmd, "")


def create_session():
    """Create a prompt_toolkit session with completer, history, and key bindings."""
    completer = FuzzyCompleter(REPLCompleter())
    history = InMemoryHistory()

    # Add some default history
    for cmd in ["fetch historical 100", "train SARIMAX", "predict 12", "compare --full", "help"]:
        history.append_string(cmd)

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)

    @bindings.add("c-d")
    def _(event):
        event.app.exit(exception=EOFError)

    style = Style.from_dict({
        "prompt": "#00D4AA bold",
        "completion-menu.completion": "bg:#161B22 #E6EDF3",
        "completion-menu.completion.current": "bg:#00D4AA #0D1117",
        "completion-menu.meta.completion": "bg:#161B22 #8B949E",
        "completion-menu.meta.completion.current": "bg:#00D4AA #0D1117",
        "scrollbar.background": "bg:#30363D",
        "scrollbar.button": "bg:#00D4AA",
        "toolbar": "bg:#161B22 #8B949E",
    })

    def get_prompt_tokens():
        return [("class:prompt", f"{ICONS['prompt']} repl ")]

    def get_bottom_toolbar():
        return HTML(
            "<b>Tab</b>: complete  "
            "<b>↑/↓</b>: history  "
            "<b>Ctrl+C</b>: cancel  "
            "<b>Ctrl+D</b>: exit  "
            "<b>?</b>: help"
        )

    return PromptSession(
        completer=completer,
        complete_style=CompleteStyle.MULTI_COLUMN,
        history=history,
        key_bindings=bindings,
        style=style,
        message=get_prompt_tokens,
        bottom_toolbar=get_bottom_toolbar,
        enable_history_search=True,
        search_ignore_case=True,
    )


def _try_prompt_toolkit() -> bool:
    """Test if prompt_toolkit can initialize on this platform."""
    try:
        create_session()
    except Exception:
        return False
    else:
        return True


def run_repl() -> None:
    """Run the inquirer-style REPL loop with fallback to classic REPL."""
    console.print()
    console.print(
        f"[brand]{ICONS['rocket']} SARIMAX OHLCV Prediction REPL[/]\n"
        f"[ui.text_dim]Tab-complete commands, ↑/↓ history, ? for help[/]",
    )

    # Try to use prompt_toolkit, fall back to classic REPL if not available
    if not _try_prompt_toolkit():
        console.print("[warning]Inquirer-style REPL not available, falling back to classic REPL[/warning]")
        run_rich_repl()
        return

    session = create_session()

    with patch_stdout():
        while True:
            try:
                line = session.prompt()
                cmd, args = parse_command(line)

                if not cmd:
                    continue

                if cmd in ("exit", "quit"):
                    console.print(f"[warning]{ICONS['cross']} Goodbye![/warning]")
                    break

                handler = _CMD_HANDLERS.get(cmd)
                if handler:
                    handler(args)
                else:
                    console.print(
                        f"[error]{ICONS['error']} Unknown command: {cmd}. "
                        f"Type 'help' or press ? for commands.[/error]"
                    )

            except KeyboardInterrupt:
                console.print(f"\n[warning]{ICONS['warning']} Use 'exit' to quit[/warning]")
            except EOFError:
                console.print(f"\n[warning]{ICONS['cross']} Goodbye![/warning]")
                break
            except Exception as e:
                console.print(f"[error]{ICONS['error']} Error: {e}[/error]")


if __name__ == "__main__":
    run_repl()