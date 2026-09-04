"""Centralized visual theme for REPL and CLI."""

from rich.theme import Theme


PALETTE = {
    "brand": "#00D4AA",
    "brand_dim": "#00A888",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
    "data_open": "#22D3EE",
    "data_high": "#10B981",
    "data_low": "#F59E0B",
    "data_close": "#EF4444",
    "data_volume": "#A855F7",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted": "#8B949E",
    "bg": "#0D1117",
    "surface": "#161B22",
    "border": "#30363D",
    "border_focus": "#00D4AA",
}


REPL_THEME = Theme({
    "brand": f"bold {PALETTE['brand']}",
    "brand.dim": PALETTE["brand_dim"],
    "success": f"bold {PALETTE['success']}",
    "warning": f"bold {PALETTE['warning']}",
    "error": f"bold {PALETTE['error']}",
    "info": f"bold {PALETTE['info']}",
    "data.open": f"bold {PALETTE['data_open']}",
    "data.high": f"bold {PALETTE['data_high']}",
    "data.low": f"bold {PALETTE['data_low']}",
    "data.close": f"bold {PALETTE['data_close']}",
    "data.volume": f"bold {PALETTE['data_volume']}",
    "ui.border": PALETTE["border"],
    "ui.border_focus": PALETTE["border_focus"],
    "ui.text": PALETTE["text_primary"],
    "ui.text_dim": PALETTE["text_secondary"],
    "ui.text_muted": PALETTE["text_muted"],
    "table.header": f"bold {PALETTE['text_primary']} on {PALETTE['surface']}",
    "table.row_even": f"on {PALETTE['bg']}",
    "table.row_odd": f"on {PALETTE['surface']}",
    "panel.title": f"bold {PALETTE['brand']}",
    "prompt": f"bold {PALETTE['brand']}",
    "status.running": PALETTE["brand"],
    "status.done": PALETTE["success"],
    "dim": PALETTE["text_muted"],
})


ICONS = {
    "prompt": "▸",
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "spinner_frames": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    "chart_up": "▲",
    "chart_down": "▼",
    "folder": "📁",
    "model": "🤖",
    "data": "📊",
    "clock": "⏱",
    "rocket": "🚀",
    "gear": "⚙",
    "chart": "📈",
    "magnifier": "🔍",
    "save": "💾",
    "load": "📂",
    "trash": "🗑",
    "check": "✅",
    "cross": "❌",
    "arrow_right": "→",
    "bullet": "•",
}


SPACING = {
    "xs": 2,
    "sm": 4,
    "md": 8,
    "lg": 12,
    "xl": 16,
}


def get_console():
    """Get a Console instance with the REPL theme applied."""
    from rich.console import Console
    return Console(theme=REPL_THEME)