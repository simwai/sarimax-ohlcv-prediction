"""PDF report generation for backtest results."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..backtest.engine import BacktestResult

# ---------------------------------------------------------------------------
# Theme - adapted from reference reporting palette
# ---------------------------------------------------------------------------
V900 = colors.HexColor("#2E1065")  # header band
V700 = colors.HexColor("#6D28D9")  # table header
V600 = colors.HexColor("#7C3AED")  # accent line
V500 = colors.HexColor("#8B5CF6")
V200 = colors.HexColor("#DDD6FE")  # borders
V100 = colors.HexColor("#EDE9FE")
V50 = colors.HexColor("#F5F3FF")
INK = colors.HexColor("#1E1B4B")
SLATE = colors.HexColor("#64748B")
GRID = colors.HexColor("#C4B5FD")

_GRADE_STYLES = {
    "A+": (colors.HexColor("#5B21B6"), colors.white),
    "A": (colors.HexColor("#7C3AED"), colors.white),
    "B+": (colors.HexColor("#8B5CF6"), colors.white),
    "B": (colors.HexColor("#A78BFA"), colors.white),
    "C+": (colors.HexColor("#DDD6FE"), INK),
    "C": (colors.HexColor("#EDE9FE"), INK),
    "D": (colors.HexColor("#E5E7EB"), SLATE),
}

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONTS: dict[str, str] = {}


def _register_fonts() -> dict[str, str]:
    """Register Inter fonts from bundled assets; fall back to Helvetica."""
    if _FONTS:
        return _FONTS
    try:
        pdfmetrics.registerFont(TTFont("Inter", str(_FONT_DIR / "Inter-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Inter-Semi", str(_FONT_DIR / "Inter-SemiBold.ttf")))
        pdfmetrics.registerFont(TTFont("Inter-Bold", str(_FONT_DIR / "Inter-Bold.ttf")))
        pdfmetrics.registerFontFamily(
            "Inter", normal="Inter", bold="Inter-Bold", italic="Inter", boldItalic="Inter-Bold"
        )
        _FONTS.update({"reg": "Inter", "semi": "Inter-Semi", "bold": "Inter-Bold"})
    except Exception:
        _FONTS.update({"reg": "Helvetica", "semi": "Helvetica-Bold", "bold": "Helvetica-Bold"})
    return _FONTS


def _unique_report_path(filename: str) -> str:
    """Return a timestamped, non-colliding report path under reports/."""
    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    candidate = reports_dir / f"{stem}_{stamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = reports_dir / f"{stem}_{stamp}_{counter}{suffix}"
        counter += 1
    return str(candidate)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fmt(v: Any, fmt: str = "{:.2f}") -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isinf(f) or math.isnan(f):
        return "–"
    return fmt.format(f)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isinf(f) or math.isnan(f):
        return "–"
    return f"{f:.2%}"


def _result_to_metrics(result: BacktestResult) -> dict[str, Any]:
    """Convert BacktestResult to the metrics dict shape expected by the report."""
    return {
        "total_return": result.total_return,
        "calmar": result.calmar_ratio,
        "sortino": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "num_trades": result.total_trades,
        "avg_trade_return": result.avg_trade_return,
        "profit_factor": result.profit_factor,
    }


def _build_kpi_table(metrics: dict[str, Any], usable: float, fonts: dict[str, str]) -> Table:
    """Build KPI cards table."""
    F, FB = fonts["reg"], fonts["bold"]

    def ps(name: str, **kw: Any) -> ParagraphStyle:
        base: dict[str, Any] = {"fontName": F, "textColor": INK}
        base.update(kw)
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **base)

    st_kpi_num = ps(
        "kpinum", fontName=FB, fontSize=15, leading=17, alignment=TA_CENTER, textColor=V700
    )
    st_kpi_cap = ps("kpicap", fontSize=6.3, leading=8, alignment=TA_CENTER, textColor=SLATE)

    kpis = [
        (f"{metrics['total_return']:+.2%}", "Total Return"),
        (f"{metrics['calmar']:.2f}", "Calmar Ratio"),
        (f"{metrics['sortino']:.2f}", "Sortino Ratio"),
        (f"{metrics['max_drawdown']:.2%}", "Max Drawdown"),
        (f"{metrics['win_rate']:.2%}", "Win Rate"),
        (f"{int(metrics['num_trades']):,}", "Total Trades"),
    ]
    kpi_row = [
        Table([[Paragraph(num, st_kpi_num)], [Paragraph(cap, st_kpi_cap)]]) for num, cap in kpis
    ]
    kpis_tbl = Table([kpi_row], colWidths=[usable / len(kpis)] * len(kpis))
    kpis_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), V50),
                ("BOX", (0, 0), (-1, -1), 0.75, V200),
                ("INNERGRID", (0, 0), (-1, -1), 0.75, V200),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    for cell_tbl in kpi_row:
        cell_tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
    return kpis_tbl


def _build_results_table(result: BacktestResult, usable: float, fonts: dict[str, str]) -> Table:
    """Build backtest results table."""
    F = fonts["reg"]

    def ps(name: str, **kw: Any) -> ParagraphStyle:
        base: dict[str, Any] = {"fontName": F, "textColor": INK}
        base.update(kw)
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **base)

    st_cell = ps("cell", fontSize=6.4, leading=8, alignment=TA_CENTER)
    st_cell_r = ps("cellr", fontSize=6.4, leading=8, alignment=TA_RIGHT)
    st_th = ps(
        "th",
        fontName=fonts["bold"],
        fontSize=6.6,
        leading=8,
        alignment=TA_CENTER,
        textColor=colors.white,
    )

    col_widths = [30, 70, 70, 70, 70, 70, 70, 70, 80]
    scale = usable / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    th = lambda t: Paragraph(t.upper(), st_th)  # noqa: E731
    header = [
        th("Metric"),
        th("Total Return"),
        th("Sharpe"),
        th("Max DD"),
        th("Win Rate"),
        th("Trades"),
        th("Avg Trade"),
        th("Profit Factor"),
        th("Streak"),
    ]
    rows: list[list[Any]] = [header]
    streak = result.stats.get("Max Losing Streak") if isinstance(result.stats, pd.Series) else None
    if streak is None:
        streak = 0
    else:
        try:
            streak = int(streak)
        except (TypeError, ValueError):
            streak = 0
    values = [
        ("Total Return", f"{result.total_return:+.2%}"),
        ("Calmar Ratio", f"{result.calmar_ratio:.2f}"),
        ("Sortino Ratio", f"{result.sortino_ratio:.2f}"),
        ("Max Drawdown", f"{result.max_drawdown:.2%}"),
        ("Win Rate", f"{result.win_rate:.2%}"),
        ("Total Trades", f"{result.total_trades:,}"),
        ("Avg Trade Return", f"{result.avg_trade_return:+.2%}"),
        ("Profit Factor", f"{result.profit_factor:.2f}"),
        ("Max Losing Streak", f"{streak}"),
    ]
    for metric, value in values:
        rows.append(
            [Paragraph(metric, st_cell), Paragraph(value, st_cell_r)] + [Paragraph("", st_cell)] * 7
        )

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), V700),
            ("FONTNAME", (0, 1), (-1, -1), F),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, V600),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("TOPPADDING", (0, 1), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
        ]
    )
    for gi in range(1, len(rows)):
        if gi % 2 == 0:
            style.add("BACKGROUND", (0, gi), (-1, gi), V50)
    tbl.setStyle(style)
    return tbl


def _build_legend(usable: float, fonts: dict[str, str]) -> Table:
    """Build methodology legend."""
    F, FB = fonts["reg"], fonts["bold"]

    def ps(name: str, **kw: Any) -> ParagraphStyle:
        base: dict[str, Any] = {"fontName": F, "textColor": INK}
        base.update(kw)
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **base)

    st_sec = ps("sec", fontName=FB, fontSize=10, leading=13, textColor=V900)
    st_leg = ps("leg", fontSize=6.8, leading=9.6)

    legend_title = Table([[Paragraph("Methodology", st_sec)]], colWidths=[usable])
    legend_title.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), V100),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBEFORE", (0, 0), (0, -1), 3, V600),
            ]
        )
    )
    legend_items = [
        Paragraph(
            "<b>Backtest</b> - vectorized portfolio simulation"
            " on OHLCV predictions with configurable fees,"
            " slippage, and frequency.",
            st_leg,
        ),
        Paragraph(
            "<b>Win Rate</b> - share of closed trades with positive"
            " return. <b>Profit Factor</b> - gross profit / gross loss.",
            st_leg,
        ),
        Paragraph(
            "<b>Max Drawdown</b> - peak-to-trough equity decline"
            " across the trade sequence.",
            st_leg,
        ),
    ]
    legend_body = Table([[legend_items]], colWidths=[usable])
    legend_body.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), V50),
                ("BOX", (0, 0), (-1, -1), 0.75, V200),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return Table([[legend_title], [legend_body]], colWidths=[usable])


def generate_pdf_report(
    result: BacktestResult,
    filename: str = "backtest_report.pdf",
    meta: dict[str, Any] | None = None,
) -> str:
    """Generate a landscape PDF report for a single backtest result.

    Returns the output path.
    """
    filename = _unique_report_path(filename)
    fonts = _register_fonts()
    F, FS, FB = fonts["reg"], fonts["semi"], fonts["bold"]
    meta = meta or {}
    metrics = _result_to_metrics(result)

    pagesize = landscape(letter)
    doc = SimpleDocTemplate(
        filename,
        pagesize=pagesize,
        leftMargin=18,
        rightMargin=18,
        topMargin=18,
        bottomMargin=26,
        title="Backtest Report",
        author="sarimax-ohlcv-prediction",
    )
    usable = pagesize[0] - doc.leftMargin - doc.rightMargin

    def ps(name: str, **kw: Any) -> ParagraphStyle:
        base: dict[str, Any] = {"fontName": F, "textColor": INK}
        base.update(kw)
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **base)

    st_h1 = ps("h1", fontName=FB, fontSize=17, leading=20, textColor=colors.white)
    st_sub = ps("sub", fontSize=7.5, leading=10, textColor=colors.HexColor("#C4B5FD"))
    st_meta = ps(
        "meta",
        fontName=FS,
        fontSize=7,
        leading=10,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#C4B5FD"),
    )

    ts_now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    meta_lines = "<br/>".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in meta.items())
    header_inner = Table(
        [
            [
                Paragraph("SARIMAX OHLCV Prediction - Backtest Report", st_h1),
                Paragraph(meta_lines if meta_lines else f"Generated {ts_now}", st_meta),
            ],
            [
                Paragraph(
                    f"Backtest results · {result.total_trades} trades"
                    f" · generated {ts_now}",
                    st_sub,
                ),
                "",
            ],
        ],
        colWidths=[usable * 0.62, usable * 0.38],
    )
    header_inner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), V900),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
            ]
        )
    )
    accent = Table([[""]], colWidths=[usable], rowHeights=[3])
    accent.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), V600)]))

    story: list[Any] = [header_inner, accent, Spacer(1, 10)]
    story.append(_build_kpi_table(metrics, usable, fonts))
    story.append(Spacer(1, 10))
    story.append(_build_results_table(result, usable, fonts))
    story.append(Spacer(1, 12))
    story.append(_build_legend(usable, fonts))

    doc.build(story)
    print(f"PDF report saved as {filename}")
    return filename
