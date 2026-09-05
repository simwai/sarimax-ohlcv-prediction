"""Plotly visualizations for Streamlit."""

import pandas as pd
import plotly.graph_objects as go

_MISSING_HIST_MSG = "Historical data missing columns: {missing}"
_MISSING_PRED_MSG = "Predictions missing columns: {missing}"
_MISSING_OHLCV_MSG = "OHLCV data missing columns: {missing}"


def create_prediction_chart(
    historical_data: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "Bitcoin Price Prediction",
) -> go.Figure:
    """Create combined historical + prediction chart."""
    missing_hist = [
        c
        for c in ("timestamp", "open", "high", "low", "close", "volume")
        if c not in historical_data.columns
    ]
    if missing_hist:
        raise ValueError(_MISSING_HIST_MSG.format(missing=missing_hist))  # noqa: TRY003
    missing_pred = [
        c for c in ("open", "high", "low", "close", "volume") if c not in predictions.columns
    ]
    if missing_pred:
        raise ValueError(_MISSING_PRED_MSG.format(missing=missing_pred))  # noqa: TRY003
    fig = go.Figure()

    # Historical data
    for column in ["open", "high", "low", "close", "volume"]:
        fig.add_trace(
            go.Scatter(
                x=historical_data["timestamp"],
                y=historical_data[column],
                mode="lines",
                name=f"Historical {column.capitalize()}",
                line={"width": 1},
            )
        )

    # Future timestamps
    start_time = historical_data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    future_timestamps = pd.date_range(start=start_time, periods=len(predictions), freq="5min")

    # Predictions
    for column in ["open", "high", "low", "close", "volume"]:
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=predictions[column],
                mode="lines",
                name=f"Predicted {column.capitalize()}",
                line={"width": 1, "dash": "dash"},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig


def create_candlestick_chart(
    data: pd.DataFrame,
    title: str = "OHLCV Candlestick Chart",
) -> go.Figure:
    """Create candlestick chart from OHLCV data."""
    missing = [c for c in ("timestamp", "open", "high", "low", "close") if c not in data.columns]
    if missing:
        raise ValueError(_MISSING_OHLCV_MSG.format(missing=missing))  # noqa: TRY003
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["timestamp"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="OHLC",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_equity_curve_chart(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Portfolio Value",
            line={"color": "green", "width": 2},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
    )

    return fig
