"""Plotly visualizations for Streamlit."""

import pandas as pd
import plotly.graph_objects as go


def create_prediction_chart(
    historical_data: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "Bitcoin Price Prediction",
) -> go.Figure:
    """Create combined historical + prediction chart."""
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
    future_timestamps = pd.date_range(start=start_time, periods=len(predictions), freq="5T")

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
