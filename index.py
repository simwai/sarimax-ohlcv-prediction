import csv
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from colorlog import ColoredFormatter
from pmdarima import auto_arima
from prophet import Prophet

from sarimax_ohlcv_prediction.data.fetcher import get_default_exchange

# Set up colored logging
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )
)
log.addHandler(handler)
log.setLevel(logging.INFO)

# Initialize Streamlit session state for calculation lock
if "calculation_lock" not in st.session_state:
    st.session_state.calculation_lock = False


def fetch_data(mode: str, lookback: int) -> pd.DataFrame:
    exchange = get_default_exchange()
    try:
        if mode == "current":
            since = exchange.milliseconds() - (24 * 60 * 60 * 1000)  # Last 24 hours
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", since=since)
        elif mode == "historical":
            since = exchange.milliseconds() - (lookback * 24 * 60 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", since=since)
        else:
            raise ValueError("Invalid mode specified")

        data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        return data
    except Exception:
        log.exception("Failed to fetch data")
        return pd.DataFrame()


@st.cache_resource
def optimize_model(
    data: pd.DataFrame, iterations: int, seasonal: bool = True, model_choice: str = "SARIMAX"
) -> dict | None:
    if st.session_state.calculation_lock:
        log.info("A calculation is already in progress. Skipping new calculation.")
        return None

    st.session_state.calculation_lock = True

    try:
        columns = ["open", "high", "low", "close", "volume"]
        model_fits = {}
        for column in columns:
            log.info("Optimizing parameters for %s using %s", column, model_choice)

            if model_choice == "SARIMAX":
                m_values = range(7, 50)
                best_aic = float("inf")
                best_m = None
                best_model = None

                for m_val in m_values:
                    try:
                        temp_model = auto_arima(
                            data[column],
                            start_p=1,
                            start_q=1,
                            max_p=1000,
                            max_q=10,
                            m=m_val,
                            start_P=0,
                            seasonal=seasonal,
                            d=1,
                            D=1,
                            trace=True,
                            error_action="ignore",
                            suppress_warnings=True,
                            stepwise=True,
                            information_criterion="aic",
                            maxiter=iterations,
                        )

                        if temp_model.aic() < best_aic:
                            best_aic = temp_model.aic()
                            best_m = m_val
                            best_model = temp_model
                    except Exception:
                        log.exception("An error occurred for m=%s", m_val)
                        log.info("Best seasonality value: %s with AIC: %s", best_m, best_aic)
                        model_fits[column] = best_model

            elif model_choice == "Prophet":
                df = data[["timestamp", column]].rename(columns={"timestamp": "ds", column: "y"})
                model = (
                    Prophet(daily_seasonality=False, yearly_seasonality=False)
                    .add_seasonality(name="daily", period=24 * 60 / 5, fourier_order=8)
                    .add_seasonality(name="weekly", period=7 * 24 * 60 / 5, fourier_order=3)
                )
                model.fit(df)
                future = model.make_future_dataframe(periods=iterations, freq="5min")
                forecast = model.predict(future)
                model_fits[column] = forecast

        return model_fits
    finally:
        st.session_state.calculation_lock = False


def predict_price(model_fits: dict, period: int, data: pd.DataFrame, model_choice: str) -> dict:
    predicted_prices = {}
    for column, model_fit in model_fits.items():
        if model_choice == "SARIMAX":
            predicted_price = model_fit.predict(start=len(data), end=len(data) + period - 1)
            predicted_prices[column] = predicted_price
        elif model_choice == "Prophet":
            predicted_prices[column] = model_fit["yhat"][-period:]
    return predicted_prices


def write_to_csv(predicted_prices: dict, filename: str, model_choice: str) -> None:
    path = Path(filename)
    file_size = path.stat().st_size if path.exists() else 0

    max_file_size = 10 * 1024 * 1024  # 10 MB

    if file_size > max_file_size:
        archive_name = f"{filename}.archive_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        path.rename(archive_name)

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if file_size == 0:
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        if model_choice == "SARIMAX":
            timestamps = predicted_prices["close"].index
            min_length = min(len(predicted_prices[col]) for col in predicted_prices)
            for i in range(min_length):
                timestamp_str = pd.to_datetime(timestamps[i], unit="ms").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                row = [timestamp_str]
                for _col in ["open", "high", "low", "close", "volume"]:
                    row.append(predicted_prices[_col].iloc[i])
                writer.writerow(row)
        elif model_choice == "Prophet":
            forecast_df = predicted_prices["close"]
            for i in range(len(forecast_df)):
                timestamp_str = forecast_df["ds"].iloc[i].strftime("%Y-%m-%d %H:%M:%S")
                row = [timestamp_str]
                for _col in ["open", "high", "low", "close", "volume"]:
                    row.append(forecast_df["yhat"].iloc[i])
                writer.writerow(row)


def main(
    gui_enabled: bool,
    iterations: int,
    prediction_periods: int,
    mode: str,
    lookback: int,
    model_choice: str,
) -> None:
    from typing import Any

    plot_placeholder: Any = None
    if gui_enabled:
        plot_placeholder = st.empty()

    while True:
        try:
            data = fetch_data(mode, lookback)
            if data.empty:
                log.error("No data fetched. Skipping this iteration.")
                time.sleep(60)
                continue

            model_fits = optimize_model(data, iterations, model_choice=model_choice)
            if model_fits is None:
                continue

            predicted_prices = predict_price(model_fits, prediction_periods, data, model_choice)
            write_to_csv(predicted_prices, "predicted_prices.csv", model_choice)

            if gui_enabled:
                fig = go.Figure()
                for column in ["open", "high", "low", "close", "volume"]:
                    fig.add_trace(
                        go.Scatter(
                            x=data["timestamp"],
                            y=data[column],
                            mode="lines",
                            name=f"Historical {column.capitalize()}",
                        )
                    )

                start_time = data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
                future_timestamps = pd.date_range(
                    start=start_time, periods=prediction_periods, freq="5T"
                )
                for column in ["open", "high", "low", "close", "volume"]:
                    fig.add_trace(
                        go.Scatter(
                            x=future_timestamps,
                            y=predicted_prices[column],
                            mode="lines",
                            name=f"Predicted {column.capitalize()}",
                        )
                    )

                fig.update_layout(
                    title="Bitcoin Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price",
                )
                if plot_placeholder is not None:
                    plot_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(15 * 60)

        except Exception:
            log.exception("An error occurred")
            time.sleep(60)


if __name__ == "__main__":
    gui_enabled = True
    iterations = 100
    prediction_periods = 12
    mode = "current"
    lookback = 100
    model_choice = "SARIMAX"
    if gui_enabled:
        _choice = st.selectbox("Select the forecasting model:", ["SARIMAX", "Prophet"])
        model_choice = _choice if isinstance(_choice, str) else "SARIMAX"
    main(gui_enabled, iterations, prediction_periods, mode, lookback, model_choice)
