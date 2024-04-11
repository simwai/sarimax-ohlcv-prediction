import ccxt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import plotly.graph_objects as go
import time
import csv
import logging
from colorlog import ColoredFormatter
from pmdarima import auto_arima
import os
from datetime import datetime

# Initialize the Binance client
binance = ccxt.binance()

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


def fetch_data(mode, lookback):
    if mode == "current":
        # Fetch the latest data (e.g., the last 24 hours)
        since = binance.milliseconds() - (24 * 60 * 60 * 1000)  # Last 24 hours
        ohlcv = binance.fetch_ohlcv("BTC/USDT", "5m", since=since)
    elif mode == "historical":
        # Fetch data starting from 'lookback' days ago
        since = binance.milliseconds() - (lookback * 24 * 60 * 60 * 1000)
        ohlcv = binance.fetch_ohlcv("BTC/USDT", "5m", since=since)
    else:
        raise ValueError("Invalid mode specified")

    # Convert the data to a pandas DataFrame
    data = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    return data


@st.cache_resource
def optimize_model(data, iterations, seasonal=True, m=12):
    # Check if a calculation is already in progress
    if st.session_state.calculation_lock:
        log.info(
            "A SARIMAX calculation is already in progress. Skipping new calculation."
        )
        return None  # Or handle this case as appropriate for your application

    # Set the lock
    st.session_state.calculation_lock = True

    try:
        columns = ["open", "high", "low", "close", "volume"]
        model_fits = {}
        for column in columns:
            log.info(f"Optimizing SARIMAX parameters for {column}")

            # Use auto_arima to find the best ARIMA model parameters
            sarimax_model = auto_arima(
                data[column],
                start_p=1,
                start_q=1,
                max_p=5,
                max_d=2,
                max_q=5,
                start_P=1,
                start_Q=1,
                max_P=2,
                max_D=1,
                max_Q=2,
                seasonal=seasonal,
                m=m,
                trace=True,
                error_action="ignore",  # don't want to know if an order does not work
                suppress_warnings=True,  # don't want convergence warnings
                stepwise=True,  # set to stepwise
                information_criterion="aic",  # Use AIC for model selection
                maxiter=iterations,
            )

            # Fit the SARIMAX model with the best found parameters
            model = SARIMAX(
                data[column],
                order=sarimax_model.order,
                seasonal_order=sarimax_model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit(disp=False)

            model_fits[column] = model_fit
        return model_fits
    finally:
        # Clear the lock after the calculation is done
        st.session_state.calculation_lock = False


def objective(params, column, data):
    p, d, q, P, D, Q, s = params
    model = SARIMAX(
        data[column],  # Use a specific column
        order=(int(p), int(d), int(q)),
        seasonal_order=(int(P), int(D), int(Q), int(s)),
    )
    model_fit = model.fit(disp=False)
    return -model_fit.aic  # We want to maximize the AIC, hence the negative sign


def predict_price(model_fits, period, data):
    predicted_prices = {}
    for column, model_fit in model_fits.items():
        predicted_price = model_fit.predict(start=len(data), end=len(data) + period - 1)
        predicted_prices[column] = predicted_price
    return predicted_prices


# def write_to_csv(predicted_prices, filename):
#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#         # Ensure we're working with aligned indices across all columns
#         min_length = min(len(predicted_prices[col]) for col in predicted_prices)
#         for i in range(min_length):
#             row = [predicted_prices['close'].index[i]]  # Assuming 'close' index is representative
#             for column in ['open', 'high', 'low', 'close', 'volume']:
#                 row.append(predicted_prices[column].iloc[i])
#             writer.writerow(row)


def write_to_csv(predicted_prices, filename):
    # Check if the file exists and its size
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)  # The output is in bytes
    else:
        file_size = 0  # File doesn't exist, so we'll be creating a new one

    max_file_size = 10 * 1024 * 1024  # 10 MB

    # If the file size exceeds the limit, archive the current file and start a new one
    if file_size > max_file_size:
        os.rename(
            filename, filename + ".archive_" + datetime.now().strftime("%Y%m%d%H%M%S")
        )

    # Assuming predicted_prices is a dictionary of pandas Series with datetime index
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        # Check if we need to write headers (file was just created)
        if file_size == 0:
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        # Ensure we're working with aligned indices across all columns
        min_length = min(len(predicted_prices[col]) for col in predicted_prices)
        for i in range(min_length):
            # Convert the timestamp to Unix time in milliseconds
            timestamp = int(predicted_prices["close"].index[i] * 1000)
            row = [timestamp]
            for column in ["open", "high", "low", "close", "volume"]:
                row.append(predicted_prices[column].iloc[i])
            writer.writerow(row)

def main(gui_enabled, iterations, prediction_periods, mode, lookback):
    # Create a placeholder for the plot
    plot_placeholder = st.empty()

    while True:
        try:
            # Fetch the data
            log.info("Fetching data...")
            data = fetch_data(mode, lookback)

            # Optimize the model
            log.info("Optimizing model...")
            model_fits = optimize_model(data, iterations)

            if model_fits is None:  # Check if optimization was skipped due to the lock
                continue  # Skip this iteration or handle as needed

            # Predict the price for the next prediction_periods x 5 minutes
            log.info("Predicting price...")
            predicted_prices = predict_price(model_fits, prediction_periods, data)

            # Write the predicted prices to a CSV file
            log.info("Writing predicted prices to CSV file...")
            write_to_csv(predicted_prices, "predicted_prices.csv")

            if gui_enabled:
                # Plot the historical and predicted prices
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

                if mode == "historical":
                    # Predict the historical prices using the model
                    historical_predictions = {}
                    for column, model_fit in model_fits.items():
                        historical_predictions[column] = model_fit.predict(
                            start=0, end=len(data) - 1
                        )

                    for column in ["open", "high", "low", "close", "volume"]:
                        fig.add_trace(
                            go.Scatter(
                                x=data["timestamp"],
                                y=historical_predictions[column],
                                mode="lines+markers",
                                name=f"Predicted Historical {column.capitalize()}",
                            )
                        )

                # Generating future timestamps for predictions if in current mode
                if mode == "current":
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
                plot_placeholder.plotly_chart(fig, use_container_width=True)

            # Wait for 15 minutes before the next prediction
            time.sleep(15 * 60)

        except Exception as e:
            log.error(f"An error occurred: {e}")


if __name__ == "__main__":
    gui_enabled = True  # Set this to False to disable the GUI
    iterations = 100  # Number of optimization iterations
    prediction_periods = 12  # Number of 5 minute periods to predict
    mode = "current"
    lookback = 100
    main(gui_enabled, iterations, prediction_periods, mode, lookback)
