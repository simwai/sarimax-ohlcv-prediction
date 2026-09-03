"""Streamlit GUI application."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st

from .config import SETTINGS
from .data.fetcher import fetch_with_retry
from .data.processor import create_future_timestamps
from .models import MODEL_REGISTRY, create_model
from .viz.plotly import create_prediction_chart

Mode = Literal["current", "historical"]


def _as_mode(mode: str) -> Mode:
    if mode not in ("current", "historical"):
        return cast(Mode, "current")
    return cast(Mode, mode)

# Initialize session state
if "calculation_lock" not in st.session_state:
    st.session_state.calculation_lock = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionConfig:
    """Configuration for prediction cycle."""

    def __init__(
        self,
        model_name: str,
        mode: str,
        lookback: int,
        iterations: int,
        prediction_periods: int,
        plot_placeholder: Any,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.lookback = lookback
        self.iterations = iterations
        self.prediction_periods = prediction_periods
        self.plot_placeholder = plot_placeholder


def write_to_csv(predicted_prices: pd.DataFrame, filename: str = SETTINGS.csv_filename) -> None:
    """Write predictions to CSV with archiving."""
    path = Path(filename)

    if path.exists() and path.stat().st_size > SETTINGS.csv_max_size_mb * 1024 * 1024:
        archive_name = f"{filename}.archive_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        path.rename(archive_name)
        logger.info("Archived large CSV to %s", archive_name)

    mode = "a" if path.exists() else "w"
    with path.open(mode, newline="") as f:
        if mode == "w":
            f.write("timestamp,open,high,low,close,volume\n")

        timestamps = create_future_timestamps(pd.Timestamp.now(), len(predicted_prices))

        for i in range(len(predicted_prices)):
            row = [
                timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                predicted_prices["open"].iloc[i],
                predicted_prices["high"].iloc[i],
                predicted_prices["low"].iloc[i],
                predicted_prices["close"].iloc[i],
                predicted_prices["volume"].iloc[i],
            ]
            f.write(",".join(str(x) for x in row) + "\n")


def run_prediction_cycle(config: PredictionConfig) -> None:
    """Run one prediction cycle."""
    if st.session_state.calculation_lock:
        logger.info("Calculation already in progress")
        return

    st.session_state.calculation_lock = True

    try:
        # Fetch data
        data = fetch_with_retry(_as_mode(config.mode), config.lookback)
        if data.empty:
            st.error("Failed to fetch data")
            return

        # Train model
        with st.spinner(f"Training {config.model_name}..."):
            model = create_model(config.model_name)
            model.fit(data, iterations=config.iterations)

        # Predict
        with st.spinner("Generating predictions..."):
            if config.model_name == "LSTM":
                predictions = model.predict_with_context(data, config.prediction_periods)
            else:
                predictions = model.predict(config.prediction_periods)

        # Save to CSV
        write_to_csv(predictions)

        # Plot
        fig = create_prediction_chart(
            data, predictions, f"Bitcoin Price Prediction ({config.model_name})"
        )
        config.plot_placeholder.plotly_chart(fig, use_container_width=True)

        # Show predictions table
        st.subheader("Predictions")
        st.dataframe(predictions.round(2))

    except Exception:
        logger.exception("Prediction cycle failed")
        st.error("Error occurred during prediction cycle")
    finally:
        st.session_state.calculation_lock = False


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Bitcoin Price Prediction",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Bitcoin Price Prediction")
    st.markdown("Real-time OHLCV forecasting with SARIMAX, Prophet, and LSTM")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")

        model_choice = st.selectbox(
            "Model",
            list(MODEL_REGISTRY.keys()),
            index=0,
            help="Select forecasting model",
        )

        mode = st.selectbox(
            "Data Mode",
            ["current", "historical"],
            index=0,
            help="Current: last 24h, Historical: specified lookback",
        )

        lookback = st.number_input(
            "Lookback (days)",
            min_value=1,
            max_value=1000,
            value=SETTINGS.default_lookback_days,
            help="Days of historical data to fetch",
        )

        iterations = st.number_input(
            "Iterations",
            min_value=10,
            max_value=1000,
            value=SETTINGS.default_iterations,
            help="Optimization iterations (SARIMAX) / epochs (LSTM)",
        )

        prediction_periods = st.number_input(
            "Prediction Periods (5-min bars)",
            min_value=1,
            max_value=100,
            value=SETTINGS.default_prediction_periods,
            help="Number of future periods to predict",
        )

        auto_refresh = st.checkbox(
            "Auto Refresh",
            value=True,
            help=f"Automatically run every {SETTINGS.sleep_interval_seconds // 60} minutes",
        )

        run_once = st.button("Run Once", type="primary")

    # Main area
    plot_placeholder = st.empty()
    status_placeholder = st.empty()

    if run_once or auto_refresh:
        while True:
            with status_placeholder.container():
                st.info(f"Running prediction cycle with {model_choice}...")

            # Cast streamlit widget returns to expected types
            model_name_str: str = model_choice if isinstance(model_choice, str) else "SARIMAX"
            mode_str: str = mode if isinstance(mode, str) else "current"

            def _to_int(v: Any, default: int) -> int:
                if isinstance(v, int):
                    return v
                if isinstance(v, float):
                    return int(v)
                return default

            lookback_int: int = _to_int(lookback, SETTINGS.default_lookback_days)
            iterations_int: int = _to_int(iterations, SETTINGS.default_iterations)
            prediction_periods_int: int = _to_int(
                prediction_periods, SETTINGS.default_prediction_periods
            )
            config = PredictionConfig(
                model_name=model_name_str,
                mode=mode_str,
                lookback=lookback_int,
                iterations=iterations_int,
                prediction_periods=prediction_periods_int,
                plot_placeholder=plot_placeholder,
            )
            run_prediction_cycle(config)

            if not auto_refresh:
                break

            with status_placeholder.container():
                wait_minutes = SETTINGS.sleep_interval_seconds // 60
                st.info(f"Waiting {wait_minutes} minutes until next cycle...")
            time.sleep(SETTINGS.sleep_interval_seconds)

    else:
        st.info("Configure settings in sidebar and click 'Run Once' or enable 'Auto Refresh'")


if __name__ == "__main__":
    main()