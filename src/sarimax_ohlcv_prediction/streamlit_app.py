"""Streamlit GUI application."""

import contextlib
import csv
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import SETTINGS
from .data.fetcher import fetch_with_retry
from .data.modes import as_mode as _as_mode
from .data.processor import create_future_timestamps
from .models import MODEL_REGISTRY, create_model
from .viz.plotly import create_prediction_chart

# Initialize session state
if "calculation_lock" not in st.session_state:
    st.session_state.calculation_lock = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CSV_COLS_MSG = "Predictions missing columns: {missing}"


def _to_int(value: object, default: int) -> int:
    """Coerce a Streamlit widget value to int with a fallback."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


class PredictionConfig:
    """Configuration for prediction cycle."""

    def __init__(
        self,
        model_name: str,
        mode: str,
        lookback: int,
        iterations: int,
        prediction_periods: int,
        plot_placeholder: Any,  # pyrefly: ignore -- Streamlit placeholder duck type
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.lookback = lookback
        self.iterations = iterations
        self.prediction_periods = prediction_periods
        self.plot_placeholder = plot_placeholder


def write_to_csv(
    predicted_prices: pd.DataFrame,
    filename: str = SETTINGS.csv_filename,
    last_timestamp: pd.Timestamp | None = None,
) -> None:
    """Write predictions to CSV atomically, preserving prior rows."""
    missing = [
        c for c in ("open", "high", "low", "close", "volume") if c not in predicted_prices.columns
    ]
    if missing:
        raise ValueError(_CSV_COLS_MSG.format(missing=missing))  # noqa: TRY003
    anchor = last_timestamp if last_timestamp is not None else pd.Timestamp.now()
    timestamps = create_future_timestamps(anchor, len(predicted_prices))
    target = Path(filename)

    if target.exists() and target.stat().st_size > SETTINGS.csv_max_size_mb * 1024 * 1024:
        archive = target.with_name(
            f"{target.name}.archive_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        )
        target.rename(archive)
        logger.info("Archived large CSV to %s", archive)

    previous: str | None = target.read_text(encoding="utf-8") if target.exists() else None
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f"{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if previous:
                f.write(previous)
                if not previous.endswith("\n"):
                    f.write("\n")
            else:
                writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for stamp, (_, row) in zip(timestamps, predicted_prices.iterrows(), strict=True):
                writer.writerow(
                    [
                        stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                    ]
                )
        Path(tmp_name).replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise


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
        write_to_csv(predictions, last_timestamp=data["timestamp"].iloc[-1])

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


def _sidebar_settings() -> tuple[str, str, int, int, int, bool, bool]:
    """Render sidebar controls and return coerced settings."""
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

    model_name: str = model_choice if isinstance(model_choice, str) else "SARIMAX"
    mode_name: str = mode if isinstance(mode, str) else "current"
    return (
        model_name,
        mode_name,
        _to_int(lookback, SETTINGS.default_lookback_days),
        _to_int(iterations, SETTINGS.default_iterations),
        _to_int(prediction_periods, SETTINGS.default_prediction_periods),
        bool(auto_refresh),
        bool(run_once),
    )


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Bitcoin Price Prediction",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Bitcoin Price Prediction")
    st.markdown("Real-time OHLCV forecasting with SARIMAX, Prophet, and LSTM")

    (
        model_name,
        mode_name,
        lookback_days,
        iteration_count,
        period_count,
        auto_refresh,
        run_once,
    ) = _sidebar_settings()

    # Main area
    plot_placeholder = st.empty()
    status_placeholder = st.empty()

    if run_once or auto_refresh:
        while True:
            with status_placeholder.container():
                st.info(f"Running prediction cycle with {model_name}...")

            config = PredictionConfig(
                model_name=model_name,
                mode=mode_name,
                lookback=lookback_days,
                iterations=iteration_count,
                prediction_periods=period_count,
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
