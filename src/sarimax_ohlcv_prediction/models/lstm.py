"""LSTM model implementation using Keras/TensorFlow."""

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import SETTINGS
from .base import ModelBase, validate_model_envelope

logger = logging.getLogger(__name__)

# Suppress TF logging and progress bars
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf  # type: ignore[import-not-found]  # pyrefly: ignore[missing-import]
    from tensorflow.keras import (  # type: ignore[import-not-found]  # pyrefly: ignore[missing-import]
        layers,
        models,
        optimizers,
    )
    from tensorflow.keras.callbacks import (  # type: ignore[import-not-found]  # pyrefly: ignore[missing-import]
        EarlyStopping,
        ReduceLROnPlateau,
    )

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    optimizers = None  # type: ignore[assignment]
    EarlyStopping = None  # type: ignore[assignment]
    ReduceLROnPlateau = None  # type: ignore[assignment]
    logger.warning("TensorFlow not available. LSTM model will not work.")


_TF_NOT_INSTALLED_TRAIN = "TensorFlow not installed. Cannot train LSTM."
_TF_NOT_INSTALLED_LOAD = "TensorFlow not installed. Cannot load LSTM."
_MODEL_NOT_FITTED = "Model not fitted. Call fit() first."
_BAD_ENVELOPE_MSG = "Unrecognized LSTM model file: {path}"
_MISSING_KERAS_MSG = "LSTM weights missing at {path}"
_NEED_DATA_POINTS = "Need at least {seq_len} data points for prediction"
_PREDICT_NOT_IMPLEMENTED = (
    "LSTM predict requires recent data for iterative forecasting. "
    "Use predict_with_context(data, periods) instead."
)


def _get_device() -> str:
    """Detect available device (GPU/CPU)."""
    if not TF_AVAILABLE or tf is None:
        return "cpu"
    gpus = tf.config.list_physical_devices("GPU")  # type: ignore[union-attr]
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # type: ignore[union-attr]
            logger.info("GPU available: %d device(s)", len(gpus))
            return "gpu"
        except RuntimeError as e:
            logger.warning("GPU setup failed: %s", e)
    return "cpu"


class LSTMModel(ModelBase):
    """LSTM model for multi-output OHLCV prediction."""

    def __init__(
        self,
        sequence_length: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        validation_split: float | None = None,
        lstm_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.sequence_length = (
            SETTINGS.lstm_sequence_length if sequence_length is None else sequence_length
        )
        self.epochs = SETTINGS.lstm_epochs if epochs is None else epochs
        self.batch_size = SETTINGS.lstm_batch_size if batch_size is None else batch_size
        self.validation_split = (
            SETTINGS.lstm_validation_split if validation_split is None else validation_split
        )
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = _get_device()
        self.keras_model: Any = None
        self.scalers: dict[str, MinMaxScaler] = {}
        self.history: Any = None

    def _build_model(self, n_features: int) -> Any:
        """Build the Keras LSTM model."""
        if not TF_AVAILABLE or layers is None or models is None or optimizers is None:
            raise RuntimeError(_TF_NOT_INSTALLED_TRAIN)
        inputs = layers.Input(shape=(self.sequence_length, n_features))  # type: ignore[union-attr]
        x = inputs

        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences)(x)  # type: ignore[union-attr]
            x = layers.Dropout(self.dropout)(x)  # type: ignore[union-attr]

        outputs = layers.Dense(n_features)(x)  # type: ignore[union-attr] # Multi-output: OHLCV

        model = models.Model(inputs=inputs, outputs=outputs)  # type: ignore[union-attr]
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),  # type: ignore[union-attr]
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _prepare_sequences(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        if len(data) <= self.sequence_length:
            raise ValueError(_NEED_DATA_POINTS.format(seq_len=self.sequence_length))
        n_features = len(self.columns)
        X = np.zeros((len(data) - self.sequence_length, self.sequence_length, n_features))
        y = np.zeros((len(data) - self.sequence_length, n_features))

        for i, col in enumerate(self.columns):
            scaled = self.scalers[col].transform(data[[col]]).flatten()
            for j in range(len(data) - self.sequence_length):
                X[j, :, i] = scaled[j : j + self.sequence_length]
                y[j, i] = scaled[j + self.sequence_length]

        return X, y

    def fit(self, data: pd.DataFrame, **kwargs) -> "LSTMModel":
        """Train the LSTM model."""
        if not TF_AVAILABLE:
            raise RuntimeError(_TF_NOT_INSTALLED_TRAIN)

        self._store_training_data(data)

        epochs = kwargs.get("epochs", self.epochs)
        batch_size = kwargs.get("batch_size", self.batch_size)
        validation_split = kwargs.get("validation_split", self.validation_split)
        progress_cb = kwargs.get("progress_callback")
        # update instance attrs for info display
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Fit scalers on each column
        for col in self.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.scalers[col] = scaler.fit(data[[col]])

        X, y = self._prepare_sequences(data)

        logger.info(
            "Training LSTM on device=%s epochs=%s - X=%s y=%s",
            self.device,
            epochs,
            tuple(X.shape),
            tuple(y.shape),
        )

        self.keras_model = self._build_model(n_features=len(self.columns))

        # default to silent; allow override via verbose kwarg
        verbose = kwargs.get("verbose", 0)

        if EarlyStopping is None or ReduceLROnPlateau is None:
            raise RuntimeError(_TF_NOT_INSTALLED_TRAIN)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=verbose,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose,
        )

        # progress callback for epochs
        callbacks: list[Any] = [early_stop, reduce_lr]
        if progress_cb is not None:
            assert tf is not None

            class _ProgressCB(tf.keras.callbacks.Callback):  # type: ignore[union-attr]
                def on_epoch_end(self, epoch, logs=None):
                    with suppress(Exception):
                        progress_cb("epoch", epoch + 1, epochs)

            callbacks.append(_ProgressCB())

        self.history = self.keras_model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

        if progress_cb is not None:
            with suppress(Exception):
                progress_cb("done", epochs, epochs)

        self.is_fitted = True
        logger.info("LSTM training complete")
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict next `periods` steps using iterative forecasting."""
        if not self.is_fitted or self.keras_model is None:
            raise RuntimeError(_MODEL_NOT_FITTED)

        raise NotImplementedError(_PREDICT_NOT_IMPLEMENTED)

    def predict_with_context(self, recent_data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Predict with recent data context for iterative forecasting."""
        if not self.is_fitted or self.keras_model is None:
            raise RuntimeError(_MODEL_NOT_FITTED)

        if len(recent_data) < self.sequence_length:
            raise ValueError(_NEED_DATA_POINTS.format(seq_len=self.sequence_length))

        # Scale recent data
        scaled_data = {}
        for col in self.columns:
            scaled_data[col] = self.scalers[col].transform(recent_data[[col]]).flatten()

        scaled_df = pd.DataFrame(scaled_data, index=recent_data.index)
        current_seq: np.ndarray = cast(
            np.ndarray, scaled_df[self.columns].iloc[-self.sequence_length :].to_numpy()
        )
        current_seq = current_seq.reshape(1, self.sequence_length, len(self.columns))

        predictions = []
        for _ in range(periods):
            pred = self.keras_model.predict(current_seq, verbose=0)
            predictions.append(pred[0])

            # Update sequence: remove first, append prediction
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = pred[0]

        # Inverse transform
        pred_array: np.ndarray = cast(np.ndarray, np.array(predictions))
        result = {}
        for i, col in enumerate(self.columns):
            col_values: Any = np.asarray(pred_array[:, i], dtype=float).reshape(-1, 1)
            result[col] = self.scalers[col].inverse_transform(col_values).flatten()

        return pd.DataFrame(result)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.keras_model is None:
            raise RuntimeError(_MODEL_NOT_FITTED)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model separately
        model_dir = Path(path).with_suffix("")
        model_dir.mkdir(parents=True, exist_ok=True)
        keras_path = model_dir / "lstm_model.keras"
        self.keras_model.save(keras_path)

        # Save metadata and scalers
        joblib.dump(
            {
                "format": "lstm-v1",
                "sequence_length": self.sequence_length,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "lstm_units": self.lstm_units,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "columns": self.columns,
                "scalers": self.scalers,
            },
            path,
        )
        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load model from disk, rejecting foreign envelopes."""
        if not TF_AVAILABLE:
            raise RuntimeError(_TF_NOT_INSTALLED_LOAD)

        data = validate_model_envelope(
            joblib.load(path),
            model_name="LSTM",
            required_keys=("sequence_length", "columns"),
            supported_formats=("lstm-v0", "lstm-v1"),
            path=path,
        )
        model = cls(
            sequence_length=data.get("sequence_length", SETTINGS.lstm_sequence_length),
            epochs=data.get("epochs", SETTINGS.lstm_epochs),
            batch_size=data.get("batch_size", SETTINGS.lstm_batch_size),
            validation_split=data.get("validation_split", SETTINGS.lstm_validation_split),
            lstm_units=data.get("lstm_units", (64, 32)),
            dropout=data.get("dropout", 0.2),
            learning_rate=data.get("learning_rate", 0.001),
        )
        model.columns = data.get("columns", model.columns)
        model.scalers = data.get("scalers", {})
        if tf is None:
            raise RuntimeError(_TF_NOT_INSTALLED_LOAD)
        keras_path = Path(path).with_suffix("") / "lstm_model.keras"
        if not keras_path.exists():
            raise ValueError(_MISSING_KERAS_MSG.format(path=keras_path))  # noqa: TRY003
        model.keras_model = tf.keras.models.load_model(keras_path)  # type: ignore[union-attr]
        model.is_fitted = True
        logger.info("LSTM model loaded from %s", path)
        return model
