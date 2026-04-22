from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from data_loader import load_csv, normal_rows, split_features_labels
from model import build_lstm_autoencoder
from predict import reconstruction_scores
from preprocess import create_sequences, fit_scaler, save_scaler, transform_features
from utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    calculate_threshold,
    ensure_parent,
    moving_average,
    save_json,
)


def train(
    input_path: str,
    model_path: str = str(DEFAULT_MODEL_PATH),
    scaler_path: str = str(DEFAULT_SCALER_PATH),
    config_path: str = str(DEFAULT_CONFIG_PATH),
    window_size: int = 60,
    epochs: int = 40,
    batch_size: int = 64,
    latent_dim: int = 32,
    validation_split: float = 0.1,
    label_col: str = "label",
    timestamp_col: str | None = None,
    threshold_method: str = "percentile",
    percentile: float = 99.0,
    k: float = 3.0,
    smoothing_window: int = 5,
) -> dict:
    df = load_csv(input_path)
    train_df = normal_rows(df, label_col=label_col)
    features, _, _ = split_features_labels(train_df, label_col=label_col, timestamp_col=timestamp_col)

    scaler = fit_scaler(features)
    scaled = transform_features(features, scaler)
    sequences = create_sequences(scaled, window_size=window_size)

    model = build_lstm_autoencoder(
        window_size=window_size,
        n_features=sequences.shape[2],
        latent_dim=latent_dim,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        sequences,
        sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    train_scores = reconstruction_scores(model, sequences)
    train_scores = moving_average(train_scores, window=smoothing_window)
    threshold = calculate_threshold(
        train_scores,
        method=threshold_method,
        percentile=percentile,
        k=k,
    )

    ensure_parent(model_path)
    ensure_parent(scaler_path)
    ensure_parent(config_path)
    model.save(model_path)
    save_scaler(scaler, scaler_path)
    save_json(
        {
            "window_size": window_size,
            "feature_columns": list(features.columns),
            "label_col": label_col,
            "timestamp_col": timestamp_col,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "percentile": percentile,
            "k": k,
            "smoothing_window": smoothing_window,
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
        },
        config_path,
    )

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "config_path": config_path,
        "threshold": threshold,
        "history": history.history,
        "feature_columns": list(features.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM autoencoder for anomaly detection.")
    parser.add_argument("--input", default="data/raw/train.csv", help="Training CSV path.")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--threshold-method", choices=["percentile", "std"], default="percentile")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    args = parser.parse_args()

    result = train(
        input_path=args.input,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        k=args.k,
        smoothing_window=args.smoothing_window,
    )
    print(f"Model saved to {result['model_path']}")
    print(f"Scaler saved to {result['scaler_path']}")
    print(f"Config saved to {result['config_path']}")
    print(f"Training threshold: {result['threshold']:.8f}")


if __name__ == "__main__":
    main()
