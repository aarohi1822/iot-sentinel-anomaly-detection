from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from data_loader import load_csv, split_features_labels
from preprocess import align_sequence_scores, create_sequences, load_scaler, transform_features
from utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    build_anomaly_report,
    calculate_threshold,
    load_json,
    moving_average,
)


def reconstruction_scores(model, sequences: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(sequences, verbose=0)
    return np.mean(np.square(sequences - reconstructed), axis=(1, 2))


def reconstruction_details(model, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reconstructed = model.predict(sequences, verbose=0)
    squared_error = np.square(sequences - reconstructed)
    sequence_scores = np.mean(squared_error, axis=(1, 2))
    feature_scores = np.mean(squared_error, axis=1)
    return sequence_scores, feature_scores


def align_feature_scores(feature_scores: np.ndarray, n_rows: int, window_size: int) -> np.ndarray:
    aligned = np.full((n_rows, feature_scores.shape[1]), np.nan, dtype=float)
    aligned[window_size - 1 :] = feature_scores

    first_valid = window_size - 1
    if first_valid < n_rows:
        aligned[:first_valid] = feature_scores[0]
    return aligned


def detect_anomalies(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    label_col: str = "label",
    timestamp_col: str | None = None,
    threshold: float | None = None,
    threshold_method: str | None = None,
    percentile: float | None = None,
    k: float | None = None,
    smoothing_window: int | None = None,
) -> dict:
    config = load_json(config_path)
    window_size = int(config["window_size"])
    feature_columns = config["feature_columns"]
    timestamp_col = timestamp_col or config.get("timestamp_col")

    features, _, timestamps = split_features_labels(df, label_col=label_col, timestamp_col=timestamp_col)

    missing = [col for col in feature_columns if col not in features.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    features = features[feature_columns]

    scaler = load_scaler(str(scaler_path))
    model = keras.models.load_model(str(model_path), compile=False)

    scaled = transform_features(features, scaler)
    sequences = create_sequences(scaled, window_size=window_size)

    sequence_scores, feature_scores = reconstruction_details(model, sequences)

    row_scores = align_sequence_scores(sequence_scores, len(df), window_size)
    row_feature_scores = align_feature_scores(feature_scores, len(df), window_size)

    smoothing_window = int(smoothing_window or config.get("smoothing_window", 5))
    smoothed_scores = moving_average(row_scores, window=smoothing_window)

    if threshold is None and "threshold" in config:
        threshold = float(config["threshold"])

    if threshold is None:
        threshold_method = threshold_method or config.get("threshold_method", "percentile")
        percentile = float(percentile or config.get("percentile", 99.0))
        k = float(k or config.get("k", 3.0))

        threshold = calculate_threshold(
            smoothed_scores,
            method=threshold_method,
            percentile=percentile,
            k=k,
        )

    anomaly_indices = np.flatnonzero(smoothed_scores > threshold)

    report = build_anomaly_report(anomaly_indices, smoothed_scores, float(threshold), timestamps)

    if len(anomaly_indices):
        root_causes = [
            feature_columns[int(np.argmax(row_feature_scores[index]))]
            for index in anomaly_indices
        ]

        report["root_cause_sensor"] = root_causes
        report["severity"] = np.where(
            report["anomaly_score"] >= float(threshold) * 1.6,
            "Critical",
            "Warning",
        )

    return {
        "scores": smoothed_scores,
        "raw_scores": row_scores,
        "feature_scores": row_feature_scores,
        "threshold": float(threshold),
        "anomaly_indices": anomaly_indices,
        "report": report,
        "timestamps": timestamps,
        "feature_columns": feature_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection on a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", default="data/processed/anomaly_report.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-method", choices=["percentile", "std"], default=None)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--timestamp-col", default=None)
    args = parser.parse_args()

    df = load_csv(args.input)

    result = detect_anomalies(
        df,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        k=args.k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result["report"].to_csv(output_path, index=False)

    print(f"Detected {len(result['anomaly_indices'])} anomalies.")
    print(f"Threshold: {result['threshold']:.8f}")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()