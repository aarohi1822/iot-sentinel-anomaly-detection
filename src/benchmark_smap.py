from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import tensorflow as tf
from tensorflow import keras

from data_loader import build_point_labels, iter_telemanom_channels, load_numpy_array
from model import build_lstm_autoencoder
from predict import reconstruction_scores
from preprocess import align_sequence_scores, create_sequences
from utils import calculate_threshold, ensure_parent, moving_average, save_json


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    auc = float("nan")
    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, scores))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
    }


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    adjusted = predictions.astype(bool).copy()
    labels = labels.astype(int)
    start = None

    for index, value in enumerate(labels):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            if adjusted[start:index].any():
                adjusted[start:index] = True
            start = None

    if start is not None and adjusted[start:].any():
        adjusted[start:] = True
    return adjusted.astype(int)


def compute_metric_bundle(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = compute_binary_metrics(labels, scores, predictions)
    adjusted_predictions = point_adjust_predictions(labels, predictions)
    adjusted_metrics = compute_binary_metrics(labels, scores, adjusted_predictions)
    metrics.update(
        {
            "adjusted_precision": adjusted_metrics["precision"],
            "adjusted_recall": adjusted_metrics["recall"],
            "adjusted_f1": adjusted_metrics["f1"],
        }
    )
    return metrics


def fit_autoencoder(
    train_values: np.ndarray,
    window_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[keras.Model, np.ndarray]:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    model = build_lstm_autoencoder(
        window_size=window_size,
        n_features=train_values.shape[1],
        latent_dim=latent_dim,
    )

    n_sequences = len(train_values) - window_size + 1
    if n_sequences <= 50000:
        sequences = create_sequences(train_values, window_size=window_size)
        callbacks: list[keras.callbacks.Callback] = []
        if validation_split > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                )
            )

        model.fit(
            sequences,
            sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )
        return model, sequences

    dataset = keras.utils.timeseries_dataset_from_array(
        train_values,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size,
    )
    dataset = dataset.map(lambda batch: (batch, batch))
    model.fit(dataset, epochs=epochs, verbose=0)
    return model, np.empty((0, window_size, train_values.shape[1]), dtype=np.float32)


def sequence_scores_stream(model: keras.Model, values: np.ndarray, window_size: int, batch_size: int) -> np.ndarray:
    dataset = keras.utils.timeseries_dataset_from_array(
        values,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )
    scores: list[np.ndarray] = []
    for batch in dataset:
        reconstructed = model(batch, training=False)
        batch_scores = tf.reduce_mean(tf.square(batch - reconstructed), axis=(1, 2)).numpy()
        scores.append(batch_scores)
    return np.concatenate(scores)


def evaluate_autoencoder(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    seed: int,
) -> dict[str, float | np.ndarray]:
    validation_split = 0.1 if len(train_values) >= 200 else 0.0
    model, train_sequences = fit_autoencoder(
        train_values=train_values,
        window_size=window_size,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    if len(train_sequences):
        train_scores = reconstruction_scores(model, train_sequences)
    else:
        train_scores = sequence_scores_stream(model, train_values, window_size, batch_size)
    train_scores = align_sequence_scores(train_scores, len(train_values), window_size)
    train_scores = moving_average(train_scores, window=smoothing_window)
    threshold = calculate_threshold(
        train_scores,
        method=threshold_method,
        percentile=percentile,
        k=k,
    )

    test_scores = sequence_scores_stream(model, test_values, window_size, batch_size)
    test_scores = align_sequence_scores(test_scores, len(test_values), window_size)
    test_scores = moving_average(test_scores, window=smoothing_window)
    predictions = test_scores > threshold
    metrics = compute_metric_bundle(labels, test_scores, predictions)

    keras.backend.clear_session()
    return {
        **metrics,
        "threshold": float(threshold),
        "predictions": predictions,
        "scores": test_scores,
    }


def evaluate_isolation_forest(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    seed: int,
) -> dict[str, float | np.ndarray]:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_values)

    train_scores = -model.score_samples(train_values)
    train_scores = moving_average(train_scores, window=smoothing_window)
    threshold = calculate_threshold(
        train_scores,
        method=threshold_method,
        percentile=percentile,
        k=k,
    )

    test_scores = -model.score_samples(test_values)
    test_scores = moving_average(test_scores, window=smoothing_window)
    predictions = test_scores > threshold
    metrics = compute_metric_bundle(labels, test_scores, predictions)
    return {
        **metrics,
        "threshold": float(threshold),
        "predictions": predictions,
        "scores": test_scores,
    }


def minmax_scale(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_min = train_values.min(axis=0)
    train_max = train_values.max(axis=0)
    denom = np.where(train_max - train_min == 0, 1.0, train_max - train_min)
    train_scaled = (train_values - train_min) / denom
    test_scaled = (test_values - train_min) / denom
    return train_scaled.astype("float32"), test_scaled.astype("float32")


def summarize_metrics(labels: list[np.ndarray], scores: list[np.ndarray], predictions: list[np.ndarray]) -> dict[str, float]:
    merged_labels = np.concatenate(labels)
    merged_scores = np.concatenate(scores)
    merged_predictions = np.concatenate(predictions)
    metrics = compute_metric_bundle(merged_labels, merged_scores, merged_predictions)
    metrics["anomaly_points"] = int(merged_labels.sum())
    metrics["evaluated_points"] = int(len(merged_labels))
    metrics["predicted_anomalies"] = int(merged_predictions.sum())
    return metrics


def load_aggregate_dataset(dataset_root: str | Path, spacecraft: str) -> list[dict]:
    dataset_root = Path(dataset_root)
    train_path = dataset_root / f"{spacecraft}_train.npy"
    test_path = dataset_root / f"{spacecraft}_test.npy"
    label_path = dataset_root / f"{spacecraft}_test_label.npy"
    if not train_path.exists() or not test_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Aggregate {spacecraft} dataset files were not found in {dataset_root}.")

    return [
        {
            "channel_id": spacecraft,
            "spacecraft": spacecraft,
            "class": "aggregate",
            "train": load_numpy_array(train_path),
            "test": load_numpy_array(test_path),
            "labels": np.load(label_path).astype(int).reshape(-1),
        }
    ]


def run_benchmark(
    dataset_root: str,
    spacecraft: str,
    output_dir: str,
    window_size: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    threshold_method: str,
    percentile: float,
    k: float,
    smoothing_window: int,
    limit_channels: int | None,
    seed: int,
) -> dict:
    try:
        channels = iter_telemanom_channels(dataset_root, spacecraft=spacecraft, limit=limit_channels)
    except FileNotFoundError:
        channels = load_aggregate_dataset(dataset_root, spacecraft=spacecraft)
        if limit_channels is not None:
            channels = channels[:limit_channels]
    if not channels:
        raise ValueError(f"No channels found for spacecraft={spacecraft} in {dataset_root}.")

    rows: list[dict] = []
    ae_labels: list[np.ndarray] = []
    ae_scores: list[np.ndarray] = []
    ae_predictions: list[np.ndarray] = []
    iso_labels: list[np.ndarray] = []
    iso_scores: list[np.ndarray] = []
    iso_predictions: list[np.ndarray] = []

    started = time.time()
    for index, channel in enumerate(channels, start=1):
        channel_id = channel["channel_id"]
        train_values, test_values = minmax_scale(channel["train"], channel["test"])
        if "labels" in channel:
            labels = channel["labels"]
        else:
            labels = build_point_labels(len(test_values), channel["anomaly_sequences"])

        autoencoder = evaluate_autoencoder(
            train_values=train_values,
            test_values=test_values,
            labels=labels,
            window_size=window_size,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            threshold_method=threshold_method,
            percentile=percentile,
            k=k,
            smoothing_window=smoothing_window,
            seed=seed + index,
        )
        isolation_forest = evaluate_isolation_forest(
            train_values=train_values,
            test_values=test_values,
            labels=labels,
            threshold_method=threshold_method,
            percentile=percentile,
            k=k,
            smoothing_window=smoothing_window,
            seed=seed + index,
        )

        rows.append(
            {
                "channel_id": channel_id,
                "class": channel["class"],
                "test_points": len(test_values),
                "anomaly_points": int(labels.sum()),
                "ae_precision": autoencoder["precision"],
                "ae_recall": autoencoder["recall"],
                "ae_f1": autoencoder["f1"],
                "ae_adjusted_precision": autoencoder["adjusted_precision"],
                "ae_adjusted_recall": autoencoder["adjusted_recall"],
                "ae_adjusted_f1": autoencoder["adjusted_f1"],
                "ae_roc_auc": autoencoder["roc_auc"],
                "ae_threshold": autoencoder["threshold"],
                "if_precision": isolation_forest["precision"],
                "if_recall": isolation_forest["recall"],
                "if_f1": isolation_forest["f1"],
                "if_adjusted_precision": isolation_forest["adjusted_precision"],
                "if_adjusted_recall": isolation_forest["adjusted_recall"],
                "if_adjusted_f1": isolation_forest["adjusted_f1"],
                "if_roc_auc": isolation_forest["roc_auc"],
                "if_threshold": isolation_forest["threshold"],
            }
        )

        ae_labels.append(labels)
        ae_scores.append(autoencoder["scores"])
        ae_predictions.append(autoencoder["predictions"])
        iso_labels.append(labels)
        iso_scores.append(isolation_forest["scores"])
        iso_predictions.append(isolation_forest["predictions"])

        print(
            f"[{index:02d}/{len(channels)}] {channel_id} | "
            f"AE F1={autoencoder['f1']:.3f} / adj {autoencoder['adjusted_f1']:.3f} | "
            f"IF F1={isolation_forest['f1']:.3f} / adj {isolation_forest['adjusted_f1']:.3f}"
        )

    ae_summary = summarize_metrics(ae_labels, ae_scores, ae_predictions)
    iso_summary = summarize_metrics(iso_labels, iso_scores, iso_predictions)
    output_dir = ensure_parent(Path(output_dir) / "placeholder.txt").parent
    metrics_path = output_dir / f"{spacecraft.lower()}_channel_metrics.csv"
    summary_path = output_dir / f"{spacecraft.lower()}_benchmark_summary.json"

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = {
        "dataset": spacecraft,
        "channel_count": len(channels),
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "threshold_method": threshold_method,
        "percentile": percentile,
        "k": k,
        "smoothing_window": smoothing_window,
        "runtime_seconds": round(time.time() - started, 2),
        "lstm_autoencoder": ae_summary,
        "isolation_forest": iso_summary,
        "f1_improvement_vs_isolation_forest_pct": round(
            (ae_summary["f1"] - iso_summary["f1"]) * 100,
            2,
        ),
        "adjusted_f1_improvement_vs_isolation_forest_pct": round(
            (ae_summary["adjusted_f1"] - iso_summary["adjusted_f1"]) * 100,
            2,
        ),
    }
    save_json(summary, summary_path)
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the LSTM autoencoder on NASA SMAP/MSL.")
    parser.add_argument("--dataset-root", default="data/nasa_telemanom")
    parser.add_argument("--spacecraft", choices=["SMAP", "MSL"], default="SMAP")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--threshold-method", choices=["percentile", "std"], default="percentile")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--limit-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(
        dataset_root=args.dataset_root,
        spacecraft=args.spacecraft,
        output_dir=args.output_dir,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        k=args.k,
        smoothing_window=args.smoothing_window,
        limit_channels=args.limit_channels,
        seed=args.seed,
    )

    summary = result["summary"]
    print(f"Saved per-channel metrics to {result['metrics_path']}")
    print(f"Saved summary to {result['summary_path']}")
    print(
        "LSTM Autoencoder | "
        f"Precision={summary['lstm_autoencoder']['precision']:.4f} "
        f"Recall={summary['lstm_autoencoder']['recall']:.4f} "
        f"F1={summary['lstm_autoencoder']['f1']:.4f} "
        f"Adj-F1={summary['lstm_autoencoder']['adjusted_f1']:.4f} "
        f"AUC={summary['lstm_autoencoder']['roc_auc']:.4f}"
    )
    print(
        "Isolation Forest | "
        f"Precision={summary['isolation_forest']['precision']:.4f} "
        f"Recall={summary['isolation_forest']['recall']:.4f} "
        f"F1={summary['isolation_forest']['f1']:.4f} "
        f"Adj-F1={summary['isolation_forest']['adjusted_f1']:.4f} "
        f"AUC={summary['isolation_forest']['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
