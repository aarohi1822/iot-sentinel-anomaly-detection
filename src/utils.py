from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/lstm_autoencoder.weights.h5")
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")
DEFAULT_CONFIG_PATH = Path("models/config.json")


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def calculate_threshold(
    scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 99.0,
    k: float = 3.0,
) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot calculate threshold from an empty score array.")

    if method == "std":
        return float(scores.mean() + k * scores.std())
    if method == "percentile":
        return float(np.percentile(scores, percentile))
    raise ValueError("method must be either 'percentile' or 'std'.")


def build_anomaly_report(
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    timestamps: pd.Series | None = None,
) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "index": indices.astype(int),
            "anomaly_score": scores[indices],
            "threshold": threshold,
        }
    )
    if timestamps is not None:
        report.insert(1, "timestamp", timestamps.iloc[indices].to_numpy())
    return report

