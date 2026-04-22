from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(features: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(features)
    return scaler


def transform_features(features: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(features)


def create_sequences(values: np.ndarray, window_size: int = 60) -> np.ndarray:
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if len(values) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows to create sequences; got {len(values)}."
        )
    return np.array([values[i : i + window_size] for i in range(len(values) - window_size + 1)])


def align_sequence_scores(sequence_scores: np.ndarray, n_rows: int, window_size: int) -> np.ndarray:
    aligned = np.full(n_rows, np.nan, dtype=float)
    aligned[window_size - 1 :] = sequence_scores

    first_valid = window_size - 1
    if first_valid < n_rows:
        aligned[:first_valid] = sequence_scores[0]
    return aligned


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)

