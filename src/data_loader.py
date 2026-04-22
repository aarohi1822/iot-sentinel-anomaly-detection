from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path_or_buffer: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df


def split_features_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    labels = df[label_col] if label_col in df.columns else None
    timestamps = df[timestamp_col] if timestamp_col and timestamp_col in df.columns else None

    drop_cols = [col for col in [label_col, timestamp_col] if col and col in df.columns]
    features = df.drop(columns=drop_cols)
    features = features.select_dtypes(include="number")

    if features.empty:
        raise ValueError("No numeric sensor columns were found in the CSV.")
    return features, labels, timestamps


def normal_rows(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in df.columns:
        return df
    return df[df[label_col] == 0].copy()

