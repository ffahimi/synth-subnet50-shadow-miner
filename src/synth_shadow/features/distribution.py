"""Kurtosis and tail-shape features for session-aware path rescaling."""

from __future__ import annotations

import pandas as pd


def add_distribution_features(df: pd.DataFrame, kurtosis_window: int) -> pd.DataFrame:
    """Add rolling kurtosis as a tail-risk proxy."""
    out = df.copy()
    out["kurtosis_4h"] = out["log_return"].rolling(kurtosis_window, min_periods=8).kurt()
    out["abs_return"] = out["log_return"].abs()
    return out
