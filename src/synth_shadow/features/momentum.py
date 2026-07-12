"""Recent 1h and 4h momentum features used to tilt sampled path shapes."""

from __future__ import annotations

import pandas as pd


def add_momentum_features(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Add rolling 1h and 4h log-return momentum."""
    out = df.copy()
    out["momentum_1h"] = out["log_return"].rolling(short_window, min_periods=1).sum()
    out["momentum_4h"] = out["log_return"].rolling(long_window, min_periods=1).sum()
    return out
