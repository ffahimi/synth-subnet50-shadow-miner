"""Realized volatility, volatility-of-volatility, and volatility slope features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_features(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Add realized vol, vol-of-vol, and short-vs-long vol slope."""
    out = df.copy()
    returns = out["log_return"]
    out["vol_1h"] = returns.rolling(short_window, min_periods=max(3, short_window // 2)).std()
    out["vol_4h"] = returns.rolling(long_window, min_periods=max(6, long_window // 2)).std()
    out["vol_of_vol_1h"] = out["vol_1h"].rolling(short_window, min_periods=max(3, short_window // 2)).std()
    out["vol_of_vol_4h"] = out["vol_4h"].rolling(long_window, min_periods=max(6, long_window // 2)).std()
    out["vol_slope"] = (out["vol_1h"] - out["vol_4h"]) / out["vol_4h"].replace(0, np.nan)
    return out
