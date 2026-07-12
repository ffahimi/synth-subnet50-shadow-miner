"""Return calculations for 5-minute BTC bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-bar log returns."""
    out = df.copy()
    out["log_return"] = np.log(out["close"]).diff().fillna(0.0)
    return out
