"""Most recent BTC regime state derived from 1h and 4h feature windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurrentState:
    timestamp: str
    price: float
    session: str
    vol_1h: float
    vol_4h: float
    vol_of_vol_1h: float
    vol_of_vol_4h: float
    vol_slope: float
    momentum_1h: float
    momentum_4h: float
    kurtosis_4h: float

    def to_dict(self) -> dict:
        return asdict(self)


def extract_current_state(features: pd.DataFrame) -> CurrentState:
    """Extract the most recent feature row as the current BTC state."""
    if features.empty:
        raise ValueError("Cannot extract current state from an empty feature frame.")
    row = features.iloc[-1]
    return CurrentState(
        timestamp=str(row["timestamp"]),
        price=float(row["close"]),
        session=str(row["session"]),
        vol_1h=_finite(row["vol_1h"]),
        vol_4h=_finite(row["vol_4h"]),
        vol_of_vol_1h=_finite(row["vol_of_vol_1h"]),
        vol_of_vol_4h=_finite(row["vol_of_vol_4h"]),
        vol_slope=_finite(row["vol_slope"]),
        momentum_1h=_finite(row["momentum_1h"]),
        momentum_4h=_finite(row["momentum_4h"]),
        kurtosis_4h=_finite(row["kurtosis_4h"]),
    )


def _finite(value: object, default: float = 0.0) -> float:
    number = float(value) if value is not None else default
    return number if np.isfinite(number) else default
