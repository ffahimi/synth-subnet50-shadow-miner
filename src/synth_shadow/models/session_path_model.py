"""Historical normalized session path library.

The first BTC model learns path fragments from prior sessions, normalizes those
fragments by their original volatility regime, and makes them available for
resampling during the next 24h forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionBlock:
    session: str
    start_time: str
    normalized_returns: np.ndarray
    source_vol: float
    source_momentum: float
    source_kurtosis: float


def build_session_library(features: pd.DataFrame, block_bars: int) -> list[SessionBlock]:
    """Build normalized historical path blocks grouped by session."""
    blocks: list[SessionBlock] = []
    if len(features) < block_bars + 1:
        raise ValueError(f"Need at least {block_bars + 1} feature rows to build session blocks.")

    returns = features["log_return"].to_numpy()
    sessions = features["session"].to_numpy()
    timestamps = features["timestamp"].to_numpy()

    for start in range(1, len(features) - block_bars + 1):
        end = start + block_bars
        block_sessions = sessions[start:end]
        session = _majority_session(block_sessions)
        block_returns = returns[start:end].astype(float)
        source_vol = float(np.std(block_returns))
        if not np.isfinite(source_vol) or source_vol <= 1e-10:
            continue
        normalized = block_returns / source_vol
        blocks.append(
            SessionBlock(
                session=session,
                start_time=str(timestamps[start]),
                normalized_returns=normalized,
                source_vol=source_vol,
                source_momentum=float(np.sum(block_returns)),
                source_kurtosis=float(pd.Series(block_returns).kurt()),
            )
        )
    return blocks


def _majority_session(values: np.ndarray) -> str:
    unique, counts = np.unique(values, return_counts=True)
    return str(unique[int(np.argmax(counts))])
