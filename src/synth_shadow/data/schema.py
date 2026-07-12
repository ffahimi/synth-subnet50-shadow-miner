"""Canonical OHLCV schema used by downstream feature and path modules."""

from __future__ import annotations

import logging

import pandas as pd

LOG = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
]


def polygon_results_to_bars(results: list[dict]) -> pd.DataFrame:
    """Convert Polygon aggregate results into canonical UTC 5-minute bars."""
    if not results:
        raise ValueError("Polygon returned no aggregate bars.")

    df = pd.DataFrame(results).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if column not in df:
            df[column] = 0.0
    df = df[CANONICAL_COLUMNS].sort_values("timestamp").drop_duplicates("timestamp")
    LOG.debug("Canonicalized %s bars from Polygon.", len(df))
    return df.reset_index(drop=True)


def repair_missing_bars(df: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    """Repair missing 5-minute BTC bars with conservative forward-filled prices."""
    if df.empty:
        raise ValueError("Cannot repair an empty bar dataframe.")

    indexed = df.set_index("timestamp").sort_index()
    freq = f"{interval_seconds}s"
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq=freq, tz="UTC")
    repaired = indexed.reindex(full_index)
    missing_count = int(repaired["close"].isna().sum())

    repaired["close"] = repaired["close"].ffill()
    for column in ["open", "high", "low", "vwap"]:
        repaired[column] = repaired[column].fillna(repaired["close"])
    repaired["volume"] = repaired["volume"].fillna(0.0)
    repaired["transactions"] = repaired["transactions"].fillna(0.0)

    repaired = repaired.reset_index(names="timestamp")
    repaired = repaired[CANONICAL_COLUMNS]
    LOG.debug("Missing bar repair inserted/fixed %s intervals.", missing_count)
    return repaired
