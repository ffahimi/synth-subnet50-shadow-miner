"""Feature and path-library creation from BTC bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBundle:
    bars_5m: pd.DataFrame
    features: pd.DataFrame
    data_cutoff: pd.Timestamp
    current_price: float


def build_feature_bundle(raw_1m_bars: pd.DataFrame, interval_seconds: int = 300) -> FeatureBundle:
    if raw_1m_bars.empty:
        raise ValueError("Cannot build features from empty raw bars.")
    if interval_seconds != 300:
        raise ValueError("private_btc_similarity_v0 supports 300-second intervals.")

    bars = _ensure_utc(raw_1m_bars).sort_values("timestamp").drop_duplicates("timestamp")
    bars_5m = _to_5m_bars(bars)
    if len(bars_5m) < 320:
        raise ValueError(f"Need at least 320 five-minute bars; got {len(bars_5m)}.")

    features = bars_5m[["timestamp", "close"]].copy()
    features["log_return"] = np.log(features["close"]).diff()
    features["return_1h"] = features["log_return"].rolling(12, min_periods=3).sum()
    features["return_4h"] = features["log_return"].rolling(48, min_periods=12).sum()
    features["volatility_1h"] = features["log_return"].rolling(12, min_periods=6).std()
    features["volatility_4h"] = features["log_return"].rolling(48, min_periods=12).std()
    features["momentum"] = features["return_1h"]
    features["vol_slope"] = (
        (features["volatility_1h"] - features["volatility_4h"])
        / features["volatility_4h"].replace(0.0, np.nan)
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError("Feature creation produced no complete rows.")

    latest_bar = bars_5m.iloc[-1]
    data_cutoff = pd.Timestamp(latest_bar["timestamp"]).tz_convert("UTC")
    current_price = float(latest_bar["close"])
    return FeatureBundle(
        bars_5m=bars_5m.reset_index(drop=True),
        features=features,
        data_cutoff=data_cutoff,
        current_price=current_price,
    )


def latest_feature_vector(features: pd.DataFrame) -> np.ndarray:
    row = features.iloc[-1]
    return np.array(
        [
            float(row["return_1h"]),
            float(row["return_4h"]),
            float(row["volatility_1h"]),
            float(row["volatility_4h"]),
            float(row["momentum"]),
            float(row["vol_slope"]),
        ],
        dtype=float,
    )


def feature_matrix(features: pd.DataFrame) -> np.ndarray:
    columns = ["return_1h", "return_4h", "volatility_1h", "volatility_4h", "momentum", "vol_slope"]
    return features[columns].to_numpy(dtype=float)


def five_minute_returns(features: pd.DataFrame) -> np.ndarray:
    return features["log_return"].to_numpy(dtype=float)


def _to_5m_bars(bars: pd.DataFrame) -> pd.DataFrame:
    indexed = bars.set_index("timestamp").sort_index()
    aggregated = indexed.resample("5min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "vwap": "mean",
            "transactions": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    return aggregated.reset_index()


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out

