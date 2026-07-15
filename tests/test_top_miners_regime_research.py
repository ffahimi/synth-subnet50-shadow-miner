from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "top_miners_regime_research.py"
SPEC = importlib.util.spec_from_file_location("top_miners_regime_research", SCRIPT_PATH)
research = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(research)


def minute_bars(rows: int = 26 * 60) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(np.sin(np.arange(rows) / 30) * 0.0001 + 0.00002))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(rows),
            "vwap": close,
            "transactions": np.ones(rows),
        }
    )


def test_market_state_features_include_requested_windows():
    features = research.build_market_state_features(minute_bars())

    for hours in [1, 3, 5, 8, 24]:
        assert f"momentum_{hours}h" in features.columns
        assert f"realized_vol_{hours}h" in features.columns
        assert f"vol_of_vol_{hours}h" in features.columns

    assert features["momentum_24h"].notna().any()
    assert features["realized_vol_24h"].notna().any()


def test_dynamic_equity_config_infers_synth_and_polygon_settings():
    base_config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "assets": {},
        "synth": {"asset": "BTC", "competition": "crypto-24h"},
    }

    config = research.dynamic_equity_config(base_config, "SPY")

    assert config["asset"] == "SPY"
    assert config["polygon_ticker"] == "SPY"
    assert config["synth"]["asset"] == "SPY"
    assert config["synth"]["competition"] == "com-equ-24h"
    assert config["assets"]["SPY"]["competition"] == "com-equ-24h"


def test_dynamic_equity_config_uses_xau_polygon_override():
    base_config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "assets": {},
        "synth": {"asset": "BTC", "competition": "crypto-24h"},
    }

    config = research.dynamic_equity_config(base_config, "XAU")

    assert config["polygon_ticker"] == "C:XAUUSD"


def test_score_feature_join_uses_forecast_origin_time():
    features = research.build_market_state_features(minute_bars())
    scores = pd.DataFrame(
        {
            "miner_uid": [1],
            "scored_time": [pd.Timestamp("2024-01-02T02:00:00Z")],
            "feature_timestamp": [pd.Timestamp("2024-01-01T02:00:00Z")],
            "crps": [10.0],
            "crps_rank": [1],
        }
    )

    joined = research.score_feature_join(scores, features, tolerance_minutes=1)

    assert len(joined) == 1
    assert joined["timestamp"].iloc[0] == pd.Timestamp("2024-01-01T02:00:00Z")
    assert joined["feature_match_lag_minutes"].iloc[0] == 0.0


def test_feature_bucket_performance_and_consistency_have_rows():
    scores = []
    for idx, ts in enumerate(pd.date_range("2024-01-02", periods=30, freq="h", tz="UTC")):
        for miner_uid in [1, 2]:
            scores.append(
                {
                    "miner_uid": miner_uid,
                    "scored_time": ts,
                    "feature_timestamp": ts - pd.Timedelta(days=1),
                    "crps": float(idx + miner_uid),
                    "crps_rank": miner_uid,
                }
            )
    scores_df = pd.DataFrame(scores)
    features = research.build_market_state_features(minute_bars(rows=60 * 60))
    joined = research.score_feature_join(scores_df, features, tolerance_minutes=1)

    bucketed = research.feature_bucket_performance(joined, top_n=1)
    miner_consistency = research.miner_feature_consistency(joined, top_n=1)
    correlations = research.score_feature_correlations(joined)

    assert not bucketed.empty
    assert not miner_consistency.empty
    assert not correlations.empty
    assert {"feature", "bucket", "topn_rate"}.issubset(bucketed.columns)
