from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


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


def test_dynamic_equity_config_maps_synth_equity_symbol_to_polygon_ticker():
    base_config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "assets": {},
        "synth": {"asset": "BTC", "competition": "crypto-24h"},
    }

    config = research.dynamic_equity_config(base_config, "NVDAX")

    assert config["asset"] == "NVDAX"
    assert config["polygon_ticker"] == "NVDA"
    assert config["synth"]["asset"] == "NVDAX"
    assert config["synth"]["competition"] == "com-equ-24h"
    assert config["assets"]["NVDAX"]["competition"] == "com-equ-24h"


def test_dynamic_equity_config_uses_xau_polygon_override():
    base_config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "assets": {},
        "synth": {"asset": "BTC", "competition": "crypto-24h"},
    }

    config = research.dynamic_equity_config(base_config, "XAU")

    assert config["polygon_ticker"] == "C:XAUUSD"


def test_dynamic_equity_config_uses_xag_polygon_override():
    base_config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "assets": {},
        "synth": {"asset": "BTC", "competition": "crypto-24h"},
    }

    config = research.dynamic_equity_config(base_config, "XAG")

    assert config["polygon_ticker"] == "C:XAGUSD"


def test_default_synth_equity_candidates_cover_common_equities_and_etfs():
    candidates = set(research.DEFAULT_SYNTH_EQUITY_CANDIDATES)

    assert {"XAU", "SPYX", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX"}.issubset(candidates)
    assert "AAPL" not in candidates
    assert "NVDA" not in candidates


def test_default_equity_assets_use_synth_validation_symbols():
    assert research.DEFAULT_EQUITY_ASSETS == (
        "XAU",
        "SPYX",
        "NVDAX",
        "GOOGLX",
        "TSLAX",
        "AAPLX",
        "WTIOIL",
        "SPCX",
    )


def test_synth_equity_polygon_proxy_mapping():
    assert research.EQUITY_TICKER_OVERRIDES["SPYX"] == "SPY"
    assert research.EQUITY_TICKER_OVERRIDES["NVDAX"] == "NVDA"
    assert research.EQUITY_TICKER_OVERRIDES["GOOGLX"] == "GOOGL"
    assert research.EQUITY_TICKER_OVERRIDES["TSLAX"] == "TSLA"
    assert research.EQUITY_TICKER_OVERRIDES["AAPLX"] == "AAPL"
    assert research.EQUITY_TICKER_OVERRIDES["WTIOIL"] == "USO"


def test_select_assets_can_use_active_discovered_assets():
    args = SimpleNamespace(
        assets=list(research.DEFAULT_ASSETS),
        equities=True,
        use_active_discovered_assets=True,
    )
    discovered = pd.DataFrame({"asset": ["XAU", "SPY", "AAPL"], "active": [True, False, True]})

    selected = research.select_assets_for_run(args, discovered)

    assert selected == ["XAU", "AAPL"]


def test_select_assets_requires_discovery_when_using_active_discovered_assets():
    args = SimpleNamespace(
        assets=list(research.DEFAULT_ASSETS),
        equities=True,
        use_active_discovered_assets=True,
    )

    with pytest.raises(SystemExit, match="requires --discover-synth-equities"):
        research.select_assets_for_run(args, None)


def test_equities_default_to_configured_equity_assets_without_discovery():
    args = SimpleNamespace(
        assets=list(research.DEFAULT_ASSETS),
        equities=True,
        use_active_discovered_assets=False,
    )

    assert research.select_assets_for_run(args, None) == list(research.DEFAULT_EQUITY_ASSETS)


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
