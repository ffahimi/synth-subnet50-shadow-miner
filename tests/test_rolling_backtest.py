from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from synth_shadow.backtest import rolling


def _config(tmp_path):
    return {
        "asset": "ETH",
        "polygon_ticker": "X:ETHUSD",
        "forecast": {
            "horizon_seconds": 30 * 60,
            "interval_seconds": 5 * 60,
            "num_paths": 99,
            "random_seed": 7,
        },
        "history": {
            "lookback_days": 1,
            "bar_multiplier": 5,
            "bar_timespan": "minute",
            "adjusted": True,
        },
        "backtest": {
            "days": 1,
            "stride_minutes": 5,
            "max_origins": None,
            "num_paths": 99,
            "compare_miners": 4,
        },
        "sampling": {"block_minutes": 10},
        "storage": {"backtest_dir": str(tmp_path / "backtests")},
    }


def _bars(periods: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-01T00:00:00Z", periods=periods, freq="5min")
    close = 1000.0 + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "vwap": close,
            "transactions": 1,
        }
    )


def _features(bars: pd.DataFrame, _config: dict) -> pd.DataFrame:
    features = bars.copy()
    features["session"] = "test"
    features["log_return"] = np.log(features["close"]).diff().fillna(0.0)
    features["vol_1h"] = 0.01
    features["vol_4h"] = 0.02
    features["vol_of_vol_1h"] = 0.001
    features["vol_of_vol_4h"] = 0.002
    features["vol_slope"] = 0.0
    features["momentum_1h"] = 0.0
    features["momentum_4h"] = 0.0
    features["kurtosis_4h"] = 0.0
    features["abs_return"] = features["log_return"].abs()
    return features


def test_rolling_backtest_is_causal_and_uses_expected_resolution_and_path_length(
    monkeypatch,
    tmp_path,
):
    config = _config(tmp_path)
    source_bars = _bars()
    expected_points = 7
    expected_paths = 4
    seen = {
        "library_max_timestamps": [],
        "states": [],
        "forecast_shapes": [],
        "realized_lengths": [],
        "realized_first_prices": [],
    }

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: source_bars)
    monkeypatch.setattr(rolling, "build_feature_frame", _features)
    monkeypatch.setattr(rolling, "_reference_miners", lambda _config: [])
    monkeypatch.setattr(rolling, "_save_backtest", lambda _rows, _result, _config: tmp_path / "saved")

    def fake_build_session_library(past_features: pd.DataFrame, block_bars: int):
        assert block_bars == 2
        assert not past_features.empty
        assert past_features["timestamp"].is_monotonic_increasing
        diffs = past_features["timestamp"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta(minutes=5)
        seen["library_max_timestamps"].append(pd.Timestamp(past_features["timestamp"].max()))
        return [SimpleNamespace(session="test")]

    def fake_extract_current_state(past_features: pd.DataFrame):
        row = past_features.iloc[-1]
        state = SimpleNamespace(
            timestamp=str(row["timestamp"]),
            price=float(row["close"]),
            session="test",
        )
        seen["states"].append(state)
        return state

    class FakePathSampler:
        def __init__(self, library, seed: int) -> None:
            assert library
            self.seed = seed

    def fake_generate_paths(state, _sampler, patched_config):
        shape = (patched_config["forecast"]["num_paths"], expected_points)
        seen["forecast_shapes"].append(shape)
        path = np.full(shape, float(state.price))
        return path, {"state_timestamp": state.timestamp}

    def fake_score(paths: np.ndarray, realized: np.ndarray):
        seen["realized_lengths"].append(len(realized))
        seen["realized_first_prices"].append(float(realized[0]))
        assert paths.shape == (expected_paths, expected_points)
        assert len(realized) == expected_points
        return {
            "raw_crps": 1.0,
            "components": {
                "crps_5m": 0.1,
                "crps_30m": 0.2,
                "crps_3h": 0.3,
                "crps_24h": 0.4,
                "crps_path_price": 0.5,
            },
        }

    monkeypatch.setattr(rolling, "build_session_library", fake_build_session_library)
    monkeypatch.setattr(rolling, "extract_current_state", fake_extract_current_state)
    monkeypatch.setattr(rolling, "PathSampler", FakePathSampler)
    monkeypatch.setattr(rolling, "generate_paths", fake_generate_paths)
    monkeypatch.setattr(rolling, "score_synth_btc_24h", fake_score)

    result = rolling.run_rolling_backtest(
        config,
        days=0.5,
        stride_minutes=10,
        max_origins=5,
        num_paths=expected_paths,
    )

    origins = [pd.Timestamp(row["origin"]) for row in result["first_rows"] + result["last_rows"]]
    scored_origins = [pd.Timestamp(state.timestamp) for state in seen["states"]]
    assert len(scored_origins) == 5
    assert scored_origins == sorted(scored_origins)
    assert all(
        later - earlier == pd.Timedelta(minutes=10)
        for earlier, later in zip(scored_origins, scored_origins[1:])
    )

    for origin, library_max, state, realized_first in zip(
        scored_origins,
        seen["library_max_timestamps"],
        seen["states"],
        seen["realized_first_prices"],
    ):
        assert library_max == origin
        assert pd.Timestamp(state.timestamp) == origin
        assert realized_first == state.price

    assert all(shape == (expected_paths, expected_points) for shape in seen["forecast_shapes"])
    assert seen["realized_lengths"] == [expected_points] * 5
    assert result["summary"]["origin_count"] == 5
    assert result["config"] == {
        "days": 0.5,
        "stride_minutes": 10,
        "num_paths": expected_paths,
    }
    assert origins
