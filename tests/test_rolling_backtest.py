from __future__ import annotations

import os
import random
import secrets
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from synth_shadow.assets import apply_asset
from synth_shadow.backtest import rolling
from synth_shadow.config import load_config
from synth_shadow.forecasting.protocol import ProviderForecast
from synth_shadow.models.protocol import ForecastContext, ForecastOutput


def _print_report(title: str, values: dict[str, object]) -> None:
    print(f"\n[{title}]")
    for key, value in values.items():
        print(f"{key}: {value}")


def test_recent_polygon_rolling_backtest_is_causal_and_shape_correct(
    monkeypatch,
    tmp_path,
):
    """Live recent-data audit for rolling backtest causality and path dimensions."""
    config = apply_asset(load_config("config/default.yaml"), "ETH")
    monkeypatch.delenv("SYNTH_MODEL_ENDPOINT", raising=False)
    config["model"]["endpoint"] = None
    if not os.getenv("POLYGON_API_KEY"):
        pytest.skip("POLYGON_API_KEY is required for the recent Polygon backtest audit.")

    audit_days = 2.0
    stride_minutes = 5
    origins_to_score = 5
    num_paths = 8
    interval_seconds = int(config["forecast"]["interval_seconds"])
    horizon_seconds = int(config["forecast"]["horizon_seconds"])
    expected_interval = pd.Timedelta(seconds=interval_seconds)
    expected_points = int(horizon_seconds / interval_seconds) + 1
    sample_seed = secrets.randbits(32)
    selected_origins: list[pd.Timestamp] = []
    audit = {
        "feature_rows_seen": None,
        "library_rows_per_origin": [],
        "library_max_timestamps": [],
        "model_bar_max_timestamps": [],
        "model_feature_max_timestamps": [],
        "state_timestamps": [],
        "state_prices": [],
        "forecast_shapes": [],
        "realized_lengths": [],
        "realized_first_prices": [],
    }

    try:
        source_bars = rolling._load_backtest_bars(config, audit_days)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403}:
            pytest.skip(
                "Polygon rejected POLYGON_API_KEY during live audit "
                f"with HTTP {status_code}. Set a valid key to run this test."
            )
        raise
    bar_diffs = source_bars["timestamp"].diff().dropna()
    assert not source_bars.empty
    assert bar_diffs.eq(expected_interval).all()

    _print_report(
        "recent polygon source data",
        {
            "asset": config["asset"],
            "polygon_ticker": config["polygon_ticker"],
            "sample_seed": sample_seed,
            "raw_bar_count": len(source_bars),
            "raw_first_timestamp": source_bars["timestamp"].min(),
            "raw_last_timestamp": source_bars["timestamp"].max(),
            "raw_resolution_seconds": interval_seconds,
            "close_min": round(float(source_bars["close"].min()), 6),
            "close_max": round(float(source_bars["close"].max()), 6),
            "close_last": round(float(source_bars["close"].iloc[-1]), 6),
        },
    )

    original_build_feature_frame = rolling.build_feature_frame
    original_select_origins = rolling._select_origins
    original_extract_current_state = rolling.extract_current_state

    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: source_bars)
    monkeypatch.setattr(rolling, "_new_backtest_output_dir", lambda _config: saved_dir)

    def audited_build_feature_frame(bars: pd.DataFrame, patched_config: dict) -> pd.DataFrame:
        features = original_build_feature_frame(bars, patched_config)
        audit["feature_rows_seen"] = len(features)
        assert features["timestamp"].is_monotonic_increasing
        feature_diffs = features["timestamp"].diff().dropna()
        assert feature_diffs.eq(expected_interval).all()
        _print_report(
            "feature frame",
            {
                "feature_row_count": len(features),
                "feature_first_timestamp": features["timestamp"].min(),
                "feature_last_timestamp": features["timestamp"].max(),
                "feature_resolution_seconds": interval_seconds,
                "sessions": {
                    session: int(count)
                    for session, count in features["session"].value_counts().sort_index().items()
                },
            },
        )
        return features

    def random_recent_origins(
        features: pd.DataFrame,
        patched_config: dict,
        days: float,
        stride: int,
        max_origins: int | None,
        origin_source: str = "polygon",
        realized_source: str = "polygon",
        score_snapshot_cache: dict | None = None,
    ) -> list[pd.Timestamp]:
        candidates = original_select_origins(
            features,
            patched_config,
            days,
            stride,
            None,
            origin_source,
            realized_source,
            score_snapshot_cache,
        )
        assert len(candidates) >= origins_to_score
        max_start = len(candidates) - origins_to_score
        start = random.Random(sample_seed).randint(0, max_start)
        sample = candidates[start : start + origins_to_score]
        selected_origins[:] = [pd.Timestamp(origin) for origin in sample]
        _print_report(
            "random recent origin sample",
            {
                "candidate_origin_count": len(candidates),
                "sample_start_index": start,
                "sample_origin_count": len(sample),
                "sample_first_origin": sample[0],
                "sample_last_origin": sample[-1],
                "requested_stride_minutes": stride,
                "max_origins_argument": max_origins,
            },
        )
        return sample

    def audited_build_session_library(past_features: pd.DataFrame, block_bars: int):
        assert not past_features.empty
        assert past_features["timestamp"].is_monotonic_increasing
        diffs = past_features["timestamp"].diff().dropna()
        assert diffs.eq(expected_interval).all()
        audit["library_rows_per_origin"].append(len(past_features))
        audit["library_max_timestamps"].append(pd.Timestamp(past_features["timestamp"].max()))
        return [SimpleNamespace(session=str(past_features["session"].iloc[-1]))]

    def audited_extract_current_state(past_features: pd.DataFrame):
        state = original_extract_current_state(past_features)
        audit["state_timestamps"].append(pd.Timestamp(state.timestamp))
        audit["state_prices"].append(float(state.price))
        return state

    class AuditedPathSampler:
        def __init__(self, library, seed: int) -> None:
            assert library
            self.library = library
            self.seed = seed

    class AuditedForecastModel:
        model_version = "test_private_model_audit"

        def generate(self, context: ForecastContext) -> ForecastOutput:
            origin = pd.Timestamp(context.origin)
            assert context.bars["timestamp"].max() == origin
            assert context.features["timestamp"].max() == origin
            assert pd.Timestamp(context.state.timestamp) == origin
            audit["model_bar_max_timestamps"].append(pd.Timestamp(context.bars["timestamp"].max()))
            audit["model_feature_max_timestamps"].append(
                pd.Timestamp(context.features["timestamp"].max())
            )
            shape = (int(context.config["forecast"]["num_paths"]), expected_points)
            audit["forecast_shapes"].append(shape)
            paths = np.full(shape, float(context.state.price))
            timestamps = pd.date_range(
                origin,
                periods=expected_points,
                freq=expected_interval,
                tz="UTC",
            )
            return ForecastOutput(
                paths=paths,
                timestamps=timestamps,
                metadata={"audit": "private-model-shaped test model"},
            )

    def audited_score(paths: np.ndarray, realized: np.ndarray):
        audit["realized_lengths"].append(len(realized))
        audit["realized_first_prices"].append(float(realized[0]))
        assert paths.shape == (num_paths, expected_points)
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

    monkeypatch.setattr(rolling, "build_feature_frame", audited_build_feature_frame)
    monkeypatch.setattr(rolling, "_select_origins", random_recent_origins)
    monkeypatch.setattr(rolling, "build_session_library", audited_build_session_library)
    monkeypatch.setattr(rolling, "extract_current_state", audited_extract_current_state)
    monkeypatch.setattr(rolling, "PathSampler", AuditedPathSampler)
    monkeypatch.setattr(rolling, "load_forecast_model", lambda _config: AuditedForecastModel())
    monkeypatch.setattr(
        rolling,
        "configured_model_entrypoint",
        lambda _config: "tests.test_rolling_backtest:AuditedForecastModel",
    )
    monkeypatch.setattr(rolling, "score_synth_btc_24h", audited_score)

    result = rolling.run_rolling_backtest(
        config,
        days=audit_days,
        stride_minutes=stride_minutes,
        max_origins=origins_to_score,
        num_paths=num_paths,
    )

    scored_origins = audit["state_timestamps"]
    assert len(scored_origins) == origins_to_score
    assert selected_origins == scored_origins
    assert all(
        later - earlier == pd.Timedelta(minutes=stride_minutes)
        for earlier, later in zip(scored_origins, scored_origins[1:])
    )

    for origin, library_max, model_bar_max, model_feature_max, state_price, realized_first in zip(
        scored_origins,
        audit["library_max_timestamps"],
        audit["model_bar_max_timestamps"],
        audit["model_feature_max_timestamps"],
        audit["state_prices"],
        audit["realized_first_prices"],
    ):
        assert library_max == origin
        assert model_bar_max == origin
        assert model_feature_max == origin
        assert realized_first == state_price

    assert all(shape == (num_paths, expected_points) for shape in audit["forecast_shapes"])
    assert audit["realized_lengths"] == [expected_points] * origins_to_score
    assert result["summary"]["origin_count"] == origins_to_score
    assert result["config"] == {
        "days": audit_days,
        "stride_minutes": stride_minutes,
        "num_paths": num_paths,
        "maturity_lag_minutes": 60,
        "checkpoint_every": 25,
        "origin_source": "polygon",
        "realized_source": "polygon",
    }

    origin_details = [
        {
            "origin": str(origin),
            "library_rows": rows,
            "state_price": round(price, 6),
            "forecast_shape": shape,
            "realized_points": realized_points,
        }
        for origin, rows, price, shape, realized_points in zip(
            scored_origins,
            audit["library_rows_per_origin"],
            audit["state_prices"],
            audit["forecast_shapes"],
            audit["realized_lengths"],
        )
    ]
    _print_report(
        "passed audit checks",
        {
            "past_only_check": "PASSED: each library/state max timestamp equals its origin",
            "model_context_causality": (
                "PASSED: private-model context receives only bars/features <= origin"
            ),
            "source_resolution_check": f"PASSED: all source/feature bars are {interval_seconds}s apart",
            "origin_stride_check": f"PASSED: {origins_to_score} origins are {stride_minutes} minutes apart",
            "forecast_shape_check": f"PASSED: every forecast is {num_paths} x {expected_points}",
            "realized_length_check": f"PASSED: every realized path has {expected_points} points",
            "realized_origin_alignment": "PASSED: realized first price equals current state price",
            "scored_origin_count": len(scored_origins),
            "scored_origins": [str(origin) for origin in scored_origins],
            "library_rows_per_origin": audit["library_rows_per_origin"],
            "model_bar_max_timestamps": [str(ts) for ts in audit["model_bar_max_timestamps"]],
            "model_feature_max_timestamps": [
                str(ts) for ts in audit["model_feature_max_timestamps"]
            ],
            "forecast_shapes": audit["forecast_shapes"],
            "origin_details": origin_details,
            "model": result["model"],
            "result_config": result["config"],
        },
    )


def test_rolling_backtest_uses_http_provider_when_endpoint_configured(monkeypatch, tmp_path):
    origin = pd.Timestamp("2026-07-10T03:00:00Z")
    interval_seconds = 300
    horizon_seconds = 600
    expected_points = 3
    timestamps = pd.date_range(origin - pd.Timedelta(minutes=20), periods=8, freq="300s", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(99.0, 106.0, len(timestamps)),
            "high": np.linspace(100.0, 107.0, len(timestamps)),
            "low": np.linspace(98.0, 105.0, len(timestamps)),
            "close": np.linspace(100.0, 107.0, len(timestamps)),
            "volume": np.ones(len(timestamps)),
        }
    )
    features = pd.DataFrame({"timestamp": timestamps})
    provider_calls = []

    config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "history": {"lookback_days": 1},
        "forecast": {
            "horizon_seconds": horizon_seconds,
            "interval_seconds": interval_seconds,
            "num_paths": 2,
            "random_seed": 7,
        },
        "sampling": {"block_minutes": 60},
        "model": {"endpoint": "http://127.0.0.1:8088/predict"},
        "backtest": {
            "days": 1,
            "stride_minutes": 5,
            "max_origins": 1,
            "num_paths": 2,
            "compare_miners": 4,
            "realized_source": "polygon",
        },
        "storage": {"backtest_dir": str(tmp_path / "backtests")},
    }

    class FakeProvider:
        def generate(self, run_config, prompt_start_time=None, origin=None):
            provider_calls.append(
                {
                    "num_paths": run_config["forecast"]["num_paths"],
                    "prompt_start_time": prompt_start_time,
                    "origin": pd.Timestamp(origin),
                }
            )
            paths = np.array([[104.0, 105.0, 106.0], [104.0, 103.0, 102.0]])
            return ProviderForecast(
                paths=paths,
                timestamps=pd.date_range(origin, periods=expected_points, freq="300s", tz="UTC"),
                metadata={
                    "provider": "http",
                    "model_version": "private_http_backtest_v1",
                    "model_entrypoint": "http://127.0.0.1:8088/predict",
                    "data_cutoff": str(origin),
                    "current_price": 104.0,
                },
            )

    def fail_load_forecast_model(_config):
        raise AssertionError("HTTP backtest should not load the local in-process model.")

    def fake_score(paths: np.ndarray, realized: np.ndarray):
        assert paths.shape == (2, expected_points)
        assert realized.tolist() == [104.0, 105.0, 106.0]
        return {
            "raw_crps": 2.0,
            "components": {
                "crps_5m": 0.1,
                "crps_30m": 0.2,
                "crps_3h": 0.3,
                "crps_24h": 0.4,
                "crps_path_price": 0.5,
            },
        }

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: bars)
    monkeypatch.setattr(rolling, "build_feature_frame", lambda _bars, _config: features)
    monkeypatch.setattr(rolling, "_select_origins", lambda *_args: [origin])
    monkeypatch.setattr(rolling, "load_forecast_provider", lambda _config: FakeProvider())
    monkeypatch.setattr(rolling, "load_forecast_model", fail_load_forecast_model)
    monkeypatch.setattr(rolling, "score_synth_btc_24h", fake_score)
    monkeypatch.setattr(
        rolling,
        "_historical_miner_scores_for_origins",
        lambda _config, _origins, _stride_minutes, **_kwargs: {
            origin: [
                {"miner_uid": 1, "crps": 1.5, "scored_time": str(origin)},
                {"miner_uid": 2, "crps": 3.0, "scored_time": str(origin)},
            ]
        },
    )

    result = rolling.run_rolling_backtest(
        config,
        days=1,
        stride_minutes=5,
        max_origins=1,
        num_paths=2,
    )

    assert provider_calls == [
        {
            "num_paths": 2,
            "prompt_start_time": "2026-07-10T03:00:00+00:00",
            "origin": origin,
        }
    ]
    assert result["summary"]["origin_count"] == 1
    assert result["model"] == {
        "model_version": "private_http_backtest_v1",
        "model_entrypoint": "http://127.0.0.1:8088/predict",
    }
    assert result["first_rows"][0]["current_price"] == 104.0
    assert result["first_rows"][0]["realized_source"] == "polygon"
    assert result["historical_miner_snapshot"]["score_snapshot_count"] == 1


def test_rolling_backtest_can_skip_historical_miner_fetch(monkeypatch, tmp_path):
    origin = pd.Timestamp("2026-07-10T03:00:00Z")
    interval_seconds = 300
    horizon_seconds = 600
    timestamps = pd.date_range(origin - pd.Timedelta(minutes=20), periods=8, freq="300s", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(99.0, 106.0, len(timestamps)),
            "high": np.linspace(100.0, 107.0, len(timestamps)),
            "low": np.linspace(98.0, 105.0, len(timestamps)),
            "close": np.linspace(100.0, 107.0, len(timestamps)),
            "volume": np.ones(len(timestamps)),
        }
    )
    features = pd.DataFrame({"timestamp": timestamps})
    config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "history": {"lookback_days": 1},
        "forecast": {
            "horizon_seconds": horizon_seconds,
            "interval_seconds": interval_seconds,
            "num_paths": 2,
            "random_seed": 7,
        },
        "sampling": {"block_minutes": 60},
        "model": {"endpoint": "http://127.0.0.1:8088/predict"},
        "backtest": {
            "days": 1,
            "stride_minutes": 5,
            "max_origins": 1,
            "num_paths": 2,
            "compare_miners": 0,
            "realized_source": "polygon",
        },
        "storage": {"backtest_dir": str(tmp_path / "backtests")},
    }

    class FakeProvider:
        def generate(self, run_config, prompt_start_time=None, origin=None):
            paths = np.array([[104.0, 105.0, 106.0], [104.0, 103.0, 102.0]])
            return ProviderForecast(
                paths=paths,
                timestamps=pd.date_range(origin, periods=3, freq="300s", tz="UTC"),
                metadata={
                    "provider": "http",
                    "model_version": "private_http_backtest_v1",
                    "model_entrypoint": "http://127.0.0.1:8088/predict",
                    "data_cutoff": str(origin),
                    "current_price": 104.0,
                },
            )

    def fail_historical_fetch(*_args, **_kwargs):
        raise AssertionError("compare_miners=0 should skip Synth historical score fetches")

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: bars)
    monkeypatch.setattr(rolling, "build_feature_frame", lambda _bars, _config: features)
    monkeypatch.setattr(rolling, "_select_origins", lambda *_args: [origin])
    monkeypatch.setattr(rolling, "load_forecast_provider", lambda _config: FakeProvider())
    monkeypatch.setattr(rolling, "_historical_miner_scores_for_origins", fail_historical_fetch)
    monkeypatch.setattr(rolling, "score_synth_btc_24h", lambda _paths, _realized: {
        "raw_crps": 2.0,
        "components": {
            "crps_5m": 0.1,
            "crps_30m": 0.2,
            "crps_3h": 0.3,
            "crps_24h": 0.4,
            "crps_path_price": 0.5,
        },
    })

    result = rolling.run_rolling_backtest(config, compare_miners=0)

    assert result["summary"]["origin_count"] == 1
    assert result["historical_miner_snapshot"]["score_snapshot_count"] == 0
    assert result["first_rows"][0]["historical_miner_count"] == 0


def test_fast_origins_bypasses_synth_score_origin_fetch(monkeypatch, tmp_path):
    origin = pd.Timestamp("2026-07-10T03:00:00Z")
    interval_seconds = 300
    horizon_seconds = 600
    timestamps = pd.date_range(origin - pd.Timedelta(minutes=20), periods=8, freq="300s", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(99.0, 106.0, len(timestamps)),
            "high": np.linspace(100.0, 107.0, len(timestamps)),
            "low": np.linspace(98.0, 105.0, len(timestamps)),
            "close": np.linspace(100.0, 107.0, len(timestamps)),
            "volume": np.ones(len(timestamps)),
        }
    )
    features = pd.DataFrame({"timestamp": timestamps})
    config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "history": {"lookback_days": 1},
        "forecast": {
            "horizon_seconds": horizon_seconds,
            "interval_seconds": interval_seconds,
            "num_paths": 2,
            "random_seed": 7,
        },
        "sampling": {"block_minutes": 60},
        "model": {"endpoint": "http://127.0.0.1:8088/predict"},
        "backtest": {
            "days": 1,
            "stride_minutes": 5,
            "max_origins": 1,
            "num_paths": 2,
            "compare_miners": 0,
            "origin_source": "synth",
            "realized_source": "synth",
        },
        "storage": {"backtest_dir": str(tmp_path / "backtests")},
    }
    selected_origin_sources = []
    selected_max_origins = []

    class FakeProvider:
        def generate(self, run_config, prompt_start_time=None, origin=None):
            return ProviderForecast(
                paths=np.array([[104.0, 105.0, 106.0], [104.0, 103.0, 102.0]]),
                timestamps=pd.date_range(origin, periods=3, freq="300s", tz="UTC"),
                metadata={"current_price": 104.0, "data_cutoff": str(origin)},
            )

    def fake_select_origins(
        _features,
        _config,
        _days,
        _stride_minutes,
        _max_origins,
        origin_source,
        _realized_source,
        _score_snapshot_cache,
    ):
        selected_origin_sources.append(origin_source)
        selected_max_origins.append(_max_origins)
        return [origin]

    def fail_historical_rows(*_args, **_kwargs):
        raise AssertionError("fast_origins should avoid Synth historical score-origin fetches")

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: bars)
    monkeypatch.setattr(rolling, "build_feature_frame", lambda _bars, _config: features)
    monkeypatch.setattr(rolling, "_select_origins", fake_select_origins)
    monkeypatch.setattr(rolling, "_fetch_historical_score_rows", fail_historical_rows)
    monkeypatch.setattr(rolling, "load_forecast_provider", lambda _config: FakeProvider())
    monkeypatch.setattr(
        rolling,
        "_load_realized_path_for_origin",
        lambda *_args, **_kwargs: np.array([104.0, 105.0, 106.0]),
    )
    monkeypatch.setattr(rolling, "score_synth_btc_24h", lambda _paths, _realized: {
        "raw_crps": 2.0,
        "components": {
            "crps_5m": 0.1,
            "crps_30m": 0.2,
            "crps_3h": 0.3,
            "crps_24h": 0.4,
            "crps_path_price": 0.5,
        },
    })

    result = rolling.run_rolling_backtest(config, compare_miners=0, fast_origins=True)

    assert selected_origin_sources == ["polygon"]
    assert selected_max_origins == [24]
    assert result["summary"]["origin_count"] == 1


def test_synth_realized_source_loads_realized_path(monkeypatch):
    origin = pd.Timestamp("2026-07-10T03:00:00Z")
    future = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    calls = []

    class FakeSynthClient:
        def __init__(self, config):
            calls.append(config["asset"])

        def realized_path(self, start_time):
            assert start_time == "2026-07-10T03:00:00Z"
            return {"real_prices": [100.0, 101.0, 102.0]}

    monkeypatch.setattr(rolling, "SynthClient", FakeSynthClient)

    realized = rolling._load_realized_path_for_origin(
        {"asset": "BTC"},
        origin,
        future,
        points_per_path=3,
        realized_source="synth",
    )

    assert calls == ["BTC"]
    assert realized.tolist() == [100.0, 101.0, 102.0]


def test_synth_backtest_skips_unavailable_realized_before_provider(monkeypatch, tmp_path):
    older_origin = pd.Timestamp("2026-07-10T03:00:00Z")
    newer_origin = pd.Timestamp("2026-07-10T03:05:00Z")
    interval_seconds = 300
    horizon_seconds = 600
    expected_points = 3
    timestamps = pd.date_range(
        older_origin - pd.Timedelta(minutes=20),
        periods=10,
        freq="300s",
        tz="UTC",
    )
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(99.0, 108.0, len(timestamps)),
            "high": np.linspace(100.0, 109.0, len(timestamps)),
            "low": np.linspace(98.0, 107.0, len(timestamps)),
            "close": np.linspace(100.0, 109.0, len(timestamps)),
            "volume": np.ones(len(timestamps)),
        }
    )
    features = pd.DataFrame({"timestamp": timestamps})
    provider_calls = []

    config = {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "history": {"lookback_days": 1},
        "forecast": {
            "horizon_seconds": horizon_seconds,
            "interval_seconds": interval_seconds,
            "num_paths": 2,
            "random_seed": 7,
        },
        "sampling": {"block_minutes": 60},
        "model": {"endpoint": "http://127.0.0.1:8088/predict"},
        "backtest": {
            "days": 1,
            "stride_minutes": 5,
            "max_origins": 1,
            "num_paths": 2,
            "compare_miners": 4,
            "origin_source": "synth",
            "realized_source": "synth",
            "synth_realized_scan_multiplier": 24,
        },
        "storage": {"backtest_dir": str(tmp_path / "backtests")},
    }

    class FakeProvider:
        def generate(self, run_config, prompt_start_time=None, origin=None):
            provider_calls.append(pd.Timestamp(origin))
            paths = np.array([[104.0, 105.0, 106.0], [104.0, 103.0, 102.0]])
            return ProviderForecast(
                paths=paths,
                timestamps=pd.date_range(origin, periods=expected_points, freq="300s", tz="UTC"),
                metadata={
                    "provider": "http",
                    "model_version": "private_http_backtest_v1",
                    "model_entrypoint": "http://127.0.0.1:8088/predict",
                    "data_cutoff": str(origin),
                    "current_price": 104.0,
                },
            )

    class FakeSynthClient:
        def __init__(self, _config, timeout_seconds=30):
            pass

        def realized_path(self, start_time):
            if start_time == "2026-07-10T03:05:00Z":
                response = requests.Response()
                response.status_code = 404
                raise requests.HTTPError("404 Client Error", response=response)
            assert start_time == "2026-07-10T03:00:00Z"
            return {"real_prices": [104.0, 105.0, 106.0]}

    def fake_select_origins(
        _features,
        _config,
        _days,
        _stride,
        selection_max,
        _origin_source,
        _realized_source,
        _score_snapshot_cache,
    ):
        assert selection_max == 24
        return [older_origin, newer_origin]

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: bars)
    monkeypatch.setattr(rolling, "build_feature_frame", lambda _bars, _config: features)
    monkeypatch.setattr(rolling, "_select_origins", fake_select_origins)
    monkeypatch.setattr(rolling, "load_forecast_provider", lambda _config: FakeProvider())
    monkeypatch.setattr(rolling, "SynthClient", FakeSynthClient)
    monkeypatch.setattr(rolling, "score_synth_btc_24h", lambda _paths, _realized: {
        "raw_crps": 2.0,
        "components": {
            "crps_5m": 0.1,
            "crps_30m": 0.2,
            "crps_3h": 0.3,
            "crps_24h": 0.4,
            "crps_path_price": 0.5,
        },
    })
    monkeypatch.setattr(rolling, "_historical_miner_scores_for_origins", lambda *_args, **_kwargs: {})

    result = rolling.run_rolling_backtest(
        config,
        max_origins=1,
        num_paths=2,
        realized_source="synth",
        origin_source="synth",
    )

    assert provider_calls == [older_origin]
    assert result["summary"]["origin_count"] == 1
    assert result["first_rows"][0]["origin"] == str(older_origin)
    assert result["first_rows"][0]["realized_source"] == "synth"


def test_synth_origin_source_uses_official_prompt_times(monkeypatch):
    feature_times = pd.date_range(
        "2026-07-10T02:00:00Z",
        periods=19,
        freq="300s",
        tz="UTC",
    )
    features = pd.DataFrame({"timestamp": feature_times})
    config = {
        "asset": "BTC",
        "backtest": {"maturity_lag_minutes": 0},
        "forecast": {"horizon_seconds": 600, "interval_seconds": 300},
    }
    prompts_seen = []

    class FakeSynthClient:
        def __init__(self, client_config):
            assert client_config is config

        def prompts(self, start=None, end=None):
            prompts_seen.append((start, end))
            return [
                "2026-07-10T02:10:00Z",
                "2026-07-10T03:01:00Z",
                "2026-07-10T03:06:00Z",
                "2026-07-10T03:25:00Z",
            ]

    monkeypatch.setattr(rolling, "SynthClient", FakeSynthClient)

    origins = rolling._select_origins(
        features,
        config,
        days=1 / 24,
        stride_minutes=5,
        max_origins=None,
        origin_source="synth",
    )

    assert len(prompts_seen) == 1
    assert prompts_seen[0][0] == pd.Timestamp("2026-07-10T02:20:00Z")
    assert prompts_seen[0][1] == pd.Timestamp("2026-07-10T03:20:00Z")
    assert origins == [
        pd.Timestamp("2026-07-10T03:01:00Z"),
        pd.Timestamp("2026-07-10T03:06:00Z"),
    ]


def test_polygon_fast_origins_apply_synth_realized_maturity_lag():
    feature_times = pd.date_range(
        "2026-07-10T02:00:00Z",
        periods=25,
        freq="300s",
        tz="UTC",
    )
    features = pd.DataFrame({"timestamp": feature_times})
    config = {
        "asset": "BTC",
        "backtest": {"maturity_lag_minutes": 30},
        "forecast": {"horizon_seconds": 600, "interval_seconds": 300},
    }

    origins = rolling._select_origins(
        features,
        config,
        days=1 / 24,
        stride_minutes=5,
        max_origins=3,
        origin_source="polygon",
        realized_source="synth",
    )

    assert origins == [
        pd.Timestamp("2026-07-10T03:10:00Z"),
        pd.Timestamp("2026-07-10T03:15:00Z"),
        pd.Timestamp("2026-07-10T03:20:00Z"),
    ]


def test_synth_realized_origin_source_uses_historical_score_snapshots(monkeypatch):
    feature_times = pd.date_range(
        "2026-07-10T02:00:00Z",
        periods=19,
        freq="300s",
        tz="UTC",
    )
    features = pd.DataFrame({"timestamp": feature_times})
    config = {
        "asset": "BTC",
        "backtest": {
            "maturity_lag_minutes": 0,
            "stride_minutes": 5,
            "historical_score_tolerance_minutes": 30,
            "historical_score_chunk_hours": 1,
        },
        "forecast": {"horizon_seconds": 600, "interval_seconds": 300},
        "synth": {"timeout_seconds": 90},
    }
    score_windows = []

    class FakeSynthClient:
        def __init__(self, client_config, timeout_seconds=30):
            assert client_config is config
            assert timeout_seconds == 90

        def historical_scores(self, start=None, end=None):
            score_windows.append((start, end))
            return [
                {"miner_uid": 1, "crps": 10.0, "scored_time": "2026-07-10T02:35:00Z"},
                {"miner_uid": 2, "crps": 11.0, "scored_time": "2026-07-10T02:35:00Z"},
                {"miner_uid": 1, "crps": 9.0, "scored_time": "2026-07-10T03:20:00Z"},
                {"miner_uid": 1, "crps": 8.0, "scored_time": "2026-07-10T03:45:00Z"},
            ]

    monkeypatch.setattr(rolling, "SynthClient", FakeSynthClient)

    cache = {}
    origins = rolling._select_origins(
        features,
        config,
        days=1 / 24,
        stride_minutes=5,
        max_origins=None,
        origin_source="synth",
        realized_source="synth",
        score_snapshot_cache=cache,
    )

    assert len(score_windows) > 1
    assert all(end - start <= pd.Timedelta(hours=1) for start, end in score_windows)
    assert len(cache["snapshots"]) == 3
    assert origins == [
        pd.Timestamp("2026-07-10T02:25:00Z"),
        pd.Timestamp("2026-07-10T03:10:00Z"),
    ]


def test_historical_score_fetch_chunks_long_windows(monkeypatch):
    calls = []
    config = {
        "asset": "BTC",
        "backtest": {"historical_score_chunk_hours": 6},
        "synth": {"timeout_seconds": 90},
    }

    class FakeSynthClient:
        def __init__(self, client_config, timeout_seconds=30):
            assert client_config is config
            assert timeout_seconds == 90

        def historical_scores(self, start=None, end=None):
            calls.append((start, end))
            return [{"scored_time": str(end), "miner_uid": len(calls), "crps": 1.0}]

    monkeypatch.setattr(rolling, "SynthClient", FakeSynthClient)

    rows = rolling._fetch_historical_score_rows(
        config,
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T03:00:00Z"),
        context="test",
    )

    assert len(calls) == 5
    assert calls[0] == (
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T06:00:00Z"),
    )
    assert calls[-1] == (
        pd.Timestamp("2026-07-02T00:00:00Z"),
        pd.Timestamp("2026-07-02T03:00:00Z"),
    )
    assert len(rows) == 5


def test_historical_miner_scores_reuse_cached_snapshots(monkeypatch):
    config = {
        "asset": "BTC",
        "forecast": {"horizon_seconds": 600},
        "backtest": {"historical_score_tolerance_minutes": 5},
    }
    origin = pd.Timestamp("2026-07-10T03:00:00Z")
    cached = {
        pd.Timestamp("2026-07-10T03:10:00Z"): [{"miner_uid": 1, "crps": 10.0}],
        pd.Timestamp("2026-07-10T04:10:00Z"): [{"miner_uid": 2, "crps": 20.0}],
    }

    class FailSynthClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cached snapshots should avoid a second Synth score fetch")

    monkeypatch.setattr(rolling, "SynthClient", FailSynthClient)

    grouped = rolling._historical_miner_scores_for_origins(
        config,
        [origin],
        stride_minutes=5,
        cached_snapshots=cached,
    )

    assert grouped == {pd.Timestamp("2026-07-10T03:10:00Z"): [{"miner_uid": 1, "crps": 10.0}]}


def test_historical_miner_score_match_uses_same_day_fallback():
    config = {
        "forecast": {"horizon_seconds": 86400},
        "backtest": {"historical_score_tolerance_minutes": 30},
    }
    origin = pd.Timestamp("2026-07-12T17:05:00Z")
    snapshots = {
        pd.Timestamp("2026-07-13T16:45:00Z"): [{"miner_uid": 1, "crps": 100.0}],
        pd.Timestamp("2026-07-13T08:00:00Z"): [{"miner_uid": 2, "crps": 200.0}],
        pd.Timestamp("2026-07-14T17:05:00Z"): [{"miner_uid": 3, "crps": 300.0}],
    }

    matched = rolling._nearest_historical_miner_scores(
        origin,
        snapshots,
        tolerance=pd.Timedelta(minutes=10),
        config=config,
    )

    assert matched is not None
    assert matched["target_scored_time"] == "2026-07-13 17:05:00+00:00"
    assert matched["scored_time"] == "2026-07-13 16:45:00+00:00"
    assert matched["delta_minutes"] == 20.0
    assert matched["match_type"] == "same_day"
    assert matched["scores"] == [{"miner_uid": 1, "crps": 100.0}]


def test_historical_miner_score_match_prefers_tolerance_match():
    config = {
        "forecast": {"horizon_seconds": 86400},
        "backtest": {"historical_score_tolerance_minutes": 30},
    }
    origin = pd.Timestamp("2026-07-12T17:05:00Z")
    snapshots = {
        pd.Timestamp("2026-07-13T17:00:00Z"): [{"miner_uid": 1, "crps": 100.0}],
        pd.Timestamp("2026-07-13T16:45:00Z"): [{"miner_uid": 2, "crps": 200.0}],
    }

    matched = rolling._nearest_historical_miner_scores(
        origin,
        snapshots,
        tolerance=pd.Timedelta(minutes=10),
        config=config,
    )

    assert matched is not None
    assert matched["scored_time"] == "2026-07-13 17:00:00+00:00"
    assert matched["delta_minutes"] == 5.0
    assert matched["match_type"] == "nearest_tolerance"


def test_historical_miner_score_match_rejects_different_day():
    config = {
        "forecast": {"horizon_seconds": 86400},
        "backtest": {"historical_score_tolerance_minutes": 30},
    }
    origin = pd.Timestamp("2026-07-12T17:05:00Z")
    snapshots = {
        pd.Timestamp("2026-07-14T17:05:00Z"): [{"miner_uid": 3, "crps": 300.0}],
    }

    matched = rolling._nearest_historical_miner_scores(
        origin,
        snapshots,
        tolerance=pd.Timedelta(minutes=10),
        config=config,
    )

    assert matched is None


def test_backtest_summary_includes_relevance_statistics():
    rows = [
        {
            "origin": "2026-07-10T01:00:00Z",
            "raw_crps": 100.0,
            "forecast_final_median": 101.0,
            "realized_final": 100.0,
            "origin_session": "outside_market_hours",
            "realized_abs_return_bps": 50.0,
            "realized_vol_5m_bps": 5.0,
            "historical_miner_count": 10,
            "historical_rank": 2,
            "historical_percentile_beaten": 0.9,
            "historical_top10_mean": 120.0,
            "historical_top10_median": 115.0,
            "historical_median_crps": 130.0,
            "gap_vs_historical_mean": -20.0,
            "gap_vs_historical_median": -30.0,
            "beats_historical_top10_mean": True,
            "beats_historical_top10_median": True,
            "beats_historical_median": True,
            "estimated_prompt_score": 10.0,
        },
        {
            "origin": "2026-07-10T14:00:00Z",
            "raw_crps": 200.0,
            "forecast_final_median": 99.0,
            "realized_final": 100.0,
            "origin_session": "us",
            "realized_abs_return_bps": 200.0,
            "realized_vol_5m_bps": 20.0,
            "historical_miner_count": 10,
            "historical_rank": 11,
            "historical_percentile_beaten": 0.0,
            "historical_top10_mean": 150.0,
            "historical_top10_median": 145.0,
            "historical_median_crps": 160.0,
            "gap_vs_historical_mean": 50.0,
            "gap_vs_historical_median": 40.0,
            "beats_historical_top10_mean": False,
            "beats_historical_top10_median": False,
            "beats_historical_median": False,
            "estimated_prompt_score": 80.0,
        },
    ]
    config = apply_asset(load_config("config/default.yaml"), "BTC")
    result = rolling._summarize_backtest(rows, config, sanity_rows=[])

    assert result["summary"]["origin_count"] == 2
    assert result["comparison"]["beat_top10_mean_rate"] == 0.5
    assert result["comparison"]["beat_miner_median_rate"] == 0.5
    assert result["comparison"]["gap_vs_top10_mean_avg"] == 15.0
    assert result["comparison"]["estimated_prompt_score_mean"] == 45.0
    assert {row["group"] for row in result["by_session"]} == {"outside_market_hours", "us"}
    assert result["by_realized_abs_return"]
    assert result["by_realized_volatility"]
