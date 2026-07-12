from __future__ import annotations

import os
import random
import secrets
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synth_shadow.assets import apply_asset
from synth_shadow.backtest import rolling
from synth_shadow.config import load_config


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
        "state_timestamps": [],
        "state_prices": [],
        "forecast_shapes": [],
        "realized_lengths": [],
        "realized_first_prices": [],
    }

    source_bars = rolling._load_backtest_bars(config, audit_days)
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

    monkeypatch.setattr(rolling, "_load_backtest_bars", lambda _config, _days: source_bars)
    monkeypatch.setattr(rolling, "_reference_miners", lambda _config: [])
    monkeypatch.setattr(rolling, "_save_backtest", lambda _rows, _result, _config: tmp_path / "saved")

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
    ) -> list[pd.Timestamp]:
        candidates = original_select_origins(features, patched_config, days, stride, None)
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

    def audited_generate_paths(state, _sampler, patched_config):
        shape = (int(patched_config["forecast"]["num_paths"]), expected_points)
        audit["forecast_shapes"].append(shape)
        paths = np.full(shape, float(state.price))
        return paths, {"state_timestamp": state.timestamp}

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
    monkeypatch.setattr(rolling, "generate_paths", audited_generate_paths)
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

    for origin, library_max, state_price, realized_first in zip(
        scored_origins,
        audit["library_max_timestamps"],
        audit["state_prices"],
        audit["realized_first_prices"],
    ):
        assert library_max == origin
        assert realized_first == state_price

    assert all(shape == (num_paths, expected_points) for shape in audit["forecast_shapes"])
    assert audit["realized_lengths"] == [expected_points] * origins_to_score
    assert result["summary"]["origin_count"] == origins_to_score
    assert result["config"] == {
        "days": audit_days,
        "stride_minutes": stride_minutes,
        "num_paths": num_paths,
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
            "source_resolution_check": f"PASSED: all source/feature bars are {interval_seconds}s apart",
            "origin_stride_check": f"PASSED: {origins_to_score} origins are {stride_minutes} minutes apart",
            "forecast_shape_check": f"PASSED: every forecast is {num_paths} x {expected_points}",
            "realized_length_check": f"PASSED: every realized path has {expected_points} points",
            "realized_origin_alignment": "PASSED: realized first price equals current state price",
            "scored_origin_count": len(scored_origins),
            "scored_origins": [str(origin) for origin in scored_origins],
            "library_rows_per_origin": audit["library_rows_per_origin"],
            "forecast_shapes": audit["forecast_shapes"],
            "origin_details": origin_details,
            "result_config": result["config"],
        },
    )
