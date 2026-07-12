"""Live forecast sanity checks with stage latency reporting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from synth_shadow.data.polygon_client import PolygonClient
from synth_shadow.data.schema import repair_missing_bars
from synth_shadow.features.pipeline import build_feature_frame
from synth_shadow.models.current_state import extract_current_state
from synth_shadow.models.loader import configured_model_entrypoint, load_forecast_model
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.models.protocol import ForecastContext
from synth_shadow.models.session_path_model import build_session_library
from synth_shadow.paths.validator import validate_paths
from synth_shadow.storage.forecast_store import (
    save_forecast_run,
    save_processed_features,
    save_raw_bars,
)
from synth_shadow.storage.registry import ForecastRegistry
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


def run_live_forecast_sanity(config: dict, prompt_start_time: str | None = None) -> dict[str, Any]:
    """Run one live forecast with detailed causality, shape, and latency checks."""
    total_start = time.perf_counter()
    stage_latencies: dict[str, float] = {}
    interval_seconds = int(config["forecast"]["interval_seconds"])
    expected_interval = pd.Timedelta(seconds=interval_seconds)
    expected_points = int(config["forecast"]["horizon_seconds"] / interval_seconds) + 1
    expected_paths = int(config["forecast"]["num_paths"])

    raw_bars = _timed(stage_latencies, "fetch_polygon_bars", lambda: PolygonClient().fetch_recent(config))
    save_raw_bars(raw_bars, config)

    bars = _timed(
        stage_latencies,
        "repair_missing_bars",
        lambda: (
            repair_missing_bars(raw_bars, interval_seconds)
            if config["history"].get("repair_missing_bars", True)
            else raw_bars
        ),
    )
    features = _timed(stage_latencies, "build_features", lambda: build_feature_frame(bars, config))
    save_processed_features(features, config)

    block_bars = int(config["sampling"]["block_minutes"] * 60 / interval_seconds)
    library = _timed(
        stage_latencies,
        "build_session_library",
        lambda: build_session_library(features, block_bars),
    )
    state = _timed(stage_latencies, "extract_current_state", lambda: extract_current_state(features))
    sampler = PathSampler(library, seed=int(config["forecast"]["random_seed"]))
    model = load_forecast_model(config)
    model_entrypoint = configured_model_entrypoint(config)
    output = _timed(
        stage_latencies,
        "generate_paths",
        lambda: model.generate(
            ForecastContext(
                config=config,
                bars=bars,
                features=features,
                library=library,
                state=state,
                sampler=sampler,
            )
        ),
    )
    paths = output.paths
    timestamps = output.timestamps

    _timed(
        stage_latencies,
        "validate_paths",
        lambda: validate_paths(paths, num_paths=expected_paths, points_per_path=expected_points),
    )

    data_checks = _data_checks(
        raw_bars=raw_bars,
        bars=bars,
        features=features,
        state_timestamp=pd.Timestamp(state.timestamp),
        expected_interval=expected_interval,
    )
    path_checks = _path_checks(
        paths=paths,
        timestamps=timestamps,
        state_price=float(state.price),
        state_timestamp=pd.Timestamp(state.timestamp),
        expected_paths=expected_paths,
        expected_points=expected_points,
        expected_interval=expected_interval,
    )

    metadata = {
        "model_version": str(getattr(model, "model_version", model.__class__.__name__)),
        "model_entrypoint": model_entrypoint,
        "asset": config["asset"],
        "polygon_ticker": config["polygon_ticker"],
        "generated_at": utc_now().isoformat(),
        "data_cutoff": state.timestamp,
        "prompt_start_time": prompt_start_time,
        "num_raw_bars": len(raw_bars),
        "num_feature_rows": len(features),
        "num_session_blocks": len(library),
        "path_shape": list(paths.shape),
        "current_price": state.price,
        "debug": bool(config.get("debug", False)),
        "model_metadata": output.metadata,
        "sanity": {
            "latency_seconds": stage_latencies,
            "data_checks": data_checks,
            "path_checks": path_checks,
        },
    }
    forecast_dir = _timed(
        stage_latencies,
        "save_forecast",
        lambda: save_forecast_run(paths, timestamps, metadata, state.to_dict(), config),
    )
    _timed(
        stage_latencies,
        "register_forecast",
        lambda: ForecastRegistry(config["storage"]["registry_path"]).register_forecast(
            str(forecast_dir),
            metadata,
            status="pending" if prompt_start_time else "debug",
        ),
    )
    total_latency = time.perf_counter() - total_start
    stage_latencies["total"] = round(total_latency, 6)
    result = {
        "forecast_dir": str(forecast_dir),
        "metadata": metadata,
        "latency_seconds": stage_latencies,
        "data_checks": data_checks,
        "path_checks": path_checks,
    }
    _print_sanity_report(result)
    LOG.info(
        "Live forecast sanity passed asset=%s forecast_dir=%s total_latency=%.3fs path_shape=%s",
        config["asset"],
        forecast_dir,
        total_latency,
        paths.shape,
    )
    return result


def _timed(stage_latencies: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    value = fn()
    stage_latencies[name] = round(time.perf_counter() - start, 6)
    return value


def _data_checks(
    raw_bars: pd.DataFrame,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    state_timestamp: pd.Timestamp,
    expected_interval: pd.Timedelta,
) -> dict[str, Any]:
    raw_diffs = raw_bars["timestamp"].diff().dropna()
    bar_diffs = bars["timestamp"].diff().dropna()
    feature_diffs = features["timestamp"].diff().dropna()
    checks = {
        "raw_bar_count": int(len(raw_bars)),
        "repaired_bar_count": int(len(bars)),
        "feature_row_count": int(len(features)),
        "raw_first_timestamp": str(raw_bars["timestamp"].min()),
        "raw_last_timestamp": str(raw_bars["timestamp"].max()),
        "feature_first_timestamp": str(features["timestamp"].min()),
        "feature_last_timestamp": str(features["timestamp"].max()),
        "state_timestamp": str(state_timestamp),
        "raw_resolution_ok": bool(raw_diffs.eq(expected_interval).all()),
        "repaired_resolution_ok": bool(bar_diffs.eq(expected_interval).all()),
        "feature_resolution_ok": bool(feature_diffs.eq(expected_interval).all()),
        "raw_monotonic": bool(raw_bars["timestamp"].is_monotonic_increasing),
        "features_monotonic": bool(features["timestamp"].is_monotonic_increasing),
        "features_are_causal_to_state": bool(features["timestamp"].max() == state_timestamp),
        "bars_are_causal_to_state": bool(bars["timestamp"].max() == state_timestamp),
        "expected_resolution_seconds": int(expected_interval.total_seconds()),
        "close_min": float(raw_bars["close"].min()),
        "close_max": float(raw_bars["close"].max()),
        "close_last": float(raw_bars["close"].iloc[-1]),
    }
    _require_checks(checks, ["raw_resolution_ok", "repaired_resolution_ok", "feature_resolution_ok"])
    _require_checks(checks, ["raw_monotonic", "features_monotonic", "features_are_causal_to_state"])
    return checks


def _path_checks(
    paths: np.ndarray,
    timestamps: pd.DatetimeIndex,
    state_price: float,
    state_timestamp: pd.Timestamp,
    expected_paths: int,
    expected_points: int,
    expected_interval: pd.Timedelta,
) -> dict[str, Any]:
    diffs = pd.Series(timestamps).diff().dropna()
    final_prices = paths[:, -1]
    checks = {
        "expected_paths": expected_paths,
        "expected_points": expected_points,
        "actual_shape": list(paths.shape),
        "timestamp_count": int(len(timestamps)),
        "timestamp_first": str(timestamps[0]),
        "timestamp_last": str(timestamps[-1]),
        "timestamp_resolution_ok": bool(diffs.eq(expected_interval).all()),
        "timestamp_starts_at_state": bool(pd.Timestamp(timestamps[0]) == state_timestamp),
        "shape_ok": bool(paths.shape == (expected_paths, expected_points)),
        "finite_ok": bool(np.isfinite(paths).all()),
        "positive_ok": bool((paths > 0).all()),
        "starts_at_current_price": bool(np.allclose(paths[:, 0], state_price)),
        "state_price": float(state_price),
        "path_min": float(paths.min()),
        "path_max": float(paths.max()),
        "final_p05": float(np.percentile(final_prices, 5)),
        "final_p50": float(np.percentile(final_prices, 50)),
        "final_p95": float(np.percentile(final_prices, 95)),
    }
    _require_checks(
        checks,
        [
            "timestamp_resolution_ok",
            "timestamp_starts_at_state",
            "shape_ok",
            "finite_ok",
            "positive_ok",
            "starts_at_current_price",
        ],
    )
    return checks


def _require_checks(checks: dict[str, Any], names: list[str]) -> None:
    failed = [name for name in names if not checks[name]]
    if failed:
        raise ValueError(f"Live forecast sanity checks failed: {failed}")


def _print_sanity_report(result: dict[str, Any]) -> None:
    metadata = result["metadata"]
    print("\n[live forecast sanity]")
    print(f"asset: {metadata['asset']}")
    print(f"forecast_dir: {result['forecast_dir']}")
    print(f"model: {metadata['model_version']} ({metadata['model_entrypoint']})")
    print(f"data_cutoff: {metadata['data_cutoff']}")
    print(f"path_shape: {metadata['path_shape']}")

    print("\n[latency seconds]")
    for stage, seconds in result["latency_seconds"].items():
        print(f"{stage}: {seconds}")

    print("\n[data sanity checks]")
    for key, value in result["data_checks"].items():
        print(f"{key}: {value}")

    print("\n[path sanity checks]")
    for key, value in result["path_checks"].items():
        print(f"{key}: {value}")
