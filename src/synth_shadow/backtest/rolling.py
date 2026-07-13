"""Rolling Polygon-realized historical backtest."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synth_shadow.data.polygon_client import PolygonClient
from synth_shadow.data.schema import repair_missing_bars
from synth_shadow.features.pipeline import build_feature_frame
from synth_shadow.forecasting.protocol import ProviderForecast
from synth_shadow.forecasting.http_provider import configured_endpoint
from synth_shadow.forecasting.loader import load_forecast_provider
from synth_shadow.models.current_state import extract_current_state
from synth_shadow.models.loader import configured_model_entrypoint, load_forecast_model
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.models.protocol import ForecastContext
from synth_shadow.models.session_path_model import build_session_library
from synth_shadow.scoring.benchmarks import select_reference_miners
from synth_shadow.scoring.crps import score_synth_btc_24h
from synth_shadow.scoring.synth_score import top_miner_crps_stats
from synth_shadow.storage.files import ensure_dir, safe_timestamp
from synth_shadow.synth.client import SynthClient
from synth_shadow.utils.logging import GREEN, YELLOW, colored_debug
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


def run_rolling_backtest(
    config: dict,
    days: float | None = None,
    stride_minutes: int | None = None,
    max_origins: int | None = None,
    num_paths: int | None = None,
) -> dict[str, Any]:
    """Run a rolling 24h historical forecast backtest.

    The backtest origins end 24 hours before the latest Polygon bar so every
    origin has a complete realized 24h future path. Each origin uses only bars
    at or before that origin to build features/session blocks.
    """
    backtest_cfg = config["backtest"]
    days = float(days if days is not None else backtest_cfg["days"])
    stride_minutes = int(stride_minutes if stride_minutes is not None else backtest_cfg["stride_minutes"])
    configured_max = backtest_cfg.get("max_origins")
    max_origins = max_origins if max_origins is not None else configured_max
    max_origins = int(max_origins) if max_origins not in (None, "") else None
    num_paths = int(num_paths if num_paths is not None else backtest_cfg["num_paths"])

    run_config = deepcopy(config)
    run_config["backtest"]["days"] = days
    run_config["backtest"]["stride_minutes"] = stride_minutes
    run_config["backtest"]["max_origins"] = max_origins
    run_config["forecast"]["num_paths"] = num_paths

    LOG.info(
        "Starting %s rolling backtest days=%s stride_minutes=%s max_origins=%s num_paths=%s",
        config["asset"],
        days,
        stride_minutes,
        max_origins,
        num_paths,
    )

    bars = _load_backtest_bars(run_config, days)
    interval_seconds = int(run_config["forecast"]["interval_seconds"])
    features = build_feature_frame(bars, run_config)
    origins = _select_origins(features, run_config, days, stride_minutes, max_origins)
    LOG.info(
        "%s backtest data ready bars=%s features=%s origins=%s first_origin=%s last_origin=%s",
        config["asset"],
        len(bars),
        len(features),
        len(origins),
        origins[0] if origins else None,
        origins[-1] if origins else None,
    )

    rows = []
    sanity_rows = []
    block_bars = int(run_config["sampling"]["block_minutes"] * 60 / interval_seconds)
    horizon_steps = int(run_config["forecast"]["horizon_seconds"] / interval_seconds)
    points_per_path = horizon_steps + 1
    endpoint = configured_endpoint(run_config)
    provider = load_forecast_provider(run_config) if endpoint else None
    model = None if provider else load_forecast_model(run_config)
    model_version = "http_provider" if provider else str(getattr(model, "model_version", model.__class__.__name__))
    model_entrypoint = endpoint if provider else configured_model_entrypoint(run_config)
    top10_stats = _latest_top_miner_crps_stats(run_config, count=10)
    if top10_stats["count"]:
        colored_debug(
            LOG,
            "[TOP10 MINERS] asset=%s count=%s mean=%.6f median=%.6f std=%.6f min=%.6f max=%.6f scored_time=%s",
            run_config["asset"],
            top10_stats["count"],
            top10_stats["mean"],
            top10_stats["median"],
            top10_stats["std"],
            top10_stats["min"],
            top10_stats["max"],
            top10_stats["scored_time"],
            color=YELLOW,
        )

    for idx, origin in enumerate(origins, start=1):
        past_bars = bars[bars["timestamp"] <= origin].copy()
        past_features = features[features["timestamp"] <= origin].copy()
        future = bars[(bars["timestamp"] >= origin)].head(points_per_path)
        if len(future) != points_per_path:
            LOG.warning("Skipping origin=%s, future path incomplete rows=%s", origin, len(future))
            continue
        try:
            if provider:
                output = provider.generate(
                    run_config,
                    prompt_start_time=_format_origin(origin),
                    origin=origin,
                )
                _validate_http_backtest_output(output, origin)
                paths = output.paths
                current_price = float(output.metadata["current_price"])
                model_version = str(output.metadata.get("model_version", model_version))
                sanity_rows.append(
                    _http_sanity_row(
                        output=output,
                        origin=origin,
                        expected_paths=num_paths,
                        expected_points=points_per_path,
                    )
                )
            else:
                library = build_session_library(past_features, block_bars)
                state = extract_current_state(past_features)
                sampler = PathSampler(library, seed=int(run_config["forecast"]["random_seed"]) + idx)
                output = model.generate(
                    ForecastContext(
                        config=run_config,
                        bars=past_bars,
                        features=past_features,
                        library=library,
                        state=state,
                        sampler=sampler,
                        origin=origin,
                    )
                )
                paths = output.paths
                current_price = state.price
            realized = future["close"].to_numpy(dtype=float)
            score = score_synth_btc_24h(paths, realized)
            row = {
                "origin": str(origin),
                "current_price": current_price,
                "realized_final": float(realized[-1]),
                "forecast_final_median": float(np.median(paths[:, -1])),
                "raw_crps": float(score["raw_crps"]),
                **{name: float(value) for name, value in score["components"].items()},
            }
            rows.append(row)
            colored_debug(
                LOG,
                (
                    "[BACKTEST CRPS] asset=%s origin=%s raw=%.6f "
                    "5m=%.6f 30m=%.6f 3h=%.6f 24h=%.6f path=%.6f "
                    "top10_mean=%s top10_median=%s top10_std=%s gap_mean=%s gap_median=%s "
                    "http_latency=%s node_latency=%s shape=%s"
                ),
                run_config["asset"],
                origin,
                row["raw_crps"],
                row["crps_5m"],
                row["crps_30m"],
                row["crps_3h"],
                row["crps_24h"],
                row["crps_path_price"],
                _format_float(top10_stats["mean"]),
                _format_float(top10_stats["median"]),
                _format_float(top10_stats["std"]),
                _format_float(_gap(row["raw_crps"], top10_stats["mean"])),
                _format_float(_gap(row["raw_crps"], top10_stats["median"])),
                _format_seconds(output.diagnostics.get("http_latency_seconds"))
                if provider
                else "n/a",
                _format_seconds(_node_total_latency(output.diagnostics))
                if provider
                else "n/a",
                tuple(paths.shape),
                color=GREEN,
            )
            if idx == 1 or idx % 12 == 0 or idx == len(origins):
                LOG.debug("Backtest checkpoint %s/%s: %s", idx, len(origins), row)
        except Exception as exc:  # noqa: BLE001 - keep rolling backtest moving.
            LOG.warning("Backtest origin failed origin=%s error=%s", origin, exc)

    if not rows:
        raise RuntimeError("Backtest produced no scored origins.")

    result = _summarize_backtest(rows, run_config, sanity_rows)
    result["top10_miner_crps_stats"] = top10_stats
    result["model"] = {
        "model_version": model_version,
        "model_entrypoint": model_entrypoint,
    }
    result["reference_miners"] = _reference_miners(run_config)
    result["miner_0_3_crps"] = [
        {
            "index": index,
            "miner_uid": row["miner_uid"],
            "crps": row["crps"],
            "reward": row["reward"],
            "scored_time": row["scored_time"],
        }
        for index, row in enumerate(result["reference_miners"][:4])
    ]
    result["summary"]["reference_miner_count"] = len(result["reference_miners"])
    if result["reference_miners"]:
        reference_crps = [float(row["crps"]) for row in result["reference_miners"]]
        result["summary"]["top_reference_miner_crps"] = min(reference_crps)
        result["summary"]["mean_reference_miner_crps"] = float(np.mean(reference_crps))
        result["summary"]["our_mean_minus_top_reference"] = (
            result["summary"]["raw_crps_mean"] - result["summary"]["top_reference_miner_crps"]
        )
        result["summary"]["our_median_minus_top_reference"] = (
            result["summary"]["raw_crps_median"] - result["summary"]["top_reference_miner_crps"]
        )
    output_dir = _save_backtest(rows, result, run_config)
    result["output_dir"] = str(output_dir)
    LOG.info("Completed rolling backtest: %s", result["summary"])
    LOG.debug("Backtest reference miners: %s", result["reference_miners"])
    return result


def _load_backtest_bars(config: dict, days: float) -> pd.DataFrame:
    now = pd.Timestamp(utc_now())
    horizon = timedelta(seconds=int(config["forecast"]["horizon_seconds"]))
    lookback = timedelta(days=float(config["history"]["lookback_days"]))
    backtest = timedelta(days=days)
    start = now - horizon - lookback - backtest - timedelta(hours=1)
    end = now
    client = PolygonClient()
    raw = client.fetch_aggregates(
        ticker=config["polygon_ticker"],
        multiplier=int(config["history"]["bar_multiplier"]),
        timespan=config["history"]["bar_timespan"],
        start=start,
        end=end,
        adjusted=bool(config["history"].get("adjusted", True)),
    )
    return repair_missing_bars(raw, int(config["forecast"]["interval_seconds"]))


def _select_origins(
    features: pd.DataFrame,
    config: dict,
    days: float,
    stride_minutes: int,
    max_origins: int | None,
) -> list[pd.Timestamp]:
    horizon = pd.Timedelta(seconds=int(config["forecast"]["horizon_seconds"]))
    latest_matured_origin = features["timestamp"].max() - horizon
    first_origin = latest_matured_origin - pd.Timedelta(days=days)
    candidates = features[
        (features["timestamp"] >= first_origin)
        & (features["timestamp"] <= latest_matured_origin)
    ]["timestamp"].tolist()
    stride_bars = max(1, int(stride_minutes / (int(config["forecast"]["interval_seconds"]) / 60)))
    origins = candidates[::stride_bars]
    if max_origins is not None:
        origins = origins[-max_origins:]
    return [pd.Timestamp(origin) for origin in origins]


def _validate_http_backtest_output(output: ProviderForecast, origin: pd.Timestamp) -> None:
    origin = _utc_timestamp(origin)
    first_timestamp = _utc_timestamp(output.timestamps[0])
    if first_timestamp != origin:
        raise ValueError(
            f"HTTP backtest forecast first timestamp {first_timestamp} does not match origin {origin}."
        )
    data_cutoff = _utc_timestamp(output.metadata["data_cutoff"])
    if data_cutoff > origin:
        raise ValueError(f"HTTP backtest data_cutoff {data_cutoff} is after origin {origin}.")


def _http_sanity_row(
    output: ProviderForecast,
    origin: pd.Timestamp,
    expected_paths: int,
    expected_points: int,
) -> dict[str, Any]:
    diagnostics = output.diagnostics or {}
    data_cutoff = _utc_timestamp(output.metadata["data_cutoff"])
    first_timestamp = _utc_timestamp(output.timestamps[0])
    shape = tuple(int(x) for x in output.paths.shape)
    latency = diagnostics.get("http_latency_seconds")
    node_latency = diagnostics.get("latency_seconds") or {}
    total_node_latency = node_latency.get("total") if isinstance(node_latency, dict) else None
    return {
        "origin": str(_utc_timestamp(origin)),
        "data_cutoff": str(data_cutoff),
        "first_timestamp": str(first_timestamp),
        "past_only": bool(data_cutoff <= _utc_timestamp(origin)),
        "first_timestamp_matches_origin": bool(first_timestamp == _utc_timestamp(origin)),
        "path_shape": shape,
        "path_shape_ok": bool(shape == (expected_paths, expected_points)),
        "finite_paths": bool(np.isfinite(output.paths).all()),
        "positive_paths": bool((output.paths > 0).all()),
        "http_latency_seconds": float(latency) if latency is not None else None,
        "node_total_latency_seconds": float(total_node_latency) if total_node_latency is not None else None,
        "data_source": diagnostics.get("data_source"),
        "num_raw_bars": diagnostics.get("num_raw_bars"),
        "num_feature_rows": diagnostics.get("num_feature_rows"),
        "feature_rows_read": diagnostics.get("feature_rows_read"),
        "nearest_neighbors": diagnostics.get("nearest_neighbors"),
    }


def _node_total_latency(diagnostics: dict[str, Any]) -> float | None:
    node_latency = diagnostics.get("latency_seconds") or {}
    if not isinstance(node_latency, dict):
        return None
    value = node_latency.get("total")
    return float(value) if value is not None else None


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _gap(ours: float, miner_stat: Any) -> float | None:
    if miner_stat is None:
        return None
    return float(ours) - float(miner_stat)


def _format_origin(origin: pd.Timestamp) -> str:
    return _utc_timestamp(origin).isoformat()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _summarize_backtest(
    rows: list[dict[str, Any]],
    config: dict,
    sanity_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = np.array([row["raw_crps"] for row in rows], dtype=float)
    final_error = np.array(
        [row["forecast_final_median"] - row["realized_final"] for row in rows],
        dtype=float,
    )
    summary = {
        "origin_count": len(rows),
        "raw_crps_mean": float(np.mean(raw)),
        "raw_crps_median": float(np.median(raw)),
        "raw_crps_p25": float(np.percentile(raw, 25)),
        "raw_crps_p75": float(np.percentile(raw, 75)),
        "final_error_mean": float(np.mean(final_error)),
        "final_abs_error_median": float(np.median(np.abs(final_error))),
    }
    sanity_summary = _summarize_sanity_rows(sanity_rows or [])
    return {
        "summary": summary,
        "sanity": sanity_summary,
        "asset": config["asset"],
        "first_rows": rows[:3],
        "last_rows": rows[-3:],
        "config": {
            "days": config["backtest"]["days"],
            "stride_minutes": config["backtest"]["stride_minutes"],
            "num_paths": config["forecast"]["num_paths"],
        },
    }


def _summarize_sanity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"enabled": False}
    http_latencies = [
        float(row["http_latency_seconds"])
        for row in rows
        if row.get("http_latency_seconds") is not None
    ]
    node_latencies = [
        float(row["node_total_latency_seconds"])
        for row in rows
        if row.get("node_total_latency_seconds") is not None
    ]
    return {
        "enabled": True,
        "checked_origins": len(rows),
        "past_only_passed": bool(all(row["past_only"] for row in rows)),
        "first_timestamp_alignment_passed": bool(
            all(row["first_timestamp_matches_origin"] for row in rows)
        ),
        "path_shape_passed": bool(all(row["path_shape_ok"] for row in rows)),
        "finite_paths_passed": bool(all(row["finite_paths"] for row in rows)),
        "positive_paths_passed": bool(all(row["positive_paths"] for row in rows)),
        "path_shapes": sorted({str(row["path_shape"]) for row in rows}),
        "first_origin": rows[0]["origin"],
        "last_origin": rows[-1]["origin"],
        "data_sources": sorted(
            {str(row["data_source"]) for row in rows if row.get("data_source")}
        ),
        "http_latency_seconds": _latency_summary(http_latencies),
        "node_total_latency_seconds": _latency_summary(node_latencies),
        "first_checks": rows[:3],
        "last_checks": rows[-3:],
    }


def _latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _reference_miners(config: dict) -> list[dict[str, Any]]:
    try:
        client = SynthClient(config)
        scores = client.latest_scores()
        leaderboard = client.latest_leaderboard()
        return select_reference_miners(
            scores,
            leaderboard,
            count=int(config["backtest"]["compare_miners"]),
        )
    except Exception as exc:  # noqa: BLE001 - backtest should still be useful without Synth.
        LOG.warning("Could not fetch reference miners for backtest: %s", exc)
        return []


def _latest_top_miner_crps_stats(config: dict, count: int = 10) -> dict[str, Any]:
    try:
        return top_miner_crps_stats(SynthClient(config).latest_scores(), count=count)
    except Exception as exc:  # noqa: BLE001 - backtest should continue without comparison stats.
        LOG.warning("Could not fetch top miner CRPS stats for backtest debug lines: %s", exc)
        return top_miner_crps_stats([], count=count)


def _save_backtest(rows: list[dict[str, Any]], result: dict[str, Any], config: dict) -> Path:
    output_dir = ensure_dir(
        Path(config["storage"]["backtest_dir"]) / config["asset"] / safe_timestamp(utc_now())
    )
    pd.DataFrame(rows).to_csv(output_dir / "rolling_results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    LOG.info("Saved backtest outputs to %s", output_dir)
    return output_dir
