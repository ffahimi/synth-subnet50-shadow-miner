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
import requests

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
from synth_shadow.scoring.crps import score_synth_btc_24h
from synth_shadow.scoring.synth_score import (
    rank_against_miners,
    top_miner_crps_stats,
    valid_miner_crps_values,
)
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
    realized_source: str | None = None,
    origin_source: str | None = None,
    maturity_lag_minutes: float | None = None,
    checkpoint_every: int | None = None,
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
    if realized_source is not None:
        run_config["backtest"]["realized_source"] = realized_source
    if origin_source is not None:
        run_config["backtest"]["origin_source"] = origin_source
    if maturity_lag_minutes is not None:
        run_config["backtest"]["maturity_lag_minutes"] = maturity_lag_minutes
    if checkpoint_every is not None:
        run_config["backtest"]["checkpoint_every"] = checkpoint_every
    realized_source = str(run_config["backtest"].get("realized_source", "polygon")).lower()
    if realized_source not in {"polygon", "synth"}:
        raise ValueError("backtest.realized_source must be 'polygon' or 'synth'.")
    origin_source = str(run_config["backtest"].get("origin_source", "polygon")).lower()
    if origin_source not in {"polygon", "synth"}:
        raise ValueError("backtest.origin_source must be 'polygon' or 'synth'.")

    LOG.info(
        (
            "Starting %s rolling backtest days=%s stride_minutes=%s max_origins=%s "
            "num_paths=%s realized_source=%s origin_source=%s"
        ),
        config["asset"],
        days,
        stride_minutes,
        max_origins,
        num_paths,
        realized_source,
        origin_source,
    )

    bars = _load_backtest_bars(run_config, days)
    interval_seconds = int(run_config["forecast"]["interval_seconds"])
    features = build_feature_frame(bars, run_config)
    selection_max_origins = _selection_max_origins(max_origins, origin_source, realized_source, run_config)
    score_snapshot_cache: dict[str, Any] = {}
    origins = _select_origins(
        features,
        run_config,
        days,
        stride_minutes,
        selection_max_origins,
        origin_source,
        realized_source,
        score_snapshot_cache,
    )
    if max_origins is not None and origin_source == "synth" and realized_source == "synth":
        origins = list(reversed(origins))
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
    historical_scores = _historical_miner_scores_for_origins(
        run_config,
        origins,
        stride_minutes,
        cached_snapshots=score_snapshot_cache.get("snapshots"),
    )
    if historical_scores:
        first_snapshot = _nearest_historical_miner_scores(
            origins[0],
            historical_scores,
            tolerance=_historical_score_tolerance(run_config, stride_minutes),
            config=run_config,
        )
        first_top10 = top_miner_crps_stats(first_snapshot["scores"], count=10) if first_snapshot else None
    else:
        first_snapshot = None
        first_top10 = None
    if first_top10 and first_top10["count"]:
        colored_debug(
            LOG,
            (
                "[HISTORICAL TOP10 MINERS] asset=%s first_origin=%s matched_scored_time=%s "
                "target_scored_time=%s delta_minutes=%.2f count=%s mean=%.6f median=%.6f std=%.6f min=%.6f max=%.6f"
            ),
            run_config["asset"],
            origins[0],
            first_snapshot["scored_time"],
            first_snapshot["target_scored_time"],
            first_snapshot["delta_minutes"],
            first_top10["count"],
            first_top10["mean"],
            first_top10["median"],
            first_top10["std"],
            first_top10["min"],
            first_top10["max"],
            color=YELLOW,
        )

    output_dir = _new_backtest_output_dir(run_config)
    LOG.info("Backtest outputs will be written to %s", output_dir)
    checkpoint_every = _checkpoint_every(run_config)

    for idx, origin in enumerate(origins, start=1):
        past_bars = bars[bars["timestamp"] <= origin].copy()
        past_features = features[features["timestamp"] <= origin].copy()
        future = bars[(bars["timestamp"] >= origin)].head(points_per_path)
        if realized_source == "polygon" and len(future) != points_per_path:
            LOG.warning("Skipping origin=%s, future path incomplete rows=%s", origin, len(future))
            continue
        try:
            realized = _load_realized_path_for_origin(
                run_config,
                origin,
                future,
                points_per_path,
                realized_source,
            )
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
            score = score_synth_btc_24h(paths, realized)
            row = {
                "origin": str(origin),
                "current_price": current_price,
                "realized_final": float(realized[-1]),
                "realized_source": realized_source,
                "forecast_final_median": float(np.median(paths[:, -1])),
                "raw_crps": float(score["raw_crps"]),
                **{name: float(value) for name, value in score["components"].items()},
            }
            if provider:
                row.update(
                    {
                        "http_latency_seconds": output.diagnostics.get("http_latency_seconds"),
                        "node_latency_seconds": _node_total_latency(output.diagnostics),
                    }
                )
            historical_snapshot = _nearest_historical_miner_scores(
                origin,
                historical_scores,
                tolerance=_historical_score_tolerance(run_config, stride_minutes),
                config=run_config,
            )
            miner_scores_at_origin = historical_snapshot["scores"] if historical_snapshot else []
            top10_stats = top_miner_crps_stats(miner_scores_at_origin, count=10)
            historical_rank = rank_against_miners(row["raw_crps"], miner_scores_at_origin)
            row.update(_origin_diagnostics(origin, past_features, realized, miner_scores_at_origin, top10_stats))
            row.update(
                {
                    "historical_rank": historical_rank["rank"],
                    "historical_miner_count": historical_rank["miner_count"],
                    "historical_miners_beaten": historical_rank["miners_beaten"],
                    "historical_percentile_beaten": historical_rank["percentile_beaten"],
                    "target_scored_time": historical_snapshot["target_scored_time"]
                    if historical_snapshot
                    else str(_score_match_time(origin, run_config)),
                    "matched_scored_time": historical_snapshot["scored_time"] if historical_snapshot else None,
                    "score_time_delta_min": historical_snapshot["delta_minutes"] if historical_snapshot else None,
                    "historical_top10_mean": top10_stats["mean"],
                    "historical_top10_median": top10_stats["median"],
                    "historical_top10_std": top10_stats["std"],
                    "gap_vs_historical_mean": _gap(row["raw_crps"], top10_stats["mean"]),
                    "gap_vs_historical_median": _gap(row["raw_crps"], top10_stats["median"]),
                }
            )
            _finalize_origin_diagnostics(row)
            rows.append(row)
            colored_debug(
                LOG,
                (
                    "[BACKTEST CRPS] asset=%s origin=%s raw=%.6f "
                    "5m=%.6f 30m=%.6f 3h=%.6f 24h=%.6f path=%.6f "
                    "historical_rank=%s/%s historical_miners_beaten=%s "
                    "historical_percentile_beaten=%s target_scored_time=%s matched_scored_time=%s score_time_delta_min=%s "
                    "historical_top10_mean=%s historical_top10_median=%s historical_top10_std=%s "
                    "gap_vs_historical_mean=%s gap_vs_historical_median=%s "
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
                _format_rank(historical_rank["rank"]),
                historical_rank["miner_count"],
                _format_rank(historical_rank["miners_beaten"]),
                _format_percent(historical_rank["percentile_beaten"]),
                historical_snapshot["target_scored_time"] if historical_snapshot else str(_score_match_time(origin, run_config)),
                historical_snapshot["scored_time"] if historical_snapshot else "n/a",
                _format_float(historical_snapshot["delta_minutes"] if historical_snapshot else None),
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
            if checkpoint_every and len(rows) % checkpoint_every == 0:
                checkpoint_result = _build_backtest_result(
                    rows,
                    run_config,
                    sanity_rows,
                    historical_scores,
                    model_version,
                    model_entrypoint,
                    output_dir,
                )
                _write_backtest_outputs(output_dir, rows, checkpoint_result)
                LOG.info(
                    "Checkpointed rolling backtest scored_origins=%s output_dir=%s",
                    len(rows),
                    output_dir,
                )
            if max_origins is not None and len(rows) >= max_origins:
                LOG.info("Reached requested scored origin limit max_origins=%s", max_origins)
                break
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if realized_source == "synth" and status_code == 404:
                LOG.warning(
                    "Skipping origin=%s because Synth realized path is not available yet.",
                    origin,
                )
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - keep rolling backtest moving.
            LOG.warning("Backtest origin failed origin=%s error=%s", origin, exc)

    if not rows:
        raise RuntimeError("Backtest produced no scored origins.")

    result = _build_backtest_result(
        rows,
        run_config,
        sanity_rows,
        historical_scores,
        model_version,
        model_entrypoint,
        output_dir,
    )
    _write_backtest_outputs(output_dir, rows, result)
    LOG.info("Completed rolling backtest: %s", result["summary"])
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
    origin_source: str = "polygon",
    realized_source: str = "polygon",
    score_snapshot_cache: dict[str, Any] | None = None,
) -> list[pd.Timestamp]:
    horizon = pd.Timedelta(seconds=int(config["forecast"]["horizon_seconds"]))
    latest_matured_origin = features["timestamp"].max() - horizon
    if origin_source == "synth":
        latest_matured_origin -= _maturity_lag(config)
    first_origin = latest_matured_origin - pd.Timedelta(days=days)
    if origin_source == "synth" and realized_source == "synth":
        origins = _select_synth_score_origins(
            features,
            config,
            first_origin,
            latest_matured_origin,
            score_snapshot_cache,
        )
    elif origin_source == "synth":
        origins = _select_synth_prompt_origins(
            features,
            config,
            first_origin,
            latest_matured_origin,
        )
    else:
        candidates = features[
            (features["timestamp"] >= first_origin)
            & (features["timestamp"] <= latest_matured_origin)
        ]["timestamp"].tolist()
        stride_bars = max(1, int(stride_minutes / (int(config["forecast"]["interval_seconds"]) / 60)))
        origins = candidates[::stride_bars]
    if max_origins is not None:
        origins = origins[-max_origins:]
    return [pd.Timestamp(origin) for origin in origins]


def _select_synth_score_origins(
    features: pd.DataFrame,
    config: dict,
    first_origin: pd.Timestamp,
    latest_matured_origin: pd.Timestamp,
    score_snapshot_cache: dict[str, Any] | None = None,
) -> list[pd.Timestamp]:
    horizon = pd.Timedelta(seconds=int(config["forecast"]["horizon_seconds"]))
    tolerance = _historical_score_tolerance(config, int(config["backtest"]["stride_minutes"]))
    start = first_origin + horizon - tolerance
    end = latest_matured_origin + horizon + tolerance
    rows = _fetch_historical_score_rows(config, start, end, context="origin selection")
    snapshots = _group_score_rows(rows)
    if score_snapshot_cache is not None:
        score_snapshot_cache["snapshots"] = snapshots
        score_snapshot_cache["rows"] = len(rows)
    earliest_feature = features["timestamp"].min()
    origins_by_time: dict[pd.Timestamp, pd.Timestamp] = {}
    for scored_time in snapshots:
        origin = scored_time - horizon
        if first_origin <= origin <= latest_matured_origin and earliest_feature <= origin:
            origins_by_time[origin] = origin
    origins = sorted(origins_by_time)
    LOG.info(
        (
            "Selected Synth score origins score_rows=%s usable_snapshots=%s "
            "first_origin=%s latest_matured_origin=%s"
        ),
        len(rows),
        len(origins),
        first_origin,
        latest_matured_origin,
    )
    return origins


def _select_synth_prompt_origins(
    features: pd.DataFrame,
    config: dict,
    first_origin: pd.Timestamp,
    latest_matured_origin: pd.Timestamp,
) -> list[pd.Timestamp]:
    prompts = SynthClient(config).prompts(start=first_origin, end=latest_matured_origin)
    earliest_feature = features["timestamp"].min()
    origins = []
    for prompt in prompts:
        ts = _utc_timestamp(prompt)
        if first_origin <= ts <= latest_matured_origin and earliest_feature <= ts:
            origins.append(ts)
    LOG.info(
        "Selected Synth prompt origins prompts=%s usable=%s first_origin=%s latest_matured_origin=%s",
        len(prompts),
        len(origins),
        first_origin,
        latest_matured_origin,
    )
    return origins


def _load_realized_path_for_origin(
    config: dict,
    origin: pd.Timestamp,
    polygon_future: pd.DataFrame,
    points_per_path: int,
    realized_source: str,
) -> np.ndarray:
    if realized_source == "polygon":
        return polygon_future["close"].to_numpy(dtype=float)
    payload = SynthClient(config).realized_path(_format_synth_time(origin))
    realized = np.asarray(payload.get("real_prices", []), dtype=float)
    if realized.shape[0] != points_per_path:
        raise ValueError(
            f"Synth realized path origin={origin} length {realized.shape[0]} "
            f"does not match expected {points_per_path}."
        )
    return realized


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


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.2f}%"


def _format_rank(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _gap(ours: float, miner_stat: Any) -> float | None:
    if miner_stat is None:
        return None
    return float(ours) - float(miner_stat)


def _format_origin(origin: pd.Timestamp) -> str:
    return _utc_timestamp(origin).isoformat()


def _format_synth_time(origin: pd.Timestamp) -> str:
    return _utc_timestamp(origin).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        "comparison": _summarize_comparison(rows),
        "by_session": _group_summary(rows, "origin_session"),
        "by_realized_abs_return": _quantile_group_summary(
            rows,
            "realized_abs_return_bps",
            "realized_abs_return_bucket",
        ),
        "by_realized_volatility": _quantile_group_summary(
            rows,
            "realized_vol_5m_bps",
            "realized_vol_bucket",
        ),
        "sanity": sanity_summary,
        "asset": config["asset"],
        "first_rows": rows[:3],
        "last_rows": rows[-3:],
        "config": {
            "days": config["backtest"]["days"],
            "stride_minutes": config["backtest"]["stride_minutes"],
            "num_paths": config["forecast"]["num_paths"],
            "maturity_lag_minutes": config["backtest"].get("maturity_lag_minutes", 0),
            "checkpoint_every": config["backtest"].get("checkpoint_every", 0),
            "origin_source": config["backtest"].get("origin_source", "polygon"),
            "realized_source": config["backtest"].get("realized_source", "polygon"),
        },
    }


def _origin_diagnostics(
    origin: pd.Timestamp,
    past_features: pd.DataFrame,
    realized: np.ndarray,
    miner_scores: list[dict[str, Any]],
    top10_stats: dict[str, Any],
) -> dict[str, Any]:
    origin = _utc_timestamp(origin)
    miner_values = valid_miner_crps_values(miner_scores)
    realized_return_bps = ((float(realized[-1]) - float(realized[0])) / float(realized[0])) * 10000.0
    realized_step_returns = np.diff(realized) / realized[:-1] * 10000.0
    session = None
    if not past_features.empty and "session" in past_features.columns:
        session = str(past_features.iloc[-1]["session"])
    row = {
        "origin_hour_utc": int(origin.hour),
        "origin_weekday": int(origin.weekday()),
        "origin_session": session,
        "realized_return_bps": float(realized_return_bps),
        "realized_abs_return_bps": float(abs(realized_return_bps)),
        "realized_vol_5m_bps": float(np.std(realized_step_returns, ddof=0)),
        "forecast_final_error_bps": None,
        "historical_best_crps": None,
        "historical_median_crps": None,
        "historical_p25_crps": None,
        "historical_p90_crps": None,
        "gap_vs_historical_best": None,
        "gap_vs_historical_median": None,
        "beats_historical_median": None,
        "beats_historical_top10_mean": None,
        "beats_historical_top10_median": None,
        "estimated_prompt_score": None,
    }
    if miner_values.size:
        row.update(
            {
                "historical_best_crps": float(np.min(miner_values)),
                "historical_median_crps": float(np.median(miner_values)),
                "historical_p25_crps": float(np.percentile(miner_values, 25)),
                "historical_p90_crps": float(np.percentile(miner_values, 90)),
            }
        )
    return row


def _finalize_origin_diagnostics(row: dict[str, Any]) -> None:
    if row.get("realized_final") not in (None, 0) and row.get("forecast_final_median") is not None:
        row["forecast_final_error_bps"] = (
            (float(row["forecast_final_median"]) - float(row["realized_final"]))
            / float(row["realized_final"])
            * 10000.0
        )
    if row.get("historical_best_crps") is not None:
        row["gap_vs_historical_best"] = float(row["raw_crps"]) - float(row["historical_best_crps"])
        p90 = row.get("historical_p90_crps")
        capped_ours = min(float(row["raw_crps"]), float(p90)) if p90 is not None else float(row["raw_crps"])
        row["estimated_prompt_score"] = capped_ours - float(row["historical_best_crps"])
    if row.get("historical_median_crps") is not None:
        row["gap_vs_historical_median"] = float(row["raw_crps"]) - float(row["historical_median_crps"])
        row["beats_historical_median"] = bool(float(row["raw_crps"]) < float(row["historical_median_crps"]))
    if row.get("historical_top10_mean") is not None:
        row["beats_historical_top10_mean"] = bool(float(row["raw_crps"]) < float(row["historical_top10_mean"]))
    if row.get("historical_top10_median") is not None:
        row["beats_historical_top10_median"] = bool(float(row["raw_crps"]) < float(row["historical_top10_median"]))


def _summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    return {
        "scored_origins": int(len(df)),
        "matched_miner_origins": int(df["historical_miner_count"].fillna(0).gt(0).sum())
        if "historical_miner_count" in df
        else 0,
        "raw_crps_mean": _series_mean(df, "raw_crps"),
        "raw_crps_median": _series_median(df, "raw_crps"),
        "gap_vs_top10_mean_avg": _series_mean(df, "gap_vs_historical_mean"),
        "gap_vs_miner_median_avg": _series_mean(df, "gap_vs_historical_median"),
        "estimated_prompt_score_mean": _series_mean(df, "estimated_prompt_score"),
        "percentile_beaten_mean": _series_mean(df, "historical_percentile_beaten"),
        "beat_top10_mean_rate": _bool_rate(df, "beats_historical_top10_mean"),
        "beat_top10_median_rate": _bool_rate(df, "beats_historical_top10_median"),
        "beat_miner_median_rate": _bool_rate(df, "beats_historical_median"),
        "median_rank": _series_median(df, "historical_rank"),
        "best_rank": _series_min(df, "historical_rank"),
        "worst_rank": _series_max(df, "historical_rank"),
        "mean_http_latency_seconds": _series_mean(df, "http_latency_seconds"),
        "mean_node_latency_seconds": _series_mean(df, "node_latency_seconds"),
    }


def _group_summary(rows: list[dict[str, Any]], group_col: str) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty or group_col not in df:
        return []
    groups = []
    for value, group in df.dropna(subset=[group_col]).groupby(group_col):
        groups.append(_summary_for_group(str(value), group))
    return sorted(groups, key=lambda row: row["count"], reverse=True)


def _quantile_group_summary(rows: list[dict[str, Any]], value_col: str, label_col: str) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty or value_col not in df:
        return []
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid = df[values.notna()].copy()
    if valid.empty:
        return []
    try:
        valid[label_col] = pd.qcut(
            pd.to_numeric(valid[value_col], errors="coerce"),
            q=min(3, len(valid)),
            labels=["low", "mid", "high"][: min(3, len(valid))],
            duplicates="drop",
        )
    except ValueError:
        valid[label_col] = "all"
    return _group_summary(valid.to_dict("records"), label_col)


def _summary_for_group(name: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "group": name,
        "count": int(len(group)),
        "raw_crps_mean": _series_mean(group, "raw_crps"),
        "raw_crps_median": _series_median(group, "raw_crps"),
        "gap_vs_top10_mean_avg": _series_mean(group, "gap_vs_historical_mean"),
        "gap_vs_miner_median_avg": _series_mean(group, "gap_vs_historical_median"),
        "percentile_beaten_mean": _series_mean(group, "historical_percentile_beaten"),
        "beat_top10_mean_rate": _bool_rate(group, "beats_historical_top10_mean"),
        "beat_miner_median_rate": _bool_rate(group, "beats_historical_median"),
        "realized_abs_return_bps_mean": _series_mean(group, "realized_abs_return_bps"),
        "realized_vol_5m_bps_mean": _series_mean(group, "realized_vol_5m_bps"),
    }


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _series_mean(df: pd.DataFrame, col: str) -> float | None:
    values = _numeric_series(df, col)
    return float(values.mean()) if not values.empty else None


def _series_median(df: pd.DataFrame, col: str) -> float | None:
    values = _numeric_series(df, col)
    return float(values.median()) if not values.empty else None


def _series_min(df: pd.DataFrame, col: str) -> float | None:
    values = _numeric_series(df, col)
    return float(values.min()) if not values.empty else None


def _series_max(df: pd.DataFrame, col: str) -> float | None:
    values = _numeric_series(df, col)
    return float(values.max()) if not values.empty else None


def _bool_rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df:
        return None
    values = df[col].dropna()
    return float(values.astype(bool).mean()) if not values.empty else None


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


def _historical_miner_scores_for_origins(
    config: dict,
    origins: list[pd.Timestamp],
    stride_minutes: int,
    cached_snapshots: dict[pd.Timestamp, list[dict[str, Any]]] | None = None,
) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    if not origins:
        return {}
    tolerance = _historical_score_tolerance(config, stride_minutes)
    scored_times = [_score_match_time(origin, config) for origin in origins]
    start = min(scored_times) - tolerance
    end = max(scored_times) + tolerance
    if cached_snapshots is not None:
        grouped = {
            ts: scores
            for ts, scores in cached_snapshots.items()
            if start <= ts <= end
        }
        LOG.info(
            (
                "Reusing cached historical miner score snapshots asset=%s "
                "snapshots=%s rows=%s first=%s last=%s"
            ),
            config["asset"],
            len(grouped),
            sum(len(scores) for scores in grouped.values()),
            min(grouped) if grouped else None,
            max(grouped) if grouped else None,
        )
        return grouped
    rows = _fetch_historical_score_rows(config, start, end, context="miner comparison")
    grouped = _group_score_rows(rows)
    LOG.info(
        "Historical miner score snapshots ready asset=%s snapshots=%s rows=%s first=%s last=%s",
        config["asset"],
        len(grouped),
        len(rows),
        min(grouped) if grouped else None,
        max(grouped) if grouped else None,
    )
    return grouped


def _group_score_rows(rows: list[dict[str, Any]]) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    grouped: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for row in rows:
        ts = _score_time(row)
        if ts is None:
            continue
        grouped.setdefault(ts, []).append(row)
    return grouped


def _fetch_historical_score_rows(
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    context: str,
) -> list[dict[str, Any]]:
    chunk_hours = float(config.get("backtest", {}).get("historical_score_chunk_hours", 24))
    client = SynthClient(config, timeout_seconds=int(config.get("synth", {}).get("timeout_seconds", 90)))
    rows: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(hours=chunk_hours), end)
        try:
            chunk = client.historical_scores(start=cursor, end=chunk_end)
            rows.extend(chunk)
            LOG.debug(
                "Fetched historical miner scores context=%s chunk_start=%s chunk_end=%s rows=%s accumulated=%s",
                context,
                cursor,
                chunk_end,
                len(chunk),
                len(rows),
            )
        except Exception as exc:  # noqa: BLE001 - long backtests can tolerate missing score chunks.
            LOG.warning(
                "Could not fetch historical miner scores context=%s chunk_start=%s chunk_end=%s: %s",
                context,
                cursor,
                chunk_end,
                exc,
            )
        cursor = chunk_end
    return rows


def _score_time(row: dict[str, Any]) -> pd.Timestamp | None:
    value = row.get("scored_time") or row.get("score_time") or row.get("timestamp")
    if not value:
        return None
    return _utc_timestamp(value).floor("s")


def _nearest_historical_miner_scores(
    origin: pd.Timestamp,
    snapshots: dict[pd.Timestamp, list[dict[str, Any]]],
    tolerance: pd.Timedelta,
    config: dict | None = None,
) -> dict[str, Any] | None:
    if not snapshots:
        return None
    target = _score_match_time(origin, config) if config else _utc_timestamp(origin)
    nearest = min(snapshots, key=lambda ts: abs(ts - target))
    delta = abs(nearest - target)
    if delta > tolerance:
        return None
    return {
        "target_scored_time": str(target),
        "scored_time": str(nearest),
        "delta_minutes": float(delta.total_seconds() / 60),
        "scores": snapshots[nearest],
    }


def _score_match_time(origin: pd.Timestamp, config: dict | None = None) -> pd.Timestamp:
    horizon_seconds = int((config or {}).get("forecast", {}).get("horizon_seconds", 86400))
    return _utc_timestamp(origin) + pd.Timedelta(seconds=horizon_seconds)


def _historical_score_tolerance(config: dict, stride_minutes: int) -> pd.Timedelta:
    configured = config.get("backtest", {}).get("historical_score_tolerance_minutes")
    minutes = float(configured) if configured not in (None, "") else max(float(stride_minutes), 30.0)
    return pd.Timedelta(minutes=minutes)


def _maturity_lag(config: dict) -> pd.Timedelta:
    configured = config.get("backtest", {}).get("maturity_lag_minutes", 0)
    minutes = float(configured) if configured not in (None, "") else 0.0
    return pd.Timedelta(minutes=minutes)


def _selection_max_origins(
    max_origins: int | None,
    origin_source: str,
    realized_source: str,
    config: dict,
) -> int | None:
    if max_origins is None or origin_source != "synth" or realized_source != "synth":
        return max_origins
    multiplier = int(config.get("backtest", {}).get("synth_realized_scan_multiplier", 24))
    return max(max_origins, max_origins * max(1, multiplier))


def _checkpoint_every(config: dict) -> int:
    configured = config.get("backtest", {}).get("checkpoint_every", 0)
    return int(configured) if configured not in (None, "") else 0


def _build_backtest_result(
    rows: list[dict[str, Any]],
    config: dict,
    sanity_rows: list[dict[str, Any]],
    historical_scores: dict[pd.Timestamp, list[dict[str, Any]]],
    model_version: str,
    model_entrypoint: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    result = _summarize_backtest(rows, config, sanity_rows)
    result["historical_miner_snapshot"] = {
        "score_snapshot_count": len(historical_scores),
        "comparison_note": "per-origin nearest historical Synth score snapshots",
    }
    result["model"] = {
        "model_version": model_version,
        "model_entrypoint": model_entrypoint,
    }
    result["output_dir"] = str(output_dir)
    return result


def _new_backtest_output_dir(config: dict) -> Path:
    return ensure_dir(
        Path(config["storage"]["backtest_dir"]) / config["asset"] / safe_timestamp(utc_now())
    )


def _write_backtest_outputs(output_dir: Path, rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "rolling_results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def _save_backtest(rows: list[dict[str, Any]], result: dict[str, Any], config: dict) -> Path:
    output_dir = _new_backtest_output_dir(config)
    _write_backtest_outputs(output_dir, rows, result)
    LOG.info("Saved backtest outputs to %s", output_dir)
    return output_dir
