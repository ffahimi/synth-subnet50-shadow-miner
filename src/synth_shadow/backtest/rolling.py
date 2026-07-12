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
from synth_shadow.models.current_state import extract_current_state
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.models.session_path_model import build_session_library
from synth_shadow.paths.generator import generate_paths
from synth_shadow.scoring.benchmarks import select_reference_miners
from synth_shadow.scoring.crps import score_synth_btc_24h
from synth_shadow.storage.files import ensure_dir, safe_timestamp
from synth_shadow.synth.client import SynthClient
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
    block_bars = int(run_config["sampling"]["block_minutes"] * 60 / interval_seconds)
    horizon_steps = int(run_config["forecast"]["horizon_seconds"] / interval_seconds)
    points_per_path = horizon_steps + 1

    for idx, origin in enumerate(origins, start=1):
        past_features = features[features["timestamp"] <= origin].copy()
        future = bars[(bars["timestamp"] >= origin)].head(points_per_path)
        if len(future) != points_per_path:
            LOG.warning("Skipping origin=%s, future path incomplete rows=%s", origin, len(future))
            continue
        try:
            library = build_session_library(past_features, block_bars)
            state = extract_current_state(past_features)
            sampler = PathSampler(library, seed=int(run_config["forecast"]["random_seed"]) + idx)
            paths, _ = generate_paths(state, sampler, run_config)
            realized = future["close"].to_numpy(dtype=float)
            score = score_synth_btc_24h(paths, realized)
            row = {
                "origin": str(origin),
                "current_price": state.price,
                "realized_final": float(realized[-1]),
                "forecast_final_median": float(np.median(paths[:, -1])),
                "raw_crps": float(score["raw_crps"]),
                **{name: float(value) for name, value in score["components"].items()},
            }
            rows.append(row)
            if idx == 1 or idx % 12 == 0 or idx == len(origins):
                LOG.debug("Backtest checkpoint %s/%s: %s", idx, len(origins), row)
        except Exception as exc:  # noqa: BLE001 - keep rolling backtest moving.
            LOG.warning("Backtest origin failed origin=%s error=%s", origin, exc)

    if not rows:
        raise RuntimeError("Backtest produced no scored origins.")

    result = _summarize_backtest(rows, run_config)
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


def _summarize_backtest(rows: list[dict[str, Any]], config: dict) -> dict[str, Any]:
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
    return {
        "summary": summary,
        "asset": config["asset"],
        "first_rows": rows[:3],
        "last_rows": rows[-3:],
        "config": {
            "days": config["backtest"]["days"],
            "stride_minutes": config["backtest"]["stride_minutes"],
            "num_paths": config["forecast"]["num_paths"],
        },
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


def _save_backtest(rows: list[dict[str, Any]], result: dict[str, Any], config: dict) -> Path:
    output_dir = ensure_dir(
        Path(config["storage"]["backtest_dir"]) / config["asset"] / safe_timestamp(utc_now())
    )
    pd.DataFrame(rows).to_csv(output_dir / "rolling_results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    LOG.info("Saved backtest outputs to %s", output_dir)
    return output_dir
