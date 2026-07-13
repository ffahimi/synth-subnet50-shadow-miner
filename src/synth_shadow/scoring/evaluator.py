"""Forecast evaluation against Synth realized paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from synth_shadow.scoring.crps import score_synth_btc_24h
from synth_shadow.scoring.synth_score import compare_to_miners, top_miner_crps_stats
from synth_shadow.storage.files import ensure_dir, safe_timestamp
from synth_shadow.storage.registry import ForecastRegistry
from synth_shadow.synth.client import SynthClient
from synth_shadow.utils.logging import GREEN, colored_debug

LOG = logging.getLogger(__name__)


def score_forecast_dir(config: dict, forecast_dir: str) -> dict[str, Any]:
    """Fetch Synth realized path and score one forecast directory."""
    target = Path(forecast_dir)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    start_time = metadata.get("prompt_start_time") or metadata["data_cutoff"]

    client = SynthClient(config)
    realized_payload = client.realized_path(str(start_time))
    realized = np.asarray(realized_payload["real_prices"], dtype=float)
    paths = np.load(target / "paths.npz")["paths"]
    if realized.shape[0] != paths.shape[1]:
        raise ValueError(
            f"Synth realized path length {realized.shape[0]} does not match paths {paths.shape[1]}."
        )

    score = score_synth_btc_24h(paths, realized)
    latest_scores = client.latest_scores()
    comparison = compare_to_miners(score["raw_crps"], latest_scores)
    top10_stats = top_miner_crps_stats(latest_scores, count=10)
    realized_file = _save_realized_path(config, realized_payload)

    ForecastRegistry(config["storage"]["registry_path"]).register_score(
        str(target),
        score,
        str(realized_file),
        comparison,
    )
    result = {
        "forecast_dir": str(target),
        "realized_path_file": str(realized_file),
        "score": score,
        "comparison": comparison,
    }
    LOG.info("Scored forecast_dir=%s raw_crps=%.6f", target, score["raw_crps"])
    diagnostics = metadata.get("diagnostics") or {}
    colored_debug(
        LOG,
        (
            "[LIVE CRPS] asset=%s prompt_start=%s raw=%.6f "
            "5m=%.6f 30m=%.6f 3h=%.6f 24h=%.6f path=%.6f "
            "latest_top10_mean=%s latest_top10_median=%s latest_top10_std=%s "
            "gap_vs_latest_mean=%s gap_vs_latest_median=%s "
            "http_latency=%s node_latency=%s forecast_dir=%s"
        ),
        metadata.get("asset", config["asset"]),
        start_time,
        float(score["raw_crps"]),
        float(score["components"]["crps_5m"]),
        float(score["components"]["crps_30m"]),
        float(score["components"]["crps_3h"]),
        float(score["components"]["crps_24h"]),
        float(score["components"]["crps_path_price"]),
        _format_float(top10_stats["mean"]),
        _format_float(top10_stats["median"]),
        _format_float(top10_stats["std"]),
        _format_float(_gap(float(score["raw_crps"]), top10_stats["mean"])),
        _format_float(_gap(float(score["raw_crps"]), top10_stats["median"])),
        _format_seconds(diagnostics.get("http_latency_seconds")),
        _format_seconds(_node_total_latency(diagnostics)),
        target,
        color=GREEN,
    )
    LOG.debug("Score result: %s", result)
    return result


def score_matured_forecasts(config: dict) -> list[dict[str, Any]]:
    """Try to score every pending forecast; skip ones whose realized path is not ready."""
    registry = ForecastRegistry(config["storage"]["registry_path"])
    results = []
    pending = registry.list_forecasts(status="pending", asset=config["asset"])
    matured = [row for row in pending if _is_matured(row, config)]
    max_attempts = int(config.get("scoring", {}).get("max_matured_score_attempts_per_cycle", 3))
    LOG.info(
        "Scoring matured forecasts asset=%s pending=%s matured=%s max_attempts=%s",
        config["asset"],
        len(pending),
        len(matured),
        max_attempts,
    )
    for row in matured[:max_attempts]:
        forecast_dir = row["forecast_dir"]
        try:
            results.append(score_forecast_dir(config, forecast_dir))
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            LOG.warning("Could not score pending forecast %s: %s", forecast_dir, exc)
            if status_code == 429:
                LOG.warning("Stopping matured scoring for this cycle after Synth 429 rate limit.")
                break
        except Exception as exc:  # noqa: BLE001 - scoring loop should continue.
            LOG.warning("Could not score pending forecast %s: %s", forecast_dir, exc)
    LOG.info("Scored %s matured forecasts.", len(results))
    return results


def _is_matured(row: dict[str, Any], config: dict) -> bool:
    prompt_start = pd.Timestamp(row["prompt_start_time"])
    if prompt_start.tzinfo is None:
        prompt_start = prompt_start.tz_localize("UTC")
    else:
        prompt_start = prompt_start.tz_convert("UTC")
    horizon = pd.Timedelta(seconds=int(config["forecast"]["horizon_seconds"]))
    grace = pd.Timedelta(seconds=int(config.get("scoring", {}).get("maturity_grace_seconds", 300)))
    return pd.Timestamp.utcnow() >= prompt_start + horizon + grace


def _save_realized_path(config: dict, payload: dict[str, Any]) -> Path:
    start_time = payload["start_time"]
    target = ensure_dir(Path(config["storage"]["realized_dir"]) / payload["asset"])
    path = target / f"realized_{safe_timestamp(start_time)}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.debug("Saved realized path to %s", path)
    return path


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
