"""Forecast evaluation against Synth realized paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from synth_shadow.scoring.crps import score_synth_btc_24h
from synth_shadow.scoring.synth_score import compare_to_miners
from synth_shadow.storage.files import ensure_dir, safe_timestamp
from synth_shadow.storage.registry import ForecastRegistry
from synth_shadow.synth.client import SynthClient

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
    LOG.debug("Score result: %s", result)
    return result


def score_matured_forecasts(config: dict) -> list[dict[str, Any]]:
    """Try to score every pending forecast; skip ones whose realized path is not ready."""
    registry = ForecastRegistry(config["storage"]["registry_path"])
    results = []
    for row in registry.list_forecasts(status="pending"):
        forecast_dir = row["forecast_dir"]
        try:
            results.append(score_forecast_dir(config, forecast_dir))
        except Exception as exc:  # noqa: BLE001 - scoring loop should continue.
            LOG.warning("Could not score pending forecast %s: %s", forecast_dir, exc)
    LOG.info("Scored %s matured forecasts.", len(results))
    return results


def _save_realized_path(config: dict, payload: dict[str, Any]) -> Path:
    start_time = payload["start_time"]
    target = ensure_dir(Path(config["storage"]["realized_dir"]) / payload["asset"])
    path = target / f"realized_{safe_timestamp(start_time)}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.debug("Saved realized path to %s", path)
    return path
