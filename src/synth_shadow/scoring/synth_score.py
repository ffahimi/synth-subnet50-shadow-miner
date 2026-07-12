"""Synth-style comparison helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def compare_to_miners(raw_crps: float, miner_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare our raw CRPS to miner raw CRPS values."""
    crps_values = np.array(
        [float(row["crps"]) for row in miner_scores if row.get("crps") is not None],
        dtype=float,
    )
    crps_values = crps_values[np.isfinite(crps_values)]
    if crps_values.size == 0:
        return {
            "miner_count": 0,
            "percentile_beaten": None,
            "estimated_prompt_score": None,
            "best_crps": None,
            "median_crps": None,
            "top_25_threshold": None,
        }

    beaten = float(np.mean(crps_values > raw_crps))
    cap = float(np.percentile(crps_values, 90))
    capped_you = min(float(raw_crps), cap)
    capped_miners = np.minimum(crps_values, cap)
    best = float(np.min(capped_miners))
    comparison = {
        "miner_count": int(crps_values.size),
        "percentile_beaten": round(beaten, 6),
        "estimated_prompt_score": float(capped_you - best),
        "best_crps": float(np.min(crps_values)),
        "median_crps": float(np.median(crps_values)),
        "top_25_threshold": float(np.percentile(crps_values, 25)),
        "p90_cap": cap,
        "best_gap": float(raw_crps - np.min(crps_values)),
        "median_gap": float(raw_crps - np.median(crps_values)),
    }
    LOG.debug("Miner comparison: %s", comparison)
    return comparison
