"""Synth-style comparison helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def compare_to_miners(raw_crps: float, miner_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare our raw CRPS to miner raw CRPS values."""
    crps_values = valid_miner_crps_values(miner_scores)
    if crps_values.size == 0:
        return {
            "miner_count": 0,
            "percentile_beaten": None,
            "estimated_prompt_score": None,
            "best_crps": None,
            "median_crps": None,
            "top_25_threshold": None,
            "estimated_rank": None,
        }

    beaten = float(np.mean(crps_values > raw_crps))
    better_count = int(np.sum(crps_values < raw_crps))
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
        "estimated_rank": better_count + 1,
        "p90_cap": cap,
        "best_gap": float(raw_crps - np.min(crps_values)),
        "median_gap": float(raw_crps - np.median(crps_values)),
    }
    LOG.debug("Miner comparison: %s", comparison)
    return comparison


def valid_miner_crps_values(miner_scores: list[dict[str, Any]]) -> np.ndarray:
    """Return finite, non-negative miner CRPS values sorted ascending."""
    values = np.array(
        [float(row["crps"]) for row in miner_scores if row.get("crps") is not None],
        dtype=float,
    )
    values = values[np.isfinite(values) & (values >= 0)]
    return np.sort(values)


def rank_against_miners(raw_crps: float, miner_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate our rank against a miner score snapshot. Lower CRPS ranks better."""
    values = valid_miner_crps_values(miner_scores)
    if values.size == 0:
        return {
            "miner_count": 0,
            "rank": None,
            "miners_beaten": None,
            "percentile_beaten": None,
            "best_crps": None,
            "rank_note": "no valid non-negative miner CRPS values",
        }
    better_count = int(np.sum(values < raw_crps))
    beaten_count = int(np.sum(values > raw_crps))
    return {
        "miner_count": int(values.size),
        "rank": better_count + 1,
        "miners_beaten": beaten_count,
        "percentile_beaten": float(beaten_count / values.size),
        "best_crps": float(values[0]),
        "rank_note": "estimated against latest valid Synth score snapshot",
    }


def top_miner_crps_stats(miner_scores: list[dict[str, Any]], count: int = 10) -> dict[str, Any]:
    """Summarize the top N finite, non-negative miner CRPS values."""
    valid = [
        {
            "miner_uid": row.get("miner_uid"),
            "crps": float(row["crps"]),
            "scored_time": row.get("scored_time"),
        }
        for row in miner_scores
        if row.get("crps") is not None
        and np.isfinite(float(row["crps"]))
        and float(row["crps"]) >= 0
    ]
    top = sorted(valid, key=lambda row: row["crps"])[:count]
    if not top:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "uids": [],
            "scored_time": None,
        }
    values = np.array([row["crps"] for row in top], dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "uids": [row["miner_uid"] for row in top],
        "scored_time": top[0].get("scored_time"),
    }
