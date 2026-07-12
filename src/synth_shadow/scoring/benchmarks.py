"""Benchmark joins between Synth score and reward surfaces."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def join_scores_to_leaderboard(
    scores: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Join validation CRPS rows to leaderboard reward rows by miner UID."""
    rewards_by_uid = {
        int(row["neuron_uid"]): row
        for row in leaderboard
        if row.get("neuron_uid") is not None
    }
    joined = []
    valid_scores = [
        row
        for row in scores
        if row.get("crps") is not None
        and np.isfinite(float(row["crps"]))
        and float(row["crps"]) >= 0
    ]
    sorted_scores = sorted(
        valid_scores,
        key=lambda row: float(row["crps"]),
    )
    for rank, score in enumerate(sorted_scores[:limit], start=1):
        uid = int(score["miner_uid"])
        reward_row = rewards_by_uid.get(uid, {})
        joined.append(
            {
                "rank_by_crps": rank,
                "miner_uid": uid,
                "crps": float(score["crps"]),
                "prompt_score": float(score.get("prompt_score", 0.0)),
                "scored_time": score.get("scored_time"),
                "reward": reward_row.get("rewards"),
                "leaderboard_updated_at": reward_row.get("updated_at"),
            }
        )
    LOG.debug("Joined score/reward benchmark rows=%s", len(joined))
    return joined


def select_reference_miners(
    scores: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    count: int = 4,
) -> list[dict[str, Any]]:
    """Return the top N miners by latest CRPS with reward context."""
    return join_scores_to_leaderboard(scores, leaderboard, limit=count)
