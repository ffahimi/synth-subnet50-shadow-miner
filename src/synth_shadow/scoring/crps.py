"""CRPS helpers for ensemble paths."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def crps_ensemble(samples: np.ndarray, observation: float) -> float:
    """CRPS for a one-dimensional ensemble forecast.

    Uses the sorted-sample identity for the pairwise absolute term, avoiding an
    explicit NxN matrix.
    """
    values = np.asarray(samples, dtype=float)
    n = values.size
    if n == 0:
        raise ValueError("CRPS requires at least one sample.")
    first = np.mean(np.abs(values - float(observation)))
    sorted_values = np.sort(values)
    weights = np.arange(1, n + 1)
    pairwise_sum = np.sum((2 * weights - n - 1) * sorted_values)
    second = pairwise_sum / (n * n)
    return float(first - second)


def crps_over_points(predicted: np.ndarray, realized: np.ndarray) -> float:
    """Average CRPS across aligned price points."""
    if predicted.shape[1] != realized.shape[0]:
        raise ValueError(f"Shape mismatch: paths={predicted.shape}, realized={realized.shape}")
    scores = [crps_ensemble(predicted[:, idx], float(realized[idx])) for idx in range(realized.shape[0])]
    return float(np.mean(scores))


def crps_over_deltas(predicted: np.ndarray, realized: np.ndarray, step: int) -> float:
    """Average CRPS over price changes at a given step length."""
    if predicted.shape[1] != realized.shape[0]:
        raise ValueError(f"Shape mismatch: paths={predicted.shape}, realized={realized.shape}")
    if step <= 0 or step >= realized.shape[0]:
        raise ValueError(f"Invalid CRPS delta step: {step}")
    predicted_delta = predicted[:, step:] - predicted[:, :-step]
    realized_delta = realized[step:] - realized[:-step]
    scores = [
        crps_ensemble(predicted_delta[:, idx], float(realized_delta[idx]))
        for idx in range(realized_delta.shape[0])
    ]
    return float(np.mean(scores))


def score_synth_btc_24h(predicted: np.ndarray, realized: np.ndarray) -> dict[str, Any]:
    """Compute shadow CRPS components for BTC 24h.

    The component set follows the documented BTC 24h evaluation scales:
    5m, 30m, 3h, and 24h. The exact validator implementation may evolve, so
    this module is intentionally isolated.
    """
    components = {
        "crps_5m": crps_over_deltas(predicted, realized, 1),
        "crps_30m": crps_over_deltas(predicted, realized, 6),
        "crps_3h": crps_over_deltas(predicted, realized, 36),
        "crps_24h": crps_over_deltas(predicted, realized, 288),
        "crps_path_price": crps_over_points(predicted, realized),
    }
    raw_crps = (
        components["crps_5m"]
        + components["crps_30m"]
        + components["crps_3h"]
        + components["crps_24h"]
    )
    score = {"raw_crps": float(raw_crps), "components": components}
    LOG.debug("Computed CRPS score: %s", score)
    return score
