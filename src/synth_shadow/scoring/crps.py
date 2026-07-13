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
    """Average CRPS across aligned price points.

    This is kept as a diagnostic path-shape score. It is not part of Synth's
    validator raw CRPS calculation.
    """
    if predicted.shape[1] != realized.shape[0]:
        raise ValueError(f"Shape mismatch: paths={predicted.shape}, realized={realized.shape}")
    scores = [crps_ensemble(predicted[:, idx], float(realized[idx])) for idx in range(realized.shape[0])]
    return float(np.mean(scores))


def crps_sum_over_interval(
    predicted: np.ndarray,
    realized: np.ndarray,
    step: int,
    *,
    absolute_price: bool = False,
) -> float:
    """Validator-style summed CRPS over non-overlapping interval points.

    Synth's validator samples paths at ``price_paths[:, ::step]``. For 5m this
    produces 288 one-step changes over a 24h/5m path; for 30m it produces 48;
    for 3h it produces 8. The 24h component is scored on the absolute final
    price and normalized to basis points by realized final price.
    """
    if predicted.shape[1] != realized.shape[0]:
        raise ValueError(f"Shape mismatch: paths={predicted.shape}, realized={realized.shape}")
    if step <= 0 or step >= realized.shape[0]:
        raise ValueError(f"Invalid CRPS delta step: {step}")

    predicted_interval = predicted[:, ::step]
    realized_interval = realized.reshape(1, -1)[:, ::step]

    if absolute_price:
        predicted_values = predicted_interval[:, 1:]
        realized_values = realized_interval[:, 1:]
    else:
        predicted_values = _basis_point_changes(predicted_interval[:, 1:], predicted_interval[:, :-1])
        realized_values = _basis_point_changes(realized_interval[:, 1:], realized_interval[:, :-1])

    total = 0.0
    for idx in range(realized_values.shape[1]):
        value = crps_ensemble(predicted_values[:, idx], float(realized_values[0, idx]))
        if absolute_price:
            value = value / float(realized[-1]) * 10000.0
        total += value
    return float(total)


def score_synth_btc_24h(predicted: np.ndarray, realized: np.ndarray) -> dict[str, Any]:
    """Compute Synth validator-compatible CRPS components for a 24h prompt.

    The validator sums CRPS over non-overlapping 5m, 30m, and 3h return
    increments in basis points, plus a 24h absolute-price CRPS normalized to
    basis points by realized final price.
    """
    components = {
        "crps_5m": crps_sum_over_interval(predicted, realized, 1),
        "crps_30m": crps_sum_over_interval(predicted, realized, 6),
        "crps_3h": crps_sum_over_interval(predicted, realized, 36),
        "crps_24h": crps_sum_over_interval(predicted, realized, 288, absolute_price=True),
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


def _basis_point_changes(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return ((current - previous) / previous) * 10000.0
