"""Placeholder similarity/bootstrap BTC path generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from private_synth_models.features.vectorizer import feature_matrix, five_minute_returns, latest_feature_vector


@dataclass(frozen=True)
class SimilarityForecast:
    paths: np.ndarray
    nearest_neighbors: int


def generate_similarity_paths(
    features: pd.DataFrame,
    current_price: float,
    num_paths: int,
    points_per_path: int,
    seed: int = 50,
    nearest_neighbors: int = 64,
) -> SimilarityForecast:
    if num_paths <= 0:
        raise ValueError("num_paths must be positive.")
    if points_per_path < 2:
        raise ValueError("points_per_path must include current price and future steps.")

    returns = five_minute_returns(features)
    matrix = feature_matrix(features)
    latest = latest_feature_vector(features)
    usable_starts = _usable_start_indices(len(features), points_per_path)
    if usable_starts.size == 0:
        raise ValueError("Not enough feature rows to bootstrap a 24h path.")

    candidate_matrix = matrix[usable_starts]
    scale = np.nanstd(candidate_matrix, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    distances = np.linalg.norm((candidate_matrix - latest) / scale, axis=1)
    k = min(nearest_neighbors, len(usable_starts))
    neighbor_positions = np.argpartition(distances, k - 1)[:k]
    neighbor_starts = usable_starts[neighbor_positions]

    rng = np.random.default_rng(seed)
    future_steps = points_per_path - 1
    paths = np.empty((num_paths, points_per_path), dtype=float)
    paths[:, 0] = current_price
    latest_vol = float(features.iloc[-1]["volatility_4h"])

    for path_idx in range(num_paths):
        start = int(rng.choice(neighbor_starts))
        block = returns[start : start + future_steps].astype(float)
        if len(block) < future_steps:
            block = rng.choice(returns[np.isfinite(returns)], size=future_steps, replace=True)
        adjusted = _stabilize_returns(block, latest_vol=latest_vol, rng=rng)
        paths[path_idx, 1:] = current_price * np.exp(np.cumsum(adjusted))

    paths = np.nan_to_num(paths, nan=current_price, posinf=current_price * 10.0, neginf=current_price * 0.1)
    paths = np.maximum(paths, 0.01)
    paths[:, 0] = current_price
    return SimilarityForecast(paths=paths, nearest_neighbors=k)


def _usable_start_indices(num_rows: int, points_per_path: int) -> np.ndarray:
    future_steps = points_per_path - 1
    # Skip the latest block to avoid copying the active state into the forecast.
    max_start = num_rows - future_steps - 1
    if max_start <= 0:
        return np.array([], dtype=int)
    return np.arange(0, max_start, dtype=int)


def _stabilize_returns(block: np.ndarray, latest_vol: float, rng: np.random.Generator) -> np.ndarray:
    clean = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    source_vol = float(np.std(clean))
    if np.isfinite(latest_vol) and latest_vol > 1e-8 and source_vol > 1e-8:
        clean = clean / source_vol * latest_vol
    jitter_scale = max(float(np.std(clean)) * 0.05, 1e-6)
    return clean + rng.normal(0.0, jitter_scale, size=clean.shape)

