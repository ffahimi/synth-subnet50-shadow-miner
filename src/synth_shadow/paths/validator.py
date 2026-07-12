"""Validate generated paths against Synth BTC 24h shape requirements."""

from __future__ import annotations

import numpy as np


def validate_paths(paths: np.ndarray, num_paths: int, points_per_path: int) -> None:
    """Validate generated forecast path shape and values."""
    if paths.shape != (num_paths, points_per_path):
        raise ValueError(f"Expected paths shape {(num_paths, points_per_path)}, got {paths.shape}.")
    if not np.isfinite(paths).all():
        raise ValueError("Generated paths contain non-finite values.")
    if (paths <= 0).any():
        raise ValueError("Generated paths contain non-positive prices.")
