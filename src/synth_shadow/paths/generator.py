"""Generate 1,000 BTC paths of 289 prices for the next 24 hours."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from synth_shadow.models.current_state import CurrentState
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.sessions.calendar import classify_timestamp

LOG = logging.getLogger(__name__)


def generate_paths(
    current_state: CurrentState,
    sampler: PathSampler,
    config: dict,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate BTC price paths using historical session blocks and current regime."""
    forecast_cfg = config["forecast"]
    interval_seconds = int(forecast_cfg["interval_seconds"])
    horizon_seconds = int(forecast_cfg["horizon_seconds"])
    num_paths = int(forecast_cfg["num_paths"])
    future_steps = horizon_seconds // interval_seconds
    points_per_path = future_steps + 1

    t0 = pd.Timestamp(current_state.timestamp)
    future_timestamps = pd.date_range(
        t0,
        periods=points_per_path,
        freq=f"{interval_seconds}s",
        tz="UTC",
    )
    future_sessions = [
        classify_timestamp(ts, config["sessions"]) for ts in future_timestamps[1:]
    ]
    block_bars = int(config["sampling"]["block_minutes"] * 60 / interval_seconds)

    paths = np.empty((num_paths, points_per_path), dtype=float)
    paths[:, 0] = current_state.price

    for path_idx in range(num_paths):
        generated_returns: list[float] = []
        while len(generated_returns) < future_steps:
            session = future_sessions[len(generated_returns)]
            block = sampler.sample(session)
            adjusted = _rescale_block(block.normalized_returns, current_state, config)
            generated_returns.extend(adjusted[:block_bars].tolist())
        returns = np.array(generated_returns[:future_steps])
        paths[path_idx, 1:] = current_state.price * np.exp(np.cumsum(returns))

    LOG.debug(
        "Generated paths shape=%s t0=%s start_price=%.2f min=%.2f max=%.2f",
        paths.shape,
        t0,
        current_state.price,
        float(paths.min()),
        float(paths.max()),
    )
    return paths, future_timestamps


def _rescale_block(normalized_returns: np.ndarray, state: CurrentState, config: dict) -> np.ndarray:
    norm_cfg = config["normalization"]
    target_vol = max(state.vol_4h, 1e-8)
    vol_slope_multiplier = np.clip(1.0 + state.vol_slope, 0.5, 1.8)
    vol_of_vol_multiplier = np.clip(1.0 + state.vol_of_vol_1h / target_vol, 0.7, 1.8)
    kurtosis_multiplier = np.clip(
        1.0 + max(state.kurtosis_4h, 0.0) / 20.0,
        float(norm_cfg["kurtosis_scale_min"]),
        float(norm_cfg["kurtosis_scale_max"]),
    )
    vol_scale = np.clip(
        vol_slope_multiplier * vol_of_vol_multiplier * kurtosis_multiplier,
        float(norm_cfg["volatility_scale_min"]),
        float(norm_cfg["volatility_scale_max"]),
    )

    momentum_bps = float(norm_cfg["momentum_adjustment_max_bps"])
    momentum_per_bar = np.clip(state.momentum_1h / 12.0, -momentum_bps / 10000, momentum_bps / 10000)
    return normalized_returns * target_vol * vol_scale + momentum_per_bar
