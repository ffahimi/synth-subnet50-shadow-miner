"""Asset 24h generator.

Generation plan:

1. Read the latest canonical Polygon bars.
2. Compute 1h and 4h regime features.
3. Build future 5-minute timestamps for the next 24h.
4. Assign each future timestamp to a liquidity session.
5. Sample matching normalized historical session fragments.
6. Rescale with current volatility, vol-of-vol, volatility slope, momentum,
   and kurtosis.
7. Emit 1,000 paths with 289 prices anchored at the current BTC price.
"""

from __future__ import annotations

import logging

from synth_shadow.forecasting.loader import load_forecast_provider
from synth_shadow.paths.validator import validate_paths
from synth_shadow.storage.forecast_store import (
    save_forecast_run,
)
from synth_shadow.storage.registry import ForecastRegistry

LOG = logging.getLogger(__name__)


def run_asset_forecast(config: dict, prompt_start_time: str | None = None) -> dict:
    """Fetch Polygon data, extract features, generate paths, and save artifacts."""
    LOG.info("Starting %s shadow forecast pipeline.", config["asset"])
    interval_seconds = int(config["forecast"]["interval_seconds"])
    provider = load_forecast_provider(config)
    forecast = provider.generate(config, prompt_start_time=prompt_start_time)
    paths = forecast.paths
    timestamps = forecast.timestamps
    validate_paths(
        paths,
        num_paths=int(config["forecast"]["num_paths"]),
        points_per_path=int(config["forecast"]["horizon_seconds"] / interval_seconds) + 1,
    )
    forecast_dir = save_forecast_run(
        paths,
        timestamps,
        forecast.metadata,
        forecast.feature_snapshot,
        config,
    )
    registry = ForecastRegistry(config["storage"]["registry_path"])
    registry.register_forecast(str(forecast_dir), forecast.metadata)
    LOG.info("Completed %s forecast: %s", config["asset"], forecast.metadata)
    return {"forecast_dir": str(forecast_dir), "metadata": forecast.metadata}


def run_btc_forecast(config: dict, prompt_start_time: str | None = None) -> dict:
    """Backward-compatible alias for existing BTC command paths."""
    return run_asset_forecast(config, prompt_start_time=prompt_start_time)
