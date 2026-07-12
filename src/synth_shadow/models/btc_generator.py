"""BTC 24h generator.

Generation plan:

1. Read the latest canonical Polygon BTC bars.
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

from synth_shadow.data.polygon_client import PolygonClient
from synth_shadow.data.schema import repair_missing_bars
from synth_shadow.features.pipeline import build_feature_frame
from synth_shadow.models.current_state import extract_current_state
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.models.session_path_model import build_session_library
from synth_shadow.paths.generator import generate_paths
from synth_shadow.paths.validator import validate_paths
from synth_shadow.storage.forecast_store import (
    save_forecast_run,
    save_processed_features,
    save_raw_bars,
)
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


def run_btc_forecast(config: dict) -> dict:
    """Fetch Polygon BTC data, extract features, generate paths, and save artifacts."""
    LOG.info("Starting Polygon BTC shadow forecast pipeline.")
    client = PolygonClient()
    raw_bars = client.fetch_recent_btc(config)
    save_raw_bars(raw_bars, config)

    interval_seconds = int(config["forecast"]["interval_seconds"])
    bars = repair_missing_bars(raw_bars, interval_seconds) if config["history"].get("repair_missing_bars", True) else raw_bars
    features = build_feature_frame(bars, config)
    save_processed_features(features, config)

    block_bars = int(config["sampling"]["block_minutes"] * 60 / interval_seconds)
    library = build_session_library(features, block_bars)
    state = extract_current_state(features)
    sampler = PathSampler(library, seed=int(config["forecast"]["random_seed"]))
    paths, timestamps = generate_paths(state, sampler, config)
    validate_paths(
        paths,
        num_paths=int(config["forecast"]["num_paths"]),
        points_per_path=int(config["forecast"]["horizon_seconds"] / interval_seconds) + 1,
    )

    metadata = {
        "model_version": "session_path_v0",
        "asset": config["asset"],
        "polygon_ticker": config["polygon_ticker"],
        "generated_at": utc_now().isoformat(),
        "data_cutoff": state.timestamp,
        "num_raw_bars": len(raw_bars),
        "num_feature_rows": len(features),
        "num_session_blocks": len(library),
        "path_shape": list(paths.shape),
        "current_price": state.price,
        "debug": bool(config.get("debug", False)),
    }
    forecast_dir = save_forecast_run(paths, timestamps, metadata, state.to_dict(), config)
    LOG.info("Completed BTC forecast: %s", metadata)
    return {"forecast_dir": str(forecast_dir), "metadata": metadata}
