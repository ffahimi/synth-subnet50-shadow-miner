"""Immutable forecast storage.

Each forecast run should write:

- paths.npz: generated price paths
- metadata.json: model version, data cutoff, random seed, path shape
- features.json: volatility, vol-of-vol, slope, momentum, kurtosis snapshot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synth_shadow.storage.files import ensure_dir, safe_timestamp

LOG = logging.getLogger(__name__)


def save_raw_bars(bars: pd.DataFrame, config: dict) -> Path:
    raw_dir = ensure_dir(config["storage"]["raw_dir"])
    filename = f"polygon_btc_5m_{safe_timestamp(bars['timestamp'].min())}_{safe_timestamp(bars['timestamp'].max())}.csv"
    path = raw_dir / filename
    bars.to_csv(path, index=False)
    LOG.debug("Saved raw bars to %s", path)
    return path


def save_processed_features(features: pd.DataFrame, config: dict) -> Path:
    processed_dir = ensure_dir(config["storage"]["processed_dir"])
    filename = f"btc_features_{safe_timestamp(features['timestamp'].min())}_{safe_timestamp(features['timestamp'].max())}.csv"
    path = processed_dir / filename
    features.to_csv(path, index=False)
    LOG.debug("Saved processed features to %s", path)
    return path


def save_forecast_run(
    paths: np.ndarray,
    timestamps: pd.DatetimeIndex,
    metadata: dict[str, Any],
    feature_snapshot: dict[str, Any],
    config: dict,
) -> Path:
    forecast_dir = ensure_dir(Path(config["storage"]["forecast_dir"]) / "BTC" / safe_timestamp(metadata["generated_at"]))
    np.savez_compressed(forecast_dir / "paths.npz", paths=paths)
    pd.DataFrame({"timestamp": timestamps}).to_csv(forecast_dir / "timestamps.csv", index=False)
    _write_json(forecast_dir / "metadata.json", metadata)
    _write_json(forecast_dir / "features.json", feature_snapshot)
    LOG.info("Saved forecast run to %s", forecast_dir)
    return forecast_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
