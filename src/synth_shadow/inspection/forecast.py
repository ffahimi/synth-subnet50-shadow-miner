"""Summaries for generated forecast paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synth_shadow.storage.files import ensure_dir
from synth_shadow.storage.registry import ForecastRegistry

LOG = logging.getLogger(__name__)


def inspect_forecast(config: dict, forecast_dir: str | None = None) -> dict[str, Any]:
    """Load a forecast directory and return aggregate path diagnostics."""
    if forecast_dir is None:
        latest = ForecastRegistry(config["storage"]["registry_path"]).latest_forecast(config["asset"])
        if latest:
            forecast_dir = latest["forecast_dir"]
        else:
            dirs = sorted(Path(config["storage"]["forecast_dir"]).glob(f"{config['asset']}/*"))
            if not dirs:
                raise FileNotFoundError("No forecast runs found.")
            forecast_dir = str(dirs[-1])

    target = Path(forecast_dir)
    paths = np.load(target / "paths.npz")["paths"]
    timestamps = pd.read_csv(target / "timestamps.csv")["timestamp"]
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))

    checkpoints = config["inspection"]["checkpoints"]
    aggregate = {}
    for label, idx in checkpoints.items():
        values = paths[:, int(idx)]
        aggregate[label] = {
            "timestamp": str(timestamps.iloc[int(idx)]),
            "p05": round(float(np.percentile(values, 5)), 2),
            "p25": round(float(np.percentile(values, 25)), 2),
            "median": round(float(np.percentile(values, 50)), 2),
            "p75": round(float(np.percentile(values, 75)), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
        }

    sample_count = int(config["inspection"]["sample_paths"])
    sample_points = int(config["inspection"]["sample_points"])
    sample_paths = {
        f"path_{idx}": [round(float(value), 2) for value in paths[idx, :sample_points]]
        for idx in range(min(sample_count, paths.shape[0]))
    }

    final = paths[:, -1]
    start = float(paths[0, 0])
    returns = final / start - 1.0
    final_distribution = {
        "start": round(start, 2),
        "final_median": round(float(np.median(final)), 2),
        "final_p05": round(float(np.percentile(final, 5)), 2),
        "final_p95": round(float(np.percentile(final, 95)), 2),
        "return_median_pct": round(float(np.median(returns) * 100), 4),
        "return_p05_pct": round(float(np.percentile(returns, 5) * 100), 4),
        "return_p95_pct": round(float(np.percentile(returns, 95) * 100), 4),
    }

    summary = {
        "forecast_dir": str(target),
        "shape": list(paths.shape),
        "metadata": metadata,
        "aggregate_checkpoints": aggregate,
        "final_distribution": final_distribution,
        "sample_paths": sample_paths,
    }
    reports_dir = ensure_dir(config["storage"]["reports_dir"])
    report_path = reports_dir / "latest_forecast_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Forecast inspection saved to %s", report_path)
    LOG.debug("Forecast inspection summary: %s", summary)
    return summary
