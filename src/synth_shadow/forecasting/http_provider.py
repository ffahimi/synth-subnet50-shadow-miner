"""HTTP forecast provider for private model inference nodes."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from synth_shadow.forecasting.protocol import ProviderForecast
from synth_shadow.utils.logging import CYAN, colored_debug
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


class HttpForecastProvider:
    """Forecast provider that calls a private model inference HTTP endpoint."""

    provider_name = "http"

    def __init__(self, endpoint: str, timeout_seconds: int = 120) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        config: dict[str, Any],
        prompt_start_time: str | None = None,
        origin: pd.Timestamp | str | None = None,
    ) -> ProviderForecast:
        interval_seconds = int(config["forecast"]["interval_seconds"])
        horizon_seconds = int(config["forecast"]["horizon_seconds"])
        expected_points = int(horizon_seconds / interval_seconds) + 1
        expected_paths = int(config["forecast"]["num_paths"])
        payload = {
            "asset": config["asset"],
            "polygon_ticker": config["polygon_ticker"],
            "prompt_start_time": prompt_start_time,
            "horizon_seconds": horizon_seconds,
            "interval_seconds": interval_seconds,
            "num_paths": expected_paths,
            "generated_at": utc_now().isoformat(),
        }
        if origin is not None:
            payload["origin"] = _format_timestamp(origin)
        started = time.perf_counter()
        LOG.info("Requesting private forecast endpoint=%s asset=%s", self.endpoint, config["asset"])
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{response.status_code} error from forecast endpoint {self.endpoint}",
                response=response,
            ) from exc
        body = response.json()
        latency = round(time.perf_counter() - started, 6)

        paths = np.asarray(body["paths"], dtype=float)
        timestamps = _timestamps_from_response(body, interval_seconds, expected_points)
        if paths.shape != (expected_paths, expected_points):
            raise ValueError(
                f"HTTP forecast paths shape {paths.shape} does not match "
                f"{(expected_paths, expected_points)}."
            )
        if len(timestamps) != expected_points:
            raise ValueError(f"HTTP forecast returned {len(timestamps)} timestamps, expected {expected_points}.")
        current_price = float(body.get("current_price", paths[0, 0]))
        model_version = str(body.get("model_version", "http_model"))
        data_cutoff = str(body.get("data_cutoff") or timestamps[0])
        diagnostics = dict(body.get("diagnostics") or {})
        diagnostics["http_latency_seconds"] = latency
        node_latency = diagnostics.get("latency_seconds") or {}
        node_total_latency = node_latency.get("total") if isinstance(node_latency, dict) else None
        metadata = {
            "provider": self.provider_name,
            "model_version": model_version,
            "model_entrypoint": self.endpoint,
            "asset": config["asset"],
            "polygon_ticker": config["polygon_ticker"],
            "generated_at": utc_now().isoformat(),
            "data_cutoff": data_cutoff,
            "prompt_start_time": prompt_start_time,
            "num_raw_bars": diagnostics.get("num_raw_bars"),
            "num_feature_rows": diagnostics.get("num_feature_rows"),
            "num_session_blocks": diagnostics.get("num_session_blocks"),
            "path_shape": list(paths.shape),
            "current_price": current_price,
            "debug": bool(config.get("debug", False)),
            "model_metadata": body.get("metadata") or {},
            "diagnostics": diagnostics,
        }
        feature_snapshot = dict(body.get("feature_snapshot") or {})
        if not feature_snapshot:
            feature_snapshot = {
                "timestamp": data_cutoff,
                "price": current_price,
                "provider": self.provider_name,
            }
        colored_debug(
            LOG,
            (
                "[HTTP FORECAST] asset=%s prompt_start=%s origin=%s "
                "latency=%.3fs node_total=%s paths=%s points=%s cutoff=%s"
            ),
            config["asset"],
            prompt_start_time,
            _format_timestamp(origin) if origin is not None else None,
            latency,
            f"{float(node_total_latency):.3f}s" if node_total_latency is not None else "n/a",
            paths.shape[0],
            paths.shape[1],
            data_cutoff,
            color=CYAN,
        )
        return ProviderForecast(
            paths=paths,
            timestamps=timestamps,
            metadata=metadata,
            feature_snapshot=feature_snapshot,
            diagnostics=diagnostics,
        )


def configured_endpoint(config: dict[str, Any]) -> str | None:
    endpoint = os.getenv("SYNTH_MODEL_ENDPOINT") or config.get("model", {}).get("endpoint")
    return str(endpoint).strip() if endpoint else None


def _format_timestamp(value: pd.Timestamp | str) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _timestamps_from_response(
    body: dict[str, Any],
    interval_seconds: int,
    expected_points: int,
) -> pd.DatetimeIndex:
    if body.get("timestamps"):
        return pd.DatetimeIndex(pd.to_datetime(body["timestamps"], utc=True))
    data_cutoff = body.get("data_cutoff")
    if not data_cutoff:
        raise ValueError("HTTP forecast response must include either timestamps or data_cutoff.")
    return pd.date_range(
        pd.Timestamp(data_cutoff).tz_convert("UTC")
        if pd.Timestamp(data_cutoff).tzinfo
        else pd.Timestamp(data_cutoff, tz="UTC"),
        periods=expected_points,
        freq=f"{interval_seconds}s",
        tz="UTC",
    )
