"""FastAPI app for private BTC forecast inference."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from private_synth_models.data.polygon_1m import Polygon1mClient
from private_synth_models.features.vectorizer import build_feature_bundle
from private_synth_models.models.similarity import generate_similarity_paths

load_dotenv()

MODEL_VERSION = "private_btc_similarity_v0"
DEFAULT_LOOKBACK_DAYS = 14

app = FastAPI(title="Private Synth BTC Inference Node", version="0.1.0")


class PredictRequest(BaseModel):
    asset: str = Field(default="BTC")
    polygon_ticker: str = Field(default="X:BTCUSD")
    prompt_start_time: datetime | None = None
    origin: datetime | None = None
    horizon_seconds: int = Field(default=86400, gt=0)
    interval_seconds: int = Field(default=300, gt=0)
    num_paths: int = Field(default=1000, gt=0, le=10000)
    generated_at: datetime | None = None


class PredictResponse(BaseModel):
    asset: str
    model_version: str
    data_cutoff: str
    current_price: float
    paths: list[list[float]]
    timestamps: list[str]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    if request.asset.upper() != "BTC":
        raise HTTPException(status_code=400, detail="private_btc_similarity_v0 only supports BTC.")
    if request.interval_seconds != 300 or request.horizon_seconds != 86400:
        raise HTTPException(status_code=400, detail="Expected 24h horizon at 300-second intervals.")

    stage_latency: dict[str, float] = {}
    try:
        fetch_started = time.perf_counter()
        end_time = _request_anchor_time(request)
        client = Polygon1mClient()
        fetch = await client.fetch_recent(
            ticker=request.polygon_ticker,
            end=end_time,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
        )
        stage_latency["fetch_data"] = round(time.perf_counter() - fetch_started, 6)

        feature_started = time.perf_counter()
        bundle = build_feature_bundle(fetch.bars, interval_seconds=request.interval_seconds)
        stage_latency["build_features"] = round(time.perf_counter() - feature_started, 6)

        model_started = time.perf_counter()
        points_per_path = int(request.horizon_seconds / request.interval_seconds) + 1
        forecast = generate_similarity_paths(
            features=bundle.features,
            current_price=bundle.current_price,
            num_paths=request.num_paths,
            points_per_path=points_per_path,
        )
        timestamps = pd.date_range(
            bundle.data_cutoff,
            periods=points_per_path,
            freq=f"{request.interval_seconds}s",
            tz="UTC",
        )
        stage_latency["generate_paths"] = round(time.perf_counter() - model_started, 6)

        paths = _validate_paths(forecast.paths, request.num_paths, points_per_path, bundle.current_price)
        timestamp_strings = [_format_ts(ts) for ts in timestamps]
        data_cutoff = timestamp_strings[0]
        diagnostics = {
            "data_source": fetch.data_source,
            "num_raw_bars": int(len(fetch.bars)),
            "num_feature_rows": int(len(bundle.features)),
            "nearest_neighbors": int(forecast.nearest_neighbors),
            "data_cutoff": data_cutoff,
            "latency_seconds": {
                **stage_latency,
                "total": round(time.perf_counter() - started, 6),
            },
        }
        return PredictResponse(
            asset=request.asset.upper(),
            model_version=MODEL_VERSION,
            data_cutoff=data_cutoff,
            current_price=float(bundle.current_price),
            paths=paths.tolist(),
            timestamps=timestamp_strings,
            diagnostics=diagnostics,
            metadata={
                "polygon_ticker": request.polygon_ticker,
                "prompt_start_time": _format_optional_dt(request.prompt_start_time),
                "origin": _format_optional_dt(request.origin),
                "generated_at": _format_optional_dt(request.generated_at),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _request_anchor_time(request: PredictRequest) -> pd.Timestamp:
    ts = request.origin or request.prompt_start_time or datetime.now(timezone.utc)
    anchor = pd.Timestamp(ts)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    else:
        anchor = anchor.tz_convert("UTC")
    return anchor.floor("min")


def _validate_paths(paths: np.ndarray, num_paths: int, points_per_path: int, current_price: float) -> np.ndarray:
    expected_shape = (num_paths, points_per_path)
    if paths.shape != expected_shape:
        raise ValueError(f"Generated paths shape {paths.shape} does not match {expected_shape}.")
    if not np.isfinite(paths).all():
        raise ValueError("Generated paths include non-finite prices.")
    if not (paths > 0).all():
        raise ValueError("Generated paths include non-positive prices.")
    paths[:, 0] = current_price
    return paths


def _format_ts(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_optional_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return _format_ts(ts)
