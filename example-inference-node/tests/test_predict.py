from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from private_synth_models import server
from private_synth_models.data.polygon_1m import PolygonFetchResult


class FakePolygonClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def fetch_recent(self, ticker: str, end: pd.Timestamp, lookback_days: int = 14) -> PolygonFetchResult:
        del ticker, lookback_days
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        periods = 3600
        timestamps = pd.date_range(end_ts - pd.Timedelta(minutes=periods - 1), periods=periods, freq="min", tz="UTC")
        drift = np.linspace(0.0, 0.04, periods)
        cycle = np.sin(np.linspace(0.0, 28.0, periods)) * 0.01
        close = 64000.0 * np.exp(drift + cycle)
        bars = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close,
                "high": close * 1.0005,
                "low": close * 0.9995,
                "close": close,
                "volume": np.full(periods, 1.0),
                "vwap": close,
                "transactions": np.ones(periods),
            }
        )
        return PolygonFetchResult(bars=bars)


def test_predict_returns_harness_compatible_shape(monkeypatch):
    monkeypatch.setattr(server, "Polygon1mClient", FakePolygonClient)
    client = TestClient(server.app)
    response = client.post(
        "/predict",
        json={
            "asset": "BTC",
            "polygon_ticker": "X:BTCUSD",
            "prompt_start_time": "2026-07-12T16:29:00Z",
            "horizon_seconds": 86400,
            "interval_seconds": 300,
            "num_paths": 8,
            "generated_at": "2026-07-12T16:29:01Z",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["asset"] == "BTC"
    assert body["model_version"] == "private_btc_similarity_v0"
    assert len(body["paths"]) == 8
    assert all(len(path) == 289 for path in body["paths"])
    assert len(body["timestamps"]) == 289
    assert body["timestamps"][0] == body["data_cutoff"]
    assert body["paths"][0][0] == body["current_price"]
    assert body["diagnostics"]["data_source"] == "polygon_1m_rest"
    assert body["diagnostics"]["num_raw_bars"] == 3600
    assert body["diagnostics"]["num_feature_rows"] > 289
    assert body["diagnostics"]["nearest_neighbors"] > 0

    timestamps = pd.to_datetime(body["timestamps"], utc=True)
    spacing = timestamps.to_series().diff().dropna().dt.total_seconds().unique().tolist()
    assert spacing == [300.0]
    flat_prices = [price for path in body["paths"] for price in path]
    assert all(math.isfinite(price) and price > 0 for price in flat_prices)
    assert all(path[0] == body["current_price"] for path in body["paths"])


def test_predict_rejects_wrong_interval(monkeypatch):
    monkeypatch.setattr(server, "Polygon1mClient", FakePolygonClient)
    client = TestClient(server.app)
    response = client.post(
        "/predict",
        json={
            "asset": "BTC",
            "polygon_ticker": "X:BTCUSD",
            "prompt_start_time": "2026-07-12T16:29:00Z",
            "horizon_seconds": 86400,
            "interval_seconds": 60,
            "num_paths": 8,
            "generated_at": "2026-07-12T16:29:01Z",
        },
    )

    assert response.status_code == 400
    assert "300-second" in response.json()["detail"]

