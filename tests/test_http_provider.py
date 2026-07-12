from __future__ import annotations

import numpy as np
import pytest

from synth_shadow.forecasting.http_provider import HttpForecastProvider


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _config():
    return {
        "asset": "BTC",
        "polygon_ticker": "X:BTCUSD",
        "forecast": {
            "horizon_seconds": 600,
            "interval_seconds": 300,
            "num_paths": 2,
        },
        "model": {"timeout_seconds": 5},
        "debug": True,
    }


def test_http_provider_posts_forecast_request_and_parses_response(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _Response(
            {
                "model_version": "private_btc_v1",
                "data_cutoff": "2026-07-12T16:30:00Z",
                "current_price": 100.0,
                "paths": [[100.0, 101.0, 102.0], [100.0, 99.0, 98.0]],
                "diagnostics": {"num_raw_bars": 1440, "num_feature_rows": 1200},
                "metadata": {"kind": "similarity"},
            }
        )

    monkeypatch.setattr("synth_shadow.forecasting.http_provider.requests.post", fake_post)

    forecast = HttpForecastProvider("http://127.0.0.1:8088/predict", timeout_seconds=5).generate(
        _config(),
        prompt_start_time="2026-07-12T16:29:00Z",
    )

    assert calls[0]["url"] == "http://127.0.0.1:8088/predict"
    assert calls[0]["json"]["asset"] == "BTC"
    assert calls[0]["json"]["num_paths"] == 2
    assert calls[0]["json"]["interval_seconds"] == 300
    assert forecast.paths.shape == (2, 3)
    assert np.allclose(forecast.paths[:, 0], 100.0)
    assert forecast.metadata["provider"] == "http"
    assert forecast.metadata["model_version"] == "private_btc_v1"
    assert forecast.metadata["prompt_start_time"] == "2026-07-12T16:29:00Z"
    assert forecast.diagnostics["num_raw_bars"] == 1440


def test_http_provider_rejects_wrong_path_shape(monkeypatch):
    def fake_post(url, json, timeout):
        return _Response(
            {
                "data_cutoff": "2026-07-12T16:30:00Z",
                "current_price": 100.0,
                "paths": [[100.0, 101.0]],
            }
        )

    monkeypatch.setattr("synth_shadow.forecasting.http_provider.requests.post", fake_post)

    with pytest.raises(ValueError, match="does not match"):
        HttpForecastProvider("http://127.0.0.1:8088/predict").generate(_config())
