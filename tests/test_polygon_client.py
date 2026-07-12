from __future__ import annotations

import sys

import pytest
import requests

from synth_shadow.data import polygon_client


def test_polygon_key_missing_in_noninteractive_shell_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(ValueError, match="POLYGON_API_KEY is required"):
        polygon_client._resolve_api_key()


def test_polygon_key_placeholder_prompts_for_real_key(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "your_valid_polygon_key")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(polygon_client.getpass, "getpass", lambda _prompt: "real_polygon_key")

    assert polygon_client._resolve_api_key() == "real_polygon_key"
    assert polygon_client.os.environ["POLYGON_API_KEY"] == "real_polygon_key"


def test_polygon_http_error_message_does_not_include_api_key():
    response = requests.Response()
    response.status_code = 401
    response.reason = "Unauthorized"
    response.url = (
        "https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/5/minute/"
        "2026-06-28/2026-07-12?adjusted=true&apiKey=secret_key"
    )

    message = polygon_client._sanitized_http_error_message(response)

    assert "secret_key" not in message
    assert "apiKey" not in message
    assert "https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/5/minute/" in message
