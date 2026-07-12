"""Forecast provider selection."""

from __future__ import annotations

import logging
from typing import Any

from synth_shadow.forecasting.http_provider import HttpForecastProvider, configured_endpoint
from synth_shadow.forecasting.local_provider import LocalForecastProvider
from synth_shadow.forecasting.protocol import ForecastProvider

LOG = logging.getLogger(__name__)


def load_forecast_provider(config: dict[str, Any]) -> ForecastProvider:
    """Load the configured forecast provider.

    If ``SYNTH_MODEL_ENDPOINT`` or ``model.endpoint`` is set, the public harness
    delegates prediction to the private HTTP node. Otherwise it uses the public
    local baseline/provider.
    """
    endpoint = configured_endpoint(config)
    if endpoint:
        timeout = int(config.get("model", {}).get("timeout_seconds", 120))
        LOG.info("Using HTTP forecast provider endpoint=%s timeout=%ss", endpoint, timeout)
        return HttpForecastProvider(endpoint=endpoint, timeout_seconds=timeout)
    LOG.debug("Using local forecast provider.")
    return LocalForecastProvider()
