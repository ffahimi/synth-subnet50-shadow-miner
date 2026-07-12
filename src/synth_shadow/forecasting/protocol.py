"""Forecast provider protocol used by the public live harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProviderForecast:
    """Forecast returned by a local or remote provider."""

    paths: np.ndarray
    timestamps: pd.DatetimeIndex
    metadata: dict[str, Any]
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ForecastProvider(Protocol):
    """Provider that returns Synth-compatible probabilistic price paths."""

    def generate(
        self,
        config: dict[str, Any],
        prompt_start_time: str | None = None,
        origin: pd.Timestamp | str | None = None,
    ) -> ProviderForecast:
        """Generate a forecast for the configured asset."""
        ...
