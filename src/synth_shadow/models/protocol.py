"""Public forecast model interface.

Private forecast packages should implement this interface and expose a class or
factory through ``SYNTH_MODEL_ENTRYPOINT=package.module:Model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from synth_shadow.models.current_state import CurrentState
from synth_shadow.models.path_sampler import PathSampler


@dataclass(frozen=True)
class ForecastContext:
    """Causal data bundle passed to a forecast model."""

    config: dict[str, Any]
    bars: pd.DataFrame
    features: pd.DataFrame
    library: list[Any]
    state: CurrentState
    sampler: PathSampler
    origin: pd.Timestamp | None = None


@dataclass(frozen=True)
class ForecastOutput:
    """Forecast model output."""

    paths: np.ndarray
    timestamps: pd.DatetimeIndex
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastModel(Protocol):
    """Protocol implemented by public or private forecast models."""

    model_version: str

    def generate(self, context: ForecastContext) -> ForecastOutput:
        """Generate probabilistic paths from a causal forecast context."""
        ...
