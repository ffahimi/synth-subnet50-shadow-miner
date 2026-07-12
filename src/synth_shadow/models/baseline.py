"""Public baseline forecast model."""

from __future__ import annotations

from synth_shadow.models.protocol import ForecastContext, ForecastOutput
from synth_shadow.paths.generator import generate_paths


class SessionPathBaselineModel:
    """Baseline model using the public historical session-path generator."""

    model_version = "session_path_v0"

    def generate(self, context: ForecastContext) -> ForecastOutput:
        paths, timestamps = generate_paths(context.state, context.sampler, context.config)
        return ForecastOutput(
            paths=paths,
            timestamps=timestamps,
            metadata={"model_entrypoint": self.__class__.__module__ + ":" + self.__class__.__name__},
        )
