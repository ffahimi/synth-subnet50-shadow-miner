"""Load public or private forecast models by import entrypoint."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from synth_shadow.models.baseline import SessionPathBaselineModel
from synth_shadow.models.protocol import ForecastModel

LOG = logging.getLogger(__name__)

DEFAULT_MODEL_ENTRYPOINT = "synth_shadow.models.baseline:SessionPathBaselineModel"


def configured_model_entrypoint(config: dict[str, Any]) -> str:
    """Return model entrypoint from env, config, or the public baseline."""
    return (
        os.getenv("SYNTH_MODEL_ENTRYPOINT")
        or config.get("model", {}).get("entrypoint")
        or DEFAULT_MODEL_ENTRYPOINT
    )


def load_forecast_model(config: dict[str, Any]) -> ForecastModel:
    """Load a forecast model from ``module:attribute``.

    The attribute may be a model instance, a class with no required constructor
    arguments, or a no-argument factory returning an object with ``generate``.
    """
    entrypoint = configured_model_entrypoint(config)
    if entrypoint == DEFAULT_MODEL_ENTRYPOINT:
        model: ForecastModel = SessionPathBaselineModel()
        LOG.debug("Loaded public baseline forecast model entrypoint=%s", entrypoint)
        return model

    module_name, separator, attribute_name = entrypoint.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "SYNTH_MODEL_ENTRYPOINT must use 'module:attribute' format, "
            f"got {entrypoint!r}"
        )

    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    model = attribute() if callable(attribute) else attribute
    if not hasattr(model, "generate"):
        raise TypeError(f"Forecast model {entrypoint!r} does not expose a generate(context) method.")

    LOG.info("Loaded forecast model entrypoint=%s version=%s", entrypoint, _model_version(model))
    return model


def _model_version(model: ForecastModel) -> str:
    return str(getattr(model, "model_version", model.__class__.__name__))
