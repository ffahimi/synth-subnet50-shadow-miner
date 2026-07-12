"""Local in-process forecast provider."""

from __future__ import annotations

from typing import Any

from synth_shadow.data.polygon_client import PolygonClient
from synth_shadow.data.schema import repair_missing_bars
from synth_shadow.features.pipeline import build_feature_frame
from synth_shadow.forecasting.protocol import ProviderForecast
from synth_shadow.models.current_state import extract_current_state
from synth_shadow.models.loader import configured_model_entrypoint, load_forecast_model
from synth_shadow.models.path_sampler import PathSampler
from synth_shadow.models.protocol import ForecastContext
from synth_shadow.models.session_path_model import build_session_library
from synth_shadow.storage.forecast_store import save_processed_features, save_raw_bars
from synth_shadow.utils.time import utc_now


class LocalForecastProvider:
    """Forecast provider that uses the public local data/feature/model pipeline."""

    provider_name = "local"

    def generate(
        self,
        config: dict[str, Any],
        prompt_start_time: str | None = None,
        origin: Any | None = None,
    ) -> ProviderForecast:
        if origin is not None:
            raise ValueError("LocalForecastProvider does not support historical origin generation.")
        client = PolygonClient()
        raw_bars = client.fetch_recent(config)
        save_raw_bars(raw_bars, config)

        interval_seconds = int(config["forecast"]["interval_seconds"])
        bars = (
            repair_missing_bars(raw_bars, interval_seconds)
            if config["history"].get("repair_missing_bars", True)
            else raw_bars
        )
        features = build_feature_frame(bars, config)
        save_processed_features(features, config)

        block_bars = int(config["sampling"]["block_minutes"] * 60 / interval_seconds)
        library = build_session_library(features, block_bars)
        state = extract_current_state(features)
        sampler = PathSampler(library, seed=int(config["forecast"]["random_seed"]))
        model = load_forecast_model(config)
        model_entrypoint = configured_model_entrypoint(config)
        output = model.generate(
            ForecastContext(
                config=config,
                bars=bars,
                features=features,
                library=library,
                state=state,
                sampler=sampler,
            )
        )
        metadata = {
            "provider": self.provider_name,
            "model_version": str(getattr(model, "model_version", model.__class__.__name__)),
            "model_entrypoint": model_entrypoint,
            "asset": config["asset"],
            "polygon_ticker": config["polygon_ticker"],
            "generated_at": utc_now().isoformat(),
            "data_cutoff": state.timestamp,
            "prompt_start_time": prompt_start_time,
            "num_raw_bars": len(raw_bars),
            "num_feature_rows": len(features),
            "num_session_blocks": len(library),
            "path_shape": list(output.paths.shape),
            "current_price": state.price,
            "debug": bool(config.get("debug", False)),
            "model_metadata": output.metadata,
        }
        return ProviderForecast(
            paths=output.paths,
            timestamps=output.timestamps,
            metadata=metadata,
            feature_snapshot=state.to_dict(),
            diagnostics={
                "raw_bar_count": len(raw_bars),
                "feature_row_count": len(features),
                "session_block_count": len(library),
                "data_source": "polygon_rest",
            },
        )
