"""Debug-friendly orchestration for the shadow miner workflow."""

from __future__ import annotations

import logging
from typing import Any

from synth_shadow.inspection.forecast import inspect_forecast
from synth_shadow.models.btc_generator import run_btc_forecast
from synth_shadow.scoring.evaluator import score_matured_forecasts
from synth_shadow.storage.registry import ForecastRegistry
from synth_shadow.synth.client import SynthClient

LOG = logging.getLogger(__name__)


def sync_prompts(config: dict) -> dict[str, Any]:
    """Fetch official Synth prompts and store them locally."""
    client = SynthClient(config)
    registry = ForecastRegistry(config["storage"]["registry_path"])
    prompts = client.prompts()
    registry.upsert_prompts(prompts)
    result = {
        "prompt_count": len(prompts),
        "earliest_prompt": prompts[0] if prompts else None,
        "latest_prompt": prompts[-1] if prompts else None,
    }
    LOG.info("Synced Synth prompts: %s", result)
    return result


def generate_for_latest_prompt(config: dict) -> dict[str, Any]:
    """Generate a Polygon BTC forecast tagged to the latest known Synth prompt."""
    registry = ForecastRegistry(config["storage"]["registry_path"])
    prompts = registry.list_prompts()
    prompt_start = prompts[0]["start_time"] if prompts else None
    if prompt_start is None:
        LOG.warning("No synced prompt available; generating unaligned forecast.")
    else:
        LOG.info("Generating forecast for latest synced prompt_start_time=%s", prompt_start)
    return run_btc_forecast(config, prompt_start_time=prompt_start)


def fetch_benchmarks(config: dict) -> dict[str, Any]:
    """Fetch public leaderboard/reward context for debug visibility."""
    client = SynthClient(config)
    latest_scores = client.latest_scores()
    leaderboard = client.latest_leaderboard()
    rewards = client.rewards_scores() if config["synth"].get("fetch_rewards_in_cycle", False) else []
    result = {
        "latest_scores_count": len(latest_scores),
        "rewards_count": len(rewards),
        "leaderboard_count": len(leaderboard),
        "latest_scores_sample": latest_scores[:3],
        "leaderboard_sample": leaderboard[:3],
    }
    LOG.info("Fetched Synth benchmarks: %s", {k: v for k, v in result.items() if not k.endswith("_sample")})
    LOG.debug("Benchmark samples: %s", result)
    return result


def run_shadow_cycle(config: dict) -> dict[str, Any]:
    """Run the first full shadow cycle with all currently implemented modules."""
    LOG.info("Starting full Synth shadow cycle.")
    prompt_result = sync_prompts(config)
    forecast_result = generate_for_latest_prompt(config)
    inspection = inspect_forecast(config, forecast_result["forecast_dir"])
    score_results = score_matured_forecasts(config)
    benchmarks = fetch_benchmarks(config)
    result = {
        "prompts": prompt_result,
        "forecast": forecast_result,
        "inspection": {
            "forecast_dir": inspection["forecast_dir"],
            "shape": inspection["shape"],
            "final_distribution": inspection["final_distribution"],
            "aggregate_checkpoints": inspection["aggregate_checkpoints"],
        },
        "scores": score_results,
        "benchmarks": benchmarks,
    }
    LOG.info("Completed full Synth shadow cycle.")
    LOG.debug("Full cycle result: %s", result)
    return result
