"""Asset configuration helpers."""

from __future__ import annotations

import copy


def apply_asset(config: dict, asset: str | None) -> dict:
    """Return a config copy with asset-specific Polygon/Synth settings applied."""
    resolved = copy.deepcopy(config)
    selected = (asset or resolved.get("asset") or "BTC").upper()
    assets = resolved.get("assets", {})
    if selected not in assets:
        available = ", ".join(sorted(assets)) or "none"
        raise ValueError(f"Unsupported asset {selected!r}. Available assets: {available}")

    asset_cfg = assets[selected]
    resolved["asset"] = selected
    resolved["polygon_ticker"] = asset_cfg["polygon_ticker"]
    resolved.setdefault("synth", {})
    resolved["synth"]["asset"] = asset_cfg.get("synth_asset", selected)
    resolved["synth"]["competition"] = asset_cfg.get(
        "competition",
        resolved["synth"].get("competition", "crypto-24h"),
    )
    return resolved
