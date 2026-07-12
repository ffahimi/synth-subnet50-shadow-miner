"""Configuration loading for the shadow forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load a YAML config file as a plain dictionary."""
    load_dotenv()
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
