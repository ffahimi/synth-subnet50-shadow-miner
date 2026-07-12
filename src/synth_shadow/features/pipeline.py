"""Feature pipeline combining session tags, returns, volatility, and momentum."""

from __future__ import annotations

import logging

import pandas as pd

from synth_shadow.features.distribution import add_distribution_features
from synth_shadow.features.momentum import add_momentum_features
from synth_shadow.features.returns import add_log_returns
from synth_shadow.features.volatility import add_volatility_features
from synth_shadow.sessions.calendar import add_session_labels

LOG = logging.getLogger(__name__)


def build_feature_frame(bars: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build the debug-friendly BTC feature dataframe."""
    feature_cfg = config["features"]
    short_window = int(feature_cfg["short_window_bars"])
    long_window = int(feature_cfg["long_window_bars"])
    kurtosis_window = int(feature_cfg["kurtosis_window_bars"])

    df = add_session_labels(bars, config)
    df = add_log_returns(df)
    df = add_volatility_features(df, short_window, long_window)
    df = add_momentum_features(df, short_window, long_window)
    df = add_distribution_features(df, kurtosis_window)
    df = df.dropna(subset=["vol_1h", "vol_4h"]).reset_index(drop=True)

    LOG.debug("Feature frame rows=%s columns=%s", len(df), list(df.columns))
    LOG.debug("Session counts: %s", df["session"].value_counts().to_dict())
    if not df.empty:
        latest = df.iloc[-1][
            [
                "timestamp",
                "close",
                "session",
                "vol_1h",
                "vol_4h",
                "vol_of_vol_1h",
                "vol_slope",
                "momentum_1h",
                "momentum_4h",
                "kurtosis_4h",
            ]
        ].to_dict()
        LOG.debug("Latest feature snapshot: %s", latest)
    return df
