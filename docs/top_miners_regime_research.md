# Top Miners Regime Research

Generated from the 90-day research run on 2026-07-13.

This note documents how persistent Synth `Crypto 24h` leaders perform across
BTC/ETH market regimes. It also defines a repeatable script for rerunning the
analysis over longer windows.

## Question

We want to know whether the miners that persistently appear in the top 25 are
better on:

- high-volatility or low-volatility days
- bullish, bearish, or flat days
- trending or mean-reverting days
- specific volatile sessions, for example US high-volatility after quiet EU or
  outside-market-hours periods

The goal is to identify where persistent leaderboard miners have edge and where
they give it back.

## Data

Run window:

```text
2026-04-14 to 2026-07-13
```

Score data:

```text
BTC: 459,272 valid score rows, 1,944 score snapshots, 91 calendar days
ETH: 456,133 valid score rows, 1,936 score snapshots, 91 calendar days
```

Market data:

```text
Polygon 5-minute BTC/ETH aggregates
```

Features:

```text
vol_1h
vol_4h
vol_of_vol_1h
vol_of_vol_4h
vol_slope
momentum_1h
momentum_4h
kurtosis_4h
session labels: eu, eu_us_overlap, us, outside_market_hours, weekend
```

## Definitions

For each prompt score snapshot, miners are ranked by CRPS ascending.

For each miner and day:

```text
daily_best_rank = best CRPS rank achieved that day
top25_day = whether the miner appeared in top 25 at least once that day
daily_mean_crps = average CRPS across that miner's scored prompts that day
```

Persistent leader cohort:

```text
top 25 miners by top25_day_rate over the analysis window
```

Performance metric:

```text
CRPS advantage = daily median CRPS of all miners - cohort mean CRPS
```

Interpretation:

```text
positive advantage = persistent cohort beat the daily median miner
negative advantage = persistent cohort underperformed the daily median miner
```

Market regimes:

```text
realized volatility = std(5m log returns) * sqrt(number of bars)
volatility-of-volatility = std(vol_1h)
trend efficiency = abs(total log return) / sum(abs(5m log returns))
direction = bullish if return > +0.5%, bearish if return < -0.5%, else flat
```

Low/mid/high regimes use terciles inside the analysis window.

## BTC Findings

Most persistent BTC top-25 miners:

| UID | Top-25 Days | Rate | Median Best Rank | Mean Advantage | Reward Rank |
|---:|---:|---:|---:|---:|---:|
| 151 | 65 | 71.4% | 11.0 | +8.81 | 105 |
| 63 | 63 | 69.2% | 7.0 | +16.54 | 99 |
| 225 | 63 | 69.2% | 7.0 | +16.03 | 94 |
| 201 | 63 | 69.2% | 9.0 | +15.64 | 91 |
| 216 | 63 | 69.2% | 7.0 | +15.56 | 103 |
| 141 | 63 | 69.2% | 10.5 | +3.13 | 142 |
| 236 | 63 | 69.2% | 9.0 | +0.54 | 129 |

BTC daily regimes:

| Regime | Days | Avg Advantage | Avg Top-25 Miners | Avg Best Rank |
|---|---:|---:|---:|---:|
| Low vol | 31 | +44.18 | 18.55 | 12.15 |
| Mid vol | 30 | -13.14 | 16.00 | 29.50 |
| High vol | 30 | -2.17 | 16.23 | 22.53 |
| Low vol-of-vol | 31 | +32.30 | 18.45 | 14.76 |
| High vol-of-vol | 30 | -15.98 | 15.77 | 22.63 |
| Flat day | 22 | +24.10 | 17.18 | 21.95 |
| Bearish day | 36 | +6.70 | 16.14 | 22.92 |
| Bullish day | 33 | +4.21 | 17.67 | 19.08 |
| Mean-reverting | 31 | +17.84 | 18.13 | 17.35 |
| Trending | 30 | +4.21 | 15.10 | 28.40 |

BTC session volatility:

| Session | Low Vol Advantage | Mid Vol Advantage | High Vol Advantage |
|---|---:|---:|---:|
| EU | +13.98 | +3.02 | -30.10 |
| EU-US overlap | +8.83 | -1.00 | -24.98 |
| Outside market hours | -0.77 | -0.67 | -11.84 |
| US | +14.82 | -15.32 | -16.94 |
| Weekend | +60.11 | +39.48 | +38.34 |

BTC interpretation:

Persistent BTC miners do best on low-volatility, low-vol-of-vol, flatter, and
mean-reverting days. Their edge weakens or turns negative when session
volatility rises, especially during EU, overlap, and US sessions.

## ETH Findings

Most persistent ETH top-25 miners:

| UID | Top-25 Days | Rate | Median Best Rank | Mean Advantage | Reward Rank |
|---:|---:|---:|---:|---:|---:|
| 158 | 68 | 74.7% | 7.0 | +21.52 | 85 |
| 201 | 68 | 74.7% | 7.0 | +21.45 | 91 |
| 108 | 68 | 74.7% | 8.0 | +9.64 | 127 |
| 59 | 68 | 74.7% | 10.0 | -1.83 | 30 |
| 73 | 67 | 73.6% | 8.0 | +21.58 | 88 |
| 84 | 67 | 73.6% | 9.0 | +21.54 | 82 |
| 132 | 67 | 73.6% | 7.0 | +21.52 | 90 |

ETH daily regimes:

| Regime | Days | Avg Advantage | Avg Top-25 Miners | Avg Best Rank |
|---|---:|---:|---:|---:|
| Low vol | 31 | +38.77 | 17.42 | 16.55 |
| Mid vol | 30 | +20.77 | 17.57 | 16.90 |
| High vol | 30 | -7.78 | 19.07 | 15.83 |
| Low vol-of-vol | 31 | +33.57 | 17.45 | 15.03 |
| High vol-of-vol | 30 | -28.43 | 18.83 | 15.33 |
| Flat day | 19 | +49.01 | 18.05 | 15.26 |
| Bullish day | 32 | +14.91 | 18.22 | 14.00 |
| Bearish day | 40 | +4.57 | 17.83 | 18.93 |
| Mean-reverting | 31 | +29.18 | 17.68 | 16.87 |
| Trending | 30 | +4.03 | 17.90 | 17.40 |

ETH session volatility:

| Session | Low Vol Advantage | Mid Vol Advantage | High Vol Advantage |
|---|---:|---:|---:|
| EU | +36.81 | -27.81 | -6.08 |
| EU-US overlap | +35.53 | +26.75 | -58.96 |
| Outside market hours | +9.45 | +44.56 | -47.80 |
| US | +19.32 | +37.35 | -52.57 |
| Weekend | +80.65 | +22.25 | +66.17 |

ETH interpretation:

The same broad pattern appears on ETH, but with stronger sensitivity to
high-volatility sessions. Persistent ETH miners are materially worse when
volatility and volatility-of-volatility are high, especially in overlap,
outside-market-hours, and US sessions.

## Modeling Implications

The persistent leaderboard miners appear strongest in "normal" regimes:

```text
low volatility
low volatility-of-volatility
flat or mean-reverting daily structure
less violent session transitions
```

They lose edge in regimes with:

```text
high volatility-of-volatility
high session volatility
rising volatility
tail-heavy behavior
strong trend efficiency
```

This suggests a model improvement direction:

1. Build regime classifiers for volatility, vol-of-vol, trend efficiency, and
   session-specific volatility.
2. Use separate path libraries/calibration for quiet, mean-reverting days vs
   volatile, tail-heavy days.
3. Focus private-model research on regimes where persistent miners give back
   edge rather than copying their average behavior.
4. Add session-transition features, especially:
   - quiet EU to volatile US
   - quiet outside-market-hours to volatile US
   - high overlap volatility

## Repeatable Command

Set the Polygon key in the environment only:

```bash
export POLYGON_API_KEY='your_polygon_key_here'
```

Run the default 90-day BTC/ETH study:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --days 90 \
  --assets BTC ETH \
  --top-n 25
```

Run a longer study:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --days 180 \
  --assets BTC ETH \
  --top-n 25 \
  --output-dir data/reports/top_miners_research_180d \
  --synth-timeout-seconds 120 \
  --max-retries 6 \
  --retry-sleep-seconds 10
```

If only Synth score consistency is needed and Polygon features can be skipped:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --days 180 \
  --assets BTC ETH \
  --top-n 25 \
  --skip-polygon
```

Probe Synth equity/commodity prompt coverage. As of the latest local probe,
`XAU` returned active prompts while common equity tickers such as `SPY`, `QQQ`,
`AAPL`, `MSFT`, `NVDA`, and `TSLA` returned no prompts for the sampled day:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --discover-synth-equities \
  --discover-only \
  --equities \
  --skip-polygon \
  --output-dir data/reports/synth_equity_coverage
```

Smoke-test Polygon 1-minute data for the active equity/commodity asset before a
long analysis. The smoke output includes row count, first/last timestamp,
minimum/median spacing, one-minute gap rate, and any error. The longer lookback
helps avoid false negatives around equity market closures:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --equities \
  --assets XAU \
  --days 1 \
  --polygon-minute-smoke \
  --polygon-smoke-lookback-hours 72 \
  --polygon-smoke-only \
  --output-dir data/reports/xau_minute_smoke
```

Configured assets such as `XAU` use `config/default.yaml`. Research-only equity
symbols discovered by the probe can also be passed directly through `--assets`;
the script dynamically maps them to Synth `com-equ-24h` and uses the same symbol
as the Polygon ticker unless an override is defined.

Run a full XAU miner-performance study with score-level market-state features:

```bash
.venv/bin/python scripts/top_miners_regime_research.py \
  --equities \
  --assets XAU \
  --days 180 \
  --top-n 25 \
  --output-dir data/reports/top_miners_xau_research_180d \
  --synth-timeout-seconds 120 \
  --max-retries 6 \
  --retry-sleep-seconds 10
```

Generated outputs include:

```text
*_historical_scores.csv
*_daily_miner_crps.csv
*_miner_consistency.csv
*_market_state_features_1m.csv
*_score_feature_rows.csv
*_feature_bucket_performance.csv
*_miner_feature_consistency.csv
*_score_feature_correlations.csv
*_full_features_5m.csv
*_daily_regimes.csv
*_session_regimes.csv
*_vol_regime_performance.csv
*_vol_of_vol_regime_performance.csv
*_direction_regime_performance.csv
*_trend_regime_performance.csv
*_session_vol_performance.csv
*_session_composite_performance.csv
*_feature_correlations.csv
summary.json
```

The score-level feature rows match each Synth score to the market state at
`scored_time - 24h`, i.e. the forecast origin time. Features are calculated from
Polygon 1-minute bars:

```text
momentum_1h,  momentum_3h,  momentum_5h,  momentum_8h,  momentum_24h
realized_vol_1h, realized_vol_3h, realized_vol_5h, realized_vol_8h, realized_vol_24h
vol_of_vol_1h, vol_of_vol_3h, vol_of_vol_5h, vol_of_vol_8h, vol_of_vol_24h
```

Outputs are written under `data/reports/...`, which is ignored by git.
