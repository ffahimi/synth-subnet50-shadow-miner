# Synth Subnet 50 Shadow Miner

Shadow forecasting prototype for Synth subnet 50. The first milestone is a
Polygon-only BTC 24h forecaster that generates 1,000 simulated paths at
5-minute resolution.

The repo is intentionally organized around the shadow workflow first:

1. Fetch recent BTC data from Polygon.
2. Split BTC history into liquidity sessions.
3. Compute recent volatility, volatility of volatility, volatility slope,
   momentum, and kurtosis on 1h and 4h windows.
4. Build a normalized library of historical session path shapes.
5. Rescale those shapes with the current regime and generate 24h paths.
6. Store immutable forecasts for later Synth realized-path scoring.

Live Bittensor miner integration comes later, after the shadow scorer proves
that the forecast distribution is competitive.

## Package Layout

```text
src/synth_shadow/
  data/          Polygon adapter and canonical bar schema
  features/      returns, volatility, momentum, kurtosis, feature pipeline
  sessions/      EU, EU-US, US, outside-hours, and weekend classifiers
  models/        current regime state, session library, and path sampler
  paths/         path normalization, generation, and validation
  scoring/       CRPS and Synth-style score helpers
  storage/       forecast and metadata persistence
  utils/         time and logging helpers
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your Polygon key in `.env`:

```bash
POLYGON_API_KEY=...
```

Run the first BTC pipeline:

```bash
synth-shadow generate-btc --debug
```

Run the first full Synth shadow version:

```bash
synth-shadow run-synth-shadow --debug
```

This executes the implemented blocks together:

```text
sync Synth BTC prompts
generate a Polygon BTC forecast for the latest prompt
inspect generated paths and percentile bands
try to score any matured forecasts
fetch latest Synth score/reward/leaderboard context
```

Individual debug commands:

```bash
synth-shadow sync-prompts --debug
synth-shadow generate-latest-prompt --debug
synth-shadow inspect-latest --debug
synth-shadow score-matured --debug
synth-shadow benchmarks --debug
synth-shadow backtest-rolling --debug --backtest-max-origins 12
```

Historical rolling backtest:

```bash
# Quick smoke test: 12 origins, configured path count
synth-shadow backtest-rolling --debug --backtest-max-origins 12

# Full default: last 1 matured day, one origin every 5 minutes
synth-shadow backtest-rolling --debug

# Heavier full-path check with 1000 paths per origin
synth-shadow backtest-rolling --debug --backtest-num-paths 1000
```

The backtest uses only Polygon bars before each origin to build the model, then
scores the generated 24h path against the next 24h of Polygon realized closes.

Outputs are written under:

```text
data/raw/          raw Polygon 5-minute bars
data/processed/    repaired bars with sessions and features
data/forecasts/    paths.npz, timestamps.csv, metadata.json, features.json
data/realized/     Synth realized paths when matured forecasts can be scored
data/reports/      latest forecast summary JSON
data/backtests/    rolling historical backtest CSV + summary
data/registry.sqlite3  local prompt/forecast/score registry
```

## First Target

```text
asset: BTC
ticker: X:BTCUSD
history: 7-14 days
forecast horizon: 24h
interval: 5 minutes
paths: 1000
path length: 289 prices
```
