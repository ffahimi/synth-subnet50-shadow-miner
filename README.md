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
