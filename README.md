# Synth Subnet 50 Shadow Miner

Shadow forecasting and scoring prototype for Synth subnet 50. The current
version is Polygon-first and supports `BTC`, `ETH`, and `XAU`: it builds
24-hour probabilistic paths at 5-minute resolution, stores the forecast
artifacts, syncs public Synth validation context, and can run a rolling
historical backtest.

Live Bittensor miner integration is intentionally not part of this first layer.
The goal is to test whether the forecast distribution is competitive before
wrapping it in a live miner.

## What Works Now

```text
Polygon 5-minute REST data fetch for BTC, ETH, XAU
canonical OHLCV repair/checks
liquidity session labels
1h/4h volatility, vol-of-vol, vol slope, momentum, kurtosis
historical normalized session-path library
1000 x 289 BTC 24h path generation
forecast inspection and percentile summaries
Synth prompt sync
Synth latest score + leaderboard fetch
CRPS/reward benchmark join by miner UID
SQLite local forecast registry
rolling historical Polygon backtest
```

Supported assets:

```text
BTC -> Polygon X:BTCUSD -> Synth BTC -> crypto-24h
ETH -> Polygon X:ETHUSD -> Synth ETH -> crypto-24h
XAU -> Polygon C:XAUUSD -> Synth XAU -> com-equ-24h
```

The main forecast target per asset is:

```text
forecast horizon: 24h
interval: 5 minutes
paths: 1000
path length: 289 prices
```

## Repository Layout

```text
src/synth_shadow/
  data/            Polygon adapter and canonical bar schema
  features/        returns, volatility, momentum, kurtosis, feature pipeline
  sessions/        EU, EU-US, US, outside-hours, weekend classifiers
  models/          current state, session library, and path sampler
  paths/           path generation and validation
  inspection/      forecast summaries and sample path reports
  synth/           public Synth API client
  scoring/         CRPS, Synth-style comparisons, benchmark joins
  backtest/        rolling historical Polygon-realized backtest
  orchestration/   full shadow-cycle runner
  storage/         forecast files and SQLite registry
  utils/           time and logging helpers
```

## Setup

Clone and install:

```bash
git clone git@github.com:ffahimi/synth-subnet50-shadow-miner.git
cd synth-subnet50-shadow-miner

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create `.env`:

```bash
cat > .env <<'EOF'
POLYGON_API_KEY=your_polygon_key_here
SYNTH_SHADOW_CONFIG=config/default.yaml
LOG_LEVEL=DEBUG
EOF
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Select An Asset

All commands default to `BTC`. Use `--asset` to run another supported asset:

```bash
--asset BTC
--asset ETH
--asset XAU
```

## Generate A New Forecast

Run a standalone Polygon forecast:

```bash
synth-shadow generate-btc --debug
```

Equivalent generic command:

```bash
synth-shadow generate-forecast --asset BTC --debug
```

ETH:

```bash
synth-shadow generate-forecast --asset ETH --debug
```

XAU:

```bash
synth-shadow generate-forecast --asset XAU --debug
```

This fetches recent Polygon bars for the selected asset, extracts features, builds the session
library, generates `1000 x 289` paths, saves files under `data/forecasts/<ASSET>/`,
and registers the run in `data/registry.sqlite3`.

Inspect the latest forecast:

```bash
synth-shadow inspect-latest --debug
```

The inspection output includes:

```text
path shape
t0, 1h, 4h, 12h, 24h percentile bands
final 24h return distribution
sample generated paths
```

You can also inspect a specific forecast directory:

```bash
synth-shadow inspect-latest \
  --forecast-dir data/forecasts/<ASSET>/<forecast_timestamp> \
  --debug
```

## Run The Full Shadow Cycle

Run all implemented modules together:

```bash
synth-shadow run-synth-shadow --debug
```

This does:

```text
1. sync Synth BTC prompts
2. generate a Polygon forecast tagged to the latest prompt
3. inspect generated path percentiles
4. try to score any matured pending forecasts
5. fetch latest Synth BTC miner scores
6. fetch latest crypto-24h leaderboard
7. join CRPS and rewards by miner UID
```

Fresh forecasts usually cannot be scored immediately because Synth's realized
path is only available after the 24h horizon has matured. In that case
`score-matured` logs a 404 warning and continues.

Useful individual commands:

```bash
synth-shadow sync-prompts --debug
synth-shadow generate-latest-prompt --asset ETH --debug
synth-shadow score-matured --debug
synth-shadow benchmarks --asset XAU --debug
```

## CRPS And Reward Benchmarks

Run:

```bash
synth-shadow benchmarks --debug
```

The benchmark command fetches:

```text
/validation/scores/latest
/v2/leaderboard/latest
```

Then it joins rows by:

```text
score.miner_uid == leaderboard.neuron_uid
```

The joined output is under `crps_reward_top10` and includes:

```text
rank_by_crps
miner_uid
crps
prompt_score
scored_time
reward
leaderboard_updated_at
```

Invalid sentinel CRPS values such as `-1` are filtered out before ranking.

## Rolling Historical Backtest

Run a quick smoke test:

```bash
synth-shadow backtest-rolling --debug --backtest-max-origins 12
```

Backtest ETH:

```bash
synth-shadow backtest-rolling --asset ETH --debug --backtest-max-origins 12
```

Backtest XAU:

```bash
synth-shadow backtest-rolling --asset XAU --debug --backtest-max-origins 12
```

Run the full default backtest:

```bash
synth-shadow backtest-rolling --debug
```

Default behavior:

```text
backtest window: last 1 matured day
origin stride: every 5 minutes
forecast horizon: 24h
realized source: Polygon close path
paths per origin: config default, currently 250 for backtest
```

The backtest is causal: for each forecast origin, it uses only Polygon bars at
or before that origin to build features and the session-path library. It then
generates a 24h forecast and scores it against the next 24h of realized Polygon
closes.

Run a heavier test with 1000 paths per origin:

```bash
synth-shadow backtest-rolling --debug --backtest-num-paths 1000
```

Run fewer origins while debugging:

```bash
synth-shadow backtest-rolling \
  --debug \
  --backtest-max-origins 3 \
  --backtest-num-paths 50
```

Run with a wider stride:

```bash
synth-shadow backtest-rolling \
  --debug \
  --backtest-stride-minutes 60
```

The backtest summary includes:

```text
raw_crps_mean
raw_crps_median
raw_crps_p25 / raw_crps_p75
final_error_mean
final_abs_error_median
top_reference_miner_crps
mean_reference_miner_crps
our_mean_minus_top_reference
our_median_minus_top_reference
miner_0_3_crps
```

`miner_0_3_crps` is the compact comparison requested for the top four valid
current Synth miners by CRPS, with reward context attached.

## Output Files

```text
data/raw/
  raw Polygon 5-minute bars

data/processed/
  repaired bars with sessions and features

data/forecasts/BTC/<timestamp>/
  paths.npz
  timestamps.csv
  metadata.json
  features.json

data/forecasts/ETH/<timestamp>/
data/forecasts/XAU/<timestamp>/

data/reports/
  latest_forecast_summary.json

data/backtests/<timestamp>/
  rolling_results.csv
  summary.json

data/backtests/BTC/<timestamp>/
data/backtests/ETH/<timestamp>/
data/backtests/XAU/<timestamp>/

data/realized/
  Synth realized paths once matured forecasts can be scored

data/registry.sqlite3
  local prompts, forecasts, and score registry
```

These generated outputs are ignored by Git except for `.gitkeep` files.

## Configuration

Main config lives in:

```text
config/default.yaml
```

Important sections:

```text
assets         Polygon ticker, Synth asset, competition mapping
forecast       horizon, interval, number of paths
history        Polygon lookback and bar size
features       1h/4h rolling windows
sessions       BTC liquidity session definitions
sampling       session block size
normalization  volatility/momentum/kurtosis scaling limits
synth          public Synth API settings
backtest       rolling backtest defaults
storage        output paths
inspection     checkpoint and sample-path settings
```

## Notes

- Polygon API keys belong in `.env`, not Git.
- The current model is a baseline session-resampled volatility projection.
- XAU uses Polygon `C:XAUUSD`; unlike crypto, it has market closures/gaps.
- Synth scoring here is a shadow approximation using documented CRPS scales.
- Single-asset testing does not estimate full competition emissions by itself.
- A fresh forecast must wait for the 24h realized path before official-like
  Synth realized-path scoring is available.
