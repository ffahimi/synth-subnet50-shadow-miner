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
public forecast model interface with private model entrypoint support
1000 x 289 BTC 24h path generation
forecast inspection and percentile summaries
Synth prompt sync
Synth latest score + leaderboard fetch
CRPS in basis points, matching Synth's cross-asset score units
CRPS/reward benchmark join by miner UID
SQLite local forecast registry
rolling historical Polygon backtest
```

## Current Public Harness Architecture

This repo is the public shadow-miner harness. It is responsible for running the
Synth-facing workflow and validating forecast artifacts. It is not meant to hold
private model alpha once the model is split into a private inference node.

Current responsibilities:

```text
sync Synth prompts
request or generate one forecast per cycle
validate returned paths and timestamps
save forecast artifacts under data/forecasts/<ASSET>/<timestamp>/
register forecasts in SQLite
inspect path percentiles
score only matured pending forecasts
fetch public miner CRPS/leaderboard benchmarks
print live sanity/latency diagnostics
```

Forecast provider modes:

```text
local provider:
  public repo fetches Polygon 5m bars, builds public features, runs baseline model

HTTP provider:
  public repo calls SYNTH_MODEL_ENDPOINT once per forecast cycle
  private service owns data access, 1m BTC data, vectorization, matching, and forecasting
```

The live loop should normally sleep `300` seconds between cycles. With
`SYNTH_MODEL_ENDPOINT` set, this means the private inference node receives one
`POST /predict` request per 5-minute cycle.

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
  models/          model interface, loader, public baseline, state/sampler tools
  paths/           path generation and validation
  inspection/      forecast summaries and sample path reports
  synth/           public Synth API client
  scoring/         CRPS, Synth-style comparisons, benchmark joins
  backtest/        rolling historical Polygon-realized backtest
  orchestration/   full shadow-cycle runner
  storage/         forecast files and SQLite registry
  utils/           time and logging helpers

example-inference-node/
  public FastAPI example of a private HTTP model node
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
# Optional: override public baseline with an installed private model package.
# SYNTH_MODEL_ENTRYPOINT=private_synth_models.eth_v1:Model
EOF
```

Do not commit `.env` or a real Polygon key. The repo ignores `.env`, and the
CLI treats placeholder values such as `your_polygon_key_here` as missing. If
`POLYGON_API_KEY` is not loaded and the command is running in an interactive
terminal, it prompts:

```text
Enter POLYGON_API_KEY:
```

For long-running `screen` jobs, export the key or create `.env` before starting
the loop:

```bash
export POLYGON_API_KEY='your_real_polygon_key'
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

Run one live forecast with detailed sanity diagnostics:

```bash
.venv/bin/python -m synth_shadow.cli forecast-sanity --asset BTC --debug
```

For a faster latency probe with fewer paths:

```bash
.venv/bin/python -m synth_shadow.cli forecast-sanity --asset BTC --debug --num-paths 8
```

The sanity output prints:

```text
stage latency in seconds: Polygon fetch, repair, features, library, generation, save, registry
raw/repaired/feature row counts
raw and feature date ranges
5-minute resolution checks
causality checks that features and bars stop at the prediction timestamp
path shape, timestamp count, timestamp spacing, finite/positive checks
current price alignment and final-path percentiles
```

`forecast-sanity` saves and registers the forecast. For live CRPS scoring
against Synth realized paths, prefer the prompt-aligned full cycle below.
Standalone sanity forecasts are registered with `debug` status unless a
prompt start time is supplied by the orchestration layer.

## Public Repo / Private Model Split

This repository is designed to be safe as a public harness. Keep data adapters,
feature extraction, scoring, backtests, storage, and orchestration here. Keep
forecast edge, experiments, and model-specific parameters in a separate private
repo.

The public baseline model is configured in `config/default.yaml`:

```yaml
model:
  entrypoint: synth_shadow.models.baseline:SessionPathBaselineModel
  endpoint:
  timeout_seconds: 120
```

There are two private model modes.

### Private HTTP Inference Node

For live deployment, prefer an HTTP inference node. In this mode the private
model service owns data access, 1-minute BTC history, feature/vector creation,
similarity matching, calibration, and path generation. The public harness only
requests a forecast, validates/saves it, scores matured forecasts, and handles
loop resiliency.

This repo includes a simple wrapper example in `example-inference-node/`. It is
safe to keep public because it is only a scaffold: it uses Polygon 1-minute REST
data and a placeholder similarity/bootstrap generator. Copy that folder into a
separate private repo before adding real model logic.

To try the example locally:

```bash
cd example-inference-node
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export POLYGON_API_KEY=your_polygon_key_here
uvicorn private_synth_models.server:app --host 127.0.0.1 --port 8088
```

Then in the public shadow miner:

```bash
export SYNTH_MODEL_ENDPOINT=http://127.0.0.1:8088/predict
.venv/bin/python -m synth_shadow.cli run-synth-shadow-sanity --asset BTC --debug
```

To turn the example into a private repo:

```bash
cd /Users/ffahimi/Documents/Code
cp -R synth-subnet50-shadow-miner/example-inference-node synth-btc-inference-private
cd synth-btc-inference-private
git init
git branch -M main
git add .
git commit -m "Initial private BTC inference node"
git remote add origin git@github.com:ffahimi/synth-btc-inference-private.git
git push -u origin main
```

Keep that GitHub repo private and replace the placeholder data/features/model
modules there.

Configure it with:

```bash
export SYNTH_MODEL_ENDPOINT=http://127.0.0.1:8088/predict
export SYNTH_SHADOW_CONFIG=config/default.yaml
export LOG_LEVEL=DEBUG
```

The public harness sends one request per forecast cycle.

Endpoint:

```text
POST /predict
Content-Type: application/json
```

Request body:

```json
{
  "asset": "BTC",
  "polygon_ticker": "X:BTCUSD",
  "prompt_start_time": "2026-07-12T16:29:00Z",
  "horizon_seconds": 86400,
  "interval_seconds": 300,
  "num_paths": 1000,
  "generated_at": "..."
}
```

Request field meanings:

```text
asset: Synth asset symbol, for example BTC
polygon_ticker: public ticker mapping, for example X:BTCUSD
prompt_start_time: latest synced Synth prompt start, nullable
horizon_seconds: forecast horizon; currently 86400
interval_seconds: output interval; currently 300
num_paths: number of probabilistic paths requested; default 1000
generated_at: public harness request time
```

Expected response from the private node:

```json
{
  "asset": "BTC",
  "model_version": "private_btc_similarity_v1",
  "data_cutoff": "2026-07-12T16:30:00Z",
  "current_price": 64124.29,
  "paths": [[64124.29, 64120.1, "... 289 points total"]],
  "timestamps": ["2026-07-12T16:30:00Z", "... 289 timestamps total"],
  "diagnostics": {
    "data_source": "private_1m_btc_store",
    "num_raw_bars": 250000,
    "num_feature_rows": 249000,
    "nearest_neighbors": 64
  },
  "metadata": {
    "notes": "optional private diagnostics"
  }
}
```

Required response fields:

```text
data_cutoff: UTC timestamp for the first forecast point
current_price: price at data_cutoff; all paths must start here
paths: list[list[float]], shape num_paths x 289
```

Optional response fields:

```text
asset
model_version
timestamps
diagnostics
metadata
feature_snapshot
```

`timestamps` may be omitted if `data_cutoff` is supplied. In that case the
public harness builds `289` timestamps at 5-minute resolution from
`data_cutoff`.

The public harness validates:

```text
paths shape == num_paths x 289
timestamp count == 289
timestamp spacing == 300 seconds
prices are finite and positive
first timestamp equals data_cutoff
first path price equals current_price
```

The public harness saves this response as a normal forecast run:

```text
paths.npz          compressed path matrix
timestamps.csv    289 output timestamps
metadata.json     model_version, model_entrypoint/endpoint, data_cutoff, diagnostics
features.json     feature_snapshot if supplied, otherwise a minimal provider snapshot
```

If `prompt_start_time` is present, the forecast is registered as `pending` and
will be eligible for CRPS scoring after the 24h horizon matures. Standalone
sanity forecasts without a prompt are registered as `debug`.

### Private In-Process Package

For research/dev, you can still install a private Python package into the same
virtualenv and point the harness at its import path:

```bash
cd ~
git clone git@github.com:ffahimi/synth-subnet50-models-private.git
cd synth-subnet50-models-private
../synth-subnet50-shadow-miner/.venv/bin/python -m pip install -e .

cd ../synth-subnet50-shadow-miner
export SYNTH_MODEL_ENTRYPOINT=private_synth_models.eth_v1:Model
export POLYGON_API_KEY=your_polygon_key_here
export SYNTH_SHADOW_CONFIG=config/default.yaml
export LOG_LEVEL=DEBUG
```

Then run the normal public commands:

```bash
.venv/bin/python -m synth_shadow.cli generate-forecast --asset ETH --debug
.venv/bin/python -m synth_shadow.cli backtest-rolling \
  --asset ETH \
  --debug \
  --backtest-days 2 \
  --backtest-stride-minutes 60 \
  --backtest-num-paths 250
```

The forecast metadata and backtest summary include:

```text
model_version
model_entrypoint
```

Private model contract:

```python
from synth_shadow.models.protocol import ForecastContext, ForecastOutput


class Model:
    model_version = "eth_private_v1"

    def generate(self, context: ForecastContext) -> ForecastOutput:
        # Causal inputs only:
        # context.bars      -> bars up to the live cutoff/backtest origin
        # context.features  -> features up to the live cutoff/backtest origin
        # context.library   -> public session-path library built from past data
        # context.state     -> current state at the cutoff/origin
        # context.sampler   -> public sampler seeded by the harness
        # context.config    -> runtime config
        # context.origin    -> backtest origin, or None for latest live forecast
        paths, timestamps = your_private_generation_logic(context)
        return ForecastOutput(
            paths=paths,
            timestamps=timestamps,
            metadata={"notes": "private diagnostics are optional"},
        )
```

Expected output shape remains:

```text
paths: num_paths x 289
timestamps: 289 UTC timestamps at 5-minute resolution
```

In rolling backtests, the public harness passes only `bars` and `features` whose
timestamps are `<= origin` into the private model context. The realized future
path is kept outside the model and is only sent to the scorer.

Before making this repo public, keep these out of git:

```gitignore
.env
private_models/
models_private/
local_models/
*.secret.yaml
```

Do not commit private model packages as subdirectories of this repo. Install
them into the virtualenv as separate private packages instead.

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

Run the same full cycle with forecast sanity diagnostics printed every time:

```bash
.venv/bin/python -m synth_shadow.cli run-synth-shadow-sanity --asset BTC --debug
```

This does:

```text
1. sync Synth BTC prompts
2. generate a prompt-aligned forecast tagged to the latest prompt
3. inspect generated path percentiles
4. try to score any matured pending forecasts
5. fetch latest Synth BTC miner scores
6. fetch latest crypto-24h leaderboard
7. join CRPS and rewards by miner UID
```

The sanity version also prints live forecast latency and path checks before it
tries to score matured pending forecasts. With `SYNTH_MODEL_ENDPOINT` set, it
calls the private HTTP model node once per cycle and prints the diagnostics
returned by that node. Without `SYNTH_MODEL_ENDPOINT`, it uses the public local
baseline.

Fresh forecasts usually cannot be scored immediately because Synth's realized
path is only available after the 24h horizon has matured. In that case
`score-matured` logs a 404 warning and continues.

Scoring is throttled:

```text
only forecasts older than horizon + maturity_grace_seconds are attempted
default maturity_grace_seconds: 300
default max_matured_score_attempts_per_cycle: 3
Synth 429 stops scoring attempts for that cycle
```

Useful individual commands:

```bash
synth-shadow sync-prompts --debug
synth-shadow generate-latest-prompt --asset ETH --debug
synth-shadow generate-sanity-latest-prompt --asset BTC --debug
synth-shadow score-matured --debug
synth-shadow benchmarks --asset XAU --debug
```

Example `screen` loop for a one-day BTC shadow run:

```bash
while true; do
  echo "===== BTC live sanity cycle $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  .venv/bin/python -m synth_shadow.cli run-synth-shadow-sanity --asset BTC --debug
  sleep 300
done 2>&1 | tee -a logs/btc_live_sanity_cycle.log
```

The loop calls the model provider once, then sleeps for 300 seconds. If the
private HTTP node is configured, that means one `/predict` call per 5-minute
cycle.

With HTTP mode enabled, the sanity output changes slightly:

```text
provider_generate: latency of private /predict call
validate_paths: public shape/finite/positive validation
path sanity checks: timestamp spacing, first timestamp, first price, final percentiles
data sanity checks: diagnostics returned by private inference node
```

The public harness does not inspect private feature vectors in HTTP mode. The
private node should include its own diagnostics, for example raw 1m row count,
feature row count, data cutoff, nearest-neighbor count, and model latency.

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

CRPS components are scored on price changes in basis points, not raw dollar
price changes. That keeps BTC, ETH, XAU, and other assets on the same unit
scale, matching Synth's documented scoring methodology.

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
