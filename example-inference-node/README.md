# Example Synth BTC Inference Node

This folder is a public example of the private HTTP inference-node contract used
by the shadow miner. It is intentionally simple and should be treated as a
wrapper/scaffold for private model development, not as a production model.

The public shadow miner calls the node through:

```bash
export SYNTH_MODEL_ENDPOINT=http://127.0.0.1:8088/predict
```

The node owns the model-side work:

- fetch latest BTC 1-minute bars from Polygon REST
- later swap Polygon REST for a local database behind the same data interface
- create 5-minute-compatible features
- run placeholder similarity/bootstrap path generation
- return diagnostics for the public harness

## Setup

```bash
cd example-inference-node
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a local `.env` file or export the key in your shell:

```bash
POLYGON_API_KEY=your_real_polygon_key
```

Do not commit `.env` or API keys.

## Run

```bash
uvicorn private_synth_models.server:app --host 127.0.0.1 --port 8088
```

Then in the public shadow miner repo:

```bash
export SYNTH_MODEL_ENDPOINT=http://127.0.0.1:8088/predict
.venv/bin/python -m synth_shadow.cli run-synth-shadow-sanity --asset BTC --debug
```

The public harness will call this node once per 5-minute cycle when your loop
sleeps for `300` seconds.

## Endpoint

`POST /predict`

```json
{
  "asset": "BTC",
  "polygon_ticker": "X:BTCUSD",
  "prompt_start_time": "2026-07-12T16:29:00Z",
  "horizon_seconds": 86400,
  "interval_seconds": 300,
  "num_paths": 1000,
  "generated_at": "2026-07-12T16:29:01Z"
}
```

The response returns `num_paths x 289` positive finite prices, with the first
timestamp equal to `data_cutoff` and every path starting at `current_price`.

## Tests

```bash
pytest
```

The tests mock Polygon/data fetching so they do not need a live API key.

## Make This A Private Model Repo

From the public shadow miner repo:

```bash
cd /Users/ffahimi/Documents/Code
cp -R synth-subnet50-shadow-miner/example-inference-node synth-btc-inference-private
cd synth-btc-inference-private
rm -rf .git
git init
git branch -M main
git add .
git commit -m "Initial private BTC inference node"
git remote add origin git@github.com:ffahimi/synth-btc-inference-private.git
git push -u origin main
```

Keep the GitHub repo private. Develop real model logic there:

```text
private_synth_models/data/
  replace Polygon REST with database/live cache adapters

private_synth_models/features/
  add 1-minute feature/vector pipeline

private_synth_models/models/
  implement similarity search, calibration, and path generation
```

The public repo should keep only this example scaffold and the HTTP contract.
