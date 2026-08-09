# Time-series forecasting (`crates/chronos2`, `crates/fincast`, `crates/kronos`)

Three foundation models — Chronos-2, FinCast, Kronos — share one contract: a
numeric context series in, a probabilistic forecast out. This page covers the
shared CLI/D-Bus surface; each model's own page has its exact weights/env var
and any model-specific behavior.

- [Chronos-2](../chronos2/readme.md) — universal probabilistic forecaster
- [FinCast](../fincast/readme.md) — financial forecaster, research/educational licence
- [Kronos](../kronos/readme.md) — financial forecaster, the only one with CLI-reachable fine-tune

## CLI: `brain forecast <compare|serve|import|finetune>`

One binary subcommand serves all three (`crates/cli/src/forecast_cli.rs`):

- **`compare`** — run the scenario battery against statistical baselines + any
  loaded foundation models: `--windows 24 --seed 1337 [--html <path>]
  [--chronos2 <weights>] [--kronos-tokenizer <dir> --kronos-decoder <dir>]
  [--fincast <weights>]`. Exits 1 if a model falsely beats the random-walk
  negative control.
- **`serve`** — start the unified JSONL server: stdio by default, or
  `--socket <path>` / `--listen <addr>`, `--max-connections 64`, plus the same
  `--chronos2` / `--kronos-tokenizer --kronos-decoder` / `--fincast` weight flags.
- **`import`** — convert a foundation checkpoint into a brain `.safetensors`
  container: `--hf <amazon/chronos-2 dir> --out chronos2.safetensors`, or
  `--fincast <FinCast safetensors> --out fincast.safetensors`.
- **`finetune`** — Kronos-only weekly gated fine-tune; see
  [`kronos/readme.md`](../kronos/readme.md).

## D-Bus: the shared `forecast` action

Every resident (`crates/cli/src/resident_forecast.rs`) exposes one `forecast`
action: input blob `context` (raw f32-LE, meta `{"shape":[...]}`, required),
param `horizon` (int, default 64), output blob `forecast` (raw f32-LE, meta
`{shape, kind, levels}`). Registered in `crates/cli/src/resident.rs`, gated on
the model's env var(s) being set. Not registered with `apiserve` — no HTTP
route for any of the three (the action is named `forecast`, not `generate`;
none are `embed`-shaped or pure-text2image-shaped).

Reference client (all three models, one script):
[`examples/forecast/forecast_client.py`](../../../examples/forecast/forecast_client.py)
— `--model brain/chronos2|brain/fincast|brain/kronos`. See
[`examples/forecast/README.md`](../../../examples/forecast/README.md) for
runnable `dbus-run-session` invocations per model.

`chronos2` and `fincast` advertise an NPU footprint and are auto-placed on the
NPU when one is budgeted (`place::pick_device`); `kronos` runs its
autoregressive rollout on CPU/GPU via a two-graph KV-cache, also NPU-eligible.

## The three models

| | Chronos-2 | FinCast | Kronos |
|---|---|---|---|
| id / env | `brain/chronos2` / `BRAIN_CHRONOS2` | `brain/fincast` / `BRAIN_FINCAST` | `brain/kronos` / `BRAIN_KRONOS_TOKENIZER`+`BRAIN_KRONOS_DECODER` |
| architecture | encoder-only T5-style patch transformer | TimesFM-style patched decoder + sparse top-2 MoE | BSQ-tokenized autoregressive decoder, dual head |
| output | `[21, horizon]` quantiles | `[horizon, 1+9]` mean + quantiles | `[horizon, feat]` sampled OHLCV bars |
| training/finetune | none (inference-only today) | trainer exists, no CLI verb | full fine-tune + LoRA, CLI-reachable |
| licence note | Apache-2.0 | research/educational use only | MIT |

## See also

- Ledger: [`docs/models/forecast/status.md`](status.md) — NPU placement, measured
  latencies, perf-gate targets
- Statistical baselines + scenario backtester: `crates/fcbench`
- Serving contract: `docs/serving-contract.md`
