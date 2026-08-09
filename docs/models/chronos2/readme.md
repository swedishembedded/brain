# Chronos-2 (`crates/chronos2`)

A from-scratch Rust+WGSL reimplementation of Amazon's Chronos-2 — an
encoder-only, T5-style patch transformer giving univariate probabilistic
forecasts (21 quantile levels) from a raw context series, imported exactly
from `amazon/chronos-2` and parity-gated to cosine 1.0.

## Model id and weights

- **Id:** `brain/chronos2` — reserved vendor `brain/`, never fetched; weights
  are imported explicitly (below), not `brain_modelstore`-fetched.
- **Weights:** `BRAIN_CHRONOS2` — path to a single brain `.safetensors` file
  produced by `import` (not a directory, not the raw HF checkpoint).
- **Import:** `brain forecast import --hf <amazon/chronos-2 dir> --out chronos2.safetensors`

## Surfaces

D-Bus (via `brain forecast serve`) and the `brain forecast compare|serve` CLI
only — no HTTP route (the action is named `forecast`, not `generate`, and it
requires an input blob so it isn't `text2image`-shaped either).

## Inference

### CLI

```bash
brain forecast compare --chronos2 chronos2.safetensors [--windows 24] [--seed 1337] [--html report.html]
brain forecast serve --chronos2 chronos2.safetensors [--socket <path> | --listen <addr>]
```

There is no `brain chronos2 ...` subcommand — `forecast` is the only CLI verb.

### D-Bus

Action `forecast`: input `context` (f32-LE, meta `{shape}`, required), param
`horizon` (int, default 64), output `forecast` (f32-LE, meta
`{shape:[21,horizon], kind:"quantiles", levels:[21 values]}`). Auto-placed on
the NPU when one is budgeted (`MemCost::with_npu`); the response's `device`
field reports where it actually ran.

```bash
BRAIN_CHRONOS2=/path/to/chronos2.safetensors dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/forecast/forecast_client.py --model brain/chronos2 --horizon 64
'
```

Reference client: [`examples/forecast/forecast_client.py`](../../../examples/forecast/forecast_client.py)

## Not supported

training, finetune, LoRA, QLoRA — per `status.md`, the backward
(`build_backward` + `impl model::Model`) is not yet implemented; today's path
is inference-only.

## See also

- Crate: `crates/chronos2`
- Workstream ledger: [`status.md`](status.md)
- Umbrella page: [`../forecast/readme.md`](../forecast/readme.md)
