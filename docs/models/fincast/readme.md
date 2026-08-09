# FinCast (`crates/fincast`)

A from-scratch Rust+WGSL reimplementation of FinCast — a ~1B-param TimesFM-style
patched decoder with a sparse top-2 mixture-of-experts and a probabilistic
quantile head, imported exactly from `Vincent05R/FinCast` and parity-gated to
cosine 1.0 against the real 991M-param weights.

**Licence:** the reference is Apache-2.0, but the authors state the model is
"for research and educational purposes only" and "does not constitute
financial advice" — see `status.md`. Flagged, not blocked.

## Model id and weights

- **Id:** `brain/fincast` — reserved vendor `brain/`, never fetched; weights
  are imported explicitly (below), not `brain_modelstore`-fetched.
- **Weights:** `BRAIN_FINCAST` — path to a single brain `.safetensors` file
  produced by `import` (not a directory, not the raw `v1.pth`).
- **Import:** the raw checkpoint (`v1.pth`, torch pickle) must first be
  converted with `tools/convert/fincast_convert.py <v1.pth> <out.safetensors>`,
  then: `brain forecast import --fincast <FinCast safetensors> --out fincast.safetensors`

## Surfaces

D-Bus (via `brain forecast serve`) and the `brain forecast compare|serve` CLI
only — no HTTP route (the action is named `forecast`, not `generate`, and it
requires an input blob so it isn't `text2image`-shaped either).

## Inference

### CLI

```bash
brain forecast compare --fincast fincast.safetensors [--windows 24] [--seed 1337]
brain forecast serve --fincast fincast.safetensors [--socket <path> | --listen <addr>]
```

There is no `brain fincast ...` subcommand — `forecast` is the only CLI verb.

### D-Bus

Action `forecast`: input `context` (f32-LE, meta `{shape}`, required), params
`horizon` (int, default 64) and `freq` (int, default 0: 0 daily / 1 weekly /
2 monthly), output `forecast` (f32-LE, meta `{shape:[horizon,10],
kind:"mean+quantiles", levels:[9 values]}`, column 0 = mean). Auto-placed on
the NPU when one is budgeted; the response's `device` field reports where it
actually ran.

```bash
BRAIN_FINCAST=/path/to/fincast.safetensors dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/forecast/forecast_client.py --model brain/fincast --freq 0
'
```

Reference client: [`examples/forecast/forecast_client.py`](../../../examples/forecast/forecast_client.py)

## Training

A real gradient-checked trainer exists in the crate
(`crates/fincast/src/train.rs` — host forward+backward of the full core,
gradcheck green, from-scratch learning test) but it is not exposed by any CLI
verb.

## Not supported

finetune (CLI), LoRA, QLoRA. Reference weights are research/educational use
only, not for financial advice.

## See also

- Crate: `crates/fincast`
- Workstream ledger: [`status.md`](status.md)
- Umbrella page: [`../forecast/readme.md`](../forecast/readme.md)
