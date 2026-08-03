# PID event/effect Transformer in brain (`crates/pid`)

A control policy over CBOR-encoded event/effect records that imitates a
per-plant pole-placed velocity-PI **PID oracle**. Trained via DAgger — an
exploration policy drives each plant, and every visited state is relabeled with
the oracle's action — so the Transformer learns to pick the actuator bin at each
`DECIDE` token. It is the model behind the WebGPU browser demo (`crates/web`).

## Architecture

Pre-norm decoder block, a port of the reference
`event_effect_transformer_pid_v3.py` (`crates/pid/src/model.rs`):

- **Embeddings**: `x = tok_emb[idx] + pos_emb[pos]` — learned **absolute**
  positional embeddings (not RoPE).
- **LayerNorm with bias** (not RMSNorm), eps `1e-5`; `ln1`/`ln2` per block plus a
  final `ln`. Dispatched through the shared `model::block` LayerNorm family
  (`layernorm_fwd` / `ln_stats_fwd` / `layernorm_dx_bwd`).
- **Per block** (pre-norm): `h = LN1(x); x += MHA(h); x += SwiGLU(LN2(x))`.
- **Attention**: fused `qkv` projection, biased linears, causal + key-padding
  mask via the `PAD` token, `attn.out` projection.
- **FFN**: SwiGLU — `SiLU(gate) * value`, then the down projection (both biased).
- **Head**: untied `u_head` → logits over `U_BINS` actuator classes.
- **Loss**: masked cross-entropy at `DECIDE` positions only
  (`ignore_index = IGNORE`), normalized by the per-batch non-ignored count.

### Token schema

A small CBOR-style event/effect vocabulary (`crates/pid/src/data.rs`):
`PAD, BOS, EV_START, EV_END, FX_START, FX_END, DECIDE` over a 263-token vocab.
An event is `setpoint / y / error` quantized to 101 bins over `[-1.5, 1.5]`; an
effect is the actuator `u` quantized to 81 bins over `[-1.0, 1.0]`. The plant
physics is first-order (τ, gain); the oracle is a pole-placed velocity-PI
controller. Training/validation plant grids are disjoint (9 train / 4
validation), with an interpolated off-grid class for the demo's generalization
badge.

## CLI

```
brain pid train     # DAgger: drive plants, relabel with the oracle, train the Transformer
brain pid rollout   # load a checkpoint, print the model-vs-oracle closed-loop report
brain pid profile   # time a single forward to the DECIDE position
```

`train` is memory-budgeted (microbatch + grad-accum planner) and ends with a
closed-loop report over training and validation plants. `rollout` reproduces
that report from a saved checkpoint; `profile` reports mean/min/max forward
latency and inferences/sec.

## WebGPU browser demo

`make web/dev` serves the Next.js app in `crates/web`. The wasm entry points
(`run_inference`, `run_inference_argmax`, `rollout_compare`) run two closed
loops on the same plant and setpoint staircase: the Transformer driving the
plant, and that plant's PID oracle. The UI renders τ / gain / steps sliders, a
train/valid/off-grid generalization badge, model-vs-oracle MSE tiles, and
tracking + control-signal charts. All physics and control run in Rust → wasm →
WebGPU; JavaScript only renders. Weights are fetched from
`/moe_pid.safetensors`. Needs a secure context (localhost or https) and a
WebGPU-capable browser (Chrome/Edge 113+).

## What's implemented

Forward + backward (cached dispatch graphs built once and reused per step),
AdamW, checkpoint save/load, and inference. The model implements the
architecture-agnostic `model::Model` / `model::ModelConfig` seam (native only).
Training, rollout, and profiling are exposed through the CLI.

### Parity / gradcheck

- `gradcheck::check_pid` — finite-difference gradient check (central difference,
  `eps=5e-3`) over a tiny config; the
  `pid_analytic_grads_match_finite_differences` test asserts no failures at
  `atol=4e-3, rtol=8e-2`.
- Inline model tests pin the param layout, config JSON round-trip, forward
  finiteness/determinism, backward finiteness, and the inference profiler.
- Inline `data.rs` tests pin quantization round-trips, the CBOR event/effect
  framing, the pole-placement and velocity-PI oracle formulas, plant step
  physics, the eval schedule, and the disjoint/interpolated plant grids.

## Limitations

- **Wasm is inference-only** — no loss readback or training in the browser; the
  `Model` trait impl and scalar forward path are native-only.
- **No `import` or `serve` subcommand** — weights come from `train` and are
  consumed by `rollout`, `profile`, and the web demo (unlike the imported
  models, there is no HF checkpoint to import).
- The demo is **green to build but unverified in this environment** (no
  WebGPU browser); see `docs/engine/web.md`.

## See also

- `docs/engine/web.md` — the browser demo, `make web/dev`, and the
  honesty-of-comparison note.
- `AGENTS.md` → Models → PID event/effect Transformer.
- `README.md` — the toy control task and the PID oracle.
