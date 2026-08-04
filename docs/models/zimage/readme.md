# Z-Image / diffusion stack in brain (`crates/{zimage,dit,diffusion,vae}`)

Z-Image (Tongyi) text-to-image: the **S³-DiT** single-stream diffusion
transformer over Qwen3-4B caption features + VAE latents. Forward/inference
first; LoRA, INT8, sharding, and a hand-written backward build on the same
forward graph. Z-Image is the in-tree sibling to FLUX.2 — both assemble the
shared `dit` / `diffusion` / `vae` / `qwen` crates.

`docs/models/zimage/status.md` is the workstream ledger. This file is the
user-facing guide.

## Architecture

- **S³-DiT** (single-stream, `crates/zimage/src/model.rs`): patchify the latent
  → embed image/caption/timestep → refine image (`noise_refiner`, modulated)
  and caption (`context_refiner`, unmodulated) → concat `[image, caption]` →
  30 main layers → FinalLayer → unpatchify. Turbo config: `dim=3840`,
  `n_layers=30`, `n_heads=30` (head_dim 128), `in_channels=16`, `patch_size=2`,
  multi-axis RoPE `axes_dims=[32,48,48]`, `rope_theta=256`.
- **adaLN fold** — the global per-channel scale/gate from the timestep
  embedding is folded into the RMSNorm weights on the host each forward
  (`rmsnorm(x,w)·scale = rmsnorm(x, w·scale)`), so no scale/gate kernels are
  needed.
- **Multi-axis RoPE** — per-axis `(cos,sin)` tables, interleaved rotation,
  f64 angles → f32, from the shared `crates/dit/src/rope.rs` (used by zimage,
  flux2, hidream).
- **Flow-matching** — `FlowMatchEulerScheduler` (rectified flow,
  `x_σ=(1-σ)x0+σε`, Euler step) from `crates/diffusion/src/scheduler.rs`, with
  Z-Image's `linspace(1,1/n,n)` sigma schedule + a dynamic shift.
- **VAE** — `AutoencoderKL` from `crates/vae` (16 latent channels, 8× upscale,
  32 GroupNorm groups, eps 1e-6). Decoder: post_quant_conv → conv_in → mid
  (resnet, self-attn, resnet) → up blocks → SiLU → conv_out.
- **Text encoder** — Qwen3-4B provides `hidden_states[-2]` caption features;
  tokenizer is `data::qwen_tokenizer::QwenBpe` with the chat template.

## CLI

There is **no dedicated `brain zimage` subcommand**. Z-Image is served through
the generalized capability interface (the serving contract, not a bespoke
command):

```
brain caps brain/z-image                 # discovery (static manifest, no weights needed)
brain do brain/z-image <action> …        # execution; also over D-Bus / the event API
```

Actions (`crates/zimage/src/caps.rs`): `text2image`, `image2image`, `inpaint`,
`outpaint`, `lora_train`. Shared params: `steps` (default 8), `guidance`
(default 0.0), `seed`, `width`/`height` (default 1024), `precision` ∈
{`int8`,`fp32`} (default `int8`). The sibling `brain flux2 generate` is the
FLUX.2 CLI, not Z-Image.

`BRAIN_ZIMAGE_{DIT,VAE,QWEN,TOKENIZER}` point the resident at its weights.

## What's implemented

- **Four DiT engines** (`dev.rs`): `ZImageModel` (reference, host round-trips),
  `ZImageDit` (device-resident, weights resident), `ZImageDitI8` (int8 DP4A,
  single GPU — the 6B fits one 24 GB P40, ~6 GB of weights), `ZImageDitShard`
  (fp32 across 2 GPUs, one host-staged residual at the cut).
- **End-to-end pipeline** `HotPipeline::generate` (`pipeline.rs`): chat-template
  + tokenize → Qwen-4B encode → seeded `randn` latent → dynamic-shifted sigmas
  → 8-step flow-matching loop → VAE decode. `generate_img` covers
  image2image/inpaint/outpaint (`Init { image, strength, mask, feather }`).
- **LoRA** (`lora.rs` + `finetune.rs`) — `W_eff = W + (α/r)·B·A`, base frozen;
  the end-to-end folder→adapter trainer VAE-encodes + Qwen-encodes once.
- **INT8** (`int8.rs`) — re-exports the engine-wide `model::int8::quantize_weight`
  (per-channel symmetric, packed 4-per-u32, dynamic per-token activation scale,
  DP4A GEMM).
- **Sharding** (`shard.rs`) — `ShardTrainer` (sequential + `grads_microbatched`
  GPipe) cuts the main-layer stack; only `[uni ‖ c]` crosses cards.
- **Hand-written backward** — `grad.rs` (host f64 block fwd/bwd, the gradcheck
  oracle), `devgrad.rs` (GPU block backward), `modelgrad.rs` (full-model f64
  fwd/bwd under flow-matching velocity-MSE), `train.rs` (GPU training step).
- **Serving contract** — `ZImageProvider` + `ZImageResident` (resident.rs),
  hot pipeline cache keyed on `(width, height, hifi, adapter)`.

## Parity

| Gate | What | Result |
|---|---|---|
| block | vs diffusers golden (small) | `cos ≥ 0.9999`, `max_abs ≤ 1e-2·|want|max` |
| full model (small) | vs diffusers golden | `cos ≥ 0.9999`, `max_abs ≤ 2e-2·|want|max` |
| device vs reference | CPU / GPU | `max_abs ≤ 1e-3` (CPU), `3e-3` (GPU) |
| real 6B fp32 (CPU) | vs diffusers (env-gated) | `cos ≥ 0.999`, `rel_l2 ≤ 0.03` |
| 2-GPU shard | vs diffusers (env-gated) | `rel_l2 ≤ 0.03` |
| real int8 (1 GPU) | vs diffusers (env-gated) | `cos ≥ 0.99` |
| block backward | vs finite difference | every tensor `worst < 1e-4` rel |
| device block backward | vs host | `cos > 0.999`, `rel_l2 < 2e-2` (GPU) |
| int8 GEMM | DP4A vs fp32 | `cos ≥ 0.999` |
| full-model gradcheck | vs FD | `worst < 1e-3`; overfit `l < l0·1e-2` |
| pipeline shard | single-device match | loss `< 1e-5`, grads `rel_l2 < 1e-4` |
| VAE decode | vs diffusers | `cos ≥ 0.999`, `PSNR ≥ 40 dB` |

Real-weight tests are env-gated (`BRAIN_ZIMAGE_DIT` / `_VAE` / `_SHARD=1` /
`_I8=1` + GPUs); CI runs the small-config goldens.

## Kernel / block reuse

Z-Image is a thin assembly of shared crates — `dit` (RoPE), `diffusion`
(scheduler), `vae` (`AutoencoderKL` + the `vae::blocks::Builder` shared with
`vqgan`), `qwen` (encoder). The bidirectional flash-attention seam
(`flash_attn_bidir` / `flash_attn_bidir_split`, dispatched through
`model::block::flash_bidir_step`) is shared with FLUX.2 — the split variant
matches the baseline to cosine 1.0 and is faster at every head_dim, selected on
the device's queried `max_workgroup_size` (`BRAIN_ZIMAGE_FLASH=1|0` forces it).
The INT8 quantizer and `hostmath::randn` were both hoisted out of zimage into
the engine-wide `model::int8` / `model::hostmath` (one implementation each).

## Limitations

- No `brain zimage` CLI subcommand — discovery/execution is via
  `brain caps` / `brain do` / D-Bus (by design).
- A batched `run_batch` and an `examples/` client are **not** present
  (the serving contract is partially met: provider + resident exist; true
  batching and a runnable D-Bus example are deferred — see
  `docs/imaging/plan.md` §3.5).
- A `caps.rs` doc comment claims the end-to-end pipeline is "the remaining
  piece" — this is stale; `HotPipeline::generate` is implemented and returns a
  real image.
- Z-Image's dynamic shift (`calc_mu` / `dynamic_shift`) is local to
  `pipeline.rs`, not unified into `crates/diffusion`.
- The reference forward path round-trips to the host between blocks
  (device-resident chaining is a later optimization).

## See also

- `docs/models/zimage/status.md` — workstream ledger.
- `docs/models/flux2/{readme,status}.md` — the sibling FLUX.2 stack.
- `docs/imaging/plan.md` — the imaging workstream plan/status.
- `docs/serving-contract.md` — the five obligations.
- `AGENTS.md` → Models → Z-Image.
