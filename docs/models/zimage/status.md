# zimage — workstream ledger

Z-Image (S³-DiT) + the shared diffusion stack (`crates/{zimage,dit,diffusion,vae}`).
The user-facing guide is `readme.md`; this file is the workstream ledger (what
landed, the parity gates, what remains).

## Done

- **Four DiT engines** — reference `ZImageModel`, device-resident `ZImageDit`,
  int8 `ZImageDitI8`, 2-GPU fp32 `ZImageDitShard` (`dev.rs`, `model.rs`).
- **VAE** — `AutoencoderKL` encode + decode (decoder-first; encoder +
  `latent` pack/unpack shared with FLUX.2) (`crates/vae`).
- **Flow-matching** — Euler scheduler + Z-Image sigma schedule + dynamic shift
  (`crates/diffusion/src/scheduler.rs`, `pipeline.rs`).
- **Multi-axis RoPE** — shared `crates/dit/src/rope.rs`.
- **Hand-written backward + gradcheck ladder** — block (`grad.rs`),
  device block (`devgrad.rs`), full model (`modelgrad.rs`), device full model
  (`train.rs`), pipeline sharding (`shard.rs`), LoRA (`lora.rs`), full LoRA
  finetune (`finetune.rs`).
- **INT8 DP4A path** + the engine-wide quantizer hoist (`int8.rs`,
  `model::int8`).
- **Import** from Comfy / original checkpoint layout (`import.rs::import_comfy`)
  + the bridge to training weights (`model_weights_from_comfy`).
- **End-to-end generation pipeline** `HotPipeline::generate` +
  `generate_img` (text2image / image2image / inpaint / outpaint) (`pipeline.rs`).
- **Serving contract (partial)** — capability `ZImageProvider` + resident
  `ZImageResident`; actions `text2image` / `image2image` / `inpaint` /
  `outpaint` / `lora_train` (`caps.rs`, `resident.rs`, `caps_cli.rs`,
  `run_cli.rs`).
- **Shared `flash_attn_bidir_split` seam** with FLUX.2
  (`crates/model/src/block.rs`).

## Parity ladder

| Gate | Result |
|---|---|
| block vs diffusers | `cos ≥ 0.9999` |
| full model (small) vs diffusers | `cos ≥ 0.9999` |
| device vs reference | `max_abs ≤ 1e-3` (CPU), `3e-3` (GPU) |
| real 6B fp32 (CPU) vs diffusers | `cos ≥ 0.999`, `rel_l2 ≤ 0.03` |
| 2-GPU shard vs diffusers | `rel_l2 ≤ 0.03` |
| real int8 (1 GPU) vs diffusers | `cos ≥ 0.99` |
| block backward vs FD | `worst < 1e-4` rel |
| device block backward vs host | `cos > 0.999`, `rel_l2 < 2e-2` |
| int8 GEMM (DP4A) vs fp32 | `cos ≥ 0.999` |
| pipeline-shard grads | `rel_l2 < 1e-4` |
| full-model gradcheck vs FD | `worst < 1e-3`; overfit `l < l0·1e-2` |
| VAE decode vs diffusers | `cos ≥ 0.999`, `PSNR ≥ 40 dB` |
| VAE (FLUX.2) encode/decode | `cos ≥ 0.9999`, pack `max_abs < 1e-4` |

`docs/imaging/plan.md` records the FLUX.2 VAE encode/decode at cosine 1.000000
after the `vae::blocks` hoist, with "zimage suites unaffected (22 suites
green)".

## Measured (quoted from code/docs)

- int8 6B fits **one 24 GB P40** (~6 GB of weights), no sharding (`int8.rs`,
  `dev.rs`).
- int8 GPU encoder ~9.5 GB resident, ~1-2 s/encode; fp32 split ~23 GB resident,
  ~1-2 s/encode; CPU encoder ~38 s/encode (`pipeline.rs`).
- `flash_attn_bidir_split` vs baseline: 29× at head_dim 128, 4.4× at head_dim 32
  (P40, `crates/model/src/block.rs`).
- No Z-Image end-to-end image-generation latency is recorded in the repo
  (FLUX.2's numbers in `docs/models/flux2/status.md` are FLUX.2, not Z-Image).

## Remaining

- **Serving contract gaps** — a true batched `run_batch` and a runnable
  `examples/` client are not present (the provider + resident exist; see
  `docs/imaging/plan.md` §3.5 for what each imaging model still owes).
- **Stale `caps.rs` doc comment** — claims the end-to-end pipeline is "the
  remaining piece"; `HotPipeline::generate` is implemented and returns a real
  image. Reconcile the comment with the code.
- **Unify the dynamic shift** — Z-Image's local `calc_mu` / `dynamic_shift`
  (`pipeline.rs`) is distinct from FLUX.2's `empirical_mu` in `crates/diffusion`.
- **Device-resident block chaining** — the reference path round-trips to the
  host between blocks (`model.rs`).
- **Continuous-CI coverage** of the real 6B path is opt-in (weight /
  `BRAIN_ZIMAGE_*` gated).

## See also

- `docs/models/zimage/readme.md` — architecture, CLI, parity.
- `docs/models/flux2/status.md` — the sibling FLUX.2 ledger.
- `docs/imaging/plan.md` — imaging workstream plan/status.
- `docs/serving-contract.md` — the five obligations.
- `AGENTS.md` → Models → Z-Image.
