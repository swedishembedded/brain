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

## Tiered weight residency (streaming, no-OOM) — 2026-08-10

Z-Image's checkpoint (~31 GB: DiT 24.6 GB bf16 sharded ×3, Qwen3-4B encoder
8.04 GB bf16 sharded ×3, VAE 168 MB, tokenizer 16 MB) does not fit in this
box's RAM+swap headroom (swap was observed >99% consumed by unrelated
processes throughout every run below; `MemAvailable` fluctuated 2-20 GiB).
Brain must never load such a checkpoint whole — weights stream from disk via
`checkpoint::weightio::WeightReader` (mmap) straight to the device, one
tensor at a time, through `checkpoint::TensorSource`'s chunked/zero-copy
accessors (shared by every model that streams through this seam, not a
zimage-only path).

**Real run** (`brain do brain/z-image text2image`, cold — first call, so this
includes streaming the whole ~31 GB checkpoint through the quantize/upload
path): prompt "a red fox in snow, photograph", 256×256, 8 steps, seed 42,
`--precision int8`.

| | |
|---|---|
| Wall-clock | **197 s** (cold: checkpoint stream + int8 quantize-build of both the encoder and DiT + 8-step sampling + VAE decode) |
| Peak `brain` process RSS | **~1.03 GiB** (`/proc/<pid>/status` `VmRSS`, sampled every 2s) |
| Peak system cgroup `memory.current` during the run | ~16.8 GiB (shared with other load on the box; not brain-exclusive) |
| Output | `results/zimage-int8-256.png`, sha256 `56a2261f043b7ea558ae4fd17bca31f11671e1b2d1d4cc0f85b59e387848ea9b` |
| Result | correct image (a red fox standing in snow), no OOM |

Before the fixes below, the identical command was OOM-killed (`SIGKILL`,
exit 137) twice, at RSS ≈ 16-19 GiB, on this same box.

### Three bugs found and fixed by this real run (not by unit tests alone)

1. **`checkpoint::weightio::WeightReader::open_hf_dir` / `safetensors::read_model_dir`
   only recognized the HF-transformers index filename**
   (`model.safetensors.index.json`). Z-Image's `transformer/` dir ships the
   diffusers convention (`diffusion_pytorch_model.safetensors.index.json`).
   Unrecognized, both silently fell back to their "no index → exactly one
   `*.safetensors` file" path and opened only the alphabetically-first shard
   — the other two (containing e.g. `layers.9.feed_forward.w3.weight`) were
   never even opened. Symptom was a late, confusing panic
   ("missing layers.9.feed_forward.w3.weight") deep in weight upload, not an
   error at open time. Fixed by recognizing both index filenames; regression
   tests added in both `weightio.rs` and `safetensors.rs`.
2. **`MmapSafetensors::advise_dontneed_tensor` had zero callers** (already
   flagged in the design plan). A whole-checkpoint streaming scan reads every
   tensor exactly once, but without evicting a tensor's page-cache pages
   after it's consumed, they accumulate for the rest of the scan — RSS grows
   toward the full file size even though nothing is ever re-read. Fixed by
   calling it from `MmapSafetensors::tensor_f32` and `with_tensor_chunks`
   themselves, right after decoding — every model streaming through
   `TensorSource` gets this for free, not just zimage.
3. **The encoder's un-set-env-var default was worse than the on-demand
   path it was supposed to be an alternative to.** With no
   `BRAIN_ZIMAGE_ENCODER_GPU`, `HotPipeline::build_adapted` built a
   permanently-resident CPU fp32 Qwen-4B encoder (~16 GiB, never dropped)
   *concurrently* with the GPU int8 DiT build — worse in both memory (16 GiB
   vs. the on-demand int8 path's ~9.5 GiB, dropped before the DiT even
   builds) and speed (~38 s vs. ~1-2 s) than simply sharing the DiT's own
   card. On a box with only one GPU there is no real "separate bulk card"
   choice to defer to the caller, so defaulting to CPU there was pure
   pessimization. Fixed via `default_bulk_gpu` in `pipeline.rs`: share the
   DiT's GPU by default when the machine has exactly one; leave the
   ambiguous multi-GPU default unchanged. This is what makes the memory fix
   automatic — the caller never has to know to set an env var for it to not
   OOM.

None of these three would have been caught by a unit test written in
isolation ahead of time — each is a real-checkpoint-shape or real-machine-
shape fact (a diffusers index filename, actual page-cache accumulation under
memory pressure, a single physical GPU) that only the end-to-end run against
the real ~31 GB checkpoint on this real box could surface. Regression tests
were added for each afterward so they can't silently regress.

**fp32 stress case, same box — UPDATED, now runs.** The paragraph originally
here recorded `--precision fp32` failing immediately with `need 2 discrete
GPUs, found 0`, since `ZImageDitShard` (the only fp32 engine at the time) is
structurally 2-discrete-GPU-only and this box has one integrated GPU. A
follow-up in the same workstream built `crates/weightset` (a generic,
model-agnostic within-instance weight window: `Schedule`/`ResidencyPlan`/
`WeightSet`, Bélády `CyclicScan` eviction over the known denoise order,
churn measured exactly) and `zimage::ZImageDitWindowed`, which streams the
main 32-layer stack through a small fixed window (default 2 blocks resident,
~1.4 GB at Z-Image-Turbo's real shape) instead of sharding across 2 GPUs.
`pipeline::DitEngine` now picks it automatically whenever fewer than 2 GPUs
are available — no flag required.

Real run, same box, same checkpoint: `--precision fp32`, 256×256, 2 steps —
**144 s total (~72 s/step)**, correct image produced
(`results/zimage-fp32-256.png`, sha256
`fbd9935b6e32a09df28c9868b7d040d624821078ea0945b25541af94ceb9bf64`), no OOM.
This confirms Risk R1's prediction from the original plan almost exactly:
fp32 streaming on this checkpoint is disk-bound, not compute-bound (~72 s/step
vs. int8's well-under-1s/step once resident) — the honest deliverable is
"fp32 streams and is correct, and is slow", not "fp32 is fast". Making it
fast needs a native bf16 device bind (dequant-in-kernel) so the window reads
half the bytes per reload, which remains a real, separate follow-up, exactly
as the original plan named it.

Proof of correctness for the windowing mechanism itself (not just "it didn't
crash"): `windowed_dit_matches_fully_resident_dit_bit_for_bit_when_window_is_
narrower_than_the_model` (`crates/zimage/src/dev.rs`) asserts the windowed
engine's output is bit-for-bit identical (`assert_eq!` on `Vec<f32>`, not a
cosine bound) to the fully-resident engine at a tiny synthetic config with a
window narrower than the model — residency is a pure memory placement
decision here, verified to never be a numerical one.

## Cross-instance Tier::Warm/Cold — a real, measured opt-in

`residency::ResidencyManager`'s eviction path now tries `Instance::demote(Warm)`
before falling back to a full drop, and `claim()` promotes a Warm instance back
to Hot instead of always rebuilding from the checkpoint (see
`crates/residency/src/manager.rs`). Every existing model still gets exactly
today's behaviour (`demote` defaults to `Err`) — this is opt-in, and for
Z-Image specifically it is only worth opting into for the plain int8 build
(`BRAIN_ZIMAGE_RETAIN_INT8_CACHE=1`): `ZImageDitI8::build_from_source_with_cache`
retains every block's already-quantized host weights (packed int8 + scales)
instead of discarding them after upload, so a later `promote` skips both the
checkpoint read and the re-quantization — at the cost of ~5 GB of permanently
retained host RAM for as long as the instance stays registered
(`zimage::pipeline::int8_cache_bytes_estimate`, reported honestly via
`estimate_at(Warm)` instead of a false `0`).

Real, measured result against the actual checkpoint
(`crates/cli/src/resident.rs::tests::zimage_demote_then_promote_produces_a_real_image_and_promote_is_faster`,
`--ignored`, `BRAIN_ZIMAGE_RETAIN_INT8_CACHE=1`): activate → demote → promote →
generate, on this same box —

| | |
|---|---|
| Fresh `activate()` (checkpoint read + quantize + upload) | **94.5 s** |
| `promote()` from cache (no checkpoint, no re-quantize) | **9.6 s** |
| Speedup | **~9.8×** |

Correctness proof independent of the real run: `dit_i8_rebuilt_from_cache_
matches_a_fresh_build_bit_for_bit` (`crates/zimage/src/dev.rs`) asserts a DiT
rebuilt purely from a `DitI8Cache` produces bit-identical output to a fresh
build off the checkpoint — the same "residency is a pure memory placement
decision" property the weight-window test above establishes, now for the
demote/promote cache too.

fp32 and any LoRA-adapter build never retain a cache (`demote` refuses for
those, matching every other non-opted-in model) — building one for those
would need caching an entire fp32/adapter-folded checkpoint's worth of host
weights, a much larger and separately-scoped undertaking, not attempted here.

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

## Resident-pipeline prompt padding — masked pad (2026-08-10)

The resident `HotPipeline` pads short prompts to its built `cap_len`. The
original scheme repeated the LAST prompt token with no attention mask, so
the caption features — all `cap_len` rows of which the S³-DiT attends
unmasked — depended on how many copies of the final token the encoder saw
(the unsoundness class the LFM ledger documents; audit F17). Now: a
dedicated `<|endoftext|>` pad token, excluded as an attention KEY past the
true content length (`Qwen::encode_padded`, the same kmask machinery
FLUX.2's `encode_hiddens_padded` parity-validated against HF's
`attention_mask` semantics). Exact-length prompts take the original
unmasked path bit-unchanged. A short-prompt parity case against the Tongyi
reference remains open (no `zimage_dump_reference.py` exists yet); the HF
pipeline's tokenizer-mask convention is the basis for the choice.

## See also

- `docs/models/zimage/readme.md` — architecture, CLI, parity.
- `docs/models/flux2/status.md` — the sibling FLUX.2 ledger.
- `docs/imaging/plan.md` — imaging workstream plan/status.
- `docs/serving-contract.md` — the five obligations.
- `AGENTS.md` → Models → Z-Image.
