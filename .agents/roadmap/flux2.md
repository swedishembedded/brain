# flux2 — roadmap

FLUX.2 Klein text-to-image / image-editing diffusion transformer (Qwen3 text
encoder + VAE + MMDiT), ported to brain's own kernels and serving stack.

## Masked editing (`--mask`) - what was measured

Blended latent diffusion: after every Euler step the masked-out region is
replaced by the source latent renoised to that step's sigma. Motivating task:
virtual staging of real-estate photographs with a rank-32 staging LoRA over
`klein-9b`.

Why not ControlNet, settled and not to be re-litigated:
`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` is **6144-dim**
(`control_transformer_blocks.*.attn.to_q.weight = [6144, 6144]`,
`ff.linear_in = [36864, 6144]`, 4 blocks) and the backbone in use is
`klein-9b` at **4096**. Residuals cannot be injected across that mismatch and
no klein ControlNet exists upstream; the staging LoRA is rank-32 over 4096, so
it only fits klein. ControlNet and that LoRA are mutually exclusive.
`crates/controlnet`'s `ControlAdapter` seam stays unimplemented for FLUX.

Measured on **2x Tesla P40**, `klein-9b` int8 DiT on gpu1 + truncated int8
Qwen3-8B text encoder on gpu0, 1024x768, 12 steps, seed 7, LoRA scale 1.0,
the model card's verbatim prompt tier wording. `edge corr` = Pearson
correlation of full-resolution Sobel gradient magnitudes against the source;
`MAD` = mean |RGB difference| in 0..255. `@keep` / `@gen` restrict the same
statistics to the mask's hard-preserve / hard-regenerate pixels. **The
absolute scale of an earlier hand-recorded edge-correlation ladder could not
be reproduced from the images it was taken on** (no definition was recorded
with it); everything below is re-measured under the definition just given, so
the rungs are comparable to each other and not to those numbers.

Bedroom (`image-02`), whole frame / preserved architecture / regenerated
interior:

| setting | edge corr | MAD | edge@keep | MAD@keep | edge@gen | MAD@gen |
|---|---|---|---|---|---|---|
| strength 0.55 | 0.731 | 13.9 | 0.804 | 8.7 | 0.671 | 18.6 |
| strength 0.85 | 0.425 | 29.5 | 0.471 | 20.8 | 0.326 | 37.4 |
| strength 0.95 | 0.272 | 48.5 | 0.280 | 34.8 | 0.202 | 60.9 |
| strength 1.00 (768 ref) | 0.166 | 61.7 | 0.175 | 50.0 | 0.110 | 72.1 |
| strength 0.999, no mask | 0.075 | 66.0 | 0.033 | 56.8 | 0.046 | 73.8 |
| **strength 0.999 + arch mask** | 0.413 | 39.0 | **0.980** | 9.3 | 0.063 | 66.6 |
| all-black mask (pure preserve) | 0.984 | 3.4 | 0.988 | 1.9 | 0.981 | 4.8 |

The last row is the **ceiling**, not a result: it is what the VAE round trip
alone costs, and no latent-space edit can beat it. Read the table as: the
mask reaches 0.980 architectural edge correlation against a 0.988 ceiling
while regenerating the interior harder than strength 1.00 did (MAD@gen 66.6
vs 72.1 with edge@gen 0.063 vs 0.110), where the best previous setting that
restaged anything at all held the architecture at 0.471.

Living/dining room (`image-01`), same settings, regions from its own mask
(49.9% preserve / 37.0% regenerate):

| setting | edge corr | MAD | edge@keep | MAD@keep | edge@gen | MAD@gen |
|---|---|---|---|---|---|---|
| strength 0.85, no adapter | 0.513 | 17.3 | 0.655 | 13.0 | 0.417 | 23.8 |
| strength 0.85 + adapter | 0.482 | 19.5 | 0.624 | 15.7 | 0.391 | 25.5 |
| strength 1.00 (768 ref) | 0.323 | 63.6 | 0.390 | 64.5 | 0.260 | 57.5 |
| strength 0.999, no mask | 0.084 | 62.5 | 0.111 | 60.7 | -0.005 | 69.0 |
| **strength 0.999 + arch mask** | 0.435 | 22.9 | **0.991** | 3.6 | 0.011 | 54.5 |
| all-black mask (pure preserve) | 0.995 | 1.2 | 0.996 | 0.9 | 0.994 | 1.7 |

Whole-frame edge correlation stays at 0.413 and **cannot** go above 0.90 for
a genuinely restaged room: half the frame is new furniture, and new furniture
has new edges. The all-black row proves the point - 0.984 is available only
by not staging. Whole-frame edge correlation conflates "did the architecture
move" with "did the furniture change", and staging requires the second.

Gate 1 (`--mask` white == no mask) was additionally confirmed on the real
9B pipeline, not only in the checkpoint-free tests: the two runs are
byte-identical PNGs, and a repeated no-mask run is byte-identical to itself,
so that comparison means something.

## Where a klein-9b int8 generation's time actually goes

Measured on **2x Tesla P40** (GP102, SM 6.1) + dual Xeon E5-2690 v3, 48
threads, warm page cache. klein-9b int8 DiT on gpu1, truncated int8 Qwen3-8B
text encoder on gpu0, 1024x768. Device numbers from `BRAIN_PROFILE`
timestamp queries; host phases from the `flux2 build:` spans and
`flux2_bench load`; the load/generate split from process wall minus the
pipeline's own stage total, cross-checked against a 10 Hz /proc RSS
timeline. `nvidia-smi` was never sampled during a timed run.

Card roofline measured on this box, not datasheet: **10 517 GFLOP/s fp32**,
**43 560 GOP/s int8 (DP4A)**, **287.5 GB/s** DRAM.

### The two halves

A single-image run is a **one-off weight load** plus a **generation**, and
before this pass the load was 41% of the process wall and completely
uninstrumented. The `--steps 10` row of the older table (~61 s) is the
PIPELINE total, not the process wall; the process wall was ~105 s.

| phase | 10 steps, 3072 img tokens | share of wall |
|---|---|---|
| DiT weight load (GGUF read, dequant, quantize, upload, free) | 20.3 s | 23% |
| text-encoder weight load | ~11 s | 12% |
| denoise, 10 steps | 54.1 s | 61% |
| VAE decode | 4.0 s | 4.5% |
| text encode | 0.8 s | 1% |
| **process wall** | **88 s** | |

### Denoise, per kernel, at two token counts

`n` is the JOINT sequence the DiT attends over: 512 text + image + any
reference tokens. Device time over 4 steps, `BRAIN_PROFILE` timestamps.

| kernel | n = 3584 | share | n = 6656 | share | scaling |
|---|---|---|---|---|---|
| `matmul_i8_dyn` | 9 678 ms | 45.9% | 18 337 ms | 35.8% | 1.90x (linear) |
| `flash_attn_bidir_reg2` | 6 914 ms | 32.8% | 23 726 ms | 46.3% | **3.43x (n^2)** |
| `matmul_reg3` (fp32) | 3 009 ms | 14.3% | 6 469 ms | 12.6% | 2.15x |
| `layernorm` | 419 ms | 2.0% | 780 ms | 1.5% | |
| everything else | 1 066 ms | 5.0% | 1 946 ms | 3.8% | |
| **total** | **21 086 ms** | | **51 258 ms** | | |

Token count is 1.857x between the two columns. Attention scales at 3.43x
against a 3.45x quadratic prediction: **the flash kernel is exactly
quadratic and nothing else is**, so the top row CHANGES IDENTITY between
the two sizes. At 3584 tokens the GEMMs dominate (60% combined); at 6656
attention alone does. Any further denoise work has to say which size it is
optimising, and a profile taken at one is not evidence about the other.

Against the measured roofs, per step at n = 3584:

| kernel | achieved | roof | % of roof |
|---|---|---|---|
| `matmul_i8_dyn` | ~19 500 GOP/s | 43 560 GOP/s int8 | **45%** |
| `flash_attn_bidir_reg2` | ~3 900 GFLOP/s | 10 517 GFLOP/s fp32 | **37%** |
| `matmul_reg3` (fp32) | ~3 910 GFLOP/s | 10 517 GFLOP/s fp32 | **37%** |

The fp32 `matmul_reg3` and the flash kernel sit at the same 37% of fp32
peak, and that figure is flat across shapes, which points at a structural
ceiling rather than a tuning gap in either.

### Denoise host occupancy: there is no bubble

Device kernel time over the 4-step denoise was 21 086 ms against a 21 730 ms
denoise stage wall: **97% of the denoise is the card working.** The
allocator/host-bubble hypothesis that paid off on ltxv does NOT apply to
flux2's denoise loop, and `gpu_core::scratch::Arena` / `Gpu::scratch_scope`
would have nothing to recover here (flux2 does not use them, and should
not on this evidence). The host bubble is entirely in the LOAD.

## Load path: what landed, and what is left

Host-side load, klein-9b, `flux2_bench load` and the `flux2 build:` spans:

| term | before | after | how |
|---|---|---|---|
| `gguf::read` (10.0 GB file) | ~9.8 s | 4.4 s | map the file instead of slurping it |
| `linear2` column split | 5.84 s | 0.58 s | `hostmath::split_cols`, row-parallel |
| `quantize_weight` (96 tensors, 31.7 GB fp32 read) | 4.2 s | 4.2 s | untouched |
| `gpu.write` (11.1 GB) | ~3.5 s | ~3.5 s | untouched, runs at 3.0-3.7 GB/s |
| staging-reclaim readback x10 | 0.14 s | 0.14 s | untouched |
| drop the 36.3 GB fp32 map | ~2.9 s | ~2.9 s | untouched |

End to end, best of 2, output **byte-identical** (same PNG md5) at every
step, since every change is bit-identical:

| config | joint tokens | pipeline before | pipeline after | **wall before** | **wall after** |
|---|---|---|---|---|---|
| 10 steps, 1024x768 | 3584 | 60.4 s | 58.9 s | **105 s** | **88 s** |
| 12 steps, 1024x768 + 768x576 ref, strength 1.0 | 5312 | 120.4 s | 120.4 s | **171 s** | **151 s** |
| 4 steps, 1024x768 | 3584 | 26.4 s | 26.6 s | **69 s** | **57 s** |

Peak host RSS is UNCHANGED at ~43 GB. Mapping the GGUF does not lower it -
mapped pages are resident too - it only makes the file's 10 GB clean,
evictable page cache instead of dirty anonymous memory, and removes the
memcpy that produced it.

### Direct Q8_0 -> int8: what it bought, and what it did not

`DitWeights::Gguf` now requantizes a Q8_0 checkpoint straight to this
engine's per-row packing, one weight matrix at a time, with no fp32 model in
between. It is **bit-identical**, not parity-gated: `deq_q8_0` yields exactly
`(q as i8 as f32) * d` (7-bit `q`, 11-bit fp16 `d`, product 18 bits against
fp32's 24, so exact rather than rounded), and the scale and packing then run
through the same `int8::row_scale` / `pack_row` that `quantize_weight` calls.
The gate is `assert_eq!` on the packed words and the scales; end to end, a
real 9B generation writes the same PNG md5 by either route, with and without
a LoRA.

Measured, 1024x768, 10 steps, best of 2:

| term | before | after |
|---|---|---|
| wall | 88 s | **81 s** |
| DiT load phase | 17.0 s | **10.4 s** |
| DiT-phase host peak | 43.2 GB | **10.4 GB** |
| whole-process host peak | 43.2 GB | **32.9 GB** |
| `gguf::read` whole-model dequant | 4.4 s | gone |
| free the 36.3 GB fp32 map | 2.9 s | gone |
| `linear2` column split | 0.6 s | gone |
| quantize / requantize | 4.6 s | 4.0 s |

**The estimate above this table used to say ~11.5 s. It was wrong, and the
error is worth keeping.** It assumed the quantize term would go to zero
because the checkpoint "already holds int8". It does not: the block-scale to
row-scale conversion still has to touch every weight, so only the whole-model
dequant and the free actually disappear. Measured saving is 7 s.

**The process peak is now bounded by the TEXT ENCODER, not the DiT.** The DiT
phase peaks at 10.4 GB; the 32.9 GB figure is Qwen3-8B's own fp32 import
arriving afterwards. That made the shard-aware TE import the next real memory
lever - **now done**, see "Streaming the text encoder" below.

### Streaming the text encoder

`pipeline.rs` built `Shard { start: 0, end: deepest tap, embed: true, head:
false }`, but the truncation happened at BUILD time, long after the import had
already insisted on the whole checkpoint: `read_model_dir` reads every shard in
the index's `weight_map` unconditionally, and `brain_init_from_hf` enforced
two-way coverage against the full `param_list()` of a 36-layer config with
`tie_embeddings: false`.

Two separate costs, fixed by two separate things. Streaming through a mapped
`WeightReader` bounds the FOOTPRINT. Deriving the required set from the shard
(`qwen3::import::hf_shard_source`, off the already-existing `shard_param_list`)
bounds the BYTES: the layers past the tap, the final norm and the LM head are
now neither read nor required.

Measured on a real Qwen3-8B (36 layers, 5 shard files, 16.4 GB bf16), dual Xeon
E5-2690 v3, for the `end: 27, embed, no head` shard this pipeline builds:

| term | before | after |
|---|---|---|
| text-encoder import, host peak | 31.67 GiB | **3.47 GiB** |
| text-encoder import, wall | 3.4 s | 5.5 s |
| tensors read | 399 | **298** |
| parameters read | 8.19 G | **5.83 G** |
| whole-process host peak | 33.35 GB | **11.27 GB** |
| whole-process wall | 49.2 s | 49.3 s |

The import figures come from `qwen3`'s env-gated real-checkpoint test and are
reproducible to the megabyte, identical under `BRAIN_DEVICE=cpu`. The
whole-process figures are a completed 1024x768 4-step klein-9b int8 generation
on otherwise idle cards, best of 2, warm page cache.

**The output PNG md5 is identical before and after**, which is the end-to-end
statement of the same bit-identity the tests assert per tensor.

**Bit-identical, not parity-gated.** Both routes decode the same bf16 bytes
with the same converter; only where the decoded f32 lives differs. All 298
tensors of the shard are pinned with `assert_eq!` on values against the eager
import, on the real checkpoint. Five gates, each mutation-verified, each
failing only for its own mutation.

**This was a memory-for-time trade until the cause of the "time" was found,
and then it stopped being one.** Streaming initially cost 13.8 s of wall (49.2
-> 63.0 s). The first hypothesis - that `advise_dontneed_tensor` was defeating
kernel readahead - was tested by disabling it and REFUTED: that costs about
0.4 s and saves 7.6 GiB, so it is close to free. The real cause was that
`mmap::decode_into`, the decoder every streaming read goes through, was serial
while the eager whole-file decoder had been fanned across the thread pool long
before. Sharing one parallel decoder took the streamed import from 15.0 s to
5.5 s and the end-to-end wall back to parity. Recorded because the wrong
hypothesis was plausible and the right one was one measurement away.

Still eager, deliberately: the UNPLACED text-encoder branch (no
`BRAIN_FLUX2_TE_DEVICE`) streams but still builds a whole encoder, because
truncating it would change what gets built rather than only where the bytes
live.

### Device budgets, and why the encoder needs its own card

Streaming the text encoder moved HOST memory and left DEVICE memory alone.
That was the intent, but it had never been measured, so there was no figure to
write a residency bound against. Measured on 2x Tesla P40, klein-9b int8,
1024x768, 4 steps, DiT on one card and the truncated int8 encoder alone on the
other, so each card's peak is attributable:

| | streamed | mapped (`BRAIN_FLUX2_TE_NO_STREAM=1`) |
|---|---|---|
| text encoder, device peak | 14097 MiB | 14101 MiB |
| DiT + VAE, device peak | 24431 MiB | 23397 MiB |
| host peak | 11.30 GB | 33.37 GB |
| output PNG md5 | identical | identical |

**Streaming costs 4 MiB of device memory, or 0.03%.** It does not defeat a
residency decision and does not double-buffer behind one: the buffers a shard
allocates are decided by `shard_param_list` and are the same either way, and
`paramstore::upload::Uploader` already bounds device-side staging with a
periodic drain for every source. The two DiT figures differ by more than the
two encoder figures do, which is run-to-run variation in the DiT's own
allocation, not an effect of the encoder route.

**Co-residency on ONE 24 GB card is NOT a supported configuration at this
size, and never was.** The DiT plus VAE alone peaks within a few hundred MiB
of a 24 GB card even at 4 steps with no adapter; adding a 14 GB encoder is not
close. This was briefly mistaken for a regression from the streaming commit;
the A/B above settles it, both routes fail identically when co-resident. Note
also that the previous in-code estimate for the truncated int8 encoder was
several times too small, which is what made co-residency look plausible.

- [ ] **Co-residency fails as a raw `wgpu error: Out of Memory` followed by a
      leaked device**, rather than a refusal that names the DiT budget, the
      encoder budget, their sum and the card's capacity. All four numbers are
      known before either build starts. A pre-flight check would turn an
      opaque driver fault into an actionable message, and is the smaller and
      more valuable half of this item.

**With a full-coverage LoRA the streamed path is a TRADE, not a win**: about
3 s slower than the map route (64 s vs 61 s at 4 steps) while holding 33.2 GB
instead of 43.3 GB. Every big tensor needs a float domain for the fold, so
the decode work remains and only the residency improves. A klein-9b adapter
touches 112 of 201 tensors and 96% of the parameters, which is why "fold only
what the adapter touches" is not by itself a saving - handling them ONE AT A
TIME is. `BRAIN_FLUX2_NO_STREAM=1` forces the map route for A/B.

## Corrections to earlier entries in this file

- "**a second query row per thread in the flash-attention kernel**" is
  already done and has been for some time: `flash_attn_bidir_reg2` is
  exactly that kernel (BR = 128 query rows, two per thread, `q0/q1/o0/o1`
  in registers), and it is what production dispatches. The remaining 37%
  of-peak gap is NOT an unclaimed register-blocking win. The kernel issues
  ~4 FLOP per shared-memory word where the SM needs ~8 to stay
  compute-bound, so it is shared-memory-bandwidth bound by about 2x. Going
  to three query rows needs a third 16 KiB `part` buffer on top of the 48
  KiB already used, which is the Vulkan/NVIDIA compute limit exactly, so
  that route is closed without an algorithmic change to the cross-lane
  reduction.
- The staging-reclaim readbacks in `Flux2Model::new_batched` (`gpu.read(b,
  1)` every ~1 GiB) cost **0.14 s over 10 flushes**, not seconds. They are
  not worth removing and removing them risks the non-ReBAR OOM they exist
  to prevent.
- `gpu.write` on this box runs at **3.0-3.7 GB/s**, not the ~0.8 GB/s
  page-fault-bound rate a fresh-destination cross-device transfer shows,
  so the upload is not a page-fault problem.

## Not yet done

- [ ] Automatic architecture-mask generation. Masks are authored today. The
      two obvious automatic recipes were tried on the real photographs and
      both fail: a monocular-depth near-field threshold (`brain zipdepth
      --view depth --colormap gray --headless`) marks the *ceiling* as
      foreground on a living room - directly above the camera it genuinely is
      the nearest surface - while leaving a sofa against the far wall as
      background; a depth top-hat (opening by reconstruction) fixes the
      ceiling but misses any object larger than the structuring element,
      which for a bed filling half the frame is the whole bed. "Near" is not
      "furniture". Two further constraints any generator must satisfy:
      staging has to ADD furniture where there is none, so a mask covering
      only the existing furniture cannot stage; and the floor is at once
      architecture (its perspective must not drift) and the place new objects
      go, which is what the grey levels exist for. A promptable segmenter
      (`crates/sam2`) plus a semantic prior is the plausible next attempt.
- [ ] `--mask` has no wire representation: `flux2::caps::gen_params_from`
      sets `mask: None`. Reaching it over D-Bus/HTTP needs a second image
      blob on the `edit` action, and `caps::refs_from` currently treats every
      blob it finds as a reference image.
- [ ] The mask's source latent is VAE-encoded a second time when
      `--strength` is 1.0 or absent (under `--strength < 1.0` the init encode
      is reused). One extra encode per generation, not per step.

- [ ] Mixed-progress admission — a new request joining an already-running
      batch. The scheduler hands a lane a fixed job slice and marks the
      instance "running" for the whole call, so no job can join a batch
      already in flight; this needs an executor-level change (an admission
      channel a lane can drain between denoise steps), not a model-side one.
- [ ] Batched text encoder forward — the text encoder's graph is built for
      one sequence at a time; prompt batching is not implemented.
- [ ] Batched VAE decode — the VAE decoder runs per request; batching it is
      not implemented.
- [ ] The VAE decoder's graph and weights are rebuilt and re-uploaded on
      every decode call instead of being cached per output resolution.
- [ ] An implicit-GEMM convolution for the VAE decode, so the convolution's
      im2col gather is folded into the GEMM's tile load instead of
      materializing a separate scratch buffer.
- [ ] A fused, tiled causal+key-mask attention kernel for the text encoder's
      score computation — the current per-element masked kernel is far off
      peak throughput and neither a repack nor a coalescing-only fix closes
      the gap.
- [ ] Several smaller kernel-efficiency gaps identified but not yet closed:
      a workgroup-per-row LayerNorm kernel (RMSNorm and softmax already got
      this treatment), a workgroup-per-row reduction for the int8 path's
      row-max step, and wider (vec4) shared-memory tile loads for the core
      GEMM kernel. (The fourth item that used to sit here, "a second query
      row per thread in the flash-attention kernel", was already done -
      `flash_attn_bidir_reg2` IS that kernel and is what production
      dispatches. See the corrections section above for what the remaining
      flash-attention gap actually is.)
- [ ] Performance sweeps do not record GPU thermal state (temperature,
      clock, throttle reason) per concurrency level, so a multi-level sweep
      on passively-cooled cards can be dominated by thermal throttling
      rather than the effect it's meant to measure.
- [ ] A GPU-backend test drives the model at dimensions that violate the
      device's minimum storage-buffer offset alignment and fails as a result;
      it needs either aligned test dimensions or an explicit alignment
      assertion so the failure is readable instead of a raw driver error.
- [ ] Klein-9B's cached-reference-attention variant is out of scope: it needs
      per-token modulation blending, which is incompatible with the current
      approach of folding modulation into the LayerNorm.
- [x] The text encoder was imported whole and then truncated. **Done** - see
      "Streaming the text encoder" below.

The core GEMM kernel already runs near a structural throughput ceiling for
its current shared-memory tiling scheme, and batching the diffusion
transformer's forward pass has a small, bounded payoff because its GEMMs are
already near their row-count-independent plateau at a single sample; most of
a served image's latency lives in the (currently unbatched) text encoder and
VAE decode rather than in the transformer itself.

## The VAE latent, measured (`flux2_latent`)

All of the below is the autoencoder alone - encode, edit the latent on the
host, decode. No DiT, no diffusion. Tesla P40 (gpu1), 1024x768 photographs of
rooms plus one generated empty room; the latent is `[32, 96, 128]`, one cell
per 8x8 pixel block (the DiT sees `[128, 48, 64]`, one token per 16x16 block,
via a 2x2 unshuffle and a frozen BatchNorm - a reshape composed with a
per-channel affine, so every linear edit below is the same edit in either
space). Metric definitions live in `flux2::latentops::ImageMetrics`; MAD is in
8-bit levels, `edge_corr` is the Pearson correlation of Sobel magnitudes on
Rec.601 luma.

Cost: 5.5 s to read the checkpoint and encode three images; about 2 s per
decode once the graph is built. The decode is **bit-identical** run to run,
across processes, and across the two cards - so any nonzero number below is
the edit, not the hardware.

**Round-trip floor.** MAD 3.39 / edge_corr 0.984 (bedroom photo), 1.24 / 0.995
(living room), 1.02 / 0.994 (the generated empty room). The often-quoted
"0.988 / 1.9" is about the mean of these; the spread across images is larger
than that single figure suggests, and a textured photograph is twice as
expensive to round-trip as a generated interior.

**Latent blending is a double exposure, and it costs contrast.** At alpha 0.5
a latent blend differs from a pixel blend of the same two decodes by MAD 10.1
and edge_corr 0.67 - not a small difference, and not a semantic morph either.
The direction of the difference is a loss: high-frequency energy (mean |3x3
Laplacian|) 9.08 vs 13.58, mean saturation 11.27 vs 15.77. Averaging latents
destroys about a third more detail and a third more colour than averaging
pixels, and the decoder restores none of it.

**Translation is equivariant; reflection and rotation are equivariant only at
latent resolution.** `decode(roll(z))` vs `roll(decode(z))`: MAD 0.67-1.6 at
every shift tried (1, 4, 16 cells, both axes) and at every spatial scale.
`decode(flip(z))` vs `flip(decode(z))`: MAD 15.6, edge_corr 0.61 at full
resolution - but box-downsample both to the latent's own 8x grid and it
becomes MAD 2.7, edge_corr 0.988, with the knee falling exactly between 4x and
8x. The images are indistinguishable mirrored bedrooms; the disagreement is
entirely in fine texture. Convolution commutes with translation but not with
reflection, and the consequence is concrete: **the latent does not store the
fine texture, the decoder synthesizes it, and the synthesis is
direction-dependent.** Rotation by a non-multiple of 90 degrees costs real
content rather than texture phase - it stays at MAD 8.9 / edge_corr 0.94 even
after 8x downsampling, and the decode is visibly soft with ringing at the
borders.

**Channel surgery: heavy-tailed, five channels of thirty-two.** Zeroing one
channel and decoding, ranked by MAD against the unedited decode: ch3 27.7,
ch8 24.2, ch15 9.1, ch19 8.0, ch11 6.2, then a flat tail of 27 channels
between 1.2 and 4.7 (max/median 13.5). Signatures: ch3 is luminance-negative
(luma -6.1, blue-weighted), ch8 luminance-positive (luma +7.6, warm), ch11 is
almost pure chroma (saturation -13.8 with no luma or hue shift), ch15 and ch19
carry both saturation and high-frequency energy. Scaling is close to linear in
the removed fraction (x0.5 gives about half of zero's MAD) and mildly
asymmetric (x2.0 gives less than zero's, the bright channels clipping at 255).
Every channel's spatial mean is near zero, so "zero the channel" and "flatten
it to its own mean" are the same edit - a per-channel DC offset is not where
the information is.

**Region splice: no artifact, aligned or not.** A hard 256x256 paste of one
latent into another, decoded, differs from the ideal pixel composite by MAD
3.8 / edge_corr 0.974 - about the round-trip floor - and looks *better* than
the composite, because the decoder smooths the seam over about three latent
cells instead of leaving a razor cut. Misaligning the rectangle by half a DiT
cell (8 px) or by half a VAE cell (4 px, so boundary cells hold a genuine
fractional mix of two unrelated latents) changes the seam-band MAD from 24.7
to 25.3 and 27.6 - a 4-12% effect, with no visible artifact at 1:1. Feathering
from 0 to 64 px monotonically reduces the seam-band excess (MAD 24.7 -> 12.3,
excess desaturation 6.3 -> 0.5). Two real costs that are *not* the seam: the
decoder's mid-block self-attention is global, so the pasted region never
reaches its own unspliced decode (MAD 9.6 even 100 px inside the rectangle)
and the untouched exterior still moves by MAD 2.1 arbitrarily far away.

**Noise, and what actually defines a valid latent.** Gaussian noise in units
of each channel's own spatial std, against the unedited decode: sigma 0.05 ->
MAD 1.2 / edge_corr 0.998; 0.2 -> 4.6 / 0.966; 0.5 -> 11.5 / 0.838; 1.0 ->
24.0 / 0.618; 2.0 -> 46.7 / 0.321. Photorealism dies between 0.5 and 1.0, the
scene stays readable to about 2.0, and gross layout survives 3.0. The failure
mode is coloured confetti at one-latent-cell granularity, not blur.
Per-channel noise sensitivity tracks the zeroing ranking at Spearman 0.95
(ch11, the chroma channel, is the one real outlier - coherent removal shows,
random jitter averages out).

The interesting part is the comparison at *matched displacement*. An alpha=0.5
blend moves the latent 0.711 sigma-units - almost exactly the sigma 0.75 noise
rung - yet decodes to a clean photograph while the noise decodes to confetti.
Three hypotheses about what makes a direction "valid" were tested and all
three failed: spatial coherence (noise smoothed to a 2/4/8-cell correlation
length at the same amplitude is *worse*, not better - MAD 17.7 -> 32.0 with
saturation blowing out), cross-channel subspace (noise confined to the top 13
or the bottom 19 principal channel directions of the latent's own per-cell
covariance is indistinguishable from noise in all 32: MAD 17.8 / 17.5 / 17.7),
and convexity (`z_bedroom + 0.5*(z_living - z_empty)`, a difference of real
latents that is *not* a convex combination of anything, decodes to a clean,
sharp bedroom; so does extrapolating past an endpoint at 1.42 sigma-units,
which produces a crisp empty room). What survives: a displacement built from
*differences of real latents* decodes to a real-looking image at any magnitude
tried, and a random displacement of the same size does not, no matter how its
spatial or channel statistics are shaped. The constraint is a joint
spatial-and-channel structure that no factorized random field reproduces, and
it was not reducible to any simple statistic measured here.

The practical consequence, and a correction: masked-generation artifacts do
**not** come from latents failing to interpolate into valid images, and they
do not come from partially-masked cells holding a linear blend of two
latents. Both of those decode cleanly, and the misalignment penalty is under
15% of an already-invisible effect. Whatever `--mask` artifacts remain have to
be explained on the DiT side - most plausibly by the per-step renoise blending
two trajectories that are at *inconsistent* sigma, which is not the same
object as a mix of two clean latents.
