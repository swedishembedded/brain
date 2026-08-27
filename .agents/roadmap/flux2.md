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

## `--strength` stops being a hidden mode switch (2026-08-27)

`pipeline::ref_skip` used to drop the FIRST reference from the conditioning
set whenever `--strength < 1.0`: it became the init latent and contributed
zero reference tokens and zero position ids. At exactly `1.0` it was not
consumed as an init latent, so it re-entered as attended conditioning. A dial
that presents as continuous was therefore a mode switch, with nothing in the
interface saying so, and the entire range `(0, 1)` ran the DiT **blind to the
photograph** - the reference reached it only as leftover signal in a
partially-noised latent.

That is now fixed. A supplied reference always conditions. Under
`strength < 1` the first reference does double duty: init latent *and*
conditioning input. Because the init role pins it to the output size, its
conditioning copy is downscaled by `GenOpts::ref_cond_scale` /
`--ref-cond-scale`, default `0.75` linear (a bit over half the tokens).
`1.0` conditions at full size, `0` is the documented escape hatch back to the
old cheap behaviour. `ref_skip` is gone; `cond_sizes` is the single rule that
the sizing entry point, the position-id builder and the denoise loop all read.

**Why a downscale and not a budget refusal.** Three options were on the table:
downscale by default, require an explicit resolution, or pre-flight the device
budget and refuse. The downscale wins on evidence: the only virtual-staging
result the user has accepted was `strength 1.00` driven by a 768x576
reference, i.e. 1728 reference tokens against 3072 generated ones. At
1024x768 the 0.75 default reproduces exactly that joint sequence, so the
default lands on the measured-good configuration rather than on a guess.
Requiring an explicit resolution leaves the broken default in place for
everyone who does not read the flag; a budget refusal is a bigger change (the
DiT and the text encoder sit on different cards under different authorities)
and is orthogonal - it should be added later, for the case where the user
raises `--ref-cond-scale` themselves. What did land for legibility is a
per-reference breakdown on stderr naming each reference's supplied size, its
conditioning size and its token count, so an OOM can be attributed to a
reference rather than read as a bare `wgpu error: Out of Memory`.

**Backwards compatibility: deliberately broken.** Every existing
`--strength` + `--ref` invocation now produces a different image. That is the
point - the old output was produced without the model seeing the reference.
`--strength 1.0` is unchanged and is fenced by byte equality, below.

Measured on **2x Tesla P40**, `klein-9b` Q8_0 int8 DiT on gpu0 + truncated
int8 Qwen3-8B text encoder on gpu1 (`BRAIN_FLUX2_TE_DEVICE=gpu1:i8`),
1024x768 out, seed 7, 12 steps, the rank-32 staging LoRA at scale 1.0, the
model card's verbatim prompt tier wording. "pipeline" is the CLI's own total,
"wall" includes weight load; peak is `nvidia-smi` sampled at 2 Hz.

| run | reference | tokens | pipeline | wall | gpu0 peak | gpu1 peak |
|---|---|---|---|---|---|---|
| before, strength 1.00 | 768x576 | 3072 + 1728 | 119.9 s | 160 s | 24301 MiB | 14105 MiB |
| after, strength 1.00 | 768x576 | 3072 + 1728 | 118.5 s | 156 s | 24038 MiB | 14101 MiB |
| before, strength 0.95 | 1024x768 | 3072 + **0** | 71.4 s | 109 s | 23403 MiB | 14101 MiB |
| after, strength 0.95 | 1024x768 | 3072 + 1728 | 121.7 s | 159 s | 24039 MiB | 14101 MiB |
| after, strength 0.90 | 1024x768 | 3072 + 1728 | 121.1 s | 166 s | 24039 MiB | 14101 MiB |
| after, strength 0.85 | 1024x768 | 3072 + 1728 | 122.3 s | 163 s | 24039 MiB | 14101 MiB |

The cost of the change at `strength 0.95`, 1024x768: +1728 reference tokens,
denoise 64.5 s -> 113.3 s (1.76x for a 1.56x joint image sequence, i.e.
sub-quadratic here), device peak +636 MiB on the DiT card, and a
reference-encode phase that goes from **absent** to 3.5 s - two VAE encodes,
the full-size init latent plus the downscaled conditioning copy. The absence
of that phase in the before run is the defect showing up directly in the
CLI's own per-phase timing: nothing was encoded as a reference at all.

**The dial now varies smoothly.** Edge correlation (Pearson of
full-resolution Sobel gradient magnitudes vs the source photograph) and MAD
(mean |RGB| in 0..255), same definitions as the masked-editing table above:

| setting | edge corr | MAD |
|---|---|---|
| after, strength 0.85 | 0.729 | 14.2 |
| after, strength 0.90 | 0.675 | 18.0 |
| after, strength 0.95 | 0.486 | 30.1 |
| after, strength 1.00 (768 ref) | 0.278 | 54.1 |
| *before*, strength 0.95 | *0.290* | *46.9* |

The old `0.95` sat at 0.290 - next to `1.00`'s 0.278, not between `0.90` and
`1.00`. That is the cliff, in numbers: the top of the range collapsed onto the
endpoint because the model had nothing to preserve *from*. The new rungs are
monotone and evenly spaced, and every one of them was rendered with the DiT
attending to the room.

**What is bit-identical, and what is parity-gated.** `strength 1.0` is
**bit-identical**: the same invocation before and after the change renders the
same PNG, sha256 `85076a98...` on the real 9B int8 weights (table row 1 vs
row 2). At the unit level the same claim is fenced by a golden FNV-1a digest
of a stub-denoiser render, measured on the pre-change `pipeline.rs` from git
and asserted by
`pipeline::tests::a_strength_one_run_is_byte_identical_to_the_pre_change_output`.
Everything under `strength < 1` is **intentionally not** bit-identical and is
not parity-gated against the old behaviour - there is no parity to claim
against a run that ignored its own input.

## `--strength` becomes a dial instead of a cliff (2026-08-27)

The change above made a supplied reference condition the model at every
`--strength`. It did not make `--strength` continuous, and the user's report
after it was precise: `1.00` stages the room - new bed, new textiles, new
furniture - and `0.99` repaints the bedding that is already there. Two
different *jobs*, one dial position apart.

**It was a true code-path discontinuity, not a steep response.** At
`strength < 1` the sampler integrated a **uniform ramp** over `[strength, 0]`;
at exactly `1.0` it integrated `klein_sigmas`. Those are different samplers.
At 12 steps and 3072 generated tokens the two schedules are nowhere near each
other:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `klein_sigmas` (`strength 1.0`) | 1.000 | 0.990 | 0.977 | 0.963 | 0.945 | 0.923 | 0.896 | 0.860 | 0.812 | 0.742 | 0.633 | 0.439 |
| uniform ramp (`strength 0.999`) | 0.999 | 0.916 | 0.833 | 0.749 | 0.666 | 0.583 | 0.500 | 0.416 | 0.333 | 0.250 | 0.167 | 0.083 |

klein is a distilled few-step sampler: its schedule is exponentially shifted
by a token-count- and step-count-dependent `mu`, it spends eleven of twelve
steps above sigma 0.43, and it resolves the image in one long final leap. The
uniform ramp descends evenly and resolves gradually. Crossing between them at
`1 − 1e-3` swapped the sampler, and the two samplers do different things with
the same weights, the same prompt and the same seed - which is exactly what
"a different job" looks like from the outside. The init latent
`(1−s)·x₀ + s·ε` was never the problem: it is continuous in `s` by
construction and at `s = 0.999` carries 0.1% of the photograph.

The fix is the one that a true discontinuity calls for, and it is not a
remap: `pipeline::img2img_sigmas` is now `klein_sigmas` **scaled** into
`[0, strength]`. `strength` still sets the noise level the init latent is
mixed at - it has to, the model must be asked to denoise the distribution it
was handed - but the *shape* of the descent is klein's own at every setting.
`strength = 1` then reproduces `klein_sigmas` bit for bit (`1.0 · x` is exact
in IEEE), so the dial **reaches** free generation rather than approaching it,
and every entry is linear in `strength`, so lowering the dial lowers every
sigma by at most `1 − strength`.

Scaling, not slicing. Slicing the distilled schedule remains wrong for the
reason recorded when the uniform ramp was written: `klein_sigmas`' lowest
non-zero entry is 0.56 at 8 steps and 0.75 at 4, so there is no low-noise
entry point to start an img2img from and the caller's step budget would
silently collapse. Scaling keeps every step the caller asked for.

### The response curve, measured

Same 2x Tesla P40 placement and the same metric definitions as the tables
above (`edge corr` = Pearson correlation of full-resolution Sobel gradient
magnitudes against the source; `MAD` = mean |RGB difference| in 0..255).
`image-02`, seed 7, 12 steps, `klein-9b` int8, the rank-32 staging LoRA at
1.0, the model card's verbatim tier wording. `1.00` is driven by a 768x576
reference; every lower rung by a 1024x768 one, whose conditioning copy the
default `--ref-cond-scale 0.75` puts back at 768x576 - so every rung in the
ladder carries the same 3072 + 1728 joint sequence.

| strength | before edge corr | before MAD | after edge corr | after MAD |
|---|---|---|---|---|
| **1.00** (768 ref) | **0.279** | **54.1** | **0.279** | **54.1** |
| 0.995 | 0.383 | 46.6 | 0.297 | 50.2 |
| 0.99 | *0.380* | 45.9 | 0.315 | 46.4 |
| 0.98 | 0.391 | 41.7 | 0.345 | 41.4 |
| 0.97 | 0.399 | 38.9 | 0.390 | 38.1 |
| 0.96 | 0.440 | 33.7 | 0.450 | 32.4 |
| 0.95 | 0.484 | 30.1 | 0.503 | 28.9 |
| 0.90 | 0.676 | 18.0 | 0.690 | 17.0 |
| 0.80 | 0.767 | 12.4 | 0.789 | 11.5 |
| 0.60 | 0.832 | 9.8 | 0.844 | 9.2 |
| 0.40 | 0.890 | 7.9 | 0.894 | 7.6 |
| 0.20 | 0.953 | 5.4 | 0.954 | 5.4 |
| 0.00 | 0.985 | 3.4 | 0.985 | 3.4 |

The `1.00` row is the same PNG in both columns - that is the bit-identity
claim below. Everything else moved.

Two things to read off it. **The first step off the top.** Before, one 0.005
move cost 0.104 of edge correlation; after, 0.018. Before, five sixths of the
dial's whole range was spent between `1.00` and `0.995` and the next eight
rungs down to `0.95` added 0.10 between them; after, those same rungs are
evenly spaced. **The reversal.** Before, `0.99` (*0.380*) is *less* faithful
to the source than `0.995` (0.383) - lowering the dial made the output drift
further from the photograph. After, both metrics fall monotonically over all
thirteen rungs with no reversal anywhere. The bottom rung is the other end of
the claim: `--strength 0` returns the photograph through the VAE round trip on
the real weights and not only in the unit gate, and it does so from both trees
- the shape of the schedule stops mattering once the trajectory has almost no
distance to travel.

The before column is flat from `0.995` down: one step off the top already
spent most of the dial's range, and everything below it was the same picture
with slightly different colour. The after column spreads that range across
the parameter. Looking at the images rather than the numbers: before, the job
changed between `1.00` and `0.995` and never changed again; after, `0.995`,
`0.99` and `0.97` are all still staging the room with progressively more of
the photograph showing through, and the source bedding comes back gradually
from about `0.98`.

### Whether to remap the dial on top of this, and why it was not done

With the sampler discontinuity gone the residual response is steep but smooth,
and it is a property of the model rather than of the code: a few percent of
the photograph in the init latent is enough to pin the room's low-frequency
layout. Fitting the after column gives

    edge corr ~ 1.07 * (1 - strength) ^ 0.26

over the whole ladder, so the transform that would make perceptual anchoring
roughly linear in the dial is `strength = 1 - (1 - dial)^4` (the fitted
exponent is 1/0.26 = 3.9). It would satisfy every constraint the current
behaviour does: exact at both ends (`dial 1 -> 1`, `dial 0 -> 0`), monotone
(`d/d(dial) = 4(1-dial)^3 >= 0`), and *more* continuous at the top, since
`dial 0.99` would map to `1 - 1e-8`.

It was **not** applied, and the reasons are worth keeping because the fit
above makes it a one-line change anyone could reach for later:

* `--strength` currently *is* the noise level sigma_0 the init latent
  `(1-sigma)*x0 + sigma*eps` is mixed at. That is the same quantity the
  trainer's forward process uses, the same one `--mask` renoises to, and the
  same one every other tool in this ecosystem spells `strength` / `denoise`.
  A remap makes the flag a private curve and leaves nothing that names the
  sigma.
* It reprices every existing invocation a second time in one day, and the
  useful band would move from `1.00..0.90` to `1.00..0.45` - so the settings
  in anyone's scripts would mean something new again, for a convenience gain
  rather than a correctness one.
* The residual steepness is real information about the model. Flattening it
  into the parameter hides that a 1% init-latent mix already decides the
  composition.

The user-facing requirement - lower the dial and get gradually more of the
photograph, with `0.99` doing `1.00`'s job - is met by the sampler fix alone,
and is met by measurement rather than by argument. A remap is a separate
product decision, and the fit above is what it should be built on.

### What is bit-identical, and what is parity-gated

`--strength 1.0` is **bit-identical**: the same invocation renders sha256
`85076a98...`, unchanged, on the real 9B int8 weights - verified by rendering
it with a binary built from the tree before this change and with one built
after, and comparing the PNGs byte for byte, not by reasoning about the
branch. Everything under `strength < 1` moves deliberately and is not
parity-gated against the old behaviour; there is no parity to claim against a
sampler that was swapped out from under the dial.

### The gates, and the mutation each one died to

| gate | mutation that kills it |
|---|---|
| `the_img2img_schedule_reaches_the_free_generation_schedule_exactly` | scale the dial by `0.999` |
| `a_hair_below_full_strength_is_a_hair_from_the_full_strength_schedule` | restore the uniform ramp (worst \|Δσ\| 0.504 against a 0.05 bound) |
| `lowering_the_strength_lowers_every_sigma_and_raises_the_source_weight` | a non-monotone remap `s + 0.2·sin(30 s)` |
| `both_spellings_of_full_strength_integrate_the_klein_schedule` | shrink the free-generation schedule by 0.8 |
| `the_img2img_branch_integrates_the_img2img_schedule` | let the pipeline build a uniform ramp again |
| `a_hair_below_full_strength_renders_what_full_strength_renders` | restore the uniform ramp (cosine 0.9984, rel_l2 0.0574 against 0.999 / 0.02) |
| `anchoring_increases_monotonically_as_strength_falls` | drop `strength` from the init latent |
| `a_vanishing_strength_returns_the_source` | drop `strength` from the init latent |

**A gate that could not fail, and what replaced it.**
`a_strength_one_run_is_byte_identical_to_the_pre_change_output` is a golden
digest over a stub-denoiser render, and it survives *any* change to the sigma
schedule. `Stub`'s velocity is `(x − g)/σ`, whose exact Euler solution is
`x = g + C·σ`: one integration lands on `g` from any init latent over any
sigma list, so the rendered bytes cannot move when the schedule does
(confirmed by mutation - shrinking `klein_sigmas` by 0.8 left the digest
untouched). It is a real fence on the conditioning and token path and it is
kept as one; it is **not** a fence on `--strength`, and reading it as one is
the trap. `both_spellings_of_full_strength_integrate_the_klein_schedule`
asserts the sigmas the DiT was evaluated at directly and does die to that
mutation.

For the same reason the end-to-end strength gates run against a second stub,
`Flow`, whose clean-image estimate is `x̂₀ = (1−σ)·x + σ·g` - it commits to
its own idea at high noise and trusts what it sees at low noise, giving the
**bounded** velocity `v = x − g`. Under `Stub` every strength renders the
same image, so a monotonicity or continuity assertion written against it
could not fail however the dial was wired.

## Device (WGSL) LoRA trainer - what was built and what it costs

`crates/flux2/src/devgrad.rs` (persistent block engine) +
`crates/flux2/src/devtrain.rs` (whole-model step) are the GPU instantiation of
the op sequence `grad.rs`/`modelgrad.rs` define. `brain flux2 finetune
--trainer device|host` selects between them; `device` is the default and the
choice is printed at the top of every run.

### Correctness

Gated against the finite-difference-gradchecked host reference on a Tesla P40:

| gate | tensors | worst cosine | worst rel_l2 |
|---|---|---|---|
| `dev_grad.rs` double block | 45 | 1.000000000 | 7.772e-7 |
| `dev_grad.rs` single block | 20 | 1.000000000 | 8.469e-7 |
| `device_train.rs` whole model | 96 | 1.000000000 | 9.632e-7 |

`device_train.rs` also gates that a `B = 0` adapter reproduces the base loss
exactly (1.369876396 host vs 1.369876531 device) and that LoRA-only training
drives a batch's loss from 1.116545 to 0.000227 with the base frozen.

Mutations each gate died to (every one run, every one killed the gate):

| mutation | outcome |
|---|---|
| LayerNorm/QK-norm eps 1e-6 → 1e-5 | **cosine stayed 1.000000000**; caught only by rel_l2 (5.454e-5 vs a 1e-5 bound) |
| RoPE backward rotates by +angle (`nsin` → `sin`) | cosine -0.049 (single), -0.242 (double) |
| drop the low-rank `dxa·A` term from `dx` | cosine 0.403 (single), 0.341 (double) |
| pack V un-transposed for the apply GEMM | cosine 0.089 (single), -0.184 (double) |
| drop the `1/√head_dim` fold from the q-norm weight | cosine 0.566 (single), 0.301 (double) |
| double-block arms swap the img/txt modulation sites | loss 1.4467 → 1.4637 |
| backward unwinds the single stack in forward order | cosine 0.995, rel_l2 9.5e-2 |

The epsilon row is the reason both metrics are asserted everywhere: a gate on
cosine alone would have passed it.

### Cost, before and after the attention rewrite

`tests/dev_step_time.rs --ignored`, klein-4b at 512 px (1536 joint tokens =
512 text + 1024 image), rank 16, fp32 frozen base resident at 13.93 GiB on one
P40, best-of-3 after a discarded warm-up, nothing polling `nvidia-smi`:

| | s / step | 1500-step run | GPU kernel time / step |
|---|---|---|---|
| naive `attn_*_bidir` attention | ~98 | ~41 h | 97 782 ms |
| GEMM attention (`head_pack` + register-tiled matmul) | **11.74** | **4.89 h** | 10 522 ms |

Before, by share of GPU kernel time: `attn_bwd_dscores_bidir` 61.6 %,
`attn_scores_bidir` 22.2 %, `attn_softmax_bidir` 4.3 %, `attn_apply_bidir`
1.5 % - 89.6 % in four attention kernels, 7.0 % in the GEMMs that do the
model's actual arithmetic. `attn_bwd_dscores_bidir` is one thread per
`(head, query)` walking a full `n·head_dim` inner product out of an
interleaved `[n, nh·hd]` layout **twice** (once for the softmax dot, once to
write), so consecutive lanes are `nh·hd` floats apart.

After, by share (90 % of wall clock is GPU kernel time):

| kernel | ms / step | share | calls / step |
|---|---|---|---|
| `matmul_reg3` | 5251 | 49.9 % | 4053 |
| `matmul_dx_reg` | 2434 | 23.1 % | 1441 |
| `attn_softmax_bidir` | 799 | 7.6 % | 1200 |
| `matmul_dw_reg` | 729 | 6.9 % | 1620 |
| `softmax_k_dx` | 420 | 4.0 % | 600 |
| `rmsnorm_dx_eps` | 284 | 2.7 % | 60 |
| `rmsnorm_dw` | 175 | 1.7 % | 60 |
| everything else | ~430 | 4.1 % | - |

79.9 % of the step is now in the three register-tiled GEMMs. Remaining
headroom, in order: `attn_softmax_bidir` (one thread per row, uncoalesced -
`softmax_rows` is the coalesced sibling, but it is a workgroup-barrier
reduction and needs a barrier-free fallback before the CPU backend can use
it), and the RMSNorm backward pair (`rmsnorm_dx_eps`/`rmsnorm_dw` are
per-element kernels at `head_dim`-wide rows; only the forward has a `_rows`
variant).

### Why the LoRA structure changes the graph

Only the rank-`r` factors are differentiated. For a targeted linear
`y = x·Wᵀ + x·Aᵀ·B̃ᵀ` the adapter gradients come straight out of the low-rank
intermediates (`dA = dxaᵀ·x`, `dB̃ = xaᵀ·dy` with `dxa = dy·B̃`), which is
algebraically the host path's `Pair::project` of a dense `dW` - asserted as
such in `dev_grad.rs`. The consequence is that the `dW` GEMM, one third of
every backward's arithmetic, is replaced by two GEMMs of rank width, no
`[out, in]` gradient buffer is ever allocated, and the frozen base is only
ever read.

### Can the frozen base be int8? What the measurement says

`tests/int8_base_grads.rs --ignored`, one REAL `klein-9b` double block out of
the released Q8_0 GGUF, 768 joint tokens (512 text + 256 image), rank 16,
adapter `B` non-zero. The base is round-tripped through brain's own
per-output-row symmetric int8 grid (`model::int8::row_scale` /`pack_row` -
exactly what `matmul_i8_dyn` consumes) and the identical backward is run on
both bases.

**The weights themselves** move by rel_l2 8.4e-3 .. 1.16e-2 per tensor
(cosine 0.99993 .. 0.99996); the `img_mlp.2` matrix is the worst and the text
stream's attention projections the best.

**The adapter gradients** that follow:

| | worst | best | where |
|---|---|---|---|
| cosine | 0.999530 | 0.999928 | worst at `img.wq.dA` |
| rel_l2 | 3.076e-2 | 1.200e-2 | worst at `img.wq.dA` |

and the block's input gradient `dx` at cosine 0.999938 / rel_l2 1.115e-2. The
error grows toward the *front* of the block (q/k/v worse than w1/w3/w2),
which is what a backward accumulating quantization error along its chain
should do.

**Reading it.** A 0.9995 cosine is a 1.8-degree direction error and 3% of
magnitude, against a per-step stochastic gradient that a single-sample
rectified-flow batch already draws a fresh sigma and a fresh noise vector for.
That is not the thing that will decide whether an adapter trains. So the
answer is: **the weight-quantization term is affordable**, and int8 is worth
building.

One thing this does NOT say: it does not cover the **activation**
quantization a real `matmul_i8_dyn` adds (a fresh per-token scale on every
activation feeding every linear). That term is unmeasured here and has to be
measured on its own before an int8 trainer is trusted.

### What an int8 base would actually buy, and what it would cost to build

The prize is **not** mainly memory, and an earlier draft of this section had
the memory arithmetic backwards. Setting it straight:

* the forward needs `W` row-quantized. The backward's `dx = dy.W` contracts
  over `W`'s ROW axis, where a per-row scale cannot be factored out of the
  sum - so a dp4a `dx` needs a SECOND, transposed int8 copy. That is 2x off
  fp32, not 4x.
* 2x is still enough. klein-9b's 9.05 G parameters are 36.2 GB at fp32 and
  about 18.1 GB as two int8 copies; next to roughly 2.4 GB of activations at
  1536 joint tokens that is about 20.5 GB, which **does** fit one 24 GiB card,
  with a few GB of headroom. So int8 collapses the two-card split back to one
  card.
* the bigger prize is the **dequantise that never happens**. The released
  klein-9b DiT is Q8_0, and both trainers today go through
  `read_dit_tensors`, which materialises the whole thing as host fp32 before a
  single step runs - that expansion, not the training, is what makes the first
  step so far away (the host trainer has been observed spending over an hour
  at roughly 100 GB RSS without reaching step 1 at size 512).
  `weights::DitWeights::try_i8_rect` already goes Q8_0 -> packed int8 with no
  fp32 intermediate for the inference path; a trainer that took its frozen
  base the same way would inherit that and start in minutes.
* what is missing to build it: the transposed copy needs a requantisation
  along the OTHER axis (Q8_0 blocks run along rows, so the transposed
  direction cannot reuse `try_i8_rect`'s block-aligned fast path and has to be
  built explicitly), plus an int8 `dx` GEMM. Neither is speculative - the
  fidelity measurement above says the weight term is affordable - but both are
  real work, and the activation term still has to be measured first.

Until then the two-card fp32 split, which exists and is gated bit-for-bit,
reaches the same place with no fidelity question at all - at the cost of the
dequantise and a second card.

## Roofline: what a training step OUGHT to cost, and what it does

Everything below is klein-4b at 512 px (1536 joint tokens = 512 text + 1024
image), rank 16, fp32 frozen base resident at 13.93 GiB on ONE Tesla P40,
best-of-3 after a discarded warm-up, nothing polling `nvidia-smi` while the
clock runs. Roofs are the MEASURED ones for this box: 10 517 GFLOP/s fp32,
287.5 GB/s DRAM, ridge point 36.6 FLOP/B.

### The model, and whether to believe it

`flux2::devtrain::step_flops` is an analytic FLOP model derived from the
config (`2*M*K*N` per GEMM, matching `gpu_core::cost`'s convention). It says
one step is **31 493.9 GFLOP**. The runtime dispatch tally over the same three
steps says **31 532.8 GFLOP** - **+0.1%**. The model is not a rule of thumb;
it is checked against the machine every time the harness runs.

| term | GFLOP | share |
|---|---|---|
| base linears, forward | 9422.1 | 29.9 % |
| base linears, recompute | 9422.1 | 29.9 % |
| base linears, backward (`dx` only) | 9422.1 | 29.9 % |
| attention, forward | 724.8 | 2.3 % |
| attention, recompute | 724.8 | 2.3 % |
| attention, backward | 1449.6 | 4.6 % |
| adapter, forward | 75.5 | 0.2 % |
| adapter, recompute | 75.5 | 0.2 % |
| adapter, backward | 151.0 | 0.5 % |
| embedders + head | 26.6 | 0.1 % |
| **device total** | **31 493.9** | |
| conditioning front (host, `m = 1`) | 0.34 | - |

Three things this makes concrete. The **adapter's own arithmetic is 0.9 % of
the step** - at rank 16 it really is free, stated as a number rather than
assumed. The **backward of a frozen linear is one GEMM, not two**: `dx` is
computed and `dW` never is, so `linear_bwd == linear_fwd` where a full
fine-tune would pay double. And the **recompute is 32.5 % of the device
total** - the reverse sweep re-runs each block's forward, and that is the
single largest tradeable term in the budget.

### Which roof binds

The step moves **247.3 GB** and does 31.5 TFLOP: arithmetic intensity
**127.3 FLOP/B**, well above the 36.6 ridge. So:

| floor | s/step | 1500 steps |
|---|---|---|
| compute-bound (31.5 TFLOP / 10 517 GFLOP/s) | **2.99** | **1.25 h** |
| memory-bound (247.3 GB / 287.5 GB/s) | 0.86 | 0.36 h |

**COMPUTE-bound, by 3.5x.** Making the memory-bound kernels faster buys only
what those kernels themselves cost; it cannot move the floor. Measured
**11.83 s/step = 25.3 % of the floor, 3.95x off**.

### Where the 11.83 s goes

Host phases first, because 11.5 % of the wall clock is not GPU kernel time:

| phase | s/step | share |
|---|---|---|
| backward sweep | 7.737 | 65.0 % |
| forward sweep | 3.191 | 26.8 % |
| optimiser (Adam) + unaccounted | 0.516 | 4.3 % |
| gradient readback | 0.317 | 2.7 % |
| upload adapter | 0.101 | 0.8 % |
| host conditioning | 0.024 | 0.2 % |
| upload mods + rope + batch | 0.007 | 0.1 % |
| head + loss | 0.007 | 0.1 % |

GPU kernel time is 10.54 s, and the two sweeps wall-clock at 10.93 s, so the
**submit/poll bubble across roughly 4000 dispatches is 0.39 s** - about
0.1 ms per dispatch. That is small, and it is a measured answer to "does
launch overhead matter now that the kernels are 8x cheaper": not yet.

### Per kernel, against the roof that actually binds it

`%roof` is against DRAM below the ridge point and against the fp32 FMA rate
above it - a memory-bound kernel compared to a compute roof would look
falsely terrible.

| kernel | ms/step | share | GFLOP/s | GB/s | FLOP/B | %roof |
|---|---|---|---|---|---|---|
| `matmul_reg3` | 5259.2 | 49.9 % | 3953 | 18.1 | 217.9 | **37.6 %** |
| `matmul_dx_reg` | 2435.7 | 23.1 % | 4066 | 18.7 | 217.0 | **38.7 %** |
| `attn_softmax_bidir` | 806.0 | 7.6 % | 21 | 28.1 | 0.8 | 9.8 % |
| `matmul_dw_reg` | 728.9 | 6.9 % | 1098 | 32.7 | 33.5 | 11.4 % |
| `softmax_k_dx` | 420.6 | 4.0 % | - | - | - | (uncovered) |
| `rmsnorm_dx_eps` | 284.2 | 2.7 % | 8 | 13.3 | 0.6 | 4.6 % |
| `rmsnorm_dw` | 175.0 | 1.7 % | 4 | 10.8 | 0.4 | 3.8 % |
| `layernorm_dx` | 79.1 | 0.8 % | 27 | 29.3 | 0.9 | 10.2 % |
| `layernorm` | 76.4 | 0.7 % | 30 | 30.0 | 1.0 | 10.4 % |
| `head_pack` | 39.9 | 0.4 % | 21 | 165.4 | 0.1 | 57.5 % |
| `silu_mul` | 31.8 | 0.3 % | 111 | 266.7 | 0.4 | **92.8 %** |
| `head_unpack` | 28.6 | 0.3 % | - | 165.1 | - | 57.4 % |
| `rope_interleave_table` | 20.9 | 0.2 % | 102 | 276.4 | 0.4 | **96.1 %** |
| `silu_bwd_da` | 20.5 | 0.2 % | 138 | 276.8 | 0.5 | **96.3 %** |
| `add2` | 19.9 | 0.2 % | 17 | 199.4 | 0.1 | 69.4 % |

The pattern is not subtle. Everything that maps **one thread per element** is
at 69-96 % of DRAM. Everything that maps **one thread per ROW** - the
softmaxes, both RMSNorm backward halves, both LayerNorms - is at 4-10 %,
because a warp's 32 lanes then sit on 32 different rows, one useful float per
sector. Together those six kernels are 1841 ms, **17.5 % of GPU time**, doing
work whose memory floor is about a tenth of that.

`gpu_core::cost` covers 91.1 % of the dispatches; `softmax_k_dx`, `film_row*`,
`gate_row_d*` and `add_inplace` have no cost formula, so their FLOPs and bytes
are missing from the totals (never counted as zero - the harness names them).

### Negative results from this pass

* **The adapter upload/readback is not the host bottleneck.** The step does
  420 small buffer writes and 420 small reads (one per adapter factor per
  direction), which looked like the obvious host cost. Measured: 0.101 s up +
  0.317 s down = 3.5 % of the step. Batching them into a few large transfers
  is not worth the packing complexity. The real top host cost was the **Adam
  optimiser at 0.516 s**, which was not on the hypothesis list.
* **Kernel launch overhead is not yet a problem.** 0.39 s of bubble across
  ~4000 dispatches. `gpu_core::scratch`'s replay arena addresses buffer
  destruction, and nothing here destroys buffers inside a step - the engine is
  persistent by construction.

### After this pass

Same harness, same box, same method, now in the TRAINER's configuration
(frozen QK gain gradient off - the parity gate's configuration is more
expensive and no run pays it):

| | s/step | 1500-step run | % of the 2.99 s floor |
|---|---|---|---|
| naive `attn_*_bidir` attention | ~98 | ~41 h | 3.1 % |
| GEMM attention | 11.83 | 4.93 h | 25.3 % |
| **+ this pass** | **10.53** | **4.39 h** | **28.4 %** |

| what | before | after |
|---|---|---|
| Adam (host) | 0.516 s | **0.028 s** |
| `attn_softmax_bidir` -> `softmax_rows` | 806.0 ms @ 9.8 % of DRAM | **211.9 ms @ 37.2 %** |
| `layernorm` + `layernorm_dx` -> `*_rows` | 155.5 ms @ ~10 % | left the top sixteen |
| frozen QK gain gradient | 206 ms | not computed |

GPU kernel time is now 93.1 % of wall clock (was 88.5 %), so the host side is
close to spent: everything outside the two sweeps is 0.43 s of a 10.53 s step.

### What is left on the table, with what it is worth

The step is now **85.8 % GEMM**, so the remaining question is one number:
`matmul_reg3` and `matmul_dx_reg` run at **37.5 % and 38.5 % of the fp32
roof**. Together they are 7.7 s of the 10.53 s step against a 2.7 s floor for
the arithmetic they do.

* **The GEMMs, ~2.9 s if they reached 60 %.** 38 % is suspiciously close to
  the occupancy an 8x8 register tile allows on Pascal: 64 accumulator
  registers plus the A/B fragments puts a 256-thread workgroup near 80
  registers per thread, which caps an SM at 3 concurrent workgroups out of the
  8 its 2048-thread budget would otherwise allow - about 37 % occupancy, and
  too few warps to hide global-load latency. That is a diagnosis, not a
  measurement, and testing it means varying the register tile in a kernel
  EVERY model in the repo shares. Highest value, highest blast radius.
* **The recompute, 3.4 s of arithmetic (32.5 %).** Stashing the eight
  per-block tensors that cost a GEMM to recreate (`q`,`k`,`v`,`ctx`,`proj`,
  `h1`,`h2`,`mlpo`) would skip the base-linear recompute entirely. That is
  226 MB per block, 5.7 GB for all 25 - against roughly 5 GB spare on a 24 GiB
  card once the 13.93 GiB base and the engine buffers are placed. So it is
  a partial win on klein-4b (about 19 of 25 blocks fit, ~1.8 s) and needs
  per-block private buffer sets to implement.
* **`softmax_k_dx`, 425 ms (4.3 %)** and **`rmsnorm_dx_eps`, 284 ms (2.9 %)**
  are the last two one-thread-per-row kernels, at 0 % and 4.6 % of the DRAM
  roof. Neither has a coalesced sibling, so each needs new WGSL - roughly
  0.55 s between them. `softmax_k_dx` additionally has no `gpu_core::cost`
  row, so its FLOPs and bytes are missing from the tally.

Adding those up: the reachable step is roughly 5-6 s without touching the
shared GEMM, and about 3.5 s with it. The floor is 2.99 s, so **1.25 h is the
hard bound for a 1500-step run on one P40** and there is no version of this
that reaches "minutes".
