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
      row-max step, wider (vec4) shared-memory tile loads for the core GEMM
      kernel, and a second query row per thread in the flash-attention
      kernel.
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
- [ ] The text encoder is imported whole and then truncated, so a large
      fraction of it is fetched, dequantised and validated only to be
      discarded. `pipeline.rs` builds `Shard { start: 0, end: deepest tap,
      embed: true, head: false }`, but that truncation happens at BUILD time,
      after the import has already insisted on the whole checkpoint:
      `checkpoint::safetensors::read_model_dir` reads every shard named in
      the index's `weight_map` unconditionally (it takes no parameter saying
      what the caller wants), and `qwen3::import::brain_init_from_hf`
      enforces two-way coverage against the full `param_list()` of a config
      whose `n_layers` is the untruncated count with `tie_embeddings: false`.
      For the Qwen3-8B encoder that means the layers past the deepest tap and
      the LM head - about 4.2 GB of 15.6 GB - are downloaded and checked, and
      the LM head is never read by any shard the pipeline builds. On a
      bandwidth-limited box that is most of an hour before the first image.
      The fix is a shard-aware import: derive the required `param_list()`
      from the `Shard` the caller will build, and let `read_model_dir` take
      the resulting name set so it can skip whole shard files. Note
      `hf_source`'s streaming path is NOT this fix - it lowers the ~32 GB
      host-RAM import peak but validates against the same full list, so it
      saves memory and not bytes. Keep the two-way coverage check: it is what
      catches a wrong checkpoint, and it must stay exact against whatever set
      is genuinely required.

The core GEMM kernel already runs near a structural throughput ceiling for
its current shared-memory tiling scheme, and batching the diffusion
transformer's forward pass has a small, bounded payoff because its GEMMs are
already near their row-count-independent plateau at a single sample; most of
a served image's latency lives in the (currently unbatched) text encoder and
VAE decode rather than in the transformer itself.
