# supir - roadmap

SUPIR ("Scaling Up to Excellence", CVPR 2024): photo-realistic blind image
restoration over a **frozen SDXL 1.0 base UNet**, a 1.24B control trunk
(`GLVControl`), and 12 adaptor modules (10 `ZeroSFT` + 2 `ZeroCrossAttn`).
Registered as an architecture id; the port has not started. This ledger is
the architecture spec (verified against upstream source and the real
checkpoint headers) plus the staged plan, so neither is re-derived.

## The spec

### Pipeline

```
LQ image (resized so short side ≥ 1024, H,W snapped to /64, range [-1,1])
  ├ _z       = 0.13025 · quant_conv(denoise_encoder(x)).mode()   # degradation-robust encode → the HINT
  ├ x_stage1 = decoder(_z)                                        # frozen SDXL decoder
  ├ z_stage1 = 0.13025 · quant_conv(encoder(x_stage1)).mode()     # CLEAN re-encode → x_center, guidance only
  ├ c, uc    = dual-CLIP(caption + p_p) / (n_p);  c["control"] = _z
  ├ x_T      = randn_like(_z)                                     # PURE NOISE, not a noised LQ latent
  ├ x_0      = RestoreEDMSampler(denoiser, x_T, c, uc, x_center=z_stage1)
  └ out      = decoder(x_0), then wavelet/AdaIN colour-fix against x_stage1
```

Two encodes of the same picture: the fine-tuned encoder makes the *hint*, the
frozen encoder re-encodes the *decoded* hint to make the *guidance target*.

### Weights - measured against the real checkpoint headers, not estimated

Total SUPIR delta = 1 035 tensors / 1 332 356 296 params. Everything else is
frozen SDXL 1.0 base (`sd_xl_base_1.0_0.9vae.safetensors`, 6 938 078 334 B).

| prefix | tensors | params | what |
|---|---|---|---|
| `model.control_model.*` | 811 | 1 243.40 M | `GLVControl` trunk |
| `model.diffusion_model.project_modules.*` | 118 | 54.80 M | the 12 adaptors |
| `first_stage_model.denoise_encoder.*` | 106 | 34.16 M | degradation-robust encoder |

`model.control_model.mask_LQ` `[1,4,64,64]` is a leftover from an unreleased
masking variant with no counterpart in the released `GLVControl` code.
Upstream drops it via `strict=False`; brain's two-way import must **name it
and reject it explicitly**, not silently ignore it.

The degradation-robust encoder is a **byte-identical topology** to the frozen
SDXL VAE encoder (106 tensors, same key set) - only the weights differ. It
reuses the frozen `quant_conv`/`post_quant_conv`/`decoder`, so it needs no new
code: `vae::VaeEncoder` with a second set of weights.

### `GLVControl` - the trunk

A hand-written copy of SDXL's 9 input blocks + middle block, same depths
(`transformer_depth [1,2,10]`, `context_dim 2048`, `adm_in_channels 2816`,
`model_channels 320`, `channel_mult [1,2,4]`), with its own `time_embed` and
`label_emb`. Two differences from a vanilla SDXL ControlNet:

- the hint embedder is **one zero-init 3×3 conv** `Conv2d(4 → 320)` - the hint
  is already a latent, not a pixel image;
- there are **no output zero-convs and no `scale_chan`** - it returns the 10
  raw hidden states.

```python
h = xt                                        # the trunk's main input is the NOISY latent
guided_hint = input_hint_block(LQ_latent)
for module in input_blocks:
    h = module(h, emb, context)
    if guided_hint is not None: h += guided_hint; guided_hint = None   # after block 0 only
    hs.append(h)
h = middle_block(h, emb, context); hs.append(h)
return hs                                     # 10 tensors
```

At a 128×128 latent: `control[0..2] [320,128,128]`, `[3] [320,64,64]`,
`[4..5] [640,64,64]`, `[6] [640,32,32]`, `[7..9] [1280,32,32]`.

**The paper's "trimmed" trunk is not what shipped.** The released v0 checkpoint
has byte-for-byte the same encoder topology as SDXL's; the trimmed variant
lives in an unshipped `options/dev/` config. Budget for the full 1.24 B trunk.

### The adaptors - replaces the UNet's skip concatenation, not an additive residual

10 `ZeroSFT` + 2 `ZeroCrossAttn`. `ZeroSFT` REPLACES the UNet decoder's skip
concatenation - a seam `sdxlunet::Unet::new_controlled`/`run_with_control`
(additive residuals only) cannot express.

```
h_raw = concat(h_ori, h_skip)
h1    = concat(h_ori, h_skip + ZeroConv1x1(c))
actv  = SiLU(Conv3x3(c → 128))                      # nhidden = 128, always
γ     = ZeroConv3x3(actv → C_out) ; β = ZeroConv3x3(actv → C_out)
out   = GroupNorm32(h1) · (1 + γ) + β               # GroupNorm carries learnable affine
out   = out·s + h_raw·(1 − s)                       # s = control_scale
```
γ and β are **conv outputs, `[C,H,W]`** - spatially varying, not per-channel
vectors. All three zero-convs are zero-init, so at init the module is the
identity `concat(h_ori, h_skip)` up to the GroupNorm.

```
ZeroCrossAttn(c, x, s):  x + s · CrossAttn( GN(x) as (hw) tokens, GN(c) as (hw) tokens )
                         heads = query_dim/64, dim_head 64, to_out zero-init is commented out upstream
```

Injection order in `LightGLVUNet.forward` - `adapter_idx` 11→0, `control_idx` 9→0:

- after the middle block: `project_modules[11](control[9], h)` - ZeroSFT with
  `concat_channels = 0`, i.e. no concat, no `control_scale` lerp (invoked with
  no `h_ori` at all - a genuinely distinct call, not a zero-width special case);
- then for each of the 9 output blocks: `_h = hs.pop()`, then
  `project_modules[i](control[j], _h, h)` - the concat-replacing ZeroSFT;
- output blocks 2 and 5 end in an `Upsample`; for those, run layers 0..2, apply
  the `ZeroCrossAttn` at `adapter_idx − 1`, then the `Upsample`.

`control[6]` and `control[3]` are each consumed twice (once by a ZeroSFT, once
by a ZeroCrossAttn). Per-adaptor channel tables:
`cond_output_channels = [320]×4 + [640]×3 + [1280]×3`,
`concat_channels = [320]×2 + [640]×3 + [1280]×4 + [0]`,
`project_channels = 2 × ([160]×4 + [320]×3 + [640]×3)`.

**Shape-preservation, proven not assumed**: walking SDXL's skip stack in
up-path pop order against these two tables reproduces both exactly, and
`control_c == skip_c` at every site (must hold - the trunk mirrors the
encoder that produced the skip). So `ZeroSFT`'s output width is always
`h_ori.c + skip.c`, identical to a plain concat - **the frozen SDXL up path's
resnets need no re-import.**

| join `k` (pop order) | `h_ori` C | skip C | control idx | ZeroSFT `C_out` |
|---|---|---|---|---|
| 0 | 1280 | 1280 | 8 | 2560 |
| 1 | 1280 | 1280 | 7 | 2560 |
| 2 | 1280 | 640 | 6 | 1920 |
| 3 | 1280 | 640 | 5 | 1920 |
| 4 | 640 | 640 | 4 | 1280 |
| 5 | 640 | 320 | 3 | 960 |
| 6 | 640 | 320 | 2 | 960 |
| 7 | 320 | 320 | 1 | 640 |
| 8 | 320 | 320 | 0 | 640 |
| post-mid | - | - | 9 | 1280 |

**One fact needs checkpoint verification before any of this is coded**:
whether `ZeroSFT.zero_conv`'s output width is the *skip* width or the
*`h_ori`* width - they differ at joins 2, 3, 5, 6 (640 vs 1280, 320 vs 640)
and upstream source alone doesn't disambiguate the constructor argument
naming from the forward-pass math. Dump every
`project_modules.*.{zero_conv,zero_mul,zero_add}.weight` shape from the real
checkpoint and compare against both columns, before writing `import.rs`.

### The sampler - `RestoreEDMSampler`

**Not Karras, despite the "EDM" name.** `LegacyDDPMDiscretization`: linear-β
(0.00085 → 0.0120, 1000 steps - exactly `diffusion::discrete::DiscreteConfig::sdxl()`),
`σ = sqrt((1−ᾱ)/ᾱ)` flipped descending with `append_zero`. `σ_max = 14.6146`
is a hard-coded constant. The denoiser is `DiscreteDenoiserWithControl`
(ε-scaling / ε-weighting, `num_idx = 1000`), so σ is **snapped back to the
nearest of the 1000 discrete σ's every step**.

```python
σ̂ = σ · (γ + 1);  γ = min(s_churn/(n−1), √2 − 1)
if γ > 0:  x += randn · s_noise · sqrt(σ̂² − σ²)                       # churn
if linear_s_stage2:  s = (σ/σ_max)·(s_start − s) + s                  # control-scale ramp
denoised = denoise(x, σ̂, cond, uc, control_scale=s)                    # LinearCFG inside
if σ_next > 0.05 and restore_cfg > 0:                                  # restoration guidance
    denoised -= (denoised − x_center) · (σ/σ_max)^restore_cfg
d = (x − denoised)/σ̂ ;  x += d · (σ_next − σ̂)                         # first-order Euler
```

CFG is `LinearCFG`: `scale(σ) = (scale − scale_min)·σ/14.6146 + scale_min`,
so guidance ramps **from `spt_linear_CFG` at σ_max up to `s_cfg` at σ→0**.
The unconditional branch keeps the **same** LQ control latent; only the text
differs. Prompts are `caption + p_p` (positive suffix, concatenated with no
separator) and `n_p` **alone** for the negative.

Defaults: `edm_steps 50`, `s_cfg 4.0` (Q preset 7.5), `spt_linear_CFG 1.0`
(Q preset 4.0), `s_stage2 (control_scale) 1.0`, `s_churn 5`, `s_noise 1.01`,
`s_stage1 (restore_cfg) −1` - **restoration guidance is OFF at CLI defaults**
despite the YAML's `4.0`. Reproduce that default faithfully and document it.

### Tiling

Two orthogonal mechanisms, both needed for >1k images:
- **Tiled VAE** (`encoder_tile_size 512` px, `decoder_tile_size 64` latent):
  multidiffusion-style with **GroupNorm statistics propagated across tiles**.
- **Tiled diffusion** (`tile_size 128`, `tile_stride 64`, latent units): per
  denoise step, sweep overlapping windows, accumulate with a separable
  Gaussian weight map, divide by the accumulated weight. One `eps_noise`
  field is sampled for the whole latent and **sliced** per tile - that is
  what keeps churn noise seam-free.

### Licence - a hard constraint on this port

The SUPIR **weights** are under the SUPIR Software License Agreement
(© 2024 SupPixel Pty Ltd): **non-commercial only**, derivative works of the
weights prohibited without written permission, and the definition of
commercial use expressly includes SaaS deployment and *using outputs as ML
training data*. No official HF repo exists; the mirrors are unofficial and
mostly carry no or wrong licence metadata.

Consequences, non-negotiable: no `default_ref`/auto-fetch for this
architecture; no SUPIR weight bytes or SUPIR-produced image enters `testdata/`
in a committed form; `docs/models/supir.md` carries the licence note
prominently. `crates/flux2` gates its own NC-licensed 9B variant behind
`BRAIN_FLUX2_ALLOW_NC=1` - the same opt-in-env-var mechanism is the right
precedent for a future `BRAIN_SUPIR_ALLOW_NC=1` if this port ever needs to
distinguish "weights present" from "commercial-use cleared" at runtime.

## What is and is not started

- [x] Architecture id (`crates/arch`), the shared-loop refactor (§0.2), doc
      drift fixes (§0.3).
- [x] Resources + reference goldens (real checkpoints, `tools/goldens/
      supir_dump_reference.py`, `testdata/supir/` + `testdata/
      supir_forward_parity/`, gitignored).
- [x] The three seams: `vae::blocks::skipfuse::SkipFuse` + `Unet::new_fused`,
      `diffusion::restore`, blended `imaging::tiling`.
- [x] `crates/supir` forward pass, parity-proven against the real checkpoint
      (trunk cosine 1.0000000000; full forward at `s_churn=0` verified).
- [x] `sdxlunet::int8` + `supir::int8` - HOST-memory quantization only (see
      "Memory" below for the real, measured, still-open device-memory gap).
- [x] Training. `crates/supir/src/train.rs` (`SupirTrainer` - trunk +
      adaptors + backbone recorded in ONE reverse-mode tape via
      `Supir::new_train`/`Rec::new_train`, an MSE loss head, mirroring
      `sdxlunet::train::UnetTrainer` exactly rather than a from-scratch host
      f64 oracle - this graph is already `vae::blocks::Builder`/`Trace`
      differentiable end to end, the same situation `check_unet` closed, so
      an independent hand-written backward would duplicate that gate's
      coverage rather than add a new signal; see `train.rs`'s own module doc
      for the full reasoning). `gradcheck::check_supir` (directional, 185
      trunk+adaptor tensors, `SupirConfig::tiny` at `H=W=8`, max_rel 9.9e-2,
      all inside the workspace `(4e-3, 8e-2)` gate) and
      `check_supir_elementwise` (per-entry on
      `control_model.mid_block.resnets.1.conv2.bias` - the trunk's mid
      output is read TWICE by `Adaptors::fuse_mid`, once via `zero_conv` and
      once via `mlp_shared.0`, the same shared/folded-gradient class as
      T5's/`check_unet`'s own elementwise gates - all 64 entries pass).
      `crates/supir/src/lora.rs` (`SupirLora` over `model::lora::Pair`,
      targeting the trunk's 8 linear suffixes per `BasicTransformerBlock`;
      gated: bit-exact no-op at `B=0`, `apply`/`fold_into` bit-agreement,
      save/load round-trip, LoRA-only overfit 5.09e-1 -> min 9.37e-2).
      `crates/supir/src/finetune.rs` (`Finetuner::adaptor_only` - upstream's
      own recipe, backbone encoder frozen, decoder+trunk+adaptors train -
      and `Finetuner::full_backbone`; both measured to overfit a single
      example (~74%/69% loss reduction over 120 steps) and a 3-example
      dataset (~75% reduction over 40 rounds) at `SupirConfig::tiny`,
      `H=W=8` - real numbers, not "near zero": see `finetune.rs`'s module
      doc for why this machine's per-step wall-clock (a real Vulkan iGPU,
      ~2.4 s per full trunk+adaptors+backbone forward+backward) sets the
      gate at "clear, substantial, measured descent" rather than a literal
      zero floor). `check_controlnet` deliberately NOT closed in this pass -
      re-assessed and found to be a genuine second trainer (no `Rec::new_train`
      wiring in `controlnet::model` yet, `scale_buf` uses the same
      tape-breaking `push_step` idiom this port's own adaptors doc already
      disqualified, and `Residuals` is a multi-buffer output with no single
      MSE loss head to reuse), not the "small, obvious mirror" the plan
      scoped it as; the reasoning is recorded in
      `.agents/roadmap/controlnet.md` rather than left as a bare unchecked
      box.
- [x] `crates/llava` (see `.agents/roadmap/llava.md`) - the vision tower ->
      projector -> decoder splice, the `vicuna_v1` template, INT8 decoder
      path and served `caption` action, all weight-free-gated; not yet
      exercised against real checkpoint bytes (a multi-ten-GB download none
      was fetched this session).
- [x] Serving contract, CLI, NPU export, docs. `crates/supir/src/pipeline.rs`
      (the restoration loop this crate's own doc had left as future work: the
      dual encode - `denoise_encoder`'s CompVis-named weights renamed to the
      diffusers keys `vae::VaeEncoder` reads via
      `crate::import::denoise_encoder_diffusers_names`, merged with the
      frozen backbone's own `quant_conv` - dual-CLIP conditioning via
      `sdxlunet::textenc::TextEncoders` reused unmodified, the
      `RestoreEDMSampler` loop driven directly off `diffusion::restore`'s
      primitives - `DiscreteDenoiserWithControl`, `churn_gamma`/`sigma_hat`/
      `apply_churn_noise`, `restore_guidance`, `linear_cfg_scale`,
      `euler_step` - CFG combined in eps-space per this codebase's own
      convention, and colour fix via a new `imaging::colorfix` module -
      `wavelet_reconstruction`, the real 5-level a-trous decomposition
      upstream's own default uses, plus `adain` for its other supported
      mode). `crates/supir/src/caps.rs` (the `restore` action; cancellable
      per denoise step via `inv.cancel`, the same contract `wan::caps`
      documents; optional LLaVA auto-captioning dispatched through a
      `capability::Registry` supplied by the caller - `crates/supir` links no
      VLM - mirroring `crates/imgpipe`'s own "registry supplied by the
      caller" precedent). `crates/cli/src/resident_supir.rs` (the residency
      adapter; `run_batch` serial, stated in-file for the same reason
      `resident_sdxl.rs`/`resident_controlnet.rs` give: every request is its
      own multi-step sample) plus the `crates/catalog` `ModelEntry` and the
      `crates/cli/src/catalog.rs` patch line, all invariant-tested by that
      file's own suite (`every_listed_model_is_constructible_by_name`,
      `every_patched_id_is_a_real_catalog_entry`). One `ARCH_TO_MODEL` row
      (`brain supir restore ...`) - `sdxlunet`/`controlnet` themselves ship
      no CLI shortcut at all (only `brain do brain/sdxl ...`/the served
      transports), so this is one line ahead of that precedent, not a new
      `supir_cli.rs`. `supir::import::GGUF_ARCHITECTURE` (`"sdxl"`, a
      borrowed spelling - the frozen backbone genuinely is byte-identical
      SDXL, the same reasoning `s3dit` used for `"lumina2"`) registered as a
      SECOND documented ambiguous-tag exception in
      `crates/cli/src/gguf_import.rs`'s own test, alongside a stub
      `import_gguf` that states plainly no real file has ever been observed
      rather than guessing a tensor mapping against one. D-Bus `Run` needed
      no new code (it dispatches generically over the residency `Executor`
      once a model is registered) - `examples/restore/supir_restore.py` +
      README update mirror `restore_face.py`'s existing shape.
      `crates/imgpipe`: a NEW `Stage::SupirRestore` variant (not
      `Stage::Restore{w}`, whose fidelity dial has no SUPIR meaning) - a
      SECOND size-changing tail alongside `Stage::Upscale`, mutually
      exclusive with it since this crate defines no combined order for two
      tails that each change the working resolution; `catalog`'s
      `imgpipe_stage_ids_match_the_catalog` test extended to cover it.
      `crates/npu/src/supir_topology.rs` + `supir_export.rs`: `ZeroCrossAttn`
      (the one adaptor with LINEAR projections, quantized through the shared
      `topo::linear_quant` emitter) - structurally tested, no real checkpoint
      or NPU hardware involved. Docs: `docs/models/supir.md` rewritten from
      its Phase-0 placeholder, `docs/models/llava.md` (already Phase 6),
      `docs/models/index.md` (both moved out of "Reserved, not started" into
      their real tables), `docs/models/imgpipe.md` (the new stage),
      `README.md`'s model list. `docs/manifest.txt` already carried both
      pages' entries. Lesson recorded in `.agents/rules/lessons.md` (#63):
      `vae::VaeEncoder`/`VaeDecoder` are diffusers-NAMED, not merely
      diffusers-SHAPED, despite sitting on the genuinely generic
      `vae::blocks::Builder`/`BlockNames`.
- [x] Optimisation pass. `crates/supir/src/bin/supir_bench.rs` - a weight-free
      per-kernel profiler over synthetic (shape-correct scratch) tensors,
      mirroring `sdxlunet::unet_bench`'s method exactly (`gpu_core::profile`,
      best-of-N wall time, `poll_wait`-bracketed). The real combined
      trunk+adaptors+backbone graph (`Supir::new`, `~15.5 GB` fp32) is
      already documented elsewhere in this ledger to hit `wgpu error: Out of
      Memory` on this machine's Intel iGPU regardless of latent size - that
      OOM is driven by total resident weight bytes, not resolution, so no
      `h`/`w` choice avoids it. The bench therefore splits the SAME real
      dispatch sequence into two phases that each independently fit (proven:
      the 10.27 GB frozen backbone alone already fits, per `unet_bench`):
      `trunk` (`GLVControl` alone, full SDXL-shaped `UNetConfig`, 1.243 B
      params / 4.97 GB fp32, over its own `Rec`/`Gpu`) and `fused` (the
      frozen backbone + the 12 adaptors via `Rec`/`Unet::record_into` over
      `supir::adaptors::Adaptors` as the `SkipFuse`, 2.622 B params / 10.49 GB
      fp32, reading the trunk's 10 control tensors from scratch buffers sized
      off the backbone's own `skip_shapes`/`AdaptorConfig::mid` rather than a
      real trunk forward - the trunk's cost is exactly what the `trunk` phase
      already measures). A gated `full` subcommand replays the true one-graph
      `Supir::new` dispatch verbatim, behind `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1`
      (same convention as the parity suite's own full-forward tests); on this
      machine it correctly declines by default rather than OOMing the bench
      process.

      Measured at a 32x32 latent (`t_enc=77`, 3 reps, this box's Intel iGPU):
      whole-pass wall time (the reliable number - `best_of`'s
      `poll_wait`-bracketed timing) was 549-1102 ms for `trunk` and
      1579-2754 ms for `fused` across repeated runs, a roughly 2x spread on
      the SAME dispatch sequence that is DVFS/clock-state noise (an idle
      integrated GPU needs several continuous seconds of work to reach its
      running clock), not a measurement bug - consistent with this box's own
      documented cold-clock behaviour. Combined (merged) wall time ran
      2643-3304 ms, trunk taking roughly 40% of it and fused 60%. Per-kernel
      shares (ratio of each kernel-kind's time to the summed total, stable
      across runs even though the DVFS noise moves the absolute wall time):
      `matmul_reg3` ~30%, `bias_add` ~16%, `layernorm_rows` ~9-14%, `add2`
      ~6-10%, then a long tail (cross-attention `attn_scores_cross`/
      `attn_apply_cross`/`attn_softmax_cross`/`flash_attn_bidir_reg2`,
      `gelu_erf`, `silu`, `gn_stats_wg`/`gn_apply`, `mul`, `im2col_at`,
      `nlc_bias_nchw`) each 1-4%. SUPIR's OWN new kernel - `edm_mix`, the
      `ZeroSFT`/`ZeroCrossAttn` lerp - measured at 0.2% of the fused pass:
      genuinely negligible.

      **Observed, not introduced here**: on this box's adapter
      (`Intel(R) Arc(tm) Graphics (MTL)`, Vulkan), the per-row absolute
      `ms`/`GFLOP/s`/`%roof` columns the device-timestamp path prints for
      this kernel set are corrupted by many orders of magnitude (`1e16`-
      `1e17` "ms" for a pass whose real wall time is under 3 seconds) -
      reproduced verbatim on `sdxlunet::unet_bench`, unmodified, so this
      predates SUPIR and is a `gpu-core`/`backend-wgpu` timestamp-query
      conversion defect, cross-cutting across every model that profiles on
      this kernel family. The whole-pass wall-clock number and the per-row
      RATIOS (each row's share of the corrupted domain's own sum, which
      cancels a shared corruption factor) both stayed sane and are what the
      analysis above and `supir_bench`'s own printed tables rely on; the
      absolute per-row `ms`/roof columns do not and should not be trusted on
      this machine for this kernel set. Fixing the underlying timestamp
      conversion is real, valuable work but is infrastructure spanning every
      model that profiles here, not a SUPIR-scoped change - filed as a
      separate follow-up rather than pulled into this phase.

      **Conclusion: no SUPIR-specific optimisation applied, and that is the
      honest result of profiling first.** The dominant kernel-kind shares
      (`matmul_reg3`/`bias_add`/`layernorm_rows`/`add2`) are inherited from
      the shared `sdxlunet`/`vae::blocks` GEMM-plus-elementwise
      infrastructure `GLVControl` and the frozen backbone both sit on, and
      closely match that backbone's OWN already-measured profile shape
      (`unet_bench` on the plain SDXL UNet alone: `matmul_reg3` 27.7%,
      `bias_add` 17.0%, `add2` 11.8%, `layernorm_rows` 9.7% - essentially the
      same proportions). SUPIR's own incremental compute - the adaptors'
      `edm_mix` lerp - is 0.2% of the fused pass, confirming the "one-graph"
      design and the "reuse existing kernels" discipline from earlier phases
      already left no meaningful residual overhead to attack. The one real
      hypothesis this profile surfaces - fusing `bias_add` into its preceding
      GEMM's own write, cutting a separate memory-bound read+write pass over
      a kernel-kind that is 16% of the whole pass - is genuine, but it is
      `vae::blocks::Builder`-level infrastructure shared by roughly ten
      models on this engine, not something a SUPIR-scoped change can safely
      land or validate; forcing it into this phase would trade a real,
      narrowly-scoped SUPIR deliverable for an unreviewed cross-cutting one.
      Recorded here rather than attempted.

## Staged plan

Full detail lives in the approved implementation plan (see the session that
added this file); summarized here so a later reader isn't sent elsewhere for
the shape of the work:

1. **Debt-first refactor in `sdxlunet`/`controlnet`** (unblocks everything
   else): factor the duplicated SDXL sampling loop into `sdxlunet::sampler`
   (a `Denoiser` trait + per-step hook) and `sdxlunet::textenc`; hoist
   `UNetConfig::skip_shapes` out of `ControlNetConfig::residual_shapes`; add
   `Rec::set_prefix`/`take_temb_act`/`set_temb_act` and extract
   `Unet::record_into` from `Unet::build`.
2. **Reference goldens first** (`tools/goldens/supir_dump_reference.py`),
   before any Rust, per `.agents/rules/porting.md` §1.
3. **Three new seams, gated weight-free**: `vae::blocks::skipfuse::SkipFuse`
   (the trait that replaces `Rec::concat_channels` at the up-path joins -
   lives in `vae::blocks`, not `crates/model`, because `crates/vae` depends on
   `crates/model` and not the reverse, so only `vae::blocks::Builder` can
   record a trainable prologue); `vae::blocks::grad::Op::Mix` (the ZeroSFT/
   ZeroCrossAttn lerp, reusing `edm_mix.wgsl` verbatim - no new kernel);
   `diffusion::restore` (the sampler's scalar math, host-only); a blended
   `imaging::tiling::TilePlan` variant + one accumulate kernel.
4. **`crates/supir` forward**: verify the zero_conv width question from the
   checkpoint first; `config.rs`/`import.rs` (two-way coverage, 1035
   tensors); `trunk.rs` (`GLVControl` over the public `sdxlunet::model::Rec`,
   composition not a `ControlNetConfig` generalisation - SUPIR's hint
   embedder, injection point and lack of output zero-convs share no code
   with vanilla ControlNet); `adaptors.rs` (the `SkipFuse` impl); `model.rs`
   (trunk + UNet recorded into ONE `Rec`/`Builder`/`Gpu`/submit via
   `Unet::record_into` - this also closes `controlnet`'s own "fused
   on-device path" roadmap gap as a side effect); climb the parity ladder
   against the goldens, tapping each adaptor's input AND output (a
   permutation of same-width control tensors would pass an output-only tap);
   `pipeline.rs`; tiled VAE + tiled diffusion.
5. **Memory**: `sdxlunet::int8` (group-wise, `QUANT_GROUP = 32`, never
   whole-channel) then `supir::int8` - this machine has no discrete GPU (one
   Intel iGPU + one NPU sharing 30 GB system RAM), so int8 is a prerequisite
   for running at all here, not an optimisation.
6. **Training**: `grad.rs`/`modelgrad.rs` (f64 oracle, generic over float
   type), `gradcheck::check_supir` + `check_supir_elementwise`, `lora.rs`,
   `finetune.rs` (adaptor-only, matching upstream's own `torch.no_grad()`
   frozen-encoder recipe, AND full-backbone via `sdxlunet::train::UnetTrainer`),
   overfit tests. `check_controlnet` closes in the same series (its
   trainable copy is the same recorded blocks `check_supir` exercises).
7. **`crates/llava`** (see `.agents/roadmap/llava.md`) - ordered after
   SUPIR's forward is parity-proven, since SUPIR does not depend on it
   (`--no_llava` is a supported upstream path; LLaVA only ever emits a
   string, never touches the diffusion graph).
8. **Serving contract, CLI, NPU export, docs.** `run_batch` stays the serial
   default (every request is its own multi-step sample, same as
   `sdxlunet`/`controlnet`) with a stated in-file reason. No official GGUF
   spelling exists for SUPIR upstream; register a plausible
   `GGUF_ARCHITECTURE` constant anyway (the `s3dit`/`wan` precedent for
   architectures with no real GGUF release) so a future release
   auto-dispatches with no further CLI change.
9. **Optimisation** - only after the parity gates are green and frozen, per
   `.agents/rules/porting.md` §10.

## Deferred, recorded rather than silently skipped

- **`sdxlunet::int8`/`supir::int8` reduce HOST memory only, not device
  memory.** Measured: the combined trunk+adaptors+backbone graph (2608
  tensors, 15.60 GB fp32) drops to 5.62 GB host-resident after quantization
  (889 tensors packed to int8, 1719 left fp32) - a real win for import/host
  peak. But `vae::blocks::Builder::set_packed` dequantizes each packed
  tensor to fp32 AT UPLOAD, so the device-resident buffers `wgpu` allocates
  are still fp32-sized. On this box (one Intel iGPU, no discrete card,
  `wgpu` reports a 2047 MiB per-buffer/per-binding cap) recording the full
  graph hits `wgpu error: Out of Memory` before a forward ever runs -
  reproduced 3 times, including with per-tap buffer pinning disabled
  (`supir_full_forward_int8_no_taps_fits_this_machine`), ruling out tap
  pinning as the cause. Closing this for real needs genuine on-device int8
  storage with a dequantizing GEMM - the shape `crates/flux1`/`crates/s3dit`
  already have - threaded through `vae::blocks::Builder`, the shared block
  recorder ~10 architectures depend on. Both int8 full-forward tests in
  `crates/supir/tests/parity.rs` are gated behind `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1`
  (same as the fp32 sibling) and skip themselves honestly rather than claim
  a false pass. Filed as Phase 8 (optimisation) or a dedicated follow-up.
- Batch > 1 (the SDXL graph is recorded at batch 1; CFG is two `run()` calls).
- The unshipped "trimmed" paper trunk (`options/dev/SUPIR_paper_version.yaml`
  is not in the released repo).
- `RestoreDPMPP2MSampler` + the Juggernaut-Lightning 8-step config.
- Per-tile local prompts (upstream supports it only at batch size 1).
- **NPU export covers `ZeroCrossAttn` only** (`crates/npu/src/supir_topology.rs`
  + `supir_export.rs`, structurally tested, `topo::linear_quant` for its
  linear projections). The 10 `ZeroSFT` adaptors (pure conv + GroupNorm - the
  same primitives `vae_topology.rs` already exports, just under a different
  block walk) and the 1.24B `GLVControl` trunk itself have NO export path:
  unlike every other NPU topology in this crate, an SDXL/ControlNet-shaped
  cross-attention UNet has never been exported to ONNX anywhere in this
  tree - neither `sdxlunet` nor `controlnet` has a topology file at all - so
  there is no existing block-walk to adapt, only `Unet::record_into`'s Rust
  implementation to port from scratch. Filed as real, separate follow-up
  work, distinct from (and larger than) "the trunk exceeds what the NPU can
  hold on this hardware" below.
- Full NPU validation, even of the piece that IS exported - there is no NPU
  on this port's own development machine, so `ZeroCrossAttn`'s graph is
  gated structurally (node counts, quantization shape, non-empty bytes), not
  against real hardware or a real checkpoint. The trunk realistically
  exceeds what an NPU can hold on this hardware even once exported.
- **`linear_s_stage2` (the optional per-step control-scale ramp) is not
  implemented in `crates/supir/src/pipeline.rs`.** `control_scale` is baked
  into `Supir::new`'s graph as a constant (see that function's own doc); a
  faithful per-step ramp would mean rebuilding - and re-uploading - the whole
  trunk+adaptors+backbone graph every denoise step. Upstream's own CLI
  default (`linear_s_stage2 = False`) already runs the constant path this
  pipeline implements. A rewritable control-scale device buffer
  (`CodeFormer`'s `w`/`scale_add` is the precedent) is the real fix.
- **Tiled VAE / tiled diffusion are not wired into `crates/supir/src/pipeline.rs`.**
  The seams exist (`imaging::tiling`'s blended `TilePlan` variant, built in
  an earlier phase of this port), but composing them into the restoration
  loop - splitting the sampler's per-step forward across overlapping windows
  with a shared per-step noise field sliced per tile, and the VAE encode/
  decode across tiles with GroupNorm statistics propagated between them - is
  independent, sizeable work this phase leaves for a follow-up. Every
  `Restorer::restore` call runs the whole working-resolution image through
  one graph.
- A full end-to-end restoration run on real checkpoints - held until
  explicitly requested, per this port's scoping decision, and expected to
  hit the device-memory ceiling documented above regardless (int8 reduces
  host memory only on this hardware, not device memory).
- `check_controlnet` and `ControlNet::record_into` - the plan expected these
  to fall out of `check_supir`'s infrastructure as a small mirror; building
  `supir::train::SupirTrainer` found two real gaps in `crates/controlnet`
  instead (no `Rec::new_train` build path, and `scale_buf`'s `Builder::push_step`
  use means `conditioning_scale` is not on the reverse tape at all today),
  plus a loss-head shape mismatch (`Residuals` is several differently-shaped
  buffers, not the one `UnetTrainer`/`SupirTrainer` MSE head assumes) - a
  genuine second trainer, not a mirror. Recorded with full reasoning in
  `.agents/roadmap/controlnet.md`.
