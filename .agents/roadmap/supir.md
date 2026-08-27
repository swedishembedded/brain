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

- [ ] Everything. The architecture id (`crates/arch`) and this placeholder
      crate are the only things that exist.

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

- Batch > 1 (the SDXL graph is recorded at batch 1; CFG is two `run()` calls).
- The unshipped "trimmed" paper trunk (`options/dev/SUPIR_paper_version.yaml`
  is not in the released repo).
- `RestoreDPMPP2MSampler` + the Juggernaut-Lightning 8-step config.
- Per-tile local prompts (upstream supports it only at batch size 1).
- Full NPU validation of the trunk - the export path should exist, but the
  trunk realistically exceeds what the NPU can hold on this hardware.
- A full end-to-end restoration run on real checkpoints - held until
  explicitly requested, per this port's scoping decision.
