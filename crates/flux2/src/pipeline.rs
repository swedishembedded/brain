// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end FLUX.2 Klein: prompt → text conditioning → 4-step (klein) or
//! CFG (base) rectified-flow Euler → FLUX.2 VAE decode; image editing via
//! reference-image token concatenation (RoPE t-offsets 10·(i+1)).
//!
//! Weight locations come from env only (`Paths::from_env`) — never baked-in
//! paths. The text encoder runs the parity-proven masked-pad path
//! (`Qwen::encode_hiddens_padded`, layers 9/18/27 concatenated).

use crate::config::Flux2Config;
use crate::model::{position_ids, Flux2Model};
use data::Tokenizer;

/// Qwen3 hidden-state taps concatenated per token (also used by
/// [`crate::finetune`]'s standalone caption encoder).
pub const TAP_LAYERS: [usize; 3] = [9, 18, 27];
/// Right-pad token for the masked-pad text-encoder path.
pub const PAD_TOKEN: u32 = 151643;

/// Weight locations, env-only:
/// `BRAIN_FLUX2_DIT` (diffusers `transformer/` dir, BFL single-file
/// safetensors, or BF16 GGUF), `BRAIN_FLUX2_VAE` (diffusers `vae/` dir or
/// file), `BRAIN_FLUX2_TE` (HF text-encoder dir), `BRAIN_FLUX2_TOKENIZER`
/// (`tokenizer.json`).
#[derive(Debug)]
pub struct Paths {
    pub dit: String,
    pub vae: String,
    pub te: String,
    pub tokenizer: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        Self::from_env_with_dit(None)
    }

    /// [`Self::from_env`], but the DiT may come from the CLI's `--model`
    /// flag instead of `BRAIN_FLUX2_DIT`: the flag names the DiT weights
    /// outright, so the variable is then not required. The other three
    /// components have no flag and stay required either way.
    pub fn from_env_with_dit(dit: Option<String>) -> Result<Paths, String> {
        let get = |k: &str| std::env::var(k).map_err(|_| format!("{k} not set"));
        Ok(Paths {
            dit: dit
                .or_else(|| std::env::var("BRAIN_FLUX2_DIT").ok())
                .ok_or_else(|| "BRAIN_FLUX2_DIT not set (pass --model to name the DiT weights)".to_string())?,
            vae: get("BRAIN_FLUX2_VAE")?,
            te: get("BRAIN_FLUX2_TE")?,
            tokenizer: get("BRAIN_FLUX2_TOKENIZER")?,
        })
    }
}

/// The pixel size at which each reference is encoded as **conditioning**, in
/// order; `None` for a reference that contributes nothing.
///
/// A supplied reference always conditions the model. `strength` decides how
/// much of the denoise starts from the init latent, not whether the DiT can
/// see the photograph - so under `strength < 1` the first reference does
/// double duty: it is the init latent *and* it is attended to. Because the
/// init role pins it to the output size, its conditioning copy is downscaled
/// by [`GenOpts::ref_cond_scale`]; reference tokens cost attention
/// quadratically, and a full-size copy of a same-size reference doubles the
/// image half of the joint sequence.
///
/// This is the ONE place the rule is written. The sizing entry point
/// ([`ref_tokens`]), the position-id builder and the denoise loop all read it,
/// so a pipeline cannot be sized for a sequence different from the one it is
/// handed.
pub fn cond_sizes(refs: &[(Vec<f32>, u32, u32)], opts: &GenOpts) -> Vec<Option<(u32, u32)>> {
    let init = opts.strength.is_some_and(|s| s < 1.0);
    refs.iter()
        .enumerate()
        .map(|(i, &(_, h, w))| {
            if i == 0 && init {
                init_cond_size(opts.ref_cond_scale, h, w)
            } else {
                Some((h, w))
            }
        })
        .collect()
}

/// Conditioning tokens `refs` actually contribute under `opts` -- what a
/// pipeline must be sized for, in latent tokens.
pub fn ref_tokens(refs: &[(Vec<f32>, u32, u32)], opts: &GenOpts) -> u32 {
    cond_sizes(refs, opts).into_iter().flatten().map(|(h, w)| (h / 16) * (w / 16)).sum()
}

/// A LoRA adapter to fold in before the model is built.
///
/// `path` selects the family by extension: a `.safetensors` is a third-party
/// (ai-toolkit / ComfyUI / diffusers) adapter over the fused matrices,
/// anything else is brain's own trained checkpoint container.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterSpec {
    pub path: String,
    /// ComfyUI's `strength_model`: multiplies the whole delta. 1.0 is the
    /// reference default. Meaningful for third-party adapters, whose files
    /// carry no alpha; brain's own adapters bake their scale into the
    /// checkpoint header and ignore this.
    pub scale: f32,
}

impl AdapterSpec {
    /// An adapter at the reference default strength.
    pub fn new(path: impl Into<String>) -> AdapterSpec {
        AdapterSpec { path: path.into(), scale: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct GenOpts {
    pub width: u32,
    pub height: u32,
    /// Image-to-image anchoring in `[0, 1]`. `None` (or 1.0) starts the
    /// denoise from pure noise, so the result keeps the composition but is a
    /// fresh generation (this is why a reference-only "colorize" reinterprets
    /// the scene). With `Some(s)` the first reference is VAE-encoded and the
    /// trajectory starts at noise level `s` from
    /// `x_σ = (1−σ)·x₀ + σ·ε` - the rectified-flow forward process - so
    /// structure is anchored to the source. Small `s` = faithful, `0` returns
    /// the source through the codec.
    ///
    /// A **smooth dial, not a mode switch**: the descent from `s` is the
    /// free-generation schedule scaled into `[0, s]`
    /// ([`img2img_sigmas`]), so the same sampler runs at every value and
    /// `0.99` renders a hair from what `1.0` renders instead of doing a
    /// different job. Lowering `s` lowers every sigma and raises the source's
    /// weight in the init latent, so preservation only ever increases.
    ///
    /// This is how much of the denoise starts from the init latent, NOT
    /// whether the model can see the reference: the reference images
    /// condition the DiT through their tokens at **every** value, including
    /// under `strength`, where the first one is both the init latent and a
    /// conditioning input ([`GenOpts::ref_cond_scale`], [`cond_sizes`]).
    pub strength: Option<f32>,
    /// None → the variant default (4 distilled / 50 base).
    pub steps: Option<u32>,
    /// CFG scale — only meaningful for the undistilled base variants.
    pub guidance: f32,
    pub seed: u64,
    /// Spatial preservation mask over the output canvas: **white regenerates,
    /// black preserves** the first reference image, which must then be at the
    /// output size. Where `strength` decides how much of the source survives
    /// *everywhere*, this decides *where* it survives - after every Euler step
    /// the masked-out region is replaced by the source latent renoised to that
    /// step's sigma ([`crate::mask::blend`]), so it tracks the source exactly
    /// instead of being softly guided toward it.
    ///
    /// `None` - and an all-white mask - are bit-for-bit the unmasked
    /// behaviour.
    pub mask: Option<crate::mask::Mask>,
    /// Linear scale of the **conditioning copy** of the init reference, in
    /// `[0, 1]`. Only `refs[0]` under `strength < 1` is affected: that is the
    /// one reference whose resolution the caller cannot choose, because the
    /// init-latent role pins it to the output size. Every other reference
    /// conditions at whatever size it was supplied at, and `strength >= 1`
    /// (or `None`) consumes no init latent at all, so this dial does not
    /// apply there.
    ///
    /// `1.0` conditions at the full output size - the largest, most faithful
    /// and most expensive setting, and the one that makes `strength 0.999`
    /// cost exactly what `strength 1.0` costs. `0.0` switches the
    /// conditioning copy off, which is the explicit opt-in to the cheap
    /// behaviour where the reference reaches the model only through the
    /// init latent. The default is a downscale, because reference tokens are
    /// quadratic in the attention and a full-size copy of a same-size
    /// reference doubles the image half of the joint sequence.
    pub ref_cond_scale: f32,
}

/// Default [`GenOpts::ref_cond_scale`]: the conditioning copy of the init
/// reference is three quarters of its linear size, i.e. a bit over half its
/// tokens. Reference *resolution* is the architecture-preservation dial, so
/// this is a fidelity/cost trade and not an implementation detail; the value
/// is the one that produced the staging results this behaviour was built
/// for. Raise it with `--ref-cond-scale` when the card has room.
pub const DEFAULT_REF_COND_SCALE: f32 = 0.75;

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts {
            width: 1024,
            height: 1024,
            strength: None,
            steps: None,
            guidance: 4.0,
            seed: 0,
            mask: None,
            ref_cond_scale: DEFAULT_REF_COND_SCALE,
        }
    }
}

/// Pixel size of the conditioning copy of an init reference that is `h x w`,
/// or `None` when `scale` switches conditioning off.
///
/// The result is floored to a multiple of 16 (one latent token) on each axis
/// independently, so the aspect ratio is preserved up to one token and a
/// non-square canvas is not a special case. `scale` is clamped to `[0, 1]`:
/// upscaling a reference past the size it was encoded at buys nothing the VAE
/// did not already throw away, and costs tokens quadratically.
pub fn init_cond_size(scale: f32, h: u32, w: u32) -> Option<(u32, u32)> {
    let s = scale.clamp(0.0, 1.0);
    let q = |d: u32| (((d as f32 * s) as u32) / 16 * 16).max(16);
    let (ch, cw) = (q(h), q(w));
    // Below one latent token in either axis there is nothing to condition on;
    // `q` floors at 16, so the off switch is the scale itself.
    (s > 0.0).then_some((ch, cw))
}

/// The noise schedule an img2img run integrates: the free-generation schedule
/// ([`diffusion::scheduler::klein_sigmas`]) **scaled** to `[0, strength]`,
/// `steps + 1` entries.
///
/// `strength` is the noise level the init latent `x_σ = (1−σ)·x₀ + σ·ε` is
/// mixed at, so the trajectory has to start at `σ₀ = strength` - the model
/// must be asked to denoise from the distribution it was actually handed. What
/// is free is the *shape* of the descent from there, and the shape has to be
/// klein's own: it is a distilled few-step sampler, its schedule is shifted by
/// a token-count- and step-count-dependent `mu`, and it spends almost all of
/// its steps at high sigma before one long final leap. A uniform ramp over
/// `[strength, 0]` is a different sampler, not a lower-noise version of the
/// same one, and switching between them at the top of the dial is what made
/// `--strength 0.99` do a different job from `--strength 1.0` rather than
/// almost the same one.
///
/// Scaling, not slicing. Slicing the distilled schedule does not work and is a
/// standing temptation: `klein_sigmas` is shifted so hard for few-step
/// sampling that its lowest non-zero entry is 0.56 at 8 steps (0.75 at 4), so
/// there is no low-noise entry point to start an img2img from and the caller's
/// step budget would silently collapse. The velocity field is defined at every
/// sigma, so the whole shape is compressed into `[0, strength]` and the caller
/// gets every step they asked for.
///
/// Two properties the dial is built on, both gated in this module:
/// * `strength = 1` reproduces `klein_sigmas` **bit for bit** (`1.0 · x` is
///   exact in IEEE), so the dial reaches free generation rather than
///   approaching it;
/// * every entry is linear in `strength` and every entry lies in `[0, 1]`, so
///   lowering the dial lowers every sigma, and lowering it by `δ` moves no
///   sigma by more than `δ`.
pub fn img2img_sigmas(strength: f32, steps: usize, n_gen: usize) -> Vec<f32> {
    diffusion::scheduler::klein_sigmas(steps, n_gen).into_iter().map(|s| strength * s).collect()
}

/// Bilinear resize of a reference image (`[-1,1]` CHW, the layout
/// [`Pipeline::generate`] takes) to `th x tw`.
///
/// Each channel plane is contiguous `[h, w]`, which is exactly a 1-channel
/// interleaved image, so this is the shared host resize applied three times
/// rather than a fourth resampler in the workspace.
fn resize_ref(chw: &[f32], h: u32, w: u32, th: u32, tw: u32) -> Vec<f32> {
    if (th, tw) == (h, w) {
        return chw.to_vec();
    }
    let plane = (h * w) as usize;
    let mut out = Vec::with_capacity(3 * (th * tw) as usize);
    for c in 0..3usize {
        out.extend(imaging::resize_bilinear_hwc(&chw[c * plane..(c + 1) * plane], 1, w, h, tw, th));
    }
    out
}

/// Read DiT weights from a diffusers `transformer/` dir, a BFL single-file
/// safetensors, or a BF16 GGUF, onto the canonical BFL names. Public so
/// [`crate::finetune`] loads the frozen base through the same importer.
pub fn read_dit_tensors(path: &str, cfg: &Flux2Config) -> Result<crate::Tensors, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("{path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors under {path}"));
        }
        let mut ts = Vec::new();
        for f in files {
            ts.extend(checkpoint::safetensors::read(f.to_str().unwrap())?);
        }
        crate::import_diffusers(ts, cfg)
    } else if p.extension().is_some_and(|x| x == "gguf") {
        crate::import_bfl(checkpoint::gguf::read(path)?, cfg)
    } else {
        let ts = checkpoint::safetensors::read(path)?;
        // single-file releases use BFL names; a consolidated diffusers dump
        // would carry `transformer_blocks.` — probe and route
        if ts.iter().any(|t| t.name.starts_with("transformer_blocks.")) {
            crate::import_diffusers(ts, cfg)
        } else {
            crate::import_bfl(ts, cfg)
        }
    }
}

// ---- automatic placement ---------------------------------------------------
//
// A FLUX.2 run is three models that must live somewhere: the DiT, the Qwen3
// text encoder, and the VAE. Which card each goes on is a CAPACITY question,
// and brain answers capacity questions in exactly one place -
// `residency::budget`/`place`/`plan`, reached through the
// `gpu_core::devices::place` seam. This module therefore DECLARES what the
// parts cost and lets the engine place them; it never names a card.
//
// What it used to do instead - and what `BRAIN_FLUX2_TE_DEVICE` still
// overrides - was require the operator to know that an int8 9B DiT and its
// text encoder do not co-reside on one 24 GiB P40, and to say so by hand on
// every run.

/// Bytes one weight of `precision` occupies on the device. Only 2-D tensors
/// (the linears) are quantised; norm scales and biases stay f32 in both
/// tiers, which is what the int8 shard builders actually do.
fn weight_bytes(shape: &[usize], precision: crate::Precision) -> u64 {
    let n: usize = shape.iter().product();
    let w = if precision == crate::Precision::Int8 && shape.len() == 2 { 1 } else { 4 };
    (n * w) as u64
}

/// The device scratch that `Flux2Model::new_from` allocates for the joint
/// sequence. This is shared by the architecture-only and GGUF-header cost
/// routes so neither can price weights accurately but omit the buffers that
/// coexist with them on the card.
fn dit_scratch_bytes(cfg: &Flux2Config, precision: crate::Precision, n_joint: u64, max_batch: u64) -> u64 {
    let batch = max_batch.max(1);
    let n = n_joint * batch;
    let d = cfg.hidden as u64;
    let mlp = cfg.mlp_hidden() as u64;
    let hd = cfg.head_dim() as u64;
    // `Scratch`: 16 D-wide words (qkv counts as three), three MLP-wide words,
    // two latent IO buffers, cos+sin totalling one head-width, context input,
    // and score/prob buffers (one word each under the fast GPU path; otherwise
    // the materialized [B,H,T,T] buffers).
    let attn_words = if precision == crate::Precision::Int8 {
        2
    } else {
        2 * batch * cfg.n_heads as u64 * n_joint * n_joint
    };
    let f32_words = n * (16 * d + 3 * mlp + 2 * cfg.in_channels as u64 + hd)
        + batch * cfg.txt_len as u64 * cfg.context_in_dim as u64
        + attn_words
        + batch * 17 * d;
    let mut bytes = f32_words * 4;
    if precision == crate::Precision::Int8 {
        // I8Scratch: one f32 scale per row plus packed [row, K] buffers for
        // hidden and MLP widths. `storage` is addressed in four-byte words.
        bytes += n * (4 + d + mlp);
    }
    bytes
}

/// The DiT's device footprint: its own weights at `precision`, plus the
/// activation scratch the joint sequence needs.
///
/// This route describes brain-native fp32 maps. A Q8_0 GGUF uses
/// [`gguf_dit_device_bytes`] instead because the constructor retains a mixture
/// of packed-int8 and deliberate-fp32 buffers that cannot be inferred from
/// tensor rank alone.
pub fn dit_bytes(cfg: &Flux2Config, precision: crate::Precision, n_joint: u64, max_batch: u64) -> u64 {
    let weights: u64 = cfg.tensor_manifest().iter().map(|(_, shape)| weight_bytes(shape, precision)).sum();
    weights + dit_scratch_bytes(cfg, precision, n_joint, max_batch)
}

/// Whether `name` becomes a retained fp32 device buffer even on FLUX.2's int8
/// Q8_0 route. This exactly follows `Flux2Model::new_from`: boundary linears,
/// QK/RMS scales, and double-stream MLP-down are intentionally not packed.
fn gguf_f32_device_weight(name: &str) -> bool {
    name == "img_in.weight"
        || name == "txt_in.weight"
        || name == "final_layer.linear.weight"
        || name.contains("norm.query_norm.scale")
        || name.contains("norm.key_norm.scale")
        || (name.starts_with("double_blocks.") && name.ends_with("_mlp.2.weight"))
}

/// Whether `name` remains a host `Vec<f32>` rather than a device buffer.
///
/// The mmap itself is file-backed and demand-paged, not committed RAM; these
/// six vectors are the owned data relevant to the constructor's device cost.
fn gguf_host_weight(name: &str) -> bool {
    matches!(
        name,
        "time_in.in_layer.weight"
            | "time_in.out_layer.weight"
            | "double_stream_modulation_img.lin.weight"
            | "double_stream_modulation_txt.lin.weight"
            | "single_stream_modulation.lin.weight"
            | "final_layer.adaLN_modulation.1.weight"
    )
}

/// The exact Q8_0-GGUF DiT device budget used by the streamed constructor.
///
/// Every tensor is classified from its own header dtype and raw byte count.
/// Q8_0 source blocks are repacked quant→quant into the DP4A layout: 34 source
/// bytes (32 values) become 32 packed bytes plus one f32 scale, or
/// `raw_bytes / 34 * 36`. Deliberate fp32 device exceptions are reconstructed
/// as `element_count * 4`; host modulation vectors upload no device bytes.
/// A non-Q8_0 tensor is rejected: accepting it would silently decode and
/// requantize a mixed-quant checkpoint, which is not a supported FLUX.2 import
/// contract. Header byte lengths, rather than a model-size label or whole-file
/// size, keep this correct for the actual validated checkpoint.
pub fn gguf_dit_device_bytes(g: &checkpoint::gguf::MmapGguf, cfg: &Flux2Config, n_joint: u64, max_batch: u64) -> Result<u64, String> {
    const Q8_FILE_BLOCK_BYTES: u64 = 34;
    const Q8_DEVICE_PACKED_BYTES: u64 = 36;
    let mut device = 0u64;
    for (name, shape) in cfg.tensor_manifest() {
        let (raw, ty) = g.raw_tensor_bytes(&name).ok_or_else(|| format!("flux2: GGUF is missing {name}"))?;
        let raw = raw.len() as u64;
        let elements = shape.iter().product::<usize>() as u64;
        if gguf_host_weight(&name) {
            continue;
        }
        let bytes = if gguf_f32_device_weight(&name) {
            // `up` dequantizes source values to host f32 then uploads this
            // deliberate F32 buffer. That behavior predates this placement
            // fix; the resulting device allocation is exactly elements * 4.
            elements.checked_mul(4).ok_or("flux2: GGUF device footprint overflow")?
        } else if ty == checkpoint::gguf::TYPE_Q8_0 {
            if !raw.is_multiple_of(Q8_FILE_BLOCK_BYTES) {
                return Err(format!("flux2: GGUF Q8_0 tensor {name} has invalid {raw}-byte payload"));
            }
            raw / Q8_FILE_BLOCK_BYTES * Q8_DEVICE_PACKED_BYTES
        } else {
            return Err(format!("flux2: GGUF tensor {name} is {}; FLUX.2 only supports Q8_0 for packed int8 linears and will not upcast it", g.dtype(&name).unwrap_or("unknown")));
        };
        device = device.checked_add(bytes).ok_or("flux2: GGUF device footprint overflow")?;
    }
    device.checked_add(dit_scratch_bytes(cfg, crate::Precision::Int8, n_joint, max_batch)).ok_or_else(|| "flux2: GGUF device footprint overflow".to_string())
}

/// Resolve the one precision that both placement and construction will use.
/// A Q8_0 GGUF is always consumed by FLUX.2's GPU-only packed-int8 path. The
/// CLI distinguishes its historical implicit fp32 default from an explicit
/// `--precision fp32`: the former follows the GGUF source, while the latter is
/// refused rather than silently planning one format and building another. The
/// header validator rejects non-Q8_0 DiT linears before construction.
pub fn effective_dit_precision(dit: &str, requested: crate::Precision, f32_was_explicit: bool) -> Result<crate::Precision, String> {
    if dit.ends_with(".gguf") {
        if requested == crate::Precision::F32 && f32_was_explicit {
            return Err("flux2: --precision fp32 is incompatible with a .gguf DiT; FLUX.2 executes Q8_0 GGUF through its GPU-only int8 path (use --precision int8)".to_string());
        }
        return Ok(crate::Precision::Int8);
    }
    Ok(requested)
}

/// The text encoder's device footprint: weights plus the activation buffers
/// its blocks hold.
///
/// `layers` is how much of the stack is actually built - a truncated shard
/// keeps `[0, deepest_tap)` and nothing past it, so a whole encoder and a
/// truncated one differ by real bytes and the placement must see the
/// difference.
///
/// A `Qwen` shard allocates its activation buffers per block rather than
/// sharing one slab, so the scratch term scales with `layers`, not with 1 -
/// which is the difference between "the f32 encoder fits a 24 GiB card on
/// paper" and the two-card layout the FLUX.2 roadmap records as the one that
/// actually runs. This is a PLACEMENT INPUT, deliberately an approximation
/// of the same shape `resident_flux2`'s own estimate uses for the DiT; the
/// per-card headroom automatic placement keeps free absorbs the remainder.
pub fn te_bytes(te_cfg: &qwen3::QwenConfig, layers: usize, seq: u64, int8: bool) -> u64 {
    let precision = if int8 { crate::Precision::Int8 } else { crate::Precision::F32 };
    let scratch = layers as u64 * seq * (16 * te_cfg.d_model as u64 + 3 * te_cfg.d_ff as u64) * 4;
    scratch
        + te_cfg
        .param_list()
        .iter()
        .filter(|(name, _)| match name.strip_prefix("blocks.") {
            Some(rest) => rest.split('.').next().and_then(|l| l.parse::<usize>().ok()).is_some_and(|l| l < layers),
            None => !name.starts_with("out.") && !name.starts_with("lm_head"),
        })
        // `param_list` gives element counts, not shapes; every quantisable
        // leaf here is a 2-D linear and every f32-in-both-tiers leaf is a 1-D
        // norm scale, so element count alone decides the width.
        .map(|(name, numel)| {
            let two_d = name.ends_with(".weight") && !name.contains("norm") && !name.contains("ln");
            *numel as u64 * if precision == crate::Precision::Int8 && two_d { 1 } else { 4 }
        })
        .sum::<u64>()
}

/// The VAE's device footprint: the largest graph this pipeline will build,
/// which is the DECODE of a full-size image.
///
/// This used to be a flat 2 GiB, and that is what put the reported failure in
/// the field: every denoise step completed and the run died in `decoding`, on
/// a card the plan believed had room. A FLUX.2 decode is dominated by its
/// activations, not its weights, and they scale with the image - at a real
/// output size it needs several times the old constant. `vae::
/// decoder_device_bytes_for_pixels` is the architecture-derived figure,
/// calibrated against the builder's own account of what it allocates and
/// gated in `crates/vae/tests/footprint.rs`.
///
/// `n_out_max` is the pipeline's GENERATED-token ceiling (not the joint
/// sequence's, which also carries reference conditioning that is encoded but
/// never decoded); one token is a 16x16 pixel patch, so the output is at most
/// `256 * n_out_max` pixels.
///
/// The ENCODE of a reference at the same size is also priced, and the larger
/// of the two wins: a reference may legitimately arrive at the output size
/// (`strength`/`mask` require exactly that). They are never resident
/// together, because `Pipeline::decode_tokens` releases the encoder cache
/// before it builds the decoder, so this is a max rather than a sum.
pub fn vae_bytes(vae_cfg: &vae::VaeConfig, n_out_max: u64) -> u64 {
    let px = 256 * n_out_max;
    vae::decoder_device_bytes_for_pixels(vae_cfg, px).max(vae::encoder_device_bytes_for_pixels(vae_cfg, px))
}

/// Where the text encoder is built, and at what width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TePlacement {
    /// The canonical card to build the truncated shard on, or `None` for the
    /// whole encoder on whatever device the caller is already on.
    pub gpu_index: Option<usize>,
    pub int8: bool,
}

impl TePlacement {
    /// The historical default: the whole encoder, here, in f32.
    pub fn here() -> TePlacement {
        TePlacement { gpu_index: None, int8: false }
    }
}

/// `BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8]`, if set. Kept as an OVERRIDE of the
/// automatic decision below, not as the way to express it: an operator who
/// has already pinned a layout keeps it, and everybody else stops needing to
/// know the layout exists.
fn te_device_override() -> Result<Option<TePlacement>, String> {
    let Some(s) = std::env::var("BRAIN_FLUX2_TE_DEVICE").ok() else { return Ok(None) };
    let Some(rest) = s.strip_prefix("gpu") else { return Ok(None) };
    let (idx_s, int8) = match rest.strip_suffix(":i8") {
        Some(p) => (p, true),
        None => (rest, false),
    };
    let idx: usize = idx_s.parse().map_err(|_| format!("bad BRAIN_FLUX2_TE_DEVICE {s} (gpu<i>[:i8])"))?;
    Ok(Some(TePlacement { gpu_index: Some(idx), int8 }))
}

/// Ask the engine where this pipeline's parts go.
///
/// Returns the DiT's home and the text encoder's placement. The VAE is
/// declared `with("dit")` - it decodes the DiT's own latents, so a cross-card
/// split there would cost a transfer per step.
///
/// The text encoder's numeric tier follows the DiT's, and is not derived from
/// an estimate. An int8 run asks for the int8 encoder; an f32 run asks for the
/// f32 one, and falls back to int8 only if that does not fit anywhere.
///
/// That is deliberate. The one configuration this pipeline has been measured
/// in on two cards is an int8 DiT beside a truncated **int8** encoder, and a
/// placement estimate is an estimate: choosing the far larger f32 encoder for
/// an int8 run on the strength of one is how a plan that arithmetically fits
/// becomes a driver out-of-memory. Truncation itself is free of that risk -
/// `Shard::owns` is `l < end` and the deepest tap is layer 27, so the layers
/// past it are never read and a truncated shard's conditioning is
/// bit-identical to a whole encoder's at the same width.
/// The parts this pipeline is made of, as the engine sees them: `[dit, te,
/// vae]`, in that order, with the costs each will occupy.
///
/// Public and pure so the two ceilings can be told apart under test. They are
/// easy to confuse and expensive to confuse: `n_joint` carries the reference
/// conditioning tokens and drives the DiT's scratch, while `n_out_max` is the
/// generated tokens alone and drives the VAE - a reference is encoded, never
/// decoded. Pricing the decode off the joint ceiling reserved most of a card
/// for an image that is never produced.
///
/// `te_int8` selects the text encoder's numeric tier (see [`plan_parts`],
/// which tries the DiT's tier first).
pub fn part_needs(
    cfg: &Flux2Config,
    vae_cfg: &vae::VaeConfig,
    precision: crate::Precision,
    n_joint: u64,
    n_out_max: u64,
    max_batch: u64,
) -> Vec<gpu_core::devices::Need> {
    let te_cfg = te_config(cfg);
    let layers = *TAP_LAYERS.iter().max().unwrap();
    let te_int8 = precision == crate::Precision::Int8;
    vec![
        gpu_core::devices::Need::sized("dit", dit_bytes(cfg, precision, n_joint, max_batch), 0).apart(),
        gpu_core::devices::Need::sized("te", te_bytes(&te_cfg, layers, cfg.txt_len as u64, te_int8), 0).apart(),
        gpu_core::devices::Need::sized("vae", vae_bytes(vae_cfg, n_out_max), 0).with("dit"),
    ]
}

pub fn plan_parts(cfg: &Flux2Config, paths: &Paths, vae_cfg: &vae::VaeConfig, precision: crate::Precision, n_joint: u64, n_out_max: u64, max_batch: u64) -> Result<(gpu_core::devices::Homes, TePlacement), String> {
    let te_cfg = te_config(cfg);
    let layers = *TAP_LAYERS.iter().max().unwrap();
    let mut needs = part_needs(cfg, vae_cfg, precision, n_joint, n_out_max, max_batch);
    if paths.dit.ends_with(".gguf") {
        let g = checkpoint::gguf::MmapGguf::open(&paths.dit)?;
        crate::import::validate_manifest(&|n| g.shape(n).map(<[usize]>::to_vec), g.names(), cfg)?;
        needs[0].vram = gguf_dit_device_bytes(&g, cfg, n_joint, max_batch)?;
    };
    let (dit, vae) = (needs[0].clone(), needs[2].clone());
    let mut why = String::new();
    // The DiT's tier first, then the smaller one as a fallback. For an int8
    // run those are the same value, so the loop tries int8 once.
    let tiers = if precision == crate::Precision::Int8 { [true, true] } else { [false, true] };
    for int8 in tiers {
        let te = gpu_core::devices::Need::sized("te", te_bytes(&te_cfg, layers, cfg.txt_len as u64, int8), 0).apart();
        let homes = match gpu_core::devices::place(&[dit.clone(), te, vae.clone()]) {
            Ok(h) => h,
            Err(e) => {
                why = e;
                continue;
            }
        };
        let Some(t) = homes.of("te") else { continue };
        // What was PLANNED must be what is BUILT. The plan costed a
        // truncated shard, so a truncated shard is what goes on the card -
        // whether or not it ended up beside the DiT. (A real two-card run
        // found this the hard way: planning the truncated encoder and then
        // building the whole one on the same card is a plan that was never
        // about the model that got built, and it OOMs.)
        if let gpu_core::devices::Home::Gpu(i) = t {
            return Ok((homes, TePlacement { gpu_index: Some(i as usize), int8 }));
        }
        return Ok((homes, TePlacement::here()));
    }
    // Nothing fits even at int8. Fall back to the historical layout rather
    // than refusing here - the operator may know something the estimate does
    // not - but say WHY first, with the real numbers, so the driver OOM that
    // probably follows is not the first news of it. A Cpu `Home` would merely
    // leave `Gpu::new` unscoped; it is not host-resident DiT execution.
    if !why.is_empty() {
        eprintln!("flux2: no automatic GPU placement fits ({why}); FLUX.2 has no host-resident DiT execution, so attempting the historical ambient-device fallback");
    }
    let here = gpu_core::devices::selected_device()
        .map(|d| gpu_core::devices::Home::Gpu(d.index))
        .unwrap_or(gpu_core::devices::Home::Cpu);
    Ok((gpu_core::devices::Homes::new(vec![("dit".into(), here), ("te".into(), here), ("vae".into(), here)]), TePlacement::here()))
}

/// Which Qwen3 the text encoder is, per DiT width. One place, so the encoder
/// COSTED for placement is provably the encoder BUILT.
fn te_config(cfg: &Flux2Config) -> qwen3::QwenConfig {
    if cfg.context_in_dim == 12288 {
        qwen3::QwenConfig::qwen3_8b()
    } else {
        qwen3::QwenConfig::qwen3_4b()
    }
}

/// Build the FLUX.2 text encoder: Qwen3-4B (klein-4b) or Qwen3-8B (klein-9b),
/// sized for one `txt_len`-token sequence, ready for the 3-tap masked-pad
/// conditioning [`Pipeline::encode_prompt`] and [`crate::finetune`] both read.
///
/// One implementation for both callers on purpose. Generation and fine-tuning
/// must condition on the *same* encoder or an adapter trained by one is
/// trained against features the other does not produce, and the two are easy
/// to drift apart: the finetune path had its own copy that still slurped the
/// whole checkpoint as an fp32 `HashMap` (`read_model_dir` +
/// `brain_init_from_hf`) long after this one stopped.
///
/// **Streamed, not slurped.** For Qwen3-8B the eager map is the largest host
/// allocation the process makes, and it is made only to be read once, tensor
/// by tensor, into device buffers and then dropped. A mapped `WeightReader` +
/// `RemapSource` hands `new_shard`/`new_shard_i8` the same bytes through the
/// `checkpoint::TensorSource` seam they already accept, so the host holds one
/// tensor at a time instead of the model.
///
/// **Inference-only.** The encoder is read, never trained: nothing in this
/// crate differentiates it. `Qwen::new`'s `train = true` gave every parameter
/// `Role::Trainable`, which allocates a gradient and two Adam moments beside
/// each weight - four times the model, for three buffers no code path here
/// ever reads. `Role::Frozen` allocates the weights only, and the forward is
/// bit-identical (`qwen3`'s `frozen_and_trainable_builds_agree_on_the_forward`).
///
/// `BRAIN_FLUX2_TE_NO_STREAM=1` forces the old whole-map route. Not a
/// correctness switch - both produce the same bytes, pinned per tensor in
/// `qwen3` - but it is how the two are A/B'd on a real checkpoint, and a valve
/// if a configuration ever needs the map route back.
pub fn build_text_encoder(cfg: &Flux2Config, paths: &Paths) -> Result<qwen3::Qwen, String> {
    // No plan in hand (the finetune path, a direct library caller): honour an
    // explicit override, else build here - exactly the historical behaviour.
    let te = te_device_override()?.unwrap_or_else(TePlacement::here);
    build_text_encoder_on(cfg, paths, te)
}

/// [`build_text_encoder`] with the placement already decided.
pub fn build_text_encoder_on(cfg: &Flux2Config, paths: &Paths, te: TePlacement) -> Result<qwen3::Qwen, String> {
    let te_cfg = te_config(cfg);
    let te_no_stream = std::env::var("BRAIN_FLUX2_TE_NO_STREAM").is_ok_and(|v| v != "0");
    let te_eager: Option<std::collections::HashMap<String, Vec<f32>>> = if te_no_stream {
        let ts = checkpoint::safetensors::read_model_dir(std::path::Path::new(&paths.te))?;
        Some(qwen3::import::brain_init_from_hf(ts, &te_cfg)?)
    } else {
        None
    };
    let te_reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&paths.te))
        .map_err(|e| format!("flux2: open text encoder {}: {e}", paths.te))?;
    // TE placement: default = ambient device; `BRAIN_FLUX2_TE_DEVICE=gpu<i>`
    // builds a truncated fp32 shard on that card, so the DiT can own the
    // other card whole. `Shard::owns` is `l < end`, so the shard keeps
    // layers `[0, deepest)` - the residual stream the deepest tap reads has
    // passed through exactly those, and the remaining layers, the final
    // norm and the LM head are never read. A `:i8` suffix (`gpu<i>:i8`)
    // uses the int8 (DP4A) shard instead, which is several times smaller.
    // The masked-pad kmask path is shared by both shard graphs, so parity
    // is unchanged (int8 is the lossy tier, gated in its own test).
    //
    // This is a TWO-CARD layout. Putting the encoder on the DiT's own card
    // is not a supported configuration at klein-9b/1024x768: measured, the
    // DiT plus VAE alone comes close to filling a 24 GB card, and even the
    // truncated int8 encoder is far too large to join it. See the FLUX.2
    // roadmap for the measured budgets. Today that combination fails as a
    // raw device out-of-memory rather than a refusal naming the two
    // budgets, which is a known gap recorded there.
    let deepest = *TAP_LAYERS.iter().max().unwrap();
    // `BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8]` is user input, parsed to a
    // canonical card index at this edge; the shard's gpu_index is what
    // places the build (device registry) - later device creation (VAE)
    // stays on the ambient card beside the DiT.
    Ok(match te.gpu_index {
        Some(idx) => {
            let te_i8 = te.int8;
            let shard = qwen3::Shard { start: 0, end: deepest, embed: true, head: false, gpu_index: idx };
            // Shard-aware coverage: this build reads the embedding and
            // layers `[0, deepest)` and nothing else, so the checkpoint is
            // required to carry exactly those. The layers past the tap,
            // the final norm and the LM head are neither read nor demanded
            // - previously they had to be present (and, on a checkpoint
            // still being fetched, downloaded) purely to be validated and
            // discarded. Narrowed, not weakened: a tensor that IS present
            // is still element-count checked, and a tensor mapping outside
            // the config's full parameter list is still a hard error.
            let streamed;
            let src: &dyn checkpoint::TensorSource = match &te_eager {
                Some(m) => m,
                None => {
                    streamed = qwen3::import::hf_shard_source(&te_reader, &te_cfg, &shard)?;
                    &streamed
                }
            };
            if te_i8 {
                qwen3::Qwen::new_shard_i8(te_cfg, 1, cfg.txt_len as u32, src, shard)
            } else {
                qwen3::Qwen::new_shard(te_cfg, 1, cfg.txt_len as u32, src, false, shard)
            }
        }
        // Placed here: the whole encoder on the ambient device, with the
        // weights arriving one tensor at a time. A whole shard requires the
        // whole `param_list()`, so the coverage check here is identical to
        // the one this path always ran.
        None => {
            let shard = qwen3::Shard::whole(te_cfg.n_layers as usize);
            let streamed;
            let src: &dyn checkpoint::TensorSource = match &te_eager {
                Some(m) => m,
                None => {
                    streamed = qwen3::import::hf_shard_source(&te_reader, &te_cfg, &shard)?;
                    &streamed
                }
            };
            qwen3::Qwen::new_shard(te_cfg, 1, cfg.txt_len as u32, src, false, shard)
        }
    })
}

/// A ready-to-generate model: DiT + VAE + text encoder held together.
pub struct Pipeline {
    pub cfg: Flux2Config,
    model: Flux2Model,
    tok: data::qwen_tokenizer::QwenBpe,
    te: qwen3::Qwen,
    vae_cfg: vae::VaeConfig,
    vae_tensors: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    /// Where this pipeline's parts were placed. Held because the VAE is built
    /// lazily, per generation, from host tensors: without it a graph would
    /// land on the process-wide default card rather than beside the DiT whose
    /// latents it is for.
    homes: gpu_core::devices::Homes,
    /// The ONE device every VAE graph is built on - see `build_batched`.
    vae_gpu: gpu_core::Gpu,
    /// The most recently built encoder, kept for the next reference of the
    /// SAME size.
    ///
    /// Bounded at one on purpose. A generation encodes every reference before
    /// it denoises, and references are commonly the same shape (one camera,
    /// one orientation), so a one-entry cache removes the rebuild in the case
    /// that actually repeats. Caching every distinct size instead would make
    /// the pipeline's peak memory a function of how many `--ref` the caller
    /// passed, which no placement estimate can predict - and the estimate is
    /// the thing that has to stay honest.
    enc_cache: std::sync::Mutex<Option<((u32, u32), vae::VaeEncoder)>>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
}

impl Pipeline {
    /// Build for a maximum joint sequence (txt + generated + reference
    /// tokens). `n_img_max` in latent tokens, e.g. 4096 for 1024×1024.
    pub fn build(cfg: &Flux2Config, paths: &Paths, n_img_max: u32) -> Result<Pipeline, String> {
        Pipeline::build_adapted(cfg, paths, n_img_max, None)
    }

    /// [`Pipeline::build`] with an optional trained LoRA adapter
    /// ([`crate::finetune`] output) folded into the DiT tensors before the
    /// model is built — a plain generation run then produces
    /// adapter-conditioned images with no model change.
    pub fn build_adapted(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>) -> Result<Pipeline, String> {
        Pipeline::build_with(cfg, paths, n_img_max, adapter, crate::Precision::F32)
    }

    /// [`Pipeline::build_adapted`] with a DiT numeric tier: `Precision::Int8`
    /// builds the DP4A DiT (~4x smaller than f32 - DiT + int8 TE fit ONE
    /// 24 GB card). A LoRA adapter (if any) is folded into the f32 tensors
    /// BEFORE quantization, so adapters work at either tier - the same order
    /// ComfyUI uses (patch the weights, then run).
    pub fn build_with(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>, precision: crate::Precision) -> Result<Pipeline, String> {
        Pipeline::build_batched(cfg, paths, n_img_max, adapter, precision, 1)
    }

    /// [`Pipeline::build_batched`] told the OUTPUT size separately.
    ///
    /// `n_img_max` is the joint sequence's image ceiling - generated tokens
    /// PLUS reference-conditioning tokens - and sizes the DiT's scratch.
    /// `n_out_max` is the generated tokens alone, and sizes the VAE: a
    /// reference is encoded, never decoded, so pricing the decode from the
    /// joint ceiling reserves for an image the pipeline will never produce.
    /// On a five-reference run that is most of a card's worth of nothing.
    ///
    /// The wrappers above pass `n_out_max = n_img_max`, which is the safe
    /// direction (a pipeline with no references decodes exactly that many
    /// tokens) - callers that know the two apart should use this.
    pub fn build_sized(
        cfg: &Flux2Config,
        paths: &Paths,
        n_img_max: u32,
        n_out_max: u32,
        adapter: Option<&AdapterSpec>,
        precision: crate::Precision,
        max_batch: u32,
    ) -> Result<Pipeline, String> {
        Pipeline::build_inner(cfg, paths, n_img_max, n_out_max, adapter, precision, max_batch)
    }

    /// The DiT half of a build: the weight source decision, any LoRA fold,
    /// and the model construction.
    ///
    /// A Q8_0 GGUF at the int8 tier never needs the fp32 model. The
    /// checkpoint already holds int8, and `DitWeights::Gguf` repacks each
    /// Q8_0 matrix quant→quant into this engine's per-row layout, one at a
    /// time; routing it through the fp32 map instead materializes the whole
    /// model (36.3 GB on klein-9b) purely as an intermediate, reads it back
    /// twice to quantize, and frees it again. The result is BIT-IDENTICAL
    /// either way - see `crate::weights` for why that is provable rather than
    /// approximate - so this is a pure cost decision, not a fidelity one.
    ///
    /// A third-party LoRA still needs a float domain, but per tensor rather
    /// than over a resident map, so it rides the same streamed path. brain's
    /// own adapter container does not: it folds through
    /// `LoraAdapter::fold_into_tensors`, which is written against the whole
    /// map. Safetensors, diffusers dirs, and the fp32 tier take the map route;
    /// a GGUF with a non-Q8_0 DiT linear is rejected during placement rather
    /// than silently converted.
    fn build_dit(
        cfg: &Flux2Config,
        paths: &Paths,
        n_max: u32,
        adapter: Option<&AdapterSpec>,
        precision: crate::Precision,
        max_batch: u32,
        gpu: gpu_core::Gpu,
    ) -> Result<Flux2Model, String> {
        let external = adapter.filter(|a| a.path.ends_with(".safetensors"));
        // `BRAIN_FLUX2_NO_STREAM=1` forces the fp32-map route. Both produce
        // the same bytes, so this is not a correctness switch - it is what
        // lets the two be A/B'd on a real checkpoint (which is how the
        // byte-identity of a real 9B generation was checked, adapter and
        // all), and a valve if a checkpoint ever trips the streamed path.
        let no_stream = std::env::var("BRAIN_FLUX2_NO_STREAM").is_ok_and(|v| v != "0");
        if no_stream && paths.dit.ends_with(".gguf") {
            return Err("flux2: BRAIN_FLUX2_NO_STREAM is incompatible with a .gguf DiT; whole-map loading would decode its quantized tensors before construction".to_string());
        }
        let streamable = precision == crate::Precision::Int8
            && paths.dit.ends_with(".gguf")
            && adapter.is_none_or(|a| a.path.ends_with(".safetensors"));
        if streamable {
            let g = checkpoint::gguf::MmapGguf::open(&paths.dit)?;
            // Two-way coverage still has to hold, and it has to hold BEFORE
            // any weight is read: it is what catches a wrong checkpoint, and
            // skipping it because the load got cheaper would trade the one
            // check that matters for the saving.
            crate::import::validate_manifest(&|n| g.shape(n).map(<[usize]>::to_vec), g.names(), cfg)?;
            let lora = match external {
                Some(ap) => {
                    let l = crate::weights::PendingLora::open(&ap.path, ap.scale, &|n| g.shape(n).map(<[usize]>::to_vec))?;
                    let (pairs, rank, scale) = l.summary();
                    // Loud on success too: a run that claims to be adapted
                    // should say how much of the model it actually moved, so
                    // a silent no-op cannot hide behind a clean exit.
                    eprintln!("flux2: folded external LoRA {} - {pairs} linears, rank {rank}, strength {scale}", ap.path);
                    Some(l)
                }
                None => None,
            };
            let src = crate::weights::DitWeights::gguf_adapted(&g, lora.as_ref());
            return Ok(Flux2Model::new_from(cfg, &src, gpu, n_max, max_batch, precision));
        }

        let mut dit_ts = read_dit_tensors(&paths.dit, cfg)?;
        if let Some(ap) = adapter {
            // Two adapter families reach this point, told apart by extension:
            // a `.safetensors` is a THIRD-PARTY (ai-toolkit / ComfyUI) file
            // over the fused matrices, anything else is brain's own trained
            // checkpoint container. Both fold into the same f32 tensor map.
            if ap.path.ends_with(".safetensors") {
                let info = crate::lora::fold_external_adapter(&ap.path, &mut dit_ts, ap.scale)?;
                eprintln!(
                    "flux2: folded external LoRA {} - {} linears, rank {}, strength {}",
                    ap.path, info.pairs, info.rank, info.scale
                );
            } else {
                // The adapter's tensor shapes depend only on the architecture, not
                // the latent grid - any (lh, lw) loads it.
                let tcfg = crate::modelgrad::Cfg::from_flux2(cfg, 1, 1);
                let ad = crate::lora::load_adapter(&ap.path, &tcfg)?;
                // `ap.scale` multiplies the checkpoint's own alpha, exactly as
                // it does on the external branch above - a strength the CLI
                // parses but the model ignores is worse than no strength.
                ad.fold_into_tensors_at(&mut dit_ts, ap.scale)?;
                eprintln!(
                    "flux2: folded brain LoRA {} - rank {}, strength {}",
                    ap.path,
                    ad.rank(),
                    ap.scale
                );
            }
        }
        let model = Flux2Model::new_batched(cfg, &dit_ts, gpu, n_max, max_batch, precision);
        drop(dit_ts);
        Ok(model)
    }

    /// [`Pipeline::build_with`] sized for up to `max_batch` concurrent
    /// generations sharing one denoise loop ([`Pipeline::generate_batch`]).
    /// Only the DiT activation scratch grows; the text encoder and VAE stay
    /// single-stream.
    pub fn build_batched(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>, precision: crate::Precision, max_batch: u32) -> Result<Pipeline, String> {
        Pipeline::build_inner(cfg, paths, n_img_max, n_img_max, adapter, precision, max_batch)
    }

    fn build_inner(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, n_out_max: u32, adapter: Option<&AdapterSpec>, precision: crate::Precision, max_batch: u32) -> Result<Pipeline, String> {
        let n_max = cfg.txt_len as u32 + n_img_max;
        // Resolve the source's executable representation before placement.
        // Planning and construction must receive this same value; otherwise a
        // Q8_0 GGUF could be budgeted as packed int8 then built as fp32.
        let precision = effective_dit_precision(&paths.dit, precision, false)?;
        // The VAE config is read FIRST because the placement estimate is
        // computed from it: the decode's footprint is a property of this
        // checkpoint's channel schedule and the output size, not a constant.
        let vp = std::path::Path::new(&paths.vae);
        let (vae_file, vae_json) = if vp.is_dir() {
            (vp.join("diffusion_pytorch_model.safetensors"), std::fs::read_to_string(vp.join("config.json")).ok())
        } else {
            (vp.to_path_buf(), None)
        };
        let vae_cfg = match vae_json {
            Some(j) => vae::VaeConfig::from_json(&serde_json::from_str(&j).map_err(|e| e.to_string())?),
            None => vae::VaeConfig::flux2(),
        };
        // Ask the engine where this pipeline's three parts go, then build
        // each one there. Nothing below names a card.
        let (homes, auto_te) = plan_parts(cfg, paths, &vae_cfg, precision, n_max as u64, n_out_max as u64, max_batch.max(1) as u64)?;
        // An operator who pinned a layout keeps it; everyone else gets the
        // automatic one. Either way the run says what it did, in the same
        // place it reports reference sizes and token counts.
        let te_place = te_device_override()?.unwrap_or(auto_te);
        eprintln!(
            "flux2: placement {} (text encoder: {})",
            homes.describe(),
            match te_place.gpu_index {
                Some(i) => format!("truncated {} shard on gpu{i}", if te_place.int8 { "int8" } else { "f32" }),
                None => "whole encoder beside the DiT".to_string(),
            }
        );
        let gpu = homes.run("dit", || gpu_core::Gpu::new(crate::model::KERNELS))?;
        let model = homes.run("dit", || Self::build_dit(cfg, paths, n_max, adapter, precision, max_batch.max(1), gpu))??;

        let tok = data::qwen_tokenizer::QwenBpe::from_file(&paths.tokenizer)?;
        let te = build_text_encoder_on(cfg, paths, te_place)?;
        // ONE device for every VAE graph this pipeline will ever build. Each
        // encode/decode used to stand up its own - recompiling every kernel,
        // and (on a two-card box) re-resolving the ambient selection, which
        // the plan may have moved since. Created here, under the part's own
        // placement, so every graph built on it lands where the plan said.
        let vae_gpu = homes.run("vae", || vae::device(None))?;

        let vae_ts = checkpoint::safetensors::read(vae_file.to_str().unwrap())?;
        let mut map = std::collections::HashMap::new();
        let (mut bn_mean, mut bn_var) = (Vec::new(), Vec::new());
        for t in vae_ts {
            if t.name == "bn.running_mean" {
                bn_mean = t.data.clone();
            }
            if t.name == "bn.running_var" {
                bn_var = t.data.clone();
            }
            map.insert(t.name, (t.shape, t.data));
        }
        if bn_mean.is_empty() || bn_var.is_empty() {
            return Err("vae checkpoint missing bn.running_{mean,var}".into());
        }

        Ok(Pipeline { cfg: cfg.clone(), model, tok, te, vae_cfg, vae_tensors: map, bn_mean, bn_var, homes, vae_gpu, enc_cache: std::sync::Mutex::new(None) })
    }

    /// Prompt → `[txt_len, context_in_dim]` conditioning (masked-pad,
    /// layers 9/18/27 concatenated per token).
    pub fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
        let templated = self.tok.apply_chat_template_no_think(&[("user", prompt)]);
        let mut ids = self.tok.encode(&templated);
        if ids.len() > self.cfg.txt_len {
            // Loud, not silent: the conditioning is computed from a PREFIX of
            // the user's prompt (audit F18).
            eprintln!("flux2: prompt is {} tokens but the model's text window is {} -- conditioning on the first {} tokens only", ids.len(), self.cfg.txt_len, self.cfg.txt_len);
        }
        ids.truncate(self.cfg.txt_len);
        let content = ids.len();
        ids.resize(self.cfg.txt_len, PAD_TOKEN);
        let taps = self.te.encode_hiddens_padded(&ids, content, &TAP_LAYERS);
        let d = taps[0].len() / self.cfg.txt_len;
        let mut ctx = Vec::with_capacity(self.cfg.txt_len * 3 * d);
        for row in 0..self.cfg.txt_len {
            for tap in &taps {
                ctx.extend_from_slice(&tap[row * d..(row + 1) * d]);
            }
        }
        ctx
    }

    /// VAE-encode an RGB image (`[-1,1]` CHW) to packed+normalized latent
    /// tokens `[lh*lw, 128]` (row-major, matching `position_ids`).
    pub fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
        let mut slot = self.enc_cache.lock().unwrap_or_else(|e| e.into_inner());
        if slot.as_ref().is_none_or(|((ch, cw), _)| (*ch, *cw) != (h, w)) {
            // Drop the previous graph BEFORE building the next one, so two
            // encoders are never resident at once: the replacement is what
            // keeps this cache's footprint equal to one encoder rather than
            // one per distinct reference size.
            *slot = None;
            // Under the part's placement, so that even a backend that cannot
            // hand out a second handle onto `vae_gpu` (and so builds a fresh
            // device) still builds it on the card the plan chose.
            let built = self
                .homes
                .run("vae", || vae::VaeEncoder::from_diffusers_on(&self.vae_gpu, self.vae_cfg.clone(), &self.vae_tensors, h, w))?;
            *slot = Some(((h, w), built));
        }
        let enc = &slot.as_ref().expect("just populated").1;
        let (lh8, lw8) = ((h / 8) as usize, (w / 8) as usize);
        let mean = enc.encode_mean(chw, lh8 as u32, lw8 as u32);
        let eps = self.vae_cfg.batch_norm_eps;
        let packed = vae::latent::pack(&mean, 32, lh8, lw8, &self.bn_mean, &self.bn_var, eps);
        // [128, lh, lw] -> tokens [lh*lw, 128]
        let (lh, lw) = (lh8 / 2, lw8 / 2);
        let mut tokens = vec![0.0f32; lh * lw * 128];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    tokens[(y * lw + x) * 128 + c] = packed[(c * lh + y) * lw + x];
                }
            }
        }
        Ok(tokens)
    }

    /// Latent tokens `[lh*lw, 128]` → RGB u8 HWC.
    pub fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
        // tokens -> [128, lh, lw]
        let mut packed = vec![0.0f32; 128 * lh * lw];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    packed[(c * lh + y) * lw + x] = tokens[(y * lw + x) * 128 + c];
                }
            }
        }
        let eps = self.vae_cfg.batch_norm_eps;
        let unpacked = vae::latent::unpack(&packed, 32, lh * 2, lw * 2, &self.bn_mean, &self.bn_var, eps);
        let (h, w) = ((lh * 16) as u32, (lw * 16) as u32);
        // Nothing reads a reference encoder again this generation, and the
        // decode is the largest graph the pipeline ever builds - so release
        // the cache first. The peak is then max(encode phase, decode phase)
        // rather than their sum, which is what the placement estimate assumes.
        *self.enc_cache.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let dec = self.homes.run("vae", || {
            vae::VaeDecoder::from_diffusers_on(&self.vae_gpu, self.vae_cfg.clone(), &self.vae_tensors, (lh * 2) as u32, (lw * 2) as u32)
        })?;
        let chw = dec.decode(&unpacked);
        // clamp FIRST, then rescale (reference order — reversed produces artifacts)
        let n = (h * w) as usize;
        let mut out = vec![0u8; n * 3];
        for c in 0..3 {
            for i in 0..n {
                let v = chw[c * n + i].clamp(-1.0, 1.0);
                out[i * 3 + c] = (127.5 * (v + 1.0)) as u8;
            }
        }
        Ok(out)
    }

    /// The largest batch [`Pipeline::generate_batch`] can put in one DiT
    /// forward (what the model's scratch was sized for at build time).
    pub fn max_batch(&self) -> u32 {
        self.model.max_batch()
    }

    /// Text-to-image (optionally with reference images for editing).
    /// `refs`: RGB `[-1,1]` CHW images, each with its (h, w) — pre-cropped to
    /// multiples of 16 (see [`ref_from_hwc`]). Returns (rgb8 HWC, width,
    /// height). `cancel` is polled once per denoise step (multi-minute CPU
    /// runs must be abortable); a `Default` token never fires.
    ///
    /// This is [`Pipeline::generate_batch`] with one request — the two share
    /// one denoise loop, so there is no second sampling implementation to
    /// drift.
    pub fn generate(
        &self,
        prompt: &str,
        refs: &[(Vec<f32>, u32, u32)],
        o: &GenOpts,
        cancel: &capability::CancelToken,
        mut progress: impl FnMut(u32, u32, &str),
    ) -> BatchOutcome {
        let req = BatchRequest { prompt: prompt.to_string(), refs: refs.to_vec(), opts: o.clone(), cancel: cancel.clone() };
        self.generate_batch(std::slice::from_ref(&req), &mut progress)
            .pop()
            .expect("one request in, one result out")
    }

    /// Generate `reqs.len()` images through ONE denoise loop: at every step the
    /// still-running requests are packed into a single batched DiT forward
    /// ([`Flux2Model::forward_batch`]).
    ///
    /// Per-request **seed, steps, guidance and prompt** are all honoured:
    ///
    /// * the seed only picks the initial latent, which is per-sample anyway;
    /// * different `steps` mean different sigma schedules, so at loop index `i`
    ///   two samples sit at *different timesteps* — which costs nothing because
    ///   modulation is a per-sample condition group. A request that runs out of
    ///   steps simply leaves the batch, which shrinks for the remainder;
    /// * CFG (undistilled `base` variants) enters as a **second slot** in the
    ///   same batch — the conditional and unconditional evaluations of one
    ///   request are two samples at the same timestep with different `ctx`,
    ///   which used to be two sequential forwards;
    /// * `cancel` is polled per request per step; a cancelled request leaves
    ///   the batch immediately with `Err("cancelled")` and the others continue.
    ///
    /// Requests whose **position ids** differ (a different reference-image
    /// layout at the same total token count) cannot share a slab, so they are
    /// partitioned into id-groups and the groups run one after another. The
    /// text encoder and the VAE stay per request (they are separate models with
    /// their own single-sequence graphs) — only the DiT, which is the whole
    /// denoise cost, batches.
    ///
    /// Results are returned in request order.
    pub fn generate_batch(
        &self,
        reqs: &[BatchRequest],
        progress: &mut dyn FnMut(u32, u32, &str),
    ) -> Vec<BatchOutcome> {
        generate_batch_on(self, reqs, progress)
    }
}

/// What the sampler needs from the models underneath it: the DiT forward, plus
/// the two codecs that bracket it.
///
/// The seam exists so the sampling logic - the sigma schedule, the img2img
/// init, the mask blending, the batching and the per-request cancellation - can
/// be exercised without a multi-gigabyte checkpoint on disk and a card to put
/// it on. [`Pipeline`] is the one production implementation; there is no second
/// sampler behind it to drift.
trait Denoiser {
    fn cfg(&self) -> &Flux2Config;
    fn encode_prompt(&self, prompt: &str) -> Vec<f32>;
    fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String>;
    fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String>;
    fn max_batch(&self) -> u32;
    fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>>;
}

impl Denoiser for Pipeline {
    fn cfg(&self) -> &Flux2Config {
        &self.cfg
    }
    fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
        Pipeline::encode_prompt(self, prompt)
    }
    fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
        Pipeline::encode_image(self, chw, h, w)
    }
    fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
        Pipeline::decode_tokens(self, tokens, lh, lw)
    }
    fn max_batch(&self) -> u32 {
        self.model.max_batch()
    }
    fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
        self.model.forward_batch(samples, ids, n_pred)
    }
}

/// [`Pipeline::generate_batch`] over any [`Denoiser`].
fn generate_batch_on<D: Denoiser>(
    d: &D,
    reqs: &[BatchRequest],
    progress: &mut dyn FnMut(u32, u32, &str),
) -> Vec<BatchOutcome> {
    let mut out: Vec<BatchOutcome> = (0..reqs.len()).map(|_| Err("not run".to_string())).collect();
    // Partition by position ids: one slab layout per group.
    let mut groups: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
    for (i, r) in reqs.iter().enumerate() {
        match plan_on(d, r) {
            Err(e) => out[i] = Err(e),
            Ok(ids) => match groups.iter_mut().find(|(g, _)| *g == ids) {
                Some((_, v)) => v.push(i),
                None => groups.push((ids, vec![i])),
            },
        }
    }
    for (ids, members) in groups {
        denoise_group_on(d, reqs, &ids, &members, &mut out, progress);
    }
    out
}

/// Validate one request and return its joint position ids (the key that
/// decides which requests can share a batched forward).
fn plan_on<D: Denoiser>(d: &D, r: &BatchRequest) -> Result<Vec<u32>, String> {
    let o = &r.opts;
    if !o.width.is_multiple_of(16) || !o.height.is_multiple_of(16) {
        return Err(format!("width/height must be multiples of 16 (got {}×{})", o.width, o.height));
    }
    let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
    // Keep in step with the token builder: a reference contributes position
    // ids at the size its CONDITIONING copy is encoded at, which under
    // `strength` is a downscale of the first reference rather than its own
    // dimensions.
    let ref_dims: Vec<(usize, usize)> = cond_sizes(&r.refs, o)
        .into_iter()
        .flatten()
        .map(|(rh, rw)| ((rh / 16) as usize, (rw / 16) as usize))
        .collect();
    Ok(position_ids(d.cfg().txt_len, lh, lw, &ref_dims))
}

/// The source content a masked lane preserves, in latent space.
///
/// Held per lane because the blend needs all three at every step: the source
/// latent, the *same* noise draw the init used (so the preserved region walks
/// the source's own forward trajectory rather than a fresh one each step), and
/// the mask resampled to this lane's latent grid.
struct Preserve {
    /// One weight per latent token, `n_gen` of them.
    mask: Vec<f32>,
    /// The source latent `x₀`, `n_gen * in_channels`.
    src: Vec<f32>,
    /// The lane's init noise `ε`, same layout.
    noise: Vec<f32>,
}

/// One id-group's shared denoise loop.
fn denoise_group_on<D: Denoiser>(
    d: &D,
    reqs: &[BatchRequest],
    ids: &[u32],
    members: &[usize],
    out: &mut [BatchOutcome],
    progress: &mut dyn FnMut(u32, u32, &str),
) {
    let cfg = d.cfg();
    // Per-member state; a member that fails to encode drops out here.
    struct Lane {
        idx: usize,
        lh: usize,
        lw: usize,
        n_gen: usize,
        steps: usize,
        guidance: f32,
        ctx: Vec<f32>,
        ctx_uncond: Option<Vec<f32>>,
        ref_tokens: Vec<f32>,
        sigmas: Vec<f32>,
        lat: Vec<f32>,
        /// First schedule index this lane runs; > 0 for img2img inits.
        start: usize,
        /// Set only when the request carries a mask; `None` leaves the
        /// trajectory bit-for-bit what it was before masking existed.
        preserve: Option<Preserve>,
    }
    let max_steps_hint = members.iter().map(|&i| reqs[i].steps_for(cfg.distilled)).max().unwrap_or(0) as u32;
    let mut lanes: Vec<Lane> = Vec::new();
    for &i in members {
        let r = &reqs[i];
        let o = &r.opts;
        let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
        let n_gen = lh * lw;
        let steps = r.steps_for(cfg.distilled);
        progress(0, max_steps_hint + 2, "encoding prompt");
        let ctx = d.encode_prompt(&r.prompt);
        let cf = !cfg.distilled && o.guidance > 1.0;
        let ctx_uncond = if cf { Some(d.encode_prompt("")) } else { None };
        let mut ref_tokens: Vec<f32> = Vec::new();
        let mut failed = None;
        // Every supplied reference conditions the model. Under `strength` the
        // first one does double duty - it is also the init latent below - and
        // is encoded a second time at its conditioning size, which is a
        // downscale of itself. That second encode is the price of the model
        // being able to SEE the reference at all: without it `strength`
        // silently turns off conditioning, and the reference reaches the
        // denoiser only as leftover signal in a partially-noised latent.
        //
        // A mask does not change this either way: it reads the source latent
        // for its preserved region but leaves the token budget alone.
        for ((chw, rh, rw), size) in r.refs.iter().zip(cond_sizes(&r.refs, o)) {
            let Some((ch, cw)) = size else { continue };
            progress(0, max_steps_hint + 2, "encoding reference");
            let small;
            let src = if (ch, cw) == (*rh, *rw) {
                chw
            } else {
                small = resize_ref(chw, *rh, *rw, ch, cw);
                &small
            };
            match d.encode_image(src, ch, cw) {
                Ok(t) => ref_tokens.extend(t),
                Err(e) => failed = Some(e),
            }
        }
        if let Some(e) = failed {
            out[i] = Err(e);
            continue;
        }
        let sigmas = diffusion::scheduler::klein_sigmas(steps, n_gen);
        let noise = model::hostmath::randn(n_gen * cfg.in_channels, o.seed);
        // The source latent `x₀`. `strength` needs it as the init, a mask
        // needs it as the preserved content at every step, and both need
        // it at the output size - so it is encoded ONCE here rather than
        // once per consumer.
        let img2img = o.strength.is_some_and(|s| s < 1.0);
        let want_src = img2img || o.mask.is_some();
        let src = if want_src {
            let why = match (img2img, o.mask.is_some()) {
                (true, true) => "strength/mask",
                (true, false) => "strength",
                _ => "mask",
            };
            let Some((chw, rh, rw)) = r.refs.first() else {
                out[i] = Err(format!("{why} needs a reference image"));
                continue;
            };
            if (*rh as usize / 16) * (*rw as usize / 16) != n_gen {
                out[i] = Err(format!(
                    "{why} needs the reference at the output size ({}x{}, got {rw}x{rh})",
                    o.width, o.height
                ));
                continue;
            }
            match d.encode_image(chw, *rh, *rw) {
                Ok(x0) => Some(x0),
                Err(e) => {
                    out[i] = Err(e);
                    continue;
                }
            }
        } else {
            None
        };
        // Resample the mask to THIS lane's latent grid; the clones are the
        // price of keeping the source and its noise draw alive for the
        // whole trajectory, and are only paid when a mask is present.
        let preserve = match (&o.mask, &src) {
            (Some(m), Some(x0)) => {
                Some(Preserve { mask: m.to_latent(lh, lw), src: x0.clone(), noise: noise.clone() })
            }
            _ => None,
        };
        // img2img: start partway down the schedule from the source latent.
        // `x_σ = (1−σ)·x₀ + σ·ε` is the same forward process the trainer
        // uses (`modelgrad::make_flow_batch`), so the model sees exactly
        // the distribution it was trained on at that σ.
        let (mut lat, start, sigmas) = if img2img {
            let st = o.strength.unwrap_or(1.0).clamp(1e-3, 1.0);
            let sigmas = img2img_sigmas(st, steps, n_gen);
            let x0 = src.as_ref().expect("img2img encodes the source above");
            let lat: Vec<f32> =
                x0.iter().zip(&noise).map(|(&a, &e)| (1.0 - st) * a + st * e).collect();
            (lat, 0usize, sigmas)
        } else {
            (noise, 0usize, sigmas)
        };
        // Seed the preserved region on the source's own trajectory before
        // the first forward, not just after each step: otherwise the model
        // spends step 1 looking at pure noise where the walls should be.
        if let Some(p) = &preserve {
            crate::mask::blend(&mut lat, &p.mask, &p.src, &p.noise, sigmas[start], cfg.in_channels);
        }
        lanes.push(Lane {
            idx: i,
            lh,
            lw,
            n_gen,
            steps,
            guidance: o.guidance,
            ctx,
            ctx_uncond,
            ref_tokens,
            sigmas,
            lat,
            start,
            preserve,
        });
    }
    if lanes.is_empty() {
        return;
    }
    let max_steps = lanes.iter().map(|l| l.steps).max().unwrap_or(0);
    let cap = d.max_batch() as usize;

    for i in 0..max_steps {
        // Cancellation is per request: a cancelled lane leaves the batch and
        // the others keep going (the scheduler handed us N independent jobs).
        lanes.retain(|l| {
            if reqs[l.idx].cancel.is_cancelled() {
                out[l.idx] = Err("cancelled".into());
                false
            } else {
                true
            }
        });
        let active: Vec<usize> =
            (0..lanes.len()).filter(|&k| i >= lanes[k].start && i < lanes[k].steps).collect();
        if active.is_empty() {
            break;
        }
        progress(i as u32 + 1, max_steps as u32 + 2, "denoising");

        // Build one slot per DiT evaluation: (lane, ctx, t). CFG adds the
        // unconditional pass as a second slot at the same timestep.
        let mut joints: Vec<Vec<f32>> = Vec::with_capacity(active.len());
        let mut slots: Vec<(usize, bool, f32)> = Vec::new(); // (active index, is_uncond, t)
        for (a, &k) in active.iter().enumerate() {
            let l = &lanes[k];
            let mut joint = Vec::with_capacity(l.lat.len() + l.ref_tokens.len());
            joint.extend_from_slice(&l.lat);
            joint.extend_from_slice(&l.ref_tokens);
            joints.push(joint);
            slots.push((a, false, l.sigmas[i]));
            if l.ctx_uncond.is_some() {
                slots.push((a, true, l.sigmas[i]));
            }
        }
        // One forward per chunk of at most `max_batch` slots.
        let mut preds: Vec<Vec<f32>> = Vec::with_capacity(slots.len());
        for chunk in slots.chunks(cap) {
            let samples: Vec<crate::model::Sample<'_>> = chunk
                .iter()
                .map(|&(a, unc, t)| {
                    let l = &lanes[active[a]];
                    let ctx = if unc { l.ctx_uncond.as_ref().unwrap() } else { &l.ctx };
                    crate::model::Sample { img_tokens: &joints[a], ctx, t }
                })
                .collect();
            preds.extend(d.forward_batch(&samples, ids, lanes[active[0]].n_gen));
        }
        // Fold CFG and take the Euler step, per lane.
        for (a, &k) in active.iter().enumerate() {
            let cond = slots.iter().position(|&(sa, unc, _)| sa == a && !unc).expect("cond slot");
            let pred: Vec<f32> = match slots.iter().position(|&(sa, unc, _)| sa == a && unc) {
                None => preds[cond].clone(),
                Some(u) => preds[cond].iter().zip(&preds[u]).map(|(&c, &un)| un + lanes[k].guidance * (c - un)).collect(),
            };
            let l = &mut lanes[k];
            let dt = l.sigmas[i + 1] - l.sigmas[i];
            for (x, v) in l.lat.iter_mut().zip(&pred) {
                *x += dt * v;
            }
            // Blended latent diffusion. Outside the mask the latent is not
            // *guided* toward the source, it IS the source renoised to the
            // sigma this step just landed on - so the preserved region is
            // re-anchored every step and reaches σ = 0 as the source
            // exactly, instead of drifting a little further with each
            // forward the way `strength` alone lets it.
            if let Some(p) = &l.preserve {
                crate::mask::blend(&mut l.lat, &p.mask, &p.src, &p.noise, l.sigmas[i + 1], cfg.in_channels);
            }
        }
    }

    progress(max_steps as u32 + 2, max_steps as u32 + 2, "decoding");
    for l in &lanes {
        let o = &reqs[l.idx].opts;
        out[l.idx] = d.decode_tokens(&l.lat, l.lh, l.lw).map(|rgb| (rgb, o.width, o.height));
    }
}


/// One generated image `(rgb8, width, height)`, or why it failed. Named because
/// it appears in the batch entry point, its per-group helper and the
/// single-image wrapper, which must not drift apart.
pub type BatchOutcome = Result<(Vec<u8>, u32, u32), String>;

/// One generation in a [`Pipeline::generate_batch`] call: everything
/// `Pipeline::generate` takes, owned, plus its cancellation token.
#[derive(Clone)]
pub struct BatchRequest {
    pub prompt: String,
    /// RGB `[-1,1]` CHW reference images with their (h, w), pre-cropped to /16.
    pub refs: Vec<(Vec<f32>, u32, u32)>,
    pub opts: GenOpts,
    /// Polled once per denoise step; a `Default` token never fires.
    pub cancel: capability::CancelToken,
}

impl BatchRequest {
    /// Resolved step count (`opts.steps` or the variant default).
    fn steps_for(&self, distilled: bool) -> usize {
        self.opts.steps.unwrap_or(if distilled { 4 } else { 50 }) as usize
    }
}

/// Convert an interleaved HWC RGB image in `[0,1]` (the shared
/// `capability::blob` wire format, also what the CLI's PPM loader produces) to
/// the reference-image layout [`Pipeline::generate`] expects: `[-1,1]` CHW,
/// **center-cropped** to multiples of 16. Returns `(chw, h, w)` with the
/// cropped dims — the ONE implementation shared by the CLI and the capability
/// provider.
pub fn ref_from_hwc(hwc: &[f32], w: u32, h: u32) -> Result<(Vec<f32>, u32, u32), String> {
    let (cw, ch) = (w - w % 16, h - h % 16);
    if cw == 0 || ch == 0 {
        return Err(format!("reference image {w}×{h} is smaller than 16×16"));
    }
    let (x0, y0) = (((w - cw) / 2) as usize, ((h - ch) / 2) as usize);
    let mut chw = vec![0.0f32; 3 * (cw * ch) as usize];
    for c in 0..3usize {
        for y in 0..ch as usize {
            for x in 0..cw as usize {
                let v = hwc[((y + y0) * w as usize + (x + x0)) * 3 + c];
                chw[(c * ch as usize + y) * cw as usize + x] = 2.0 * v - 1.0;
            }
        }
    }
    Ok((chw, ch, cw))
}

/// The largest size with `w x h`'s aspect ratio whose **long edge** is at most
/// `max_edge`. Downscale-only: an image already inside the bound, and a
/// `max_edge` of 0, come back unchanged.
///
/// This is the sizing half of [`ref_from_hwc_bounded`], separated because it is
/// what a caller needs to *predict* a reference's cost before paying it: a
/// reference contributes `(w/16)*(h/16)` tokens, so a 2048x1536 photo is 12288
/// of them - more than a whole 1024x768 generation - and nothing downstream can
/// recover from that. Rounding is to the nearest pixel, not to a multiple of
/// 16: squashing the aspect to land on a /16 grid is visible, and the centre
/// crop in [`ref_from_hwc`] already takes the <16px remainder off each axis.
pub fn fit_long_edge(w: u32, h: u32, max_edge: u32) -> (u32, u32) {
    let long = w.max(h);
    if max_edge == 0 || long <= max_edge {
        return (w, h);
    }
    let s = max_edge as f64 / long as f64;
    (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1))
}

/// [`ref_from_hwc`], with the reference first downscaled so its long edge is at
/// most `max_edge` pixels.
///
/// `None` is not "an unlimited bound" evaluated to a no-op - it takes the
/// bound-free path, so an unbounded call is the same arithmetic on the same
/// pixels it always was. `Some(m)` that the image already satisfies does the
/// same: the bound never upscales, and never resamples an image it would leave
/// the same size.
///
/// The resample is `imaging`'s one host resampler, which is bit-equivalent to
/// the `resize_bilinear` kernel under `AlignCorners::HalfPixel`. It is the host
/// copy and not [`imaging::Ctx::resize`] because references are loaded before
/// the pipeline - and therefore before the `Gpu` - exists.
pub fn ref_from_hwc_bounded(
    hwc: &[f32],
    w: u32,
    h: u32,
    max_edge: Option<u32>,
) -> Result<(Vec<f32>, u32, u32), String> {
    let (tw, th) = match max_edge {
        Some(m) => fit_long_edge(w, h, m),
        None => (w, h),
    };
    if (tw, th) == (w, h) {
        return ref_from_hwc(hwc, w, h);
    }
    ref_from_hwc(&imaging::resize_bilinear_hwc(hwc, 3, w, h, tw, th), tw, th)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(h: u32, w: u32) -> (Vec<f32>, u32, u32) {
        (Vec::new(), h, w)
    }

    /// `--model` names the DiT outright, so `BRAIN_FLUX2_DIT` must not be
    /// required when the CLI hands one in -- while the three components with
    /// no flag stay required, and `from_env` itself is unchanged.
    #[test]
    fn from_env_with_dit_lets_the_flag_stand_in_for_the_variable() {
        std::env::remove_var("BRAIN_FLUX2_DIT");
        for (k, v) in [("BRAIN_FLUX2_VAE", "v"), ("BRAIN_FLUX2_TE", "t"), ("BRAIN_FLUX2_TOKENIZER", "k")] {
            std::env::set_var(k, v);
        }

        let p = Paths::from_env_with_dit(Some("d".to_string())).unwrap();
        assert_eq!(p.dit, "d");
        assert_eq!(p.vae, "v");

        // Without a flag the variable is still required -- and the error
        // teaches the flag that replaces it.
        let err = Paths::from_env_with_dit(None).unwrap_err();
        assert!(err.contains("BRAIN_FLUX2_DIT") && err.contains("--model"), "{err}");

        // The other three have no flag: still required with one present.
        std::env::remove_var("BRAIN_FLUX2_VAE");
        assert!(Paths::from_env_with_dit(Some("d".to_string())).is_err());

        // `from_env` keeps requiring all four.
        std::env::set_var("BRAIN_FLUX2_VAE", "v");
        assert!(Paths::from_env().is_err());
    }

    /// A [`Denoiser`] with no checkpoint behind it, so the sampler itself can
    /// be gated: schedule, img2img init, mask blending, decode.
    ///
    /// * `encode_image` / `decode_tokens` are a real, lossy round trip - the
    ///   mean of each 16x16 pixel block per channel, broadcast back over the
    ///   block. Like the VAE it is many-to-one, so "reproduces the source"
    ///   has to be asserted against the round trip AND against the original
    ///   with a stated tolerance, exactly as it does on real weights.
    /// * `forward_batch` is the exact velocity field `v = (x − target)/σ`,
    ///   whose Euler solution is `x = target + C·σ`. One integration therefore
    ///   lands on `target` - a fixed, deterministic "generated image" that
    ///   depends on the prompt and not at all on the source, which is what
    ///   makes "this region was regenerated" unambiguous.
    ///
    /// Unlike the real VAE this decoder is block-local, so the pixel-level
    /// equalities below are exact. On real weights the same guarantees hold in
    /// *latent* space, and the decoder's receptive field smears them across
    /// the mask seam by a few pixels: preserved regions are exact latents, not
    /// exact pixels, within a few pixels of a mask boundary.
    struct Stub {
        cfg: Flux2Config,
        /// Every `(joint image sequence, position-id count)` the sampler
        /// handed to the DiT. What the model *attends to* is not observable
        /// from the returned image, so the gates below read it here.
        seen: std::cell::RefCell<Vec<(Vec<f32>, usize)>>,
        /// Every sigma the sampler evaluated the DiT at, in order. What
        /// schedule a run integrates is not observable from the returned
        /// image either - see the gate that reads this.
        sigmas: std::cell::RefCell<Vec<f32>>,
    }

    impl Stub {
        fn new() -> Stub {
            Stub {
                cfg: Flux2Config { in_channels: 4, txt_len: 8, ..Flux2Config::klein_4b() },
                seen: Default::default(),
                sigmas: Default::default(),
            }
        }
    }

    impl Denoiser for Stub {
        fn cfg(&self) -> &Flux2Config {
            &self.cfg
        }
        fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
            let d = self.cfg.context_in_dim;
            (0..self.cfg.txt_len * d)
                .map(|i| ((i + prompt.len()) as f32 * 0.017).sin())
                .collect()
        }
        fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
            let (h, w) = (h as usize, w as usize);
            let (lh, lw) = (h / 16, w / 16);
            let ch = self.cfg.in_channels;
            let mut t = vec![0.0f32; lh * lw * ch];
            for c in 0..3.min(ch) {
                for y in 0..lh {
                    for x in 0..lw {
                        let mut s = 0.0f32;
                        for dy in 0..16 {
                            for dx in 0..16 {
                                s += chw[c * h * w + (y * 16 + dy) * w + x * 16 + dx];
                            }
                        }
                        t[(y * lw + x) * ch + c] = s / 256.0;
                    }
                }
            }
            Ok(t)
        }
        fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
            let (h, w) = (lh * 16, lw * 16);
            let ch = self.cfg.in_channels;
            let mut out = vec![0u8; h * w * 3];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let v = tokens[((y / 16) * lw + x / 16) * ch + c].clamp(-1.0, 1.0);
                        out[(y * w + x) * 3 + c] = (127.5 * (v + 1.0)) as u8;
                    }
                }
            }
            Ok(out)
        }
        fn max_batch(&self) -> u32 {
            4
        }
        fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
            let ch = self.cfg.in_channels;
            samples
                .iter()
                .map(|s| {
                    self.seen.borrow_mut().push((s.img_tokens.to_vec(), ids.len()));
                    self.sigmas.borrow_mut().push(s.t);
                    // The conditioning tail shifts the target's phase. Without
                    // this the stub's output is blind to the reference tokens
                    // and a byte-identity gate over a rendered image could not
                    // see a conditioning change at all - which is the very
                    // thing being fenced. The summary is deliberately
                    // POSITION-WEIGHTED: a plain mean is almost invariant to
                    // resampling the same photograph, so it cannot tell a
                    // full-size conditioning copy from a downscaled one. It is
                    // still one scalar shared by every token, so the velocity
                    // reads only its own latent and the mask gates' exact
                    // equalities hold.
                    free_target(n_pred, ch, s)
                        .into_iter()
                        .enumerate()
                        .map(|(i, g)| (s.img_tokens[i] - g) / s.t.max(1e-6))
                        .collect()
                })
                .collect()
        }
    }

    /// The stub denoisers' "free generation": a fixed image determined by the
    /// prompt and by the conditioning tail, and **independent of the init
    /// latent** - which is what makes "this region was regenerated"
    /// unambiguous.
    ///
    /// The conditioning summary is deliberately POSITION-WEIGHTED: a plain
    /// mean is almost invariant to resampling the same photograph, so it could
    /// not tell a full-size conditioning copy from a downscaled one, and a
    /// byte-identity gate over a rendered image would be blind to exactly the
    /// change it is fencing. It is still one scalar shared by every token, so
    /// each velocity reads only its own latent and the mask gates' exact
    /// per-region equalities hold.
    fn free_target(n_pred: usize, ch: usize, s: &crate::model::Sample<'_>) -> Vec<f32> {
        let tail = &s.img_tokens[n_pred * ch..];
        let cond = tail
            .iter()
            .enumerate()
            .map(|(j, &v)| v * (j as f32 * 0.37).sin())
            .sum::<f32>()
            / (tail.len().max(1)) as f32;
        (0..n_pred * ch).map(|i| (i as f32 * 0.031 + s.ctx[0] + cond).sin() * 0.8).collect()
    }

    /// A deterministic source photo: `[-1,1]` CHW, structured enough that a
    /// left half and a right half are visibly different.
    fn source(h: u32, w: u32) -> (Vec<f32>, u32, u32) {
        let (hu, wu) = (h as usize, w as usize);
        let mut chw = vec![0.0f32; 3 * hu * wu];
        for c in 0..3 {
            for y in 0..hu {
                for x in 0..wu {
                    // Low spatial frequency, like a real photograph relative to
                    // a 16x16 latent cell: the codec's block average is then a
                    // faithful round trip rather than a blur beyond
                    // recognition, which is what the VAE's own fidelity looks
                    // like and what the tolerances below are calibrated to.
                    chw[c * hu * wu + y * wu + x] =
                        ((x as f32 * 0.006 + y as f32 * 0.004 + c as f32).sin()) * 0.9;
                }
            }
        }
        (chw, h, w)
    }

    fn run(mask: Option<crate::mask::Mask>, w: u32, h: u32) -> Vec<u8> {
        let d = Stub::new();
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(0.9),
            steps: Some(4),
            guidance: 4.0,
            seed: 11,
            mask,
            ref_cond_scale: DEFAULT_REF_COND_SCALE,
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs: vec![source(h, w)],
            opts,
            cancel: Default::default(),
        };
        generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation")
            .0
    }

    fn solid(v: f32, w: u32, h: u32) -> crate::mask::Mask {
        crate::mask::Mask::new(vec![v; (w * h) as usize], w, h).unwrap()
    }

    /// **Gate 1 - masking is free when it is not used.** An all-white mask must
    /// be BIT-IDENTICAL to no mask at all. Anything weaker leaves every
    /// existing unmasked generation one rounding step away from its previous
    /// output, and this feature would be a silent regression for everyone who
    /// never asked for it.
    #[test]
    fn an_all_white_mask_is_bit_identical_to_no_mask() {
        let (w, h) = (128u32, 96u32); // 4:3, the aspect this was built for
        assert_eq!(run(None, w, h), run(Some(solid(1.0, w, h)), w, h));
    }

    /// **Gate 2 - an all-black mask reproduces the source.** Exactly, against
    /// the codec round trip (the best any latent-space edit can do), and within
    /// a stated tolerance against the original - asserted on rel_l2 as well as
    /// cosine, because cosine alone is scale-invariant and would pass a
    /// uniformly brightened image.
    #[test]
    fn an_all_black_mask_reproduces_the_source() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let round_trip = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let got = run(Some(solid(0.0, w, h)), w, h);
        assert_eq!(got, round_trip, "black must land on the source latent exactly");

        // ... and that round trip is genuinely the source, not a grey field.
        let n = (h * w) as usize;
        let mut src8 = vec![0u8; n * 3];
        for i in 0..n {
            for c in 0..3 {
                src8[i * 3 + c] = (127.5 * (chw[c * n + i].clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
        let (cos, rel) = agreement(&got, &src8);
        assert!(cos > 0.99, "cosine {cos}");
        assert!(rel < 0.15, "rel_l2 {rel}");

        // A generation with no preservation at all must NOT pass that bar -
        // otherwise the gate above proves nothing about the mask.
        let (cos_free, rel_free) = agreement(&run(Some(solid(1.0, w, h)), w, h), &src8);
        assert!(cos_free < 0.99 || rel_free > 0.15, "free run: cosine {cos_free}, rel_l2 {rel_free}");
    }

    /// **Gate 3 - a mask is spatial.** With the left half white the right half
    /// must match the preserved baseline and the left half must not; and the
    /// mirror image must hold against the regenerated baseline. Both directions
    /// are asserted, because a mask that changes nothing and a mask that
    /// changes everything are both failures and each assertion alone catches
    /// only one of them.
    ///
    /// Neither baseline is itself produced by a mask: "preserved" is the codec
    /// round trip of the source and "regenerated" is a plain unmasked run. An
    /// earlier version of this test compared against all-black and all-white
    /// *mask* runs and was therefore blind to a global mask inversion, which
    /// flips the baselines in lockstep with the result.
    #[test]
    fn a_split_mask_regenerates_one_half_and_preserves_the_other() {
        let (w, h) = (128u32, 96u32);
        let mut v = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..(w / 2) as usize {
                v[y * w as usize + x] = 1.0;
            }
        }
        let split = run(Some(crate::mask::Mask::new(v, w, h).unwrap()), w, h);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let kept = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let freed = run(None, w, h);

        let half = |img: &[u8], left: bool| -> Vec<u8> {
            let mut o = Vec::new();
            for y in 0..h as usize {
                let (a, b) = if left { (0, w as usize / 2) } else { (w as usize / 2, w as usize) };
                for x in a..b {
                    o.extend_from_slice(&img[(y * w as usize + x) * 3..][..3]);
                }
            }
            o
        };
        assert_eq!(half(&split, false), half(&kept, false), "the black half must be the source");
        assert_ne!(half(&split, true), half(&kept, true), "the white half must NOT be the source");
        assert_ne!(half(&split, false), half(&freed, false), "the black half must not be a free generation");
        // Gate 1, restricted to a region: the white half is untouched by the
        // blend, so it reproduces the unmasked run exactly. (Exact here because
        // this stub's velocity reads only its own token; with real attention
        // the white half still *sees* the preserved half, which is the point of
        // masking, so on real weights this is a resemblance, not an equality.)
        assert_eq!(half(&split, true), half(&freed, true), "the white half must be the unmasked generation");
    }

    /// Cosine and relative L2 between two u8 images, on the `[-1,1]` scale the
    /// latents live on.
    fn agreement(a: &[u8], b: &[u8]) -> (f32, f32) {
        let f = |x: &[u8]| -> Vec<f32> { x.iter().map(|&v| v as f32 / 127.5 - 1.0).collect() };
        let (a, b) = (f(a), f(b));
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let d2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        (dot / (na * nb).max(1e-12), d2 / nb.max(1e-12))
    }

    /// **Gate 4 - a supplied reference always conditions the model.**
    /// `strength` decides how much denoising starts from the init latent; it
    /// must NOT decide whether the DiT can see the photograph. Under
    /// `strength < 1` the first reference does double duty: it is the init
    /// latent AND it contributes conditioning tokens, at
    /// [`GenOpts::ref_cond_scale`] of its own size (the init role pins it to
    /// the output size, so it is the one reference whose conditioning
    /// resolution the caller cannot pick by choosing a file).
    #[test]
    fn a_strength_reference_still_contributes_conditioning_tokens() {
        let refs = vec![img(768, 1024), img(768, 1024)];
        let base = GenOpts { width: 1024, height: 768, ..GenOpts::default() };

        // No strength: every reference conditions at its own size.
        let no_str = GenOpts { strength: None, ..base.clone() };
        assert_eq!(ref_tokens(&refs, &no_str), 2 * 48 * 64);

        // With strength: the first is BOTH the init latent and conditioning,
        // downscaled by the default 0.75 (1024x768 -> 768x576 -> 48x36).
        let with_str = GenOpts { strength: Some(0.4), ..base.clone() };
        assert_eq!(ref_tokens(&refs, &with_str), 36 * 48 + 48 * 64);

        // A lone reference under strength conditions on ITSELF - this is the
        // whole point. It used to contribute nothing.
        assert_eq!(ref_tokens(&refs[..1], &with_str), 36 * 48);

        // The dial reaches both ends: 1.0 is the full-size conditioning copy
        // (exactly what strength 1.0 costs), 0.0 switches it off entirely -
        // the documented escape hatch back to the old, cheap behaviour.
        let full_cond = GenOpts { ref_cond_scale: 1.0, ..with_str.clone() };
        assert_eq!(ref_tokens(&refs[..1], &full_cond), 48 * 64);
        let off = GenOpts { ref_cond_scale: 0.0, ..with_str.clone() };
        assert_eq!(ref_tokens(&refs[..1], &off), 0);

        // strength == 1.0 consumes no init latent, so nothing is downscaled
        // and the dial does not apply. This is the path that already works.
        for scale in [0.0, 0.75, 1.0] {
            let full = GenOpts { strength: Some(1.0), ref_cond_scale: scale, ..base.clone() };
            assert_eq!(ref_tokens(&refs, &full), 2 * 48 * 64, "scale {scale}");
        }
    }

    /// **Gate 5 - a pipeline is sized for exactly the tokens it attends to.**
    /// The invariant that motivated the removed
    /// `the_strength_init_reference_contributes_no_conditioning_tokens`: the
    /// attention scratch is allocated from [`ref_tokens`], so if the denoise
    /// loop builds a joint sequence of any other length the graph is either
    /// too small (a wrong-shaped forward) or wastefully too large. Only the
    /// answer changed; the invariant did not. Asserted against what the DiT
    /// was actually handed, on both sides of the sizing decision.
    #[test]
    fn the_joint_sequence_is_exactly_what_the_pipeline_was_sized_for() {
        let (w, h) = (128u32, 96u32);
        let refs = [source(h, w), source(h, w)];
        let base = GenOpts { width: w, height: h, steps: Some(2), seed: 5, ..GenOpts::default() };
        let cases = [
            GenOpts { strength: None, ..base.clone() },
            GenOpts { strength: Some(0.4), ..base.clone() },
            GenOpts { strength: Some(0.4), ref_cond_scale: 1.0, ..base.clone() },
            GenOpts { strength: Some(0.4), ref_cond_scale: 0.0, ..base.clone() },
            GenOpts { strength: Some(1.0), ..base.clone() },
        ];
        for (n, opts) in cases.iter().enumerate() {
            for k in 1..=refs.len() {
                let d = Stub::new();
                let req = BatchRequest {
                    prompt: "a staged living room".into(),
                    refs: refs[..k].to_vec(),
                    opts: opts.clone(),
                    cancel: Default::default(),
                };
                generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
                    .pop()
                    .unwrap()
                    .expect("stub generation");
                let n_gen = ((h / 16) * (w / 16)) as usize;
                let want = n_gen + ref_tokens(&refs[..k], opts) as usize;
                let seen = d.seen.borrow();
                assert!(!seen.is_empty(), "case {n}/{k}: no forward ran");
                for (joint, n_ids) in seen.iter() {
                    assert_eq!(joint.len(), want * d.cfg.in_channels, "case {n}/{k}: joint tokens");
                    assert_eq!(*n_ids, 4 * (d.cfg.txt_len + want), "case {n}/{k}: position ids");
                }
            }
        }
    }

    /// **Gate 6 - the model actually receives the photograph.** Gate 5 pins
    /// the *length* of the joint sequence; a pipeline that padded it with
    /// zeros would pass. This pins the *content*: the tail of what the DiT
    /// attends to is the encoding of the reference, downscaled by the
    /// conditioning dial. Under the old behaviour that tail was empty - which
    /// is exactly the defect: at `--strength 0.95` the DiT never saw the
    /// user's photograph at all, and only the leftover signal in a
    /// partially-noised init latent stood between the result and a fresh
    /// generation.
    #[test]
    fn the_denoiser_attends_to_the_downscaled_init_reference() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let src = source(h, w);
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(0.95),
            steps: Some(2),
            seed: 3,
            ..GenOpts::default()
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs: vec![src.clone()],
            opts: opts.clone(),
            cancel: Default::default(),
        };
        generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation");

        let ch = d.cfg.in_channels;
        let n_gen = ((h / 16) * (w / 16)) as usize;
        let seen = d.seen.borrow();
        let (joint, _) = seen.first().expect("at least one forward");
        let tail = &joint[n_gen * ch..];
        assert!(
            !tail.is_empty(),
            "a supplied reference must be attended to, not merely renoised into the init latent"
        );

        // 96x128 at the default 0.75 -> 72x96 floored to /16 -> 64x96 -> 4x6.
        let (ch_px, cw_px) = init_cond_size(opts.ref_cond_scale, h, w).expect("dial is on");
        assert_eq!((ch_px, cw_px), (64, 96));
        let small = resize_ref(&src.0, h, w, ch_px, cw_px);
        let want = d.encode_image(&small, ch_px, cw_px).expect("stub encode");
        assert_eq!(tail, &want[..], "the conditioning tail must BE the reference");
    }

    /// **Gate 7 - `strength == 1.0` is byte-for-byte what it always was.**
    /// The digest below was taken on the code as it stood *before* the
    /// conditioning change, so it is a genuine before/after fence on the one
    /// path users already depend on. If a future edit to the reference
    /// pipeline moves this, it moved a rendered image, not an abstraction.
    #[test]
    fn a_strength_one_run_is_byte_identical_to_the_pre_change_output() {
        let (w, h) = (128u32, 96u32);
        let refs = vec![source(h, w), source(h, w)];
        let d = Stub::new();
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(1.0),
            steps: Some(4),
            seed: 11,
            ..GenOpts::default()
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs,
            opts,
            cancel: Default::default(),
        };
        let rgb = generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation")
            .0;
        assert_eq!(fnv1a(&rgb), 0x0d96_f927_7211_6425u64);
    }

    /// **Gate 7b - both full-strength spellings integrate the klein schedule.**
    /// Gate 7 above is a byte fence on a rendered image and it is deliberately
    /// **blind to the sampler**: [`Stub`]'s velocity `(x − g)/σ` drives one
    /// exact Euler integration onto `g` from any init latent over any sigma
    /// list, so its digest does not move when the schedule does - verified by
    /// mutation, and the reason that digest alone must not be read as a fence
    /// on `--strength`. The sigmas the DiT is evaluated at are therefore
    /// asserted directly, on both spellings of full strength: no `strength` at
    /// all (the free-generation branch) and `strength = 1.0` (the img2img
    /// branch at the top of its range). They are different code paths and they
    /// must integrate the same schedule, or the dial does not reach the
    /// setting it is supposed to reach.
    #[test]
    fn both_spellings_of_full_strength_integrate_the_klein_schedule() {
        let (w, h) = (128u32, 96u32);
        let (steps, n_gen) = (12usize, ((h / 16) * (w / 16)) as usize);
        let want = diffusion::scheduler::klein_sigmas(steps, n_gen);
        for strength in [None, Some(1.0f32)] {
            let d = Stub::new();
            let req = BatchRequest {
                prompt: "a staged bedroom".into(),
                refs: vec![source(h, w)],
                opts: GenOpts {
                    width: w,
                    height: h,
                    strength,
                    steps: Some(steps as u32),
                    seed: 7,
                    ..GenOpts::default()
                },
                cancel: Default::default(),
            };
            generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
                .pop()
                .unwrap()
                .expect("stub generation");
            // The terminal 0 is the endpoint of the last step, never a
            // timestep the model is evaluated at.
            assert_eq!(d.sigmas.borrow().as_slice(), &want[..steps], "strength {strength:?}");
        }
    }

    // ---- `--strength` as a smooth anchoring dial ---------------------------

    /// A denoiser whose clean-image estimate is `x̂₀ = (1−σ)·x + σ·g`: at high
    /// noise it commits to its own idea `g`, at low noise it trusts the
    /// structure it is already looking at. The velocity is then
    /// `v = (x − x̂₀)/σ = x − g`, which is **bounded**, so a trajectory
    /// starting at `σ₀` displaces the latent by `O(σ₀)` and a low-strength run
    /// stays near its init latent - the qualitative behaviour every real
    /// diffusion denoiser has, and the one `--strength` is a dial on.
    ///
    /// [`Stub`]'s velocity `(x − g)/σ` is deliberately *un*bounded: one exact
    /// Euler integration lands on `g` from any init latent over any schedule,
    /// which is what makes the mask gates' equalities exact. That also makes
    /// it useless here - under it every strength renders the same image, so a
    /// monotonicity or continuity assertion written against it could not fail
    /// however the dial was wired. This denoiser exists so those assertions
    /// can fail.
    struct Flow(Stub);

    impl Denoiser for Flow {
        fn cfg(&self) -> &Flux2Config {
            self.0.cfg()
        }
        fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
            self.0.encode_prompt(prompt)
        }
        fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
            self.0.encode_image(chw, h, w)
        }
        fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
            self.0.decode_tokens(tokens, lh, lw)
        }
        fn max_batch(&self) -> u32 {
            self.0.max_batch()
        }
        fn forward_batch(&self, samples: &[crate::model::Sample<'_>], _ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
            let ch = self.0.cfg.in_channels;
            samples
                .iter()
                .map(|s| {
                    self.0.sigmas.borrow_mut().push(s.t);
                    free_target(n_pred, ch, s)
                        .into_iter()
                        .enumerate()
                        .map(|(i, g)| s.img_tokens[i] - g)
                        .collect()
                })
                .collect()
        }
    }

    /// One [`Flow`] render of `source(h, w)` at `strength`, everything else
    /// fixed. `ref_cond_scale` is 1.0 so the conditioning copy is the
    /// reference at its own size on **both** sides of the `strength < 1`
    /// branch: the only thing the gates below vary is the dial.
    fn run_flow(strength: f32, w: u32, h: u32) -> Vec<u8> {
        let req = BatchRequest {
            prompt: "a staged bedroom".into(),
            refs: vec![source(h, w)],
            opts: GenOpts {
                width: w,
                height: h,
                strength: Some(strength),
                steps: Some(12),
                seed: 7,
                ref_cond_scale: 1.0,
                ..GenOpts::default()
            },
            cancel: Default::default(),
        };
        generate_batch_on(&Flow(Stub::new()), std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation")
            .0
    }

    /// The strengths the gates below walk, high to low.
    const LADDER: [f32; 12] =
        [1.0, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.80, 0.60, 0.40, 0.20];

    /// **Gate 8 - the anchoring dial reaches free generation exactly.**
    /// `--strength 1.0` takes a different branch (no init latent, no source
    /// encode), so the only way the dial can be continuous at the top is for
    /// the img2img schedule to *become* the free-generation schedule there -
    /// bit for bit, not approximately, because `strength 1.0` is a shipped
    /// output that must not move by a ULP.
    #[test]
    fn the_img2img_schedule_reaches_the_free_generation_schedule_exactly() {
        for &(steps, n) in &[(4usize, 3072usize), (12, 3072), (12, 1024), (28, 8192)] {
            assert_eq!(
                img2img_sigmas(1.0, steps, n),
                diffusion::scheduler::klein_sigmas(steps, n),
                "steps {steps}, {n} tokens"
            );
        }
    }

    /// **Gate 9 - the dial is continuous at the top.** This is the user-facing
    /// complaint in one number: `0.99` must be a hair from `1.00`, not a
    /// different job. Every sigma lies in `[0, 1]`, so a schedule that is the
    /// full-strength one scaled by `strength` moves no entry by more than
    /// `1 − strength`. A schedule that changes *shape* at the top misses that
    /// bound by more than an order of magnitude, whatever the metric.
    #[test]
    fn a_hair_below_full_strength_is_a_hair_from_the_full_strength_schedule() {
        let (steps, n) = (12usize, 3072usize);
        let full = diffusion::scheduler::klein_sigmas(steps, n);
        for &s in &[0.95f32, 0.99, 0.995, 0.999] {
            let got = img2img_sigmas(s, steps, n);
            let worst = got.iter().zip(&full).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            // `1 − s` evaluated the way the schedule does, so the bound is the
            // property and not a float-rounding contest.
            assert!(worst <= 1.0 - s, "strength {s}: worst |Δσ| = {worst}, bound {}", 1.0 - s);
        }
    }

    /// **Gate 10 - the dial is monotone.** Lowering `--strength` may never
    /// raise the noise level the trajectory starts from or passes through, and
    /// may never lower the weight the source carries in the init latent.
    /// Preservation is not directly assertable on a schedule; these two are
    /// the mechanism behind it, and any remap of the dial has to keep both.
    #[test]
    fn lowering_the_strength_lowers_every_sigma_and_raises_the_source_weight() {
        let (steps, n) = (12usize, 3072usize);
        for pair in LADDER.windows(2) {
            let (hi, lo) = (pair[0], pair[1]);
            let (a, b) = (img2img_sigmas(hi, steps, n), img2img_sigmas(lo, steps, n));
            for (k, (&x, &y)) in a.iter().zip(&b).enumerate() {
                assert!(y <= x, "σ[{k}] rose from {x} at strength {hi} to {y} at {lo}");
                assert!(x <= 0.0 || y < x, "σ[{k}] did not fall from strength {hi} to {lo}");
            }
            // The init latent is `(1−s)·x₀ + s·ε`; its source weight is the
            // preservation dial the schedule is only half of.
            assert!(1.0 - lo > 1.0 - hi, "source weight did not rise from {hi} to {lo}");
        }
    }

    /// The pipeline integrates the schedule the gates above reason about.
    /// Without this they are assertions on a pure function that nothing has to
    /// call.
    #[test]
    fn the_img2img_branch_integrates_the_img2img_schedule() {
        let (w, h) = (128u32, 96u32);
        let (steps, n_gen) = (12usize, ((h / 16) * (w / 16)) as usize);
        for s in [0.9f32, 0.4] {
            let d = Flow(Stub::new());
            let req = BatchRequest {
                prompt: "a staged bedroom".into(),
                refs: vec![source(h, w)],
                opts: GenOpts {
                    width: w,
                    height: h,
                    strength: Some(s),
                    steps: Some(steps as u32),
                    seed: 7,
                    ..GenOpts::default()
                },
                cancel: Default::default(),
            };
            generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
                .pop()
                .unwrap()
                .expect("stub generation");
            assert_eq!(
                d.0.sigmas.borrow().as_slice(),
                &img2img_sigmas(s, steps, n_gen)[..steps],
                "strength {s}"
            );
        }
    }

    /// **Gate 11 - `0.99` renders what `1.00` renders.** Gate 9 fences the
    /// schedule; this fences the picture, end to end through the sampler, the
    /// init latent and the decoder. Asserted on cosine AND relative L2,
    /// because cosine alone is scale-invariant and would pass an image that
    /// merely has the same structure at a different contrast.
    ///
    /// The bound is the one the mechanism supports: at `1 − s = δ` the init
    /// latent carries `δ` of the source and every sigma moves by at most `δ`,
    /// so the rendered difference is `O(δ)` and shrinks with `δ`. A sampler
    /// that changes *shape* at the top instead lands a fixed distance away
    /// however small `δ` is - which is what this bound separates.
    #[test]
    fn a_hair_below_full_strength_renders_what_full_strength_renders() {
        let (w, h) = (128u32, 96u32);
        let full = run_flow(1.0, w, h);
        for &s in &[0.995f32, 0.99] {
            let (cos, rel) = agreement(&run_flow(s, w, h), &full);
            assert!(cos > 0.999, "strength {s} vs 1.0: cosine {cos}, rel_l2 {rel}");
            assert!(rel < 0.02, "strength {s} vs 1.0: cosine {cos}, rel_l2 {rel}");
        }
    }

    /// **Gate 12 - anchoring rises as the dial falls, with no reversal.**
    /// Measured as the relative L2 between the render and the source's own
    /// codec round trip - the best any latent-space edit can do - so 0 is
    /// "returned the photograph" and larger is "redrew more of it".
    #[test]
    fn anchoring_increases_monotonically_as_strength_falls() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let rt = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let dist: Vec<f32> = LADDER.iter().map(|&s| agreement(&run_flow(s, w, h), &rt).1).collect();
        for (pair, ss) in dist.windows(2).zip(LADDER.windows(2)) {
            assert!(
                pair[1] < pair[0],
                "strength {} -> {}: distance from the source went {} -> {} (must fall)",
                ss[0],
                ss[1],
                pair[0],
                pair[1]
            );
        }
        // ... and the two ends are genuinely far apart, so the run above is
        // not a flat line that trivially satisfies a strict inequality on
        // rounding noise.
        assert!(dist[0] - dist[dist.len() - 1] > 0.3, "dial has no range: {dist:?}");
    }

    /// **Gate 13 - `--strength 0` returns the photograph.** Within the VAE
    /// round trip, which is the floor for anything that edits in latent space.
    /// The free run is measured against the same bar to prove the bar is not
    /// one anything would clear.
    #[test]
    fn a_vanishing_strength_returns_the_source() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let rt = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let (cos, rel) = agreement(&run_flow(0.0, w, h), &rt);
        assert!(cos > 0.9995, "strength 0: cosine {cos}, rel_l2 {rel}");
        assert!(rel < 0.02, "strength 0: cosine {cos}, rel_l2 {rel}");
        let (cf, rf) = agreement(&run_flow(1.0, w, h), &rt);
        assert!(cf < 0.9995 || rf > 0.02, "a free run cleared the bar: cosine {cf}, rel_l2 {rf}");
    }

    /// A `w x h` interleaved-RGB `[0,1]` horizontal ramp. Smooth, so a correct
    /// bilinear downscale of it is predictable analytically (see below), and
    /// per-channel distinct, so a channel or axis swap shows up.
    fn ramp(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / (w - 1) as f32;
                let u = y as f32 / (h - 1) as f32;
                let p = ((y * w + x) * 3) as usize;
                v[p] = t;
                v[p + 1] = u;
                v[p + 2] = 0.5 * t + 0.5 * u;
            }
        }
        v
    }

    /// **The reference-size bound is opt-in.** Without a bound - and with one
    /// the image already satisfies - `ref_from_hwc_bounded` must be the
    /// existing `ref_from_hwc`, byte for byte. This is what lets a
    /// bound-less invocation keep producing exactly what it produced before
    /// the bound existed.
    #[test]
    fn an_absent_or_slack_ref_bound_changes_nothing() {
        for &(w, h) in &[(64u32, 48u32), (100, 100), (33, 240), (384, 288)] {
            let src = ramp(w, h);
            let want = ref_from_hwc(&src, w, h).unwrap();
            assert_eq!(ref_from_hwc_bounded(&src, w, h, None).unwrap(), want, "{w}x{h}: no bound");
            let long = w.max(h);
            for m in [long, long + 1, long * 3] {
                assert_eq!(
                    ref_from_hwc_bounded(&src, w, h, Some(m)).unwrap(),
                    want,
                    "{w}x{h}: --ref-size {m} is slack and must not upscale or resample"
                );
            }
        }
    }

    /// **The bound caps the token count, preserving aspect.** A phone photo
    /// costs `(w/16)*(h/16)` reference tokens, which is what decides whether a
    /// run fits the card; the bound is the dial that makes that affordable.
    #[test]
    fn a_ref_bound_caps_the_token_count() {
        let (w, h) = (2048u32, 1536u32);
        let src = ramp(w, h);
        assert_eq!(ref_tokens(&[ref_from_hwc(&src, w, h).unwrap()], &GenOpts::default()), 12288);
        let (chw, ch, cw) = ref_from_hwc_bounded(&src, w, h, Some(384)).unwrap();
        assert!(cw <= 384 && ch <= 384, "long edge not bounded: {cw}x{ch}");
        assert_eq!((cw % 16, ch % 16), (0, 0), "not /16-aligned: {cw}x{ch}");
        // 4:3 in, 4:3 out - within the /16 crop, which can shave <16px an axis.
        let ar = |a: u32, b: u32| a as f64 / b as f64;
        assert!((ar(cw, ch) - ar(w, h)).abs() < 0.05, "aspect drifted: {cw}x{ch}");
        assert_eq!(ref_tokens(&[(chw.clone(), ch, cw)], &GenOpts::default()), (cw / 16) * (ch / 16));
        assert!(ref_tokens(&[(chw.clone(), ch, cw)], &GenOpts::default()) < 700);

        // And it is a real resample, not a crop or a transpose: the ramp is
        // linear, so the downscaled `[-1,1]` CHW plane must agree with the
        // analytic ramp evaluated at the half-pixel source positions.
        let mut want = vec![0f32; chw.len()];
        for c in 0..3usize {
            for y in 0..ch {
                let fy = (((y as f32 + 0.5) * h as f32 / ch as f32) - 0.5) / (h - 1) as f32;
                for x in 0..cw {
                    let fx = (((x as f32 + 0.5) * w as f32 / cw as f32) - 0.5) / (w - 1) as f32;
                    let v = [fx, fy, 0.5 * fx + 0.5 * fy][c];
                    want[(c * ch as usize + y as usize) * cw as usize + x as usize] = 2.0 * v - 1.0;
                }
            }
        }
        let (cos, rel) = brain_testutil::parity::compare(&chw, &want);
        assert!(cos > 0.9999, "downscale is not the ramp: cosine {cos}, rel_l2 {rel}");
        assert!(rel < 2e-3, "downscale is not the ramp: cosine {cos}, rel_l2 {rel}");
    }

    /// FNV-1a 64 over the rendered bytes. A whole reference image is too large
    /// to inline and a tolerance would defeat the purpose, so the fence is a
    /// digest - written here rather than pulled in as a dependency because a
    /// hash of 36 kB in a unit test needs nothing stronger.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}
