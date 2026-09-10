// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FLUX.1 text-to-image pipeline: a prompt in, an image out.
//!
//! Mirrors `sdxlunet::pipeline::Sdxl` and `flux2::pipeline::Pipeline`'s shape
//! (build the conditioning towers, denoise, VAE decode) but is its own loop:
//! FLUX.1 conditions on T5-XXL context + a CLIP-L pooled vector (not FLUX.2's
//! Qwen3 taps), has no undistilled base variant (so no CFG branch at all -
//! `dev`/`kontext-dev` fold a guidance SCALAR into the conditioning via
//! `guidance_in`, `schnell` ignores it), and its 16-channel VAE uses the
//! model's own scalar `(shift_factor, scaling_factor)` affine, NOT
//! `vae::latent::pack`/`unpack` (that module is explicitly FLUX.2's
//! BatchNorm-based packing - reusing it here would silently apply the wrong
//! normalization).
//!
//! # The schedule is FLUX.1's own, not FLUX.2's
//!
//! `diffusion::scheduler::empirical_mu` is FLUX.2 Klein's own empirical fit
//! (its doc says so) - wrong constants for FLUX.1. FLUX.1's `dev`/
//! `kontext-dev` use BFL's `calculate_shift`/`get_lin_function`
//! (`flux/sampling.py`, verbatim in every diffusers `FluxPipeline`): a
//! LINEAR `mu(image_seq_len)` between `(256, 0.5)` and `(4096, 1.15)`, fed
//! through the same [`diffusion::scheduler::time_shift_exponential`] FLUX.2
//! reuses. `schnell` applies **no shift at all** (`shift=(not is_schnell)` in
//! BFL's own CLI) - a plain `linspace(1, 0, steps+1)`.
//!
//! The denoise loop steps this schedule directly (`dt = sigmas[i+1] -
//! sigmas[i]`), the same manual style `flux2::pipeline::Pipeline` uses,
//! rather than through `FlowMatchEulerScheduler` (that wrapper appends its
//! own terminal `0` unconditionally, which double-counts against a
//! `steps+1`-length input like this one - `flux2` avoids it for the same
//! reason).
//!
//! # Not yet in scope
//!
//! Kontext reference-image editing, img2img (`strength`), LoRA adapters, and
//! batched serving are all deferred - this is a single-image text-to-image
//! loop. `int8` DiT precision and automatic DiT/T5-XXL device placement DO
//! exist now (`load_with`/`plan_flux1`). `flux2::pipeline` is the fuller
//! reference for what the still-deferred items need when they land here.
//!
//! # An honest note on verification
//!
//! Every piece this composes (the DiT forward, the T5/CLIP towers, the VAE)
//! is independently parity-gated elsewhere in this workspace. The GLUE
//! written here - patchify layout, position ids, the schedule, the affine
//! latent normalization - is NOT: there is no FLUX.1 checkpoint or reference
//! pipeline dump in this environment to run it against end to end. Treat a
//! first real generation as the actual test of this file.

use std::path::Path;

use clip::config::ClipTextConfig;
use clip::model::ClipText;
use data::unigram::UnigramTokenizer;
use diffusion::scheduler::time_shift_exponential;
use gpu_core::devices::{Homes, Need};
use gpu_core::Gpu;
use t5encoder::config::T5Config;
use t5encoder::model::T5Encoder;
use vae::config::VaeConfig;
use vae::VaeDecoder;

use crate::config::Flux1Config;
use crate::model::{position_ids, Flux1Model, Precision, KERNELS};

/// How the latent is seeded and how many steps to take.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// `None` -> the variant default (4 for schnell, 50 for dev/kontext-dev -
    /// matching BFL's own CLI defaults).
    pub steps: Option<usize>,
    /// `guidance_in`'s scalar. Only meaningful for `dev`/`kontext-dev`
    /// (`guidance_embed = true`); `schnell` ignores it. BFL's CLI default is
    /// 3.5.
    pub guidance: f32,
    pub seed: u64,
    /// Generated size in pixels; must be a multiple of 16 (the VAE's 8x
    /// downscale composed with the DiT's 2x2 patchify).
    pub height: u32,
    pub width: u32,
    /// The denoising step at which `generate_injected`'s conditioning starts
    /// applying (steps before it forward WITHOUT `inject`, same as `inject:
    /// None`). Meaningless when `inject` is `None` (plain `flux1::caps`
    /// always passes 0 here and never reads it). Upstream PuLID-FLUX's own
    /// `start_step`: smaller injects identity sooner (more fidelity, less
    /// editability of the base structure); their guidance is ~4 for
    /// photorealism, ~0-1 for stylization. Default 0 preserves this crate's
    /// prior always-inject behavior.
    pub start_step: usize,
    /// Upstream PuLID-FLUX's OPTIONAL true CFG: `Some(cfg)` runs a SECOND,
    /// un-injected FLUX forward each step from `cfg.start_step` onward on a
    /// negative prompt's own conditioning, and combines `neg + cfg.scale *
    /// (pos - neg)` - on top of (not instead of) the distilled `guidance`
    /// scalar every variant already has. `None` (the default) keeps the
    /// original single-forward-per-step cost and behavior. Meaningless
    /// without a `negative_prompt` passed to `generate_injected` - see its
    /// own doc.
    pub true_cfg: Option<TrueCfg>,
}

/// Parameters of upstream PuLID-FLUX's optional true-CFG branch. See
/// [`GenerateOptions::true_cfg`].
#[derive(Clone, Copy, Debug)]
pub struct TrueCfg {
    pub scale: f32,
    pub start_step: usize,
}

impl Default for GenerateOptions {
    fn default() -> GenerateOptions {
        GenerateOptions { steps: None, guidance: 3.5, seed: 0, height: 1024, width: 1024, start_step: 0, true_cfg: None }
    }
}

/// A loaded FLUX.1 stack.
///
/// # Only the DiT stays resident, for the same reason as SDXL
///
/// FLUX.1-dev is ~12 B params (~48 GB fp32) - even more VRAM-constrained than
/// SDXL. The T5-XXL/CLIP-L towers and the VAE are built for one encode/decode
/// and dropped, the same tiering `sdxlunet::pipeline::Sdxl` uses and for the
/// same reason (documented on its `Sdxl` struct).
pub struct Flux1 {
    root: String,
    cfg: Flux1Config,
    variant: String,
    dit: Flux1Model,
    vae_cfg: VaeConfig,
    hw: (u32, u32),
    /// Where each part landed - `"te"` is read by [`Flux1::clip_l`]/[`Flux1::t5_xxl`]
    /// so the text towers build on whichever card [`plan_flux1`] gave them,
    /// never assumed to share the DiT's.
    homes: Homes,
}

/// Bytes the DiT's own weights occupy at `precision`, closed-form from
/// [`Flux1Config`] (no per-tensor manifest exists for this crate, unlike
/// `flux2::pipeline::dit_bytes`'s `tensor_manifest()` route) - a double block
/// carries independent img/txt weights (`qkv` + `proj` + a 2-layer MLP each),
/// a single block fuses `qkv+mlp_in` and `proj+mlp_out` into shared linears.
/// Only 2-D linears quantize under [`Precision::Int8`]; norm/mod tables stay
/// f32 in both tiers, same rule `flux2::pipeline::weight_bytes` uses.
///
/// Pinned by `dit_weight_bytes_matches_measured_footprint` against the
/// measured fp32 footprint (`AGENTS.md`/`dit_parity.rs`: ~11.9 B params ≈
/// 47.6 GiB) - a PLACEMENT INPUT, not claimed exact.
fn dit_weight_bytes(cfg: &Flux1Config, precision: Precision) -> u64 {
    let d = cfg.hidden as u64;
    let mlp = cfg.mlp_hidden() as u64;
    let lin = |rows: u64, cols: u64| -> u64 {
        let n = rows * cols;
        let w = if precision == Precision::Int8 { 1 } else { 4 };
        n * w + if precision == Precision::Int8 { n / 32 * 4 } else { 0 } // group-32 scales
    };
    // img + txt halves: qkv, proj, 2-layer MLP, and each stream's own
    // `Modulation(dim, double=true)` (shift/scale/gate for BOTH the attn and
    // mlp sub-layers = a [6d, d] linear) - omitting this term is what made an
    // earlier version of this function undercount by 32% against the
    // measured ~11.9 B params (see `dit_weight_bytes_matches_measured_footprint`).
    let double_block = 2 * (lin(3 * d, d) + lin(d, d) + lin(mlp, d) + lin(d, mlp) + lin(6 * d, d));
    // fused qkv+mlp_in, fused proj+mlp_out, and one `Modulation(dim,
    // double=false)` ([3d, d] - single blocks share one modulation output
    // across attn and mlp).
    let single_block = lin(3 * d + mlp, d) + lin(d, d + mlp) + lin(3 * d, d);
    let boundary = lin(d, cfg.in_channels as u64) // img_in
        + lin(d, cfg.context_in_dim as u64) // txt_in
        + lin(cfg.in_channels as u64, d) // final_layer.linear
        + lin(d, cfg.vec_in_dim as u64) // vector_in (2-layer MLP, approximated as one)
        + if cfg.guidance_embed { lin(d, 256) } else { 0 };
    cfg.depth_double as u64 * double_block + cfg.depth_single as u64 * single_block + boundary
}

/// The DiT's device scratch for one joint (image-only, text2image) sequence -
/// same shape as `flux2::pipeline::dit_scratch_bytes`, without FLUX.2's
/// reference-token/batch axes this pipeline never fills.
fn dit_scratch_bytes(cfg: &Flux1Config, precision: Precision, n_joint: u64) -> u64 {
    let d = cfg.hidden as u64;
    let mlp = cfg.mlp_hidden() as u64;
    let hd = cfg.head_dim() as u64;
    let attn_words = if precision == Precision::Int8 { 2 } else { 2 * cfg.n_heads as u64 * n_joint * n_joint };
    let f32_words = n_joint * (16 * d + 3 * mlp + 2 * cfg.in_channels as u64 + hd) + attn_words + 17 * d;
    let mut bytes = f32_words * 4;
    if precision == Precision::Int8 {
        bytes += n_joint * (4 + d + mlp);
    }
    bytes
}

/// The DiT part's total device footprint: its weights plus scratch, plus
/// whatever the caller folds in on top (`dit_extra_bytes` - PuLID's resident
/// `PulidCa` module, ~1.6 GB fp32, when this pipeline is conditioned).
pub fn dit_bytes(cfg: &Flux1Config, precision: Precision, n_joint: u64, dit_extra_bytes: u64) -> u64 {
    dit_weight_bytes(cfg, precision) + dit_scratch_bytes(cfg, precision, n_joint) + dit_extra_bytes
}

/// The hard cap on T5-XXL context length any `Flux1` built here can encode -
/// BFL's own released default, and the upper bound both `flux1::caps` and
/// `pulid::caps` set on their `max_len` param. [`Flux1::n_max`] reserves
/// exactly this many joint-token rows for text on top of the image tokens,
/// so a caller's param bound must never exceed this constant: raising one
/// without the other either wastes VRAM headroom or reintroduces the "sized
/// for N joint tokens, got M" panic this constant exists to prevent.
pub const MAX_TXT_LEN: u32 = 512;

/// T5-XXL's device footprint - always fp32 (`t5encoder` has no int8 tier) -
/// weights plus the per-layer scratch `T5Encoder::new_on` allocates (see
/// `crates/flux1/src/pipeline.rs`'s module docs on why this cannot share the
/// DiT's card at `max_len=512`).
fn te_bytes(max_len: u64) -> u64 {
    let c = T5Config::xxl();
    let (d, ff, layers) = (c.d_model as u64, c.d_ff as u64, c.layers as u64);
    let weights = (2 * d * ff + 4 * 64 * c.heads as u64 * d + 2 * d) * layers * 4 + d * c.vocab as u64 * 4;
    let scratch = layers * max_len * (11 * d + 4 * ff) * 4;
    weights + scratch
}

/// The pipeline stage the text towers are live in. T5-XXL and CLIP-L are
/// built inside [`Flux1::encode`], read once, and dropped when it returns -
/// BEFORE the denoise loop starts (`t5` is a local variable there, and this
/// struct's own doc says so: "the T5-XXL/CLIP-L towers and the VAE are built
/// for one encode/decode and dropped"). The DiT, by contrast, is resident for
/// the whole generation.
///
/// Declaring that difference is what stops an 18 GiB transient encoder taking
/// the only card that could hold the 14 GiB permanent DiT - the shape behind
/// the reported `cannot place 'dit' ... after placing te=gpu0` failure. See
/// `residency::plan::plan`'s ordering note.
const PHASE_ENCODE: u32 = 1;

/// The two parts this pipeline ever needs placed: the DiT (`dit`, including
/// any `dit_extra_bytes` a conditioning adapter adds) and T5-XXL (`te`,
/// CLIP-L rides with the DiT - ~0.5 GB, not worth its own part). `.apart()`
/// on both so the placer never puts them on one card when two are available;
/// with one card (or none installed) [`gpu_core::devices::place`] falls back
/// to the ambient device for everything, unchanged from today's behavior.
///
/// `te` declares [`PHASE_ENCODE`] and the DiT declares nothing (permanent),
/// which is their real relationship: they DO coexist in VRAM while `encode`
/// runs, so this is not a claim that the card is charged for only one of them
/// - it is what tells the placer which of the two can afford a slower tier.
pub fn part_needs(cfg: &Flux1Config, precision: Precision, n_joint: u64, dit_extra_bytes: u64) -> Vec<Need> {
    vec![
        Need::sized("dit", dit_bytes(cfg, precision, n_joint, dit_extra_bytes), 0).apart(),
        Need::sized("te", te_bytes(MAX_TXT_LEN as u64), 0).apart().phase(PHASE_ENCODE),
    ]
}

/// Ask the installed placement policy where the DiT and T5-XXL go. A caller
/// conditioning the DiT (PuLID) folds its own extra device bytes in via
/// `dit_extra_bytes` so the plan prices what will actually be built, not a
/// bare FLUX.1 DiT.
pub fn plan_flux1(cfg: &Flux1Config, precision: Precision, n_joint: u64, dit_extra_bytes: u64) -> Result<Homes, String> {
    gpu_core::devices::place(&part_needs(cfg, precision, n_joint, dit_extra_bytes))
}

/// `BRAIN_FLUX1_TE_DEVICE` overrides the automatic T5-XXL/CLIP-L placement -
/// the same escape hatch `BRAIN_FLUX2_TE_DEVICE`/`wan`'s T5 selector give an
/// operator when the automatic plan doesn't fit some other box.
fn te_device_override() -> Result<Option<gpu_core::devices::Home>, String> {
    use gpu_core::devices::Home;
    match std::env::var("BRAIN_FLUX1_TE_DEVICE") {
        Err(_) => Ok(None),
        Ok(s) if s == "cpu" => Ok(Some(Home::Cpu)),
        Ok(s) => {
            let i: u32 = s.strip_prefix("gpu").unwrap_or(&s).parse().map_err(|_| format!("flux1: BRAIN_FLUX1_TE_DEVICE={s:?} - expected cpu or gpu<N>"))?;
            Ok(Some(Home::Gpu(i)))
        }
    }
}

/// Run `f` with the ambient device scoped to `home` - the `Home`-typed
/// sibling of `Homes::run` (which resolves by NAME out of one `Homes`; this
/// resolves a `Home` this function already picked, honoring
/// `BRAIN_FLUX1_TE_DEVICE` over the plan). A `Home::Cpu` part builds on the
/// CPU backend, same as `Homes::run` - it used to run UNSCOPED, so a T5-XXL
/// "placed on the host tier" (or an operator's explicit
/// `BRAIN_FLUX1_TE_DEVICE=cpu`) still allocated ~18 GiB on whatever card was
/// ambient.
fn run_on_home<R>(home: gpu_core::devices::Home, f: impl FnOnce() -> R) -> Result<R, String> {
    match home {
        gpu_core::devices::Home::Gpu(i) if !gpu_core::devices::gpus().is_empty() => gpu_core::devices::with_gpu(i, f),
        gpu_core::devices::Home::Cpu => Ok(gpu_core::devices::with_host_tier(f)),
        _ => Ok(f()),
    }
}

/// Inverse of the DiT's token patchify: predicted/denoised tokens
/// `[lh*lw, 4c]` (row-major, matching [`position_ids`]'s (h, w) order) back
/// to the VAE's `[c, h, w]` latent mean, undoing FLUX.1's own affine
/// normalization on the way. `h, w` are the UNPACKED (VAE-latent) dims.
///
/// Released `ae.safetensors`/diffusers `vae/` carries no `bn.running_{mean,
/// var}` (that is FLUX.2's scheme, `vae::latent::pack`/`unpack` is wrong
/// here); just a scalar `(shift_factor, scaling_factor)` affine, the inverse
/// of BFL's `AutoEncoder.encode`: `z = (posterior.mean - shift) * scale`.
/// A `pack_tokens` (forward direction) has no caller yet, per the module
/// docs' "not yet in scope" list, and is deliberately not written until
/// img2img needs it, rather than shipped untested.
fn unpack_tokens(tokens: &[f32], c: usize, h: usize, w: usize, shift: f32, scale: f32) -> Vec<f32> {
    assert!(h.is_multiple_of(2) && w.is_multiple_of(2), "flux1: latent {h}x{w} must be even");
    let (lh, lw) = (h / 2, w / 2);
    let mut out = vec![0.0f32; c * h * w];
    for ci in 0..c {
        for pi in 0..2 {
            for pj in 0..2 {
                let oc = ci * 4 + pi * 2 + pj;
                for y in 0..lh {
                    for x in 0..lw {
                        out[(ci * h + 2 * y + pi) * w + 2 * x + pj] =
                            tokens[(y * lw + x) * (4 * c) + oc] / scale + shift;
                    }
                }
            }
        }
    }
    out
}

/// BFL's `linspace(1, 0, steps+1)`, optionally shifted by the linear
/// `calculate_shift` mu - see the module docs for why this is not
/// `diffusion::scheduler::klein_sigmas`/`empirical_mu`.
fn flux1_sigmas(steps: usize, image_seq_len: usize, dynamic_shift: bool) -> Vec<f32> {
    let base: Vec<f32> = (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
    if !dynamic_shift {
        return base;
    }
    // BFL `get_lin_function(base_seq_len=256, max_seq_len=4096, base_shift=0.5,
    // max_shift=1.15)` - the exact constants `flux/sampling.py` and every
    // diffusers `FluxPipeline` use for `dev`/`kontext-dev`.
    let (base_seq_len, max_seq_len, base_shift, max_shift) = (256.0f32, 4096.0f32, 0.5f32, 1.15f32);
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    let b = base_shift - m * base_seq_len;
    let mu = image_seq_len as f32 * m + b;
    time_shift_exponential(mu, &base)
}

fn read_json(p: &Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("flux1: reading {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("flux1: parsing {}: {e}", p.display()))
}

/// Read the DiT weights from a diffusers `transformer/` dir, a BFL
/// single-file safetensors, or a GGUF, onto the canonical BFL names - the
/// same probe `flux2::pipeline::read_dit_tensors` uses.
fn read_dit_tensors(path: &str, cfg: &Flux1Config) -> Result<crate::import::Tensors, String> {
    let p = Path::new(path);
    if p.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("flux1: {path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("flux1: no .safetensors under {path}"));
        }
        let mut ts = Vec::new();
        for f in files {
            ts.extend(checkpoint::safetensors::read(f.to_str().ok_or("flux1: non-UTF8 path")?)?);
        }
        crate::import_diffusers(ts, cfg)
    } else if p.extension().is_some_and(|x| x == "gguf") {
        crate::import_bfl(checkpoint::gguf::read(path)?, cfg)
    } else {
        let ts = checkpoint::safetensors::read(path)?;
        if ts.iter().any(|t| t.name.starts_with("transformer_blocks.")) {
            crate::import_diffusers(ts, cfg)
        } else {
            crate::import_bfl(ts, cfg)
        }
    }
}

impl Flux1 {
    /// Load from a released FLUX.1 directory (the HF `black-forest-labs/
    /// FLUX.1-*` layout: `transformer/`, `vae/`, `text_encoder/`+`tokenizer/`
    /// for CLIP-L, `text_encoder_2/`+`tokenizer_2/` for T5-XXL - the same
    /// `text_encoder_2/`+`tokenizer_2/` shape `t5encoder::caps`'s `flux_xxl`
    /// variant already expects, since it is the same checkpoint family).
    ///
    /// `variant` is `dev` | `kontext-dev` | `schnell`
    /// ([`Flux1Config::from_name`]). `h`/`w` are the generated size: the DiT's
    /// max joint-token budget is sized for exactly this latent, so a
    /// different size needs a different `Flux1`.
    pub fn load(root: &str, variant: &str, h: u32, w: u32) -> Result<Flux1, String> {
        Flux1::load_with(root, variant, h, w, Precision::F32)
    }

    /// [`Flux1::load`] at a numeric tier - `precision` only governs the DiT
    /// (T5-XXL/CLIP-L have no int8 tier, per [`te_bytes`]'s doc). Places the
    /// DiT and T5-XXL automatically via [`plan_flux1`] (`.apart()`, so a
    /// two-card box never stacks both) and builds the DiT from `flux1::KERNELS`
    /// on its own `Gpu` - the shared entry point for a caller (PuLID) that
    /// needs its own kernel list on the SAME handle is [`Flux1::load_shared`].
    pub fn load_with(root: &str, variant: &str, h: u32, w: u32, precision: Precision) -> Result<Flux1, String> {
        // Plan AND build inside the retry: a lost VRAM race is only
        // recoverable if the retry re-plans against freshly probed capacity
        // rather than repeating the placement that just failed. See
        // `gpu_core::devices::build_with_retry` for exactly what this covers
        // (a transient neighbouring process) and what it does not (a driver
        // `abort`, or a true cross-process reservation).
        gpu_core::devices::build_with_retry("flux1", || {
            let cfg = Flux1Config::from_name(variant)?;
            let n_max = Flux1::n_max(h, w)?;
            let homes = plan_flux1(&cfg, precision, n_max as u64, 0)?;
            eprintln!("flux1: placement {}", homes.describe());
            let gpu = homes.run("dit", || Gpu::new(KERNELS))?;
            Flux1::build(root, variant, h, w, cfg, n_max, gpu, precision, homes)
        })
    }

    /// [`Flux1::load_with`] for a caller that owns the DiT's placement and
    /// kernel list itself (PuLID's `Bundle::load`, sharing one `Gpu` built
    /// from `pulid::joint_kernels()` between the DiT and `PulidCa` - see
    /// `crates/flux1/src/inject.rs`'s "same `Gpu` handle" contract). `gpu` MUST
    /// have been built under `homes.run("dit", ...)` (or the caller's own
    /// equivalent scoping) so its device matches what `homes` planned.
    pub fn load_shared(root: &str, variant: &str, h: u32, w: u32, gpu: Gpu, precision: Precision, homes: Homes) -> Result<Flux1, String> {
        let cfg = Flux1Config::from_name(variant)?;
        let n_max = Flux1::n_max(h, w)?;
        Flux1::build(root, variant, h, w, cfg, n_max, gpu, precision, homes)
    }

    fn n_max(h: u32, w: u32) -> Result<u32, String> {
        let scale = 16u32; // VAE downscale 8 * DiT 2x2 patchify
        if !h.is_multiple_of(scale) || !w.is_multiple_of(scale) {
            return Err(format!("flux1: {w}x{h} is not a multiple of {scale}"));
        }
        let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
        Ok((lh * lw) as u32 + MAX_TXT_LEN)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(root: &str, variant: &str, h: u32, w: u32, cfg: Flux1Config, n_max: u32, gpu: Gpu, precision: Precision, homes: Homes) -> Result<Flux1, String> {
        let r = Path::new(root);
        let dit_dir = r.join("transformer");
        let dit_path = if dit_dir.exists() { dit_dir } else { r.to_path_buf() };
        let ts = read_dit_tensors(dit_path.to_str().ok_or("flux1: non-UTF8 transformer path")?, &cfg)?;
        // `n_max` (image tokens + MAX_TXT_LEN) is sized for the WORST case
        // this pipeline ever calls it with - matches `Flux1Model::new`'s own
        // doc: "at most n_max joint tokens (txt + image + reference)".
        // Text2image submits `ctx` and `img_tokens` as separate arguments to
        // `forward`, which sums their row counts (`nt + ni`) and asserts it
        // against `n_max` - so `n_max` must cover BOTH, not image tokens
        // alone (an earlier version of this function omitted the text term
        // entirely, panicking "sized for 1024 joint tokens, got 1536" on the
        // first real T5-XXL-conditioned forward).
        let dit = Flux1Model::new_with(&cfg, &ts, gpu, n_max, precision);

        let vae_json = r.join("vae").join("config.json");
        let vae_cfg = if vae_json.exists() {
            VaeConfig::from_json(&read_json(&vae_json)?)
        } else {
            // BFL's released `ae.safetensors` (as opposed to the diffusers
            // `vae/config.json` release layout) ships no config at all.
            // `VaeConfig::from_json`'s own fallbacks already ARE FLUX.1's
            // architecture (16 latent channels, [128,256,512,512], the
            // published scaling_factor/shift_factor - this crate's own doc
            // names FLUX.1 as the reference case) with ONE exception:
            // `use_quant_conv`/`use_post_quant_conv` default true (the
            // SDXL/SD1.x-family default) because a real config.json only
            // carries the keys it OVERRIDES - but FLUX.1/Z-Image's real
            // released configs explicitly set both false, so an empty json
            // here must too.
            VaeConfig {
                use_quant_conv: false,
                use_post_quant_conv: false,
                ..VaeConfig::from_json(&serde_json::json!({}))
            }
        };

        Ok(Flux1 { root: root.into(), cfg, variant: variant.into(), dit, vae_cfg, hw: (h, w), homes })
    }

    /// T5-XXL/CLIP-L's home, honoring `BRAIN_FLUX1_TE_DEVICE` over the plan -
    /// resolved once per call rather than cached, since it is cheap and an
    /// operator may change the env var between requests.
    fn te_home(&self) -> Result<gpu_core::devices::Home, String> {
        Ok(te_device_override()?.unwrap_or_else(|| self.homes.of("te").unwrap_or(gpu_core::devices::Home::Cpu)))
    }

    fn clip_l(&self) -> Result<ClipText, String> {
        let cfg = ClipTextConfig::clip_l();
        let t = clip::import::read_text_encoder(&Path::new(&self.root).join("text_encoder"))?;
        let init = clip::import::import_text(t, &cfg)?;
        let map: std::collections::HashMap<String, Vec<f32>> =
            init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        // CLIP-L is small (~0.5 GB) - rides on whichever card T5-XXL landed on
        // rather than costing its own placement slot.
        run_on_home(self.te_home()?, || ClipText::new_on(Gpu::new(clip::model::TEXT_PIPELINES), cfg, 1, 77, &map))
    }

    fn t5_xxl(&self, max_len: usize) -> Result<T5Encoder, String> {
        let cfg = T5Config::xxl();
        let dir = Path::new(&self.root).join("text_encoder_2");
        let tensors = t5encoder::import::read_encoder(&dir)?;
        let init: std::collections::HashMap<String, Vec<f32>> =
            t5encoder::import::import_hf(tensors, &cfg)?.into_iter().map(|(k, (_, d))| (k, d)).collect();
        run_on_home(self.te_home()?, || T5Encoder::new_on(Gpu::new(t5encoder::model::PIPELINES), cfg, 1, max_len as u32, &init))
    }

    /// `(pooled[768], ctx[max_len*4096])` - CLIP-L's pooled EOS row (it does
    /// not project; only OpenCLIP-bigG does) and T5-XXL's unmasked context
    /// (FLUX passes no `attention_mask`, so right-pad positions are
    /// attended as ordinary keys - `t5encoder::caps`'s `flux_xxl` variant
    /// documents the same choice).
    fn encode(&self, prompt: &str, max_len: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let clip_tok = data::clip_bpe::ClipBpe::from_dir(&Path::new(&self.root).join("tokenizer"))
            .map_err(|e| format!("flux1: CLIP tokenizer: {e}"))?;
        let clip = self.clip_l()?;
        clip.set_tokens(&clip_tok.encode_with_context(prompt, 77).ids);
        clip.forward();
        let pooled = clip.read_pooled();
        drop(clip);

        let t5_tok = UnigramTokenizer::from_dir(Path::new(&self.root).join("tokenizer_2").to_str().ok_or("flux1: non-UTF8 path")?)
            .map_err(|e| format!("flux1: T5 tokenizer: {e}"))?;
        let (ids, _mask) = t5_tok.encode_padded(prompt, max_len);
        let t5 = self.t5_xxl(max_len)?;
        t5.set_tokens(&ids);
        t5.forward();
        let ctx = t5.read_hidden(); // unmasked: no pad-row zeroing (see `encode`'s doc)

        Ok((pooled, ctx))
    }

    /// Generate one image. Returns HWC RGB in `[0,1]`.
    pub fn generate(&self, prompt: &str, o: &GenerateOptions, max_len: usize) -> Result<Vec<f32>, String> {
        self.generate_injected(prompt, None, o, max_len, None)
    }

    /// [`Flux1::generate`] with every DiT step routed through
    /// `Flux1Model::forward_injected` when `inject` is `Some` - the seam
    /// `pulid::caps` uses to condition on an identity, and `crates/flux1`'s
    /// own `inject::BlockInject` trait so this needs no dependency on
    /// `pulid` (or any other adapter crate) to exist.
    ///
    /// `negative_prompt` only has an effect when `o.true_cfg` is also
    /// `Some` (see [`GenerateOptions::true_cfg`]'s doc) - passing one without
    /// the other is accepted, not an error, and behaves as if neither were
    /// given (today's single-forward-per-step cost).
    pub fn generate_injected(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        o: &GenerateOptions,
        max_len: usize,
        inject: Option<&dyn crate::inject::BlockInject>,
    ) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
        let n_gen = lh * lw;

        let (pooled, ctx) = self.encode(prompt, max_len)?;
        // Only pay for the second tower encode when true CFG is actually on.
        let neg = match (negative_prompt, o.true_cfg) {
            (Some(np), Some(_)) => Some(self.encode(np, max_len)?),
            _ => None,
        };

        let dynamic_shift = self.variant != "schnell";
        let steps = o.steps.unwrap_or(if self.variant == "schnell" { 4 } else { 50 });
        let sigmas = flux1_sigmas(steps, n_gen, dynamic_shift);

        let ids = position_ids(max_len, lh, lw, &[]);
        let mut lat = model::hostmath::gaussian(n_gen * self.cfg.in_channels, o.seed);

        for i in 0..steps {
            let t = sigmas[i];
            let pos_pred = match inject {
                Some(inj) if i >= o.start_step => self.dit.forward_injected(&lat, &ctx, &pooled, t, o.guidance, &ids, n_gen, inj),
                _ => self.dit.forward(&lat, &ctx, &pooled, t, o.guidance, &ids, n_gen),
            };
            // True CFG: a SECOND, un-injected forward on the negative
            // prompt's own conditioning - "unconditional" here means no
            // identity injection at all, not a blank/zero prompt, matching
            // upstream's own `id=None` unconditional branch. Doubles this
            // step's DiT cost, which is why it is gated on `cfg.start_step`
            // rather than applied unconditionally.
            let pred = match (&neg, o.true_cfg) {
                (Some((neg_pooled, neg_ctx)), Some(cfg)) if i >= cfg.start_step => {
                    let neg_pred = self.dit.forward(&lat, neg_ctx, neg_pooled, t, o.guidance, &ids, n_gen);
                    neg_pred.iter().zip(&pos_pred).map(|(n, p)| n + cfg.scale * (p - n)).collect()
                }
                _ => pos_pred,
            };
            let dt = sigmas[i + 1] - t;
            for (x, v) in lat.iter_mut().zip(&pred) {
                *x += dt * v;
            }
        }

        // `in_channels = 16*2*2`; unpack back to the VAE's [16, h_lat, w_lat]
        // latent, undo the affine, then decode.
        let c = self.cfg.in_channels / 4;
        let unpacked = unpack_tokens(&lat, c, lh * 2, lw * 2, self.vae_cfg.shift_factor, self.vae_cfg.scaling_factor);
        let vt = read_any_safetensors(&Path::new(&self.root).join("vae"))?;
        let vmap: vae::blocks::Tensors = vt.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
        let vdev = std::env::var("BRAIN_FLUX1_VAE_DEVICE").unwrap_or_else(|_| "cpu".into());
        let dec = VaeDecoder::from_diffusers(self.vae_cfg.clone(), &vmap, (lh * 2) as u32, (lw * 2) as u32, Some(&vdev));
        let chw = dec.decode(&unpacked);
        // diffusers maps the decoder's [-1,1] output to [0,1].
        let rgb: Vec<f32> = chw.iter().map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)).collect();
        Ok(imaging::pixels::chw_to_hwc(&rgb, 3, h as usize, w as usize))
    }
}

fn read_any_safetensors(dir: &Path) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("flux1: reading {}: {e}", dir.display()))?;
    let mut files: Vec<std::path::PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("flux1: no *.safetensors under {}", dir.display()));
    }
    let mut out = Vec::new();
    for f in files {
        out.extend(checkpoint::safetensors::read(f.to_str().ok_or("flux1: non-UTF8 path")?)?);
    }
    Ok(out)
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    /// `dit_weight_bytes` is a closed-form PLACEMENT INPUT, not a real
    /// tensor manifest (flux1 has none) - pinned against the measured fp32
    /// footprint (`AGENTS.md`/`dit_parity.rs`: ~11.9 B params ≈ 47.6 GiB) so
    /// a future architecture-formula bug (like the missing modulation
    /// linears this test would have caught) shows up as a failing test, not
    /// as a placement that silently doesn't fit.
    #[test]
    fn dit_weight_bytes_matches_measured_footprint() {
        let cfg = Flux1Config::dev();
        let gib = dit_weight_bytes(&cfg, Precision::F32) as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((40.0..55.0).contains(&gib), "dit_weight_bytes = {gib:.2} GiB, expected ~47.6 GiB (measured)");
    }

    #[test]
    fn int8_dit_weight_bytes_is_smaller_than_fp32() {
        let cfg = Flux1Config::dev();
        let f32_bytes = dit_weight_bytes(&cfg, Precision::F32);
        let i8_bytes = dit_weight_bytes(&cfg, Precision::Int8);
        assert!(i8_bytes < f32_bytes / 3, "int8 ({i8_bytes}) should be well under 1/3 of fp32 ({f32_bytes})");
    }

    /// `part_needs` keeps the DiT and T5-XXL `.apart()` so a 2-card box never
    /// stacks a ~17 GB DiT and a ~21 GB T5-XXL on one card.
    #[test]
    fn dit_and_te_are_placed_apart() {
        let cfg = Flux1Config::dev();
        let needs = part_needs(&cfg, Precision::Int8, 1024, 0);
        assert_eq!(needs.len(), 2);
        assert!(needs.iter().all(|n| n.affinity == gpu_core::devices::Affinity::Apart));
    }
}

