// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements structure-preserving video-to-video
// conditioning for diffusion transformers for its clients. If your team needs
// expertise in controllable video generation then you can procure our services
// by sending an email to info@swedishembedded.com.

//! IC-LoRA ("In-Context LoRA") reference-video conditioning.
//!
//! Port of `ltx_core.conditioning.types.reference_video_cond.
//! VideoConditionByReferenceLatent` plus the attention-strength wrapper
//! (`ltx_core.conditioning.types.attention_strength_wrapper`) and
//! `ltx_pipelines.iclora_utils.downsample_mask_video_to_latent`.
//!
//! # What this is, and what it is NOT
//!
//! An IC-LoRA conditions generation on a *reference video* that is positionally
//! aligned with the clip being generated: the reference's latent tokens are
//! appended to the sequence as CLEAN, frozen tokens carrying the reference's own
//! RoPE position bounds re-expressed in the target's coordinate frame. The model
//! then attends across both halves.
//!
//! The base checkpoint does **not** know what to do with those extra tokens.
//! `reference_video_cond.py`'s own docstring is explicit: "IC-LoRAs are trained
//! by concatenating reference (control signal) and target tokens, *learning to
//! attend across both*. This class replicates that setup at inference." The
//! attending is the ADAPTER's contribution, not the base model's. Appending
//! reference tokens without a matching IC-LoRA loaded is not a weaker control
//! signal, it is an out-of-distribution sequence - so this module is the
//! plumbing for an adapter, and is only meaningful with one applied.
//!
//! `IC` is `In-Context`, from the reference's own config header
//! (`packages/ltx-trainer/configs/v2v_ic_lora.yaml`) and model cards. It is not
//! an identity mechanism: nothing in this path takes a face, a portrait, or any
//! per-subject embedding. Identity, if wanted, has to come from the prompt or
//! from a separately trained subject LoRA.
//!
//! # Reference downscale / temporal scale
//!
//! An IC-LoRA may be trained with a reference SMALLER than the target, to spend
//! fewer tokens on it: `downscale_factor` = target/reference spatial ratio,
//! `temporal_scale_factor` = the temporal one. Both are recorded in the
//! adapter's own safetensors metadata (`reference_downscale_factor`,
//! `reference_temporal_scale_factor`) and MUST match training, because they are
//! what maps reference positions onto target positions. Scaling the spatial axes
//! stretches the reference grid over the target's extent; the temporal branch
//! re-spaces the reference at `target_fps / S` and then translates it so the
//! reference's LAST patch ends with the target's last patch, clamping the causal
//! first patch's negative start back to zero.
//!
//! # Attention mask: the per-region control dial
//!
//! `ConditioningItemAttentionStrengthWrapper` scales how strongly each reference
//! token exchanges attention with the noisy tokens. The reference materialises a
//! dense `(N+M, N+M)` matrix, but that matrix is fully determined by the
//! `M`-vector of per-token weights plus a fixed block structure
//! (`mask_utils.build_attention_mask`'s own diagram): identity-ish `1`s inside
//! each group, the cross-mask on both noisy<->new blocks, zeros between distinct
//! reference groups. This module therefore stores the `M` values and rebuilds
//! the dense form only where a test demands it - `(N+M)^2` is quadratic in the
//! token count and would dwarf the latents themselves at real resolutions. The
//! golden asserts the two forms are EXACTLY equal, which is what licenses that.
//!
//! A weight of `1.0` means "this region of the reference constrains the output
//! fully"; `0.0` means "ignore the reference here, generate freely from the
//! prompt". That is the honest mechanism behind pinning WHICH part of a frame a
//! control signal governs.

use crate::pipeline::{real_pixel_positions, VAE_SPATIAL_SCALE, VAE_TEMPORAL_SCALE};

/// Geometry and content for one appended reference-video conditioning block.
///
/// Layout matches this crate's existing image-conditioning builder: positions
/// are `[3, T, 2]` axis-major, per-token masks are `[T]`, content is
/// `[T, channels]` token-major.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceVideoConditioning {
    /// `[3, base_t + m, 2]` - the caller's base positions verbatim on
    /// `[0, base_t)`, the reference's remapped bounds after it.
    pub positions: Vec<f32>,
    /// `[base_t + m]` - the caller's base mask verbatim, then zeros: reference
    /// tokens are never keyframe-marked (`extend_keyframes_mask(marked=false)`).
    pub keyframes_mask: Vec<f32>,
    /// `[base_t + m]` - `1.0` (denoise fully) on the base range, `1 - strength`
    /// on the appended range (`strength = 1.0` freezes the reference).
    pub denoise_mask: Vec<f32>,
    /// `[(base_t + m) * channels]` - zeros on the base range (never read there),
    /// the reference's patchified latent after it.
    pub clean: Vec<f32>,
    /// `[m]` per-token attention weights in `[0, 1]`, or `None` for full
    /// attention. See the module doc for why this is not the dense matrix.
    pub cross_mask: Option<Vec<f32>>,
}

/// `[3, f*h*w, 2]` RoPE position bounds for a reference video latent, expressed
/// in the TARGET clip's coordinate frame.
///
/// Ports `VideoConditionByReferenceLatent::apply_to`'s position block exactly:
/// `get_pixel_coords(..., causal_fix=True)`, then `/= fps / S`, then (for
/// `S != 1`) `clamp(x - (S - 1) / fps, min=0)`, then `*= downscale` on the two
/// spatial axes. [`real_pixel_positions`] already is `get_pixel_coords` followed
/// by a division of the time axis by its `fps` argument, so passing `fps / S`
/// performs the reference's own `/= fps / S` in one step.
///
/// # Panics
/// If `fps` is not finite and positive, or either scale factor is zero.
pub fn reference_video_positions(
    ref_f: usize,
    ref_h: usize,
    ref_w: usize,
    fps: f64,
    downscale_factor: usize,
    temporal_scale_factor: usize,
) -> Vec<f32> {
    assert!(fps.is_finite() && fps > 0.0, "reference_video_positions: fps must be finite and positive, got {fps}");
    assert!(downscale_factor >= 1, "reference_video_positions: downscale_factor must be >= 1");
    assert!(temporal_scale_factor >= 1, "reference_video_positions: temporal_scale_factor must be >= 1");

    let s = temporal_scale_factor as f64;
    let mut out = real_pixel_positions(ref_f, ref_h, ref_w, fps / s);
    let t = ref_f * ref_h * ref_w;

    if temporal_scale_factor != 1 {
        // `t_target` is the target's own first-token time END = 1/target_fps
        // (`latent_state.positions[:, 0, 0:1, 1:2]`), so the shift is
        // `(S - 1) / fps`. Applied to BOTH bounds, then clamped at zero.
        let shift = (s - 1.0) / fps;
        for v in out[..t * 2].iter_mut() {
            *v = ((*v as f64 - shift).max(0.0)) as f32;
        }
    }
    if downscale_factor != 1 {
        let d = downscale_factor as f32;
        for v in out[t * 2..].iter_mut() {
            *v *= d;
        }
    }
    out
}

/// Area-average a pixel-space `[f_pix, h_pix, w_pix]` mask down to `[lat_f *
/// lat_h * lat_w]` latent-token weights.
///
/// Ports `ltx_pipelines.iclora_utils.downsample_mask_video_to_latent`: bilinear
/// `mode="area"` spatially (an exact box average, which for these integer
/// ratios is a plain mean over each cell), then the causal temporal split - the
/// first latent frame takes the first downsampled pixel frame ALONE (the VAE is
/// causal, so its first temporal latent frame covers one pixel frame while the
/// rest cover `t`), and each later latent frame averages its own `t`-frame run.
///
/// # Panics
/// If `mask` is not `f_pix * h_pix * w_pix` long, or the frame counts are
/// incompatible (`(f_pix - 1)` must divide evenly by `(lat_f - 1)`, the
/// reference's own assertion).
pub fn downsample_mask_to_latent(
    mask: &[f32],
    f_pix: usize,
    h_pix: usize,
    w_pix: usize,
    lat_f: usize,
    lat_h: usize,
    lat_w: usize,
) -> Vec<f32> {
    assert_eq!(mask.len(), f_pix * h_pix * w_pix, "downsample_mask_to_latent: mask has {} values, expected {}", mask.len(), f_pix * h_pix * w_pix);
    assert!(lat_f >= 1 && lat_h >= 1 && lat_w >= 1, "downsample_mask_to_latent: latent dims must be >= 1");
    assert!(h_pix >= lat_h && w_pix >= lat_w, "downsample_mask_to_latent: cannot upsample ({h_pix}x{w_pix} -> {lat_h}x{lat_w})");

    // Spatial area-average, per pixel frame: `[f_pix, lat_h, lat_w]`.
    let mut spatial = vec![0f32; f_pix * lat_h * lat_w];
    for f in 0..f_pix {
        for oh in 0..lat_h {
            // `F.interpolate(mode="area")` = adaptive average pooling: cell
            // `i` spans `[floor(i*H/O), ceil((i+1)*H/O))`.
            let (h0, h1) = (oh * h_pix / lat_h, ((oh + 1) * h_pix).div_ceil(lat_h));
            for ow in 0..lat_w {
                let (w0, w1) = (ow * w_pix / lat_w, ((ow + 1) * w_pix).div_ceil(lat_w));
                let mut acc = 0f64;
                for hh in h0..h1 {
                    for ww in w0..w1 {
                        acc += mask[(f * h_pix + hh) * w_pix + ww] as f64;
                    }
                }
                spatial[(f * lat_h + oh) * lat_w + ow] = (acc / (((h1 - h0) * (w1 - w0)) as f64)) as f32;
            }
        }
    }

    let plane = lat_h * lat_w;
    let mut out = vec![0f32; lat_f * plane];
    // First latent frame is the first pixel frame alone (causal VAE).
    out[..plane].copy_from_slice(&spatial[..plane]);
    if f_pix > 1 && lat_f > 1 {
        assert_eq!((f_pix - 1) % (lat_f - 1), 0, "downsample_mask_to_latent: pixel frames ({f_pix}) not compatible with latent frames ({lat_f}): (f_pix - 1) must divide by (lat_f - 1)");
        let t = (f_pix - 1) / (lat_f - 1);
        for lf in 1..lat_f {
            for p in 0..plane {
                let mut acc = 0f64;
                for k in 0..t {
                    acc += spatial[(1 + (lf - 1) * t + k) * plane + p] as f64;
                }
                out[lf * plane + p] = (acc / t as f64) as f32;
            }
        }
    }
    out
}

/// Append one reference-video conditioning block after `base_t` noisy tokens.
///
/// `ref_tokens` is the reference latent already patchified to `[m, channels]`
/// token-major (`m = ref_f * ref_h * ref_w`), i.e. what this crate's VAE encoder
/// produces run through [`crate::pipeline::chw_to_tc`].
///
/// `strength` is the reference's own freeze level: `1.0` keeps it perfectly
/// clean (the normal case), `0.0` would let it be denoised along with the
/// target. `cross_mask`, when given, is `[m]` per-token attention weights.
///
/// # Panics
/// On any shape disagreement between the arguments.
pub fn append_reference_video_conditioning(
    base_t: usize,
    base_positions: &[f32],
    base_keyframes_mask: &[f32],
    channels: usize,
    fps: f64,
    ref_dims: (usize, usize, usize),
    ref_tokens: &[f32],
    downscale_factor: usize,
    temporal_scale_factor: usize,
    strength: f32,
    cross_mask: Option<&[f32]>,
) -> ReferenceVideoConditioning {
    let (ref_f, ref_h, ref_w) = ref_dims;
    let m = ref_f * ref_h * ref_w;
    assert!(m > 0, "append_reference_video_conditioning: reference must be non-empty");
    assert_eq!(base_positions.len(), 3 * base_t * 2, "append_reference_video_conditioning: base_positions has {} values, expected {}", base_positions.len(), 3 * base_t * 2);
    assert_eq!(base_keyframes_mask.len(), base_t, "append_reference_video_conditioning: base_keyframes_mask has {} values, expected {base_t}", base_keyframes_mask.len());
    assert_eq!(ref_tokens.len(), m * channels, "append_reference_video_conditioning: ref_tokens has {} values, expected {}", ref_tokens.len(), m * channels);
    assert!((0.0..=1.0).contains(&strength), "append_reference_video_conditioning: strength must be in [0, 1], got {strength}");
    if let Some(cm) = cross_mask {
        assert_eq!(cm.len(), m, "append_reference_video_conditioning: cross_mask has {} values, expected {m}", cm.len());
        assert!(cm.iter().all(|v| (0.0..=1.0).contains(v)), "append_reference_video_conditioning: cross_mask values must be in [0, 1]");
    }

    let total_t = base_t + m;
    let ref_positions = reference_video_positions(ref_f, ref_h, ref_w, fps, downscale_factor, temporal_scale_factor);

    let mut positions = vec![0f32; 3 * total_t * 2];
    for axis in 0..3 {
        let dst = axis * total_t * 2;
        positions[dst..dst + base_t * 2].copy_from_slice(&base_positions[axis * base_t * 2..(axis + 1) * base_t * 2]);
        positions[dst + base_t * 2..dst + total_t * 2].copy_from_slice(&ref_positions[axis * m * 2..(axis + 1) * m * 2]);
    }

    let mut keyframes_mask = vec![0f32; total_t];
    keyframes_mask[..base_t].copy_from_slice(base_keyframes_mask);

    let mut denoise_mask = vec![1.0f32 - strength; total_t];
    denoise_mask[..base_t].fill(1.0);

    let mut clean = vec![0f32; total_t * channels];
    clean[base_t * channels..].copy_from_slice(ref_tokens);

    ReferenceVideoConditioning { positions, keyframes_mask, denoise_mask, clean, cross_mask: cross_mask.map(<[f32]>::to_vec) }
}

/// Rebuild the reference's dense `[(base_t + m)^2]` self-attention matrix from a
/// per-token cross-mask.
///
/// Ports `mask_utils.build_attention_mask` for the single-reference case this
/// crate appends: `1` inside the noisy block and inside the reference block, the
/// cross-mask on both noisy<->reference blocks. Provided so the factored form
/// can be gated against the reference's own dense output; the generation path
/// never materialises this.
///
/// # Panics
/// If `cross_mask` is not `m` long.
pub fn dense_attention_mask(base_t: usize, cross_mask: &[f32]) -> Vec<f32> {
    let m = cross_mask.len();
    let total = base_t + m;
    let mut out = vec![0f32; total * total];
    for r in 0..base_t {
        out[r * total..r * total + base_t].fill(1.0);
        out[r * total + base_t..(r + 1) * total].copy_from_slice(cross_mask);
    }
    for (j, &w) in cross_mask.iter().enumerate() {
        let r = base_t + j;
        out[r * total..r * total + base_t].fill(w);
        out[r * total + base_t..(r + 1) * total].fill(1.0);
    }
    out
}

/// Pixel-space extent a reference latent of `ref_dims` covers, for sizing the
// perf-number: the VAE's spatial/temporal downsample factors are architecture constants, not measured speedups
/// mask video a caller has to supply: the VAE's own `32x32` spatial and `8x`
/// causal temporal factors.
pub fn reference_pixel_extent(ref_dims: (usize, usize, usize)) -> (usize, usize, usize) {
    let (f, h, w) = ref_dims;
    (if f <= 1 { 1 } else { (f - 1) * VAE_TEMPORAL_SCALE + 1 }, h * VAE_SPATIAL_SCALE, w * VAE_SPATIAL_SCALE)
}
