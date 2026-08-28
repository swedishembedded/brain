// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements region-preserving masked conditioning for
// video diffusion transformers for its clients. If your team needs expertise
// in controllable video generation then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Masked conditioning - `ltx_core.conditioning.types.mask_cond.
//! VideoConditionByMask`.
//!
//! # What it is
//!
//! A per-token dial that says, for every position of the latent grid, whether
//! that position is GENERATED or CARRIED. Unlike [`crate::refcond`]'s IC-LoRA
//! path it appends no tokens, needs no adapter, and does not depend on a model
//! having learned to attend across a second half of the sequence: it is the
//! `clean_latent`/`denoise_mask` pair the base checkpoint already honours on
//! every step. Where it says "carry", the base model's own
//! `post_process_latent` re-pins the token to the supplied clean latent after
//! every step, so at `strength = 1.0` that region comes out **bit-exactly** the
//! latent that was handed in - not "structurally similar", identical.
//!
//! That is what makes it the mechanism for a character swap on LTX-2.5: mask
//! the background as conditioning and the character region as generated, and
//! the set, the camera move and the lighting survive the trip through the
//! denoiser untouched while the masked-out region is redrawn from the prompt.
//!
//! # Polarity - inverted from intuition, and the single most dangerous detail
//!
//! From the class's own docstring: **`mask = 1` is the CONDITIONING
//! position**. It receives the clean latent and is EXCLUDED from denoising
//! (`denoise_mask` set to `1 - strength`). **`mask = 0` is the GENERATED
//! position**: untouched, noised and denoised normally.
//!
//! So a character swap masks the **background at 1** and the **character
//! region at 0**. Every human-facing mask this repo consumes uses the opposite
//! convention (a tracker paints the SUBJECT), so the flip has exactly one
//! named home - [`MaskSeqPolarity::to_conditioning`] - and everything else in
//! this module speaks the reference's polarity.
//!
//! # The two blends, verbatim
//!
//! ```text
//! inv        = 1 - mask
//! clean      = clean * inv + tokens * mask
//! denoise    = denoise * inv + (1 - strength) * mask
//! ```
//!
//! Both terms of both blends matter and neither is redundant:
//!
//! * `clean * inv` only shows up when the state's clean latent was already
//!   non-zero - a long-form continuation window, or a second conditioning item
//!   applied first. On a plain text-to-video state it is all zeros and a port
//!   that dropped the term would look perfect.
//! * `denoise * inv` only shows up when the base denoise mask is not all ones,
//!   i.e. again when something conditioned the state first. Drop it and you get
//!   `1 - strength * mask`, which is right everywhere else.
//!
//! `tools/goldens/ltxv_maskcond_dump_reference.py` therefore dumps a case with
//! a non-zero initial latent and a case with
//! `VideoConditionByLatentIndex` applied first, so neither term can be dropped
//! silently.
//!
//! **A fractional mask is not a lerp between the two endpoints.** Feed
//! `mask = 0.5` at `strength = 1.0` and the reference gives `clean = 0.5 *
//! tokens` and `denoise = 0.5`, so [`noised_initial_latent`]'s
//! `lerp(clean, noise, denoise)` starts that token at `0.25 * tokens + 0.5 *
//! noise` - the mask enters the initial latent QUADRATICALLY. That is what the
//! reference does, it is a direct consequence of the class documenting its mask
//! as binary, and it is what soft mask edges (which is what any real
//! downsampled mask has) actually get. It is ported as-is.
//!
//! # Storage: `[N]`, never `[N, C]`
//!
//! The reference multiplies a `[B, N, 1]` mask against a `[B, N, C]` latent and
//! lets torch broadcast. Materialising that broadcast - one weight per token
//! per channel - costs `C` times the mask for zero information; at the real
// perf-number: 128 is the checkpoint's own latent-channel count, not a measured runtime ratio
//! checkpoint's 128 latent channels that is a 128x blow-up of the one tensor a
//! caller is most likely to keep several copies of. This module stores `[N]`
//! and broadcasts at the point of use, and the golden asserts the two forms are
//! EXACTLY equal, which is what licenses that. Same discipline, same reason, as
//! [`crate::refcond`]'s factored attention mask.
//!
//! The other blow-up on this path is the PIXEL mask. A 121-frame 1280x704 mask
//! sequence is 109 M samples - larger than the clip it masks - and it is only
//! ever read once, on the way to a `[F, H, W]` latent grid thousands of times
//! smaller. [`LatentMaskAccumulator`] therefore consumes it one frame at a
//! time and is gated bit-for-bit against the whole-buffer
//! [`crate::refcond::downsample_mask_to_latent`], so nothing is traded for the
//! memory.
//!
//! # Pixel space to latent space
//!
//! The mask arrives at source resolution and has to reach the latent grid,
// perf-number: the VAE's spatial/temporal downsample factors are architecture constants, not measured speedups
//! which the causal VAE reduces by **32x32 spatially and 8x temporally, except
//! for the first latent frame, which covers ONE pixel frame**. The reduction is
//! `ltx_pipelines.iclora_utils.downsample_mask_video_to_latent`, already ported
//! as [`crate::refcond::downsample_mask_to_latent`]: an area average spatially,
//! and temporally a MEAN over each latent frame's own run of pixel frames -
//! not `any`, not `all`, not `max`. Those alternatives are all plausible and
//! all wrong, and they differ from the mean only in the soft band around the
//! subject, which is exactly where a wrong answer looks almost right.

use std::path::Path;

/// One [`apply_video_condition_by_mask`] result - the two tensors
/// `VideoConditionByMask.apply_to` replaces on the latent state.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskConditioning {
    /// `[t * channels]` token-major clean latent: `clean * inv + tokens * m`.
    pub clean: Vec<f32>,
    /// `[t]` per-token denoise mask: `denoise * inv + (1 - strength) * m`.
    /// `1.0` denoises fully, `0.0` freezes the token at [`Self::clean`].
    pub denoise_mask: Vec<f32>,
}

/// `VideoConditionByMask::apply_to`, exactly.
///
/// `mask_tokens` is `[t]` in the reference's polarity (`1` = conditioning,
/// `0` = generated) and in TOKEN order. At LTX-2.5's `patch_size = 1` the
/// patchifier's `b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)` is a plain
/// C-order flatten of the `[F, H, W]` latent mask, so a latent mask laid out
/// frame-major/row-major IS the token vector with no permutation - asserted
/// against the live reference by the golden dumper, not assumed.
///
/// `cond_tokens` is `[t * channels]`, the conditioning latent already
/// patchified (`[C, F, H, W]` through [`crate::pipeline::chw_to_tc`]).
///
/// # Panics
/// On any shape disagreement, on a non-finite or out-of-range `strength`, or
/// on a mask value outside `[0, 1]`.
pub fn apply_video_condition_by_mask(
    base_clean: &[f32],
    base_denoise_mask: &[f32],
    cond_tokens: &[f32],
    mask_tokens: &[f32],
    channels: usize,
    strength: f32,
) -> MaskConditioning {
    let t = base_denoise_mask.len();
    assert!(channels > 0, "apply_video_condition_by_mask: channels must be non-zero");
    assert_eq!(base_clean.len(), t * channels, "apply_video_condition_by_mask: base_clean has {} values, expected {}", base_clean.len(), t * channels);
    assert_eq!(cond_tokens.len(), t * channels, "apply_video_condition_by_mask: cond_tokens has {} values, expected {}", cond_tokens.len(), t * channels);
    assert_eq!(mask_tokens.len(), t, "apply_video_condition_by_mask: mask_tokens has {} values, expected {t}", mask_tokens.len());
    assert!((0.0..=1.0).contains(&strength), "apply_video_condition_by_mask: strength must be in [0, 1], got {strength}");
    assert!(mask_tokens.iter().all(|v| (0.0..=1.0).contains(v)), "apply_video_condition_by_mask: mask values must be in [0, 1]");

    let mut clean = Vec::with_capacity(t * channels);
    let mut denoise_mask = Vec::with_capacity(t);
    // Walked as ROWS, never by index. The two tensors here live on different
    // axes - `[t]` for the mask, `[t, channels]` for the latents - and this is
    // the one function where confusing them produces a plausible picture
    // rather than a crash: an off-by-one on the token axis shifts the whole
    // conditioning region by a latent cell. Zipping `chunks_exact(channels)`
    // against the mask hands each token its channels as one slice, so the
    // correspondence is structural and there is no arithmetic to get wrong.
    for (((base_row, cond_row), &m), &d) in
        base_clean.chunks_exact(channels).zip(cond_tokens.chunks_exact(channels)).zip(mask_tokens).zip(base_denoise_mask)
    {
        let inv = 1.0 - m;
        denoise_mask.push(d * inv + (1.0 - strength) * m);
        clean.extend(base_row.iter().zip(cond_row).map(|(&b, &c)| b * inv + c * m));
    }
    MaskConditioning { clean, denoise_mask }
}

/// `ltx_core.components.noisers.GaussianNoiser.__call__`, for one already
/// conditioned state: `lerp(clean, lerp(latent, noise, noise_scale), denoise)`.
///
/// The reference runs the noiser AFTER every conditioning item, which is why
/// this is a separate step rather than folded into
/// [`apply_video_condition_by_mask`]. At `denoise = 0` (a `strength = 1.0`
/// conditioning position) the result is exactly `clean` - the bit-exactness
/// claim in this module's doc, and the reason a masked region survives the whole
/// trajectory rather than merely starting near it.
///
/// # Panics
/// On any shape disagreement.
pub fn noised_initial_latent(latent: &[f32], noise: &[f32], clean: &[f32], denoise_mask: &[f32], channels: usize, noise_scale: f32) -> Vec<f32> {
    let t = denoise_mask.len();
    assert_eq!(latent.len(), t * channels, "noised_initial_latent: latent has {} values, expected {}", latent.len(), t * channels);
    assert_eq!(noise.len(), t * channels, "noised_initial_latent: noise has {} values, expected {}", noise.len(), t * channels);
    assert_eq!(clean.len(), t * channels, "noised_initial_latent: clean has {} values, expected {}", clean.len(), t * channels);

    let mut out = Vec::with_capacity(t * channels);
    for (((lat_row, noise_row), clean_row), &d) in
        latent.chunks_exact(channels).zip(noise.chunks_exact(channels)).zip(clean.chunks_exact(channels)).zip(denoise_mask)
    {
        out.extend(lat_row.iter().zip(noise_row).zip(clean_row).map(|((&l, &n), &c)| {
            // `torch.lerp(a, b, w) = a + w * (b - a)`, in that order - the
            // reference's own two nested calls, kept nested so the rounding
            // matches rather than being algebraically flattened.
            let stepped = l + noise_scale * (n - l);
            c + d * (stepped - c)
        }));
    }
    out
}

// ===========================================================================
// Pixel-space mask sequence -> latent conditioning mask
// ===========================================================================

/// Streaming form of [`crate::refcond::downsample_mask_to_latent`]: consumes
/// one `[h_pix * w_pix]` pixel frame at a time and never holds the sequence.
///
/// Bit-for-bit identical to the whole-buffer function (gated in
/// `crates/ltxv/tests/maskcond_parity.rs`), because the accumulation order is
/// the same: an `f64` area average per pixel frame rounded to `f32`, then an
/// `f64` mean of those `f32` values over each latent frame's run, rounded to
/// `f32`. The causal split is the same too - latent frame 0 is the FIRST pixel
/// frame alone, every later latent frame the mean of its own `t`-frame run.
pub struct LatentMaskAccumulator {
    h_pix: usize,
    w_pix: usize,
    lat_h: usize,
    lat_w: usize,
    /// Pixel frames per latent frame after the first; `0` when `lat_f == 1`.
    run: usize,
    seen: usize,
    total_pix_frames: usize,
    acc: Vec<f64>,
    in_run: usize,
    out: Vec<f32>,
}

impl LatentMaskAccumulator {
    /// # Panics
    /// If any dimension is zero, if the latent grid is larger than the pixel
    /// grid, or if `(f_pix - 1)` is not divisible by `(lat_f - 1)` - the
    /// reference's own compatibility assertion.
    pub fn new(f_pix: usize, h_pix: usize, w_pix: usize, lat_f: usize, lat_h: usize, lat_w: usize) -> LatentMaskAccumulator {
        assert!(f_pix >= 1 && h_pix >= 1 && w_pix >= 1, "LatentMaskAccumulator: pixel dims must be >= 1");
        assert!(lat_f >= 1 && lat_h >= 1 && lat_w >= 1, "LatentMaskAccumulator: latent dims must be >= 1");
        assert!(h_pix >= lat_h && w_pix >= lat_w, "LatentMaskAccumulator: cannot upsample ({h_pix}x{w_pix} -> {lat_h}x{lat_w})");
        let run = if lat_f > 1 {
            assert!(f_pix > 1, "LatentMaskAccumulator: {lat_f} latent frames need more than one pixel frame");
            assert_eq!((f_pix - 1) % (lat_f - 1), 0, "LatentMaskAccumulator: pixel frames ({f_pix}) not compatible with latent frames ({lat_f}): (f_pix - 1) must divide by (lat_f - 1)");
            (f_pix - 1) / (lat_f - 1)
        } else {
            0
        };
        LatentMaskAccumulator {
            h_pix,
            w_pix,
            lat_h,
            lat_w,
            run,
            seen: 0,
            total_pix_frames: f_pix,
            acc: vec![0f64; lat_h * lat_w],
            in_run: 0,
            out: Vec::with_capacity(lat_f * lat_h * lat_w),
        }
    }

    /// Feed the next pixel frame, `[h_pix * w_pix]` row-major.
    ///
    /// # Panics
    /// On the wrong length, or if more frames are pushed than declared.
    pub fn push_frame(&mut self, frame: &[f32]) {
        assert_eq!(frame.len(), self.h_pix * self.w_pix, "LatentMaskAccumulator::push_frame: {} values, expected {}", frame.len(), self.h_pix * self.w_pix);
        assert!(self.seen < self.total_pix_frames, "LatentMaskAccumulator::push_frame: more frames pushed than the {} declared", self.total_pix_frames);
        let plane = self.area_average(frame);
        if self.seen == 0 {
            // Causal VAE: the first latent frame covers this pixel frame ALONE.
            self.out.extend_from_slice(&plane);
        } else {
            for (a, &v) in self.acc.iter_mut().zip(&plane) {
                *a += v as f64;
            }
            self.in_run += 1;
            if self.in_run == self.run {
                let n = self.run as f64;
                self.out.extend(self.acc.iter().map(|a| (a / n) as f32));
                self.acc.iter_mut().for_each(|a| *a = 0.0);
                self.in_run = 0;
            }
        }
        self.seen += 1;
    }

    /// The `[lat_f * lat_h * lat_w]` latent mask, frame-major/row-major.
    ///
    /// # Panics
    /// If fewer frames were pushed than declared - a short mask sequence is a
    /// caller error, never something to pad over.
    pub fn finish(self) -> Vec<f32> {
        assert_eq!(self.seen, self.total_pix_frames, "LatentMaskAccumulator::finish: {} frames pushed, {} declared", self.seen, self.total_pix_frames);
        assert_eq!(self.in_run, 0, "LatentMaskAccumulator::finish: {} frames left in an incomplete latent frame", self.in_run);
        self.out
    }

    /// `F.interpolate(mode="area")` = adaptive average pooling: output cell `i`
    /// spans `[floor(i*H/O), ceil((i+1)*H/O))`.
    fn area_average(&self, frame: &[f32]) -> Vec<f32> {
        let mut plane = vec![0f32; self.lat_h * self.lat_w];
        for oh in 0..self.lat_h {
            let (h0, h1) = (oh * self.h_pix / self.lat_h, ((oh + 1) * self.h_pix).div_ceil(self.lat_h));
            for ow in 0..self.lat_w {
                let (w0, w1) = (ow * self.w_pix / self.lat_w, ((ow + 1) * self.w_pix).div_ceil(self.lat_w));
                let mut sum = 0f64;
                for hh in h0..h1 {
                    for ww in w0..w1 {
                        sum += frame[hh * self.w_pix + ww] as f64;
                    }
                }
                plane[oh * self.lat_w + ow] = (sum / (((h1 - h0) * (w1 - w0)) as f64)) as f32;
            }
        }
        plane
    }
}

// ===========================================================================
// The `brain/sam2-maskseq/1` interchange format
// ===========================================================================

/// The manifest's declared `format` string. Anything else is refused: a mask
/// directory whose layout is not known is not something to guess at.
pub const MASKSEQ_FORMAT: &str = "brain/sam2-maskseq/1";

/// How a mask sequence encodes the TRACKED OBJECT - the region a swap
/// regenerates - in its 8-bit PNGs.
///
/// Read from the manifest's `polarity` field and never inferred from the
/// pixels. A sequence whose polarity is missing or unrecognised is rejected:
/// guessing it backwards preserves the character and regenerates the entire
/// background, which is the exact inverse of the intent and is not the kind of
/// mistake a run reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSeqPolarity {
    /// `"object=255"` - white is the tracked object. SAM 2's native output.
    ObjectWhite,
    /// `"object=0"` - black is the tracked object.
    ObjectBlack,
}

impl MaskSeqPolarity {
    /// Parse the manifest's own spelling.
    pub fn parse(s: &str) -> Result<MaskSeqPolarity, String> {
        match s {
            "object=255" => Ok(MaskSeqPolarity::ObjectWhite),
            "object=0" => Ok(MaskSeqPolarity::ObjectBlack),
            other => Err(format!(
                "mask sequence declares polarity {other:?}, which is not one of \"object=255\" / \"object=0\" - refusing to guess, because guessing it backwards preserves the tracked subject and regenerates everything else"
            )),
        }
    }

    /// **The one place the polarity flip happens.**
    ///
    /// `px` is one mask sample normalised to `[0, 1]` from the PNG's channel 0.
    /// The result is that sample in `VideoConditionByMask`'s polarity: `1` =
    /// conditioning (carried verbatim), `0` = generated.
    ///
    /// The tracked object is what a swap REPLACES, so it must land at `0` and
    /// everything else at `1`, whichever way the producer wrote it. Soft edges
    /// survive as soft edges: this is an affine map, and it commutes with the
    /// area average [`LatentMaskAccumulator`] applies afterwards.
    #[inline]
    pub fn to_conditioning(self, px: f32) -> f32 {
        match self {
            MaskSeqPolarity::ObjectWhite => 1.0 - px,
            MaskSeqPolarity::ObjectBlack => px,
        }
    }
}

/// A parsed `masks.json`.
#[derive(Debug, Clone)]
pub struct MaskSeqManifest {
    /// `printf`-style file name pattern, e.g. `mask_%06d.png`.
    pub pattern: String,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub polarity: MaskSeqPolarity,
    /// Per-frame `object_score`, when the producer recorded one. A NEGATIVE
    /// score is SAM 2 saying the object is absent or occluded on that frame -
    /// see [`read_mask_sequence`] for what this port does about it.
    pub object_score: Option<Vec<f32>>,
}

impl MaskSeqManifest {
    /// Parse `masks.json` from its bytes.
    ///
    /// Every field this port depends on is REQUIRED. A sequence missing one is
    /// refused rather than defaulted: each of these defaults would silently
    /// produce a plausible-looking clip that conditioned the wrong region.
    pub fn parse(bytes: &[u8]) -> Result<MaskSeqManifest, String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("masks.json is not valid JSON: {e}"))?;
        let fmt = v.get("format").and_then(|f| f.as_str()).ok_or("masks.json has no \"format\" field")?;
        if fmt != MASKSEQ_FORMAT {
            return Err(format!("masks.json declares format {fmt:?}, expected {MASKSEQ_FORMAT:?}"));
        }
        let req_str = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string).ok_or_else(|| format!("masks.json has no string \"{k}\""));
        let req_usize = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).map(|x| x as usize).ok_or_else(|| format!("masks.json has no integer \"{k}\""));
        let polarity = MaskSeqPolarity::parse(&req_str("polarity")?)?;
        let object_score = v.get("per_frame").and_then(|p| p.as_array()).map(|rows| {
            rows.iter().map(|r| r.get("object_score").and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32).collect::<Vec<f32>>()
        });
        Ok(MaskSeqManifest {
            pattern: req_str("pattern")?,
            frames: req_usize("frames")?,
            width: req_usize("width")?,
            height: req_usize("height")?,
            fps: v.get("fps").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
            polarity,
            object_score,
        })
    }

    /// Expand [`Self::pattern`] for one frame index.
    ///
    /// Supports the one thing the format uses - a single `%0Nd` (or `%d`)
    /// field - and refuses anything else rather than mis-naming a file.
    pub fn file_name(&self, frame: usize) -> Result<String, String> {
        let p = &self.pattern;
        let start = p.find('%').ok_or_else(|| format!("mask pattern {p:?} has no % field"))?;
        let rest = &p[start + 1..];
        let end = rest.find('d').ok_or_else(|| format!("mask pattern {p:?} is not a %[0N]d field"))?;
        let spec = &rest[..end];
        let width: usize = if spec.is_empty() {
            0
        } else {
            let digits = spec.strip_prefix('0').ok_or_else(|| format!("mask pattern {p:?} has an unsupported field {spec:?}"))?;
            digits.parse().map_err(|_| format!("mask pattern {p:?} has an unsupported field {spec:?}"))?
        };
        if p[start + 1 + end + 1..].contains('%') {
            return Err(format!("mask pattern {p:?} has more than one % field"));
        }
        Ok(format!("{}{:0width$}{}", &p[..start], frame, &p[start + 1 + end + 1..], width = width))
    }
}

/// Read a `brain/sam2-maskseq/1` directory and reduce it to the `[lat_f *
/// lat_h * lat_w]` latent conditioning mask [`apply_video_condition_by_mask`]
/// takes, in the reference's polarity.
///
/// One PNG is resident at a time (see [`LatentMaskAccumulator`]).
///
/// **Frame count is checked against the clip and never reconciled.** A mask
/// sequence that is shorter or longer than the video it masks is a mismatch
/// between two separate runs; truncating or padding would shift every mask
/// after the first discrepancy, which is unrecoverable and invisible.
///
/// **Occluded frames condition EVERYTHING.** A negative `object_score` is SAM 2
/// reporting that it cannot see the tracked object on that frame - normal in a
/// real shot, where the subject passes behind something. The mask it emits for
/// such a frame is empty, and an empty object mask means "regenerate the whole
/// frame" once inverted. That is the worst possible answer: one dropped track
/// and a second of the clip is redrawn from scratch. This port instead
/// conditions the frame FULLY (all `1` in the reference's polarity), so an
/// occluded frame carries its source content verbatim and the swap simply does
/// not happen there. That is a deliberate, conservative choice - it is visible
/// as "the swap flickers off" rather than as "the shot dissolves" - and it is
/// reported on `tracing` at `warn` so a caller can see how much of the clip it
/// covered.
///
/// # Errors
/// Missing/invalid manifest, unknown format or polarity, a frame count or
/// resolution that disagrees with the clip, a missing or wrongly sized PNG.
pub fn read_mask_sequence(dir: &Path, clip_frames: usize, clip_width: usize, clip_height: usize, lat_f: usize, lat_h: usize, lat_w: usize) -> Result<Vec<f32>, String> {
    let mpath = dir.join("masks.json");
    let bytes = std::fs::read(&mpath).map_err(|e| format!("{}: {e}", mpath.display()))?;
    let man = MaskSeqManifest::parse(&bytes).map_err(|e| format!("{}: {e}", mpath.display()))?;
    if man.frames != clip_frames {
        return Err(format!(
            "the mask sequence in {} covers {} frames but the clip has {clip_frames}: these are two different runs, and matching them up by truncating or padding would silently shift every mask",
            dir.display(),
            man.frames
        ));
    }
    if man.width != clip_width || man.height != clip_height {
        return Err(format!("the mask sequence in {} is {}x{} but the clip is {clip_width}x{clip_height}", dir.display(), man.width, man.height));
    }
    if let Some(scores) = &man.object_score {
        if scores.len() != man.frames {
            return Err(format!("{}: per_frame has {} entries but frames is {}", mpath.display(), scores.len(), man.frames));
        }
    } else {
        tracing::warn!(dir = %dir.display(), "mask sequence carries no per_frame block: every frame is treated as a frame the tracker could see");
    }

    let mut acc = LatentMaskAccumulator::new(clip_frames, clip_height, clip_width, lat_f, lat_h, lat_w);
    let mut occluded = 0usize;
    let mut plane = vec![0f32; clip_height * clip_width];
    for f in 0..clip_frames {
        let occl = man.object_score.as_ref().is_some_and(|s| s[f] < 0.0);
        if occl {
            occluded += 1;
            // Fully conditioned: carry the source frame, generate nothing.
            plane.fill(1.0);
            acc.push_frame(&plane);
            continue;
        }
        let name = man.file_name(f)?;
        let p = dir.join(&name);
        let img = image::open(&p).map_err(|e| format!("{}: {e}", p.display()))?.to_rgb8();
        if img.width() as usize != clip_width || img.height() as usize != clip_height {
            return Err(format!("{} is {}x{}, expected {clip_width}x{clip_height}", p.display(), img.width(), img.height()));
        }
        let raw = img.as_raw();
        for (i, px) in plane.iter_mut().enumerate() {
            // Channel 0 of the replicated luminance, the repo-wide convention
            // (`capability::blob::decode_plane`).
            *px = man.polarity.to_conditioning(raw[i * 3] as f32 / 255.0);
        }
        acc.push_frame(&plane);
    }
    if occluded > 0 {
        tracing::warn!(occluded, frames = clip_frames, "the tracker reported the object absent on some frames; those frames are conditioned fully and the swap does not happen there");
    }
    let latent = acc.finish();
    let conditioned = latent.iter().filter(|&&v| v > 0.5).count();
    tracing::info!(latent_tokens = latent.len(), conditioned, polarity = ?man.polarity, "mask sequence reduced to the latent grid");
    Ok(latent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The polarity flip is an involution pair, and it is the ONLY place the
    /// inversion lives. A tracked object must reach the reference's polarity as
    /// `0` (generated) from either encoding.
    #[test]
    fn polarity_puts_the_tracked_object_on_the_generated_side() {
        assert_eq!(MaskSeqPolarity::ObjectWhite.to_conditioning(1.0), 0.0, "white object -> generated");
        assert_eq!(MaskSeqPolarity::ObjectWhite.to_conditioning(0.0), 1.0, "white background -> conditioned");
        assert_eq!(MaskSeqPolarity::ObjectBlack.to_conditioning(0.0), 0.0, "black object -> generated");
        assert_eq!(MaskSeqPolarity::ObjectBlack.to_conditioning(1.0), 1.0, "black background -> conditioned");
        // Soft edges survive as soft edges rather than being thresholded.
        assert!((MaskSeqPolarity::ObjectWhite.to_conditioning(0.25) - 0.75).abs() < 1e-7);
    }

    #[test]
    fn an_unknown_or_absent_polarity_is_refused_rather_than_defaulted() {
        assert!(MaskSeqPolarity::parse("object=1").is_err());
        assert!(MaskSeqPolarity::parse("").is_err());
        let no_polarity = br#"{"format":"brain/sam2-maskseq/1","pattern":"mask_%06d.png","frames":9,"width":64,"height":64}"#;
        let e = MaskSeqManifest::parse(no_polarity).expect_err("a manifest without a polarity must not parse");
        assert!(e.contains("polarity"), "{e}");
    }

    #[test]
    fn a_foreign_format_is_refused() {
        let other = br#"{"format":"brain/sam2-maskseq/2","pattern":"m_%d.png","frames":1,"width":1,"height":1,"polarity":"object=255"}"#;
        assert!(MaskSeqManifest::parse(other).expect_err("must refuse").contains("format"));
    }

    #[test]
    fn the_manifest_carries_the_fields_this_port_reads() {
        let js = br#"{"format":"brain/sam2-maskseq/1","pattern":"mask_%06d.png","frames":9,"width":128,"height":64,
                      "fps":24.0,"polarity":"object=255","binary":true,"threshold":0.5,"object_id":0,
                      "per_frame":[{"frame":0,"object_score":1.5},{"frame":1,"object_score":-0.25}]}"#;
        let m = MaskSeqManifest::parse(js).expect("parses");
        assert_eq!(m.frames, 9);
        assert_eq!((m.width, m.height), (128, 64));
        assert_eq!(m.polarity, MaskSeqPolarity::ObjectWhite);
        assert_eq!(m.object_score.as_deref(), Some(&[1.5f32, -0.25][..]));
        assert_eq!(m.file_name(7).unwrap(), "mask_000007.png");
    }

    #[test]
    fn patterns_expand_the_way_the_producer_writes_them() {
        let mk = |p: &str| MaskSeqManifest {
            pattern: p.to_string(),
            frames: 1,
            width: 1,
            height: 1,
            fps: 0.0,
            polarity: MaskSeqPolarity::ObjectWhite,
            object_score: None,
        };
        assert_eq!(mk("mask_%06d.png").file_name(12).unwrap(), "mask_000012.png");
        assert_eq!(mk("%d.png").file_name(12).unwrap(), "12.png");
        assert!(mk("mask.png").file_name(0).is_err(), "a pattern with no field cannot name frame 1");
        assert!(mk("%06d_%06d.png").file_name(0).is_err(), "two fields is ambiguous");
    }

    /// The mask enters [`noised_initial_latent`] quadratically at fractional
    /// values (see this module's doc). Pinned here so the surprise is a
    /// documented, tested property rather than something to "fix" later.
    #[test]
    fn a_conditioning_position_starts_bit_exactly_at_the_conditioning_latent() {
        let (t, c) = (3usize, 2usize);
        let cond: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.5 - 1.0).collect();
        let noise: Vec<f32> = (0..t * c).map(|i| 3.0 - i as f32).collect();
        let mask = vec![1.0, 0.0, 0.5];
        let m = apply_video_condition_by_mask(&vec![0f32; t * c], &vec![1f32; t], &cond, &mask, c, 1.0);
        assert_eq!(m.denoise_mask, vec![0.0, 1.0, 0.5]);
        let l = noised_initial_latent(&noise, &noise, &m.clean, &m.denoise_mask, c, 1.0);
        assert_eq!(&l[..c], &cond[..c], "mask=1 at strength 1.0 starts EXACTLY at the conditioning latent");
        assert_eq!(&l[c..2 * c], &noise[c..2 * c], "mask=0 starts at pure noise");
        for i in 0..c {
            let want = 0.5 * cond[2 * c + i] + 0.5 * (noise[2 * c + i] - 0.5 * cond[2 * c + i]);
            assert!((l[2 * c + i] - want).abs() < 1e-6, "fractional mask enters the initial latent quadratically");
        }
    }
}
