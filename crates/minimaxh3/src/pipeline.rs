// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's `t2va` (text only) and `fl2va` (first/last keyframe) end-to-
//! end pipelines: prompt (+ optional keyframes) -> packed-sequence layout ->
//! the dual-schedule denoise loop over [`crate::model::H3Transformer`] ->
//! [`crate::video_vae`] + [`crate::vocoder`] decode -> pixels + waveform.
//!
//! Follows `wan::pipeline`'s phase-sequential-residency shape: text/vision
//! conditioning is built by the CALLER (see [`TextConditioning`]'s own doc
//! for why - the Qwen3-VL splice this needs is a real, documented gap, not
//! hidden here), then [`generate`] loads the DiT, drives every denoise step
//! resident, drops it, and only then opens the two VAEs to decode - the DiT
//! and the two VAEs are never resident together, matching every other
//! pipeline in this workspace's memory discipline.
//!
//! Every piece of packing math below (`position_ids`/`token_tags`/the three
//! index arrays/the per-step row-timestep plan/the DiT-level patchify) is
//! ported directly from the real installed `diffusers==0.40.0` source, read
//! in full for this phase (not the roadmap's earlier secondhand summary):
//! `diffusers/modular_pipelines/minimax_h3/{before_denoise,denoise,encoders,
//! decoders,modular_pipeline}.py`. Every function below cites the exact
//! class/method/line it reproduces.
//!
//! ## `ref2va` - deliberately not implemented here
//!
//! `ref2va`'s packed layout (`MiniMaxH3Ref2VAPrepareLayoutStep`) is
//! structurally different enough from `t2va`/`fl2va` (per-reference
//! sub-blocks in request order, a second summation-order convention for the
//! video-reference rotary span, up to 12 mixed image/video/audio references)
//! that it needs its own layout builder, not a generalization of
//! [`build_packed_sequence`] below. Per this phase's own scope, `ref2va` is
//! deferred rather than guessed - see this crate's roadmap for the settled
//! `ref2va` facts a follow-up phase can build from.
//!
//! Swedish Embedded AB implements end-to-end diffusion pipelines like this
//! one for its clients. If your team needs expertise in composing large
//! multi-modal diffusion models into a working generation loop, you can
//! procure our services by sending an email to info@swedishembedded.com.

use vae::blocks::Tensors;

use crate::config::{H3TransformerConfig, TAG_AUDIO, TAG_VIDEO};
use crate::model::{H3Transformer, PackedInputs};
use crate::schedule::{DualSchedule, H3Scheduler};
use crate::video_vae::VideoVaeConfig;
use crate::vocoder::VocoderConfig;

// ============================================================================
// Model-fixed constants (`MiniMaxH3ModularPipeline`'s own properties/module
// constants, `modular_pipeline.py`) - real values, not assumed.
// ============================================================================

/// `MINIMAX_H3_FPS` - MiniMax-H3's fixed generation frame rate.
pub const FPS: f32 = 24.0;
/// `MINIMAX_H3_AUDIO_LATENTS_PER_SECOND` - the audio VAE's latent rate.
pub const AUDIO_LATENTS_PER_SECOND: f32 = 40.0;
/// `MINIMAX_H3_AUDIO_CHANNELS` - stereo, packed channel-major.
pub const AUDIO_CHANNELS: u32 = 2;
/// `MINIMAX_H3_MIN_ASPECT_RATIO`.
pub const MIN_ASPECT_RATIO: f64 = 0.25;
/// `MINIMAX_H3_MAX_ASPECT_RATIO`.
pub const MAX_ASPECT_RATIO: f64 = 4.0;
/// `MiniMaxH3PrepareLayoutStep.expected_configs`' `canvas_short_edge`.
pub const CANVAS_SHORT_EDGE: u32 = 768;
/// `MiniMaxH3PrepareLayoutStep.expected_configs`' `canvas_max_pixels`.
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;
/// `MiniMaxH3ModularPipeline.min_duration`, in seconds.
pub const MIN_DURATION_S: f32 = 5.0;
/// `MiniMaxH3ModularPipeline.max_duration`, in seconds.
pub const MAX_DURATION_S: f32 = 15.0;
/// `MiniMaxH3ModularPipeline.keyframe_noise_aug` - the `t` a keyframe anchor
/// is held at (just short of clean, `t=1` in H3's own convention).
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;
/// `MiniMaxH3ModularPipeline.text_encoder_layer` - Phase 3's confirmed depth.
pub const TEXT_ENCODER_LAYER: usize = 50;
/// `MiniMaxH3ModularPipeline.pixel_mean` - ImageNet, applied to the video
/// VAE's RGB input/output.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// `MiniMaxH3ModularPipeline.pixel_std`.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

// The rotary-time constants `before_denoise.py` module-level defines.
const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE: f64 = 32.0;

// ============================================================================
// Canvas / frame-count arithmetic (`modular_pipeline.py`'s free functions)
// ============================================================================

/// `resolve_canvas_size`: resolve a display aspect ratio into a `(height,
/// width)` MiniMax-H3 canvas - short edge `short_edge`, area capped at
/// `max_pixels`, both axes then rounded to the nearest `canvas_multiple`.
pub fn resolve_canvas_size(aspect_width: f64, aspect_height: f64, canvas_multiple: u32, short_edge: u32, max_pixels: u32) -> Result<(u32, u32), String> {
    if aspect_width <= 0.0 || aspect_height <= 0.0 {
        return Err(format!("resolve_canvas_size: aspect ratio must be positive, got {aspect_width}:{aspect_height}"));
    }
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(format!("resolve_canvas_size: MiniMax-H3 supports aspect ratios from 1:{} to {}:1, got {aspect_width}:{aspect_height} ({ratio})", 1.0 / MIN_ASPECT_RATIO, MAX_ASPECT_RATIO));
    }
    let (mut width, mut height) = if ratio >= 1.0 { (short_edge as f64 * ratio, short_edge as f64) } else { (short_edge as f64, short_edge as f64 / ratio) };
    let area = width * height;
    if area > max_pixels as f64 {
        let scale = (max_pixels as f64 / area).sqrt();
        width *= scale;
        height *= scale;
    }
    let m = canvas_multiple as f64;
    let round_to_multiple = |v: f64| -> u32 { (canvas_multiple.max((v / m).round() as u32 * canvas_multiple)).max(canvas_multiple) };
    Ok((round_to_multiple(height), round_to_multiple(width)))
}

/// `align_num_frames`: snap `num_frames` up to the next `frames_per_chunk*n +
/// latents_per_chunk` the video VAE can encode.
pub fn align_num_frames(num_frames: u32, frames_per_chunk: u32, latents_per_chunk: u32) -> Result<u32, String> {
    if num_frames < 1 {
        return Err(format!("align_num_frames: num_frames must be positive, got {num_frames}"));
    }
    let mut n = num_frames;
    while n % frames_per_chunk != latents_per_chunk {
        n += 1;
    }
    Ok(n)
}

/// `video_latent_num_frames`: the number of latent frames the video VAE
/// produces for an ALIGNED frame count.
pub fn video_latent_num_frames(num_frames: u32, frames_per_chunk: u32, latents_per_chunk: u32) -> u32 {
    assert_eq!(num_frames % frames_per_chunk, latents_per_chunk, "video_latent_num_frames: num_frames {num_frames} is not aligned to {frames_per_chunk}*n+{latents_per_chunk}");
    (num_frames - latents_per_chunk) / frames_per_chunk * latents_per_chunk + 2
}

/// `audio_latent_num_frames`: the number of audio latents covering a video
/// of `num_frames` frames.
pub fn audio_latent_num_frames(num_frames: u32, fps: f32, latents_per_second: f32) -> u32 {
    (num_frames as f32 / fps * latents_per_second).round() as u32
}

// ============================================================================
// Rotary position grid (`before_denoise.py`: `_spatial_position_grid`,
// `_frame_position_grid`, `_temporal_position_grid`, and the `"last"`
// keyframe anchor's numpy-pairwise-summed span)
// ============================================================================

/// `_spatial_position_grid`: one aspect-normalized spatial rotary axis,
/// `dim//patch` coordinates centred on the unit interval and scaled by 32,
/// built with `np.linspace(left, left+ratio, dim//patch, endpoint=False)` -
/// `start + arange(num)*(stop-start)/num`, NOT `torch.linspace`'s
/// endpoint-inclusive formula. Computed in f64, matching the reference's own
/// float64 grid.
fn spatial_position_grid(dim: u32, patch: u32, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = (dim / patch) as usize;
    (0..n).map(|i| (left + i as f64 * ratio / n as f64) * ROPE_SPATIAL_SCALE).collect()
}

/// `_frame_position_grid`: the `(h, w)` rotary coordinates of one latent
/// frame (`torch.meshgrid(height_grid, width_grid, indexing="ij")`, flattened
/// row-major - height varies slowest), and the width axis they were built
/// from (needed again by audio rows, which pin to its two extremes).
fn frame_position_grid(latent_height: u32, latent_width: u32, patch_h: u32, patch_w: u32) -> (Vec<[f64; 2]>, Vec<f64>) {
    let sqrt_area = (latent_height as f64 * latent_width as f64).sqrt();
    let height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area);
    let width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area);
    let mut grid = Vec::with_capacity(height_grid.len() * width_grid.len());
    for &h in &height_grid {
        for &w in &width_grid {
            grid.push([h, w]);
        }
    }
    (grid, width_grid)
}

/// `_temporal_position_grid`: the rotary time of every latent frame,
/// starting at `origin`. Spacing is non-uniform: `5/3 * (1,4,4,4,4)`
/// cyclically.
fn temporal_position_grid(num_latent_frames: usize, origin: f64) -> Vec<f64> {
    let spans: Vec<f64> = (0..num_latent_frames).map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()]).collect();
    let mut out = vec![0f64; num_latent_frames];
    if num_latent_frames > 0 {
        out[0] = origin;
    }
    let mut cum = 0f64;
    for i in 1..num_latent_frames {
        cum += spans[i - 1];
        out[i] = origin + cum;
    }
    out
}

/// numpy's pairwise summation algorithm (`pairwise_sum` in numpy's own
/// `loops.c.src`, transcribed): base case <=8 elements is a plain sequential
/// sum; 8 < n <= 128 accumulates 8 running partial sums 8-wide then combines
/// them in a balanced tree (`((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))`), any
/// remainder added on afterward; n > 128 splits at the midpoint (rounded down
/// to a multiple of 8) and recurses. This is what the `"last"` keyframe
/// anchor's rotary-time span uses (`before_denoise.py`'s
/// `build_packed_sequence`, the `spans.sum()` call) rather than a plain
/// sequential sum - the two orders differ in the last ULP from 16 elements
/// onward per that code's own comment, which is why this is reproduced
/// faithfully rather than simplified to `iter().sum()`. Not asserted
/// bit-exact against numpy itself (porting.md's own bar for pure schedule
/// math is "within a couple of ULPs", not bit-identical).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n <= 8 {
        return a.iter().sum();
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        while i + 8 <= n {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

// ============================================================================
// Packed-sequence layout (`before_denoise.py`:
// `MiniMaxH3PrepareLayoutStep.build_packed_sequence`)
// ============================================================================

/// Which end of the video a keyframe conditioning block anchors to
/// (`keyframe_anchors`' entries).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    First,
    Last,
}

/// [`build_packed_sequence`]'s output: the `[text | keyframe conditions |
/// target audio | target video]` layout's `(t,h,w)` rotary grid, modality
/// tags, and the three row-index arrays every later stage addresses rows
/// through.
#[derive(Clone, Debug)]
pub struct PackedSequence {
    /// `[seq_len, 3]` row-major `(t, h, w)`.
    pub position_ids: Vec<f32>,
    pub token_tags: Vec<u32>,
    /// Sequence positions of the video rows, conditioning rows first.
    pub video_indices: Vec<u32>,
    /// Sequence positions of the audio rows (always the generated rows only,
    /// for `t2va`/`fl2va` - never any conditioning audio).
    pub audio_indices: Vec<u32>,
    pub text_indices: Vec<u32>,
    /// How many leading rows of `video_indices` are keyframe conditioning
    /// rows rather than generated rows.
    pub num_condition_video_rows: usize,
    /// Always `0` for `t2va`/`fl2va` (only `ref2va` has conditioning audio).
    pub num_condition_audio_rows: usize,
}

/// `MiniMaxH3PrepareLayoutStep.build_packed_sequence`, ported exactly:
/// `[text | keyframe conditions | target audio | target video]`.
///
/// `text_token_tags` is the per-row modality tag of every text-presentation
/// row (`TAG_TEXT` for plain text, `TAG_VIDEO` for a keyframe's own vision-
/// block rows in the text presentation - see [`TextConditioning`]'s own
/// doc); `keyframe_anchors` is one entry per keyframe conditioning block, in
/// packed order (empty for `t2va`).
#[allow(clippy::too_many_arguments)]
pub fn build_packed_sequence(text_token_tags: &[u32], num_latent_frames: u32, latent_height: u32, latent_width: u32, num_audio_latents: u32, patch_size: [u32; 3], audio_channels: u32, keyframe_anchors: &[Anchor]) -> PackedSequence {
    let (patch_h, patch_w) = (patch_size[1], patch_size[2]);
    let rows_per_frame = (latent_height / patch_h) * (latent_width / patch_w);
    let num_text_tokens = text_token_tags.len() as u32;
    let num_condition_rows = keyframe_anchors.len() as u32 * rows_per_frame;
    let num_audio_rows = num_audio_latents * audio_channels;
    let num_video_rows = num_latent_frames * rows_per_frame;
    let sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows;

    let condition_start = num_text_tokens;
    let audio_start = condition_start + num_condition_rows;
    let video_start = audio_start + num_audio_rows;

    let mut position_ids = vec![0f64; sequence_length as usize * 3];
    for i in 0..num_text_tokens {
        position_ids[(i * 3) as usize] = i as f64;
    }

    let (frame_grid, width_grid) = frame_position_grid(latent_height, latent_width, patch_h, patch_w);

    for (index, anchor) in keyframe_anchors.iter().enumerate() {
        let anchor_time = match anchor {
            Anchor::First => num_text_tokens as f64,
            Anchor::Last => {
                let spans: Vec<f64> = (0..num_latent_frames).map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[(i as usize) % ROPE_FRAMES_PER_LATENT.len()]).collect();
                num_text_tokens as f64 + pairwise_sum(&spans) - ROPE_FRAME_RESCALE
            }
        };
        let row0 = condition_start + index as u32 * rows_per_frame;
        for r in 0..rows_per_frame {
            let row = (row0 + r) as usize;
            position_ids[row * 3] = anchor_time;
            position_ids[row * 3 + 1] = frame_grid[r as usize][0];
            position_ids[row * 3 + 2] = frame_grid[r as usize][1];
        }
    }

    // Audio rows: channel-major, `audio_time.repeat(audio_channels)` (torch
    // `repeat` on a 1D tensor TILES, it does not repeat-each), pinned to the
    // width grid's extremes - the first `num_audio_latents` rows (channel 0)
    // at `width_grid[0]`, every remaining row (every other channel) at
    // `width_grid[-1]`.
    let w_first = width_grid[0];
    let w_last = width_grid[width_grid.len() - 1];
    for c in 0..audio_channels {
        for k in 0..num_audio_latents {
            let row = (audio_start + c * num_audio_latents + k) as usize;
            position_ids[row * 3] = num_text_tokens as f64 + k as f64;
            position_ids[row * 3 + 2] = if c == 0 { w_first } else { w_last };
        }
    }

    let temporal = temporal_position_grid(num_latent_frames as usize, num_text_tokens as f64);
    for f in 0..num_latent_frames {
        for r in 0..rows_per_frame {
            let row = (video_start + f * rows_per_frame + r) as usize;
            position_ids[row * 3] = temporal[f as usize];
            position_ids[row * 3 + 1] = frame_grid[r as usize][0];
            position_ids[row * 3 + 2] = frame_grid[r as usize][1];
        }
    }

    let video_indices: Vec<u32> = (condition_start..audio_start).chain(video_start..sequence_length).collect();
    let audio_indices: Vec<u32> = (audio_start..video_start).collect();
    let text_indices: Vec<u32> = (0..num_text_tokens).collect();

    let mut token_tags = vec![0u32; sequence_length as usize];
    for (row, &tag) in text_indices.iter().zip(text_token_tags) {
        token_tags[*row as usize] = tag;
    }
    for &row in &audio_indices {
        token_tags[row as usize] = TAG_AUDIO;
    }
    for &row in &video_indices {
        token_tags[row as usize] = TAG_VIDEO;
    }

    PackedSequence {
        position_ids: position_ids.iter().map(|&v| v as f32).collect(),
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows: num_condition_rows as usize,
        num_condition_audio_rows: 0,
    }
}

// ============================================================================
// Per-step row-timestep plan (`before_denoise.py`:
// `MiniMaxH3SetTimestepsStep.build_row_timesteps`)
// ============================================================================

/// `torch.unique(values, sorted=True, return_inverse=True)`: the sorted
/// distinct values, and every input row's index into them.
fn unique_sorted_with_inverse(values: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("unique_sorted_with_inverse: NaN timestep"));
    sorted.dedup();
    let indices = values.iter().map(|v| sorted.binary_search_by(|probe| probe.partial_cmp(v).expect("unique_sorted_with_inverse: NaN timestep")).expect("unique_sorted_with_inverse: value missing from its own sorted set") as u32).collect();
    (sorted, indices)
}

/// `MiniMaxH3SetTimestepsStep.build_row_timesteps`: assign a timestep to
/// every row of the packed sequence, then reduce to the transformer's
/// `(timestep, timestep_indices)` pair. Generated video/audio rows step down
/// their own schedule; conditioning rows stay pinned at their own noise
/// level; TEXT rows (never touched by either `video_indices`/`audio_indices`
/// assignment) inherit the default `video_timestep` - matching
/// `crate::precompute_adaln::timestep_grid`'s own documented choice, now
/// confirmed from source rather than assumed.
#[allow(clippy::too_many_arguments)]
fn build_row_timesteps(video_indices: &[u32], audio_indices: &[u32], num_condition_video_rows: usize, num_condition_audio_rows: usize, num_text_tokens: usize, video_timestep: f32, audio_timestep: f32, condition_video_timestep: f32, condition_audio_timestep: f32) -> (Vec<f32>, Vec<u32>) {
    let seq_len = video_indices.len() + audio_indices.len() + num_text_tokens;
    let mut row_timesteps = vec![video_timestep; seq_len];
    for &row in &video_indices[..num_condition_video_rows] {
        row_timesteps[row as usize] = condition_video_timestep;
    }
    for &row in &audio_indices[num_condition_audio_rows..] {
        row_timesteps[row as usize] = audio_timestep;
    }
    for &row in &audio_indices[..num_condition_audio_rows] {
        row_timesteps[row as usize] = condition_audio_timestep;
    }
    unique_sorted_with_inverse(&row_timesteps)
}

// ============================================================================
// DiT-level patchify/unpatchify (`before_denoise.py::patchify_video_latents`
// and its inverse, `decoders.py::MiniMaxH3AfterDenoiseStep`)
// ============================================================================

/// `patchify_video_latents`: pack a channel-major `[C,T,H,W]` latent into
/// `[T/pt*H/ph*W/pw, C*pt*ph*pw]` rows, frame-major then row-major, each
/// row's own columns ordered `(c, pt_i, ph_i, pw_i)`.
pub fn patchify_video(latents: &[f32], c: u32, t: u32, h: u32, w: u32, patch: [u32; 3]) -> Vec<f32> {
    let (pt, ph, pw) = (patch[0], patch[1], patch[2]);
    assert_eq!(t % pt, 0, "patchify_video: t {t} not divisible by patch_t {pt}");
    assert_eq!(h % ph, 0, "patchify_video: h {h} not divisible by patch_h {ph}");
    assert_eq!(w % pw, 0, "patchify_video: w {w} not divisible by patch_w {pw}");
    assert_eq!(latents.len(), (c * t * h * w) as usize, "patchify_video: input length mismatch");
    let (tp, hp, wp) = (t / pt, h / ph, w / pw);
    let row_len = (c * pt * ph * pw) as usize;
    let mut out = vec![0f32; (tp * hp * wp) as usize * row_len];
    for tt in 0..tp {
        for hh in 0..hp {
            for ww in 0..wp {
                let row = ((tt * hp + hh) * wp + ww) as usize;
                for cc in 0..c {
                    for pti in 0..pt {
                        for phi in 0..ph {
                            for pwi in 0..pw {
                                let src_t = tt * pt + pti;
                                let src_h = hh * ph + phi;
                                let src_w = ww * pw + pwi;
                                let src_idx = (((cc * t + src_t) * h + src_h) * w + src_w) as usize;
                                let dst_col = (((cc * pt + pti) * ph + phi) * pw + pwi) as usize;
                                out[row * row_len + dst_col] = latents[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`patchify_video`] (`MiniMaxH3AfterDenoiseStep`'s
/// `reshape`+`permute`+`reshape`, run backward): `[num_rows, C*pt*ph*pw]`
/// rows back to a channel-major `[C,T,H,W]` latent.
pub fn unpatchify_video(rows: &[f32], c: u32, t: u32, h: u32, w: u32, patch: [u32; 3]) -> Vec<f32> {
    let (pt, ph, pw) = (patch[0], patch[1], patch[2]);
    let (tp, hp, wp) = (t / pt, h / ph, w / pw);
    let row_len = (c * pt * ph * pw) as usize;
    assert_eq!(rows.len(), (tp * hp * wp) as usize * row_len, "unpatchify_video: input length mismatch");
    let mut out = vec![0f32; (c * t * h * w) as usize];
    for tt in 0..tp {
        for hh in 0..hp {
            for ww in 0..wp {
                let row = ((tt * hp + hh) * wp + ww) as usize;
                for cc in 0..c {
                    for pti in 0..pt {
                        for phi in 0..ph {
                            for pwi in 0..pw {
                                let dst_t = tt * pt + pti;
                                let dst_h = hh * ph + phi;
                                let dst_w = ww * pw + pwi;
                                let dst_idx = (((cc * t + dst_t) * h + dst_h) * w + dst_w) as usize;
                                let src_col = (((cc * pt + pti) * ph + phi) * pw + pwi) as usize;
                                out[dst_idx] = rows[row * row_len + src_col];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// Text conditioning - built by the CALLER, consumed here
// ============================================================================

/// One request's text conditioning: the Qwen3-VL hidden state MiniMax-H3
/// reads (`qwen3vl::Qwen3Vl::encode_hidden`/`encode_hidden_with_image` at
/// [`TEXT_ENCODER_LAYER`]) plus the per-row modality tag of every row of it
/// (`TAG_TEXT` for plain prompt text, `TAG_VIDEO` for a keyframe's own
/// `"<Picture i>: "` vision-block rows in the presentation -
/// `MiniMaxH3FL2VATextEncoderStep`'s own tagging).
///
/// **Built by the caller, not this module** - `get_qwen3vl_prompt_embeds`
/// (`encoders.py`) needs the full tokenized presentation (label + vision-pad
/// tokens per keyframe, then the prompt) built by a Qwen3-VL processor/
/// tokenizer this crate does not own, and `t2va`/`fl2va` share this one
/// type: a text-only presentation is just `token_tags` all `TAG_TEXT` and no
/// image.
///
/// **`fl2va` with 2 keyframes has a real, documented gap here**:
/// `qwen3vl::Qwen3Vl::splice_vision` (which `encode_hidden_with_image` calls)
/// supports exactly ONE spliced image at a FIXED row position baked in at
/// construction - it cannot place two keyframes' vision blocks at two
/// different token-stream positions in one forward. [`encode_text`] below
/// wires the ONE-image case (`t2va` needs no image at all; a single-keyframe
/// `fl2va` request needs exactly one); a two-keyframe `fl2va` request's text
/// presentation cannot be built through this crate's dependencies as they
/// stand today - `qwen3vl`'s splice needs generalizing to multiple images at
/// multiple positions in one forward first. This is a text-conditioning gap
/// ONLY: the DiT-side keyframe conditioning rows built by
/// [`build_packed_sequence`]/[`encode_keyframe_condition`] below - the
/// mechanism that actually anchors the generated video to the keyframes -
/// support up to 2 keyframes regardless, since they never touch the text
/// encoder at all.
pub struct TextConditioning {
    /// `[num_text_tokens, text_dim]`, row-major.
    pub embeds: Vec<f32>,
    pub token_tags: Vec<u32>,
}

/// Encode one `t2va`/`fl2va` text presentation - `get_qwen3vl_prompt_embeds`
/// (`encoders.py`), with `tokens`/`token_tags` and (for a single-keyframe
/// request) `image` already built by the caller. `image = None` is `t2va`'s
/// own path (`MiniMaxH3TextEncoderStep`); `image = Some(..)` is `fl2va`'s
/// single-keyframe path (`MiniMaxH3FL2VATextEncoderStep`) - see
/// [`TextConditioning`]'s own doc for the two-keyframe gap.
pub fn encode_text(qwen: &qwen3vl::Qwen3Vl, tokens: &[u32], token_tags: &[u32], image: Option<qwen3vl::model::ImageInput<'_>>) -> TextConditioning {
    assert_eq!(tokens.len(), token_tags.len(), "encode_text: tokens/token_tags length mismatch");
    let embeds = match image {
        Some(img) => qwen.encode_hidden_with_image(tokens, img, TEXT_ENCODER_LAYER),
        None => qwen.encode_hidden(tokens, TEXT_ENCODER_LAYER),
    };
    TextConditioning { embeds, token_tags: token_tags.to_vec() }
}

// ============================================================================
// Keyframe conditioning (`encoders.py::encode_vae_condition` +
// `MiniMaxH3PrepareConditionLatentsStep`)
// ============================================================================

/// One noised, patchified keyframe conditioning block, ready to prepend to
/// the generated video rows ([`generate`]'s `condition_rows` argument).
pub struct KeyframeCondition {
    pub anchor: Anchor,
    /// `[rows_per_frame, video_patch_dim]` flattened - one latent frame's
    /// worth of patchified rows.
    pub rows: Vec<f32>,
}

/// `encode_vae_condition` + `scale_noise` + `patchify_video_latents`: encode
/// one keyframe (already resized onto the TARGET canvas the request will
/// generate at - resizing is the caller's job, this function does not crop
/// or letterbox) into a noised, patchified conditioning block.
///
/// `pixels_rgb` is `[3, height, width]` channel-major, values in `[0, 255]`
/// (`uint8` range as `f32`, matching `encode_vae_condition`'s own
/// `pixels.div(255.0)`). `video_latents_mean`/`video_latents_std` are the
/// video VAE's own per-channel latent normalization (`vae.config.
/// latents_mean`/`latents_std`), read from the checkpoint by
/// [`crate::caps::read_latent_stats`]. They are a required input rather than
/// something defaulted here: `mean=0`/`std=1` is the IDENTITY of this affine
/// transform, so a default would not approximate the real values, it would
/// delete the normalization step and silently feed the DiT a latent in the
/// wrong space - see [`crate::caps::read_latent_stats`]'s own doc.
///
/// **Deliberate deviation**: the reference SAMPLES the encoder's posterior
/// (`posterior.sample(generator=...)`, seeded independently of the request);
/// this uses [`crate::video_vae::posterior_mode`] (the deterministic mean)
/// instead, since brain never claims torch RNG bit-parity anywhere in this
/// workspace (`wan::pipeline`'s own module doc states the same choice) - the
/// mode is the mean of the same posterior the sample would have been drawn
/// from, so this changes the conditioning by exactly the posterior's own
/// (small, VAE-regularized) variance, not a different quantity in kind.
#[allow(clippy::too_many_arguments)]
pub fn encode_keyframe_condition(video_vae_cfg: &VideoVaeConfig, video_vae_tensors: &Tensors, device: Option<&str>, pixels_rgb: &[f32], height: u32, width: u32, video_latents_mean: &[f32], video_latents_std: &[f32], patch_size: [u32; 3], anchor: Anchor, seed: u64) -> KeyframeCondition {
    let c = video_vae_cfg.in_channels;
    assert_eq!(pixels_rgb.len(), (c * height * width) as usize, "encode_keyframe_condition: pixels_rgb length mismatch");
    let mut px = vec![0f32; pixels_rgb.len()];
    for cc in 0..c as usize {
        let (mean, std) = (PIXEL_MEAN[cc], PIXEL_STD[cc]);
        for i in 0..(height * width) as usize {
            px[cc * (height * width) as usize + i] = (pixels_rgb[cc * (height * width) as usize + i] / 255.0 - mean) / std;
        }
    }

    let (moments, t1, h1, w1) = crate::video_vae::encode(video_vae_cfg, video_vae_tensors, device, &px, 1, height, width);
    let latent_channels = video_vae_cfg.latent_channels;
    let mean_latent = crate::video_vae::posterior_mode(&moments, latent_channels, t1, h1, w1);
    assert_eq!(video_latents_mean.len(), latent_channels as usize, "encode_keyframe_condition: video_latents_mean length must equal latent_channels");
    assert_eq!(video_latents_std.len(), latent_channels as usize, "encode_keyframe_condition: video_latents_std length must equal latent_channels");

    let plane = (t1 * h1 * w1) as usize;
    let mut normalized = vec![0f32; mean_latent.len()];
    for cc in 0..latent_channels as usize {
        for i in 0..plane {
            normalized[cc * plane + i] = (mean_latent[cc * plane + i] - video_latents_mean[cc]) / video_latents_std[cc];
        }
    }

    let mut rng = data::rng::Rng::new(seed);
    let noise: Vec<f32> = (0..normalized.len()).map(|_| rng.next_gaussian() as f32).collect();
    let noised = H3Scheduler::scale_noise(&normalized, KEYFRAME_NOISE_AUG, &noise);
    let rows = patchify_video(&noised, latent_channels, t1, h1, w1, patch_size);
    KeyframeCondition { anchor, rows }
}

// ============================================================================
// The checkpoint bundle + request options + output
// ============================================================================

/// Every weight source [`generate`] needs, none held resident together - see
/// this module's own doc for the phase-sequential loading order.
pub struct H3Checkpoint<'a> {
    /// A streaming [`checkpoint::weightio::WeightReader`] (never
    /// materializes the ~33B-param checkpoint as a whole-map host copy - see
    /// [`crate::model::H3Transformer::load`]'s own doc) or, in a test, the
    /// eager `Tensors` map every real `TensorSource` implementor still
    /// satisfies.
    pub dit_tensors: &'a dyn checkpoint::TensorSource,
    pub dit_cfg: H3TransformerConfig,
    pub video_vae_tensors: &'a Tensors,
    pub video_vae_cfg: VideoVaeConfig,
    pub vocoder_tensors: &'a Tensors,
    pub vocoder_cfg: VocoderConfig,
    /// The video VAE's own per-channel latent normalization
    /// (`vae.config.latents_mean`/`latents_std`) - see
    /// [`encode_keyframe_condition`]'s own doc for why these are a required,
    /// explicit input rather than a baked-in constant.
    pub video_latents_mean: Vec<f32>,
    pub video_latents_std: Vec<f32>,
    /// The audio VAE's own per-channel latent normalization
    /// (`audio_vae.config.latents_mean`/`latents_std`).
    pub audio_latents_mean: Vec<f32>,
    pub audio_latents_std: Vec<f32>,
}

/// One `t2va`/`fl2va` request's generation options
/// (`MiniMaxH3PrepareLayoutStep`'s own resolved inputs).
#[derive(Clone, Debug)]
pub struct GenOpts {
    /// `None` resolves the canvas at 16:9 (`resolve_canvas_size`'s own
    /// `t2va` default); `Some((height, width))` is an explicit canvas
    /// (already a multiple of `canvas_multiple`, see [`generate`]'s own
    /// validation).
    pub canvas: Option<(u32, u32)>,
    pub num_frames: u32,
    /// `linspace(1, 0, num_inference_steps)`'s own point count - drives
    /// `num_inference_steps - 1` model evaluations (fewer if the shift's
    /// dedup collapses any); see `crate::schedule`'s own doc.
    pub num_inference_steps: usize,
    pub seed: u64,
    pub device: Option<String>,
}

/// One decoded `t2va`/`fl2va` result: RGB video in `[0, 1]` plus a stereo
/// waveform.
pub struct GeneratedAv {
    /// `[3, num_video_frames, height, width]` channel-major, `[0, 1]`.
    pub video: Vec<f32>,
    pub num_video_frames: u32,
    pub height: u32,
    pub width: u32,
    /// One `[num_samples]` waveform per channel (`AUDIO_CHANNELS` of them),
    /// each already clamped to `[-1, 1]` by [`crate::vocoder::decode`].
    pub audio: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

// ============================================================================
// The pipelines
// ============================================================================

/// `t2va`: text only, no keyframe conditioning rows at all
/// (`MiniMaxH3NoKeyframeAnchorsStep`'s own `keyframe_anchors = ()`). Loads
/// (and drops) its own [`H3Transformer`] - the cold path; see [`t2va_hot`]
/// for a call that reuses an already-resident one.
pub fn t2va(ckpt: &H3Checkpoint, text: &TextConditioning, opts: &GenOpts) -> Result<GeneratedAv, String> {
    generate(ckpt, text, &[], opts, None)
}

/// `fl2va`: text plus up to 2 keyframe conditioning rows, one `"first"`-
/// and/or one `"last"`-anchored ([`Anchor`]), built via
/// [`encode_keyframe_condition`] in packed order. Cold path - see
/// [`fl2va_hot`] for a call that reuses an already-resident
/// [`H3Transformer`].
pub fn fl2va(ckpt: &H3Checkpoint, text: &TextConditioning, keyframes: &[KeyframeCondition], opts: &GenOpts) -> Result<GeneratedAv, String> {
    if keyframes.is_empty() {
        return Err("fl2va: at least one keyframe is required - use t2va for a text-only request".to_string());
    }
    generate(ckpt, text, keyframes, opts, None)
}

/// [`t2va`], against a caller-held resident [`H3Transformer`] instead of a
/// freshly loaded one - the residency-scheduled serving path's entry point.
/// `model`'s own config must match `ckpt.dit_cfg`; a mismatch is a caller
/// bug (wrong resident handed to the wrong checkpoint), asserted rather than
/// silently forwarded into a shape error deep in `forward`.
pub fn t2va_hot(ckpt: &H3Checkpoint, text: &TextConditioning, opts: &GenOpts, model: &H3Transformer) -> Result<GeneratedAv, String> {
    assert_eq!(model.config(), &ckpt.dit_cfg, "t2va_hot: the resident H3Transformer's config does not match ckpt.dit_cfg");
    generate(ckpt, text, &[], opts, Some(model))
}

/// [`fl2va`]'s [`t2va_hot`] analogue.
pub fn fl2va_hot(ckpt: &H3Checkpoint, text: &TextConditioning, keyframes: &[KeyframeCondition], opts: &GenOpts, model: &H3Transformer) -> Result<GeneratedAv, String> {
    if keyframes.is_empty() {
        return Err("fl2va: at least one keyframe is required - use t2va for a text-only request".to_string());
    }
    assert_eq!(model.config(), &ckpt.dit_cfg, "fl2va_hot: the resident H3Transformer's config does not match ckpt.dit_cfg");
    generate(ckpt, text, keyframes, opts, Some(model))
}

/// The shared `t2va`/`fl2va` body: resolve geometry, build the packed
/// layout, denoise, decode. `keyframes` is empty for `t2va`. `model`, when
/// given, is used in place of a freshly loaded one - [`H3Transformer::
/// forward`] takes an explicit packed sequence of ANY length per call (no
/// latent-extent-sized graph is baked in at `load` time, unlike this
/// workspace's other DiT residency precedents - `wan::pipeline::HotDit`'s
/// compiled RoPE/kernel graph genuinely is sized per `(frames,width,height)`),
/// so one resident transformer serves every request shape at a given
/// checkpoint/device.
fn generate(ckpt: &H3Checkpoint, text: &TextConditioning, keyframes: &[KeyframeCondition], opts: &GenOpts, model: Option<&H3Transformer>) -> Result<GeneratedAv, String> {
    let device = opts.device.as_deref();
    let dit_cfg = &ckpt.dit_cfg;
    let vae_cfg = &ckpt.video_vae_cfg;

    // 1. Geometry (`MiniMaxH3PrepareLayoutStep.__call__`).
    let canvas_multiple = vae_cfg.spatial_compression_ratio() * dit_cfg.patch_size[2];
    let (height, width) = match opts.canvas {
        Some((h, w)) => {
            if h % canvas_multiple != 0 || w % canvas_multiple != 0 {
                return Err(format!("generate: height/width must be multiples of {canvas_multiple}, got {h}x{w}"));
            }
            (h, w)
        }
        None => resolve_canvas_size(16.0, 9.0, canvas_multiple, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS)?,
    };
    let frames_per_chunk = vae_cfg.clip_length;
    let latents_per_chunk = vae_cfg.tokens_chunk_size();
    let aligned_num_frames = align_num_frames(opts.num_frames, frames_per_chunk, latents_per_chunk)?;
    let duration = aligned_num_frames as f32 / FPS;
    if !(MIN_DURATION_S..=MAX_DURATION_S).contains(&duration) {
        return Err(format!("generate: MiniMax-H3 generates between {MIN_DURATION_S}s and {MAX_DURATION_S}s at {FPS}fps, got num_frames={} (aligned {aligned_num_frames}, {duration}s)", opts.num_frames));
    }
    let num_latent_frames = video_latent_num_frames(aligned_num_frames, frames_per_chunk, latents_per_chunk);
    let ratio = vae_cfg.spatial_compression_ratio();
    let (latent_height, latent_width) = (height / ratio, width / ratio);
    let num_audio_latents = audio_latent_num_frames(aligned_num_frames, FPS, AUDIO_LATENTS_PER_SECOND);

    // 2. The packed layout.
    let anchors: Vec<Anchor> = keyframes.iter().map(|k| k.anchor).collect();
    let packed = build_packed_sequence(&text.token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents, dit_cfg.patch_size, AUDIO_CHANNELS, &anchors);

    let rows_per_frame = (latent_height / dit_cfg.patch_size[1]) * (latent_width / dit_cfg.patch_size[2]);
    let video_patch_dim = dit_cfg.video_patch_dim() as usize;
    for k in keyframes {
        if k.rows.len() != rows_per_frame as usize * video_patch_dim {
            return Err(format!("generate: keyframe condition has {} values, expected {} (rows_per_frame {rows_per_frame} * video_patch_dim {video_patch_dim})", k.rows.len(), rows_per_frame as usize * video_patch_dim));
        }
    }

    // 3. Compact video/audio rows: conditioning first (unchanged all loop
    // long), then the drawn noise of the generated rows
    // (`MiniMaxH3PrepareConditionLatentsStep` + `MiniMaxH3PrepareLatentsStep`
    // + `MiniMaxH3FL2VAPrepareLatentsStep`).
    let mut rng = data::rng::Rng::new(opts.seed);
    let num_condition_video_rows = packed.num_condition_video_rows;
    let num_generated_video_rows = packed.video_indices.len() - num_condition_video_rows;
    let mut compact_video = Vec::with_capacity(packed.video_indices.len() * video_patch_dim);
    for k in keyframes {
        compact_video.extend_from_slice(&k.rows);
    }
    for _ in 0..(num_generated_video_rows * video_patch_dim) {
        compact_video.push(rng.next_gaussian() as f32);
    }

    let audio_in_channels = dit_cfg.audio_in_channels as usize;
    let mut compact_audio = Vec::with_capacity(packed.audio_indices.len() * audio_in_channels);
    for _ in 0..(packed.audio_indices.len() * audio_in_channels) {
        compact_audio.push(rng.next_gaussian() as f32);
    }

    // 4. Denoise, both VAEs still unopened. A caller-supplied `model` (the
    // "hot"/resident serving path, `t2va_hot`/`fl2va_hot`) stays fully
    // resident and is reused across requests unchanged. Otherwise this
    // streams each step's forward one block at a time
    // (`H3Transformer::forward_streaming`) rather than eagerly loading every
    // block's weights up front - see that function's own doc for why this is
    // what actually lets `device` name a real GPU here at all, not only
    // "cpu": the eager load's ~132GB fp32 resident footprint never fits in
    // any single GPU's VRAM on this box, streaming's ~2.6GB/block does.
    let mut sched = DualSchedule::new();
    sched.set_timesteps(opts.num_inference_steps);
    let num_steps = sched.num_steps();
    let num_text_tokens = packed.text_indices.len();

    for i in 0..num_steps {
        let (video_ts, audio_ts) = sched.current_timesteps(i);
        let condition_video_ts = video_ts.max(KEYFRAME_NOISE_AUG);
        let (unique_ts, ts_idx) = build_row_timesteps(&packed.video_indices, &packed.audio_indices, num_condition_video_rows, packed.num_condition_audio_rows, num_text_tokens, video_ts, audio_ts, condition_video_ts, 1.0);

        let inp = PackedInputs {
            hidden_states: &compact_video,
            audio_hidden_states: &compact_audio,
            encoder_hidden_states: &text.embeds,
            timestep: &unique_ts,
            timestep_indices: &ts_idx,
            token_tags: &packed.token_tags,
            position_ids: &packed.position_ids,
            video_indices: &packed.video_indices,
            audio_indices: &packed.audio_indices,
            text_indices: &packed.text_indices,
        };
        let out = match model {
            Some(m) => m.forward(&inp),
            None => H3Transformer::forward_streaming(ckpt.dit_tensors, dit_cfg, device, &inp),
        };

        let gen_video_pred = &out.video[num_condition_video_rows * video_patch_dim..];
        let gen_video_sample = &compact_video[num_condition_video_rows * video_patch_dim..];
        let (video_next, audio_next) = (sched.video().step(gen_video_pred, video_ts, gen_video_sample, i), sched.audio().step(&out.audio, audio_ts, &compact_audio, i));
        compact_video[num_condition_video_rows * video_patch_dim..].copy_from_slice(&video_next);
        compact_audio.copy_from_slice(&audio_next);
    }

    // 5. Video decode (`MiniMaxH3AfterDenoiseStep` + `MiniMaxH3VideoDecodeStep`).
    let generated_video_rows = &compact_video[num_condition_video_rows * video_patch_dim..];
    let latent_channels = vae_cfg.latent_channels;
    let mut latent = unpatchify_video(generated_video_rows, latent_channels, num_latent_frames, latent_height, latent_width, dit_cfg.patch_size);
    assert_eq!(ckpt.video_latents_mean.len(), latent_channels as usize, "generate: video_latents_mean length must equal latent_channels");
    assert_eq!(ckpt.video_latents_std.len(), latent_channels as usize, "generate: video_latents_std length must equal latent_channels");
    let plane = (num_latent_frames * latent_height * latent_width) as usize;
    for cc in 0..latent_channels as usize {
        for v in &mut latent[cc * plane..(cc + 1) * plane] {
            *v = *v * ckpt.video_latents_std[cc] + ckpt.video_latents_mean[cc];
        }
    }
    let (mut pixels, out_t, out_h, out_w) = crate::video_vae::decode(vae_cfg, ckpt.video_vae_tensors, device, &latent, num_latent_frames, latent_height, latent_width);
    let out_plane = (out_t * out_h * out_w) as usize;
    for cc in 0..3usize {
        let (mean, std) = (PIXEL_MEAN[cc], PIXEL_STD[cc]);
        for v in &mut pixels[cc * out_plane..(cc + 1) * out_plane] {
            *v = (*v * std + mean).clamp(0.0, 1.0);
        }
    }

    // 6. Audio decode (`MiniMaxH3AfterDenoiseStep` + `MiniMaxH3AudioDecodeStep`).
    let generated_audio_rows = &compact_audio[packed.num_condition_audio_rows * audio_in_channels..];
    let num_audio_latents_gen = num_audio_latents as usize;
    assert_eq!(ckpt.audio_latents_mean.len(), audio_in_channels, "generate: audio_latents_mean length must equal audio_in_channels");
    assert_eq!(ckpt.audio_latents_std.len(), audio_in_channels, "generate: audio_latents_std length must equal audio_in_channels");
    let mut audio_channels_out = Vec::with_capacity(AUDIO_CHANNELS as usize);
    for c in 0..AUDIO_CHANNELS as usize {
        // Transpose this channel's `[T, C]` row-major block to the `[C, T]`
        // channel-major layout `crate::vocoder::decode` reads, denormalizing
        // by the audio VAE's own per-channel mean/std on the way.
        let mut z = vec![0f32; audio_in_channels * num_audio_latents_gen];
        for k in 0..num_audio_latents_gen {
            let row = &generated_audio_rows[(c * num_audio_latents_gen + k) * audio_in_channels..(c * num_audio_latents_gen + k + 1) * audio_in_channels];
            for cc in 0..audio_in_channels {
                z[cc * num_audio_latents_gen + k] = row[cc] * ckpt.audio_latents_std[cc] + ckpt.audio_latents_mean[cc];
            }
        }
        let wave = crate::vocoder::decode(&ckpt.vocoder_cfg, ckpt.vocoder_tensors, &z, num_audio_latents as u32, device);
        audio_channels_out.push(wave);
    }

    Ok(GeneratedAv { video: pixels, num_video_frames: out_t, height: out_h, width: out_w, audio: audio_channels_out, sample_rate: 32000 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TAG_TEXT;

    // ------------------------------------------------------------------
    // Pure-math unit tests (no weights, no model)
    // ------------------------------------------------------------------

    #[test]
    fn resolve_canvas_size_matches_a_hand_computed_16_9_case() {
        // short_edge=768, 16:9 -> width = 768*16/9 = 1365.33, area =
        // 768*1365.33 = 1_048_576 < max_pixels (768*1344=1_032_192)? Check
        // against the real defaults instead, where the ratio pushes area
        // over budget and the scale-down branch is exercised.
        let (h, w) = resolve_canvas_size(16.0, 9.0, 32, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h > 0 && w > 0);
        // 16:9 at a 32-multiple must keep w > h.
        assert!(w > h);
    }

    #[test]
    fn resolve_canvas_size_rejects_an_out_of_range_aspect_ratio() {
        assert!(resolve_canvas_size(10.0, 1.0, 32, 768, 768 * 1344).is_err());
    }

    #[test]
    fn align_num_frames_snaps_up_to_17n_plus_5() {
        assert_eq!(align_num_frames(1, 17, 5).unwrap(), 5);
        assert_eq!(align_num_frames(5, 17, 5).unwrap(), 5);
        assert_eq!(align_num_frames(6, 17, 5).unwrap(), 22);
        assert_eq!(align_num_frames(124, 17, 5).unwrap(), 124);
    }

    #[test]
    fn video_latent_num_frames_matches_the_5n_plus_2_formula() {
        assert_eq!(video_latent_num_frames(5, 17, 5), 2);
        assert_eq!(video_latent_num_frames(124, 17, 5), 37);
    }

    #[test]
    fn audio_latent_num_frames_matches_the_documented_example() {
        assert_eq!(audio_latent_num_frames(124, 24.0, 40.0), 207);
    }

    #[test]
    fn pairwise_sum_matches_sequential_sum_below_the_base_case() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        assert!((pairwise_sum(&a) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_sum_matches_sequential_sum_closely_above_128() {
        let a: Vec<f64> = (0..300).map(|i| (i as f64) * 0.37 + 1.0).collect();
        let seq: f64 = a.iter().sum();
        let pw = pairwise_sum(&a);
        // "A couple of ULPs" per porting.md's own bar for schedule math, not
        // bit-identical to a naive sequential sum (that is the whole point
        // of pairwise summation existing).
        assert!((pw - seq).abs() < 1e-6 * seq.abs().max(1.0), "pairwise {pw} vs sequential {seq}");
    }

    #[test]
    fn patchify_then_unpatchify_round_trips() {
        let (c, t, h, w) = (2u32, 2u32, 4u32, 4u32);
        let n = (c * t * h * w) as usize;
        let latents: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let patch = [1u32, 2, 2];
        let rows = patchify_video(&latents, c, t, h, w, patch);
        assert_eq!(rows.len(), n);
        let back = unpatchify_video(&rows, c, t, h, w, patch);
        assert_eq!(back, latents);
    }

    #[test]
    fn build_packed_sequence_t2va_has_no_condition_rows() {
        let text_tags = vec![TAG_TEXT; 3];
        let packed = build_packed_sequence(&text_tags, 4, 2, 2, 5, [1, 2, 2], 2, &[]);
        assert_eq!(packed.num_condition_video_rows, 0);
        assert_eq!(packed.num_condition_audio_rows, 0);
        // text(3) + audio(5*2=10) + video(4*1=4) = 17
        assert_eq!(packed.token_tags.len(), 17);
        assert_eq!(packed.text_indices, vec![0, 1, 2]);
        assert_eq!(packed.audio_indices, (3..13).collect::<Vec<u32>>());
        assert_eq!(packed.video_indices, (13..17).collect::<Vec<u32>>());
    }

    #[test]
    fn build_packed_sequence_fl2va_reserves_one_condition_block_per_anchor() {
        let text_tags = vec![TAG_TEXT; 2];
        let packed = build_packed_sequence(&text_tags, 4, 2, 2, 3, [1, 2, 2], 2, &[Anchor::First, Anchor::Last]);
        // rows_per_frame = 1, so 2 condition rows (one per keyframe).
        assert_eq!(packed.num_condition_video_rows, 2);
        // video_indices: condition rows first (physically right after
        // text), then target video rows - never contiguous with audio in
        // between.
        assert_eq!(packed.video_indices.len(), 2 + 4);
        assert_eq!(&packed.video_indices[..2], &[2u32, 3u32], "condition rows sit right after the 2 text rows");
    }

    #[test]
    fn build_packed_sequence_first_and_last_anchors_get_different_rotary_times() {
        let text_tags = vec![TAG_TEXT; 1];
        let packed = build_packed_sequence(&text_tags, 8, 2, 2, 1, [1, 2, 2], 2, &[Anchor::First, Anchor::Last]);
        let row_first = packed.video_indices[0] as usize;
        let row_last = packed.video_indices[1] as usize;
        let t_first = packed.position_ids[row_first * 3];
        let t_last = packed.position_ids[row_last * 3];
        assert_ne!(t_first, t_last, "a 'first' and a 'last' anchor at 8 latent frames must land at different rotary times");
        assert!(t_last > t_first, "the 'last' anchor must sit later on the rotary clock than the 'first' anchor");
    }

    #[test]
    fn build_row_timesteps_pins_condition_rows_and_lets_text_inherit_video() {
        // 1 text row, 1 condition video row, 1 generated video row, 1
        // generated audio row.
        let video_indices = vec![1u32, 2u32];
        let audio_indices = vec![3u32];
        let (unique_ts, idx) = build_row_timesteps(&video_indices, &audio_indices, 1, 0, 1, 0.3, 0.7, 0.999, 1.0);
        // row order: text(0), cond-video(1), gen-video(2), gen-audio(3).
        assert_eq!(unique_ts.len(), idx.iter().collect::<std::collections::HashSet<_>>().len());
        let ts = |row: usize| unique_ts[idx[row] as usize];
        assert_eq!(ts(0), 0.3, "text row inherits the video timestep");
        assert_eq!(ts(1), 0.999, "condition video row is pinned to keyframe_noise_aug");
        assert_eq!(ts(2), 0.3, "generated video row gets the video timestep");
        assert_eq!(ts(3), 0.7, "generated audio row gets the audio timestep");
    }

    // ------------------------------------------------------------------
    // Weight-free tiny-config end-to-end smoke test
    // ------------------------------------------------------------------

    /// A DiT config whose `in_channels`/`audio_in_channels` are chosen to
    /// match this test's own tiny video/audio VAE configs below - the three
    /// phases that built [`H3TransformerConfig::tiny`],
    /// [`VideoVaeConfig::tiny`] and `vocoder`'s own private `tiny_cfg` each
    /// picked proportions independently and never needed to agree with each
    /// other's latent widths, since none of them exercised the full
    /// pipeline. This module's own end-to-end test is the first one that
    /// does, so it builds its own mutually-consistent trio rather than
    /// reusing those three `tiny()` presets as-is.
    fn tiny_dit_cfg() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 4,
            attention_head_dim: 32,
            hidden_size: 20,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 64,
            in_channels: 4,       // == tiny_video_vae_cfg().latent_channels
            audio_in_channels: 4, // == tiny_vocoder_cfg().vae_latent_channels
            patch_size: [1, 2, 2],
            text_dim: 12,
            freq_dim: 16,
            time_embed_hidden_dim: 40,
            time_embed_dim: 24,
            rope_freq_dim: 4,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    fn tiny_video_vae_cfg() -> VideoVaeConfig {
        VideoVaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: [8, 16, 16, 32, 32, 64],
            layers_per_block: 2,
            spatial_downsample_factors: [2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: [1, 2, 2, 1, 1, 1],
            norm_num_groups: 2,
            norm_eps: 1e-6,
            decoder_num_layers: 2,
            decoder_num_attention_heads: 2,
            decoder_attention_head_dim: 8,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
        }
    }

    fn tiny_vocoder_cfg() -> VocoderConfig {
        VocoderConfig {
            vae_latent_channels: 4,
            upsample_initial_channel: 128,
            upsample_rates: [2, 2, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: [4, 4, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 5, 7],
            resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            mel_channels: 12,
            out_channels: 1,
        }
    }

    /// Duplicates `crate::model`'s own private test-only `rand_tensors` (not
    /// reusable from here: it is `#[cfg(test)]`-private to that module) -
    /// the exact tensor names `H3Transformer::load` reads, seeded random
    /// values.
    fn rand_dit_tensors(cfg: &H3TransformerConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        let mut t: Tensors = std::collections::HashMap::new();
        let mut put = |name: &str, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            t.insert(name.to_string(), (shape, rng.vec_scaled(n, 0.2)));
        };
        let (hidden, inner, hd, ffn, te) = (cfg.hidden_size as usize, cfg.inner_dim() as usize, cfg.attention_head_dim as usize, cfg.ffn_dim as usize, cfg.time_embed_dim as usize);

        let put_attn = |name: &mut dyn FnMut(&str, Vec<usize>), prefix: &str| {
            name(&format!("{prefix}.to_q.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.to_k.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.to_v.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.norm_q.weight"), vec![hd]);
            name(&format!("{prefix}.norm_k.weight"), vec![hd]);
            name(&format!("{prefix}.to_out.0.weight"), vec![hidden, inner]);
        };
        let put_ff = |name: &mut dyn FnMut(&str, Vec<usize>), prefix: &str| {
            name(&format!("{prefix}.ff.net.0.proj.weight"), vec![2 * ffn, hidden]);
            name(&format!("{prefix}.ff.net.2.weight"), vec![hidden, ffn]);
        };

        for i in 0..cfg.num_refiner_layers {
            let p = format!("token_refiner.refiner_blocks.{i}");
            put_attn(&mut put, &format!("{p}.attn"));
            put_ff(&mut put, &p);
            put(&format!("{p}.norm1.weight"), vec![hidden]);
            put(&format!("{p}.norm2.weight"), vec![hidden]);
        }
        put("token_refiner.final_norm.weight", vec![hidden]);

        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            put_attn(&mut put, &format!("{p}.attn"));
            put_ff(&mut put, &p);
            put(&format!("{p}.norm1.weight"), vec![hidden]);
            put(&format!("{p}.norm2.weight"), vec![hidden]);
            put(&format!("{p}.adaln_proj.linear.weight"), vec![18 * hidden, te]);
            put(&format!("{p}.adaln_proj.linear.bias"), vec![18 * hidden]);
        }

        put("proj_in.weight", vec![hidden, cfg.video_patch_dim() as usize]);
        put("proj_in.bias", vec![hidden]);
        put("audio_proj_in.weight", vec![hidden, cfg.audio_in_channels as usize]);
        put("audio_proj_in.bias", vec![hidden]);
        put("context_embedder.weight", vec![hidden, cfg.text_dim as usize]);
        put("context_embedder.bias", vec![hidden]);
        put("time_embedder.linear_1.weight", vec![cfg.time_embed_hidden_dim as usize, cfg.freq_dim as usize]);
        put("time_embedder.linear_1.bias", vec![cfg.time_embed_hidden_dim as usize]);
        put("time_embedder.linear_2.weight", vec![te, cfg.time_embed_hidden_dim as usize]);
        put("time_embedder.linear_2.bias", vec![te]);
        put("norm_out.norm.weight", vec![hidden]);
        put("norm_out.linear.weight", vec![2 * hidden, te]);
        put("norm_out.linear.bias", vec![2 * hidden]);
        put("proj_out.weight", vec![cfg.video_patch_dim() as usize, hidden]);
        put("proj_out.bias", vec![cfg.video_patch_dim() as usize]);
        put("audio_proj_out.weight", vec![cfg.audio_in_channels as usize, hidden]);
        put("audio_proj_out.bias", vec![cfg.audio_in_channels as usize]);
        t
    }

    fn rand_manifest_tensors(manifest: Vec<(String, Vec<usize>)>, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        manifest
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let vals = rng.vec_scaled(n, 0.2);
                (name, (shape, vals))
            })
            .collect()
    }

    fn tiny_checkpoint<'a>(dit_tensors: &'a Tensors, video_vae_tensors: &'a Tensors, vocoder_tensors: &'a Tensors) -> H3Checkpoint<'a> {
        let video_vae_cfg = tiny_video_vae_cfg();
        let vocoder_cfg = tiny_vocoder_cfg();
        H3Checkpoint {
            dit_tensors,
            dit_cfg: tiny_dit_cfg(),
            video_vae_tensors,
            video_vae_cfg: video_vae_cfg.clone(),
            vocoder_tensors,
            vocoder_cfg,
            video_latents_mean: vec![0.0; video_vae_cfg.latent_channels as usize],
            video_latents_std: vec![1.0; video_vae_cfg.latent_channels as usize],
            audio_latents_mean: vec![0.0; 4],
            audio_latents_std: vec![1.0; 4],
        }
    }

    fn tiny_text(num_tokens: usize, text_dim: usize, seed: u64) -> TextConditioning {
        let mut rng = data::rng::Lcg::new(seed);
        TextConditioning { embeds: rng.vec_scaled(num_tokens * text_dim, 0.3), token_tags: vec![TAG_TEXT; num_tokens] }
    }

    /// The smallest request geometry that still satisfies MiniMax-H3's real
    /// 5-15 second duration bound (`generate`'s own validation, not
    /// bypassed) - the "tiny" in this test's name is the WEIGHT/channel
    /// scale, not the frame count, since the duration bound is a genuine
    /// request-level business rule this pipeline enforces for a real
    /// caller too.
    fn tiny_gen_opts(seed: u64) -> GenOpts {
        GenOpts { canvas: Some((32, 32)), num_frames: 124, num_inference_steps: 3, seed, device: Some("cpu".to_string()) }
    }

    /// Phase 9's own end-to-end gate: encode (pre-built) -> pack -> denoise
    /// N steps -> decode, weight-free, exercising every piece this phase
    /// added together for the first time. `t2va` first, per this phase's own
    /// scope ("build and gate this before fl2va").
    #[test]
    fn t2va_tiny_config_end_to_end_is_finite_and_the_right_shape() {
        let dit_cfg = tiny_dit_cfg();
        let video_vae_cfg = tiny_video_vae_cfg();
        let vocoder_cfg = tiny_vocoder_cfg();
        let dit_tensors = rand_dit_tensors(&dit_cfg, 1);
        let video_vae_tensors = rand_manifest_tensors(video_vae_cfg.tensor_manifest(), 2);
        let vocoder_tensors = rand_manifest_tensors(vocoder_cfg.tensor_manifest(), 3);
        let ckpt = tiny_checkpoint(&dit_tensors, &video_vae_tensors, &vocoder_tensors);

        let text = tiny_text(3, dit_cfg.text_dim as usize, 4);
        let opts = tiny_gen_opts(5);

        let av = t2va(&ckpt, &text, &opts).expect("t2va");

        assert_eq!(av.video.len(), (3 * av.num_video_frames * av.height * av.width) as usize);
        assert!(av.video.iter().all(|v| v.is_finite()), "video output must be finite");
        assert!(av.video.iter().all(|&v| (0.0..=1.0).contains(&v)), "video output must be clamped to [0,1]");
        assert!(av.video.iter().any(|&v| v != 0.0), "video output must not be trivially all-zero");

        assert_eq!(av.audio.len(), AUDIO_CHANNELS as usize);
        for ch in &av.audio {
            assert!(ch.iter().all(|v| v.is_finite()), "audio output must be finite");
            assert!(ch.iter().all(|&v| (-1.0..=1.0).contains(&v)), "audio output must be clamped to [-1,1]");
            assert!(ch.iter().any(|&v| v != 0.0), "audio output must not be trivially all-zero");
        }
    }

    /// [`t2va_tiny_config_end_to_end_is_finite_and_the_right_shape`] plus one
    /// `"first"`-anchored keyframe conditioning block - the same pieces plus
    /// [`encode_keyframe_condition`] and a non-empty `condition_rows` prefix
    /// riding through every denoise step unchanged.
    #[test]
    fn fl2va_tiny_config_end_to_end_is_finite_and_the_right_shape() {
        let dit_cfg = tiny_dit_cfg();
        let video_vae_cfg = tiny_video_vae_cfg();
        let vocoder_cfg = tiny_vocoder_cfg();
        let dit_tensors = rand_dit_tensors(&dit_cfg, 11);
        let video_vae_tensors = rand_manifest_tensors(video_vae_cfg.tensor_manifest(), 12);
        let vocoder_tensors = rand_manifest_tensors(vocoder_cfg.tensor_manifest(), 13);
        let ckpt = tiny_checkpoint(&dit_tensors, &video_vae_tensors, &vocoder_tensors);

        let text = tiny_text(3, dit_cfg.text_dim as usize, 14);
        let opts = tiny_gen_opts(15);

        let mut rng = data::rng::Lcg::new(16);
        let keyframe_pixels = rng.vec_scaled((3 * 32 * 32) as usize, 128.0); // ~[0,255]-ish range
        let keyframe_pixels: Vec<f32> = keyframe_pixels.iter().map(|&v| (v + 128.0).clamp(0.0, 255.0)).collect();
        let kf = encode_keyframe_condition(&video_vae_cfg, &video_vae_tensors, Some("cpu"), &keyframe_pixels, 32, 32, &ckpt.video_latents_mean, &ckpt.video_latents_std, dit_cfg.patch_size, Anchor::First, 17);

        let av = fl2va(&ckpt, &text, &[kf], &opts).expect("fl2va");

        assert_eq!(av.video.len(), (3 * av.num_video_frames * av.height * av.width) as usize);
        assert!(av.video.iter().all(|v| v.is_finite()), "video output must be finite");
        assert!(av.video.iter().any(|&v| v != 0.0), "video output must not be trivially all-zero");
        assert_eq!(av.audio.len(), AUDIO_CHANNELS as usize);
        for ch in &av.audio {
            assert!(ch.iter().all(|v| v.is_finite()), "audio output must be finite");
        }
    }

    #[test]
    fn fl2va_rejects_an_empty_keyframe_list() {
        let dit_cfg = tiny_dit_cfg();
        let video_vae_cfg = tiny_video_vae_cfg();
        let vocoder_cfg = tiny_vocoder_cfg();
        let dit_tensors = rand_dit_tensors(&dit_cfg, 21);
        let video_vae_tensors = rand_manifest_tensors(video_vae_cfg.tensor_manifest(), 22);
        let vocoder_tensors = rand_manifest_tensors(vocoder_cfg.tensor_manifest(), 23);
        let ckpt = tiny_checkpoint(&dit_tensors, &video_vae_tensors, &vocoder_tensors);
        let text = tiny_text(2, dit_cfg.text_dim as usize, 24);
        let opts = tiny_gen_opts(25);
        assert!(fl2va(&ckpt, &text, &[], &opts).is_err());
    }

    // ------------------------------------------------------------------
    // Real-geometry numeric parity for the pipeline GLUE
    // ------------------------------------------------------------------

    /// The geometry a real 128x128 / 124-frame `t2va` request resolves to -
    /// the same constants `tools/minimaxh3_layout_dump_reference.py` dumps at.
    const GOLDEN_LATENT_H: u32 = 8;
    const GOLDEN_LATENT_W: u32 = 8;
    const GOLDEN_LATENT_FRAMES: u32 = 37;
    const GOLDEN_AUDIO_LATENTS: u32 = 207;
    const GOLDEN_TEXT_TOKENS: usize = 37;
    const GOLDEN_PATCH: [u32; 3] = [1, 2, 2];
    const GOLDEN_LATENT_CHANNELS: u32 = 24;
    const GOLDEN_STEPS: usize = 20;

    /// `crate::model`'s real-weight golden proves the DiT's own block math is
    /// right GIVEN a correctly packed input; it hand-builds that input, so
    /// everything that ASSEMBLES it is invisible to it. This test closes that
    /// gap: it replays the REAL installed `diffusers==0.40.0`
    /// `build_packed_sequence` / `build_row_timesteps` / `MiniMaxH3Scheduler` /
    /// `patchify_video_latents` outputs, dumped at the geometry an actual
    /// generation runs at (not a toy - 37 latent frames wraps the
    /// `(1,4,4,4,4)` rotary spacing seven times), against this module's own
    /// equivalents.
    ///
    /// One thing this does NOT cover: the shift's `unique_consecutive` pass
    /// never actually collapses anything at either of H3's two shifts - the
    /// float32 grid stays strictly decreasing at every step count checked up
    /// to 5000, so `sigmas.len() == num_inference_steps` throughout and that
    /// branch is unreachable in practice. It is ported because the reference
    /// has it, not because it fires.
    ///
    /// Both the `t2va` (no keyframes) and the `fl2va` (`first`+`last` anchors,
    /// which is what exercises the pairwise-summed `"last"` anchor time and
    /// the conditioning-row pinning) layouts are covered.
    #[test]
    fn pipeline_layout_matches_the_real_reference_numerically() {
        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/layout");
        let fixture_file = fixture_dir.join("minimaxh3_layout.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/minimaxh3_layout_dump_reference.py --out {}", fixture_file.display(), fixture_dir.display()));
            return;
        }
        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/minimaxh3_layout_dump_reference.py") else {
            return;
        };
        let ok = src.require(&[
            ("latent_height", GOLDEN_LATENT_H as i64),
            ("latent_width", GOLDEN_LATENT_W as i64),
            ("num_latent_frames", GOLDEN_LATENT_FRAMES as i64),
            ("num_audio_latents", GOLDEN_AUDIO_LATENTS as i64),
            ("num_text_tokens", GOLDEN_TEXT_TOKENS as i64),
            ("patch_t", GOLDEN_PATCH[0] as i64),
            ("patch_h", GOLDEN_PATCH[1] as i64),
            ("patch_w", GOLDEN_PATCH[2] as i64),
            ("audio_channels", AUDIO_CHANNELS as i64),
            ("latent_channels", GOLDEN_LATENT_CHANNELS as i64),
            ("num_inference_steps", GOLDEN_STEPS as i64),
            ("video_shift_x1000", (crate::schedule::H3_VIDEO_SHIFT * 1000.0) as i64),
            ("audio_shift_x1000", (crate::schedule::H3_AUDIO_SHIFT * 1000.0) as i64),
        ]);
        if !ok {
            return;
        }

        let raw = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = raw.into_iter().map(|t| (t.name.clone(), t)).collect();
        let get = |name: &str| -> &[f32] { &fx.get(name).unwrap_or_else(|| panic!("golden fixture entry {name:?} missing")).data };
        let get_u32 = |name: &str| -> Vec<u32> { get(name).iter().map(|&v| v.round() as u32).collect() };

        let mut r = brain_testutil::parity::Report::new(0.9999);

        for (prefix, anchors) in [("t2va", &[][..]), ("fl2va", &[Anchor::First, Anchor::Last][..])] {
            let text_tags = get_u32(&format!("{prefix}_text_token_tags"));
            assert_eq!(text_tags.len(), GOLDEN_TEXT_TOKENS, "{prefix}: golden text-tag count");
            let packed = build_packed_sequence(&text_tags, GOLDEN_LATENT_FRAMES, GOLDEN_LATENT_H, GOLDEN_LATENT_W, GOLDEN_AUDIO_LATENTS, GOLDEN_PATCH, AUDIO_CHANNELS, anchors);

            // The rotary grid is the highest-value comparison here: it is
            // per-token, and lesson #49's own generalizable point is that a
            // wrong per-token quantity produces confident structured garbage
            // rather than an obviously broken picture.
            r.check(&format!("{prefix} position_ids"), &packed.position_ids, get(&format!("{prefix}_position_ids")));

            // Index/tag arrays are exact integers, so they are asserted
            // equal outright rather than run through a cosine.
            assert_eq!(packed.token_tags, get_u32(&format!("{prefix}_token_tags")), "{prefix}: token_tags");
            assert_eq!(packed.video_indices, get_u32(&format!("{prefix}_video_indices")), "{prefix}: video_indices");
            assert_eq!(packed.audio_indices, get_u32(&format!("{prefix}_audio_indices")), "{prefix}: audio_indices");
            assert_eq!(packed.text_indices, get_u32(&format!("{prefix}_text_indices")), "{prefix}: text_indices");
            let counts = get_u32(&format!("{prefix}_row_counts"));
            assert_eq!(packed.num_condition_video_rows as u32, counts[0], "{prefix}: num_condition_video_rows");
            assert_eq!(packed.num_condition_audio_rows as u32, counts[1], "{prefix}: num_condition_audio_rows");
            assert_eq!(packed.token_tags.len() as u32, counts[2], "{prefix}: sequence_length");

            // The full per-step row-timestep plan, driven off this port's own
            // two schedules exactly as `generate` drives it.
            let mut sched = DualSchedule::new();
            sched.set_timesteps(GOLDEN_STEPS);
            let unique_counts = get_u32(&format!("{prefix}_row_unique_counts"));
            let golden_unique = get(&format!("{prefix}_row_unique_timesteps"));
            let golden_indices = get_u32(&format!("{prefix}_row_timestep_indices"));
            assert_eq!(sched.num_steps(), unique_counts.len(), "{prefix}: step count");
            let seq_len = packed.token_tags.len();
            let mut cursor = 0usize;
            for step in 0..sched.num_steps() {
                let (video_ts, audio_ts) = sched.current_timesteps(step);
                let condition_video_ts = video_ts.max(KEYFRAME_NOISE_AUG);
                let (unique_ts, ts_idx) = build_row_timesteps(&packed.video_indices, &packed.audio_indices, packed.num_condition_video_rows, packed.num_condition_audio_rows, packed.text_indices.len(), video_ts, audio_ts, condition_video_ts, 1.0);
                let n = unique_counts[step] as usize;
                assert_eq!(unique_ts.len(), n, "{prefix} step {step}: number of distinct timesteps");
                // A handful of scalars, one of which is legitimately exactly
                // 0.0 (at step 0 the shift maps sigma=1 to 1 under both
                // shifts, so t = 1 - sigma = 0 for video and audio alike).
                // Compared elementwise rather than through the cosine report,
                // which - rightly - refuses an all-zero reference as
                // degenerate.
                for (k, (&got, &want)) in unique_ts.iter().zip(&golden_unique[cursor..cursor + n]).enumerate() {
                    assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "{prefix} step {step}: unique timestep {k} is {got}, reference has {want}");
                }
                assert_eq!(ts_idx, golden_indices[step * seq_len..(step + 1) * seq_len], "{prefix} step {step}: per-row timestep index");
                cursor += n;
            }
            assert_eq!(cursor, golden_unique.len(), "{prefix}: consumed the whole ragged timestep plan");
        }

        // Both shifted-sigma schedules, plus one real Euler step each - the
        // step is what catches a swapped sigma source or a flipped velocity
        // sign, neither of which the sigma grid alone can see.
        for (name, shift) in [("video", crate::schedule::H3_VIDEO_SHIFT), ("audio", crate::schedule::H3_AUDIO_SHIFT)] {
            let mut sched = H3Scheduler::new(shift);
            sched.set_timesteps(GOLDEN_STEPS);
            r.check(&format!("sched {name} sigmas"), sched.sigmas(), get(&format!("sched_{name}_sigmas")));
            r.check(&format!("sched {name} timesteps"), sched.timesteps(), get(&format!("sched_{name}_timesteps")));
            let step_index = get_u32(&format!("sched_{name}_step_index"))[0] as usize;
            let sample = get(&format!("sched_{name}_step_sample"));
            let velocity = get(&format!("sched_{name}_step_velocity"));
            let stepped = sched.step(velocity, sched.timesteps()[step_index], sample, step_index);
            r.check(&format!("sched {name} euler step"), &stepped, get(&format!("sched_{name}_step_out")));
        }

        let scaled = H3Scheduler::scale_noise(get("scale_noise_clean"), KEYFRAME_NOISE_AUG, get("scale_noise_noise"));
        r.check("scale_noise at keyframe_noise_aug", &scaled, get("scale_noise_out"));

        // The DiT-level patchify at the real latent shape, and its inverse
        // against the reference's OWN reshape/permute (not merely against
        // this module's forward direction, which a matched pair of wrong
        // permutations would satisfy).
        let latents = get("patchify_latents");
        let rows = patchify_video(latents, GOLDEN_LATENT_CHANNELS, GOLDEN_LATENT_FRAMES, GOLDEN_LATENT_H, GOLDEN_LATENT_W, GOLDEN_PATCH);
        r.check("patchify_video rows", &rows, get("patchify_rows"));
        let back = unpatchify_video(get("patchify_rows"), GOLDEN_LATENT_CHANNELS, GOLDEN_LATENT_FRAMES, GOLDEN_LATENT_H, GOLDEN_LATENT_W, GOLDEN_PATCH);
        r.check("unpatchify_video latents", &back, get("patchify_roundtrip"));

        r.finish("minimaxh3 pipeline layout/schedule/patchify vs the real reference");
    }
}
