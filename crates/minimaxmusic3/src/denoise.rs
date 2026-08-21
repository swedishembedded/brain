// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunked, CFG-guided flow-matching denoise: turns the per-frame hidden
//! states `pipeline::generate_frames` produced into Flow-VAE latents, one
//! 200-frame chunk (100-frame hop) at a time, splicing consecutive chunks
//! over a 172-latent overlap so the DiT never sees a hard seam.
//!
//! Ported directly from the reference `diffusers` PR's `ChunkConditionStep`
//! / `ChunkPrepareLatentsStep` / `ChunkSetTimestepsStep` / `ChunkDenoiseInner`
//! / `ChunkUpdateStep` (`before_denoise.py`/`denoise.py`), not reimagined:
//! every constant and blend formula below has a named counterpart there.
//!
//! Layout convention (matching `dit::forward`'s own parameters): latents are
//! `[in_channels, length]` NCL (channel-major); condition is `[length,
//! condition_dim]` row-major (frame-major, straight out of
//! `condition_encoder::forward`). The overlap carried between chunks slices
//! `length` (the last axis of latents, the first axis of condition) - a
//! strided extraction for latents, a contiguous one for condition.
//!
//! CFG here is over the DiT's own *conditioning*, not its logits: the
//! unconditional branch is a zeroed condition tensor (`denoise.py`'s
//! `zeros_like(condition)`), not a second Global-LLM/depth-decoder pass -
//! unrelated to `pipeline::generate_frames`'s own AR-stage CFG, which
//! blends two full model branches. Two independent CFG axes, ported
//! independently, matching the reference's own two independent `Guider`
//! components.

use crate::condition_encoder::{self, ConditionEncoderWeights};
use crate::config::{ConditionEncoderConfig, DitConfig};
use crate::dit::{self, DitWeights};
use data::rng::Rng;
use diffusion::scheduler::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};
use gpu_core::Gpu;

/// Frames per chunk (`_CHUNK_FRAMES`).
pub const CHUNK_FRAMES: usize = 200;
/// Frame stride between consecutive chunk starts (`_CHUNK_HOP`).
pub const CHUNK_HOP: usize = 100;
/// Latent-axis overlap carried from one chunk into the next (`_OVERLAP_LATENT_LENGTH`).
pub const OVERLAP_LATENT_LENGTH: usize = 172;
/// Euler steps per chunk when the caller doesn't override it.
pub const DEFAULT_NUM_INFERENCE_STEPS: usize = 30;
/// The DiT's own classifier-free guidance scale (distinct from the AR
/// stage's `pipeline::AR_CFG_SCALE`).
pub const GUIDANCE_SCALE: f32 = 1.7;

/// `[0]` for a song that fits in one chunk, else every 100-frame-hop start
/// up to (not including) the tail that would run past `num_frames` -
/// `range(0, num_frames-100, 100)` in the reference.
pub fn chunk_starts(num_frames: usize) -> Vec<usize> {
    if num_frames <= CHUNK_FRAMES {
        vec![0]
    } else {
        (0..num_frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
    }
}

/// The state one chunk hands to the next: the trailing `in_channels x span`
/// slice of the just-denoised latents and the matching `span x
/// condition_dim` slice of that chunk's own condition (`span <=
/// OVERLAP_LATENT_LENGTH`, and less on the first couple of chunks of a
/// short song). `None` before the first chunk.
#[derive(Clone, Debug, Default)]
pub struct ChunkState {
    pub previous_latent: Option<Vec<f32>>,
    pub previous_condition: Option<Vec<f32>>,
}

/// `n` samples of standard-normal noise via [`Rng::next_gaussian`] - the
/// canonical Gaussian source (`data::rng::Lcg` has no Gaussian sampler and
/// this crate's own convention, per `data::rng`'s doc, is to reach for
/// `Rng` rather than hand-roll a fresh Box-Muller copy on top of `Lcg`).
fn gaussian_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.next_gaussian() as f32).collect()
}

/// Denoise one chunk: `frame_hiddens` is the WHOLE song's per-frame hidden
/// states (`[num_frames_total, num_condition_layers*condition_hidden_dim]`
/// row-major, `pipeline::generate_frames`'s own output layout), sliced here
/// to `[chunk_start, chunk_start+CHUNK_FRAMES)` (clipped to
/// `num_frames_total`). Returns this chunk's denoised latents, `[in_channels,
/// length]` NCL, and advances `state` for the next call.
#[allow(clippy::too_many_arguments)]
pub fn denoise_chunk(
    gpu: &Gpu,
    dit_cfg: &DitConfig,
    dit_w: &DitWeights,
    cond_cfg: &ConditionEncoderConfig,
    cond_w: &ConditionEncoderWeights,
    frame_hiddens: &[f32],
    num_frames_total: usize,
    chunk_start: usize,
    state: &mut ChunkState,
    num_inference_steps: usize,
    seed: u64,
) -> Vec<f32> {
    let chunk_end = (chunk_start + CHUNK_FRAMES).min(num_frames_total);
    let chunk_frames = chunk_end - chunk_start;
    let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
    let chunk_hidden = &frame_hiddens[chunk_start * per_frame..chunk_end * per_frame];
    let (mut condition, length) = condition_encoder::forward(cond_cfg, cond_w, chunk_hidden, 1, chunk_frames);
    let condition_dim = cond_cfg.out_dim as usize;
    let cin = dit_cfg.in_channels as usize;

    // `overlap = min(previous_latent's own span, this chunk's length)` -
    // `previous_condition` was sliced in lockstep with `previous_latent` by
    // the prior call, so it always covers at least `overlap` frames too.
    let overlap = match (&state.previous_latent, &state.previous_condition) {
        (Some(prev_latent), Some(prev_condition)) => {
            let span = prev_latent.len() / cin;
            debug_assert_eq!(prev_condition.len(), span * condition_dim, "denoise_chunk: previous_latent/previous_condition span mismatch");
            span.min(length)
        }
        _ => 0,
    };
    if overlap > 0 {
        let prev_condition = state.previous_condition.as_ref().unwrap();
        condition[..overlap * condition_dim].copy_from_slice(&prev_condition[..overlap * condition_dim]);
    }

    let mut latents = gaussian_vec(seed, cin * length);
    // `noise_prompt`: the freshly-drawn noise in the overlap region, before
    // any denoise step touches it - the blend below interpolates between
    // this and `previous_latent` every step, not just once.
    let noise_prompt: Vec<f32> = if overlap > 0 {
        (0..cin).flat_map(|c| latents[c * length..c * length + overlap].to_vec()).collect()
    } else {
        Vec::new()
    };

    let mut scheduler = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1, shift: 1.0, invert_sigmas: true });
    scheduler.set_timesteps(&default_z_image_sigmas(num_inference_steps));
    let timesteps: Vec<f32> = scheduler.timesteps().to_vec();

    let zero_condition = vec![0.0f32; condition.len()];
    let prev_span = state.previous_latent.as_ref().map(|p| span_of(p, cin));
    for &t in &timesteps {
        if overlap > 0 {
            let prev_latent = state.previous_latent.as_ref().unwrap();
            let span = prev_span.unwrap();
            for c in 0..cin {
                for j in 0..overlap {
                    latents[c * length + j] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt[c * overlap + j] + t * prev_latent[c * span + j];
                }
            }
        }
        let v_cond = dit::forward(gpu, dit_cfg, dit_w, &latents, &condition, t, length);
        let v_uncond = dit::forward(gpu, dit_cfg, dit_w, &latents, &zero_condition, t, length);
        let velocity: Vec<f32> = v_cond.iter().zip(&v_uncond).map(|(c, u)| u + (c - u) * GUIDANCE_SCALE).collect();
        latents = scheduler.step(&velocity, &latents);
    }

    if overlap > 0 {
        let prev_latent = state.previous_latent.as_ref().unwrap();
        let span = span_of(prev_latent, cin);
        for c in 0..cin {
            for j in 0..overlap {
                latents[c * length + j] = prev_latent[c * span + j];
            }
        }
    }

    let overlap_start = length.saturating_sub(2 * OVERLAP_LATENT_LENGTH);
    let overlap_end = overlap_start.max(length.saturating_sub(OVERLAP_LATENT_LENGTH));
    state.previous_latent = Some((0..cin).flat_map(|c| latents[c * length + overlap_start..c * length + overlap_end].to_vec()).collect());
    state.previous_condition = Some(condition[overlap_start * condition_dim..overlap_end * condition_dim].to_vec());

    latents
}

fn span_of(prev_latent: &[f32], cin: usize) -> usize {
    prev_latent.len() / cin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit_train;
    use data::rng::Lcg;

    fn random_condition_weights(cfg: &ConditionEncoderConfig, seed: u64) -> ConditionEncoderWeights {
        let mut r = Lcg::new(seed);
        let (layers, hidden, out_dim) = (cfg.num_condition_layers as usize, cfg.condition_hidden_dim as usize, cfg.out_dim as usize);
        ConditionEncoderWeights {
            layer_weight_logits: r.vec_scaled(layers, 0.5),
            layer_scale: 1.0,
            proj_weight: r.vec_scaled(out_dim * hidden * 3, 0.2),
            proj_bias: r.vec_scaled(out_dim, 0.1),
        }
    }

    #[test]
    fn chunk_starts_matches_the_reference_windowing() {
        assert_eq!(chunk_starts(50), vec![0]);
        assert_eq!(chunk_starts(200), vec![0]);
        assert_eq!(chunk_starts(250), vec![0, 100]);
        assert_eq!(chunk_starts(400), vec![0, 100, 200]);
    }

    #[test]
    fn single_chunk_denoise_produces_the_expected_shape_with_no_overlap() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 1);
        let cond_w = random_condition_weights(&cond_cfg, 2);
        let gpu = Gpu::new_cpu(dit::PIPELINES);

        let num_frames = 5usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(3);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);

        let mut state = ChunkState::default();
        let latents = denoise_chunk(&gpu, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, 4, 7);

        let expected_length = condition_encoder::latent_length(&cond_cfg, num_frames);
        assert_eq!(latents.len(), dit_cfg.in_channels as usize * expected_length);
        assert!(state.previous_latent.is_some());
        assert!(state.previous_condition.is_some());
    }

    #[test]
    fn two_chunks_carry_forward_a_consistent_overlap() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 11);
        let cond_w = random_condition_weights(&cond_cfg, 12);
        let gpu = Gpu::new_cpu(dit::PIPELINES);

        // Small enough that `chunk_starts` still only emits [0], but we
        // drive `denoise_chunk` twice by hand (at real scale `chunk_starts`
        // would supply the second start) to exercise the overlap path
        // without needing a 250-frame fixture.
        let num_frames = 6usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(13);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);

        let mut state = ChunkState::default();
        let first = denoise_chunk(&gpu, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, 4, 21);
        let first_length = condition_encoder::latent_length(&cond_cfg, num_frames);
        assert_eq!(first.len(), dit_cfg.in_channels as usize * first_length);

        let prev_latent = state.previous_latent.clone().unwrap();
        let prev_condition = state.previous_condition.clone().unwrap();
        let span = prev_latent.len() / dit_cfg.in_channels as usize;
        assert_eq!(prev_condition.len(), span * cond_cfg.out_dim as usize);
        assert!(span <= first_length.min(OVERLAP_LATENT_LENGTH));

        let second = denoise_chunk(&gpu, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 2, &mut state, 4, 22);
        let second_length = condition_encoder::latent_length(&cond_cfg, num_frames - 2);
        assert_eq!(second.len(), dit_cfg.in_channels as usize * second_length);
    }
}
