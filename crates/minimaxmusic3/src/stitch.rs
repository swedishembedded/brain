// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vocoder crop-and-stitch: turns `denoise::denoise_chunk`'s per-chunk
//! Flow-VAE latents into one continuous stereo waveform.
//!
//! Ported from the reference `diffusers` PR's `decoders.py` chunk-decode
//! step: run the vocoder over each chunk's full latent span (it needs the
//! extra context to decode cleanly at the edges), then crop away that
//! context before concatenating - `CROP_LEFT_LATENT` latents off the
//! start (except the song's first chunk, which has no left neighbor to
//! blend with) and `CROP_RIGHT_LATENT` off the end (except the last
//! chunk). Both crop widths are in *latent* units; `hop_length` (the
//! vocoder's own upsample factor) converts them to samples.

use crate::config::VocoderConfig;
use crate::vocoder::{self, VocoderWeights};
use gpu_core::Gpu;

/// Latents cropped off the start of every non-first chunk's decoded
/// waveform (`_CROP_LEFT_LATENT`).
pub const CROP_LEFT_LATENT: usize = 86;
/// Latents cropped off the end of every non-last chunk's decoded waveform
/// (`_CROP_RIGHT_LATENT`) - `2*denoise::OVERLAP_LATENT_LENGTH -
/// CROP_LEFT_LATENT`, but the reference states it as its own literal
/// constant, so this does too rather than deriving it.
pub const CROP_RIGHT_LATENT: usize = 258;

/// Samples produced per latent step: `product(upsampling_ratios)`.
pub fn hop_length(cfg: &VocoderConfig) -> usize {
    cfg.upsampling_ratios.iter().map(|&r| r as usize).product()
}

/// Accumulates cropped per-chunk waveforms into one continuous stereo
/// signal (planar: left/right accumulated separately, matching
/// `vocoder::forward`'s own `[batch, 2, samples]` output split).
#[derive(Default)]
pub struct Stitcher {
    left: Vec<f32>,
    right: Vec<f32>,
}

impl Stitcher {
    pub fn new() -> Self {
        Stitcher::default()
    }

    /// Decode one chunk's latents and append its cropped waveform.
    /// `is_first`/`is_last` are the caller's own knowledge of this chunk's
    /// position among `denoise::chunk_starts`' full list - not derivable
    /// from `latents` alone.
    pub fn push_chunk(&mut self, gpu: &Gpu, cfg: &VocoderConfig, w: &VocoderWeights, latents: &[f32], length: usize, is_first: bool, is_last: bool) {
        let waveform = vocoder::forward(gpu, cfg, w, latents, 1, length);
        let total_samples = waveform.len() / 2;
        let hop = hop_length(cfg);
        let left_crop = if is_first { 0 } else { CROP_LEFT_LATENT * hop };
        let right_crop = if is_last { 0 } else { CROP_RIGHT_LATENT * hop };
        assert!(left_crop + right_crop <= total_samples, "stitch::push_chunk: crop ({left_crop}+{right_crop}) exceeds this chunk's own {total_samples} samples");
        let start = left_crop;
        let end = total_samples - right_crop;
        self.left.extend_from_slice(&waveform[start..end]);
        self.right.extend_from_slice(&waveform[total_samples + start..total_samples + end]);
    }

    /// The final stereo waveform, `(left, right)`, each clamped to
    /// `[-1, 1]` (matching the reference's own explicit final clamp - a
    /// no-op in practice since `vocoder::forward` already runs every
    /// sample through `tanh`, but kept for exact parity with the reference
    /// rather than relying on that being true forever).
    pub fn finish(self) -> (Vec<f32>, Vec<f32>) {
        let clamp = |v: Vec<f32>| v.into_iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        (clamp(self.left), clamp(self.right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train;
    use data::rng::Lcg;

    #[test]
    fn hop_length_is_the_product_of_the_upsampling_ratios() {
        let cfg = VocoderConfig::real();
        assert_eq!(hop_length(&cfg), 8 * 8 * 4 * 2);
        let tiny = VocoderConfig::tiny();
        assert_eq!(hop_length(&tiny), 2 * 2);
    }

    #[test]
    fn single_chunk_needs_no_crop_and_matches_vocoder_forward_directly() {
        let cfg = VocoderConfig::tiny();
        let w = train::random_weights(&cfg, 1);
        let gpu = Gpu::new_cpu(vocoder::PIPELINES);
        let length = 5usize;
        let mut r = Lcg::new(2);
        let latents = r.vec_scaled(cfg.latent_channels as usize * length, 0.3);

        let direct = vocoder::forward(&gpu, &cfg, &w, &latents, 1, length);
        let mut stitcher = Stitcher::new();
        stitcher.push_chunk(&gpu, &cfg, &w, &latents, length, true, true);
        let (left, right) = stitcher.finish();

        let total_samples = direct.len() / 2;
        assert_eq!(left.len(), total_samples);
        assert_eq!(right.len(), total_samples);
        assert_eq!(left, direct[..total_samples]);
        assert_eq!(right, direct[total_samples..]);
    }

    #[test]
    fn two_chunks_crop_the_shared_edge_and_concatenate() {
        let cfg = VocoderConfig::tiny();
        let w = train::random_weights(&cfg, 3);
        let gpu = Gpu::new_cpu(vocoder::PIPELINES);
        // A length generous enough that both crops fit inside one chunk's
        // own decoded span at `hop_length=4` (tiny's upsampling product).
        let length = 400usize;
        let mut r = Lcg::new(4);
        let latents_a = r.vec_scaled(cfg.latent_channels as usize * length, 0.3);
        let latents_b = r.vec_scaled(cfg.latent_channels as usize * length, 0.3);

        let mut stitcher = Stitcher::new();
        stitcher.push_chunk(&gpu, &cfg, &w, &latents_a, length, true, false);
        stitcher.push_chunk(&gpu, &cfg, &w, &latents_b, length, false, true);
        let (left, right) = stitcher.finish();

        let hop = hop_length(&cfg);
        let per_chunk_samples = length * hop;
        let expected_len = (per_chunk_samples - CROP_RIGHT_LATENT * hop) + (per_chunk_samples - CROP_LEFT_LATENT * hop);
        assert_eq!(left.len(), expected_len);
        assert_eq!(right.len(), expected_len);
        for &s in left.iter().chain(&right) {
            assert!((-1.0..=1.0).contains(&s), "sample out of [-1,1]: {s}");
        }
    }
}
