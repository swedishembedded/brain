// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The audio stream's generation-side geometry and its decode tail: how many
//! audio tokens a clip of `frames` at `fps` carries, where each of those
//! tokens sits on the shared time axis, and how the denoised audio latent
//! becomes a waveform.
//!
//! LTX-2.5 is natively audio-visual - one diffusion transformer denoises a
//! video-latent stream and an audio-latent stream together - so this module
//! is not a separate audio model bolted onto a video one. It is the half of
//! the joint model's I/O that the video path alone never had to express.
//!
//! ## The audio latent's shape is derived from the VIDEO request
//!
//! `AudioLatentShape.from_video_pixel_shape` (`ltx_core.types`) turns the
//! clip's own duration into a token count and nothing else is free:
//!
//! ```text
//! latents_per_second = sample_rate / hop_length / audio_latent_downsample_factor
//!                    = 16000 / 160 / 4 = 25
//! ta = round(frames / fps * 25)
//! ```
//!
//! and the latent is `[channels = 8, ta, mel_bins = 16]`. The DiT sees it
//! patchified as `ta` tokens of `8 * 16 = 128` channels
//! (`AudioPatchifier.patchify`'s `b c t f -> b t (c f)`, `patch_size = 1`, an
//! identity reshape) - which is exactly the `audio_patchify_proj.weight`'s
//! `[128, 2048]` input width in the real checkpoint, and the reason the audio
//! stream shares the video stream's `in_channels` field.
//!
//! ## Alignment is a property of the arithmetic, not of a later resample
//!
//! Each audio token carries a `[start, end)` bound **in seconds** on the same
//! time axis the video tokens use ([`positions`]), and the A<->V
//! cross-attention's shared RoPE space is built from those seconds on both
//! sides. So a video frame and an audio window at the same moment attend to
//! each other because they carry the same coordinate, not because anything
//! resamples afterwards.
//!
//! The decode tail then lands `(4*ta - 3) * 160` samples at 16 kHz, i.e.
//! `(4*ta - 3) / 100` seconds. At the exact `ta = 25 * duration` that is
//! `duration - 3 * HOP_LENGTH / SAMPLE_RATE`: the causal audio VAE's first
//! latent frame covers one mel frame rather than four, so a decoded clip is
//! exactly THREE MEL FRAMES shorter than its video, whatever its length.
//! That is the reference's own arithmetic
//! (`AudioDecoder._denormalize_latents`' `frames * 4 - 3`), not a rounding
//! bug here, and [`AudioClip::pad_to_seconds`] is what makes the two tracks
//! exactly equal length for a container that wants them to be.

use crate::audio_vae::{self, AudioVaeConfig};
use crate::vocoder::{self, VocoderConfig};

/// `sample_rate / hop_length / audio_latent_downsample_factor` - the audio
/// latent's own frame rate, 25 Hz at every constant below. Named rather than
/// written as 25 so the derivation stays visible next to the constants it
/// comes from.
pub const LATENTS_PER_SECOND: f64 = SAMPLE_RATE as f64 / HOP_LENGTH as f64 / LATENT_DOWNSAMPLE as f64;

/// The base vocoder's output rate. The 48 kHz bandwidth-extension stage
/// (`vocoder.bwe_generator.*`) is present in the checkpoint but not
/// implemented - see [`crate::vocoder`]'s module doc.
pub const SAMPLE_RATE: u32 = 16_000;
/// The mel spectrogram's hop, in samples - also the vocoder's own total
/// upsample ratio (`product([5,2,2,2,2,2]) == 160`), which is why one mel
/// frame becomes exactly this many samples.
pub const HOP_LENGTH: u32 = 160;
/// Mel frames per audio latent frame, on the time axis, through the audio
/// VAE's two stride-2 stages.
pub const LATENT_DOWNSAMPLE: u32 = 4;
/// The audio latent's channel count (`z_channels`).
pub const LATENT_CHANNELS: u32 = 8;
/// The audio latent's frequency-bin count - `mel_bins / 4`, the two freq-axis
/// halvings the encoder applies to its 64-bin mel input.
pub const LATENT_MEL_BINS: u32 = 16;
/// Audio channels the model generates (`stereo: true`).
pub const CHANNELS: u32 = 2;

/// The DiT token width for the audio stream, `LATENT_CHANNELS *
/// LATENT_MEL_BINS`. Equals the video stream's `in_channels` in the real
/// checkpoint, which is why one `in_channels` field serves both.
pub const TOKEN_DIM: u32 = LATENT_CHANNELS * LATENT_MEL_BINS;

/// How many audio latent frames (= DiT audio tokens) a clip of `frames` at
/// `fps` carries - `AudioLatentShape.from_video_pixel_shape`'s own
/// `round(duration * latents_per_second)`.
///
/// At least 1: a single-frame "clip" still has a duration, and a zero-token
/// audio stream would make every A<->V cross-attention degenerate rather than
/// merely short.
pub fn latent_frames(frames: usize, fps: usize) -> usize {
    let duration = frames as f64 / fps.max(1) as f64;
    ((duration * LATENTS_PER_SECOND).round() as usize).max(1)
}

/// The seconds one audio latent frame's own `[start, end)` window covers -
/// `AudioPatchifier._get_audio_latent_time_in_sec` at `is_causal = true`.
///
/// The causal offset is the reference's own `+1 - downsample_factor` clipped
/// at zero, applied to the MEL frame index before the hop/rate divide: it
/// makes a token's timestamp the first sample fully available to it rather
/// than the first one it partly overlaps, so a token never claims a moment it
/// could not have seen.
fn latent_bounds(i: usize) -> (f32, f32) {
    let sec = |latent_idx: usize| -> f32 {
        let mel = latent_idx as i64 * i64::from(LATENT_DOWNSAMPLE) + 1 - i64::from(LATENT_DOWNSAMPLE);
        mel.max(0) as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32
    };
    (sec(i), sec(i + 1))
}

/// `[1 axis, ta, 2]` row-major RoPE position bounds for the audio stream -
/// the single-axis (time, in seconds) counterpart of
/// [`crate::pipeline::real_pixel_positions`], and in the SAME layout that
/// function uses (`out[(axis * t + tok) * 2 + {0,1}]`), because
/// `crate::rope::ltx_rope_tables` reads both through one indexing rule.
///
/// Seconds, not indices: `audio_positional_embedding_max_pos` is `[20]` in
/// the real config, which is 20 SECONDS, and the shared cross-modal RoPE
/// space normalizes video's frame axis by the same number. Feeding latent
/// indices here would put the two streams on incomparable scales and quietly
/// destroy the audio/video correspondence the cross-attention exists for.
pub fn positions(ta: usize) -> Vec<f32> {
    let mut out = vec![0f32; ta * 2];
    for (i, slot) in out.chunks_mut(2).enumerate() {
        let (s, e) = latent_bounds(i);
        slot[0] = s;
        slot[1] = e;
    }
    out
}

/// A decoded waveform, one `Vec<f32>` per channel.
#[derive(Clone, Debug, Default)]
pub struct AudioClip {
    /// `channels.len()` planes of `samples_per_channel` values in `[-1, 1]`.
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

impl AudioClip {
    pub fn samples_per_channel(&self) -> usize {
        self.channels.first().map(Vec::len).unwrap_or(0)
    }

    pub fn seconds(&self) -> f32 {
        self.samples_per_channel() as f32 / self.sample_rate.max(1) as f32
    }

    /// Extend every channel to exactly `seconds` by repeating its LAST sample.
    ///
    /// Only ever a handful of samples: the audio VAE's causal first frame
    /// makes a decode 3 mel frames (30 ms) shorter than the video it belongs
    /// to (see this module's doc), and a container is happier with two tracks
    /// of equal length than with one that ends early. Holding the last sample
    /// rather than zero-filling avoids a step discontinuity at the very end,
    /// which is audible as a click where 30 ms of silence is not. Shortening
    /// is deliberately NOT done here - a clip that came out LONGER than asked
    /// means the geometry is wrong, and truncating would hide it.
    pub fn pad_to_seconds(&mut self, seconds: f32) {
        let want = (seconds * self.sample_rate as f32).round() as usize;
        for ch in &mut self.channels {
            let Some(&last) = ch.last() else { continue };
            if ch.len() < want {
                ch.resize(want, last);
            }
        }
    }
}

/// Turn the DiT's own `[ta, TOKEN_DIM]` token-major audio latent into the
/// audio VAE's `[LATENT_CHANNELS, ta, LATENT_MEL_BINS]` channel-major one -
/// `AudioPatchifier.unpatchify`'s `b t (c f) -> b c t f`.
///
/// The one transpose this path needs, and the one place a channel/frequency
/// mix-up could hide: within a token the 128 values run frequency-fastest
/// inside channel (`j = c * mel_bins + f`), so reading them as
/// `f * channels + c` would decode a spectrogram whose bins are shuffled -
/// audible as noise, not as an error.
pub fn unpatchify(tokens: &[f32], ta: usize) -> Vec<f32> {
    let (c, f) = (LATENT_CHANNELS as usize, LATENT_MEL_BINS as usize);
    assert_eq!(tokens.len(), ta * c * f, "ltxv audio unpatchify: {} values, expected {}", tokens.len(), ta * c * f);
    let mut out = vec![0f32; c * ta * f];
    for t in 0..ta {
        for ci in 0..c {
            for fi in 0..f {
                out[ci * ta * f + t * f + fi] = tokens[t * c * f + ci * f + fi];
            }
        }
    }
    out
}

/// Decode a denoised audio latent all the way to a waveform: unpatchify ->
/// audio VAE decoder (latent -> log-mel) -> base vocoder (log-mel -> 16 kHz
/// stereo samples).
///
/// `tokens` is the DiT's own `[ta, TOKEN_DIM]` output. `vae_weights` and
/// `vocoder_weights` both come from the SAME `ltx-2.5-audio-vae-*` file - two
/// disjoint tensor subsets of one checkpoint, see [`crate::import`]'s module
/// doc.
///
/// Both stages are the real-weight, real-parity implementations this crate
/// already gates ([`crate::audio_vae`], [`crate::vocoder`]); nothing here
/// re-derives their math, it only composes them and states the shapes that
/// connect them.
pub fn decode(vae_weights: &vae::blocks::Tensors, vocoder_weights: &vae::blocks::Tensors, tokens: &[f32], ta: usize, device: Option<&str>) -> AudioClip {
    let vcfg = AudioVaeConfig::ltx25();
    let latent = unpatchify(tokens, ta);
    let mel = audio_vae::decode(&vcfg, vae_weights, &latent, ta as u32, LATENT_MEL_BINS, device);

    let mel_frames = LATENT_DOWNSAMPLE * ta as u32 - 3;
    let mel_bins = LATENT_MEL_BINS * 4;
    assert_eq!(mel.len(), (CHANNELS * mel_frames * mel_bins) as usize, "ltxv audio decode: mel is {} values, expected {}", mel.len(), CHANNELS * mel_frames * mel_bins);

    let ccfg = VocoderConfig::ltx25();
    let wave = vocoder::synthesize(&ccfg, vocoder_weights, &mel, CHANNELS, mel_frames, mel_bins, device);
    let n = (mel_frames * HOP_LENGTH) as usize;
    assert_eq!(wave.len(), CHANNELS as usize * n, "ltxv audio decode: waveform is {} values, expected {}", wave.len(), CHANNELS as usize * n);

    AudioClip { channels: wave.chunks(n).map(<[f32]>::to_vec).collect(), sample_rate: SAMPLE_RATE }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token count is the reference's own `round(duration * 25)`, and the
    /// decoded LENGTH that follows from it must land within one latent frame
    /// of the clip it belongs to - the whole audio/video alignment claim in
    /// one arithmetic gate, checked across the frame counts `1 + 8k` allows
    /// and both frame rates the CLI offers.
    #[test]
    fn the_audio_track_is_the_same_length_as_the_clip_it_belongs_to() {
        for fps in [24usize, 25, 30] {
            for k in 0..40usize {
                let frames = 1 + 8 * k;
                let ta = latent_frames(frames, fps);
                let video_seconds = frames as f64 / fps as f64;
                let audio_seconds = f64::from(LATENT_DOWNSAMPLE * ta as u32 - 3) * f64::from(HOP_LENGTH) / f64::from(SAMPLE_RATE);
                // The gap is bounded by exactly two known terms, not by a
                // round number: the causal VAE's fixed three-mel-frame trim
                // (see this module's doc) plus at most half a latent frame of
                // rounding in `round(duration * LATENTS_PER_SECOND)`.
                // Anything larger means the geometry is wrong, not coarse.
                let slack = 3.0 * f64::from(HOP_LENGTH) / f64::from(SAMPLE_RATE) + 0.5 / LATENTS_PER_SECOND;
                assert!(
                    (video_seconds - audio_seconds).abs() <= slack + 1e-9,
                    "{frames} frames at {fps} fps: video {video_seconds:.4}s vs audio {audio_seconds:.4}s (ta={ta}), off by more than the causal trim plus half a latent frame ({slack:.4}s)"
                );
                // Never LONGER than the video: the pad-to-length step only
                // ever extends, so an over-long track would survive as a
                // desynchronised tail rather than being corrected.
                assert!(audio_seconds <= video_seconds + 1e-9, "{frames} frames at {fps} fps: audio {audio_seconds:.4}s is longer than video {video_seconds:.4}s");
            }
        }
    }

    /// Positions are seconds on the same axis the video stream uses, strictly
    /// increasing, starting at zero, and each token's window is contiguous
    /// with the next one's - the property the shared cross-modal RoPE space
    /// depends on. A gap or an overlap here is silent: it produces audio that
    /// plays but is attending to the wrong moment of the picture.
    #[test]
    fn audio_positions_tile_the_timeline_without_gaps() {
        let ta = 64;
        let p = positions(ta);
        assert_eq!(p.len(), ta * 2);
        assert_eq!(p[0], 0.0, "the first token must start at t=0");
        for i in 0..ta {
            let (s, e) = (p[i * 2], p[i * 2 + 1]);
            assert!(e > s, "token {i}: [{s}, {e}) is not a forward window");
            if i + 1 < ta {
                assert_eq!(e, p[(i + 1) * 2], "token {i} ends at {e} but token {} starts at {}", i + 1, p[(i + 1) * 2]);
            }
        }
        // The steady-state cadence IS the latent rate: every window past the
        // causal first one covers exactly one latent frame of time.
        let step = p[5] - p[3];
        assert!((f64::from(step) - 1.0 / LATENTS_PER_SECOND).abs() < 1e-6, "steady-state window is {step}s, expected {}s", 1.0 / LATENTS_PER_SECOND);
    }

    /// `unpatchify` must read a token's 128 values as frequency-fastest
    /// inside channel. Built from a value that encodes its own coordinates so
    /// a transposed read cannot accidentally agree.
    #[test]
    fn unpatchify_reads_frequency_fastest_inside_channel() {
        let (ta, c, f) = (3usize, LATENT_CHANNELS as usize, LATENT_MEL_BINS as usize);
        let tokens: Vec<f32> = (0..ta * c * f).map(|i| i as f32).collect();
        let out = unpatchify(&tokens, ta);
        for t in 0..ta {
            for ci in 0..c {
                for fi in 0..f {
                    assert_eq!(out[ci * ta * f + t * f + fi], (t * c * f + ci * f + fi) as f32, "mismatch at (c={ci}, t={t}, f={fi})");
                }
            }
        }
    }

    /// Padding only ever extends, holds the last sample, and never shortens.
    #[test]
    fn padding_extends_and_never_truncates() {
        let mut clip = AudioClip { channels: vec![vec![0.0, 0.5], vec![0.0, -0.5]], sample_rate: 100 };
        clip.pad_to_seconds(0.05);
        assert_eq!(clip.samples_per_channel(), 5);
        assert_eq!(clip.channels[0], vec![0.0, 0.5, 0.5, 0.5, 0.5]);
        assert_eq!(clip.channels[1], vec![0.0, -0.5, -0.5, -0.5, -0.5]);
        clip.pad_to_seconds(0.01);
        assert_eq!(clip.samples_per_channel(), 5, "pad_to_seconds must never truncate");
    }
}
