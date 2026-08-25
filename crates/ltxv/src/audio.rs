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
//!
//! ## Carrying the stream across a window seam
//!
//! A clip too long for one denoising window is generated as several
//! ([`crate::longform`]), and the audio stream has to cross every seam the
//! video latent crosses. It does it the same way and by the same mechanism -
//! the previous window's own last tokens, sliced out of the denoised latent
//! ([`carry_tail`]), frozen at sigma 0 at the head of the next window's
//! sequence - but the two streams do NOT share a time resolution, and that is
//! where a seam can go quietly wrong.
//!
//! One video latent frame is [`crate::pipeline::VAE_TEMPORAL_SCALE`] pixel
//! frames, so it is `VAE_TEMPORAL_SCALE * LATENT_RATE / fps` audio tokens -
//! [`TOKENS_PER_VIDEO_LATENT_FRAME_NUM`] over `fps`, a RATIO that is not an
//! integer in general. At a 24-frame-a-second clip it is `200/24 = 25/3`,
//! so a seam placed at
//! an arbitrary video latent frame lands a third or two thirds of the way
//! into an audio token, and there is no honest way to freeze two thirds of a
//! token.
//!
//! **The rule this module implements: a window boundary must fall on a whole
//! number of audio tokens, and a plan that cannot place one there is
//! refused.** [`window_latent_frame_quantum`] is the smallest number of video
//! latent frames a window may advance by for that to hold (3 at 24 and 30
//! and 30 frames a second, 1 at 25), [`crate::longform::window_plan_aligned`]
//! plans in
//! multiples of it, and [`audio_plan`] re-derives the whole token layout from
//! the finished plan and REFUSES rather than rounding if any seam missed.
//!
//! Two consequences fall out of that rule, and both are what make the result
//! exact rather than merely close:
//!
//! * every seam shifts BOTH streams' local time axes by the same amount, so
//!   a carried audio token sits at exactly the moment of the picture it sat
//!   at in the window that generated it;
//! * the windows' token counts sum to the clip's own
//!   `round(frames / fps * LATENT_RATE)` exactly, so a multi-window clip
//!   decodes to the same number of samples a single-window clip of the same
//!   length would - the container's two stream durations are the same
//!   numbers they already were.
//!
//! The LAST window needs no quantum: it has no successor, so nothing is
//! carried out of it. Leaving it free is also what lets any legal `1 + 8k`
//! length be planned - constraining it too would restrict the clip's total
//! length to one residue class.

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

/// [`LATENTS_PER_SECOND`] as an exact integer.
///
/// The seam arithmetic below must never be decided by a float: a boundary
/// that lands half a token off does not fail, it drifts, and a rounded
/// division is exactly how that happens. This constant divides exactly at
/// LTX-2.5's own constants and the assertion under it is what says so at
/// compile time rather than in a comment.
pub const LATENT_RATE: u32 = SAMPLE_RATE / (HOP_LENGTH * LATENT_DOWNSAMPLE);
const _: () = assert!(SAMPLE_RATE.is_multiple_of(HOP_LENGTH * LATENT_DOWNSAMPLE), "the audio latent rate is not a whole number of latents per second at these constants");

/// The numerator of "audio tokens per VIDEO latent frame": one video latent
/// frame is [`crate::pipeline::VAE_TEMPORAL_SCALE`] pixel frames, hence
/// `VAE_TEMPORAL_SCALE / fps` seconds, hence this many tokens over `fps`.
///
/// Kept as a numerator rather than evaluated, because the whole point is
/// that the ratio is not an integer at every frame rate - see this module's
/// doc on carrying the stream across a seam.
pub const TOKENS_PER_VIDEO_LATENT_FRAME_NUM: usize = crate::pipeline::VAE_TEMPORAL_SCALE * LATENT_RATE as usize;

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

/// The audio tokens `n` video latent frames of clip time are worth at `fps`,
/// or `None` when that is not a WHOLE number of tokens.
///
/// `None` is the whole safety property of the seam arithmetic: the caller
/// either gets an exact token count or is told there isn't one, and never a
/// rounded stand-in.
pub fn tokens_for_video_latent_frames(n: usize, fps: usize) -> Option<usize> {
    let fps = fps.max(1);
    let num = TOKENS_PER_VIDEO_LATENT_FRAME_NUM * n;
    num.is_multiple_of(fps).then(|| num / fps)
}

/// The smallest number of video latent frames a window may advance by for a
/// seam to land on a whole audio token at `fps`.
///
/// Written as a search for the property rather than as `fps / gcd(200, fps)`
/// so the definition IS the requirement: the answer is the first `n` whose
/// clip time is a whole number of audio tokens. `n = fps` always satisfies it
/// (`TOKENS_PER_VIDEO_LATENT_FRAME_NUM * fps` is trivially divisible by
/// `fps`), so the search never comes back empty.
pub fn window_latent_frame_quantum(fps: usize) -> usize {
    let fps = fps.max(1);
    (1..=fps).find(|&n| tokens_for_video_latent_frames(n, fps).is_some()).unwrap_or(fps)
}

/// Why an audio-visual plan at `fps` has to advance in whole quanta, in the
/// terms the caller can act on.
///
/// Appended to a [`crate::longform::window_plan_aligned`] refusal by every
/// caller that passed a quantum, because that function is deliberately
/// frame-rate-blind - it takes a number, and only the audio stream knows
/// where the number came from. Without this the refusal names a constraint
/// with no visible cause, which is the kind of error message that gets worked
/// around instead of understood.
pub fn quantum_note(fps: usize) -> String {
    let q = window_latent_frame_quantum(fps);
    format!(
        "With --audio a window seam has to land on a whole audio token: at this frame rate one video latent frame is {}/{fps} audio tokens, so a plan may only advance by multiples of {q} latent frames. Generate at a smaller size, lower --context-frames, or pick a frame rate whose quantum is 1 (any rate that divides {}) - a fraction of a token cannot be carried, and carrying a rounded one is sound that drifts against the picture.",
        TOKENS_PER_VIDEO_LATENT_FRAME_NUM, TOKENS_PER_VIDEO_LATENT_FRAME_NUM
    )
}

/// Audio tokens a carried context of `context_latent_frames` video latent
/// frames covers at the HEAD of a window's own sequence.
///
/// The carried prefix is re-based to the new window's own time origin exactly
/// as the video's is - its first latent frame covers one pixel frame there
/// rather than eight - so the pixel span it occupies is
/// `(N - 1) * VAE_TEMPORAL_SCALE + 1`, which is
/// [`crate::longform::Window::dropped_frames`]. This is therefore the audio
/// token count of that prefix considered as a clip in its own right, which is
/// what makes it the same rule [`latent_frames`] applies to everything else.
pub fn context_tokens(context_latent_frames: usize, fps: usize) -> usize {
    latent_frames(1 + crate::pipeline::VAE_TEMPORAL_SCALE * context_latent_frames.saturating_sub(1), fps)
}

/// The audio token layout of a whole multi-window plan.
///
/// One place, so the generation loop, the gate and the CLI preview cannot
/// disagree about where a window's sound begins and ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioPlan {
    /// Tokens every continuation window carries in from its predecessor and
    /// holds at sigma 0 - [`context_tokens`], the same at every seam.
    pub context: usize,
    /// Tokens in each window's OWN sequence, `context` of them carried.
    pub per_window: Vec<usize>,
    /// Tokens the finished clip holds, which must be exactly the clip's own
    /// [`latent_frames`].
    pub total: usize,
}

impl AudioPlan {
    /// Tokens window `i` contributes to the finished clip - everything past
    /// the carried prefix.
    pub fn new_tokens(&self, i: usize) -> usize {
        self.per_window[i] - if i == 0 { 0 } else { self.context }
    }
}

/// Derive the audio token layout of `plan`, refusing any plan whose seams do
/// not land on whole audio tokens at `fps`.
///
/// **This is the alignment rule, and it is a check rather than a
/// construction on purpose.** [`crate::longform::window_plan_aligned`] builds
/// a plan that satisfies it; this re-derives the answer from the finished
/// plan and refuses if it does not, so a planner change that quietly breaks
/// the correspondence cannot reach a generation. Both halves of the rule are
/// enforced:
///
/// * every window with a SUCCESSOR must advance the clip's time by a whole
///   number of audio tokens (`w.latent_frames() - context`), or the carried
///   prefix would land a fraction of a token away from the picture it belongs
///   to - inaudible as a fault, audible as sound that drifts;
/// * the windows' contributions must sum to the clip's own token count, so a
///   multi-window clip's two container streams carry the same durations a
///   single-window clip's do.
pub fn audio_plan(plan: &[crate::longform::Window], context: usize, frames: usize, fps: usize) -> Result<AudioPlan, String> {
    if plan.is_empty() {
        return Err("an audio plan needs at least one window".into());
    }
    let ctx_tokens = context_tokens(context, fps);
    let quantum = window_latent_frame_quantum(fps);
    let mut per_window = Vec::with_capacity(plan.len());
    for (i, w) in plan.iter().enumerate() {
        let ta = latent_frames(w.decoded_frames(), fps);
        if i > 0 && ta <= ctx_tokens {
            return Err(format!(
                "window {i} of this plan holds {ta} audio tokens and carries {ctx_tokens} of them, so it would generate no sound at all - lower the context, or ask for longer windows"
            ));
        }
        // Only a window with a successor shifts the time base, so only that
        // window's advance has to be a whole number of tokens. The last one
        // is free, which is also what lets any legal clip length be planned.
        if i + 1 < plan.len() {
            let advance = w.latent_frames().saturating_sub(context);
            if tokens_for_video_latent_frames(advance, fps).is_none() {
                return Err(format!(
                    "window {i} of this plan advances the clip by {advance} latent frames, which is {}/{fps} audio tokens and not a whole number of them, so the next window's carried sound would sit a fraction of a token away from the picture it belongs to - a plan at {fps} fps has to advance in multiples of {quantum} latent frames",
                    TOKENS_PER_VIDEO_LATENT_FRAME_NUM * advance
                ));
            }
        }
        per_window.push(ta);
    }
    let total: usize = per_window.iter().enumerate().map(|(i, &ta)| ta - if i == 0 { 0 } else { ctx_tokens }).sum();
    let want = latent_frames(frames, fps);
    if total != want {
        return Err(format!(
            "this plan's windows carry {total} audio tokens between them but a {frames}-frame clip at {fps} fps is {want} - the sound would not be the same length as the picture"
        ));
    }
    Ok(AudioPlan { context: ctx_tokens, per_window, total })
}

/// The last `k` tokens of a `[ta, TOKEN_DIM]` token-major audio latent - the
/// audio counterpart of [`crate::longform::carry_tail`].
///
/// A slice and a copy, nothing else: no decode, no re-encode, no resample.
/// What the next window freezes is bit-identical to what this one produced,
/// which is the same promise the video half makes.
pub fn carry_tail(latent: &[f32], k: usize) -> Vec<f32> {
    let dim = TOKEN_DIM as usize;
    assert!(latent.len().is_multiple_of(dim), "audio carry_tail: {} values is not a whole number of {dim}-wide tokens", latent.len());
    let ta = latent.len() / dim;
    assert!(k <= ta, "audio carry_tail: cannot carry {k} of {ta} tokens");
    latent[(ta - k) * dim..].to_vec()
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
