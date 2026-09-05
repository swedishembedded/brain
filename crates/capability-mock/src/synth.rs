// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure, deterministic content generators shared by [`crate::MockProvider`]
//! (today's mirror-a-manifest-and-infer-shape mock) and [`crate::expect`]'s
//! scripted [`crate::expect::MockBlob`] (a test declares exactly what a call
//! returns). No RNG crate, no wall-clock, no GPU, no file I/O - every
//! function here is wrapping arithmetic over an explicit `u32` seed, so the
//! same seed always produces the same bytes.

use std::f32::consts::TAU;

use capability::{CancelToken, Invocation, Progress};

/// How many synthetic "steps" a mock action ticks through before returning,
/// polling `cancel` between each - enough to make cancellation observably
/// interruptible without slowing tests down.
pub const STEPS: u32 = 3;

/// Poll `cancel` between [`STEPS`] synthetic progress ticks, returning
/// `Err("cancelled")` the moment it fires (checked BEFORE the first tick too,
/// so an already-cancelled invocation never emits progress at all).
pub fn run_steps(cancel: &CancelToken, progress: &mut dyn FnMut(Progress)) -> Result<(), String> {
    for s in 0..STEPS {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(Progress::step(s + 1, STEPS, "mock generating"));
    }
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    Ok(())
}

/// Fold `(model_id, action, an optional numeric extra)` into one `u32` seed -
/// plain wrapping arithmetic (FNV-1a over the bytes, then an avalanche mix),
/// no RNG crate, matching `resident_mock.rs::text2image`'s "no randomness,
/// just wrapping arithmetic" style but extended to the identity strings so
/// distinct mock providers/actions never coincide.
pub fn fold_seed(model_id: &str, action: &str, extra: i64) -> u32 {
    let mut h: u32 = 0x811C9DC5; // FNV-1a offset basis
    for b in model_id.bytes().chain(std::iter::once(b'/')).chain(action.bytes()) {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193); // FNV prime
    }
    h ^= extra as u32;
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h
}

/// The invocation's `seed` param, or `0` - folded into [`fold_seed`]'s extra
/// term so a caller-supplied seed actually changes the output.
pub fn seed_param(inv: &Invocation) -> i64 {
    inv.get_i64("seed").unwrap_or(0)
}

/// A deterministic interleaved-HWC f32 `[0,1]` RGB gradient - the same shape
/// `resident_mock.rs::text2image` produces (three independent per-channel
/// ramps offset by `seed`), so a mock image/mask "looks like" the real mock
/// resident's own output, not an unrelated pattern.
pub fn gradient_hwc(seed: u32, w: u32, h: u32) -> Vec<f32> {
    let mut hwc = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            hwc.push((x.wrapping_add(seed) % 256) as f32 / 255.0);
            hwc.push((y.wrapping_add(seed >> 8) % 256) as f32 / 255.0);
            hwc.push((x.wrapping_add(y).wrapping_add(seed) % 256) as f32 / 255.0);
        }
    }
    hwc
}

/// One frame of a moving-gradient clip: the same "bright block sweeps across
/// a per-axis ramp" idea as `crates/imaging/src/video.rs`'s private test
/// helper `moving_block`, reimplemented directly as an f32 HWC `[0,1]` plane
/// (no u8 round trip, no `Rgb8`) so [`capability::blob::video_blob`] can
/// encode it straight - the point of the fixture is the same: frames must
/// NOT all be identical.
pub fn video_frame_hwc(seed: u32, w: u32, h: u32, frame: u32) -> Vec<f32> {
    let mut hwc = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let period = w.max(1);
    for y in 0..h {
        for x in 0..w {
            let on = (x.wrapping_add(frame).wrapping_add(seed)) % period == 0;
            hwc.push(if on { 1.0 } else { (x % 256) as f32 / 255.0 });
            hwc.push((y.wrapping_add(seed >> 8) % 256) as f32 / 255.0);
            hwc.push((frame.wrapping_mul(20).wrapping_add(seed) % 256) as f32 / 255.0);
        }
    }
    hwc
}

/// A sine tone as raw interleaved f32-LE PCM: frequency `220 + seed % 440` Hz
/// (always audibly non-silent and seed-distinguishable), `channels` identical
/// copies per frame (a real stereo/mono split has no meaning for a synthetic
/// tone) - no WAV container, matching `audio::asr_caps`'s raw-PCM-plus-meta
/// convention for an untagged-format audio blob.
pub fn sine_pcm(seed: u32, seconds: f32, sample_rate: u32, channels: u32) -> Vec<u8> {
    let freq = 220.0 + (seed % 440) as f32;
    let n_frames = ((seconds.max(0.0)) * sample_rate as f32).round() as usize;
    let mut bytes = Vec::with_capacity(n_frames * channels as usize * 4);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate.max(1) as f32;
        let sample = (TAU * freq * t).sin() * 0.5;
        for _ in 0..channels {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    bytes
}

/// A short deterministic string: the invocation's last user turn / `prompt`
/// (via [`capability::last_user_text`], the same extraction every real chat
/// model uses), prefixed with the model/action identity - or, when the
/// invocation carries no such param, a fixed string keyed by model+action.
pub fn mock_text(model_id: &str, action: &str, inv: &Invocation) -> String {
    let prompt = capability::last_user_text(inv);
    if prompt.trim().is_empty() {
        format!("[{model_id}/{action}] deterministic mock output")
    } else {
        format!("[{model_id}/{action}] {}", prompt.trim())
    }
}

/// `n` deterministic bytes: a repeating `(i + seed) % 256` counter pattern.
pub fn counter_bytes(seed: u32, n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i as u32).wrapping_add(seed) % 256) as u8).collect()
}
