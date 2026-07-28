// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Audio foundation for the Qwen3-TTS stack.
//!
//! Three independent pieces, all fp32 and dependency-light so they share the
//! brain engine's portability constraints:
//!   * [`wav`]   — canonical PCM WAV read/write (mono f32).
//!   * [`conv`]  — 1D conv / transposed-conv `Step`-builders over the shared
//!                 WGSL engine (+ CPU reference oracles), the audio analogue of
//!                 `model::block`. Backs the codec, ECAPA speaker encoder and
//!                 GAN vocoder.
//!   * resampling — simple linear-interpolation rate conversion (24 kHz codec
//!                 vs 16 kHz inputs).
//!
//! STFT / mel-spectrogram features land in [`mel`] (built out for the speaker
//! encoder in Phase 3).

pub mod asr_frontend;
pub mod conv;
pub mod mel;
pub mod wav;

/// Linear-interpolation resample from `src_rate` to `dst_rate`.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let i0 = src_pos.floor() as usize;
        let frac = (src_pos - i0 as f64) as f32;
        let a = samples[i0.min(samples.len() - 1)];
        let b = samples[(i0 + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}
