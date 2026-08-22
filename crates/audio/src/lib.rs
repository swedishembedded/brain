// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared audio engine infrastructure - the front-end/back-end pieces every
//! spectrogram- or waveform-based model composes from, all fp32 and
//! dependency-light so they share the brain engine's portability constraints:
//!   * [`wav`]   - canonical PCM WAV read/write (mono f32).
//!   * [`conv`]  - 1D conv / transposed-conv `Step`-builders over the shared
//!     WGSL engine (+ CPU reference oracles), the audio analogue of
//!     `model::block`. Backs the codec, ECAPA speaker encoder and GAN vocoder.
//!     Also hosts [`conv::fold_weight_norm`], the `nn.utils.weight_norm`
//!     fold every weight-normalized checkpoint import needs.
//!   * [`act`] - elementwise activation `Step`-builders (currently ELU).
//!   * [`mel`] - STFT + mel-spectrogram features (forward), with a
//!     mixed-radix FFT for non-power-of-two `n_fft`.
//!   * [`kaldi_fbank`] - `torchaudio.compliance.kaldi.fbank`-compatible
//!     mel features (a genuinely different filter shape from [`mel`]'s -
//!     Kaldi's own triangular filters are linear in the MEL domain, not the
//!     Hz domain), for models (CosyVoice's CAM++ speaker encoder) whose
//!     reference front end is that Kaldi convention rather than
//!     librosa/HTK-style.
//!   * [`istft`] - the inverse: overlap-add ISTFT with window-sum-square
//!     normalization, for models (like a HiFT-style vocoder) that predict a
//!     spectrum and need it turned back into a waveform.
//!   * [`resample_linear`] - cheap linear-interpolation rate conversion;
//!     [`resample::rational`] - the accurate Kaiser-windowed-sinc sibling for
//!     rate changes that need it (e.g. 24 kHz -> 16 kHz).

pub mod act;
pub mod asr_caps;
pub mod asr_frontend;
pub mod conv;
pub mod istft;
pub mod kaldi_fbank;
pub mod mel;
pub mod resample;
pub mod snake;
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
