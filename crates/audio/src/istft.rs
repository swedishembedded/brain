// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Inverse STFT (overlap-add + window-sum-square / NOLA normalization).
//! Nothing in this workspace computed one before this module -
//! `crates/ltxv/src/vocoder.rs` explicitly notes ISTFT is out of scope there.
//! CosyVoice's HiFT vocoder needs exactly this: a neural head predicts a
//! per-frame spectrum, and the waveform is reconstructed by overlap-adding
//! its inverse FFT.
//!
//! [`istft`] computes each frame's inverse FFT via the conjugate trick
//! (negate the imaginary part, run the *forward* FFT, the real part of the
//! result scaled by `1/n_fft` is the real-valued time-domain frame) -
//! `asr_frontend::fft_any`'s Bluestein inverse already uses the same trick,
//! so `mel::fft` (radix-2 or Bluestein depending on `n_fft`) is the only FFT
//! implementation this needs, reused rather than duplicated.
//!
//! [`stft`] is this module's forward counterpart, existing only to compose
//! with [`istft`] for round-trip verification: it is deliberately
//! independent of `mel::power_spectrogram` (which discards phase and carries
//! mel-specific config unrelated to a plain STFT/ISTFT pair) and does no
//! centering/padding, so the two are the exact mathematical inverse of one
//! another at every non-edge frame.

use crate::mel::fft;
use std::f32::consts::PI;

/// Plain STFT/ISTFT shape: `n_fft`-point transform, `hop`-sample stride, a
/// `win`-sample periodic Hann analysis/synthesis window (`win <= n_fft`, the
/// window zero-padded to `n_fft` when shorter).
#[derive(Clone, Copy, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
}

impl StftConfig {
    /// Frame count for [`stft`] given a signal of length `len` (no padding).
    pub fn n_frames(&self, len: usize) -> usize {
        if len < self.n_fft {
            0
        } else {
            1 + (len - self.n_fft) / self.hop
        }
    }
}

fn hann_periodic(win: usize) -> Vec<f32> {
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

/// Forward complex STFT: `[n_frames, bins]` row-major real/imag one-sided
/// spectrum, `bins = n_fft/2+1`. No centering: frame `fr` covers samples
/// `[fr*hop, fr*hop+n_fft)`.
pub fn stft(samples: &[f32], cfg: &StftConfig) -> (Vec<f32>, Vec<f32>, usize) {
    let window = hann_periodic(cfg.win);
    let bins = cfg.n_fft / 2 + 1;
    let n_frames = cfg.n_frames(samples.len());
    let mut re_out = vec![0.0f32; n_frames * bins];
    let mut im_out = vec![0.0f32; n_frames * bins];
    let mut re = vec![0.0f32; cfg.n_fft];
    let mut im = vec![0.0f32; cfg.n_fft];
    for fr in 0..n_frames {
        let start = fr * cfg.hop;
        for v in im.iter_mut() {
            *v = 0.0;
        }
        for i in 0..cfg.n_fft {
            re[i] = if i < cfg.win { samples[start + i] * window[i] } else { 0.0 };
        }
        fft(&mut re, &mut im);
        re_out[fr * bins..(fr + 1) * bins].copy_from_slice(&re[..bins]);
        im_out[fr * bins..(fr + 1) * bins].copy_from_slice(&im[..bins]);
    }
    (re_out, im_out, n_frames)
}

/// Inverse STFT via overlap-add with window-sum-square (NOLA) normalization.
/// `re`/`im` are `[n_frames, bins]` row-major one-sided spectra (`bins =
/// n_fft/2+1`); the negative-frequency half is reconstructed by Hermitian
/// symmetry (`X[n_fft-k] = conj(X[k])`), so a real-valued waveform comes back
/// out even though only the non-negative frequencies were provided. Returns
/// a waveform of length `(n_frames-1)*hop + n_fft` (0 when `n_frames == 0`),
/// UN-trimmed - a caller that padded before the forward transform trims the
/// same amount back off both ends.
pub fn istft(re: &[f32], im: &[f32], n_frames: usize, cfg: &StftConfig) -> Vec<f32> {
    if n_frames == 0 {
        return Vec::new();
    }
    let bins = cfg.n_fft / 2 + 1;
    assert_eq!(re.len(), n_frames * bins, "istft: re length != n_frames*bins");
    assert_eq!(im.len(), n_frames * bins, "istft: im length != n_frames*bins");
    let window = hann_periodic(cfg.win);
    let out_len = (n_frames - 1) * cfg.hop + cfg.n_fft;
    let mut out = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let mut fr_re = vec![0.0f32; cfg.n_fft];
    let mut fr_im = vec![0.0f32; cfg.n_fft];
    for fr in 0..n_frames {
        for b in 0..bins {
            fr_re[b] = re[fr * bins + b];
            fr_im[b] = im[fr * bins + b];
        }
        for k in bins..cfg.n_fft {
            let src = cfg.n_fft - k;
            fr_re[k] = re[fr * bins + src];
            fr_im[k] = -im[fr * bins + src];
        }
        // inverse FFT via the conjugate trick: negate im, forward FFT, the
        // real part scaled by 1/n_fft is the (real-valued) time frame.
        for v in fr_im.iter_mut() {
            *v = -*v;
        }
        fft(&mut fr_re, &mut fr_im);
        let inv = 1.0 / cfg.n_fft as f32;
        let start = fr * cfg.hop;
        for i in 0..cfg.n_fft {
            let w = if i < window.len() { window[i] } else { 0.0 };
            out[start + i] += fr_re[i] * inv * w;
            wsum[start + i] += w * w;
        }
    }
    for i in 0..out_len {
        if wsum[i] > 1e-8 {
            out[i] /= wsum[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// `istft(stft(x)) ~= x` on the interior (away from the un-covered edge
    /// frames, per this module's own doc), at CosyVoice's HiFT shape:
    /// n_fft=16, hop=4, win=16.
    #[test]
    fn roundtrip_hift_shape() {
        let cfg = StftConfig { n_fft: 16, hop: 4, win: 16 };
        let mut r = Lcg::new(0xC057_100E);
        let n = 400;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 0.6 + r.signed() * 0.1).collect();
        let (re, im, n_frames) = stft(&x, &cfg);
        let y = istft(&re, &im, n_frames, &cfg);
        assert_eq!(y.len(), (n_frames - 1) * cfg.hop + cfg.n_fft);
        // interior: skip the first/last n_fft samples, where fewer frames
        // overlap and edge effects (per this module's doc) are expected.
        let lo = cfg.n_fft;
        let hi = y.len() - cfg.n_fft;
        assert!(hi > lo, "signal too short for an interior region");
        let d = max_abs(&x[lo..hi], &y[lo..hi]);
        assert!(d < 1e-4, "istft(stft(x)) interior maxdiff {d}");
    }

    /// Same round-trip at a second, distinct shape (win < n_fft, non-unit
    /// hop/win ratio) so the gate isn't calibrated to one config only.
    #[test]
    fn roundtrip_second_shape() {
        let cfg = StftConfig { n_fft: 32, hop: 8, win: 24 };
        let mut r = Lcg::new(0x0005_EED1);
        let n = 600;
        let x: Vec<f32> = (0..n).map(|_| r.signed() * 0.5).collect();
        let (re, im, n_frames) = stft(&x, &cfg);
        let y = istft(&re, &im, n_frames, &cfg);
        let lo = cfg.n_fft;
        let hi = y.len() - cfg.n_fft;
        assert!(hi > lo);
        let d = max_abs(&x[lo..hi], &y[lo..hi]);
        assert!(d < 1e-3, "istft(stft(x)) interior maxdiff {d}");
    }

    #[test]
    fn istft_empty_frames_returns_empty() {
        let cfg = StftConfig { n_fft: 16, hop: 4, win: 16 };
        assert!(istft(&[], &[], 0, &cfg).is_empty());
    }
}
