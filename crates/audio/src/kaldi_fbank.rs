// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kaldi-compatible mel filterbank features (`torchaudio.compliance.kaldi.fbank`),
//! the front end CosyVoice's CAM++ speaker encoder needs:
//! `kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)`.
//! Every other parameter is left at `torchaudio.compliance.kaldi.fbank`'s own
//! documented default: `frame_length=25ms`, `frame_shift=10ms`, `povey`
//! window, `preemphasis_coefficient=0.97`, `remove_dc_offset=true`,
//! `round_to_power_of_two=true`, `low_freq=20`, `high_freq=0` (meaning
//! Nyquist), `use_power=true`, `use_log_fbank=true`, `use_energy=false`,
//! `snip_edges=true`.
//!
//! Ported directly from `torchaudio/compliance/kaldi.py`'s `fbank`/
//! `get_mel_banks` (read line-for-line in this checkout's own vendored venv,
//! not from the Kaldi C++ source or a secondhand description). This is a
//! GENUINELY DIFFERENT filter shape from [`crate::mel::mel_filterbank`] /
//! [`crate::asr_frontend::mel_filterbank_slaney`]: Kaldi computes each
//! triangular filter's slope as a ratio of MEL-domain distances (`(mel -
//! left_mel) / (center_mel - left_mel)`), not a ratio of Hz-domain distances
//! between mel-spaced edges the way HTK/librosa/slaney filters do - the two
//! are numerically close but not identical, so this module cannot reuse
//! either existing filterbank and is not a copy of one with different
//! constants.
//!
//! **Honest gap**: no golden dump of a real `torchaudio.compliance.kaldi.fbank`
//! run exists in this workspace to check this implementation against
//! bit-for-bit - CAM++'s own real-weight parity test
//! (`crates/campplus/tests/parity.rs`) reads its `[346,80]` fbank input
//! straight from a captured golden rather than computing it in Rust. This
//! implementation is verified structurally (frame count, window shape,
//! mel-scale formula match Kaldi's documented algorithm exactly, checked
//! against the literal `torchaudio.compliance.kaldi` source) and exercised by
//! the composed pipeline's own end-to-end smoke test, not by a per-stage
//! numeric comparison against a captured reference. A real parity gate
//! against a captured `torchaudio.compliance.kaldi.fbank` output is a
//! recorded follow-up. Dithering (`dither != 0`) is not implemented -
//! CosyVoice's own reference call always passes `dither=0`, so this is not a
//! gap for that caller.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::f32::consts::PI;

use crate::mel::fft;

/// `torchaudio.compliance.kaldi.fbank`'s parameters this port models.
#[derive(Clone, Copy, Debug)]
pub struct KaldiFbankConfig {
    pub sample_rate: f32,
    pub num_mel_bins: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    /// Low cutoff frequency for mel bins (Kaldi default `20.0`).
    pub low_freq: f32,
    /// High cutoff; `<= 0.0` means "Nyquist plus this" (Kaldi default `0.0`,
    /// i.e. exactly Nyquist).
    pub high_freq: f32,
    pub preemphasis_coefficient: f32,
}

impl KaldiFbankConfig {
    /// CosyVoice's own `_extract_spk_embedding` call:
    /// `kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)`,
    /// every other parameter at `torchaudio.compliance.kaldi.fbank`'s default.
    pub fn cosyvoice() -> KaldiFbankConfig {
        KaldiFbankConfig { sample_rate: 16000.0, num_mel_bins: 80, frame_length_ms: 25.0, frame_shift_ms: 10.0, low_freq: 20.0, high_freq: 0.0, preemphasis_coefficient: 0.97 }
    }
}

/// `torch.hann_window(n, periodic=False).pow(0.85)` - Kaldi's "povey" window
/// (like Hanning but goes to zero at the edges more steeply).
fn povey_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f32 / denom).cos()).powf(0.85)).collect()
}

/// `1127 * ln(1 + f/700)` - Kaldi's own mel scale (`mel_scale_scalar`).
fn mel_scale(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// `get_mel_banks` (`vtln_warp_factor == 1.0` path, no VTLN): `[num_bins,
/// padded_window_size/2 + 1]` row-major - the trailing column is the zero pad
/// `torch.nn.functional.pad(mel_energies, (0, 1))` appends so the bank lines
/// up with `rfft`'s own `padded_window_size/2 + 1`-wide output.
fn mel_banks(num_bins: usize, padded_window_size: usize, sample_freq: f32, low_freq: f32, high_freq: f32) -> Vec<f32> {
    let num_fft_bins = padded_window_size / 2;
    let nyquist = 0.5 * sample_freq;
    let high_freq = if high_freq <= 0.0 { high_freq + nyquist } else { high_freq };
    let fft_bin_width = sample_freq / padded_window_size as f32;
    let mel_low = mel_scale(low_freq);
    let mel_high = mel_scale(high_freq);
    let delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);

    let bins_out = num_fft_bins + 1;
    let mut out = vec![0.0f32; num_bins * bins_out];
    for i in 0..num_bins {
        let left = mel_low + i as f32 * delta;
        let center = mel_low + (i as f32 + 1.0) * delta;
        let right = mel_low + (i as f32 + 2.0) * delta;
        for j in 0..num_fft_bins {
            let mel = mel_scale(fft_bin_width * j as f32);
            let up = (mel - left) / (center - left);
            let down = (right - mel) / (right - center);
            out[i * bins_out + j] = up.min(down).max(0.0);
        }
        // out[i * bins_out + num_fft_bins] stays 0.0 - the padded Nyquist column.
    }
    out
}

/// `_get_strided`'s own `snip_edges=true` frame count: `1 + (n -
/// window_size) / window_shift` (floor division), `0` when `n < window_size`.
fn num_frames(n: usize, window_size: usize, window_shift: usize) -> usize {
    if n < window_size {
        0
    } else {
        1 + (n - window_size) / window_shift
    }
}

/// Kaldi-compatible mel filterbank features: `[n_frames, num_mel_bins]`
/// row-major (time-major, matching `kaldi.fbank`'s own `(m, num_mel_bins)`
/// output), and `n_frames`.
pub fn fbank(samples: &[f32], cfg: &KaldiFbankConfig) -> (Vec<f32>, usize) {
    let window_shift = (cfg.sample_rate * cfg.frame_shift_ms / 1000.0) as usize;
    let window_size = (cfg.sample_rate * cfg.frame_length_ms / 1000.0) as usize;
    let padded = window_size.next_power_of_two();
    let n_frames = num_frames(samples.len(), window_size, window_shift);
    let window = povey_window(window_size);
    let banks = mel_banks(cfg.num_mel_bins, padded, cfg.sample_rate, cfg.low_freq, cfg.high_freq);
    let bins = padded / 2 + 1;

    let mut out = vec![0.0f32; n_frames * cfg.num_mel_bins];
    let mut frame = vec![0.0f32; window_size];
    let mut re = vec![0.0f32; padded];
    let mut im = vec![0.0f32; padded];
    for fr in 0..n_frames {
        let start = fr * window_shift;
        frame.copy_from_slice(&samples[start..start + window_size]);

        // remove_dc_offset: subtract the frame's own mean.
        let mean = frame.iter().sum::<f32>() / window_size as f32;
        for v in frame.iter_mut() {
            *v -= mean;
        }

        // preemphasis: new[0] = old[0]*(1-coeff); new[j] = old[j] - coeff*old[j-1]
        // for j>=1 - the `replicate`-pad boundary `_get_window` implements.
        if cfg.preemphasis_coefficient != 0.0 {
            for j in (1..window_size).rev() {
                frame[j] -= cfg.preemphasis_coefficient * frame[j - 1];
            }
            frame[0] *= 1.0 - cfg.preemphasis_coefficient;
        }

        for j in 0..window_size {
            re[j] = frame[j] * window[j];
        }
        for v in re[window_size..].iter_mut() {
            *v = 0.0;
        }
        for v in im.iter_mut() {
            *v = 0.0;
        }

        fft(&mut re, &mut im);

        let mut spec = vec![0.0f32; bins];
        for (b, s) in spec.iter_mut().enumerate() {
            *s = re[b] * re[b] + im[b] * im[b]; // |rfft|^2 == use_power's spectrum
        }
        for m in 0..cfg.num_mel_bins {
            let mut acc = 0.0f32;
            let row = &banks[m * bins..(m + 1) * bins];
            for b in 0..bins {
                acc += row[b] * spec[b];
            }
            out[fr * cfg.num_mel_bins + m] = acc.max(f32::EPSILON).ln();
        }
    }
    (out, n_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_frames_matches_kaldi_snip_edges_formula() {
        assert_eq!(num_frames(400, 400, 160), 1);
        assert_eq!(num_frames(560, 400, 160), 2);
        assert_eq!(num_frames(720, 400, 160), 3);
        assert_eq!(num_frames(399, 400, 160), 0, "shorter than one window -> zero frames");
    }

    #[test]
    fn povey_window_matches_hann_pow_0_85_by_hand() {
        let w = povey_window(5);
        assert_eq!(w.len(), 5);
        assert!((w[0]).abs() < 1e-6, "edge sample must be exactly zero, got {}", w[0]);
        let hann_mid = 0.5 - 0.5 * (2.0 * PI * 2.0 / 4.0).cos(); // n=2 of 5, N-1=4
        assert!((w[2] - hann_mid.powf(0.85)).abs() < 1e-5);
    }

    #[test]
    fn mel_banks_rows_are_bounded_unit_triangles_that_sum_to_something_nonzero() {
        let cfg = KaldiFbankConfig::cosyvoice();
        let banks = mel_banks(cfg.num_mel_bins, 512, cfg.sample_rate, cfg.low_freq, cfg.high_freq);
        assert_eq!(banks.len(), cfg.num_mel_bins * (512 / 2 + 1));
        for &w in &banks {
            assert!((0.0..=1.0).contains(&w), "mel bank weight {w} out of [0,1]");
        }
        for m in 0..cfg.num_mel_bins {
            let row = &banks[m * (512 / 2 + 1)..(m + 1) * (512 / 2 + 1)];
            assert!(row.iter().any(|&w| w > 0.0), "mel bin {m} has no nonzero weight");
        }
    }

    #[test]
    fn fbank_on_a_synthetic_tone_produces_finite_frames_of_the_right_shape() {
        let cfg = KaldiFbankConfig::cosyvoice();
        let sr = cfg.sample_rate;
        let n = sr as usize * 2; // 2 seconds
        let freq = 440.0f32;
        let samples: Vec<f32> = (0..n).map(|i| 0.2 * (2.0 * PI * freq * i as f32 / sr).sin()).collect();

        let (feat, t) = fbank(&samples, &cfg);
        let window_shift = (cfg.sample_rate * cfg.frame_shift_ms / 1000.0) as usize;
        let window_size = (cfg.sample_rate * cfg.frame_length_ms / 1000.0) as usize;
        assert_eq!(t, num_frames(n, window_size, window_shift));
        assert_eq!(feat.len(), t * cfg.num_mel_bins);
        assert!(feat.iter().all(|v| v.is_finite()), "every fbank value must be finite");
    }
}
