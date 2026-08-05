// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! STFT + mel-spectrogram features (CPU, fp32). Used as the front-end for the
//! ECAPA speaker encoder and the codec's mel reconstruction loss.
//!
//! The exact windowing / mel constants are matched to the Qwen3-TTS reference
//! front-end in Phase 3; the parameters here are configurable so that match is a
//! config change, not a rewrite. FFT is an iterative radix-2 Cooley–Tukey
//! (n_fft must be a power of two).

use std::f32::consts::PI;

/// STFT / mel configuration.
#[derive(Clone, Copy, Debug)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    /// Slaney-style mel (true) vs HTK (false).
    pub slaney: bool,
}

impl MelConfig {
    /// 24 kHz, 128-mel default (Qwen speaker-encoder class front-end).
    pub fn default_24k() -> MelConfig {
        MelConfig { sample_rate: 24000, n_fft: 1024, hop: 256, win: 1024, n_mels: 128, fmin: 0.0, fmax: 12000.0, slaney: true }
    }
}

fn hann(win: usize) -> Vec<f32> {
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

/// In-place iterative radix-2 FFT (n a power of two). `re`/`im` length n.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = cr * re[b] - ci * im[b];
                let ti = cr * im[b] + ci * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Power spectrogram: `[n_frames, n_fft/2+1]` row-major. Centered, reflect pad.
pub fn power_spectrogram(samples: &[f32], cfg: &MelConfig) -> (Vec<f32>, usize, usize) {
    let window = hann(cfg.win);
    let pad = cfg.n_fft / 2;
    let mut padded = vec![0.0f32; samples.len() + 2 * pad];
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    // reflect padding at the edges
    for i in 0..pad {
        padded[pad - 1 - i] = samples.get(i + 1).copied().unwrap_or(0.0);
        let s = samples.len();
        padded[pad + s + i] = samples.get(s.wrapping_sub(2 + i)).copied().unwrap_or(0.0);
    }
    let bins = cfg.n_fft / 2 + 1;
    let n_frames = if padded.len() >= cfg.n_fft { 1 + (padded.len() - cfg.n_fft) / cfg.hop } else { 0 };
    let mut spec = vec![0.0f32; n_frames * bins];
    let mut re = vec![0.0f32; cfg.n_fft];
    let mut im = vec![0.0f32; cfg.n_fft];
    for fr in 0..n_frames {
        let start = fr * cfg.hop;
        for v in im.iter_mut() {
            *v = 0.0;
        }
        for i in 0..cfg.n_fft {
            re[i] = if i < cfg.win { padded[start + i] * window[i] } else { 0.0 };
        }
        fft(&mut re, &mut im);
        for b in 0..bins {
            spec[fr * bins + b] = re[b] * re[b] + im[b] * im[b];
        }
    }
    (spec, n_frames, bins)
}

fn hz_to_mel(f: f32, slaney: bool) -> f32 {
    if slaney {
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            (f - f_min) / f_sp
        }
    } else {
        2595.0 * (1.0 + f / 700.0).log10()
    }
}

fn mel_to_hz(m: f32, slaney: bool) -> f32 {
    if slaney {
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_min + f_sp * m
        }
    } else {
        700.0 * (10f32.powf(m / 2595.0) - 1.0)
    }
}

/// Mel filterbank `[n_mels, n_fft/2+1]` (triangular, optional Slaney norm).
pub fn mel_filterbank(cfg: &MelConfig) -> Vec<f32> {
    let bins = cfg.n_fft / 2 + 1;
    let mel_min = hz_to_mel(cfg.fmin, cfg.slaney);
    let mel_max = hz_to_mel(cfg.fmax, cfg.slaney);
    let points: Vec<f32> = (0..cfg.n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (cfg.n_mels + 1) as f32, cfg.slaney))
        .collect();
    let bin_hz = |b: usize| b as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32;
    let mut fb = vec![0.0f32; cfg.n_mels * bins];
    for m in 0..cfg.n_mels {
        let (l, c, r) = (points[m], points[m + 1], points[m + 2]);
        for b in 0..bins {
            let f = bin_hz(b);
            let w = if f >= l && f <= c {
                (f - l) / (c - l).max(1e-9)
            } else if f > c && f <= r {
                (r - f) / (r - c).max(1e-9)
            } else {
                0.0
            };
            let w = if cfg.slaney { w * 2.0 / (r - l).max(1e-9) } else { w };
            fb[m * bins + b] = w;
        }
    }
    fb
}

/// Log-mel spectrogram `[n_frames, n_mels]` row-major: `log(mel·power + eps)`.
pub fn log_mel(samples: &[f32], cfg: &MelConfig) -> (Vec<f32>, usize) {
    let (spec, n_frames, bins) = power_spectrogram(samples, cfg);
    let fb = mel_filterbank(cfg);
    let mut out = vec![0.0f32; n_frames * cfg.n_mels];
    for fr in 0..n_frames {
        for m in 0..cfg.n_mels {
            let mut acc = 0.0f32;
            for b in 0..bins {
                acc += fb[m * bins + b] * spec[fr * bins + b];
            }
            out[fr * cfg.n_mels + m] = (acc + 1e-6).ln();
        }
    }
    (out, n_frames)
}
