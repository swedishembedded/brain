// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Log-mel front-end for the ECAPA speaker encoder, matching the Qwen3-TTS
//! reference `mel_spectrogram` / `extract_speaker_embedding` exactly:
//!   n_fft = win = 1024, hop = 256, num_mels = 128, fmin = 0, fmax = 12000,
//!   sample_rate = 24000, Hann window, librosa slaney mel filterbank,
//!   `center=False` with an explicit reflect pad of `(n_fft - hop)//2 = 384`,
//!   magnitude = sqrt(power + 1e-9), compression = log(clamp(mel, 1e-5)).
//!
//! The output is `[n_frames, 128]` row-major (time-major), i.e. the reference's
//! `mel.transpose(1, 2)` — exactly what [`crate::SpeakerEncoder::embed`] expects.
//! Only the front-end (`embed_wav`); the parity test feeds a pre-dumped mel.

use std::f32::consts::PI;

use audio::mel::{mel_filterbank, MelConfig};

pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const WIN: usize = 1024;
pub const N_MELS: usize = 128;
pub const FMIN: f32 = 0.0;
pub const FMAX: f32 = 12000.0;

fn hann(win: usize) -> Vec<f32> {
    // torch.hann_window default (periodic): denominator N, not N-1.
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

/// In-place iterative radix-2 Cooley–Tukey FFT (n a power of two).
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
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

/// PyTorch-style `F.pad(mode="reflect")` of a 1-D signal by `p` on each side
/// (no border-element repeat). Requires `p < len`.
fn reflect_pad_signal(x: &[f32], p: usize) -> Vec<f32> {
    let l = x.len();
    let mut o = vec![0.0f32; l + 2 * p];
    for (j, slot) in o.iter_mut().enumerate() {
        let mut idx = j as isize - p as isize;
        if idx < 0 {
            idx = -idx;
        }
        if idx as usize >= l {
            idx = 2 * (l as isize - 1) - idx;
        }
        *slot = x[idx as usize];
    }
    o
}

/// Reference log-mel: `[n_frames, 128]` row-major. `samples` is mono 24 kHz.
pub fn log_mel(samples: &[f32]) -> (Vec<f32>, usize) {
    let cfg = MelConfig {
        sample_rate: 24000,
        n_fft: N_FFT,
        hop: HOP,
        win: WIN,
        n_mels: N_MELS,
        fmin: FMIN,
        fmax: FMAX,
        slaney: true,
        center: false,
    };
    let fb = mel_filterbank(&cfg); // [N_MELS, bins]
    let bins = N_FFT / 2 + 1;
    let window = hann(WIN);

    // center=False -> explicit reflect pad of (n_fft - hop)//2.
    let pad = (N_FFT - HOP) / 2;
    let padded = if samples.is_empty() { Vec::new() } else { reflect_pad_signal(samples, pad) };
    let n_frames = if padded.len() >= N_FFT { 1 + (padded.len() - N_FFT) / HOP } else { 0 };

    let mut out = vec![0.0f32; n_frames * N_MELS];
    let mut re = vec![0.0f32; N_FFT];
    let mut im = vec![0.0f32; N_FFT];
    for fr in 0..n_frames {
        let start = fr * HOP;
        for v in im.iter_mut() {
            *v = 0.0;
        }
        for i in 0..N_FFT {
            re[i] = if i < WIN { padded[start + i] * window[i] } else { 0.0 };
        }
        fft(&mut re, &mut im);
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for b in 0..bins {
                let mag = (re[b] * re[b] + im[b] * im[b] + 1e-9).sqrt();
                acc += fb[m * bins + b] * mag;
            }
            out[fr * N_MELS + m] = acc.max(1e-5).ln();
        }
    }
    (out, n_frames)
}
