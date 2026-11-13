// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rational-ratio polyphase resampling (Kaiser-windowed sinc), the accurate
//! sibling of [`crate::resample_linear`] - plain linear interpolation has no
//! anti-alias filter at all, so it cannot hold parity against a reference
//! resampler (e.g. torchaudio's windowed-sinc `resample`) for a rate change
//! like 24 kHz -> 16 kHz. Filter design follows the same Kaiser-windowed-sinc
//! technique `crates/ltxv/src/vocoder.rs`'s anti-aliasing filters are BUILT
//! from - that filter ships pre-computed inside the checkpoint and is only
//! ever loaded there, never derived; the `kaiser_sinc_lowpass` helper below is
//! this workspace's first from-scratch derivation of one. [`sinc_hann`] is
//! torchaudio's own Hann-windowed resampler, for references that resample
//! with it.

use std::f64::consts::PI;

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Zeroth-order modified Bessel function of the first kind, via its power
/// series - converges in a handful of terms for the beta values a Kaiser
/// window design needs.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let mut k = 1.0f64;
    while term > sum * 1e-16 {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Symmetric Kaiser window, `n` points.
fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = bessel_i0(beta);
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = (2.0 * i as f64 - m) / m;
            bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / denom
        })
        .collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Kaiser beta from a target stopband attenuation (Kaiser's own empirical
/// fit; Oppenheim & Schafer, *Discrete-Time Signal Processing*).
fn kaiser_beta(atten_db: f64) -> f64 {
    if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db >= 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    }
}

/// Kaiser-windowed sinc lowpass at cutoff `fc` (cycles/sample, half-sampling-rate =
/// 0.5), `half_taps` zero crossings on each side of the interpolation factor
/// `up`, scaled by `gain` (the `up`-factor amplitude compensation a
/// zero-stuffed interpolation filter needs). Length `2*half_taps*up + 1`.
fn kaiser_sinc_lowpass(fc: f64, half_taps: usize, up: u32, beta: f64, gain: f64) -> Vec<f64> {
    let flen = 2 * half_taps * up as usize + 1;
    let center = (flen - 1) as f64 / 2.0;
    let win = kaiser(flen, beta);
    (0..flen).map(|n| gain * 2.0 * fc * sinc(2.0 * fc * (n as f64 - center)) * win[n]).collect()
}

/// Filter quality knobs: zero crossings per phase and target stopband
/// attenuation. Fixed rather than exposed - every caller wants "as accurate
/// as `resample_linear` is fast", not a tuning surface.
const HALF_TAPS: usize = 32;
const ATTEN_DB: f64 = 80.0;

/// Resample `samples` by the exact rational ratio `up/down` (reduced by their
/// gcd) via a Kaiser-windowed-sinc polyphase filter. Implements the direct
/// polyphase convolution (zero-stuff by `up`, lowpass at
/// `0.5/max(up,down)`, decimate by `down`) without materializing the
/// zero-stuffed intermediate: output sample `j` reads
/// `h[delay + j*down - i*up]` against input sample `i`, for every `i` the
/// filter's support reaches.
pub fn rational(samples: &[f32], up: u32, down: u32) -> Vec<f32> {
    assert!(up > 0 && down > 0, "resample::rational: up/down must be nonzero");
    if samples.is_empty() {
        return Vec::new();
    }
    let g = gcd(up, down);
    let (l, m) = (up / g, down / g);
    if l == m {
        return samples.to_vec();
    }
    let fc = 0.5 / l.max(m) as f64;
    let beta = kaiser_beta(ATTEN_DB);
    let filt = kaiser_sinc_lowpass(fc, HALF_TAPS, l.max(m), beta, l as f64);
    let flen = filt.len() as i64;
    let delay = (flen - 1) / 2;
    let n_in = samples.len() as i64;
    let out_len = ((samples.len() as u64 * l as u64).div_ceil(m as u64)) as usize;
    let li = l as i64;
    let mut out = vec![0.0f32; out_len];
    for (j, o) in out.iter_mut().enumerate() {
        let t = delay + j as i64 * m as i64;
        // k = t - i*l must land in [0, flen) -> i in ((t-flen)/l, t/l].
        let i_max = t.div_euclid(li);
        let i_min = (t - flen + li).div_euclid(li);
        let i_lo = i_min.max(0);
        let i_hi = i_max.min(n_in - 1);
        let mut acc = 0.0f64;
        let mut i = i_lo;
        while i <= i_hi {
            let k = (t - i * li) as usize;
            acc += filt[k] * samples[i as usize] as f64;
            i += 1;
        }
        *o = acc as f32;
    }
    out
}


/// `torchaudio.transforms.Resample(orig_rate, new_rate)` with its defaults,
/// to float rounding: a Hann-windowed sinc (`lowpass_filter_width = 6`,
/// `rolloff = 0.99`), one polyphase filter per output phase of the reduced
/// ratio, applied as a stride-`orig` convolution over the zero-padded input
/// and cut to `ceil(new * len / orig)` samples. Use it where a reference
/// resamples with torchaudio and parity matters; [`rational`] is the
/// workspace's own higher-attenuation design.
///
/// The kernel is built in f64 and narrowed to f32 as torchaudio caches it -
/// except the per-phase offset `-j / new`, which torchaudio divides as an
/// integer tensor and so computes in f32 before promoting it.
pub fn sinc_hann(samples: &[f32], orig_rate: u32, new_rate: u32) -> Vec<f32> {
    const LOWPASS_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let g = gcd(orig_rate, new_rate);
    let (orig, new) = ((orig_rate / g) as usize, (new_rate / g) as usize);
    if orig == new {
        return samples.to_vec();
    }
    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_WIDTH * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    let kernels: Vec<f32> = (0..new)
        .flat_map(|j| {
            let phase = (-(j as f32) / new as f32) as f64;
            (0..taps).map(move |k| {
                let t = ((phase + (k as f64 - width as f64) / orig as f64) * base).clamp(-LOWPASS_WIDTH, LOWPASS_WIDTH);
                let window = (t * PI / LOWPASS_WIDTH / 2.0).cos().powi(2);
                let t = t * PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                (sinc * window * base / orig as f64) as f32
            })
        })
        .collect();

    let len = samples.len();
    let mut padded = vec![0f32; width + len + width + orig];
    padded[width..width + len].copy_from_slice(samples);
    let frames = (padded.len() - taps) / orig + 1;
    let target = ((new as f64 * len as f64 / orig as f64) as f32).ceil() as usize;
    let mut out = Vec::with_capacity(frames * new);
    for m in 0..frames {
        let window = &padded[m * orig..m * orig + taps];
        for kernel in kernels.chunks_exact(taps) {
            out.push(window.iter().zip(kernel).map(|(&x, &k)| x as f64 * k as f64).sum::<f64>() as f32);
        }
    }
    out.truncate(target);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_i0_at_zero_is_one() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = kaiser(5, 0.0);
        for v in w {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn gcd_reduces_ratio() {
        assert_eq!(gcd(16000, 24000), 8000);
    }
}
