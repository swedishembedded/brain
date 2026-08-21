// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! STFT + mel-spectrogram features (CPU, fp32). Used as the front-end for the
//! ECAPA speaker encoder and the codec's mel reconstruction loss.
//!
//! The exact windowing / mel constants are matched to the Qwen3-TTS reference
//! front-end in Phase 3; the parameters here are configurable so that match is a
//! config change, not a rewrite. FFT is an iterative radix-2 Cooley–Tukey when
//! `n_fft` is a power of two, else Bluestein's algorithm (arbitrary `n_fft`,
//! e.g. CosyVoice's 1920 = 2^7*3*5) - see this module's `fft` function.

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
    /// `torch.stft(center=True)` convention (reflect-pad by `n_fft/2`, the
    /// long-standing default here) vs `center=False` (reflect-pad by
    /// `(n_fft-hop)/2` and no further padding) - the convention
    /// `matcha.utils.audio.mel_spectrogram` uses, which CosyVoice's mel
    /// front end is built on.
    pub center: bool,
}

impl MelConfig {
    /// 24 kHz, 128-mel default (Qwen speaker-encoder class front-end).
    pub fn default_24k() -> MelConfig {
        MelConfig { sample_rate: 24000, n_fft: 1024, hop: 256, win: 1024, n_mels: 128, fmin: 0.0, fmax: 12000.0, slaney: true, center: true }
    }

    /// CosyVoice's mel front end (`matcha.utils.audio.mel_spectrogram`):
    /// 24 kHz, n_fft/win 1920, hop 480, 80 mel bins, fmin 0 / fmax 8000,
    /// librosa Slaney-normalised mel, `center=False`.
    pub fn cosyvoice_24k() -> MelConfig {
        MelConfig { sample_rate: 24000, n_fft: 1920, hop: 480, win: 1920, n_mels: 80, fmin: 0.0, fmax: 8000.0, slaney: true, center: false }
    }
}

fn hann(win: usize) -> Vec<f32> {
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

/// In-place FFT, `re`/`im` length n: radix-2 Cooley–Tukey when `n` is a power
/// of two, else Bluestein's algorithm (chirp-z), which reduces an arbitrary-
/// length DFT to a power-of-two convolution - the technique
/// `asr_frontend::fft_any` already uses in f64 for the ASR front ends,
/// adapted here to this module's f32 precision domain (kept a separate
/// implementation deliberately, per this module's own header comment: f32
/// vs f64 are different precision domains, not a copy to converge).
pub(crate) fn fft(re: &mut [f32], im: &mut [f32]) {
    if re.len().is_power_of_two() {
        fft_radix2(re, im);
    } else {
        fft_bluestein(re, im);
    }
}

/// In-place iterative radix-2 FFT (n a power of two). `re`/`im` length n.
fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
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

/// Bluestein's algorithm for arbitrary-length `n`: `a_k = x_k * w^{k^2/2}`
/// convolved with `b_k = w^{-k^2/2}` (`w = exp(-2pi*i/n)`), the convolution
/// done via a power-of-two radix-2 FFT. Mirrors `asr_frontend::fft_any`'s f64
/// derivation exactly, in f32.
fn fft_bluestein(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let m = (2 * n - 1).next_power_of_two();
    let mut wr = vec![0.0f32; n];
    let mut wi = vec![0.0f32; n];
    for k in 0..n {
        // k^2 mod 2n keeps the angle accurate for large k.
        let kk = (k as u128 * k as u128 % (2 * n as u128)) as f32;
        let ang = -PI * kk / n as f32;
        wr[k] = ang.cos();
        wi[k] = ang.sin();
    }
    let mut ar = vec![0.0f32; m];
    let mut ai = vec![0.0f32; m];
    for k in 0..n {
        ar[k] = re[k] * wr[k] - im[k] * wi[k];
        ai[k] = re[k] * wi[k] + im[k] * wr[k];
    }
    let mut br = vec![0.0f32; m];
    let mut bi = vec![0.0f32; m];
    br[0] = wr[0];
    bi[0] = -wi[0];
    for k in 1..n {
        br[k] = wr[k];
        bi[k] = -wi[k];
        br[m - k] = wr[k];
        bi[m - k] = -wi[k];
    }
    fft_radix2(&mut ar, &mut ai);
    fft_radix2(&mut br, &mut bi);
    for k in 0..m {
        let tr = ar[k] * br[k] - ai[k] * bi[k];
        let ti = ar[k] * bi[k] + ai[k] * br[k];
        ar[k] = tr;
        ai[k] = ti;
    }
    // inverse FFT via the conjugate trick, 1/m scale.
    for v in ai.iter_mut() {
        *v = -*v;
    }
    fft_radix2(&mut ar, &mut ai);
    let inv = 1.0 / m as f32;
    for k in 0..n {
        let cr = ar[k] * inv;
        let ci = -ai[k] * inv;
        re[k] = cr * wr[k] - ci * wi[k];
        im[k] = cr * wi[k] + ci * wr[k];
    }
}

/// Power spectrogram: `[n_frames, n_fft/2+1]` row-major. Reflect-padded by
/// `n_fft/2` each side when `cfg.center` (the `torch.stft(center=True)`
/// convention), or by `(n_fft-hop)/2` when not (the `center=False` manual
/// reflect-pad `matcha.utils.audio.mel_spectrogram` uses) - either way the
/// framing loop below adds no further padding.
pub fn power_spectrogram(samples: &[f32], cfg: &MelConfig) -> (Vec<f32>, usize, usize) {
    let window = hann(cfg.win);
    let pad = if cfg.center { cfg.n_fft / 2 } else { (cfg.n_fft - cfg.hop) / 2 };
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

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    /// O(n^2) reference DFT (f64 accumulation), used only as a test oracle.
    fn naive_dft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = re.len();
        let mut cos_tab = vec![0.0f64; n];
        let mut sin_tab = vec![0.0f64; n];
        for (i, (c, s)) in cos_tab.iter_mut().zip(sin_tab.iter_mut()).enumerate() {
            let ang = -2.0 * std::f64::consts::PI * i as f64 / n as f64;
            *c = ang.cos();
            *s = ang.sin();
        }
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        for k in 0..n {
            let mut sr = 0.0f64;
            let mut si = 0.0f64;
            for t in 0..n {
                let (c, s) = (cos_tab[(k * t) % n], sin_tab[(k * t) % n]);
                sr += re[t] as f64 * c - im[t] as f64 * s;
                si += re[t] as f64 * s + im[t] as f64 * c;
            }
            out_re[k] = sr as f32;
            out_im[k] = si as f32;
        }
        (out_re, out_im)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn check_fft(n: usize, seed: u64) {
        let mut r = Lcg::new(seed);
        let re0 = r.vec(n);
        let im0 = r.vec(n);
        let (want_re, want_im) = naive_dft(&re0, &im0);
        let mut re = re0.clone();
        let mut im = im0.clone();
        fft(&mut re, &mut im);
        let d_re = max_abs(&re, &want_re);
        let d_im = max_abs(&im, &want_im);
        // absolute tolerance scaled by n: an n-point sum of O(1) terms carries
        // O(n) f32 rounding error in the worst case.
        let tol = 1e-4 * n as f32;
        assert!(d_re < tol && d_im < tol, "fft n={n}: maxdiff re={d_re} im={d_im} (tol={tol})");
    }

    #[test]
    fn fft_matches_naive_dft_power_of_two() {
        for &n in &[16usize, 64, 256] {
            check_fft(n, n as u64 + 1);
        }
    }

    /// Mixed-radix (non-power-of-two) sizes, including 1920 = 2^7*3*5, the
    /// exact `n_fft` CosyVoice's mel front end needs.
    #[test]
    fn fft_matches_naive_dft_composite_sizes() {
        for &n in &[15usize, 45, 100, 300, 1920] {
            check_fft(n, n as u64 + 7);
        }
    }

    #[test]
    fn cosyvoice_24k_preset_matches_spec() {
        let cfg = MelConfig::cosyvoice_24k();
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.n_fft, 1920);
        assert_eq!(cfg.hop, 480);
        assert_eq!(cfg.win, 1920);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.fmin, 0.0);
        assert_eq!(cfg.fmax, 8000.0);
        assert!(cfg.slaney);
        assert!(!cfg.center);
    }

    /// `center=False` pads by `(n_fft-hop)/2` each side (vs. `center=True`'s
    /// `n_fft/2`) and applies NO further padding in the framing loop -
    /// matching `matcha.utils.audio.mel_spectrogram`'s manual reflect-pad +
    /// `torch.stft(center=False)` convention CosyVoice's reference uses.
    /// Hand-computed: n_fft=4, hop=2, win=4, samples=[1,3] -> pad=1 each side
    /// -> padded=[3,1,3,1] (reflect, no edge repeat) -> one 4-sample frame ->
    /// windowed=[0, 0.5, 3.0, 0.5] (Hann-4) -> DFT bins [16, 9, 4].
    #[test]
    fn power_spectrogram_center_false_pins_frames_and_matches_hand_computed() {
        let cfg = MelConfig { sample_rate: 8, n_fft: 4, hop: 2, win: 4, n_mels: 1, fmin: 0.0, fmax: 4.0, slaney: true, center: false };
        let samples = [1.0f32, 3.0];
        let (spec, n_frames, bins) = power_spectrogram(&samples, &cfg);
        assert_eq!(n_frames, 1, "one frame: padded len 4 == n_fft, hop 2");
        assert_eq!(bins, 3);
        let want = [16.0f32, 9.0, 4.0];
        for (i, (&got, &w)) in spec.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-3, "bin {i}: got {got} want {w}");
        }
    }

    /// `center=True` keeps the original `n_fft/2` pad (unchanged behaviour).
    #[test]
    fn power_spectrogram_center_true_unchanged() {
        let cfg = MelConfig { sample_rate: 8, n_fft: 4, hop: 2, win: 4, n_mels: 1, fmin: 0.0, fmax: 4.0, slaney: true, center: true };
        let samples = [1.0f32, 3.0, 5.0, 2.0];
        let (_, n_frames, _) = power_spectrogram(&samples, &cfg);
        // pad = n_fft/2 = 2 -> padded len = 4+4 = 8 -> n_frames = 1+(8-4)/2 = 3
        assert_eq!(n_frames, 3);
    }

    /// Cross-check this module's f32 Slaney filterbank against
    /// `asr_frontend::mel_filterbank_slaney` (f64), which is itself
    /// parity-gated against real librosa/transformers goldens - both are
    /// meant to implement the identical `norm="slaney"` formula, so they must
    /// agree bit-for-bit up to f32/f64 rounding.
    #[test]
    fn mel_filterbank_matches_librosa_slaney_reference() {
        let cfg = MelConfig { sample_rate: 16000, n_fft: 400, hop: 160, win: 400, n_mels: 40, fmin: 0.0, fmax: 8000.0, slaney: true, center: true };
        let ours = mel_filterbank(&cfg);
        let theirs = crate::asr_frontend::mel_filterbank_slaney(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin as f64, cfg.fmax as f64);
        assert_eq!(ours.len(), theirs.len());
        let max_abs = ours.iter().zip(&theirs).map(|(&a, &b)| (a - b as f32).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "mel filterbank vs asr_frontend reference maxdiff {max_abs}");
    }
}
