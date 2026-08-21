// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `HiFTGenerator.decode` forward (CosyVoice 2, NON-causal): mel -> f0
//! (`ConvRNNF0Predictor`) -> NSF harmonic source excitation
//! (`SourceModuleHnNSF`/`SineGen2`) -> conv trunk (BigVGAN-style upsample +
//! Snake `ResBlock`s, source-fused per stage) -> ISTFT head -> 24 kHz
//! waveform. Ported algorithm-for-algorithm from
//! `resources/cosyvoice/source/cosyvoice/hifigan/{generator,f0_predictor}.py`
//! (read directly, not from the paper). See `hift_config`'s module doc for
//! the CausalHiFTGenerator (CosyVoice 3) gap this does not cover.
//!
//! Every conv reuses `audio::conv`'s CPU reference kernels
//! (`conv1d_ref`/`convtr1d_ref`, the same math `wgsl/conv1d.wgsl` computes -
//! see that module's doc), `audio::snake`/`audio::act` for the Snake/ELU
//! activations, and `audio::istft` for the STFT/ISTFT pair - a host (not
//! device-dispatched) forward, matching this crate's own `llm.rs` (which
//! wraps `qwen3::Qwen`'s own compute strategy rather than re-deriving GPU
//! Step dispatch here). Getting this correct first and fast later is the
//! deliberate order: a GPU-dispatched conv trunk is a natural follow-up
//! performance milestone once this forward is parity-proven, not a requirement of this
//! one.
//!
//! ## The RNG-crossing gap (read before touching this file's noise draw)
//!
//! `SourceModuleHnNSF`'s `SineGen2` draws `torch.rand`/`torch.randn` from
//! PyTorch's global Mersenne-Twister RNG on every call (verified directly
//! against `generator.py`, not assumed) - real HiFT output is NOT
//! reproducible run-to-run without reseeding, exactly the gap
//! `crate::sampling`'s module doc names for `ras_sampling`'s `multinomial`
//! draws.
//!
//! **One empirical finding narrows that gap further than it first looks.**
//! `SineGen2._f02sine`'s `rand_ini` draw (the "initial phase noise", added to
//! `rad_values` at the single FIRST full-waveform-rate sample, before the
//! downsample-cumsum-upsample chain) is PROVABLY INERT at HiFT's real
//! `upsample_scale=480`: the downsample step's `F.interpolate(...,
//! mode="linear", align_corners=False)` samples its first output element at
//! fractional input position `(0+0.5)*480 - 0.5 = 239.5`, reading input
//! indices 239/240 - NEVER index 0, the one `rand_ini` perturbs. Verified
//! empirically against the real `SineGen2._f02sine` (not derived by hand):
//! two runs seeded differently on a realistic f0 contour produce
//! `(sines_a - sines_b).abs().max() == 0.0`, bit-exact. So `rand_ini` is not
//! modeled here at all - the reference's own math discards it regardless of
//! its value. The ONLY random draw that genuinely reaches HiFT's output is
//! `torch.randn_like(sine_waves)` (the additive per-sample noise, `[T,
//! harmonics]`), and [`nsf_source_forward`] takes that as an explicit
//! parameter rather than drawing it internally - a parity test injects the
//! exact values captured from a real, reseeded `hift.inference()` run
//! (`testdata/golden/cosyvoice/hift_real_nsf_noise.f32`, captured by an
//! ad-hoc script against the real checkpoint - see
//! `crates/cosyvoice/tests/hift_parity.rs`'s module doc for exactly how),
//! which lets the conv-trunk + NSF-source + ISTFT math be verified
//! bit-exactly without reimplementing PyTorch's Mersenne-Twister transform in
//! Rust. [`forward_seeded`] (the production entry point) draws its own
//! stream from `data::rng::Rng::next_gaussian` (matching `crate::sampling`'s
//! production RNG choice) - NOT bit-exact with the reference.

use audio::act::elu_ref;
use audio::conv::{conv1d_ref, convtr1d_ref, Conv1d};
use audio::istft::{istft as istft_overlap_add, stft as stft_forward, StftConfig};
use audio::snake::{snake1d_ref, Snake1d};
use data::rng::Rng;

use crate::hift_config::{HiftConfig, RESBLOCK_DILATIONS};
use crate::hift_import::{ConvW, F0PredictorW, HiftWeights, ResBlockW};

// ---------------------------------------------------------------------------
// small host-side numeric helpers
// ---------------------------------------------------------------------------

fn leaky_relu(x: &[f32], slope: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { v * slope }).collect()
}

fn add_bias_ncl(y: &mut [f32], bias: &[f32], l: usize) {
    for (ci, b) in bias.iter().enumerate() {
        for li in 0..l {
            y[ci * l + li] += *b;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d(x: &[f32], w: &ConvW, cin: usize, l: usize, cout: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> (Vec<f32>, usize) {
    let lo = Conv1d::out_len(l as u32, k as u32, stride as u32, pad as u32, pad as u32, dilation as u32) as usize;
    let c = Conv1d {
        n: 1,
        cin: cin as u32,
        l: l as u32,
        cout: cout as u32,
        k: k as u32,
        stride: stride as u32,
        pad: pad as u32,
        dilation: dilation as u32,
        groups: 1,
        lo: lo as u32,
    };
    let mut y = conv1d_ref(&c, x, &w.weight);
    add_bias_ncl(&mut y, &w.bias, lo);
    (y, lo)
}

#[allow(clippy::too_many_arguments)]
fn convtr1d(x: &[f32], w: &ConvW, cin: usize, l: usize, cout: usize, k: usize, stride: usize, pad: usize) -> (Vec<f32>, usize) {
    let lo = Conv1d::out_len_transposed(l as u32, k as u32, stride as u32, pad as u32, 0, 1) as usize;
    let c = Conv1d { n: 1, cin: cin as u32, l: l as u32, cout: cout as u32, k: k as u32, stride: stride as u32, pad: pad as u32, dilation: 1, groups: 1, lo: lo as u32 };
    let mut y = convtr1d_ref(&c, x, &w.weight);
    add_bias_ncl(&mut y, &w.bias, lo);
    (y, lo)
}

const SNAKE_EPS: f32 = 1e-9;

fn snake(x: &[f32], alpha: &[f32], c: usize, l: usize) -> Vec<f32> {
    let sc = Snake1d { rows: 1, c: c as u32, inner: l as u32, eps: SNAKE_EPS };
    snake1d_ref(&sc, x, alpha)
}

/// `ResBlock.forward`: 3 sequential `(Snake -> conv1[dilation] -> Snake ->
/// conv2[dilation=1]) + x` branches (dilations `[1,3,5]`), the SAME `x`
/// threaded through all three - NOT the 3-way average `decode` applies one
/// level up to the 3 `resblocks[i*3+j]`/`kernel_sizes` instances.
fn resblock_forward(x0: &[f32], rb: &ResBlockW, c: usize, l: usize, k: usize) -> Vec<f32> {
    let mut x = x0.to_vec();
    // `idx` indexes several parallel arrays (`RESBLOCK_DILATIONS`, `alpha1`,
    // `convs1`, ...) -- clippy's `needless_range_loop` heuristic only sees
    // the first use.
    #[allow(clippy::needless_range_loop)]
    for idx in 0..3 {
        let d = RESBLOCK_DILATIONS[idx] as usize;
        let pad1 = (k - 1) * d / 2;
        let xt = snake(&x, &rb.alpha1[idx], c, l);
        let (xt, lo1) = conv1d(&xt, &rb.convs1[idx], c, l, c, k, 1, pad1, d);
        debug_assert_eq!(lo1, l, "resblock convs1[{idx}] changed length");
        let xt = snake(&xt, &rb.alpha2[idx], c, l);
        let pad2 = (k - 1) / 2;
        let (xt, lo2) = conv1d(&xt, &rb.convs2[idx], c, l, c, k, 1, pad2, 1);
        debug_assert_eq!(lo2, l, "resblock convs2[{idx}] changed length");
        for (xi, xti) in x.iter_mut().zip(&xt) {
            *xi += *xti;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// ConvRNNF0Predictor (no RNN, despite the name)
// ---------------------------------------------------------------------------

/// `mel [in_channels, T] -> condnet (5x weight-normed Conv1d(k=3,pad=1) +
/// ELU) -> classifier (Linear(cond_channels,1)) -> abs()` -> `f0 [T]` (Hz).
pub fn f0_predictor_forward(w: &F0PredictorW, mel: &[f32], in_channels: usize, cond_channels: usize, t: usize) -> Vec<f32> {
    let mut cur = mel.to_vec();
    let mut cin = in_channels;
    for cw in &w.condnet {
        let (y, lo) = conv1d(&cur, cw, cin, t, cond_channels, 3, 1, 1, 1);
        debug_assert_eq!(lo, t);
        cur = elu_ref(&y, 1.0);
        cin = cond_channels;
    }
    let mut f0 = vec![0f32; t];
    for (ti, out) in f0.iter_mut().enumerate() {
        let mut acc = w.classifier_b;
        for c in 0..cond_channels {
            acc += w.classifier_w[c] * cur[c * t + ti];
        }
        *out = acc.abs();
    }
    f0
}

// ---------------------------------------------------------------------------
// NSF harmonic source (SourceModuleHnNSF / SineGen2, causal=False)
// ---------------------------------------------------------------------------

fn nearest_upsample(x: &[f32], scale: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() * scale);
    for &v in x {
        for _ in 0..scale {
            out.push(v);
        }
    }
    out
}

/// `F.interpolate(x, mode="linear", align_corners=False)`'s exact half-pixel
/// coordinate formula, used for BOTH the down- and up-sampling legs of
/// `SineGen2._f02sine`'s phase accumulation (same op either direction).
/// Verified numerically against the real op (max diff ~2e-4 at magnitude
/// ~3400, i.e. float32 accumulation-order noise, not a formula error).
fn linear_interp(x: &[f32], lin: usize, lout: usize) -> Vec<f32> {
    if lin == 1 {
        return vec![x[0]; lout];
    }
    let scale = lin as f32 / lout as f32;
    let mut out = vec![0f32; lout];
    for (i, o) in out.iter_mut().enumerate() {
        let mut src = (i as f32 + 0.5) * scale - 0.5;
        src = src.max(0.0).min((lin - 1) as f32);
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(lin - 1);
        let frac = src - i0 as f32;
        *o = x[i0] * (1.0 - frac) + x[i1] * frac;
    }
    out
}

/// `SourceModuleHnNSF.forward` (`sine_merge` output only - `noise`/`uv`, the
/// other two return values, are dropped by every real caller in this
/// codepath, matching the reference's own `s, _, _ = self.m_source(s)`).
/// `f0_mel` is `[T_mel]` Hz (from [`f0_predictor_forward`]); `randn_noise` is
/// the caller-supplied `[T_full * harmonics]` standard-normal draw (`T_full =
/// T_mel * cfg.nsf_upsample_scale()`) - see this module's doc for why
/// `rand_ini` needs no parameter at all. Returns the excitation `s`, `[T_full]`.
pub fn nsf_source_forward(f0_mel: &[f32], cfg: &HiftConfig, w: &HiftWeights, randn_noise: &[f32]) -> Vec<f32> {
    let scale = cfg.nsf_upsample_scale() as usize;
    let harm = cfg.harmonics() as usize;
    let t_mel = f0_mel.len();
    let t_full = t_mel * scale;
    assert_eq!(randn_noise.len(), t_full * harm, "nsf_source_forward: randn_noise length");

    let f0_full = nearest_upsample(f0_mel, scale);
    let sr = cfg.sampling_rate as f32;

    let mut sines = vec![0f32; t_full * harm]; // [T_full, harm] row-major
    for h in 0..harm {
        let mult = (h + 1) as f32;
        let rad: Vec<f32> = f0_full.iter().map(|&f| (f * mult / sr).rem_euclid(1.0)).collect();
        // rand_ini deliberately not added here - see module doc: provably
        // inert at this upsample_scale, discarded by the very next line's
        // downsample interpolation.
        let down = linear_interp(&rad, t_full, t_mel);
        let mut phase_mel = vec![0f32; t_mel];
        let mut acc = 0f32;
        for (t, d) in down.iter().enumerate() {
            acc += *d;
            phase_mel[t] = acc * std::f32::consts::TAU;
        }
        let scaled: Vec<f32> = phase_mel.iter().map(|&p| p * scale as f32).collect();
        let phase_full = linear_interp(&scaled, t_mel, t_full);
        for (t, p) in phase_full.iter().enumerate() {
            sines[t * harm + h] = p.sin();
        }
    }

    let mut sine_waves = vec![0f32; t_full * harm];
    for (t, &f0) in f0_full.iter().enumerate() {
        let uv = if f0 > cfg.nsf_voiced_threshold { 1.0f32 } else { 0.0 };
        let noise_amp = uv * cfg.nsf_sigma + (1.0 - uv) * cfg.nsf_alpha / 3.0;
        for h in 0..harm {
            let idx = t * harm + h;
            let s = sines[idx] * cfg.nsf_alpha;
            sine_waves[idx] = s * uv + noise_amp * randn_noise[idx];
        }
    }

    let mut s = vec![0f32; t_full];
    for (t, out) in s.iter_mut().enumerate() {
        let mut acc = w.m_source_linear_b;
        for h in 0..harm {
            acc += w.m_source_linear_w[h] * sine_waves[t * harm + h];
        }
        *out = acc.tanh();
    }
    s
}

// ---------------------------------------------------------------------------
// _stft / _istft (center=True, reflect pad - torch.stft/istft's own default)
// ---------------------------------------------------------------------------

/// Reflect-pad by `pad` each side, matching `audio::mel::power_spectrogram`'s
/// `center=True` convention (`torch.stft(center=True)`'s default reflect,
/// excluding the edge sample itself) - duplicated here rather than shared
/// since this file may not touch `crates/audio`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0f32; n + 2 * pad];
    out[pad..pad + n].copy_from_slice(x);
    for i in 0..pad {
        out[pad - 1 - i] = x.get(i + 1).copied().unwrap_or(0.0);
        out[pad + n + i] = x.get(n.wrapping_sub(2 + i)).copied().unwrap_or(0.0);
    }
    out
}

fn stft_center(x: &[f32], cfg: &StftConfig) -> (Vec<f32>, Vec<f32>, usize) {
    let padded = reflect_pad(x, cfg.n_fft / 2);
    stft_forward(&padded, cfg)
}

fn istft_center(re: &[f32], im: &[f32], n_frames: usize, cfg: &StftConfig) -> Vec<f32> {
    let raw = istft_overlap_add(re, im, n_frames, cfg);
    let pad = cfg.n_fft / 2;
    raw[pad..raw.len() - pad].to_vec()
}

// ---------------------------------------------------------------------------
// decode: conv trunk + source fusion + ISTFT head
// ---------------------------------------------------------------------------

pub struct DecodeOutput {
    /// `[stft_bins, n_frames]` channel-major, pre-ISTFT.
    pub magnitude: Vec<f32>,
    /// `[stft_bins, n_frames]` channel-major, pre-ISTFT.
    pub phase: Vec<f32>,
    /// `[T_full]`, clamped to `[-audio_limit, audio_limit]`.
    pub waveform: Vec<f32>,
}

/// `HiFTGenerator.decode`: `mel [in_channels, T_mel]` + the NSF `excitation
/// [T_full]` -> `DecodeOutput`.
pub fn decode(mel: &[f32], t_mel: usize, excitation: &[f32], cfg: &HiftConfig, w: &HiftWeights) -> DecodeOutput {
    let base = cfg.base_channels as usize;
    let bins = cfg.stft_bins() as usize;
    let stft_cfg = StftConfig { n_fft: cfg.n_fft as usize, hop: cfg.hop_len as usize, win: cfg.n_fft as usize };

    let (s_re, s_im, n_frames_s) = stft_center(excitation, &stft_cfg);
    // s_stft: [n_fft+2, n_frames] = real bins then imag bins, channel-major
    // (torch.cat([real, imag], dim=1) on a [B, bins, T] pair).
    let mut s_stft = vec![0f32; 2 * bins * n_frames_s];
    for b in 0..bins {
        for f in 0..n_frames_s {
            s_stft[b * n_frames_s + f] = s_re[f * bins + b];
            s_stft[(bins + b) * n_frames_s + f] = s_im[f * bins + b];
        }
    }

    let (mut x, mut l) = conv1d(mel, &w.conv_pre, cfg.in_channels as usize, t_mel, base, 7, 1, 3, 1);
    let mut c = base;
    let down_strides = cfg.source_downsample_strides();

    // `i` indexes several parallel arrays plus a raw `i == 2` special case --
    // clippy's `needless_range_loop` heuristic only sees the first use.
    #[allow(clippy::needless_range_loop)]
    for i in 0..3usize {
        let u = cfg.upsample_rates[i] as usize;
        let k = cfg.upsample_kernel_sizes[i] as usize;
        let pad = (k - u) / 2;
        let cout = base / (1 << (i + 1));

        let xa = leaky_relu(&x, cfg.lrelu_slope);
        let (mut xu, mut lu) = convtr1d(&xa, &w.ups[i], c, l, cout, k, u, pad);
        c = cout;

        if i == 2 {
            // ReflectionPad1d((1, 0)): 1-sample left pad, mirrored (output[0]
            // = input[1]).
            let mirror = 1usize.min(lu - 1);
            let mut padded = vec![0f32; c * (lu + 1)];
            for ch in 0..c {
                padded[ch * (lu + 1)] = xu[ch * lu + mirror];
                padded[ch * (lu + 1) + 1..ch * (lu + 1) + 1 + lu].copy_from_slice(&xu[ch * lu..ch * lu + lu]);
            }
            xu = padded;
            lu += 1;
        }
        l = lu;

        let stride_i = down_strides[i] as usize;
        let (kd, padd) = if stride_i == 1 { (1usize, 0usize) } else { (stride_i * 2, stride_i / 2) };
        let (si_raw, lo) = conv1d(&s_stft, &w.source_downs[i], 2 * bins, n_frames_s, c, kd, stride_i, padd, 1);
        assert_eq!(lo, l, "source_downs[{i}]: length {lo} != trunk length {l}");
        let src_k = cfg.source_resblock_kernel_sizes[i] as usize;
        let si = resblock_forward(&si_raw, &w.source_resblocks[i], c, l, src_k);
        for (xi, si_v) in xu.iter_mut().zip(&si) {
            *xi += *si_v;
        }
        x = xu;

        let mut acc = vec![0f32; x.len()];
        for j in 0..3usize {
            let k = cfg.resblock_kernel_sizes[j] as usize;
            let r = resblock_forward(&x, &w.resblocks[i * 3 + j], c, l, k);
            for (a, rv) in acc.iter_mut().zip(&r) {
                *a += *rv;
            }
        }
        for v in acc.iter_mut() {
            *v /= 3.0;
        }
        x = acc;
    }

    let xa = leaky_relu(&x, 0.01); // F.leaky_relu(x) default slope - NOT cfg.lrelu_slope.
    let (post, lo) = conv1d(&xa, &w.conv_post, c, l, 2 * bins, 7, 1, 3, 1);
    assert_eq!(lo, l);

    let mut magnitude = vec![0f32; bins * l];
    let mut phase = vec![0f32; bins * l];
    for b in 0..bins {
        for t in 0..l {
            magnitude[b * l + t] = post[b * l + t].exp();
            phase[b * l + t] = post[(bins + b) * l + t].sin(); // "actually, sin is redundancy" - reference's own comment.
        }
    }

    let mut re = vec![0f32; bins * l];
    let mut im = vec![0f32; bins * l];
    for b in 0..bins {
        for t in 0..l {
            let m = magnitude[b * l + t].min(1e2);
            let p = phase[b * l + t];
            re[t * bins + b] = m * p.cos();
            im[t * bins + b] = m * p.sin();
        }
    }

    let raw_waveform = istft_center(&re, &im, l, &stft_cfg);
    let waveform: Vec<f32> = raw_waveform.iter().map(|&v| v.clamp(-cfg.audio_limit, cfg.audio_limit)).collect();

    DecodeOutput { magnitude, phase, waveform }
}

// ---------------------------------------------------------------------------
// top level
// ---------------------------------------------------------------------------

pub struct HiftOutput {
    pub f0: Vec<f32>,
    pub magnitude: Vec<f32>,
    pub phase: Vec<f32>,
    pub waveform: Vec<f32>,
}

/// Full forward: `mel [in_channels, T_mel]` -> waveform, given an explicit
/// NSF noise draw (see this module's doc for why it is a parameter rather
/// than an internal RNG call - this is the entry point
/// `crates/cosyvoice/tests/hift_parity.rs` uses with the captured golden
/// noise). [`forward_seeded`] is the production entry point.
pub fn forward(mel: &[f32], t_mel: usize, cfg: &HiftConfig, w: &HiftWeights, randn_noise: &[f32]) -> HiftOutput {
    assert_eq!(mel.len(), cfg.in_channels as usize * t_mel, "forward: mel length != in_channels*T_mel");
    let f0 = f0_predictor_forward(&w.f0_predictor, mel, cfg.in_channels as usize, cfg.f0_cond_channels as usize, t_mel);
    let excitation = nsf_source_forward(&f0, cfg, w, randn_noise);
    let out = decode(mel, t_mel, &excitation, cfg, w);
    HiftOutput { f0, magnitude: out.magnitude, phase: out.phase, waveform: out.waveform }
}

/// Production entry point: draws the NSF source's additive noise from
/// `data::rng::Rng::next_gaussian` (matching `crate::sampling`'s own
/// production RNG choice), seeded by the caller. NOT bit-exact with the
/// reference's `torch.randn_like` stream - see this module's doc.
pub fn forward_seeded(mel: &[f32], t_mel: usize, cfg: &HiftConfig, w: &HiftWeights, seed: u64) -> HiftOutput {
    let n = t_mel * cfg.nsf_upsample_scale() as usize * cfg.harmonics() as usize;
    let mut rng = Rng::new(seed);
    let randn: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
    forward(mel, t_mel, cfg, w, &randn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hift_import::ResBlockW;

    fn tiny_cfg() -> HiftConfig {
        HiftConfig::cosyvoice2()
    }

    fn filled(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
    }

    /// Fan-in-scaled random fill (Xavier-ish) so a chain of ~30 conv/resblock
    /// layers with real-sized (up to 512) channel counts stays numerically
    /// bounded through `exp()` in the ISTFT head - a flat `filled(..., 0.2)`
    /// scale blows up across that much depth even though it is only a
    /// synthetic-weight smoke test.
    fn filled_fanin(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
        let scale = 1.0 / (fan_in.max(1) as f32).sqrt();
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (rng.next_f32() - 0.5) * scale).collect()
    }

    fn tiny_convw(cout: usize, cin: usize, k: usize, seed: u64) -> ConvW {
        ConvW { weight: filled_fanin(cout * cin * k, cin * k, seed), bias: filled(cout, seed + 1) }
    }

    /// `ConvTranspose1d` weight layout is `[Cin, Cout, K]` but its bias is
    /// dimensioned by `Cout` (the actual output channels) - distinct from
    /// [`tiny_convw`], which conflates its first arg with both the weight's
    /// leading dim and the bias length (correct for plain `Conv1d`, wrong
    /// here).
    fn tiny_convtrw(cin: usize, cout: usize, k: usize, seed: u64) -> ConvW {
        ConvW { weight: filled_fanin(cin * cout * k, cin * k, seed), bias: filled(cout, seed + 1) }
    }

    fn tiny_resblock(c: usize, k: usize, seed: u64) -> ResBlockW {
        let mk = |s: u64| tiny_convw(c, c, k, s);
        ResBlockW {
            convs1: [mk(seed), mk(seed + 10), mk(seed + 20)],
            convs2: [mk(seed + 30), mk(seed + 40), mk(seed + 50)],
            alpha1: [filled(c, seed + 60), filled(c, seed + 61), filled(c, seed + 62)],
            alpha2: [filled(c, seed + 63), filled(c, seed + 64), filled(c, seed + 65)],
        }
    }

    /// Tiny end-to-end smoke test: fake weights at the REAL shapes, no real
    /// checkpoint needed. Exercises
    /// every step kind (conv/convtr/snake/elu/leaky_relu/reflection-pad/
    /// source-fusion/resblock-average/ISTFT) and asserts the whole pipeline
    /// stays finite end to end.
    #[test]
    fn tiny_forward_produces_a_finite_waveform_at_the_real_shapes() {
        let cfg = tiny_cfg();
        let base = cfg.base_channels as usize;
        let bins = cfg.source_stft_channels() as usize; // n_fft+2

        let f0_predictor = F0PredictorW {
            condnet: [
                tiny_convw(base, 80, 3, 1),
                tiny_convw(base, base, 3, 2),
                tiny_convw(base, base, 3, 3),
                tiny_convw(base, base, 3, 4),
                tiny_convw(base, base, 3, 5),
            ],
            classifier_w: filled(base, 6),
            classifier_b: 0.01,
        };

        let mut ups = Vec::with_capacity(3);
        let mut source_downs = Vec::with_capacity(3);
        let mut source_resblocks = Vec::with_capacity(3);
        let mut resblocks = Vec::with_capacity(9);
        // `i` indexes several parallel arrays -- clippy's `needless_range_loop`
        // heuristic only sees the first use.
        #[allow(clippy::needless_range_loop)]
        for i in 0..3usize {
            let cin = base / (1 << i);
            let cout = base / (1 << (i + 1));
            let k = cfg.upsample_kernel_sizes[i] as usize;
            ups.push(tiny_convtrw(cin, cout, k, 100 + i as u64));
            let stride = cfg.source_downsample_strides()[i] as usize;
            let dk = if stride == 1 { 1 } else { stride * 2 };
            source_downs.push(tiny_convw(cout, bins, dk, 200 + i as u64));
            source_resblocks.push(tiny_resblock(cout, cfg.source_resblock_kernel_sizes[i] as usize, 300 + i as u64 * 10));
            for j in 0..3usize {
                resblocks.push(tiny_resblock(cout, cfg.resblock_kernel_sizes[j] as usize, 400 + (i * 3 + j) as u64 * 10));
            }
        }
        let final_c = base / (1 << 3);

        let w = HiftWeights {
            f0_predictor,
            m_source_linear_w: filled(9, 7),
            m_source_linear_b: 0.0,
            conv_pre: tiny_convw(base, 80, 7, 8),
            ups: ups.try_into().unwrap_or_else(|_| unreachable!()),
            source_downs: source_downs.try_into().unwrap_or_else(|_| unreachable!()),
            source_resblocks: source_resblocks.try_into().unwrap_or_else(|_| unreachable!()),
            resblocks: resblocks.try_into().unwrap_or_else(|_| unreachable!()),
            conv_post: tiny_convw(bins, final_c, 7, 9),
        };

        let t_mel = 4usize;
        let mel = filled(80 * t_mel, 42);

        let out = forward_seeded(&mel, t_mel, &cfg, &w, 1234);
        assert_eq!(out.f0.len(), t_mel);
        assert!(out.f0.iter().all(|v| v.is_finite() && *v >= 0.0), "f0 must be finite and non-negative (abs())");
        let t_full = t_mel * cfg.nsf_upsample_scale() as usize;
        assert_eq!(out.waveform.len(), t_full, "waveform length must equal T_mel * upsample_scale");
        assert!(out.waveform.iter().all(|v| v.is_finite()), "waveform must be finite");
        assert!(out.waveform.iter().all(|&v| (-cfg.audio_limit..=cfg.audio_limit).contains(&v)), "waveform must respect audio_limit clamp");
        assert!(out.magnitude.iter().all(|v| v.is_finite() && *v > 0.0), "magnitude = exp(.) must be finite and positive");
        assert!(out.phase.iter().all(|v| v.is_finite()), "phase must be finite");
    }

    #[test]
    fn linear_interp_is_identity_at_equal_lengths() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y = linear_interp(&x, 4, 4);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn nsf_source_forward_is_pure_given_the_same_noise() {
        let cfg = tiny_cfg();
        let w = HiftWeights {
            f0_predictor: F0PredictorW { condnet: std::array::from_fn(|_| tiny_convw(1, 1, 1, 0)), classifier_w: vec![], classifier_b: 0.0 },
            m_source_linear_w: filled(9, 1),
            m_source_linear_b: 0.05,
            conv_pre: tiny_convw(1, 1, 1, 0),
            ups: std::array::from_fn(|_| tiny_convw(1, 1, 1, 0)),
            source_downs: std::array::from_fn(|_| tiny_convw(1, 1, 1, 0)),
            source_resblocks: std::array::from_fn(|_| tiny_resblock(1, 1, 0)),
            resblocks: std::array::from_fn(|_| tiny_resblock(1, 1, 0)),
            conv_post: tiny_convw(1, 1, 1, 0),
        };
        let f0 = vec![120.0f32, 0.0, 200.0];
        let t_full = f0.len() * cfg.nsf_upsample_scale() as usize;
        let noise = filled(t_full * cfg.harmonics() as usize, 99);
        let a = nsf_source_forward(&f0, &cfg, &w, &noise);
        let b = nsf_source_forward(&f0, &cfg, &w, &noise);
        assert_eq!(a, b, "same f0 + same noise must reproduce the same excitation");
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)), "excitation is a tanh() output, must be in [-1,1]");
    }
}
