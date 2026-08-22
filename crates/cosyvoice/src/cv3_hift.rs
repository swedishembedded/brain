// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalHiFTGenerator` (CosyVoice 3) forward: reuses the EXACT SAME
//! `ConvRNNF0Predictor` (well, its causal sibling) -> NSF -> conv-trunk ->
//! ISTFT topology `crate::hift` already implements for CosyVoice 2, but every
//! conv is causal: `conv_pre` is a RIGHT-looking causal conv (kernel
//! `conv_pre_look_right+1=5`, right-only padding, NOT CosyVoice 2's
//! symmetric `Conv1d(k=7,p=3)`), `ups[i]` are nearest-upsample-then-LEFT-
//! causal-`Conv1d` (`CausalConv1dUpsample`), NOT `ConvTranspose1d`, and every
//! `ResBlock`/`source_downs` conv is one-sided (left- or right-) causal
//! instead of symmetric.
//!
//! Reuses `crate::hift::nsf_source_forward_causal` - the SAME NSF math as
//! CosyVoice 2's `nsf_source_forward`, with exactly ONE real formula
//! difference (`SineGen2._f02sine`'s phase-upsample interpolation mode,
//! `"nearest"` here vs `"linear"` for CosyVoice 2 - a genuine, empirically
//! caught divergence, not a stylistic split; see that function's own doc for
//! how it was found) - and `audio::act::elu_ref`/`audio::snake`/
//! `audio::istft` the same way `crate::hift` does.
//!
//! ## The RNG story is DIFFERENT from, and simpler than, CosyVoice 2's
//! `SineGen2(causal=True)` in eval mode reads FIXED buffers (`rand_ini`,
//! `sine_waves`, `uv`) drawn ONCE at `__init__` time (plain tensor
//! attributes, never registered as `nn.Buffer`s, so never saved in
//! `hift.pt`), not redrawn per `inference()` call the way CosyVoice 2's
//! non-causal `SineGen2` redraws from the global RNG every time. This
//! port's faithful equivalent: draw the noise buffer ONCE when
//! [`Cv3HiftInstance`] is constructed, then reuse it for every `forward()`
//! call on that instance - see [`Cv3HiftInstance::new_seeded`].
//! `rand_ini` is provably inert here too (the SAME empirical argument
//! `crate::hift`'s module doc makes: the downsample interpolation inside
//! `SineGen2._f02sine` uses `mode="linear"` regardless of `causal`, so its
//! first sampled input index is never index 0, the one `rand_ini`
//! perturbs), so it needs no buffer at all.
//!
//! **`f0_predictor` precision**: the reference explicitly upcasts
//! `self.f0_predictor` to `float64` for the duration of `inference()`
//! ("f0_predictor precision is crucial for causal inference" - the
//! reference's own comment). This port runs the f0 predictor in `f32`
//! throughout; `crates/cosyvoice/tests/cv3_hift_parity.rs` reports the
//! actual resulting numeric delta against the real f64 reference rather than
//! assuming it is negligible.
//!
//! **`CausalHiFTGenerator.inference()` has NO `cache_source` parameter** - a
//! real signature difference from CosyVoice 2's `HiFTGenerator.inference`
//! (verified against the real reference source, not by trial and error) -
//! [`forward`]/[`forward_seeded`] below reflect that: no cache-source
//! argument exists here at all.

use audio::act::elu_ref;
use audio::conv::{conv1d_ref, Conv1d};
use audio::istft::{istft as istft_overlap_add, stft as stft_forward, StftConfig};
use audio::snake::{snake1d_ref, Snake1d};
use data::rng::Rng;

use crate::cv3_hift_config::{Cv3HiftConfig, RESBLOCK_DILATIONS};
use crate::hift::nsf_source_forward_causal;
use crate::hift_import::{ConvW, F0PredictorW, HiftWeights, ResBlockW};

// ---------------------------------------------------------------------------
// causal conv helpers
// ---------------------------------------------------------------------------

fn add_bias_ncl(y: &mut [f32], bias: &[f32], l: usize) {
    for (ci, b) in bias.iter().enumerate() {
        for li in 0..l {
            y[ci * l + li] += *b;
        }
    }
}

/// One fully-general conv1d call: `pad` shifts the read origin (a pure
/// index shift, per `audio::conv::conv1d_ref`'s own out-of-range-is-zero
/// convention - the same trick `crate::flow`'s conv helpers exploit for
/// one-sided padding), `lo` is the caller-computed output length.
#[allow(clippy::too_many_arguments)]
fn conv1d_generic(x: &[f32], w: &ConvW, cin: usize, cout: usize, l: usize, lo: usize, k: usize, pad: usize, dilation: usize, stride: usize) -> Vec<f32> {
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
    y
}

/// `CausalConv1d(causal_type='left')`: left-pad `dilation*(k-1)`, output
/// length == input length.
fn causal_left(x: &[f32], w: &ConvW, cin: usize, cout: usize, l: usize, k: usize, dilation: usize) -> Vec<f32> {
    conv1d_generic(x, w, cin, cout, l, l, k, dilation * (k - 1), dilation, 1)
}

/// `CausalConv1d(causal_type='right')`: right-only padding (via the
/// `pad=0, lo=l` free-right-zero-fill trick), output length == input length.
fn causal_right(x: &[f32], w: &ConvW, cin: usize, cout: usize, l: usize, k: usize) -> Vec<f32> {
    conv1d_generic(x, w, cin, cout, l, l, k, 0, 1, 1)
}

/// `CausalConv1dDownSample`: left-pad `stride-1`, kernel `2*stride`,
/// strided - `lo` is the caller-supplied (already-known) trunk length this
/// stage must land on, matching `crate::hift::decode`'s own
/// assert-the-length-lines-up convention for the non-causal case.
fn causal_downsample(x: &[f32], w: &ConvW, cin: usize, cout: usize, l: usize, lo: usize, stride: usize) -> Vec<f32> {
    if stride == 1 {
        return causal_left(x, w, cin, cout, l, 1, 1);
    }
    conv1d_generic(x, w, cin, cout, l, lo, stride * 2, stride - 1, 1, stride)
}

fn nearest_upsample_cl(x: &[f32], c: usize, l: usize, scale: usize) -> Vec<f32> {
    let lo = l * scale;
    let mut y = vec![0.0f32; c * lo];
    for ci in 0..c {
        for i in 0..lo {
            y[ci * lo + i] = x[ci * l + i / scale];
        }
    }
    y
}

/// `CausalConv1dUpsample`: nearest-upsample by `scale`, then a LEFT-causal
/// `Conv1d(k, stride=1)` - output length `l*scale`.
fn causal_upsample(x: &[f32], w: &ConvW, cin: usize, cout: usize, l: usize, k: usize, scale: usize) -> Vec<f32> {
    let up = nearest_upsample_cl(x, cin, l, scale);
    causal_left(&up, w, cin, cout, l * scale, k, 1)
}

const SNAKE_EPS: f32 = 1e-9;

fn snake(x: &[f32], alpha: &[f32], c: usize, l: usize) -> Vec<f32> {
    let sc = Snake1d { rows: 1, c: c as u32, inner: l as u32, eps: SNAKE_EPS };
    snake1d_ref(&sc, x, alpha)
}

fn leaky_relu(x: &[f32], slope: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { v * slope }).collect()
}

/// Causal `ResBlock`: 3 sequential `(Snake -> CausalConv1d[dilation, left] ->
/// Snake -> CausalConv1d[dilation=1, left]) + x` branches - same structure as
/// `crate::hift::resblock_forward`, but every conv is one-sided causal
/// instead of symmetric.
fn resblock_forward_causal(x0: &[f32], rb: &ResBlockW, c: usize, l: usize, k: usize) -> Vec<f32> {
    let mut x = x0.to_vec();
    #[allow(clippy::needless_range_loop)]
    for idx in 0..3 {
        let d = RESBLOCK_DILATIONS[idx] as usize;
        let xt = snake(&x, &rb.alpha1[idx], c, l);
        let xt = causal_left(&xt, &rb.convs1[idx], c, c, l, k, d);
        let xt = snake(&xt, &rb.alpha2[idx], c, l);
        let xt = causal_left(&xt, &rb.convs2[idx], c, c, l, k, 1);
        for (xi, xti) in x.iter_mut().zip(&xt) {
            *xi += *xti;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// CausalConvRNNF0Predictor (right-causal first conv, then 4 left-causal)
// ---------------------------------------------------------------------------

/// `mel [in_channels, T] -> condnet (1x RIGHT-causal k=4 + 4x LEFT-causal
/// k=3, weight-normed Conv1d + ELU) -> classifier (Linear(cond_channels,1))
/// -> abs()` -> `f0 [T]` (Hz). Runs in `f32` - see this module's doc on the
/// reference's `float64` precision requirement.
pub fn f0_predictor_forward(w: &F0PredictorW, mel: &[f32], in_channels: usize, cond_channels: usize, t: usize) -> Vec<f32> {
    let mut cur = causal_right(mel, &w.condnet[0], in_channels, cond_channels, t, 4);
    cur = elu_ref(&cur, 1.0);
    for cw in &w.condnet[1..] {
        cur = causal_left(&cur, cw, cond_channels, cond_channels, t, 3, 1);
        cur = elu_ref(&cur, 1.0);
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
// _stft / _istft (center=False - the SAME excitation-STFT convention
// `crate::hift` uses; CausalHiFTGenerator's `_stft`/`_istft` are inherited
// UNCHANGED from `HiFTGenerator`, not overridden)
// ---------------------------------------------------------------------------

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
// decode: causal conv trunk + source fusion + ISTFT head
// ---------------------------------------------------------------------------

pub struct DecodeOutput {
    pub magnitude: Vec<f32>,
    pub phase: Vec<f32>,
    pub waveform: Vec<f32>,
}

/// `CausalHiFTGenerator.decode` (`finalize=True` only - the streaming/
/// chunked path is a documented, not-yet-implemented gap matching
/// `crate::flow`'s own streaming gap).
pub fn decode(mel: &[f32], t_mel: usize, excitation: &[f32], cfg: &Cv3HiftConfig, w: &HiftWeights) -> DecodeOutput {
    let base = cfg.base_channels as usize;
    let bins = cfg.stft_bins() as usize;
    let stft_cfg = StftConfig { n_fft: cfg.n_fft as usize, hop: cfg.hop_len as usize, win: cfg.n_fft as usize };

    let (s_re, s_im, n_frames_s) = stft_center(excitation, &stft_cfg);
    let mut s_stft = vec![0f32; 2 * bins * n_frames_s];
    for b in 0..bins {
        for f in 0..n_frames_s {
            s_stft[b * n_frames_s + f] = s_re[f * bins + b];
            s_stft[(bins + b) * n_frames_s + f] = s_im[f * bins + b];
        }
    }

    let mut x = causal_right(mel, &w.conv_pre, cfg.in_channels as usize, base, t_mel, cfg.conv_pre_kernel() as usize);
    let mut l = t_mel;
    let mut c = base;
    let down_strides = cfg.source_downsample_strides();

    #[allow(clippy::needless_range_loop)]
    for i in 0..3usize {
        let u = cfg.upsample_rates[i] as usize;
        let k = cfg.upsample_kernel_sizes[i] as usize;
        let cout = base / (1 << (i + 1));

        let xa = leaky_relu(&x, cfg.lrelu_slope);
        let mut xu = causal_upsample(&xa, &w.ups[i], c, cout, l, k, u);
        let mut lu = l * u;
        c = cout;

        if i == 2 {
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
        let si_raw = causal_downsample(&s_stft, &w.source_downs[i], 2 * bins, c, n_frames_s, l, stride_i);
        let src_k = cfg.source_resblock_kernel_sizes[i] as usize;
        let si = resblock_forward_causal(&si_raw, &w.source_resblocks[i], c, l, src_k);
        for (xi, si_v) in xu.iter_mut().zip(&si) {
            *xi += *si_v;
        }
        x = xu;

        let mut acc = vec![0f32; x.len()];
        for j in 0..3usize {
            let k = cfg.resblock_kernel_sizes[j] as usize;
            let r = resblock_forward_causal(&x, &w.resblocks[i * 3 + j], c, l, k);
            for (a, rv) in acc.iter_mut().zip(&r) {
                *a += *rv;
            }
        }
        for v in acc.iter_mut() {
            *v /= 3.0;
        }
        x = acc;
    }

    let xa = leaky_relu(&x, 0.01);
    let post = causal_left(&xa, &w.conv_post, c, 2 * bins, l, 7, 1);

    let mut magnitude = vec![0f32; bins * l];
    let mut phase = vec![0f32; bins * l];
    for b in 0..bins {
        for t in 0..l {
            magnitude[b * l + t] = post[b * l + t].exp();
            phase[b * l + t] = post[(bins + b) * l + t].sin();
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

/// Full forward given an explicit NSF noise draw - the entry point
/// `crates/cosyvoice/tests/cv3_hift_parity.rs` uses with the captured
/// per-instance noise buffer (see this module's doc). [`forward_seeded`] is
/// the production entry point.
pub fn forward(mel: &[f32], t_mel: usize, cfg: &Cv3HiftConfig, w: &HiftWeights, randn_noise: &[f32]) -> HiftOutput {
    assert_eq!(mel.len(), cfg.in_channels as usize * t_mel, "forward: mel length != in_channels*T_mel");
    let f0 = f0_predictor_forward(&w.f0_predictor, mel, cfg.in_channels as usize, cfg.f0_cond_channels as usize, t_mel);
    let nsf_cfg = crate::hift_config::HiftConfig {
        in_channels: cfg.in_channels,
        base_channels: cfg.base_channels,
        nb_harmonics: cfg.nb_harmonics,
        sampling_rate: cfg.sampling_rate,
        nsf_alpha: cfg.nsf_alpha,
        nsf_sigma: cfg.nsf_sigma,
        nsf_voiced_threshold: cfg.nsf_voiced_threshold,
        upsample_rates: cfg.upsample_rates,
        upsample_kernel_sizes: cfg.upsample_kernel_sizes,
        n_fft: cfg.n_fft,
        hop_len: cfg.hop_len,
        resblock_kernel_sizes: cfg.resblock_kernel_sizes,
        source_resblock_kernel_sizes: cfg.source_resblock_kernel_sizes,
        lrelu_slope: cfg.lrelu_slope,
        audio_limit: cfg.audio_limit,
        f0_cond_channels: cfg.f0_cond_channels,
    };
    let excitation = nsf_source_forward_causal(&f0, &nsf_cfg, w, randn_noise);
    let out = decode(mel, t_mel, &excitation, cfg, w);
    HiftOutput { f0, magnitude: out.magnitude, phase: out.phase, waveform: out.waveform }
}

/// A constructed `CausalHiFTGenerator` "instance": the imported weights plus
/// the ONE fixed NSF noise buffer this port draws at construction time,
/// reused for every [`Self::forward`] call - the faithful equivalent of the
/// reference's `SineGen2(causal=True)` fixed `sine_waves` attribute (see
/// this module's doc). Two `forward()` calls on the SAME instance are
/// therefore bit-exact without reseeding between them, mirroring the
/// golden's own self-validation.
pub struct Cv3HiftInstance {
    pub weights: HiftWeights,
    noise: Vec<f32>,
}

impl Cv3HiftInstance {
    /// `max_t_mel` bounds how large a single `forward()` call's mel input may
    /// be (the noise buffer is drawn once, sized for the largest call this
    /// instance will ever serve - mirroring the reference's own
    /// `torch.rand(1, 300*24000, 9)` fixed-size buffer, just sized to the
    /// caller's real budget instead of a hardcoded 300 s).
    pub fn new_seeded(weights: HiftWeights, cfg: &Cv3HiftConfig, max_t_mel: usize, seed: u64) -> Cv3HiftInstance {
        let n = max_t_mel * cfg.nsf_upsample_scale() as usize * cfg.harmonics() as usize;
        let mut rng = Rng::new(seed);
        let noise: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
        Cv3HiftInstance { weights, noise }
    }

    /// Production forward: draws no new randomness, reuses this instance's
    /// fixed noise buffer - NOT bit-exact with the reference's own specific
    /// draw (a fresh, unrelated RNG stream, same honest gap
    /// `crate::hift::forward_seeded`'s doc already names), but faithfully
    /// reproduces the reference's real "fixed across calls" behavior.
    pub fn forward(&self, mel: &[f32], t_mel: usize, cfg: &Cv3HiftConfig) -> HiftOutput {
        let need = t_mel * cfg.nsf_upsample_scale() as usize * cfg.harmonics() as usize;
        assert!(self.noise.len() >= need, "Cv3HiftInstance::forward: t_mel exceeds this instance's max_t_mel budget");
        forward(mel, t_mel, cfg, &self.weights, &self.noise[..need])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hift_import::ResBlockW;

    fn tiny_cfg() -> Cv3HiftConfig {
        Cv3HiftConfig::cosyvoice3()
    }

    fn filled(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
    }

    fn filled_fanin(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
        let scale = 1.0 / (fan_in.max(1) as f32).sqrt();
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (rng.next_f32() - 0.5) * scale).collect()
    }

    fn tiny_convw(cout: usize, cin: usize, k: usize, seed: u64) -> ConvW {
        ConvW { weight: filled_fanin(cout * cin * k, cin * k, seed), bias: filled(cout, seed + 1) }
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

    #[test]
    fn tiny_forward_produces_a_finite_waveform_at_the_real_shapes() {
        let cfg = tiny_cfg();
        let base = cfg.base_channels as usize;
        let bins = cfg.source_stft_channels() as usize;

        let f0_predictor = F0PredictorW {
            condnet: [
                tiny_convw(base, 80, 4, 1),
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
        #[allow(clippy::needless_range_loop)]
        for i in 0..3usize {
            let cin = base / (1 << i);
            let cout = base / (1 << (i + 1));
            let k = cfg.upsample_kernel_sizes[i] as usize;
            ups.push(tiny_convw(cout, cin, k, 100 + i as u64)); // plain Conv1d layout: [Cout,Cin,K]
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
            conv_pre: tiny_convw(base, 80, cfg.conv_pre_kernel() as usize, 8),
            ups: ups.try_into().unwrap_or_else(|_| unreachable!()),
            source_downs: source_downs.try_into().unwrap_or_else(|_| unreachable!()),
            source_resblocks: source_resblocks.try_into().unwrap_or_else(|_| unreachable!()),
            resblocks: resblocks.try_into().unwrap_or_else(|_| unreachable!()),
            conv_post: tiny_convw(bins, final_c, 7, 9),
        };

        let t_mel = 4usize;
        let mel = filled(80 * t_mel, 42);
        let inst = Cv3HiftInstance::new_seeded(w, &cfg, t_mel, 1234);
        let out = inst.forward(&mel, t_mel, &cfg);

        assert_eq!(out.f0.len(), t_mel);
        assert!(out.f0.iter().all(|v| v.is_finite() && *v >= 0.0));
        let t_full = t_mel * cfg.nsf_upsample_scale() as usize;
        assert_eq!(out.waveform.len(), t_full);
        assert!(out.waveform.iter().all(|v| v.is_finite()));
        assert!(out.waveform.iter().all(|&v| (-cfg.audio_limit..=cfg.audio_limit).contains(&v)));
        assert!(out.magnitude.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(out.phase.iter().all(|v| v.is_finite()));

        let out2 = inst.forward(&mel, t_mel, &cfg);
        assert_eq!(out.waveform, out2.waveform, "two forward() calls on the SAME instance must be bit-exact (fixed noise buffer)");
    }
}
