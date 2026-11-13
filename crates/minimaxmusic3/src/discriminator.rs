// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The STFT-magnitude discriminator of the vocoder's adversarial fine-tune: a PatchGAN over `|STFT(waveform)|`, the
//! LSGAN adversarial losses and an L1 feature-matching loss, run at a list of STFT resolutions ([`MrdConfig`], one
//! discriminator per `(n_fft, hop)`; a single resolution is a list of one).
//!
//! Architecture, per resolution: `|STFT(waveform)| -> Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Conv2d -> patch
//! logits` (one real/fake score per time-frequency patch, not a single scalar: the BigVGAN/HiFi-GAN discriminator
//! shape). Every conv reuses the `conv2d`/`conv2d_dx`/`conv2d_dw` device kernels, `leaky_relu`/`leaky_relu_bwd`, and
//! the `add_chan_inplace`/`bias_grad_ncl` bias pair (layout-generic over `[rows, C, inner]`; NCHW's `inner = H*W`
//! fits it unchanged). The feature-matching gradient joins the adversarial one at the first layer's post-activation
//! (`a1`) with one `add2`.
//!
//! The STFT is a direct DFT-matrix formulation (`O(n_fft^2)` per frame, not an FFT butterfly), deliberately: a
//! windowed product with a FIXED cos/sin basis (evaluated in the loops, not stored) has a backward that is the same
//! product against the transposed basis ([`stft_mag_bwd`]), where an FFT's butterfly network would be additional
//! backward work for no benefit at the frame sizes a short training clip needs.
//!
//! Entry points: [`mrd_discriminator_step`] (discriminator loss and weight gradients on real vs generated audio) and
//! [`generator_adversarial`] (the generator's adversarial and feature-matching losses and their gradient on the
//! generated waveform, which the vocoder's backward continues from; [`crate::adversarial`] composes both with the
//! vocoder trainer). MiniMax publishes no vocoder training code, so the resolutions and weights this crate defaults
//! to are the published BigVGAN/HiFi-GAN recipe's ([`crate::adversarial::GanConfig`]), not MiniMax's.

use gpu_core::{DeviceBuffer, Gpu, Step};
use std::f32::consts::PI;

/// `PIPELINES` this module's forward/backward needs: conv2d's family, the bias pair, leaky ReLU and `add2`.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("bias_grad_ncl", kernels::BIAS_GRAD_NCL),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("add2", kernels::ADD2),
];
const CONV2D: usize = 0;
const CONV2D_DX: usize = 1;
const CONV2D_DW: usize = 2;
const BIAS_ADD: usize = 3;
const BIAS_GRAD: usize = 4;
const LEAKY_RELU: usize = 5;
const LEAKY_RELU_BWD: usize = 6;
const ADD2: usize = 7;
const SLOPE: f32 = 0.1;

/// One frame's magnitude STFT config: `n_fft`-point direct DFT, `hop`-size
/// stride, a Hann analysis window.
#[derive(Clone, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop: usize,
}

impl StftConfig {
    fn window(&self) -> Vec<f32> {
        (0..self.n_fft).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / self.n_fft as f32).cos()).collect()
    }
    fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
    pub fn num_frames(&self, x_len: usize) -> usize {
        if x_len < self.n_fft {
            0
        } else {
            (x_len - self.n_fft) / self.hop + 1
        }
    }
}

/// The magnitude spectrogram plus the real/imag parts backward needs.
/// `mag`/`re`/`im` are `[freq_bins, num_frames]` row-major (NCHW with
/// `N=1, C=1, H=freq_bins, W=num_frames`).
pub struct StftOut {
    pub mag: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    win: Vec<f32>,
    cfg: StftConfig,
    x_len: usize,
}

const MAG_EPS: f32 = 1e-6;

/// Direct-DFT magnitude STFT forward, `O(freq_bins * num_frames * n_fft)`.
pub fn stft_mag_fwd(x: &[f32], cfg: &StftConfig) -> StftOut {
    let win = cfg.window();
    let (f_bins, frames) = (cfg.freq_bins(), cfg.num_frames(x.len()));
    let mut re = vec![0.0f32; f_bins * frames];
    let mut im = vec![0.0f32; f_bins * frames];
    let mut mag = vec![0.0f32; f_bins * frames];
    for t in 0..frames {
        let base = t * cfg.hop;
        for f in 0..f_bins {
            let (mut re_ft, mut im_ft) = (0.0f32, 0.0f32);
            let w0 = 2.0 * PI * f as f32 / cfg.n_fft as f32;
            for n in 0..cfg.n_fft {
                let v = win[n] * x[base + n];
                re_ft += v * (w0 * n as f32).cos();
                im_ft -= v * (w0 * n as f32).sin();
            }
            re[f * frames + t] = re_ft;
            im[f * frames + t] = im_ft;
            mag[f * frames + t] = (re_ft * re_ft + im_ft * im_ft + MAG_EPS).sqrt();
        }
    }
    StftOut { mag, re, im, win, cfg: cfg.clone(), x_len: x.len() }
}

/// `dx` from `dmag` (gradient w.r.t. the magnitude spectrogram).
pub fn stft_mag_bwd(out: &StftOut, dmag: &[f32]) -> Vec<f32> {
    let (f_bins, frames) = (out.cfg.freq_bins(), out.cfg.num_frames(out.x_len));
    let mut dx = vec![0.0f32; out.x_len];
    for t in 0..frames {
        let base = t * out.cfg.hop;
        for f in 0..f_bins {
            let idx = f * frames + t;
            let (re_ft, im_ft, mag_ft) = (out.re[idx], out.im[idx], out.mag[idx]);
            let d = dmag[idx] / mag_ft; // d(mag)/d(re)=re/mag, d(mag)/d(im)=im/mag.
            let w0 = 2.0 * PI * f as f32 / out.cfg.n_fft as f32;
            for n in 0..out.cfg.n_fft {
                let c = (w0 * n as f32).cos();
                let s = (w0 * n as f32).sin();
                dx[base + n] += out.win[n] * d * (re_ft * c - im_ft * s);
            }
        }
    }
    dx
}

#[derive(Clone)]
pub struct Conv2dW {
    pub weight: Vec<f32>, // [cout, cin, k, k]
    pub bias: Vec<f32>,   // [cout]
}

#[derive(Clone)]
pub struct DiscWeights {
    pub conv1: Conv2dW, // 1 -> c1
    pub conv2: Conv2dW, // c1 -> c2
    pub conv_out: Conv2dW, // c2 -> 1
}

#[derive(Clone, Debug)]
pub struct DiscConfig {
    pub c1: usize,
    pub c2: usize,
    pub k: usize,
}

/// Random discriminator weights, deterministic from `seed`.
pub fn random_weights(cfg: &DiscConfig, seed: u64) -> DiscWeights {
    let mut r = data::rng::Lcg::new(seed);
    let conv = |cout: usize, cin: usize, k: usize, r: &mut data::rng::Lcg| Conv2dW { weight: r.vec_scaled(cout * cin * k * k, 0.3), bias: r.vec_scaled(cout, 0.05) };
    DiscWeights { conv1: conv(cfg.c1, 1, cfg.k, &mut r), conv2: conv(cfg.c2, cfg.c1, cfg.k, &mut r), conv_out: conv(1, cfg.c2, cfg.k, &mut r) }
}

fn conv2d_out(h: usize, w: usize, k: usize, pad: usize) -> (usize, usize) {
    (h + 2 * pad - k + 1, w + 2 * pad - k + 1)
}

struct Cache {
    mag_in: DeviceBuffer,
    h1: DeviceBuffer, // pre-activation, conv1 out
    a1: DeviceBuffer, // post-LeakyReLU
    h2: DeviceBuffer,
    a2: DeviceBuffer,
    dims: [(usize, usize, usize); 4], // (c,h,w) at input, after conv1, after conv2, after conv_out
}

/// Device forward: `mag [f_bins, frames]` -> patch logits. Returns the
/// logits (host, for the LSGAN loss) and the device-resident cache
/// `backward` needs.
fn forward(gpu: &Gpu, w: &DiscWeights, cfg: &DiscConfig, mag: &[f32], f_bins: usize, frames: usize) -> (Vec<f32>, Cache) {
    let mag_in = gpu.storage_init("mag", mag);
    let mut steps: Vec<Step> = Vec::new();

    let (h1o, w1o) = conv2d_out(f_bins, frames, cfg.k, 1);
    let h1 = conv2d_bias(gpu, &mut steps, &w.conv1, 1, f_bins, frames, cfg.c1, cfg.k, 1, h1o, w1o, &mag_in);
    let a1 = leaky_relu(gpu, &mut steps, &h1, cfg.c1 * h1o * w1o);

    let (h2o, w2o) = conv2d_out(h1o, w1o, cfg.k, 1);
    let h2 = conv2d_bias(gpu, &mut steps, &w.conv2, cfg.c1, h1o, w1o, cfg.c2, cfg.k, 1, h2o, w2o, &a1);
    let a2 = leaky_relu(gpu, &mut steps, &h2, cfg.c2 * h2o * w2o);

    let (h3o, w3o) = conv2d_out(h2o, w2o, cfg.k, 1);
    let logits = conv2d_bias(gpu, &mut steps, &w.conv_out, cfg.c2, h2o, w2o, 1, cfg.k, 1, h3o, w3o, &a2);

    gpu.submit(&[], &steps);
    let got = gpu.read(&logits, h3o * w3o);
    (got, Cache { mag_in, h1, a1, h2, a2, dims: [(1, f_bins, frames), (cfg.c1, h1o, w1o), (cfg.c2, h2o, w2o), (1, h3o, w3o)] })
}

/// The first layer's post-activation (`a1`), the feature map the feature-matching loss compares.
fn features(gpu: &Gpu, cache: &Cache) -> Vec<f32> {
    let (c, h, w) = cache.dims[1];
    gpu.read(&cache.a1, c * h * w)
}

/// `dW`/`db` for each of the 3 convs and the gradient on the input magnitudes (`d_mag`, which [`stft_mag_bwd`]
/// carries on to the waveform).
struct Grads {
    d_conv1: Conv2dW,
    d_conv2: Conv2dW,
    d_conv_out: Conv2dW,
    d_mag: Vec<f32>,
}

/// Backward from `d_logits`; `d_feat`, when given, is a gradient on the first layer's post-activation `a1` (the
/// feature-matching loss) added to the one flowing back from the logits before the first layer's backward.
fn backward(gpu: &Gpu, w: &DiscWeights, cfg: &DiscConfig, cache: &Cache, d_logits: &[f32], d_feat: Option<&[f32]>) -> Grads {
    let [(_, f_bins, frames), (_, h1o, w1o), (_, h2o, w2o), (_, h3o, w3o)] = cache.dims;
    let mut steps: Vec<Step> = Vec::new();
    let d_logits_b = gpu.storage_init("d_logits", d_logits);

    let dw_out = gpu.storage(w.conv_out.weight.len() as u64);
    let db_out = gpu.storage(w.conv_out.bias.len() as u64);
    let d_a2 = conv2d_bias_bwd(gpu, &mut steps, &w.conv_out, cfg.c2, h2o, w2o, 1, cfg.k, 1, h3o, w3o, &cache.a2, &d_logits_b, &dw_out, &db_out);

    let d_h2 = leaky_relu_bwd(gpu, &mut steps, &cache.h2, &d_a2, cfg.c2 * h2o * w2o);
    let dw2 = gpu.storage(w.conv2.weight.len() as u64);
    let db2 = gpu.storage(w.conv2.bias.len() as u64);
    let d_a1_logits = conv2d_bias_bwd(gpu, &mut steps, &w.conv2, cfg.c1, h1o, w1o, cfg.c2, cfg.k, 1, h2o, w2o, &cache.a1, &d_h2, &dw2, &db2);
    let a1_len = cfg.c1 * h1o * w1o;
    let d_a1 = match d_feat {
        Some(feat) => {
            assert_eq!(feat.len(), a1_len, "feature gradient must match the first layer's activation");
            let sum = gpu.storage(a1_len as u64);
            steps.push(gpu.step(ADD2, &[&d_a1_logits, &gpu.storage_init("d_feat", feat), &sum], &[a1_len as u32], a1_len as u32));
            sum
        }
        None => d_a1_logits,
    };

    let d_h1 = leaky_relu_bwd(gpu, &mut steps, &cache.h1, &d_a1, cfg.c1 * h1o * w1o);
    let dw1 = gpu.storage(w.conv1.weight.len() as u64);
    let db1 = gpu.storage(w.conv1.bias.len() as u64);
    let d_mag = conv2d_bias_bwd(gpu, &mut steps, &w.conv1, 1, f_bins, frames, cfg.c1, cfg.k, 1, h1o, w1o, &cache.mag_in, &d_h1, &dw1, &db1);

    gpu.submit(&[], &steps);
    Grads {
        d_conv1: Conv2dW { weight: gpu.read(&dw1, w.conv1.weight.len()), bias: gpu.read(&db1, w.conv1.bias.len()) },
        d_conv2: Conv2dW { weight: gpu.read(&dw2, w.conv2.weight.len()), bias: gpu.read(&db2, w.conv2.bias.len()) },
        d_conv_out: Conv2dW { weight: gpu.read(&dw_out, w.conv_out.weight.len()), bias: gpu.read(&db_out, w.conv_out.bias.len()) },
        d_mag: gpu.read(&d_mag, f_bins * frames),
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d_bias(gpu: &Gpu, steps: &mut Vec<Step>, w: &Conv2dW, cin: usize, h: usize, wid: usize, cout: usize, k: usize, pad: usize, ho: usize, wo: usize, x: &DeviceBuffer) -> DeviceBuffer {
    let params = [1u32, cin as u32, h as u32, wid as u32, cout as u32, k as u32, 1, pad as u32, ho as u32, wo as u32];
    let wb = gpu.storage_init("dw", &w.weight);
    let y = gpu.storage((cout * ho * wo) as u64);
    steps.push(gpu.step(CONV2D, &[x, &wb, &y], &params, (cout * ho * wo) as u32));
    let bb = gpu.storage_init("db", &w.bias);
    steps.push(gpu.step(BIAS_ADD, &[&y, &bb], &[(cout * ho * wo) as u32, cout as u32, (ho * wo) as u32], (cout * ho * wo) as u32));
    y
}

#[allow(clippy::too_many_arguments)]
fn conv2d_bias_bwd(gpu: &Gpu, steps: &mut Vec<Step>, w: &Conv2dW, cin: usize, h: usize, wid: usize, cout: usize, k: usize, pad: usize, ho: usize, wo: usize, x: &DeviceBuffer, dy: &DeviceBuffer, dw: &DeviceBuffer, db: &DeviceBuffer) -> DeviceBuffer {
    let params = [1u32, cin as u32, h as u32, wid as u32, cout as u32, k as u32, 1, pad as u32, ho as u32, wo as u32];
    let dx = gpu.storage((cin * h * wid) as u64);
    steps.push(gpu.step(CONV2D_DX, &[dy, &gpu.storage_init("w", &w.weight), &dx], &params, (cin * h * wid) as u32));
    steps.push(gpu.step(CONV2D_DW, &[dy, x, dw], &params, (cout * cin * k * k) as u32));
    steps.push(gpu.step(BIAS_GRAD, &[dy, db], &[1, cout as u32, (ho * wo) as u32], cout as u32));
    dx
}

fn leaky_relu(gpu: &Gpu, steps: &mut Vec<Step>, x: &DeviceBuffer, n: usize) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    steps.push(gpu.step(LEAKY_RELU, &[x, &y], &[n as u32, SLOPE.to_bits()], n as u32));
    y
}
fn leaky_relu_bwd(gpu: &Gpu, steps: &mut Vec<Step>, x: &DeviceBuffer, dy: &DeviceBuffer, n: usize) -> DeviceBuffer {
    let dx = gpu.storage(n as u64);
    steps.push(gpu.step(LEAKY_RELU_BWD, &[x, dy, &dx], &[n as u32, SLOPE.to_bits()], n as u32));
    dx
}

/// LSGAN discriminator loss `mean((D(real)-1)^2) + mean(D(fake)^2)`, and
/// its gradient w.r.t. each logit map (`d_real`, `d_fake`).
pub fn lsgan_d_loss(real_logits: &[f32], fake_logits: &[f32]) -> (f32, Vec<f32>, Vec<f32>) {
    let n_r = real_logits.len() as f32;
    let n_f = fake_logits.len() as f32;
    let loss_real: f32 = real_logits.iter().map(|&l| (l - 1.0).powi(2)).sum::<f32>() / n_r;
    let loss_fake: f32 = fake_logits.iter().map(|&l| l.powi(2)).sum::<f32>() / n_f;
    let d_real: Vec<f32> = real_logits.iter().map(|&l| 2.0 * (l - 1.0) / n_r).collect();
    let d_fake: Vec<f32> = fake_logits.iter().map(|&l| 2.0 * l / n_f).collect();
    (loss_real + loss_fake, d_real, d_fake)
}

/// LSGAN generator loss `mean((D(fake)-1)^2)` and its gradient w.r.t. the
/// fake logits.
pub fn lsgan_g_loss(fake_logits: &[f32]) -> (f32, Vec<f32>) {
    let n = fake_logits.len() as f32;
    let loss = fake_logits.iter().map(|&l| (l - 1.0).powi(2)).sum::<f32>() / n;
    let grad = fake_logits.iter().map(|&l| 2.0 * (l - 1.0) / n).collect();
    (loss, grad)
}

/// Feature-matching loss `mean(|feat_fake - feat_real|)` (D's first-layer
/// post-activation, `a1`) and its gradient w.r.t. `feat_fake`.
pub fn feature_matching_loss(feat_real: &[f32], feat_fake: &[f32]) -> (f32, Vec<f32>) {
    let n = feat_real.len() as f32;
    let loss = feat_real.iter().zip(feat_fake).map(|(r, f)| (f - r).abs()).sum::<f32>() / n;
    let grad = feat_real.iter().zip(feat_fake).map(|(r, f)| (f - r).signum() / n).collect();
    (loss, grad)
}

/// One discriminator update's result: the LSGAN D loss and the gradient of each of its three convolutions.
pub struct DStep {
    pub loss: f32,
    pub d_conv1: Conv2dW,
    pub d_conv2: Conv2dW,
    pub d_conv_out: Conv2dW,
}

pub fn discriminator_step(gpu: &Gpu, w: &DiscWeights, disc_cfg: &DiscConfig, stft_cfg: &StftConfig, real: &[f32], fake: &[f32]) -> DStep {
    let real_stft = stft_mag_fwd(real, stft_cfg);
    let fake_stft = stft_mag_fwd(fake, stft_cfg);
    let (f_bins, frames) = (stft_cfg.freq_bins(), stft_cfg.num_frames(real.len()));
    assert_eq!(frames, stft_cfg.num_frames(fake.len()), "real/fake must have the same length");

    let (real_logits, real_cache) = forward(gpu, w, disc_cfg, &real_stft.mag, f_bins, frames);
    let (fake_logits, fake_cache) = forward(gpu, w, disc_cfg, &fake_stft.mag, f_bins, frames);
    let (loss, d_real, d_fake) = lsgan_d_loss(&real_logits, &fake_logits);

    let gr = backward(gpu, w, disc_cfg, &real_cache, &d_real, None);
    let gf = backward(gpu, w, disc_cfg, &fake_cache, &d_fake, None);
    let sum = |a: &Conv2dW, b: &Conv2dW| Conv2dW {
        weight: a.weight.iter().zip(&b.weight).map(|(x, y)| x + y).collect(),
        bias: a.bias.iter().zip(&b.bias).map(|(x, y)| x + y).collect(),
    };
    DStep { loss, d_conv1: sum(&gr.d_conv1, &gf.d_conv1), d_conv2: sum(&gr.d_conv2, &gf.d_conv2), d_conv_out: sum(&gr.d_conv_out, &gf.d_conv_out) }
}

impl DiscWeights {
    /// Every tensor in the fixed order `conv1.weight, conv1.bias, conv2.weight, conv2.bias, conv_out.weight,
    /// conv_out.bias`, for an optimiser that updates them in place.
    pub fn tensors_mut(&mut self) -> [&mut [f32]; 6] {
        [&mut self.conv1.weight, &mut self.conv1.bias, &mut self.conv2.weight, &mut self.conv2.bias, &mut self.conv_out.weight, &mut self.conv_out.bias]
    }
}

impl DStep {
    /// The gradients in [`DiscWeights::tensors_mut`]'s order.
    pub fn tensors(&self) -> [&[f32]; 6] {
        [&self.d_conv1.weight, &self.d_conv1.bias, &self.d_conv2.weight, &self.d_conv2.bias, &self.d_conv_out.weight, &self.d_conv_out.bias]
    }
}

/// A multi-resolution discriminator: one [`DiscWeights`] per STFT resolution, all of the architecture `disc`.
#[derive(Clone, Debug)]
pub struct MrdConfig {
    pub resolutions: Vec<StftConfig>,
    pub disc: DiscConfig,
}

/// One discriminator's weights per resolution of `cfg`, deterministic from `seed` (resolution `i` draws from
/// `seed + i`, so the resolutions do not start identical).
pub fn random_mrd_weights(cfg: &MrdConfig, seed: u64) -> Vec<DiscWeights> {
    (0..cfg.resolutions.len()).map(|i| random_weights(&cfg.disc, seed.wrapping_add(i as u64))).collect()
}

/// [`discriminator_step`] at every resolution: the summed loss and each resolution's weight gradients.
pub struct MrdDStep {
    pub loss: f32,
    pub steps: Vec<DStep>,
}

fn check_resolutions(cfg: &MrdConfig, ws: &[DiscWeights], len: usize) {
    assert_eq!(cfg.resolutions.len(), ws.len(), "one set of discriminator weights per resolution");
    for r in &cfg.resolutions {
        assert!(r.num_frames(len) > 0, "a {len}-sample waveform is shorter than n_fft={}", r.n_fft);
    }
}

pub fn mrd_discriminator_step(gpu: &Gpu, ws: &[DiscWeights], cfg: &MrdConfig, real: &[f32], fake: &[f32]) -> MrdDStep {
    check_resolutions(cfg, ws, real.len());
    let steps: Vec<DStep> = cfg.resolutions.iter().zip(ws).map(|(sc, w)| discriminator_step(gpu, w, &cfg.disc, sc, real, fake)).collect();
    MrdDStep { loss: steps.iter().map(|s| s.loss).sum(), steps }
}

/// The generator's side of the adversarial game on one generated waveform.
pub struct GenAdv {
    /// LSGAN generator loss, summed over resolutions (unweighted).
    pub adv_loss: f32,
    /// Feature-matching loss, summed over resolutions (unweighted).
    pub fm_loss: f32,
    /// Gradient of `adv_weight * adv_loss + fm_weight * fm_loss` on the generated waveform: through the logits and,
    /// for feature matching, the first-layer features, then each resolution's [`stft_mag_bwd`].
    pub d_fake: Vec<f32>,
}

/// The generator's adversarial and feature-matching losses on `fake` against `real`, and their gradient on `fake`,
/// with the discriminators held fixed. The feature-matching target is the discriminator's own `a1` on `real`.
pub fn generator_adversarial(gpu: &Gpu, ws: &[DiscWeights], cfg: &MrdConfig, real: &[f32], fake: &[f32], adv_weight: f32, fm_weight: f32) -> GenAdv {
    assert_eq!(real.len(), fake.len(), "real/fake must have the same length");
    check_resolutions(cfg, ws, real.len());
    let mut out = GenAdv { adv_loss: 0.0, fm_loss: 0.0, d_fake: vec![0.0; fake.len()] };
    for (sc, w) in cfg.resolutions.iter().zip(ws) {
        let (f_bins, frames) = (sc.freq_bins(), sc.num_frames(fake.len()));
        let real_stft = stft_mag_fwd(real, sc);
        let fake_stft = stft_mag_fwd(fake, sc);
        let (_, real_cache) = forward(gpu, w, &cfg.disc, &real_stft.mag, f_bins, frames);
        let (fake_logits, fake_cache) = forward(gpu, w, &cfg.disc, &fake_stft.mag, f_bins, frames);

        let (adv, d_logits) = lsgan_g_loss(&fake_logits);
        let (fm, d_feat) = feature_matching_loss(&features(gpu, &real_cache), &features(gpu, &fake_cache));
        let d_logits: Vec<f32> = d_logits.iter().map(|g| g * adv_weight).collect();
        let d_feat: Vec<f32> = d_feat.iter().map(|g| g * fm_weight).collect();
        let grads = backward(gpu, w, &cfg.disc, &fake_cache, &d_logits, Some(&d_feat));

        out.adv_loss += adv;
        out.fm_loss += fm;
        for (acc, g) in out.d_fake.iter_mut().zip(stft_mag_bwd(&fake_stft, &grads.d_mag)) {
            *acc += g;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    fn tiny_stft() -> StftConfig {
        StftConfig { n_fft: 8, hop: 4 }
    }
    fn tiny_disc() -> DiscConfig {
        DiscConfig { c1: 2, c2: 2, k: 3 }
    }

    #[test]
    fn stft_forward_matches_a_hand_dft_at_one_bin() {
        // Bin 0 (DC) of a windowed frame is just sum(window[n]*x[n]) - no
        // trig needed to check it independently of stft_mag_fwd's own loop.
        let cfg = tiny_stft();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = stft_mag_fwd(&x, &cfg);
        let win = cfg.window();
        let want_re0: f32 = win.iter().zip(&x).map(|(w, v)| w * v).sum();
        assert!((out.re[0] - want_re0).abs() < 1e-4, "re[0,0]={} want={want_re0}", out.re[0]);
        assert!(out.im[0].abs() < 1e-4, "im at DC must be ~0, got {}", out.im[0]);
    }

    #[test]
    fn stft_backward_matches_finite_differences() {
        let cfg = tiny_stft();
        let mut r = Lcg::new(data::rng::random_seed());
        let x = r.vec_scaled(20, 0.5);
        let out = stft_mag_fwd(&x, &cfg);
        let (f_bins, frames) = (cfg.freq_bins(), cfg.num_frames(x.len()));
        let dmag = r.vec_scaled(f_bins * frames, 1.0);
        let dx = stft_mag_bwd(&out, &dmag);

        let loss = |x: &[f32]| -> f32 {
            let o = stft_mag_fwd(x, &cfg);
            o.mag.iter().zip(&dmag).map(|(m, d)| m * d).sum()
        };
        // Round-off allowance from `rounding::agreement_bound`: `S = sum |mag * dmag|`, and a chain is under 64
        // binary32 roundings (an 8-term windowed DFT sum, its square root, then the accumulation over bins).
        let s_abs: f64 = stft_mag_fwd(&x, &cfg).mag.iter().zip(&dmag).map(|(m, d)| f64::from((m * d).abs())).sum();
        let eps = 1e-3f32;
        let abs_tol = (brain_testutil::rounding::agreement_bound(64, s_abs) / (2.0 * f64::from(eps))) as f32;
        for i in (0..x.len()).step_by(3) {
            let mut p = x.to_vec();
            p[i] = x[i] + eps;
            let lp = loss(&p);
            p[i] = x[i] - eps;
            let lm = loss(&p);
            let num = (lp - lm) / (2.0 * eps);
            // A bin whose magnitude is near zero bends within a step (the magnitude is a norm): the two one-sided
            // differences then disagree and the sample measures no gradient, as in `brain_testutil::fd::Directional`.
            let l0 = loss(&x);
            let (ahead, behind) = ((lp - l0) / eps, (l0 - lm) / eps);
            let tol = abs_tol + 1e-2 * num.abs().max(dx[i].abs());
            if (ahead - behind).abs() > tol {
                continue;
            }
            assert!((num - dx[i]).abs() < tol, "dx[{i}]: numeric={num} analytic={}", dx[i]);
        }
    }

    #[test]
    fn feature_matching_loss_matches_finite_differences() {
        let mut r = Lcg::new(data::rng::random_seed());
        let real = r.vec_scaled(12, 1.0);
        let fake = r.vec_scaled(12, 1.0);
        let (_, grad) = feature_matching_loss(&real, &fake);
        let eps = 1e-3f32;
        for i in 0..fake.len() {
            let mut p = fake.clone();
            p[i] = fake[i] + eps;
            let lp = feature_matching_loss(&real, &p).0;
            p[i] = fake[i] - eps;
            let lm = feature_matching_loss(&real, &p).0;
            let num = (lp - lm) / (2.0 * eps);
            // L1 grad is a step function at 0 - the finite-diff estimate can
            // land on either side if lp/lm straddle the kink, so this only
            // checks magnitude/sign agreement, not tight closeness.
            assert!((num - grad[i]).abs() < 0.05, "feat[{i}]: numeric={num} analytic={}", grad[i]);
        }
    }

    /// Full-chain gradcheck: perturbing the FAKE waveform must move the
    /// LSGAN generator loss the way `stft_mag_bwd(backward(...).d_mag)`
    /// predicts - the seam [`generator_adversarial`] hands the vocoder's backward.
    #[test]
    fn full_chain_waveform_gradient_matches_finite_differences() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let stft_cfg = tiny_stft();
        let disc_cfg = tiny_disc();
        let w = random_weights(&disc_cfg, 3);
        let mut r = Lcg::new(data::rng::random_seed());
        let fake = r.vec_scaled(20, 0.5);
        let (f_bins, frames) = (stft_cfg.freq_bins(), stft_cfg.num_frames(fake.len()));

        let loss_of = |fake: &[f32]| -> f32 {
            let stft = stft_mag_fwd(fake, &stft_cfg);
            let (logits, _) = forward(&gpu, &w, &disc_cfg, &stft.mag, f_bins, frames);
            lsgan_g_loss(&logits).0
        };

        let stft = stft_mag_fwd(&fake, &stft_cfg);
        let (logits, cache) = forward(&gpu, &w, &disc_cfg, &stft.mag, f_bins, frames);
        let (_, d_logits) = lsgan_g_loss(&logits);
        let grads = backward(&gpu, &w, &disc_cfg, &cache, &d_logits, None);
        let dx = stft_mag_bwd(&stft, &grads.d_mag);

        let eps = 2e-3f32;
        for i in (0..fake.len()).step_by(4) {
            let mut p = fake.clone();
            p[i] = fake[i] + eps;
            let lp = loss_of(&p);
            p[i] = fake[i] - eps;
            let lm = loss_of(&p);
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dx[i]).abs() < 5e-2 + 5e-2 * num.abs().max(dx[i].abs()), "dx[{i}]: numeric={num} analytic={}", dx[i]);
        }
    }

    #[test]
    fn discriminator_backward_matches_finite_differences() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let stft_cfg = tiny_stft();
        let disc_cfg = tiny_disc();
        let w = random_weights(&disc_cfg, 5);
        let mut r = Lcg::new(data::rng::random_seed());
        let real = r.vec_scaled(20, 0.5);
        let fake = r.vec_scaled(20, 0.5);

        let loss_of = |w: &DiscWeights| -> f32 {
            let rs = stft_mag_fwd(&real, &stft_cfg);
            let fs = stft_mag_fwd(&fake, &stft_cfg);
            let (fb, fr) = (stft_cfg.freq_bins(), stft_cfg.num_frames(real.len()));
            let (rl, _) = forward(&gpu, w, &disc_cfg, &rs.mag, fb, fr);
            let (fl, _) = forward(&gpu, w, &disc_cfg, &fs.mag, fb, fr);
            lsgan_d_loss(&rl, &fl).0
        };

        let step = discriminator_step(&gpu, &w, &disc_cfg, &stft_cfg, &real, &fake);
        let eps = 5e-3f32;

        // conv1.weight
        {
            let mut wp = w.clone();
            for i in (0..wp.conv1.weight.len()).step_by((wp.conv1.weight.len() / 3).max(1)) {
                let orig = wp.conv1.weight[i];
                wp.conv1.weight[i] = orig + eps;
                let lp = loss_of(&wp);
                wp.conv1.weight[i] = orig - eps;
                let lm = loss_of(&wp);
                wp.conv1.weight[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                let ana = step.d_conv1.weight[i];
                assert!((num - ana).abs() < 2e-2 + 2e-2 * num.abs().max(ana.abs()), "conv1.weight[{i}]: numeric={num} analytic={ana}");
            }
        }
        // conv2.bias
        {
            let mut wp = w.clone();
            for i in 0..wp.conv2.bias.len() {
                let orig = wp.conv2.bias[i];
                wp.conv2.bias[i] = orig + eps;
                let lp = loss_of(&wp);
                wp.conv2.bias[i] = orig - eps;
                let lm = loss_of(&wp);
                wp.conv2.bias[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                let ana = step.d_conv2.bias[i];
                assert!((num - ana).abs() < 2e-2 + 2e-2 * num.abs().max(ana.abs()), "conv2.bias[{i}]: numeric={num} analytic={ana}");
            }
        }
        // conv_out.weight
        {
            let mut wp = w.clone();
            for i in 0..wp.conv_out.weight.len() {
                let orig = wp.conv_out.weight[i];
                wp.conv_out.weight[i] = orig + eps;
                let lp = loss_of(&wp);
                wp.conv_out.weight[i] = orig - eps;
                let lm = loss_of(&wp);
                wp.conv_out.weight[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                let ana = step.d_conv_out.weight[i];
                assert!((num - ana).abs() < 2e-2 + 2e-2 * num.abs().max(ana.abs()), "conv_out.weight[{i}]: numeric={num} analytic={ana}");
            }
        }
    }

    #[test]
    fn discriminator_learns_to_separate_real_and_fake() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let stft_cfg = tiny_stft();
        let disc_cfg = tiny_disc();
        let mut w = random_weights(&disc_cfg, 7);
        let mut r = Lcg::new(data::rng::random_seed());
        // Two visibly different populations: a low-frequency-ish "real" and
        // a noisier "fake" - what matters for this test is only that they
        // differ, not that either resembles real audio.
        let real: Vec<f32> = (0..24).map(|i| (i as f32 * 0.5).sin()).collect();
        let fake: Vec<f32> = r.vec_scaled(24, 1.0);
        let lr = 0.05f32;

        let loss0 = discriminator_step(&gpu, &w, &disc_cfg, &stft_cfg, &real, &fake).loss;
        let mut loss = loss0;
        for _ in 0..200 {
            let step = discriminator_step(&gpu, &w, &disc_cfg, &stft_cfg, &real, &fake);
            loss = step.loss;
            let upd = |wgt: &mut Conv2dW, d: &Conv2dW| {
                for (v, dv) in wgt.weight.iter_mut().zip(&d.weight) {
                    *v -= lr * dv;
                }
                for (v, dv) in wgt.bias.iter_mut().zip(&d.bias) {
                    *v -= lr * dv;
                }
            };
            upd(&mut w.conv1, &step.d_conv1);
            upd(&mut w.conv2, &step.d_conv2);
            upd(&mut w.conv_out, &step.d_conv_out);
        }
        assert!(loss < loss0 * 0.5, "discriminator did not learn to separate real/fake: start={loss0} end={loss}");
    }

    /// Spec: the multi-resolution discriminator is the single-resolution one run per `(n_fft, hop)`, losses summed:
    /// one implementation, parameterised, no second copy.
    #[test]
    fn the_multi_resolution_step_sums_the_single_resolution_steps() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let mrd = MrdConfig { resolutions: vec![StftConfig { n_fft: 8, hop: 4 }, StftConfig { n_fft: 16, hop: 8 }], disc: tiny_disc() };
        let ws = random_mrd_weights(&mrd, 9);
        let mut r = Lcg::new(data::rng::random_seed());
        let (real, fake) = (r.vec_scaled(32, 0.5), r.vec_scaled(32, 0.5));
        let multi = mrd_discriminator_step(&gpu, &ws, &mrd, &real, &fake);
        assert_eq!(multi.steps.len(), 2);
        let mut sum = 0.0f32;
        for (i, sc) in mrd.resolutions.iter().enumerate() {
            let one = discriminator_step(&gpu, &ws[i], &mrd.disc, sc, &real, &fake);
            assert_eq!(multi.steps[i].d_conv1.weight, one.d_conv1.weight);
            sum += one.loss;
        }
        assert!((multi.loss - sum).abs() <= 1e-6 * sum.abs().max(1.0), "{} vs {sum}", multi.loss);
    }

    /// Spec: `generator_adversarial`'s waveform gradient is the derivative of `adv_weight * sum LSGAN-G +
    /// fm_weight * sum FM` over every resolution, through the STFT back to the waveform. Central differences, the
    /// round-off allowance derived from `rounding::agreement_bound` (the loss is a sum of non-negative terms; a
    /// chain is under 512 binary32 roundings: three 3x3 convolutions of at most 18 terms and a 16-point DFT).
    #[test]
    fn the_generator_adversarial_gradient_matches_central_differences() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let mrd = MrdConfig { resolutions: vec![StftConfig { n_fft: 8, hop: 4 }, StftConfig { n_fft: 16, hop: 8 }], disc: tiny_disc() };
        let ws = random_mrd_weights(&mrd, 13);
        let mut r = Lcg::new(data::rng::random_seed());
        let (real, fake) = (r.vec_scaled(32, 0.5), r.vec_scaled(32, 0.5));
        let (adv_w, fm_w) = (1.0f32, 2.0f32);
        let total = |f: &[f32]| {
            let g = generator_adversarial(&gpu, &ws, &mrd, &real, f, adv_w, fm_w);
            adv_w * g.adv_loss + fm_w * g.fm_loss
        };
        let analytic = generator_adversarial(&gpu, &ws, &mrd, &real, &fake, adv_w, fm_w).d_fake;
        let eps = 5e-3f32;
        let abs_tol = (brain_testutil::rounding::agreement_bound(512, f64::from(total(&fake))) / (2.0 * f64::from(eps))) as f32;
        for i in 0..fake.len() {
            let mut p = fake.clone();
            p[i] = fake[i] + eps;
            let lp = total(&p);
            p[i] = fake[i] - eps;
            let lm = total(&p);
            let num = (lp - lm) / (2.0 * eps);
            // The L1 feature-matching term has kinks; a step straddling one measures nothing, so a sample whose two
            // one-sided differences disagree is skipped (as `brain_testutil::fd::Directional` does).
            let (ahead, behind) = ((lp - total(&fake)) / eps, (total(&fake) - lm) / eps);
            if (ahead - behind).abs() > abs_tol + 2e-2 * ahead.abs().max(behind.abs()) {
                continue;
            }
            assert!((num - analytic[i]).abs() <= abs_tol + 2e-2 * num.abs().max(analytic[i].abs()), "d_fake[{i}]: numeric={num} analytic={}", analytic[i]);
        }
    }
}
