// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The neural-codec **trainer**: reconstruction, multi-scale mel, and the adversarial and feature-matching terms.
//!
//!   wav -> conv encoder -> straight-through VQ -> conv decoder -> wav'
//!   generator loss = waveform L1+L2 + commitment
//!                    + mel_weight * multi-scale log-mel L1
//!                    + adv_weight * LSGAN generator loss + feat_weight * feature matching   (the last two from
//!                      `GanConfig::start_step` on, when a [`GanConfig`] is set)
//!   discriminator loss = LSGAN over the multi-period and multi-scale waveform discriminators ([`crate::disc`]),
//!                        trained by its own AdamW
//!
//! The differentiable pieces are:
//!   * [`waveform_recon`] - `mean(|y-x|) + 0.5 mean((y-x)^2)`, exact analytic grad w.r.t. the prediction;
//!   * [`multiscale_mel_l1`] - the log-mel L1 averaged over STFT scales, with its gradient w.r.t. the waveform
//!     (`audio::mel::log_mel_backward`);
//!   * [`StraightThroughVq`] - vector-quantize each frame to its nearest codebook entry; the decoder sees the
//!     quantized vectors, the gradient flows to the encoder by the **straight-through estimator** (identity) plus a
//!     commitment term `beta ||z - sg(q)||^2`;
//!   * [`ConvReconAE`] - a tiny causal-conv encoder/decoder over the shared WGSL engine (`audio::conv` fwd +
//!     `conv1d_dx`/`conv1d_dw` bwd), so the conv gradients are the real kernel gradients.
//!
//! [`ConvReconAE::gradcheck`] freezes the VQ assignment (the argmin is piecewise constant; the straight-through
//! estimator is exactly the frozen-assignment Jacobian) and compares the analytic encoder/decoder weight gradients to
//! central finite differences of the end-to-end loss, mel term included.
//!
//! # Weights and where they come from
//!
//! The mel, adversarial and feature-matching weights are [`ReconObjective`] and [`GanConfig`] fields. Their defaults
//! (45, 1, 2) are HiFi-GAN's `lambda_mel`, adversarial and `lambda_fm`, the recipe whose multi-period and multi-scale
//! discriminators this trainer uses; the waveform and commitment terms keep the unit weight they had before. The mel
//! scales (FFT 512, 1024, 2048 at 24 kHz, hop a quarter of the window, 80 mel bins) are this crate's choice for the
//! 24 kHz codec: the multi-resolution form the DAC/Encodec family uses instead of HiFi-GAN's single 1024 window.
//!
//! The published codec recipe also distils a WavLM teacher into the first quantizer's output; that needs a teacher
//! model brain does not have and is not part of this trainer.

use audio::conv::{Conv1d, ConvKernels};
use audio::mel::{log_mel, log_mel_backward, MelConfig};
use bytemuck::cast_slice;
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

use crate::disc::{feature_matching, lsgan_d_loss, lsgan_g_loss, DiscConfig, WaveDiscriminators};

/// Pipelines this trainer needs: conv forward + the two conv gradients.
const RC_CONV1D: usize = 0;
const RC_CONV1D_DX: usize = 1;
const RC_CONV1D_DW: usize = 2;
const RECON_PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
];

/// Waveform reconstruction loss `mean(|y-x|) + 0.5·mean((y-x)²)` and its gradient
/// w.r.t. `y` (`pred`). Both terms are averaged over the (shared) length.
pub fn waveform_recon(pred: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    assert_eq!(pred.len(), target.len(), "pred/target length mismatch");
    let n = pred.len().max(1) as f32;
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; pred.len()];
    for i in 0..pred.len() {
        let d = pred[i] - target[i];
        loss += d.abs() + 0.5 * d * d;
        grad[i] = (d.signum() + d) / n;
    }
    (loss / n, grad)
}

/// Multi-scale log-mel L1 between a prediction and a target waveform, and its gradient w.r.t. the prediction:
/// `(1/S) sum_s mean_i |logmel_s(pred)_i - logmel_s(target)_i|` over the `S` scales. A scale with no frames (a
/// waveform shorter than its window) contributes nothing to the sum but still counts in the average.
///
/// Per scale the gradient w.r.t. the log-mel is `sign(pred - target) / (n S)`, with `sign(0) = 0` (a subgradient of
/// the kink of `|.|`, so a perfect reconstruction is a stationary point), pulled back to the waveform by
/// [`log_mel_backward`]. The target is a constant. The log-mel is built on the power spectrum, which is smooth at
/// zero magnitude, so no convention is needed there.
pub fn multiscale_mel_l1(pred: &[f32], target: &[f32], scales: &[MelConfig]) -> (f32, Vec<f32>) {
    let s = scales.len().max(1) as f32;
    let mut total = 0.0f32;
    let mut grad = vec![0.0f32; pred.len()];
    for cfg in scales {
        let (mp, _) = log_mel(pred, cfg);
        let (mt, _) = log_mel(target, cfg);
        let n = mp.len().min(mt.len());
        if n == 0 {
            continue;
        }
        total += (0..n).map(|i| (mp[i] - mt[i]).abs()).sum::<f32>() / n as f32;
        let d_log_mel: Vec<f32> = (0..n)
            .map(|i| {
                let d = mp[i] - mt[i];
                (if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 }) / (n as f32 * s)
            })
            .collect();
        for (g, d) in grad.iter_mut().zip(log_mel_backward(pred, cfg, &d_log_mel)) {
            *g += d;
        }
    }
    (total / s, grad)
}

/// The 24 kHz multi-resolution mel scales of [`ReconObjective::default`]: FFT and window 512, 1024 and 2048, hop a
/// quarter of the window, 80 Slaney mel bins up to 12 kHz.
pub fn multiscale_mel_24k() -> Vec<MelConfig> {
    [512usize, 1024, 2048].iter().map(|&n| MelConfig { sample_rate: 24000, n_fft: n, hop: n / 4, win: n, n_mels: 80, fmin: 0.0, fmax: 12000.0, slaney: true, center: true }).collect()
}

/// The adversarial half of the objective: the discriminators, and the weights and start of the generator's terms
/// against them. [`GanConfig::default`] is HiFi-GAN's recipe (adversarial 1, feature matching 2, discriminators on
/// from the first step).
#[derive(Clone, Debug)]
pub struct GanConfig {
    pub disc: DiscConfig,
    /// Weight of the LSGAN generator loss.
    pub adv_weight: f32,
    /// Weight of the feature-matching loss.
    pub feat_weight: f32,
    /// First step (counting from 0) on which the discriminators train and the generator sees their terms; earlier
    /// steps train the generator on reconstruction alone.
    pub start_step: u32,
}

impl Default for GanConfig {
    fn default() -> GanConfig {
        GanConfig { disc: DiscConfig::default(), adv_weight: 1.0, feat_weight: 2.0, start_step: 0 }
    }
}

/// What the generator trains on besides the waveform and commitment terms.
#[derive(Clone, Debug)]
pub struct ReconObjective {
    /// Weight of [`multiscale_mel_l1`] over [`ReconObjective::mel_scales`]; 0 turns the term off (HiFi-GAN: 45).
    pub mel_weight: f32,
    pub mel_scales: Vec<MelConfig>,
    /// The discriminators and their terms; `None` trains without them (the default: the HiFi-GAN-sized
    /// discriminators are far larger than the toy autoencoder).
    pub gan: Option<GanConfig>,
}

impl Default for ReconObjective {
    fn default() -> ReconObjective {
        ReconObjective { mel_weight: 45.0, mel_scales: multiscale_mel_24k(), gan: None }
    }
}

/// The terms of one training step, measured on its forward (before the step's updates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLosses {
    pub wave: f32,
    pub commit: f32,
    /// The unweighted multi-scale mel L1.
    pub mel: f32,
    /// The unweighted adversarial and feature-matching losses and the discriminator's loss; 0 while the
    /// discriminators are off.
    pub adv: f32,
    pub feat: f32,
    pub disc: f32,
    /// The generator objective: every term times its weight.
    pub generator: f32,
}

/// A frozen vector-quantizer codebook `[bins, dim]` with the straight-through
/// estimator. Quantizes each `dim`-vector to its nearest entry.
pub struct StraightThroughVq {
    pub bins: usize,
    pub dim: usize,
    pub codebook: Vec<f32>, // [bins, dim]
    pub beta: f32,          // commitment weight
}

impl StraightThroughVq {
    /// A small deterministic codebook (for tests / the toy trainer).
    pub fn demo(bins: usize, dim: usize, beta: f32) -> StraightThroughVq {
        let mut codebook = vec![0.0f32; bins * dim];
        for b in 0..bins {
            for d in 0..dim {
                // spread entries over [-1,1] deterministically
                codebook[b * dim + d] = ((b * 31 + d * 17) % 200) as f32 / 100.0 - 1.0;
            }
        }
        StraightThroughVq { bins, dim, codebook, beta }
    }

    /// Nearest codebook index for one `dim`-vector (argmin squared-Euclidean).
    pub fn nearest(&self, z: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for b in 0..self.bins {
            let row = &self.codebook[b * self.dim..(b + 1) * self.dim];
            let mut d = 0.0f32;
            for c in 0..self.dim {
                let e = z[c] - row[c];
                d += e * e;
            }
            if d < bd {
                bd = d;
                best = b;
            }
        }
        best
    }

    /// Quantize `z` (`[n, dim]`) to `(q, indices)`. With `frozen` indices supplied,
    /// reuse them (so the forward is smooth for finite-difference gradchecks).
    pub fn quantize(&self, z: &[f32], frozen: Option<&[usize]>) -> (Vec<f32>, Vec<usize>) {
        let n = z.len() / self.dim;
        let mut q = vec![0.0f32; z.len()];
        let mut idx = vec![0usize; n];
        for t in 0..n {
            let zt = &z[t * self.dim..(t + 1) * self.dim];
            let b = frozen.map(|f| f[t]).unwrap_or_else(|| self.nearest(zt));
            idx[t] = b;
            q[t * self.dim..(t + 1) * self.dim].copy_from_slice(&self.codebook[b * self.dim..(b + 1) * self.dim]);
        }
        (q, idx)
    }

    /// Commitment loss `β·mean(‖z − sg(q)‖²)` and its gradient w.r.t. `z`
    /// (`β·2/n·(z−q)`); `sg(q)` is a constant so q carries no grad.
    pub fn commitment(&self, z: &[f32], q: &[f32]) -> (f32, Vec<f32>) {
        let n = (z.len() / self.dim).max(1) as f32;
        let mut loss = 0.0f32;
        let mut grad = vec![0.0f32; z.len()];
        for i in 0..z.len() {
            let d = z[i] - q[i];
            loss += d * d;
            grad[i] = self.beta * 2.0 * d / n;
        }
        (self.beta * loss / n, grad)
    }
}

/// A tiny causal-conv reconstruction autoencoder over the shared engine.
///
/// Encoder: `conv1d` 1->`c` (kernel `k`, causal). Quantizer: per-time-step VQ over
/// the `c` channels (straight-through). Decoder: `conv1d` `c`->1 (kernel `k`,
/// causal). Trained to minimise the generator objective of this module's header: [`waveform_recon`], the
/// commitment, the mel term, and, with a [`GanConfig`], the adversarial and feature-matching terms.
pub struct ConvReconAE {
    gpu: Gpu,
    pub c: u32,
    pub k: u32,
    pub vq: StraightThroughVq,
    pub we: Vec<f32>, // encoder weight [c, 1, k]
    pub wd: Vec<f32>, // decoder weight [1, c, k]
    pub objective: ReconObjective,
    /// Present when `objective.gan` is set.
    disc: Option<WaveDiscriminators>,
    /// Training steps taken (the clock `GanConfig::start_step` is read against) and discriminator updates made
    /// (AdamW's step count).
    step: u32,
    disc_step: u32,
}

impl ConvReconAE {
    pub fn new(c: u32, k: u32, beta: f32, seed: u64) -> ConvReconAE {
        let gpu = Gpu::new_cpu(RECON_PIPELINES);
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let we: Vec<f32> = (0..c * k).map(|_| rnd() * 0.5).collect();
        let wd: Vec<f32> = (0..c * k).map(|_| rnd() * 0.5).collect();
        let vq = StraightThroughVq::demo(8, c as usize, beta);
        ConvReconAE { gpu, c, k, vq, we, wd, objective: ReconObjective::default(), disc: None, step: 0, disc_step: 0 }
    }

    /// This autoencoder trained on `objective`; its discriminators (when it has a [`GanConfig`]) are built on a
    /// handle of this device.
    pub fn with_objective(mut self, objective: ReconObjective) -> ConvReconAE {
        self.disc = objective.gan.as_ref().map(|g| WaveDiscriminators::new(self.gpu.new_like(crate::disc::PIPELINES), g.disc.clone()));
        self.objective = objective;
        self
    }

    fn up(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu.buffer("rc", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST);
        self.gpu.write(&b, cast_slice(data));
        b
    }
    fn conv(&self, cin: u32, cout: u32, l: u32) -> Conv1d {
        // causal: left pad k-1, lo == l
        Conv1d { n: 1, cin, l, cout, k: self.k, stride: 1, pad: self.k - 1, dilation: 1, groups: 1, lo: l }
    }

    /// Forward to the encoder latent `z` `[c, L]` (NCL). Returns `(z, x_buf)`.
    fn encode(&self, x: &[f32]) -> (Vec<f32>, u32) {
        let l = x.len() as u32;
        let xb = self.up(x);
        let web = self.up(&self.we);
        let zb = self.gpu.storage((self.c * l) as u64);
        let c = self.conv(1, self.c, l);
        self.gpu.submit(&[], &[audio::conv::conv1d_fwd(&self.gpu, &ConvKernels { fwd: RC_CONV1D, dx: 0, dw: 0 }, &c, &xb, &web, &zb)]);
        (self.gpu.read(&zb, (self.c * l) as usize), l)
    }

    /// Decode quantized latent `q` `[c, L]` (NCL, time-major per channel) to a
    /// waveform `[L]`.
    fn decode(&self, q_ncl: &[f32], l: u32) -> Vec<f32> {
        let qb = self.up(q_ncl);
        let wdb = self.up(&self.wd);
        let yb = self.gpu.storage(l as u64);
        let c = self.conv(self.c, 1, l);
        self.gpu.submit(&[], &[audio::conv::conv1d_fwd(&self.gpu, &ConvKernels { fwd: RC_CONV1D, dx: 0, dw: 0 }, &c, &qb, &wdb, &yb)]);
        self.gpu.read(&yb, l as usize)
    }

    /// `[c, L]` NCL latent <-> `[L, c]` per-frame vectors (VQ operates per frame).
    fn ncl_to_frames(z: &[f32], c: usize, l: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; l * c];
        for ch in 0..c {
            for t in 0..l {
                o[t * c + ch] = z[ch * l + t];
            }
        }
        o
    }
    fn frames_to_ncl(f: &[f32], c: usize, l: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; c * l];
        for t in 0..l {
            for ch in 0..c {
                o[ch * l + t] = f[t * c + ch];
            }
        }
        o
    }

    /// The mel term at `y` against `x`: `(unweighted loss, gradient of mel_weight * loss w.r.t. y)`; zeros and no
    /// work when the weight is 0.
    fn mel_term(&self, y: &[f32], x: &[f32]) -> (f32, Vec<f32>) {
        if self.objective.mel_weight == 0.0 {
            return (0.0, vec![0.0; y.len()]);
        }
        let (loss, mut grad) = multiscale_mel_l1(y, x, &self.objective.mel_scales);
        grad.iter_mut().for_each(|g| *g *= self.objective.mel_weight);
        (loss, grad)
    }

    /// Full forward; returns `(y, recon_loss, commit_loss, indices)`. The decoder
    /// sees the quantized latent `q` (the real codec forward).
    pub fn forward(&self, x: &[f32], frozen: Option<&[usize]>) -> (Vec<f32>, f32, f32, Vec<usize>) {
        let (z, l) = self.encode(x);
        let zf = Self::ncl_to_frames(&z, self.c as usize, l as usize);
        let (qf, idx) = self.vq.quantize(&zf, frozen);
        let qn = Self::frames_to_ncl(&qf, self.c as usize, l as usize);
        let y = self.decode(&qn, l);
        let (recon, _) = waveform_recon(&y, x);
        let (commit, _) = self.vq.commitment(&zf, &qf);
        (y, recon, commit, idx)
    }

    /// The generator's non-adversarial loss `recon + commitment + mel_weight * mel` (frozen VQ assignment optional).
    pub fn loss(&self, x: &[f32], frozen: Option<&[usize]>) -> f32 {
        let (y, r, c, _) = self.forward(x, frozen);
        r + c + self.objective.mel_weight * self.mel_term(&y, x).0
    }

    /// The **straight-through surrogate** loss: the decoder is fed
    /// `z + latch`, where `latch = q0 - z0` is the quantization offset taken at
    /// the latch point and then held fixed. At the latch it IS [`Self::loss`]
    /// (the decoder sees `q`), and its gradient is the straight-through one:
    /// the decoder differentiated at `q`, its input gradient passed to `z`
    /// unchanged. That is what [`Self::grads`] computes, so the two agree under
    /// finite differences. (Feeding the decoder `z` itself instead checks a
    /// different function than the one trained.)
    pub fn surrogate_loss(&self, x: &[f32], frozen: &[usize], latch: &[f32]) -> f32 {
        let l = x.len() as u32;
        let (z, _) = self.encode(x);
        let zf = Self::ncl_to_frames(&z, self.c as usize, l as usize);
        let (qf, _) = self.vq.quantize(&zf, Some(frozen));
        let fed: Vec<f32> = zf.iter().zip(latch).map(|(z, d)| z + d).collect();
        let y = self.decode(&Self::frames_to_ncl(&fed, self.c as usize, l as usize), l);
        let (recon, _) = waveform_recon(&y, x);
        let (commit, _) = self.vq.commitment(&zf, &qf);
        recon + commit + self.objective.mel_weight * self.mel_term(&y, x).0
    }

    /// Analytic gradients of the straight-through objective w.r.t. the
    /// encoder/decoder weights, via the real conv backward kernels. The decoder
    /// runs forward and backward on the quantized latent `q`, as in the real
    /// forward; its input gradient passes to `z` unchanged (straight-through)
    /// and the commitment term is added on the encoder side. Returns `(dWe, dWd)` for the frozen assignment.
    pub fn grads(&self, x: &[f32], frozen: &[usize]) -> (Vec<f32>, Vec<f32>) {
        self.grads_with(x, frozen, None)
    }

    /// [`Self::grads`] with an extra gradient w.r.t. the decoded waveform (the adversarial and feature-matching
    /// terms, which the discriminators differentiate to `y`) added to the waveform and mel terms'.
    fn grads_with(&self, x: &[f32], frozen: &[usize], extra_dy: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let l = x.len() as u32;
        let (z, _) = self.encode(x);
        let zf = Self::ncl_to_frames(&z, self.c as usize, l as usize);
        let (qf, _) = self.vq.quantize(&zf, Some(frozen));
        let qn = Self::frames_to_ncl(&qf, self.c as usize, l as usize);
        let y = self.decode(&qn, l);

        // dL/dy (recon, mel, and any discriminator terms) and dL/dz (commitment, straight-through identity on q).
        let (_, mut dy) = waveform_recon(&y, x);
        for (d, m) in dy.iter_mut().zip(self.mel_term(&y, x).1) {
            *d += m;
        }
        if let Some(extra) = extra_dy {
            for (d, e) in dy.iter_mut().zip(extra) {
                *d += e;
            }
        }
        let (_, dz_commit) = self.vq.commitment(&zf, &qf);

        // decoder backward on input q: dq (input grad) + dWd (weight grad).
        let cdec = self.conv(self.c, 1, l);
        let kdec = ConvKernels { fwd: RC_CONV1D, dx: RC_CONV1D_DX, dw: RC_CONV1D_DW };
        let dyb = self.up(&dy);
        let qb = self.up(&qn);
        let wdb = self.up(&self.wd);
        let dqb = self.gpu.storage((self.c * l) as u64);
        let dwdb = self.gpu.storage((self.c * self.k) as u64);
        self.gpu.write(&dwdb, cast_slice(&vec![0.0f32; (self.c * self.k) as usize]));
        self.gpu.submit(&[], &audio::conv::conv1d_bwd(&self.gpu, &kdec, &cdec, &dyb, &qb, &wdb, Some(&dqb), Some(&dwdb)));
        let dq = self.gpu.read(&dqb, (self.c * l) as usize); // [c,L] NCL
        let dwd = self.gpu.read(&dwdb, (self.c * self.k) as usize);

        // straight-through: dz = dq (passthrough) + commitment grad.
        let dq_frames = Self::ncl_to_frames(&dq, self.c as usize, l as usize);
        let mut dz_frames = dq_frames;
        for i in 0..dz_frames.len() {
            dz_frames[i] += dz_commit[i];
        }
        let dz_ncl = Self::frames_to_ncl(&dz_frames, self.c as usize, l as usize);

        // encoder backward: only the weight grad dWe (input is the fixed wav).
        let cenc = self.conv(1, self.c, l);
        let kenc = ConvKernels { fwd: RC_CONV1D, dx: RC_CONV1D_DX, dw: RC_CONV1D_DW };
        let dzb = self.up(&dz_ncl);
        let xb = self.up(x);
        let web = self.up(&self.we);
        let dweb = self.gpu.storage((self.c * self.k) as u64);
        self.gpu.write(&dweb, cast_slice(&vec![0.0f32; (self.c * self.k) as usize]));
        self.gpu.submit(&[], &audio::conv::conv1d_bwd(&self.gpu, &kenc, &cenc, &dzb, &xb, &web, None, Some(&dweb)));
        let dwe = self.gpu.read(&dweb, (self.c * self.k) as usize);
        (dwe, dwd)
    }

    /// One plain SGD step on the toy waveform; returns the post-step [`Self::loss`] (the non-adversarial terms).
    pub fn train_step(&mut self, x: &[f32], lr: f32) -> f32 {
        self.train_step_report(x, lr);
        self.loss(x, None)
    }

    /// One training step, returning every term of its forward.
    ///
    /// With discriminators and `step >= start_step`: the discriminators take one AdamW step on `(x, y)` (LSGAN), then
    /// the generator's adversarial and feature-matching gradients are taken through the *updated* discriminators
    /// (HiFi-GAN's order), added to the reconstruction and mel gradients at `y`, and the generator takes its SGD step.
    pub fn train_step_report(&mut self, x: &[f32], lr: f32) -> StepLosses {
        // Use the live VQ assignment for the step (straight-through).
        let (y, wave, commit, idx) = self.forward(x, None);
        let mel = self.mel_term(&y, x).0;
        let mut losses = StepLosses { wave, commit, mel, adv: 0.0, feat: 0.0, disc: 0.0, generator: 0.0 };
        let mut extra_dy = None;
        if let (Some(d), Some(gan)) = (&self.disc, &self.objective.gan) {
            if self.step >= gan.start_step {
                let (fake, real) = (d.forward(&y), d.forward(x));
                let dl = lsgan_d_loss(&real.logits, &fake.logits);
                d.zero_grads();
                d.backward(&real, &dl.d_real, None, false);
                d.backward(&fake, &dl.d_fake, None, false);
                self.disc_step += 1;
                d.adamw_step(self.disc_step);
                losses.disc = dl.loss as f32;

                let (real, fake) = (d.forward(x), d.forward(&y));
                let (adv, mut d_logits) = lsgan_g_loss(&fake.logits);
                let (feat, mut d_feats) = feature_matching(&real.features, &fake.features);
                d_logits.iter_mut().flatten().for_each(|v| *v *= gan.adv_weight);
                d_feats.iter_mut().flatten().flatten().for_each(|v| *v *= gan.feat_weight);
                d.zero_grads();
                extra_dy = d.backward(&fake, &d_logits, Some(&d_feats), true);
                (losses.adv, losses.feat) = (adv as f32, feat as f32);
            }
        }
        let weights = self.objective.gan.as_ref().map_or((0.0, 0.0), |g| (g.adv_weight, g.feat_weight));
        losses.generator = wave + commit + self.objective.mel_weight * mel + weights.0 * losses.adv + weights.1 * losses.feat;

        let (dwe, dwd) = self.grads_with(x, &idx, extra_dy.as_deref());
        for (i, w) in self.we.iter_mut().enumerate() {
            *w -= lr * dwe[i];
        }
        for (i, w) in self.wd.iter_mut().enumerate() {
            *w -= lr * dwd[i];
        }
        self.step += 1;
        losses
    }

    /// Inline finite-difference gradcheck of the **whole reconstruction path**
    /// (encoder conv -> straight-through VQ -> decoder conv -> recon+commit loss),
    /// with the VQ assignment frozen so the forward is smooth. Returns the max
    /// relative error over a few probed weights of each tensor.
    pub fn gradcheck(&self, x: &[f32], eps: f32) -> f32 {
        let (_, _, _, idx) = self.forward(x, None);
        let (dwe, dwd) = self.grads(x, &idx);
        let (z0, _) = self.encode(x);
        let z0 = Self::ncl_to_frames(&z0, self.c as usize, x.len());
        let (q0, _) = self.vq.quantize(&z0, Some(&idx));
        let latch: Vec<f32> = q0.iter().zip(&z0).map(|(q, z)| q - z).collect();

        // Central difference of the loss w.r.t. weight `i` with the VQ assignment
        // frozen, for both the encoder (`enc=true`) and decoder weight tensors.
        let probe = |enc: bool, i: usize| -> f32 {
            let mut m = self.clone_weights();
            if enc { m.we[i] += eps } else { m.wd[i] += eps }
            let lp = m.surrogate_loss(x, &idx, &latch);
            let mut m = self.clone_weights();
            if enc { m.we[i] -= eps } else { m.wd[i] -= eps }
            let lm = m.surrogate_loss(x, &idx, &latch);
            (lp - lm) / (2.0 * eps)
        };

        let mut max_rel = 0.0f32;
        for (enc, ana) in [(true, &dwe), (false, &dwd)] {
            for &i in &[0usize, ana.len() / 2, ana.len() - 1] {
                let num = probe(enc, i);
                let rel = (ana[i] - num).abs() / ana[i].abs().max(num.abs()).max(1e-6);
                max_rel = max_rel.max(rel);
            }
        }
        max_rel
    }

    /// A copy of the generator's weights and non-adversarial objective, for finite-difference probes (no
    /// discriminators: the probes differentiate the deterministic terms).
    fn clone_weights(&self) -> ConvReconAE {
        ConvReconAE {
            gpu: Gpu::new_cpu(RECON_PIPELINES),
            c: self.c,
            k: self.k,
            vq: StraightThroughVq { bins: self.vq.bins, dim: self.vq.dim, codebook: self.vq.codebook.clone(), beta: self.vq.beta },
            we: self.we.clone(),
            wd: self.wd.clone(),
            objective: ReconObjective { gan: None, ..self.objective.clone() },
            disc: None,
            step: 0,
            disc_step: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disc::tests::tiny_config;
    use brain_testutil::fd::Directional;
    use brain_testutil::rounding::agreement_bound;
    use data::rng::Lcg;

    /// The reconstruction-only objective the original trainer tests were written against.
    fn recon_only() -> ReconObjective {
        ReconObjective { mel_weight: 0.0, ..ReconObjective::default() }
    }

    /// Two mel scales small enough for a 64-sample toy waveform.
    fn toy_scales() -> Vec<MelConfig> {
        [(16usize, 4usize), (32, 8)].iter().map(|&(n, hop)| MelConfig { sample_rate: 8000, n_fft: n, hop, win: n, n_mels: 8, fmin: 0.0, fmax: 4000.0, slaney: true, center: true }).collect()
    }

    fn toy_objective(mel_weight: f32, gan: Option<GanConfig>) -> ReconObjective {
        ReconObjective { mel_weight, mel_scales: toy_scales(), gan }
    }

    fn toy_wav(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.3).sin() * 0.7 + (i as f32 * 0.11).cos() * 0.2).collect()
    }

    #[test]
    fn waveform_recon_grad_matches_finite_diff() {
        let target = toy_wav(32);
        let mut pred = toy_wav(32);
        for (i, p) in pred.iter_mut().enumerate() {
            *p += 0.05 * (i as f32).cos();
        }
        let (_, grad) = waveform_recon(&pred, &target);
        let eps = 1e-3;
        for &i in &[0usize, 7, 31] {
            let mut pp = pred.clone();
            pp[i] += eps;
            let mut pm = pred.clone();
            pm[i] -= eps;
            let num = (waveform_recon(&pp, &target).0 - waveform_recon(&pm, &target).0) / (2.0 * eps);
            assert!((grad[i] - num).abs() < 1e-3, "recon grad[{i}] {} vs {num}", grad[i]);
        }
    }

    #[test]
    fn commitment_grad_matches_finite_diff() {
        let vq = StraightThroughVq::demo(8, 4, 0.25);
        let z: Vec<f32> = (0..5 * 4).map(|i| (i as f32 * 0.2).sin()).collect();
        let (q, idx) = vq.quantize(&z, None);
        let (_, grad) = vq.commitment(&z, &q);
        let eps = 1e-3;
        for &i in &[0usize, 9, 19] {
            // freeze the assignment so q is constant across the perturbation
            let mut zp = z.clone();
            zp[i] += eps;
            let (qp, _) = vq.quantize(&zp, Some(&idx));
            let mut zm = z.clone();
            zm[i] -= eps;
            let (qm, _) = vq.quantize(&zm, Some(&idx));
            let num = (vq.commitment(&zp, &qp).0 - vq.commitment(&zm, &qm).0) / (2.0 * eps);
            assert!((grad[i] - num).abs() < 1e-3, "commit grad[{i}] {} vs {num}", grad[i]);
        }
    }

    #[test]
    fn conv_recon_path_gradchecks() {
        let ae = ConvReconAE::new(4, 3, 0.25, 42).with_objective(recon_only());
        let x = toy_wav(48);
        let max_rel = ae.gradcheck(&x, 1e-3);
        eprintln!("conv-recon gradcheck max_rel = {max_rel:.2e}");
        assert!(max_rel < 2e-2, "reconstruction-path gradcheck failed: max_rel {max_rel:.3e}");
    }

    /// The decoder sees the QUANTIZED latent in the real forward, and with the
    /// assignment frozen the loss is smooth in the decoder's weights, so their
    /// straight-through gradient must be the plain gradient of the real loss.
    #[test]
    fn decoder_weight_grad_is_taken_at_the_quantized_latent() {
        let ae = ConvReconAE::new(4, 3, 0.25, 42).with_objective(recon_only());
        let x = toy_wav(48);
        let (_, _, _, idx) = ae.forward(&x, None);
        let (_, dwd) = ae.grads(&x, &idx);
        let eps = 1e-3;
        for i in [0usize, dwd.len() / 2, dwd.len() - 1] {
            let mut p = ae.clone_weights();
            p.wd[i] += eps;
            let mut m = ae.clone_weights();
            m.wd[i] -= eps;
            let num = (p.loss(&x, Some(&idx)) - m.loss(&x, Some(&idx))) / (2.0 * eps);
            let rel = (dwd[i] - num).abs() / dwd[i].abs().max(num.abs()).max(1e-6);
            assert!(rel < 2e-2, "dWd[{i}] {} vs the real loss's {num}", dwd[i]);
        }
    }

    #[test]
    fn conv_recon_trains_down() {
        let mut ae = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, None));
        let x = toy_wav(64);
        let initial = ae.loss(&x, None);
        let mut last = initial;
        for _ in 0..300 {
            last = ae.train_step(&x, 0.05);
        }
        eprintln!("conv-recon train: {initial:.4} -> {last:.4}");
        assert!(last < initial * 0.85, "reconstruction loss did not drop: {initial:.3} -> {last:.3}");
    }

    /// Sum over scales of `mean |logmel(pred) - logmel(target)|` in f64, from `audio::mel::log_mel` alone.
    fn mel_l1_f64(pred: &[f32], target: &[f32], scales: &[MelConfig]) -> f64 {
        scales
            .iter()
            .map(|cfg| {
                let (mp, mt) = (log_mel(pred, cfg).0, log_mel(target, cfg).0);
                mp.iter().zip(&mt).map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs()).sum::<f64>() / mp.len() as f64
            })
            .sum::<f64>()
            / scales.len() as f64
    }

    /// Spec: the multi-scale mel loss is the scale-average of the log-mel L1, zero for identical waveforms with a
    /// zero gradient there, and its gradient w.r.t. the waveform is the loss's derivative.
    #[test]
    fn multiscale_mel_loss_and_gradient() {
        let scales = toy_scales();
        let (pred, target) = (Lcg::new(data::rng::random_seed()).vec(64), Lcg::new(data::rng::random_seed()).vec(64));
        let (loss, grad) = multiscale_mel_l1(&pred, &target, &scales);
        let want = mel_l1_f64(&pred, &target, &scales);
        assert!((f64::from(loss) - want).abs() <= agreement_bound(64, want), "loss {loss} vs the definition {want}");

        let (same, zero) = multiscale_mel_l1(&pred, &pred, &scales);
        assert_eq!(same, 0.0);
        assert!(zero.iter().all(|g| *g == 0.0), "a perfect reconstruction is a stationary point");

        // Directional central differences of the loss in f64 (a step crossing an |.| kink is redrawn). The loss is
        // evaluated through the f32 log-mel, so each evaluation is within `gamma_k` of its terms' magnitude: `k`
        // counts the longest chain (an FFT of `8 log2 n_fft` operations, the filterbank sum over the bins, the window,
        // and the mean over the frames and bins), and the quotient divides the two evaluations' error by `2 eps`.
        let eps = 5e-3f32;
        let (bins, frames) = (scales.iter().map(|c| c.n_fft / 2 + 1).max().unwrap(), 64 / scales[0].hop + 1);
        let k = 8 * 5 + bins + 16 + frames * 8;
        let level: f64 = scales.iter().map(|c| log_mel(&pred, c).0.iter().map(|v| f64::from(v.abs())).sum::<f64>() / (frames * c.n_mels) as f64).sum::<f64>() / scales.len() as f64;
        let abs_tol = (2.0 * agreement_bound(k, level) / f64::from(eps)) as f32;
        let mut rng = Lcg::new(data::rng::random_seed());
        let check = Directional { eps, abs_tol, rel_tol: eps * eps };
        let (a, fd) = check
            .check(&grad, &pred, || rng.vec(64).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect(), |p| mel_l1_f64(p, &target, &scales))
            .expect("the analytic gradient is the loss's derivative");
        eprintln!("multiscale mel gradient: analytic {a:.6e} vs central difference {fd:.6e} (tolerance {abs_tol:.2e})");
    }

    /// Spec: with `mel_weight > 0` the trainer's weight gradients include the mel term, i.e. they are the gradient of
    /// `loss()`. Directional central differences on the decoder weights, assignment frozen.
    #[test]
    fn the_trainer_gradient_includes_the_mel_term() {
        let ae = ConvReconAE::new(4, 3, 0.25, 42).with_objective(toy_objective(2.0, None));
        let x = toy_wav(64);
        let (_, _, _, idx) = ae.forward(&x, None);
        let (_, dwd) = ae.grads(&x, &idx);
        let eps = 1e-2f32;
        // The loss is O(10) and its f32 evaluation spans a few hundred rounded operations.
        let abs_tol = (2.0 * agreement_bound(400, 20.0) / f64::from(eps)) as f32;
        let mut rng = Lcg::new(data::rng::random_seed());
        let check = Directional { eps, abs_tol, rel_tol: eps * eps };
        let probe = |w: &[f32]| {
            let mut m = ae.clone_weights();
            m.wd.copy_from_slice(w);
            f64::from(m.loss(&x, Some(&idx)))
        };
        let (a, fd) = check.check(&dwd, &ae.wd, || rng.vec(dwd.len()).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect(), probe).expect("trainer gradient includes the mel term");
        eprintln!("trainer mel gradient: analytic {a:.6e} vs central difference {fd:.6e} (tolerance {abs_tol:.2e})");
        // And the mel term really contributes: the same weights with the term off have a different gradient.
        let off = ConvReconAE::new(4, 3, 0.25, 42).with_objective(toy_objective(0.0, None));
        let (_, dwd_off) = off.grads(&x, &idx);
        assert!(dwd.iter().zip(&dwd_off).any(|(a, b)| (a - b).abs() > 1e-3), "mel_weight had no effect on the gradient");
    }

    fn toy_gan(start_step: u32) -> GanConfig {
        GanConfig { disc: tiny_config(), adv_weight: 0.5, feat_weight: 1.0, start_step }
    }

    /// Spec: with every term on, every loss is finite, and until `start_step` the discriminators neither train nor
    /// contribute.
    #[test]
    fn the_adversarial_terms_start_at_start_step() {
        let mut ae = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, Some(toy_gan(2))));
        let x = toy_wav(64);
        let reports: Vec<StepLosses> = (0..4).map(|_| ae.train_step_report(&x, 0.01)).collect();
        for (i, r) in reports.iter().enumerate() {
            let all = [r.wave, r.commit, r.mel, r.adv, r.feat, r.disc, r.generator];
            assert!(all.iter().all(|v| v.is_finite()), "step {i}: {r:?}");
            let on = i >= 2;
            assert_eq!(r.disc != 0.0 && r.adv != 0.0 && r.feat != 0.0, on, "step {i}: adversarial terms {} at start_step 2: {r:?}", if on { "missing" } else { "present" });
        }
    }

    /// Spec: with the generator frozen (rate 0) on a fixed batch, only the discriminators train and their loss falls.
    #[test]
    fn the_discriminator_loss_falls_with_the_generator_frozen() {
        let mut ae = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, Some(toy_gan(0))));
        let x = toy_wav(64);
        let losses: Vec<f32> = (0..20).map(|_| ae.train_step_report(&x, 0.0).disc).collect();
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(last < 0.7 * first, "discriminator loss went {first} -> {last}: {losses:?}");
    }

    /// Spec: with every term on, the generator objective falls over a short overfit run on a fixed batch. The
    /// discriminators train alongside, so the adversarial terms are a moving target; the run is judged on the first
    /// steps against the last ones.
    #[test]
    fn the_full_objective_falls_over_an_overfit_run() {
        let mut ae = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, Some(toy_gan(0))));
        let x = toy_wav(64);
        let totals: Vec<f32> = (0..60).map(|_| ae.train_step_report(&x, 0.05).generator).collect();
        assert!(totals.iter().all(|v| v.is_finite()));
        let mean = |w: &[f32]| w.iter().sum::<f32>() / w.len() as f32;
        let (head, tail) = (mean(&totals[..5]), mean(&totals[totals.len() - 5..]));
        eprintln!("full objective: first 5 steps {head:.4} -> last 5 steps {tail:.4}");
        assert!(tail < 0.85 * head, "the generator objective did not fall: {head:.3} -> {tail:.3}");
    }

    /// Spec: the discriminators' gradient reaches the generator. One step from identical weights moves them
    /// differently with the adversarial and feature-matching terms on than with them off.
    #[test]
    fn the_adversarial_gradient_reaches_the_generator() {
        let x = toy_wav(64);
        let mut with = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, Some(toy_gan(0))));
        let mut without = ConvReconAE::new(6, 3, 0.1, 7).with_objective(toy_objective(1.0, None));
        with.train_step_report(&x, 0.05);
        without.train_step_report(&x, 0.05);
        let moved = with.we.iter().chain(&with.wd).zip(without.we.iter().chain(&without.wd)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-6, "the adversarial terms left the generator update unchanged ({moved:e})");
    }
}
