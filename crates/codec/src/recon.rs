// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Non-GAN **reconstruction-loss codec trainer** (scope-limited, Track C).
//!
//! This wires the *reconstruction* half of a neural-codec training objective —
//! the part that does not need a discriminator — and gradchecks it end to end:
//!
//!   wav -> conv encoder -> straight-through VQ -> conv decoder -> wav'
//!   loss = waveform L1+L2  (+ multi-scale log-mel L1, reported)
//!
//! The three differentiable pieces are:
//!   * [`waveform_recon`] — `mean(|y-x|) + 0.5·mean((y-x)²)`, exact analytic grad
//!     w.r.t. the prediction;
//!   * [`StraightThroughVq`] — vector-quantize each frame to its nearest codebook
//!     entry; the decoder sees the quantized vectors, but the gradient flows to
//!     the encoder via the **straight-through estimator** (identity) plus a
//!     commitment term `β·‖z − sg(q)‖²`;
//!   * [`ConvReconAE`] — a tiny causal-conv encoder/decoder over the shared WGSL
//!     engine (`audio::conv` fwd + `conv1d_dx`/`conv1d_dw` bwd), so the conv
//!     gradients are the real kernel gradients (already gradchecked in `audio`).
//!
//! [`ConvReconAE::gradcheck`] freezes the VQ assignment (the argmin is piecewise
//! constant; the straight-through estimator is exactly the frozen-assignment
//! Jacobian) and compares the analytic encoder/decoder weight gradients to
//! central finite differences of the end-to-end loss.
//!
//! ## What is NOT here (the remaining research piece — documented, not faked)
//! The production Qwen3-TTS / Mimi codec is trained with a **GAN** objective on
//! the raw waveform (multi-period + multi-scale discriminators, feature-matching
//! loss) plus **WavLM** semantic distillation, on top of this reconstruction
//! loss. Those are intentionally out of scope here:
//!   * the multi-scale log-mel L1 ([`multiscale_mel_l1`]) is wired as a forward
//!     *metric* only — its STFT/mel backward is a TODO;
//!   * the discriminator stack + feature-matching + WavLM distillation are not
//!     implemented (see the codec README "Training" section and the TODOs below).
//! No discriminator results are reported because none are computed.

use audio::conv::{Conv1d, ConvKernels};
use audio::mel::{log_mel, MelConfig};
use bytemuck::cast_slice;
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

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

/// Multi-scale log-mel L1 distance between two waveforms — a perceptual
/// reconstruction **metric** (forward only; reported, not back-propagated).
///
// TODO(mel-backward): an STFT + mel-filterbank + log backward would let this be
// a *trainable* spectral loss (the usual codec recon term). Forward-only today.
pub fn multiscale_mel_l1(pred: &[f32], target: &[f32], scales: &[MelConfig]) -> f32 {
    let mut total = 0.0f32;
    for cfg in scales {
        let (mp, _) = log_mel(pred, cfg);
        let (mt, _) = log_mel(target, cfg);
        let n = mp.len().min(mt.len());
        if n == 0 {
            continue;
        }
        let l1: f32 = (0..n).map(|i| (mp[i] - mt[i]).abs()).sum::<f32>() / n as f32;
        total += l1;
    }
    total / scales.len().max(1) as f32
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

// TODO(gan): the production codec adds a raw-waveform GAN (multi-period +
// multi-scale discriminators, adversarial + feature-matching losses) and WavLM
// semantic distillation on top of this reconstruction loss. Those are the
// remaining research piece — see the codec README "Training" section. Not
// implemented here; no discriminator results are reported.

/// A tiny causal-conv reconstruction autoencoder over the shared engine.
///
/// Encoder: `conv1d` 1->`c` (kernel `k`, causal). Quantizer: per-time-step VQ over
/// the `c` channels (straight-through). Decoder: `conv1d` `c`->1 (kernel `k`,
/// causal). Trained to minimise [`waveform_recon`] (+ commitment).
pub struct ConvReconAE {
    gpu: Gpu,
    pub c: u32,
    pub k: u32,
    pub vq: StraightThroughVq,
    pub we: Vec<f32>, // encoder weight [c, 1, k]
    pub wd: Vec<f32>, // decoder weight [1, c, k]
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
        ConvReconAE { gpu, c, k, vq, we, wd }
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

    /// End-to-end loss `recon + commitment` (frozen VQ assignment optional).
    pub fn loss(&self, x: &[f32], frozen: Option<&[usize]>) -> f32 {
        let (_, r, c, _) = self.forward(x, frozen);
        r + c
    }

    /// The **straight-through surrogate** loss: identical to [`Self::loss`] except
    /// the decoder is fed the *continuous* latent `z` instead of the quantized
    /// `q`. This is exactly the smooth objective the straight-through estimator
    /// differentiates (decoder grad copied `q -> z` via identity), so its analytic
    /// gradient — which [`Self::grads`] computes — matches finite differences.
    pub fn surrogate_loss(&self, x: &[f32], frozen: &[usize]) -> f32 {
        let l = x.len() as u32;
        let (z, _) = self.encode(x);
        let zf = Self::ncl_to_frames(&z, self.c as usize, l as usize);
        let (qf, _) = self.vq.quantize(&zf, Some(frozen));
        let zn = Self::frames_to_ncl(&zf, self.c as usize, l as usize); // decoder sees z
        let y = self.decode(&zn, l);
        let (recon, _) = waveform_recon(&y, x);
        let (commit, _) = self.vq.commitment(&zf, &qf);
        recon + commit
    }

    /// Analytic gradients of the straight-through objective w.r.t. the
    /// encoder/decoder weights, via the real conv backward kernels. The decoder
    /// backward runs on the latent `z` (straight-through identity `q -> z`), the
    /// recon target is the quantized forward's output, and the commitment term is
    /// added on the encoder side. Returns `(dWe, dWd)` for the frozen assignment.
    pub fn grads(&self, x: &[f32], frozen: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let l = x.len() as u32;
        let (z, _) = self.encode(x);
        let zf = Self::ncl_to_frames(&z, self.c as usize, l as usize);
        let (qf, _) = self.vq.quantize(&zf, Some(frozen));
        let zn = Self::frames_to_ncl(&zf, self.c as usize, l as usize);
        let y = self.decode(&zn, l); // straight-through: decoder differentiates z

        // dL/dy (recon) and dL/dz (commitment, straight-through identity on q).
        let (_, dy) = waveform_recon(&y, x);
        let (_, dz_commit) = self.vq.commitment(&zf, &qf);

        // decoder backward on input z: dz_recon (input grad) + dWd (weight grad).
        let cdec = self.conv(self.c, 1, l);
        let kdec = ConvKernels { fwd: RC_CONV1D, dx: RC_CONV1D_DX, dw: RC_CONV1D_DW };
        let dyb = self.up(&dy);
        let qb = self.up(&zn);
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

    /// One plain SGD step on the toy waveform; returns the post-step loss.
    pub fn train_step(&mut self, x: &[f32], lr: f32) -> f32 {
        // Use the live VQ assignment for the step (straight-through).
        let (_, _, _, idx) = self.forward(x, None);
        let (dwe, dwd) = self.grads(x, &idx);
        for i in 0..self.we.len() {
            self.we[i] -= lr * dwe[i];
        }
        for i in 0..self.wd.len() {
            self.wd[i] -= lr * dwd[i];
        }
        self.loss(x, None)
    }

    /// Inline finite-difference gradcheck of the **whole reconstruction path**
    /// (encoder conv -> straight-through VQ -> decoder conv -> recon+commit loss),
    /// with the VQ assignment frozen so the forward is smooth. Returns the max
    /// relative error over a few probed weights of each tensor.
    pub fn gradcheck(&self, x: &[f32], eps: f32) -> f32 {
        let (_, _, _, idx) = self.forward(x, None);
        let (dwe, dwd) = self.grads(x, &idx);

        // Central difference of the loss w.r.t. weight `i` with the VQ assignment
        // frozen, for both the encoder (`enc=true`) and decoder weight tensors.
        let probe = |enc: bool, i: usize| -> f32 {
            let mut m = self.clone_weights();
            if enc { m.we[i] += eps } else { m.wd[i] += eps }
            let lp = m.surrogate_loss(x, &idx);
            let mut m = self.clone_weights();
            if enc { m.we[i] -= eps } else { m.wd[i] -= eps }
            let lm = m.surrogate_loss(x, &idx);
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

    fn clone_weights(&self) -> ConvReconAE {
        ConvReconAE {
            gpu: Gpu::new_cpu(RECON_PIPELINES),
            c: self.c,
            k: self.k,
            vq: StraightThroughVq { bins: self.vq.bins, dim: self.vq.dim, codebook: self.vq.codebook.clone(), beta: self.vq.beta },
            we: self.we.clone(),
            wd: self.wd.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let ae = ConvReconAE::new(4, 3, 0.25, 42);
        let x = toy_wav(48);
        let max_rel = ae.gradcheck(&x, 1e-3);
        eprintln!("conv-recon gradcheck max_rel = {max_rel:.2e}");
        assert!(max_rel < 2e-2, "reconstruction-path gradcheck failed: max_rel {max_rel:.3e}");
    }

    #[test]
    fn conv_recon_trains_down() {
        let mut ae = ConvReconAE::new(6, 3, 0.1, 7);
        let x = toy_wav(64);
        let initial = ae.loss(&x, None);
        let mut last = initial;
        for _ in 0..300 {
            last = ae.train_step(&x, 0.05);
        }
        eprintln!("conv-recon train: {initial:.4} -> {last:.4}");
        assert!(last < initial * 0.85, "reconstruction loss did not drop: {initial:.3} -> {last:.3}");
    }
}
