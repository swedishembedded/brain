// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements adversarial training of image generators for its clients. If your team needs
// expertise in stable GAN losses and discriminator design for image tokenizers and restoration, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The PatchGAN discriminator VQGAN trains against (`NLayerDiscriminator` of taming-transformers, which LlamaGen and
//! CodeFormer's VQGAN stage copy), its hinge loss, and its own AdamW.
//!
//! # Architecture
//!
//! `n_layers = 3`, `ndf = 64` (taming's defaults) is
//!
//! ```text
//! Conv(in, ndf, 4, stride 2, pad 1, bias)                     LeakyReLU(0.2)
//! Conv(ndf*m_{n-1}, ndf*m_n, 4, stride 2, pad 1)  BatchNorm2d  LeakyReLU(0.2)    n = 1 .. n_layers-1
//! Conv(ndf*m_{n-1}, ndf*m_n, 4, stride 1, pad 1)  BatchNorm2d  LeakyReLU(0.2)    n = n_layers
//! Conv(ndf*m_n, 1, 4, stride 1, pad 1, bias)                                    one logit per patch
//! ```
//!
//! with `m_n = min(2^n, 8)`. The strided, normalised convolutions number `n_layers - 1` (taming's loop runs
//! `range(1, n_layers)`); the stride-1 one after them is the `n_layers`-th. The normalisation is **BatchNorm**
//! (taming's default `norm_layer = nn.BatchNorm2d`; its `ActNorm` alternative is a flag LlamaGen leaves off), so its
//! convolutions carry no bias. The output is a map of logits, one per receptive-field patch, not a scalar.
//!
//! It is built from [`vision::Conv`] (conv, batch-statistics BatchNorm, [`vision::Act::LeakyRelu`]) and nothing of its
//! own: forward, the input gradient and every weight gradient are those blocks'. The weights are initialised as
//! taming's `weights_init`: convolutions `N(0, 0.02)`, BatchNorm scale `N(1, 0.02)` and shift 0 (a convolution's bias
//! keeps torch's default, uniform in `+-1/sqrt(fan_in)`).
//!
//! # Where it lives
//!
//! In this crate, not a crate of its own: the VQGAN trainer is its only consumer. A second image trainer that needs a
//! patch discriminator (a restoration net's adversarial stage) hoists it then; it depends on `vision` alone, so the
//! move is a file. The temporal STFT discriminator in `minimaxmusic3` is a different network over a different domain.
//!
//! # The two branches
//!
//! The reference calls the discriminator twice per step, on the real batch and on the reconstruction, each with its
//! own batch statistics, and the generator's adversarial term is the second call differentiated to the image. Here
//! that is two [`Branch`]es of the same graph sharing one [`ParamStore`]: the real branch's gradient reaches only the
//! weights, the fake branch's also reaches the image. Both are staged from the caller's device (the discriminator
//! runs on a handle of its own: `Gpu::new_like`), so a caller hands over device buffers and reads one back.
//!
//! # The losses
//!
//! Hinge, as taming's `hinge_d_loss`: `0.5 * (mean(relu(1 - D(x))) + mean(relu(1 + D(x_hat))))`, and the generator's
//! `-mean(D(x_hat))`. They are two-line reductions over the logit map (a few thousand values), computed on the host
//! from one readback of it: [`hinge_d_loss`], [`generator_loss`].

use data::rng::Rng;
use gpu_core::{DeviceBuffer, Gpu};
use optim::{Adam, Optim};
use paramstore::{ParamStore, Role};
use vision::{Act, Conv, ConvKernelIds, ConvSpec, Ctx, Norm, Shape};

/// Every kernel the discriminator dispatches (forward, both backwards, batch-statistics BatchNorm) and its AdamW.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_pack", kernels::BN_PACK),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dparams", kernels::BN_DPARAMS),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// The discriminator's shape: taming's `NLayerDiscriminator(input_nc, ndf, n_layers)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchGanConfig {
    pub in_channels: u32,
    /// Width of the first convolution; the widest is `8 * ndf`.
    pub ndf: u32,
    /// Depth, as taming counts it (at least 1): `n_layers - 1` strided normalised convolutions follow the first.
    pub n_layers: u32,
}

impl PatchGanConfig {
    /// taming's `NLayerDiscriminator` defaults, which LlamaGen's `PatchGANDiscriminator` keeps.
    pub fn taming() -> PatchGanConfig {
        PatchGanConfig { in_channels: 3, ndf: 64, n_layers: 3 }
    }

    /// One unit per convolution, in order.
    fn specs(&self) -> Vec<ConvSpec> {
        assert!(self.n_layers >= 1 && self.ndf >= 1 && self.in_channels >= 1, "patchgan: {self:?} is not a discriminator");
        let width = |n: u32| self.ndf * (1u32 << n.min(3));
        let unit = |cout: u32, stride: u32, norm: Norm, bias: bool| ConvSpec { cout, k: 4, stride, pad: 1, groups: 1, dilation: 1, norm, act: Act::LeakyRelu, bias };
        let mut specs = vec![unit(self.ndf, 2, Norm::None, true)];
        for n in 1..self.n_layers {
            specs.push(unit(width(n), 2, Norm::Bn, false));
        }
        specs.push(unit(width(self.n_layers), 1, Norm::Bn, false));
        specs.push(ConvSpec { act: Act::None, ..unit(1, 1, Norm::None, true) });
        specs
    }

    /// The logit map for an `h x w` input.
    pub fn logit_shape(&self, h: u32, w: u32) -> Shape {
        self.specs().iter().fold(Shape::new(1, self.in_channels, h, w), |s, spec| spec.out_shape(s))
    }
}

/// One forward-backward chain of the discriminator over a staged input.
struct Branch {
    convs: Vec<Conv>,
    input: DeviceBuffer,
    /// The gradient wrt each convolution's input (`d_in[0]` is the image's).
    d_in: Vec<DeviceBuffer>,
    d_logits: DeviceBuffer,
}

impl Branch {
    fn new(ctx: &Ctx, cfg: &PatchGanConfig, h: u32, w: u32) -> Branch {
        let mut shape = Shape::new(1, cfg.in_channels, h, w);
        let input = ctx.act(shape.numel());
        let (mut convs, mut d_in) = (Vec::new(), Vec::new());
        for (i, spec) in cfg.specs().into_iter().enumerate() {
            d_in.push(ctx.act(shape.numel()));
            let conv = Conv::with_spec(ctx, &format!("disc.{i}"), shape, spec, true);
            shape = conv.out_shape;
            convs.push(conv);
        }
        Branch { convs, input, d_in, d_logits: ctx.act(shape.numel()) }
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore) {
        let mut x = &self.input;
        for conv in &self.convs {
            conv.forward(ctx, ps, x);
            x = conv.out();
        }
    }

    fn logits(&self) -> &DeviceBuffer {
        self.convs.last().expect("a discriminator has convolutions").out()
    }

    /// Reverse from `d_logits` (already written) to the weights, and to the image when `to_input`.
    fn backward(&self, ctx: &Ctx, ps: &ParamStore, to_input: bool) {
        let n = self.convs.len();
        let mut d_out = &self.d_logits;
        for i in (0..n).rev() {
            let x_in = if i == 0 { &self.input } else { self.convs[i - 1].out() };
            if i == 0 && !to_input {
                self.convs[0].backward_params(ctx, ps, x_in, d_out);
            } else {
                self.convs[i].backward(ctx, ps, x_in, d_out, &self.d_in[i]);
                d_out = &self.d_in[i];
            }
        }
    }
}

/// The discriminator with its parameters, its optimiser and its real and fake branches, for one image size.
pub struct PatchGan {
    gpu: Gpu,
    ids: ConvKernelIds,
    cfg: PatchGanConfig,
    ps: ParamStore,
    opt: Optim,
    real: Branch,
    fake: Branch,
    h: u32,
    w: u32,
}

impl PatchGan {
    /// A discriminator for `[in_channels, h, w]` images on `gpu`, which must have been built with (or `new_like`'d to)
    /// [`PIPELINES`], weights drawn from `seed` (taming's `weights_init`).
    pub fn new(gpu: Gpu, cfg: PatchGanConfig, h: u32, w: u32, seed: u64) -> PatchGan {
        let ids = ConvKernelIds::resolve(PIPELINES);
        let ctx = Ctx::new(&gpu, &ids);
        let (real, fake) = (Branch::new(&ctx, &cfg, h, w), Branch::new(&ctx, &cfg, h, w));
        // Both branches name the same tensors; the store is built from one.
        let mut roles = Vec::new();
        let mut init = std::collections::HashMap::new();
        let mut rng = Rng::new(seed);
        for conv in &real.convs {
            let spec = conv.spec;
            let fan_in = (conv.in_shape.c * spec.k * spec.k) as f32;
            for (name, numel) in conv.param_list() {
                let role = if name.ends_with(".run_mean") || name.ends_with(".run_var") { Role::Frozen } else { Role::Trainable };
                let data: Vec<f32> = if name.ends_with(".conv.weight") {
                    (0..numel).map(|_| (0.02 * rng.next_gaussian()) as f32).collect()
                } else if name.ends_with(".conv.bias") {
                    let bound = 1.0 / fan_in.sqrt();
                    (0..numel).map(|_| (rng.uniform(-1.0, 1.0) as f32) * bound).collect()
                } else if name.ends_with(".bn.gamma") {
                    (0..numel).map(|_| (1.0 + 0.02 * rng.next_gaussian()) as f32).collect()
                } else if name.ends_with(".bn.run_var") {
                    vec![1.0; numel]
                } else {
                    vec![0.0; numel]
                };
                init.insert(name.clone(), data);
                roles.push((name, numel, role));
            }
        }
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let kernel = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("patchgan: kernel `{name}` is not registered on this device; build it with patchgan::PIPELINES"));
        let opt = Optim::new(kernel("adamw"), kernel("gradnorm_sq"), kernel("grad_scale"), kernel("clip_coef"), kernel("grad_scale_buf"));
        PatchGan { gpu, ids, cfg, ps, opt, real, fake, h, w }
    }

    pub fn config(&self) -> &PatchGanConfig {
        &self.cfg
    }

    /// The logit map's element count.
    pub fn n_logits(&self) -> usize {
        self.cfg.logit_shape(self.h, self.w).numel() as usize
    }

    /// Every trainable tensor, `(name, element count)`.
    pub fn param_names(&self) -> Vec<(String, usize)> {
        self.ps.trained_params().cloned().collect()
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.ps.numel(name), "patchgan: {name} has {} values, got {}", self.ps.numel(name), data.len());
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    fn branch(&self, fake: bool) -> &Branch {
        if fake {
            &self.fake
        } else {
            &self.real
        }
    }

    /// Stage an image (`[in_channels, h, w]` on `src_gpu`, a buffer of that handle) as the real or fake branch's input.
    pub fn stage(&self, fake: bool, src_gpu: &Gpu, src: &DeviceBuffer) {
        let b = self.branch(fake);
        let bytes = 4 * (self.cfg.in_channels * self.h * self.w) as u64;
        src_gpu.copy_to(src, 0, &self.gpu, &b.input, 0, bytes);
    }

    /// Stage an image from the host.
    pub fn stage_host(&self, fake: bool, image: &[f32]) {
        assert_eq!(image.len(), (self.cfg.in_channels * self.h * self.w) as usize, "patchgan: image has {} values", image.len());
        self.gpu.write(&self.branch(fake).input, bytemuck::cast_slice(image));
    }

    /// Run a branch's forward and read its logit map (`[1, 1, ho, wo]` row-major).
    pub fn forward(&self, fake: bool) -> Vec<f32> {
        let b = self.branch(fake);
        b.forward(&Ctx::new(&self.gpu, &self.ids), &self.ps);
        self.gpu.read(b.logits(), self.n_logits())
    }

    /// Reverse a branch from `d_logits` (the loss's gradient wrt each logit), accumulating every weight gradient and,
    /// with `to_input`, leaving the gradient wrt the staged image in [`PatchGan::input_grad`]. Needs the branch's
    /// forward, and gradients zeroed by [`PatchGan::zero_grads`] beforehand.
    pub fn backward(&self, fake: bool, d_logits: &[f32], to_input: bool) {
        assert_eq!(d_logits.len(), self.n_logits(), "patchgan: {} logit gradients for {} logits", d_logits.len(), self.n_logits());
        let b = self.branch(fake);
        self.gpu.write(&b.d_logits, bytemuck::cast_slice(d_logits));
        b.backward(&Ctx::new(&self.gpu, &self.ids), &self.ps, to_input);
    }

    /// The gradient wrt the image the fake branch was staged with, from the last `backward(true, _, true)`, on this
    /// discriminator's device.
    pub fn input_grad(&self) -> &DeviceBuffer {
        &self.fake.d_in[0]
    }

    /// Zero every weight gradient.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// One AdamW step (`t` counts from 1) on the weights, from the gradients the backwards accumulated.
    pub fn adamw_step(&self, t: u32, lr: f32, weight_decay: f32, adam: Adam, clip: Option<f32>) {
        self.opt.step(&self.gpu, &self.ps, t, lr, weight_decay, adam, clip, 1.0);
    }

    /// The device the discriminator runs on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

/// The hinge discriminator loss and its gradient wrt each logit.
#[derive(Clone, Debug, PartialEq)]
pub struct HingeLoss {
    pub loss: f64,
    pub d_real: Vec<f32>,
    pub d_fake: Vec<f32>,
}

/// `0.5 * (mean(relu(1 - real)) + mean(relu(1 + fake)))` (taming's `hinge_d_loss`) over the two logit maps, and its
/// gradient: `-0.5 / n` on a real logit below 1, `+0.5 / n` on a fake logit above -1, zero elsewhere (the kink itself
/// takes the zero, torch's `relu` subgradient).
pub fn hinge_d_loss(real: &[f32], fake: &[f32]) -> HingeLoss {
    let (nr, nf) = (real.len() as f64, fake.len() as f64);
    let loss_real: f64 = real.iter().map(|v| (1.0 - *v as f64).max(0.0)).sum::<f64>() / nr;
    let loss_fake: f64 = fake.iter().map(|v| (1.0 + *v as f64).max(0.0)).sum::<f64>() / nf;
    HingeLoss {
        loss: 0.5 * (loss_real + loss_fake),
        d_real: real.iter().map(|v| if *v < 1.0 { (-0.5 / nr) as f32 } else { 0.0 }).collect(),
        d_fake: fake.iter().map(|v| if *v > -1.0 { (0.5 / nf) as f32 } else { 0.0 }).collect(),
    }
}

/// The generator's adversarial loss `-mean(D(x_hat))` and its gradient wrt each logit, `-1 / n`.
pub fn generator_loss(fake: &[f32]) -> (f64, Vec<f32>) {
    let n = fake.len() as f64;
    (-fake.iter().map(|v| *v as f64).sum::<f64>() / n, vec![(-1.0 / n) as f32; fake.len()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::rounding::agreement_bound;
    use std::collections::HashMap;

    const SIDE: u32 = 32;

    /// A discriminator small enough for an element-wise sweep: 2 layers, 4 filters, on 32x32 (logit map 6x6).
    fn tiny_config() -> PatchGanConfig {
        PatchGanConfig { in_channels: 3, ndf: 4, n_layers: 2 }
    }

    fn tiny() -> PatchGan {
        PatchGan::new(gpu_core::testgpu::dev(PIPELINES), tiny_config(), SIDE, SIDE, data::rng::random_seed())
    }

    fn image(seed: u64) -> Vec<f32> {
        data::rng::Lcg::new(seed).vec((3 * SIDE * SIDE) as usize)
    }

    /// A loss that is smooth in the logits and weights every one differently: `sum_j r_j logit_j`.
    fn logit_weights(n: usize) -> Vec<f32> {
        data::rng::Lcg::new(0x5eed).vec(n)
    }

    // ---- the f64 oracle: the same network written out, sharing no code with `vision` ------------------------------

    /// `x` (`[cin, h, w]`) convolved 4x4 with `w` (`[cout, cin, 4, 4]`), stride and pad 1, plus an optional bias.
    fn conv64(x: &[f64], (cin, h, wd): (usize, usize, usize), w: &[f64], bias: Option<&[f64]>, cout: usize, stride: usize) -> (Vec<f64>, (usize, usize)) {
        let (ho, wo) = ((h + 2 - 4) / stride + 1, (wd + 2 - 4) / stride + 1);
        let mut y = vec![0.0; cout * ho * wo];
        for o in 0..cout {
            for oy in 0..ho {
                for ox in 0..wo {
                    let mut acc = bias.map_or(0.0, |b| b[o]);
                    for c in 0..cin {
                        for ky in 0..4 {
                            for kx in 0..4 {
                                let (iy, ix) = ((oy * stride + ky) as isize - 1, (ox * stride + kx) as isize - 1);
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < wd {
                                    acc += w[((o * cin + c) * 4 + ky) * 4 + kx] * x[(c * h + iy as usize) * wd + ix as usize];
                                }
                            }
                        }
                    }
                    y[(o * ho + oy) * wo + ox] = acc;
                }
            }
        }
        (y, (ho, wo))
    }

    /// The logits of `x`, batch-statistics BatchNorm (biased variance, eps 1e-5), LeakyReLU 0.2.
    fn logits64(cfg: &PatchGanConfig, p: &HashMap<String, Vec<f64>>, x: &[f64]) -> Vec<f64> {
        let (mut cur, mut dims, mut cin) = (x.to_vec(), (SIDE as usize, SIDE as usize), cfg.in_channels as usize);
        for (i, spec) in cfg.specs().iter().enumerate() {
            let bias = spec.bias.then(|| &p[&format!("disc.{i}.conv.bias")][..]);
            let (mut y, d) = conv64(&cur, (cin, dims.0, dims.1), &p[&format!("disc.{i}.conv.weight")], bias, spec.cout as usize, spec.stride as usize);
            let hw = d.0 * d.1;
            if spec.norm == Norm::Bn {
                let (gamma, beta) = (&p[&format!("disc.{i}.bn.gamma")], &p[&format!("disc.{i}.bn.beta")]);
                for c in 0..spec.cout as usize {
                    let ch = &mut y[c * hw..(c + 1) * hw];
                    let mean = ch.iter().sum::<f64>() / hw as f64;
                    let var = ch.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / hw as f64;
                    ch.iter_mut().for_each(|v| *v = (*v - mean) / (var + 1e-5).sqrt() * gamma[c] + beta[c]);
                }
            }
            if spec.act == Act::LeakyRelu {
                y.iter_mut().for_each(|v| *v = if *v >= 0.0 { *v } else { 0.2 * *v });
            }
            (cur, dims, cin) = (y, d, spec.cout as usize);
        }
        cur
    }

    /// Terms on the longest chain the backward walks to a first-layer weight: every convolution's `cin k^2`
    /// contraction, the per-position sum of the weight gradient and of the BatchNorm statistics.
    fn chain_terms(cfg: &PatchGanConfig) -> usize {
        let (mut terms, mut cin) = (0, cfg.in_channels as usize);
        for spec in cfg.specs() {
            terms += cin * 16 + (SIDE * SIDE / 4) as usize;
            cin = spec.cout as usize;
        }
        terms
    }

    fn params64(d: &PatchGan) -> HashMap<String, Vec<f64>> {
        d.param_names().into_iter().map(|(n, _)| (n.clone(), d.read_weight(&n).into_iter().map(f64::from).collect())).collect()
    }

    /// Element-wise central differences of `loss` in f64 against `analytic`, tolerance the round-off of the backward
    /// chain at the largest gradient entry (a defect moves entries by a fraction of their size, round-off by `gamma_k`
    /// of it); entries whose step crosses a LeakyReLU kink are drawn out by their disagreeing one-sided differences.
    fn check_elementwise(what: &str, analytic: &[f32], k: usize, mut loss: impl FnMut(usize, f64) -> f64) {
        let scale = analytic.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
        let tol = agreement_bound(k, scale);
        assert!(scale > 1e-6, "{what}: an all-zero gradient proves nothing");
        assert!(tol < 0.01 * scale, "{what}: tolerance {tol:e} would hide a one percent error on the largest entry {scale:e}");
        let eps = 1e-6;
        let mut kinks = 0;
        for (j, g) in analytic.iter().enumerate() {
            let (l0, lh, ll) = (loss(j, 0.0), loss(j, eps), loss(j, -eps));
            let (ahead, behind) = ((lh - l0) / eps, (l0 - ll) / eps);
            if (ahead - behind).abs() > 1e-3 * scale {
                kinks += 1;
                continue;
            }
            let fd = (lh - ll) / (2.0 * eps);
            assert!((*g as f64 - fd).abs() <= tol, "{what}[{j}]: analytic {g:e} vs central difference {fd:e} (tolerance {tol:e}, largest entry {scale:e})");
        }
        assert!(kinks * 100 <= analytic.len().max(100), "{what}: {kinks} of {} entries cross a kink", analytic.len());
    }

    #[test]
    fn the_logit_map_follows_taming_s_geometry() {
        // 256 -> 128 -> 64 -> 32 (two strided normalised convs at n_layers 3) -> 31 -> 30.
        let s = PatchGanConfig::taming().logit_shape(256, 256);
        assert_eq!((s.c, s.h, s.w), (1, 30, 30));
        let specs = PatchGanConfig::taming().specs();
        assert_eq!(specs.iter().map(|c| c.cout).collect::<Vec<_>>(), [64, 128, 256, 512, 1]);
        assert_eq!(specs.iter().map(|c| c.stride).collect::<Vec<_>>(), [2, 2, 2, 1, 1]);
        assert_eq!(specs.iter().map(|c| c.bias).collect::<Vec<_>>(), [true, false, false, false, true], "BatchNorm convs have no bias");
        assert_eq!(specs.iter().map(|c| c.norm).collect::<Vec<_>>(), [Norm::None, Norm::Bn, Norm::Bn, Norm::Bn, Norm::None]);
    }

    #[test]
    fn hinge_matches_the_definition_and_its_subgradient() {
        let (real, fake) = ([2.0f32, 0.5, -1.0, 1.0], [-3.0f32, 0.0, 0.5, -1.0]);
        let h = hinge_d_loss(&real, &fake);
        // real: relu(1 - x) = 0, 0.5, 2, 0 -> mean 0.625; fake: relu(1 + x) = 0, 1, 1.5, 0 -> mean 0.625.
        assert!((h.loss - 0.5 * (0.625 + 0.625)).abs() < 1e-12, "{}", h.loss);
        // 0.5 / 4 = 0.125 on the active logits only; the kinks (real = 1, fake = -1) are inactive.
        assert_eq!(h.d_real, [0.0, -0.125, -0.125, 0.0]);
        assert_eq!(h.d_fake, [0.0, 0.125, 0.125, 0.0]);
        // -mean(fake) = -(-3 + 0 + 0.5 - 1) / 4 = 0.875, each logit's gradient -1/4.
        let (g, d) = generator_loss(&fake);
        assert!((g - 0.875).abs() < 1e-12 && d == [-0.25; 4], "{g} {d:?}");
    }

    /// The device forward is the f64 oracle's, which pins the oracle the gradient checks below stand on.
    #[test]
    fn the_forward_matches_the_f64_oracle() {
        let d = tiny();
        let x = image(8);
        d.stage_host(true, &x);
        let got = d.forward(true);
        let want = logits64(&tiny_config(), &params64(&d), &x.iter().map(|v| *v as f64).collect::<Vec<_>>());
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tol = agreement_bound(chain_terms(&tiny_config()), scale);
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((*g as f64 - w).abs() <= tol, "logit {j}: {g} vs {w} (tolerance {tol:e})");
        }
    }

    /// Every weight's gradient (convolutions, biases, BatchNorm scale and shift) on both branches, element by element.
    #[test]
    fn the_weight_gradients_match_central_differences() {
        let d = tiny();
        let r = logit_weights(d.n_logits());
        let r64: Vec<f64> = r.iter().map(|v| *v as f64).collect();
        for fake in [false, true] {
            let x = image(1 + fake as u64);
            d.stage_host(fake, &x);
            d.zero_grads();
            d.forward(fake);
            d.backward(fake, &r, false);
            let x64: Vec<f64> = x.iter().map(|v| *v as f64).collect();
            let base = params64(&d);
            for (name, _) in d.param_names() {
                let analytic = d.read_grad(&name);
                let loss = |j: usize, delta: f64| {
                    let mut p = base.clone();
                    p.get_mut(&name).unwrap()[j] += delta;
                    logits64(&tiny_config(), &p, &x64).iter().zip(&r64).map(|(l, r)| l * r).sum::<f64>()
                };
                check_elementwise(&format!("{name} ({} branch)", if fake { "fake" } else { "real" }), &analytic, chain_terms(&tiny_config()), loss);
            }
        }
    }

    /// The gradient wrt the image the generator's adversarial term differentiates, element by element.
    #[test]
    fn the_input_gradient_matches_central_differences() {
        let d = tiny();
        let x = image(3);
        d.stage_host(true, &x);
        let r = logit_weights(d.n_logits());
        d.zero_grads();
        d.forward(true);
        d.backward(true, &r, true);
        let analytic = d.gpu().read(d.input_grad(), x.len());
        let (p, x64, r64) = (params64(&d), x.iter().map(|v| *v as f64).collect::<Vec<_>>(), r.iter().map(|v| *v as f64).collect::<Vec<_>>());
        let loss = |j: usize, delta: f64| {
            let mut x = x64.clone();
            x[j] += delta;
            logits64(&tiny_config(), &p, &x).iter().zip(&r64).map(|(l, r)| l * r).sum::<f64>()
        };
        check_elementwise("input", &analytic, chain_terms(&tiny_config()), loss);
    }

    /// Spec: trained alone on a fixed real batch and a fixed fake batch, the discriminator's hinge loss falls.
    #[test]
    fn the_hinge_loss_falls_when_only_the_discriminator_trains() {
        let d = tiny();
        d.stage_host(false, &image(4));
        d.stage_host(true, &image(5));
        let mut losses = Vec::new();
        for t in 1..=30 {
            let (real, fake) = (d.forward(false), d.forward(true));
            let h = hinge_d_loss(&real, &fake);
            losses.push(h.loss);
            d.zero_grads();
            d.backward(false, &h.d_real, false);
            d.backward(true, &h.d_fake, false);
            d.adamw_step(t, 5e-3, 0.0, Adam::default(), None);
        }
        assert!(losses.iter().all(|l| l.is_finite()));
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(last < 0.7 * first, "the hinge loss went {first} -> {last}: {losses:?}");
    }
}
