// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements adversarial training of neural audio codecs and vocoders for its clients. If your
// team needs expertise in waveform discriminators, stable GAN losses and feature matching then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The waveform discriminators a codec trains against (HiFi-GAN: multi-period and multi-scale), their LSGAN and
//! feature-matching losses, and their own AdamW.
//!
//! # The two families
//!
//! * **Multi-period (MPD).** For each period `p` in [`DiscConfig::periods`] the waveform `x[T]` is reshaped to
//!   `[T/p, p]` (`x2[t][c] = x[t p + c]`, the last `T mod p` samples cropped) and convolved with `(k, 1)` kernels, so
//!   the `p` columns are independent sequences of samples spaced `p` apart. A `(k, 1)` 2-D convolution is a 1-D
//!   convolution over the first axis shared by every column, so it runs on `conv1d` with the `p` columns as the batch.
//! * **Multi-scale (MSD).** The same stack of grouped 1-D convolutions on the waveform and on copies of it average-pooled
//!   by 2 once per further scale (`AvgPool1d(4, 2, pad 2)`, which is a fixed-weight strided `conv1d`).
//!
//! Each sub-discriminator is `[conv, bias, LeakyReLU] x n` followed by a `k = 3` convolution to one channel (no
//! activation): the logits. The hidden layers' outputs are its **feature maps**. Convolutions are zero-padded
//! `(k - 1) / 2` and carry torch's default initialisation, uniform in `+-1 / sqrt(fan_in)`; the reference's weight and
//! spectral normalisation are not reproduced (the checkpoint-free trainer has no use for them).
//!
//! The recipe the crate's trainer follows is HiFi-GAN's, waveform discriminators only: its widths are
//! [`DiscConfig::default`].
//!
//! # Losses
//!
//! LSGAN, as HiFi-GAN: `D = mean((D(x) - 1)^2) + mean(D(x_hat)^2)` and `G = mean((D(x_hat) - 1)^2)`, each averaged over
//! the `K` sub-discriminators. Least squares rather than hinge because it is smooth in the logits (no kink for a
//! finite-difference check to straddle) and it is the loss the recipe was published with. Feature matching is
//! `(1 / (K L)) sum_k sum_l mean(|f_l(x) - f_l(x_hat)|)`; the real features are constants.
//!
//! # What the backward keeps
//!
//! Only each layer's post-activation output (the feature map itself) and the input. LeakyReLU with a positive slope
//! has the sign of its input, so its backward reads the output (`y >= 0` iff `x >= 0`) and the pre-activation is not
//! kept. The discriminators are small and run on the CPU backend, so there is no layer stack worth recomputing.

use data::rng::Rng;
use model::saved::SavedPolicy;
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu};
use optim::{Adam, Optim};
use paramstore::{ParamStore, Role};

use audio::conv::{Conv1d, ConvKernels};

/// Every kernel the discriminators dispatch: the convolution and both gradients, the bias, LeakyReLU, the bias
/// gradient, the accumulations, and AdamW.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("bias_grad_ncl", kernels::BIAS_GRAD_NCL),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// One convolution of a multi-scale stack: `cout` filters of width `k` at `stride` over `groups` channel groups,
/// zero-padded `(k - 1) / 2` on each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvLayer {
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub groups: u32,
}

/// The discriminators' shape, optimiser and seed.
#[derive(Clone, Debug)]
pub struct DiscConfig {
    /// Periods of the multi-period discriminators (HiFi-GAN: 2, 3, 5, 7, 11). Empty: none.
    pub periods: Vec<u32>,
    /// Widths of a multi-period stack's hidden layers: `k = 5` convolutions, stride 3 except the last (stride 1).
    pub mpd_channels: Vec<u32>,
    /// Number of multi-scale discriminators: the waveform, then it average-pooled by 2 once per further scale. 0: none.
    pub scales: u32,
    /// The hidden layers of a multi-scale stack.
    pub msd_layers: Vec<ConvLayer>,
    /// LeakyReLU's negative slope (HiFi-GAN: 0.1). Must be positive: the backward reads the output's sign.
    pub leaky_slope: f32,
    /// AdamW learning rate (HiFi-GAN: 2e-4).
    pub lr: f32,
    /// AdamW betas and epsilon (HiFi-GAN: 0.8, 0.99).
    pub adam: Adam,
    pub weight_decay: f32,
    /// Seed of the weight initialisation. Random by default; a test or a resumed run pins it.
    pub seed: u64,
}

impl Default for DiscConfig {
    /// HiFi-GAN V1: periods 2, 3, 5, 7, 11 with widths 32, 128, 512, 1024, 1024, and three scales of the
    /// 128-128-256-512-1024-1024-1024 grouped stack; AdamW at 2e-4 with betas (0.8, 0.99).
    fn default() -> DiscConfig {
        let layer = |cout, k, stride, groups| ConvLayer { cout, k, stride, groups };
        DiscConfig {
            periods: vec![2, 3, 5, 7, 11],
            mpd_channels: vec![32, 128, 512, 1024, 1024],
            scales: 3,
            msd_layers: vec![
                layer(128, 15, 1, 1),
                layer(128, 41, 2, 4),
                layer(256, 41, 2, 16),
                layer(512, 41, 4, 16),
                layer(1024, 41, 4, 64),
                layer(1024, 41, 1, 16),
                layer(1024, 5, 1, 1),
            ],
            leaky_slope: 0.1,
            lr: 2e-4,
            adam: Adam { beta1: 0.8, beta2: 0.99, eps: 1e-8 },
            weight_decay: 0.01,
            seed: data::rng::random_seed(),
        }
    }
}

/// A layer of one sub-discriminator with its input width resolved.
#[derive(Clone, Copy, Debug)]
struct Layer {
    cin: u32,
    spec: ConvLayer,
    /// LeakyReLU follows (every layer but the logit convolution).
    act: bool,
}

impl Layer {
    fn pad(&self) -> u32 {
        (self.spec.k - 1) / 2
    }

    fn conv(&self, n: u32, l: u32) -> Conv1d {
        let pad = self.pad();
        Conv1d { n, cin: self.cin, l, cout: self.spec.cout, k: self.spec.k, stride: self.spec.stride, pad, dilation: 1, groups: self.spec.groups, lo: Conv1d::out_len(l, self.spec.k, self.spec.stride, pad, pad, 1) }
    }
}

/// One sub-discriminator: its parameter prefix, its batch (`period` columns for a multi-period stack, else 1) and
/// its layers.
struct Stack {
    name: String,
    /// `Some(p)`: multi-period over `[T/p, p]`. `None`: multi-scale over the waveform pooled `scale` times.
    period: Option<u32>,
    scale: u32,
    layers: Vec<Layer>,
}

impl Stack {
    fn new(name: String, period: Option<u32>, scale: u32, hidden: &[ConvLayer]) -> Stack {
        let mut layers = Vec::new();
        let mut cin = 1;
        for spec in hidden {
            assert!(spec.cout % spec.groups == 0 && cin % spec.groups == 0, "{name}: {cin} -> {} channels do not split into {} groups", spec.cout, spec.groups);
            layers.push(Layer { cin, spec: *spec, act: true });
            cin = spec.cout;
        }
        layers.push(Layer { cin, spec: ConvLayer { cout: 1, k: 3, stride: 1, groups: 1 }, act: false });
        Stack { name, period, scale, layers }
    }

    fn batch(&self) -> u32 {
        self.period.unwrap_or(1)
    }

    fn weight(&self, i: usize) -> String {
        format!("{}.{i}.weight", self.name)
    }

    fn bias(&self, i: usize) -> String {
        format!("{}.{i}.bias", self.name)
    }
}

/// What one sub-discriminator's forward leaves for its backward.
struct Pass {
    /// The stack's input, `[batch, 1, l0]`.
    x: DeviceBuffer,
    /// Each layer's output (post-activation for a hidden layer, the logits for the last), `[batch, cout, len]`.
    /// Empty under a recomputing [`SavedPolicy`]: the backward rebuilds them from `x`.
    outs: Vec<DeviceBuffer>,
    /// Each layer's output length.
    lens: Vec<u32>,
    l0: u32,
}

/// A forward over one waveform: the saved passes and the host copy of what the losses read.
pub struct Forward {
    passes: Vec<Pass>,
    wave_len: usize,
    /// Per sub-discriminator, its logits (flattened `[batch, 1, len]`).
    pub logits: Vec<Vec<f32>>,
    /// Per sub-discriminator, per hidden layer, its feature map.
    pub features: Vec<Vec<Vec<f32>>>,
}

/// The multi-period and multi-scale discriminators with their parameters and AdamW.
pub struct WaveDiscriminators {
    gpu: Gpu,
    cfg: DiscConfig,
    ps: ParamStore,
    opt: Optim,
    stacks: Vec<Stack>,
    conv: ConvKernels,
    k_leaky: usize,
    k_leaky_bwd: usize,
    k_add_chan: usize,
    k_bias_grad: usize,
    k_add2: usize,
    k_axpy: usize,
    /// The `AvgPool1d(4, 2, pad 2)` weights, `[0.25; 4]`.
    pool_w: DeviceBuffer,
    /// What a forward keeps for the backward ([`Self::set_saved_policy`]).
    saved: std::cell::Cell<SavedPolicy>,
}

impl WaveDiscriminators {
    /// Discriminators for `cfg` on `gpu`, which must have been built with (or `new_like`'d to) [`PIPELINES`].
    pub fn new(gpu: Gpu, cfg: DiscConfig) -> WaveDiscriminators {
        assert!(cfg.leaky_slope > 0.0, "disc: the LeakyReLU backward reads the output's sign, which needs a positive slope");
        let mut stacks = Vec::new();
        for &p in &cfg.periods {
            assert!(p >= 1, "disc: a period must be at least 1");
            let last = cfg.mpd_channels.len().saturating_sub(1);
            let hidden: Vec<ConvLayer> = cfg.mpd_channels.iter().enumerate().map(|(i, &cout)| ConvLayer { cout, k: 5, stride: if i == last { 1 } else { 3 }, groups: 1 }).collect();
            stacks.push(Stack::new(format!("mpd{p}"), Some(p), 0, &hidden));
        }
        for s in 0..cfg.scales {
            stacks.push(Stack::new(format!("msd{s}"), None, s, &cfg.msd_layers));
        }
        assert!(!stacks.is_empty(), "disc: no sub-discriminators configured");

        let mut rng = Rng::new(cfg.seed);
        let (mut roles, mut init) = (Vec::new(), std::collections::HashMap::new());
        for stack in &stacks {
            for (i, layer) in stack.layers.iter().enumerate() {
                let fan_in = (layer.cin / layer.spec.groups * layer.spec.k) as f32;
                let bound = 1.0 / fan_in.sqrt();
                let uniform = |rng: &mut Rng, n: usize| -> Vec<f32> { (0..n).map(|_| rng.uniform(-1.0, 1.0) as f32 * bound).collect() };
                let wn = (layer.spec.cout * (layer.cin / layer.spec.groups) * layer.spec.k) as usize;
                for (name, data) in [(stack.weight(i), uniform(&mut rng, wn)), (stack.bias(i), uniform(&mut rng, layer.spec.cout as usize))] {
                    roles.push((name.clone(), data.len(), Role::Trainable));
                    init.insert(name, data);
                }
            }
        }
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let kernel = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("disc: kernel `{name}` is not registered on this device; build it with disc::PIPELINES"));
        let opt = Optim::new(kernel("adamw"), kernel("gradnorm_sq"), kernel("grad_scale"), kernel("clip_coef"), kernel("grad_scale_buf"));
        let conv = ConvKernels { fwd: kernel("conv1d"), dx: kernel("conv1d_dx"), dw: kernel("conv1d_dw") };
        let (k_leaky, k_leaky_bwd, k_add_chan) = (kernel("leaky_relu"), kernel("leaky_relu_bwd"), kernel("add_chan_inplace"));
        let (k_bias_grad, k_add2, k_axpy) = (kernel("bias_grad_ncl"), kernel("add2"), kernel("axpy"));
        let pool_w = gpu.buffer("pool", 16, BufUsage::STORAGE | BufUsage::COPY_DST);
        gpu.write(&pool_w, bytemuck::cast_slice(&[0.25f32; 4]));
        let saved = std::cell::Cell::new(SavedPolicy::Keep);
        WaveDiscriminators { gpu, cfg, ps, opt, stacks, conv, k_leaky, k_leaky_bwd, k_add_chan, k_bias_grad, k_add2, k_axpy, pool_w, saved }
    }

    /// Choose what a forward keeps for the backward. [`SavedPolicy::Keep`] (the default) holds every layer's output
    /// on the device; [`SavedPolicy::Recompute`] holds only each sub-discriminator's input and the backward re-runs
    /// that stack's forward (one more forward's work per backward, and the outputs live only while one stack is
    /// reversed). The feature maps the losses read are host copies either way. Narrowed boundary formats do not
    /// apply: a stack's only boundary is its f32 input, so a [`SavedPolicy::Quantized`] choice is refused.
    pub fn set_saved_policy(&self, policy: SavedPolicy) -> Result<(), String> {
        if let SavedPolicy::Quantized(format) = policy {
            return Err(format!("mimi discriminators recompute from their f32 inputs; {format:?} saved boundaries do not apply"));
        }
        self.saved.set(policy);
        Ok(())
    }

    /// One sub-discriminator's forward from its input: each layer's output and length.
    fn run_stack(&self, stack: &Stack, x: &DeviceBuffer, l0: u32) -> (Vec<DeviceBuffer>, Vec<u32>) {
        let n = stack.batch();
        let (mut outs, mut lens): (Vec<DeviceBuffer>, Vec<u32>) = (Vec::new(), Vec::new());
        let mut l = l0;
        for (i, layer) in stack.layers.iter().enumerate() {
            let c = layer.conv(n, l);
            let total = n * c.cout * c.lo;
            let pre = self.gpu.storage(u64::from(total));
            let input = if i == 0 { x } else { &outs[i - 1] };
            let mut steps = vec![audio::conv::conv1d_fwd(&self.gpu, &self.conv, &c, input, self.ps.w(&stack.weight(i)), &pre)];
            steps.push(self.gpu.step(self.k_add_chan, &[&pre, self.ps.w(&stack.bias(i))], &[total, c.cout, c.lo], total));
            let out = if layer.act {
                let post = self.gpu.storage(u64::from(total));
                steps.push(self.gpu.step(self.k_leaky, &[&pre, &post], &[total, f(self.cfg.leaky_slope)], total));
                post
            } else {
                pre
            };
            self.gpu.submit(&[], &steps);
            outs.push(out);
            lens.push(c.lo);
            l = c.lo;
        }
        (outs, lens)
    }

    pub fn config(&self) -> &DiscConfig {
        &self.cfg
    }

    /// Number of sub-discriminators `K`.
    pub fn len(&self) -> usize {
        self.stacks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Every trainable tensor, `(name, element count)`.
    pub fn param_names(&self) -> Vec<(String, usize)> {
        self.ps.trained_params().cloned().collect()
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.ps.numel(name), "disc: {name} has {} values, got {}", self.ps.numel(name), data.len());
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// One AdamW step (`t` counts from 1) on the weights from the gradients the backwards accumulated, at the
    /// configured rate, betas and weight decay.
    pub fn adamw_step(&self, t: u32) {
        self.opt.step(&self.gpu, &self.ps, t, self.cfg.lr, self.cfg.weight_decay, self.cfg.adam, None, 1.0);
    }

    fn upload(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu.buffer("disc", (data.len().max(1) * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
        self.gpu.write(&b, bytemuck::cast_slice(data));
        b
    }

    /// The `AvgPool1d(4, 2, pad 2)` of a `[1, 1, l]` buffer, as a fixed-weight convolution.
    fn pool_desc(l: u32) -> Conv1d {
        Conv1d { n: 1, cin: 1, l, cout: 1, k: 4, stride: 2, pad: 2, dilation: 1, groups: 1, lo: Conv1d::out_len(l, 4, 2, 2, 2, 1) }
    }

    /// The input of every stack for `wave`: multi-period stacks get the host reshape, multi-scale stacks the pooled
    /// chain (`scale` pools of the waveform). Returns `(buffer, l0)` per stack.
    fn stack_inputs(&self, wave: &[f32]) -> Vec<(DeviceBuffer, u32)> {
        let wave_buf = self.upload(wave);
        // pooled[s] is the waveform average-pooled s times.
        let mut pooled: Vec<(DeviceBuffer, u32)> = vec![(wave_buf, wave.len() as u32)];
        let scales = self.stacks.iter().filter(|s| s.period.is_none()).map(|s| s.scale).max().unwrap_or(0);
        for _ in 0..scales {
            let (prev, l) = pooled.last().expect("the waveform is the first pooled copy");
            let c = Self::pool_desc(*l);
            let out = self.gpu.storage(u64::from(c.lo));
            self.gpu.submit(&[], &[audio::conv::conv1d_fwd(&self.gpu, &self.conv, &c, prev, &self.pool_w, &out)]);
            pooled.push((out, c.lo));
        }
        self.stacks
            .iter()
            .map(|s| match s.period {
                Some(p) => {
                    let (p, l0) = (p as usize, wave.len() / p as usize);
                    let mut x2 = vec![0.0f32; p * l0];
                    for c in 0..p {
                        for t in 0..l0 {
                            x2[c * l0 + t] = wave[t * p + c];
                        }
                    }
                    (self.upload(&x2), l0 as u32)
                }
                None => {
                    let (buf, l) = &pooled[s.scale as usize];
                    (clone_buf(&self.gpu, buf, *l), *l)
                }
            })
            .collect()
    }

    /// Run every sub-discriminator over `wave` and read back its logits and feature maps.
    pub fn forward(&self, wave: &[f32]) -> Forward {
        for s in &self.stacks {
            if let Some(p) = s.period {
                assert!(wave.len() >= p as usize, "disc: a {}-sample waveform is shorter than the period {p}", wave.len());
            }
        }
        let inputs = self.stack_inputs(wave);
        let mut passes = Vec::new();
        let (mut logits, mut features) = (Vec::new(), Vec::new());
        let recompute = self.saved.get().recomputes();
        for (stack, (x, l0)) in self.stacks.iter().zip(inputs) {
            let n = stack.batch();
            let (outs, lens) = self.run_stack(stack, &x, l0);
            let last = stack.layers.len() - 1;
            logits.push(self.gpu.read(&outs[last], (n * lens[last]) as usize));
            features.push((0..last).map(|i| self.gpu.read(&outs[i], (n * stack.layers[i].spec.cout * lens[i]) as usize)).collect());
            let outs = if recompute { Vec::new() } else { outs };
            passes.push(Pass { x, outs, lens, l0 });
        }
        Forward { passes, wave_len: wave.len(), logits, features }
    }

    /// Reverse `fwd` from `d_logits` (the loss's gradient wrt each sub-discriminator's logits) and, when given,
    /// `d_features` (wrt each hidden feature map), accumulating every weight gradient (zero them with
    /// [`Self::zero_grads`] first). With `to_input`, returns the gradient wrt the waveform, summed over the
    /// sub-discriminators.
    pub fn backward(&self, fwd: &Forward, d_logits: &[Vec<f32>], d_features: Option<&[Vec<Vec<f32>>]>, to_input: bool) -> Option<Vec<f32>> {
        assert_eq!(d_logits.len(), self.stacks.len(), "disc: {} logit gradients for {} sub-discriminators", d_logits.len(), self.stacks.len());
        let mut d_wave = vec![0.0f32; fwd.wave_len];
        // Input gradients of the multi-scale stacks, per pooling depth (summed over the stacks of one depth: there is one).
        let mut d_pooled: Vec<Option<Vec<f32>>> = Vec::new();
        for (si, (stack, pass)) in self.stacks.iter().zip(&fwd.passes).enumerate() {
            let n = stack.batch();
            let last = stack.layers.len() - 1;
            assert_eq!(d_logits[si].len(), (n * pass.lens[last]) as usize, "disc: logit gradient length for {}", stack.name);
            // A recomputing forward kept only the input: rebuild this stack's outputs, which live until it is reversed.
            let rebuilt;
            let outs: &[DeviceBuffer] = if pass.outs.is_empty() {
                rebuilt = self.run_stack(stack, &pass.x, pass.l0).0;
                &rebuilt
            } else {
                &pass.outs
            };
            // The gradient arriving at layer `i`'s output.
            let mut d_out = self.upload(&d_logits[si]);
            for i in (0..=last).rev() {
                let layer = &stack.layers[i];
                let c = layer.conv(n, if i == 0 { pass.l0 } else { pass.lens[i - 1] });
                let total = n * c.cout * c.lo;
                if i < last {
                    if let Some(extra) = d_features.map(|d| &d[si][i]).filter(|e| !e.is_empty()) {
                        assert_eq!(extra.len(), total as usize, "disc: feature gradient length for {} layer {i}", stack.name);
                        let (extra, sum) = (self.upload(extra), self.gpu.storage(u64::from(total)));
                        self.gpu.submit(&[], &[self.gpu.step(self.k_add2, &[&d_out, &extra, &sum], &[total], total)]);
                        d_out = sum;
                    }
                }
                let d_pre_act;
                let d_pre: &DeviceBuffer = if layer.act {
                    d_pre_act = self.gpu.storage(u64::from(total));
                    self.gpu.submit(&[], &[self.gpu.step(self.k_leaky_bwd, &[&outs[i], &d_out, &d_pre_act], &[total, f(self.cfg.leaky_slope)], total)]);
                    &d_pre_act
                } else {
                    &d_out
                };
                let input = if i == 0 { &pass.x } else { &outs[i - 1] };
                let need_dx = i > 0 || to_input;
                let d_in = need_dx.then(|| self.gpu.storage(u64::from(n * c.cin * c.l)));
                let (wname, bname) = (stack.weight(i), stack.bias(i));
                let mut steps = audio::conv::conv1d_bwd(&self.gpu, &self.conv, &c, d_pre, input, self.ps.w(&wname), d_in.as_ref(), Some(self.ps.g(&wname)));
                let db = self.gpu.storage(u64::from(c.cout));
                steps.push(self.gpu.step(self.k_bias_grad, &[d_pre, &db], &[n, c.cout, c.lo], c.cout));
                steps.push(self.gpu.step(self.k_axpy, &[self.ps.g(&bname), &db], &[c.cout, f(1.0)], c.cout));
                self.gpu.submit(&[], &steps);
                if let Some(d_in) = d_in {
                    if i > 0 {
                        d_out = d_in;
                    } else {
                        let dx = self.gpu.read(&d_in, (n * c.l) as usize);
                        match stack.period {
                            Some(p) => {
                                let (p, l0) = (p as usize, pass.l0 as usize);
                                for col in 0..p {
                                    for t in 0..l0 {
                                        d_wave[t * p + col] += dx[col * l0 + t];
                                    }
                                }
                            }
                            None => {
                                let depth = stack.scale as usize;
                                if d_pooled.len() <= depth {
                                    d_pooled.resize(depth + 1, None);
                                }
                                d_pooled[depth] = Some(dx);
                            }
                        }
                    }
                }
            }
        }
        if !to_input {
            return None;
        }
        // Through the pooling chain, deepest first: the gradient at depth s reaches depth s-1 through the pool's adjoint.
        let mut lens = vec![fwd.wave_len as u32];
        for _ in 1..d_pooled.len() {
            lens.push(Self::pool_desc(*lens.last().expect("non-empty")).lo);
        }
        let mut carry: Option<Vec<f32>> = None;
        for depth in (0..d_pooled.len()).rev() {
            let mut g = d_pooled[depth].clone().unwrap_or_else(|| vec![0.0; lens[depth] as usize]);
            if let Some(from_deeper) = carry.take() {
                let c = Self::pool_desc(lens[depth]);
                let (dy, dx) = (self.upload(&from_deeper), self.gpu.storage(u64::from(lens[depth])));
                // The pool's input gradient only: its weights are constants (no dw), so the `x` operand is unused.
                let steps = audio::conv::conv1d_bwd(&self.gpu, &self.conv, &c, &dy, &dy, &self.pool_w, Some(&dx), None);
                self.gpu.submit(&[], &steps);
                for (a, b) in g.iter_mut().zip(self.gpu.read(&dx, lens[depth] as usize)) {
                    *a += b;
                }
            }
            carry = Some(g);
        }
        for (a, b) in d_wave.iter_mut().zip(carry.unwrap_or_default()) {
            *a += b;
        }
        Some(d_wave)
    }
}

fn clone_buf(gpu: &Gpu, src: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let v = gpu.read(src, n as usize);
    let b = gpu.buffer("disc", u64::from(n.max(1)) * 4, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    gpu.write(&b, bytemuck::cast_slice(&v));
    b
}

/// The LSGAN discriminator loss and its gradient wrt every sub-discriminator's real and fake logits.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscLoss {
    pub loss: f64,
    pub d_real: Vec<Vec<f32>>,
    pub d_fake: Vec<Vec<f32>>,
}

/// `(1/K) sum_k [mean((D_k(x) - 1)^2) + mean(D_k(x_hat)^2)]` over the real and fake logits, and its gradient:
/// `2 (r - 1) / (K n)` on a real logit, `2 f / (K n)` on a fake one.
pub fn lsgan_d_loss(real: &[Vec<f32>], fake: &[Vec<f32>]) -> DiscLoss {
    assert_eq!(real.len(), fake.len(), "lsgan: {} real and {} fake sub-discriminators", real.len(), fake.len());
    let k = real.len() as f64;
    let mut loss = 0.0f64;
    let (mut d_real, mut d_fake) = (Vec::new(), Vec::new());
    for (r, f) in real.iter().zip(fake) {
        let (nr, nf) = (r.len() as f64, f.len() as f64);
        loss += r.iter().map(|v| (f64::from(*v) - 1.0).powi(2)).sum::<f64>() / nr + f.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / nf;
        d_real.push(r.iter().map(|v| (2.0 * (f64::from(*v) - 1.0) / (k * nr)) as f32).collect());
        d_fake.push(f.iter().map(|v| (2.0 * f64::from(*v) / (k * nf)) as f32).collect());
    }
    DiscLoss { loss: loss / k, d_real, d_fake }
}

/// The generator's adversarial loss `(1/K) sum_k mean((D_k(x_hat) - 1)^2)` and its gradient wrt the fake logits.
pub fn lsgan_g_loss(fake: &[Vec<f32>]) -> (f64, Vec<Vec<f32>>) {
    let k = fake.len() as f64;
    let loss = fake.iter().map(|f| f.iter().map(|v| (f64::from(*v) - 1.0).powi(2)).sum::<f64>() / f.len() as f64).sum::<f64>() / k;
    let grad = fake.iter().map(|f| f.iter().map(|v| (2.0 * (f64::from(*v) - 1.0) / (k * f.len() as f64)) as f32).collect()).collect();
    (loss, grad)
}

/// Feature matching `(1 / (K L)) sum_k sum_l mean(|real - fake|)` over the per-sub-discriminator, per-layer feature
/// maps (`L` counted per sub-discriminator and averaged over them), and its gradient wrt the fake features:
/// `sign(fake - real) / (K L n)`, 0 at a tie.
pub fn feature_matching(real: &[Vec<Vec<f32>>], fake: &[Vec<Vec<f32>>]) -> (f64, Vec<Vec<Vec<f32>>>) {
    assert_eq!(real.len(), fake.len(), "feature matching: {} real and {} fake sub-discriminators", real.len(), fake.len());
    let k = real.len() as f64;
    let (mut loss, mut grad) = (0.0f64, Vec::new());
    for (rk, fk) in real.iter().zip(fake) {
        let layers = rk.len().max(1) as f64;
        let mut gk = Vec::new();
        for (r, f) in rk.iter().zip(fk) {
            assert_eq!(r.len(), f.len(), "feature matching: real and fake feature maps differ in size");
            let n = r.len() as f64;
            loss += r.iter().zip(f).map(|(r, f)| (f64::from(*f) - f64::from(*r)).abs()).sum::<f64>() / n / layers / k;
            gk.push(r.iter().zip(f).map(|(r, f)| (if f > r { 1.0 } else if f < r { -1.0 } else { 0.0 } / (k * layers * n)) as f32).collect());
        }
        grad.push(gk);
    }
    (loss, grad)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use brain_testutil::rounding::agreement_bound;
    use data::rng::Lcg;
    use std::collections::HashMap;

    /// Waveform length of the checks: a multiple of every tiny period so none crops.
    pub const T: usize = 60;

    /// Two periods (2, 3) of widths 3 and 4, and two scales of a grouped three-layer stack: every code path
    /// (period reshape, stride-3 and stride-1 layers, groups, the pooling chain) at a size an element-wise sweep affords.
    pub fn tiny_config() -> DiscConfig {
        let layer = |cout, k, stride, groups| ConvLayer { cout, k, stride, groups };
        DiscConfig {
            periods: vec![2, 3],
            mpd_channels: vec![3, 4],
            scales: 2,
            msd_layers: vec![layer(4, 5, 1, 1), layer(4, 5, 2, 2), layer(6, 3, 1, 2)],
            leaky_slope: 0.1,
            lr: 5e-3,
            adam: Adam::default(),
            weight_decay: 0.0,
            seed: data::rng::random_seed(),
        }
    }

    pub fn tiny() -> WaveDiscriminators {
        WaveDiscriminators::new(Gpu::new_cpu(PIPELINES), tiny_config())
    }

    fn wave(seed: u64) -> Vec<f32> {
        Lcg::new(seed).vec(T)
    }

    // ---- the f64 oracle: the same networks written out, sharing no code with the device path ------------------------

    /// `x` (`[n, cin, l]`) convolved with `w` (`[cout, cin/g, k]`), zero-padded `(k-1)/2`, plus `b`.
    fn conv64(x: &[f64], (n, cin, l): (usize, usize, usize), w: &[f64], b: &[f64], layer: &Layer) -> (Vec<f64>, usize) {
        let (cout, k, stride, groups) = (layer.spec.cout as usize, layer.spec.k as usize, layer.spec.stride as usize, layer.spec.groups as usize);
        let pad = (k - 1) / 2;
        let lo = (l + 2 * pad - k) / stride + 1;
        let (cg, og) = (cin / groups, cout / groups);
        let mut y = vec![0.0; n * cout * lo];
        for bi in 0..n {
            for o in 0..cout {
                let g = o / og;
                for t in 0..lo {
                    let mut acc = b[o];
                    for c in 0..cg {
                        for j in 0..k {
                            let pos = (t * stride + j) as isize - pad as isize;
                            if pos >= 0 && (pos as usize) < l {
                                acc += w[(o * cg + c) * k + j] * x[(bi * cin + g * cg + c) * l + pos as usize];
                            }
                        }
                    }
                    y[(bi * cout + o) * lo + t] = acc;
                }
            }
        }
        (y, lo)
    }

    /// `AvgPool1d(4, 2, padding = 2)` with torch's `count_include_pad`.
    fn pool64(x: &[f64]) -> Vec<f64> {
        let lo = x.len() / 2 + 1;
        (0..lo).map(|t| (0..4).filter_map(|j| (t * 2 + j).checked_sub(2).and_then(|p| x.get(p))).sum::<f64>() / 4.0).collect()
    }

    /// Per sub-discriminator, `(logits, hidden feature maps)` of `wave` under parameters `p`.
    fn outputs64(d: &WaveDiscriminators, p: &HashMap<String, Vec<f64>>, wave: &[f64]) -> Vec<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut pooled = vec![wave.to_vec()];
        for _ in 1..d.cfg.scales {
            pooled.push(pool64(pooled.last().unwrap()));
        }
        d.stacks
            .iter()
            .map(|s| {
                let (n, mut cur, mut l) = match s.period {
                    Some(pp) => {
                        let (pp, l0) = (pp as usize, wave.len() / pp as usize);
                        let mut x2 = vec![0.0; pp * l0];
                        for c in 0..pp {
                            for t in 0..l0 {
                                x2[c * l0 + t] = wave[t * pp + c];
                            }
                        }
                        (pp, x2, l0)
                    }
                    None => (1, pooled[s.scale as usize].clone(), pooled[s.scale as usize].len()),
                };
                let mut feats = Vec::new();
                for (i, layer) in s.layers.iter().enumerate() {
                    let (mut y, lo) = conv64(&cur, (n, layer.cin as usize, l), &p[&s.weight(i)], &p[&s.bias(i)], layer);
                    if layer.act {
                        y.iter_mut().for_each(|v| *v = if *v >= 0.0 { *v } else { f64::from(d.cfg.leaky_slope) * *v });
                        feats.push(y.clone());
                    }
                    (cur, l) = (y, lo);
                }
                (cur, feats)
            })
            .collect()
    }

    fn params64(d: &WaveDiscriminators) -> HashMap<String, Vec<f64>> {
        d.param_names().into_iter().map(|(n, _)| (n.clone(), d.read_weight(&n).into_iter().map(f64::from).collect())).collect()
    }

    /// A loss that touches every logit and every feature map with a different weight: `sum r.logits + sum s.features`.
    struct Probe {
        r: Vec<Vec<f32>>,
        s: Vec<Vec<Vec<f32>>>,
    }

    impl Probe {
        fn new(fwd: &Forward) -> Probe {
            let mut rng = Lcg::new(data::rng::random_seed());
            Probe { r: fwd.logits.iter().map(|l| rng.vec(l.len())).collect(), s: fwd.features.iter().map(|k| k.iter().map(|f| rng.vec(f.len())).collect()).collect() }
        }

        fn loss(&self, out: &[(Vec<f64>, Vec<Vec<f64>>)]) -> f64 {
            out.iter().zip(&self.r).zip(&self.s).map(|(((lg, ft), r), s)| lg.iter().zip(r).map(|(a, b)| a * f64::from(*b)).sum::<f64>() + ft.iter().zip(s).map(|(f, s)| f.iter().zip(s).map(|(a, b)| a * f64::from(*b)).sum::<f64>()).sum::<f64>()).sum()
        }
    }

    /// Terms on the longest chain the backward walks to one weight gradient: every layer's `cin / g * k` contraction
    /// and the per-position sum of the weight gradient.
    fn chain_terms(d: &WaveDiscriminators) -> usize {
        d.stacks.iter().map(|s| s.layers.iter().map(|l| (l.cin / l.spec.groups * l.spec.k) as usize + T).sum::<usize>()).max().unwrap_or(1)
    }

    /// Element-wise central differences of `loss` in f64 against `analytic`; the tolerance is the round-off of the
    /// backward chain at the largest gradient entry. Entries whose step crosses a LeakyReLU kink show disagreeing
    /// one-sided differences and are drawn out.
    fn check_elementwise(what: &str, analytic: &[f32], k: usize, mut loss: impl FnMut(usize, f64) -> f64) {
        let scale = analytic.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
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
            assert!((f64::from(*g) - fd).abs() <= tol, "{what}[{j}]: analytic {g:e} vs central difference {fd:e} (tolerance {tol:e}, largest entry {scale:e})");
        }
        assert!(kinks * 20 <= analytic.len().max(20), "{what}: {kinks} of {} entries cross a kink", analytic.len());
    }

    /// The device forward is the f64 oracle's, which pins the oracle the gradient checks stand on (it covers the
    /// period reshape, the grouped strided convolutions and the pooling chain).
    #[test]
    fn the_forward_matches_the_f64_oracle() {
        let d = tiny();
        let x = wave(1);
        let fwd = d.forward(&x);
        let want = outputs64(&d, &params64(&d), &x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
        for (si, (logits, feats)) in want.iter().enumerate() {
            let scale = logits.iter().chain(feats.iter().flatten()).fold(0.0f64, |m, v| m.max(v.abs()));
            let tol = agreement_bound(chain_terms(&d), scale);
            assert_eq!(fwd.logits[si].len(), logits.len(), "{} logit count", d.stacks[si].name);
            for (j, (g, w)) in fwd.logits[si].iter().zip(logits).enumerate() {
                assert!((f64::from(*g) - w).abs() <= tol, "{} logit {j}: {g} vs {w} (tolerance {tol:e})", d.stacks[si].name);
            }
            for (l, (g, w)) in fwd.features[si].iter().zip(feats).enumerate() {
                assert_eq!(g.len(), w.len(), "{} feature map {l} size", d.stacks[si].name);
                for (j, (g, w)) in g.iter().zip(w).enumerate() {
                    assert!((f64::from(*g) - w).abs() <= tol, "{} feature {l}[{j}]: {g} vs {w} (tolerance {tol:e})", d.stacks[si].name);
                }
            }
        }
    }

    /// Every weight and bias gradient, element by element, of a loss on the logits and the feature maps.
    #[test]
    fn recomputing_the_forward_gives_the_kept_gradients_bitwise() {
        let d = tiny();
        let x = wave(3);
        // One probe for both runs: the loss must be the same function of the outputs.
        let probe = Probe::new(&d.forward(&x));
        let grads = |policy: SavedPolicy| {
            d.set_saved_policy(policy).expect("a supported policy");
            let fwd = d.forward(&x);
            d.zero_grads();
            let dx = d.backward(&fwd, &probe.r, Some(&probe.s), true).expect("input gradient");
            (d.param_names().into_iter().map(|(n, _)| d.read_grad(&n)).collect::<Vec<_>>(), dx)
        };
        let keep = grads(SavedPolicy::Keep);
        let recompute = grads(SavedPolicy::Recompute);
        assert_eq!(keep, recompute, "the rebuilt outputs must equal the kept ones");
        let narrow = SavedPolicy::Quantized(model::saved::SavedFormat::Bf16);
        assert!(d.set_saved_policy(narrow).unwrap_err().contains("do not apply"));
    }

    #[test]
    fn the_weight_gradients_match_central_differences() {
        let d = tiny();
        let x = wave(2);
        let fwd = d.forward(&x);
        let probe = Probe::new(&fwd);
        d.zero_grads();
        d.backward(&fwd, &probe.r, Some(&probe.s), false);
        let (x64, base) = (x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), params64(&d));
        for (name, _) in d.param_names() {
            let analytic = d.read_grad(&name);
            let loss = |j: usize, delta: f64| {
                let mut p = base.clone();
                p.get_mut(&name).unwrap()[j] += delta;
                probe.loss(&outputs64(&d, &p, &x64))
            };
            check_elementwise(&name, &analytic, chain_terms(&d), loss);
        }
    }

    /// The gradient wrt the waveform the generator's terms differentiate, element by element: through the period
    /// reshape and, for the deeper scales, back through the pooling chain.
    #[test]
    fn the_input_gradient_matches_central_differences() {
        let d = tiny();
        let x = wave(3);
        let fwd = d.forward(&x);
        let probe = Probe::new(&fwd);
        d.zero_grads();
        let analytic = d.backward(&fwd, &probe.r, Some(&probe.s), true).expect("input gradient requested");
        let (p, x64) = (params64(&d), x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
        let loss = |j: usize, delta: f64| {
            let mut x = x64.clone();
            x[j] += delta;
            probe.loss(&outputs64(&d, &p, &x))
        };
        check_elementwise("input", &analytic, chain_terms(&d) + 8, loss);
    }

    #[test]
    fn the_losses_match_their_definitions() {
        let (real, fake) = (vec![vec![0.5f32, 1.5], vec![1.0]], vec![vec![0.0f32, 1.0], vec![-1.0]]);
        // K = 2. sub 0: mean((r-1)^2) = 0.25, mean(f^2) = 0.5; sub 1: 0, 1.  (0.25 + 0.5 + 0 + 1) / 2.
        let d = lsgan_d_loss(&real, &fake);
        assert!((d.loss - 0.875).abs() < 1e-12, "{}", d.loss);
        assert_eq!(d.d_real, [vec![-0.25f32, 0.25], vec![0.0]]);
        assert_eq!(d.d_fake, [vec![0.0f32, 0.5], vec![-1.0]]);
        // G: sub 0 mean((f-1)^2) = 0.5, sub 1 = 4: (0.5 + 4) / 2.
        let (g, dg) = lsgan_g_loss(&fake);
        assert!((g - 2.25).abs() < 1e-12 && dg == [vec![-0.5f32, 0.0], vec![-2.0]], "{g} {dg:?}");
        // FM: sub 0 layers [|1-0|, |2-0|] -> means 1 and ... one layer each here: sub 0 mean |r-f| = 1, sub 1 = 2.
        let (fm, dfm) = feature_matching(&[vec![vec![1.0f32, 2.0]], vec![vec![3.0f32]]], &[vec![vec![0.0f32, 3.0]], vec![vec![1.0f32]]]);
        assert!((fm - (1.0 + 2.0) / 2.0).abs() < 1e-12, "{fm}");
        assert_eq!(dfm, [vec![vec![-0.25f32, 0.25]], vec![vec![-0.5f32]]]);
    }

    /// Spec: trained alone on a fixed real waveform and a fixed fake one, the discriminator's loss falls.
    #[test]
    fn the_discriminator_loss_falls_when_only_the_discriminator_trains() {
        let d = tiny();
        let (real, fake) = (wave(4), wave(5));
        let mut losses = Vec::new();
        for t in 1..=40 {
            let (fr, ff) = (d.forward(&real), d.forward(&fake));
            let l = lsgan_d_loss(&fr.logits, &ff.logits);
            losses.push(l.loss);
            d.zero_grads();
            d.backward(&fr, &l.d_real, None, false);
            d.backward(&ff, &l.d_fake, None, false);
            d.adamw_step(t);
        }
        assert!(losses.iter().all(|l| l.is_finite()));
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(last < 0.7 * first, "the LSGAN loss went {first} -> {last}: {losses:?}");
    }
}
