// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS v0.1 / AlexNet on the device, composed entirely from existing
//! kernels and blocks - this crate adds none:
//!
//! | LPIPS step (`lpips/lpips.py`) | how it runs here |
//! |---|---|
//! | `[0,1]` HWC -> NCHW | `imaging::Ctx::to_chw` (`nlc_nchw`) |
//! | `2x-1`, then `ScalingLayer` | one per-channel affine, `imaging::Ctx::affine` (`film_chan`) |
//! | AlexNet `features[0..12]` | `vision::Conv` (bias, ReLU: `conv_bias_reg` + `leaky_relu`) and `vision::MaxPool` (`maxpool2d`) |
//! | `normalize_tensor` over channels | `l2norm_scale2d` with a unit gain |
//! | `(f0 - f1)^2` | `sub` (the two images are one batch of two), `mul` |
//! | `NetLinLayer` (1x1, no bias) | `vision::Conv`, `Norm::None`, no bias (`conv2d`) |
//! | `spatial_average` | `avgpool2d` to `1x1` |
//!
//! Both images go through the trunk as ONE batch of two, so the taps of the
//! pair sit in one buffer and their difference is a single offset `sub`.
//!
//! `l2norm_scale2d` computes `x * rsqrt(sum_c x^2 + eps)` where the reference
//! divides by `sqrt(sum_c x^2) + 1e-10`. With `eps = 1e-20` the two agree to
//! within `1e-10 / |x|` relative, and both are exactly zero where a ReLU map
//! is zero in every channel; `tests/reference.rs` holds the device to an f64
//! transcription of the reference formula.
//!
//! A mask weights the spatial average. At each tap the mask is area-averaged
//! onto that tap's grid (`avgpool2d`, torch's adaptive rule), and the layer's
//! distance is `sum(d * m) / sum(m)` - the unmasked metric exactly when the
//! mask is all ones. A feature position is weighted by how much of its cell
//! the mask covers; its receptive field still reaches past the mask's edge,
//! which is inherent to a convolutional metric.

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{Act, Conv, ConvKernelIds, ConvNames, ConvSpec, Ctx, MaxPool, Norm, PoolSpec, Shape};

use crate::config::{head_weight, trunk_bias, trunk_weight, MIN_SIDE, POOL_K, POOL_STRIDE, SCALE, SHIFT, TRUNK};
use crate::import::Tensors;

/// Every kernel the metric dispatches, by name.
pub const PIPELINES: &[(&str, &str)] = &[
    // The trunk: biased convolutions, ReLU, max-pool.
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("leaky_relu", kernels::LEAKY_RELU),
    // `vision::Conv` resolves an activation's forward and backward together.
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("maxpool2d", kernels::MAXPOOL2D),
    // The heads: a bias-free 1x1 convolution.
    ("conv2d", kernels::CONV2D),
    // The input: layout, and the scaling layer's affine.
    ("nlc_nchw", kernels::NLC_NCHW),
    ("film_chan", kernels::FILM_CHAN),
    // The distance.
    ("l2norm_scale2d", kernels::L2NORM_SCALE2D),
    ("sub", kernels::SUB),
    ("mul", kernels::MUL),
    ("avgpool2d", kernels::AVGPOOL2D),
];

/// The epsilon `l2norm_scale2d` adds to the channel sum of squares: the
/// square of the reference's `1e-10` added to the norm (module docs).
const NORM_EPS: f32 = 1e-20;

/// An LPIPS distance: the sum over the five taps, and each tap's share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distance {
    pub total: f64,
    pub layers: [f64; 5],
}

/// The trunk and heads for one input size. Rebuilt when the size changes:
/// `vision`'s blocks own buffers sized for their input.
struct Graph {
    w: u32,
    h: u32,
    convs: Vec<Conv>,
    pools: Vec<Option<MaxPool>>,
    heads: Vec<Conv>,
}

/// The LPIPS v0.1 / AlexNet metric, resident on one device.
pub struct Lpips {
    gpu: Gpu,
    ids: ConvKernelIds,
    ps: ParamStore,
    ones: DeviceBuffer,
    k_sub: usize,
    k_l2: usize,
    graph: Option<Graph>,
}

impl Lpips {
    /// Put `weights` ([`crate::import::read`]) on `gpu`, which must have been
    /// built with (or `new_like`'d to) [`PIPELINES`]. The metric runs on
    /// whatever device `gpu` is; it never moves itself elsewhere.
    pub fn new(gpu: Gpu, weights: &Tensors) -> Result<Lpips, String> {
        crate::import::validate(weights)?;
        let kernel = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("lpips: kernel `{name}` is not registered on this device; build it with lpips::PIPELINES"));
        for (name, _) in PIPELINES {
            kernel(name)?;
        }
        let (k_sub, k_l2) = (kernel("sub")?, kernel("l2norm_scale2d")?);
        let mut roles: Vec<(String, usize, Role)> = weights.iter().map(|(n, (_, d))| (n.clone(), d.len(), Role::Frozen)).collect();
        roles.sort_by(|a, b| a.0.cmp(&b.0));
        let init = weights.iter().map(|(n, (_, d))| (n.clone(), d.clone())).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let widest = TRUNK.iter().map(|c| c.cout).max().expect("five convolutions");
        let ones = gpu.storage_init("lpips.ones", &vec![1.0; widest as usize]);
        Ok(Lpips { ids: ConvKernelIds::resolve(PIPELINES), gpu, ps, ones, k_sub, k_l2, graph: None })
    }

    /// [`Lpips::new`] from the model store: the trunk and heads
    /// [`crate::spec::resolve`] finds, on a `new_like` of `gpu`.
    pub fn from_store(gpu: &Gpu) -> Result<Lpips, String> {
        let (trunk, heads) = crate::spec::resolve()?;
        let weights = crate::import::read(&trunk, &heads)?;
        Lpips::new(gpu.new_like(PIPELINES), &weights)
    }

    /// The device the metric runs on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn build(&self, w: u32, h: u32) -> Graph {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let mut shape = Shape::new(2, 3, h, w);
        let (mut convs, mut pools, mut heads) = (Vec::new(), Vec::new(), Vec::new());
        for (i, c) in TRUNK.iter().enumerate() {
            let pool = c.pooled.then(|| MaxPool::new(&ctx, shape, PoolSpec::new(POOL_K, POOL_STRIDE, 0)));
            if let Some(p) = &pool {
                shape = p.out_shape;
            }
            let names = ConvNames { weight: trunk_weight(i), bias: trunk_bias(i), ..ConvNames::torch_flat("") };
            let spec = ConvSpec { cout: c.cout, k: c.k, stride: c.stride, pad: c.pad, groups: 1, dilation: 1, norm: Norm::None, act: Act::Relu, bias: true };
            let conv = Conv::with_names(&ctx, &format!("features.{}", c.index), names, shape, spec, false);
            shape = conv.out_shape;
            let head_names = ConvNames { weight: head_weight(i), ..ConvNames::torch_flat("") };
            let head_spec = ConvSpec { cout: 1, k: 1, stride: 1, pad: 0, groups: 1, dilation: 1, norm: Norm::None, act: Act::None, bias: false };
            heads.push(Conv::with_names(&ctx, &format!("lin{i}"), head_names, Shape::new(1, c.cout, shape.h, shape.w), head_spec, false));
            pools.push(pool);
            convs.push(conv);
        }
        Graph { w, h, convs, pools, heads }
    }

    /// The LPIPS distance between `a` and `b`, interleaved RGB in `[0, 1]`,
    /// `w x h` each. `mask`, one weight per pixel, weights the spatial
    /// average (module docs); `None` is the reference metric.
    pub fn distance(&mut self, a: &[f32], b: &[f32], w: u32, h: u32, mask: Option<&[f32]>) -> Result<Distance, String> {
        let px = w as usize * h as usize;
        if w < MIN_SIDE || h < MIN_SIDE {
            return Err(format!("lpips: {w}x{h} is smaller than the {MIN_SIDE}x{MIN_SIDE} AlexNet needs"));
        }
        if a.len() != 3 * px || b.len() != 3 * px {
            return Err(format!("lpips: {w}x{h} RGB is {} values, got {} and {}", 3 * px, a.len(), b.len()));
        }
        if let Some(m) = mask {
            if m.len() != px {
                return Err(format!("lpips: a {w}x{h} mask is {px} values, got {}", m.len()));
            }
            if m.iter().any(|v| !v.is_finite() || *v < 0.0) || m.iter().all(|v| *v == 0.0) {
                return Err("lpips: a mask needs finite, non-negative weights, not all zero".to_string());
            }
        }
        if self.graph.as_ref().is_none_or(|g| (g.w, g.h) != (w, h)) {
            self.graph = None;
            self.gpu.poll_wait();
            self.graph = Some(self.build(w, h));
        }
        let result = self.run(a, b, w, h, mask);
        self.gpu.poll_wait();
        result
    }

    fn run(&self, a: &[f32], b: &[f32], w: u32, h: u32, mask: Option<&[f32]>) -> Result<Distance, String> {
        let g = self.graph.as_ref().expect("built by distance");
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let img = imaging::Ctx::new(&self.gpu);
        let input = Shape::new(2, 3, h, w);

        let hwc = self.gpu.storage(input.numel() as u64);
        self.gpu.write_f32_at(&hwc, 0, a);
        self.gpu.write_f32_at(&hwc, a.len() as u64, b);
        let chw = img.to_chw(&hwc, input);
        // (2x - 1 - shift) / scale = x * (2 / scale) + (-1 - shift) / scale
        let scale: Vec<f32> = SCALE.iter().map(|s| 2.0 / s).collect();
        let shift: Vec<f32> = SHIFT.iter().zip(&SCALE).map(|(t, s)| (-1.0 - t) / s).collect();
        let mut x = img.affine(&chw, input, &scale, &shift);

        let mask = mask.map(|m| (img.upload("lpips.mask", m), Shape::new(1, 1, h, w)));
        let mut layers = [0.0f64; 5];
        for (i, conv) in g.convs.iter().enumerate() {
            if let Some(p) = &g.pools[i] {
                p.forward(&ctx, &x);
                x = p.out().clone();
            }
            conv.forward(&ctx, &self.ps, &x);
            x = conv.out().clone();
            let s = conv.out_shape;
            let (per, hw) = (s.c * s.h * s.w, s.h * s.w);

            // Unit-normalize both images' features over channels, then the
            // squared difference of image 0 and image 1.
            let unit = ctx.act(s.numel());
            let norm = self.gpu.step(self.k_l2, &[&x, &self.ones, &unit], &[s.n, s.c, hw, NORM_EPS.to_bits()], s.n * hw);
            let diff = ctx.act(per);
            let sub = self.gpu.step(self.k_sub, &[&unit, &unit, &diff], &[per, 0, per], per);
            let sq = ctx.act(per);
            let mul = self.gpu.step(self.ids.need(self.ids.mul, "mul"), &[&diff, &diff, &sq], &[per], per);
            self.gpu.submit(&[], &[norm, sub, mul]);

            let head = &g.heads[i];
            head.forward(&ctx, &self.ps, &sq);
            let map = Shape::new(1, 1, s.h, s.w);
            let mean = |buf: &DeviceBuffer| -> f64 {
                let (m, _) = imaging::mask::downsample(&img, buf, map, 1, 1);
                self.gpu.read(&m, 1)[0] as f64
            };
            layers[i] = match &mask {
                None => mean(head.out()),
                Some((m, full)) => {
                    let (cell, _) = imaging::mask::downsample(&img, m, *full, s.h, s.w);
                    let weighted = imaging::mask::intersect(&img, head.out(), &cell, map);
                    let covered = mean(&cell);
                    if covered <= 0.0 {
                        return Err(format!("lpips: the mask covers nothing on tap {i}'s {}x{} grid", s.w, s.h));
                    }
                    mean(&weighted) / covered
                }
            };
        }
        Ok(Distance { total: layers.iter().sum(), layers })
    }
}
