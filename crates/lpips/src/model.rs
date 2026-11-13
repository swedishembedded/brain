// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS v0.1 on the device, with either released trunk ([`Trunk`]: AlexNet, or VGG16 as the perceptual losses of
//! VQGAN-family trainers use it), composed entirely from existing kernels and blocks - this crate adds none:
//!
//! | LPIPS step (`lpips/lpips.py`) | how it runs here |
//! |---|---|
//! | `[0,1]` HWC -> NCHW | `imaging::Ctx::to_chw` (`nlc_nchw`) |
//! | `2x-1`, then `ScalingLayer` | one per-channel affine, `imaging::Ctx::affine` (`film_chan`) |
//! | the trunk's `features` | `vision::Conv` (bias, ReLU: `conv_bias_reg` + `leaky_relu`) and `vision::MaxPool` (`maxpool2d`); AlexNet taps every convolution, VGG16 the last of each block |
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
//! is zero in every channel; `tests/it/reference.rs` holds the device to an f64
//! transcription of the reference formula.
//!
//! # The gradient
//!
//! [`Lpips::distance_grad`] also differentiates the distance with respect to its input images, the trunk and the
//! heads frozen (what a perceptual loss needs of a metric it does not train). The reverse walks the same stages
//! backwards, each from a kernel or block that already has an adjoint:
//!
//! | stage | its adjoint here |
//! |---|---|
//! | `spatial_average` | a constant seed `2 / (h w)` per tap (the 2 is the square's) |
//! | head, 1x1 conv | `Conv::backward_input` (`conv2d_dx`) |
//! | `(f0 - f1)^2` | `mul` of the stored difference and the head's input gradient; `+d` into image 0, `-d` into image 1 (`axpy` on slices) |
//! | channel normalisation | `nchw_nlc`, `l2norm_scale_dx` (a unit gain), `nlc_nchw`: the NCHW normalisation is the row one with a layout swap either side |
//! | ReLU, conv | `Conv::backward_input` |
//! | max-pool | `MaxPool::backward` (`maxpool2d_dx`, a gather over the stored argmax) |
//! | the scaling layer | a per-channel `film_chan` by `1 / SCALE` |
//!
//! A tap's gradient and the one flowing back from the next layer meet in an `add2`. The gradient is unmasked: a mask
//! weights the reported distance only. Its entry point reads and writes planar `[-1, 1]` images (the range a generator
//! produces), where [`Lpips::distance`] reads interleaved `[0, 1]`.
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

use crate::config::{head_weight, Trunk, SCALE, SHIFT};
use crate::import::Tensors;

/// Every kernel the metric dispatches, by name.
pub const PIPELINES: &[(&str, &str)] = &FORWARD;

const FORWARD: [(&str, &str); 12] = [
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

/// The kernels only [`Lpips::distance_grad`] dispatches, behind [`PIPELINES`].
const BACKWARD: [(&str, &str); 6] = [
    ("conv2d_dx", kernels::CONV2D_DX),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
];

/// [`PIPELINES`] and the backward kernels, in that order: the set a device needs for [`Lpips::distance_grad`]. The
/// forward set is a prefix, so a kernel's position in one is its position in the other.
pub const GRAD_PIPELINES: &[(&str, &str)] = &concat();

const fn concat() -> [(&'static str, &'static str); FORWARD.len() + BACKWARD.len()] {
    let mut all = [("", ""); FORWARD.len() + BACKWARD.len()];
    let mut i = 0;
    while i < FORWARD.len() {
        all[i] = FORWARD[i];
        i += 1;
    }
    let mut j = 0;
    while j < BACKWARD.len() {
        all[FORWARD.len() + j] = BACKWARD[j];
        j += 1;
    }
    all
}

/// The epsilon `l2norm_scale2d` adds to the channel sum of squares: the
/// square of the reference's `1e-10` added to the norm (module docs).
const NORM_EPS: f32 = 1e-20;

/// An LPIPS distance: the sum over the five taps, and each tap's share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distance {
    pub total: f64,
    pub layers: [f64; 5],
}

/// One tap's buffers: the unit-normalised features of the pair, their difference and its square. Held so the
/// reverse reads what the forward wrote.
struct Tap {
    unit: DeviceBuffer,
    diff: DeviceBuffer,
    sq: DeviceBuffer,
}

/// The reverse's buffers, per tap or per convolution, allocated on the first gradient request.
struct Reverse {
    /// `2 / (h w)` over a tap's `[1, 1, h, w]` head output: the average's `1 / (h w)` and the square's 2.
    seed: Vec<DeviceBuffer>,
    /// The gradient wrt a head's input (`sq`), then wrt the difference.
    d_sq: Vec<DeviceBuffer>,
    d_diff: Vec<DeviceBuffer>,
    /// The gradient wrt the pair of unit-normalised features, and its channel-last twins around the normalisation's
    /// reverse.
    d_unit: Vec<DeviceBuffer>,
    rows_x: Vec<DeviceBuffer>,
    rows_dy: Vec<DeviceBuffer>,
    rows_dx: Vec<DeviceBuffer>,
    /// Per tap, the gradient wrt the tapped convolution's output from the tap itself, and its sum with the gradient
    /// from the next layer.
    d_tap: Vec<DeviceBuffer>,
    d_out: Vec<DeviceBuffer>,
    /// Per convolution, the gradient wrt its input.
    d_in: Vec<DeviceBuffer>,
    /// The scaling layer's `film_chan` parameters for the reverse (`1 / scale`, no shift) and the gradient wrt the
    /// `[-1, 1]` pair they produce: what [`Lpips::distance_grad`] returns.
    unscale: DeviceBuffer,
    d_pair: DeviceBuffer,
}

/// The trunk and heads for one input size. Rebuilt when the size changes:
/// `vision`'s blocks own buffers sized for their input.
struct Graph {
    w: u32,
    h: u32,
    /// Every trunk convolution, and the pool (if any) in front of each.
    convs: Vec<Conv>,
    pools: Vec<Option<MaxPool>>,
    /// Per tap: its head, its buffers, and the convolution it follows.
    heads: Vec<Conv>,
    taps: Vec<Tap>,
    tap_conv: [usize; 5],
    reverse: Option<Reverse>,
}

/// The LPIPS v0.1 metric, resident on one device.
pub struct Lpips {
    trunk: Trunk,
    gpu: Gpu,
    ids: ConvKernelIds,
    ps: ParamStore,
    ones: DeviceBuffer,
    k_sub: usize,
    k_l2: usize,
    /// Whether the device has [`GRAD_PIPELINES`]' backward kernels.
    gradients: bool,
    graph: Option<Graph>,
}

impl Lpips {
    /// Put `weights` ([`crate::import::read`]) on `gpu`, which must have been built with (or `new_like`'d to)
    /// [`PIPELINES`], or [`GRAD_PIPELINES`] to differentiate ([`Lpips::distance_grad`]). The metric runs on whatever
    /// device `gpu` is; it never moves itself elsewhere.
    pub fn new(gpu: Gpu, weights: &Tensors) -> Result<Lpips, String> {
        Lpips::with_trunk(gpu, Trunk::Alex, weights)
    }

    /// [`Lpips::new`] for either trunk; `weights` must be that trunk's ([`crate::import::read`]).
    pub fn with_trunk(gpu: Gpu, trunk: Trunk, weights: &Tensors) -> Result<Lpips, String> {
        crate::import::validate(trunk, weights)?;
        let kernel = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("lpips: kernel `{name}` is not registered on this device; build it with lpips::PIPELINES"));
        for (name, _) in PIPELINES {
            kernel(name)?;
        }
        let (k_sub, k_l2) = (kernel("sub")?, kernel("l2norm_scale2d")?);
        let gradients = BACKWARD.iter().all(|(name, _)| gpu.kernel_index(name).is_some());
        let mut roles: Vec<(String, usize, Role)> = weights.iter().map(|(n, (_, d))| (n.clone(), d.len(), Role::Frozen)).collect();
        roles.sort_by(|a, b| a.0.cmp(&b.0));
        let init = weights.iter().map(|(n, (_, d))| (n.clone(), d.clone())).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let widest = trunk.convs().iter().map(|c| c.cout).max().expect("a trunk has convolutions");
        let ones = gpu.storage_init("lpips.ones", &vec![1.0; widest as usize]);
        // The backward set extends the forward one, so its ids resolve against the longer list wherever the device
        // has it and every forward id is the same either way.
        let ids = ConvKernelIds::resolve(if gradients { GRAD_PIPELINES } else { PIPELINES });
        Ok(Lpips { trunk, ids, gpu, ps, ones, k_sub, k_l2, gradients, graph: None })
    }

    /// [`Lpips::new`] from the model store: the trunk and heads
    /// [`crate::spec::resolve`] finds, on a `new_like` of `gpu`. Built with [`GRAD_PIPELINES`], so it can differentiate.
    pub fn from_store(gpu: &Gpu) -> Result<Lpips, String> {
        Lpips::from_store_for(gpu, Trunk::Alex)
    }

    /// [`Lpips::from_store`] for either trunk; a missing file is refused by name ([`crate::spec::resolve_for`]).
    pub fn from_store_for(gpu: &Gpu, trunk: Trunk) -> Result<Lpips, String> {
        let (trunk_file, heads) = crate::spec::resolve_for(trunk)?;
        let weights = crate::import::read(trunk, &trunk_file, &heads)?;
        Lpips::with_trunk(gpu.new_like(GRAD_PIPELINES), trunk, &weights)
    }

    /// Which trunk this metric runs.
    pub fn trunk(&self) -> Trunk {
        self.trunk
    }

    /// The device the metric runs on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn build(&self, w: u32, h: u32) -> Graph {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let mut shape = Shape::new(2, 3, h, w);
        let (pool_k, pool_stride) = self.trunk.pool();
        let tap_conv = self.trunk.tap_convs();
        let (mut convs, mut pools, mut heads, mut taps) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, c) in self.trunk.convs().iter().enumerate() {
            let pool = c.pooled.then(|| MaxPool::new(&ctx, shape, PoolSpec::new(pool_k, pool_stride, 0)));
            if let Some(p) = &pool {
                shape = p.out_shape;
            }
            let names = ConvNames { weight: c.weight(), bias: c.bias(), ..ConvNames::torch_flat("") };
            let spec = ConvSpec { cout: c.cout, k: c.k, stride: c.stride, pad: c.pad, groups: 1, dilation: 1, norm: Norm::None, act: Act::Relu, bias: true };
            let conv = Conv::with_names(&ctx, &format!("features.{}", c.index), names, shape, spec, false);
            shape = conv.out_shape;
            if let Some(t) = tap_conv.iter().position(|&x| x == i) {
                let head_names = ConvNames { weight: head_weight(t), ..ConvNames::torch_flat("") };
                let head_spec = ConvSpec { cout: 1, k: 1, stride: 1, pad: 0, groups: 1, dilation: 1, norm: Norm::None, act: Act::None, bias: false };
                heads.push(Conv::with_names(&ctx, &format!("lin{t}"), head_names, Shape::new(1, c.cout, shape.h, shape.w), head_spec, false));
                let per = c.cout * shape.h * shape.w;
                taps.push(Tap { unit: ctx.act(shape.numel()), diff: ctx.act(per), sq: ctx.act(per) });
            }
            pools.push(pool);
            convs.push(conv);
        }
        Graph { w, h, convs, pools, heads, taps, tap_conv, reverse: None }
    }

    /// The reverse's buffers for `g`'s shapes.
    fn build_reverse(&self, g: &Graph) -> Reverse {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let mut r = Reverse {
            seed: vec![],
            d_sq: vec![],
            d_diff: vec![],
            d_unit: vec![],
            rows_x: vec![],
            rows_dy: vec![],
            rows_dx: vec![],
            d_tap: vec![],
            d_out: vec![],
            d_in: vec![],
            // `film_chan` computes `x * (1 + s) + b` from `[N, 2C]` rows of `[s.., b..]` (`imaging::Ctx::affine`).
            unscale: self.gpu.storage_init(
                "lpips.unscale",
                &(0..2).flat_map(|_| SCALE.iter().map(|s| 1.0 / s - 1.0).chain([0.0; 3])).collect::<Vec<f32>>(),
            ),
            d_pair: self.gpu.storage(2 * 3 * (g.h * g.w) as u64),
        };
        for conv in &g.convs {
            r.d_in.push(ctx.act(conv.in_shape.numel()));
        }
        for &c in &g.tap_conv {
            let s = g.convs[c].out_shape;
            let (per, hw) = (s.c * s.h * s.w, s.h * s.w);
            r.seed.push(self.gpu.storage_init("lpips.seed", &vec![2.0 / hw as f32; hw as usize]));
            r.d_sq.push(ctx.act(per));
            r.d_diff.push(ctx.act(per));
            r.d_unit.push(ctx.act(s.numel()));
            r.rows_x.push(ctx.act(s.numel()));
            r.rows_dy.push(ctx.act(s.numel()));
            r.rows_dx.push(ctx.act(s.numel()));
            r.d_tap.push(ctx.act(s.numel()));
            r.d_out.push(ctx.act(s.numel()));
        }
        r
    }

    /// The graph for `w x h`, built on first use and after a size change.
    fn graph_for(&mut self, w: u32, h: u32) -> Result<(), String> {
        let min = self.trunk.min_side();
        if w < min || h < min {
            return Err(format!("lpips: {w}x{h} is smaller than the {min}x{min} the {} trunk needs", self.trunk.name()));
        }
        if self.graph.as_ref().is_none_or(|g| (g.w, g.h) != (w, h)) {
            self.graph = None;
            self.gpu.poll_wait();
            self.graph = Some(self.build(w, h));
        }
        Ok(())
    }

    /// The LPIPS distance between `a` and `b`, interleaved RGB in `[0, 1]`,
    /// `w x h` each. `mask`, one weight per pixel, weights the spatial
    /// average (module docs); `None` is the reference metric.
    pub fn distance(&mut self, a: &[f32], b: &[f32], w: u32, h: u32, mask: Option<&[f32]>) -> Result<Distance, String> {
        let px = w as usize * h as usize;
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
        self.graph_for(w, h)?;
        let result = self.run(a, b, w, h, mask);
        self.gpu.poll_wait();
        result
    }

    /// The distance of the pair `pair` (`[2, 3, h, w]`, planar RGB in `[-1, 1]`: image `a`, then image `b`, on this
    /// metric's device) and its gradient with respect to the pair, a `[2, 3, h, w]` buffer in the same layout and
    /// range, valid until the next call. The reference is unmasked. Needs a device built with [`GRAD_PIPELINES`].
    pub fn distance_grad(&mut self, pair: &DeviceBuffer, w: u32, h: u32) -> Result<(Distance, &DeviceBuffer), String> {
        if !self.gradients {
            return Err("lpips: differentiating needs a device built with lpips::GRAD_PIPELINES".to_string());
        }
        self.graph_for(w, h)?;
        let mut graph = self.graph.take().expect("built by graph_for");
        if graph.reverse.is_none() {
            graph.reverse = Some(self.build_reverse(&graph));
        }
        let shape = Shape::new(2, 3, h, w);
        let img = imaging::Ctx::new(&self.gpu);
        // `(x - shift) / scale` on an image already in [-1, 1].
        let scale: Vec<f32> = SCALE.iter().map(|s| 1.0 / s).collect();
        let shift: Vec<f32> = SHIFT.iter().zip(&SCALE).map(|(t, s)| -t / s).collect();
        let x0 = img.affine(pair, shape, &scale, &shift);
        let layers = self.taps(&graph, &x0, None);
        if layers.is_ok() {
            self.reverse(&graph, &x0);
        }
        self.gpu.poll_wait();
        self.graph = Some(graph);
        let layers = layers?;
        let reverse = self.graph.as_ref().and_then(|g| g.reverse.as_ref()).expect("put back above");
        Ok((Distance { total: layers.iter().sum(), layers }, &reverse.d_pair))
    }

    fn run(&self, a: &[f32], b: &[f32], w: u32, h: u32, mask: Option<&[f32]>) -> Result<Distance, String> {
        let g = self.graph.as_ref().expect("built by distance");
        let img = imaging::Ctx::new(&self.gpu);
        let input = Shape::new(2, 3, h, w);

        let hwc = self.gpu.storage(input.numel() as u64);
        self.gpu.write_f32_at(&hwc, 0, a);
        self.gpu.write_f32_at(&hwc, a.len() as u64, b);
        let chw = img.to_chw(&hwc, input);
        // (2x - 1 - shift) / scale = x * (2 / scale) + (-1 - shift) / scale
        let scale: Vec<f32> = SCALE.iter().map(|s| 2.0 / s).collect();
        let shift: Vec<f32> = SHIFT.iter().zip(&SCALE).map(|(t, s)| (-1.0 - t) / s).collect();
        let x = img.affine(&chw, input, &scale, &shift);

        let mask = mask.map(|m| (img.upload("lpips.mask", m), Shape::new(1, 1, h, w)));
        let layers = self.taps(g, &x, mask.as_ref().map(|(m, s)| (m, *s)))?;
        Ok(Distance { total: layers.iter().sum(), layers })
    }

    /// The five taps' distances of the scaled pair `x`, leaving every intermediate in `g` for the reverse.
    fn taps(&self, g: &Graph, x: &DeviceBuffer, mask: Option<(&DeviceBuffer, Shape)>) -> Result<[f64; 5], String> {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let img = imaging::Ctx::new(&self.gpu);
        let mut x = x.clone();
        let mut layers = [0.0f64; 5];
        for (i, conv) in g.convs.iter().enumerate() {
            if let Some(p) = &g.pools[i] {
                p.forward(&ctx, &x);
                x = p.out().clone();
            }
            conv.forward(&ctx, &self.ps, &x);
            x = conv.out().clone();
            let Some(t) = g.tap_conv.iter().position(|&c| c == i) else { continue };
            let s = conv.out_shape;
            let (per, hw) = (s.c * s.h * s.w, s.h * s.w);

            // Unit-normalize both images' features over channels, then the
            // squared difference of image 0 and image 1.
            let tap = &g.taps[t];
            let norm = self.gpu.step(self.k_l2, &[&x, &self.ones, &tap.unit], &[s.n, s.c, hw, NORM_EPS.to_bits()], s.n * hw);
            let sub = self.gpu.step(self.k_sub, &[&tap.unit, &tap.unit, &tap.diff], &[per, 0, per], per);
            let mul = self.gpu.step(self.ids.need(self.ids.mul, "mul"), &[&tap.diff, &tap.diff, &tap.sq], &[per], per);
            self.gpu.submit(&[], &[norm, sub, mul]);

            let head = &g.heads[t];
            head.forward(&ctx, &self.ps, &tap.sq);
            let map = Shape::new(1, 1, s.h, s.w);
            let mean = |buf: &DeviceBuffer| -> f64 {
                let (m, _) = imaging::mask::downsample(&img, buf, map, 1, 1);
                self.gpu.read(&m, 1)[0] as f64
            };
            layers[t] = match &mask {
                None => mean(head.out()),
                Some((m, full)) => {
                    let (cell, _) = imaging::mask::downsample(&img, m, *full, s.h, s.w);
                    let weighted = imaging::mask::intersect(&img, head.out(), &cell, map);
                    let covered = mean(&cell);
                    if covered <= 0.0 {
                        return Err(format!("lpips: the mask covers nothing on tap {t}'s {}x{} grid", s.w, s.h));
                    }
                    mean(&weighted) / covered
                }
            };
        }
        Ok(layers)
    }

    /// Tap `t`'s own gradient wrt its convolution's output, into `d_tap[t]`: through the head, the square, the
    /// difference of the pair and the channel normalisation.
    fn tap_reverse(&self, g: &Graph, r: &Reverse, t: usize, (k_nchw_nlc, k_nlc_nchw, k_dx, k_axpy): (usize, usize, usize, usize)) {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let (conv, tap) = (&g.convs[g.tap_conv[t]], &g.taps[t]);
        let s = conv.out_shape;
        let (per, hw) = (s.c * s.h * s.w, s.h * s.w);
        let total = s.numel();

        // The head, then the square: d_diff = diff * (2 / (h w)) * (head's input gradient).
        g.heads[t].backward_input(&ctx, &self.ps, &tap.sq, &r.seed[t], &r.d_sq[t]);
        let square = self.gpu.step(self.ids.need(self.ids.mul, "mul"), &[&tap.diff, &r.d_sq[t], &r.d_diff[t]], &[per], per);
        // The difference of the pair's two images: +d into image 0, -d into image 1.
        let to_a = self.gpu.step_sliced(k_axpy, &[&r.d_unit[t], &r.d_diff[t]], &[(0, per as u64), (0, per as u64)], &[per, gpu_core::f(1.0)], per);
        let to_b = self.gpu.step_sliced(k_axpy, &[&r.d_unit[t], &r.d_diff[t]], &[(per as u64, per as u64), (0, per as u64)], &[per, gpu_core::f(-1.0)], per);
        self.gpu.submit(&[], &[square]);
        self.gpu.submit(&[&r.d_unit[t]], &[to_a, to_b]);

        // The channel normalisation, as its row form between two layout swaps.
        let x = conv.out();
        let swap = |kind, from: &DeviceBuffer, to: &DeviceBuffer| self.gpu.step(kind, &[from, to], &[total, s.c, hw], total);
        let rows = s.n * hw;
        let reverse_norm = self.gpu.step(k_dx, &[&r.rows_x[t], &self.ones, &r.rows_dy[t], &r.rows_dx[t]], &[rows, s.c, NORM_EPS.to_bits()], total);
        self.gpu.submit(
            &[],
            &[swap(k_nchw_nlc, x, &r.rows_x[t]), swap(k_nchw_nlc, &r.d_unit[t], &r.rows_dy[t]), reverse_norm, swap(k_nlc_nchw, &r.rows_dx[t], &r.d_tap[t])],
        );
    }

    /// The reverse of [`Lpips::taps`] (unmasked) after its forward: the gradient wrt the `[-1, 1]` pair, in the
    /// buffer it returns. See the module docs for each stage's adjoint.
    fn reverse(&self, g: &Graph, x0: &DeviceBuffer) {
        let ctx = Ctx::new(&self.gpu, &self.ids);
        let r = g.reverse.as_ref().expect("built by distance_grad");
        let kernel = |name: &str| self.gpu.kernel_index(name).expect("checked by `gradients`");
        let (k_nchw_nlc, k_nlc_nchw, k_dx, k_add2, k_axpy) = (kernel("nchw_nlc"), kernel("nlc_nchw"), kernel("l2norm_scale_dx"), kernel("add2"), kernel("axpy"));
        let n = g.convs.len();
        for i in (0..n).rev() {
            let conv = &g.convs[i];
            // What the next convolution (or the pool in front of it) sends back to this one's output.
            let from_next = (i + 1 < n).then(|| match &g.pools[i + 1] {
                Some(p) => p.d_in(),
                None => &r.d_in[i + 1],
            });
            let s = conv.out_shape;
            let total = s.numel();
            let d_out = match g.tap_conv.iter().position(|&c| c == i) {
                // A tapped convolution also receives its own tap's gradient.
                Some(t) => {
                    self.tap_reverse(g, r, t, (k_nchw_nlc, k_nlc_nchw, k_dx, k_axpy));
                    match from_next {
                        Some(next) => {
                            self.gpu.submit(&[], &[self.gpu.step(k_add2, &[&r.d_tap[t], next, &r.d_out[t]], &[total], total)]);
                            &r.d_out[t]
                        }
                        None => &r.d_tap[t],
                    }
                }
                None => from_next.expect("the last convolution is a tap"),
            };

            // The convolution, then the pool in front of it.
            let x_in = match &g.pools[i] {
                Some(p) => p.out(),
                None if i == 0 => x0,
                None => g.convs[i - 1].out(),
            };
            conv.backward_input(&ctx, &self.ps, x_in, d_out, &r.d_in[i]);
            if let Some(p) = &g.pools[i] {
                p.backward(&ctx, &r.d_in[i]);
            }
        }
        // The scaling layer's `1 / scale` per channel, back to the image's own units.
        let total = 2 * 3 * g.h * g.w;
        let unscale = self.gpu.step(kernel("film_chan"), &[&r.d_in[0], &r.unscale, &r.d_pair], &[2, 3, g.h, g.w], total);
        self.gpu.submit(&[], &[unscale]);
    }
}
