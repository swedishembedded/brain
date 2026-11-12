// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forward pass, composed from `crates/vision`'s shared blocks: every
//! conv is a [`Conv`] (BatchNorm already folded into its bias, so
//! `Norm::None` with a bias), squeeze-excite is [`SqueezeExcite`], the skip
//! concat is [`Cat`] and the detector's peak pool is [`MaxPool`]. The only
//! kernels dispatched directly are the residual `add2` and the decoder's
//! corner-aligned `resize_bilinear`.
//!
//! SSA: every stage writes a fresh buffer; nothing is computed in place.
//!
//! The forward is RECORDED once and replayed: a frame differs from the last
//! only in the input buffer's contents, and the blocks submit each stage
//! eagerly, so building and submitting a couple of hundred small dispatches
//! per frame would cost more than running them. The first
//! [`Net::forward`] records the whole pass on a `gpu_core` tape; every
//! later one rewrites the input and replays the tape as one submission.

use std::path::Path;
use std::sync::OnceLock;

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::ParamStore;
use vision::{Act, Cat, Conv, ConvKernelIds, ConvSpec, Ctx, MaxPool, Norm, PoolSpec, SeSpec, Shape, SqueezeExcite};

use crate::config::{Block, Kind, NetConfig};

/// Every kernel the networks dispatch, by name; the blocks resolve their own
/// ids from it.
///
/// The `_bwd` halves are registered only because `vision`'s activation seam
/// resolves an activation's forward and backward together; nothing here runs
/// a backward.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("hardswish", kernels::HARDSWISH),
    ("hardswish_bwd", kernels::HARDSWISH_BWD),
    ("hardsigmoid", kernels::HARDSIGMOID),
    ("hardsigmoid_bwd", kernels::HARDSIGMOID_BWD),
    ("relu6", kernels::RELU6),
    ("relu6_bwd", kernels::RELU6_BWD),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("scale_chan", kernels::SCALE_CHAN),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("add2", kernels::ADD2),
];

/// Kernel ids resolved once against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

fn kernel(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("kernel `{name}` is not in openseeface::PIPELINES"))
}

/// Whether a block adds its input to its output: stride 1, same width, and
/// not the head's pointwise conv.
pub fn residual(b: &Block, cin: u32) -> bool {
    b.kind != Kind::Cn && b.stride == 1 && b.out == cin
}

fn spec(cout: u32, k: u32, stride: u32, groups: u32, act: Act) -> ConvSpec {
    ConvSpec { cout, k, stride, pad: k / 2, groups, dilation: 1, norm: Norm::None, act, bias: true }
}

enum Unit {
    Conv(Box<Conv>),
    Se(Box<SqueezeExcite>),
}

impl Unit {
    fn out(&self) -> &DeviceBuffer {
        match self {
            Unit::Conv(c) => c.out(),
            Unit::Se(s) => s.out(),
        }
    }
    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        match self {
            Unit::Conv(c) => c.forward(ctx, ps, x),
            Unit::Se(s) => s.forward(ctx, ps, x),
        }
    }
}

struct BlockNet {
    units: Vec<Unit>,
    /// The residual sum, when the block has one.
    sum: Option<DeviceBuffer>,
    out_shape: Shape,
}

impl BlockNet {
    fn out(&self) -> &DeviceBuffer {
        self.sum.as_ref().unwrap_or_else(|| self.units.last().expect("a block has units").out())
    }
}

struct UpNet {
    resized: DeviceBuffer,
    resized_shape: Shape,
    in_shape: Shape,
    skip: usize,
    cat: Cat,
    dw: Conv,
    pw: Conv,
}

/// One released network, resident on a device.
pub struct Net {
    pub cfg: NetConfig,
    gpu: Gpu,
    ps: ParamStore,
    input: DeviceBuffer,
    stem: Conv,
    /// Blocks per stage, in order.
    stages: Vec<Vec<BlockNet>>,
    ups: Vec<UpNet>,
    head: (Conv, Conv),
    pool: Option<MaxPool>,
    /// The recorded forward, after the first call.
    tape: std::sync::Mutex<Option<gpu_core::tape::Tape>>,
}

impl Net {
    /// Import `dir/<cfg.file>` and build the network on `device` (shared,
    /// per the one-device-per-process rule: the detector and the landmark
    /// network of one tracker run on the same device).
    pub fn load(device: &Gpu, dir: &Path, cfg: NetConfig) -> Result<Net, String> {
        let t = crate::import::import(&dir.join(cfg.file), &cfg)?;
        Ok(Net::with_weights(device.share_or_new(PIPELINES), cfg, &t))
    }

    /// Build on `gpu`, which must carry [`PIPELINES`], from imported tensors.
    pub fn with_weights(gpu: Gpu, cfg: NetConfig, t: &onnx::walk::Tensors) -> Net {
        let ps = ParamStore::frozen(&gpu, t, cfg.file);
        let ctx = Ctx::new(&gpu, ids());
        let n = cfg.input;
        let in_shape = Shape::new(1, 3, n, n);
        let input = gpu.storage(in_shape.numel() as u64);
        let stem = Conv::with_spec(&ctx, "stem", in_shape, spec(cfg.stem, 3, 2, 1, cfg.stem_act), false);
        let mut shape = stem.out_shape;
        let mut stages = Vec::new();
        for (s, stage) in cfg.stages.iter().enumerate() {
            let mut blocks = Vec::new();
            for (b, blk) in stage.iter().enumerate() {
                let p = format!("blocks.{s}.{b}");
                let cin = shape.c;
                let mut units = Vec::new();
                let push_conv = |units: &mut Vec<Unit>, name: &str, x: Shape, sp: ConvSpec| -> Shape {
                    let c = Conv::with_spec(&ctx, &format!("{p}.{name}"), x, sp, false);
                    let o = c.out_shape;
                    units.push(Unit::Conv(Box::new(c)));
                    o
                };
                let se = |units: &mut Vec<Unit>, x: Shape| {
                    if blk.se > 0 {
                        units.push(Unit::Se(Box::new(SqueezeExcite::new(&ctx, &format!("{p}.se"), x, SeSpec::mobilenet_v3(blk.se)))));
                    }
                };
                let out = match blk.kind {
                    Kind::Ds => {
                        let x = push_conv(&mut units, "dw", shape, spec(cin, blk.k, blk.stride, cin, blk.act));
                        se(&mut units, x);
                        push_conv(&mut units, "pw", x, spec(blk.out, 1, 1, 1, Act::None))
                    }
                    Kind::Ir => {
                        let x = push_conv(&mut units, "pw", shape, spec(blk.mid, 1, 1, 1, blk.act));
                        let x = push_conv(&mut units, "dw", x, spec(blk.mid, blk.k, blk.stride, blk.mid, blk.act));
                        se(&mut units, x);
                        push_conv(&mut units, "pwl", x, spec(blk.out, 1, 1, 1, Act::None))
                    }
                    Kind::Cn => push_conv(&mut units, "conv", shape, spec(blk.out, 1, 1, 1, blk.act)),
                };
                let sum = residual(blk, cin).then(|| gpu.storage(out.numel() as u64));
                blocks.push(BlockNet { units, sum, out_shape: out });
                shape = out;
            }
            stages.push(blocks);
        }
        let stage_shape = |s: usize| stages[s].last().expect("a stage has blocks").out_shape;
        let mut ups = Vec::new();
        for (k, up) in cfg.ups.iter().enumerate() {
            let skip = stage_shape(up.skip);
            let resized_shape = Shape::new(1, shape.c, skip.h, skip.w);
            let cat = Cat::new(&ctx, skip, resized_shape);
            let cs = cat.out_shape();
            let dw = Conv::with_spec(&ctx, &format!("up{}.dw", k + 1), cs, spec(cs.c, 3, 1, cs.c, up.dw_act), false);
            let pw = Conv::with_spec(&ctx, &format!("up{}.pw", k + 1), dw.out_shape, spec(up.out, 1, 1, 1, Act::Relu6), false);
            let next = pw.out_shape;
            ups.push(UpNet { resized: gpu.storage(resized_shape.numel() as u64), resized_shape, in_shape: shape, skip: up.skip, cat, dw, pw });
            shape = next;
        }
        let h = cfg.head;
        let mid = Conv::with_spec(&ctx, "head.mid", shape, spec(h.mid, 3, 1, h.groups, h.mid_act), false);
        let out = Conv::with_spec(&ctx, "head.out", mid.out_shape, spec(h.out, 1, 1, h.out_groups, h.out_act), false);
        let pool = cfg.peak_pool.then(|| MaxPool::new(&ctx, out.out_shape, PoolSpec::new(3, 1, 1)));
        drop(ctx);
        Net { cfg, gpu, ps, input, stem, stages, ups, head: (mid, out), pool, tape: std::sync::Mutex::new(None) }
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The output's shape: `[1, C, H, W]`.
    pub fn out_shape(&self) -> Shape {
        self.head.1.out_shape
    }

    /// Run on one normalized `[1, 3, input, input]` tensor (NCHW, RGB,
    /// ImageNet-normalized - see [`crate::decode::normalize`]).
    pub fn forward(&self, x: &[f32]) {
        let n = self.cfg.input as usize;
        assert_eq!(x.len(), 3 * n * n, "openseeface: input must be 3x{n}x{n}");
        self.gpu.write_f32(&self.input, x);
        let mut tape = self.tape.lock().unwrap_or_else(|e| e.into_inner());
        if tape.is_none() {
            self.gpu.begin_tape();
            self.record();
            *tape = Some(self.gpu.end_tape());
        }
        self.gpu.replay_tape(tape.as_ref().expect("recorded above"));
    }

    /// Every dispatch of one forward pass, in order. Reads only the input
    /// buffer, so a recording of it is valid for any input.
    fn record(&self) {
        let ctx = Ctx::new(&self.gpu, ids());
        let ps = &self.ps;
        self.stem.forward(&ctx, ps, &self.input);
        let mut x: &DeviceBuffer = self.stem.out();
        let mut stage_out: Vec<&DeviceBuffer> = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for b in stage {
                let mut y = x;
                for u in &b.units {
                    u.forward(&ctx, ps, y);
                    y = u.out();
                }
                if let Some(sum) = &b.sum {
                    let t = b.out_shape.numel();
                    let s = ctx.step(kernel("add2"), &[y, x, sum], &[t], t);
                    self.gpu.submit(&[], &[s]);
                }
                x = b.out();
            }
            stage_out.push(x);
        }
        for u in &self.ups {
            let (i, o) = (u.in_shape, u.resized_shape);
            let s = ctx.step(kernel("resize_bilinear"), &[x, &u.resized], &[1, i.c, i.h, i.w, o.h, o.w, 1], o.numel());
            self.gpu.submit(&[], &[s]);
            u.cat.forward(&ctx, stage_out[u.skip], &u.resized);
            u.dw.forward(&ctx, ps, &u.cat.out);
            u.pw.forward(&ctx, ps, u.dw.out());
            x = u.pw.out();
        }
        self.head.0.forward(&ctx, ps, x);
        self.head.1.forward(&ctx, ps, self.head.0.out());
        if let Some(p) = &self.pool {
            p.forward(&ctx, self.head.1.out());
        }
    }

    /// The output map after [`Net::forward`], `[C, H, W]` flattened.
    pub fn read_output(&self) -> Vec<f32> {
        self.gpu.read(self.head.1.out(), self.out_shape().numel() as usize)
    }

    /// The detector's 3x3 max-pooled output, for peak picking.
    pub fn read_pooled(&self) -> Option<Vec<f32>> {
        self.pool.as_ref().map(|p| self.gpu.read(p.out(), p.out_shape.numel() as usize))
    }

    /// Stage taps for parity bisection, named as the goldens name them:
    /// `stem`, `blocks.<s>.<b>`, `up<k>`.
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        let read = |b: &DeviceBuffer, s: Shape| self.gpu.read(b, s.numel() as usize);
        if name == "stem" {
            return Some(read(self.stem.out(), self.stem.out_shape));
        }
        if let Some(k) = name.strip_prefix("up").and_then(|k| k.parse::<usize>().ok()) {
            return self.ups.get(k.checked_sub(1)?).map(|u| read(u.pw.out(), u.pw.out_shape));
        }
        let mut it = name.strip_prefix("blocks.")?.split('.');
        let (s, b) = (it.next()?.parse::<usize>().ok()?, it.next()?.parse::<usize>().ok()?);
        let blk = self.stages.get(s)?.get(b)?;
        Some(read(blk.out(), blk.out_shape))
    }
}
