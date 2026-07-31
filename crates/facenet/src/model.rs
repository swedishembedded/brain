// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward graphs: the ArcFace IResNet-100 embedding and the SCRFD detector.
//!
//! Both are composed from the SHARED conv-net blocks in `crates/vision`
//! (`Conv`, `BatchNorm`, `PReLU`, `MaxPool`, `AvgPool`) over the shared
//! kernel-id seam (`ConvKernelIds::resolve`, by NAME) — this crate adds no conv,
//! no norm, no pool and no activation of its own.
//!
//! SSA discipline (`AGENTS.md`): every stage writes a FRESH buffer, which is
//! also the activation cache the deferred backward will read. Nothing is
//! computed in place.
//!
//! # Both networks are BatchNorm-FOLDED
//!
//! The released ONNX graphs have their BatchNorms folded into the preceding
//! convolutions, so nearly every conv here is `Norm::None` **with a bias**, and
//! the activation follows directly. Only ArcFace keeps three real BatchNorms
//! (each block's entry `bn1`, the final `bn2`, and `features` after `fc`); SCRFD
//! keeps none. Modelling either from its *torch* module definition would need
//! weights that are not in the file.

use std::sync::OnceLock;

use gpu_core::{f, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{
    Act, AvgPool, BatchNorm, BnNames, Conv, ConvKernelIds, ConvNames, ConvSpec, Ctx, MaxPool, Norm,
    PoolSpec, PReLU, Shape,
};

use crate::config::{ArcFaceConfig, ScrfdConfig};
use crate::import::Tensors;

/// Every kernel the two forward graphs dispatch, by name.
///
/// The blocks resolve their own ids from this list via
/// [`ConvKernelIds::resolve`], so the ORDER here is free — that is the whole
/// point of the name-resolved seam. The backward-only kernels (`*_dx`, `*_dw`,
/// `prelu_bwd*`) are registered even though this workflow is forward-only:
/// leaving them out would make the deferred trainer fail at its first dispatch
/// with a `not registered` panic rather than at build time, and they cost
/// nothing but a pipeline slot.
pub const PIPELINES: &[(&str, &str)] = &[
    // conv (dense; every conv in both models has groups=1, dilation=1)
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    // batchnorm (ArcFace's bn1 / bn2 / features)
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    // activations. ReLU = leaky_relu at slope 0 (identical in both directions).
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    // PReLU: a LEARNED per-channel slope. Both backward variants — selecting
    // between them on `DeviceCaps::workgroup_reductions` is a correctness gate.
    ("prelu", kernels::PRELU),
    ("prelu_bwd", kernels::PRELU_BWD),
    ("prelu_bwd_wg", kernels::PRELU_BWD_WG),
    // spatial
    ("maxpool2d", kernels::MAXPOOL2D),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),
    ("resize_nearest", kernels::RESIZE_NEAREST),
    ("resize_nearest_dx", kernels::RESIZE_NEAREST_DX),
    // the 5-point alignment warp (see `crate::align`)
    ("grid_sample", kernels::GRID_SAMPLE),
    // elementwise / linear
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    // preprocessing: brain's one per-channel affine, dispatched via `imaging::Ctx`.
    ("film_chan", kernels::FILM_CHAN),
];

/// Kernel ids resolved once against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// The pipeline index of a kernel this crate dispatches directly (i.e. one with
/// no block in `crates/vision` behind it: `matmul`, `bias_add`, `axpy`,
/// `sigmoid`, `grid_sample`). Resolved by NAME against [`PIPELINES`] for the
/// same reason [`ConvKernelIds`] is — a bare index means nothing outside the
/// list that declared it.
pub fn kernel(name: &str) -> usize {
    PIPELINES
        .iter()
        .position(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("kernel `{name}` is not in facenet::PIPELINES"))
}

/// Build a frozen (inference) ParamStore from an imported tensor map, checking
/// every tensor's element count against the manifest that produced it.
fn frozen_store(gpu: &Gpu, t: &Tensors) -> ParamStore {
    let mut init = std::collections::HashMap::with_capacity(t.len());
    let mut roles = Vec::with_capacity(t.len());
    for (name, (shape, data)) in t {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "tensor {name}: {} values for shape {shape:?}", data.len());
        roles.push((name.clone(), n, Role::Frozen));
        init.insert(name.clone(), data.clone());
    }
    roles.sort_by(|a, b| a.0.cmp(&b.0));
    ParamStore::new_with_roles(gpu, roles, &init)
}

/// A bias-free conv spec with a learned bias and no norm — the shape EVERY conv
/// in both folded graphs takes.
fn folded(cout: u32, k: u32, stride: u32, pad: u32, act: Act) -> ConvSpec {
    ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::None, act, bias: true }
}

// ===========================================================================
// ArcFace / IResNet-100
// ===========================================================================

/// One IResNet residual block:
/// `y = conv2(prelu(conv1(bn1(x)))) + (downsample(x) or x)`.
///
/// Note that the shortcut reads the block's **input**, not `bn1`'s output — a
/// pre-activation ("v3") residual. Feeding it `bn1(x)` instead still runs, still
/// produces plausible features, and is wrong; it is visible only against the
/// `s{n}b0_branch` golden.
struct ResBlock {
    bn1: BatchNorm,
    conv1: Conv,
    prelu: PReLU,
    conv2: Conv,
    downsample: Option<Conv>,
    out: DeviceBuffer,
    out_shape: Shape,
}

impl ResBlock {
    fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, stride: u32) -> ResBlock {
        let bn1 = BatchNorm::new(ctx, BnNames::torch(&format!("{prefix}.bn1")), in_shape, false);
        bn1.set_eval(true);
        let conv1 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv1"),
            ConvNames::torch_flat(&format!("{prefix}.conv1")),
            in_shape,
            folded(cout, 3, 1, 1, Act::None),
            false,
        );
        let prelu = PReLU::new(ctx, &format!("{prefix}.prelu"), conv1.out_shape);
        let conv2 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv2"),
            ConvNames::torch_flat(&format!("{prefix}.conv2")),
            conv1.out_shape,
            folded(cout, 3, stride, 1, Act::None),
            false,
        );
        let downsample = if stride != 1 || in_shape.c != cout {
            Some(Conv::with_names(
                ctx,
                &format!("{prefix}.downsample"),
                ConvNames::torch_flat(&format!("{prefix}.downsample")),
                in_shape,
                folded(cout, 1, stride, 0, Act::None),
                false,
            ))
        } else {
            None
        };
        let out_shape = conv2.out_shape;
        ResBlock { bn1, conv1, prelu, conv2, downsample, out: ctx.act(out_shape.numel()), out_shape }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.bn1.param_list();
        v.extend(self.conv1.param_list());
        v.extend(self.prelu.param_list());
        v.extend(self.conv2.param_list());
        if let Some(d) = &self.downsample {
            v.extend(d.param_list());
        }
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> &DeviceBuffer {
        self.bn1.forward(ctx, ps, x);
        self.conv1.forward(ctx, ps, self.bn1.out());
        self.prelu.forward(ctx, ps, self.conv1.out());
        self.conv2.forward(ctx, ps, self.prelu.out());
        let ident: &DeviceBuffer = match &self.downsample {
            Some(d) => {
                d.forward(ctx, ps, x);
                d.out()
            }
            None => x,
        };
        let n = self.out_shape.numel();
        let s = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[self.conv2.out(), ident, &self.out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        &self.out
    }
}

/// Per-stage taps the parity test replays against. Named for the goldens
/// (`stem`, `layer1..4`, `bn2`, `flatten`, `fc`, `embedding`) so the test is a
/// pure lookup rather than a translation table.
#[derive(Clone, Debug, Default)]
pub struct ArcFaceTaps {
    pub stem: Vec<f32>,
    pub layers: [Vec<f32>; 4],
    /// Every residual `Add` output, in block order — the bisection ladder.
    pub blocks: Vec<Vec<f32>>,
    pub bn2: Vec<f32>,
    pub fc: Vec<f32>,
    /// The graph's raw 512-d output (the `features` BN). NOT L2-normalised.
    pub embedding: Vec<f32>,
    /// First-block internals of each stage, in `[bn_in, conv1, prelu, conv2,
    /// branch]` order — the goldens' `s{n}b0_*` taps.
    pub stage_b0: [[Vec<f32>; 5]; 4],
}

/// Inference-only ArcFace IResNet-100 embedder.
pub struct ArcFace {
    gpu: Gpu,
    cfg: ArcFaceConfig,
    ps: ParamStore,
    input: DeviceBuffer,
    stem_conv: Conv,
    stem_prelu: PReLU,
    blocks: Vec<ResBlock>,
    /// `(stage, index of the stage's LAST block in `blocks`)`.
    stage_last: [usize; 4],
    bn2: BatchNorm,
    fc_out: DeviceBuffer,
    features: BatchNorm,
}

impl ArcFace {
    /// Build on a shared device handle (`Gpu::share` / `testgpu::dev`) from an
    /// imported tensor map.
    pub fn new(gpu: Gpu, cfg: ArcFaceConfig, weights: &Tensors) -> ArcFace {
        let ids = ids();
        let ctx = Ctx::new(&gpu, ids);
        let sz = cfg.image_size;
        let in_shape = Shape::new(1, 3, sz, sz);

        let stem_conv = Conv::with_names(
            &ctx,
            "stem.conv",
            ConvNames::torch_flat("stem.conv"),
            in_shape,
            folded(cfg.stem_channels, 3, 1, 1, Act::None),
            false,
        );
        let stem_prelu = PReLU::new(&ctx, "stem.prelu", stem_conv.out_shape);

        let mut shape = stem_conv.out_shape;
        let mut blocks: Vec<ResBlock> = Vec::new();
        let mut stage_last = [0usize; 4];
        for s in 0..4usize {
            for b in 0..cfg.layers[s] as usize {
                let stride = if b == 0 { 2 } else { 1 };
                let blk = ResBlock::new(&ctx, &format!("layer{}.{}", s + 1, b), shape, cfg.channels[s], stride);
                shape = blk.out_shape;
                blocks.push(blk);
            }
            stage_last[s] = blocks.len() - 1;
        }

        let bn2 = BatchNorm::new(&ctx, BnNames::torch("bn2"), shape, false);
        bn2.set_eval(true);
        let e = cfg.embedding;
        let fc_out = ctx.act(e);
        // The `features` BN runs over a flat [1, 512] vector: N=1, C=512, H=W=1.
        let features = BatchNorm::new(&ctx, BnNames::torch("features"), Shape::new(1, e, 1, 1), false);
        features.set_eval(true);

        let mut params: Vec<(String, usize)> = stem_conv.param_list();
        params.extend(stem_prelu.param_list());
        for b in &blocks {
            params.extend(b.param_list());
        }
        params.extend(bn2.param_list());
        params.push(("fc.weight".into(), (e * cfg.flatten()) as usize));
        params.push(("fc.bias".into(), e as usize));
        params.extend(features.param_list());

        // The graph's own parameter list vs the import manifest: two independently
        // derived lists of the same 462 tensors. Comparing them here catches a
        // block wired at the wrong width before a single kernel runs.
        assert_eq!(
            params.len(),
            weights.len(),
            "arcface: the graph reads {} tensors but the checkpoint has {}",
            params.len(),
            weights.len()
        );
        for (name, n) in &params {
            match weights.get(name) {
                None => panic!("arcface: checkpoint is missing {name}"),
                Some((shape, _)) => {
                    let have: usize = shape.iter().product();
                    assert_eq!(have, *n, "arcface: {name} has {have} values, the graph wants {n}");
                }
            }
        }

        let ps = frozen_store(&gpu, weights);
        let input = gpu.storage(in_shape.numel() as u64);
        ArcFace { gpu, cfg, ps, input, stem_conv, stem_prelu, blocks, stage_last, bn2, fc_out, features }
    }

    pub fn config(&self) -> &ArcFaceConfig {
        &self.cfg
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, ids())
    }

    /// Run the network on an NCHW `[1, 3, 112, 112]` blob and return the raw
    /// 512-d embedding (not normalised — the graph has no normalisation).
    ///
    /// Captures no taps: reading them back is ~1.9 M floats and 60 device syncs
    /// per image, and computing them only to drop them would put the parity
    /// ladder's instrumentation in the production path.
    pub fn embed_blob(&self, blob: &[f32]) -> Vec<f32> {
        self.run(blob, false).embedding
    }

    /// Forward with every stage tap captured — what the parity test replays.
    pub fn forward(&self, blob: &[f32]) -> ArcFaceTaps {
        self.run(blob, true)
    }

    /// The graph. `capture` gates ONLY the host readbacks, never a dispatch, so
    /// the two paths compute bit-identical values.
    fn run(&self, blob: &[f32], capture: bool) -> ArcFaceTaps {
        let ctx = self.ctx();
        let want = (3 * self.cfg.image_size * self.cfg.image_size) as usize;
        assert_eq!(blob.len(), want, "arcface: blob must be [3,{0},{0}]", self.cfg.image_size);
        self.gpu.write(&self.input, bytemuck::cast_slice(blob));

        self.stem_conv.forward(&ctx, &self.ps, &self.input);
        self.stem_prelu.forward(&ctx, &self.ps, self.stem_conv.out());

        let mut taps = ArcFaceTaps::default();
        if capture {
            taps.stem = self.read(self.stem_prelu.out(), self.stem_conv.out_shape.numel());
        }

        let mut x: &DeviceBuffer = self.stem_prelu.out();
        let mut stage = 0usize;
        let mut first_of_stage = true;
        for (i, blk) in self.blocks.iter().enumerate() {
            let y = blk.forward(&ctx, &self.ps, x);
            if capture && first_of_stage {
                taps.stage_b0[stage] = [
                    self.read(blk.bn1.out(), blk.bn1.shape.numel()),
                    self.read(blk.conv1.out(), blk.conv1.out_shape.numel()),
                    self.read(blk.prelu.out(), blk.prelu.shape.numel()),
                    self.read(blk.conv2.out(), blk.conv2.out_shape.numel()),
                    // Stage-first blocks always stride, so the shortcut conv
                    // always exists; `None` cannot occur here and an empty tap
                    // would be silently skipped by the parity test.
                    match &blk.downsample {
                        Some(d) => self.read(d.out(), d.out_shape.numel()),
                        None => panic!("layer{}.0 has no downsample conv", stage + 1),
                    },
                ];
            }
            first_of_stage = false;
            if capture {
                taps.blocks.push(self.read(y, blk.out_shape.numel()));
            }
            if i == self.stage_last[stage] {
                if capture {
                    taps.layers[stage] = taps.blocks.last().cloned().unwrap_or_default();
                }
                stage = (stage + 1).min(3);
                first_of_stage = true;
            }
            x = y;
        }

        self.bn2.forward(&ctx, &self.ps, x);
        if capture {
            taps.bn2 = self.read(self.bn2.out(), self.bn2.shape.numel());
        }

        // Flatten is a no-op: `bn2`'s NCHW buffer IS the [1, 25088] row.
        // fc: `matmul` is out = x · Wᵀ with x [m,k], W [n,k] — exactly the ONNX
        // Gemm's transB=1 layout. Params [m, k, n]; one invocation per output.
        let (e, fl) = (self.cfg.embedding, self.cfg.flatten());
        let s_mm = ctx.step(
            kernel("matmul"),
            &[self.bn2.out(), self.ps.w("fc.weight"), &self.fc_out],
            &[1, fl, e],
            e,
        );
        // `bias_add` is the [M, N] LINEAR bias (`out[idx] += bias[idx % N]`), which
        // is right for a flat [1, 512] row — NOT the NCHW `add_chan_inplace`.
        let s_b = ctx.step(kernel("bias_add"), &[&self.fc_out, self.ps.w("fc.bias")], &[1, e], e);
        ctx.gpu.submit(&[], &[s_mm, s_b]);
        if capture {
            taps.fc = self.read(&self.fc_out, e);
        }

        // The embedding is the RESULT, not a tap — always read.
        self.features.forward(&ctx, &self.ps, &self.fc_out);
        taps.embedding = self.read(self.features.out(), e);
        taps
    }

    fn read(&self, b: &DeviceBuffer, n: u32) -> Vec<f32> {
        self.gpu.read(b, n as usize)
    }
}

// The embedding's consumer-side math — L2-normalisation and cosine — is
// `model::hostmath::{l2_normalize, cosine}`. It is not face-specific (the ECAPA
// speaker embedding wants the identical pair), so it lives in host math's one
// home and is called directly; a local wrapper here is how a shared function
// becomes a private copy at the next edit (`AGENTS.md`).

// ===========================================================================
// SCRFD
// ===========================================================================

/// A ResNet basic block: `relu(conv2(relu(conv1(x))) + shortcut(x))`.
///
/// The strided shortcut is **`AveragePool(2, 2)` then a 1×1 conv** (the
/// "ResNet-D" downsample), not a strided 1×1. Both produce the right shape; only
/// one produces the right numbers.
struct BasicBlock {
    conv1: Conv,
    conv2: Conv,
    /// `(2x2/2 average-pool, 1x1 conv)` — present exactly where the block
    /// changes shape; the pool only where it also changes size.
    down: Option<(Option<AvgPool>, Conv)>,
    sum: DeviceBuffer,
    out: DeviceBuffer,
    out_shape: Shape,
}

impl BasicBlock {
    fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, stride: u32) -> BasicBlock {
        let conv1 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv1"),
            ConvNames::torch_flat(&format!("{prefix}.conv1")),
            in_shape,
            folded(cout, 3, stride, 1, Act::Relu),
            false,
        );
        let conv2 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv2"),
            ConvNames::torch_flat(&format!("{prefix}.conv2")),
            conv1.out_shape,
            folded(cout, 3, 1, 1, Act::None),
            false,
        );
        let out_shape = conv2.out_shape;
        let down = if stride != 1 || in_shape.c != cout {
            let pool = if stride != 1 { Some(AvgPool::half(ctx, in_shape)) } else { None };
            let pin = pool.as_ref().map(|p| p.out_shape).unwrap_or(in_shape);
            let proj = Conv::with_names(
                ctx,
                &format!("{prefix}.downsample"),
                ConvNames::torch_flat(&format!("{prefix}.downsample")),
                pin,
                folded(cout, 1, 1, 0, Act::None),
                false,
            );
            Some((pool, proj))
        } else {
            None
        };
        BasicBlock { conv1, conv2, down, sum: ctx.act(out_shape.numel()), out: ctx.act(out_shape.numel()), out_shape }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.conv1.param_list();
        v.extend(self.conv2.param_list());
        if let Some((_, p)) = &self.down {
            v.extend(p.param_list());
        }
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> &DeviceBuffer {
        self.conv1.forward(ctx, ps, x);
        self.conv2.forward(ctx, ps, self.conv1.out());
        let ident: &DeviceBuffer = match &self.down {
            Some((pool, proj)) => {
                let src = match pool {
                    Some(p) => {
                        p.forward(ctx, x);
                        p.out()
                    }
                    None => x,
                };
                proj.forward(ctx, ps, src);
                proj.out()
            }
            None => x,
        };
        let n = self.out_shape.numel();
        let s_add = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[self.conv2.out(), ident, &self.sum], &[n], n);
        // ReLU = leaky_relu at slope 0.
        let s_relu = ctx.step(
            ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
            &[&self.sum, &self.out],
            &[n, f(0.0)],
            n,
        );
        ctx.gpu.submit(&[], &[s_add, s_relu]);
        &self.out
    }
}

/// One stride's detection head: 3 conv+ReLU, then cls / reg (× learned scale) /
/// kps branches.
struct Head {
    stem: Vec<Conv>,
    cls: Conv,
    reg: Conv,
    kps: Conv,
    scale_name: String,
    /// `sigmoid(cls)` and `reg * scale`, in NCHW. The graph's transpose+reshape
    /// is a pure layout permute and is done on the host in [`Scrfd::detect`].
    cls_sig: DeviceBuffer,
    reg_scaled: DeviceBuffer,
}

impl Head {
    fn new(ctx: &Ctx, prefix: &str, scale_name: &str, in_shape: Shape, cfg: &ScrfdConfig) -> Head {
        let mut stem = Vec::new();
        let mut shape = in_shape;
        for d in 0..cfg.head_depth as usize {
            let c = Conv::with_names(
                ctx,
                &format!("{prefix}.stem.{d}"),
                ConvNames::torch_flat(&format!("{prefix}.stem.{d}")),
                shape,
                folded(cfg.head_channels, 3, 1, 1, Act::Relu),
                false,
            );
            shape = c.out_shape;
            stem.push(c);
        }
        let br = |name: &str, cout: u32| {
            Conv::with_names(
                ctx,
                &format!("{prefix}.{name}"),
                ConvNames::torch_flat(&format!("{prefix}.{name}")),
                shape,
                folded(cout, 3, 1, 1, Act::None),
                false,
            )
        };
        let na = cfg.num_anchors;
        let cls = br("cls", na);
        let reg = br("reg", 4 * na);
        let kps = br("kps", 10 * na);
        let (cn, rn) = (cls.out_shape.numel(), reg.out_shape.numel());
        Head {
            stem,
            cls,
            reg,
            kps,
            scale_name: scale_name.to_string(),
            cls_sig: ctx.act(cn),
            reg_scaled: ctx.act(rn),
        }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.stem.iter().flat_map(|c| c.param_list()).collect();
        v.extend(self.cls.param_list());
        v.extend(self.reg.param_list());
        v.extend(self.kps.param_list());
        v.push((self.scale_name.clone(), 1));
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, scale: f32) {
        let mut h = x;
        for c in &self.stem {
            c.forward(ctx, ps, h);
            h = c.out();
        }
        self.cls.forward(ctx, ps, h);
        self.reg.forward(ctx, ps, h);
        self.kps.forward(ctx, ps, h);
        let cn = self.cls.out_shape.numel();
        let s_sig = ctx.step(kernel("sigmoid"), &[self.cls.out(), &self.cls_sig], &[cn], cn);
        // `axpy` is `out += s*in`; clearing `out` in the same submit turns it
        // into the scaled COPY the graph's `Mul` node is, with SSA preserved.
        let rn = self.reg.out_shape.numel();
        let s_scale = ctx.step(kernel("axpy"), &[&self.reg_scaled, self.reg.out()], &[rn, f(scale)], rn);
        ctx.gpu.submit(&[&self.reg_scaled], &[s_sig, s_scale]);
    }
}

/// SCRFD forward taps, named for the goldens.
#[derive(Clone, Debug, Default)]
pub struct ScrfdTaps {
    pub stem_pre_pool: Vec<f32>,
    pub stem: Vec<f32>,
    /// `c2, c3, c4, c5`.
    pub c: [Vec<f32>; 4],
    /// `lat3, lat4, lat5`.
    pub lat: [Vec<f32>; 3],
    pub fpn4: Vec<f32>,
    pub fpn3: Vec<f32>,
    pub pafpn16_pre: Vec<f32>,
    pub pafpn32_pre: Vec<f32>,
    pub pafpn16: Vec<f32>,
    pub pafpn32: Vec<f32>,
    /// `neck8, neck16, neck32`.
    pub neck: [Vec<f32>; 3],
    /// Per stride: the head stem output.
    pub head_feat: [Vec<f32>; 3],
    /// Per stride: raw cls logits (pre-sigmoid), NCHW.
    pub cls_raw: [Vec<f32>; 3],
    /// Per stride: `reg * scale`, NCHW.
    pub bbox_scaled: [Vec<f32>; 3],
    /// Per stride: raw kps, NCHW.
    pub kps_raw: [Vec<f32>; 3],
    /// Per stride, in the graph's OUTPUT layout: `[rows, 1] / [rows, 4] /
    /// [rows, 10]` after transpose(2,3,0,1) + reshape.
    pub out_score: [Vec<f32>; 3],
    pub out_bbox: [Vec<f32>; 3],
    pub out_kps: [Vec<f32>; 3],
}

/// Inference-only SCRFD-10GF detector.
pub struct Scrfd {
    gpu: Gpu,
    cfg: ScrfdConfig,
    ps: ParamStore,
    input: DeviceBuffer,
    stem: Vec<Conv>,
    stem_pool: MaxPool,
    stages: Vec<Vec<BasicBlock>>,
    lateral: Vec<Conv>,
    /// Upsampled `lat5 -> lat4` and `fpn4 -> lat3` (nearest, ONNX asymmetric).
    up: [(DeviceBuffer, Shape, Shape); 2],
    fpn_add: [DeviceBuffer; 2],
    fpn: Vec<Conv>,
    down: Vec<Conv>,
    pa_add: [DeviceBuffer; 2],
    pafpn: Vec<Conv>,
    heads: Vec<Head>,
}

impl Scrfd {
    pub fn new(gpu: Gpu, cfg: ScrfdConfig, weights: &Tensors) -> Scrfd {
        let ctx = Ctx::new(&gpu, ids());
        let sz = cfg.image_size;
        let mut shape = Shape::new(1, 3, sz, sz);

        let mut stem = Vec::new();
        for i in 0..3usize {
            let stride = if i == 0 { 2 } else { 1 };
            let c = Conv::with_names(
                &ctx,
                &format!("backbone.stem.{i}"),
                ConvNames::torch_flat(&format!("backbone.stem.{i}")),
                shape,
                folded(cfg.stem_channels[i], 3, stride, 1, Act::Relu),
                false,
            );
            shape = c.out_shape;
            stem.push(c);
        }
        let stem_pool = MaxPool::new(&ctx, shape, PoolSpec::half());
        shape = stem_pool.out_shape;

        let mut stages: Vec<Vec<BasicBlock>> = Vec::new();
        let mut c_out: Vec<Shape> = Vec::new();
        for s in 0..4usize {
            let mut blocks = Vec::new();
            for b in 0..cfg.layers[s] as usize {
                let stride = if b == 0 && cfg.stage_stride2[s] { 2 } else { 1 };
                let blk = BasicBlock::new(&ctx, &format!("backbone.layer{}.{}", s + 1, b), shape, cfg.channels[s], stride);
                shape = blk.out_shape;
                blocks.push(blk);
            }
            c_out.push(shape);
            stages.push(blocks);
        }

        let nk = cfg.neck_channels;
        let lateral: Vec<Conv> = (0..3)
            .map(|i| {
                Conv::with_names(
                    &ctx,
                    &format!("neck.lateral_convs.{i}.conv"),
                    ConvNames::torch_flat(&format!("neck.lateral_convs.{i}.conv")),
                    c_out[i + 1],
                    folded(nk, 1, 1, 0, Act::None),
                    false,
                )
            })
            .collect();
        let l3 = lateral[0].out_shape;
        let l4 = lateral[1].out_shape;
        let l5 = lateral[2].out_shape;
        let up = [
            (ctx.act(l4.numel()), l5, l4), // lat5 -> lat4 size
            (ctx.act(l3.numel()), l4, l3), // fpn4 -> lat3 size
        ];
        let fpn_add = [ctx.act(l4.numel()), ctx.act(l3.numel())];
        let fpn_in = [l3, l4, l5];
        let fpn: Vec<Conv> = (0..3)
            .map(|i| {
                Conv::with_names(
                    &ctx,
                    &format!("neck.fpn_convs.{i}.conv"),
                    ConvNames::torch_flat(&format!("neck.fpn_convs.{i}.conv")),
                    fpn_in[i],
                    folded(nk, 3, 1, 1, Act::None),
                    false,
                )
            })
            .collect();
        let down: Vec<Conv> = (0..2)
            .map(|i| {
                Conv::with_names(
                    &ctx,
                    &format!("neck.downsample_convs.{i}.conv"),
                    ConvNames::torch_flat(&format!("neck.downsample_convs.{i}.conv")),
                    if i == 0 { fpn[0].out_shape } else { l4 },
                    folded(nk, 3, 2, 1, Act::None),
                    false,
                )
            })
            .collect();
        let pa_add = [ctx.act(l4.numel()), ctx.act(l5.numel())];
        let pafpn: Vec<Conv> = (0..2)
            .map(|i| {
                Conv::with_names(
                    &ctx,
                    &format!("neck.pafpn_convs.{i}.conv"),
                    ConvNames::torch_flat(&format!("neck.pafpn_convs.{i}.conv")),
                    if i == 0 { l4 } else { l5 },
                    folded(nk, 3, 1, 1, Act::None),
                    false,
                )
            })
            .collect();

        let neck_shapes = [fpn[0].out_shape, pafpn[0].out_shape, pafpn[1].out_shape];
        let heads: Vec<Head> = cfg
            .strides
            .iter()
            .enumerate()
            .map(|(i, st)| Head::new(&ctx, &format!("head.stride{st}"), &format!("head.scales.{i}"), neck_shapes[i], &cfg))
            .collect();

        let mut params: Vec<(String, usize)> = stem.iter().flat_map(|c| c.param_list()).collect();
        for st in &stages {
            for b in st {
                params.extend(b.param_list());
            }
        }
        for c in lateral.iter().chain(&fpn).chain(&down).chain(&pafpn) {
            params.extend(c.param_list());
        }
        for h in &heads {
            params.extend(h.param_list());
        }
        assert_eq!(
            params.len(),
            weights.len(),
            "scrfd: the graph reads {} tensors but the checkpoint has {}",
            params.len(),
            weights.len()
        );
        for (name, n) in &params {
            match weights.get(name) {
                None => panic!("scrfd: checkpoint is missing {name}"),
                Some((shape, _)) => {
                    let have: usize = shape.iter().product();
                    assert_eq!(have, *n, "scrfd: {name} has {have} values, the graph wants {n}");
                }
            }
        }

        let ps = frozen_store(&gpu, weights);
        let input = gpu.storage((3 * sz * sz) as u64);
        Scrfd { gpu, cfg, ps, input, stem, stem_pool, stages, lateral, up, fpn_add, fpn, down, pa_add, pafpn, heads }
    }

    pub fn config(&self) -> &ScrfdConfig {
        &self.cfg
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn read(&self, b: &DeviceBuffer, n: u32) -> Vec<f32> {
        self.gpu.read(b, n as usize)
    }

    /// Nearest-neighbour upsample to an explicit target size.
    ///
    /// `resize_nearest` Params `[N, C, H, W, Ho, Wo]`, one invocation per OUTPUT
    /// element, with the rule `src = floor(o*in/out)` — which IS the graph's
    /// `Resize(mode=nearest, coordinate_transformation_mode=asymmetric,
    /// nearest_mode=floor)`. `upsample2` would agree only because the ratio here
    /// happens to be exactly 2; the explicit target keeps it right at any size.
    fn upsample(&self, ctx: &Ctx, x: &DeviceBuffer, i: usize) {
        let (dst, src_shape, dst_shape) = &self.up[i];
        let s = ctx.step(
            ctx.ids.need(ctx.ids.resize_nearest, "resize_nearest"),
            &[x, dst],
            &[src_shape.n, src_shape.c, src_shape.h, src_shape.w, dst_shape.h, dst_shape.w],
            dst_shape.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
    }

    fn add(&self, ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, n: u32) {
        let s = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[a, b, out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }

    /// Full forward on an NCHW `[1, 3, 640, 640]` blob, capturing every tap.
    pub fn forward(&self, blob: &[f32]) -> ScrfdTaps {
        let ctx = Ctx::new(&self.gpu, ids());
        let sz = self.cfg.image_size;
        assert_eq!(blob.len(), (3 * sz * sz) as usize, "scrfd: blob must be [3,{sz},{sz}]");
        self.gpu.write(&self.input, bytemuck::cast_slice(blob));

        let mut t = ScrfdTaps::default();
        let mut x: &DeviceBuffer = &self.input;
        for c in &self.stem {
            c.forward(&ctx, &self.ps, x);
            x = c.out();
        }
        t.stem_pre_pool = self.read(x, self.stem[2].out_shape.numel());
        self.stem_pool.forward(&ctx, x);
        x = self.stem_pool.out();
        t.stem = self.read(x, self.stem_pool.out_shape.numel());

        let mut c_bufs: Vec<&DeviceBuffer> = Vec::new();
        for (si, st) in self.stages.iter().enumerate() {
            for b in st {
                x = b.forward(&ctx, &self.ps, x);
            }
            t.c[si] = self.read(x, st.last().expect("a stage has blocks").out_shape.numel());
            c_bufs.push(x);
        }

        // laterals over c3, c4, c5
        for i in 0..3usize {
            self.lateral[i].forward(&ctx, &self.ps, c_bufs[i + 1]);
            t.lat[i] = self.read(self.lateral[i].out(), self.lateral[i].out_shape.numel());
        }
        // top-down: fpn4 = lat4 + up(lat5); fpn3 = lat3 + up(fpn4)
        self.upsample(&ctx, self.lateral[2].out(), 0);
        let l4n = self.lateral[1].out_shape.numel();
        self.add(&ctx, self.lateral[1].out(), &self.up[0].0, &self.fpn_add[0], l4n);
        t.fpn4 = self.read(&self.fpn_add[0], l4n);
        self.upsample(&ctx, &self.fpn_add[0], 1);
        let l3n = self.lateral[0].out_shape.numel();
        self.add(&ctx, self.lateral[0].out(), &self.up[1].0, &self.fpn_add[1], l3n);
        t.fpn3 = self.read(&self.fpn_add[1], l3n);

        // fpn convs: /8 from fpn3, and the two PAFPN inputs from fpn4 / lat5
        self.fpn[0].forward(&ctx, &self.ps, &self.fpn_add[1]);
        self.fpn[1].forward(&ctx, &self.ps, &self.fpn_add[0]);
        self.fpn[2].forward(&ctx, &self.ps, self.lateral[2].out());
        t.neck[0] = self.read(self.fpn[0].out(), self.fpn[0].out_shape.numel());
        t.pafpn16_pre = self.read(self.fpn[1].out(), self.fpn[1].out_shape.numel());
        t.pafpn32_pre = self.read(self.fpn[2].out(), self.fpn[2].out_shape.numel());

        // bottom-up
        self.down[0].forward(&ctx, &self.ps, self.fpn[0].out());
        self.add(&ctx, self.fpn[1].out(), self.down[0].out(), &self.pa_add[0], l4n);
        t.pafpn16 = self.read(&self.pa_add[0], l4n);
        self.down[1].forward(&ctx, &self.ps, &self.pa_add[0]);
        let l5n = self.lateral[2].out_shape.numel();
        self.add(&ctx, self.fpn[2].out(), self.down[1].out(), &self.pa_add[1], l5n);
        t.pafpn32 = self.read(&self.pa_add[1], l5n);

        self.pafpn[0].forward(&ctx, &self.ps, &self.pa_add[0]);
        self.pafpn[1].forward(&ctx, &self.ps, &self.pa_add[1]);
        t.neck[1] = self.read(self.pafpn[0].out(), self.pafpn[0].out_shape.numel());
        t.neck[2] = self.read(self.pafpn[1].out(), self.pafpn[1].out_shape.numel());

        let neck_bufs: [&DeviceBuffer; 3] = [self.fpn[0].out(), self.pafpn[0].out(), self.pafpn[1].out()];
        for (i, h) in self.heads.iter().enumerate() {
            let scale = self.ps.read_weight(&self.gpu, &h.scale_name)[0];
            h.forward(&ctx, &self.ps, neck_bufs[i], scale);
            let last = h.stem.last().expect("head_depth >= 1");
            t.head_feat[i] = self.read(last.out(), last.out_shape.numel());
            t.cls_raw[i] = self.read(h.cls.out(), h.cls.out_shape.numel());
            t.bbox_scaled[i] = self.read(&h.reg_scaled, h.reg.out_shape.numel());
            t.kps_raw[i] = self.read(h.kps.out(), h.kps.out_shape.numel());

            let s = h.cls.out_shape;
            let sig = self.read(&h.cls_sig, s.numel());
            t.out_score[i] = nchw_to_rows(&sig, s.c, s.h, s.w, self.cfg.num_anchors);
            t.out_bbox[i] = nchw_to_rows(&t.bbox_scaled[i], h.reg.out_shape.c, s.h, s.w, self.cfg.num_anchors);
            t.out_kps[i] = nchw_to_rows(&t.kps_raw[i], h.kps.out_shape.c, s.h, s.w, self.cfg.num_anchors);
        }
        t
    }
}

/// The graph's `Transpose(perm=[2,3,0,1]) -> Reshape(-1, cols)` on an NCHW head
/// output, for N = 1.
///
/// `[C, H, W]` becomes `[H*W*num_anchors, C/num_anchors]`: row
/// `(h*W + w)*A + a`, column `k`, reads channel `a*cols + k`. A pure layout
/// permutation, which `crates/imaging`'s own line puts on the host — there is no
/// arithmetic here, and the tensor is at most 12800×10.
fn nchw_to_rows(x: &[f32], c: u32, h: u32, w: u32, anchors: u32) -> Vec<f32> {
    assert_eq!(c % anchors, 0, "channel count {c} is not a multiple of {anchors} anchors");
    let cols = (c / anchors) as usize;
    let (h, w, a) = (h as usize, w as usize, anchors as usize);
    let hw = h * w;
    let mut out = vec![0.0f32; hw * a * cols];
    for p in 0..hw {
        for ai in 0..a {
            for k in 0..cols {
                out[(p * a + ai) * cols + k] = x[(ai * cols + k) * hw + p];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permute must invert the reference's `transpose(2,3,0,1).reshape(-1,k)`
    /// exactly — an off-by-one in the anchor/column split still produces the
    /// right SHAPE and completely wrong boxes.
    #[test]
    fn head_output_permute_matches_transpose_then_reshape() {
        // C=4 (2 anchors x 2 cols), H=W=2. x[c, p] = c*10 + p.
        let (c, h, w, a) = (4u32, 2u32, 2u32, 2u32);
        let x: Vec<f32> = (0..(c * h * w)).map(|i| (i / (h * w) * 10 + i % (h * w)) as f32).collect();
        let rows = nchw_to_rows(&x, c, h, w, a);
        assert_eq!(rows.len(), (h * w * a * (c / a)) as usize);
        // row for (p=1, anchor=1) must be channels [1*2+0, 1*2+1] at p=1.
        let r = (1 * a + 1) as usize;
        assert_eq!(rows[r * 2], 21.0);
        assert_eq!(rows[r * 2 + 1], 31.0);
    }

    #[test]
    fn kernel_lookup_is_by_name_and_panics_on_an_unregistered_one() {
        assert_eq!(PIPELINES[kernel("matmul")].0, "matmul");
        assert_eq!(PIPELINES[kernel("grid_sample")].0, "grid_sample");
        assert!(std::panic::catch_unwind(|| kernel("no_such_kernel")).is_err());
    }

    /// Every kernel the shared blocks will look up must actually be registered,
    /// or the first forward panics at a dispatch instead of at build time.
    #[test]
    fn every_kernel_the_blocks_need_is_registered() {
        let i = ids();
        for (id, name) in [
            (i.conv_bias_reg, "conv_bias_reg"),
            (i.bn_eval, "bn_eval"),
            (i.prelu, "prelu"),
            (i.prelu_bwd, "prelu_bwd"),
            (i.prelu_bwd_wg, "prelu_bwd_wg"),
            (i.maxpool2d, "maxpool2d"),
            (i.avgpool2d, "avgpool2d"),
            (i.resize_nearest, "resize_nearest"),
            (i.add2, "add2"),
            (i.leaky_relu, "leaky_relu"),
        ] {
            assert_ne!(id, vision::NONE, "{name} missing from facenet::PIPELINES");
        }
    }
}
