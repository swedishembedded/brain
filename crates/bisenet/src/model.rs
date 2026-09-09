// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BiSeNet(`num_class=19`) face parsing - inference-only, composed entirely
//! from the shared conv-net blocks in `crates/vision` (`Conv`, `MaxPool`,
//! `Shape`, `Ctx`) over the shared kernel-id seam (`ConvKernelIds::resolve`,
//! by NAME). This crate adds no conv, no norm and no activation of its own -
//! see `config.rs` for the fixed architecture this composes and
//! `tools/goldens/pulid_face_parsing_dump_reference.py`'s header for the
//! upstream reference (`facexlib.parsing.bisenet.BiSeNet`) every prefix here
//! is transcribed from, tensor name for tensor name.
//!
//! Only the path the reference inference call actually reads is built:
//! `conv_out16`/`conv_out32` (upstream's two training-only auxiliary heads)
//! are not imported and not part of this graph at all - see
//! `import.rs`/`config.rs`'s docs.

use gpu_core::{f, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{Act, Conv, ConvKernelIds, ConvNames, ConvSpec, Ctx, MaxPool, Norm, PoolSpec, Shape};

use crate::config::{BiSeNetConfig, ARM16, ARM32, FFM, OUTPUT_MID, RESNET18_STAGES};
use crate::import::Tensors;

/// Every kernel this graph dispatches, by name - resolved once against
/// [`ConvKernelIds::resolve`], the same name-indirection seam every other
/// `crates/vision`-based model uses.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),
    ("resize_nearest", kernels::RESIZE_NEAREST),
    ("resize_nearest_dx", kernels::RESIZE_NEAREST_DX),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_bilinear_dx", kernels::RESIZE_BILINEAR_DX),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("scale_chan", kernels::SCALE_CHAN),
    ("scale_chan_dg", kernels::SCALE_CHAN_DG),
    ("conv_act", kernels::CONV_ACT),
    ("conv_act_tiled", kernels::CONV_ACT_TILED),
    ("conv_act_reg", kernels::CONV_ACT_REG),
    // `crate::align`'s warp - not one of `ConvKernelIds`'s named fields (a
    // face-alignment primitive, not a generic conv-net one), resolved by
    // name via `kernel()` below instead.
    ("grid_sample", kernels::GRID_SAMPLE),
];

pub fn ids() -> &'static ConvKernelIds {
    static IDS: std::sync::OnceLock<ConvKernelIds> = std::sync::OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// The pipeline index of a kernel this crate dispatches directly by name
/// (today: only `grid_sample`, which has no `ConvKernelIds` field) - see
/// `arcface::model::kernel`'s identical helper.
pub fn kernel(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("kernel `{name}` is not in bisenet::PIPELINES"))
}

fn frozen_store(gpu: &Gpu, t: &Tensors) -> ParamStore {
    let mut init = std::collections::HashMap::with_capacity(t.len());
    let mut roles = Vec::with_capacity(t.len());
    for (name, (shape, data)) in t {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "bisenet: tensor {name}: {} values for shape {shape:?}", data.len());
        roles.push((name.clone(), n, Role::Frozen));
        init.insert(name.clone(), data.clone());
    }
    roles.sort_by(|a, b| a.0.cmp(&b.0));
    ParamStore::new_with_roles(gpu, roles, &init)
}

/// `ConvNames` for an explicit `(conv weight name, BN module prefix)` pair -
/// this checkpoint's convs and their BNs are separately-named siblings
/// (`conv1`/`bn1`, `conv.conv`/`conv.bn`, ...), not one of the three
/// pre-built `ConvNames` conventions (`brain`/`torch_conv_bn`/`torch_seq`).
/// `bias` is left empty and is never read (every BN-fed conv here is
/// `bias=False`, per the reference source).
fn bn_names(conv_weight: &str, bn_prefix: &str) -> ConvNames {
    ConvNames {
        bias: String::new(),
        weight: conv_weight.to_string(),
        gamma: format!("{bn_prefix}.weight"),
        beta: format!("{bn_prefix}.bias"),
        run_mean: format!("{bn_prefix}.running_mean"),
        run_var: format!("{bn_prefix}.running_var"),
    }
}

/// A bare, BN-free, bias-free conv (`FFM.conv1`/`conv2`, `conv_out.conv_out`
/// - all `nn.Conv2d(..., bias=False)` with no following BatchNorm).
fn raw_names(conv_weight: &str) -> ConvNames {
    ConvNames { bias: String::new(), weight: conv_weight.to_string(), gamma: String::new(), beta: String::new(), run_mean: String::new(), run_var: String::new() }
}

fn bn_spec(cout: u32, k: u32, stride: u32, pad: u32, act: Act) -> ConvSpec {
    ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::Bn, act, bias: false }
}
fn raw_spec(cout: u32, k: u32, stride: u32, pad: u32, act: Act) -> ConvSpec {
    ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::None, act, bias: false }
}

/// `1x1` global average pool (`F.avg_pool2d(x, x.size()[2:])`) - `avgpool2d`'s
/// adaptive rule at `Ho=Wo=1` is exactly a box mean over the whole map.
fn global_avgpool(ctx: &Ctx, shape: Shape, x: &DeviceBuffer) -> DeviceBuffer {
    let y = ctx.act(shape.n * shape.c);
    let s = ctx.step(ctx.ids.avgpool2d, &[x, &y], &[shape.n, shape.c, shape.h, shape.w, 1, 1], shape.n * shape.c);
    ctx.gpu.submit(&[], &[s]);
    y
}

/// Nearest-neighbor upsample of a `[N,C,H,W]` map to `(oh, ow)` -
/// `F.interpolate(..., mode='nearest')`. At `H=W=1` (the two `avg`/`atten`
/// broadcasts) this is a pure tile/broadcast, the same operator.
fn nearest_upsample(ctx: &Ctx, shape: Shape, oh: u32, ow: u32, x: &DeviceBuffer) -> (DeviceBuffer, Shape) {
    let out_shape = Shape::new(shape.n, shape.c, oh, ow);
    let y = ctx.act(out_shape.numel());
    let s = ctx.step(ctx.ids.resize_nearest, &[x, &y], &[shape.n, shape.c, shape.h, shape.w, oh, ow], out_shape.numel());
    ctx.gpu.submit(&[], &[s]);
    (y, out_shape)
}

/// `y = a + b`, both `[n]`.
fn add(ctx: &Ctx, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
    let y = ctx.act(n);
    let s = ctx.step(ctx.ids.add2, &[a, b, &y], &[n], n);
    ctx.gpu.submit(&[], &[s]);
    y
}

/// `y = x * scale[c]` - `scale` is `[C]` (broadcast over batch AND space);
/// correct at this crate's fixed `N=1`.
fn scale_chan(ctx: &Ctx, shape: Shape, x: &DeviceBuffer, scale: &DeviceBuffer) -> DeviceBuffer {
    let n = shape.numel();
    let y = ctx.act(n);
    let s = ctx.step(ctx.ids.scale_chan, &[x, scale, &y], &[n, shape.c, shape.h * shape.w], n);
    ctx.gpu.submit(&[], &[s]);
    y
}

/// The stand-alone post-residual-add ReLU every `BasicBlock` needs (`Conv`'s
/// own fused activation runs BEFORE the shortcut add, never after) -
/// `leaky_relu` at slope 0 is ReLU in both directions, so this costs no
/// kernel of its own (the same reuse every other ReLU-activated model here
/// takes).
fn relu(ctx: &Ctx, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let y = ctx.act(n);
    let s = ctx.step(ctx.ids.leaky_relu, &[x, &y], &[n, f(0.0)], n);
    ctx.gpu.submit(&[], &[s]);
    y
}

// =========================================================================
// ResNet18 BasicBlock
// =========================================================================

struct BasicBlock {
    conv1: Conv,
    conv2: Conv,
    downsample: Option<Conv>,
    out_shape: Shape,
}

impl BasicBlock {
    fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, stride: u32, train: bool) -> BasicBlock {
        let conv1 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv1"),
            bn_names(&format!("{prefix}.conv1.weight"), &format!("{prefix}.bn1")),
            in_shape,
            bn_spec(cout, 3, stride, 1, Act::Relu),
            train,
        );
        let conv2 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv2"),
            bn_names(&format!("{prefix}.conv2.weight"), &format!("{prefix}.bn2")),
            conv1.out_shape,
            bn_spec(cout, 3, 1, 1, Act::None),
            train,
        );
        let downsample = if stride != 1 || in_shape.c != cout {
            Some(Conv::with_names(
                ctx,
                &format!("{prefix}.downsample"),
                bn_names(&format!("{prefix}.downsample.0.weight"), &format!("{prefix}.downsample.1")),
                in_shape,
                bn_spec(cout, 1, stride, 0, Act::None),
                train,
            ))
        } else {
            None
        };
        let out_shape = conv2.out_shape;
        BasicBlock { conv1, conv2, downsample, out_shape }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.conv1.param_list();
        v.extend(self.conv2.param_list());
        if let Some(d) = &self.downsample {
            v.extend(d.param_list());
        }
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> DeviceBuffer {
        self.conv1.forward(ctx, ps, x);
        self.conv2.forward(ctx, ps, self.conv1.out());
        let ident: &DeviceBuffer = match &self.downsample {
            Some(d) => {
                d.forward(ctx, ps, x);
                d.out()
            }
            None => x,
        };
        let sum = add(ctx, self.out_shape.numel(), self.conv2.out(), ident);
        relu(ctx, self.out_shape.numel(), &sum)
    }
}

/// One `ResNet18` stage: `n` `BasicBlock`s, the first stride-`stride`
/// (downsampling), the rest stride-1.
struct Stage {
    blocks: Vec<BasicBlock>,
    out_shape: Shape,
}

impl Stage {
    fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, n: u32, first_stride: u32, train: bool) -> Stage {
        let mut shape = in_shape;
        let mut blocks = Vec::new();
        for b in 0..n {
            let stride = if b == 0 { first_stride } else { 1 };
            let blk = BasicBlock::new(ctx, &format!("{prefix}.{b}"), shape, cout, stride, train);
            shape = blk.out_shape;
            blocks.push(blk);
        }
        Stage { blocks, out_shape: shape }
    }
    fn param_list(&self) -> Vec<(String, usize)> {
        self.blocks.iter().flat_map(BasicBlock::param_list).collect()
    }
    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> DeviceBuffer {
        let mut cur = x.clone();
        for b in &self.blocks {
            cur = b.forward(ctx, ps, &cur);
        }
        cur
    }
}

// =========================================================================
// AttentionRefinementModule: conv(BN,ReLU) -> sigmoid(BN(conv1x1(gap))) gate
// =========================================================================

struct Arm {
    conv: Conv,
    atten: Conv,
    out_shape: Shape,
}

impl Arm {
    fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, io: (u32, u32), train: bool) -> Arm {
        let (_cin, cout) = io;
        let conv = Conv::with_names(
            ctx,
            &format!("{prefix}.conv"),
            bn_names(&format!("{prefix}.conv.conv.weight"), &format!("{prefix}.conv.bn")),
            in_shape,
            bn_spec(cout, 3, 1, 1, Act::Relu),
            train,
        );
        let pooled_shape = Shape::new(conv.out_shape.n, cout, 1, 1);
        let atten = Conv::with_names(
            ctx,
            &format!("{prefix}.conv_atten"),
            bn_names(&format!("{prefix}.conv_atten.weight"), &format!("{prefix}.bn_atten")),
            pooled_shape,
            bn_spec(cout, 1, 1, 0, Act::Sigmoid),
            train,
        );
        Arm { out_shape: conv.out_shape, conv, atten }
    }
    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.conv.param_list();
        v.extend(self.atten.param_list());
        v
    }
    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> DeviceBuffer {
        self.conv.forward(ctx, ps, x);
        let feat = self.conv.out();
        let pooled = global_avgpool(ctx, self.out_shape, feat);
        self.atten.forward(ctx, ps, &pooled);
        scale_chan(ctx, self.out_shape, feat, self.atten.out())
    }
}

// =========================================================================
// ContextPath: ResNet18 + global-context tail + 2 ARMs + 2 head convs
// =========================================================================

struct ContextPath {
    stem: Conv,
    pool: PoolSpec,
    stage1: Stage,
    stage2: Stage,
    stage3: Stage,
    stage4: Stage,
    conv_avg: Conv,
    arm16: Arm,
    arm32: Arm,
    conv_head32: Conv,
    conv_head16: Conv,
}

/// What `ContextPath::forward` returns: `feat_res8` (the RAW, un-refined
/// stage-2 output - upstream's spatial-path substitute) and `feat_cp8`
/// (the fully context-refined output at the same 1/8 resolution).
struct ContextOut {
    feat_res8: DeviceBuffer,
    feat_cp8: DeviceBuffer,
    shape8: Shape,
}

impl ContextPath {
    #[allow(clippy::too_many_arguments)]
    fn new(ctx: &Ctx, in_shape: Shape, train: bool) -> ContextPath {
        let stem = Conv::with_names(
            ctx,
            "cp.resnet.conv1",
            bn_names("cp.resnet.conv1.weight", "cp.resnet.bn1"),
            in_shape,
            bn_spec(64, 7, 2, 3, Act::Relu),
            train,
        );
        let pool = PoolSpec { k: 3, stride: 2, pad: 1 };
        let pooled_shape = pool.out_shape(stem.out_shape);

        let (c1, n1) = RESNET18_STAGES[0];
        let (c2, n2) = RESNET18_STAGES[1];
        let (c3, n3) = RESNET18_STAGES[2];
        let (c4, n4) = RESNET18_STAGES[3];
        let stage1 = Stage::new(ctx, "cp.resnet.layer1", pooled_shape, c1, n1, 1, train);
        let stage2 = Stage::new(ctx, "cp.resnet.layer2", stage1.out_shape, c2, n2, 2, train);
        let stage3 = Stage::new(ctx, "cp.resnet.layer3", stage2.out_shape, c3, n3, 2, train);
        let stage4 = Stage::new(ctx, "cp.resnet.layer4", stage3.out_shape, c4, n4, 2, train);

        let conv_avg = Conv::with_names(
            ctx,
            "cp.conv_avg",
            bn_names("cp.conv_avg.conv.weight", "cp.conv_avg.bn"),
            Shape::new(stage4.out_shape.n, stage4.out_shape.c, 1, 1),
            bn_spec(128, 1, 1, 0, Act::Relu),
            train,
        );
        let arm16 = Arm::new(ctx, "cp.arm16", stage3.out_shape, ARM16, train);
        let arm32 = Arm::new(ctx, "cp.arm32", stage4.out_shape, ARM32, train);
        let conv_head32 = Conv::with_names(
            ctx,
            "cp.conv_head32",
            bn_names("cp.conv_head32.conv.weight", "cp.conv_head32.bn"),
            Shape::new(1, 128, stage3.out_shape.h, stage3.out_shape.w),
            bn_spec(128, 3, 1, 1, Act::Relu),
            train,
        );
        let conv_head16 = Conv::with_names(
            ctx,
            "cp.conv_head16",
            bn_names("cp.conv_head16.conv.weight", "cp.conv_head16.bn"),
            Shape::new(1, 128, stage2.out_shape.h, stage2.out_shape.w),
            bn_spec(128, 3, 1, 1, Act::Relu),
            train,
        );

        ContextPath { stem, pool, stage1, stage2, stage3, stage4, conv_avg, arm16, arm32, conv_head32, conv_head16 }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.stem.param_list();
        v.extend(self.stage1.param_list());
        v.extend(self.stage2.param_list());
        v.extend(self.stage3.param_list());
        v.extend(self.stage4.param_list());
        v.extend(self.conv_avg.param_list());
        v.extend(self.arm16.param_list());
        v.extend(self.arm32.param_list());
        v.extend(self.conv_head32.param_list());
        v.extend(self.conv_head16.param_list());
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> ContextOut {
        self.stem.forward(ctx, ps, x);
        let pool = MaxPool::new(ctx, self.stem.out_shape, self.pool);
        pool.forward(ctx, self.stem.out());
        let feat_res8 = self.stage1.forward(ctx, ps, pool.out());
        let feat_res8 = self.stage2.forward(ctx, ps, &feat_res8);
        let feat16 = self.stage3.forward(ctx, ps, &feat_res8);
        let feat32 = self.stage4.forward(ctx, ps, &feat16);

        let avg = global_avgpool(ctx, self.stage4.out_shape, &feat32);
        self.conv_avg.forward(ctx, ps, &avg);
        let (avg_up, _) = nearest_upsample(ctx, Shape::new(1, 128, 1, 1), self.stage4.out_shape.h, self.stage4.out_shape.w, self.conv_avg.out());

        let feat32_arm = self.arm32.forward(ctx, ps, &feat32);
        let feat32_sum = add(ctx, self.arm32.out_shape.numel(), &feat32_arm, &avg_up);
        let (feat32_up, _) = nearest_upsample(ctx, self.arm32.out_shape, self.stage3.out_shape.h, self.stage3.out_shape.w, &feat32_sum);
        self.conv_head32.forward(ctx, ps, &feat32_up);

        let feat16_arm = self.arm16.forward(ctx, ps, &feat16);
        let feat16_sum = add(ctx, self.arm16.out_shape.numel(), &feat16_arm, self.conv_head32.out());
        let (feat16_up, _) = nearest_upsample(ctx, self.arm16.out_shape, self.stage2.out_shape.h, self.stage2.out_shape.w, &feat16_sum);
        self.conv_head16.forward(ctx, ps, &feat16_up);

        ContextOut { feat_res8, feat_cp8: self.conv_head16.out().clone(), shape8: self.stage2.out_shape }
    }
}

// =========================================================================
// FeatureFusionModule
// =========================================================================

struct Ffm {
    convblk: Conv,
    conv1: Conv,
    conv2: Conv,
    out_shape: Shape,
}

impl Ffm {
    fn new(ctx: &Ctx, shape8: Shape, train: bool) -> Ffm {
        let (in_chan, out_chan) = FFM;
        let cat_shape = Shape::new(shape8.n, in_chan, shape8.h, shape8.w);
        let convblk = Conv::with_names(ctx, "ffm.convblk", bn_names("ffm.convblk.conv.weight", "ffm.convblk.bn"), cat_shape, bn_spec(out_chan, 1, 1, 0, Act::Relu), train);
        let pooled_shape = Shape::new(1, out_chan, 1, 1);
        let hidden = out_chan / 4;
        let conv1 = Conv::with_names(ctx, "ffm.conv1", raw_names("ffm.conv1.weight"), pooled_shape, raw_spec(hidden, 1, 1, 0, Act::Relu), train);
        let conv2 = Conv::with_names(ctx, "ffm.conv2", raw_names("ffm.conv2.weight"), conv1.out_shape, raw_spec(out_chan, 1, 1, 0, Act::Sigmoid), train);
        let out_shape = convblk.out_shape;
        Ffm { convblk, conv1, conv2, out_shape }
    }
    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.convblk.param_list();
        v.extend(self.conv1.param_list());
        v.extend(self.conv2.param_list());
        v
    }
    fn forward(&self, ctx: &Ctx, ps: &ParamStore, fsp: &DeviceBuffer, fcp: &DeviceBuffer, shape8: Shape) -> DeviceBuffer {
        let (ca, cb) = (shape8.c, shape8.c);
        let cat_shape = Shape::new(shape8.n, ca + cb, shape8.h, shape8.w);
        let cat = ctx.act(cat_shape.numel());
        let s = ctx.step(ctx.ids.concat2, &[fsp, fcp, &cat], &[shape8.n, ca, cb, shape8.h, shape8.w], cat_shape.numel());
        ctx.gpu.submit(&[], &[s]);

        self.convblk.forward(ctx, ps, &cat);
        let feat = self.convblk.out();
        let pooled = global_avgpool(ctx, self.out_shape, feat);
        self.conv1.forward(ctx, ps, &pooled);
        self.conv2.forward(ctx, ps, self.conv1.out());
        let feat_atten = scale_chan(ctx, self.out_shape, feat, self.conv2.out());
        add(ctx, self.out_shape.numel(), &feat_atten, feat)
    }
}

// =========================================================================
// BiSeNet: ContextPath -> FFM -> BiSeNetOutput(conv_out only) -> upsample
// =========================================================================

pub struct BiSeNet {
    gpu: Gpu,
    cfg: BiSeNetConfig,
    ps: ParamStore,
    in_shape: Shape,
    cp: ContextPath,
    ffm: Ffm,
    conv_out: Conv,
    conv_out_final: Conv,
}

impl BiSeNet {
    pub fn new(gpu: Gpu, cfg: BiSeNetConfig, weights: &Tensors) -> BiSeNet {
        let ids = ids();
        let ctx = Ctx::new(&gpu, ids);
        let in_shape = Shape::new(1, 3, cfg.input_size, cfg.input_size);

        let cp = ContextPath::new(&ctx, in_shape, false);
        let shape8 = Shape::new(1, 128, cp.stage2.out_shape.h, cp.stage2.out_shape.w);
        let ffm = Ffm::new(&ctx, shape8, false);
        let conv_out = Conv::with_names(&ctx, "conv_out.conv", bn_names("conv_out.conv.conv.weight", "conv_out.conv.bn"), ffm.out_shape, bn_spec(OUTPUT_MID, 3, 1, 1, Act::Relu), false);
        let conv_out_final = Conv::with_names(&ctx, "conv_out.conv_out", raw_names("conv_out.conv_out.weight"), conv_out.out_shape, raw_spec(cfg.num_class, 1, 1, 0, Act::None), false);

        let mut params = cp.param_list();
        params.extend(ffm.param_list());
        params.extend(conv_out.param_list());
        params.extend(conv_out_final.param_list());
        assert_eq!(
            params.len(),
            weights.len(),
            "bisenet: the graph reads {} tensors but the checkpoint has {}",
            params.len(),
            weights.len()
        );
        for (name, n) in &params {
            match weights.get(name) {
                None => panic!("bisenet: checkpoint is missing {name}"),
                Some((shape, _)) => {
                    let have: usize = shape.iter().product();
                    assert_eq!(have, *n, "bisenet: {name} has {have} values, the graph wants {n}");
                }
            }
        }

        let ps = frozen_store(&gpu, weights);
        BiSeNet { gpu, cfg, ps, in_shape, cp, ffm, conv_out, conv_out_final }
    }

    pub fn config(&self) -> &BiSeNetConfig {
        &self.cfg
    }

    /// The device this model was built on - `crate::align::norm_crop_512`
    /// needs it to warp on the SAME handle (its kernel list is this crate's
    /// own `PIPELINES`, which is what this `Gpu` was built from).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, ids())
    }

    /// Run on a `[1,3,input_size,input_size]` CHW `[0,1]` blob, IMAGENET-
    /// normalized by the CALLER (`crate::preprocess`'s job, not this one -
    /// this crate is the network only). Returns the `[num_class,
    /// input_size, input_size]` class-logit map, upsampled bilinear
    /// (`align_corners=true`, matching the reference) from the network's
    /// native 1/8 resolution.
    pub fn forward(&self, chw_normalized: &[f32]) -> Vec<f32> {
        let ctx = self.ctx();
        assert_eq!(chw_normalized.len(), self.in_shape.numel() as usize, "bisenet: input size mismatch");
        let x = self.gpu.storage(self.in_shape.numel() as u64);
        self.gpu.write_f32(&x, chw_normalized);

        let out = self.cp.forward(&ctx, &self.ps, &x);
        let fused = self.ffm.forward(&ctx, &self.ps, &out.feat_res8, &out.feat_cp8, out.shape8);
        self.conv_out.forward(&ctx, &self.ps, &fused);
        self.conv_out_final.forward(&ctx, &self.ps, self.conv_out.out());

        let logits_shape = self.conv_out_final.out_shape;
        let final_shape = Shape::new(1, self.cfg.num_class, self.cfg.input_size, self.cfg.input_size);
        let upsampled = ctx.act(final_shape.numel());
        let s = ctx.step(
            ctx.ids.resize_bilinear,
            &[self.conv_out_final.out(), &upsampled],
            &[1, logits_shape.c, logits_shape.h, logits_shape.w, final_shape.h, final_shape.w, 1 /* align_corners = true */],
            final_shape.numel(),
        );
        self.gpu.submit(&[], &[s]);
        self.gpu.read(&upsampled, final_shape.numel() as usize)
    }
}

/// The ONE normalization every BiSeNet forward needs on its `[0,1]` RGB CHW
/// input - torchvision's standard ImageNet `(mean, std)`, per the reference
/// pipeline (`pipeline_flux.py`'s own `normalize(input, [0.485,0.456,0.406],
/// [0.229,0.224,0.225])`). Public here (not private to a caller) because
/// EVERY caller of [`BiSeNet::forward`] needs to apply it first, and
/// duplicating these six constants per caller is exactly the drift risk
/// `AGENTS.md`'s "one implementation" rule exists for.
pub fn imagenet_normalize(chw: &[f32]) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let hw = chw.len() / 3;
    let mut out = vec![0.0f32; chw.len()];
    for c in 0..3 {
        for i in 0..hw {
            out[c * hw + i] = (chw[c * hw + i] - MEAN[c]) / STD[c];
        }
    }
    out
}
