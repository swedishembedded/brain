// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared convolutional building blocks. NCHW throughout.
//!
//! * conv family — [`Conv`] (spec-driven: grouped/dilated/biased, fused and
//!   register-tiled eval paths), [`ConvTranspose`].
//! * composites — [`Bottleneck`], [`C2f`], [`SPPF`], [`CXBlock`] (ConvNeXt).
//! * stages — [`MaxPool`], [`LayerNorm2d`] (channels-first, composed).
//!
//! ## Block abstraction (the pattern, mirroring `gpt::Gpt`)
//!
//! Every block is constructed once ([`Conv::new`] etc.): it registers its
//! parameters (see [`Conv::param_list`]) and pre-allocates its SSA activation +
//! backward-temporary buffers from [`Ctx::act`]. Thereafter it offers a
//! `forward(ctx, ps, x_in) -> &out` and a `backward(ctx, ps, x_in, d_out, d_in)`
//! that record + submit their dispatch [`Step`]s.
//!
//! Why imperative submit (vs. one pre-recorded replay like `gpt`): BatchNorm's
//! `bn_train`/`bn_dx` kernels read the per-channel stats as INTERLEAVED packed
//! buffers (`mv[2C]` = mean|var, `mvg[3C]` = mean|var|gamma), but `bn_stats`
//! emits `mean[C]`/`var[C]` as separate tensors and there is no interleave
//! kernel (and P2 must add none). So between `bn_stats` and `bn_train` we must
//! interleave on the host. A model can't splice a host write into the middle of
//! a single recorded `submit`, so each block submits its forward in the natural
//! data-dependency order, host-packing the BN stats at the one boundary where it
//! is required. Backward is likewise submitted block-by-block. Buffers are still
//! SSA and the grad-accumulating kernels (`*_dw`, `bn_dgamma/beta`, residual
//! `add2`) compose exactly as in `gpt`.
//!
//! SSA discipline: every forward stage writes a FRESH buffer that doubles as the
//! activation cache the backward reads. Residual / multi-consumer grads
//! accumulate out-of-place via `add2`.
//!
//! Buffers: weights + grads come from a [`ParamStore`] keyed by the names each
//! block registers; activations + backward temporaries are plain
//! [`Ctx::act`] storage.
//!
//! Param-naming scheme (feeds the pretrained-weight mapping later), under a
//! caller-supplied prefix `P`:
//!   * `Conv`: `P.conv.weight [Cout,Cin,K,K]` (bias-free), `P.bn.gamma [C]`,
//!     `P.bn.beta [C]`, `P.bn.run_mean [C]`, `P.bn.run_var [C]`.
//!   * `Bottleneck`: convs `P.cv1`, `P.cv2`.
//!   * `C2f`: `P.cv1` (in 1x1), `P.cv2` (out 1x1), bottlenecks `P.m.{i}`.
//!   * `SPPF`: `P.cv1` (in 1x1), `P.cv2` (out 1x1).
//! This matches Ultralytics' `cv1`/`cv2`/`m` naming for later string-mapping.

use gpu_core::{f, DeviceBuffer};
use model::block as mblock;
use paramstore::ParamStore;

use crate::net::{Ctx, Shape};

/// Whether to use the weight-staged (workgroup-memory) tiled conv on the GPU.
/// Off by default: staging the full weight tile costs up to ~10–32 KiB of
/// workgroup memory, which collapses GPU occupancy and was measured SLOWER than
/// the naive conv on Intel Arc. Kept opt-in (`BRAIN_TILED_CONV=1`) for the
/// follow-up proper input+weight tiled GEMM. The work-group JIT path (solution
/// B) it exercises is correct either way.
fn use_tiled_conv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRAIN_TILED_CONV").map(|v| v != "0").unwrap_or(false))
}

/// Minimum output channels for the conv-as-GEMM eval path (`BRAIN_CONV_GEMM_MIN`,
/// default 32). Below this the direct register-tiled conv wins (large spatial /
/// few channels under-fills the GEMM tile); at/above it, im2col + matmul_reg2 is
/// 2-5x faster on the P40. Whole YOLOv8n@640 forward: 2.36x at 32.
fn gemm_conv_min_cout() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_CONV_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

/// `BRAIN_NAIVE_CONV=1` forces the naive one-output-per-invocation fused conv
/// (the previous default) instead of the register-tiled one — for comparison.
fn use_naive_conv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRAIN_NAIVE_CONV").map(|v| v != "0").unwrap_or(false))
}

/// Interleave two per-channel vectors into a `[2C]` packed buffer.
fn pack2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * a.len());
    for i in 0..a.len() {
        v.push(a[i]);
        v.push(b[i]);
    }
    v
}

// ===========================================================================
// Conv = conv2d (bias-free) -> BatchNorm -> SiLU
// ===========================================================================

/// A single `Conv` unit. Supports stride 1/2 and K=3 (pad 1) / K=1 (pad 0).

/// A conv unit's activation. An enum rather than a bare kernel id: the id alone
/// cannot express the unfused path's fwd+bwd PAIR, nor whether a fused kernel
/// exists for it, and it would let a caller set an activation the fused path
/// silently ignores.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    /// No activation (a bare conv+BN).
    None,
    /// SiLU/swish. yolo. Has a fused conv->BN(eval)->act kernel.
    Silu,
    /// ReLU. ZipDepth. Dispatched as `leaky_relu` with slope 0 — which IS relu in
    /// both directions — so it costs no kernel of its own.
    Relu,
    /// Sigmoid. A conv unit whose output is a GATE rather than a feature map:
    /// ZipDepth's `StripPoolingAttention` ends `Conv->BN->Sigmoid` and multiplies
    /// the result into `x`. Never fuses (the fused kernel is SiLU-only).
    Sigmoid,
    /// GELU, **tanh approximation** (`0.5x(1+tanh(k(x+0.044715x^3)))`) — the
    /// GPT-2 spelling. Not the same function as [`Act::GeluErf`]; picking the
    /// wrong one is a silent ~1e-3 output shift, not an error.
    Gelu,
    /// GELU, **exact erf form** (`0.5x(1+erf(x/sqrt 2))`) — torch's default
    /// `nn.GELU()`, and therefore what ConvNeXt / SAM 2 / VQGAN checkpoints were
    /// trained with. The default for [`CXBlock`].
    GeluErf,
}

impl Act {
    /// The FUSED kernels' activation selector — `conv_act*`'s and `bn_eval`'s
    /// `p.act` — or `None` for an activation those kernels do not implement.
    ///
    /// `None` is load-bearing, not a formality. Their WGSL branches on exactly
    /// `1u`/`2u`/`3u` and falls through to the IDENTITY otherwise, so pushing a
    /// `4` for GELU would produce a conv whose activation silently vanished:
    /// plausible numbers, wrong model. Everything that consults this must treat
    /// `None` as "take the unfused path", never as "use 0".
    pub fn fused_code(self) -> Option<u32> {
        match self {
            Act::None => Some(0),
            Act::Relu => Some(1),
            Act::Silu => Some(2),
            Act::Sigmoid => Some(3),
            Act::Gelu | Act::GeluErf => None,
        }
    }
}

/// The unfused activation's `(forward, backward)` kernel pair, or `None` for
/// [`Act::None`]. ReLU maps to `leaky_relu` at slope 0 — identical in both
/// directions — so it needs no kernel of its own.
///
/// A free function rather than a method: `Conv`, [`ConvTranspose`] and
/// [`CXBlock`]'s pointwise stage all need the same pairing, and a second copy is
/// how the ReLU-is-leaky_relu-at-slope-0 trick drifts.
fn act_pair(ctx: &Ctx, act: Act) -> Option<(usize, usize)> {
    match act {
        Act::None => None,
        Act::Silu => Some((ctx.ids.need(ctx.ids.silu, "silu"), ctx.ids.need(ctx.ids.silu_bwd, "silu_bwd"))),
        Act::Relu => Some((
            ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
            ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"),
        )),
        Act::Sigmoid => {
            Some((ctx.ids.need(ctx.ids.sigmoid, "sigmoid"), ctx.ids.need(ctx.ids.sigmoid_bwd, "sigmoid_bwd")))
        }
        Act::Gelu => Some((ctx.ids.need(ctx.ids.gelu, "gelu"), ctx.ids.need(ctx.ids.gelu_bwd, "gelu_bwd"))),
        Act::GeluErf => {
            Some((ctx.ids.need(ctx.ids.gelu_erf, "gelu_erf"), ctx.ids.need(ctx.ids.gelu_erf_bwd, "gelu_erf_bwd")))
        }
    }
}

/// The activation kernels' uniform stream. `leaky_relu`'s is `[total, slope]`
/// with `slope` a bit-cast f32; every other one's is `[total]`. Slope 0 makes
/// `leaky_relu` exactly ReLU.
fn act_params(act: Act, n: u32) -> Vec<u32> {
    match act {
        Act::Relu => vec![n, f(0.0)],
        _ => vec![n],
    }
}

/// Accumulate a per-output-channel bias gradient from `d_out`, the grad wrt the
/// biased unit's output: `dbias[c] = sum over n,h,w of d_out[n,c,h,w]`.
///
/// The `bias_grad` kernel is `[M,N]` row-major and reduces over M, so `d_out` is
/// viewed as `[M=N, N=C*HW]`: element `(n, c*HW+p)` IS `d_out[n,c,h,w]`, and the
/// kernel gives `dbcast[c*HW+p] = sum_n d_out[n,c,h,w]`. The remaining spatial
/// sum is done on the host — the same host-reduce split `gradnorm_sq` uses.
///
/// Shared by [`Conv`] and [`ConvTranspose`]: the reduction is a property of the
/// NCHW layout, not of which convolution produced it.
///
/// Must run AFTER the caller's submit — it reads `d_out` back.
fn accumulate_bias_grad(
    ctx: &Ctx,
    ps: &ParamStore,
    name: &str,
    dbcast: &DeviceBuffer,
    shape: Shape,
    d_out: &DeviceBuffer,
) {
    let (cout, hw) = (shape.c, shape.h * shape.w);
    let n = cout * hw;
    // bias_grad ACCUMULATES into its output, and `dbcast` persists across
    // backward passes -> it must be zeroed first (submit's clear list).
    let s = ctx.step(ctx.ids.need(ctx.ids.bias_grad, "bias_grad"), &[d_out, dbcast], &[shape.n, n], n);
    ctx.gpu.submit(&[dbcast], &[s]);
    let host = ctx.gpu.read(dbcast, n as usize);
    let cur = ctx.gpu.read(ps.g(name), cout as usize);
    let merged: Vec<f32> = (0..cout as usize)
        .map(|ch| cur[ch] + host[ch * hw as usize..(ch + 1) * hw as usize].iter().sum::<f32>())
        .collect();
    ctx.gpu.write(ps.g(name), bytemuck::cast_slice(&merged));
}

/// Read a unit's input to the host, let an installed [`crate::ActTap`] observe
/// (and optionally rewrite in place — quant->dequant) it keyed by `prefix`, and
/// return a device copy of the tapped bytes. `None` when no tap is installed
/// (every normal inference), so the caller convolves its input directly and pays
/// zero cost.
///
/// Fires ONCE per unit at the unit's input — the same point the exported ONNX
/// graph inserts its Q/DQ pair. `scratch` is the caller's lazily-allocated
/// staging buffer, reused across frames.
fn apply_tap(
    ctx: &Ctx,
    prefix: &str,
    in_numel: usize,
    scratch: &std::cell::RefCell<Option<DeviceBuffer>>,
    x_in: &DeviceBuffer,
) -> Option<DeviceBuffer> {
    let tap = ctx.tap?;
    let mut h = ctx.gpu.read(x_in, in_numel);
    tap.tap(prefix, &mut h);
    if scratch.borrow().is_none() {
        *scratch.borrow_mut() = Some(ctx.gpu.storage(in_numel as u64));
    }
    let q = scratch.borrow();
    let qbuf = q.as_ref().unwrap().clone();
    ctx.gpu.write(&qbuf, bytemuck::cast_slice(&h));
    Some(qbuf)
}

/// Whether a conv unit carries its own BatchNorm.
///
/// `None` is not an optimisation — several ZipDepth blocks genuinely have raw
/// convs: `MinimalMultiScale`'s two depthwise branches share ONE BN over their
/// sum (so the branches must have none of their own), and SE's two 1x1s have no
/// BN at all. A unit whose BN is mandatory cannot express either.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    None,
    Bn,
}

/// Everything that varies between conv units. Immutable after construction:
/// `sb` caches the BN-eval collapse and is invalidated only on a train/eval
/// flip, so a runtime-mutable activation or grouping would leave it stale.
#[derive(Clone, Copy, Debug)]
pub struct ConvSpec {
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub groups: u32,
    pub dilation: u32,
    pub norm: Norm,
    pub act: Act,
    /// A learned per-output-channel bias (`nn.Conv2d(.., bias=True)`).
    ///
    /// TWO dispatch paths, picked by [`ConvSpec::is_dense`]:
    ///   * dense -> the FUSED `conv_bias` / `conv_bias_reg` kernel, one pass.
    ///   * grouped/dilated -> `conv2d_gd` followed by `add_chan_inplace`
    ///     (`out[n,c,hw] += bias[c]`), because `conv_bias`'s uniform carries no
    ///     groups/dilation and would silently convolve as if dense.
    /// The second path exists for ConvNeXt: [`CXBlock`]'s 7x7 depthwise conv is
    /// grouped AND biased. Every biased conv in ZipDepth is dense and keeps the
    /// fused path unchanged.
    ///
    /// `bias` is independent of `norm`: ZipDepth's `GlobalContextBlock.transform.0`
    /// is a biased conv followed by BN, where the bias is mathematically redundant
    /// (BN subtracts the batch mean) but is PRESENT in the checkpoint and must be
    /// loaded and updated, or a strict load fails.
    pub bias: bool,
}

impl ConvSpec {
    /// yolo's unit: dense, SiLU.
    pub const fn silu(cout: u32, k: u32, stride: u32, pad: u32) -> ConvSpec {
        ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::Bn, act: Act::Silu, bias: false }
    }
    /// ZipDepth's `ConvBN`: grouped/dilated, ReLU.
    pub const fn relu(cout: u32, k: u32, stride: u32, pad: u32) -> ConvSpec {
        ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::Bn, act: Act::Relu, bias: false }
    }
    /// Add a learned per-channel bias.
    ///
    /// No longer restricted to dense units: a grouped/dilated biased conv runs
    /// `conv2d_gd -> add_chan_inplace` (ConvNeXt's 7x7 depthwise conv is grouped
    /// AND biased). Dense units keep the fused `conv_bias` / `conv_bias_reg`
    /// single pass unchanged — see [`ConvSpec::bias`].
    pub const fn with_bias(self) -> ConvSpec {
        ConvSpec { bias: true, ..self }
    }
    pub const fn with_groups(self, groups: u32) -> ConvSpec {
        ConvSpec { groups, ..self }
    }
    pub const fn with_dilation(self, dilation: u32) -> ConvSpec {
        ConvSpec { dilation, ..self }
    }
    pub const fn with_act(self, act: Act) -> ConvSpec {
        ConvSpec { act, ..self }
    }
    /// Drop this unit's BatchNorm — it becomes a raw conv.
    pub const fn with_norm(self, norm: Norm) -> ConvSpec {
        ConvSpec { norm, ..self }
    }
    /// Depthwise: groups == cin == cout, weight `[C,1,k,k]`.
    pub const fn depthwise(ch: u32, k: u32, stride: u32, pad: u32, act: Act) -> ConvSpec {
        ConvSpec { cout: ch, k, stride, pad, groups: ch, dilation: 1, norm: Norm::Bn, act, bias: false }
    }
    /// Is this the plain dense case the pre-existing `conv2d` kernel covers?
    ///
    /// The distinction is load-bearing: `backend-cpu` binds an AVX2/winograd fast
    /// path to the NAME `conv2d`, and that path is dense — it ignores `groups`
    /// entirely. Routing a grouped conv there would compute wrong results with no
    /// error, so grouping/dilation MUST go to `conv2d_gd`.
    pub fn is_dense(&self) -> bool {
        self.groups == 1 && self.dilation == 1
    }
    pub fn out_shape(&self, x: Shape) -> Shape {
        x.conv_out_dilated(self.cout, self.k, self.stride, self.pad, self.dilation)
    }
}


/// The six tensor names a conv-shaped unit owns.
///
/// Shared by [`Conv`] and [`ConvTranspose`] — a transposed conv spells its
/// weight and bias exactly like a forward one; only the tensor LAYOUT differs,
/// and that is the spec's business, not the name's. A norm-free unit
/// (`Norm::None`, and every `ConvTranspose`) simply never reads the four
/// BatchNorm names.
///
/// A property of WHERE THE WEIGHTS CAME FROM, not of the block — which is why it
/// is data rather than a hardcoded `format!`. Two models share this block and
/// spell BatchNorm differently:
///   * yolo mirrors Ultralytics: `P.conv.weight` + `P.bn.{gamma,beta,run_mean,run_var}`
///   * ZipDepth mirrors its own torch checkpoint: `P.bn.{weight,bias,running_mean,
///     running_var}`, and inside a `nn.Sequential` the conv+BN are indexed
///     POSITIONALLY (`P.0.weight`, `P.1.weight`) with no `.conv`/`.bn` at all.
///
/// Making this configurable is what lets each model's `param_list` mirror its own
/// checkpoint exactly, so import is a 1:1 name match instead of the hand-written
/// translation table every other brain importer carries.
#[derive(Clone, Debug)]
pub struct ConvNames {
    /// The conv's own bias tensor. Read only when `ConvSpec::bias` is set; every
    /// name-builder fills it with torch's own convention (`<conv>.bias`) so a
    /// caller that flips `with_bias()` on gets the right name for free.
    pub bias: String,
    pub weight: String,
    pub gamma: String,
    pub beta: String,
    pub run_mean: String,
    pub run_var: String,
}

impl ConvNames {
    /// brain / yolo / Ultralytics: `P.conv.weight` + `P.bn.{gamma,beta,run_mean,run_var}`.
    pub fn brain(prefix: &str) -> ConvNames {
        ConvNames {
            bias: format!("{prefix}.conv.bias"),
            weight: format!("{prefix}.conv.weight"),
            gamma: format!("{prefix}.bn.gamma"),
            beta: format!("{prefix}.bn.beta"),
            run_mean: format!("{prefix}.bn.run_mean"),
            run_var: format!("{prefix}.bn.run_var"),
        }
    }
    /// A torch module with `.conv` / `.bn` attributes (ZipDepth's `ConvBN`):
    /// `P.conv.weight` + `P.bn.{weight,bias,running_mean,running_var}`.
    pub fn torch_conv_bn(prefix: &str) -> ConvNames {
        ConvNames {
            bias: format!("{prefix}.conv.bias"),
            weight: format!("{prefix}.conv.weight"),
            gamma: format!("{prefix}.bn.weight"),
            beta: format!("{prefix}.bn.bias"),
            run_mean: format!("{prefix}.bn.running_mean"),
            run_var: format!("{prefix}.bn.running_var"),
        }
    }
    /// A BARE torch conv module at path `P`: `P.weight` + `P.bias`, with no
    /// `.conv`/`.bn` infix at all.
    ///
    /// This is the shape every `nn.ConvTranspose2d` and every `nn.Linear`-as-1x1
    /// takes, and it covers the positional case too because the caller owns the
    /// prefix: SAM 2's mask decoder is
    /// `nn.Sequential(ConvTranspose2d, LayerNorm2d, GELU, ConvTranspose2d, GELU)`,
    /// so its first deconv is `torch_flat("output_upscaling.0")`, while a
    /// VQGAN/CodeFormer decoder names the same module `…up.3.upsample.conv`.
    /// One convention, two checkpoints, no translation table.
    ///
    /// The BatchNorm names are filled with torch's `P.bn.*` spelling so a unit
    /// that later gains a `Norm::Bn` gets sane names for free; a `ConvTranspose`
    /// never reads them.
    pub fn torch_flat(prefix: &str) -> ConvNames {
        ConvNames {
            bias: format!("{prefix}.bias"),
            weight: format!("{prefix}.weight"),
            gamma: format!("{prefix}.bn.weight"),
            beta: format!("{prefix}.bn.bias"),
            run_mean: format!("{prefix}.bn.running_mean"),
            run_var: format!("{prefix}.bn.running_var"),
        }
    }
    /// A torch `nn.Sequential(Conv2d, BatchNorm2d)` (ZipDepth's QARepBlock
    /// branches, gate_conv, transform, mask_pred): children are indexed by
    /// POSITION, so `P.{ci}.weight` + `P.{bi}.{weight,bias,running_mean,running_var}`.
    pub fn torch_seq(prefix: &str, conv_idx: u32, bn_idx: u32) -> ConvNames {
        ConvNames {
            bias: format!("{prefix}.{conv_idx}.bias"),
            weight: format!("{prefix}.{conv_idx}.weight"),
            gamma: format!("{prefix}.{bn_idx}.weight"),
            beta: format!("{prefix}.{bn_idx}.bias"),
            run_mean: format!("{prefix}.{bn_idx}.running_mean"),
            run_var: format!("{prefix}.{bn_idx}.running_var"),
        }
    }
}

/// Which checkpoint naming convention a composite block gives its inner convs.
///
/// A composite (`SPPF`, `C2f`, ...) builds its own children's prefixes, so it —
/// not the caller — decides their tensor names. Passing the convention rather than
/// the names lets one block serve models whose checkpoints disagree: yolo's SPPF
/// is `cv1.bn.gamma`, ZipDepth's is `cv1.bn.weight`, and the wiring is identical.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NameStyle {
    /// brain / yolo: `P.conv.weight` + `P.bn.{gamma,beta,run_mean,run_var}`.
    Brain,
    /// torch `ConvBN`-style module: `P.conv.weight` + `P.bn.{weight,bias,running_*}`.
    TorchConvBn,
    /// A bare torch conv/deconv module: `P.weight` + `P.bias`.
    TorchFlat,
}

impl NameStyle {
    pub fn names(self, prefix: &str) -> ConvNames {
        match self {
            NameStyle::Brain => ConvNames::brain(prefix),
            NameStyle::TorchConvBn => ConvNames::torch_conv_bn(prefix),
            NameStyle::TorchFlat => ConvNames::torch_flat(prefix),
        }
    }
}

/// Everything that varies between SPPF instances.
///
/// The pooled width is `hidden`, NOT a fixed fraction: Ultralytics derives it from
/// the OUTPUT channels (`c_out/2`) and ZipDepth's `LightweightSPPF` from the INPUT
/// channels (`c1/4`). Those coincide often enough to hide the difference, so the
/// caller states it.
#[derive(Clone, Copy, Debug)]
pub struct SppfSpec {
    /// cv1's output channels — the width the 3 pools and the 4-way concat run at.
    pub hidden: u32,
    pub c_out: u32,
    pub act: Act,
}

pub struct Conv {
    /// The ActTap key. MUST equal the exported ONNX node name — the NPU
    /// calibrator maps its per-tensor scales by this string. Kept separate from
    /// [`ConvNames`]: the tap identifies the CONV SITE, the names identify its
    /// weights, and they are not the same concept.
    prefix: String,
    names: ConvNames,
    pub in_shape: Shape,
    pub out_shape: Shape,
    pub spec: ConvSpec,
    k: u32,
    stride: u32,
    pad: u32,
    /// Train-mode BN (batch stats) vs eval-mode BN (running stats). Interior-
    /// mutable so inference can flip every Conv to eval-mode BN via
    /// [`Conv::set_eval`] WITHOUT rebuilding the graph (the only thing the flag
    /// changes is which BN kernel `forward` dispatches; see P6 infer).
    train: std::cell::Cell<bool>,
    momentum: f32,
    /// Apply the running-stat momentum EMA update during forward. Interior-mutable
    /// (like `train`) so real training can enable it via [`Conv::set_update_running`]
    /// without rebuilding the graph. Disabled for the gradient check (it mutates
    /// `run_mean`/`run_var`, breaking forward determinism; those tensors carry no
    /// train-mode gradient anyway), so it defaults OFF.
    update_running: std::cell::Cell<bool>,

    conv_out: DeviceBuffer, // post-conv pre-BN [out]
    mean: DeviceBuffer,     // batch mean [C]
    var: DeviceBuffer,      // batch var  [C]
    mv: DeviceBuffer,       // packed mean|var [2C]
    gb: DeviceBuffer,       // packed gamma|beta [2C]
    sb: DeviceBuffer,       // packed scale|bias [2C] = BN-eval collapsed (fused conv_act)
    /// Whether `sb` holds the current BN-eval collapse. Computed lazily on the
    /// first eval-mode forward and reused across frames (constant in inference);
    /// invalidated when the block re-enters train mode.
    sb_ready: std::cell::Cell<bool>,
    mvg: DeviceBuffer,      // packed mean|var|gamma [3C]
    bn_out: DeviceBuffer,   // post-BN pre-SiLU [out]
    act: DeviceBuffer,      // SiLU output (block output) [out]

    d_bn: DeviceBuffer,   // grad wrt bn_out [out]
    bp: DeviceBuffer,     // packed [5C] from bn_dstats
    d_conv: DeviceBuffer, // grad wrt conv_out [out]

    /// `bias_grad`'s output before the host spatial reduce: `[C*HW]`. Allocated
    /// only for a biased unit (see `bias_backward` for why the view is [N, C*HW]).
    dbcast: Option<DeviceBuffer>,

    /// Lazily-allocated [in] scratch holding the tapped (possibly fake-quantized)
    /// conv input, used only when a [`crate::ActTap`] is installed (NPU
    /// calibration / fake-quant). Never allocated on the normal inference path.
    q_in: std::cell::RefCell<Option<DeviceBuffer>>,
    /// Lazily-allocated `[Ho*Wo, Cin*K*K]` im2col scratch for the conv-as-GEMM
    /// eval path (dense convs with large Cout, where `conv_act_reg` collapses on
    /// the P40). Allocated once, reused across frames.
    col: std::cell::RefCell<Option<DeviceBuffer>>,
}

impl Conv {
    /// yolo's ctor, unchanged: dense conv + BN + SiLU.
    // The 8 positional args ARE this ctor's contract (yolo's call sites are
    // frozen by its checkpoint layout); the spec-driven `with_spec` is the
    // non-positional alternative.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        prefix: &str,
        in_shape: Shape,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        train: bool,
    ) -> Conv {
        Conv::with_spec(ctx, prefix, in_shape, ConvSpec::silu(cout, k, stride, pad), train)
    }

    /// The general ctor: any grouping/dilation/activation, brain-style names.
    pub fn with_spec(ctx: &Ctx, prefix: &str, in_shape: Shape, spec: ConvSpec, train: bool) -> Conv {
        Conv::with_names(ctx, prefix, ConvNames::brain(prefix), in_shape, spec, train)
    }

    /// The fully general ctor: the caller supplies the tensor names, so each
    /// model's param list can mirror its own checkpoint.
    pub fn with_names(
        ctx: &Ctx,
        prefix: &str,
        names: ConvNames,
        in_shape: Shape,
        spec: ConvSpec,
        train: bool,
    ) -> Conv {
        let (cout, k, stride, pad) = (spec.cout, spec.k, spec.stride, spec.pad);
        assert_eq!(in_shape.c % spec.groups, 0, "cin {} not divisible by groups {}", in_shape.c, spec.groups);
        assert_eq!(cout % spec.groups, 0, "cout {cout} not divisible by groups {}", spec.groups);
        let out_shape = spec.out_shape(in_shape);
        let on = out_shape.numel();
        let c = cout;
        Conv {
            prefix: prefix.to_string(),
            names,
            in_shape,
            out_shape,
            spec,
            k,
            stride,
            pad,
            train: std::cell::Cell::new(train),
            // Running-stat EMA momentum. PyTorch's default (0.03) converges too
            // slowly for the short from-scratch runs here, so use 0.1 — the BN
            // running mean/var reach usable eval-mode values in a few hundred
            // steps (validated by the p11 eval-inference test).
            momentum: 0.1,
            update_running: std::cell::Cell::new(false),
            conv_out: ctx.act(on),
            mean: ctx.act(c),
            var: ctx.act(c),
            mv: ctx.act(2 * c),
            gb: ctx.act(2 * c),
            sb: ctx.act(2 * c),
            sb_ready: std::cell::Cell::new(false),
            mvg: ctx.act(3 * c),
            bn_out: ctx.act(on),
            act: ctx.act(on),
            d_bn: ctx.act(on),
            bp: ctx.act(5 * c),
            d_conv: ctx.act(on),
            dbcast: if spec.bias { Some(ctx.act(cout * out_shape.h * out_shape.w)) } else { None },
            q_in: std::cell::RefCell::new(None),
            col: std::cell::RefCell::new(None),
        }
    }

    /// This unit's conv-weight tensor name.
    pub fn names_weight(&self) -> &str {
        &self.names.weight
    }

    pub fn out(&self) -> &DeviceBuffer {
        // A raw conv with no activation IS its conv output: dispatching a
        // slope-1 leaky_relu just to copy it into `act` cost a full extra
        // memory pass AND a dependent-dispatch hop per unit (ZipDepth has ~12
        // such units per frame: the fusion/cross-scale projections and the
        // MinimalMultiScale branches). Alias instead; forward/backward skip
        // the copy dispatches to match.
        if self.spec.norm == Norm::None && self.spec.act == Act::None {
            &self.conv_out
        } else {
            &self.act
        }
    }

    /// Flip this Conv's BN to eval-mode (running stats) or train-mode (batch
    /// stats). Inference-only concern: it changes which BN kernel `forward`
    /// dispatches, never the graph or any buffer.
    pub fn set_eval(&self, eval: bool) {
        // Re-entering train mode invalidates the cached BN-eval collapse so the
        // next eval recomputes it from the (now updated) running stats / affine.
        if !eval {
            self.sb_ready.set(false);
        }
        self.train.set(!eval);
    }

    /// True iff this Conv is in eval-mode BN (running stats).
    pub fn is_eval(&self) -> bool {
        !self.train.get()
    }

    /// Enable/disable the BN running-stat momentum EMA update during train-mode
    /// forward. Must be ON during real training so `run_mean`/`run_var` track the
    /// data and eval-mode inference works; left OFF for the gradient check.
    pub fn set_update_running(&self, on: bool) {
        self.update_running.set(on);
    }

    /// The unit's tensor names — read-only. Lets a block-level fusion (e.g.
    /// depth's QARep RepVGG collapse) read this conv's weights/BN stats from
    /// the ParamStore without duplicating the naming rules.
    pub fn names(&self) -> &ConvNames {
        &self.names
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.out_shape.c as usize;
        // Grouped conv weights are `[cout, cin/groups, k, k]` — NOT `[cout, cin,
        // k, k]`. Depthwise (groups == cin) makes the second axis 1.
        let cin_g = (self.in_shape.c / self.spec.groups) as usize;
        let k = self.k as usize;
        let mut v = vec![(self.names.weight.clone(), c * cin_g * k * k)];
        if self.spec.bias {
            v.push((self.names.bias.clone(), c));
        }
        if self.spec.norm == Norm::Bn {
            v.push((self.names.gamma.clone(), c));
            v.push((self.names.beta.clone(), c));
            v.push((self.names.run_mean.clone(), c));
            v.push((self.names.run_var.clone(), c));
        }
        v
    }

    /// The conv uniform stream. TWO ABIs, picked by the spec:
    ///   dense  -> conv2d's 10-u32 `[N,Cin,H,W,Cout,K,stride,pad,Ho,Wo]`
    ///   gd     -> conv2d_gd's 12-u32, with `dilation,groups` before `Ho,Wo`
    /// The legacy form is preserved exactly for the dense case so yolo's uniform
    /// layout — and its AVX2/winograd fast path, which is bound to the NAME
    /// `conv2d` — are untouched.
    fn conv_params(&self) -> Vec<u32> {
        let mut v = vec![
            self.in_shape.n,
            self.in_shape.c,
            self.in_shape.h,
            self.in_shape.w,
            self.out_shape.c,
            self.k,
            self.stride,
            self.pad,
        ];
        if !self.spec.is_dense() {
            v.push(self.spec.dilation);
            v.push(self.spec.groups);
        }
        v.push(self.out_shape.h);
        v.push(self.out_shape.w);
        v
    }

    /// Forward conv kernel: the dense fast-pathed one, or the grouped/dilated one.
    fn conv_kind(&self, ctx: &Ctx) -> usize {
        if self.spec.is_dense() {
            ctx.ids.need(ctx.ids.conv2d, "conv2d")
        } else {
            ctx.ids.need(ctx.ids.conv2d_gd, "conv2d_gd")
        }
    }
    /// The forward conv dispatch, biased or not. EVERY forward path goes through
    /// this — the raw-conv path, the train path and the unfused-eval path — so a
    /// biased unit cannot silently lose its bias on one of the three.
    ///
    /// Returns one step for the dense cases and TWO for a grouped/dilated biased
    /// unit, which has no fused kernel. The caller must submit them **in order**:
    /// the second reads what the first wrote.
    ///
    /// `conv_bias` fuses the per-channel NCHW add into the conv's own pass. It is
    /// NOT `bias_add`, which is an [M,N] LINEAR-layer bias (`out[idx] += b[idx % N]`,
    /// biased dim TRAILING) and indexes garbage in NCHW.
    fn conv_steps(&self, ctx: &Ctx, ps: &ParamStore, src: &DeviceBuffer, dst: &DeviceBuffer) -> Vec<gpu_core::Step> {
        let on = self.out_shape.numel();
        if self.spec.bias && self.spec.is_dense() {
            // Register-tiled variant when registered: same math, ~8x less input
            // traffic on the GPU (`conv_bias` is dense-only, so the reg tiling
            // always applies). CPU routes both names to the same fast path.
            let (kind, threads) = if ctx.ids.conv_bias_reg != crate::NONE {
                (ctx.ids.conv_bias_reg, self.reg_threads())
            } else {
                (ctx.ids.need(ctx.ids.conv_bias, "conv_bias"), on)
            };
            return vec![ctx.step(
                kind,
                &[src, ps.w(&self.names.weight), ps.w(&self.names.bias), dst],
                &self.conv_params(),
                threads,
            )];
        }
        let conv = if !self.spec.is_dense() && ctx.ids.conv2d_gd_reg != crate::NONE {
            // Grouped/dilated register-tiled variant: group-aligned 8x4 tile,
            // ~8x less input traffic for the grouped 1x1 projections. Same math
            // as conv2d_gd (CPU routes both names to one fast path).
            ctx.step(
                ctx.ids.conv2d_gd_reg,
                &[src, ps.w(&self.names.weight), dst],
                &self.conv_params(),
                self.gd_reg_threads(),
            )
        } else {
            ctx.step(self.conv_kind(ctx), &[src, ps.w(&self.names.weight), dst], &self.conv_params(), on)
        };
        let mut v = vec![conv];
        if self.spec.bias {
            // Grouped/dilated + bias: `conv_bias` has no grouped form, so the
            // per-channel add is a second pass. `add_chan_inplace`'s uniform is
            // [total, C, HW] with a SINGLE read_write output binding — it is the
            // NCHW form; `bias_add` is the [M,N] linear one and would index
            // garbage here.
            v.push(ctx.step(
                ctx.ids.need(ctx.ids.add_chan_inplace, "add_chan_inplace"),
                &[dst, ps.w(&self.names.bias)],
                &[on, self.out_shape.c, self.out_shape.h * self.out_shape.w],
                on,
            ));
        }
        v
    }
    fn conv_dx_kind(&self, ctx: &Ctx) -> usize {
        if self.spec.is_dense() {
            ctx.ids.need(ctx.ids.conv2d_dx, "conv2d_dx")
        } else {
            ctx.ids.need(ctx.ids.conv2d_gd_dx, "conv2d_gd_dx")
        }
    }
    fn conv_dw_kind(&self, ctx: &Ctx) -> usize {
        if self.spec.is_dense() {
            ctx.ids.need(ctx.ids.conv2d_dw, "conv2d_dw")
        } else {
            ctx.ids.need(ctx.ids.conv2d_gd_dw, "conv2d_gd_dw")
        }
    }
    /// Can this unit take the fused conv->BN(eval)->act path?
    ///
    /// Any activation the `conv_act*` kernels implement fuses — they take the act
    /// selector in their uniform (0 identity, 1 relu, 2 silu, 3 sigmoid), so a
    /// ReLU model (ZipDepth) fuses exactly like a SiLU one (yolo). What still
    /// can't fuse: GELU (no selector code — see [`Act::fused_code`]),
    /// grouped/dilated convs (the fused kernels and the CPU fast path they route
    /// to are dense — binding a grouped unit would silently ignore `groups`),
    /// and biased units (`conv_bias` is its own kernel). Those run the unfused
    /// `conv -> bn_eval -> act`, whose eps matches `pack_sb`'s (`1e-5`), so the
    /// two paths agree numerically by construction.
    ///
    /// `pub` so tests can pin that a given spec actually takes the fused path —
    /// a silent fall-back to unfused is a 3x-dispatch perf regression that no
    /// output comparison would ever catch.
    pub fn can_fuse(&self, ctx: &Ctx) -> bool {
        self.spec.norm == Norm::Bn
            && !self.spec.bias
            && self.spec.is_dense()
            && self.spec.act.fused_code().is_some()
            && ctx.ids.conv_act_reg != crate::NONE
    }

    /// `conv_params()` + the act selector — the 11-u32 uniform stream of the
    /// fused `conv_act*` kernels. Only reachable when [`Conv::can_fuse`] holds,
    /// which is what guarantees the selector exists.
    fn fused_params(&self) -> Vec<u32> {
        let mut v = self.conv_params();
        v.push(self.spec.act.fused_code().expect("can_fuse() gates this on a selector existing"));
        v
    }
    /// Invocation count for the register-tiled conv kernels (`conv_act_reg`,
    /// `conv_bias_reg`): one invocation per 8-channel x 4-position tile.
    fn reg_threads(&self) -> u32 {
        let ntc = self.out_shape.c.div_ceil(8);
        let npq = (self.out_shape.h * self.out_shape.w).div_ceil(4);
        self.out_shape.n * ntc * npq
    }

    /// Invocation count for `conv2d_gd_reg`: octets are GROUP-ALIGNED, so the
    /// channel-tile count is `groups * ceil(cout_g/8)`, not `ceil(cout/8)` —
    /// they differ whenever `cout_g % 8 != 0` (e.g. depthwise: cout_g = 1).
    fn gd_reg_threads(&self) -> u32 {
        let cout_g = self.out_shape.c / self.spec.groups;
        let ntc = self.spec.groups * cout_g.div_ceil(8);
        let npq = (self.out_shape.h * self.out_shape.w).div_ceil(4);
        self.out_shape.n * ntc * npq
    }

    /// Invocation count for the weight-tiled conv: one workgroup (64 invocations)
    /// per `(n, output-channel, 64-output-position block)`.
    fn tiled_threads(&self) -> u32 {
        let psz = self.out_shape.h * self.out_shape.w;
        let blocks = psz.div_ceil(64);
        self.out_shape.n * self.out_shape.c * blocks * 64
    }
    fn nchw(&self) -> [u32; 4] {
        [self.out_shape.n, self.out_shape.c, self.out_shape.h, self.out_shape.w]
    }

    /// Pack gamma|beta into `gb` from the current weights (host). BN affine
    /// params can't be aliased as the interleaved buffer the kernel wants.
    fn pack_gb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        let beta = ctx.gpu.read(ps.w(&self.names.beta), c);
        ctx.gpu.write(&self.gb, bytemuck::cast_slice(&pack2(&gamma, &beta)));
    }

    /// Run the full forward and return this block's output buffer. Submits in
    /// dependency order, host-packing the BN stats at the one required boundary.
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let on = self.out_shape.numel();
        let c = self.out_shape.c;

        if self.spec.norm == Norm::None {
            // Raw conv -> act. No stats, no host interleave, no mode distinction:
            // without BN the two modes compute the same function. The
            // calibration/fake-quant tap still fires here — many of ZipDepth's
            // DECODER convs (the fusion projections, the head, SE, GCB's raw
            // convs) are Norm::None, and those are exactly the layers QuartDepth
            // flags as quant-sensitive, so skipping them would blind the report.
            let tapped = self.tap_input(ctx, x_in);
            let src = tapped.as_ref().unwrap_or(x_in);
            let s_conv = self.conv_steps(ctx, ps, src, &self.conv_out);
            ctx.gpu.submit(&[], &s_conv);
            // Act::None: `out()` aliases `conv_out` — no copy dispatch.
            if let Some((fwd, _)) = act_pair(ctx, self.spec.act) {
                let s_act = ctx.step(fwd, &[&self.conv_out, &self.act], &act_params(self.spec.act, on), on);
                ctx.gpu.submit(&[], &[s_act]);
            }
            return;
        }

        if !self.train.get() {
            // Inference: one fused conv -> BN(eval) -> SiLU dispatch. The BN-eval
            // transform is collapsed per channel into `sb` once (constant across
            // frames), so there is no per-frame host stat packing nor separate
            // bn_eval/silu passes.
            if !self.can_fuse(ctx) {
                // Unfused eval: conv -> bn_eval -> act. Taken by the units the
                // fused path cannot serve — grouped/dilated convs (the fused
                // kernels and the dense CPU fast path ignore `groups`) and any
                // unit whose registry omits `conv_act_reg`.
                self.forward_eval_unfused(ctx, ps, x_in);
                return;
            }
            if !self.sb_ready.get() {
                self.pack_sb(ctx, ps);
                self.sb_ready.set(true);
            }
            // Conv-as-GEMM fast path (P40 / compute-bound GPUs): a dense conv
            // with many output channels is far faster as im2col + matmul_reg2 +
            // per-channel-affine+SiLU epilogue than the direct register-tiled
            // conv, which collapses on deep small-spatial layers (measured 2-5x
            // on YOLOv8n's stage2-4). Same math — parity-gated by the detection
            // test. Gated on: the three kernels registered, a dense single-batch
            // SiLU conv past the tile threshold, and no calibration tap (the tap
            // path stays on the direct conv). `BRAIN_CONV_GEMM=0` disables it.
            if self.gemm_eval_eligible(ctx) {
                self.forward_eval_gemm(ctx, ps, x_in);
                return;
            }
            // Fused conv -> BN(eval) -> SiLU. The weight-tiled variant (opt-in)
            // exercises the single-source work-group kernel; the default naive
            // variant is faster on current GPUs (full occupancy). Both route to
            // the same native AVX2 fast path on CPU.
            let (kind, threads) = if use_tiled_conv() {
                (ctx.ids.conv_act_tiled, self.tiled_threads())
            } else if use_naive_conv() {
                (ctx.ids.conv_act, self.out_shape.numel())
            } else {
                // Default: register-tiled — each invocation computes an 8x4 tile
                // (8 output channels x 4 positions), reusing weight + input loads
                // across it (no workgroup memory, full occupancy). Each strided
                // NCHW input load feeds all 8 channels -> input traffic ~/8.
                (ctx.ids.conv_act_reg, self.reg_threads())
            };
            // Calibration / fake-quant tap (NPU INT8): route the conv input
            // through the host so the tap can read its range and/or rewrite it
            // (quant→dequant), then convolve the tapped copy. `tap_input` is the
            // ONE tap implementation — the raw-conv and unfused-eval paths call
            // the same one, so a tap can never fire on two of the three and be
            // silently skipped on the third. `None` on every untapped forward,
            // which then convolves `x_in` directly at zero cost.
            let tapped = self.tap_input(ctx, x_in);
            let src = tapped.as_ref().unwrap_or(x_in);
            let s = ctx.step(
                kind,
                &[src, ps.w(&self.names.weight), &self.sb, &self.act],
                &self.fused_params(),
                threads,
            );
            ctx.gpu.submit(&[], &[s]);
            return;
        }

        // Train mode: conv -> bn_stats, host-pack mv/mvg, then bn_train -> silu.
        self.pack_gb(ctx, ps);
        let mut pre = self.conv_steps(ctx, ps, x_in, &self.conv_out);
        pre.push(ctx.step(ctx.ids.bn_stats, &[&self.conv_out, &self.mean, &self.var], &self.nchw(), c));
        if self.update_running.get() {
            pre.push(ctx.step(
                ctx.ids.bn_running,
                &[&self.mean, &self.var, ps.w(&self.names.run_mean), ps.w(&self.names.run_var)],
                &[c, f(self.momentum)],
                c,
            ));
        }
        ctx.gpu.submit(&[], &pre);
        self.pack_stats_host(ctx, ps);
        let s_train = ctx.step(ctx.ids.bn_train, &[&self.conv_out, &self.mv, &self.gb, &self.bn_out], &self.nchw(), on);
        let s_act = match act_pair(ctx, self.spec.act) {
            Some((fwd, _)) => ctx.step(fwd, &[&self.bn_out, &self.act], &act_params(self.spec.act, on), on),
            // Act::None: the block's output IS the BN output. Copying via a
            // slope-1 leaky_relu would be a wasted dispatch, so alias instead.
            None => ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.bn_out, &self.act], &[on, f(1.0)], on),
        };
        ctx.gpu.submit(&[], &[s_train, s_act]);
    }

    /// This conv's calibration / fake-quant tap — see [`apply_tap`]. Every conv
    /// variant routes through here so the calibrator sees a complete, consistent
    /// set of tensors.
    fn tap_input(&self, ctx: &Ctx, x_in: &DeviceBuffer) -> Option<DeviceBuffer> {
        apply_tap(ctx, &self.prefix, self.in_shape.numel() as usize, &self.q_in, x_in)
    }

    /// Eligible for the conv-as-GEMM eval path (see the call site).
    fn gemm_eval_eligible(&self, ctx: &Ctx) -> bool {
        std::env::var("BRAIN_CONV_GEMM").map(|v| v != "0").unwrap_or(true)
            && ctx.tap.is_none()
            && self.spec.is_dense()
            && self.out_shape.n == 1
            && self.out_shape.c >= gemm_conv_min_cout()
            && matches!(self.spec.act, Act::Silu)
            && ctx.ids.im2col != crate::NONE
            && ctx.ids.matmul_reg2 != crate::NONE
            && ctx.ids.conv_epilogue != crate::NONE
    }

    /// im2col + matmul_reg2 (raw conv into `act`) + conv_epilogue (per-channel
    /// affine from `sb` + SiLU, in place). Reproduces the fused conv_act exactly.
    fn forward_eval_gemm(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let (cin, h, w) = (self.in_shape.c, self.in_shape.h, self.in_shape.w);
        let (k, stride, pad) = (self.spec.k, self.spec.stride, self.spec.pad);
        let (cout, ho, wo) = (self.out_shape.c, self.out_shape.h, self.out_shape.w);
        let cinkk = cin * k * k;
        let hw = ho * wo;
        if self.col.borrow().is_none() {
            *self.col.borrow_mut() = Some(ctx.gpu.storage((hw as u64) * (cinkk as u64)));
        }
        let col_ref = self.col.borrow();
        let col = col_ref.as_ref().unwrap();
        let s_col = ctx.step(
            ctx.ids.im2col,
            &[x_in, col],
            &[cin, h, w, k, stride, pad, ho, wo, cinkk],
            hw * cinkk,
        );
        // y[Cout, HW] = W[Cout, CinKK] . col[HW, CinKK]^T into conv_out (raw conv).
        let reg_threads = cout.div_ceil(128) * hw.div_ceil(128) * 256;
        let s_gemm = ctx.step(
            ctx.ids.matmul_reg2,
            &[ps.w(&self.names.weight), col, &self.conv_out],
            &[cout, cinkk, hw],
            reg_threads,
        );
        // per-channel affine (BN-eval collapsed in `sb`) + SiLU (act=2): conv_out -> act.
        let s_epi = ctx.step(ctx.ids.conv_epilogue, &[&self.sb, &self.conv_out, &self.act], &[cout, hw, 2], cout * hw);
        ctx.gpu.submit(&[], &[s_col, s_gemm, s_epi]);
    }

    /// Unfused eval: `conv -> bn_eval(+act)`.
    ///
    /// Taken by every unit the fused path cannot serve — any grouped/dilated one
    /// (i.e. all of ZipDepth), any biased one, and any whose activation has no
    /// selector code (GELU). The fused `conv_act*` kernels' uniform carries no
    /// groups/dilation and the dense conv they route to on CPU ignores `groups`,
    /// so fusing those would be wrong rather than merely slower.
    ///
    /// This is what gives `bn_eval` a consumer: it has been registered and tested
    /// since P2 but nothing dispatched it, because yolo always fused. Its eps
    /// (`1e-5`) matches `pack_sb`'s, so the fused and unfused paths agree
    /// numerically by construction rather than by coincidence.
    ///
    /// NOTE `bn_eval` takes the SAME four buffers as `bn_train` — `x, mv, gb,
    /// out` — with the RUNNING mean|var in `mv`, not the collapsed `scale|bias`
    /// in `sb`. `sb` exists only for the fused `conv_act*` kernels. Binding `sb`
    /// here instead reads binding 3 out of bounds and, since the CPU JIT compiles
    /// with `MemFlags::trusted()` (no bounds checks), SEGFAULTS rather than
    /// erroring. The per-channel packing is still cached across frames via
    /// `sb_ready`, which here gates `mv`+`gb` instead.
    fn forward_eval_unfused(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        if self.spec.norm == Norm::None {
            // Handled by the shared raw-conv path in `forward`.
            unreachable!("Norm::None short-circuits before reaching the eval path");
        }
        if !self.sb_ready.get() {
            self.pack_running_mv(ctx, ps);
            self.pack_gb(ctx, ps);
            self.sb_ready.set(true);
        }
        let on = self.out_shape.numel();
        let tapped = self.tap_input(ctx, x_in);
        let src = tapped.as_ref().unwrap_or(x_in);

        // conv -> bn_eval(+act): where `bn_eval`'s selector covers the
        // activation it rides along, straight into `self.act` — one dispatch
        // (and one full memory pass) fewer than conv -> bn -> act. Eval-only, so
        // nothing reads the pre-activation `bn_out` cache that path skips.
        //
        // GELU has NO selector code (see `Act::fused_code`), so it must run
        // `bn_eval` at act=0 into `bn_out` and then its own kernel. Pushing a
        // fabricated code would silently drop the activation.
        let mut steps = self.conv_steps(ctx, ps, src, &self.conv_out);
        let mut bn_params = self.nchw().to_vec();
        match self.spec.act.fused_code() {
            Some(code) => {
                bn_params.push(code);
                steps.push(ctx.step(
                    ctx.ids.need(ctx.ids.bn_eval, "bn_eval"),
                    &[&self.conv_out, &self.mv, &self.gb, &self.act],
                    &bn_params,
                    on,
                ));
            }
            None => {
                bn_params.push(0);
                steps.push(ctx.step(
                    ctx.ids.need(ctx.ids.bn_eval, "bn_eval"),
                    &[&self.conv_out, &self.mv, &self.gb, &self.bn_out],
                    &bn_params,
                    on,
                ));
                let (fwd, _) = act_pair(ctx, self.spec.act).expect("fused_code()==None implies a real activation");
                steps.push(ctx.step(fwd, &[&self.bn_out, &self.act], &act_params(self.spec.act, on), on));
            }
        }
        ctx.gpu.submit(&[], &steps);
    }

    /// Interleave the RUNNING mean/var into `mv` for `bn_eval` (which shares
    /// `bn_train`'s signature and simply expects running stats there).
    fn pack_running_mv(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let rmean = ctx.gpu.read(ps.w(&self.names.run_mean), c);
        let rvar = ctx.gpu.read(ps.w(&self.names.run_var), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&rmean, &rvar)));
    }

    /// Collapse the BN-eval transform into per-channel `scale|bias` packed in
    /// `sb` (`sb[2c] = gamma/sqrt(run_var+eps)`, `sb[2c+1] = beta - run_mean*scale`)
    /// — the constant the FUSED `conv_act*` kernel consumes. Eps matches
    /// `bn_eval`'s, so the fused and unfused paths agree by construction.
    ///
    /// Only the fused path uses this. `bn_eval` wants `mv`+`gb`, not `sb` — see
    /// [`Conv::pack_running_mv`].
    fn pack_sb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        let beta = ctx.gpu.read(ps.w(&self.names.beta), c);
        let rmean = ctx.gpu.read(ps.w(&self.names.run_mean), c);
        let rvar = ctx.gpu.read(ps.w(&self.names.run_var), c);
        let mut sb = Vec::with_capacity(2 * c);
        for i in 0..c {
            // `crate::fold::BN_EPS`, not a `1e-5` literal: `fold.rs` is the one
            // place this constant is stated, and its own doc names `pack_sb` as
            // a consumer. A second literal is how the folded ONNX export and the
            // fused eval path drift apart by an epsilon nothing compares.
            let scale = gamma[i] / (rvar[i] + crate::fold::BN_EPS).sqrt();
            sb.push(scale);
            sb.push(beta[i] - rmean[i] * scale);
        }
        ctx.gpu.write(&self.sb, bytemuck::cast_slice(&sb));
    }

    /// Interleave the freshly-computed batch mean/var into `mv` and mean|var|gamma
    /// into `mvg` (the BN-backward input). Called between `bn_stats` and
    /// `bn_train` during forward.
    fn pack_stats_host(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let mean = ctx.gpu.read(&self.mean, c);
        let var = ctx.gpu.read(&self.var, c);
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&mean, &var)));
        let mut mvg = Vec::with_capacity(3 * c);
        for i in 0..c {
            mvg.push(mean[i]);
            mvg.push(var[i]);
            mvg.push(gamma[i]);
        }
        ctx.gpu.write(&self.mvg, bytemuck::cast_slice(&mvg));
    }

    /// Accumulate this unit's bias gradient from `d_conv` (the grad wrt the
    /// conv+bias output) via [`accumulate_bias_grad`]. No-op for an unbiased unit.
    ///
    /// `d_conv` is passed in rather than read from `self.d_conv` because the
    /// Act::None raw path binds the caller's `d_out` DIRECTLY (no act-backward
    /// copy dispatch) — reading `self.d_conv` there would reduce a buffer that
    /// was never written.
    fn bias_backward(&self, ctx: &Ctx, ps: &ParamStore, d_conv: &DeviceBuffer) {
        let Some(dbcast) = self.dbcast.as_ref() else { return };
        accumulate_bias_grad(ctx, ps, &self.names.bias, dbcast, self.out_shape, d_conv);
    }

    /// Backward. `d_out` = grad wrt this block's output; `d_in` receives the grad
    /// wrt `x_in` (overwritten). Param grads accumulate into the ParamStore.
    /// Assumes `forward` already ran (caches + `mv`/`mvg`/`bp` populated).
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let on = self.out_shape.numel();
        let c = self.out_shape.c;
        let dw_n = self.out_shape.c * self.in_shape.c * self.k * self.k;

        if self.spec.norm == Norm::None {
            // Raw conv: act backward straight into d_conv, then the conv
            // adjoints. Act::None needs no act backward at all — `out()`
            // aliases `conv_out`, so `d_out` IS d(conv_out); bind it directly.
            let d_conv: &DeviceBuffer = if let Some((_, bwd)) = act_pair(ctx, self.spec.act) {
                let s_a = ctx.step(bwd, &[&self.conv_out, d_out, &self.d_conv], &act_params(self.spec.act, on), on);
                ctx.gpu.submit(&[], &[s_a]);
                &self.d_conv
            } else {
                d_out
            };
            let dw_n = self.out_shape.c * (self.in_shape.c / self.spec.groups) * self.k * self.k;
            let s_dw = ctx.step(self.conv_dw_kind(ctx), &[d_conv, x_in, ps.g(&self.names.weight)], &self.conv_params(), dw_n);
            let s_dxin = ctx.step(self.conv_dx_kind(ctx), &[d_conv, ps.w(&self.names.weight), d_in], &self.conv_params(), self.in_shape.numel());
            ctx.gpu.submit(&[], &[s_dw, s_dxin]);
            self.bias_backward(ctx, ps, d_conv);
            return;
        }
        let s_act = match act_pair(ctx, self.spec.act) {
            Some((_, bwd)) => ctx.step(bwd, &[&self.bn_out, d_out, &self.d_bn], &act_params(self.spec.act, on), on),
            None => ctx.step(ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"), &[&self.bn_out, d_out, &self.d_bn], &[on, f(1.0)], on),
        };
        let s_dstats = ctx.step(ctx.ids.bn_dstats, &[&self.conv_out, &self.d_bn, &self.mvg, &self.bp], &self.nchw(), c);
        // bn_dgamma / bn_dbeta accumulate -> their grad buffers are pre-zeroed by
        // the model's zero_grads (clears list), exactly like gpt.
        let s_dgamma = ctx.step(ctx.ids.bn_dgamma, &[&self.conv_out, &self.d_bn, &self.mv, ps.g(&self.names.gamma)], &self.nchw(), c);
        let s_dbeta = ctx.step(ctx.ids.bn_dbeta, &[&self.d_bn, ps.g(&self.names.beta)], &self.nchw(), c);
        let s_dx = ctx.step(ctx.ids.bn_dx, &[&self.conv_out, &self.d_bn, &self.bp, &self.d_conv], &self.nchw(), on);
        let s_dw = ctx.step(self.conv_dw_kind(ctx), &[&self.d_conv, x_in, ps.g(&self.names.weight)], &self.conv_params(), dw_n);
        let s_dxin = ctx.step(self.conv_dx_kind(ctx), &[&self.d_conv, ps.w(&self.names.weight), d_in], &self.conv_params(), self.in_shape.numel());
        // s_silu must precede s_dstats/s_dgamma/s_dbeta (they read d_bn); s_dstats
        // must precede s_dx (reads bp). Submit in this order.
        ctx.gpu.submit(&[], &[s_act, s_dstats, s_dgamma, s_dbeta, s_dx, s_dw, s_dxin]);
        self.bias_backward(ctx, ps, &self.d_conv);
    }
}

// ===========================================================================
// Bottleneck = Conv(K3,s1) -> Conv(K3,s1) [+ residual]
// ===========================================================================

/// CSP bottleneck. Two K3/s1 convs; optional residual `add2` shortcut when
/// `c_in == c_out && shortcut`.
pub struct Bottleneck {
    pub cv1: Conv,
    pub cv2: Conv,
    pub shortcut: bool,
    pub in_shape: Shape,
    pub out_shape: Shape,
    sum: DeviceBuffer,   // residual sum (output when shortcut) [out]
    d_mid: DeviceBuffer, // grad wrt cv1.out [out]
}

impl Bottleneck {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, shortcut: bool, train: bool) -> Bottleneck {
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, c_out, 3, 1, 1, train);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cv1.out_shape, c_out, 3, 1, 1, train);
        let out_shape = cv2.out_shape;
        let use_short = shortcut && in_shape.c == c_out;
        let on = out_shape.numel();
        Bottleneck {
            cv1,
            cv2,
            shortcut: use_short,
            in_shape,
            out_shape,
            sum: ctx.act(on),
            d_mid: ctx.act(on),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        if self.shortcut {
            &self.sum
        } else {
            self.cv2.out()
        }
    }

    /// Propagate the eval/train BN toggle to both convs.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to both convs.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        v.extend(self.cv2.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cv1.forward(ctx, ps, x_in);
        self.cv2.forward(ctx, ps, self.cv1.out());
        if self.shortcut {
            let on = self.out_shape.numel();
            let s = ctx.step(ctx.ids.add2, &[x_in, self.cv2.out(), &self.sum], &[on], on);
            ctx.gpu.submit(&[], &[s]);
        }
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        // output = (shortcut ? x_in + cv2 : cv2). In both cases d(cv2_out)=d_out.
        self.cv2.backward(ctx, ps, self.cv1.out(), d_out, &self.d_mid);
        self.cv1.backward(ctx, ps, x_in, &self.d_mid, d_in);
        if self.shortcut {
            let on = self.in_shape.numel();
            let s = ctx.step(ctx.ids.add_inplace, &[d_in, d_out], &[on], on);
            ctx.gpu.submit(&[], &[s]);
        }
    }
}

// ===========================================================================
// C2f = Conv1x1(2c) -> split [y0|y1] -> y1 thru n Bottlenecks -> concat -> Conv1x1
// ===========================================================================

/// CSP "C2f" block (`c = C_out/2`): a 1x1 conv expands to `2c` channels, split
/// into halves `y0`/`y1`; `y1` runs through `n` bottlenecks (each output
/// retained); everything `[y0, y1, b1..bn]` is concatenated along C and a final
/// 1x1 conv projects to `C_out`.
pub struct C2f {
    pub cv1: Conv, // in: Cin -> 2c
    pub cv2: Conv, // out: (2+n)*c -> Cout
    pub m: Vec<Bottleneck>,
    pub in_shape: Shape,
    pub out_shape: Shape,
    c: u32, // half width
    n: u32, // bottleneck count
    sh: Shape, // spatial shape of each c-channel chunk

    // forward caches
    y0: DeviceBuffer,       // first half [n,c,h,w]
    y1: DeviceBuffer,       // second half [n,c,h,w]
    concat: DeviceBuffer,   // [n,(2+n)*c,h,w]
    // intermediate concat buffers (left-fold of concat2)
    cat_tmp: Vec<DeviceBuffer>,

    // backward caches
    d_split: DeviceBuffer,  // grad wrt cv1.out ([n,2c,h,w]); chunk grads scattered in
    d_chunk: Vec<DeviceBuffer>, // grad wrt each chunk [y0,y1,b1..bn]
    d_y1: DeviceBuffer,     // accumulated grad wrt y1 (from chain + concat slice)
}

impl C2f {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, n: u32, shortcut: bool, train: bool) -> C2f {
        assert!(c_out % 2 == 0, "C2f C_out must be even");
        let c = c_out / 2;
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, 2 * c, 1, 1, 0, train);
        let sh = Shape::new(cv1.out_shape.n, c, cv1.out_shape.h, cv1.out_shape.w);
        let mut m = Vec::new();
        let mut prev = sh;
        for i in 0..n {
            let b = Bottleneck::new(ctx, &format!("{prefix}.m.{i}"), prev, c, shortcut, train);
            prev = b.out_shape;
            m.push(b);
        }
        let cat_c = (2 + n) * c;
        let cat_shape = Shape::new(sh.n, cat_c, sh.h, sh.w);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cat_shape, c_out, 1, 1, 0, train);
        let out_shape = cv2.out_shape;
        let chunk_n = sh.numel();

        // left-fold concat needs (chunks-1) intermediate buffers; the last is the
        // full concat. chunks = 2 + n.
        let chunks = 2 + n;
        let mut cat_tmp = Vec::new();
        for k in 2..=chunks {
            cat_tmp.push(ctx.act(k * c * sh.h * sh.w * sh.n));
        }
        let mut d_chunk = Vec::new();
        for _ in 0..chunks {
            d_chunk.push(ctx.act(chunk_n));
        }
        C2f {
            cv1,
            cv2,
            m,
            in_shape,
            out_shape,
            c,
            n,
            sh,
            y0: ctx.act(chunk_n),
            y1: ctx.act(chunk_n),
            concat: ctx.act(cat_c * sh.h * sh.w * sh.n),
            cat_tmp,
            d_split: ctx.act(2 * chunk_n),
            d_chunk,
            d_y1: ctx.act(chunk_n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.cv2.out()
    }

    /// Propagate the eval/train BN toggle to all convs + bottlenecks.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        for b in &self.m {
            b.set_eval(eval);
        }
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to all convs + bottlenecks.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        for b in &self.m {
            b.set_update_running(on);
        }
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        for b in &self.m {
            v.extend(b.param_list());
        }
        v.extend(self.cv2.param_list());
        v
    }

    fn split_params(&self, c_off: u32) -> [u32; 6] {
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        [self.sh.n, 2 * self.c, self.c, c_off, self.sh.h, self.sh.w]
    }

    /// All retained chunk buffers in concat order `[y0, y1, b1..bn]`.
    fn chunks(&self) -> Vec<&DeviceBuffer> {
        let mut v = vec![&self.y0, &self.y1];
        for b in &self.m {
            v.push(b.out());
        }
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        // 1x1 conv -> 2c channels.
        self.cv1.forward(ctx, ps, x_in);
        // split into y0 (c_off=0) and y1 (c_off=c) via concat_split copies.
        let chunk_n = self.sh.numel();
        let s0 = ctx.step(ctx.ids.concat_split, &[self.cv1.out(), &self.y0], &self.split_params(0), chunk_n);
        let s1 = ctx.step(ctx.ids.concat_split, &[self.cv1.out(), &self.y1], &self.split_params(self.c), chunk_n);
        ctx.gpu.submit(&[], &[s0, s1]);

        // run bottlenecks on y1, chaining each output to the next input.
        let mut prev = &self.y1;
        for b in &self.m {
            b.forward(ctx, ps, prev);
            prev = b.out();
        }

        // concat [y0, y1, b1..bn] along C in a SINGLE pass: each chunk (c
        // channels) is placed once into its slice of `concat`, instead of a
        // left-fold of concat2 that re-copies the growing prefix (O(n^2) data
        // movement). The placements write disjoint channel ranges, so they share
        // one submit.
        let chunks = self.chunks();
        let cat_c = chunks.len() as u32 * self.c;
        let mut steps = Vec::with_capacity(chunks.len());
        let chunk_n = self.sh.numel();
        for (i, chunk) in chunks.iter().enumerate() {
            let c_off = i as u32 * self.c;
            // chan_place ABI: [N, Ctot, Csrc, c_off, H, W], bufs [src, dst]
            let params = [self.sh.n, cat_c, self.c, c_off, self.sh.h, self.sh.w];
            steps.push(ctx.step(ctx.ids.chan_place, &[chunk, &self.concat], &params, chunk_n));
        }
        ctx.gpu.submit(&[], &steps);

        // final 1x1 conv -> C_out.
        self.cv2.forward(ctx, ps, &self.concat);
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        // cv2 backward: grad wrt the concat buffer -> reuse `concat`-shaped temp.
        // We need a buffer of concat size for the grad; allocate via d_split? No,
        // d_split is 2c. Use the largest cat_tmp (which is the full concat size)
        // as the cv2 input-grad target.
        let d_concat = self.cat_tmp.last().unwrap();
        self.cv2.backward(ctx, ps, &self.concat, d_out, d_concat);

        // concat_split: distribute d_concat to each chunk grad [y0,y1,b1..bn].
        let chunks_n = (2 + self.n) as usize;
        let chunk_n = self.sh.numel();
        let cat_c = (2 + self.n) * self.c;
        let mut steps = Vec::new();
        for i in 0..chunks_n {
            let c_off = i as u32 * self.c;
            // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
            let params = [self.sh.n, cat_c, self.c, c_off, self.sh.h, self.sh.w];
            steps.push(ctx.step(ctx.ids.concat_split, &[d_concat, &self.d_chunk[i]], &params, chunk_n));
        }
        ctx.gpu.submit(&[], &steps);

        // bottleneck chain backward (reverse). The grad into bottleneck i is its
        // own chunk grad PLUS the input-grad produced by bottleneck i+1.
        // Process from last to first; carry `d_carry` = grad wrt that bottleneck's
        // input (= grad wrt the previous chunk's output).
        // d_chunk indices: 0=y0, 1=y1, 2+j = bottleneck j output.
        // Use a scratch (d_split's first half region is wrong shape) -> reuse
        // per-bottleneck d_mid via a dedicated accumulation buffer `d_y1`.
        // We accumulate into d_chunk[k] for the input of bottleneck k.
        let nb = self.m.len();
        // grad flowing back into each bottleneck input; start with chunk grads.
        // For bottleneck j (0-based), its output chunk index is 2+j, its input is
        // chunk (2+j-1) for j>0, or y1 (index 1) for j==0.
        for j in (0..nb).rev() {
            let out_idx = 2 + j;
            let in_idx = if j == 0 { 1 } else { 2 + j - 1 };
            let b = &self.m[j];
            let x_in_b: &DeviceBuffer = if j == 0 { &self.y1 } else { self.m[j - 1].out() };
            // backward produces grad wrt input into d_y1 (scratch), then we add
            // it onto the running chunk grad of the input chunk.
            b.backward(ctx, ps, x_in_b, &self.d_chunk[out_idx], &self.d_y1);
            let s = ctx.step(ctx.ids.add_inplace, &[&self.d_chunk[in_idx], &self.d_y1], &[chunk_n], chunk_n);
            ctx.gpu.submit(&[], &[s]);
        }

        // Now d_chunk[0] = grad wrt y0, d_chunk[1] = grad wrt y1 (fully merged).
        // Re-merge into d_split [2c] = grad wrt cv1.out: copy y0-grad to channels
        // [0,c) and y1-grad to [c,2c). concat2 does exactly this.
        let params = [self.sh.n, self.c, self.c, self.sh.h, self.sh.w];
        let s = ctx.step(ctx.ids.concat2, &[&self.d_chunk[0], &self.d_chunk[1], &self.d_split], &params, 2 * chunk_n);
        ctx.gpu.submit(&[], &[s]);

        // cv1 backward -> d_in.
        self.cv1.backward(ctx, ps, x_in, &self.d_split, d_in);
    }
}

// ===========================================================================
// SPPF = Conv1x1 -> m1,m2,m3 = maxpool2d chain -> concat[x,m1,m2,m3] -> Conv1x1
// ===========================================================================

/// Spatial-Pyramid-Pooling-Fast. A 1x1 conv, three chained 5x5 maxpools, a
/// channel-concat of `[x, m1, m2, m3]` (4*c channels), and a final 1x1 conv.
///
/// The pools are [`MaxPool`] units, not an inline `maxpool2d` dispatch: there is
/// ONE max-pool in this crate and this block composes it.
pub struct SPPF {
    pub cv1: Conv,
    pub cv2: Conv,
    /// The three chained pools, `m[0] = pool(cv1.out)`, `m[i] = pool(m[i-1])`.
    pub m: Vec<MaxPool>,
    pub in_shape: Shape,
    pub out_shape: Shape,
    c: u32,
    sh: Shape, // [n,c,h,w] of the inner maps

    // forward caches
    cat1: DeviceBuffer, // [x,m1]            -> 2c
    cat2: DeviceBuffer, // [x,m1,m2]         -> 3c
    concat: DeviceBuffer, // [x,m1,m2,m3]    -> 4c

    // backward caches
    d_x: DeviceBuffer,  // grad wrt cv1.out (accumulated)
    d_x_cat: DeviceBuffer, // grad slice of x from concat
    d_m1: DeviceBuffer,
    d_m2: DeviceBuffer,
    d_m3: DeviceBuffer,
    d_m1_cat: DeviceBuffer,
    d_m2_cat: DeviceBuffer,
}

impl SPPF {
    /// yolo's ctor, unchanged: Ultralytics SPPF — cv1 halves channels to `c_out/2`,
    /// SiLU, brain names.
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, train: bool) -> SPPF {
        let spec = SppfSpec { hidden: c_out / 2, c_out, act: Act::Silu };
        SPPF::with_spec(ctx, prefix, in_shape, spec, NameStyle::Brain, train)
    }

    /// The general ctor. `cv1` narrows to `spec.hidden`, three K=5/pad=2 max-pools
    /// chain off it, and `cv2` maps the 4-way concat to `spec.c_out`.
    ///
    /// K is fixed at 5 (the kernel is `maxpool2d` at K=5/stride=1/pad=2):
    /// Ultralytics and ZipDepth both use 5, and ZipDepth's `k` argument is never
    /// passed a different value. Widen this when something needs it.
    pub fn with_spec(
        ctx: &Ctx,
        prefix: &str,
        in_shape: Shape,
        spec: SppfSpec,
        style: NameStyle,
        train: bool,
    ) -> SPPF {
        let (c, c_out) = (spec.hidden, spec.c_out);
        let cspec = |cout: u32| ConvSpec {
            cout,
            k: 1,
            stride: 1,
            pad: 0,
            groups: 1,
            dilation: 1,
            norm: Norm::Bn,
            act: spec.act,
            bias: false,
        };
        let p1 = format!("{prefix}.cv1");
        let cv1 = Conv::with_names(ctx, &p1, style.names(&p1), in_shape, cspec(c), train);
        let sh = cv1.out_shape;
        let cat_shape = Shape::new(sh.n, 4 * c, sh.h, sh.w);
        let p2 = format!("{prefix}.cv2");
        let cv2 = Conv::with_names(ctx, &p2, style.names(&p2), cat_shape, cspec(c_out), train);
        let out_shape = cv2.out_shape;
        let n1 = sh.numel();
        // K=5/stride=1/pad=2 is shape-preserving (Ho = (H + 2*2 - 5)/1 + 1 = H),
        // which is what the 4-way concat below depends on. `PoolSpec::out_shape`
        // states it once; the pools then all share `sh`.
        let m: Vec<MaxPool> = (0..3).map(|_| MaxPool::new(ctx, sh, PoolSpec::same5())).collect();
        assert_eq!(m[0].out_shape, sh, "SPPF's pools must be shape-preserving for the concat to line up");
        SPPF {
            cv1,
            cv2,
            m,
            in_shape,
            out_shape,
            c,
            sh,
            cat1: ctx.act(2 * n1),
            cat2: ctx.act(3 * n1),
            concat: ctx.act(4 * n1),
            d_x: ctx.act(n1),
            d_x_cat: ctx.act(n1),
            d_m1: ctx.act(n1),
            d_m2: ctx.act(n1),
            d_m3: ctx.act(n1),
            d_m1_cat: ctx.act(n1),
            d_m2_cat: ctx.act(n1),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.cv2.out()
    }

    /// Propagate the eval/train BN toggle to both convs.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to both convs.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        v.extend(self.cv2.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cv1.forward(ctx, ps, x_in);
        let x = self.cv1.out();
        let n1 = self.sh.numel();
        // m1 = pool(x); m2 = pool(m1); m3 = pool(m2). Sequential dependency.
        let mut prev = x;
        for p in &self.m {
            p.forward(ctx, prev);
            prev = p.out();
        }

        // concat [x, m1, m2, m3] via left-fold.
        let c = self.c;
        let (h, w, n) = (self.sh.h, self.sh.w, self.sh.n);
        let sc1 = ctx.step(ctx.ids.concat2, &[x, self.m[0].out(), &self.cat1], &[n, c, c, h, w], 2 * n1);
        ctx.gpu.submit(&[], &[sc1]);
        let sc2 = ctx.step(ctx.ids.concat2, &[&self.cat1, self.m[1].out(), &self.cat2], &[n, 2 * c, c, h, w], 3 * n1);
        ctx.gpu.submit(&[], &[sc2]);
        let sc3 = ctx.step(ctx.ids.concat2, &[&self.cat2, self.m[2].out(), &self.concat], &[n, 3 * c, c, h, w], 4 * n1);
        ctx.gpu.submit(&[], &[sc3]);

        self.cv2.forward(ctx, ps, &self.concat);
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let n1 = self.sh.numel();
        let c = self.c;
        let (h, w, n) = (self.sh.h, self.sh.w, self.sh.n);
        let cat_c = 4 * c;

        // cv2 backward -> grad wrt concat (reuse `concat` buffer? no, need fresh).
        // Use cat2 (3c) is too small; allocate d_concat via the `concat` buffer is
        // also a live cache. Use a dedicated approach: split d_out's contribution
        // directly. We need a 4c-channel grad buffer; reuse `concat` is unsafe
        // because cv2.backward reads it (it's cv2's x_in). So we add a scratch.
        // To avoid another field, route cv2 grad into `concat`-sized cat via the
        // last cat buffer is 4c == concat size; allocate one more temp here.
        let d_concat = ctx.act(cat_c * h * w * n);
        self.cv2.backward(ctx, ps, &self.concat, d_out, &d_concat);

        // split d_concat into [x, m1, m2, m3] grad slices.
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        let sx = ctx.step(ctx.ids.concat_split, &[&d_concat, &self.d_x_cat], &[n, cat_c, c, 0, h, w], n1);
        let s1 = ctx.step(ctx.ids.concat_split, &[&d_concat, &self.d_m1_cat], &[n, cat_c, c, c, h, w], n1);
        let s2 = ctx.step(ctx.ids.concat_split, &[&d_concat, &self.d_m2_cat], &[n, cat_c, c, 2 * c, h, w], n1);
        let s3 = ctx.step(ctx.ids.concat_split, &[&d_concat, &self.d_m3], &[n, cat_c, c, 3 * c, h, w], n1);
        ctx.gpu.submit(&[], &[sx, s1, s2, s3]);

        // Backprop the maxpool chain. m3 = pool(m2): grad wrt m2 from m3.
        // d_m2 = d_m2_cat + maxpool_dx(d_m3)
        self.m[2].backward(ctx, &self.d_m3);
        let a3 = ctx.step(ctx.ids.add2, &[&self.d_m2_cat, self.m[2].d_in(), &self.d_m2], &[n1], n1);
        ctx.gpu.submit(&[], &[a3]);

        // m2 = pool(m1): grad wrt m1 = d_m1_cat + maxpool_dx(d_m2)
        self.m[1].backward(ctx, &self.d_m2);
        let a2 = ctx.step(ctx.ids.add2, &[&self.d_m1_cat, self.m[1].d_in(), &self.d_m1], &[n1], n1);
        ctx.gpu.submit(&[], &[a2]);

        // m1 = pool(x): grad wrt x = d_x_cat + maxpool_dx(d_m1)
        self.m[0].backward(ctx, &self.d_m1);
        let a1 = ctx.step(ctx.ids.add2, &[&self.d_x_cat, self.m[0].d_in(), &self.d_x], &[n1], n1);
        ctx.gpu.submit(&[], &[a1]);

        // cv1 backward -> d_in.
        self.cv1.backward(ctx, ps, x_in, &self.d_x, d_in);
    }
}

// ===========================================================================
// ConvTranspose = convtr2d (+ optional bias) -> optional activation
// ===========================================================================

/// Everything that varies between transposed-conv units.
///
/// Deliberately NOT a flag on [`ConvSpec`]. A transposed conv is a different
/// operator with a different WEIGHT LAYOUT (`[Cin, Cout/G, K, K]`, the transpose
/// of `conv2d_gd`'s `[Cout, Cin/G, K, K]`), a different output-shape formula,
/// and one hyperparameter a forward conv does not have (`out_pad`). Both layouts
/// hold exactly `Cin*Cout/G` elements, so a mis-specified unit ALWAYS binds and
/// silently computes the wrong operator — a separate type is what makes that
/// unrepresentable.
///
/// No `norm` field: nothing that needs a transposed conv in brain's imaging
/// models pairs it with BatchNorm (SAM 2's mask decoder follows it with
/// [`LayerNorm2d`], VQGAN's decoder with GroupNorm), and [`crate::BatchNorm`]
/// already exists as a standalone unit for anything that does. Welding a second
/// BN implementation in here would be the duplicate, not the convenience.
#[derive(Clone, Copy, Debug)]
pub struct ConvTrSpec {
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    /// torch's `output_padding`. It only WIDENS `Ho`/`Wo`; it is not zero-fill.
    ///
    /// At `stride > 1` the far-side `pad` crop hides output positions that
    /// genuinely receive input, and `output_padding` is exactly what un-crops
    /// them — which is why torch documents it as resolving a strided conv's
    /// output-shape ambiguity rather than as padding. Verified against PyTorch:
    /// the extra bottom/right band is NOT zeros.
    pub out_pad: u32,
    pub dilation: u32,
    pub groups: u32,
    pub act: Act,
    /// `nn.ConvTranspose2d(.., bias=True)` — torch's default, so this defaults
    /// to `true`. There is no fused `convtr2d_bias` kernel, so the bias is a
    /// second `add_chan_inplace` pass over the output.
    pub bias: bool,
}

impl ConvTrSpec {
    /// A plain `nn.ConvTranspose2d(cin, cout, k, stride, pad)`: dense, biased,
    /// no activation — torch's defaults.
    pub const fn new(cout: u32, k: u32, stride: u32, pad: u32) -> ConvTrSpec {
        ConvTrSpec { cout, k, stride, pad, out_pad: 0, dilation: 1, groups: 1, act: Act::None, bias: true }
    }
    pub const fn with_act(self, act: Act) -> ConvTrSpec {
        ConvTrSpec { act, ..self }
    }
    pub const fn with_groups(self, groups: u32) -> ConvTrSpec {
        ConvTrSpec { groups, ..self }
    }
    pub const fn with_dilation(self, dilation: u32) -> ConvTrSpec {
        ConvTrSpec { dilation, ..self }
    }
    pub const fn with_out_pad(self, out_pad: u32) -> ConvTrSpec {
        ConvTrSpec { out_pad, ..self }
    }
    /// Drop the learned bias (`bias=False`).
    pub const fn bias_free(self) -> ConvTrSpec {
        ConvTrSpec { bias: false, ..self }
    }
    /// `Ho = (H-1)*stride - 2*pad + dilation*(K-1) + out_pad + 1`, likewise `Wo`.
    ///
    /// The kernel takes `Ho`/`Wo` as Params words and never recomputes them, so
    /// this formula is the single place the shape is decided — a caller that
    /// passes a `Ho` from the FORWARD conv formula gets a kernel that runs and
    /// gathers out of the map it meant to.
    pub fn out_shape(&self, x: Shape) -> Shape {
        let ext = self.dilation * (self.k - 1);
        let (nh, nw) = (
            (x.h - 1) * self.stride + ext + self.out_pad + 1,
            (x.w - 1) * self.stride + ext + self.out_pad + 1,
        );
        // These are u32: an over-large `pad` would WRAP to a ~4-billion extent and
        // the block would try to allocate it, rather than reporting the bad spec.
        assert!(
            nh > 2 * self.pad && nw > 2 * self.pad,
            "ConvTranspose pad {} crops the entire output of a {}x{} input (k={}, stride={}, dilation={})",
            self.pad,
            x.h,
            x.w,
            self.k,
            self.stride,
            self.dilation
        );
        Shape::new(x.n, self.cout, nh - 2 * self.pad, nw - 2 * self.pad)
    }
}

/// A transposed-convolution unit: `convtr2d` (+ optional per-channel bias) and
/// an optional activation. NCHW, SSA.
///
/// Decoder upsampling — SAM 2's mask decoder (2x twice), the VQGAN/CodeFormer
/// decoder. Naming is [`ConvNames`], the same data-driven scheme [`Conv`] uses,
/// because the two checkpoints spell the module differently
/// (`output_upscaling.0.weight` vs `…upsample.conv.weight`).
pub struct ConvTranspose {
    /// The ActTap key — MUST equal the exported ONNX node name (see [`Conv`]).
    prefix: String,
    names: ConvNames,
    pub in_shape: Shape,
    pub out_shape: Shape,
    pub spec: ConvTrSpec,

    pre: DeviceBuffer,  // convtr2d output (+ bias), pre-activation [out]
    act: DeviceBuffer,  // activation output [out]; unused for Act::None
    d_pre: DeviceBuffer,// grad wrt `pre` [out]
    /// `bias_grad`'s output before the host spatial reduce, `[C*HW]`. Allocated
    /// only for a biased unit.
    dbcast: Option<DeviceBuffer>,
    q_in: std::cell::RefCell<Option<DeviceBuffer>>,
}

impl ConvTranspose {
    /// brain-style names (`P.conv.weight`), for a unit brain owns end to end.
    pub fn with_spec(ctx: &Ctx, prefix: &str, in_shape: Shape, spec: ConvTrSpec) -> ConvTranspose {
        ConvTranspose::with_names(ctx, prefix, ConvNames::brain(prefix), in_shape, spec)
    }

    /// A bare torch `nn.ConvTranspose2d` at `prefix`: `P.weight` + `P.bias`.
    pub fn torch(ctx: &Ctx, prefix: &str, in_shape: Shape, spec: ConvTrSpec) -> ConvTranspose {
        ConvTranspose::with_names(ctx, prefix, ConvNames::torch_flat(prefix), in_shape, spec)
    }

    /// The fully general ctor: the caller supplies the tensor names, so each
    /// model's param list mirrors its own checkpoint.
    pub fn with_names(
        ctx: &Ctx,
        prefix: &str,
        names: ConvNames,
        in_shape: Shape,
        spec: ConvTrSpec,
    ) -> ConvTranspose {
        assert_eq!(in_shape.c % spec.groups, 0, "cin {} not divisible by groups {}", in_shape.c, spec.groups);
        assert_eq!(spec.cout % spec.groups, 0, "cout {} not divisible by groups {}", spec.cout, spec.groups);
        let out_shape = spec.out_shape(in_shape);
        let on = out_shape.numel();
        ConvTranspose {
            prefix: prefix.to_string(),
            names,
            in_shape,
            out_shape,
            spec,
            pre: ctx.act(on),
            // Act::None never writes `act` (`out()` aliases `pre`), but a
            // zero-sized buffer is a hard error on wgpu — allocate one element.
            act: ctx.act(if spec.act == Act::None { 1 } else { on }),
            d_pre: ctx.act(on),
            dbcast: if spec.bias { Some(ctx.act(spec.cout * out_shape.h * out_shape.w)) } else { None },
            q_in: std::cell::RefCell::new(None),
        }
    }

    /// This unit's tensor names — read-only, for importers and fusions.
    pub fn names(&self) -> &ConvNames {
        &self.names
    }

    pub fn out(&self) -> &DeviceBuffer {
        // Act::None: the block's output IS the (biased) convtr output. Copying it
        // through a slope-1 leaky_relu would be a wasted full memory pass.
        if self.spec.act == Act::None {
            &self.pre
        } else {
            &self.act
        }
    }

    /// Weight `[Cin, Cout/G, K, K]` — torch's `ConvTranspose2d` layout, with the
    /// INPUT channel outermost. This is the transpose of [`Conv::param_list`]'s
    /// `[Cout, Cin/G, K, K]` and holds the same number of elements, so a weight
    /// stored in the wrong one never fails a size check.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let cin = self.in_shape.c as usize;
        let cout_g = (self.spec.cout / self.spec.groups) as usize;
        let k = self.spec.k as usize;
        let mut v = vec![(self.names.weight.clone(), cin * cout_g * k * k)];
        if self.spec.bias {
            v.push((self.names.bias.clone(), self.spec.cout as usize));
        }
        v
    }

    /// convtr2d's 12-u32 uniform: `[N, Cin, H, W, Cout, K, stride, pad, dilation,
    /// groups, Ho, Wo]` — byte-identical to `conv2d_gd`'s, and shared by
    /// `convtr2d`, `convtr2d_dw` and `convtr2d_dx`.
    fn params(&self) -> [u32; 12] {
        [
            self.in_shape.n,
            self.in_shape.c,
            self.in_shape.h,
            self.in_shape.w,
            self.spec.cout,
            self.spec.k,
            self.spec.stride,
            self.spec.pad,
            self.spec.dilation,
            self.spec.groups,
            self.out_shape.h,
            self.out_shape.w,
        ]
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let on = self.out_shape.numel();
        let tapped = apply_tap(ctx, &self.prefix, self.in_shape.numel() as usize, &self.q_in, x_in);
        let src = tapped.as_ref().unwrap_or(x_in);
        // One invocation per OUTPUT element: the forward is a gather over the
        // INVERTED forward map.
        let mut steps = vec![ctx.step(
            ctx.ids.need(ctx.ids.convtr2d, "convtr2d"),
            &[src, ps.w(&self.names.weight), &self.pre],
            &self.params(),
            on,
        )];
        if self.spec.bias {
            steps.push(ctx.step(
                ctx.ids.need(ctx.ids.add_chan_inplace, "add_chan_inplace"),
                &[&self.pre, ps.w(&self.names.bias)],
                &[on, self.spec.cout, self.out_shape.h * self.out_shape.w],
                on,
            ));
        }
        if let Some((fwd, _)) = act_pair(ctx, self.spec.act) {
            steps.push(ctx.step(fwd, &[&self.pre, &self.act], &act_params(self.spec.act, on), on));
        }
        ctx.gpu.submit(&[], &steps);
    }

    /// `d_out` = grad wrt this unit's output; `d_in` receives the grad wrt
    /// `x_in` (overwritten). Weight/bias grads ACCUMULATE into the ParamStore —
    /// `convtr2d_dw` composes with a prior `dw`, so the caller's `zero_grads`
    /// must have cleared it.
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let on = self.out_shape.numel();
        // Act::None: `out()` aliases `pre`, so `d_out` IS d(pre) — bind it
        // directly rather than paying a copy dispatch.
        let d_pre: &DeviceBuffer = if let Some((_, bwd)) = act_pair(ctx, self.spec.act) {
            let s = ctx.step(bwd, &[&self.pre, d_out, &self.d_pre], &act_params(self.spec.act, on), on);
            ctx.gpu.submit(&[], &[s]);
            &self.d_pre
        } else {
            d_out
        };
        let dw_n = self.in_shape.c * (self.spec.cout / self.spec.groups) * self.spec.k * self.spec.k;
        // _dw: one invocation per WEIGHT element (it owns the sum over n,hi,wi
        // instead of needing an atomic). _dx: one per INPUT element.
        let s_dw = ctx.step(
            ctx.ids.need(ctx.ids.convtr2d_dw, "convtr2d_dw"),
            &[d_pre, x_in, ps.g(&self.names.weight)],
            &self.params(),
            dw_n,
        );
        let s_dx = ctx.step(
            ctx.ids.need(ctx.ids.convtr2d_dx, "convtr2d_dx"),
            &[d_pre, ps.w(&self.names.weight), d_in],
            &self.params(),
            self.in_shape.numel(),
        );
        ctx.gpu.submit(&[], &[s_dw, s_dx]);
        if let Some(dbcast) = self.dbcast.as_ref() {
            accumulate_bias_grad(ctx, ps, &self.names.bias, dbcast, self.out_shape, d_pre);
        }
    }
}

// ===========================================================================
// MaxPool = maxpool2d (+ maxpool2d_dx)
// ===========================================================================

/// A max-pool's geometry. Square window, square stride, symmetric zero pad —
/// the shape `maxpool2d` covers.
#[derive(Clone, Copy, Debug)]
pub struct PoolSpec {
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
}

impl PoolSpec {
    pub const fn new(k: u32, stride: u32, pad: u32) -> PoolSpec {
        PoolSpec { k, stride, pad }
    }
    /// SPPF's / Hiera's shape-preserving pool: K=5, stride 1, pad 2.
    pub const fn same5() -> PoolSpec {
        PoolSpec { k: 5, stride: 1, pad: 2 }
    }
    /// Hiera's `q_pool` / SCRFD's downsampling pool: K=stride=2, no pad.
    pub const fn half() -> PoolSpec {
        PoolSpec { k: 2, stride: 2, pad: 0 }
    }
    /// `Ho = (H + 2*pad - K)/stride + 1`. Identical to a conv's output formula
    /// at `dilation = 1` with the channel count carried through, so it reuses
    /// [`Shape::conv_out`] rather than restating the arithmetic.
    pub fn out_shape(&self, x: Shape) -> Shape {
        x.conv_out(x.c, self.k, self.stride, self.pad)
    }
}

/// A generic max-pool stage (forward + backward), SSA. Owns its output, its
/// `argmax` side-output and its input-gradient buffer, in the shape of
/// [`crate::Up`].
///
/// One max-pool, not two: `maxpool2d` replaced the stride-pinned `maxpool5`, and
/// [`SPPF`] composes THIS unit rather than dispatching the kernel a second time.
pub struct MaxPool {
    pub in_shape: Shape,
    pub out_shape: Shape,
    pub spec: PoolSpec,
    out: DeviceBuffer,
    /// The winning tap's INPUT flat index per output element, stored as f32 —
    /// the forward's side-output that makes the backward a gather. Exact while
    /// `N*C*H*W < 2^24`.
    argmax: DeviceBuffer,
    d_in: DeviceBuffer,
}

impl MaxPool {
    pub fn new(ctx: &Ctx, in_shape: Shape, spec: PoolSpec) -> MaxPool {
        let out_shape = spec.out_shape(in_shape);
        MaxPool {
            in_shape,
            out_shape,
            spec,
            out: ctx.act(out_shape.numel()),
            argmax: ctx.act(out_shape.numel()),
            d_in: ctx.act(in_shape.numel()),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    /// The grad wrt this stage's input, valid after [`MaxPool::backward`].
    pub fn d_in(&self) -> &DeviceBuffer {
        &self.d_in
    }

    /// `maxpool2d` / `maxpool2d_dx` ABI: `[N, C, H, W, K, stride, pad, Ho, Wo]`.
    /// `stride` sits BEFORE `pad`, matching `conv2d_gd`'s hyperparameter order —
    /// this is NOT the old `maxpool5` `[N,C,H,W,K,pad]` list with words appended.
    /// `Ho`/`Wo` are passed in and never recomputed by the kernel.
    fn params(&self) -> [u32; 9] {
        [
            self.in_shape.n,
            self.in_shape.c,
            self.in_shape.h,
            self.in_shape.w,
            self.spec.k,
            self.spec.stride,
            self.spec.pad,
            self.out_shape.h,
            self.out_shape.w,
        ]
    }

    /// One invocation per OUTPUT element.
    pub fn forward(&self, ctx: &Ctx, x: &DeviceBuffer) {
        let s = ctx.step(
            ctx.ids.need(ctx.ids.maxpool2d, "maxpool2d"),
            &[x, &self.out, &self.argmax],
            &self.params(),
            self.out_shape.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
    }

    /// `d_out` -> [`MaxPool::d_in`]. One invocation per INPUT element (gather
    /// form: every `d_in` element is written by exactly one invocation, so no
    /// pre-zeroing and no atomics).
    pub fn backward(&self, ctx: &Ctx, d_out: &DeviceBuffer) {
        let s = ctx.step(
            ctx.ids.need(ctx.ids.maxpool2d_dx, "maxpool2d_dx"),
            &[d_out, &self.argmax, &self.d_in],
            &self.params(),
            self.in_shape.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// LayerNorm2d = channels-first LayerNorm, COMPOSED (nchw_nlc -> LN -> nlc_nchw)
// ===========================================================================

/// The two tensor names a [`LayerNorm2d`] owns.
#[derive(Clone, Debug)]
pub struct Ln2dNames {
    pub gamma: String,
    pub beta: String,
}

impl Ln2dNames {
    /// torch's `nn.LayerNorm` / the SAM 2 & ConvNeXt `LayerNorm2d` module:
    /// `P.{weight,bias}`.
    pub fn torch(prefix: &str) -> Ln2dNames {
        Ln2dNames { gamma: format!("{prefix}.weight"), beta: format!("{prefix}.bias") }
    }
    /// brain's spelling: `P.{gamma,beta}`.
    pub fn brain(prefix: &str) -> Ln2dNames {
        Ln2dNames { gamma: format!("{prefix}.gamma"), beta: format!("{prefix}.beta") }
    }
}

/// Channels-first LayerNorm over an NCHW map: each spatial position is
/// normalized across its C channels, then scaled by `gamma[c]` and shifted by
/// `beta[c]`.
///
/// ## Why this is COMPOSED and not a fused `layernorm2d` kernel
///
/// A fused channels-first kernel has to give one thread the C-strided walk over
/// a position's channels — i.e. **one thread per row**, which is the documented
/// coalescing trap: a warp's 32 loads land `H*W` floats apart, each 32-byte
/// sector fetched serves ONE useful float, and the kernel runs at ~1/8 of memory
/// bandwidth no matter how many positions there are. `layernorm_rows` is the
/// kernel that already fixed exactly this (a 64-thread workgroup walks one row;
/// measured 2.3-9.1x over the per-element form on a P40), and it wants its rows
/// contiguous.
///
/// So: permute NCHW -> NLC (`nchw_nlc`), which makes each position's C channels
/// a contiguous row, run the row-oriented LayerNorm family, permute back
/// (`nlc_nchw`). The two permutations are each other's inverse AND adjoint, so
/// the backward is the same pair swapped — no extra kernel in either direction.
///
/// **Measured** (Tesla P40, wgpu/Vulkan, release,
/// `crates/vision/tests/imaging_blocks.rs::layernorm2d_composition_cost`, SAM 2
/// Hiera-B+ @1024 feature-map shapes; every timed region bracketed by
/// `Gpu::poll_wait` — `Gpu::submit` alone only appends to a pending list and
/// times the HOST):
///
/// | shape | bytes | total | 2 permutes | permute GB/s | norm |
/// |---|---|---|---|---|---|
/// | 1x112x256x256 | 28.0 MiB | 2.789 ms | 2.442 ms (88%) |  48 | 0.348 ms |
/// | 1x224x128x128 | 14.0 MiB | 0.793 ms | 0.687 ms (87%) |  86 | 0.107 ms |
/// | 1x448x64x64   |  7.0 MiB | 0.333 ms | 0.240 ms (72%) | 122 | 0.093 ms |
/// | 1x896x32x32   |  3.5 MiB | 0.179 ms | 0.110 ms (61%) | 133 | 0.069 ms |
/// | 1x96x64x64    |  1.5 MiB | 0.124 ms | 0.079 ms (63%) |  80 | 0.046 ms |
///
/// (Run-to-run spread is ~10-20 %; a second run gave 56-84 % and 46-113 GB/s.
/// Nothing below turns on a figure that tight.)
///
/// **Read this table the way it actually reads.** The permutes are ~60-88 % of
/// the composed cost and they run at 46-133 GB/s on a ~346 GB/s card — they are
/// NOT at the memory roof. That is not a surprise once the kernels are read:
/// `nchw_nlc` writes coalesced but *gathers* `x[(n*C+ch)*HW + l]` with `ch`
/// varying fastest, so a warp's 32 loads land `H*W` floats apart; `nlc_nchw` is
/// the mirror image. Both permutes already pay the sector amplification that a
/// fused channels-first kernel is usually rejected for, and the amplification
/// gets *worse* as `H*W` grows (48 GB/s at `HW = 65536`, 133 GB/s at
/// `HW = 1024`). The row-oriented LayerNorm in the middle, by contrast, moves
/// 2n floats at ~169 GB/s at the largest shape — it is the only coalesced stage
/// here.
///
/// So this composition is the right FIRST implementation — it is correct, it
/// reuses the one selection site for the coalesced `*_rows` family, and it adds
/// no kernel — but the measurement does **not** establish that a fused
/// `layernorm2d` would be slower. It points the other way: a fused kernel would
/// do ~2 strided passes where the composition does ~6 (of which 4 are strided),
/// so it is the leading candidate for the next optimisation, not a rejected one.
/// The earlier version of this comment claimed the permutes ran at "377 GB/s,
/// at the memory roof" — that number came from a timing loop of bare `submit`s
/// that never reached the device, and 377 GB/s on a 346 GB/s card is
/// self-refuting. **Anyone adding the kernel must re-run the test above and
/// show the fused numbers next to these; anyone NOT adding it should not cite
/// this table as proof that it cannot win.**
///
/// Which LayerNorm kernel actually runs is decided by `model::block`'s
/// `LayerNormIds` seam on the queried `DeviceCaps` — the one selection site in
/// the workspace, so this block picks up `layernorm_rows` wherever the owning
/// model registered it, with no per-model wiring.
pub struct LayerNorm2d {
    names: Ln2dNames,
    pub shape: Shape,
    eps: f32,
    ln: mblock::LayerNormIds,

    xt: DeviceBuffer,   // NLC view of the input  [N*HW, C] — the backward's cache
    yt: DeviceBuffer,   // NLC normalized output
    out: DeviceBuffer,  // NCHW output
    dyt: DeviceBuffer,  // NLC view of d_out
    dxt: DeviceBuffer,  // NLC grad wrt xt
    mean: DeviceBuffer, // per-row mean [N*HW]
    inv: DeviceBuffer,  // per-row 1/sqrt(var+eps) [N*HW]
}

impl LayerNorm2d {
    /// `eps` is a parameter and not a constant: torch's `nn.LayerNorm` defaults
    /// to `1e-5`, but ConvNeXt's and SAM 2's `LayerNorm2d` are constructed with
    /// `1e-6`, and the difference is visible in a parity comparison.
    pub fn new(ctx: &Ctx, names: Ln2dNames, shape: Shape, eps: f32) -> LayerNorm2d {
        let rows = shape.n * shape.h * shape.w;
        let total = shape.numel();
        LayerNorm2d {
            names,
            shape,
            eps,
            ln: mblock::LayerNormIds::resolve(
                ctx.gpu,
                ctx.ids.need(ctx.ids.layernorm, "layernorm"),
                ctx.ids.need(ctx.ids.ln_stats, "ln_stats"),
                ctx.ids.need(ctx.ids.layernorm_dx, "layernorm_dx"),
            ),
            xt: ctx.act(total),
            yt: ctx.act(total),
            out: ctx.act(total),
            dyt: ctx.act(total),
            dxt: ctx.act(total),
            mean: ctx.act(rows),
            inv: ctx.act(rows),
        }
    }

    /// torch-named (`P.weight` / `P.bias`) at `eps = 1e-6` — ConvNeXt / SAM 2.
    pub fn torch(ctx: &Ctx, prefix: &str, shape: Shape) -> LayerNorm2d {
        LayerNorm2d::new(ctx, Ln2dNames::torch(prefix), shape, 1e-6)
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.shape.c as usize;
        vec![(self.names.gamma.clone(), c), (self.names.beta.clone(), c)]
    }

    /// `nchw_nlc` / `nlc_nchw` ABI: `[total, C, HW]`, one invocation per element.
    /// `hw` is the L axis — a call site that passes `rows = N*HW` here instead
    /// permutes a different tensor and still runs.
    fn perm_params(&self) -> [u32; 3] {
        [self.shape.numel(), self.shape.c, self.shape.h * self.shape.w]
    }
    /// Rows the LayerNorm sees: one per (image, spatial position).
    fn rows(&self) -> u32 {
        self.shape.n * self.shape.h * self.shape.w
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        let (total, c, rows) = (self.shape.numel(), self.shape.c, self.rows());
        let s_in = ctx.step(ctx.ids.need(ctx.ids.nchw_nlc, "nchw_nlc"), &[x, &self.xt], &self.perm_params(), total);
        let s_ln = mblock::layernorm_fwd(
            ctx.gpu,
            &self.ln,
            &self.xt,
            ps.w(&self.names.gamma),
            ps.w(&self.names.beta),
            &self.yt,
            c,
            rows,
            self.eps,
        );
        let s_out =
            ctx.step(ctx.ids.need(ctx.ids.nlc_nchw, "nlc_nchw"), &[&self.yt, &self.out], &self.perm_params(), total);
        // Strictly sequential: each step reads what the previous wrote. Steps
        // within one submit execute in the order given.
        ctx.gpu.submit(&[], &[s_in, s_ln, s_out]);
    }

    /// `d_out` (NCHW) -> `d_in` (NCHW, overwritten); `gamma`/`beta` grads
    /// ACCUMULATE into the ParamStore.
    ///
    /// Takes no `x_in`: the forward cached the NLC image of the input in `xt`,
    /// which is what every backward kernel here reads.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let (total, c, rows) = (self.shape.numel(), self.shape.c, self.rows());
        let steps = [
            // The adjoint of nlc_nchw is nchw_nlc — the same permutation matrix
            // transposed, which for a permutation IS its inverse.
            ctx.step(ctx.ids.need(ctx.ids.nchw_nlc, "nchw_nlc"), &[d_out, &self.dyt], &self.perm_params(), total),
            // dgamma/dbeta need mean + 1/sqrt(var+eps) recomputed from the cache.
            mblock::ln_stats_fwd(ctx.gpu, &self.ln, &self.xt, &self.mean, &self.inv, c, rows, self.eps),
            ctx.step(
                ctx.ids.need(ctx.ids.layernorm_dgamma, "layernorm_dgamma"),
                &[&self.dyt, &self.xt, &self.mean, &self.inv, ps.g(&self.names.gamma)],
                &[c, rows],
                c,
            ),
            ctx.step(
                ctx.ids.need(ctx.ids.layernorm_dbeta, "layernorm_dbeta"),
                &[&self.dyt, ps.g(&self.names.beta)],
                &[c, rows],
                c,
            ),
            mblock::layernorm_dx_bwd(
                ctx.gpu,
                &self.ln,
                &self.xt,
                ps.w(&self.names.gamma),
                &self.dyt,
                &self.dxt,
                c,
                rows,
                self.eps,
            ),
            ctx.step(ctx.ids.need(ctx.ids.nlc_nchw, "nlc_nchw"), &[&self.dxt, d_in], &self.perm_params(), total),
        ];
        ctx.gpu.submit(&[], &steps);
    }
}

// ===========================================================================
// CXBlock = ConvNeXt block
// ===========================================================================

/// Everything that varies between ConvNeXt blocks.
#[derive(Clone, Copy, Debug)]
pub struct CxSpec {
    /// Depthwise kernel size (7) and its `same` padding (3). `conv2d_gd` is
    /// fully general over K/stride/pad/dilation/groups, so 7x7 depthwise needs
    /// no kernel of its own.
    pub k: u32,
    pub pad: u32,
    /// Hidden width of the inverted bottleneck as a multiple of `dim` (4).
    pub mlp_ratio: u32,
    /// The `LayerNorm2d` epsilon — `1e-6` in ConvNeXt and SAM 2, not torch's
    /// `1e-5` default.
    pub eps: f32,
    /// `use_dwconv=False` makes the first conv DENSE (ConvNeXt's ablation, and
    /// SAM 2's `CXBlock(use_dwconv=...)` argument).
    pub depthwise: bool,
    /// Whether the block carries LayerScale (`layer_scale_init_value > 0`). When
    /// off there is no `gamma` tensor at all, and a strict checkpoint load would
    /// fail on a spurious one.
    pub layer_scale: bool,
    /// The activation between the two pointwise stages. Defaults to
    /// [`Act::GeluErf`] — torch's `nn.GELU()` — because that is what the
    /// checkpoints were trained with; [`Act::Gelu`] is the tanh approximation
    /// and a different function.
    pub act: Act,
}

impl CxSpec {
    /// ConvNeXt's / SAM 2's defaults: 7x7 depthwise, pad 3, 4x MLP, eps 1e-6,
    /// LayerScale on, exact GELU.
    pub const fn new() -> CxSpec {
        CxSpec { k: 7, pad: 3, mlp_ratio: 4, eps: 1e-6, depthwise: true, layer_scale: true, act: Act::GeluErf }
    }
    pub const fn with_act(self, act: Act) -> CxSpec {
        CxSpec { act, ..self }
    }
    pub const fn dense(self) -> CxSpec {
        CxSpec { depthwise: false, ..self }
    }
    pub const fn without_layer_scale(self) -> CxSpec {
        CxSpec { layer_scale: false, ..self }
    }
}

impl Default for CxSpec {
    fn default() -> CxSpec {
        CxSpec::new()
    }
}

/// A ConvNeXt block: `KxK depthwise conv -> LayerNorm2d -> pointwise(dim -> r*dim)
/// -> GELU -> pointwise(r*dim -> dim) -> LayerScale -> + input`.
///
/// Channel-preserving and spatially shape-preserving, so it drops into a trunk
/// or a neck anywhere.
///
/// **The two pointwise stages are `nn.Linear` in the reference**, applied to a
/// channels-last view. They are built here as 1x1 [`Conv`] units with
/// `Norm::None` and a bias, which is the same operator: a 1x1 conv weight
/// `[Cout, Cin, 1, 1]` has byte-identical flat layout to a Linear's
/// `[Cout, Cin]`, so the checkpoint tensor loads unchanged and no permutation
/// is needed. Building them as convs also means brain's fused/register-tiled
/// conv paths apply to them for free.
pub struct CXBlock {
    pub dwconv: Conv,
    pub norm: LayerNorm2d,
    pub pwconv1: Conv,
    pub pwconv2: Conv,
    /// The LayerScale tensor name, `None` when `spec.layer_scale` is off.
    gamma: Option<String>,
    pub spec: CxSpec,
    pub shape: Shape,

    scaled: DeviceBuffer, // gamma * pwconv2.out  [shape]
    sum: DeviceBuffer,    // input + scaled       [shape]
    d_scaled: DeviceBuffer,
    d_pw1: DeviceBuffer,
    d_norm: DeviceBuffer,
    d_dw: DeviceBuffer,
}

impl CXBlock {
    /// torch-named, matching the reference module: `P.dwconv.{weight,bias}`,
    /// `P.norm.{weight,bias}`, `P.pwconv1.{weight,bias}`, `P.pwconv2.{weight,bias}`,
    /// `P.gamma`.
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape, spec: CxSpec, train: bool) -> CXBlock {
        let dim = shape.c;
        let hidden = dim * spec.mlp_ratio;
        // dwconv: KxK, stride 1, `same` pad, groups = dim (or 1 when dense), BIASED,
        // no norm, no activation. Grouped + biased routes through
        // `conv2d_gd -> add_chan_inplace`; there is no fused grouped-bias kernel.
        let dw_spec = ConvSpec {
            cout: dim,
            k: spec.k,
            stride: 1,
            pad: spec.pad,
            groups: if spec.depthwise { dim } else { 1 },
            dilation: 1,
            norm: Norm::None,
            act: Act::None,
            bias: true,
        };
        let dwp = format!("{prefix}.dwconv");
        let dwconv = Conv::with_names(ctx, &dwp, ConvNames::torch_flat(&dwp), shape, dw_spec, train);
        let norm = LayerNorm2d::new(ctx, Ln2dNames::torch(&format!("{prefix}.norm")), shape, spec.eps);
        let pw = |cin_shape: Shape, cout: u32, act: Act, name: String| {
            let s = ConvSpec {
                cout,
                k: 1,
                stride: 1,
                pad: 0,
                groups: 1,
                dilation: 1,
                norm: Norm::None,
                act,
                bias: true,
            };
            Conv::with_names(ctx, &name, ConvNames::torch_flat(&name), cin_shape, s, train)
        };
        let pwconv1 = pw(shape, hidden, spec.act, format!("{prefix}.pwconv1"));
        let pwconv2 = pw(pwconv1.out_shape, dim, Act::None, format!("{prefix}.pwconv2"));
        let n = shape.numel();
        let hidden_n = pwconv1.out_shape.numel();
        CXBlock {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma: spec.layer_scale.then(|| format!("{prefix}.gamma")),
            spec,
            shape,
            // Unused without LayerScale, but never zero-sized: wgpu rejects a
            // zero-length buffer outright.
            scaled: ctx.act(if spec.layer_scale { n } else { 1 }),
            sum: ctx.act(n),
            d_scaled: ctx.act(if spec.layer_scale { n } else { 1 }),
            d_pw1: ctx.act(hidden_n),
            d_norm: ctx.act(n),
            d_dw: ctx.act(n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.sum
    }

    /// In the reference module's declaration order — `dwconv`, `norm`, `pwconv1`,
    /// `pwconv2`, `gamma` — so a checkpoint's key order matches 1:1.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.dwconv.param_list();
        v.extend(self.norm.param_list());
        v.extend(self.pwconv1.param_list());
        v.extend(self.pwconv2.param_list());
        if let Some(g) = &self.gamma {
            v.push((g.clone(), self.shape.c as usize));
        }
        v
    }

    /// Propagate the eval/train BN toggle. All three convs are `Norm::None`, so
    /// this is a no-op today — kept so a composite that owns `CXBlock`s can call
    /// it uniformly and so a future normed variant cannot forget it.
    pub fn set_eval(&self, eval: bool) {
        self.dwconv.set_eval(eval);
        self.pwconv1.set_eval(eval);
        self.pwconv2.set_eval(eval);
    }

    /// `scale_chan` / `scale_chan_dg` ABI: `[total, C, inner]` with the channel
    /// read as `(idx / inner) % C`. For NCHW that is `inner = H*W`, which makes
    /// the gain SHARED across the batch — correct here, because LayerScale is a
    /// learned parameter, not a per-image gate. (An `[N,C]` per-image gate needs
    /// `film_chan`/`add_chan_bcast` instead; using `scale_chan` for one applies
    /// image 0's values to the whole batch.)
    fn ls_params(&self) -> [u32; 3] {
        [self.shape.numel(), self.shape.c, self.shape.h * self.shape.w]
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.dwconv.forward(ctx, ps, x_in);
        self.norm.forward(ctx, ps, self.dwconv.out());
        self.pwconv1.forward(ctx, ps, self.norm.out());
        self.pwconv2.forward(ctx, ps, self.pwconv1.out());
        let n = self.shape.numel();
        let branch: &DeviceBuffer = match &self.gamma {
            Some(g) => {
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.scale_chan, "scale_chan"),
                    &[self.pwconv2.out(), ps.w(g), &self.scaled],
                    &self.ls_params(),
                    n,
                );
                ctx.gpu.submit(&[], &[s]);
                &self.scaled
            }
            None => self.pwconv2.out(),
        };
        let s = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[x_in, branch, &self.sum], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }

    /// `d_out` -> `d_in` (grad wrt `x_in`, overwritten). The residual means
    /// `d_in` receives the branch gradient AND `d_out` itself, accumulated last.
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let n = self.shape.numel();
        // sum = x_in + scaled  =>  d(scaled) = d_out and d(x_in) += d_out.
        let d_branch: &DeviceBuffer = match &self.gamma {
            Some(g) => {
                // dgamma ACCUMULATES (like every *_dw kernel), so the caller's
                // zero_grads must have cleared it. dx is scale_chan with the
                // same gain — no separate kernel.
                let s_dg = ctx.step(
                    ctx.ids.need(ctx.ids.scale_chan_dg, "scale_chan_dg"),
                    &[self.pwconv2.out(), d_out, ps.g(g)],
                    &self.ls_params(),
                    self.shape.c,
                );
                let s_dx = ctx.step(
                    ctx.ids.need(ctx.ids.scale_chan, "scale_chan"),
                    &[d_out, ps.w(g), &self.d_scaled],
                    &self.ls_params(),
                    n,
                );
                ctx.gpu.submit(&[], &[s_dg, s_dx]);
                &self.d_scaled
            }
            None => d_out,
        };
        self.pwconv2.backward(ctx, ps, self.pwconv1.out(), d_branch, &self.d_pw1);
        self.pwconv1.backward(ctx, ps, self.norm.out(), &self.d_pw1, &self.d_norm);
        self.norm.backward(ctx, ps, &self.d_norm, &self.d_dw);
        self.dwconv.backward(ctx, ps, x_in, &self.d_dw, d_in);
        let s = ctx.step(ctx.ids.need(ctx.ids.add_inplace, "add_inplace"), &[d_in, d_out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }
}
