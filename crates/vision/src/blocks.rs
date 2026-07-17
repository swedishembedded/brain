// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 convolutional building blocks (P2): `Conv`, `Bottleneck`, `C2f`,
//! `SPPF`. NCHW throughout.
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
}

impl ConvSpec {
    /// yolo's unit: dense, SiLU.
    pub const fn silu(cout: u32, k: u32, stride: u32, pad: u32) -> ConvSpec {
        ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::Bn, act: Act::Silu }
    }
    /// ZipDepth's `ConvBN`: grouped/dilated, ReLU.
    pub const fn relu(cout: u32, k: u32, stride: u32, pad: u32) -> ConvSpec {
        ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::Bn, act: Act::Relu }
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
        ConvSpec { cout: ch, k, stride, pad, groups: ch, dilation: 1, norm: Norm::Bn, act }
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


/// The five tensor names a `Conv` unit owns.
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
            weight: format!("{prefix}.conv.weight"),
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
            weight: format!("{prefix}.{conv_idx}.weight"),
            gamma: format!("{prefix}.{bn_idx}.weight"),
            beta: format!("{prefix}.{bn_idx}.bias"),
            run_mean: format!("{prefix}.{bn_idx}.running_mean"),
            run_var: format!("{prefix}.{bn_idx}.running_var"),
        }
    }
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

    /// Lazily-allocated [in] scratch holding the tapped (possibly fake-quantized)
    /// conv input, used only when a [`crate::ActTap`] is installed (NPU
    /// calibration / fake-quant). Never allocated on the normal inference path.
    q_in: std::cell::RefCell<Option<DeviceBuffer>>,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    /// yolo's ctor, unchanged: dense conv + BN + SiLU.
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
            q_in: std::cell::RefCell::new(None),
        }
    }

    /// This unit's conv-weight tensor name.
    pub fn names_weight(&self) -> &str {
        &self.names.weight
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.act
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

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.out_shape.c as usize;
        // Grouped conv weights are `[cout, cin/groups, k, k]` — NOT `[cout, cin,
        // k, k]`. Depthwise (groups == cin) makes the second axis 1.
        let cin_g = (self.in_shape.c / self.spec.groups) as usize;
        let k = self.k as usize;
        let mut v = vec![(self.names.weight.clone(), c * cin_g * k * k)];
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
    /// Only SiLU has a fused kernel (`conv_act*` hardcodes it in the WGSL body),
    /// and only the dense conv is fast-pathed. Everything else runs the unfused
    /// `conv -> bn_eval -> act` — which is why `bn_eval`, registered and tested
    /// but previously dead, now has a consumer. Its eps matches `pack_sb`'s
    /// (`1e-5`), so the two paths agree numerically by construction.
    fn can_fuse(&self, ctx: &Ctx) -> bool {
        self.spec.act == Act::Silu && self.spec.norm == Norm::Bn && self.spec.is_dense() && ctx.ids.conv_act_reg != crate::NONE
    }

    /// The unfused activation's (forward, backward) kernel pair, or `None` for
    /// `Act::None`. ReLU maps to `leaky_relu` at slope 0 — identical in both
    /// directions — so it needs no kernel of its own.
    fn act_pair(&self, ctx: &Ctx) -> Option<(usize, usize)> {
        match self.spec.act {
            Act::None => None,
            Act::Silu => Some((ctx.ids.need(ctx.ids.silu, "silu"), ctx.ids.need(ctx.ids.silu_bwd, "silu_bwd"))),
            Act::Relu => Some((
                ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
                ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"),
            )),
            Act::Sigmoid => {
                Some((ctx.ids.need(ctx.ids.sigmoid, "sigmoid"), ctx.ids.need(ctx.ids.sigmoid_bwd, "sigmoid_bwd")))
            }
        }
    }
    /// `leaky_relu`'s uniform is `[total, slope]` with slope a bit-cast f32;
    /// `silu`'s and `sigmoid`'s are `[total]`. Slope 0 makes leaky_relu exactly relu.
    fn act_params(&self, n: u32) -> Vec<u32> {
        match self.spec.act {
            Act::Relu => vec![n, f(0.0)],
            _ => vec![n],
        }
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
        let _ = c;

        if self.spec.norm == Norm::None {
            // Raw conv -> act. No stats, no host interleave, no mode distinction:
            // without BN the two modes compute the same function.
            let s_conv = ctx.step(self.conv_kind(ctx), &[x_in, ps.w(&self.names.weight), &self.conv_out], &self.conv_params(), on);
            ctx.gpu.submit(&[], &[s_conv]);
            let s_act = match self.act_pair(ctx) {
                Some((fwd, _)) => ctx.step(fwd, &[&self.conv_out, &self.act], &self.act_params(on), on),
                None => ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.conv_out, &self.act], &[on, f(1.0)], on),
            };
            ctx.gpu.submit(&[], &[s_act]);
            return;
        }

        if !self.train.get() {
            // Inference: one fused conv -> BN(eval) -> SiLU dispatch. The BN-eval
            // transform is collapsed per channel into `sb` once (constant across
            // frames), so there is no per-frame host stat packing nor separate
            // bn_eval/silu passes.
            if !self.can_fuse(ctx) {
                // Unfused eval: conv -> bn_eval -> act. Taken by every ReLU or
                // grouped/dilated unit (i.e. all of ZipDepth), since the fused
                // conv_act* kernels hardcode SiLU in their WGSL body and the dense
                // fast path ignores `groups`.
                self.forward_eval_unfused(ctx, ps, x_in);
                return;
            }
            if !self.sb_ready.get() {
                self.pack_sb(ctx, ps);
                self.sb_ready.set(true);
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
                // threads = N * ceil(Cout/8) * ceil(Ho*Wo/4).
                let ntc = self.out_shape.c.div_ceil(8);
                let npq = (self.out_shape.h * self.out_shape.w).div_ceil(4);
                (ctx.ids.conv_act_reg, self.out_shape.n * ntc * npq)
            };
            // Calibration / fake-quant tap (NPU INT8): route the conv input
            // through the host so the tap can read its range and/or rewrite it
            // (quant→dequant), then convolve the tapped copy. Only taken when a
            // tap is installed; the normal inference path skips this entirely.
            if let Some(tap) = ctx.tap {
                let in_n = self.in_shape.numel() as usize;
                let mut h = ctx.gpu.read(x_in, in_n);
                tap.tap(&self.prefix, &mut h);
                if self.q_in.borrow().is_none() {
                    *self.q_in.borrow_mut() = Some(ctx.gpu.storage(in_n as u64));
                }
                let q = self.q_in.borrow();
                let qbuf = q.as_ref().unwrap();
                ctx.gpu.write(qbuf, bytemuck::cast_slice(&h));
                let s = ctx.step(
                    kind,
                    &[qbuf, ps.w(&self.names.weight), &self.sb, &self.act],
                    &self.conv_params(),
                    threads,
                );
                ctx.gpu.submit(&[], &[s]);
                return;
            }
            let s = ctx.step(
                kind,
                &[x_in, ps.w(&self.names.weight), &self.sb, &self.act],
                &self.conv_params(),
                threads,
            );
            ctx.gpu.submit(&[], &[s]);
            return;
        }

        // Train mode: conv -> bn_stats, host-pack mv/mvg, then bn_train -> silu.
        self.pack_gb(ctx, ps);
        let s_conv = ctx.step(self.conv_kind(ctx), &[x_in, ps.w(&self.names.weight), &self.conv_out], &self.conv_params(), on);
        let s_stats = ctx.step(ctx.ids.bn_stats, &[&self.conv_out, &self.mean, &self.var], &self.nchw(), c);
        let mut pre = vec![s_conv, s_stats];
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
        let s_act = match self.act_pair(ctx) {
            Some((fwd, _)) => ctx.step(fwd, &[&self.bn_out, &self.act], &self.act_params(on), on),
            // Act::None: the block's output IS the BN output. Copying via a
            // slope-1 leaky_relu would be a wasted dispatch, so alias instead.
            None => ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.bn_out, &self.act], &[on, f(1.0)], on),
        };
        ctx.gpu.submit(&[], &[s_train, s_act]);
    }

    /// Unfused eval: `conv -> bn_eval -> act`.
    ///
    /// Taken by every unit the fused path cannot serve — any ReLU unit, and any
    /// grouped/dilated one (i.e. all of ZipDepth). The fused `conv_act*` kernels
    /// hardcode SiLU in their WGSL body, and the dense conv they route to on CPU
    /// ignores `groups`, so fusing here would be wrong rather than merely slower.
    ///
    /// This is what finally gives `bn_eval` a consumer: it has been registered and
    /// tested since P2 but nothing dispatched it, because yolo always fused. Its
    /// eps (`1e-5`) matches `pack_sb`'s, so the fused and unfused paths agree
    /// numerically by construction rather than by coincidence.
    ///
    /// NOTE `bn_eval` takes the SAME four buffers as `bn_train` — `x, mv, gb,
    /// out` — with the RUNNING mean|var in `mv`, not the collapsed `scale|bias`
    /// in `sb`. `sb` exists only for the fused `conv_act*` kernels. Binding `sb`
    /// here instead reads binding 3 out of bounds and, since the CPU JIT compiles
    /// with `MemFlags::trusted()` (no bounds checks), SEGFAULTS rather than
    /// erroring. The per-channel packing is still cached across frames via
    /// `sb_ready`, which now gates `mv`+`gb` instead.
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
        // The calibration / fake-quant tap lives on the eval path, and must fire
        // exactly once per conv at the same point as it does when fused — the NPU
        // calibrator keys its scale map on `self.prefix`.
        let tapped: Option<DeviceBuffer> = ctx.tap.map(|tap| {
            let in_n = self.in_shape.numel() as usize;
            let mut h = ctx.gpu.read(x_in, in_n);
            tap.tap(&self.prefix, &mut h);
            if self.q_in.borrow().is_none() {
                *self.q_in.borrow_mut() = Some(ctx.gpu.storage(in_n as u64));
            }
            let q = self.q_in.borrow();
            let qbuf = q.as_ref().unwrap().clone();
            ctx.gpu.write(&qbuf, bytemuck::cast_slice(&h));
            qbuf
        });
        let src = tapped.as_ref().unwrap_or(x_in);

        let s_conv = ctx.step(
            self.conv_kind(ctx),
            &[src, ps.w(&self.names.weight), &self.conv_out],
            &self.conv_params(),
            on,
        );
        let s_bn = ctx.step(
            ctx.ids.need(ctx.ids.bn_eval, "bn_eval"),
            &[&self.conv_out, &self.mv, &self.gb, &self.bn_out],
            &self.nchw(),
            on,
        );
        let s_act = match self.act_pair(ctx) {
            Some((fwd, _)) => ctx.step(fwd, &[&self.bn_out, &self.act], &self.act_params(on), on),
            None => ctx.step(
                ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
                &[&self.bn_out, &self.act],
                &[on, f(1.0)],
                on,
            ),
        };
        ctx.gpu.submit(&[], &[s_conv, s_bn, s_act]);
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
            let scale = gamma[i] / (rvar[i] + 1e-5).sqrt();
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
            // Raw conv: act backward straight into d_conv, then the conv adjoints.
            let s_a = match self.act_pair(ctx) {
                Some((_, bwd)) => ctx.step(bwd, &[&self.conv_out, d_out, &self.d_conv], &self.act_params(on), on),
                None => ctx.step(ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"), &[&self.conv_out, d_out, &self.d_conv], &[on, f(1.0)], on),
            };
            ctx.gpu.submit(&[], &[s_a]);
            let dw_n = self.out_shape.c * (self.in_shape.c / self.spec.groups) * self.k * self.k;
            let s_dw = ctx.step(self.conv_dw_kind(ctx), &[&self.d_conv, x_in, ps.g(&self.names.weight)], &self.conv_params(), dw_n);
            let s_dxin = ctx.step(self.conv_dx_kind(ctx), &[&self.d_conv, ps.w(&self.names.weight), d_in], &self.conv_params(), self.in_shape.numel());
            ctx.gpu.submit(&[], &[s_dw, s_dxin]);
            return;
        }
        let s_act = match self.act_pair(ctx) {
            Some((_, bwd)) => ctx.step(bwd, &[&self.bn_out, d_out, &self.d_bn], &self.act_params(on), on),
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
            let s = ctx.step(ctx.ids.add2, &[d_in, d_out, d_in], &[on], on);
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
            let s = ctx.step(ctx.ids.add2, &[&self.d_chunk[in_idx], &self.d_y1, &self.d_chunk[in_idx]], &[chunk_n], chunk_n);
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
// SPPF = Conv1x1 -> m1,m2,m3 = maxpool5 chain -> concat[x,m1,m2,m3] -> Conv1x1
// ===========================================================================

/// Spatial-Pyramid-Pooling-Fast. A 1x1 conv, three chained 5x5 maxpools, a
/// channel-concat of `[x, m1, m2, m3]` (4*c channels), and a final 1x1 conv.
pub struct SPPF {
    pub cv1: Conv,
    pub cv2: Conv,
    pub in_shape: Shape,
    pub out_shape: Shape,
    c: u32,
    sh: Shape, // [n,c,h,w] of the inner maps

    // forward caches
    m1: DeviceBuffer,
    m2: DeviceBuffer,
    m3: DeviceBuffer,
    am1: DeviceBuffer,
    am2: DeviceBuffer,
    am3: DeviceBuffer,
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
    d_tmp: DeviceBuffer, // scratch for maxpool dx contributions
}

impl SPPF {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, train: bool) -> SPPF {
        // Ultralytics SPPF: cv1 halves channels to c = c_out/2, cv2 maps 4c->c_out.
        let c = c_out / 2;
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, c, 1, 1, 0, train);
        let sh = cv1.out_shape;
        let cat_shape = Shape::new(sh.n, 4 * c, sh.h, sh.w);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cat_shape, c_out, 1, 1, 0, train);
        let out_shape = cv2.out_shape;
        let n1 = sh.numel();
        SPPF {
            cv1,
            cv2,
            in_shape,
            out_shape,
            c,
            sh,
            m1: ctx.act(n1),
            m2: ctx.act(n1),
            m3: ctx.act(n1),
            am1: ctx.act(n1),
            am2: ctx.act(n1),
            am3: ctx.act(n1),
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
            d_tmp: ctx.act(n1),
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

    fn pool_params(&self) -> [u32; 6] {
        // maxpool5 ABI: [N, C, H, W, K, pad], K=5 pad=2.
        [self.sh.n, self.c, self.sh.h, self.sh.w, 5, 2]
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cv1.forward(ctx, ps, x_in);
        let x = self.cv1.out();
        let n1 = self.sh.numel();
        // m1 = pool(x); m2 = pool(m1); m3 = pool(m2). Sequential dependency.
        let s1 = ctx.step(ctx.ids.maxpool5, &[x, &self.m1, &self.am1], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s1]);
        let s2 = ctx.step(ctx.ids.maxpool5, &[&self.m1, &self.m2, &self.am2], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s2]);
        let s3 = ctx.step(ctx.ids.maxpool5, &[&self.m2, &self.m3, &self.am3], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s3]);

        // concat [x, m1, m2, m3] via left-fold.
        let c = self.c;
        let (h, w, n) = (self.sh.h, self.sh.w, self.sh.n);
        let sc1 = ctx.step(ctx.ids.concat2, &[x, &self.m1, &self.cat1], &[n, c, c, h, w], 2 * n1);
        ctx.gpu.submit(&[], &[sc1]);
        let sc2 = ctx.step(ctx.ids.concat2, &[&self.cat1, &self.m2, &self.cat2], &[n, 2 * c, c, h, w], 3 * n1);
        ctx.gpu.submit(&[], &[sc2]);
        let sc3 = ctx.step(ctx.ids.concat2, &[&self.cat2, &self.m3, &self.concat], &[n, 3 * c, c, h, w], 4 * n1);
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
        // d_m2 = d_m2_cat + maxpool_dx(d_m3 -> via am3)
        let sd3 = ctx.step(ctx.ids.maxpool5_dx, &[&self.d_m3, &self.am3, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd3]);
        let a3 = ctx.step(ctx.ids.add2, &[&self.d_m2_cat, &self.d_tmp, &self.d_m2], &[n1], n1);
        ctx.gpu.submit(&[], &[a3]);

        // m2 = pool(m1): grad wrt m1 = d_m1_cat + maxpool_dx(d_m2 -> via am2)
        let sd2 = ctx.step(ctx.ids.maxpool5_dx, &[&self.d_m2, &self.am2, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd2]);
        let a2 = ctx.step(ctx.ids.add2, &[&self.d_m1_cat, &self.d_tmp, &self.d_m1], &[n1], n1);
        ctx.gpu.submit(&[], &[a2]);

        // m1 = pool(x): grad wrt x = d_x_cat + maxpool_dx(d_m1 -> via am1)
        let sd1 = ctx.step(ctx.ids.maxpool5_dx, &[&self.d_m1, &self.am1, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd1]);
        let a1 = ctx.step(ctx.ids.add2, &[&self.d_x_cat, &self.d_tmp, &self.d_x], &[n1], n1);
        ctx.gpu.submit(&[], &[a1]);

        // cv1 backward -> d_in.
        self.cv1.backward(ctx, ps, x_in, &self.d_x, d_in);
    }
}
