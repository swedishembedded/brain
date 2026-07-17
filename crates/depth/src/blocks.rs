// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's model-specific blocks.
//!
//! Everything conv-shaped composes `vision::Conv` (via [`ConvNames::torch_seq`] /
//! [`ConvNames::torch_conv_bn`], so each block's param list mirrors the reference
//! checkpoint exactly). Only the wiring that is genuinely ZipDepth's lives here.
//!
//! Same discipline as `vision::blocks`: constructed once (registering params,
//! pre-allocating SSA activation + backward-temporary buffers), then
//! `forward(ctx, ps, x) -> &out` and `backward(ctx, ps, x, d_out, d_in)`.

use gpu_core::{f, DeviceBuffer};
use paramstore::ParamStore;
use vision::blocks::{Act, Conv, ConvNames, ConvSpec, Norm};
use vision::{Ctx, Shape};

// ===========================================================================
// QARepBlock — RepVGG: conv3x3+BN  +  conv1x1+BN  [+ identity]  -> ReLU
// ===========================================================================

/// The encoder's workhorse: 15 instances in the base config.
///
/// Train-time this is three branches summed then ReLU'd. Inference-time the whole
/// thing collapses to ONE biased 3x3 (see [`crate::fuse_qarep`]) — that collapse
/// is what takes the released model from 6.79M params to its 6.1M headline.
/// Training runs the unfused form because the branches carry separate BN
/// statistics that only exist while training.
///
/// ⚠️ The identity branch has NO BatchNorm (`architecture.py:89-108`), unlike
/// canonical RepVGG. It is a raw `+ x`.
pub struct QARepBlock {
    pub in_shape: Shape,
    pub out_shape: Shape,
    has_identity: bool,
    b3: Conv,
    b1: Conv,
    sum: DeviceBuffer,   // branch_3x3 + branch_1x1 [+ x]  (pre-activation)
    act: DeviceBuffer,   // relu(sum) — the block output
    d_sum: DeviceBuffer, // grad wrt sum
    d_b1: DeviceBuffer,  // grad wrt x from the 1x1 branch
    acc: DeviceBuffer,   // out-of-place accumulator for the multi-consumer x grad
}

impl QARepBlock {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, stride: u32, train: bool) -> QARepBlock {
        // Both branches are conv+BN with NO activation — the ReLU is applied once
        // to their sum, not per branch.
        let s3 = ConvSpec::relu(cout, 3, stride, 1).with_act(Act::None);
        let s1 = ConvSpec::relu(cout, 1, stride, 0).with_act(Act::None);
        let b3 = Conv::with_names(
            ctx,
            &format!("{prefix}.branch_3x3"),
            ConvNames::torch_seq(&format!("{prefix}.branch_3x3"), 0, 1),
            in_shape,
            s3,
            train,
        );
        let b1 = Conv::with_names(
            ctx,
            &format!("{prefix}.branch_1x1"),
            ConvNames::torch_seq(&format!("{prefix}.branch_1x1"), 0, 1),
            in_shape,
            s1,
            train,
        );
        let out_shape = b3.out_shape;
        assert_eq!(out_shape, b1.out_shape, "QARepBlock branches must agree on shape");
        // The reference's own condition (`architecture.py:89`).
        let has_identity = in_shape.c == cout && stride == 1;
        let on = out_shape.numel();
        QARepBlock {
            in_shape,
            out_shape,
            has_identity,
            b3,
            b1,
            sum: ctx.act(on),
            act: ctx.act(on),
            d_sum: ctx.act(on),
            d_b1: ctx.act(in_shape.numel()),
            acc: ctx.act(in_shape.numel()),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.act
    }
    pub fn set_eval(&self, on: bool) {
        self.b3.set_eval(on);
        self.b1.set_eval(on);
    }
    pub fn set_update_running(&self, on: bool) {
        self.b3.set_update_running(on);
        self.b1.set_update_running(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.b3.param_list();
        v.extend(self.b1.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let on = self.out_shape.numel();
        self.b3.forward(ctx, ps, x_in);
        self.b1.forward(ctx, ps, x_in);
        // sum = b3 + b1 [+ x]. `add2` is SSA (distinct in/out buffers), so with an
        // identity the pair sums through `d_sum` as scratch — it is free here,
        // since backward overwrites it before reading.
        if self.has_identity {
            let s_pair = ctx.step(ctx.ids.add2, &[self.b3.out(), self.b1.out(), &self.d_sum], &[on], on);
            let s_id = ctx.step(ctx.ids.add2, &[&self.d_sum, x_in, &self.sum], &[on], on);
            ctx.gpu.submit(&[], &[s_pair, s_id]);
        } else {
            let s_pair = ctx.step(ctx.ids.add2, &[self.b3.out(), self.b1.out(), &self.sum], &[on], on);
            ctx.gpu.submit(&[], &[s_pair]);
        }
        let s_relu = ctx.step(
            ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
            &[&self.sum, &self.act],
            &[on, f(0.0)],
            on,
        );
        ctx.gpu.submit(&[], &[s_relu]);
    }

    /// `d_out` -> `d_in`. `x` feeds up to THREE consumers (both branches and the
    /// identity), so their gradients accumulate out-of-place via `add2`.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let on = self.out_shape.numel();
        let inn = self.in_shape.numel();
        // ReLU backward: d_sum = d_out * [sum > 0]
        let s_relu = ctx.step(
            ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"),
            &[&self.sum, d_out, &self.d_sum],
            &[on, f(0.0)],
            on,
        );
        ctx.gpu.submit(&[], &[s_relu]);
        // The sum is linear, so BOTH branches (and the identity) see d_sum.
        self.b3.backward(ctx, ps, x_in, &self.d_sum, d_in);
        self.b1.backward(ctx, ps, x_in, &self.d_sum, &self.d_b1);
        let s_acc = ctx.step(ctx.ids.add2, &[d_in, &self.d_b1, &self.acc], &[inn], inn);
        ctx.gpu.submit(&[], &[s_acc]);
        if self.has_identity {
            // + the identity's own d_sum (it is `+ x`, so its adjoint is d_sum).
            let s_id = ctx.step(ctx.ids.add2, &[&self.acc, &self.d_sum, d_in], &[inn], inn);
            ctx.gpu.submit(&[], &[s_id]);
        } else {
            let s_cp = ctx.step(
                ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
                &[&self.acc, d_in],
                &[inn, f(1.0)],
                inn,
            );
            ctx.gpu.submit(&[], &[s_cp]);
        }
    }
}

// ===========================================================================
// ChannelAttention (SE) — pool -> 1x1 -> relu -> 1x1 -> sigmoid -> scale
// ===========================================================================

/// Squeeze-and-excitation. `reduction = 8`, `hidden = max(dim/8, 4)`.
///
/// Both 1x1 convs are **bias-free and have no BatchNorm** (`architecture.py:
/// 148-162`), so they are raw convs rather than `vision::Conv` units.
///
/// The gate is `[N,C,1,1]` — PER IMAGE. `scale_chan` indexes its scale by
/// `(idx/inner) % c`, so the naive `(c=C, inner=H*W)` would apply image 0's gate
/// to every image in the batch. Passing **`c = N*C, inner = H*W`** makes the index
/// `n*C + c`, i.e. exactly the per-image gate — the existing kernel covers this
/// under the right arguments, so no new one is needed. (Same class of trap as
/// `bias_add`, which does NOT have such an escape.)
pub struct ChannelAttention {
    prefix: String,
    pub shape: Shape,
    hidden: u32,
    pooled: DeviceBuffer,
    h: DeviceBuffer,
    h_act: DeviceBuffer,
    g: DeviceBuffer,
    gate: DeviceBuffer,
    out: DeviceBuffer,
    prod: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_g: DeviceBuffer,
    d_h_act: DeviceBuffer,
    d_h: DeviceBuffer,
    d_pooled: DeviceBuffer,
    d_x_pool: DeviceBuffer,
    d_x_mul: DeviceBuffer,
}

impl ChannelAttention {
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape) -> ChannelAttention {
        let (n, c) = (shape.n, shape.c);
        let hidden = (c / 8).max(4);
        ChannelAttention {
            prefix: prefix.to_string(),
            shape,
            hidden,
            pooled: ctx.act(n * c),
            h: ctx.act(n * hidden),
            h_act: ctx.act(n * hidden),
            g: ctx.act(n * c),
            gate: ctx.act(n * c),
            out: ctx.act(shape.numel()),
            prod: ctx.act(shape.numel()),
            d_gate: ctx.act(n * c),
            d_g: ctx.act(n * c),
            d_h_act: ctx.act(n * hidden),
            d_h: ctx.act(n * hidden),
            d_pooled: ctx.act(n * c),
            d_x_pool: ctx.act(shape.numel()),
            d_x_mul: ctx.act(shape.numel()),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (c, h) = (self.shape.c as usize, self.hidden as usize);
        vec![
            (format!("{}.fc.0.weight", self.prefix), h * c),
            (format!("{}.fc.2.weight", self.prefix), c * h),
        ]
    }

    /// A 1x1 conv over a `[N,C,1,1]` map. Expressed with the same `conv2d` the
    /// rest of the model uses, at H=W=1 — no separate GEMM path to keep in sync.
    fn c1(&self, ctx: &Ctx, w: &DeviceBuffer, x: &DeviceBuffer, y: &DeviceBuffer, cin: u32, cout: u32) {
        let n = self.shape.n;
        let s = ctx.step(
            ctx.ids.need(ctx.ids.conv2d, "conv2d"),
            &[x, w, y],
            &[n, cin, 1, 1, cout, 1, 1, 0, 1, 1],
            n * cout,
        );
        ctx.gpu.submit(&[], &[s]);
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let (n, c, h, w) = (self.shape.n, self.shape.c, self.shape.h, self.shape.w);
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let s_pool = ctx.step(
            ctx.ids.need(ctx.ids.avgpool2d, "avgpool2d"),
            &[x_in, &self.pooled],
            &[n, c, h, w, 1, 1],
            n * c,
        );
        ctx.gpu.submit(&[], &[s_pool]);
        self.c1(ctx, ps.w(&p("fc.0.weight")), &self.pooled, &self.h, c, self.hidden);
        let nh = n * self.hidden;
        let s_relu = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.h, &self.h_act], &[nh, f(0.0)], nh);
        ctx.gpu.submit(&[], &[s_relu]);
        self.c1(ctx, ps.w(&p("fc.2.weight")), &self.h_act, &self.g, self.hidden, c);
        let s_sig = ctx.step(ctx.ids.need(ctx.ids.sigmoid, "sigmoid"), &[&self.g, &self.gate], &[n * c], n * c);
        ctx.gpu.submit(&[], &[s_sig]);
        // x * gate, per-image per-channel (see the struct doc for why c = N*C).
        let s_mul = ctx.step(
            ctx.ids.need(ctx.ids.scale_chan, "scale_chan"),
            &[x_in, &self.gate, &self.out],
            &[self.shape.numel(), n * c, h * w],
            self.shape.numel(),
        );
        ctx.gpu.submit(&[], &[s_mul]);
    }

    /// `x` feeds TWO consumers — the squeeze (pool) and the scale (multiply) — so
    /// their gradients accumulate.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let (n, c, h, w) = (self.shape.n, self.shape.c, self.shape.h, self.shape.w);
        let hw = h * w;
        let tot = self.shape.numel();
        let p = |s: &str| format!("{}.{s}", self.prefix);

        // --- the multiply: out = x * gate ---
        // d_x_mul = d_out * gate  (the same broadcast as the forward)
        let s = ctx.step(
            ctx.ids.need(ctx.ids.scale_chan, "scale_chan"),
            &[d_out, &self.gate, &self.d_x_mul],
            &[tot, n * c, hw],
            tot,
        );
        ctx.gpu.submit(&[], &[s]);
        // d_gate[n,c] = sum_hw d_out * x. Elementwise product, then the per-(n,c)
        // spatial sum — which is exactly `add_chan_bcast_dv`'s adjoint-of-broadcast.
        let s = ctx.step(ctx.ids.need(ctx.ids.mul, "mul"), &[d_out, x_in, &self.prod], &[tot], tot);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(
            ctx.ids.need(ctx.ids.add_chan_bcast_dv, "add_chan_bcast_dv"),
            &[&self.prod, &self.d_gate],
            &[n, c, hw],
            n * c,
        );
        ctx.gpu.submit(&[], &[s]);

        // --- the excite chain, backwards ---
        let s = ctx.step(ctx.ids.need(ctx.ids.sigmoid_bwd, "sigmoid_bwd"), &[&self.g, &self.d_gate, &self.d_g], &[n * c], n * c);
        ctx.gpu.submit(&[], &[s]);
        // fc.2: [hidden -> c] at 1x1
        let p2 = [n, self.hidden, 1, 1, c, 1, 1, 0, 1, 1];
        let s_dw = ctx.step(ctx.ids.need(ctx.ids.conv2d_dw, "conv2d_dw"), &[&self.d_g, &self.h_act, ps.g(&p("fc.2.weight"))], &p2, (c * self.hidden) as u32);
        let s_dx = ctx.step(ctx.ids.need(ctx.ids.conv2d_dx, "conv2d_dx"), &[&self.d_g, ps.w(&p("fc.2.weight")), &self.d_h_act], &p2, n * self.hidden);
        ctx.gpu.submit(&[], &[s_dw, s_dx]);
        let nh = n * self.hidden;
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"), &[&self.h, &self.d_h_act, &self.d_h], &[nh, f(0.0)], nh);
        ctx.gpu.submit(&[], &[s]);
        // fc.0: [c -> hidden] at 1x1
        let p0 = [n, c, 1, 1, self.hidden, 1, 1, 0, 1, 1];
        let s_dw = ctx.step(ctx.ids.need(ctx.ids.conv2d_dw, "conv2d_dw"), &[&self.d_h, &self.pooled, ps.g(&p("fc.0.weight"))], &p0, (self.hidden * c) as u32);
        let s_dx = ctx.step(ctx.ids.need(ctx.ids.conv2d_dx, "conv2d_dx"), &[&self.d_h, ps.w(&p("fc.0.weight")), &self.d_pooled], &p0, n * c);
        ctx.gpu.submit(&[], &[s_dw, s_dx]);
        // the squeeze's adjoint: spread d_pooled back over space / (H*W)
        let s = ctx.step(
            ctx.ids.need(ctx.ids.avgpool2d_dx, "avgpool2d_dx"),
            &[&self.d_pooled, &self.d_x_pool],
            &[n, c, h, w, 1, 1],
            tot,
        );
        ctx.gpu.submit(&[], &[s]);

        // --- x's two paths sum ---
        let s = ctx.step(ctx.ids.add2, &[&self.d_x_mul, &self.d_x_pool, d_in], &[tot], tot);
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// MinimalMultiScale — two depthwise 3x3 branches (dilation 1 & 2), BN, residual
// ===========================================================================

/// `x + BN(dw_d1(x) + dw_d2(x))` (`architecture.py:285-294`).
///
/// Unconditional in stage2 — NOT gated by `global_mode`. Both branches are
/// bias-free depthwise 3x3 convs with no activation; there is no ReLU anywhere,
/// and the single BN sits on their SUM.
///
/// That last point is why this needs `vision::BatchNorm` rather than a
/// `vision::Conv`: a Conv's BN is welded to its own conv, and here one BN spans
/// two. The dilated branch (pad 2, dilation 2) is shape-preserving exactly as the
/// dilation-1 branch (pad 1) is — two receptive fields, one output shape, no
/// resampling.
pub struct MinimalMultiScale {
    pub shape: Shape,
    b1: Conv,
    b2: Conv,
    bn: vision::BatchNorm,
    sum: DeviceBuffer,
    out: DeviceBuffer,
    d_sum: DeviceBuffer,
    d_b2: DeviceBuffer,
    acc: DeviceBuffer,
}

impl MinimalMultiScale {
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape, train: bool) -> MinimalMultiScale {
        let c = shape.c;
        // Depthwise, bias-free, NO BN of their own (Act::None + a shared BN after).
        let mk = |name: &str, dil: u32, pad: u32| {
            // Norm::None: a RAW depthwise conv. The reference has exactly ONE bn
            // here, over the branches' SUM — giving each branch its own would run
            // BatchNorm twice and register BN tensors that do not exist in the
            // checkpoint.
            let spec = ConvSpec::depthwise(c, 3, 1, pad, Act::None)
                .with_dilation(dil)
                .with_norm(Norm::None);
            Conv::with_names(
                ctx,
                &format!("{prefix}.{name}"),
                // A bare `nn.Conv2d` attribute: `P.branchN.weight` and nothing
                // else. The BN names are unused at Norm::None (param_list omits
                // them and no dispatch reads them).
                ConvNames {
                    bias: String::new(),
                    weight: format!("{prefix}.{name}.weight"),
                    gamma: String::new(),
                    beta: String::new(),
                    run_mean: String::new(),
                    run_var: String::new(),
                },
                shape,
                spec,
                train,
            )
        };
        let b1 = mk("branch1", 1, 1);
        let b2 = mk("branch2", 2, 2);
        assert_eq!(b1.out_shape, shape, "MinimalMultiScale must be shape-preserving");
        assert_eq!(b2.out_shape, shape, "the dilated branch must also preserve shape");
        let bn = vision::BatchNorm::new(ctx, vision::BnNames::torch(&format!("{prefix}.bn")), shape, train);
        let n = shape.numel();
        MinimalMultiScale {
            shape,
            b1,
            b2,
            bn,
            sum: ctx.act(n),
            out: ctx.act(n),
            d_sum: ctx.act(n),
            d_b2: ctx.act(n),
            acc: ctx.act(n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn set_eval(&self, on: bool) {
        self.b1.set_eval(on);
        self.b2.set_eval(on);
        self.bn.set_eval(on);
    }
    pub fn set_update_running(&self, on: bool) {
        self.bn.set_update_running(on);
    }
    /// The two branch weights + the ONE shared BN. Note the branches contribute no
    /// BN tensors of their own — the reference has exactly one `bn` here.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.b1.param_list();
        v.extend(self.b2.param_list());
        v.extend(self.bn.param_list());
        v
    }
}

impl MinimalMultiScale {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let n = self.shape.numel();
        self.b1.forward(ctx, ps, x_in);
        self.b2.forward(ctx, ps, x_in);
        let s = ctx.step(ctx.ids.add2, &[self.b1.out(), self.b2.out(), &self.sum], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        self.bn.forward(ctx, ps, &self.sum);
        // The residual. No activation anywhere in this block.
        let s = ctx.step(ctx.ids.add2, &[x_in, self.bn.out(), &self.out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }

    /// `x` feeds THREE consumers — both branches and the residual — so the grads
    /// accumulate.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let n = self.shape.numel();
        // The residual is `+`, so d_out passes straight through to both the BN and x.
        self.bn.backward(ctx, ps, &self.sum, d_out, &self.d_sum);
        // The sum is linear -> both branches see d_sum.
        self.b1.backward(ctx, ps, x_in, &self.d_sum, &self.acc);
        self.b2.backward(ctx, ps, x_in, &self.d_sum, &self.d_b2);
        let s = ctx.step(ctx.ids.add2, &[&self.acc, &self.d_b2, d_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        // + the residual's own path.
        let s = ctx.step(ctx.ids.add2, &[d_in, d_out, &self.acc], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.acc, d_in], &[n, f(1.0)], n);
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// StripPoolingAttention — orthogonal strip pooling into a depthwise gate
// ===========================================================================

/// `x * sigmoid(BN(dw1x1(mean_W(x) + mean_H(x))))` (`architecture.py:236-247`).
///
/// The two strips are `x.mean(dim=3, keepdim=True)` -> `[B,C,H,1]` and
/// `x.mean(dim=2, keepdim=True)` -> `[B,C,1,W]`, and torch broadcasts their sum
/// back to `[B,C,H,W]`. Both means are [`avgpool2d`] at a degenerate output size
/// (`Ho=H,Wo=1` and `Ho=1,Wo=W`) — the reuse audit established these are
/// bit-identical to the dedicated `strip_pool` kernels that were deleted.
///
/// The gate is a depthwise 1x1 conv: `groups == dim`, so each channel's gate is
/// its own scalar weight — no cross-channel mixing at all. It is bias-free
/// (`bias=False`) with the BN supplying the shift.
///
/// `x` reaches the output by THREE routes (the two strips and the final
/// multiply), so its gradient is a three-way accumulation.
pub struct StripPoolingAttention {
    pub shape: Shape,
    h_shape: Shape,
    w_shape: Shape,
    gate: Conv,
    h_strip: DeviceBuffer,
    w_strip: DeviceBuffer,
    sum: DeviceBuffer,
    out: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_sum: DeviceBuffer,
    d_h: DeviceBuffer,
    d_w: DeviceBuffer,
    d_xh: DeviceBuffer,
    d_xw: DeviceBuffer,
    acc: DeviceBuffer,
}

impl StripPoolingAttention {
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape, train: bool) -> StripPoolingAttention {
        let c = shape.c;
        let h_shape = Shape { w: 1, ..shape };
        let w_shape = Shape { h: 1, ..shape };
        // nn.Sequential(Conv2d(dim,dim,1,groups=dim,bias=False), BatchNorm2d, Sigmoid)
        // -> `.0.weight` and `.1.{weight,bias,running_*}`.
        let gate = Conv::with_names(
            ctx,
            &format!("{prefix}.gate_conv"),
            ConvNames::torch_seq(&format!("{prefix}.gate_conv"), 0, 1),
            shape,
            ConvSpec::depthwise(c, 1, 1, 0, Act::Sigmoid),
            train,
        );
        assert_eq!(gate.out_shape, shape, "the gate must match x elementwise");
        let n = shape.numel();
        StripPoolingAttention {
            shape,
            h_shape,
            w_shape,
            gate,
            h_strip: ctx.act(h_shape.numel()),
            w_strip: ctx.act(w_shape.numel()),
            sum: ctx.act(n),
            out: ctx.act(n),
            d_gate: ctx.act(n),
            d_sum: ctx.act(n),
            d_h: ctx.act(h_shape.numel()),
            d_w: ctx.act(w_shape.numel()),
            d_xh: ctx.act(n),
            d_xw: ctx.act(n),
            acc: ctx.act(n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn set_eval(&self, on: bool) {
        self.gate.set_eval(on);
    }
    pub fn set_update_running(&self, on: bool) {
        self.gate.set_update_running(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        self.gate.param_list()
    }

    /// `avgpool2d`'s uniform, which is `[N, C, H, W, Ho, Wo]`.
    fn pool_params(&self, o: Shape) -> Vec<u32> {
        vec![self.shape.n, self.shape.c, self.shape.h, self.shape.w, o.h, o.w]
    }
}

impl StripPoolingAttention {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let n = self.shape.numel();
        let pool = ctx.ids.need(ctx.ids.avgpool2d, "avgpool2d");
        // mean over W -> [N,C,H,1]; mean over H -> [N,C,1,W].
        let sh = ctx.step(pool, &[x_in, &self.h_strip], &self.pool_params(self.h_shape), self.h_shape.numel());
        let sw = ctx.step(pool, &[x_in, &self.w_strip], &self.pool_params(self.w_shape), self.w_shape.numel());
        ctx.gpu.submit(&[], &[sh, sw]);
        // torch's broadcast of `h_strip + w_strip` back to the full map.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.broadcast_add_hw, "broadcast_add_hw"),
            &[&self.h_strip, &self.w_strip, &self.sum],
            &[self.shape.n, self.shape.c, self.shape.h, self.shape.w],
            n,
        );
        ctx.gpu.submit(&[], &[s]);
        self.gate.forward(ctx, ps, &self.sum);
        let s = ctx.step(ctx.ids.mul, &[x_in, self.gate.out(), &self.out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }

    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let n = self.shape.numel();
        // `out = x * gate` -> d_gate = d_out * x, and x's FIRST route: d_out * gate.
        let sg = ctx.step(ctx.ids.mul, &[d_out, x_in, &self.d_gate], &[n], n);
        let sx = ctx.step(ctx.ids.mul, &[d_out, self.gate.out(), &self.acc], &[n], n);
        ctx.gpu.submit(&[], &[sg, sx]);
        self.gate.backward(ctx, ps, &self.sum, &self.d_gate, &self.d_sum);
        // The adjoint of the broadcast: sum the map gradient over each broadcast axis.
        let da = ctx.ids.need(ctx.ids.broadcast_add_hw_da, "broadcast_add_hw_da");
        let p = |axis: u32| vec![self.shape.n, self.shape.c, self.shape.h, self.shape.w, axis];
        let sh = ctx.step(da, &[&self.d_sum, &self.d_h], &p(0), self.h_shape.numel());
        let sw = ctx.step(da, &[&self.d_sum, &self.d_w], &p(1), self.w_shape.numel());
        ctx.gpu.submit(&[], &[sh, sw]);
        // ...then back through each pool: x's second and third routes.
        let pool_dx = ctx.ids.need(ctx.ids.avgpool2d_dx, "avgpool2d_dx");
        let sh = ctx.step(pool_dx, &[&self.d_h, &self.d_xh], &self.pool_params(self.h_shape), n);
        let sw = ctx.step(pool_dx, &[&self.d_w, &self.d_xw], &self.pool_params(self.w_shape), n);
        ctx.gpu.submit(&[], &[sh, sw]);
        let s = ctx.step(ctx.ids.add2, &[&self.d_xh, &self.d_xw, d_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.add2, &[d_in, &self.acc, &self.d_xh], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.d_xh, d_in], &[n, f(1.0)], n);
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// GlobalContextBlock — GCNet-style learned global context
// ===========================================================================

/// `x + transform(bmm(x, softmax(context_weight(x))))` (`architecture.py:255-278`).
///
/// A LEARNED weighted global average pool: a 1x1 conv scores every spatial
/// position, softmax over `H*W` turns the scores into weights summing to 1, and the
/// feature map is contracted against them into a single `[B,C,1,1]` context vector.
/// That contraction is upstream's `bmm(x.view(B,C,HW), mask.view(B,HW,1))` and is
/// brain's [`weighted_gap`]; the residual add of a per-(image,channel) scalar is
/// `add_chan_bcast` (NOT `bias_add`, whose `[C]` vector is shared across the batch —
/// at N>1 it would add the wrong image's context).
///
/// The softmax is over the map, dispatched as `softmax_k` at `M=1`: a stride of 1
/// over `K=H*W` IS a contiguous softmax, which the reuse audit established is
/// bit-identical to the `softmax_hw` kernel that was written and then deleted.
///
/// All three convs are BIASED (`nn.Conv2d(..)` defaults `bias=True`) and all are
/// dense. `transform.0`'s bias is mathematically redundant — the BN right after it
/// subtracts the batch mean — but it is in the checkpoint, so it is carried.
///
/// ⚠️ TWO of the five bias/affine tensors here are MATHEMATICALLY DEAD — the loss is
/// exactly invariant to them, so their gradient is identically zero and they can
/// never learn:
///   * `context_weight.bias` is one scalar added to every position of the softmax
///     axis, and `softmax(z + b) == softmax(z)`.
///   * `transform.0.bias` is immediately followed by BN, which subtracts the mean —
///     the classic `bias=False`-before-BN redundancy, here left `True` upstream.
/// Both are in the checkpoint and must be loaded, and both are carried faithfully.
/// `gcb_two_biases_are_provably_dead` pins the invariance (measured: the loss moves
/// by EXACTLY 0.0f32 when either is shifted by +-0.5, and by 2 ULP at +-5.0, where
/// a live parameter would move it by ~850). That is also why neither can be
/// finite-difference-checked: their FD is 100% round-off noise.
///
/// NOTE: upstream's ONNX export monkey-patches this block into a uniform
/// `avg_pool2d`, DROPPING the learned softmax (`export.py:68-74`). brain implements
/// the real thing; P6 adds the avg-pool variant as an ablation row.
pub struct GlobalContextBlock {
    pub shape: Shape,
    /// dim -> 1, biased, no BN, no act. Scores each position.
    score: Conv,
    t0: Conv,
    t1: Conv,
    sm: DeviceBuffer,
    gap: DeviceBuffer,
    out: DeviceBuffer,
    d_t: DeviceBuffer,
    d_gap: DeviceBuffer,
    d_sm: DeviceBuffer,
    d_mask: DeviceBuffer,
    d_xg: DeviceBuffer,
    d_xs: DeviceBuffer,
    acc: DeviceBuffer,
}

impl GlobalContextBlock {
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape, reduction: u32, train: bool) -> GlobalContextBlock {
        let c = shape.c;
        let hidden = (c / reduction).max(8);
        let score = Conv::with_names(
            ctx,
            &format!("{prefix}.context_weight"),
            // A bare `nn.Conv2d` attribute: `.weight` + `.bias`, no BN.
            ConvNames {
                bias: format!("{prefix}.context_weight.bias"),
                weight: format!("{prefix}.context_weight.weight"),
                gamma: String::new(),
                beta: String::new(),
                run_mean: String::new(),
                run_var: String::new(),
            },
            shape,
            ConvSpec::relu(1, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            train,
        );
        // The transform runs on the CONTRACTED [B,C,1,1] vector, not the map.
        let ctx_shape = Shape { h: 1, w: 1, ..shape };

        // nn.Sequential(Conv2d(dim,hidden,1), BatchNorm2d, ReLU, Conv2d(hidden,dim,1))
        // -> indices 0,1 and 3.
        let t0 = Conv::with_names(
            ctx,
            &format!("{prefix}.transform.0"),
            ConvNames::torch_seq(&format!("{prefix}.transform"), 0, 1),
            ctx_shape,
            ConvSpec::relu(hidden, 1, 1, 0).with_bias(),
            train,
        );
        let t1 = Conv::with_names(
            ctx,
            &format!("{prefix}.transform.3"),
            ConvNames {
                bias: format!("{prefix}.transform.3.bias"),
                weight: format!("{prefix}.transform.3.weight"),
                gamma: String::new(),
                beta: String::new(),
                run_mean: String::new(),
                run_var: String::new(),
            },
            t0.out_shape,
            ConvSpec::relu(c, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            train,
        );
        let n = shape.numel();
        let hw = shape.h * shape.w;
        let t0_out = t0.out_shape.numel();
        GlobalContextBlock {
            shape,
            score,
            t0,
            t1,
            sm: ctx.act(shape.n * hw),
            gap: ctx.act(shape.n * c),
            out: ctx.act(n),
            d_t: ctx.act(shape.n * c),
            // Sized from t0's OUTPUT, not from `c`. `hidden` is `max(c/reduction, 8)`
            // and the clamp makes it EXCEED c whenever c < 32 — at which point a
            // `n*c` buffer here is an out-of-bounds write, and the CPU JIT's
            // MemFlags::trusted() turns that into silent heap corruption rather
            // than an error. (The two happen to be equal at c=32/reduction=4, which
            // is precisely why this must be derived, not assumed.)
            d_gap: ctx.act(t0_out),
            d_sm: ctx.act(shape.n * hw),
            d_mask: ctx.act(shape.n * hw),
            d_xg: ctx.act(n),
            d_xs: ctx.act(n),
            acc: ctx.act(n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn set_eval(&self, on: bool) {
        self.score.set_eval(on);
        self.t0.set_eval(on);
        self.t1.set_eval(on);
    }
    pub fn set_update_running(&self, on: bool) {
        self.t0.set_update_running(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.score.param_list();
        v.extend(self.t0.param_list());
        v.extend(self.t1.param_list());
        v
    }
    fn gap_params(&self) -> Vec<u32> {
        vec![self.shape.n, self.shape.c, self.shape.h * self.shape.w]
    }
}

impl GlobalContextBlock {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let n = self.shape.numel();
        let hw = self.shape.h * self.shape.w;
        self.score.forward(ctx, ps, x_in);
        // softmax over the map: softmax_k at M=1 (a stride of 1 over K=H*W).
        let s = ctx.step(
            ctx.ids.need(ctx.ids.softmax_k, "softmax_k"),
            &[self.score.out(), &self.sm],
            &[self.shape.n, hw, 1],
            self.shape.n,
        );
        ctx.gpu.submit(&[], &[s]);
        // The bmm: contract x against the softmax'd weights, per image.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.weighted_gap, "weighted_gap"),
            &[x_in, &self.sm, &self.gap],
            &self.gap_params(),
            self.shape.n * self.shape.c,
        );
        ctx.gpu.submit(&[], &[s]);
        self.t0.forward(ctx, ps, &self.gap);
        self.t1.forward(ctx, ps, self.t0.out());
        // The residual: x + context, broadcasting the per-(image,channel) scalar.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.add_chan_bcast, "add_chan_bcast"),
            &[x_in, self.t1.out(), &self.out],
            &self.gap_params(),
            n,
        );
        ctx.gpu.submit(&[], &[s]);
    }

    /// `x` has THREE routes: the score conv, the contraction, and the residual.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let n = self.shape.numel();
        let hw = self.shape.h * self.shape.w;
        // The residual's adjoint: d_out passes to x unchanged, and reduces over
        // space into the context's grad.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.add_chan_bcast_dv, "add_chan_bcast_dv"),
            &[d_out, &self.d_t],
            &self.gap_params(),
            self.shape.n * self.shape.c,
        );
        ctx.gpu.submit(&[], &[s]);
        self.t1.backward(ctx, ps, self.t0.out(), &self.d_t, &self.d_gap);
        // reuse d_t as the t0-input grad scratch ([N,C], same size as d_gap).
        self.t0.backward(ctx, ps, &self.gap, &self.d_gap, &self.d_t);
        // weighted_gap's two adjoints: wrt x (route 2) and wrt the mask.
        let s_dx = ctx.step(
            ctx.ids.need(ctx.ids.weighted_gap_dx, "weighted_gap_dx"),
            &[&self.d_t, &self.sm, &self.d_xg],
            &self.gap_params(),
            n,
        );
        let s_dm = ctx.step(
            ctx.ids.need(ctx.ids.weighted_gap_dm, "weighted_gap_dm"),
            &[&self.d_t, x_in, &self.d_sm],
            &self.gap_params(),
            self.shape.n * hw,
        );
        ctx.gpu.submit(&[], &[s_dx, s_dm]);
        // One invocation per softmax GROUP (the sum_j term couples all K), not per
        // element: total = N*M = N*1.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.softmax_k_dx, "softmax_k_dx"),
            &[&self.sm, &self.d_sm, &self.d_mask],
            &[self.shape.n, hw, 1],
            self.shape.n,
        );
        ctx.gpu.submit(&[], &[s]);
        // Route 3: back through the score conv.
        self.score.backward(ctx, ps, x_in, &self.d_mask, &self.d_xs);
        let s = ctx.step(ctx.ids.add2, &[&self.d_xg, &self.d_xs, &self.acc], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        // ...plus the residual's own straight-through path.
        let s = ctx.step(ctx.ids.add2, &[&self.acc, d_out, d_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }
}
