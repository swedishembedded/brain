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
