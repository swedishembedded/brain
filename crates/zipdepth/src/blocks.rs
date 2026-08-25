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
    stride: u32,
    b3: Conv,
    b1: Conv,
    sum: DeviceBuffer,   // branch_3x3 + branch_1x1 [+ x]  (pre-activation)
    act: DeviceBuffer,   // relu(sum) — the block output
    d_sum: DeviceBuffer, // grad wrt sum
    d_b1: DeviceBuffer,  // grad wrt x from the 1x1 branch
    acc: DeviceBuffer,   // out-of-place accumulator for the multi-consumer x grad
    /// Eval mode: the whole block runs as ONE fused dispatch (see `forward`).
    eval: std::cell::Cell<bool>,
    /// The RepVGG-collapsed `(weight, scale|bias)` device tensors, built lazily
    /// on the first eval forward from the ParamStore (host-side `fuse_qarep`)
    /// and invalidated by `set_eval(false)` — training changes the weights.
    fused: std::cell::RefCell<Option<(DeviceBuffer, DeviceBuffer)>>,
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
            stride,
            b3,
            b1,
            sum: ctx.act(on),
            act: ctx.act(on),
            d_sum: ctx.act(on),
            d_b1: ctx.act(in_shape.numel()),
            acc: ctx.act(in_shape.numel()),
            eval: std::cell::Cell::new(false),
            fused: std::cell::RefCell::new(None),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.act
    }
    pub fn set_eval(&self, on: bool) {
        self.eval.set(on);
        if !on {
            // Leaving eval invalidates the collapse: training will move the
            // branch weights the fused tensors were derived from.
            *self.fused.borrow_mut() = None;
        }
        self.b3.set_eval(on);
        self.b1.set_eval(on);
    }

    /// Whether eval runs the RepVGG-collapsed single dispatch. Needs the fused
    /// kernel in the registry, and steps aside when a calibration tap is
    /// installed — the tap observes the two BRANCH convs' inputs, which the
    /// collapsed form no longer has.
    pub fn eval_fused(&self, ctx: &Ctx) -> bool {
        self.eval.get() && ctx.tap.is_none() && ctx.ids.conv_act_reg != vision::NONE
    }

    /// Build the collapsed `(kernel, scale|bias)` on device: read the ten
    /// branch tensors, run the (verified) host-side [`crate::fuse_qarep`], and
    /// upload. `scale` is 1 — the BN folding already happened inside the fuse —
    /// so the fused conv is dispatched through `conv_act_reg` as
    /// `relu(conv(x, k) * 1 + bias)`, one dispatch for the whole block.
    fn ensure_fused(&self, ctx: &Ctx, ps: &ParamStore) {
        if self.fused.borrow().is_some() {
            return;
        }
        let cin = self.in_shape.c as usize;
        let cout = self.out_shape.c as usize;
        let rd = |name: &str, len: usize| ctx.gpu.read(ps.w(name), len);
        let n3 = self.b3.names();
        let n1 = self.b1.names();
        let (w3, g3, be3, m3, v3) = (
            rd(&n3.weight, cout * cin * 9),
            rd(&n3.gamma, cout),
            rd(&n3.beta, cout),
            rd(&n3.run_mean, cout),
            rd(&n3.run_var, cout),
        );
        let (w1, g1, be1, m1, v1) = (
            rd(&n1.weight, cout * cin),
            rd(&n1.gamma, cout),
            rd(&n1.beta, cout),
            rd(&n1.run_mean, cout),
            rd(&n1.run_var, cout),
        );
        let br3 = crate::fuse::Branch { weight: &w3, gamma: &g3, beta: &be3, run_mean: &m3, run_var: &v3 };
        let br1 = crate::fuse::Branch { weight: &w1, gamma: &g1, beta: &be1, run_mean: &m1, run_var: &v1 };
        let (k, b) = crate::fuse::fuse_qarep(&br3, &br1, cin, cout, 1, self.has_identity);
        let sb: Vec<f32> = b.iter().flat_map(|&bias| [1.0, bias]).collect();
        let kb = ctx.gpu.storage_init("qarep.fused.w", &k);
        let sbb = ctx.gpu.storage_init("qarep.fused.sb", &sb);
        *self.fused.borrow_mut() = Some((kb, sbb));
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
        if self.eval_fused(ctx) {
            // The RepVGG collapse: relu(conv3x3(x, k_fused) + b_fused) — the
            // exact function of the three branches (crate::fuse, tested for
            // all x), as ONE register-tiled dispatch instead of ~5 dependent
            // ones (2 convs + 1-2 adds + relu). On the measured Intel Arc,
            // every dependent dispatch pays a serialization hop, so the win is
            // latency as much as arithmetic.
            self.ensure_fused(ctx, ps);
            let fused = self.fused.borrow();
            let (k, sb) = fused.as_ref().unwrap();
            let s = self.out_shape;
            let params = [
                s.n,
                self.in_shape.c,
                self.in_shape.h,
                self.in_shape.w,
                s.c,
                3,
                self.stride,
                1,
                s.h,
                s.w,
                1, // act = relu
            ];
            let threads = s.n * s.c.div_ceil(8) * (s.h * s.w).div_ceil(4);
            let step = ctx.step(ctx.ids.conv_act_reg, &[x_in, k, sb, &self.act], &params, threads);
            ctx.gpu.submit(&[], &[step]);
            return;
        }
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
        let s_dw = ctx.step(ctx.ids.need(ctx.ids.conv2d_dw, "conv2d_dw"), &[&self.d_g, &self.h_act, ps.g(&p("fc.2.weight"))], &p2, c * self.hidden);
        let s_dx = ctx.step(ctx.ids.need(ctx.ids.conv2d_dx, "conv2d_dx"), &[&self.d_g, ps.w(&p("fc.2.weight")), &self.d_h_act], &p2, n * self.hidden);
        ctx.gpu.submit(&[], &[s_dw, s_dx]);
        let nh = n * self.hidden;
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"), &[&self.h, &self.d_h_act, &self.d_h], &[nh, f(0.0)], nh);
        ctx.gpu.submit(&[], &[s]);
        // fc.0: [c -> hidden] at 1x1
        let p0 = [n, c, 1, 1, self.hidden, 1, 1, 0, 1, 1];
        let s_dw = ctx.step(ctx.ids.need(ctx.ids.conv2d_dw, "conv2d_dw"), &[&self.d_h, &self.pooled, ps.g(&p("fc.0.weight"))], &p0, self.hidden * c);
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
///
/// Both are in the checkpoint and must be loaded, and both are carried faithfully.
/// `gcb_two_biases_are_provably_dead` pins the invariance (measured: the loss moves
/// by EXACTLY 0.0f32 when either is shifted by +-0.5, and by 2 ULP at +-5.0, where
/// a live parameter would move it by ~850). That is also why neither can be
/// finite-difference-checked: their FD is entirely round-off noise.
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

// ===========================================================================
// LightweightSPPF — IS vision::SPPF, at a different width/act/name convention
// ===========================================================================

/// ZipDepth's `LightweightSPPF` (`architecture.py:322-337`).
///
/// Not a new block: it is [`vision::SPPF`] exactly — cv1 narrows, three K=5/pad=2
/// max-pools chain off it, the 4-way concat feeds cv2. Only three things differ
/// from Ultralytics', and all three are configuration:
///   * the pooled width is `c1/4` (from the INPUT channels) rather than `c_out/2`
///     (from the output) — the reason `SppfSpec` states `hidden` instead of
///     deriving it,
///   * ReLU rather than SiLU,
///   * torch `ConvBN` names (`cv1.bn.weight`) rather than brain's (`cv1.bn.gamma`).
///
/// The pool chain, the concat fold and the whole backward are shared verbatim.
pub fn lightweight_sppf(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, train: bool) -> vision::SPPF {
    let spec = vision::SppfSpec { hidden: in_shape.c / 4, c_out, act: Act::Relu };
    vision::SPPF::with_spec(ctx, prefix, in_shape, spec, vision::NameStyle::TorchConvBn, train)
}

// ===========================================================================
// MinimalCrossScale — bidirectional 0.3-weighted exchange between two scales
// ===========================================================================

/// `(x_high + 0.3*up(l2h(x_low)), x_low + 0.3*pool(h2l(x_high)))`
/// (`architecture.py:302-315`).
///
/// The only block with TWO inputs AND TWO outputs. Both projections are bias-free
/// 1x1 grouped convs with no BN and no activation; the group counts come from the
/// reference's own `_pick_groups(in, out, 4)` rule.
///
/// ⚠️ The order is PROJECT-then-RESAMPLE here, and RESAMPLE-then-PROJECT in
/// [`UltraLightFusion`]. Both are 1x1 convs, so the two orders differ only in
/// arithmetic cost — but they are not the same graph, and the checkpoint's shapes
/// pin which is which.
///
/// The `* 0.3` fuses into `axpy` (`out += s*in`), so the residual costs one
/// dispatch and no new kernel. The low->high resample is NEAREST, the high->low is
/// `adaptive_avg_pool2d` — i.e. [`avgpool2d`], whose adaptive rule this is.
pub struct MinimalCrossScale {
    pub high: Shape,
    pub low: Shape,
    l2h: Conv,
    h2l: Conv,
    up: DeviceBuffer,
    down: DeviceBuffer,
    out_high: DeviceBuffer,
    out_low: DeviceBuffer,
    d_up: DeviceBuffer,
    d_down: DeviceBuffer,
    d_l2h: DeviceBuffer,
    d_h2l: DeviceBuffer,
    acc_h: DeviceBuffer,
    acc_l: DeviceBuffer,
}

/// The reference's residual weight (`architecture.py:314`).
const CROSS_SCALE_W: f32 = 0.3;

impl MinimalCrossScale {
    pub fn new(ctx: &Ctx, prefix: &str, high: Shape, low: Shape, train: bool) -> MinimalCrossScale {
        let raw = |cout: u32, groups: u32| {
            ConvSpec::relu(cout, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_groups(groups)
        };
        let names = |n: &str| ConvNames {
            bias: String::new(),
            weight: format!("{prefix}.{n}.weight"),
            gamma: String::new(),
            beta: String::new(),
            run_mean: String::new(),
            run_var: String::new(),
        };
        // _pick_groups(in_ch, out_ch, 4) — note the argument order follows the
        // DIRECTION of each projection, not a fixed (high, low) pair.
        let g_h = crate::config::pick_groups(low.c, high.c, 4);
        let g_l = crate::config::pick_groups(high.c, low.c, 4);
        let l2h = Conv::with_names(ctx, &format!("{prefix}.low_to_high"), names("low_to_high"), low, raw(high.c, g_h), train);
        let h2l = Conv::with_names(ctx, &format!("{prefix}.high_to_low"), names("high_to_low"), high, raw(low.c, g_l), train);
        MinimalCrossScale {
            high,
            low,
            l2h,
            h2l,
            up: ctx.act(high.numel()),
            down: ctx.act(low.numel()),
            out_high: ctx.act(high.numel()),
            out_low: ctx.act(low.numel()),
            d_up: ctx.act(high.numel()),
            d_down: ctx.act(low.numel()),
            // The projections' OWN output shapes: l2h emits [N, high.c, low.h, low.w]
            // (project first, resample after), h2l emits [N, low.c, high.h, high.w].
            d_l2h: ctx.act(low.n * high.c * low.h * low.w),
            d_h2l: ctx.act(high.n * low.c * high.h * high.w),
            // Per-scale accumulators. NOT `d_l2h`/`d_h2l` reused as scratch: those
            // are sized to the PROJECTIONS' outputs (`[N, high.c, low.h, low.w]` and
            // `[N, low.c, high.h, high.w]`), which are neither scale's own numel.
            acc_h: ctx.act(high.numel()),
            acc_l: ctx.act(low.numel()),
        }
    }

    pub fn out_high(&self) -> &DeviceBuffer {
        &self.out_high
    }
    pub fn out_low(&self) -> &DeviceBuffer {
        &self.out_low
    }
    pub fn set_eval(&self, on: bool) {
        self.l2h.set_eval(on);
        self.h2l.set_eval(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.l2h.param_list();
        v.extend(self.h2l.param_list());
        v
    }
    /// resize/pool ABI: `[N, C, H, W, Ho, Wo]`.
    fn rs(&self, c: u32, from: Shape, to: Shape) -> Vec<u32> {
        vec![from.n, c, from.h, from.w, to.h, to.w]
    }
}

impl MinimalCrossScale {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_high: &DeviceBuffer, x_low: &DeviceBuffer) {
        // low -> high: project at the LOW resolution, then upsample (nearest).
        self.l2h.forward(ctx, ps, x_low);
        let s = ctx.step(
            ctx.ids.need(ctx.ids.resize_nearest, "resize_nearest"),
            &[self.l2h.out(), &self.up],
            &self.rs(self.high.c, self.low, self.high),
            self.high.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
        // high -> low: project at the HIGH resolution, then adaptive-avg-pool down.
        self.h2l.forward(ctx, ps, x_high);
        let s = ctx.step(
            ctx.ids.need(ctx.ids.avgpool2d, "avgpool2d"),
            &[self.h2l.out(), &self.down],
            &self.rs(self.low.c, self.high, self.low),
            self.low.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
        // x + 0.3*delta, as copy-then-axpy. `axpy` is read-modify-write, so the
        // copy is what keeps this SSA.
        self.axpy_residual(ctx, x_high, &self.up, &self.out_high, self.high.numel());
        self.axpy_residual(ctx, x_low, &self.down, &self.out_low, self.low.numel());
    }

    /// `out = x + 0.3*delta`. leaky_relu at slope 1 is brain's aliased copy.
    fn axpy_residual(&self, ctx: &Ctx, x: &DeviceBuffer, delta: &DeviceBuffer, out: &DeviceBuffer, n: u32) {
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[x, out], &[n, f(1.0)], n);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[out, delta], &[n, f(CROSS_SCALE_W)], n);
        ctx.gpu.submit(&[], &[s]);
    }

    /// Both outputs carry gradients, so this takes both. `x_high` reaches BOTH
    /// outputs (its own residual and the high->low projection), likewise `x_low`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_high: &DeviceBuffer,
        x_low: &DeviceBuffer,
        d_high: &DeviceBuffer,
        d_low: &DeviceBuffer,
        d_in_high: &DeviceBuffer,
        d_in_low: &DeviceBuffer,
    ) {
        let (nh, nl) = (self.high.numel(), self.low.numel());
        // The 0.3-scaled residual's adjoint: d_delta = 0.3 * d_out. Clearing via
        // submit's clear-list then axpy gives the scale without a second kernel.
        let s = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.d_up, d_high], &[nh, f(CROSS_SCALE_W)], nh);
        ctx.gpu.submit(&[&self.d_up], &[s]);
        let s = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.d_down, d_low], &[nl, f(CROSS_SCALE_W)], nl);
        ctx.gpu.submit(&[&self.d_down], &[s]);

        // Back through the nearest upsample, then the low->high projection: this is
        // x_low's SECOND route (the first is its own residual).
        let s = ctx.step(
            ctx.ids.need(ctx.ids.resize_nearest_dx, "resize_nearest_dx"),
            &[&self.d_up, &self.d_l2h],
            &self.rs(self.high.c, self.low, self.high),
            self.low.n * self.high.c * self.low.h * self.low.w,
        );
        ctx.gpu.submit(&[], &[s]);
        self.l2h.backward(ctx, ps, x_low, &self.d_l2h, d_in_low);

        // Back through the pool, then the high->low projection: x_high's second route.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.avgpool2d_dx, "avgpool2d_dx"),
            &[&self.d_down, &self.d_h2l],
            &self.rs(self.low.c, self.high, self.low),
            self.high.n * self.low.c * self.high.h * self.high.w,
        );
        ctx.gpu.submit(&[], &[s]);
        self.h2l.backward(ctx, ps, x_high, &self.d_h2l, &self.acc_h);

        // ...plus each input's own straight-through residual (weight 1, not 0.3).
        let s = ctx.step(ctx.ids.add2, &[&self.acc_h, d_high, d_in_high], &[nh], nh);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.add2, &[d_in_low, d_low, &self.acc_l], &[nl], nl);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.acc_l, d_in_low], &[nl, f(1.0)], nl);
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// UltraLightFusion — the decoder's two-scale merge
// ===========================================================================

/// `relu(BN(proj_high(x_high) + proj_low(up_bilinear(x_low))))`
/// (`architecture.py:345-359`).
///
/// One BN over the SUM of two projections — the same shape as
/// [`MinimalMultiScale`], so the same `vision::BatchNorm` + `Norm::None` branches
/// apply. Both projections are bias-free 1x1 grouped convs.
///
/// ⚠️ RESAMPLE-then-PROJECT, the opposite order to [`MinimalCrossScale`]'s: the
/// reference rebinds `x_low = F.interpolate(x_low, ...)` BEFORE `self.proj_low(x_low)`.
/// The upsample is bilinear with `align_corners=False`.
pub struct UltraLightFusion {
    pub high: Shape,
    pub low: Shape,
    pub out_shape: Shape,
    proj_high: Conv,
    proj_low: Conv,
    bn: vision::BatchNorm,
    up: DeviceBuffer,
    sum: DeviceBuffer,
    out: DeviceBuffer,
    d_sum: DeviceBuffer,
    d_pl: DeviceBuffer,
    acc: DeviceBuffer,
    /// Eval fuses the trailing ReLU into `bn_eval`'s act selector; `out()`
    /// then aliases the BN's own output (one dispatch + one memory pass fewer).
    eval: std::cell::Cell<bool>,
}

impl UltraLightFusion {
    pub fn new(ctx: &Ctx, prefix: &str, high: Shape, low: Shape, out_ch: u32, train: bool) -> UltraLightFusion {
        let raw = |cout: u32, groups: u32| {
            ConvSpec::relu(cout, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_groups(groups)
        };
        let names = |n: &str| ConvNames {
            bias: String::new(),
            weight: format!("{prefix}.{n}.weight"),
            gamma: String::new(),
            beta: String::new(),
            run_mean: String::new(),
            run_var: String::new(),
        };
        let g_high = crate::config::pick_groups(high.c, out_ch, 4);
        let g_low = crate::config::pick_groups(low.c, out_ch, 4);
        // proj_low runs on the UPSAMPLED low map: low's channels at high's geometry.
        let up_shape = Shape { c: low.c, ..high };
        let proj_high = Conv::with_names(ctx, &format!("{prefix}.proj_high"), names("proj_high"), high, raw(out_ch, g_high), train);
        let proj_low = Conv::with_names(ctx, &format!("{prefix}.proj_low"), names("proj_low"), up_shape, raw(out_ch, g_low), train);
        let out_shape = proj_high.out_shape;
        assert_eq!(out_shape, proj_low.out_shape, "both projections must land on the same map");
        let bn = vision::BatchNorm::new(ctx, vision::BnNames::torch(&format!("{prefix}.bn")), out_shape, train);
        let n = out_shape.numel();
        UltraLightFusion {
            high,
            low,
            out_shape,
            proj_high,
            proj_low,
            bn,
            up: ctx.act(up_shape.numel()),
            sum: ctx.act(n),
            out: ctx.act(n),
            d_sum: ctx.act(n),
            d_pl: ctx.act(up_shape.numel()),
            acc: ctx.act(n),
            eval: std::cell::Cell::new(false),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        if self.eval.get() {
            self.bn.out() // relu fused into bn_eval; the BN output IS the block output
        } else {
            &self.out
        }
    }
    pub fn set_eval(&self, on: bool) {
        self.eval.set(on);
        self.proj_high.set_eval(on);
        self.proj_low.set_eval(on);
        self.bn.set_eval(on);
    }
    pub fn set_update_running(&self, on: bool) {
        self.bn.set_update_running(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.proj_high.param_list();
        v.extend(self.proj_low.param_list());
        v.extend(self.bn.param_list());
        v
    }
    /// resize_bilinear ABI: `[N, C, H, W, Ho, Wo, align_corners]`.
    fn rs(&self) -> Vec<u32> {
        vec![self.low.n, self.low.c, self.low.h, self.low.w, self.high.h, self.high.w, 0]
    }
    fn up_n(&self) -> u32 {
        self.low.n * self.low.c * self.high.h * self.high.w
    }
}

impl UltraLightFusion {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_high: &DeviceBuffer, x_low: &DeviceBuffer) {
        let n = self.out_shape.numel();
        // Resample FIRST (align_corners=False), then project.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.resize_bilinear, "resize_bilinear"),
            &[x_low, &self.up],
            &self.rs(),
            self.up_n(),
        );
        ctx.gpu.submit(&[], &[s]);
        self.proj_high.forward(ctx, ps, x_high);
        self.proj_low.forward(ctx, ps, &self.up);
        let s = ctx.step(ctx.ids.add2, &[self.proj_high.out(), self.proj_low.out(), &self.sum], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        // Eval: the ReLU rides in bn_eval's act selector and `out()` aliases
        // the BN output. Train keeps the separate ReLU — its backward needs
        // the pre-activation BN output as a cache.
        if !self.bn.forward_act(ctx, ps, &self.sum, 1) {
            let s = ctx.step(
                ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"),
                &[self.bn.out(), &self.out],
                &[n, f(0.0)],
                n,
            );
            ctx.gpu.submit(&[], &[s]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_high: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in_high: &DeviceBuffer,
        d_in_low: &DeviceBuffer,
    ) {
        let n = self.out_shape.numel();
        let s = ctx.step(
            ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"),
            &[self.bn.out(), d_out, &self.acc],
            &[n, f(0.0)],
            n,
        );
        ctx.gpu.submit(&[], &[s]);
        self.bn.backward(ctx, ps, &self.sum, &self.acc, &self.d_sum);
        // The sum is linear -> both projections see d_sum.
        self.proj_high.backward(ctx, ps, x_high, &self.d_sum, d_in_high);
        self.proj_low.backward(ctx, ps, &self.up, &self.d_sum, &self.d_pl);
        // ...and back through the bilinear upsample to x_low.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.resize_bilinear_dx, "resize_bilinear_dx"),
            &[&self.d_pl, d_in_low],
            &self.rs(),
            self.low.numel(),
        );
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// FastConvexUpsample — the decoder's learned S-times upsample, two variants
// ===========================================================================

/// Which upsampler ZipDepth's decoder ends with (`architecture.py:367-430`).
///
/// The reference calls these the GPU/TensorRT path and the NPU path, and they are
/// NOT two implementations of one function — they are different architectures with
/// different parameters, and the two released checkpoints differ by exactly this
/// (`zipdepth_base.pth` has `mask_pred.*`, `zipdepth_base_npu.pth` has
/// `where_conv.*`; 278 vs 283 tensors). Picking the wrong one fails a strict load.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UpsampleKind {
    /// `use_unfold=True`: predict a 3x3 convex-combination mask per sub-pixel.
    /// Learns arbitrary local interpolation; needs the 9-way softmax and the
    /// `unfold`-shaped gather.
    Unfold,
    /// `use_unfold=False`: predict a scalar per pixel and blend nearest against
    /// bilinear. Cheap and NPU-friendly, but can only interpolate BETWEEN those
    /// two, never outside them.
    Blend,
}

/// `relu(pixel_shuffle(sum_k softmax(mask)_k * unfold(pad_replicate(depth))_k))`
/// (`architecture.py:399-424`), or the nearest/bilinear blend (`:388-397`).
///
/// Takes TWO inputs: the decoder feature map (which predicts *how* to upsample)
/// and the half-resolution depth (*what* to upsample), both at `[N, ., H, W]`.
/// Emits `[N, 1, H*S, W*S]`.
///
/// The `Unfold` path needs no `pad`/`unfold`/`pixel_shuffle` dispatches at all:
/// `convex_upsample` folds the replicate-pad, the 9-way gather and the shuffle
/// into one kernel indexed by the output pixel, so the `[N,9,S*S,H,W]`
/// intermediate never exists. Its softmax is `softmax_k` over the strided 9 axis
/// (`K=9, M=S*S*H*W`) — the shape that kernel was written for.
pub struct FastConvexUpsample {
    pub kind: UpsampleKind,
    pub feat: Shape,
    pub depth: Shape,
    pub out_shape: Shape,
    scale: u32,
    temperature: f32,
    /// Unfold: mask_pred.0 / .3. Blend: where_conv.0 / .3 / .6.
    c0: Conv,
    c1: Conv,
    c2: Option<Conv>,
    // Unfold path
    sm: DeviceBuffer,
    d_sm: DeviceBuffer,
    d_logits: DeviceBuffer,
    // Blend path
    nn_up: DeviceBuffer,
    bi_up: DeviceBuffer,
    alpha_up: DeviceBuffer,
    alpha: DeviceBuffer,
    t1: DeviceBuffer,
    t2: DeviceBuffer,
    d_alpha: DeviceBuffer,
    d_alpha_lo: DeviceBuffer,
    d_nn: DeviceBuffer,
    d_bi: DeviceBuffer,
    /// The two `[N, where_hidden, H, W]` links of the where_conv chain, plus the
    /// two half-res depth-grad partials. Sized from the convs' OWN out_shapes.
    d_c1: DeviceBuffer,
    d_c0: DeviceBuffer,
    dd1: DeviceBuffer,
    dd2: DeviceBuffer,
    // Shared
    pre: DeviceBuffer,
    out: DeviceBuffer,
    d_pre: DeviceBuffer,
    acc: DeviceBuffer,
}

impl FastConvexUpsample {
    pub fn new(
        ctx: &Ctx,
        prefix: &str,
        kind: UpsampleKind,
        feat: Shape,
        depth: Shape,
        scale: u32,
        temperature: f32,
        train: bool,
    ) -> FastConvexUpsample {
        assert_eq!(depth.c, 1, "the depth map is single-channel");
        assert_eq!((feat.h, feat.w), (depth.h, depth.w), "feat and depth share the half-res grid");
        let s = scale;
        let out_shape = Shape::new(depth.n, 1, depth.h * s, depth.w * s);
        let bare = |p: String| ConvNames {
            bias: format!("{p}.bias"),
            weight: format!("{p}.weight"),
            gamma: String::new(),
            beta: String::new(),
            run_mean: String::new(),
            run_var: String::new(),
        };
        let (c0, c1, c2) = match kind {
            UpsampleKind::Unfold => {
                // Sequential(Conv2d(in,hidden,3,padding=1,bias=False), BN, ReLU,
                //            Conv2d(hidden, 9*S*S, 1))   -> indices 0, 1, 3
                let hidden = (feat.c / 4).max(8);
                let p = format!("{prefix}.mask_pred");
                let a = Conv::with_names(
                    ctx,
                    &format!("{p}.0"),
                    ConvNames::torch_seq(&p, 0, 1),
                    feat,
                    ConvSpec::relu(hidden, 3, 1, 1),
                    train,
                );
                // The last 1x1 is BIASED (nn.Conv2d's default) and has no BN/act.
                let b = Conv::with_names(
                    ctx,
                    &format!("{p}.3"),
                    bare(format!("{p}.3")),
                    a.out_shape,
                    ConvSpec::relu(9 * s * s, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
                    train,
                );
                (a, b, None)
            }
            UpsampleKind::Blend => {
                // Sequential(Conv2d(in,wh,1,bias=False), BN, ReLU,
                //            Conv2d(wh,wh,5,padding=2,groups=wh,bias=False), BN, ReLU,
                //            Conv2d(wh,1,1,bias=False))  -> indices 0, 3, 6
                let wh = (feat.c / 2).max(8);
                let p = format!("{prefix}.where_conv");
                let a = Conv::with_names(
                    ctx,
                    &format!("{p}.0"),
                    ConvNames::torch_seq(&p, 0, 1),
                    feat,
                    ConvSpec::relu(wh, 1, 1, 0),
                    train,
                );
                let b = Conv::with_names(
                    ctx,
                    &format!("{p}.3"),
                    ConvNames::torch_seq(&p, 3, 4),
                    a.out_shape,
                    ConvSpec::depthwise(wh, 5, 1, 2, Act::Relu),
                    train,
                );
                let c = Conv::with_names(
                    ctx,
                    &format!("{p}.6"),
                    ConvNames {
                        bias: String::new(),
                        weight: format!("{p}.6.weight"),
                        gamma: String::new(),
                        beta: String::new(),
                        run_mean: String::new(),
                        run_var: String::new(),
                    },
                    b.out_shape,
                    ConvSpec::relu(1, 1, 1, 0).with_norm(Norm::None).with_act(Act::None),
                    train,
                );
                (a, b, Some(c))
            }
        };
        let mask_n = depth.n * 9 * s * s * depth.h * depth.w;
        let on = out_shape.numel();
        let dn = depth.numel();
        // From the producing units' own shapes — never re-derived. (The GCB bug:
        // `hidden` can exceed the nominal channel count, and the CPU JIT turns an
        // undersized buffer into silent heap corruption.)
        let (c0_out, c1_out) = (c0.out_shape.numel(), c1.out_shape.numel());
        FastConvexUpsample {
            kind,
            feat,
            depth,
            out_shape,
            scale: s,
            temperature,
            c0,
            c1,
            c2,
            sm: ctx.act(if kind == UpsampleKind::Unfold { mask_n } else { 1 }),
            d_sm: ctx.act(if kind == UpsampleKind::Unfold { mask_n } else { 1 }),
            d_logits: ctx.act(if kind == UpsampleKind::Unfold { mask_n } else { 1 }),
            nn_up: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            bi_up: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            alpha_up: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            alpha: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            t1: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            t2: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            d_alpha: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            d_alpha_lo: ctx.act(if kind == UpsampleKind::Blend { dn } else { 1 }),
            d_nn: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            d_bi: ctx.act(if kind == UpsampleKind::Blend { on } else { 1 }),
            d_c1: ctx.act(if kind == UpsampleKind::Blend { c1_out } else { 1 }),
            // Both kinds need this: it is the grad wrt c0's OUTPUT, i.e. c1's input.
            // NOT `acc` (sized to the block's output) — c0's output lives on the
            // half-res grid at `hidden` channels and is a different size entirely.
            d_c0: ctx.act(c0_out),
            dd1: ctx.act(if kind == UpsampleKind::Blend { dn } else { 1 }),
            dd2: ctx.act(if kind == UpsampleKind::Blend { dn } else { 1 }),
            pre: ctx.act(on),
            out: ctx.act(on),
            d_pre: ctx.act(on),
            acc: ctx.act(on),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn set_eval(&self, on: bool) {
        self.c0.set_eval(on);
        self.c1.set_eval(on);
        if let Some(c) = &self.c2 {
            c.set_eval(on);
        }
    }
    pub fn set_update_running(&self, on: bool) {
        self.c0.set_update_running(on);
        self.c1.set_update_running(on);
        if let Some(c) = &self.c2 {
            c.set_update_running(on);
        }
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.c0.param_list();
        v.extend(self.c1.param_list());
        if let Some(c) = &self.c2 {
            v.extend(c.param_list());
        }
        v
    }
    /// convex_upsample ABI: `[N, H, W, S]` (H/W are the HALF-res grid).
    fn cu(&self) -> Vec<u32> {
        vec![self.depth.n, self.depth.h, self.depth.w, self.scale]
    }
    /// softmax_k over the 9-neighbour axis: `[N, K=9, M=S*S*H*W]`.
    fn sk(&self) -> Vec<u32> {
        vec![self.depth.n, 9, self.scale * self.scale * self.depth.h * self.depth.w]
    }
    /// Params for the resize kernels of the Blend upsampler:
    /// `[N, C, H, W, Ho, Wo]`, plus an optional 7th `align_corners` word.
    ///
    /// `align_word` is `Some(0)` for `resize_bilinear` / `resize_bicubic`
    /// (which read a 7th word; `0` means `align_corners = FALSE`, i.e. the
    /// half-pixel rule the ZipDepth reference uses) and `None` for
    /// `resize_nearest`, which reads only six and must not be handed a seventh.
    ///
    /// This was a `bool align` whose `true` branch pushed `0` — it read as
    /// "align_corners = true" at every call site and meant the opposite. A
    /// mismatched param list is silently wrong rather than a crash,
    /// so the parameter now carries the word it
    /// actually pushes instead of a name that inverts it.
    fn up_params(&self, c: u32, align_word: Option<u32>) -> Vec<u32> {
        let mut v = vec![self.depth.n, c, self.depth.h, self.depth.w, self.out_shape.h, self.out_shape.w];
        if let Some(w) = align_word {
            v.push(w);
        }
        v
    }
}

impl FastConvexUpsample {
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, feat: &DeviceBuffer, depth: &DeviceBuffer) {
        let on = self.out_shape.numel();
        match self.kind {
            UpsampleKind::Unfold => {
                self.c0.forward(ctx, ps, feat);
                self.c1.forward(ctx, ps, self.c0.out());
                // softmax(logits / T) over the 9 axis. At T == 1 the divide is the
                // identity and is skipped rather than dispatched.
                let logits = if (self.temperature - 1.0).abs() < f32::EPSILON {
                    self.c1.out().clone()
                } else {
                    let n = self.depth.n * 9 * self.scale * self.scale * self.depth.h * self.depth.w;
                    let s = ctx.step(
                        ctx.ids.need(ctx.ids.axpy, "axpy"),
                        &[&self.d_sm, self.c1.out()],
                        &[n, f(1.0 / self.temperature)],
                        n,
                    );
                    ctx.gpu.submit(&[&self.d_sm], &[s]);
                    self.d_sm.clone()
                };
                let groups = self.depth.n * self.scale * self.scale * self.depth.h * self.depth.w;
                let s = ctx.step(ctx.ids.need(ctx.ids.softmax_k, "softmax_k"), &[&logits, &self.sm], &self.sk(), groups);
                ctx.gpu.submit(&[], &[s]);
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.convex_upsample, "convex_upsample"),
                    &[&self.sm, depth, &self.pre],
                    &self.cu(),
                    on,
                );
                ctx.gpu.submit(&[], &[s]);
            }
            UpsampleKind::Blend => {
                let s_nn = ctx.step(
                    ctx.ids.need(ctx.ids.resize_nearest, "resize_nearest"),
                    &[depth, &self.nn_up],
                    &self.up_params(1, None),
                    on,
                );
                let s_bi = ctx.step(
                    ctx.ids.need(ctx.ids.resize_bilinear, "resize_bilinear"),
                    &[depth, &self.bi_up],
                    &self.up_params(1, Some(0)),
                    on,
                );
                ctx.gpu.submit(&[], &[s_nn, s_bi]);
                self.c0.forward(ctx, ps, feat);
                self.c1.forward(ctx, ps, self.c0.out());
                let c2 = self.c2.as_ref().expect("Blend has a third conv");
                c2.forward(ctx, ps, self.c1.out());
                // The reference upsamples alpha BEFORE the sigmoid.
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.resize_bilinear, "resize_bilinear"),
                    &[c2.out(), &self.alpha_up],
                    &self.up_params(1, Some(0)),
                    on,
                );
                ctx.gpu.submit(&[], &[s]);
                let s = ctx.step(ctx.ids.need(ctx.ids.sigmoid, "sigmoid"), &[&self.alpha_up, &self.alpha], &[on], on);
                ctx.gpu.submit(&[], &[s]);
                // out = a*nn + (1-a)*bi = bi + a*nn - a*bi. axpy at s=-1 is the
                // subtract, so no sub kernel is needed.
                let cp = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.bi_up, &self.pre], &[on, f(1.0)], on);
                let m1 = ctx.step(ctx.ids.mul, &[&self.alpha, &self.nn_up, &self.t1], &[on], on);
                let m2 = ctx.step(ctx.ids.mul, &[&self.alpha, &self.bi_up, &self.t2], &[on], on);
                ctx.gpu.submit(&[], &[cp, m1, m2]);
                let a1 = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.pre, &self.t1], &[on, f(1.0)], on);
                ctx.gpu.submit(&[], &[a1]);
                let a2 = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.pre, &self.t2], &[on, f(-1.0)], on);
                ctx.gpu.submit(&[], &[a2]);
            }
        }
        // Both paths end in the same ReLU: the model's output is non-negative
        // inverse depth.
        let s = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.pre, &self.out], &[on, f(0.0)], on);
        ctx.gpu.submit(&[], &[s]);
    }

    /// `d_feat` and `d_depth_in` both receive gradients: the feature map decides
    /// HOW to upsample and the depth map WHAT, and both are learned upstream.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        feat: &DeviceBuffer,
        depth: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_feat: &DeviceBuffer,
        d_depth_in: &DeviceBuffer,
    ) {
        let on = self.out_shape.numel();
        let s = ctx.step(
            ctx.ids.need(ctx.ids.leaky_relu_bwd, "leaky_relu_bwd"),
            &[&self.pre, d_out, &self.d_pre],
            &[on, f(0.0)],
            on,
        );
        ctx.gpu.submit(&[], &[s]);
        match self.kind {
            UpsampleKind::Unfold => {
                let mask_n = self.depth.n * 9 * self.scale * self.scale * self.depth.h * self.depth.w;
                // convex_upsample is bilinear in (mask, d): two independent adjoints.
                let dm = ctx.step(
                    ctx.ids.need(ctx.ids.convex_upsample_dmask, "convex_upsample_dmask"),
                    &[&self.d_pre, depth, &self.d_sm],
                    &self.cu(),
                    mask_n,
                );
                let dd = ctx.step(
                    ctx.ids.need(ctx.ids.convex_upsample_dd, "convex_upsample_dd"),
                    &[&self.d_pre, &self.sm, d_depth_in],
                    &self.cu(),
                    self.depth.numel(),
                );
                ctx.gpu.submit(&[], &[dm, dd]);
                let groups = self.depth.n * self.scale * self.scale * self.depth.h * self.depth.w;
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.softmax_k_dx, "softmax_k_dx"),
                    &[&self.sm, &self.d_sm, &self.d_logits],
                    &self.sk(),
                    groups,
                );
                ctx.gpu.submit(&[], &[s]);
                // ...and the 1/T scale's adjoint is the same 1/T.
                if (self.temperature - 1.0).abs() >= f32::EPSILON {
                    let s = ctx.step(
                        ctx.ids.need(ctx.ids.axpy, "axpy"),
                        &[&self.d_sm, &self.d_logits],
                        &[mask_n, f(1.0 / self.temperature)],
                        mask_n,
                    );
                    ctx.gpu.submit(&[&self.d_sm], &[s]);
                    self.c1.backward(ctx, ps, self.c0.out(), &self.d_sm, &self.d_c0);
                } else {
                    self.c1.backward(ctx, ps, self.c0.out(), &self.d_logits, &self.d_c0);
                }
                self.c0.backward(ctx, ps, feat, &self.d_c0, d_feat);
            }
            UpsampleKind::Blend => {
                // pre = bi + a*nn - a*bi
                //   d_a  = d_pre * (nn - bi)
                //   d_nn = d_pre * a
                //   d_bi = d_pre * (1 - a)
                let dn = ctx.step(ctx.ids.mul, &[&self.d_pre, &self.alpha, &self.d_nn], &[on], on);
                ctx.gpu.submit(&[], &[dn]);
                // d_a = d_pre*nn - d_pre*bi, reusing t1/t2 (dead after forward).
                let m1 = ctx.step(ctx.ids.mul, &[&self.d_pre, &self.nn_up, &self.t1], &[on], on);
                let m2 = ctx.step(ctx.ids.mul, &[&self.d_pre, &self.bi_up, &self.t2], &[on], on);
                ctx.gpu.submit(&[], &[m1, m2]);
                let c1 = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.t1, &self.d_alpha], &[on, f(1.0)], on);
                ctx.gpu.submit(&[], &[c1]);
                let a1 = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.d_alpha, &self.t2], &[on, f(-1.0)], on);
                ctx.gpu.submit(&[], &[a1]);
                // d_bi = d_pre - d_pre*a  (t2 is free again after d_alpha).
                let cb = ctx.step(ctx.ids.need(ctx.ids.leaky_relu, "leaky_relu"), &[&self.d_pre, &self.d_bi], &[on, f(1.0)], on);
                ctx.gpu.submit(&[], &[cb]);
                let ab = ctx.step(ctx.ids.need(ctx.ids.axpy, "axpy"), &[&self.d_bi, &self.d_nn], &[on, f(-1.0)], on);
                ctx.gpu.submit(&[], &[ab]);

                // Back through sigmoid, then the alpha upsample, then where_conv.
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.sigmoid_bwd, "sigmoid_bwd"),
                    &[&self.alpha_up, &self.d_alpha, &self.acc],
                    &[on],
                    on,
                );
                ctx.gpu.submit(&[], &[s]);
                let s = ctx.step(
                    ctx.ids.need(ctx.ids.resize_bilinear_dx, "resize_bilinear_dx"),
                    &[&self.acc, &self.d_alpha_lo],
                    &self.up_params(1, Some(0)),
                    self.depth.numel(),
                );
                ctx.gpu.submit(&[], &[s]);
                let c2 = self.c2.as_ref().expect("Blend has a third conv");
                c2.backward(ctx, ps, self.c1.out(), &self.d_alpha_lo, &self.d_c1);
                self.c1.backward(ctx, ps, self.c0.out(), &self.d_c1, &self.d_c0);
                self.c0.backward(ctx, ps, feat, &self.d_c0, d_feat);

                // ...and `depth`'s own two routes, through each upsample.
                let s1 = ctx.step(
                    ctx.ids.need(ctx.ids.resize_nearest_dx, "resize_nearest_dx"),
                    &[&self.d_nn, &self.dd1],
                    &self.up_params(1, None),
                    self.depth.numel(),
                );
                let s2 = ctx.step(
                    ctx.ids.need(ctx.ids.resize_bilinear_dx, "resize_bilinear_dx"),
                    &[&self.d_bi, &self.dd2],
                    &self.up_params(1, Some(0)),
                    self.depth.numel(),
                );
                ctx.gpu.submit(&[], &[s1, s2]);
                let dn_ = self.depth.numel();
                let s = ctx.step(ctx.ids.add2, &[&self.dd1, &self.dd2, d_depth_in], &[dn_], dn_);
                ctx.gpu.submit(&[], &[s]);
            }
        }
    }
}
