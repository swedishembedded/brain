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
use vision::blocks::{Act, Conv, ConvNames, ConvSpec};
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
