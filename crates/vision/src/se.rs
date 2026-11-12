// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Squeeze-and-excitation (Hu, Shen and Sun, CVPR 2018): reweight every
//! channel of a feature map by a gate computed from the map's own global
//! average.
//!
//! ```text
//! x [N,C,H,W] -> global average pool [N,C,1,1]
//!             -> 1x1 conv C -> R (+bias), `inner` activation
//!             -> 1x1 conv R -> C (+bias), `gate` activation
//!             -> y = x * gate[n, c]
//! ```
//!
//! The two variants in released checkpoints differ only in the activations,
//! which is why they are data here rather than two blocks: the original uses
//! ReLU then a logistic sigmoid; MobileNetV3 uses ReLU then a HARD sigmoid
//! ([`Act::HardSigmoid`]), and the two gates disagree by up to ~0.1, far beyond
//! any parity tolerance.
//!
//! Built from the shared units ([`AvgPool`], [`Conv`]) and the `scale_chan`
//! kernel, so it adds no kernel of its own. Inference only: the block has no
//! backward yet, and a caller that needs one gets a compile error rather than a
//! silent zero gradient.

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;

use crate::blocks::{Act, AvgPool, Conv, ConvSpec, Norm};
use crate::net::{Ctx, Shape};

/// The activations and the bottleneck width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeSpec {
    /// Channels of the squeezed representation.
    pub reduced: u32,
    pub inner: Act,
    pub gate: Act,
}

impl SeSpec {
    /// MobileNetV3's gate: ReLU, then hard sigmoid.
    pub const fn mobilenet_v3(reduced: u32) -> SeSpec {
        SeSpec { reduced, inner: Act::Relu, gate: Act::HardSigmoid }
    }
}

/// One squeeze-and-excitation gate over a fixed input shape.
pub struct SqueezeExcite {
    pub shape: Shape,
    pool: AvgPool,
    reduce: Conv,
    expand: Conv,
    out: DeviceBuffer,
}

impl SqueezeExcite {
    /// The gate's two 1x1 convs are named `{prefix}.reduce` and
    /// `{prefix}.expand` (each `.conv.weight` and `.conv.bias`, the
    /// [`crate::ConvNames::brain`] spelling).
    pub fn new(ctx: &Ctx, prefix: &str, shape: Shape, spec: SeSpec) -> SqueezeExcite {
        assert!(spec.reduced > 0, "SqueezeExcite: the squeezed width must be positive");
        let pool = AvgPool::new(ctx, shape, 1, 1);
        let one_by_one = |cout, act| ConvSpec { cout, k: 1, stride: 1, pad: 0, groups: 1, dilation: 1, norm: Norm::None, act, bias: true };
        let reduce = Conv::with_spec(ctx, &format!("{prefix}.reduce"), pool.out_shape, one_by_one(spec.reduced, spec.inner), false);
        let expand = Conv::with_spec(ctx, &format!("{prefix}.expand"), reduce.out_shape, one_by_one(shape.c, spec.gate), false);
        SqueezeExcite { shape, pool, reduce, expand, out: ctx.act(shape.numel()) }
    }

    /// `(name, element count)` of every parameter the gate reads.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.reduce.param_list();
        p.extend(self.expand.param_list());
        p
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }

    /// The gate alone, `[N, C]`, valid after [`SqueezeExcite::forward`].
    pub fn gate(&self) -> &DeviceBuffer {
        self.expand.out()
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        self.pool.forward(ctx, x);
        self.reduce.forward(ctx, ps, self.pool.out());
        self.expand.forward(ctx, ps, self.reduce.out());
        let total = self.shape.numel();
        // `scale_chan` indexes its gain by `(idx / inner) % c`; with c = N*C
        // that is exactly the per-(sample, channel) gate.
        let s = ctx.step(
            ctx.ids.need(ctx.ids.scale_chan, "scale_chan"),
            &[x, self.expand.out(), &self.out],
            &[total, self.shape.n * self.shape.c, self.shape.h * self.shape.w],
            total,
        );
        ctx.gpu.submit(&[], &[s]);
    }
}
