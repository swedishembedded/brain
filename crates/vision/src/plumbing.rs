// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Neck plumbing shared by conv-net models: nearest upsample, channel concat,
//! and an out-of-place gradient accumulator — each with its backward, SSA.
//!
//! These were private to yolo's `model.rs`. They are not detection-specific:
//! any FPN-shaped decoder needs exactly this trio, and ZipDepth's decoder needs
//! all three. The FPN *topology* stays with each model — yolo's PAN-FPN is 34
//! lines of straight-line stage wiring plus a hand-written backward, and
//! ZipDepth's is a different shape — so the right granularity to share is the
//! stages, not an `Fpn` abstraction neither model would fit.

use gpu_core::DeviceBuffer;

use crate::net::{Ctx, Shape};

/// A 2x nearest-neighbour upsample stage (forward + backward), SSA.
pub struct Up {
    pub in_shape: Shape,
    pub out_shape: Shape,
    pub out: DeviceBuffer,
    pub d_in: DeviceBuffer,
}
impl Up {
    pub fn new(ctx: &Ctx, in_shape: Shape) -> Up {
        let out_shape = Shape::new(in_shape.n, in_shape.c, in_shape.h * 2, in_shape.w * 2);
        Up { in_shape, out_shape, out: ctx.act(out_shape.numel()), d_in: ctx.act(in_shape.numel()) }
    }
    pub fn params(&self) -> [u32; 4] {
        [self.in_shape.n, self.in_shape.c, self.in_shape.h, self.in_shape.w]
    }
    pub fn forward(&self, ctx: &Ctx, x: &DeviceBuffer) {
        let s = ctx.step(ctx.ids.upsample2, &[x, &self.out], &self.params(), self.out_shape.numel());
        ctx.gpu.submit(&[], &[s]);
    }
    /// `d_out` (grad wrt upsampled output) -> `self.d_in` (grad wrt input).
    pub fn backward(&self, ctx: &Ctx, d_out: &DeviceBuffer) {
        let s = ctx.step(ctx.ids.upsample2_dx, &[d_out, &self.d_in], &self.params(), self.in_shape.numel());
        ctx.gpu.submit(&[], &[s]);
    }
}

/// A channel-concat of two equal-spatial feature maps `[a | b]` (forward +
/// backward via concat_split). SSA.
pub struct Cat {
    pub n: u32,
    pub ca: u32,
    pub cb: u32,
    pub h: u32,
    pub w: u32,
    pub out: DeviceBuffer,
    pub d_a: DeviceBuffer,
    pub d_b: DeviceBuffer,
}
impl Cat {
    pub fn new(ctx: &Ctx, a: Shape, b: Shape) -> Cat {
        assert_eq!((a.n, a.h, a.w), (b.n, b.h, b.w), "concat spatial mismatch");
        let out = Shape::new(a.n, a.c + b.c, a.h, a.w);
        Cat {
            n: a.n,
            ca: a.c,
            cb: b.c,
            h: a.h,
            w: a.w,
            out: ctx.act(out.numel()),
            d_a: ctx.act(a.numel()),
            d_b: ctx.act(b.numel()),
        }
    }
    pub fn out_shape(&self) -> Shape {
        Shape::new(self.n, self.ca + self.cb, self.h, self.w)
    }
    pub fn forward(&self, ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer) {
        let threads = (self.ca + self.cb) * self.h * self.w * self.n;
        let s = ctx.step(ctx.ids.concat2, &[a, b, &self.out], &[self.n, self.ca, self.cb, self.h, self.w], threads);
        ctx.gpu.submit(&[], &[s]);
    }
    /// Split `d_out` into `self.d_a` (channels [0,ca)) and `self.d_b` ([ca,..)).
    pub fn backward(&self, ctx: &Ctx, d_out: &DeviceBuffer) {
        let ctot = self.ca + self.cb;
        let na = self.ca * self.h * self.w * self.n;
        let nb = self.cb * self.h * self.w * self.n;
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        let sa = ctx.step(ctx.ids.concat_split, &[d_out, &self.d_a], &[self.n, ctot, self.ca, 0, self.h, self.w], na);
        let sb = ctx.step(ctx.ids.concat_split, &[d_out, &self.d_b], &[self.n, ctot, self.cb, self.ca, self.h, self.w], nb);
        ctx.gpu.submit(&[], &[sa, sb]);
    }
}

/// A small out-of-place grad accumulator `dst = a + b` (the multi-consumer
/// pattern). Owns the destination so it survives across the backward chain.
pub struct Acc {
    pub out: DeviceBuffer,
    pub n: u32,
}
impl Acc {
    pub fn new(ctx: &Ctx, shape: Shape) -> Acc {
        Acc { out: ctx.act(shape.numel()), n: shape.numel() }
    }
    pub fn add(&self, ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer) {
        let s = ctx.step(ctx.ids.add2, &[a, b, &self.out], &[self.n], self.n);
        ctx.gpu.submit(&[], &[s]);
    }
}

