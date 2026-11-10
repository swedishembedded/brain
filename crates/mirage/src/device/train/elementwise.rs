// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Elementwise primitives of the device tape: sums, products, scales, activations, the loss reduction.

use super::DevTape;
use crate::train::tape::Id;

impl DevTape<'_> {
    pub(super) fn op_add(&mut self, a: Id, b: Id) -> Id {
        let n = self.len_of(a);
        assert_eq!(n, self.len_of(b), "add: operand lengths differ");
        let (ab, bb) = (self.buf(a), self.buf(b));
        let out = self.rec.alloc(n);
        self.rec.ew(0, &ab, &bb, &out, n, n.max(1), 0.0);
        self.op(out, n, &[a, b], move |cx, gy| {
            for x in [a, b] {
                if let Some(g) = cx.target(x) {
                    cx.rec.accumulate(gy, &g, n);
                }
            }
        })
    }

    pub(super) fn op_mul(&mut self, a: Id, b: Id) -> Id {
        let n = self.len_of(a);
        assert_eq!(n, self.len_of(b), "mul: operand lengths differ");
        let (ab, bb) = (self.buf(a), self.buf(b));
        let out = self.rec.alloc(n);
        self.rec.ew(1, &ab, &bb, &out, n, n.max(1), 0.0);
        self.op(out, n, &[a, b], move |cx, gy| {
            for (x, other) in [(a, b), (b, a)] {
                if let Some(g) = cx.target(x) {
                    let o = cx.val(other);
                    cx.rec.mul_acc(gy, &o, &g, n, n.max(1), false);
                }
            }
        })
    }

    pub(super) fn op_scale(&mut self, a: Id, c: f64) -> Id {
        let n = self.len_of(a);
        let c = c as f32;
        let ab = self.buf(a);
        let out = self.rec.alloc(n);
        self.rec.ew(2, &ab, &ab, &out, n, n.max(1), c);
        self.op(out, n, &[a], move |cx, gy| {
            if let Some(g) = cx.target(a) {
                cx.rec.axpy(c, gy, &g, n);
            }
        })
    }

    /// Identity forward (the value buffer is shared, values are immutable); the gradient is scaled by `factor`.
    pub(super) fn op_grad_scale(&mut self, x: Id, factor: f64) -> Id {
        let n = self.len_of(x);
        let f = factor as f32;
        let buf = self.buf(x);
        self.op(buf, n, &[x], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                cx.rec.axpy(f, gy, &g, n);
            }
        })
    }

    /// `x[i] + b[i % len(b)]`.
    pub(super) fn op_bias_add(&mut self, x: Id, b: Id) -> Id {
        let (n, bl) = (self.len_of(x), self.len_of(b));
        assert!(bl > 0 && n % bl == 0, "bias_add: bias of {bl} does not tile {n} elements");
        let (xb, bb) = (self.buf(x), self.buf(b));
        let out = self.rec.alloc(n);
        self.rec.ew(0, &xb, &bb, &out, n, bl, 0.0);
        self.op(out, n, &[x, b], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                cx.rec.accumulate(gy, &g, n);
            }
            if let Some(g) = cx.target(b) {
                cx.rec.col_red(gy, gy, &g, n / bl, bl, false);
            }
        })
    }

    /// `x[i] * s[i % len(s)]`; a scalar scale is the period-1 case.
    pub(super) fn op_chan_mul(&mut self, x: Id, s: Id) -> Id {
        let (n, sl) = (self.len_of(x), self.len_of(s));
        assert!(sl > 0 && n % sl == 0, "chan_mul: scale of {sl} does not tile {n} elements");
        let (xb, sb) = (self.buf(x), self.buf(s));
        let out = self.rec.alloc(n);
        self.rec.ew(1, &xb, &sb, &out, n, sl, 0.0);
        self.op(out, n, &[x, s], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                let sv = cx.val(s);
                cx.rec.mul_acc(gy, &sv, &g, n, sl, false);
            }
            if let Some(g) = cx.target(s) {
                let xv = cx.val(x);
                cx.rec.col_red(gy, &xv, &g, n / sl, sl, true);
            }
        })
    }

    pub(super) fn op_scalar_mul(&mut self, x: Id, s: Id) -> Id {
        assert_eq!(self.len_of(s), 1, "scalar_mul: the scale is one element");
        self.op_chan_mul(x, s)
    }

    /// A shared unary kernel with its derivative: `mode` selects the backward (0 SiLU, 1 sigmoid, 2 GELU).
    pub(super) fn op_activation(&mut self, kernel: &'static str, mode: u32, x: Id) -> Id {
        let n = self.len_of(x);
        let xb = self.buf(x);
        let out = self.rec.alloc(n);
        self.rec.unary(kernel, &xb, &out, n);
        self.op(out, n, &[x], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                let xv = cx.val(x);
                cx.rec.act_bwd(&xv, gy, &g, n, mode);
            }
        })
    }

    /// `weight * sum(x)`: a scalar.
    pub(super) fn op_sum_weighted(&mut self, x: Id, weight: f64) -> Id {
        let n = self.len_of(x);
        let w = weight as f32;
        let xb = self.buf(x);
        let out = self.rec.alloc(1);
        self.rec.sum(&xb, &out, n, w);
        self.op(out, 1, &[x], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                cx.rec.bcast_acc(gy, &g, n, w);
            }
        })
    }
}
