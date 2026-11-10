// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Row-wise primitives of the device tape: softmax and per-row dot products and scales.

use super::DevTape;
use crate::train::tape::Id;

impl DevTape<'_> {
    /// Row-wise softmax over width `d`.
    pub(super) fn op_softmax_rows(&mut self, x: Id, d: usize) -> Id {
        let n = self.len_of(x);
        assert!(d > 0 && n.is_multiple_of(d), "softmax_rows: input does not tile {d}");
        let rows = n / d;
        let xb = self.buf(x);
        let p = self.rec.alloc(n);
        self.rec.softmax(&xb, &p, rows, d);
        let probs = p.clone();
        self.op(p, n, &[x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                cx.rec.softmax_bwd(&probs, gy, &gx, rows, d);
            }
        })
    }

    /// Per-row dot product of `a` and `b` (width `d`): `[rows]`.
    pub(super) fn op_row_dot(&mut self, a: Id, b: Id, d: usize) -> Id {
        let n = self.len_of(a);
        assert_eq!(n, self.len_of(b), "row_dot: operand lengths differ");
        assert!(d > 0 && n.is_multiple_of(d), "row_dot: operands do not tile {d}");
        let rows = n / d;
        let (ab, bb) = (self.buf(a), self.buf(b));
        let out = self.rec.alloc(rows);
        self.rec.row_red(&ab, &bb, &out, rows, d, false);
        self.op(out, rows, &[a, b], move |cx, gy| {
            for (x, other) in [(a, b), (b, a)] {
                if let Some(g) = cx.target(x) {
                    let o = cx.val(other);
                    cx.rec.mul_acc(&o, gy, &g, n, d, true);
                }
            }
        })
    }

    /// `x[r, :] * s[r]` for one scale per row of width `d`.
    pub(super) fn op_row_mul(&mut self, s: Id, x: Id, d: usize) -> Id {
        let n = self.len_of(x);
        let rows = self.len_of(s);
        assert_eq!(n, rows * d, "row_mul: one scale per row");
        let (sb, xb) = (self.buf(s), self.buf(x));
        let out = self.rec.alloc(n);
        self.rec.ew(4, &xb, &sb, &out, n, d, 0.0);
        self.op(out, n, &[s, x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let sv = cx.val(s);
                cx.rec.mul_acc(gy, &sv, &gx, n, d, true);
            }
            if let Some(gs) = cx.target(s) {
                let xv = cx.val(x);
                cx.rec.row_red(gy, &xv, &gs, rows, d, true);
            }
        })
    }
}
