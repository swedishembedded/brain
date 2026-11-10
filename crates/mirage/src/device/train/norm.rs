// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Row normalisation and row masking on the device tape.

use super::DevTape;
use crate::train::tape::Id;

impl DevTape<'_> {
    /// `(1 + scale) * x / sqrt(mean(x^2) + eps)` over rows of width `d`; without a `scale` the unit RMS norm.
    pub(super) fn op_zc_norm(&mut self, x: Id, d: usize, scale: Option<Id>) -> Id {
        let n = self.len_of(x);
        assert!(d > 0 && n.is_multiple_of(d), "zc_norm: input does not tile {d}");
        let rows = n / d;
        let xb = self.buf(x);
        // The norm kernel multiplies by a weight: `1 + scale`, or ones for the unit norm.
        let weight = match scale {
            Some(s) => {
                assert_eq!(self.len_of(s), d, "zc_norm: scale is not {d} wide");
                let sb = self.buf(s);
                let w = self.rec.alloc(d);
                self.rec.ew(3, &sb, &sb, &w, d, d, 1.0);
                w
            }
            None => self.ones(d),
        };
        let out = self.rec.alloc(n);
        self.rec.rmsnorm(&xb, &weight, &out, rows, d);
        let inputs: Vec<Id> = std::iter::once(x).chain(scale).collect();
        self.op(out, n, &inputs, move |cx, gy| {
            let xv = cx.val(x);
            let stats = cx.rec.alloc(2 * rows);
            cx.rec.rms_stats(&xv, gy, &weight, &stats, rows, d);
            if let Some(gx) = cx.target(x) {
                cx.rec.rms_dx(&xv, gy, &weight, &stats, &gx, n, d);
            }
            if let Some(gs) = scale.and_then(|s| cx.target(s)) {
                cx.rec.rms_dscale(&xv, gy, &stats, &gs, rows, d);
            }
        })
    }

    /// Zero the rows of width `d` whose `valid` entry is false: a per-row product with a constant 0/1 mask.
    pub(super) fn op_mask_rows(&mut self, x: Id, d: usize, valid: &[bool]) -> Id {
        let n = self.len_of(x);
        assert_eq!(n, valid.len() * d, "mask_rows: mask does not match the rows");
        let mask: Vec<f32> = valid.iter().map(|&ok| f32::from(u8::from(ok))).collect();
        let m = self.upload(&mask);
        let xb = self.buf(x);
        let out = self.rec.alloc(n);
        self.rec.ew(4, &xb, &m, &out, n, d, 0.0);
        self.op(out, n, &[x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                cx.rec.mul_acc(gy, &m, &gx, n, d, true);
            }
        })
    }
}
