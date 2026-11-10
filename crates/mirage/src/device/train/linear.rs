// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense projections of the device tape. Kernels keep the checkpoint's Flax `[in, out]` layout (nothing is
//! transposed): the shared GEMM family reads one `[in, out]` weight as `matmul_dx` (forward), `matmul` (input
//! gradient) and `matmul_dw` (weight gradient).

use super::DevTape;
use crate::train::tape::Id;

impl DevTape<'_> {
    /// `y = x W` for `x: [rows, inn]`, `w: [inn, out]`.
    pub(super) fn op_linear(&mut self, x: Id, w: Id, rows: usize, inn: usize, out: usize) -> Id {
        assert_eq!(self.len_of(x), rows * inn, "linear: input is not {rows}x{inn}");
        assert_eq!(self.len_of(w), inn * out, "linear: kernel is not {inn}x{out}");
        let (xb, wb) = (self.buf(x), self.buf(w));
        let y = self.rec.alloc(rows * out);
        self.rec.matmul_in_out(&wb, &xb, &y, rows, inn, out);
        self.op(y, rows * out, &[x, w], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                // The GEMM family assigns, so the product lands in a scratch buffer and is added in.
                let wv = cx.val(w);
                let tmp = cx.rec.alloc(rows * inn);
                cx.rec.matmul_nk(&wv, gy, &tmp, rows, inn, out);
                cx.rec.accumulate(&tmp, &gx, rows * inn);
            }
            if let Some(gw) = cx.target(w) {
                let xv = cx.val(x);
                cx.rec.matmul_dw(&xv, gy, &gw, rows, inn, out);
            }
        })
    }

    /// `y = x W^T` for `x: [rows, inn]`, `w: [out, inn]`.
    pub(super) fn op_linear_nt(&mut self, x: Id, w: Id, rows: usize, inn: usize, out: usize) -> Id {
        assert_eq!(self.len_of(x), rows * inn, "linear_nt: input is not {rows}x{inn}");
        assert_eq!(self.len_of(w), out * inn, "linear_nt: table is not {out}x{inn}");
        let (xb, wb) = (self.buf(x), self.buf(w));
        let y = self.rec.alloc(rows * out);
        self.rec.matmul_nk(&wb, &xb, &y, rows, out, inn);
        self.op(y, rows * out, &[x, w], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let wv = cx.val(w);
                cx.rec.matmul_dx_acc(&wv, gy, &gx, rows, out, inn);
            }
            if let Some(gw) = cx.target(w) {
                let xv = cx.val(x);
                cx.rec.matmul_dw(gy, &xv, &gw, rows, out, inn);
            }
        })
    }
}
