// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spatial and channel-mixing primitives of the device tape: depthwise and dense convolutions, nearest-neighbour
//! upsampling, the sequency Walsh transform, the banded channel convolution and global response normalisation.
//!
//! The forward convolutions are the inference kernels; their gradients are owner-computes kernels (one thread per
//! input element or per kernel element, no atomics). The Walsh transform runs as a GEMM against its (symmetric,
//! orthonormal) matrix, built once per width from the host transform, so the backward is the same product.

use super::launch::Conv;
use super::DevTape;
use crate::train::spatial::walsh_rows;
use crate::train::tape::Id;
use crate::vision::ops::Map;

impl DevTape<'_> {
    /// Depthwise (`dense_cout == None`, kernel `[kh * kw, c]`) or dense (HWIO) convolution with zero padding `k / 2`.
    pub(super) fn op_conv(&mut self, x: Id, m: Map, kernel: Id, k: (usize, usize), dense_cout: Option<usize>, stride: (usize, usize)) -> (Id, Map) {
        let g = Conv { h: m.h, w: m.w, c: m.c, kh: k.0, kw: k.1, sy: stride.0, sx: stride.1 };
        let cout = dense_cout.unwrap_or(m.c);
        assert_eq!(self.len_of(x), m.h * m.w * m.c, "conv: input does not match its map");
        let kernel_len = if dense_cout.is_some() { k.0 * k.1 * m.c * cout } else { k.0 * k.1 * m.c };
        assert_eq!(self.len_of(kernel), kernel_len, "conv: kernel does not match");
        let o = Map { h: g.ho(), w: g.wo(), c: cout };
        let (xb, kb) = (self.buf(x), self.buf(kernel));
        let y = self.rec.alloc(o.h * o.w * cout);
        self.rec.conv_fwd(g, dense_cout, &xb, &kb, &y);
        let id = self.op(y, o.h * o.w * cout, &[x, kernel], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let kv = cx.val(kernel);
                cx.rec.conv_dx(g, dense_cout, gy, &kv, &gx);
            }
            if let Some(gk) = cx.target(kernel) {
                let xv = cx.val(x);
                cx.rec.conv_dk(g, dense_cout, &xv, gy, &gk);
            }
        });
        (id, o)
    }

    /// Nearest-neighbour 2x upsample of a map (a plan: every output pixel copies its parent).
    pub(super) fn op_upsample2(&mut self, x: Id, m: Map) -> (Id, Map) {
        let o = Map { h: m.h * 2, w: m.w * 2, c: m.c };
        let mut segs = Vec::with_capacity(o.h * o.w);
        for oy in 0..o.h {
            for ox in 0..o.w {
                segs.push(crate::train::ops::Seg::copy((oy * o.w + ox) * m.c, ((oy / 2) * m.w + ox / 2) * m.c, m.c));
            }
        }
        (self.op_plan(x, o.h * o.w * o.c, segs), o)
    }

    /// The `[n, n]` matrix of the normalised sequency Walsh transform (the transform of the identity's rows).
    fn walsh_matrix(&mut self, n: usize) -> gpu_core::DeviceBuffer {
        if let Some(w) = self.walsh.get(&n) {
            return w.clone();
        }
        let mut id = vec![0f32; n * n];
        (0..n).for_each(|i| id[i * n + i] = 1.0);
        walsh_rows(&mut id, n);
        let w = self.upload(&id);
        self.walsh.insert(n, w.clone());
        w
    }

    /// Normalised sequency-ordered Walsh transform of every row of width `n`. Orthonormal and symmetric, so the
    /// backward is the same product with the output gradient.
    pub(super) fn op_walsh(&mut self, x: Id, n: usize) -> Id {
        let len = self.len_of(x);
        assert!(n.is_power_of_two() && len.is_multiple_of(n), "walsh: width {n} is not a power of two tiling the input");
        let rows = len / n;
        let (w, xb) = (self.walsh_matrix(n), self.buf(x));
        let y = self.rec.alloc(len);
        self.rec.matmul_in_out(&w, &xb, &y, rows, n, n);
        self.op(y, len, &[x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let tmp = cx.rec.alloc(len);
                cx.rec.matmul_nk(&w, gy, &tmp, rows, n, n);
                cx.rec.accumulate(&tmp, &gx, len);
            }
        })
    }

    /// Per-channel banded convolution along the channel axis, `w: [n, k]`, `k` odd.
    pub(super) fn op_band(&mut self, u: Id, n: usize, w: Id, k: usize) -> Id {
        assert_eq!(self.len_of(w), n * k, "band: weights are not {n}x{k}");
        let len = self.len_of(u);
        assert!(n > 0 && len.is_multiple_of(n), "band: input does not tile {n}");
        let rows = len / n;
        let (ub, wb) = (self.buf(u), self.buf(w));
        let y = self.rec.alloc(len);
        self.rec.band(&ub, &wb, &y, rows, n, k);
        self.op(y, len, &[u, w], move |cx, gy| {
            if let Some(gu) = cx.target(u) {
                let wv = cx.val(w);
                cx.rec.band_dx(gy, &wv, &gu, rows, n, k);
            }
            if let Some(gw) = cx.target(w) {
                let uv = cx.val(u);
                cx.rec.band_dw(&uv, gy, &gw, rows, n, k);
            }
        })
    }

    /// The per-channel scale of global response normalisation over the valid rows of `y` (width `d`).
    pub(super) fn op_grn_scale(&mut self, y: Id, d: usize, valid: &[bool], gamma: Id) -> Id {
        let len = self.len_of(y);
        assert_eq!(len, valid.len() * d, "grn_scale: mask does not match the rows");
        assert_eq!(self.len_of(gamma), d, "grn_scale: gamma is not {d} wide");
        let mask: Vec<f32> = valid.iter().map(|&ok| f32::from(u8::from(ok))).collect();
        let (mask, yb, gb) = (self.upload(&mask), self.buf(y), self.buf(gamma));
        let energy = self.rec.alloc(d);
        let scale = self.rec.alloc(d);
        self.rec.grn_scale(&yb, &mask, &gb, &energy, &scale, valid.len(), d);
        self.op(scale, d, &[y, gamma], move |cx, gs| {
            let de = cx.rec.alloc(d);
            let gv = cx.val(gamma);
            // The statistics kernel always writes a gamma gradient; without a target it lands in scratch.
            let dgamma = cx.target(gamma).unwrap_or_else(|| cx.rec.alloc_zeroed(d));
            cx.rec.grn_de(&energy, &gv, gs, &de, &dgamma, d);
            if let Some(gy) = cx.target(y) {
                let yv = cx.val(y);
                cx.rec.grn_dy(&yv, &mask, &energy, &de, &gy, len, d);
            }
        })
    }
}
