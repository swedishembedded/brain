// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sequence primitives of the device tape: rotary embeddings, masked grouped-query attention, causal taps, Sinkhorn
//! normalisation, lane mixing and the Kronecker rotations of the Hadamard MLP.
//!
//! Every backward is owner-computes: one thread per output element, gathering what it needs (attention's key and
//! value gradients loop over the queries that see the key), so there are no atomics and the sums have a fixed order.

use super::DevTape;
use crate::train::seq::AttnShape;
use crate::train::tape::Id;

impl DevTape<'_> {
    /// Rotate pairs `(j, j + hd/2)` of every head of every row by `angles[row * hd/2 + j]` (radians).
    pub(super) fn op_rope(&mut self, x: Id, heads: usize, hd: usize, angles: &[f64]) -> Id {
        let pairs = hd / 2;
        let width = heads * hd;
        let n = self.len_of(x);
        assert!(hd.is_multiple_of(2) && n.is_multiple_of(width), "rope: input does not tile {heads} heads of {hd}");
        let rows = n / width;
        assert_eq!(angles.len(), rows * pairs, "rope: one angle per row and pair");
        let cs: Vec<f32> = angles.iter().flat_map(|a| [a.cos() as f32, a.sin() as f32]).collect();
        let cs = self.upload(&cs);
        let xb = self.buf(x);
        let y = self.rec.alloc(n);
        self.rec.rope(&xb, &cs, &y, rows, heads, hd, 1.0, false);
        self.op(y, n, &[x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                cx.rec.rope(gy, &cs, &gx, rows, heads, hd, -1.0, true);
            }
        })
    }

    /// Scaled dot-product attention with softmax over the visible keys. `q: [tq, hq*qk]`, `k: [tk, hk*qk]`,
    /// `v: [tk, hk*vh]`; output `[tq, hq*vh]`. The forward keeps the log-sum-exp of every (query, head) row; the
    /// backward rebuilds each probability from q, k and that statistic, so no `tq * hq * tk` matrix is ever stored.
    pub(super) fn op_attention(&mut self, q: Id, k: Id, v: Id, s: AttnShape) -> Id {
        assert!(s.hk > 0 && s.hq.is_multiple_of(s.hk), "attention: {} query heads do not group over {} key heads", s.hq, s.hk);
        assert_eq!(self.len_of(q), s.tq * s.hq * s.qk, "attention: q shape");
        assert_eq!(self.len_of(k), s.tk * s.hk * s.qk, "attention: k shape");
        assert_eq!(self.len_of(v), s.tk * s.hk * s.vh, "attention: v shape");
        let (qb, kb, vb) = (self.buf(q), self.buf(k), self.buf(v));
        let out_len = s.tq * s.hq * s.vh;
        let out = self.rec.alloc(out_len);
        let lse = self.rec.alloc(s.tq * s.hq);
        self.rec.attn_fwd(&qb, &kb, &vb, &out, &lse, &s);
        let out_b = out.clone();
        self.op(out, out_len, &[q, k, v], move |cx, go| {
            let (qv, kv, vv) = (cx.val(q), cx.val(k), cx.val(v));
            let delta = cx.rec.alloc(s.tq * s.hq);
            cx.rec.attn_delta(go, &out_b, &delta, &s);
            if let Some(gq) = cx.target(q) {
                cx.rec.attn_dq(&qv, &kv, &vv, go, &lse, &delta, &gq, &s);
            }
            // dk and dv share one pass; a frozen side is written to a scratch accumulator.
            let (gk, gv) = (cx.target(k), cx.target(v));
            if gk.is_some() || gv.is_some() {
                let gk = gk.unwrap_or_else(|| cx.rec.alloc_zeroed(s.tk * s.hk * s.qk));
                let gv = gv.unwrap_or_else(|| cx.rec.alloc_zeroed(s.tk * s.hk * s.vh));
                cx.rec.attn_dkv(&qv, &kv, &vv, go, &lse, &delta, &gk, &gv, &s);
            }
        })
    }

    /// Causal dilated taps: `y[r] = sum_j taps[j] * x[r - j * dilation]`, `x: [t, w]`, `taps: [kt, w]`.
    pub(super) fn op_causal_taps(&mut self, x: Id, taps: Id, w: usize, dilation: usize) -> Id {
        let (n, tl) = (self.len_of(x), self.len_of(taps));
        assert!(w > 0 && tl.is_multiple_of(w) && n.is_multiple_of(w), "causal_taps: taps and input do not tile {w}");
        let (t, kt) = (n / w, tl / w);
        let (xb, tb) = (self.buf(x), self.buf(taps));
        let y = self.rec.alloc(n);
        self.rec.taps(&xb, &tb, &y, t, w, kt, dilation);
        self.op(y, n, &[x, taps], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let tv = cx.val(taps);
                cx.rec.taps_dx(gy, &tv, &gx, t, w, kt, dilation);
            }
            if let Some(gt) = cx.target(taps) {
                let xv = cx.val(x);
                cx.rec.taps_dt(&xv, gy, &gt, t, w, kt, dilation);
            }
        })
    }

    /// Sinkhorn normalisation in log space of each `n x n` block of `logits` (`[t, n*n]`), `iters` rounds.
    pub(super) fn op_sinkhorn(&mut self, logits: Id, n: usize, iters: usize) -> Id {
        let nn = n * n;
        let len = self.len_of(logits);
        assert!(n > 0 && n <= 6, "sinkhorn: blocks of {n}x{n} exceed the device kernel's 6x6");
        assert!(len.is_multiple_of(nn), "sinkhorn: logits do not tile {n}x{n}");
        let blocks = len / nn;
        let xb = self.buf(logits);
        let y = self.rec.alloc(len);
        self.rec.sinkhorn(&xb, &y, blocks, n, iters);
        let yb = y.clone();
        self.op(y, len, &[logits], move |cx, gy| {
            if let Some(gl) = cx.target(logits) {
                let traj = cx.rec.alloc(blocks * 2 * iters * nn);
                let xv = cx.val(logits);
                cx.rec.sinkhorn_bwd(&xv, &yb, gy, &traj, &gl, blocks, n, iters);
            }
        })
    }

    /// `out[r, i, :] = sum_j h[r, i*b + j] * x[r, j, :]` with `h: [t, a*b]`, `x: [t, b*d]`, output `[t, a*d]`.
    pub(super) fn op_row_mix(&mut self, h: Id, x: Id, a: usize, b: usize, d: usize) -> Id {
        let t = self.len_of(h) / (a * b);
        assert_eq!(self.len_of(h), t * a * b, "row_mix: weights do not tile {a}x{b}");
        assert_eq!(self.len_of(x), t * b * d, "row_mix: sources do not tile {b}x{d}");
        let dims = [t, a, b, d];
        let (hb, xb) = (self.buf(h), self.buf(x));
        let y = self.rec.alloc(t * a * d);
        self.rec.mix(&hb, &xb, &y, dims);
        self.op(y, t * a * d, &[h, x], move |cx, gy| {
            if let Some(gh) = cx.target(h) {
                let xv = cx.val(x);
                cx.rec.mix_dh(gy, &xv, &gh, dims);
            }
            if let Some(gx) = cx.target(x) {
                let hv = cx.val(h);
                cx.rec.mix_dx(&hv, gy, &gx, dims);
            }
        })
    }

    /// Kronecker rotation of each row viewed as `[fa, fb]`: `out = A^T Z B` with `a: [fa, fa]`, `b: [fb, fb]`.
    pub(super) fn op_kron(&mut self, z: Id, a: Id, b: Id, fa: usize, fb: usize) -> Id {
        let n = fa * fb;
        let len = self.len_of(z);
        assert_eq!(self.len_of(a), fa * fa, "kron: left factor");
        assert_eq!(self.len_of(b), fb * fb, "kron: right factor");
        assert!(len.is_multiple_of(n), "kron: input does not tile {fa}x{fb}");
        let rows = len / n;
        let (zb, ab, bb) = (self.buf(z), self.buf(a), self.buf(b));
        // tmp = Z B, out = A^T tmp.
        let tmp = self.rec.alloc(len);
        self.rec.kron_right(&zb, &bb, &tmp, rows * fa, fb, fb, (fb, 1), false);
        let y = self.rec.alloc(len);
        self.rec.kron_left(&tmp, &ab, &y, rows, (fa, fb), (fa, 1), false);
        self.op(y, len, &[z, a, b], move |cx, g| {
            let (zv, av, bv) = (cx.val(z), cx.val(a), cx.val(b));
            // dtmp = A dOut; dZ = dtmp B^T; dA = tmp dOut^T summed over rows; dB = Z^T dtmp summed over rows.
            let dtmp = cx.rec.alloc(len);
            cx.rec.kron_left(g, &av, &dtmp, rows, (fa, fb), (1, fa), false);
            if let Some(gz) = cx.target(z) {
                cx.rec.kron_right(&dtmp, &bv, &gz, rows * fa, fb, fb, (1, fb), true);
            }
            if let Some(ga) = cx.target(a) {
                cx.rec.kron_da(&tmp, g, &ga, rows, fa, fb);
            }
            if let Some(gb) = cx.target(b) {
                cx.rec.kron_db(&zv, &dtmp, &gb, rows, fa, fb);
            }
        })
    }

    /// `out[r, i] = z[r, perm[i]]` for rows of width `n`; `perm` must be a permutation (the backward gathers
    /// through its inverse, so every input column has one owner).
    pub(super) fn op_permute_cols(&mut self, z: Id, n: usize, perm: &[u32]) -> Id {
        assert_eq!(perm.len(), n, "permute_cols: one source per column");
        let len = self.len_of(z);
        let mut inverse = vec![u32::MAX; n];
        for (i, &p) in perm.iter().enumerate() {
            assert!((p as usize) < n && inverse[p as usize] == u32::MAX, "permute_cols: `perm` is not a permutation of 0..{n}");
            inverse[p as usize] = i as u32;
        }
        let (fwd, inv) = (self.upload_words(perm), self.upload_words(&inverse));
        let zb = self.buf(z);
        let y = self.rec.alloc(len);
        self.rec.gather_cols(&zb, &fwd, &y, len, n, false);
        self.op(y, len, &[z], move |cx, gy| {
            if let Some(gz) = cx.target(z) {
                cx.rec.gather_cols(gy, &inv, &gz, len, n, true);
            }
        })
    }
}
