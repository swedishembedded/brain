// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loss primitives of the device tape: weighted token cross-entropy, CenterNet focal loss, weighted L1,
//! BCE plus Dice, and CTC. Each is a scalar. Per-element or per-row terms are written by one thread each and summed
//! by a single workgroup in a fixed order (`mirage_t_sum`), so the value is deterministic and needs no atomics.

use super::DevTape;
use crate::train::loss::ctc_feasible;
use crate::train::tape::Id;

impl DevTape<'_> {
    /// `sum_r weight[r] * (logsumexp(logits[r]) - logits[r][target[r]])`; a zero-weight row is ignored.
    pub(super) fn op_softmax_ce(&mut self, logits: Id, v: usize, targets: &[u32], weights: &[f64]) -> Id {
        let n = targets.len();
        assert_eq!(self.len_of(logits), n * v, "softmax_ce: logits do not match the targets");
        assert_eq!(weights.len(), n, "softmax_ce: one weight per row");
        let (xb, tb, wb) = (self.buf(logits), self.upload_words(targets), self.upload_f64(weights));
        let (rowloss, lse) = (self.rec.alloc(n), self.rec.alloc(n));
        let out = self.rec.alloc(1);
        self.rec.ce_rows(&xb, &tb, &wb, &rowloss, &lse, n, v);
        self.rec.sum(&rowloss, &out, n, 1.0);
        self.op(out, 1, &[logits], move |cx, gy| {
            if let Some(gl) = cx.target(logits) {
                let xv = cx.val(logits);
                cx.rec.ce_bwd(&xv, &tb, &wb, &lse, gy, &gl, n, v);
            }
        })
    }

    /// CenterNet focal loss summed over weighted cells and divided by `norm`.
    pub(super) fn op_focal_heat(&mut self, logits: Id, target: &[f64], weight: &[f64], alpha: f64, norm: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len_of(logits), n, "focal_heat: logits do not match the target");
        assert_eq!(weight.len(), n, "focal_heat: one weight per cell");
        let (xb, tb, wb) = (self.buf(logits), self.upload_f64(target), self.upload_f64(weight));
        let (alpha, norm) = (alpha as f32, norm as f32);
        let terms = self.rec.alloc(n);
        let out = self.rec.alloc(1);
        self.rec.focal(&xb, &tb, &wb, &xb, &terms, n, alpha, norm, false);
        self.rec.sum(&terms, &out, n, 1.0 / norm);
        self.op(out, 1, &[logits], move |cx, gy| {
            if let Some(gl) = cx.target(logits) {
                let xv = cx.val(logits);
                cx.rec.focal(&xv, &tb, &wb, gy, &gl, n, alpha, norm, true);
            }
        })
    }

    /// `sum_i weight[i] * |x[i] - target[i]| / norm`.
    pub(super) fn op_l1(&mut self, x: Id, target: &[f64], weight: &[f64], norm: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len_of(x), n, "l1: prediction does not match the target");
        assert_eq!(weight.len(), n, "l1: one weight per element");
        let (xb, tb, wb) = (self.buf(x), self.upload_f64(target), self.upload_f64(weight));
        let norm = norm as f32;
        let terms = self.rec.alloc(n);
        let out = self.rec.alloc(1);
        self.rec.l1(&xb, &tb, &wb, &xb, &terms, n, norm, false);
        self.rec.sum(&terms, &out, n, 1.0 / norm);
        self.op(out, 1, &[x], move |cx, gy| {
            if let Some(gx) = cx.target(x) {
                let xv = cx.val(x);
                cx.rec.l1(&xv, &tb, &wb, gy, &gx, n, norm, true);
            }
        })
    }

    /// Weighted mean BCE of `logits` plus the Dice loss, both against the 0/1 `target`.
    pub(super) fn op_bce_dice(&mut self, logits: Id, target: &[f64], bce_weight: f64, dice_weight: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len_of(logits), n, "bce_dice: logits do not match the target");
        assert!(n > 0, "bce_dice: an empty region");
        let (xb, tb) = (self.buf(logits), self.upload_f64(target));
        let (bw, dw) = (bce_weight as f32, dice_weight as f32);
        let (out, stats) = (self.rec.alloc(1), self.rec.alloc(3));
        self.rec.bce_dice(&xb, &tb, &out, &stats, n, bw, dw);
        self.op(out, 1, &[logits], move |cx, gy| {
            if let Some(gl) = cx.target(logits) {
                let xv = cx.val(logits);
                cx.rec.bce_dice_bwd(&xv, &tb, &stats, gy, &gl, n, bw, dw);
            }
        })
    }

    /// CTC loss `-log p(labels | logits)` for `logits: [frames, classes]`, blank class 0. When no alignment exists
    /// the loss is zero with no gradient, as on the host.
    pub(super) fn op_ctc(&mut self, logits: Id, classes: usize, labels: &[u32]) -> Id {
        let frames = self.len_of(logits) / classes;
        let out = self.rec.alloc_zeroed(1);
        if !ctc_feasible(frames, labels) {
            return self.op(out, 1, &[logits], |_, _| {});
        }
        let ext: Vec<u32> = std::iter::once(0).chain(labels.iter().flat_map(|&l| [l, 0])).collect();
        let (xb, eb) = (self.buf(logits), self.upload_words(&ext));
        let grad = self.rec.alloc(frames * classes);
        self.rec.ctc(&xb, &eb, &out, &grad, frames, classes, ext.len());
        self.op(out, 1, &[logits], move |cx, gy| {
            if let Some(gl) = cx.target(logits) {
                cx.rec.mul_acc(&grad, gy, &gl, frames * classes, 1, false);
            }
        })
    }
}
