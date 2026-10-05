// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The additive baseline on the same contract: no attention, no value head.
//!
//! ```text
//! z          = sum over a subject's real tokens of e            (pooling)
//! log lambda = (z @ Wz^T)[subject of row] + time_features @ Wt^T + b   [B*P, K]
//! ```
//!
//! Each token's embedding is FiLM over soft bins of its value, so `w . e` is
//! a smooth function of each variable's value, and their sum is an ADDITIVE
//! proportional-hazards model (a piecewise-exponential GAM): per-variable
//! effects, a baseline over age, calendar and piece, no interactions. It is the
//! baseline the set encoder has to beat on the same inputs and the same
//! evaluation; restricted to a few standard risk factors it is the
//! conventional risk-score baseline.
//!
//! A subject's rows are contiguous, so the pooling is a masked sum over each
//! fixed-length segment of rows (`segment_sum_rows`) and its gradient the
//! segment's gradient broadcast back to its rows (`segment_bcast_rows`).

use gpu_core::Step;

use super::*;
use crate::batch::HostBatch;

/// `[B*N]` 1 for a real, non-summary token row: the rows the pooling sums.
pub(super) fn pool_mask(hb: &HostBatch, n: usize) -> Vec<u32> {
    let mut m = hb.keep.clone();
    for s in 0..hb.summary_rows.len() {
        m[s * n] = 0;
    }
    m
}

impl Horizon {
    pub(super) fn additive_forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let (d, k, nf) = (c.d_model, c.n_codes, c.time_features());
        let (bp, b) = (self.b * c.pieces(), self.b);
        let mut s = self.embedding_steps();
        s.extend([
            g.step(
                SEGMENT_SUM,
                &[&i.pool_mask, &self.res[0], &self.z],
                &[b, c.max_tokens, d],
                b * d,
            ),
            self.mm(&self.z, self.w("additive.state.weight"), &self.lz, b, d, k),
            g.step(
                EMBED,
                &[&i.piece_subject, &self.lz, &self.lzrep],
                &[k, bp],
                bp * k,
            ),
            self.mm(
                &i.time_features,
                self.w("additive.time.weight"),
                &self.tf,
                bp,
                nf,
                k,
            ),
            g.step(
                ADD2,
                &[&self.lzrep, &self.tf, &self.loglam],
                &[bp * k],
                bp * k,
            ),
            g.step(
                BIAS_ADD,
                &[&self.loglam, self.w("hazard.code.bias")],
                &[bp, k],
                bp * k,
            ),
            g.step(
                PEXP_VALUE,
                &[
                    &self.loglam,
                    &i.event,
                    &i.exposure,
                    &i.subject_weight,
                    &self.hloss,
                ],
                &[bp, k, c.pieces(), gpu_core::f(1.0)],
                bp * k,
            ),
        ]);
        s
    }

    /// `d_lz` is cleared before this list runs (it is accumulated into).
    pub(super) fn additive_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (d, k, nf) = (c.d_model, c.n_codes, c.time_features());
        let (bn, bp, b) = (self.b * c.max_tokens, self.b * c.pieces(), self.b);
        let mut s = vec![
            g.step(
                PEXP_GRAD,
                &[
                    &self.loglam,
                    &i.event,
                    &i.exposure,
                    &i.subject_weight,
                    &self.d_loglam,
                ],
                &[bp, k, c.pieces(), gpu_core::f(1.0)],
                bp * k,
            ),
            self.bias_grad_part(&self.d_loglam, bp, k),
            self.bias_grad_final(gr("hazard.code.bias"), bp, k),
            self.mm_dw(
                &self.d_loglam,
                &i.time_features,
                gr("additive.time.weight"),
                bp,
                nf,
                k,
            ),
            g.step(
                EMB_BWD,
                &[&i.piece_subject, &self.d_loglam, &self.d_lz],
                &[bp, k, b],
                b * k,
            ),
            self.mm_dw(&self.d_lz, &self.z, gr("additive.state.weight"), b, d, k),
            self.mm_dx(
                &self.d_lz,
                self.w("additive.state.weight"),
                &self.d_z,
                b,
                d,
                k,
                0,
            ),
            g.step(
                SEGMENT_BCAST,
                &[&i.pool_mask, &self.d_z, &self.dres[0]],
                &[b, c.max_tokens, d],
                bn * d,
            ),
        ];
        s.extend(self.embedding_backward_steps());
        s
    }
}
