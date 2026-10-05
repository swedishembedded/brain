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
//! The pooling sum is `pool @ e` with `pool` the `[B, B*N]` membership
//! matrix, computed as `matmul_dx` (which multiplies without transposing the
//! right operand); its gradient back to the token rows is `pool^T @ dz`, a
//! `matmul_dw`.

use gpu_core::Step;

use super::*;
use crate::batch::HostBatch;

/// The `[B, B*N]` membership matrix: 1 where row `j` is a real, non-summary
/// token of subject `b`.
pub(super) fn pool_matrix(hb: &HostBatch, n: usize) -> Vec<f32> {
    let b = hb.summary_rows.len();
    let mut m = vec![0.0; b * b * n];
    for s in 0..b {
        for row in s * n + 1..(s + 1) * n {
            if hb.keep[row] == 1 {
                m[s * b * n + row] = 1.0;
            }
        }
    }
    m
}

impl Horizon {
    pub(super) fn additive_forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let (d, k, nf) = (c.d_model, c.n_codes, c.time_features());
        let (bn, bp, b) = (self.b * c.max_tokens, self.b * c.pieces(), self.b);
        let mut s = self.embedding_steps();
        s.extend([
            g.step(
                MATMUL_DX,
                &[&i.pool, &self.res[0], &self.z],
                &[b, d, bn, 0],
                b * d,
            ),
            g.step(
                MATMUL,
                &[&self.z, self.w("additive.state.weight"), &self.lz],
                &[b, d, k],
                b * k,
            ),
            g.step(
                EMBED,
                &[&i.piece_subject, &self.lz, &self.lzrep],
                &[k, bp],
                bp * k,
            ),
            g.step(
                MATMUL,
                &[&i.time_features, self.w("additive.time.weight"), &self.tf],
                &[bp, nf, k],
                bp * k,
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

    /// `d_lz` and `dres[0]` are cleared before this list runs (both are
    /// accumulated into).
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
            g.step(
                BIAS_GRAD,
                &[&self.d_loglam, gr("hazard.code.bias")],
                &[bp, k],
                k,
            ),
            g.step(
                MATMUL_DW,
                &[&self.d_loglam, &i.time_features, gr("additive.time.weight")],
                &[bp, nf, k],
                k * nf,
            ),
            g.step(
                EMB_BWD,
                &[&i.piece_subject, &self.d_loglam, &self.d_lz],
                &[bp, k, b],
                b * k,
            ),
            g.step(
                MATMUL_DW,
                &[&self.d_lz, &self.z, gr("additive.state.weight")],
                &[b, d, k],
                k * d,
            ),
            g.step(
                MATMUL_DX,
                &[&self.d_lz, self.w("additive.state.weight"), &self.d_z],
                &[b, d, k, 0],
                b * d,
            ),
            g.step(
                MATMUL_DW,
                &[&i.pool, &self.d_z, &self.dres[0]],
                &[b, d, bn],
                bn * d,
            ),
        ];
        s.extend(self.embedding_backward_steps());
        s
    }
}
