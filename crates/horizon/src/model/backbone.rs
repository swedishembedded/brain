// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The continuous-time state across visits (`HorizonConfig::visits > 0`).
//!
//! ```text
//! u   = xf[summary row of each visit]                                  [S, D]
//! xg  = u @ Win^T + bin = [x | gate pre-activation]                    [S, 2D]
//! per subject and channel, r = softplus(rate), m = population state:
//!   h = m;  per visit, dt after the previous one:
//!     q = m + exp(-r dt) (h - m);   h = q + sigmoid(gate) (x - q)
//!   z = m + exp(-r T) (h - m),  T from the last visit to the prediction time
//! ```
//!
//! Between visits the state forgets towards the population at a rate per
//! channel, over the time that actually passed: a gap longer than any seen in
//! training is extrapolated by the same exponential, not by an embedding of
//! the gap that was never trained there. Each visit moves the state towards
//! its own evidence by a learned gate - the diagonal form of the delta rule.
//! A new visit costs one more step; the history is never re-encoded.

use gpu_core::{DeviceBuffer, Gpu, Step};

use super::*;

/// Device buffers of the state across visits, allocated when `cfg.visits > 0`.
pub(super) struct BackboneBufs {
    u: DeviceBuffer,
    xg: DeviceBuffer,
    hs: DeviceBuffer,
    d_u: DeviceBuffer,
    d_xg: DeviceBuffer,
    part: DeviceBuffer,
}

impl BackboneBufs {
    pub(super) fn new(gpu: &Gpu, cfg: &HorizonConfig, b: u32) -> Option<BackboneBufs> {
        if cfg.visits == 0 {
            return None;
        }
        let (s, d, bb) = (
            (b * cfg.sets_per_subject()) as u64,
            cfg.d_model as u64,
            b as u64,
        );
        let st = |x: u64| gpu.storage(x);
        Some(BackboneBufs {
            u: st(s * d),
            xg: st(s * 2 * d),
            hs: st(s * d),
            d_u: st(s * d),
            d_xg: st(s * 2 * d),
            part: st(bb * 2 * d),
        })
    }
}

impl Horizon {
    /// From the encoder's output to the state `z` the heads read: the summary
    /// row of each subject, or, across visits, the state the visits leave.
    pub(super) fn state_forward_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let Some(v) = &self.bb else {
            return vec![g.step(EMBED, &[&i.summary_rows, &self.xf, &self.z], &[d, b], b * d)];
        };
        vec![
            g.step(EMBED, &[&i.summary_rows, &self.xf, &v.u], &[d, s], s * d),
            self.mm(&v.u, self.w("visit.in.weight"), &v.xg, s, d, 2 * d),
            g.step(
                BIAS_ADD,
                &[&v.xg, self.w("visit.in.bias")],
                &[s, 2 * d],
                s * 2 * d,
            ),
            g.step(
                CT_SCAN,
                &[&v.xg, &i.visit_dt, self.w("visit.state"), &v.hs, &self.z],
                &[b, self.cfg.visits, d],
                b * d,
            ),
        ]
    }

    /// The adjoint of [`Horizon::state_forward_steps`]: `d_z` (complete) into
    /// the summary rows of `d_xf`, and the backbone's own gradients.
    pub(super) fn state_backward_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let bn = s * self.cfg.max_tokens;
        // The summary rows carry no value target (d_xf is zero there), so
        // their gradient is exactly the state's.
        let Some(v) = &self.bb else {
            return vec![g.step(
                ROW_SCATTER,
                &[&i.summary_rows, &self.d_z, &self.d_xf],
                &[b, d, bn],
                b * d,
            )];
        };
        let mut steps = vec![g.step(
            CT_SCAN_BWD,
            &[
                &v.xg,
                &i.visit_dt,
                self.w("visit.state"),
                &v.hs,
                &self.d_z,
                &v.d_xg,
                &v.part,
            ],
            &[b, self.cfg.visits, d],
            b * d,
        )];
        // Each subject's share of the rate and population-state gradients,
        // summed over subjects.
        steps.extend(self.bias_grad(&v.part, gr("visit.state"), b, 2 * d));
        steps.extend(self.bias_grad(&v.d_xg, gr("visit.in.bias"), s, 2 * d));
        steps.extend([
            self.mm_dw(&v.d_xg, &v.u, gr("visit.in.weight"), s, d, 2 * d),
            self.mm_dx(&v.d_xg, self.w("visit.in.weight"), &v.d_u, s, d, 2 * d, 0),
            g.step(
                ROW_SCATTER,
                &[&i.summary_rows, &v.d_u, &self.d_xf],
                &[s, d, bn],
                s * d,
            ),
        ]);
        steps
    }
}
