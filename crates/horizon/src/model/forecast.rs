// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forecast head: a variable's value at a time after entry, read from the
//! summary state.
//!
//! ```text
//! q    = z[subject of slot] + var[token] + ahead_bins @ Wt^T        [B*F, D]
//! (mu, log sigma) = (GELU(q @ Wh^T + bh)) @ Wo^T + bo               [B*F, 2]
//! loss = gauss_cens_nll(...)            scored where the slot holds a measurement
//! ```
//!
//! A forecast query never enters the set encoder: the existence of a
//! measurement at a later time says the subject was alive then, which an
//! encoder input would leak into every hazard. The head reads the same state
//! the hazards read, so training it shapes that state towards what the
//! subject's measurements will be - the multi-task pressure that makes the
//! state a state of health rather than a shortcut to one outcome.

use gpu_core::{DeviceBuffer, Step};

use super::*;
use crate::batch::HostBatch;

/// Device buffers of the forecast head, allocated when `cfg.forecasts > 0`.
pub(super) struct ForecastBufs {
    token: DeviceBuffer,
    subject: DeviceBuffer,
    time: DeviceBuffer,
    target: DeviceBuffer,
    state: DeviceBuffer,
    weight: DeviceBuffer,
    zf: DeviceBuffer,
    vf: DeviceBuffer,
    tf: DeviceBuffer,
    q1: DeviceBuffer,
    q: DeviceBuffer,
    h0: DeviceBuffer,
    h: DeviceBuffer,
    pred: DeviceBuffer,
    loss: DeviceBuffer,
    d_pred: DeviceBuffer,
    d_h: DeviceBuffer,
    d_h0: DeviceBuffer,
    d_q: DeviceBuffer,
}

impl ForecastBufs {
    pub(super) fn new(gpu: &Gpu, cfg: &HorizonConfig, b: u32) -> Option<ForecastBufs> {
        if cfg.forecasts == 0 {
            return None;
        }
        let (bf, d, nt) = (
            (b * cfg.forecasts) as u64,
            cfg.d_model as u64,
            cfg.time_bins as u64,
        );
        let st = |x: u64| gpu.storage(x);
        let input = |label: &str, words: u64| {
            gpu.buffer(label, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
        };
        Some(ForecastBufs {
            token: input("forecast_token", bf),
            subject: input("forecast_subject", bf),
            time: input("forecast_time", bf * nt),
            target: input("forecast_target", bf),
            state: input("forecast_state", bf),
            weight: input("forecast_weight", bf),
            zf: st(bf * d),
            vf: st(bf * d),
            tf: st(bf * d),
            q1: st(bf * d),
            q: st(bf * d),
            h0: st(bf * d),
            h: st(bf * d),
            pred: st(bf * 2),
            loss: st(bf),
            d_pred: st(bf * 2),
            d_h: st(bf * d),
            d_h0: st(bf * d),
            d_q: st(bf * d),
        })
    }

    pub(super) fn write(&self, gpu: &Gpu, hb: &HostBatch) {
        gpu.write(&self.token, &hb.forecast_token);
        gpu.write(&self.subject, &hb.forecast_subject);
        gpu.write_f32(&self.time, &hb.forecast_time);
        gpu.write_f32(&self.target, &hb.forecast_target);
        gpu.write(&self.state, &hb.forecast_state);
        gpu.write_f32(&self.weight, &hb.forecast_weight);
    }
}

impl Horizon {
    fn bf(&self) -> u32 {
        self.b * self.cfg.forecasts
    }

    pub(super) fn forecast_forward_steps(&self) -> Vec<Step> {
        let Some(f) = &self.fc else { return Vec::new() };
        let g = &self.gpu;
        let (d, nt, bf) = (self.cfg.d_model, self.cfg.time_bins, self.bf());
        vec![
            g.step(EMBED, &[&f.subject, &self.z, &f.zf], &[d, bf], bf * d),
            g.step(
                EMBED,
                &[&f.token, self.w("forecast.var"), &f.vf],
                &[d, bf],
                bf * d,
            ),
            self.mm(&f.time, self.w("forecast.time.weight"), &f.tf, bf, nt, d),
            g.step(ADD2, &[&f.zf, &f.vf, &f.q1], &[bf * d], bf * d),
            g.step(ADD2, &[&f.q1, &f.tf, &f.q], &[bf * d], bf * d),
            self.mm(&f.q, self.w("forecast.hidden.weight"), &f.h0, bf, d, d),
            g.step(
                BIAS_ADD,
                &[&f.h0, self.w("forecast.hidden.bias")],
                &[bf, d],
                bf * d,
            ),
            g.step(GELU, &[&f.h0, &f.h], &[bf * d], bf * d),
            self.mm(&f.h, self.w("forecast.out.weight"), &f.pred, bf, d, 2),
            g.step(
                BIAS_ADD,
                &[&f.pred, self.w("forecast.out.bias")],
                &[bf, 2],
                bf * 2,
            ),
            g.step(
                GAUSS_VALUE,
                &[&f.pred, &f.target, &f.state, &f.weight, &f.loss],
                &[bf],
                bf,
            ),
        ]
    }

    /// Accumulates the head's share of the state gradient into `d_z`, which
    /// the backward list clears before it runs.
    pub(super) fn forecast_backward_steps(&self) -> Vec<Step> {
        let Some(f) = &self.fc else { return Vec::new() };
        let g = &self.gpu;
        let gr = |name: &str| self.ps.g(name);
        let (d, nt, bf, b, v) = (
            self.cfg.d_model,
            self.cfg.time_bins,
            self.bf(),
            self.b,
            self.cfg.vocab,
        );
        vec![
            g.step(
                GAUSS_GRAD,
                &[&f.pred, &f.target, &f.state, &f.weight, &f.d_pred],
                &[bf],
                bf,
            ),
            self.bias_grad_part(&f.d_pred, bf, 2),
            self.bias_grad_final(gr("forecast.out.bias"), bf, 2),
            self.mm_dw(&f.d_pred, &f.h, gr("forecast.out.weight"), bf, d, 2),
            self.mm_dx(
                &f.d_pred,
                self.w("forecast.out.weight"),
                &f.d_h,
                bf,
                d,
                2,
                0,
            ),
            g.step(GELU_BWD, &[&f.h0, &f.d_h, &f.d_h0], &[bf * d], bf * d),
            self.bias_grad_part(&f.d_h0, bf, d),
            self.bias_grad_final(gr("forecast.hidden.bias"), bf, d),
            self.mm_dw(&f.d_h0, &f.q, gr("forecast.hidden.weight"), bf, d, d),
            self.mm_dx(
                &f.d_h0,
                self.w("forecast.hidden.weight"),
                &f.d_q,
                bf,
                d,
                d,
                0,
            ),
            // q = z[subject] + var[token] + time: the same gradient to all three.
            self.mm_dw(&f.d_q, &f.time, gr("forecast.time.weight"), bf, nt, d),
            g.step(
                EMB_BWD,
                &[&f.token, &f.d_q, gr("forecast.var")],
                &[bf, d, v],
                v * d,
            ),
            g.step(
                EMB_BWD,
                &[&f.subject, &f.d_q, &self.d_z],
                &[bf, d, b],
                b * d,
            ),
        ]
    }

    /// The forecast head's loss of the last forward (the weighted NLL).
    pub(super) fn forecast_loss(&self) -> f64 {
        let Some(f) = &self.fc else { return 0.0 };
        self.gpu
            .read(&f.loss, self.bf() as usize)
            .iter()
            .map(|&x| x as f64)
            .sum()
    }

    /// `(mu, log sigma)` per forecast slot of the last forward, `[B*F, 2]`,
    /// on the normal-score scale of each slot's variable.
    pub fn read_forecasts(&self) -> Vec<f32> {
        match &self.fc {
            Some(f) => self.gpu.read(&f.pred, (self.bf() * 2) as usize),
            None => Vec::new(),
        }
    }
}
