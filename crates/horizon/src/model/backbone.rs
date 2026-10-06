// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What carries a subject's visits to the prediction time
//! (`HorizonConfig::visits > 0`), in one of two forms.
//!
//! **State** (`Backbone::State`):
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
//!
//! **Attention** (`Backbone::Attention`), the comparison arm:
//!
//! ```text
//! seq = [u of the subject's visit slots | query]                       [B*(V+1), D]
//! qkv = seq @ Wqkv^T + bqkv, q and k rotated by their row's real time
//!       relative to the prediction time (rotary angles, unused slots masked)
//! z   = query + (Attn(qkv)[query row] @ Wo^T + bo)                     [B, D]
//! ```
//!
//! The query attends to every visit by content and by how long before the
//! prediction time it was; nothing ties a gap's effect to a decay.

use gpu_core::{DeviceBuffer, Gpu, Step};

use super::*;
use super::stack::StackBufs;
use crate::config::Backbone;

/// The rotary base of the attention arm: with times in years and heads of a
/// few channels, the rotation frequencies span about a radian a year down to
/// a few thousandths - months to decades.
const ROPE_THETA: f32 = 1000.0;

/// Device buffers of the state arm.
pub(super) struct StateBufs {
    u: DeviceBuffer,
    xg: DeviceBuffer,
    hs: DeviceBuffer,
    d_u: DeviceBuffer,
    d_xg: DeviceBuffer,
    part: DeviceBuffer,
}

/// Device buffers of the attention arm.
pub(super) struct AttnBufs {
    /// `[S]` each visit set's row in the sequences (constant).
    seq_rows: DeviceBuffer,
    /// `[B]` each subject's query row (constant).
    query_rows: DeviceBuffer,
    /// `[B]` zeros: every subject reads the one query embedding (constant).
    query_index: DeviceBuffer,
    pos: DeviceBuffer,
    keep: DeviceBuffer,
    u: DeviceBuffer,
    qe: DeviceBuffer,
    seq: DeviceBuffer,
    qkv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    cq: DeviceBuffer,
    proj: DeviceBuffer,
    d_cq: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_seq: DeviceBuffer,
    d_u: DeviceBuffer,
    d_qa: DeviceBuffer,
    d_qe: DeviceBuffer,
}

/// The visit backbone's buffers, allocated when `cfg.visits > 0`.
pub(super) enum BackboneBufs {
    State(StateBufs),
    Attention(AttnBufs),
    Stack(Box<StackBufs>),
}

impl BackboneBufs {
    pub(super) fn new(gpu: &Gpu, cfg: &HorizonConfig, b: u32) -> Option<BackboneBufs> {
        if cfg.visits == 0 {
            return None;
        }
        let vs = cfg.sets_per_subject();
        let (s, d, bb) = ((b * vs) as u64, cfg.d_model as u64, b as u64);
        let st = |x: u64| gpu.storage(x);
        Some(match cfg.backbone {
            Backbone::State => BackboneBufs::State(StateBufs {
                u: st(s * d),
                xg: st(s * 2 * d),
                hs: st(s * d),
                d_u: st(s * d),
                d_xg: st(s * 2 * d),
                part: st(bb * 2 * d),
            }),
            Backbone::Stack(_) => BackboneBufs::Stack(Box::new(StackBufs::new(gpu, cfg, b))),
            Backbone::Attention => {
                let t = vs + 1;
                let rows = (b * t) as u64;
                let att = bb * cfg.n_heads as u64 * (t * t) as u64;
                let input = |label: &str, words: u64| {
                    gpu.buffer(label, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
                };
                let (seq_rows, query_rows, query_index) = (
                    input("seq_rows", s),
                    input("query_rows", bb),
                    input("query_index", bb),
                );
                let set_rows: Vec<u32> = (0..b * vs).map(|x| (x / vs) * t + x % vs).collect();
                let q_rows: Vec<u32> = (0..b).map(|i| i * t + vs).collect();
                gpu.write(&seq_rows, &set_rows);
                gpu.write(&query_rows, &q_rows);
                gpu.write(&query_index, &vec![0u32; b as usize]);
                BackboneBufs::Attention(AttnBufs {
                    seq_rows,
                    query_rows,
                    query_index,
                    pos: input("seq_pos", rows),
                    keep: input("seq_keep", rows),
                    u: st(s * d),
                    qe: st(bb * d),
                    seq: st(rows * d),
                    qkv: st(rows * 3 * d),
                    scores: st(att),
                    probs: st(att),
                    ctx: st(rows * d),
                    cq: st(bb * d),
                    proj: st(bb * d),
                    d_cq: st(bb * d),
                    d_ctx: st(rows * d),
                    d_scores: st(att),
                    d_qkv: st(rows * 3 * d),
                    d_seq: st(rows * d),
                    d_u: st(s * d),
                    d_qa: st(bb * d),
                    d_qe: st(bb * d),
                })
            }
        })
    }

    /// Upload the batch's per-visit inputs.
    pub(super) fn write(&self, gpu: &Gpu, hb: &HostBatch) {
        match self {
            BackboneBufs::State(_) => {}
            BackboneBufs::Attention(a) => {
                gpu.write_f32(&a.pos, &hb.seq_pos);
                gpu.write(&a.keep, &hb.seq_keep);
            }
            BackboneBufs::Stack(k) => k.write(gpu, hb),
        }
    }

    /// Buffers the forward pass must find zeroed (it accumulates into them).
    pub(super) fn forward_cleared(&self) -> Vec<&DeviceBuffer> {
        match self {
            BackboneBufs::Stack(k) => k.forward_cleared(),
            _ => Vec::new(),
        }
    }

    /// Buffers the backward pass must find zeroed.
    pub(super) fn cleared(&self) -> Vec<&DeviceBuffer> {
        match self {
            BackboneBufs::State(_) => Vec::new(),
            BackboneBufs::Attention(a) => vec![&a.d_ctx],
            BackboneBufs::Stack(k) => k.cleared(),
        }
    }
}

impl Horizon {
    /// From the encoder's output to the state `z` the heads read: the summary
    /// row of each subject, or what its visits leave at the prediction time.
    pub(super) fn state_forward_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        match &self.bb {
            None => vec![g.step(EMBED, &[&i.summary_rows, &self.xf, &self.z], &[d, b], b * d)],
            Some(BackboneBufs::State(v)) => vec![
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
            ],
            Some(BackboneBufs::Attention(a)) => self.attention_forward_steps(a),
            Some(BackboneBufs::Stack(k)) => self.stack_forward_steps(k),
        }
    }

    fn attention(&self) -> Bidir {
        let d = self.cfg.d_model;
        Bidir {
            b: self.b,
            t: self.cfg.visits + 1,
            n_heads: self.cfg.n_heads,
            head_dim: d / self.cfg.n_heads,
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    /// Rotate the q and k regions of `qkv` (`rows` rows) by each row's time
    /// in `pos` (`dir` 1), or back (`dir` -1, the adjoint).
    pub(super) fn rope(
        &self,
        pos: &DeviceBuffer,
        qkv: &DeviceBuffer,
        rows: u32,
        dir: f32,
    ) -> [Step; 2] {
        let d = self.cfg.d_model;
        let (h, hd) = (self.cfg.n_heads, d / self.cfg.n_heads);
        let threads = rows * h * (hd / 2);
        let step = |off: u32| {
            self.gpu.step(
                ROPE_POS,
                &[pos, qkv],
                &[
                    rows,
                    h,
                    hd,
                    3 * d,
                    off,
                    gpu_core::f(ROPE_THETA),
                    gpu_core::f(dir),
                ],
                threads,
            )
        };
        [step(0), step(d)]
    }

    fn attention_forward_steps(&self, a: &AttnBufs) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let t = self.cfg.visits + 1;
        let rows = b * t;
        let att = self.attention();
        let mut steps = vec![
            g.step(EMBED, &[&i.summary_rows, &self.xf, &a.u], &[d, s], s * d),
            g.step(
                ROW_SCATTER,
                &[&a.seq_rows, &a.u, &a.seq],
                &[s, d, rows],
                s * d,
            ),
            g.step(
                EMBED,
                &[&a.query_index, self.w("visit.query"), &a.qe],
                &[d, b],
                b * d,
            ),
            g.step(
                ROW_SCATTER,
                &[&a.query_rows, &a.qe, &a.seq],
                &[b, d, rows],
                b * d,
            ),
            self.mm(
                &a.seq,
                self.w("visit.attn.qkv.weight"),
                &a.qkv,
                rows,
                d,
                3 * d,
            ),
            g.step(
                BIAS_ADD,
                &[&a.qkv, self.w("visit.attn.qkv.bias")],
                &[rows, 3 * d],
                rows * 3 * d,
            ),
        ];
        steps.extend(self.rope(&a.pos, &a.qkv, rows, 1.0));
        let mut attn = block::bidir_fwd(g, &BIDIR, &att, &a.qkv, &a.scores, &a.probs, &a.ctx);
        // Unused visit slots are never keys.
        attn.insert(
            1,
            g.step(
                KEYPAD,
                &[&a.keep, &a.scores],
                &[b, self.cfg.n_heads, t],
                b * self.cfg.n_heads * t * t,
            ),
        );
        steps.extend(attn);
        steps.extend([
            g.step(EMBED, &[&a.query_rows, &a.ctx, &a.cq], &[d, b], b * d),
            self.mm(&a.cq, self.w("visit.attn.out.weight"), &a.proj, b, d, d),
            g.step(
                BIAS_ADD,
                &[&a.proj, self.w("visit.attn.out.bias")],
                &[b, d],
                b * d,
            ),
            g.step(ADD2, &[&a.qe, &a.proj, &self.z], &[b * d], b * d),
        ]);
        steps
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
        let v = match &self.bb {
            None => {
                return vec![g.step(
                    ROW_SCATTER,
                    &[&i.summary_rows, &self.d_z, &self.d_xf],
                    &[b, d, bn],
                    b * d,
                )]
            }
            Some(BackboneBufs::Attention(a)) => return self.attention_backward_steps(a),
            Some(BackboneBufs::Stack(k)) => return self.stack_backward_steps(k),
            Some(BackboneBufs::State(v)) => v,
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
        steps.extend(self.mm_dw(&v.d_xg, &v.u, gr("visit.in.weight"), s, d, 2 * d));
        steps.extend([
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

    fn attention_backward_steps(&self, a: &AttnBufs) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let rows = b * (self.cfg.visits + 1);
        let bn = s * self.cfg.max_tokens;
        // z = qe + proj: the out-projection, then the query rows of d_ctx
        // (the rest stay zero: only the query's context is read).
        let mut steps: Vec<Step> = self
            .bias_grad(&self.d_z, gr("visit.attn.out.bias"), b, d)
            .into();
        steps.extend(self.mm_dw(&self.d_z, &a.cq, gr("visit.attn.out.weight"), b, d, d));
        steps.extend([
            self.mm_dx(
                &self.d_z,
                self.w("visit.attn.out.weight"),
                &a.d_cq,
                b,
                d,
                d,
                0,
            ),
            g.step(
                ROW_SCATTER,
                &[&a.query_rows, &a.d_cq, &a.d_ctx],
                &[b, d, rows],
                b * d,
            ),
        ]);
        steps.extend(block::bidir_bwd(
            g,
            &BIDIR,
            &self.attention(),
            &a.qkv,
            &a.probs,
            &a.d_ctx,
            &a.d_scores,
            &a.d_qkv,
        ));
        steps.extend(self.rope(&a.pos, &a.d_qkv, rows, -1.0));
        steps.extend(self.bias_grad(&a.d_qkv, gr("visit.attn.qkv.bias"), rows, 3 * d));
        steps.extend(self.mm_dw(
            &a.d_qkv,
            &a.seq,
            gr("visit.attn.qkv.weight"),
            rows,
            d,
            3 * d,
        ));
        steps.extend([
            self.mm_dx(
                &a.d_qkv,
                self.w("visit.attn.qkv.weight"),
                &a.d_seq,
                rows,
                d,
                3 * d,
                0,
            ),
            // The sequence rows back to the visits and to the query.
            g.step(EMBED, &[&a.seq_rows, &a.d_seq, &a.d_u], &[d, s], s * d),
            g.step(EMBED, &[&a.query_rows, &a.d_seq, &a.d_qa], &[d, b], b * d),
            g.step(ADD2, &[&a.d_qa, &self.d_z, &a.d_qe], &[b * d], b * d),
            g.step(
                EMB_BWD,
                &[&a.query_index, &a.d_qe, gr("visit.query")],
                &[b, d, 1],
                d,
            ),
            g.step(
                ROW_SCATTER,
                &[&i.summary_rows, &a.d_u, &self.d_xf],
                &[s, d, bn],
                s * d,
            ),
        ]);
        steps
    }
}
