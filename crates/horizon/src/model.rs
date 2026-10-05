// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device model: set encoder, hazard head and value head, forward and
//! backward as WGSL dispatch lists (the engine's hand-written-backward idiom),
//! gradient-checked through the blanket `gradcheck::CheckModel`.
//!
//! ```text
//! e   = gamma[tok] * (value_bins @ Wv^T) + beta[tok] + time_bins @ Wt^T     [B*N, D]
//! x   = L x pre-LN { x + Attn(LN x) masked to real keys;  x + MLP(LN x) }
//! xf  = LN_f(x)
//! value head:  (mu, log sigma) = xf @ Wval^T + b     scored where a value was hidden
//! z   = xf[summary row of each subject]                                        [B, D]
//! h   = GELU( (z @ Ws^T + bs)[subject of row] + time_features @ Wtime^T )     [B*P, R]
//! log lambda = h @ Wcode^T + bcode                                             [B*P, K]
//! loss = sum pexp_nll(log lambda, event, exposure, weight) + sum gauss_cens_nll(...)
//! ```
//!
//! The summary token attends over the subject's whole set and is the state the
//! hazards read; a hazard depends on the state, the subject's age and the
//! calendar at each piece (two clocks) and the piece itself.

use std::collections::HashMap;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, Bidir, BidirIds, LayerNormIds};
use optim::Optim;
use paramstore::ParamStore;
use serde_json::Value;

use crate::batch::HostBatch;
use crate::config::HorizonConfig;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const BIAS_ADD: usize = 2;
const BIAS_GRAD: usize = 3;
const MATMUL_DX: usize = 4;
const MATMUL_DW: usize = 5;
const MUL: usize = 6;
const ADD2: usize = 7;
const LAYERNORM: usize = 8;
const LN_STATS: usize = 9;
const LN_DX: usize = 10;
const LN_DGAMMA: usize = 11;
const LN_DBETA: usize = 12;
const LAYERNORM_ROWS: usize = 13;
const LN_STATS_ROWS: usize = 14;
const LN_DX_ROWS: usize = 15;
const KEYPAD: usize = 23;
const GELU: usize = 24;
const GELU_BWD: usize = 25;
const EMB_BWD: usize = 26;
const ROW_SCATTER: usize = 27;
const PEXP_VALUE: usize = 28;
const PEXP_GRAD: usize = 29;
const GAUSS_VALUE: usize = 30;
const GAUSS_GRAD: usize = 31;
const GRADNORM_SQ: usize = 32;
const GRAD_SCALE: usize = 33;
const ADAMW: usize = 34;
const CLIP_COEF: usize = 35;
const GRAD_SCALE_BUF: usize = 36;

const BIDIR: BidirIds = BidirIds {
    scores: 16,
    softmax: 17,
    apply: 18,
    dscores: 19,
    dv: 20,
    dq: 21,
    dk: 22,
};
const LN_IDS: LayerNormIds = LayerNormIds {
    layernorm: LAYERNORM,
    layernorm_rows: Some(LAYERNORM_ROWS),
    ln_stats: LN_STATS,
    ln_stats_rows: Some(LN_STATS_ROWS),
    layernorm_dx: LN_DX,
    layernorm_dx_rows: Some(LN_DX_ROWS),
};
const LN_EPS: f32 = 1e-5;

/// The kernels this model dispatches, in index order.
pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("attn_keypad_mask", kernels::ATTN_KEYPAD_MASK),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("row_scatter", kernels::ROW_SCATTER),
    ("pexp_nll_value", kernels::PEXP_NLL_VALUE),
    ("pexp_nll_grad", kernels::PEXP_NLL_GRAD),
    ("gauss_cens_nll_value", kernels::GAUSS_CENS_NLL_VALUE),
    ("gauss_cens_nll_grad", kernels::GAUSS_CENS_NLL_GRAD),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // Cooperative grad-norm, resolved by name by `optim::Optim`.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

struct Layer {
    ln1_out: DeviceBuffer,
    qkv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    ln2_out: DeviceBuffer,
    up_pre: DeviceBuffer,
    up: DeviceBuffer,
}

/// Batch inputs, written once per batch.
struct Inputs {
    token_ids: DeviceBuffer,
    keep: DeviceBuffer,
    value_bins: DeviceBuffer,
    time_bins: DeviceBuffer,
    summary_rows: DeviceBuffer,
    value_target: DeviceBuffer,
    value_state: DeviceBuffer,
    value_weight: DeviceBuffer,
    piece_subject: DeviceBuffer,
    time_features: DeviceBuffer,
    event: DeviceBuffer,
    exposure: DeviceBuffer,
    subject_weight: DeviceBuffer,
}

/// The timeline model on one device, sized for `b` subjects per batch.
pub struct Horizon {
    /// The device.
    pub gpu: Gpu,
    /// Its configuration.
    pub cfg: HorizonConfig,
    /// Its parameters (weights, gradients, moments).
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    inp: Inputs,
    gam: DeviceBuffer,
    bet: DeviceBuffer,
    phi: DeviceBuffer,
    gp: DeviceBuffer,
    e1: DeviceBuffer,
    te: DeviceBuffer,
    res: Vec<DeviceBuffer>,
    layers: Vec<Layer>,
    proj: DeviceBuffer,
    ffn_out: DeviceBuffer,
    xf: DeviceBuffer,
    vpred: DeviceBuffer,
    vloss: DeviceBuffer,
    z: DeviceBuffer,
    az: DeviceBuffer,
    rep: DeviceBuffer,
    cf: DeviceBuffer,
    h0: DeviceBuffer,
    h: DeviceBuffer,
    loglam: DeviceBuffer,
    hloss: DeviceBuffer,
    // backward
    d_loglam: DeviceBuffer,
    d_h: DeviceBuffer,
    d_h0: DeviceBuffer,
    d_az: DeviceBuffer,
    d_z: DeviceBuffer,
    d_vpred: DeviceBuffer,
    d_xf: DeviceBuffer,
    dres: Vec<DeviceBuffer>,
    d_branch: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_up: DeviceBuffer,
    d_up_pre: DeviceBuffer,
    d_gam: DeviceBuffer,
    d_phi: DeviceBuffer,
    ln_mean: DeviceBuffer,
    ln_inv: DeviceBuffer,
    fwd: Vec<Step>,
    bwd: Vec<Step>,
}

impl Horizon {
    /// Build on the device `BRAIN_DEVICE` selects.
    pub fn new(cfg: HorizonConfig, b: u32, init: &HashMap<String, Vec<f32>>) -> Horizon {
        Horizon::new_on(Gpu::new(PIPELINES), cfg, b, init)
    }

    /// Build on an existing device handle (created with [`PIPELINES`]).
    pub fn new_on(
        gpu: Gpu,
        cfg: HorizonConfig,
        b: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> Horizon {
        cfg.validate().expect("horizon config");
        let ps = ParamStore::new(&gpu, cfg.param_list(), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);
        let (n, d, ff, r) = (
            cfg.max_tokens as u64,
            cfg.d_model as u64,
            cfg.d_ff as u64,
            cfg.rank as u64,
        );
        let (p, k, nf) = (
            cfg.pieces() as u64,
            cfg.n_codes as u64,
            cfg.time_features() as u64,
        );
        let (bn, bp, bb) = (b as u64 * n, b as u64 * p, b as u64);
        let bhnn = bb * cfg.n_heads as u64 * n * n;
        let st = |x: u64| gpu.storage(x);
        let input = |label: &str, words: u64| {
            gpu.buffer(label, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
        };
        let inp = Inputs {
            token_ids: input("token_ids", bn),
            keep: input("keep", bn),
            value_bins: input("value_bins", bn * cfg.value_table() as u64),
            time_bins: input("time_bins", bn * cfg.time_table() as u64),
            summary_rows: input("summary_rows", bb),
            value_target: input("value_target", bn),
            value_state: input("value_state", bn),
            value_weight: input("value_weight", bn),
            piece_subject: input("piece_subject", bp),
            time_features: input("time_features", bp * nf),
            event: input("event", bp * k),
            exposure: input("exposure", bp * k),
            subject_weight: input("subject_weight", bb),
        };
        let layers = (0..cfg.n_layers)
            .map(|_| Layer {
                ln1_out: st(bn * d),
                qkv: st(bn * 3 * d),
                scores: st(bhnn),
                probs: st(bhnn),
                ctx: st(bn * d),
                xmid: st(bn * d),
                ln2_out: st(bn * d),
                up_pre: st(bn * ff),
                up: st(bn * ff),
            })
            .collect();
        let mut m = Horizon {
            b,
            inp,
            gam: st(bn * d),
            bet: st(bn * d),
            phi: st(bn * d),
            gp: st(bn * d),
            e1: st(bn * d),
            te: st(bn * d),
            res: (0..=cfg.n_layers).map(|_| st(bn * d)).collect(),
            layers,
            proj: st(bn * d),
            ffn_out: st(bn * d),
            xf: st(bn * d),
            vpred: st(bn * 2),
            vloss: st(bn),
            z: st(bb * d),
            az: st(bb * r),
            rep: st(bp * r),
            cf: st(bp * r),
            h0: st(bp * r),
            h: st(bp * r),
            loglam: st(bp * k),
            hloss: st(bp * k),
            d_loglam: st(bp * k),
            d_h: st(bp * r),
            d_h0: st(bp * r),
            d_az: st(bb * r),
            d_z: st(bb * d),
            d_vpred: st(bn * 2),
            d_xf: st(bn * d),
            dres: (0..=cfg.n_layers).map(|_| st(bn * d)).collect(),
            d_branch: st(bn * d),
            d_tmp: st(bn * d),
            dxmid: st(bn * d),
            d_ctx: st(bn * d),
            d_scores: st(bhnn),
            d_qkv: st(bn * 3 * d),
            d_up: st(bn * ff),
            d_up_pre: st(bn * ff),
            d_gam: st(bn * d),
            d_phi: st(bn * d),
            ln_mean: st(bn),
            ln_inv: st(bn),
            fwd: Vec::new(),
            bwd: Vec::new(),
            cfg,
            ps,
            opt,
            gpu,
        };
        m.fwd = m.forward_steps();
        m.bwd = m.backward_steps();
        m
    }

    /// Subjects per batch this model was sized for.
    pub fn batch_size(&self) -> u32 {
        self.b
    }

    /// Upload one assembled batch.
    pub fn set_batch(&self, hb: &HostBatch) {
        let g = &self.gpu;
        let i = &self.inp;
        assert_eq!(
            hb.summary_rows.len(),
            self.b as usize,
            "batch assembled for {} slots, model sized for {}",
            hb.summary_rows.len(),
            self.b
        );
        g.write(&i.token_ids, &hb.token_ids);
        g.write(&i.keep, &hb.keep);
        g.write_f32(&i.value_bins, &hb.value_bins);
        g.write_f32(&i.time_bins, &hb.time_bins);
        g.write(&i.summary_rows, &hb.summary_rows);
        g.write_f32(&i.value_target, &hb.value_target);
        g.write(&i.value_state, &hb.value_state);
        g.write_f32(&i.value_weight, &hb.value_weight);
        g.write(&i.piece_subject, &hb.piece_subject);
        g.write_f32(&i.time_features, &hb.time_features);
        g.write_f32(&i.event, &hb.event);
        g.write_f32(&i.exposure, &hb.exposure);
        g.write_f32(&i.subject_weight, &hb.subject_weight);
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn bidir(&self) -> Bidir {
        let d = self.cfg.d_model;
        Bidir {
            b: self.b,
            t: self.cfg.max_tokens,
            n_heads: self.cfg.n_heads,
            head_dim: d / self.cfg.n_heads,
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let (n, d, ff, r) = (c.max_tokens, c.d_model, c.d_ff, c.rank);
        let (p, k, nf) = (c.pieces(), c.n_codes, c.time_features());
        let (bn, bp, b) = (self.b * n, self.b * p, self.b);
        let (vt, tt) = (c.value_table(), c.time_table());
        let mut s = vec![
            g.step(
                EMBED,
                &[&i.token_ids, self.w("tok.gamma"), &self.gam],
                &[d, bn],
                bn * d,
            ),
            g.step(
                EMBED,
                &[&i.token_ids, self.w("tok.beta"), &self.bet],
                &[d, bn],
                bn * d,
            ),
            g.step(
                MATMUL,
                &[&i.value_bins, self.w("value_bins.weight"), &self.phi],
                &[bn, vt, d],
                bn * d,
            ),
            g.step(MUL, &[&self.gam, &self.phi, &self.gp], &[bn * d], bn * d),
            g.step(ADD2, &[&self.gp, &self.bet, &self.e1], &[bn * d], bn * d),
            g.step(
                MATMUL,
                &[&i.time_bins, self.w("time_bins.weight"), &self.te],
                &[bn, tt, d],
                bn * d,
            ),
            g.step(ADD2, &[&self.e1, &self.te, &self.res[0]], &[bn * d], bn * d),
        ];
        let a = self.bidir();
        for (l, lb) in self.layers.iter().enumerate() {
            let pn = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::layernorm_fwd(
                g,
                &LN_IDS,
                &self.res[l],
                self.w(&pn("ln1.weight")),
                self.w(&pn("ln1.bias")),
                &lb.ln1_out,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.ln1_out, self.w(&pn("attn.qkv.weight")), &lb.qkv],
                &[bn, d, 3 * d],
                bn * 3 * d,
            ));
            s.push(g.step(
                BIAS_ADD,
                &[&lb.qkv, self.w(&pn("attn.qkv.bias"))],
                &[bn, 3 * d],
                bn * 3 * d,
            ));
            let mut attn = block::bidir_fwd(g, &BIDIR, &a, &lb.qkv, &lb.scores, &lb.probs, &lb.ctx);
            // Padding is never a key: the mask goes between scores and softmax.
            attn.insert(
                1,
                g.step(
                    KEYPAD,
                    &[&i.keep, &lb.scores],
                    &[b, c.n_heads, n],
                    b * c.n_heads * n * n,
                ),
            );
            s.extend(attn);
            s.push(g.step(
                MATMUL,
                &[&lb.ctx, self.w(&pn("attn.out.weight")), &self.proj],
                &[bn, d, d],
                bn * d,
            ));
            s.push(g.step(
                BIAS_ADD,
                &[&self.proj, self.w(&pn("attn.out.bias"))],
                &[bn, d],
                bn * d,
            ));
            s.push(g.step(
                ADD2,
                &[&self.res[l], &self.proj, &lb.xmid],
                &[bn * d],
                bn * d,
            ));
            s.push(block::layernorm_fwd(
                g,
                &LN_IDS,
                &lb.xmid,
                self.w(&pn("ln2.weight")),
                self.w(&pn("ln2.bias")),
                &lb.ln2_out,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.ln2_out, self.w(&pn("ffn.up.weight")), &lb.up_pre],
                &[bn, d, ff],
                bn * ff,
            ));
            s.push(g.step(
                BIAS_ADD,
                &[&lb.up_pre, self.w(&pn("ffn.up.bias"))],
                &[bn, ff],
                bn * ff,
            ));
            s.push(g.step(GELU, &[&lb.up_pre, &lb.up], &[bn * ff], bn * ff));
            s.push(g.step(
                MATMUL,
                &[&lb.up, self.w(&pn("ffn.down.weight")), &self.ffn_out],
                &[bn, ff, d],
                bn * d,
            ));
            s.push(g.step(
                BIAS_ADD,
                &[&self.ffn_out, self.w(&pn("ffn.down.bias"))],
                &[bn, d],
                bn * d,
            ));
            s.push(g.step(
                ADD2,
                &[&lb.xmid, &self.ffn_out, &self.res[l + 1]],
                &[bn * d],
                bn * d,
            ));
        }
        let last = &self.res[c.n_layers as usize];
        s.extend([
            block::layernorm_fwd(
                g,
                &LN_IDS,
                last,
                self.w("ln_f.weight"),
                self.w("ln_f.bias"),
                &self.xf,
                d,
                bn,
                LN_EPS,
            ),
            // value head
            g.step(
                MATMUL,
                &[&self.xf, self.w("value_head.weight"), &self.vpred],
                &[bn, d, 2],
                bn * 2,
            ),
            g.step(
                BIAS_ADD,
                &[&self.vpred, self.w("value_head.bias")],
                &[bn, 2],
                bn * 2,
            ),
            g.step(
                GAUSS_VALUE,
                &[
                    &self.vpred,
                    &i.value_target,
                    &i.value_state,
                    &i.value_weight,
                    &self.vloss,
                ],
                &[bn],
                bn,
            ),
            // hazard head
            g.step(EMBED, &[&i.summary_rows, &self.xf, &self.z], &[d, b], b * d),
            g.step(
                MATMUL,
                &[&self.z, self.w("hazard.state.weight"), &self.az],
                &[b, d, r],
                b * r,
            ),
            g.step(
                BIAS_ADD,
                &[&self.az, self.w("hazard.state.bias")],
                &[b, r],
                b * r,
            ),
            g.step(
                EMBED,
                &[&i.piece_subject, &self.az, &self.rep],
                &[r, bp],
                bp * r,
            ),
            g.step(
                MATMUL,
                &[&i.time_features, self.w("hazard.time.weight"), &self.cf],
                &[bp, nf, r],
                bp * r,
            ),
            g.step(ADD2, &[&self.rep, &self.cf, &self.h0], &[bp * r], bp * r),
            g.step(GELU, &[&self.h0, &self.h], &[bp * r], bp * r),
            g.step(
                MATMUL,
                &[&self.h, self.w("hazard.code.weight"), &self.loglam],
                &[bp, r, k],
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
                &[bp, k, p, gpu_core::f(1.0)],
                bp * k,
            ),
        ]);
        s
    }

    fn backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (n, d, ff, r) = (c.max_tokens, c.d_model, c.d_ff, c.rank);
        let (p, k, nf) = (c.pieces(), c.n_codes, c.time_features());
        let (bn, bp, b) = (self.b * n, self.b * p, self.b);
        let (vt, tt) = (c.value_table(), c.time_table());
        let last = c.n_layers as usize;
        let mut s = vec![
            // hazard head
            g.step(
                PEXP_GRAD,
                &[
                    &self.loglam,
                    &i.event,
                    &i.exposure,
                    &i.subject_weight,
                    &self.d_loglam,
                ],
                &[bp, k, p, gpu_core::f(1.0)],
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
                &[&self.d_loglam, &self.h, gr("hazard.code.weight")],
                &[bp, r, k],
                k * r,
            ),
            g.step(
                MATMUL_DX,
                &[&self.d_loglam, self.w("hazard.code.weight"), &self.d_h],
                &[bp, r, k, 0],
                bp * r,
            ),
            g.step(
                GELU_BWD,
                &[&self.h0, &self.d_h, &self.d_h0],
                &[bp * r],
                bp * r,
            ),
            g.step(
                MATMUL_DW,
                &[&self.d_h0, &i.time_features, gr("hazard.time.weight")],
                &[bp, nf, r],
                r * nf,
            ),
            // d_az is cleared before this list runs (emb_bwd accumulates).
            g.step(
                EMB_BWD,
                &[&i.piece_subject, &self.d_h0, &self.d_az],
                &[bp, r, b],
                b * r,
            ),
            g.step(
                BIAS_GRAD,
                &[&self.d_az, gr("hazard.state.bias")],
                &[b, r],
                r,
            ),
            g.step(
                MATMUL_DW,
                &[&self.d_az, &self.z, gr("hazard.state.weight")],
                &[b, d, r],
                r * d,
            ),
            g.step(
                MATMUL_DX,
                &[&self.d_az, self.w("hazard.state.weight"), &self.d_z],
                &[b, d, r, 0],
                b * d,
            ),
            // value head
            g.step(
                GAUSS_GRAD,
                &[
                    &self.vpred,
                    &i.value_target,
                    &i.value_state,
                    &i.value_weight,
                    &self.d_vpred,
                ],
                &[bn],
                bn,
            ),
            g.step(
                BIAS_GRAD,
                &[&self.d_vpred, gr("value_head.bias")],
                &[bn, 2],
                2,
            ),
            g.step(
                MATMUL_DW,
                &[&self.d_vpred, &self.xf, gr("value_head.weight")],
                &[bn, d, 2],
                2 * d,
            ),
            g.step(
                MATMUL_DX,
                &[&self.d_vpred, self.w("value_head.weight"), &self.d_xf],
                &[bn, d, 2, 0],
                bn * d,
            ),
            // The summary rows carry no value target (d_xf is zero there), so
            // their gradient is exactly the pooled state's.
            g.step(
                ROW_SCATTER,
                &[&i.summary_rows, &self.d_z, &self.d_xf],
                &[b, d, bn],
                b * d,
            ),
            // final norm
            block::ln_stats_fwd(
                g,
                &LN_IDS,
                &self.res[last],
                &self.ln_mean,
                &self.ln_inv,
                d,
                bn,
                LN_EPS,
            ),
            g.step(
                LN_DGAMMA,
                &[
                    &self.d_xf,
                    &self.res[last],
                    &self.ln_mean,
                    &self.ln_inv,
                    gr("ln_f.weight"),
                ],
                &[d, bn],
                d,
            ),
            g.step(LN_DBETA, &[&self.d_xf, gr("ln_f.bias")], &[d, bn], d),
            block::layernorm_dx_bwd(
                g,
                &LN_IDS,
                &self.res[last],
                self.w("ln_f.weight"),
                &self.d_xf,
                &self.dres[last],
                d,
                bn,
                LN_EPS,
            ),
        ];
        let a = self.bidir();
        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");
            // MLP
            s.push(g.step(
                BIAS_GRAD,
                &[&self.dres[l + 1], gr(&pn("ffn.down.bias"))],
                &[bn, d],
                d,
            ));
            s.push(g.step(
                MATMUL_DW,
                &[&self.dres[l + 1], &lb.up, gr(&pn("ffn.down.weight"))],
                &[bn, ff, d],
                d * ff,
            ));
            s.push(g.step(
                MATMUL_DX,
                &[
                    &self.dres[l + 1],
                    self.w(&pn("ffn.down.weight")),
                    &self.d_up,
                ],
                &[bn, ff, d, 0],
                bn * ff,
            ));
            s.push(g.step(
                GELU_BWD,
                &[&lb.up_pre, &self.d_up, &self.d_up_pre],
                &[bn * ff],
                bn * ff,
            ));
            s.push(g.step(
                BIAS_GRAD,
                &[&self.d_up_pre, gr(&pn("ffn.up.bias"))],
                &[bn, ff],
                ff,
            ));
            s.push(g.step(
                MATMUL_DW,
                &[&self.d_up_pre, &lb.ln2_out, gr(&pn("ffn.up.weight"))],
                &[bn, d, ff],
                ff * d,
            ));
            s.push(g.step(
                MATMUL_DX,
                &[&self.d_up_pre, self.w(&pn("ffn.up.weight")), &self.d_branch],
                &[bn, d, ff, 0],
                bn * d,
            ));
            s.push(block::ln_stats_fwd(
                g,
                &LN_IDS,
                &lb.xmid,
                &self.ln_mean,
                &self.ln_inv,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                LN_DGAMMA,
                &[
                    &self.d_branch,
                    &lb.xmid,
                    &self.ln_mean,
                    &self.ln_inv,
                    gr(&pn("ln2.weight")),
                ],
                &[d, bn],
                d,
            ));
            s.push(g.step(
                LN_DBETA,
                &[&self.d_branch, gr(&pn("ln2.bias"))],
                &[d, bn],
                d,
            ));
            s.push(block::layernorm_dx_bwd(
                g,
                &LN_IDS,
                &lb.xmid,
                self.w(&pn("ln2.weight")),
                &self.d_branch,
                &self.d_tmp,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                ADD2,
                &[&self.dres[l + 1], &self.d_tmp, &self.dxmid],
                &[bn * d],
                bn * d,
            ));
            // attention
            s.push(g.step(
                BIAS_GRAD,
                &[&self.dxmid, gr(&pn("attn.out.bias"))],
                &[bn, d],
                d,
            ));
            s.push(g.step(
                MATMUL_DW,
                &[&self.dxmid, &lb.ctx, gr(&pn("attn.out.weight"))],
                &[bn, d, d],
                d * d,
            ));
            s.push(g.step(
                MATMUL_DX,
                &[&self.dxmid, self.w(&pn("attn.out.weight")), &self.d_ctx],
                &[bn, d, d, 0],
                bn * d,
            ));
            s.extend(block::bidir_bwd(
                g,
                &BIDIR,
                &a,
                &lb.qkv,
                &lb.probs,
                &self.d_ctx,
                &self.d_scores,
                &self.d_qkv,
            ));
            s.push(g.step(
                BIAS_GRAD,
                &[&self.d_qkv, gr(&pn("attn.qkv.bias"))],
                &[bn, 3 * d],
                3 * d,
            ));
            s.push(g.step(
                MATMUL_DW,
                &[&self.d_qkv, &lb.ln1_out, gr(&pn("attn.qkv.weight"))],
                &[bn, d, 3 * d],
                3 * d * d,
            ));
            s.push(g.step(
                MATMUL_DX,
                &[&self.d_qkv, self.w(&pn("attn.qkv.weight")), &self.d_branch],
                &[bn, d, 3 * d, 0],
                bn * d,
            ));
            s.push(block::ln_stats_fwd(
                g,
                &LN_IDS,
                &self.res[l],
                &self.ln_mean,
                &self.ln_inv,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                LN_DGAMMA,
                &[
                    &self.d_branch,
                    &self.res[l],
                    &self.ln_mean,
                    &self.ln_inv,
                    gr(&pn("ln1.weight")),
                ],
                &[d, bn],
                d,
            ));
            s.push(g.step(
                LN_DBETA,
                &[&self.d_branch, gr(&pn("ln1.bias"))],
                &[d, bn],
                d,
            ));
            s.push(block::layernorm_dx_bwd(
                g,
                &LN_IDS,
                &self.res[l],
                self.w(&pn("ln1.weight")),
                &self.d_branch,
                &self.d_tmp,
                d,
                bn,
                LN_EPS,
            ));
            s.push(g.step(
                ADD2,
                &[&self.dxmid, &self.d_tmp, &self.dres[l]],
                &[bn * d],
                bn * d,
            ));
        }
        // token embedding: e = gamma[tok] * phi + beta[tok] + te
        let d0 = &self.dres[0];
        s.extend([
            g.step(
                MATMUL_DW,
                &[d0, &i.time_bins, gr("time_bins.weight")],
                &[bn, tt, d],
                d * tt,
            ),
            g.step(
                EMB_BWD,
                &[&i.token_ids, d0, gr("tok.beta")],
                &[bn, d, c.vocab],
                c.vocab * d,
            ),
            g.step(MUL, &[d0, &self.phi, &self.d_gam], &[bn * d], bn * d),
            g.step(
                EMB_BWD,
                &[&i.token_ids, &self.d_gam, gr("tok.gamma")],
                &[bn, d, c.vocab],
                c.vocab * d,
            ),
            g.step(MUL, &[d0, &self.gam, &self.d_phi], &[bn * d], bn * d),
            g.step(
                MATMUL_DW,
                &[&self.d_phi, &i.value_bins, gr("value_bins.weight")],
                &[bn, vt, d],
                d * vt,
            ),
        ]);
        s
    }

    /// Submit the forward pass (no readback).
    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd);
    }

    /// The batch loss of the last forward: the weighted mean event NLL plus
    /// the weighted value NLL.
    pub fn loss(&self) -> f32 {
        let (event, value) = self.loss_parts();
        event + value
    }

    /// `(event NLL, value NLL)` of the last forward.
    pub fn loss_parts(&self) -> (f32, f32) {
        let c = &self.cfg;
        let bpk = (self.b * c.pieces() * c.n_codes) as usize;
        let bn = (self.b * c.max_tokens) as usize;
        let event: f64 = self
            .gpu
            .read(&self.hloss, bpk)
            .iter()
            .map(|&x| x as f64)
            .sum();
        let value: f64 = self
            .gpu
            .read(&self.vloss, bn)
            .iter()
            .map(|&x| x as f64)
            .sum();
        (event as f32, value as f32)
    }

    /// Forward and loss.
    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    /// Accumulate gradients of the last forward's loss.
    pub fn backward(&self) {
        self.gpu.submit(&[&self.d_az], &self.bwd);
    }

    /// Log-hazards of the last forward, `[b, pieces, codes]`.
    pub fn read_log_hazards(&self) -> Vec<f32> {
        self.gpu.read(
            &self.loglam,
            (self.b * self.cfg.pieces() * self.cfg.n_codes) as usize,
        )
    }

    /// The summary state of the last forward, `[b, d_model]`.
    pub fn read_state(&self) -> Vec<f32> {
        self.gpu.read(&self.z, (self.b * self.cfg.d_model) as usize)
    }

    /// The value head's `(mu, log sigma)` per token row of the last forward.
    pub fn read_value_predictions(&self) -> Vec<f32> {
        self.gpu
            .read(&self.vpred, (self.b * self.cfg.max_tokens * 2) as usize)
    }

    /// Zero every gradient.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// One AdamW step.
    pub fn adamw_step(
        &self,
        t: u32,
        lr: f32,
        wd: f32,
        adam: model::Adam,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        self.opt
            .step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale);
    }

    /// Write the weights and the configuration to a safetensors checkpoint.
    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| {
                (
                    name.clone(),
                    vec![self.ps.numel(name) as u64],
                    self.ps.read_weight(&self.gpu, name),
                )
            })
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }

    /// Load a checkpoint sized for batches of `b`.
    pub fn load(path: &str, b: u32) -> Result<Horizon, String> {
        let c = checkpoint::load(path);
        let cfg = HorizonConfig::from_json(&c.header["config"])?;
        Ok(Horizon::new(cfg, b, &c.by_role("")))
    }
}

impl model::ModelConfig for HorizonConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        HorizonConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        HorizonConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        HorizonConfig::from_json(v).expect("horizon config in checkpoint")
    }
    /// No token head over a vocabulary of outputs.
    fn vocab(&self) -> u32 {
        0
    }
    /// The per-subject token capacity.
    fn block_size(&self) -> u32 {
        self.max_tokens
    }
    /// The configuration is fixed by the fitted vocabulary, not by a text dataset.
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

impl model::Model for Horizon {
    type Config = HorizonConfig;

    fn new(cfg: HorizonConfig, b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Horizon::new(cfg, b, init)
    }
    fn init_weights(cfg: &HorizonConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &HorizonConfig {
        &self.cfg
    }
    /// A timeline batch is not one of the generic shapes; use [`Horizon::set_batch`].
    fn set_batch(&self, _batch: model::Batch) {
        panic!("horizon::Horizon takes a timeline batch: call Horizon::set_batch with a HostBatch");
    }
    fn forward(&self) -> f32 {
        Horizon::forward(self)
    }
    fn backward(&self) {
        Horizon::backward(self)
    }
    fn zero_grads(&self) {
        Horizon::zero_grads(self)
    }
    fn adamw_step(
        &self,
        t: u32,
        lr: f32,
        wd: f32,
        adam: model::Adam,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        Horizon::adamw_step(self, t, lr, wd, adam, clip, extra_scale)
    }
    fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.w(name), data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, path: &str) {
        Horizon::save(self, path)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}
