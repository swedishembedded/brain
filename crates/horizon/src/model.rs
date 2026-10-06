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

mod additive;
mod backbone;
mod forecast;
mod stack;

use crate::batch::HostBatch;
use crate::config::HorizonConfig;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const BIAS_ADD: usize = 2;
const MATMUL_DX: usize = 3;
const MATMUL_DW: usize = 4;
const MUL: usize = 5;
const ADD2: usize = 6;
const LAYERNORM: usize = 7;
const LN_STATS: usize = 8;
const LN_DX: usize = 9;
const LAYERNORM_ROWS: usize = 12;
const LN_STATS_ROWS: usize = 13;
const LN_DX_ROWS: usize = 14;
const KEYPAD: usize = 22;
const GELU: usize = 23;
const GELU_BWD: usize = 24;
const EMB_BWD: usize = 25;
const ROW_SCATTER: usize = 26;
const PEXP_VALUE: usize = 27;
const PEXP_GRAD: usize = 28;
const GAUSS_VALUE: usize = 29;
const GAUSS_GRAD: usize = 30;
const GRADNORM_SQ: usize = 31;
const GRAD_SCALE: usize = 32;
const ADAMW: usize = 33;
const CLIP_COEF: usize = 34;
const GRAD_SCALE_BUF: usize = 35;
const MATMUL_REG3: usize = 36;
const MATMUL_DX_REG: usize = 37;
const MATMUL_DW_REG: usize = 38;
const BIAS_GRAD_PART: usize = 39;
const BIAS_GRAD_FINAL: usize = 40;
const SEGMENT_SUM: usize = 41;
const SEGMENT_BCAST: usize = 42;
const CT_SCAN: usize = 43;
const CT_SCAN_BWD: usize = 44;
const ROPE_POS: usize = 45;
const MATMUL_DW_SPLITK: usize = 46;
const DW_SPLITK_REDUCE: usize = 47;
const LN_DGAMMA_PART: usize = 48;
const EMB_BWD_PART: usize = 49;
// The Gated DeltaNet recurrence (`model::gdn`) and its two gates.
const BMM: usize = 50;
const BMM_ACC: usize = 51;
const GDN_CHUNK_CUMSUM_STEP: usize = 52;
const GDN_DECAY_MASK: usize = 53;
const GDN_MASK_STRICT_LOWER: usize = 54;
const GDN_UT_STEP: usize = 55;
const GDN_ADD_IDENTITY: usize = 56;
const SCALE_ROW: usize = 57;
const GDN_ROW_SCALE_OFF: usize = 58;
const GDN_DECAY_SCALE: usize = 59;
const GDN_STATE_DECAY: usize = 60;
const EXP: usize = 61;
const SUB: usize = 62;
const REGION_COPY: usize = 63;
const SPLICE_ADD: usize = 64;
const ROW_DOT: usize = 65;
const SCALE_ADD: usize = 66;
const GDN_CHUNK_REVERSE_CUMSUM_STEP: usize = 67;
const GDN_UT_BWD_DATTN0: usize = 68;
const GDN_UT_BWD_DTMAT: usize = 69;
const GDN_MASK_STRICT_LOWER_BWD: usize = 70;
const GDN_DECAY_MASK_BWD: usize = 71;
const GDN_DECAY_SCALE_BWD: usize = 72;
const GDN_DECAY_SCALE_BWD_LAST: usize = 73;
const GDN_STATE_DECAY_BWD_DSCALE: usize = 74;
const GDN_UT_FWD: usize = 75;
const BMM_TILED: usize = 76;
const GDN_LAYOUT_PERMUTE: usize = 77;
const L2NORM_SCALE: usize = 78;
const L2NORM_SCALE_DX: usize = 79;
const GDN_GAP_GATE: usize = 80;
const GDN_GAP_GATE_BWD: usize = 81;
/// Partial gradients the token-table backward may hold at once: bounds its
/// row blocks for a large vocabulary.
const EMB_PART_ELEMS: u64 = 1 << 22;
/// Row blocks of the token-table backward at most.
const EMB_PART_MAX_BLOCKS: u64 = 64;
/// Workgroups the split-K weight gradient aims to occupy: a few per
/// compute unit of a large device, so the contraction over tens of thousands
/// of rows is spread across the whole card.
const DW_SPLITK_TARGET_WGS: u32 = 512;
/// Slices of the split-K weight gradient at most (bounds its scratch).
const DW_SPLITK_MAX_SLICES: u32 = 128;
/// Row chunks per column of the two-stage bias gradient: one serial walk
/// over every row per column is a handful of threads on a wide device.
const BIAS_GRAD_CHUNKS: u32 = 64;

const BIDIR: BidirIds = BidirIds {
    scores: 15,
    softmax: 16,
    apply: 17,
    dscores: 18,
    dv: 19,
    dq: 20,
    dk: 21,
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
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    // Unused since the two-stage LayerNorm gradients; kept so the indices
    // after them stay put.
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
    // The tiled GEMMs and the two-stage bias gradient, chosen per shape.
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_grad_part", kernels::BIAS_GRAD_PART),
    ("bias_grad_final", kernels::BIAS_GRAD_FINAL),
    ("segment_sum_rows", kernels::SEGMENT_SUM_ROWS),
    ("segment_bcast_rows", kernels::SEGMENT_BCAST_ROWS),
    ("ct_state_scan", kernels::CT_STATE_SCAN),
    ("ct_state_scan_bwd", kernels::CT_STATE_SCAN_BWD),
    ("rope_pos", kernels::ROPE_POS),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
    ("layernorm_dgamma_part", kernels::LAYERNORM_DGAMMA_PART),
    ("emb_bwd_part", kernels::EMB_BWD_PART),
    // The Gated DeltaNet mixer of the visit stack.
    ("bmm", kernels::BMM),
    ("bmm_acc", kernels::BMM_ACC),
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP),
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK),
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER),
    ("gdn_ut_step", kernels::GDN_UT_STEP),
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY),
    ("scale_row", kernels::SCALE_ROW),
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF),
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE),
    ("gdn_state_decay", kernels::GDN_STATE_DECAY),
    ("exp", kernels::EXP),
    ("sub", kernels::SUB),
    ("region_copy", kernels::REGION_COPY),
    ("splice_add", kernels::SPLICE_ADD),
    ("row_dot", kernels::ROW_DOT),
    ("scale_add", kernels::SCALE_ADD),
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP),
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0),
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT),
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD),
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD),
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD),
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST),
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE),
    ("gdn_ut_fwd", kernels::GDN_UT_FWD),
    ("bmm_tiled", kernels::BMM_TILED),
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("gdn_gap_gate", kernels::GDN_GAP_GATE),
    ("gdn_gap_gate_bwd", kernels::GDN_GAP_GATE_BWD),
    // Cooperative grad-norm, resolved by name by `optim::Optim`. Kept last:
    // the optimiser finds them by name, not by index.
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
    /// Time gaps of the visit slots and to entry (see `HostBatch::visit_dt`).
    visit_dt: DeviceBuffer,
    value_target: DeviceBuffer,
    value_state: DeviceBuffer,
    value_weight: DeviceBuffer,
    piece_subject: DeviceBuffer,
    time_features: DeviceBuffer,
    event: DeviceBuffer,
    exposure: DeviceBuffer,
    subject_weight: DeviceBuffer,
    /// `[B*N]` 1 for a real, non-summary token row: what the additive mode sums.
    pool_mask: DeviceBuffer,
}

/// Whether `gpu` takes the split-K weight gradient: a GPU that runs
/// workgroup-cooperative kernels. The CPU backend keeps its native one-stage
/// kernels, and so does a device without workgroup reductions.
fn splitk_dw(gpu: &Gpu) -> bool {
    gpu.kind() != "cpu" && gpu.caps().workgroup_reductions
}

/// Row blocks of the token-table backward: as many as fit the partial-sum
/// budget, at most [`EMB_PART_MAX_BLOCKS`].
fn emb_blocks(cfg: &HorizonConfig) -> u32 {
    let per = cfg.vocab as u64 * cfg.d_model as u64;
    (EMB_PART_ELEMS / per.max(1)).clamp(1, EMB_PART_MAX_BLOCKS) as u32
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
    /// Sets the encoder reads per batch: `b` times the sets per subject.
    sets: u32,
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
    // additive mode
    lz: DeviceBuffer,
    lzrep: DeviceBuffer,
    tf: DeviceBuffer,
    d_lz: DeviceBuffer,
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
    /// Partial column sums of the two-stage bias gradient.
    bias_part: DeviceBuffer,
    /// Per-slice partial weight gradients of the split-K path (`None` where
    /// the device runs the one-stage kernels).
    dw_part: Option<DeviceBuffer>,
    /// Per-block partial gradients of the token tables.
    emb_part: DeviceBuffer,
    /// Row blocks of the token-table backward.
    emb_blocks: u32,
    /// The forecast head, when the configuration asks for one.
    fc: Option<forecast::ForecastBufs>,
    /// The state across visits, when the configuration asks for one.
    bb: Option<backbone::BackboneBufs>,
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
        let sets = b * cfg.sets_per_subject();
        let (bn, bp, bb) = (sets as u64 * n, b as u64 * p, b as u64);
        let bhnn = sets as u64 * cfg.n_heads as u64 * n * n;
        let st = |x: u64| gpu.storage(x);
        let input = |label: &str, words: u64| {
            gpu.buffer(label, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
        };
        // Rows the stack backbone's blocks normalise at once (LayerNorm
        // statistics are shared scratch).
        let stack_rows = if cfg.visits > 0 && cfg.stack().is_some() {
            b as u64 * cfg.stack_layout().1 as u64
        } else {
            0
        };
        let inp = Inputs {
            token_ids: input("token_ids", bn),
            keep: input("keep", bn),
            value_bins: input("value_bins", bn * cfg.value_table() as u64),
            time_bins: input("time_bins", bn * cfg.time_table() as u64),
            summary_rows: input("summary_rows", sets as u64),
            visit_dt: input("visit_dt", sets as u64 + bb),
            value_target: input("value_target", bn),
            value_state: input("value_state", bn),
            value_weight: input("value_weight", bn),
            piece_subject: input("piece_subject", bp),
            time_features: input("time_features", bp * nf),
            event: input("event", bp * k),
            exposure: input("exposure", bp * k),
            subject_weight: input("subject_weight", bb),
            pool_mask: input("pool_mask", bn),
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
            sets,
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
            lz: st(bb * k),
            lzrep: st(bp * k),
            tf: st(bp * k),
            d_lz: st(bb * k),
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
            ln_mean: st(bn.max(stack_rows)),
            ln_inv: st(bn.max(stack_rows)),
            bias_part: st(BIAS_GRAD_CHUNKS as u64 * (3 * d).max(ff).max(k).max(r)),
            emb_blocks: emb_blocks(&cfg),
            emb_part: st(emb_blocks(&cfg) as u64 * cfg.vocab as u64 * d),
            dw_part: splitk_dw(&gpu).then(|| {
                // The largest weight any backward GEMM writes.
                let largest = cfg
                    .param_list()
                    .iter()
                    .filter(|(name, _)| name.ends_with(".weight"))
                    .map(|(_, numel)| *numel as u64)
                    .max()
                    .unwrap_or(1);
                st(DW_SPLITK_MAX_SLICES as u64 * largest)
            }),
            fc: forecast::ForecastBufs::new(&gpu, &cfg, b),
            bb: backbone::BackboneBufs::new(&gpu, &cfg, b),
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
            self.sets as usize,
            "batch assembled for {} sets, model sized for {}",
            hb.summary_rows.len(),
            self.sets
        );
        g.write(&i.token_ids, &hb.token_ids);
        g.write(&i.keep, &hb.keep);
        g.write_f32(&i.value_bins, &hb.value_bins);
        g.write_f32(&i.time_bins, &hb.time_bins);
        g.write(&i.summary_rows, &hb.summary_rows);
        if let Some(bb) = &self.bb {
            g.write_f32(&i.visit_dt, &hb.visit_dt);
            bb.write(g, hb);
        }
        g.write_f32(&i.value_target, &hb.value_target);
        g.write(&i.value_state, &hb.value_state);
        g.write_f32(&i.value_weight, &hb.value_weight);
        g.write(&i.piece_subject, &hb.piece_subject);
        g.write_f32(&i.time_features, &hb.time_features);
        g.write_f32(&i.event, &hb.event);
        g.write_f32(&i.exposure, &hb.exposure);
        g.write_f32(&i.subject_weight, &hb.subject_weight);
        if let Some(f) = &self.fc {
            f.write(g, hb);
        }
        if self.cfg.additive {
            g.write(
                &i.pool_mask,
                &additive::pool_mask(hb, self.cfg.max_tokens as usize),
            );
        }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn bidir(&self) -> Bidir {
        let d = self.cfg.d_model;
        Bidir {
            b: self.sets,
            t: self.cfg.max_tokens,
            n_heads: self.cfg.n_heads,
            head_dim: d / self.cfg.n_heads,
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    /// `out = x @ w^T` (`[m, k] x [n, k]`), tiled when the shape pays for it.
    #[allow(clippy::too_many_arguments)]
    fn mm(
        &self,
        x: &DeviceBuffer,
        w: &DeviceBuffer,
        out: &DeviceBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Step {
        let (kern, grid) = block::pick_gemm(m as usize, n as usize, MATMUL, MATMUL_REG3, false);
        self.gpu.dispatch(kern, &[x, w, out], &[m, k, n], grid)
    }

    /// `dx = dy @ w` (`[m, n] x [n, k]`), assigned or (`acc = 1`) accumulated.
    #[allow(clippy::too_many_arguments)]
    fn mm_dx(
        &self,
        dy: &DeviceBuffer,
        w: &DeviceBuffer,
        dx: &DeviceBuffer,
        m: u32,
        k: u32,
        n: u32,
        acc: u32,
    ) -> Step {
        let (kern, grid) =
            block::pick_gemm(m as usize, k as usize, MATMUL_DX, MATMUL_DX_REG, false);
        self.gpu.dispatch(kern, &[dy, w, dx], &[m, k, n, acc], grid)
    }

    /// `dw += dy^T @ x` (`[m, n]^T x [m, k]`). The contraction runs over the
    /// batch's rows - tens of thousands - while the output is a small weight,
    /// so a device that can run workgroup-cooperative kernels splits the
    /// rows across many workgroups (`matmul_dw_reg_splitk`) and sums the
    /// slices (`dw_splitk_reduce`); others keep the one-stage kernels.
    #[allow(clippy::too_many_arguments)]
    fn mm_dw(
        &self,
        dy: &DeviceBuffer,
        x: &DeviceBuffer,
        dw: &DeviceBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Vec<Step> {
        let Some(part) = &self.dw_part else {
            let (kern, grid) =
                block::pick_gemm(n as usize, k as usize, MATMUL_DW, MATMUL_DW_REG, false);
            return vec![self.gpu.dispatch(kern, &[dy, x, dw], &[m, k, n], grid)];
        };
        let tiles = n.div_ceil(128) * k.div_ceil(128);
        // Each slice is at least one BK chunk of rows.
        let slices = DW_SPLITK_TARGET_WGS
            .div_ceil(tiles)
            .min(DW_SPLITK_MAX_SLICES)
            .min(m.div_ceil(8))
            .max(1);
        vec![
            self.gpu.dispatch(
                MATMUL_DW_SPLITK,
                &[dy, x, part],
                &[m, k, n, slices],
                gpu_core::Dispatch::Workgroups(slices * tiles),
            ),
            self.gpu.step(DW_SPLITK_REDUCE, &[part, dw], &[n * k, slices, 1], n * k),
        ]
    }

    /// `table += scatter of dx by token` over `rows` rows: the partial sums
    /// of contiguous row blocks, then the blocks folded into the table.
    fn emb_grad(&self, tokens: &DeviceBuffer, dx: &DeviceBuffer, table: &DeviceBuffer, rows: u32) -> [Step; 2] {
        let (d, v, blocks) = (self.cfg.d_model, self.cfg.vocab, self.emb_blocks);
        [
            self.gpu.step(
                EMB_BWD_PART,
                &[tokens, dx, &self.emb_part],
                &[rows, d, v, blocks],
                blocks * v * d,
            ),
            self.gpu.step(DW_SPLITK_REDUCE, &[&self.emb_part, table], &[v * d, blocks, 1], v * d),
        ]
    }

    /// The LayerNorm parameter gradients of `y = LN(x) * gamma + beta` from
    /// `dy` (`[rows, d]`), with the row statistics `ln_stats_fwd` left in
    /// `ln_mean`/`ln_inv`: both as two-stage column reductions over row
    /// chunks, accumulated into `dgamma` and `dbeta`.
    fn ln_param_grads(
        &self,
        dy: &DeviceBuffer,
        x: &DeviceBuffer,
        dgamma: &DeviceBuffer,
        dbeta: &DeviceBuffer,
        d: u32,
        rows: u32,
    ) -> [Step; 4] {
        [
            self.gpu.step(
                LN_DGAMMA_PART,
                &[dy, x, &self.ln_mean, &self.ln_inv, &self.bias_part],
                &[rows, d, BIAS_GRAD_CHUNKS],
                d * BIAS_GRAD_CHUNKS,
            ),
            self.bias_grad_final(dgamma, rows, d),
            self.bias_grad_part(dy, rows, d),
            self.bias_grad_final(dbeta, rows, d),
        ]
    }

    /// Stage one of `db += column sums of dy` (`[m, n]`): partial sums over row chunks.
    fn bias_grad_part(&self, dy: &DeviceBuffer, m: u32, n: u32) -> Step {
        self.gpu.step(
            BIAS_GRAD_PART,
            &[dy, &self.bias_part],
            &[m, n, BIAS_GRAD_CHUNKS],
            n * BIAS_GRAD_CHUNKS,
        )
    }

    /// Stage two: fold the partial sums into `db` (accumulating).
    fn bias_grad_final(&self, db: &DeviceBuffer, m: u32, n: u32) -> Step {
        self.gpu.step(
            BIAS_GRAD_FINAL,
            &[&self.bias_part, db],
            &[m, n, BIAS_GRAD_CHUNKS],
            n,
        )
    }

    /// Both stages of the bias gradient.
    fn bias_grad(&self, dy: &DeviceBuffer, db: &DeviceBuffer, m: u32, n: u32) -> [Step; 2] {
        [
            self.bias_grad_part(dy, m, n),
            self.bias_grad_final(db, m, n),
        ]
    }

    /// The adjoint of `y = LN(x) * gamma + beta` from `dy`: the parameter
    /// gradients and `dx` (assigned), for the LayerNorm whose parameters are
    /// named `(gamma, beta)`.
    fn ln_backward_steps(
        &self,
        x: &DeviceBuffer,
        (gamma, beta): (&str, &str),
        dy: &DeviceBuffer,
        dx: &DeviceBuffer,
        rows: u32,
    ) -> Vec<Step> {
        let d = self.cfg.d_model;
        let mut s = vec![block::ln_stats_fwd(
            &self.gpu,
            &LN_IDS,
            x,
            &self.ln_mean,
            &self.ln_inv,
            d,
            rows,
            LN_EPS,
        )];
        s.extend(self.ln_param_grads(dy, x, self.ps.g(gamma), self.ps.g(beta), d, rows));
        s.push(block::layernorm_dx_bwd(
            &self.gpu,
            &LN_IDS,
            x,
            self.w(gamma),
            dy,
            dx,
            d,
            rows,
            LN_EPS,
        ));
        s
    }

    /// The feed-forward sublayer of a pre-norm block over `rows` rows,
    /// `out = x_mid + MLP(LN(x_mid))` with the parameters named
    /// `{prefix}ln2.*` and `{prefix}ffn.*`. `(ln2_out, up_pre, up, ffn_out)`
    /// are the block's activation buffers (the backward reads the first three).
    fn mlp_forward_steps(
        &self,
        prefix: &str,
        rows: u32,
        x_mid: &DeviceBuffer,
        (ln2_out, up_pre, up, ffn_out): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        out: &DeviceBuffer,
    ) -> Vec<Step> {
        let g = &self.gpu;
        let (d, ff) = (self.cfg.d_model, self.cfg.d_ff);
        let pn = |name: &str| self.w(&format!("{prefix}{name}"));
        vec![
            block::layernorm_fwd(
                g,
                &LN_IDS,
                x_mid,
                pn("ln2.weight"),
                pn("ln2.bias"),
                ln2_out,
                d,
                rows,
                LN_EPS,
            ),
            self.mm(ln2_out, pn("ffn.up.weight"), up_pre, rows, d, ff),
            g.step(
                BIAS_ADD,
                &[up_pre, pn("ffn.up.bias")],
                &[rows, ff],
                rows * ff,
            ),
            g.step(GELU, &[up_pre, up], &[rows * ff], rows * ff),
            self.mm(up, pn("ffn.down.weight"), ffn_out, rows, ff, d),
            g.step(
                BIAS_ADD,
                &[ffn_out, pn("ffn.down.bias")],
                &[rows, d],
                rows * d,
            ),
            g.step(ADD2, &[x_mid, ffn_out, out], &[rows * d], rows * d),
        ]
    }

    /// The adjoint of [`Self::mlp_forward_steps`]: from `d_out` (the gradient
    /// of the sublayer's output) the sublayer's parameter gradients and the
    /// gradient of `x_mid` (`d_xmid`, assigned: the skip path plus the MLP
    /// path). `scratch` is `(d_up, d_up_pre, d_branch, d_tmp)`.
    fn mlp_backward_steps(
        &self,
        prefix: &str,
        rows: u32,
        x_mid: &DeviceBuffer,
        (ln2_out, up_pre, up): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        d_out: &DeviceBuffer,
        (d_up, d_up_pre, d_branch, d_tmp): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        d_xmid: &DeviceBuffer,
    ) -> Vec<Step> {
        let g = &self.gpu;
        let (d, ff) = (self.cfg.d_model, self.cfg.d_ff);
        let pn = |name: &str| format!("{prefix}{name}");
        let gr = |name: &str| self.ps.g(&pn(name));
        let w = |name: &str| self.w(&pn(name));
        let mut s: Vec<Step> = self.bias_grad(d_out, gr("ffn.down.bias"), rows, d).into();
        s.extend(self.mm_dw(d_out, up, gr("ffn.down.weight"), rows, ff, d));
        s.extend([
            self.mm_dx(d_out, w("ffn.down.weight"), d_up, rows, ff, d, 0),
            g.step(
                GELU_BWD,
                &[up_pre, d_up, d_up_pre],
                &[rows * ff],
                rows * ff,
            ),
        ]);
        s.extend(self.bias_grad(d_up_pre, gr("ffn.up.bias"), rows, ff));
        s.extend(self.mm_dw(d_up_pre, ln2_out, gr("ffn.up.weight"), rows, d, ff));
        s.push(self.mm_dx(d_up_pre, w("ffn.up.weight"), d_branch, rows, d, ff, 0));
        s.extend(self.ln_backward_steps(
            x_mid,
            (&pn("ln2.weight"), &pn("ln2.bias")),
            d_branch,
            d_tmp,
            rows,
        ));
        s.push(g.step(ADD2, &[d_out, d_tmp, d_xmid], &[rows * d], rows * d));
        s
    }

    /// The token embedding `e = gamma[tok] * (value_bins @ Wv^T) + beta[tok]
    /// + time_bins @ Wt^T`, into `res[0]`; shared by both encoders.
    fn embedding_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let (n, d) = (c.max_tokens, c.d_model);
        let bn = self.sets * n;
        let (vt, tt) = (c.value_table(), c.time_table());
        vec![
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
            self.mm(
                &i.value_bins,
                self.w("value_bins.weight"),
                &self.phi,
                bn,
                vt,
                d,
            ),
            g.step(MUL, &[&self.gam, &self.phi, &self.gp], &[bn * d], bn * d),
            g.step(ADD2, &[&self.gp, &self.bet, &self.e1], &[bn * d], bn * d),
            self.mm(
                &i.time_bins,
                self.w("time_bins.weight"),
                &self.te,
                bn,
                tt,
                d,
            ),
            g.step(ADD2, &[&self.e1, &self.te, &self.res[0]], &[bn * d], bn * d),
        ]
    }

    fn forward_steps(&self) -> Vec<Step> {
        if self.cfg.additive {
            return self.additive_forward_steps();
        }
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let (n, d, r) = (c.max_tokens, c.d_model, c.rank);
        let (p, k, nf) = (c.pieces(), c.n_codes, c.time_features());
        let (bn, bp, b) = (self.sets * n, self.b * p, self.b);
        let mut s = self.embedding_steps();
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
            s.push(self.mm(
                &lb.ln1_out,
                self.w(&pn("attn.qkv.weight")),
                &lb.qkv,
                bn,
                d,
                3 * d,
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
                    &[self.sets, c.n_heads, n],
                    self.sets * c.n_heads * n * n,
                ),
            );
            s.extend(attn);
            s.push(self.mm(
                &lb.ctx,
                self.w(&pn("attn.out.weight")),
                &self.proj,
                bn,
                d,
                d,
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
            s.extend(self.mlp_forward_steps(
                &format!("blocks.{l}."),
                bn,
                &lb.xmid,
                (&lb.ln2_out, &lb.up_pre, &lb.up, &self.ffn_out),
                &self.res[l + 1],
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
            self.mm(&self.xf, self.w("value_head.weight"), &self.vpred, bn, d, 2),
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
        ]);
        s.extend(self.state_forward_steps());
        s.extend([
            // hazard head
            self.mm(&self.z, self.w("hazard.state.weight"), &self.az, b, d, r),
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
            self.mm(
                &i.time_features,
                self.w("hazard.time.weight"),
                &self.cf,
                bp,
                nf,
                r,
            ),
            g.step(ADD2, &[&self.rep, &self.cf, &self.h0], &[bp * r], bp * r),
            g.step(GELU, &[&self.h0, &self.h], &[bp * r], bp * r),
            self.mm(
                &self.h,
                self.w("hazard.code.weight"),
                &self.loglam,
                bp,
                r,
                k,
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
        s.extend(self.forecast_forward_steps());
        s
    }

    fn backward_steps(&self) -> Vec<Step> {
        if self.cfg.additive {
            return self.additive_backward_steps();
        }
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (n, d, r) = (c.max_tokens, c.d_model, c.rank);
        let (p, k, nf) = (c.pieces(), c.n_codes, c.time_features());
        let (bn, bp, b) = (self.sets * n, self.b * p, self.b);
        let last = c.n_layers as usize;
        // The forecast head first: its share of the state gradient goes into
        // d_z (cleared before this list), the hazard head's is added to it.
        let mut s = self.forecast_backward_steps();
        s.extend(vec![
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
            self.bias_grad_part(&self.d_loglam, bp, k),
            self.bias_grad_final(gr("hazard.code.bias"), bp, k),
        ]);
        s.extend(self.mm_dw(&self.d_loglam, &self.h, gr("hazard.code.weight"), bp, r, k));
        s.extend(vec![
            self.mm_dx(
                &self.d_loglam,
                self.w("hazard.code.weight"),
                &self.d_h,
                bp,
                r,
                k,
                0,
            ),
            g.step(
                GELU_BWD,
                &[&self.h0, &self.d_h, &self.d_h0],
                &[bp * r],
                bp * r,
            ),
        ]);
        s.extend(self.mm_dw(
            &self.d_h0,
            &i.time_features,
            gr("hazard.time.weight"),
            bp,
            nf,
            r,
        ));
        s.extend(vec![
            // d_az is cleared before this list runs (emb_bwd accumulates).
            g.step(
                EMB_BWD,
                &[&i.piece_subject, &self.d_h0, &self.d_az],
                &[bp, r, b],
                b * r,
            ),
            self.bias_grad_part(&self.d_az, b, r),
            self.bias_grad_final(gr("hazard.state.bias"), b, r),
        ]);
        s.extend(self.mm_dw(&self.d_az, &self.z, gr("hazard.state.weight"), b, d, r));
        s.extend(vec![
            self.mm_dx(
                &self.d_az,
                self.w("hazard.state.weight"),
                &self.d_z,
                b,
                d,
                r,
                1,
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
            self.bias_grad_part(&self.d_vpred, bn, 2),
            self.bias_grad_final(gr("value_head.bias"), bn, 2),
        ]);
        s.extend(self.mm_dw(&self.d_vpred, &self.xf, gr("value_head.weight"), bn, d, 2));
        s.extend(vec![
            self.mm_dx(
                &self.d_vpred,
                self.w("value_head.weight"),
                &self.d_xf,
                bn,
                d,
                2,
                0,
            ),
        ]);
        s.extend(self.state_backward_steps());
        s.extend(self.ln_backward_steps(
            &self.res[last],
            ("ln_f.weight", "ln_f.bias"),
            &self.d_xf,
            &self.dres[last],
            bn,
        ));
        let a = self.bidir();
        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");
            s.extend(self.mlp_backward_steps(
                &format!("blocks.{l}."),
                bn,
                &lb.xmid,
                (&lb.ln2_out, &lb.up_pre, &lb.up),
                &self.dres[l + 1],
                (&self.d_up, &self.d_up_pre, &self.d_branch, &self.d_tmp),
                &self.dxmid,
            ));
            // attention
            s.extend(self.bias_grad(&self.dxmid, gr(&pn("attn.out.bias")), bn, d));
            s.extend(self.mm_dw(&self.dxmid, &lb.ctx, gr(&pn("attn.out.weight")), bn, d, d));
            s.push(self.mm_dx(
                &self.dxmid,
                self.w(&pn("attn.out.weight")),
                &self.d_ctx,
                bn,
                d,
                d,
                0,
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
            s.extend(self.bias_grad(&self.d_qkv, gr(&pn("attn.qkv.bias")), bn, 3 * d));
            s.extend(self.mm_dw(
                &self.d_qkv,
                &lb.ln1_out,
                gr(&pn("attn.qkv.weight")),
                bn,
                d,
                3 * d,
            ));
            s.push(self.mm_dx(
                &self.d_qkv,
                self.w(&pn("attn.qkv.weight")),
                &self.d_branch,
                bn,
                d,
                3 * d,
                0,
            ));
            s.extend(self.ln_backward_steps(
                &self.res[l],
                (&pn("ln1.weight"), &pn("ln1.bias")),
                &self.d_branch,
                &self.d_tmp,
                bn,
            ));
            s.push(g.step(
                ADD2,
                &[&self.dxmid, &self.d_tmp, &self.dres[l]],
                &[bn * d],
                bn * d,
            ));
        }
        s.extend(self.embedding_backward_steps());
        s
    }

    /// Backward of [`Self::embedding_steps`] from the gradient in `dres[0]`.
    fn embedding_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (d, bn) = (c.d_model, self.sets * c.max_tokens);
        let (vt, tt) = (c.value_table(), c.time_table());
        let d0 = &self.dres[0];
        let mut s = self.mm_dw(d0, &i.time_bins, gr("time_bins.weight"), bn, tt, d);
        s.extend(self.emb_grad(&i.token_ids, d0, gr("tok.beta"), bn));
        s.push(g.step(MUL, &[d0, &self.phi, &self.d_gam], &[bn * d], bn * d));
        s.extend(self.emb_grad(&i.token_ids, &self.d_gam, gr("tok.gamma"), bn));
        s.push(g.step(MUL, &[d0, &self.gam, &self.d_phi], &[bn * d], bn * d));
        s.extend(self.mm_dw(
            &self.d_phi,
            &i.value_bins,
            gr("value_bins.weight"),
            bn,
            vt,
            d,
        ));
        s
    }

    /// Submit the forward pass (no readback).
    pub fn forward_submit(&self) {
        let clears = self.bb.as_ref().map_or_else(Vec::new, |bb| bb.forward_cleared());
        self.gpu.submit(&clears, &self.fwd);
    }

    /// The batch loss of the last forward: the weighted mean event NLL plus
    /// the weighted value NLL and, with a forecast head, the weighted
    /// forecast NLL.
    pub fn loss(&self) -> f32 {
        let (event, value) = self.loss_parts();
        event + value + self.forecast_loss() as f32
    }

    /// `(event NLL, value NLL)` of the last forward; the event NLL covers every
    /// hazard column, the next-event group included (the training objective).
    pub fn loss_parts(&self) -> (f32, f32) {
        let (outcome, next, value) = self.loss_split();
        (outcome + next, value)
    }

    /// `(outcome-code NLL, next-event-group NLL, value NLL)` of the last
    /// forward. The first is what held-out evaluation of the outcomes uses:
    /// the group is a training signal, not a result.
    pub fn loss_split(&self) -> (f32, f32, f32) {
        let c = &self.cfg;
        let (k, outcomes) = (c.n_codes as usize, c.outcome_codes() as usize);
        let bpk = (self.b * c.pieces() * c.n_codes) as usize;
        let bn = (self.sets * c.max_tokens) as usize;
        let (mut outcome, mut next) = (0.0f64, 0.0f64);
        for (e, &x) in self.gpu.read(&self.hloss, bpk).iter().enumerate() {
            if e % k < outcomes {
                outcome += x as f64;
            } else {
                next += x as f64;
            }
        }
        let value: f64 = if c.additive {
            0.0 // no value head
        } else {
            self.gpu
                .read(&self.vloss, bn)
                .iter()
                .map(|&x| x as f64)
                .sum()
        };
        (outcome as f32, next as f32, value as f32)
    }

    /// Forward and loss.
    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    /// Accumulate gradients of the last forward's loss.
    pub fn backward(&self) {
        let mut clears = vec![&self.d_az, &self.d_lz, &self.d_z];
        if let Some(bb) = &self.bb {
            clears.extend(bb.cleared());
        }
        self.gpu.submit(&clears, &self.bwd);
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
            .read(&self.vpred, (self.sets * self.cfg.max_tokens * 2) as usize)
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
