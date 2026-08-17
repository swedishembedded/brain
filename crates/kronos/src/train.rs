// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Differentiable Kronos AR-decoder — the trainable twin of the inference
//! [`crate::decoder`], built as a recorded Step tape over a [`ParamStore`] so it
//! backprops and fine-tunes. Reuses brain's gradcheck-validated transformer
//! backward (`model::block`: RMSNorm/RoPE/GQA/SwiGLU fwd+bwd) plus `bias_grad`,
//! `matmul_dw/dx`, `embed`/`emb_bwd`, and `ce_*_masked` — no new gradient kernels.
//! The block topology matches `nn::Ops::transformer_block` exactly (pre-norm
//! RMSNorm → q/k/v linear **+bias** → NeoX RoPE → causal scaled MHA → out_proj
//! **+bias** → residual → RMSNorm → SwiGLU FFN → residual), so a fine-tuned
//! checkpoint stays inference-compatible.
//!
//! Milestone B surface (gradcheck-gated): the trainable **hierarchical + temporal
//! embeddings** (emb_s1/emb_s2 ×√d → fused via the split `fusion_proj` → + Σ
//! calendar embeddings) feeding the transformer stack + `proj_s1` + CE on s1.
//! `fusion_proj` `[d, 2d]` is split at construction into two `[d, d]` halves
//! (`fusion_l`/`fusion_r`) so no concat kernel is needed; export merges them back.
//! Milestone C adds the dep cross-attention + `proj_s2`.

use crate::config::KronosConfig;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use model::IGNORE;
use optim::Optim;
use paramstore::ParamStore;
use std::cell::Cell;
use std::collections::HashMap;

// Kernel pipeline indices (order must match PIPELINES below).
const MATMUL: usize = 0;
const RMSNORM: usize = 1;
const RMS_INV: usize = 2;
const RMSNORM_DX: usize = 3;
const RMSNORM_DW: usize = 4;
const ROPE: usize = 5;
const ROPE_BWD: usize = 6;
const GQA_SCORES: usize = 7;
const ATTN_SOFTMAX: usize = 8;
const GQA_APPLY: usize = 9;
const GQA_DSCORES: usize = 10;
const GQA_DV: usize = 11;
const GQA_DQ: usize = 12;
const GQA_DK: usize = 13;
const SILU_MUL: usize = 14;
const SILU_DA: usize = 15;
const SILU_DB: usize = 16;
const ADD2: usize = 17;
const CE_VALUE: usize = 18;
const CE_GRAD: usize = 19;
const MATMUL_DX: usize = 20;
const MATMUL_DW: usize = 21;
const BIAS_ADD: usize = 22;
const BIAS_GRAD: usize = 23;
const ADAMW: usize = 24;
const GRADNORM_SQ: usize = 25;
const GRAD_SCALE: usize = 26;
const CLIP_COEF: usize = 27;
const GRAD_SCALE_BUF: usize = 28;
const EMBED: usize = 29;
const EMB_BWD: usize = 30;
const SC_BIDIR: usize = 31;
const SM_BIDIR: usize = 32;
const AP_BIDIR: usize = 33;
const DSC_BIDIR: usize = 34;
const DQ_BIDIR: usize = 35;
const DK_BIDIR: usize = 36;
const DV_BIDIR: usize = 37;
const CONCAT2: usize = 38;
const CONCAT_SPLIT: usize = 39;
const AXPY: usize = 40;

pub(crate) const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("embed", kernels::EMBED),
    ("emb_bwd", kernels::EMB_BWD),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("axpy", kernels::AXPY),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// LoRA fine-tuning config: rank-`r` adapters (scale `alpha/r`) on the linear
/// projections whose weight name ends with one of `targets` (e.g. `q_proj.weight`).
/// With LoRA on, the base weights are frozen and only `*.lora_a`/`*.lora_b` train —
/// the anti-overfit default for weekly fine-tuning.
#[derive(Clone, Debug)]
pub struct LoraCfg {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
}

impl LoraCfg {
    /// Default surface: the self-attention projections in every transformer block.
    pub fn attn(rank: usize, alpha: f32) -> LoraCfg {
        LoraCfg {
            rank,
            alpha,
            targets: [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.out_proj.weight",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
    fn hits(&self, wname: &str) -> bool {
        self.targets.iter().any(|t| wname.ends_with(t))
    }
}

const ROPE_THETA: f32 = 10000.0;

/// Calendar tables (name, cardinality), in the reference stamp order.
pub const CAL: [(&str, usize); 5] =
    [("minute", 60), ("hour", 24), ("weekday", 7), ("day", 32), ("month", 13)];

/// Per-layer forward-activation buffers (cached for the backward pass).
struct Layer {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

/// The trainable Kronos decoder (milestone B surface). `b`=1 (one series per
/// forward); `t` is the context length.
pub struct KronosTrain {
    gpu: Gpu,
    cfg: KronosConfig,
    ps: ParamStore,
    opt: Optim,
    t: u32,
    b: u32,
    count: Cell<f32>,
    sqrt_d: f32,
    lora: Option<LoraCfg>,
    lora_a_buf: DeviceBuffer,
    lora_da_buf: DeviceBuffer,
    lora_out_buf: DeviceBuffer,

    // inputs (uploaded per batch)
    s1_ids: DeviceBuffer,
    s2_ids: DeviceBuffer,
    cal_ids: Vec<DeviceBuffer>, // 5 × [t]
    targets: DeviceBuffer,      // [t] s1 token ids

    // embedding activations
    e1: DeviceBuffer,
    e2: DeviceBuffer,
    x1: DeviceBuffer,
    x2: DeviceBuffer,
    xf: DeviceBuffer,
    te: Vec<DeviceBuffer>, // 5 gathers
    s01: DeviceBuffer,
    s23: DeviceBuffer,
    s0123: DeviceBuffer,
    tesum: DeviceBuffer,

    // dep cross-attention + s2 head (milestone C)
    sampled_s1_ids: DeviceBuffer,
    s2_targets: DeviceBuffer,
    sib: DeviceBuffer,
    dep_q: DeviceBuffer,
    dep_k: DeviceBuffer,
    dep_v: DeviceBuffer,
    qk: DeviceBuffer,
    qkv: DeviceBuffer,
    dep_scores: DeviceBuffer,
    dep_probs: DeviceBuffer,
    dep_ctxo: DeviceBuffer,
    dep_sum: DeviceBuffer,
    dep_normed: DeviceBuffer,
    s2_logits: DeviceBuffer,
    ce_buf2: DeviceBuffer,
    ce_grad_uni2: DeviceBuffer,
    d_s2logits: DeviceBuffer,
    d_normed: DeviceBuffer,
    d_sum: DeviceBuffer,
    d_ctxo: DeviceBuffer,
    d_dscores: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_dq: DeviceBuffer,
    d_dk: DeviceBuffer,
    d_dv: DeviceBuffer,
    d_sib: DeviceBuffer,
    d_xn2: DeviceBuffer,

    res: Vec<DeviceBuffer>,
    layers: Vec<Layer>,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    scores: DeviceBuffer,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    // backward temporaries
    dres: Vec<DeviceBuffer>,
    d_logits: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_up: DeviceBuffer,
    d_e1: DeviceBuffer,
    d_e2: DeviceBuffer,
    inv: DeviceBuffer,

    ce_grad_uni: DeviceBuffer,
    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
}

/// Trainable transformer-block + norm + s1-head param names (milestone A subset).
pub fn param_list_blocks(cfg: &KronosConfig) -> Vec<(String, usize)> {
    let d = cfg.d_model;
    let ff = cfg.ff_dim;
    let mut p: Vec<(String, usize)> = Vec::new();
    for i in 0..cfg.n_layers {
        let pre = format!("transformer.{i}");
        p.push((format!("{pre}.norm1.weight"), d));
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            p.push((format!("{pre}.self_attn.{proj}.weight"), d * d));
            p.push((format!("{pre}.self_attn.{proj}.bias"), d));
        }
        p.push((format!("{pre}.norm2.weight"), d));
        p.push((format!("{pre}.ffn.w1.weight"), ff * d));
        p.push((format!("{pre}.ffn.w3.weight"), ff * d));
        p.push((format!("{pre}.ffn.w2.weight"), d * ff));
    }
    p.push(("norm.weight".into(), d));
    p.push(("head.proj_s1.weight".into(), cfg.s1_vocab() * d));
    p.push(("head.proj_s1.bias".into(), cfg.s1_vocab()));
    p
}

/// Full milestone-B trainable param list: embeddings (with `fusion_proj` split
/// into `fusion_l`/`fusion_r`) + the block/head params.
pub fn param_list_b(cfg: &KronosConfig) -> Vec<(String, usize)> {
    let d = cfg.d_model;
    let mut p: Vec<(String, usize)> = vec![
        ("embedding.emb_s1.weight".into(), cfg.s1_vocab() * d),
        ("embedding.emb_s2.weight".into(), cfg.s2_vocab() * d),
        ("embedding.fusion_l".into(), d * d),
        ("embedding.fusion_r".into(), d * d),
        ("embedding.fusion_proj.bias".into(), d),
    ];
    for (name, size) in CAL {
        p.push((format!("time_emb.{name}_embed.weight"), size * d));
    }
    p.extend(param_list_blocks(cfg));
    p
}

/// Full milestone-C trainable param list: everything in B plus the dependency
/// cross-attention layer and the s2 head (the whole decoder).
pub fn param_list_c(cfg: &KronosConfig) -> Vec<(String, usize)> {
    let d = cfg.d_model;
    let mut p = param_list_b(cfg);
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        p.push((format!("dep_layer.cross_attn.{proj}.weight"), d * d));
        p.push((format!("dep_layer.cross_attn.{proj}.bias"), d));
    }
    p.push(("dep_layer.norm.weight".into(), d));
    p.push(("head.proj_s2.weight".into(), cfg.s2_vocab() * d));
    p.push(("head.proj_s2.bias".into(), cfg.s2_vocab()));
    p
}

/// Split the reference fused `embedding.fusion_proj.weight` `[d, 2d]` into the two
/// `[d, d]` column halves this model trains (left = s1 side, right = s2 side).
/// Also usable to seed from a real checkpoint; export does the inverse merge.
fn split_fusion(init: &HashMap<String, Vec<f32>>, d: usize) -> (Vec<f32>, Vec<f32>) {
    let w = init.get("embedding.fusion_proj.weight").expect("missing embedding.fusion_proj.weight");
    assert_eq!(w.len(), d * 2 * d);
    let mut l = vec![0.0f32; d * d];
    let mut r = vec![0.0f32; d * d];
    for o in 0..d {
        l[o * d..o * d + d].copy_from_slice(&w[o * 2 * d..o * 2 * d + d]);
        r[o * d..o * d + d].copy_from_slice(&w[o * 2 * d + d..o * 2 * d + 2 * d]);
    }
    (l, r)
}

impl KronosTrain {
    /// Full-backprop constructor: every param trainable. `init` uses reference
    /// names (incl. `embedding.fusion_proj.weight` `[d,2d]`, split internally).
    pub fn new(cfg: KronosConfig, t: u32, init: &HashMap<String, Vec<f32>>) -> KronosTrain {
        Self::with_lora(cfg, t, init, None)
    }

    /// LoRA (or full) constructor. With `lora = Some(..)` every base weight is
    /// frozen and only the adapters train (the anti-overfit weekly-fine-tune path);
    /// with `None` it is full-backprop. Adapter weights are seeded here (A small
    /// random, B zero → the initial LoRA delta is zero, so the model starts == base).
    pub fn with_lora(cfg: KronosConfig, t: u32, init: &HashMap<String, Vec<f32>>, lora: Option<LoraCfg>) -> KronosTrain {
        Self::with_lora_on(Gpu::new(PIPELINES), cfg, t, init, lora)
    }

    /// Build on an existing device handle (`gpu_core::Gpu::share`) — one device
    /// per process, however many trainers/evaluators a run constructs.
    pub fn with_lora_on(gpu: Gpu, cfg: KronosConfig, t: u32, init: &HashMap<String, Vec<f32>>, lora: Option<LoraCfg>) -> KronosTrain {
        Self::with_lora_batch_on(gpu, cfg, t, 1, init, lora)
    }

    /// Full-backprop batched trainer (`b` sequences of length `t` per step).
    pub fn new_batch(cfg: KronosConfig, t: u32, b: u32, init: &HashMap<String, Vec<f32>>) -> KronosTrain {
        Self::with_lora_batch_on(Gpu::new(PIPELINES), cfg, t, b, init, None)
    }

    /// LoRA (or full, `lora = None`) batched trainer — the weekly-fine-tune path
    /// with `b` windows per step.
    pub fn with_lora_batch(cfg: KronosConfig, t: u32, b: u32, init: &HashMap<String, Vec<f32>>, lora: Option<LoraCfg>) -> KronosTrain {
        Self::with_lora_batch_on(Gpu::new(PIPELINES), cfg, t, b, init, lora)
    }

    /// Batched full/LoRA constructor: `b` independent sequences of length `t` are
    /// trained per step (their tokens are uploaded batch-major — all `t` of
    /// sequence 0, then sequence 1, …). Activation buffers scale by `b` (rows =
    /// `b*t`); the per-sequence attention (main GQA + the dep bidir head) keeps a
    /// `t×t` score matrix *per batch element* (`b*heads*t*t`), so sequences never
    /// attend across the batch. Every weight-grad accumulates over the whole
    /// `b*t` token set and the CE loss is the mean over all non-ignored tokens —
    /// so a `b`-batched step equals the sum of `b` single-sequence steps'
    /// gradients (validated bit-close by the batch-parity gate + gradcheck).
    pub fn with_lora_batch_on(gpu: Gpu, cfg: KronosConfig, t: u32, b: u32, init: &HashMap<String, Vec<f32>>, lora: Option<LoraCfg>) -> KronosTrain {
        let d = cfg.d_model;
        // build the internal init (fusion split) then the ParamStore.
        let mut init2 = init.clone();
        let (fl, fr) = split_fusion(init, d);
        init2.remove("embedding.fusion_proj.weight");
        init2.insert("embedding.fusion_l".into(), fl);
        init2.insert("embedding.fusion_r".into(), fr);
        let ps = if let Some(lc) = &lora {
            // seed adapters + build role list (base Frozen, adapters Trainable).
            let mut lcg = data::rng::Lcg::new(0x51A7); // the unified LCG (audit F40) -- the sanctioned home for production deterministic init
            let mut roles: Vec<(String, usize, paramstore::Role)> = Vec::new();
            for (name, numel) in param_list_c(&cfg) {
                roles.push((name.clone(), numel, paramstore::Role::Frozen));
                if lc.hits(&name) {
                    let (r, k, nout) = (lc.rank, d, numel / d); // targeted projections are d×k
                    init2.entry(format!("{name}.lora_a")).or_insert_with(|| {
                        (0..r * k).map(|_| lcg.scaled(0.01)).collect()
                    });
                    init2.entry(format!("{name}.lora_b")).or_insert_with(|| vec![0.0f32; nout * r]);
                    roles.push((format!("{name}.lora_a"), r * k, paramstore::Role::Trainable));
                    roles.push((format!("{name}.lora_b"), nout * r, paramstore::Role::Trainable));
                }
            }
            ParamStore::new_with_roles(&gpu, roles, &init2)
        } else {
            ParamStore::new(&gpu, param_list_c(&cfg), &init2)
        };
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);
        let lora_r = lora.as_ref().map(|l| l.rank as u64).unwrap_or(1).max(1);

        let d = cfg.d_model as u64;
        let ff = cfg.ff_dim as u64;
        let s1v = cfg.s1_vocab() as u64;
        let s2v = cfg.s2_vocab() as u64;
        let bb = b as u64;
        let tt = t as u64;
        let n = bb * tt; // total rows across the batch (batch-major: b sequences of t)
        let heads = cfg.n_heads as u64;
        let heads_dep = cfg.dep_n_heads as u64;
        // Attention scores/probs are per-batch-element t×t (sequences do not attend
        // across the batch), so scale by b — NOT by n=b*t squared.
        let hh = bb * heads * tt * tt;
        let dep_hh = bb * heads_dep * tt * tt;
        let st = |x: u64| gpu.storage(x);
        let idbuf = |name: &str| gpu.buffer(name, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);

        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                xn1: st(n * d), q: st(n * d), k: st(n * d), v: st(n * d),
                probs: st(hh), ctx: st(n * d), xmid: st(n * d), xn2: st(n * d),
                gate: st(n * ff), up: st(n * ff), h: st(n * ff),
            });
        }
        let mut m = KronosTrain {
            cfg,
            t,
            b,
            count: Cell::new(1.0),
            sqrt_d: (d as f32).sqrt(),
            lora,
            lora_a_buf: st(n * lora_r),
            lora_da_buf: st(n * lora_r),
            lora_out_buf: st(n * d),
            ps,
            opt,
            s1_ids: idbuf("s1_ids"),
            s2_ids: idbuf("s2_ids"),
            cal_ids: (0..5).map(|c| idbuf(&format!("cal{c}"))).collect(),
            targets: idbuf("targets"),
            e1: st(n * d), e2: st(n * d), x1: st(n * d), x2: st(n * d), xf: st(n * d),
            te: (0..5).map(|_| st(n * d)).collect(),
            s01: st(n * d), s23: st(n * d), s0123: st(n * d), tesum: st(n * d),
            sampled_s1_ids: idbuf("sampled_s1"),
            s2_targets: idbuf("s2_targets"),
            sib: st(n * d), dep_q: st(n * d), dep_k: st(n * d), dep_v: st(n * d),
            qk: st(n * 2 * d), qkv: st(n * 3 * d),
            dep_scores: st(dep_hh), dep_probs: st(dep_hh),
            dep_ctxo: st(n * d), dep_sum: st(n * d), dep_normed: st(n * d),
            s2_logits: st(n * s2v), ce_buf2: st(n), ce_grad_uni2: gpu.uniform_dynamic(4),
            d_s2logits: st(n * s2v), d_normed: st(n * d), d_sum: st(n * d),
            d_ctxo: st(n * d), d_dscores: st(dep_hh), d_qkv: st(n * 3 * d),
            d_dq: st(n * d), d_dk: st(n * d), d_dv: st(n * d), d_sib: st(n * d),
            d_xn2: st(n * d),
            res,
            layers,
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(hh),
            xn_final: st(n * d),
            logits: st(n * s1v),
            ce_buf: st(n),
            dres,
            d_logits: st(n * s1v),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_ctx: st(n * d),
            d_scores: st(hh),
            d_q: st(n * d),
            d_k: st(n * d),
            d_v: st(n * d),
            d_h: st(n * ff),
            d_gate: st(n * ff),
            d_up: st(n * ff),
            d_e1: st(n * d),
            d_e2: st(n * d),
            inv: st(n),
            ce_grad_uni: gpu.uniform_dynamic(4),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        m.bwd_steps = m.backward_steps();
        m
    }

    fn ids() -> KernelIds {
        KernelIds {
            rmsnorm: RMSNORM, rms_inv: RMS_INV, rmsnorm_dx: RMSNORM_DX, rmsnorm_dw: RMSNORM_DW,
            rope: ROPE, rope_bwd: ROPE_BWD, gqa_scores: GQA_SCORES, gqa_apply: GQA_APPLY,
            attn_softmax: ATTN_SOFTMAX, gqa_dscores: GQA_DSCORES, gqa_dv: GQA_DV, gqa_dq: GQA_DQ,
            gqa_dk: GQA_DK, silu_mul: SILU_MUL, silu_da: SILU_DA, silu_db: SILU_DB,
        }
    }
    fn gqa(&self) -> Gqa {
        let hd = (self.cfg.d_model / self.cfg.n_heads) as u32;
        Gqa { b: self.b, t: self.t, n_heads: self.cfg.n_heads as u32, n_kv_heads: self.cfg.n_heads as u32, head_dim: hd }
    }
    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }
    fn matmul(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        s.push(self.gpu.step(MATMUL, &[x, self.w(wname), out], &[m, k, nout], m * nout));
    }
    fn bias(&self, s: &mut Vec<Step>, out: &DeviceBuffer, bname: &str, m: u32, n: u32) {
        s.push(self.gpu.step(BIAS_ADD, &[out, self.w(bname)], &[m, n], m * n));
    }
    fn rms_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        // gain grad only when the norm weight trains (frozen under LoRA → dx only).
        let gw = self.trainable(wname).then(|| self.g(wname));
        s.extend(block::rmsnorm_bwd(&self.gpu, &Self::ids(), x, self.w(wname), dy, dx, &self.inv, gw, dim, rows));
    }
    /// A weight gradient that runs only for a trainable weight (frozen → skipped).
    fn dw(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, m: u32, k: u32, nout: u32) {
        if self.trainable(wname) {
            s.push(self.gpu.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
        }
    }
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }
    fn lora_for(&self, wname: &str) -> Option<(u32, f32)> {
        self.lora.as_ref().filter(|lc| lc.hits(wname)).map(|lc| (lc.rank as u32, lc.alpha / lc.rank as f32))
    }
    /// Forward LoRA delta `y += (alpha/r)·(x·Aᵀ)·Bᵀ` for a targeted projection
    /// (no-op otherwise). Must be wired at the SAME projections `proj_bwd`'s LoRA
    /// branch fires on, so forward and backward stay consistent.
    fn lora_fwd(&self, s: &mut Vec<Step>, wname: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(wname) else { return };
        let a = format!("{wname}.lora_a");
        let b = format!("{wname}.lora_b");
        s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a_buf], &[m, k, r], m * r));
        s.push(self.gpu.step(MATMUL, &[&self.lora_a_buf, self.w(&b), &self.lora_out_buf], &[m, r, nout], m * nout));
        s.push(self.gpu.step(AXPY, &[y, &self.lora_out_buf], &[m * nout, f(scale)], m * nout));
    }
    /// Backward for a linear `y = x·Wᵀ + b`. Full: dW, db, dX. LoRA: base weight
    /// frozen (dX only, no dW), adapter grads gA/gB (Qwen pattern). Frozen params
    /// (LoRA base, frozen bias) get no weight/bias grad — only dX flows.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, bname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        if self.trainable(bname) {
            s.push(self.gpu.step(BIAS_GRAD, &[d_out, self.g(bname)], &[m, nout], nout));
        }
        match self.lora_for(wname) {
            Some((r, scale)) => {
                // frozen base: dx += d_out·W (no dW)
                s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
                let a = format!("{wname}.lora_a");
                let b = format!("{wname}.lora_b");
                s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a_buf], &[m, k, r], m * r));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_a_buf], &[m * r, f(scale)], m * r));
                s.push(self.gpu.step(MATMUL_DW, &[d_out, &self.lora_a_buf, self.g(&b)], &[m, r, nout], nout * r));
                s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(&b), &self.lora_da_buf], &[m, r, nout, 0], m * r));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_da_buf], &[m * r, f(scale)], m * r));
                s.push(self.gpu.step(MATMUL_DW, &[&self.lora_da_buf, x, self.g(&a)], &[m, k, r], r * k));
                s.push(self.gpu.step(MATMUL_DX, &[&self.lora_da_buf, self.w(&a), dx], &[m, k, r, 1], m * k));
            }
            None => {
                if self.trainable(wname) {
                    s.push(self.gpu.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
                }
                s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
            }
        }
    }

    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let d = c.d_model as u32;
        let ff = c.ff_dim as u32;
        let t = self.t; // per-sequence length (rope period + attention time dim)
        let b = self.b;
        let n = b * t; // total rows (batch-major)
        let s1v = c.s1_vocab() as u32;
        let heads = c.n_heads as u32;
        let hd = d / heads;
        let ids = Self::ids();
        let ga = self.gqa();
        // ---- hierarchical + temporal embedding → res[0] ----
        let mut s: Vec<Step> = vec![
            self.gpu.step(EMBED, &[&self.s1_ids, self.w("embedding.emb_s1.weight"), &self.e1], &[d, n], n * d),
            self.gpu.step(GRAD_SCALE, &[&self.e1], &[n * d, f(self.sqrt_d)], n * d),
            self.gpu.step(EMBED, &[&self.s2_ids, self.w("embedding.emb_s2.weight"), &self.e2], &[d, n], n * d),
            self.gpu.step(GRAD_SCALE, &[&self.e2], &[n * d, f(self.sqrt_d)], n * d),
        ];
        self.matmul(&mut s, &self.e1, "embedding.fusion_l", &self.x1, n, d, d);
        self.matmul(&mut s, &self.e2, "embedding.fusion_r", &self.x2, n, d, d);
        s.push(self.gpu.step(ADD2, &[&self.x1, &self.x2, &self.xf], &[n * d], n * d));
        self.bias(&mut s, &self.xf, "embedding.fusion_proj.bias", n, d);
        for (ci, (name, _)) in CAL.iter().enumerate() {
            s.push(self.gpu.step(EMBED, &[&self.cal_ids[ci], self.w(&format!("time_emb.{name}_embed.weight")), &self.te[ci]], &[d, n], n * d));
        }
        s.push(self.gpu.step(ADD2, &[&self.te[0], &self.te[1], &self.s01], &[n * d], n * d));
        s.push(self.gpu.step(ADD2, &[&self.te[2], &self.te[3], &self.s23], &[n * d], n * d));
        s.push(self.gpu.step(ADD2, &[&self.s01, &self.s23, &self.s0123], &[n * d], n * d));
        s.push(self.gpu.step(ADD2, &[&self.s0123, &self.te[4], &self.tesum], &[n * d], n * d));
        s.push(self.gpu.step(ADD2, &[&self.xf, &self.tesum, &self.res[0]], &[n * d], n * d));

        // ---- transformer stack ----
        for l in 0..c.n_layers {
            let lb = &self.layers[l];
            let p = |nm: &str| format!("transformer.{l}.{nm}");
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &self.res[l], self.w(&p("norm1.weight")), &lb.xn1, d, n));
            self.matmul(&mut s, &lb.xn1, &p("self_attn.q_proj.weight"), &lb.q, n, d, d);
            self.bias(&mut s, &lb.q, &p("self_attn.q_proj.bias"), n, d);
            self.lora_fwd(&mut s, &p("self_attn.q_proj.weight"), &lb.xn1, &lb.q, n, d, d);
            self.matmul(&mut s, &lb.xn1, &p("self_attn.k_proj.weight"), &lb.k, n, d, d);
            self.bias(&mut s, &lb.k, &p("self_attn.k_proj.bias"), n, d);
            self.lora_fwd(&mut s, &p("self_attn.k_proj.weight"), &lb.xn1, &lb.k, n, d, d);
            self.matmul(&mut s, &lb.xn1, &p("self_attn.v_proj.weight"), &lb.v, n, d, d);
            self.bias(&mut s, &lb.v, &p("self_attn.v_proj.bias"), n, d);
            self.lora_fwd(&mut s, &p("self_attn.v_proj.weight"), &lb.xn1, &lb.v, n, d, d);
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.q, n, heads, hd, d, t, ROPE_THETA));
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.k, n, heads, hd, d, t, ROPE_THETA));
            s.extend(block::gqa_fwd(&self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &self.scores, &lb.probs, &lb.ctx));
            self.matmul(&mut s, &lb.ctx, &p("self_attn.out_proj.weight"), &self.proj, n, d, d);
            self.bias(&mut s, &self.proj, &p("self_attn.out_proj.bias"), n, d);
            self.lora_fwd(&mut s, &p("self_attn.out_proj.weight"), &lb.ctx, &self.proj, n, d, d);
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &lb.xmid, self.w(&p("norm2.weight")), &lb.xn2, d, n));
            self.matmul(&mut s, &lb.xn2, &p("ffn.w1.weight"), &lb.gate, n, d, ff);
            self.matmul(&mut s, &lb.xn2, &p("ffn.w3.weight"), &lb.up, n, d, ff);
            s.push(block::swiglu_fwd(&self.gpu, &ids, &lb.gate, &lb.up, &lb.h, n * ff));
            self.matmul(&mut s, &lb.h, &p("ffn.w2.weight"), &self.mlp_out, n, ff, d);
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
        }
        let last = c.n_layers;
        s.push(block::rmsnorm_fwd(&self.gpu, &ids, &self.res[last], self.w("norm.weight"), &self.xn_final, d, n));
        self.matmul(&mut s, &self.xn_final, "head.proj_s1.weight", &self.logits, n, d, s1v);
        self.bias(&mut s, &self.logits, "head.proj_s1.bias", n, s1v);
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, s1v, IGNORE], n));

        // ---- dependency cross-attention (q from sampled-s1 sibling, k/v from
        // ctx=xn_final; non-causal) + s2 head ----
        let s2v = c.s2_vocab() as u32;
        let dep_h = c.dep_n_heads as u32;
        let dep_hd = d / dep_h;
        let s3 = 3 * d; // fused qkv row stride
        // sib = RAW emb_s1[sampled_s1] (no √d scale, matching decode_s2)
        s.push(self.gpu.step(EMBED, &[&self.sampled_s1_ids, self.w("embedding.emb_s1.weight"), &self.sib], &[d, n], n * d));
        let dp = |nm: &str| format!("dep_layer.cross_attn.{nm}");
        self.matmul(&mut s, &self.sib, &dp("q_proj.weight"), &self.dep_q, n, d, d);
        self.bias(&mut s, &self.dep_q, &dp("q_proj.bias"), n, d);
        self.matmul(&mut s, &self.xn_final, &dp("k_proj.weight"), &self.dep_k, n, d, d);
        self.bias(&mut s, &self.dep_k, &dp("k_proj.bias"), n, d);
        self.matmul(&mut s, &self.xn_final, &dp("v_proj.weight"), &self.dep_v, n, d, d);
        self.bias(&mut s, &self.dep_v, &dp("v_proj.bias"), n, d);
        s.push(block::rope_fwd(&self.gpu, &ids, &self.dep_q, n, dep_h, dep_hd, d, t, ROPE_THETA));
        s.push(block::rope_fwd(&self.gpu, &ids, &self.dep_k, n, dep_h, dep_hd, d, t, ROPE_THETA));
        // pack q,k,v → fused qkv[n,3d]
        s.push(self.gpu.step(CONCAT2, &[&self.dep_q, &self.dep_k, &self.qk], &[n, d, d, 1, 1], n * 2 * d));
        s.push(self.gpu.step(CONCAT2, &[&self.qk, &self.dep_v, &self.qkv], &[n, 2 * d, d, 1, 1], n * 3 * d));
        // non-causal attention — per-batch-element t×t (leading dim = b)
        s.push(self.gpu.step(SC_BIDIR, &[&self.qkv, &self.dep_scores], &[b, dep_h, t, dep_hd, s3, 0, d], b * dep_h * t * t));
        s.push(self.gpu.step(SM_BIDIR, &[&self.dep_scores, &self.dep_probs], &[b, dep_h, t], b * dep_h * t));
        s.push(self.gpu.step(AP_BIDIR, &[&self.dep_probs, &self.qkv, &self.dep_ctxo], &[b, dep_h, t, dep_hd, s3, 2 * d, d], b * dep_h * t * dep_hd));
        // out proj (reuse mlp_out scratch as dep_o), residual with ctx, norm, s2 head
        self.matmul(&mut s, &self.dep_ctxo, &dp("out_proj.weight"), &self.mlp_out, n, d, d);
        self.bias(&mut s, &self.mlp_out, &dp("out_proj.bias"), n, d);
        s.push(self.gpu.step(ADD2, &[&self.xn_final, &self.mlp_out, &self.dep_sum], &[n * d], n * d));
        s.push(block::rmsnorm_fwd(&self.gpu, &ids, &self.dep_sum, self.w("dep_layer.norm.weight"), &self.dep_normed, d, n));
        self.matmul(&mut s, &self.dep_normed, "head.proj_s2.weight", &self.s2_logits, n, d, s2v);
        self.bias(&mut s, &self.s2_logits, "head.proj_s2.bias", n, s2v);
        s.push(self.gpu.step(CE_VALUE, &[&self.s2_logits, &self.s2_targets, &self.ce_buf2], &[n, s2v, IGNORE], n));
        s
    }

    fn backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let d = c.d_model as u32;
        let ff = c.ff_dim as u32;
        let t = self.t; // per-sequence length (rope period + attention time dim)
        let b = self.b;
        let n = b * t; // total rows (batch-major)
        let s1v = c.s1_vocab() as u32;
        let s2v = c.s2_vocab() as u32;
        let heads = c.n_heads as u32;
        let hd = d / heads;
        let ids = Self::ids();
        let ga = self.gqa();
        let mut s: Vec<Step> = Vec::new();

        let last = c.n_layers;
        let dep_h = c.dep_n_heads as u32;
        let dep_hd = d / dep_h;
        let s3 = 3 * d;
        let dp = |nm: &str| format!("dep_layer.cross_attn.{nm}");

        // ---- s2 / dep-layer backward ----
        // grad wrt s2 logits; s2 head → d_normed
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni2, &[&self.s2_logits, &self.s2_targets, &self.d_s2logits], n * s2v));
        self.proj_bwd(&mut s, &self.d_s2logits, &self.dep_normed, "head.proj_s2.weight", "head.proj_s2.bias", &self.d_normed, n, d, s2v, 0);
        // dep_norm rmsnorm bwd → d_sum (grad wrt dep_sum = xn_final + o)
        self.rms_bwd(&mut s, &self.dep_sum, "dep_layer.norm.weight", &self.d_normed, &self.d_sum, d, n);
        // dep out_proj bwd (d_o = d_sum) → d_ctxo
        self.proj_bwd(&mut s, &self.d_sum, &self.dep_ctxo, &dp("out_proj.weight"), &dp("out_proj.bias"), &self.d_ctxo, n, d, d, 0);
        // non-causal attention backward → d_qkv (fused; disjoint q/k/v slices)
        s.push(self.gpu.step(DSC_BIDIR, &[&self.d_ctxo, &self.qkv, &self.dep_probs, &self.d_dscores], &[b, dep_h, t, dep_hd, s3, 2 * d, d], b * dep_h * t * t));
        s.push(self.gpu.step(DQ_BIDIR, &[&self.d_dscores, &self.qkv, &self.d_qkv], &[b, dep_h, t, dep_hd, s3, 0, d], b * dep_h * t * dep_hd));
        s.push(self.gpu.step(DK_BIDIR, &[&self.d_dscores, &self.qkv, &self.d_qkv], &[b, dep_h, t, dep_hd, s3, 0, d], b * dep_h * t * dep_hd));
        s.push(self.gpu.step(DV_BIDIR, &[&self.dep_probs, &self.d_ctxo, &self.d_qkv], &[b, dep_h, t, dep_hd, s3, 2 * d, d], b * dep_h * t * dep_hd));
        // unpack d_qkv → d_dq/d_dk/d_dv
        s.push(self.gpu.step(CONCAT_SPLIT, &[&self.d_qkv, &self.d_dq], &[n, s3, d, 0, 1, 1], n * d));
        s.push(self.gpu.step(CONCAT_SPLIT, &[&self.d_qkv, &self.d_dk], &[n, s3, d, d, 1, 1], n * d));
        s.push(self.gpu.step(CONCAT_SPLIT, &[&self.d_qkv, &self.d_dv], &[n, s3, d, 2 * d, 1, 1], n * d));
        s.push(block::rope_bwd(&self.gpu, &ids, &self.d_dq, n, dep_h, dep_hd, d, t, ROPE_THETA));
        s.push(block::rope_bwd(&self.gpu, &ids, &self.d_dk, n, dep_h, dep_hd, d, t, ROPE_THETA));
        // q proj bwd → d_sib (grad to the sibling embedding); k/v proj bwd → d_xn
        self.proj_bwd(&mut s, &self.d_dq, &self.sib, &dp("q_proj.weight"), &dp("q_proj.bias"), &self.d_sib, n, d, d, 0);
        self.proj_bwd(&mut s, &self.d_dk, &self.xn_final, &dp("k_proj.weight"), &dp("k_proj.bias"), &self.d_xn, n, d, d, 0);
        self.proj_bwd(&mut s, &self.d_dv, &self.xn_final, &dp("v_proj.weight"), &dp("v_proj.bias"), &self.d_xn, n, d, d, 1);
        // sibling embedding grad scatters into emb_s1 at the SAMPLED rows (RAW, no √d).
        if self.trainable("embedding.emb_s1.weight") {
            s.push(self.gpu.step(EMB_BWD, &[&self.sampled_s1_ids, &self.d_sib, self.g("embedding.emb_s1.weight")], &[n, d, s1v], s1v * d));
        }

        // ---- s1 head backward, accumulating into d_xn ----
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * s1v));
        self.proj_bwd(&mut s, &self.d_logits, &self.xn_final, "head.proj_s1.weight", "head.proj_s1.bias", &self.d_xn, n, d, s1v, 1);
        // xn_final also feeds the dep residual (dep_sum = xn_final + o): d_xn += d_sum.
        s.push(self.gpu.step(ADD2, &[&self.d_xn, &self.d_sum, &self.d_xn2], &[n * d], n * d));
        self.rms_bwd(&mut s, &self.res[last], "norm.weight", &self.d_xn2, &self.dres[last], d, n);

        for l in (0..c.n_layers).rev() {
            let lb = &self.layers[l];
            let p = |nm: &str| format!("transformer.{l}.{nm}");
            // FFN backward (input grad = dres[l+1]); dW guarded for frozen (LoRA) params
            self.dw(&mut s, &self.dres[l + 1], &lb.h, &p("ffn.w2.weight"), n, ff, d);
            s.push(self.gpu.step(MATMUL_DX, &[&self.dres[l + 1], self.w(&p("ffn.w2.weight")), &self.d_h], &[n, ff, d, 0], n * ff));
            s.extend(block::swiglu_bwd(&self.gpu, &ids, &lb.gate, &lb.up, &self.d_h, &self.d_gate, &self.d_up, n * ff));
            self.dw(&mut s, &self.d_up, &lb.xn2, &p("ffn.w3.weight"), n, d, ff);
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_up, self.w(&p("ffn.w3.weight")), &self.d_xn], &[n, d, ff, 0], n * d));
            self.dw(&mut s, &self.d_gate, &lb.xn2, &p("ffn.w1.weight"), n, d, ff);
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_gate, self.w(&p("ffn.w1.weight")), &self.d_xn], &[n, d, ff, 1], n * d));
            self.rms_bwd(&mut s, &lb.xmid, &p("norm2.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));
            // attention backward (input grad = dxmid)
            self.proj_bwd(&mut s, &self.dxmid, &lb.ctx, &p("self_attn.out_proj.weight"), &p("self_attn.out_proj.bias"), &self.d_ctx, n, d, d, 0);
            s.extend(block::gqa_bwd(&self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &lb.probs, &self.d_ctx, &self.d_scores, &self.d_q, &self.d_k, &self.d_v));
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_q, n, heads, hd, d, t, ROPE_THETA));
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_k, n, heads, hd, d, t, ROPE_THETA));
            self.proj_bwd(&mut s, &self.d_v, &lb.xn1, &p("self_attn.v_proj.weight"), &p("self_attn.v_proj.bias"), &self.d_xn, n, d, d, 0);
            self.proj_bwd(&mut s, &self.d_k, &lb.xn1, &p("self_attn.k_proj.weight"), &p("self_attn.k_proj.bias"), &self.d_xn, n, d, d, 1);
            self.proj_bwd(&mut s, &self.d_q, &lb.xn1, &p("self_attn.q_proj.weight"), &p("self_attn.q_proj.bias"), &self.d_xn, n, d, d, 1);
            self.rms_bwd(&mut s, &self.res[l], &p("norm1.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // ---- embedding backward (input grad = dres[0]) ----
        // Input layer: weight grads only, no downstream dX. Under LoRA every
        // embedding param is frozen (shared role) → skip the whole section.
        if self.trainable("embedding.emb_s1.weight") {
            // temporal tables: each receives dres[0] (they enter by addition).
            for (ci, (name, _)) in CAL.iter().enumerate() {
                let size = CAL[ci].1 as u32;
                s.push(self.gpu.step(EMB_BWD, &[&self.cal_ids[ci], &self.dres[0], self.g(&format!("time_emb.{name}_embed.weight"))], &[n, d, size], size * d));
            }
            // fusion: xf = e1·fl^T + e2·fr^T + bias
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dres[0], self.g("embedding.fusion_proj.bias")], &[n, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dres[0], &self.e1, self.g("embedding.fusion_l")], &[n, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dres[0], self.w("embedding.fusion_l"), &self.d_e1], &[n, d, d, 0], n * d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dres[0], &self.e2, self.g("embedding.fusion_r")], &[n, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dres[0], self.w("embedding.fusion_r"), &self.d_e2], &[n, d, d, 0], n * d));
            // e = √d · gather(emb) → scale the grad, then scatter into the table.
            s.push(self.gpu.step(GRAD_SCALE, &[&self.d_e1], &[n * d, f(self.sqrt_d)], n * d));
            s.push(self.gpu.step(GRAD_SCALE, &[&self.d_e2], &[n * d, f(self.sqrt_d)], n * d));
            s.push(self.gpu.step(EMB_BWD, &[&self.s1_ids, &self.d_e1, self.g("embedding.emb_s1.weight")], &[n, d, s1v], s1v * d));
            s.push(self.gpu.step(EMB_BWD, &[&self.s2_ids, &self.d_e2, self.g("embedding.emb_s2.weight")], &[n, d, s2v], s2v * d));
        }
        s
    }

    // ---- driver surface (used by the bespoke gradcheck + train loop) ----

    /// Upload one batch: s1/s2 context token ids `[t]`, the five calendar-stamp
    /// columns `[t]` (minute,hour,weekday,day,month), the `sampled_s1` ids `[t]`
    /// feeding the dep sibling (a fixed **detached** input — the caller samples it
    /// from the model's own s1 prediction during training, matching the reference
    /// exposure-bias recipe), and the s1/s2 CE targets `[t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_batch(&self, s1: &[u32], s2: &[u32], cal: &[&[u32]; 5], sampled_s1: &[u32], s1_targets: &[u32], s2_targets: &[u32]) {
        // Batch-major: b sequences of t tokens concatenated → b*t total rows.
        let t = (self.b * self.t) as usize;
        assert!(s1.len() == t && s2.len() == t && s1_targets.len() == t && s2_targets.len() == t && sampled_s1.len() == t);
        self.gpu.write(&self.s1_ids, s1);
        self.gpu.write(&self.s2_ids, s2);
        for (buf, col) in self.cal_ids.iter().zip(cal.iter()) {
            assert_eq!(col.len(), t);
            self.gpu.write(buf, col);
        }
        self.gpu.write(&self.sampled_s1_ids, sampled_s1);
        self.gpu.write(&self.targets, s1_targets);
        self.gpu.write(&self.s2_targets, s2_targets);
        let cnt = s1_targets.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(cnt.max(1) as f32);
    }

    /// The combined objective `(CE_s1 + CE_s2) / 2`, mean over non-ignored positions.
    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        let t = (self.b * self.t) as usize;
        let l1: f32 = self.gpu.read(&self.ce_buf, t).iter().sum();
        let l2: f32 = self.gpu.read(&self.ce_buf2, t).iter().sum();
        (l1 + l2) / (2.0 * self.count.get())
    }
    pub fn backward(&self) {
        // 2·count folds the /2 of (CE_s1+CE_s2)/2 into both grad scalings.
        let c2 = f(2.0 * self.count.get());
        let rows = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[rows, self.cfg.s1_vocab() as u32, IGNORE, c2]);
        self.gpu.write(&self.ce_grad_uni2, &[rows, self.cfg.s2_vocab() as u32, IGNORE, c2]);
        self.gpu.submit(&[], &self.bwd_steps);
    }
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, 1.0);
    }
    pub fn config(&self) -> &KronosConfig {
        &self.cfg
    }
    /// Per-sequence context length `t`.
    pub fn seq_len(&self) -> u32 {
        self.t
    }
    /// Number of sequences trained per step (batch dimension `b`).
    pub fn batch(&self) -> u32 {
        self.b
    }
    pub fn param_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.ps.grad.keys().cloned().collect();
        v.sort();
        v
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
}

/// One pre-tokenized training example: the frozen-tokenizer output for a single
/// leak-safe OHLCV window (s1/s2 context tokens, the five calendar-stamp columns,
/// the exposure-bias sampled-s1, and the s1/s2 next-token targets).
#[derive(Clone)]
pub struct TokenBatch {
    pub s1: Vec<u32>,
    pub s2: Vec<u32>,
    pub cal: [Vec<u32>; 5],
    pub sampled_s1: Vec<u32>,
    pub s1_targets: Vec<u32>,
    pub s2_targets: Vec<u32>,
}

impl KronosTrain {
    /// Upload a [`TokenBatch`].
    pub fn set(&self, b: &TokenBatch) {
        let calr: [&[u32]; 5] = [&b.cal[0], &b.cal[1], &b.cal[2], &b.cal[3], &b.cal[4]];
        self.set_batch(&b.s1, &b.s2, &calr, &b.sampled_s1, &b.s1_targets, &b.s2_targets);
    }

    /// Upload `b` [`TokenBatch`]es batch-major (all `t` of window 0, then window
    /// 1, …) — the batched training step. `batches.len()` must equal the trainer's
    /// batch dim `b` and every window must have the trainer's length `t`.
    pub fn set_many(&self, batches: &[TokenBatch]) {
        assert_eq!(batches.len(), self.b as usize, "set_many expects exactly b={} windows", self.b);
        let cat = |f: &dyn Fn(&TokenBatch) -> &Vec<u32>| -> Vec<u32> { batches.iter().flat_map(|w| f(w).iter().copied()).collect() };
        let s1 = cat(&|w| &w.s1);
        let s2 = cat(&|w| &w.s2);
        let sampled_s1 = cat(&|w| &w.sampled_s1);
        let s1_targets = cat(&|w| &w.s1_targets);
        let s2_targets = cat(&|w| &w.s2_targets);
        let cal: [Vec<u32>; 5] = std::array::from_fn(|c| batches.iter().flat_map(|w| w.cal[c].iter().copied()).collect());
        let calr: [&[u32]; 5] = [&cal[0], &cal[1], &cal[2], &cal[3], &cal[4]];
        self.set_batch(&s1, &s2, &calr, &sampled_s1, &s1_targets, &s2_targets);
    }
    /// Mean forward loss over a set of batches (no gradient) — the held-out metric
    /// the promotion gate compares.
    pub fn mean_loss(&self, batches: &[TokenBatch]) -> f32 {
        if batches.is_empty() {
            return f32::NAN;
        }
        let mut sum = 0.0;
        for b in batches {
            self.set(b);
            sum += self.forward();
        }
        sum / batches.len() as f32
    }
    /// Every decoder weight in REFERENCE names, merging `fusion_l`/`fusion_r` back
    /// into `embedding.fusion_proj.weight` — ready for `checkpoint::save` / reload.
    pub fn to_reference_weights(&self) -> HashMap<String, Vec<f32>> {
        let d = self.cfg.d_model;
        let mut w = HashMap::new();
        for (name, _) in param_list_c(&self.cfg) {
            if name == "embedding.fusion_l" || name == "embedding.fusion_r" {
                continue;
            }
            let mut wv = self.ps.read_weight(&self.gpu, &name);
            // Under LoRA the base weight is frozen; fold the trained adapter into it
            // so the saved checkpoint carries the adaptation: W += (α/r)·(B·A),
            // A = lora_a [r, k], B = lora_b [nout, r], W [nout, k].
            if let Some((r, scale)) = self.lora_for(&name) {
                let r = r as usize;
                let a = self.ps.read_weight(&self.gpu, &format!("{name}.lora_a"));
                let b = self.ps.read_weight(&self.gpu, &format!("{name}.lora_b"));
                let k = a.len() / r; // A is [r, k]
                let nout = wv.len() / k;
                for o in 0..nout {
                    for j in 0..k {
                        let mut acc = 0.0f32;
                        for x in 0..r {
                            acc += b[o * r + x] * a[x * k + j];
                        }
                        wv[o * k + j] += scale * acc;
                    }
                }
            }
            w.insert(name.clone(), wv);
        }
        let fl = self.ps.read_weight(&self.gpu, "embedding.fusion_l");
        let fr = self.ps.read_weight(&self.gpu, "embedding.fusion_r");
        let mut fp = vec![0.0f32; d * 2 * d];
        for o in 0..d {
            fp[o * 2 * d..o * 2 * d + d].copy_from_slice(&fl[o * d..o * d + d]);
            fp[o * 2 * d + d..o * 2 * d + 2 * d].copy_from_slice(&fr[o * d..o * d + d]);
        }
        w.insert("embedding.fusion_proj.weight".into(), fp);
        w
    }
    /// Save as a brain decoder `.safetensors` checkpoint (loadable by the inference
    /// `KronosDecoder`/`KronosForecaster`).
    pub fn save(&self, path: &str) {
        let w = self.to_reference_weights();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .cfg
            .param_list()
            .into_iter()
            .map(|(n, shape)| (n.clone(), shape.iter().map(|&x| x as u64).collect(), w[&n].clone()))
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }
}

/// Fine-tuning hyperparameters (small LR + the anti-overfit defaults from the
/// reference recipe). `lora = Some(..)` freezes the base and trains adapters only.
#[derive(Clone)]
pub struct FinetuneOpts {
    pub epochs: u32,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub lora: Option<LoraCfg>,
    /// Windows trained per optimizer step (the batch dim `b`). `1` = the original
    /// one-window-per-step path. Higher `b` amortises the per-step overhead and
    /// fills the AVX/GPU lanes; the step is mathematically the mean of the `b`
    /// single steps (validated by the batch-parity gate). Uses drop_last — a
    /// trailing partial group of `< b` windows is skipped each epoch.
    pub batch: u32,
    /// Print per-batch progress (epoch, step/total, running loss, elapsed + ETA).
    pub progress: bool,
}

impl Default for FinetuneOpts {
    fn default() -> Self {
        FinetuneOpts { epochs: 8, lr: 4e-5, wd: 0.1, clip: 3.0, lora: None, batch: 1, progress: false }
    }
}

/// The gate decision: whether the fine-tuned checkpoint beat the base on held-out
/// data (and by how much).
#[derive(Clone, Debug)]
pub struct FinetuneReport {
    pub promoted: bool,
    pub base_val: f32,
    pub ft_val: f32,
    pub steps: u32,
}

/// Fine-tune the Kronos decoder on `train`, then **promote only if the held-out
/// `val` loss beats the untouched base model** — the walk-forward gate that makes
/// a weekly update earn its place instead of chasing noise. Returns the report and,
/// the fine-tuned decoder weights (reference-named). Returned unconditionally so
/// the caller can also evaluate generalization on a non-promoted candidate;
/// **save iff `report.promoted`** — the gate decision is the report, not the
/// presence of weights.
/// The tokenizer is not touched here (frozen upstream), matching the recipe.
pub fn finetune(
    cfg: KronosConfig,
    t: u32,
    base_init: &HashMap<String, Vec<f32>>,
    train: &[TokenBatch],
    val: &[TokenBatch],
    opts: &FinetuneOpts,
) -> (FinetuneReport, Option<HashMap<String, Vec<f32>>>) {
    // One device for the whole finetune: base evaluation and the fine-tuned
    // trainer share it instead of each building their own.
    let dev = Gpu::new(PIPELINES);
    let base_val = KronosTrain::with_lora_on(dev.share_or_new(PIPELINES), cfg.clone(), t, base_init, None).mean_loss(val);
    if opts.progress {
        eprintln!("  [finetune] base held-out loss {base_val:.4} · {} train / {} val windows · {} epoch(s)",
            train.len(), val.len(), opts.epochs);
    }
    // Batch `bsz` windows per step. `bsz` is clamped to the window count so a
    // small universe still trains (never zero steps); a trailing partial group is
    // dropped each epoch (a fixed-`b` trainer can't take a short group).
    let bsz = opts.batch.max(1).min(train.len().max(1) as u32);
    let ft = KronosTrain::with_lora_batch_on(dev, cfg.clone(), t, bsz, base_init, opts.lora.clone());
    let groups = train.chunks_exact(bsz as usize);
    let dropped = train.len() % bsz as usize;
    let per_epoch = train.len() / bsz as usize;
    if opts.progress && bsz > 1 {
        eprintln!("  [finetune] batch={bsz} → {per_epoch} step(s)/epoch ({dropped} window(s) dropped/epoch by drop_last)");
    }
    let total = (opts.epochs as usize * per_epoch).max(1);
    let every = (total / 40).max(1);
    let t0 = std::time::Instant::now();
    let (mut run, mut cnt) = (0.0f32, 0u32);
    let mut step = 0u32;
    for ep in 0..opts.epochs {
        for group in groups.clone() {
            step += 1;
            ft.set_many(group);
            ft.zero_grads();
            let l = ft.forward();
            ft.backward();
            ft.adamw_step(step, opts.lr, opts.wd, Some(opts.clip));
            run += l;
            cnt += 1;
            if opts.progress && ((step as usize).is_multiple_of(every) || step as usize == total) {
                let el = t0.elapsed().as_secs_f32();
                let frac = step as f32 / total as f32;
                let eta = if frac > 0.0 { el / frac - el } else { 0.0 };
                eprintln!(
                    "  [finetune] epoch {}/{}  step {step}/{total} ({:.0}%)  loss {:.4}  |  {:.0}s elapsed, ETA {:.0}s",
                    ep + 1, opts.epochs, frac * 100.0, run / cnt as f32, el, eta
                );
                run = 0.0;
                cnt = 0;
            }
        }
    }
    let ft_weights = ft.to_reference_weights();
    // Held-out eval: a batched trainer (b>1) has buffers sized for b*t tokens and
    // can't score single val windows, so evaluate on a fresh b=1 model of the
    // fine-tuned weights (identical to the base_val path). For b=1 the live trainer
    // scores directly, preserving the original gate value exactly.
    let ft_val = if bsz == 1 {
        ft.mean_loss(val)
    } else {
        KronosTrain::with_lora_on(Gpu::new(PIPELINES), cfg, t, &ft_weights, None).mean_loss(val)
    };
    let promoted = ft_val.is_finite() && base_val.is_finite() && ft_val < base_val;
    // Return the fine-tuned weights unconditionally so the caller can also evaluate
    // generalization (held-out names); the caller saves iff `promoted`.
    (FinetuneReport { promoted, base_val, ft_val, steps: step }, Some(ft_weights))
}
