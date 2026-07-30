// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encoder-decoder Transformer (ADR 0001 §5), forward + backprop as WGSL compute
//! dispatches. Shares the engine seam (`gpu_core`, `paramstore`, `optim`,
//! `kernels`) with the GPT/MoE/PID models and implements [`model::Model`] so the
//! generic trainer, sampler, and the blanket `gradcheck::CheckModel` cover it.
//!
//! Architecture (pre-LN throughout; GELU MLPs; dropout disabled):
//!   ENCODER (bidirectional self-attention):
//!     e = tok_emb[src] + enc_pos[pos]
//!     per enc block: h = LN1(e); e += Wo · BidirAttn(h)
//!                    h = LN2(e); e += proj · GELU(fc · h)
//!     enc_mem = e                                  (the encoder memory)
//!   DECODER (causal self-attention + cross-attention to enc_mem):
//!     d = tok_emb[tgt] + dec_pos[pos]
//!     per dec block: h = LN1(d); d += Wo · CausalAttn(h)
//!                    h = LN2(d); d += Wo · CrossAttn(q=h, kv=enc_mem)
//!                    h = LN3(d); d += proj · GELU(fc · h)
//!     logits = lm_head( LN_f(d) )                  (untied token head)
//!     loss   = masked cross-entropy over `labels` (ignore_index = IGNORE).
//!
//! Cross-attention buffer layout (matches the `attn_*_cross` kernels):
//!   * Q: a contiguous decoder buffer [B*T_dec, d] (q_stride = d, q_off = 0).
//!   * K/V: a per-layer encoder-memory FUSED-KV buffer [B*T_enc, 2*d]
//!     (kv_stride = 2*d, K at 0, V at d), produced by a fused `enc_mem @ Wkv`.
//!   * scores: ((b*H + h)*T_dec + i)*T_enc + j; softmax over T_enc.
//!
//! Design choices for v1 (documented per the SPEC):
//!   * Shared source/target token embeddings (`tok.weight`) — common when src/tgt
//!     share a vocabulary; separate positional tables for encoder vs decoder.
//!   * The encoder memory is the raw final encoder residual (no extra final
//!     encoder LayerNorm); each decoder cross-attn layer owns its own K/V
//!     projection of that memory.
//!   * `lm_head` is untied from `tok.weight` (one grad written per tensor —
//!     matches the rest of the engine and the FD gradient check).

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{f, Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;

/// Cross-entropy ignore index (masked label positions). Mirrors `gpt::IGNORE`.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const POS_ADD: usize = 1;
const LAYERNORM: usize = 2;
const LN_STATS: usize = 3;
const LN_DX: usize = 4;
const LN_DGAMMA: usize = 5;
const LN_DBETA: usize = 6;
const MATMUL: usize = 7;
const BIAS_ADD: usize = 8;
const BIAS_GRAD: usize = 9;
const GELU: usize = 10;
const GELU_BWD: usize = 11;
const CE_VALUE: usize = 12;
const CE_GRAD: usize = 13;
const MATMUL_DX: usize = 14;
const MATMUL_DW: usize = 15;
const POS_BWD: usize = 16;
const EMB_BWD: usize = 17;
const ADD2: usize = 18;
// causal self-attention (decoder)
const ATTN_SCORES: usize = 19;
const ATTN_SOFTMAX: usize = 20;
const ATTN_APPLY: usize = 21;
const ATTN_DSCORES: usize = 22;
const ATTN_DV: usize = 23;
const ATTN_DQ: usize = 24;
const ATTN_DK: usize = 25;
// bidirectional self-attention (encoder)
const ATTN_SCORES_BIDIR: usize = 26;
const ATTN_SOFTMAX_BIDIR: usize = 27;
const ATTN_APPLY_BIDIR: usize = 28;
const ATTN_DSCORES_BIDIR: usize = 29;
const ATTN_DV_BIDIR: usize = 30;
const ATTN_DQ_BIDIR: usize = 31;
const ATTN_DK_BIDIR: usize = 32;
// cross-attention (decoder -> encoder memory)
const ATTN_SCORES_CROSS: usize = 33;
const ATTN_SOFTMAX_CROSS: usize = 34;
const ATTN_APPLY_CROSS: usize = 35;
const ATTN_DSCORES_CROSS: usize = 36;
const ATTN_DV_CROSS: usize = 37;
const ATTN_DQ_CROSS: usize = 38;
const ATTN_DK_CROSS: usize = 39;
// optimizer
const GRADNORM_SQ: usize = 40;
const GRAD_SCALE: usize = 41;
const ADAMW: usize = 42;
const CLIP_COEF: usize = 43;
const GRAD_SCALE_BUF: usize = 44;
// Workgroup-per-row LayerNorm (2.3-9.1x the per-element kernels on a P40 — see
// `model::block::LayerNormIds`). Appended, so every index above is unchanged.
const LAYERNORM_ROWS: usize = 45;
const LN_STATS_ROWS: usize = 46;
const LN_DX_ROWS: usize = 47;

/// The LayerNorm family this model dispatches through `model::block`, which
/// picks the coalesced variant per device (`backend_api::select`).
const LN_IDS: model::block::LayerNormIds = model::block::LayerNormIds {
    layernorm: LAYERNORM,
    layernorm_rows: Some(LAYERNORM_ROWS),
    ln_stats: LN_STATS,
    ln_stats_rows: Some(LN_STATS_ROWS),
    layernorm_dx: LN_DX,
    layernorm_dx_rows: Some(LN_DX_ROWS),
};


pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("pos_add", kernels::POS_ADD),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    ("ce_value_masked", kernels::CE_VALUE_MASKED),
    ("ce_grad_masked", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("pos_bwd", kernels::POS_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("add2", kernels::ADD2),
    ("attn_scores", kernels::ATTN_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// Encoder-decoder configuration. `vocab`/`block_size` follow the `ModelConfig`
/// convention (`block_size` is the *decoder* block size, i.e. the seq length the
/// generic trainer/sampler reasons about); `src_block_size` is the encoder
/// length.
#[derive(Clone, Debug)]
pub struct Seq2SeqConfig {
    pub vocab: u32,
    /// Decoder (target) block size — the `ModelConfig::block_size`.
    pub block_size: u32,
    /// Encoder (source) block size.
    pub src_block_size: u32,
    pub n_enc: u32,
    pub n_dec: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_ff: u32,
}

impl Seq2SeqConfig {
    /// A tiny config for tests / gradient checks.
    pub fn tiny() -> Seq2SeqConfig {
        Seq2SeqConfig {
            vocab: 23,
            block_size: 8,
            src_block_size: 6,
            n_enc: 1,
            n_dec: 1,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
        }
    }

    /// nanogpt's `4 * d_model` feed-forward width default.
    pub fn with_ff_default(mut self) -> Self {
        if self.d_ff == 0 {
            self.d_ff = 4 * self.d_model;
        }
        self
    }

    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "seq2seq",
            "vocab_size": self.vocab,
            "block_size": self.block_size,
            "src_block_size": self.src_block_size,
            "n_enc": self.n_enc, "n_dec": self.n_dec,
            "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff,
            "shared_embeddings": true
        })
    }

    pub fn from_json(c: &Value) -> Seq2SeqConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        Seq2SeqConfig {
            vocab: g("vocab_size", 23),
            block_size: g("block_size", 8),
            src_block_size: g("src_block_size", 8),
            n_enc: g("n_enc", 1),
            n_dec: g("n_dec", 1),
            d_model: g("d_model", 16),
            n_heads: g("n_heads", 2),
            d_ff: g("d_ff", 32),
        }
    }

    /// Parameter list: `(name, numel)`. Order is irrelevant to correctness.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let mut out = vec![
            ("tok.weight".to_string(), v * d), // shared src/tgt embedding
            ("enc_pos.weight".to_string(), self.src_block_size as usize * d),
            ("dec_pos.weight".to_string(), self.block_size as usize * d),
        ];
        // Encoder blocks: pre-LN bidir self-attn + GELU MLP.
        for l in 0..self.n_enc {
            let p = |s: &str| format!("enc.blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("ln1.bias"), d));
            out.push((p("attn.qkv.weight"), 3 * d * d));
            out.push((p("attn.qkv.bias"), 3 * d));
            out.push((p("attn.out.weight"), d * d));
            out.push((p("attn.out.bias"), d));
            out.push((p("ln2.weight"), d));
            out.push((p("ln2.bias"), d));
            out.push((p("mlp.fc.weight"), ff * d));
            out.push((p("mlp.fc.bias"), ff));
            out.push((p("mlp.proj.weight"), d * ff));
            out.push((p("mlp.proj.bias"), d));
        }
        // Decoder blocks: pre-LN causal self-attn + cross-attn + GELU MLP.
        for l in 0..self.n_dec {
            let p = |s: &str| format!("dec.blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("ln1.bias"), d));
            out.push((p("attn.qkv.weight"), 3 * d * d));
            out.push((p("attn.qkv.bias"), 3 * d));
            out.push((p("attn.out.weight"), d * d));
            out.push((p("attn.out.bias"), d));
            out.push((p("ln2.weight"), d));
            out.push((p("ln2.bias"), d));
            // cross-attention: q from decoder hidden (d), fused K|V from enc memory (2d).
            out.push((p("cross.q.weight"), d * d));
            out.push((p("cross.q.bias"), d));
            out.push((p("cross.kv.weight"), 2 * d * d));
            out.push((p("cross.kv.bias"), 2 * d));
            out.push((p("cross.out.weight"), d * d));
            out.push((p("cross.out.bias"), d));
            out.push((p("ln3.weight"), d));
            out.push((p("ln3.bias"), d));
            out.push((p("mlp.fc.weight"), ff * d));
            out.push((p("mlp.fc.bias"), ff));
            out.push((p("mlp.proj.weight"), d * ff));
            out.push((p("mlp.proj.bias"), d));
        }
        out.push(("ln.weight".to_string(), d));
        out.push(("ln.bias".to_string(), d));
        out.push(("lm_head.weight".to_string(), v * d)); // untied, no bias
        out
    }
}

/// Per-encoder-block forward activation cache (SSA buffers).
struct EncLayer {
    ln1_out: gpu_core::DeviceBuffer,
    qkv: gpu_core::DeviceBuffer,
    scores: gpu_core::DeviceBuffer,
    probs: gpu_core::DeviceBuffer,
    attn_ctx: gpu_core::DeviceBuffer,
    proj: gpu_core::DeviceBuffer,
    xmid: gpu_core::DeviceBuffer,
    ln2_out: gpu_core::DeviceBuffer,
    fc: gpu_core::DeviceBuffer,
    gelu: gpu_core::DeviceBuffer,
    ffn_out: gpu_core::DeviceBuffer,
}

/// Per-decoder-block forward activation cache (SSA buffers).
struct DecLayer {
    // causal self-attention
    ln1_out: gpu_core::DeviceBuffer,
    qkv: gpu_core::DeviceBuffer,
    scores: gpu_core::DeviceBuffer,
    probs: gpu_core::DeviceBuffer,
    attn_ctx: gpu_core::DeviceBuffer,
    sa_proj: gpu_core::DeviceBuffer,
    xa: gpu_core::DeviceBuffer, // residual after self-attn
    // cross-attention
    ln2_out: gpu_core::DeviceBuffer,
    cq: gpu_core::DeviceBuffer,  // [n_dec, d]
    ckv: gpu_core::DeviceBuffer, // [n_enc, 2d] fused K|V (encoder memory)
    cscores: gpu_core::DeviceBuffer,
    cprobs: gpu_core::DeviceBuffer,
    cctx: gpu_core::DeviceBuffer,
    ca_proj: gpu_core::DeviceBuffer,
    xc: gpu_core::DeviceBuffer, // residual after cross-attn
    // MLP
    ln3_out: gpu_core::DeviceBuffer,
    fc: gpu_core::DeviceBuffer,
    gelu: gpu_core::DeviceBuffer,
    ffn_out: gpu_core::DeviceBuffer,
}

pub struct Seq2Seq {
    pub gpu: Gpu,
    pub cfg: Seq2SeqConfig,
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    t_dec: u32,
    t_enc: u32,
    count: Cell<f32>,

    src: gpu_core::DeviceBuffer,
    tgt: gpu_core::DeviceBuffer,
    labels: gpu_core::DeviceBuffer,

    // encoder forward
    enc_res: Vec<gpu_core::DeviceBuffer>, // [n_enc+1] residual stream
    enc_layers: Vec<EncLayer>,

    // decoder forward
    dec_res: Vec<gpu_core::DeviceBuffer>, // [n_dec+1] residual stream
    dec_layers: Vec<DecLayer>,
    xn_final: gpu_core::DeviceBuffer,
    logits: gpu_core::DeviceBuffer,
    ce_buf: gpu_core::DeviceBuffer,

    // backward temporaries (decoder residual grads + scratch)
    dec_dres: Vec<gpu_core::DeviceBuffer>,
    enc_dres: Vec<gpu_core::DeviceBuffer>,
    // Encoder-memory grad accumulators (ping-pong, out-of-place ADD2 to avoid
    // binding the same buffer as both read and read_write). Each decoder layer's
    // cross-attn contributes its K/V grad; `d_enc_mem[01]` alternate as the
    // running sum, seeded from a cleared buffer.
    d_enc_mem0: gpu_core::DeviceBuffer,
    d_enc_mem1: gpu_core::DeviceBuffer,
    d_enc_mem_tmp: gpu_core::DeviceBuffer,
    d_logits: gpu_core::DeviceBuffer,
    d_xn: gpu_core::DeviceBuffer,
    d_branch: gpu_core::DeviceBuffer,
    d_tmp: gpu_core::DeviceBuffer,
    d_acc: gpu_core::DeviceBuffer,
    d_acc2: gpu_core::DeviceBuffer,
    d_attn_ctx: gpu_core::DeviceBuffer,
    d_scores: gpu_core::DeviceBuffer,
    d_qkv: gpu_core::DeviceBuffer,
    d_gelu: gpu_core::DeviceBuffer,
    d_fc: gpu_core::DeviceBuffer,
    // cross-attn backward scratch
    d_cctx: gpu_core::DeviceBuffer,
    d_cscores: gpu_core::DeviceBuffer,
    d_cq: gpu_core::DeviceBuffer,  // [n_dec, d]
    d_ckv: gpu_core::DeviceBuffer, // [n_enc, 2d]
    // encoder-scoped scratch (sized to n_enc)
    enc_d_branch: gpu_core::DeviceBuffer,
    enc_d_tmp: gpu_core::DeviceBuffer,
    enc_d_acc: gpu_core::DeviceBuffer,
    enc_d_attn_ctx: gpu_core::DeviceBuffer,
    enc_d_scores: gpu_core::DeviceBuffer,
    enc_d_qkv: gpu_core::DeviceBuffer,
    enc_d_gelu: gpu_core::DeviceBuffer,
    enc_d_fc: gpu_core::DeviceBuffer,
    ln_mean: gpu_core::DeviceBuffer,
    ln_inv: gpu_core::DeviceBuffer,
    enc_ln_mean: gpu_core::DeviceBuffer,
    enc_ln_inv: gpu_core::DeviceBuffer,

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: gpu_core::DeviceBuffer,
}

impl Seq2Seq {
    pub fn load(path: &str, b: u32, t: u32) -> Seq2Seq {
        let c = checkpoint::load(path);
        let cfg = Seq2SeqConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Seq2Seq::new(cfg, b, t, &init)
    }

    pub fn new(cfg: Seq2SeqConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Seq2Seq {
        Seq2Seq::new_on(Gpu::new(PIPELINES), cfg, b, t, init)
    }

    /// Build on an existing device handle — see `Gpt::new_on`.
    pub fn new_on(gpu: Gpu, cfg: Seq2SeqConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Seq2Seq {
        let ps = ParamStore::new(&gpu, cfg.param_list(), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let t_dec = t;
        let t_enc = cfg.src_block_size;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let h = cfg.n_heads;
        let nd = (b * t_dec) as u64; // decoder tokens
        let ne = (b * t_enc) as u64; // encoder tokens
        let bht_dd = (b * h * t_dec * t_dec) as u64; // causal self-attn scores
        let bht_de = (b * h * t_dec * t_enc) as u64; // cross-attn scores
        let bht_ee = (b * h * t_enc * t_enc) as u64; // encoder self-attn scores
        let st = |x: u64| gpu.storage(x);

        let mk_tok = |label: &str, n: u64| {
            gpu.buffer(label, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };
        let src = mk_tok("src", ne);
        let tgt = mk_tok("tgt", nd);
        let labels = mk_tok("labels", nd);
        let ce_grad_uni = gpu.uniform_dynamic(4);

        let mut enc_res = Vec::new();
        let mut enc_dres = Vec::new();
        for _ in 0..=cfg.n_enc {
            enc_res.push(st(ne * d));
            enc_dres.push(st(ne * d));
        }
        let mut enc_layers = Vec::new();
        for _ in 0..cfg.n_enc {
            enc_layers.push(EncLayer {
                ln1_out: st(ne * d),
                qkv: st(ne * 3 * d),
                scores: st(bht_ee),
                probs: st(bht_ee),
                attn_ctx: st(ne * d),
                proj: st(ne * d),
                xmid: st(ne * d),
                ln2_out: st(ne * d),
                fc: st(ne * ff),
                gelu: st(ne * ff),
                ffn_out: st(ne * d),
            });
        }

        let mut dec_res = Vec::new();
        let mut dec_dres = Vec::new();
        for _ in 0..=cfg.n_dec {
            dec_res.push(st(nd * d));
            dec_dres.push(st(nd * d));
        }
        let mut dec_layers = Vec::new();
        for _ in 0..cfg.n_dec {
            dec_layers.push(DecLayer {
                ln1_out: st(nd * d),
                qkv: st(nd * 3 * d),
                scores: st(bht_dd),
                probs: st(bht_dd),
                attn_ctx: st(nd * d),
                sa_proj: st(nd * d),
                xa: st(nd * d),
                ln2_out: st(nd * d),
                cq: st(nd * d),
                ckv: st(ne * 2 * d),
                cscores: st(bht_de),
                cprobs: st(bht_de),
                cctx: st(nd * d),
                ca_proj: st(nd * d),
                xc: st(nd * d),
                ln3_out: st(nd * d),
                fc: st(nd * ff),
                gelu: st(nd * ff),
                ffn_out: st(nd * d),
            });
        }

        let mut m = Seq2Seq {
            cfg,
            b,
            t_dec,
            t_enc,
            count: Cell::new(1.0),
            ps,
            opt,
            src,
            tgt,
            labels,
            enc_res,
            enc_layers,
            dec_res,
            dec_layers,
            xn_final: st(nd * d),
            logits: st(nd * v),
            ce_buf: st(nd),
            dec_dres,
            enc_dres,
            d_enc_mem0: st(ne * d),
            d_enc_mem1: st(ne * d),
            d_enc_mem_tmp: st(ne * d),
            d_logits: st(nd * v),
            d_xn: st(nd * d),
            d_branch: st(nd * d),
            d_tmp: st(nd * d),
            d_acc: st(nd * d),
            d_acc2: st(nd * d),
            d_attn_ctx: st(nd * d),
            d_scores: st(bht_dd),
            d_qkv: st(nd * 3 * d),
            d_gelu: st(nd * ff),
            d_fc: st(nd * ff),
            d_cctx: st(nd * d),
            d_cscores: st(bht_de),
            d_cq: st(nd * d),
            d_ckv: st(ne * 2 * d),
            enc_d_branch: st(ne * d),
            enc_d_tmp: st(ne * d),
            enc_d_acc: st(ne * d),
            enc_d_attn_ctx: st(ne * d),
            enc_d_scores: st(bht_ee),
            enc_d_qkv: st(ne * 3 * d),
            enc_d_gelu: st(ne * ff),
            enc_d_fc: st(ne * ff),
            ln_mean: st(nd),
            ln_inv: st(nd),
            enc_ln_mean: st(ne),
            enc_ln_inv: st(ne),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        m.bwd_steps = m.build_backward_steps();
        m
    }

    /// Upload one seq2seq batch. `src` feeds the encoder ([`Seq2SeqConfig::src_block_size`]
    /// tokens per sequence), `tgt` the decoder, `labels` the decoder next-token
    /// targets (IGNORE masks padding).
    pub fn set_batch(&self, src: &[u32], tgt: &[u32], labels: &[u32]) {
        self.gpu.write(&self.src, src);
        self.gpu.write(&self.tgt, tgt);
        self.gpu.write(&self.labels, labels);
        let c = labels.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.ps.w(name)
    }

    /// Kernel-index map for the shared bidirectional-attention builders.
    fn bidir_ids() -> model::block::BidirIds {
        model::block::BidirIds {
            scores: ATTN_SCORES_BIDIR,
            softmax: ATTN_SOFTMAX_BIDIR,
            apply: ATTN_APPLY_BIDIR,
            dscores: ATTN_DSCORES_BIDIR,
            dv: ATTN_DV_BIDIR,
            dq: ATTN_DQ_BIDIR,
            dk: ATTN_DK_BIDIR,
        }
    }

    /// Encoder self-attention shape over the fused `[q|k|v]` buffer.
    fn enc_bidir(&self) -> model::block::Bidir {
        let d = self.cfg.d_model;
        model::block::Bidir {
            b: self.b,
            t: self.t_enc,
            n_heads: self.cfg.n_heads,
            head_dim: self.cfg.head_dim(),
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let h = c.n_heads;
        let b = self.b;
        let te = self.t_enc;
        let td = self.t_dec;
        let ne = b * te;
        let nd = b * td;
        let mut s: Vec<Step> = Vec::new();

        // ---- ENCODER ----
        s.push(self.gpu.step(EMBED, &[&self.src, self.w("tok.weight"), &self.enc_res[0]], &[d, ne], ne * d));
        s.push(self.gpu.step(POS_ADD, &[&self.enc_res[0], self.w("enc_pos.weight")], &[ne * d, d, te], ne * d));
        for l in 0..c.n_enc as usize {
            let lb = &self.enc_layers[l];
            let p = |name: &str| format!("enc.blocks.{l}.{name}");
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &self.enc_res[l], self.w(&p("ln1.weight")), self.w(&p("ln1.bias")), &lb.ln1_out, d, ne, 1e-5));
            s.push(self.gpu.step(MATMUL, &[&lb.ln1_out, self.w(&p("attn.qkv.weight")), &lb.qkv], &[ne, d, 3 * d], ne * 3 * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.qkv, self.w(&p("attn.qkv.bias"))], &[ne, 3 * d], ne * 3 * d));
            // bidir self-attn via the shared builder: q_off=0, k_off=d, v_off=2d
            s.extend(model::block::bidir_fwd(&self.gpu, &Self::bidir_ids(), &self.enc_bidir(), &lb.qkv, &lb.scores, &lb.probs, &lb.attn_ctx));
            s.push(self.gpu.step(MATMUL, &[&lb.attn_ctx, self.w(&p("attn.out.weight")), &lb.proj], &[ne, d, d], ne * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.proj, self.w(&p("attn.out.bias"))], &[ne, d], ne * d));
            s.push(self.gpu.step(ADD2, &[&self.enc_res[l], &lb.proj, &lb.xmid], &[ne * d], ne * d));
            // MLP
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &lb.xmid, self.w(&p("ln2.weight")), self.w(&p("ln2.bias")), &lb.ln2_out, d, ne, 1e-5));
            s.push(self.gpu.step(MATMUL, &[&lb.ln2_out, self.w(&p("mlp.fc.weight")), &lb.fc], &[ne, d, ff], ne * ff));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.fc, self.w(&p("mlp.fc.bias"))], &[ne, ff], ne * ff));
            s.push(self.gpu.step(GELU, &[&lb.fc, &lb.gelu], &[ne * ff], ne * ff));
            s.push(self.gpu.step(MATMUL, &[&lb.gelu, self.w(&p("mlp.proj.weight")), &lb.ffn_out], &[ne, ff, d], ne * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.ffn_out, self.w(&p("mlp.proj.bias"))], &[ne, d], ne * d));
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &lb.ffn_out, &self.enc_res[l + 1]], &[ne * d], ne * d));
        }
        let enc_mem = &self.enc_res[c.n_enc as usize];

        // ---- DECODER ----
        s.push(self.gpu.step(EMBED, &[&self.tgt, self.w("tok.weight"), &self.dec_res[0]], &[d, nd], nd * d));
        s.push(self.gpu.step(POS_ADD, &[&self.dec_res[0], self.w("dec_pos.weight")], &[nd * d, d, td], nd * d));
        for l in 0..c.n_dec as usize {
            let lb = &self.dec_layers[l];
            let p = |name: &str| format!("dec.blocks.{l}.{name}");
            // causal self-attention
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &self.dec_res[l], self.w(&p("ln1.weight")), self.w(&p("ln1.bias")), &lb.ln1_out, d, nd, 1e-5));
            s.push(self.gpu.step(MATMUL, &[&lb.ln1_out, self.w(&p("attn.qkv.weight")), &lb.qkv], &[nd, d, 3 * d], nd * 3 * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.qkv, self.w(&p("attn.qkv.bias"))], &[nd, 3 * d], nd * 3 * d));
            s.push(self.gpu.step(ATTN_SCORES, &[&lb.qkv, &lb.scores], &[b, h, td, hd, 3 * d, 0, d], b * h * td * td));
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&lb.scores, &lb.probs], &[b, h, td], b * h * td));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_ctx], &[b, h, td, hd, 3 * d, 2 * d, d], b * h * td * hd));
            s.push(self.gpu.step(MATMUL, &[&lb.attn_ctx, self.w(&p("attn.out.weight")), &lb.sa_proj], &[nd, d, d], nd * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.sa_proj, self.w(&p("attn.out.bias"))], &[nd, d], nd * d));
            s.push(self.gpu.step(ADD2, &[&self.dec_res[l], &lb.sa_proj, &lb.xa], &[nd * d], nd * d));

            // cross-attention to encoder memory
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &lb.xa, self.w(&p("ln2.weight")), self.w(&p("ln2.bias")), &lb.ln2_out, d, nd, 1e-5));
            // q = ln2_out @ Wq + bq  -> contiguous [nd, d] (q_stride=d)
            s.push(self.gpu.step(MATMUL, &[&lb.ln2_out, self.w(&p("cross.q.weight")), &lb.cq], &[nd, d, d], nd * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.cq, self.w(&p("cross.q.bias"))], &[nd, d], nd * d));
            // kv = enc_mem @ Wkv + bkv -> fused [ne, 2d] (kv_stride=2d, K@0, V@d)
            s.push(self.gpu.step(MATMUL, &[enc_mem, self.w(&p("cross.kv.weight")), &lb.ckv], &[ne, d, 2 * d], ne * 2 * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.ckv, self.w(&p("cross.kv.bias"))], &[ne, 2 * d], ne * 2 * d));
            // cross scores/softmax/apply: q_stride=d, kv_stride=2d, q_off=0, k_off=0, v_off=d
            s.push(self.gpu.step(ATTN_SCORES_CROSS, &[&lb.cq, &lb.ckv, &lb.cscores], &[b, h, td, te, hd, d, 2 * d, 0, 0], b * h * td * te));
            s.push(self.gpu.step(ATTN_SOFTMAX_CROSS, &[&lb.cscores, &lb.cprobs], &[b, h, td, te], b * h * td));
            s.push(self.gpu.step(ATTN_APPLY_CROSS, &[&lb.cprobs, &lb.ckv, &lb.cctx], &[b, h, td, te, hd, 2 * d, d, d], b * h * td * hd));
            s.push(self.gpu.step(MATMUL, &[&lb.cctx, self.w(&p("cross.out.weight")), &lb.ca_proj], &[nd, d, d], nd * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.ca_proj, self.w(&p("cross.out.bias"))], &[nd, d], nd * d));
            s.push(self.gpu.step(ADD2, &[&lb.xa, &lb.ca_proj, &lb.xc], &[nd * d], nd * d));

            // MLP
            s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &lb.xc, self.w(&p("ln3.weight")), self.w(&p("ln3.bias")), &lb.ln3_out, d, nd, 1e-5));
            s.push(self.gpu.step(MATMUL, &[&lb.ln3_out, self.w(&p("mlp.fc.weight")), &lb.fc], &[nd, d, ff], nd * ff));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.fc, self.w(&p("mlp.fc.bias"))], &[nd, ff], nd * ff));
            s.push(self.gpu.step(GELU, &[&lb.fc, &lb.gelu], &[nd * ff], nd * ff));
            s.push(self.gpu.step(MATMUL, &[&lb.gelu, self.w(&p("mlp.proj.weight")), &lb.ffn_out], &[nd, ff, d], nd * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.ffn_out, self.w(&p("mlp.proj.bias"))], &[nd, d], nd * d));
            s.push(self.gpu.step(ADD2, &[&lb.xc, &lb.ffn_out, &self.dec_res[l + 1]], &[nd * d], nd * d));
        }

        let last = c.n_dec as usize;
        s.push(model::block::layernorm_fwd(&self.gpu, &LN_IDS, &self.dec_res[last], self.w("ln.weight"), self.w("ln.bias"), &self.xn_final, d, nd, 1e-5));
        s.push(self.gpu.step(MATMUL, &[&self.xn_final, self.w("lm_head.weight"), &self.logits], &[nd, d, v], nd * v));
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.labels, &self.ce_buf], &[nd, v, IGNORE], nd));
        s
    }

    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd_steps);
    }

    pub fn loss(&self) -> f32 {
        let nd = (self.b * self.t_dec) as usize;
        let losses = self.gpu.read(&self.ce_buf, nd);
        losses.iter().sum::<f32>() / self.count.get()
    }

    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    pub fn backward(&self) {
        let nd = self.b * self.t_dec;
        let v = self.cfg.vocab;
        self.gpu.write(&self.ce_grad_uni, &[nd, v, IGNORE, f(self.count.get())]);
        // Clear the encoder-memory grad accumulator seed (index 0) so the first
        // cross-attn contribution adds to zero.
        self.gpu.submit(&[&self.d_enc_mem0], &self.bwd_steps);
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let h = c.n_heads;
        let b = self.b;
        let te = self.t_enc;
        let td = self.t_dec;
        let ne = b * te;
        let nd = b * td;
        let g = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();
        let enc_mem = &self.enc_res[c.n_enc as usize];
        // Ping-pong accumulator parity for the encoder-memory grad: starts at the
        // cleared `d_enc_mem0`; each decoder layer flips it.
        let mem_acc = |i: u32| if i % 2 == 0 { &self.d_enc_mem0 } else { &self.d_enc_mem1 };
        let mut mem_idx: u32 = 0;

        // ---- head + final LN ----
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.labels, &self.d_logits], nd * v));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_logits, &self.xn_final, g("lm_head.weight")], &[nd, d, v], v * d));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_logits, self.w("lm_head.weight"), &self.d_xn], &[nd, d, v, 0], nd * d));
        let last = c.n_dec as usize;
        s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &self.dec_res[last], &self.ln_mean, &self.ln_inv, d, nd, 1e-5));
        s.push(self.gpu.step(LN_DGAMMA, &[&self.d_xn, &self.dec_res[last], &self.ln_mean, &self.ln_inv, g("ln.weight")], &[d, nd], d));
        s.push(self.gpu.step(LN_DBETA, &[&self.d_xn, g("ln.bias")], &[d, nd], d));
        s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &self.dec_res[last], self.w("ln.weight"), &self.d_xn, &self.dec_dres[last], d, nd, 1e-5));

        // ---- DECODER blocks (reverse) ----
        // d_enc_mem accumulates cross-attn K/V grads from every decoder layer; it
        // is cleared once per backward submit (see `backward`).
        for l in (0..c.n_dec as usize).rev() {
            let lb = &self.dec_layers[l];
            let p = |name: &str| format!("dec.blocks.{l}.{name}");

            // MLP backward; input grad = dec_dres[l+1]
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dec_dres[l + 1], g(&p("mlp.proj.bias"))], &[nd, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dec_dres[l + 1], &lb.gelu, g(&p("mlp.proj.weight"))], &[nd, ff, d], d * ff));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dec_dres[l + 1], self.w(&p("mlp.proj.weight")), &self.d_gelu], &[nd, ff, d, 0], nd * ff));
            s.push(self.gpu.step(GELU_BWD, &[&lb.fc, &self.d_gelu, &self.d_fc], &[nd * ff], nd * ff));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_fc, g(&p("mlp.fc.bias"))], &[nd, ff], ff));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_fc, &lb.ln3_out, g(&p("mlp.fc.weight"))], &[nd, d, ff], ff * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_fc, self.w(&p("mlp.fc.weight")), &self.d_branch], &[nd, d, ff, 0], nd * d));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &lb.xc, &self.ln_mean, &self.ln_inv, d, nd, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &lb.xc, &self.ln_mean, &self.ln_inv, g(&p("ln3.weight"))], &[d, nd], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p("ln3.bias"))], &[d, nd], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &lb.xc, self.w(&p("ln3.weight")), &self.d_branch, &self.d_tmp, d, nd, 1e-5));
            // grad into xc residual = dec_dres[l+1] + d_tmp
            s.push(self.gpu.step(ADD2, &[&self.dec_dres[l + 1], &self.d_tmp, &self.d_acc], &[nd * d], nd * d));

            // cross-attention backward; input (to xc) grad = d_acc
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_acc, g(&p("cross.out.bias"))], &[nd, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_acc, &lb.cctx, g(&p("cross.out.weight"))], &[nd, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_acc, self.w(&p("cross.out.weight")), &self.d_cctx], &[nd, d, d, 0], nd * d));
            // d_cctx -> d_cscores (softmax jac, uses ckv V), d_ckv (V), d_cq, d_ckv (K)
            s.push(self.gpu.step(ATTN_DSCORES_CROSS, &[&self.d_cctx, &lb.ckv, &lb.cprobs, &self.d_cscores], &[b, h, td, te, hd, 2 * d, d, d], b * h * td));
            s.push(self.gpu.step(ATTN_DV_CROSS, &[&lb.cprobs, &self.d_cctx, &self.d_ckv], &[b, h, td, te, hd, 2 * d, d, d], b * h * te * hd));
            s.push(self.gpu.step(ATTN_DQ_CROSS, &[&self.d_cscores, &lb.ckv, &self.d_cq], &[b, h, td, te, hd, d, 2 * d, 0, 0], b * h * td * hd));
            s.push(self.gpu.step(ATTN_DK_CROSS, &[&self.d_cscores, &lb.cq, &self.d_ckv], &[b, h, td, te, hd, d, 2 * d, 0, 0], b * h * te * hd));
            // d_cq -> grad through q proj into ln2_out branch, + cross q bias
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_cq, g(&p("cross.q.bias"))], &[nd, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_cq, &lb.ln2_out, g(&p("cross.q.weight"))], &[nd, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_cq, self.w(&p("cross.q.weight")), &self.d_branch], &[nd, d, d, 0], nd * d));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &lb.xa, &self.ln_mean, &self.ln_inv, d, nd, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &lb.xa, &self.ln_mean, &self.ln_inv, g(&p("ln2.weight"))], &[d, nd], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p("ln2.bias"))], &[d, nd], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &lb.xa, self.w(&p("ln2.weight")), &self.d_branch, &self.d_tmp, d, nd, 1e-5));
            // grad into xa residual = d_acc + d_tmp (self-attn-branch input grad).
            // Out-of-place into d_acc2 (avoid binding d_acc as read + read_write).
            s.push(self.gpu.step(ADD2, &[&self.d_acc, &self.d_tmp, &self.d_acc2], &[nd * d], nd * d));

            // d_ckv -> grad through kv proj into encoder memory; accumulate into the
            // ping-pong encoder-memory grad (out-of-place: prev acc + contrib -> next).
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_ckv, g(&p("cross.kv.bias"))], &[ne, 2 * d], 2 * d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_ckv, enc_mem, g(&p("cross.kv.weight"))], &[ne, d, 2 * d], 2 * d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_ckv, self.w(&p("cross.kv.weight")), &self.d_enc_mem_tmp], &[ne, d, 2 * d, 0], ne * d));
            s.push(self.gpu.step(ADD2, &[mem_acc(mem_idx), &self.d_enc_mem_tmp, mem_acc(mem_idx + 1)], &[ne * d], ne * d));
            mem_idx += 1;

            // causal self-attention backward; input grad = d_acc2
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_acc2, g(&p("attn.out.bias"))], &[nd, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_acc2, &lb.attn_ctx, g(&p("attn.out.weight"))], &[nd, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_acc2, self.w(&p("attn.out.weight")), &self.d_attn_ctx], &[nd, d, d, 0], nd * d));
            s.push(self.gpu.step(ATTN_DSCORES, &[&self.d_attn_ctx, &lb.qkv, &lb.probs, &self.d_scores], &[b, h, td, hd, 3 * d, 2 * d, d], b * h * td));
            s.push(self.gpu.step(ATTN_DV, &[&lb.probs, &self.d_attn_ctx, &self.d_qkv], &[b, h, td, hd, 3 * d, 2 * d, d], b * h * td * hd));
            s.push(self.gpu.step(ATTN_DQ, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[b, h, td, hd, 3 * d, 0, d], b * h * td * hd));
            s.push(self.gpu.step(ATTN_DK, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[b, h, td, hd, 3 * d, 0, d], b * h * td * hd));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_qkv, g(&p("attn.qkv.bias"))], &[nd, 3 * d], 3 * d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_qkv, &lb.ln1_out, g(&p("attn.qkv.weight"))], &[nd, d, 3 * d], 3 * d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_qkv, self.w(&p("attn.qkv.weight")), &self.d_branch], &[nd, d, 3 * d, 0], nd * d));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &self.dec_res[l], &self.ln_mean, &self.ln_inv, d, nd, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &self.dec_res[l], &self.ln_mean, &self.ln_inv, g(&p("ln1.weight"))], &[d, nd], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p("ln1.bias"))], &[d, nd], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &self.dec_res[l], self.w(&p("ln1.weight")), &self.d_branch, &self.d_tmp, d, nd, 1e-5));
            // grad into dec_res[l] = d_acc2 + d_tmp
            s.push(self.gpu.step(ADD2, &[&self.d_acc2, &self.d_tmp, &self.dec_dres[l]], &[nd * d], nd * d));
        }

        // ---- ENCODER blocks (reverse) ----
        // Seed the encoder residual grad from the accumulated cross-attn memory
        // grad. After `mem_idx` flips, `mem_acc(mem_idx)` holds the running sum
        // (the final accumulator); use it directly as the encoder's top grad. If
        // there are no decoder layers, `mem_acc(0)` is the cleared zero seed.
        let enc_last = c.n_enc as usize;
        let enc_top = mem_acc(mem_idx);

        for l in (0..c.n_enc as usize).rev() {
            let lb = &self.enc_layers[l];
            let p = |name: &str| format!("enc.blocks.{l}.{name}");
            // MLP backward; input grad = grad into enc_res[l+1]. The top encoder
            // layer receives the accumulated cross-attn memory grad (`enc_top`);
            // lower layers receive the residual grad written by the layer above.
            let upstream = if l + 1 == enc_last { enc_top } else { &self.enc_dres[l + 1] };
            s.push(self.gpu.step(BIAS_GRAD, &[upstream, g(&p("mlp.proj.bias"))], &[ne, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[upstream, &lb.gelu, g(&p("mlp.proj.weight"))], &[ne, ff, d], d * ff));
            s.push(self.gpu.step(MATMUL_DX, &[upstream, self.w(&p("mlp.proj.weight")), &self.enc_d_gelu], &[ne, ff, d, 0], ne * ff));
            s.push(self.gpu.step(GELU_BWD, &[&lb.fc, &self.enc_d_gelu, &self.enc_d_fc], &[ne * ff], ne * ff));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.enc_d_fc, g(&p("mlp.fc.bias"))], &[ne, ff], ff));
            s.push(self.gpu.step(MATMUL_DW, &[&self.enc_d_fc, &lb.ln2_out, g(&p("mlp.fc.weight"))], &[ne, d, ff], ff * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.enc_d_fc, self.w(&p("mlp.fc.weight")), &self.enc_d_branch], &[ne, d, ff, 0], ne * d));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &lb.xmid, &self.enc_ln_mean, &self.enc_ln_inv, d, ne, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.enc_d_branch, &lb.xmid, &self.enc_ln_mean, &self.enc_ln_inv, g(&p("ln2.weight"))], &[d, ne], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.enc_d_branch, g(&p("ln2.bias"))], &[d, ne], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &lb.xmid, self.w(&p("ln2.weight")), &self.enc_d_branch, &self.enc_d_tmp, d, ne, 1e-5));
            s.push(self.gpu.step(ADD2, &[upstream, &self.enc_d_tmp, &self.enc_d_acc], &[ne * d], ne * d));

            // bidir self-attention backward; input grad = enc_d_acc
            s.push(self.gpu.step(BIAS_GRAD, &[&self.enc_d_acc, g(&p("attn.out.bias"))], &[ne, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.enc_d_acc, &lb.attn_ctx, g(&p("attn.out.weight"))], &[ne, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.enc_d_acc, self.w(&p("attn.out.weight")), &self.enc_d_attn_ctx], &[ne, d, d, 0], ne * d));
            s.extend(model::block::bidir_bwd(
                &self.gpu, &Self::bidir_ids(), &self.enc_bidir(), &lb.qkv, &lb.probs,
                &self.enc_d_attn_ctx, &self.enc_d_scores, &self.enc_d_qkv,
            ));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.enc_d_qkv, g(&p("attn.qkv.bias"))], &[ne, 3 * d], 3 * d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.enc_d_qkv, &lb.ln1_out, g(&p("attn.qkv.weight"))], &[ne, d, 3 * d], 3 * d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.enc_d_qkv, self.w(&p("attn.qkv.weight")), &self.enc_d_branch], &[ne, d, 3 * d, 0], ne * d));
            s.push(model::block::ln_stats_fwd(&self.gpu, &LN_IDS, &self.enc_res[l], &self.enc_ln_mean, &self.enc_ln_inv, d, ne, 1e-5));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.enc_d_branch, &self.enc_res[l], &self.enc_ln_mean, &self.enc_ln_inv, g(&p("ln1.weight"))], &[d, ne], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.enc_d_branch, g(&p("ln1.bias"))], &[d, ne], d));
            s.push(model::block::layernorm_dx_bwd(&self.gpu, &LN_IDS, &self.enc_res[l], self.w(&p("ln1.weight")), &self.enc_d_branch, &self.enc_d_tmp, d, ne, 1e-5));
            s.push(self.gpu.step(ADD2, &[&self.enc_d_acc, &self.enc_d_tmp, &self.enc_dres[l]], &[ne * d], ne * d));
        }

        // ---- embeddings backward ----
        // Decoder embeddings.
        s.push(self.gpu.step(POS_BWD, &[&self.dec_dres[0], g("dec_pos.weight")], &[b, td, d], td * d));
        s.push(self.gpu.step(EMB_BWD, &[&self.tgt, &self.dec_dres[0], g("tok.weight")], &[nd, d, v], v * d));
        // Encoder embeddings (tok.weight is SHARED: accumulate into the same grad).
        s.push(self.gpu.step(POS_BWD, &[&self.enc_dres[0], g("enc_pos.weight")], &[b, te, d], te * d));
        s.push(self.gpu.step(EMB_BWD, &[&self.src, &self.enc_dres[0], g("tok.weight")], &[ne, d, v], v * d));
        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }

    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }
}

// ---- the architecture-agnostic Model seam (ADR 0001 §2.2/§2.3) ----

impl model::ModelConfig for Seq2SeqConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        Seq2SeqConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        Seq2SeqConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        Seq2SeqConfig::from_json(v)
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.vocab = vocab;
        self.block_size = block_size;
        self.with_ff_default()
    }
}

impl model::Model for Seq2Seq {
    type Config = Seq2SeqConfig;

    fn new(cfg: Seq2SeqConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Seq2Seq::new(cfg, b, t, init)
    }

    fn init_weights(cfg: &Seq2SeqConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &Seq2SeqConfig {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Seq2Seq { src, tgt, labels } => Seq2Seq::set_batch(self, src, tgt, labels),
            _ => panic!("seq2seq::Seq2Seq only supports Batch::Seq2Seq"),
        }
    }

    fn forward(&self) -> f32 {
        Seq2Seq::forward(self)
    }
    fn backward(&self) {
        Seq2Seq::backward(self)
    }
    fn zero_grads(&self) {
        Seq2Seq::zero_grads(self)
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Seq2Seq::adamw_step(self, t, lr, wd, clip, extra_scale)
    }

    fn poll_wait(&self) {
        Seq2Seq::poll_wait(self)
    }

    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Seq2Seq::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Seq2Seq::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Seq2Seq::read_grad(self, name)
    }

    /// seq2seq is encoder-decoder: there is no single-stream `logits_all` (a bare
    /// token list has no source/target split). Returns `None`; sampling uses the
    /// dedicated seq2seq decode path instead.
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }

    fn save(&self, path: &str) {
        Seq2Seq::save(self, path)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}
