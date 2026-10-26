// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped-transformer execution - one shared
// stack run for a configurable number of passes with per-pass readouts - for
// its clients. If your team needs expertise in recurrent-depth inference or in
// training through repeated uses of the same weights then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Ouro's execution graph.
//!
//! One forward runs the shared layer stack `total_ut_steps` times over the
//! whole sequence. Every pass is a full causal transformer pass over the same
//! weights; pass `p > 0` starts from pass `p-1`'s **normed** output (the
//! reference reassigns `hidden_states = norm(hidden_states)` at the end of
//! every pass), and each pass ends with the final RMSNorm plus the exit gate.
//!
//! Inference selects one pass - by step, or by the cumulative exit mass
//! crossing `early_exit_threshold`. Training follows the released objective:
//! every pass's cross-entropy is weighted by that pass's exit probability and
//! the weighted rows are summed, so the one shared parameter set accumulates
//! the gradient of every use.
//!
//! Boundary residuals are kept per `(pass, layer)` - `T · L + 1` buffers - so
//! the parity taps can hand back any pass's residual stream and the backward
//! pass rebuilds a layer from its input.

use std::cell::{Cell, RefCell};

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::OuroConfig;

/// Cross-entropy ignore index (unsupervised positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

// Slots the decode engine composes its own graph from.
pub(crate) const EMBED_ID: usize = EMBED;
pub(crate) const MATMUL_ID: usize = MATMUL;
pub(crate) const ADD2_ID: usize = ADD2;
pub(crate) const BIAS_ADD_ID: usize = BIAS_ADD;
pub(crate) const ROPE_AT_ID: usize = ROPE_AT;
pub(crate) const FLASH_GQA_ID: usize = FLASH_GQA_FWD;
pub(crate) const MATMUL_REG2_ID: usize = MATMUL_REG2;
pub(crate) const MATMUL_GEMV_ID: usize = MATMUL_GEMV;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const RMSNORM_DX: usize = 4;
const RMSNORM_DW: usize = 5;
const ROPE: usize = 6;
const ROPE_BWD: usize = 7;
const GQA_SCORES: usize = 8;
const ATTN_SOFTMAX: usize = 9;
const GQA_APPLY: usize = 10;
const GQA_DSCORES: usize = 11;
const GQA_DV: usize = 12;
const GQA_DQ: usize = 13;
const GQA_DK: usize = 14;
const SILU_MUL: usize = 15;
const SILU_DA: usize = 16;
const SILU_DB: usize = 17;
const ADD2: usize = 18;
const BIAS_ADD: usize = 19;
const CE_VALUE: usize = 20;
const CE_GRAD: usize = 21;
const MATMUL_DX: usize = 22;
const MATMUL_DW: usize = 23;
const EMB_BWD: usize = 24;
const ADAMW: usize = 25;
const GRADNORM_SQ: usize = 26;
const GRAD_SCALE: usize = 27;
const CLIP_COEF: usize = 28;
const GRAD_SCALE_BUF: usize = 29;
const SCALE_ROW: usize = 30;
const AXPY: usize = 31;
const BIAS_GRAD: usize = 32;
const KV_APPEND: usize = 33;
const ATTN_DECODE_SCORES: usize = 34;
const DECODE_SOFTMAX: usize = 35;
const ATTN_DECODE_APPLY: usize = 36;
const ROPE_AT: usize = 37;
const PAGED_KV_APPEND_BATCHED: usize = 38;
const PAGED_DECODE_SCORES_BATCHED: usize = 39;
const DECODE_SOFTMAX_BATCHED: usize = 40;
const PAGED_DECODE_APPLY_BATCHED: usize = 41;
const FLASH_GQA_FWD: usize = 42;
const MATMUL_REG2: usize = 43;
const MATMUL_GEMV: usize = 44;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
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
    ("bias_add", kernels::BIAS_ADD),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("scale_row", kernels::SCALE_ROW),
    ("axpy", kernels::AXPY),
    ("bias_grad", kernels::BIAS_GRAD),
    ("kv_append", kernels::KV_APPEND),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("rope_at", kernels::ROPE_AT),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("matmul_gemv", kernels::MATMUL_GEMV),
];

pub(crate) fn decode_batched_ids() -> block::GqaDecodeBatchedIds {
    block::GqaDecodeBatchedIds {
        kv_append_batched: PAGED_KV_APPEND_BATCHED,
        scores_batched: PAGED_DECODE_SCORES_BATCHED,
        softmax_batched: DECODE_SOFTMAX_BATCHED,
        apply_batched: PAGED_DECODE_APPLY_BATCHED,
    }
}

pub(crate) fn decode_ids() -> block::GqaDecodeIds {
    block::GqaDecodeIds {
        kv_append: KV_APPEND,
        attn_decode_scores: ATTN_DECODE_SCORES,
        decode_softmax: DECODE_SOFTMAX,
        attn_decode_apply: ATTN_DECODE_APPLY,
    }
}

pub(crate) fn kernel_ids() -> KernelIds {
    ids()
}

fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dw: RMSNORM_DW,
        rope: ROPE,
        rope_bwd: ROPE_BWD,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: GQA_DSCORES,
        gqa_dv: GQA_DV,
        gqa_dq: GQA_DQ,
        gqa_dk: GQA_DK,
        silu_mul: SILU_MUL,
        silu_da: SILU_DA,
        silu_db: SILU_DB,
        rmsnorm_rows: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
    }
}

/// Per-layer forward scratch. Reused by every layer and every pass: a layer
/// consumes its input boundary residual and produces the next one, so internal
/// activations are dead as soon as the next layer starts. The backward pass
/// rebuilds each layer from its boundary residual before differentiating it.
struct Scratch {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xn1_2: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    mlp_h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn2_2: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

/// Backward scratch (training builds only).
struct Grad {
    /// `T * L + 1` boundary gradients, `[n, d]` each.
    dres: Vec<DeviceBuffer>,
    /// `T` pass-output gradients, `[n, d]` each; the head writes it first and
    /// the next pass's first layer accumulates into it.
    dhs: Vec<DeviceBuffer>,
    /// `T` per-row exit probabilities used to weight each pass's CE gradient.
    exit_scale: Vec<DeviceBuffer>,
    /// `T` per-row gate-logit gradients (the last pass's gate is unused by the
    /// released objective, so its entry stays zero).
    gate_grad: Vec<DeviceBuffer>,
    dlogits: DeviceBuffer,
    dlogits_scaled: DeviceBuffer,
    rms_inv: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_xmid: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_tmp2: DeviceBuffer,
    d_mlp_out: DeviceBuffer,
    d_mlp_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_xn2: DeviceBuffer,
    d_proj: DeviceBuffer,
    d_xn1: DeviceBuffer,
    d_in: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
}

pub struct Ouro {
    pub gpu: Gpu,
    pub cfg: OuroConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    train: bool,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    /// `T * L + 1` boundary residuals, `[n, d]` each.
    res: Vec<DeviceBuffer>,
    /// `T` pass outputs: the final-normed hidden state each pass ends on.
    hs: Vec<DeviceBuffer>,
    /// `T` exit-gate logits, `[n]` each (pre-sigmoid).
    gates: Vec<DeviceBuffer>,
    /// Logit buffers: `T` in training (every pass's loss is needed at once),
    /// one reused buffer in inference (only the selected pass is read).
    logits: Vec<DeviceBuffer>,
    /// Per-pass CE rows (training only).
    ce_rows: Vec<DeviceBuffer>,

    scratch: Scratch,
    grad: Option<Grad>,
    /// Trunk: embed once, then every pass of the shared stack plus its gate.
    trunk: Vec<Step>,
    /// One LM-head matmul per pass.
    heads: Vec<Vec<Step>>,
    /// Training forward: trunk + every pass's head + its CE rows.
    fwd_steps: Vec<Step>,
    /// Training backward: weighted head gradients, then the stack in reverse.
    bwd_steps: Vec<Step>,
    /// Runtime CE-gradient uniform (`[rows, vocab, ignore, count]`).
    ce_grad_uniform: DeviceBuffer,
    opt: Option<optim::Optim>,
    /// Last pass selection made by the inference forward, per row.
    exit: Cell<Vec<u32>>,
    /// Per-pass gate-logit gradients computed by the training forward and
    /// written to the device by `backward`.
    dgate: RefCell<Vec<Vec<f32>>>,
}

impl Ouro {
    /// Frozen-weight model over a checkpoint source (a brain checkpoint, an HF
    /// directory source, or an in-memory init map).
    pub fn new_inference(cfg: OuroConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Ouro {
        Ouro::new_impl(cfg, b, t, src, false)
    }

    /// Trainable model: weights, gradients and AdamW moments for every
    /// parameter, plus the backward graph.
    pub fn new_train(cfg: OuroConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource) -> Ouro {
        Ouro::new_impl(cfg, b, t, init, true)
    }

    fn new_impl(cfg: OuroConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> Ouro {
        cfg.validate().expect("ouro config");
        assert!(t <= cfg.block_size, "sequence {t} exceeds positional capacity {}", cfg.block_size);
        assert!(b > 0 && t > 0);

        let gpu = Gpu::new(&optim::with_moment_kernels(PIPELINES));
        let role = if train { paramstore::Role::Trainable } else { paramstore::Role::Frozen };
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let opt = train.then(|| optim::Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF));

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer("tokens", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let targets = gpu.buffer("targets", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let boundaries = (cfg.total_ut_steps * cfg.n_layers + 1) as usize;
        let res = (0..boundaries).map(|_| st(n * d)).collect::<Vec<_>>();
        let hs = (0..cfg.total_ut_steps).map(|_| st(n * d)).collect::<Vec<_>>();
        let gates = (0..cfg.total_ut_steps).map(|_| st(n)).collect::<Vec<_>>();
        let logits = if train {
            (0..cfg.total_ut_steps).map(|_| st(n * v)).collect()
        } else {
            vec![st(n * v)]
        };
        let ce_rows = if train {
            (0..cfg.total_ut_steps).map(|_| st(n)).collect()
        } else {
            Vec::new()
        };

        let bht2 = (b as u64) * cfg.n_heads as u64 * (t as u64) * (t as u64);
        assert!(bht2 <= u32::MAX as u64, "materialized attention exceeds u32 index space");
        let scratch = Scratch {
            xn1: st(n * d),
            q: st(n * hq),
            k: st(n * hkv),
            v: st(n * hkv),
            ctx: st(n * hq),
            proj: st(n * d),
            xn1_2: st(n * d),
            xmid: st(n * d),
            xn2: st(n * d),
            gate_pre: st(n * ff),
            up: st(n * ff),
            mlp_h: st(n * ff),
            mlp_out: st(n * d),
            xn2_2: st(n * d),
            scores: st(bht2),
            probs: st(bht2),
        };

        let grad = train.then(|| Grad {
            dres: (0..boundaries).map(|_| st(n * d)).collect(),
            dhs: (0..cfg.total_ut_steps).map(|_| st(n * d)).collect(),
            exit_scale: (0..cfg.total_ut_steps).map(|_| st(n)).collect(),
            gate_grad: (0..cfg.total_ut_steps).map(|_| st(n)).collect(),
            dlogits: st(n * v),
            dlogits_scaled: st(n * v),
            rms_inv: st(n),
            d_scores: st(bht2),
            d_xmid: st(n * d),
            d_tmp: st(n * d),
            d_tmp2: st(n * d),
            d_mlp_out: st(n * d),
            d_mlp_h: st(n * ff),
            d_gate_pre: st(n * ff),
            d_up: st(n * ff),
            d_xn2: st(n * d),
            d_proj: st(n * d),
            d_xn1: st(n * d),
            d_in: st(n * d),
            d_ctx: st(n * hq),
            d_q: st(n * hq),
            d_k: st(n * hkv),
            d_v: st(n * hkv),
        });

        let ce_grad_uniform = gpu.uniform_dynamic(4);
        let mut m = Ouro {
            gpu,
            cfg: cfg.clone(),
            ps,
            b,
            t,
            train,
            count: Cell::new(1.0),
            tokens,
            targets,
            res,
            hs,
            gates,
            logits,
            ce_rows,
            scratch,
            grad,
            trunk: Vec::new(),
            heads: Vec::new(),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uniform,
            opt,
            exit: Cell::new(vec![0; b as usize]),
            dgate: RefCell::new(Vec::new()),
        };
        m.trunk = m.trunk_steps();
        m.heads = (0..cfg.total_ut_steps).map(|p| m.head_steps(p as usize)).collect();
        if train {
            m.fwd_steps = m.training_forward_steps();
            m.bwd_steps = m.training_backward_steps();
        }
        m
    }

    pub fn batch(&self) -> u32 {
        self.b
    }
    pub fn seq(&self) -> u32 {
        self.t
    }
    pub fn steps(&self) -> u32 {
        self.cfg.total_ut_steps
    }
    pub fn is_training(&self) -> bool {
        self.train
    }
    fn rows(&self) -> u32 {
        self.b * self.t
    }
    fn k(&self) -> KernelIds {
        ids()
    }

    /// Upload token ids for the next forward.
    pub fn set_tokens(&self, tokens: &[u32]) {
        assert_eq!(tokens.len(), self.rows() as usize, "tokens must be b*t");
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
    }

    /// Upload a causal-LM batch (`targets` predict the next token; `IGNORE`
    /// masks a row).
    pub fn set_tokens_targets(&self, tokens: &[u32], targets: &[u32]) {
        assert_eq!(tokens.len(), self.rows() as usize, "tokens must be b*t");
        assert_eq!(targets.len(), self.rows() as usize, "targets must be b*t");
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
        self.gpu.write(&self.targets, bytemuck::cast_slice(targets));
        let count = targets.iter().filter(|v| **v != IGNORE).count();
        self.count.set(count.max(1) as f32);
    }

    /// One layer of the shared stack: sandwich RMSNorm, MHA with half-split
    /// RoPE, SwiGLU, two residual adds. Writes `h_out` and fills the reusable
    /// scratch the backward pass will differentiate.
    fn layer_steps_into(&self, steps: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, h_out: &DeviceBuffer, kv: Option<(&DeviceBuffer, &DeviceBuffer)>) {
        let cfg = &self.cfg;
        let ids = self.k();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let s = &self.scratch;
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        steps.push(block::rmsnorm_fwd(&self.gpu, &ids, h_in, self.ps.w(&format!("blocks.{l}.ln1.weight")), &s.xn1, d, n, cfg.norm_eps));
        steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wq.weight")), &s.q], &[n, d, hq], n * hq));
        steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wk.weight")), &s.k], &[n, d, hkv], n * hkv));
        steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wv.weight")), &s.v], &[n, d, hkv], n * hkv));
        steps.push(block::rope_fwd(&self.gpu, &ids, &s.q, n, nh, hd, hq, self.t, cfg.rope_theta));
        steps.push(block::rope_fwd(&self.gpu, &ids, &s.k, n, nkv, hd, hkv, self.t, cfg.rope_theta));
        if let Some((kc, vc)) = kv {
            steps.push(block::kv_cache_fill(&self.gpu, KV_APPEND, &s.k, kc, n, nkv, hd));
            steps.push(block::kv_cache_fill(&self.gpu, KV_APPEND, &s.v, vc, n, nkv, hd));
        }
        steps.extend(block::gqa_fwd(&self.gpu, &ids, &ga, &s.q, &s.k, &s.v, &s.scores, &s.probs, &s.ctx));
        steps.push(self.gpu.step(MATMUL, &[&s.ctx, self.ps.w(&format!("blocks.{l}.attn.wo.weight")), &s.proj], &[n, hq, d], n * d));
        // Sandwich normalization: the attention output is normalized before the
        // residual add, and so is the MLP output.
        steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.proj, self.ps.w(&format!("blocks.{l}.ln1_2.weight")), &s.xn1_2, d, n, cfg.norm_eps));
        steps.push(self.gpu.step(ADD2, &[h_in, &s.xn1_2, &s.xmid], &[n * d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.xmid, self.ps.w(&format!("blocks.{l}.ln2.weight")), &s.xn2, d, n, cfg.norm_eps));
        steps.push(self.gpu.step(MATMUL, &[&s.xn2, self.ps.w(&format!("blocks.{l}.mlp.gate.weight")), &s.gate_pre], &[n, d, ff], n * ff));
        steps.push(self.gpu.step(MATMUL, &[&s.xn2, self.ps.w(&format!("blocks.{l}.mlp.up.weight")), &s.up], &[n, d, ff], n * ff));
        steps.push(block::swiglu_fwd(&self.gpu, &ids, &s.gate_pre, &s.up, &s.mlp_h, n * ff));
        steps.push(self.gpu.step(MATMUL, &[&s.mlp_h, self.ps.w(&format!("blocks.{l}.mlp.down.weight")), &s.mlp_out], &[n, ff, d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.mlp_out, self.ps.w(&format!("blocks.{l}.ln2_2.weight")), &s.xn2_2, d, n, cfg.norm_eps));
        steps.push(self.gpu.step(ADD2, &[&s.xmid, &s.xn2_2, h_out], &[n * d], n * d));
    }

    /// The trunk: embed once, then `T` passes of the shared stack, each ending
    /// in the final norm and the exit gate. The embedding is applied exactly
    /// once - pass `p` consumes the previous pass's state, never `embed(x)`.
    fn trunk_steps(&self) -> Vec<Step> {
        let cfg = &self.cfg;
        let ids = self.k();
        let n = self.rows();
        let d = cfg.d_model;
        let t = cfg.total_ut_steps;

        let mut steps = vec![self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &self.res[0]], &[d, n], n * d)];
        for p in 0..t as usize {
            for l in 0..cfg.n_layers as usize {
                let i = p * cfg.n_layers as usize + l;
                // The reference reassigns `hidden_states = norm(hidden_states)`
                // at the end of every pass, so pass p > 0 consumes the previous
                // pass's NORMED output, not its pre-norm residual.
                let h_in: &DeviceBuffer = if l == 0 && p > 0 { &self.hs[p - 1] } else { &self.res[i] };
                self.layer_steps_into(&mut steps, l, h_in, &self.res[i + 1], None);
            }
            let last = &self.res[(p + 1) * cfg.n_layers as usize];
            steps.push(block::rmsnorm_fwd(&self.gpu, &ids, last, self.ps.w("norm.weight"), &self.hs[p], d, n, cfg.norm_eps));
            // Exit gate: one linear over the model width, plus its bias.
            steps.push(self.gpu.step(MATMUL, &[&self.hs[p], self.ps.w("gate.weight"), &self.gates[p]], &[n, d, 1], n));
            steps.push(self.gpu.step(BIAS_ADD, &[&self.gates[p], self.ps.w("gate.bias")], &[n, 1], n));
        }
        steps
    }

    fn head_steps(&self, pass: usize) -> Vec<Step> {
        let n = self.rows();
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let out = &self.logits[if self.train { pass } else { 0 }];
        vec![self.gpu.step(MATMUL, &[&self.hs[pass], self.ps.w("lm_head.weight"), out], &[n, d, v], n * v)]
    }

    fn training_forward_steps(&self) -> Vec<Step> {
        let n = self.rows();
        let v = self.cfg.vocab;
        let mut steps = self.trunk.clone();
        for p in 0..self.cfg.total_ut_steps as usize {
            steps.extend(self.heads[p].iter().cloned());
            steps.push(self.gpu.step(CE_VALUE, &[&self.logits[p], &self.targets, &self.ce_rows[p]], &[n, v, IGNORE], n));
        }
        steps
    }

    /// The backward graph. Pass 1 of every layer's gradient is the update of
    /// the one shared parameter set; the pass loop is the "same weight, K
    /// uses" accumulation, and the kernels that write a parameter gradient
    /// (`matmul_dw`, `rmsnorm_dw`, `emb_bwd`, `matmul_dx` with `acc=1`) add
    /// into the buffer the trainer zeroed once per step.
    fn training_backward_steps(&self) -> Vec<Step> {
        let cfg = &self.cfg;
        let ids = self.k();
        let n = self.rows();
        let d = cfg.d_model;
        let v = cfg.vocab;
        let t = cfg.total_ut_steps as usize;
        let l_count = cfg.n_layers as usize;
        let g = self.grad.as_ref().expect("training graph on an inference build");
        let mut steps = Vec::new();

        // Head phase, every pass. The CE gradient is divided by the supervised
        // count and then scaled per row by that row's exit probability for this
        // pass, so the shared head and trunk see the weighted sum.
        for p in 0..t {
            steps.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uniform, &[&self.logits[p], &self.targets, &g.dlogits], n * v));
            steps.push(self.gpu.step(SCALE_ROW, &[&g.dlogits, &g.exit_scale[p], &g.dlogits_scaled], &[n * v, v], n * v));
            steps.push(self.gpu.step(MATMUL_DX, &[&g.dlogits_scaled, self.ps.w("lm_head.weight"), &g.dhs[p]], &[n, d, v, 0], n * d));
            steps.push(self.gpu.step(MATMUL_DW, &[&g.dlogits_scaled, &self.hs[p], self.ps.g("lm_head.weight")], &[n, d, v], v * d));
        }

        // Exit-gate phase. The loss depends on every pass's gate through its
        // exit probability, so the gate's input gradient lands on that pass's
        // hidden state (accumulated onto the head's assignment above). The
        // final pass's gate never enters the released objective: its
        // probability is the remaining mass, so its gradient stays zero.
        for p in 0..t.saturating_sub(1) {
            steps.push(self.gpu.step(MATMUL_DX, &[&g.gate_grad[p], self.ps.w("gate.weight"), &g.dhs[p]], &[n, d, 1, 1], n * d));
            steps.push(self.gpu.step(MATMUL_DW, &[&g.gate_grad[p], &self.hs[p], self.ps.g("gate.weight")], &[n, d, 1], d));
            steps.push(self.gpu.step(BIAS_GRAD, &[&g.gate_grad[p], self.ps.g("gate.bias")], &[n, 1], 1));
        }

        // Trunk phase, passes in reverse. Every boundary gradient has exactly
        // one producer except `dhs[p]`, which the head wrote and the next
        // pass's first layer accumulates into.
        for p in (0..t).rev() {
            let last = &self.res[(p + 1) * l_count];
            steps.extend(block::rmsnorm_bwd(&self.gpu, &ids, last, self.ps.w("norm.weight"), &g.dhs[p], &g.dres[(p + 1) * l_count], &g.rms_inv, Some(self.ps.g("norm.weight")), d, n, cfg.norm_eps));
            for l in (0..l_count).rev() {
                let i = p * l_count + l;
                let h_in: &DeviceBuffer = if l == 0 && p > 0 { &self.hs[p - 1] } else { &self.res[i] };
                // Rebuild the layer's internals from its boundary residual.
                self.layer_steps_into(&mut steps, l, h_in, &self.res[i + 1], None);
                self.layer_bwd_into(&mut steps, l, h_in, &g.dres[i + 1], if l == 0 && p > 0 { &g.dhs[p - 1] } else { &g.dres[i] }, l == 0 && p > 0);
            }
        }

        steps.push(self.gpu.step(EMB_BWD, &[&self.tokens, &g.dres[0], self.ps.g("tok.weight")], &[n, d, v], v * d));
        steps
    }

    /// One layer's backward. `d_h_out` is the gradient of the layer's output;
    /// `dest` receives the input gradient, accumulated with [`AXPY`] when the
    /// input is a pass output that the head already wrote (`acc`).
    #[allow(clippy::too_many_arguments)]
    fn layer_bwd_into(&self, steps: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, d_h_out: &DeviceBuffer, dest: &DeviceBuffer, acc: bool) {
        let cfg = &self.cfg;
        let ids = self.k();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let s = &self.scratch;
        let g = self.grad.as_ref().expect("training graph on an inference build");
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        let p = |name: &str| format!("blocks.{l}.{name}");

        // h_out = xmid + xn2_2
        steps.extend(block::rmsnorm_bwd(&self.gpu, &ids, &s.mlp_out, self.ps.w(&p("ln2_2.weight")), d_h_out, &g.d_mlp_out, &g.rms_inv, Some(self.ps.g(&p("ln2_2.weight"))), d, n, cfg.norm_eps));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_mlp_out, &s.mlp_h, self.ps.g(&p("mlp.down.weight"))], &[n, ff, d], d * ff));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_mlp_out, self.ps.w(&p("mlp.down.weight")), &g.d_mlp_h], &[n, ff, d, 0], n * ff));
        steps.extend(block::swiglu_bwd(&self.gpu, &ids, &s.gate_pre, &s.up, &g.d_mlp_h, &g.d_gate_pre, &g.d_up, n * ff));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_gate_pre, &s.xn2, self.ps.g(&p("mlp.gate.weight"))], &[n, d, ff], ff * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_gate_pre, self.ps.w(&p("mlp.gate.weight")), &g.d_xn2], &[n, d, ff, 0], n * d));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_up, &s.xn2, self.ps.g(&p("mlp.up.weight"))], &[n, d, ff], ff * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_up, self.ps.w(&p("mlp.up.weight")), &g.d_xn2], &[n, d, ff, 1], n * d));
        // xmid = h_in + xn1_2
        steps.extend(block::rmsnorm_bwd(&self.gpu, &ids, &s.xmid, self.ps.w(&p("ln2.weight")), &g.d_xn2, &g.d_tmp, &g.rms_inv, Some(self.ps.g(&p("ln2.weight"))), d, n, cfg.norm_eps));
        steps.push(self.gpu.step(ADD2, &[d_h_out, &g.d_tmp, &g.d_xmid], &[n * d], n * d));
        // Attention output (normalized before the residual add).
        steps.extend(block::rmsnorm_bwd(&self.gpu, &ids, &s.proj, self.ps.w(&p("ln1_2.weight")), &g.d_xmid, &g.d_proj, &g.rms_inv, Some(self.ps.g(&p("ln1_2.weight"))), d, n, cfg.norm_eps));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_proj, &s.ctx, self.ps.g(&p("attn.wo.weight"))], &[n, hq, d], d * hq));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_proj, self.ps.w(&p("attn.wo.weight")), &g.d_ctx], &[n, hq, d, 0], n * hq));
        steps.extend(block::gqa_bwd(&self.gpu, &ids, &ga, &s.q, &s.k, &s.v, &s.probs, &g.d_ctx, &g.d_scores, &g.d_q, &g.d_k, &g.d_v));
        steps.push(block::rope_bwd(&self.gpu, &ids, &g.d_q, n, nh, hd, hq, self.t, cfg.rope_theta));
        steps.push(block::rope_bwd(&self.gpu, &ids, &g.d_k, n, nkv, hd, hkv, self.t, cfg.rope_theta));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_q, &s.xn1, self.ps.g(&p("attn.wq.weight"))], &[n, d, hq], hq * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_q, self.ps.w(&p("attn.wq.weight")), &g.d_xn1], &[n, d, hq, 0], n * d));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_k, &s.xn1, self.ps.g(&p("attn.wk.weight"))], &[n, d, hkv], hkv * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_k, self.ps.w(&p("attn.wk.weight")), &g.d_xn1], &[n, d, hkv, 1], n * d));
        steps.push(self.gpu.step(MATMUL_DW, &[&g.d_v, &s.xn1, self.ps.g(&p("attn.wv.weight"))], &[n, d, hkv], hkv * d));
        steps.push(self.gpu.step(MATMUL_DX, &[&g.d_v, self.ps.w(&p("attn.wv.weight")), &g.d_xn1], &[n, d, hkv, 1], n * d));
        steps.extend(block::rmsnorm_bwd(&self.gpu, &ids, h_in, self.ps.w(&p("ln1.weight")), &g.d_xn1, &g.d_tmp2, &g.rms_inv, Some(self.ps.g(&p("ln1.weight"))), d, n, cfg.norm_eps));
        if acc {
            steps.push(self.gpu.step(ADD2, &[&g.d_xmid, &g.d_tmp2, &g.d_in], &[n * d], n * d));
            steps.push(self.gpu.step(AXPY, &[dest, &g.d_in], &[n * d, gpu_core::f(1.0)], n * d));
        } else {
            steps.push(self.gpu.step(ADD2, &[&g.d_xmid, &g.d_tmp2, dest], &[n * d], n * d));
        }
    }

    /// Run the trunk only (all passes and gates).
    pub fn trunk_forward(&self) {
        self.gpu.submit(&[], &self.trunk);
    }

    /// Run the trunk, then the LM head for pass `pass`.
    pub fn forward_pass(&self, pass: usize) {
        self.trunk_forward();
        self.head_forward(pass);
    }

    /// Run the LM head for `pass` against the already-computed pass output.
    pub fn head_forward(&self, pass: usize) {
        assert!(pass < self.cfg.total_ut_steps as usize, "pass {pass} out of range");
        self.gpu.submit(&[], &self.heads[pass]);
        self.exit.set(vec![pass as u32; self.b as usize]);
    }

    /// Per-row exit **probabilities** for every pass: `p_t = λ_t·Π_{j<t}(1-λ_j)`
    /// and the final pass takes the remaining mass, exactly the reference's
    /// normalized distribution.
    pub fn exit_probabilities(&self) -> Vec<Vec<f32>> {
        let n = self.rows() as usize;
        let t = self.cfg.total_ut_steps as usize;
        let mut out = vec![vec![0.0f32; n]; t];
        let mut remaining = vec![1.0f64; n];
        for (pass, out_pass) in out.iter_mut().enumerate() {
            let gate = self.gpu.read(&self.gates[pass], n);
            for r in 0..n {
                let lambda = 1.0 / (1.0 + (-(gate[r] as f64)).exp());
                if pass + 1 < t {
                    out_pass[r] = (lambda * remaining[r]) as f32;
                    remaining[r] *= 1.0 - lambda;
                } else {
                    out_pass[r] = remaining[r] as f32;
                }
            }
        }
        out
    }

    /// The per-row exit pass implied by the gates and `early_exit_threshold`
    /// (the reference's CDF rule): the first pass whose cumulative exit mass
    /// reaches the threshold, else the final pass. Requires the trunk to have
    /// run.
    pub fn exit_steps(&self) -> Vec<u32> {
        let n = self.rows() as usize;
        let t = self.cfg.total_ut_steps as usize;
        let mut out = vec![t as u32 - 1; n];
        let mut remaining = vec![1.0f64; n];
        let mut cdf = vec![0.0f64; n];
        for pass in 0..t {
            let gate = self.gpu.read(&self.gates[pass], n);
            for r in 0..n {
                let lambda = 1.0 / (1.0 + (-(gate[r] as f64)).exp());
                let p = if pass + 1 < t { lambda * remaining[r] } else { remaining[r] };
                remaining[r] *= 1.0 - lambda;
                cdf[r] += p;
                if cdf[r] >= self.cfg.early_exit_threshold as f64 {
                    out[r] = pass as u32;
                }
            }
        }
        out
    }

    /// Inference forward: trunk, exit selection, head for the selected pass.
    /// The selection must be uniform across rows for this entry point (batch
    /// 1, or equal prompts); a heterogeneous batch selects per row in the
    /// serving engine.
    pub fn forward_logits(&self) -> Vec<f32> {
        self.trunk_forward();
        let steps = self.exit_steps();
        let step = steps[0];
        assert!(steps.iter().all(|s| *s == step), "heterogeneous exit steps {steps:?} need the serving path");
        self.gpu.submit(&[], &self.heads[step as usize]);
        self.exit.set(steps);
        self.read_logits()
    }

    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits[0], self.rows() as usize * self.cfg.vocab as usize)
    }

    /// That pass's logits (training mode keeps one buffer per pass).
    pub fn read_logits_pass(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.logits[pass], self.rows() as usize * self.cfg.vocab as usize)
    }

    /// `[n, d]` final-normed hidden state of pass `pass`.
    pub fn read_hidden(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.hs[pass], self.rows() as usize * self.cfg.d_model as usize)
    }

    /// `[n]` exit-gate logits of pass `pass` (pre-sigmoid).
    pub fn read_gate(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.gates[pass], self.rows() as usize)
    }

    /// `[n, d]` residual stream entering layer `layer` of pass `pass` (layer
    /// `n_layers` is that pass's pre-norm output).
    pub fn read_res(&self, pass: usize, layer: usize) -> Vec<f32> {
        let i = pass * self.cfg.n_layers as usize + layer;
        self.gpu.read(&self.res[i], self.rows() as usize * self.cfg.d_model as usize)
    }
}

impl model::Model for Ouro {
    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: &self.gpu, store: &mut self.ps, optim: self.opt.as_ref()? })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: &self.gpu, store: &self.ps, optim: self.opt.as_ref()? })
    }

    type Config = OuroConfig;

    fn new(cfg: Self::Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>) -> Self {
        Ouro::new_train(cfg, b, t, init)
    }

    fn init_weights(cfg: &Self::Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => self.set_tokens_targets(tokens, targets),
            other => panic!("ouro: unsupported batch shape {:?}", std::mem::discriminant(&other)),
        }
    }

    /// The released training objective: every pass's CE weighted by that pass's
    /// exit probability, summed over passes, averaged over supervised rows.
    fn forward(&self) -> f32 {
        assert!(self.train, "forward on an inference-only build has no objective; read pass logits instead");
        self.gpu.submit(&[], &self.fwd_steps);
        let probs = self.exit_probabilities();
        for (p, row) in probs.iter().enumerate() {
            self.gpu.write_f32(&self.grad.as_ref().unwrap().exit_scale[p], row);
        }
        let n = self.rows() as usize;
        let t = self.cfg.total_ut_steps as usize;
        let count = self.count.get() as f64;
        let mut ce_rows = Vec::with_capacity(t);
        let mut loss = 0.0f64;
        for (p, row) in probs.iter().enumerate() {
            let ce = self.gpu.read(&self.ce_rows[p], n);
            for r in 0..n {
                loss += (row[r] as f64) * (ce[r] as f64);
            }
            ce_rows.push(ce);
        }
        // dL/dλ_t for every pass but the last, then through the sigmoid. The
        // products are evaluated directly rather than by dividing out
        // (1-λ_t), which is the wrong answer when a gate saturates.
        let mut lambda = vec![vec![0.0f64; n]; t];
        for (p, lambda_p) in lambda.iter_mut().enumerate() {
            let gate = self.gpu.read(&self.gates[p], n);
            for (l, g) in lambda_p.iter_mut().zip(&gate) {
                *l = 1.0 / (1.0 + (-(*g as f64)).exp());
            }
        }
        let mut dgate = vec![vec![0.0f32; n]; t];
        for p in 0..t.saturating_sub(1) {
            for r in 0..n {
                let mut d = 0.0f64;
                for j in p..t {
                    let mut prod = 1.0f64;
                    for (k, lambda_k) in lambda.iter().enumerate().take(j) {
                        if k != p {
                            prod *= 1.0 - lambda_k[r];
                        }
                    }
                    // p_j carries its own λ_j only for j != p; the derivative
                    // w.r.t. λ_p is the product of the other survival factors.
                    let coef = if j == p { 1.0 } else if j + 1 < t { lambda[j][r] } else { 1.0 };
                    let term = ce_rows[j][r] as f64 * prod * coef;
                    d += if j == p { term } else { -term };
                }
                dgate[p][r] = (d / count * lambda[p][r] * (1.0 - lambda[p][r])) as f32;
            }
        }
        *self.dgate.borrow_mut() = dgate;
        (loss / count) as f32
    }

    fn backward(&self) {
        assert!(self.train, "backward on an inference-only build");
        let (n, v) = (self.rows(), self.cfg.vocab);
        self.gpu.write(
            &self.ce_grad_uniform,
            &[n, v, IGNORE, gpu_core::f(self.count.get())],
        );
        let dgate = self.dgate.borrow();
        assert_eq!(dgate.len(), self.cfg.total_ut_steps as usize, "backward needs the training forward's gate gradients");
        let grad = self.grad.as_ref().expect("training graph on an inference build");
        for (p, row) in dgate.iter().enumerate() {
            self.gpu.write_f32(&grad.gate_grad[p], row);
        }
        self.gpu.submit(&[], &self.bwd_steps);
    }

    fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        self.opt.as_ref().expect("adamw_step on an inference-only build").step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale);
    }

    fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    fn param_names(&self) -> Vec<String> {
        self.cfg.param_list().into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.ps.w(name), data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.set_tokens(tokens);
        self.trunk_forward();
        let last = self.cfg.total_ut_steps as usize - 1;
        self.gpu.submit(&[], &self.heads[last]);
        Some(self.gpu.read(&self.logits[if self.train { last } else { 0 }], self.rows() as usize * self.cfg.vocab as usize))
    }

    fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        let card = checkpoint::st::ModelCard::new("brain/ouro", "ouro");
        checkpoint::save_carded(path, self.cfg.to_json(), &tensors, &card);
    }
}

impl model::ModelConfig for OuroConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        OuroConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        OuroConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        OuroConfig::from_json(v).expect("ouro config")
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
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hand-numbered slots index `PIPELINES` positionally; inserting a
    /// kernel without updating them would dispatch the wrong WGSL silently on
    /// the CPU backend. Pin every slot the graphs use.
    #[test]
    fn kernel_slots_match_the_pipeline_list() {
        for (slot, want) in [
            (EMBED, "embed"),
            (MATMUL, "matmul"),
            (RMSNORM, "rmsnorm"),
            (ROPE, "rope_base"),
            (GQA_SCORES, "gqa_scores"),
            (ATTN_SOFTMAX, "attn_softmax"),
            (GQA_APPLY, "gqa_apply"),
            (SILU_MUL, "silu_mul"),
            (ADD2, "add2"),
            (BIAS_ADD, "bias_add"),
            (CE_VALUE, "ce_value"),
            (CE_GRAD, "ce_grad"),
            (MATMUL_DX, "matmul_dx"),
            (MATMUL_DW, "matmul_dw"),
            (EMB_BWD, "emb_bwd"),
            (ADAMW, "adamw"),
            (GRADNORM_SQ, "gradnorm_sq"),
            (GRAD_SCALE, "grad_scale"),
            (CLIP_COEF, "clip_coef"),
            (GRAD_SCALE_BUF, "grad_scale_buf"),
            (SCALE_ROW, "scale_row"),
            (AXPY, "axpy"),
            (BIAS_GRAD, "bias_grad"),
            (ROPE_AT, "rope_at"),
            (PAGED_KV_APPEND_BATCHED, "paged_kv_append_batched"),
            (PAGED_DECODE_SCORES_BATCHED, "paged_decode_scores_batched"),
            (DECODE_SOFTMAX_BATCHED, "decode_softmax_batched"),
            (PAGED_DECODE_APPLY_BATCHED, "paged_decode_apply_batched"),
            (FLASH_GQA_FWD, "flash_attn_causal_gqa"),
        ] {
            assert_eq!(PIPELINES[slot].0, want, "slot {slot} is not {want}");
        }
    }

    #[test]
    fn every_registered_kernel_has_a_cost_formula() {
        for (name, _) in PIPELINES {
            assert!(gpu_core::cost::covers(name), "kernel {name} has no cost formula");
        }
    }
}
