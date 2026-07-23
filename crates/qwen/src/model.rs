// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 dense decoder Transformer — forward + backprop as WGSL compute
//! dispatches, sharing the engine with the GPT/MoE/PID models (`gpu_core`,
//! `paramstore`, `optim`, `kernels`).
//!
//! Per pre-norm block (no biases anywhere):
//!   h  = RMSNorm(x)·ln1
//!   q,k,v = h·Wq, h·Wk, h·Wv         (separate, GQA: Wk/Wv narrower)
//!   q  = RoPE(QKNorm(q)·q_norm) ;  k = RoPE(QKNorm(k)·k_norm)
//!   x += Wo · GQA-attention(q,k,v)
//!   h  = RMSNorm(x)·ln2
//!   x += Wdown · ( SiLU(Wgate·h) ⊙ (Wup·h) )
//!   logits = tok.weightᵀ · RMSNorm(x)·norm    (tied head)
//!   loss   = masked cross-entropy (ignore_index = IGNORE)
//!
//! RoPE uses Qwen's half-split convention + base 1e6 (`rope_base.wgsl`), QK-norm
//! reuses `rmsnorm` over `head_dim`, and the tied head accumulates both the
//! lm_head and embedding gradients into `tok.weight` (matmul_dw then emb_bwd).

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use optim::Optim;
use paramstore::ParamStore;

use crate::config::QwenConfig;

/// Cross-entropy ignore index (masked target positions); the loader's `-1 i32`
/// reinterpreted as `u32`.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
/// Plain (untiled) embedding gather — kept in PIPELINES at index 0 for stable
/// indexing; the forward uses the vocab-tiled `EMBED_TILE` instead.
#[allow(dead_code)]
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
const CE_VALUE: usize = 19;
#[allow(dead_code)]
const CE_GRAD: usize = 20;
const MATMUL_DX: usize = 21;
const MATMUL_DW: usize = 22;
const EMB_BWD: usize = 23;
const ADAMW: usize = 24;
const GRADNORM_SQ: usize = 25;
const GRAD_SCALE: usize = 26;
const CLIP_COEF: usize = 27;
const GRAD_SCALE_BUF: usize = 28;
const AXPY: usize = 29;
const EMBED_TILE: usize = 30;
const MATMUL_TILE: usize = 31;
const MATMUL_REG2: usize = 32;
const MATMUL_DX_REG: usize = 33;
const MATMUL_DW_REG: usize = 34;
const CE_STATS: usize = 35;
const CE_GRAD_STATS: usize = 36;

const PIPELINES: &[(&str, &str)] = &[
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
    ("axpy", kernels::AXPY),
    ("embed_tile", kernels::EMBED_TILE),
    ("matmul_tile", kernels::MATMUL_TILE),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
];

/// Pick the GEMM kernel + dispatch thread count for a forward linear
/// `[m,k]·[n,k]ᵀ`. The software-pipelined `matmul_reg2` (128×128 tile, 256
/// threads, ~4 TFLOP/s on a P40) wins once both output dims fill a tile; below
/// that the naive one-thread-per-output `matmul` is better. Same math either way
/// (parity gated by `tests/backend_parity` + gradcheck), so this only changes
/// speed. `BRAIN_QWEN_NAIVE_MM=1` forces the naive kernel.
fn linear_kernel(m: usize, n: usize) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    if naive || m < 128 || n < 128 {
        (MATMUL, (m * n) as u32)
    } else {
        (MATMUL_REG2, (m.div_ceil(128) * n.div_ceil(128) * 256) as u32)
    }
}

/// Backward GEMM pickers: tiled `matmul_{dx,dw}_reg` (bit-identical to naive,
/// ~34% of P40 peak) once both output dims fill a 128-tile, else naive. Small
/// LoRA-rank matmuls fall back automatically. `BRAIN_QWEN_NAIVE_MM=1` forces naive.
/// Full fine-tuning with AdamW moments offloaded to system RAM (`BRAIN_OFFLOAD_ADAM=1`).
fn offload_adam() -> bool {
    std::env::var("BRAIN_OFFLOAD_ADAM").map(|v| v != "0").unwrap_or(false)
}

fn dx_kernel_bw(m: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    if naive || m < 128 || k < 128 { (MATMUL_DX, m * k) }
    else { (MATMUL_DX_REG, m.div_ceil(128) * k.div_ceil(128) * 256) }
}
fn dw_kernel_bw(nrows: u32, k: u32) -> (usize, u32) {
    let naive = std::env::var("BRAIN_QWEN_NAIVE_MM").map(|v| v != "0").unwrap_or(false);
    if naive || nrows < 128 || k < 128 { (MATMUL_DW, nrows * k) }
    else { (MATMUL_DW_REG, nrows.div_ceil(128) * k.div_ceil(128) * 256) }
}


/// Per-binding budget (f32 words) for tiling the embedding / lm_head over vocab,
/// so each storage binding stays under a backend's `max_storage_buffer_binding_
/// size` (e.g. 128MB on Mesa-GL). ~96 MiB; small models collapse to one tile.
/// `BRAIN_TILE_BUDGET_WORDS` overrides it (e.g. tiny, to force tiling in tests).
const TILE_BUDGET_WORDS: u64 = 24 * 1024 * 1024;

fn tile_budget_words() -> u64 {
    std::env::var("BRAIN_TILE_BUDGET_WORDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(TILE_BUDGET_WORDS)
}

struct Layer {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

pub struct Qwen {
    pub gpu: Gpu,
    pub cfg: QwenConfig,
    pub ps: ParamStore,
    opt: Optim,
    /// Host AdamW for offloaded params (RAM-resident moments); None otherwise.
    offload_opt: std::cell::RefCell<Option<optim::OffloadAdam>>,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
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
    ce_stats: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    dq_pre: DeviceBuffer,
    dk_pre: DeviceBuffer,
    d_v: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    inv: DeviceBuffer,

    // LoRA scratch (sized for rank `r`; trivially small when LoRA is off).
    lora_a: DeviceBuffer,   // [n*r] : a = x @ A^T
    lora_da: DeviceBuffer,  // [n*r] : grad wrt a
    lora_out: DeviceBuffer, // [n*max_out] : delta = a @ B^T

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: DeviceBuffer,
}

impl Qwen {
    /// Load a trainable model (weights + grad + AdamW moments) from a checkpoint.
    pub fn load(path: &str, b: u32, t: u32) -> Qwen {
        let c = checkpoint::load(path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Qwen::new(cfg, b, t, &init)
    }

    /// Load an **inference-only** model: parameters are frozen (weights only, no
    /// grad/AdamW buffers), cutting device memory ~4× — essential for loading a
    /// real 0.6B checkpoint for generation. Builds only the forward graph.
    pub fn load_inference(path: &str, b: u32, t: u32) -> Qwen {
        let c = checkpoint::load(path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Qwen::new_impl(cfg, b, t, &init, false)
    }

    pub fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen {
        Qwen::new_impl(cfg, b, t, init, true)
    }

    fn new_impl(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, train: bool) -> Qwen {
        let gpu = Gpu::new(PIPELINES);
        // Role assignment:
        //  - inference (`!train`): every parameter Frozen (weights only).
        //  - LoRA training: only `*.lora_a`/`*.lora_b` trainable; base Frozen.
        //  - full training: every parameter Trainable.
        let ps = if !train {
            let roles = cfg
                .param_list()
                .into_iter()
                .map(|(n, c)| (n, c, paramstore::Role::Frozen))
                .collect();
            ParamStore::new_with_roles(&gpu, roles, init)
        } else if cfg.lora.is_some() {
            let roles = cfg
                .param_list()
                .into_iter()
                .map(|(n, c)| {
                    let role = if n.ends_with(".lora_a") || n.ends_with(".lora_b") {
                        paramstore::Role::Trainable
                    } else {
                        paramstore::Role::Frozen
                    };
                    (n, c, role)
                })
                .collect();
            ParamStore::new_with_roles(&gpu, roles, init)
        } else if offload_adam() {
            // Full fine-tuning with the AdamW moments in system RAM (Role::Offload):
            // GPU holds only weight+grad (2×model) instead of 4×model.
            let roles = cfg
                .param_list()
                .into_iter()
                .map(|(n, c)| (n, c, paramstore::Role::Offload))
                .collect();
            ParamStore::new_with_roles(&gpu, roles, init)
        } else {
            ParamStore::new(&gpu, cfg.param_list(), init)
        };
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);
        // Lazily-built host optimiser for the offloaded params (None unless any
        // parameter took Role::Offload).
        let offload_opt: std::cell::RefCell<Option<optim::OffloadAdam>> = std::cell::RefCell::new(None);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let bht2 = (b * cfg.n_heads * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer(
            "tokens",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let targets = gpu.buffer(
            "targets",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let ce_grad_uni = gpu.uniform_dynamic(4); // [n, vocab, IGNORE, count]

        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                xn1: st(n * d),
                q_pre: st(n * hq),
                q: st(n * hq),
                k_pre: st(n * hkv),
                k: st(n * hkv),
                v: st(n * hkv),
                probs: st(bht2),
                ctx: st(n * hq),
                xmid: st(n * d),
                xn2: st(n * d),
                gate_pre: st(n * ff),
                up: st(n * ff),
                h: st(n * ff),
            });
        }
        // `inv` must hold the per-row RMS for the largest norm: QK-norm-q has
        // n*n_heads rows (>= n*n_kv and >= n).
        let inv_rows = n * cfg.n_heads as u64;
        // LoRA scratch (rank r; max projection output across all sites).
        let r = cfg.lora.as_ref().map(|l| l.rank as u64).unwrap_or(0).max(1);
        let max_out = hq.max(ff).max(d).max(hkv);

        let mut m = Qwen {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            offload_opt,
            opt,
            tokens,
            targets,
            res,
            layers,
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(bht2),
            xn_final: st(n * d),
            logits: st(n * v),
            ce_buf: st(n),
            dres,
            d_logits: st(n * v),
            ce_stats: st(n * 2),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_ctx: st(n * hq),
            d_scores: st(bht2),
            d_q: st(n * hq),
            d_k: st(n * hkv),
            dq_pre: st(n * hq),
            dk_pre: st(n * hkv),
            d_v: st(n * hkv),
            d_h: st(n * ff),
            d_gate_pre: st(n * ff),
            d_up: st(n * ff),
            inv: st(inv_rows),
            lora_a: st(n * r),
            lora_da: st(n * r),
            lora_out: st(n * max_out),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            gpu,
        };
        m.fwd_steps = m.forward_steps(m.b, m.t);
        m.bwd_steps = if train { m.build_backward_steps() } else { Vec::new() };
        m
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    /// True if `name` has a gradient buffer (i.e. is optimised). Frozen
    /// parameters (LoRA base, inference) have none, so their weight-gradient
    /// dispatches must be skipped — only the input-gradient (dX) path runs to
    /// keep backprop flowing to lower-layer adapters.
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// Kernel-index map for the shared `model::block` Step-builders.
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
        }
    }

    /// GQA shape for `b`×`t` (the buffers are sized for the max `b`/`t`).
    fn gqa(&self, b: u32, t: u32) -> Gqa {
        Gqa { b, t, n_heads: self.cfg.n_heads, n_kv_heads: self.cfg.n_kv_heads, head_dim: self.cfg.head_dim }
    }

    /// RMSNorm backward via the shared builder: input grad always, gain grad only
    /// when the gain is trainable (frozen LoRA base / inference skip it).
    fn rmsnorm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let gw = self.trainable(wname).then(|| self.g(wname));
        s.extend(block::rmsnorm_bwd(&self.gpu, &Self::ids(), x, self.w(wname), dy, dx, &self.inv, gw, dim, rows));
    }

    /// True if a LoRA adapter is configured for the given projection leaf.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg
            .lora
            .as_ref()
            .filter(|lc| lc.targets_leaf(leaf))
            .map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)·(x·Aᵀ)·Bᵀ`.
    /// No-op for an untargeted leaf. `m`×`k` is the input, `nout` the output.
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(self.gpu.step(MATMUL, &[&self.lora_a, self.w(&bnm), &self.lora_out], &[m, r, nout], m * nout));
        s.push(self.gpu.step(AXPY, &[y, &self.lora_out], &[m * nout, f(scale)], m * nout));
    }

    /// Backward for a (possibly-LoRA) linear `y = x·Wᵀ`. Accumulates the input
    /// gradient into `dx` (flag `acc`). For a full weight: base dW + dX. For a
    /// LoRA-targeted leaf: the base weight is frozen (dX only, no dW) and the
    /// adapter grads gA/gB are produced (scale folded in by scaling `d_out`).
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, s: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        match self.lora_for(leaf) {
            Some((r, scale)) => {
                // base: dx += d_out·W (frozen weight — no dW). d_out is NOT mutated
                // here: for `wo` it is `dxmid`, reused downstream as the residual
                // grad, so the adapter scale is folded into the private scratch.
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[d_out, self.w(wname), dx], &[m, k, nout, acc], bt));
                let a = format!("{wname}.lora_a");
                let bnm = format!("{wname}.lora_b");
                // a = (alpha/r)·(x·Aᵀ)  -> gB += d_outᵀ·a
                s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_a], &[m * r, f(scale)], m * r));
                let (bk, bt) = dw_kernel_bw(nout, r);
                s.push(self.gpu.step(bk, &[d_out, &self.lora_a, self.g(&bnm)], &[m, r, nout], bt));
                // da = (alpha/r)·(d_out·B) -> gA += daᵀ·x ; dx += da·A
                let (bk, bt) = dx_kernel_bw(m, r);
                s.push(self.gpu.step(bk, &[d_out, self.w(&bnm), &self.lora_da], &[m, r, nout, 0], bt));
                s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_da], &[m * r, f(scale)], m * r));
                let (bk, bt) = dw_kernel_bw(r, k);
                s.push(self.gpu.step(bk, &[&self.lora_da, x, self.g(&a)], &[m, k, r], bt));
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[&self.lora_da, self.w(&a), dx], &[m, k, r, 1], bt));
            }
            None => {
                if self.trainable(wname) {
                    let (bk, bt) = dw_kernel_bw(nout, k);
                    s.push(self.gpu.step(bk, &[d_out, x, self.g(wname)], &[m, k, nout], bt));
                }
                let (bk, bt) = dx_kernel_bw(m, k);
                s.push(self.gpu.step(bk, &[d_out, self.w(wname), dx], &[m, k, nout, acc], bt));
            }
        }
    }

    /// Vocab tiles `(v0, count)` sized so a `[count, d_model]` weight slice stays
    /// within the per-binding budget. Small vocabularies yield a single tile.
    fn vocab_tiles(&self) -> Vec<(u32, u32)> {
        let d = self.cfg.d_model as u64;
        let v = self.cfg.vocab as u64;
        let rows = (tile_budget_words() / d.max(1)).max(1);
        let mut out = Vec::new();
        let mut v0 = 0u64;
        while v0 < v {
            let cnt = rows.min(v - v0);
            out.push((v0 as u32, cnt as u32));
            v0 += cnt;
        }
        out
    }

    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let ids = Self::ids();
        let ga = self.gqa(b_use, t_use);
        let theta = c.rope_theta;
        let mut s: Vec<Step> = Vec::new();
        let dw = d as u64;
        let tiles = self.vocab_tiles();

        // Token embedding, tiled over vocab so each `tok.weight` binding stays
        // under the backend's max-binding size (GL: 128MB).
        for &(v0, cnt) in &tiles {
            s.push(self.gpu.step_sliced(
                EMBED_TILE,
                &[&self.tokens, self.w("tok.weight"), &self.res[0]],
                &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                &[d, n, v0, cnt],
                n * d,
            ));
        }

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // --- attention --- (projections stay here: they carry LoRA/bias;
            // norms/RoPE/attention-core come from the shared block builders)
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &self.res[l], self.w(&p("ln1.weight")), &lb.xn1, d, n));
            let (mk, mt) = linear_kernel(n as usize, hq as usize);
            s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wq.weight")), &lb.q_pre], &[n, d, hq], mt));
            self.lora_fwd(&mut s, "wq", &lb.xn1, &p("attn.wq.weight"), &lb.q_pre, n, d, hq);
            let (mk, mt) = linear_kernel(n as usize, hkv as usize);
            s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wk.weight")), &lb.k_pre], &[n, d, hkv], mt));
            self.lora_fwd(&mut s, "wk", &lb.xn1, &p("attn.wk.weight"), &lb.k_pre, n, d, hkv);
            let (mk, mt) = linear_kernel(n as usize, hkv as usize);
            s.push(self.gpu.step(mk, &[&lb.xn1, self.w(&p("attn.wv.weight")), &lb.v], &[n, d, hkv], mt));
            self.lora_fwd(&mut s, "wv", &lb.xn1, &p("attn.wv.weight"), &lb.v, n, d, hkv);
            // QK-norm over head_dim then half-split RoPE on q/k.
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &lb.q_pre, self.w(&p("attn.q_norm.weight")), &lb.q, hd, n * nh));
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &lb.k_pre, self.w(&p("attn.k_norm.weight")), &lb.k, hd, n * nkv));
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.q, n, nh, hd, hq, t_use, theta));
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.k, n, nkv, hd, hkv, t_use, theta));
            s.extend(block::gqa_fwd(&self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &self.scores, &lb.probs, &lb.ctx));
            let (mk, mt) = linear_kernel(n as usize, d as usize);
            s.push(self.gpu.step(mk, &[&lb.ctx, self.w(&p("attn.wo.weight")), &self.proj], &[n, hq, d], mt));
            self.lora_fwd(&mut s, "wo", &lb.ctx, &p("attn.wo.weight"), &self.proj, n, hq, d);
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // --- SwiGLU MLP ---
            s.push(block::rmsnorm_fwd(&self.gpu, &ids, &lb.xmid, self.w(&p("ln2.weight")), &lb.xn2, d, n));
            let (mk, mt) = linear_kernel(n as usize, ff as usize);
            s.push(self.gpu.step(mk, &[&lb.xn2, self.w(&p("mlp.gate.weight")), &lb.gate_pre], &[n, d, ff], mt));
            self.lora_fwd(&mut s, "gate", &lb.xn2, &p("mlp.gate.weight"), &lb.gate_pre, n, d, ff);
            let (mk, mt) = linear_kernel(n as usize, ff as usize);
            s.push(self.gpu.step(mk, &[&lb.xn2, self.w(&p("mlp.up.weight")), &lb.up], &[n, d, ff], mt));
            self.lora_fwd(&mut s, "up", &lb.xn2, &p("mlp.up.weight"), &lb.up, n, d, ff);
            s.push(block::swiglu_fwd(&self.gpu, &ids, &lb.gate_pre, &lb.up, &lb.h, n * ff));
            let (mk, mt) = linear_kernel(n as usize, d as usize);
            s.push(self.gpu.step(mk, &[&lb.h, self.w(&p("mlp.down.weight")), &self.mlp_out], &[n, ff, d], mt));
            self.lora_fwd(&mut s, "down", &lb.h, &p("mlp.down.weight"), &self.mlp_out, n, ff, d);
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
        }

        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(&self.gpu, &ids, &self.res[last], self.w("norm.weight"), &self.xn_final, d, n));
        // lm_head. When the whole vocab fits one tile (v0=0, cnt=v — the common
        // case for a small vocab like the TTS Talker's 3072), it is a plain
        // `[n,d]·[v,d]ᵀ` matmul, so dispatch the size-adaptive fast kernel
        // (`matmul_reg2`) instead of the naive column-tiled `matmul_tile` — the
        // Talker lm_head was ~50 ms (naive) vs ~2 ms (reg2). Only when the weight
        // genuinely exceeds a binding budget do we fall back to the tiled path.
        let head = c.head_weight();
        if tiles.len() == 1 && tiles[0] == (0, v) {
            let (mk, mt) = linear_kernel(n as usize, v as usize);
            s.push(self.gpu.step(mk, &[&self.xn_final, self.w(head), &self.logits], &[n, d, v], mt));
        } else {
            for &(v0, cnt) in &tiles {
                s.push(self.gpu.step_sliced(
                    MATMUL_TILE,
                    &[&self.xn_final, self.w(head), &self.logits],
                    &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                    &[n, d, v, v0, cnt],
                    n * cnt,
                ));
            }
        }
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
        s
    }

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        let n = (self.b * self.t) as usize;
        let losses = self.gpu.read(&self.ce_buf, n);
        losses.iter().sum::<f32>() / self.count.get()
    }

    pub fn backward(&self) {
        let n = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let theta = c.rope_theta;
        let head = c.head_weight();
        let b = self.b;
        let t = self.t;
        let ids = Self::ids();
        let ga = self.gqa(b, t);
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final norm ----
        // Two-pass CE gradient: compute per-row softmax stats ONCE (ce_stats),
        // then the per-element gradient reads them — O(rows*vocab) instead of the
        // naive per-element softmax recompute's O(rows*vocab^2). At vocab 151936
        // this is the difference between ~10 ms and ~56 s per backward.
        s.push(self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &self.ce_stats], &[n, v, IGNORE], n));
        s.push(self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.ce_stats, &self.d_logits], n * v));
        if self.trainable(head) {
            let (bk, bt) = dw_kernel_bw(v, d);
            s.push(self.gpu.step(bk, &[&self.d_logits, &self.xn_final, self.g(head)], &[n, d, v], bt));
        }
        let (bk, bt) = dx_kernel_bw(n, d);
        s.push(self.gpu.step(bk, &[&self.d_logits, self.w(head), &self.d_xn], &[n, d, v, 0], bt));
        let last = c.n_layers as usize;
        self.rmsnorm_bwd(&mut s, &self.res[last], "norm.weight", &self.d_xn, &self.dres[last], d, n);

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- SwiGLU MLP backward (input grad = dres[l+1]) ----
            self.proj_bwd(&mut s, "down", &self.dres[l + 1], &lb.h, &p("mlp.down.weight"), &self.d_h, n, ff, d, 0);
            s.extend(block::swiglu_bwd(&self.gpu, &ids, &lb.gate_pre, &lb.up, &self.d_h, &self.d_gate_pre, &self.d_up, n * ff));
            self.proj_bwd(&mut s, "up", &self.d_up, &lb.xn2, &p("mlp.up.weight"), &self.d_xn, n, d, ff, 0);
            self.proj_bwd(&mut s, "gate", &self.d_gate_pre, &lb.xn2, &p("mlp.gate.weight"), &self.d_xn, n, d, ff, 1);
            self.rmsnorm_bwd(&mut s, &lb.xmid, &p("ln2.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // ---- attention backward (input grad = dxmid) ----
            self.proj_bwd(&mut s, "wo", &self.dxmid, &lb.ctx, &p("attn.wo.weight"), &self.d_ctx, n, hq, d, 0);
            s.extend(block::gqa_bwd(
                &self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &lb.probs, &self.d_ctx, &self.d_scores, &self.d_q, &self.d_k, &self.d_v,
            ));
            // RoPE backward (in place on d_q/d_k -> grad wrt normed q/k)
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_q, n, nh, hd, hq, t, theta));
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_k, n, nkv, hd, hkv, t, theta));
            // QK-norm backward: grad wrt q_pre/k_pre -> dq_pre/dk_pre
            self.rmsnorm_bwd(&mut s, &lb.q_pre, &p("attn.q_norm.weight"), &self.d_q, &self.dq_pre, hd, n * nh);
            self.rmsnorm_bwd(&mut s, &lb.k_pre, &p("attn.k_norm.weight"), &self.d_k, &self.dk_pre, hd, n * nkv);
            // q/k/v projection backward -> accumulate into d_xn (= grad wrt xn1)
            self.proj_bwd(&mut s, "wv", &self.d_v, &lb.xn1, &p("attn.wv.weight"), &self.d_xn, n, d, hkv, 0);
            self.proj_bwd(&mut s, "wk", &self.dk_pre, &lb.xn1, &p("attn.wk.weight"), &self.d_xn, n, d, hkv, 1);
            self.proj_bwd(&mut s, "wq", &self.dq_pre, &lb.xn1, &p("attn.wq.weight"), &self.d_xn, n, d, hq, 1);
            // ln1 backward -> d_tmp ; dres[l] = dxmid + d_tmp
            self.rmsnorm_bwd(&mut s, &self.res[l], &p("ln1.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embedding backward (tied: accumulates onto the head grad in tok.weight)
        if self.trainable("tok.weight") {
            s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("tok.weight")], &[n, d, v], v * d));
        }
        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // GPU optimiser for `Trainable` params.
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
        // Host (RAM-resident) optimiser for `Offload` params — built lazily on the
        // first step from the store's current weights.
        if !self.ps.offload.is_empty() {
            let mut slot = self.offload_opt.borrow_mut();
            if slot.is_none() {
                *slot = Some(optim::OffloadAdam::new(&self.gpu, &self.ps));
            }
            slot.as_mut().unwrap().step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
        }
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

    /// The maximum sequence length this instance was sized for (the `t` it was
    /// built/loaded with) — generation must keep its context within this.
    pub fn ctx_len(&self) -> usize {
        self.t as usize
    }

    /// Logits for every position of a single sequence (B must be 1, t>=len).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "qwen decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab) as usize)
    }

    pub fn save(&self, path: &str) {
        self.save_with_itos(path, None);
    }

    pub fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        let mut config = self.cfg.to_json();
        if let Some(itos) = itos {
            let arr: Vec<Value> = itos.iter().map(|c| Value::from(c.to_string())).collect();
            config["itos"] = Value::Array(arr);
        }
        checkpoint::save(path, config, &tensors);
    }
}

// ---- architecture-agnostic Model seam ----

impl model::ModelConfig for QwenConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        QwenConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        QwenConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        QwenConfig::from_json(v)
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
        self.with_defaults()
    }
}

impl model::Model for Qwen {
    type Config = QwenConfig;

    fn new(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Qwen::new(cfg, b, t, init)
    }
    fn init_weights(cfg: &QwenConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &QwenConfig {
        &self.cfg
    }
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Qwen::set_batch(self, tokens, targets),
            _ => panic!("qwen::Qwen only supports Batch::Lm"),
        }
    }
    fn forward(&self) -> f32 {
        Qwen::forward(self)
    }
    fn backward(&self) {
        Qwen::backward(self)
    }
    fn zero_grads(&self) {
        Qwen::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Qwen::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Qwen::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        // The *trainable* set (full: all params; LoRA: adapters only). Used by
        // gradcheck and the optimiser-facing surface — frozen params have no grad.
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Qwen::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Qwen::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Qwen::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Qwen::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        Qwen::save(self, path)
    }
    fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        Qwen::save_with_itos(self, path, itos)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn param_list_shapes_and_tied_head() {
        let cfg = QwenConfig::tiny(); // v23 d16 L2 nh4 nkv2 hd8 ff32 tied
        let m: HashMap<_, _> = cfg.param_list().into_iter().collect();
        assert_eq!(m["tok.weight"], 23 * 16);
        assert_eq!(m["blocks.0.attn.wq.weight"], (4 * 8) * 16); // Hq x d
        assert_eq!(m["blocks.0.attn.wk.weight"], (2 * 8) * 16); // Hkv x d
        assert_eq!(m["blocks.0.attn.q_norm.weight"], 8); // head_dim
        assert_eq!(m["blocks.1.attn.wo.weight"], 16 * (4 * 8)); // d x Hq
        assert_eq!(m["blocks.0.mlp.gate.weight"], 32 * 16);
        assert_eq!(m["blocks.0.mlp.down.weight"], 16 * 32);
        // tied head: no separate lm_head tensor.
        assert!(!m.contains_key("lm_head.weight"));
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = QwenConfig::qwen3_0_6b();
        let back = QwenConfig::from_json(&cfg.to_json());
        assert_eq!(back.vocab, 151936);
        assert_eq!(back.n_kv_heads, 8);
        assert_eq!(back.head_dim, 128);
        assert_eq!(back.group(), 2);
        assert!((back.rope_theta - 1.0e6).abs() < 1.0);
        assert!(back.tie_embeddings);
    }

    #[test]
    fn forward_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let model = Qwen::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 3 % 23) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 3 + 1) % 23) as u32).collect();
        model.set_batch(&x, &y);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss {l1}");
        assert!((l1 - l2).abs() < 1e-6, "not deterministic");
        assert!(l1 < 2.0 * (23f32).ln(), "loss implausibly large: {l1}");
    }

    #[test]
    fn one_overfit_run_reduces_loss() {
        if gpu_disabled() {
            return;
        }
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 11);
        let model = Qwen::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
        model.set_batch(&x, &y);
        let before = model.forward();
        for step in 1..=50 {
            model.zero_grads();
            model.forward();
            model.backward();
            model.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
            model.poll_wait();
        }
        let after = model.forward();
        assert!(after < before, "overfit did not reduce loss: {before} -> {after}");
    }
}
