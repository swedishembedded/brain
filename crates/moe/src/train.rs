// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full training pipeline (forward + backprop + AdamW) as a WGSL compute
//! pipeline. Mirrors `tiny_sparse_moe.py`'s training step exactly, so the Rust
//! executable can train the model from scratch on the GPU.
//!
//! Entry point:
//!   * `train(args)`    — generate the toy corpus, init weights, and run the
//!     optimisation loop, then save weights in the inference engine's format.
//!     Numerical correctness is gated by the finite-difference `brain-gradcheck`.
//!
//! Design notes:
//!   * fp32 only; <=5 storage buffers per kernel; single bind group. We request
//!     `max_storage_buffers_per_shader_stage = 8` (well within Pascal/sm_61).
//!   * The forward pass is written SSA-style: every stage writes a fresh buffer
//!     which doubles as its activation cache, so backprop has everything it
//!     needs and no mid-pass copies are required.
//!   * MoE uses the dense top-k formulation (all experts evaluated, masked by
//!     the renormalised gate). With capacity dropping disabled this is exactly
//!     what the Python reference computes.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::IGNORE;
use optim::Optim;
use paramstore::ParamStore;

// ---- kernel indices (order matches `PIPELINES`) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE: usize = 3;
const ATTN_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const ATTN_APPLY: usize = 6;
const ROUTER: usize = 7;
const SILU: usize = 8;
const SCALE_ADD: usize = 9;
const ADD2: usize = 10;
const CE_GRAD: usize = 11;
const CE_VALUE: usize = 12;
const MATMUL_DX: usize = 13;
const MATMUL_DW: usize = 14;
const RMS_INV: usize = 15;
const RMSNORM_DX: usize = 16;
const RMSNORM_DW: usize = 17;
const SILU_DA: usize = 18;
const SILU_DB: usize = 19;
const SCALE_ADD_DEXP: usize = 20;
const SCALE_ADD_DGATE: usize = 21;
const EXPERT_COUNTS: usize = 22;
const ROUTER_BWD: usize = 23;
const ROPE_BWD: usize = 24;
const ATTN_BWD_DSCORES: usize = 25;
const ATTN_BWD_DV: usize = 26;
const ATTN_BWD_DQ: usize = 27;
const ATTN_BWD_DK: usize = 28;
const EMB_BWD: usize = 29;
const ADAMW: usize = 30;
const GRADNORM_SQ: usize = 31;
const GRAD_SCALE: usize = 32;
const CLIP_COEF: usize = 33;
const GRAD_SCALE_BUF: usize = 34;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_train", kernels::ROPE_TRAIN),
    ("attn_scores", kernels::ATTN_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("router_gate_train", kernels::ROUTER_GATE_TRAIN),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("add2", kernels::ADD2),
    // Masked cross-entropy (ignore_index = IGNORE), so masked-label datasets
    // (e.g. the benchmark battery's answer-masking recipe) work: an IGNORE
    // target must NOT index `logits[base + target]` (an unmasked CE would read
    // out of bounds for the 0xFFFF_FFFF sentinel). Mirrors GPT/PID. With no
    // masking (count == n) these are numerically identical to the unmasked CE,
    // so the gradcheck path is unchanged.
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
    ("expert_counts", kernels::EXPERT_COUNTS),
    ("router_bwd", kernels::ROUTER_BWD),
    ("rope_train_bwd", kernels::ROPE_TRAIN_BWD),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// MoE's fixed AdamW betas/eps (the established values matched against the
/// PyTorch reference). The unified trait [`model::Model::adamw_step`] does not
/// thread per-call betas, so the trait path uses these; the inherent
/// [`Trainer::adamw_step_betas`] still accepts explicit betas for callers that
/// need them (e.g. federated expert training).
const MOE_BETA1: f32 = 0.9;
const MOE_BETA2: f32 = 0.95;
const MOE_EPS: f32 = 1e-8;

#[derive(Clone)]
pub struct Config {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub d_ff: u32,
    pub aux_coef: f32,
    pub z_coef: f32,
}

impl Config {
    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
}

/// Names + element counts of the unique (trainable) parameters.
fn param_list(c: &Config) -> Vec<(String, usize)> {
    let d = c.d_model as usize;
    let ff = c.d_ff as usize;
    let mut v = vec![("token_emb.weight".to_string(), c.vocab as usize * d)];
    for l in 0..c.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        v.push((p("norm1.weight"), d));
        v.push((p("attn.qkv.weight"), 3 * d * d));
        v.push((p("attn.out.weight"), d * d));
        v.push((p("norm2.weight"), d));
        v.push((p("moe.router.weight"), c.n_experts as usize * d));
        for e in 0..c.n_experts {
            v.push((format!("blocks.{l}.moe.experts.{e}.w_gate.weight"), ff * d));
            v.push((format!("blocks.{l}.moe.experts.{e}.w_up.weight"), ff * d));
            v.push((format!("blocks.{l}.moe.experts.{e}.w_down.weight"), d * ff));
        }
    }
    v.push(("norm.weight".to_string(), d));
    v
}

struct LayerBufs {
    xn1: DeviceBuffer,
    qkv: DeviceBuffer,
    probs: DeviceBuffer,
    attn_out: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    router_logits: DeviceBuffer,
    router_probs: DeviceBuffer,
    gate: DeviceBuffer,
    gate_pre: Vec<DeviceBuffer>,
    up: Vec<DeviceBuffer>,
    h: Vec<DeviceBuffer>,
    expert_out: Vec<DeviceBuffer>,
    dxmid: DeviceBuffer,
}

pub struct Trainer {
    gpu: Gpu,
    cfg: Config,
    b: u32,
    t: u32,

    // Parameter storage + optimizer are shared with the GPT/PID models (ADR §7),
    // which gives MoE the unified clip + grad-accum AdamW path for free.
    ps: ParamStore,
    opt: Optim,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,

    /// Count of non-IGNORE target rows in the current batch (the masked-CE mean
    /// divisor + backward grad scale). Set by [`set_batch`](Self::set_batch).
    count: Cell<f32>,
    /// Dynamic uniform for the masked CE-grad step: `[n, vocab, IGNORE, count]`,
    /// rewritten per batch (the count varies with the masking) before backward.
    ce_grad_uni: DeviceBuffer,

    res: Vec<DeviceBuffer>,  // residual stream, len n_layers+1 (res[0]=emb out, res[L]=x_final)
    dres: Vec<DeviceBuffer>, // its gradient
    layers: Vec<LayerBufs>,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,

    // forward temporaries
    scores: DeviceBuffer,
    proj: DeviceBuffer,
    moe_acc: DeviceBuffer,
    ce_buf: DeviceBuffer,
    fe: DeviceBuffer,

    // backward temporaries
    d_logits: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_attn_out: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_router_logits: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_h: DeviceBuffer,
    d_expert_out: DeviceBuffer,
    inv: DeviceBuffer,

    // Cached forward/backward dispatch graphs. They are structurally identical
    // on every iteration, so we build them once and reuse them. This avoids
    // allocating a fresh `params` uniform buffer and a fresh bind group per
    // dispatch (~210/step), which otherwise exhausts the GPU memory aperture
    // after a few thousand steps and triggers a device reset. The bind groups
    // keep their referenced buffers alive; only the *contents* of
    // `tokens`/`targets` (set_batch) change in place via `write_buffer`. The
    // AdamW graph is cached inside the shared `Optim`.
    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
}

impl Trainer {
    pub fn new(cfg: Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Trainer {
        // One shared accelerator (wgpu or native CPU, chosen at runtime). All the
        // device-init + dispatch plumbing that used to live here now lives in
        // `gpu_core`, shared with the GPT and PID models.
        Trainer::new_on(Gpu::new(PIPELINES), cfg, b, t, init)
    }

    /// Build on an existing device handle — see `Gpt::new_on`.
    pub(crate) fn new_on(gpu: Gpu, cfg: Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Trainer {
        let c = cfg.clone();
        // Parameter weights/grads/Adam-moment buffers (all zero-initialised) live
        // in the shared ParamStore; the shared Optim drives the AdamW + clip path.
        let ps = ParamStore::new(&gpu, param_list(&c), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let e = c.n_experts as u64;
        let bht2 = (b * c.n_heads * t * t) as u64;

        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);
        let ce_grad_uni = gpu.uniform_dynamic(4); // [n, vocab, IGNORE, count]

        let st = |x: u64| gpu.storage(x);
        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=c.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..c.n_layers {
            layers.push(LayerBufs {
                xn1: st(n * d),
                qkv: st(n * 3 * d),
                probs: st(bht2),
                attn_out: st(n * d),
                xmid: st(n * d),
                xn2: st(n * d),
                router_logits: st(n * e),
                router_probs: st(n * e),
                gate: st(n * e),
                gate_pre: (0..e).map(|_| st(n * ff)).collect(),
                up: (0..e).map(|_| st(n * ff)).collect(),
                h: (0..e).map(|_| st(n * ff)).collect(),
                expert_out: (0..e).map(|_| st(n * d)).collect(),
                dxmid: st(n * d),
            });
        }

        let mut trainer = Trainer {
            cfg: c,
            b,
            t,
            ps,
            opt,
            tokens,
            targets,
            count: Cell::new(1.0),
            ce_grad_uni,
            res,
            dres,
            layers,
            xn_final: st(n * d),
            logits: st(n * cfg.vocab as u64),
            scores: st(bht2),
            proj: st(n * d),
            moe_acc: st(n * d),
            ce_buf: st(n),
            fe: st(e),
            d_logits: st(n * cfg.vocab as u64),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            d_qkv: st(n * 3 * d),
            d_attn_out: st(n * d),
            d_scores: st(bht2),
            d_gate: st(n * e),
            d_router_logits: st(n * e),
            d_gate_pre: st(n * ff),
            d_up: st(n * ff),
            d_h: st(n * ff),
            d_expert_out: st(n * d),
            inv: st(n),
            gpu,
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
        };
        trainer.build_graphs();
        trainer
    }

    /// Build the forward, backward, and AdamW dispatch graphs once. Called from
    /// `new` after all buffers exist; the cached step lists are then reused on
    /// every training iteration.
    fn build_graphs(&mut self) {
        self.fwd_steps = self.build_forward();
        self.bwd_steps = self.build_backward();
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, bytemuck::cast_slice(x));
        self.gpu.write(&self.targets, bytemuck::cast_slice(y));
        // Masked-CE divisor = number of non-IGNORE target rows (mean over scored
        // positions, matching F.cross_entropy(ignore_index=...)). `.max(1)` guards
        // an all-masked batch.
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    /// Run the (cached) forward pass and return mean cross-entropy. The
    /// `read` also polls the device, which is what lets wgpu reclaim the
    /// transient staging buffers this step allocated.
    pub fn forward(&self) -> f32 {
        let n = self.b * self.t;
        self.gpu.submit(&[], &self.fwd_steps);
        let losses = self.gpu.read(&self.ce_buf, n as usize);
        // Masked CE writes 0 for IGNORE rows; mean over the non-ignored count.
        losses.iter().sum::<f32>() / self.count.get()
    }

    /// Build the forward dispatch graph; caches all activations. Buffers and
    /// uniform contents are identical every step, so this is built once.
    fn build_forward(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;
        let hd = c.head_dim();
        let half = hd / 2;
        let mut s: Vec<Step> = Vec::new();

        // embedding -> res[0]
        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("token_emb.weight"), &self.res[0]], &[d, n], n * d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");
            // attention
            s.push(self.gpu.step(RMSNORM, &[&self.res[l], self.w(&pn("norm1.weight")), &lb.xn1], &[d, n], n));
            s.push(self.gpu.step(MATMUL, &[&lb.xn1, self.w(&pn("attn.qkv.weight")), &lb.qkv], &[n, d, 3 * d], n * 3 * d));
            s.push(self.gpu.step(ROPE, &[&lb.qkv], &[n, c.n_heads, hd, 3 * d, 0, self.t], n * c.n_heads * half));
            s.push(self.gpu.step(ROPE, &[&lb.qkv], &[n, c.n_heads, hd, 3 * d, d, self.t], n * c.n_heads * half));
            s.push(self.gpu.step(ATTN_SCORES, &[&lb.qkv, &self.scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * self.t));
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&self.scores, &lb.probs], &[self.b, c.n_heads, self.t], self.b * c.n_heads * self.t));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_out], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(MATMUL, &[&lb.attn_out, self.w(&pn("attn.out.weight")), &self.proj], &[n, d, d], n * d));
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // moe
            s.push(self.gpu.step(RMSNORM, &[&lb.xmid, self.w(&pn("norm2.weight")), &lb.xn2], &[d, n], n));
            s.push(self.gpu.step(MATMUL, &[&lb.xn2, self.w(&pn("moe.router.weight")), &lb.router_logits], &[n, d, e], n * e));
            s.push(self.gpu.step(ROUTER, &[&lb.router_logits, &lb.gate, &lb.router_probs], &[n, e, c.top_k], n));
            for ei in 0..e as usize {
                let ep = |name: &str| format!("blocks.{l}.moe.experts.{ei}.{name}");
                s.push(self.gpu.step(MATMUL, &[&lb.xn2, self.w(&ep("w_gate.weight")), &lb.gate_pre[ei]], &[n, d, ff], n * ff));
                s.push(self.gpu.step(MATMUL, &[&lb.xn2, self.w(&ep("w_up.weight")), &lb.up[ei]], &[n, d, ff], n * ff));
                s.push(self.gpu.step(SILU, &[&lb.gate_pre[ei], &lb.up[ei], &lb.h[ei]], &[n * ff], n * ff));
                s.push(self.gpu.step(MATMUL, &[&lb.h[ei], self.w(&ep("w_down.weight")), &lb.expert_out[ei]], &[n, ff, d], n * d));
                let acc = if ei == 0 { 0 } else { 1 };
                s.push(self.gpu.step(SCALE_ADD, &[&lb.gate, &lb.expert_out[ei], &self.moe_acc], &[n, d, e, ei as u32, acc], n * d));
            }
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.moe_acc, &self.res[l + 1]], &[n * d], n * d));
        }

        // final norm + tied lm_head
        s.push(self.gpu.step(RMSNORM, &[&self.res[c.n_layers as usize], self.w("norm.weight"), &self.xn_final], &[d, n], n));
        s.push(self.gpu.step(MATMUL, &[&self.xn_final, self.w("token_emb.weight"), &self.logits], &[n, d, c.vocab], n * c.vocab));
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, c.vocab, IGNORE], n));

        s
    }

    /// Run the (cached) backward pass: accumulate every parameter gradient
    /// (does NOT zero them — the caller zeroes once per effective batch via
    /// [`Self::zero_grads`], matching the GPT/PID grad-accum contract that the
    /// generic trainer relies on).
    pub fn backward(&self) {
        // Masked CE-grad: zero for IGNORE rows, normalised by the non-ignored
        // count (rewritten per batch since the masking varies).
        let n = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    /// Build the backward dispatch graph (constant across steps).
    fn build_backward(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;
        let hd = c.head_dim();
        let half = hd / 2;
        let n_layers = c.n_layers as usize;
        let mut s: Vec<Step> = Vec::new();

        // ---- output: cross-entropy grad, lm_head, final norm ----
        // Masked CE-grad reads its `[n, vocab, IGNORE, count]` from the dynamic
        // uniform (written per batch in `backward`), since `count` varies.
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * c.vocab));
        // lm_head dW (tied -> grad_emb) and dX -> d_xn (=d_xn_final)
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_logits, &self.xn_final, self.g("token_emb.weight")], &[n, d, c.vocab], c.vocab * d));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_logits, self.w("token_emb.weight"), &self.d_xn], &[n, d, c.vocab, 0], n * d));
        // final norm backward -> dres[L]
        s.push(self.gpu.step(RMS_INV, &[&self.res[n_layers], &self.inv], &[d, n], n));
        s.push(self.gpu.step(RMSNORM_DW, &[&self.d_xn, &self.res[n_layers], &self.inv, self.g("norm.weight")], &[d, n], d));
        s.push(self.gpu.step(RMSNORM_DX, &[&self.res[n_layers], self.w("norm.weight"), &self.d_xn, &self.dres[n_layers]], &[d, n], n));

        for l in (0..n_layers).rev() {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");

            // ===== MoE backward (d_moe_acc = dres[l+1]) =====
            // Phase A: per-expert gate gradient
            for ei in 0..e as usize {
                s.push(self.gpu.step(SCALE_ADD_DGATE, &[&lb.expert_out[ei], &self.dres[l + 1], &self.d_gate], &[n, d, e, ei as u32], n));
            }
            // Phase B: router gradient -> d_xn (init), grad_Wrouter
            s.push(self.gpu.step(EXPERT_COUNTS, &[&lb.gate, &self.fe], &[n, e, c.top_k], e));
            s.push(self.gpu.step(ROUTER_BWD, &[&lb.router_logits, &lb.gate, &self.d_gate, &self.fe, &self.d_router_logits], &[n, e, c.top_k, 0, f(c.aux_coef), f(c.z_coef)], n));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_router_logits, &lb.xn2, self.g(&pn("moe.router.weight"))], &[n, d, e], e * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_router_logits, self.w(&pn("moe.router.weight")), &self.d_xn], &[n, d, e, 0], n * d));
            // Phase C: per-expert SwiGLU backward, accumulate into d_xn
            for ei in 0..e as usize {
                let ep = |name: &str| format!("blocks.{l}.moe.experts.{ei}.{name}");
                s.push(self.gpu.step(SCALE_ADD_DEXP, &[&lb.gate, &self.dres[l + 1], &self.d_expert_out], &[n, d, e, ei as u32], n * d));
                s.push(self.gpu.step(MATMUL_DW, &[&self.d_expert_out, &lb.h[ei], self.g(&ep("w_down.weight"))], &[n, ff, d], d * ff));
                s.push(self.gpu.step(MATMUL_DX, &[&self.d_expert_out, self.w(&ep("w_down.weight")), &self.d_h], &[n, ff, d, 0], n * ff));
                s.push(self.gpu.step(SILU_DA, &[&lb.gate_pre[ei], &lb.up[ei], &self.d_h, &self.d_gate_pre], &[n * ff], n * ff));
                s.push(self.gpu.step(SILU_DB, &[&lb.gate_pre[ei], &self.d_h, &self.d_up], &[n * ff], n * ff));
                s.push(self.gpu.step(MATMUL_DW, &[&self.d_up, &lb.xn2, self.g(&ep("w_up.weight"))], &[n, d, ff], ff * d));
                s.push(self.gpu.step(MATMUL_DX, &[&self.d_up, self.w(&ep("w_up.weight")), &self.d_xn], &[n, d, ff, 1], n * d));
                s.push(self.gpu.step(MATMUL_DW, &[&self.d_gate_pre, &lb.xn2, self.g(&ep("w_gate.weight"))], &[n, d, ff], ff * d));
                s.push(self.gpu.step(MATMUL_DX, &[&self.d_gate_pre, self.w(&ep("w_gate.weight")), &self.d_xn], &[n, d, ff, 1], n * d));
            }
            // norm2 backward -> d_tmp ; dxmid = dres[l+1] + d_tmp
            s.push(self.gpu.step(RMS_INV, &[&lb.xmid, &self.inv], &[d, n], n));
            s.push(self.gpu.step(RMSNORM_DW, &[&self.d_xn, &lb.xmid, &self.inv, self.g(&pn("norm2.weight"))], &[d, n], d));
            s.push(self.gpu.step(RMSNORM_DX, &[&lb.xmid, self.w(&pn("norm2.weight")), &self.d_xn, &self.d_tmp], &[d, n], n));
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &lb.dxmid], &[n * d], n * d));

            // ===== attention backward (d_proj = dxmid) =====
            s.push(self.gpu.step(MATMUL_DW, &[&lb.dxmid, &lb.attn_out, self.g(&pn("attn.out.weight"))], &[n, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&lb.dxmid, self.w(&pn("attn.out.weight")), &self.d_attn_out], &[n, d, d, 0], n * d));
            s.push(self.gpu.step(ATTN_BWD_DSCORES, &[&self.d_attn_out, &lb.qkv, &lb.probs, &self.d_scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t));
            s.push(self.gpu.step(ATTN_BWD_DV, &[&lb.probs, &self.d_attn_out, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_BWD_DQ, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_BWD_DK, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            // rope backward on q and k regions of d_qkv
            s.push(self.gpu.step(ROPE_BWD, &[&self.d_qkv], &[n, c.n_heads, hd, 3 * d, 0, self.t], n * c.n_heads * half));
            s.push(self.gpu.step(ROPE_BWD, &[&self.d_qkv], &[n, c.n_heads, hd, 3 * d, d, self.t], n * c.n_heads * half));
            // qkv matmul backward -> grad_Wqkv, d_xn (=d_xn1)
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_qkv, &lb.xn1, self.g(&pn("attn.qkv.weight"))], &[n, d, 3 * d], 3 * d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_qkv, self.w(&pn("attn.qkv.weight")), &self.d_xn], &[n, d, 3 * d, 0], n * d));
            // norm1 backward -> d_tmp ; dres[l] = dxmid + d_tmp
            s.push(self.gpu.step(RMS_INV, &[&self.res[l], &self.inv], &[d, n], n));
            s.push(self.gpu.step(RMSNORM_DW, &[&self.d_xn, &self.res[l], &self.inv, self.g(&pn("norm1.weight"))], &[d, n], d));
            s.push(self.gpu.step(RMSNORM_DX, &[&self.res[l], self.w(&pn("norm1.weight")), &self.d_xn, &self.d_tmp], &[d, n], n));
            s.push(self.gpu.step(ADD2, &[&lb.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embedding backward (accumulates onto grad_emb which holds lm_head grad)
        s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("token_emb.weight")], &[n, d, c.vocab], c.vocab * d));

        s
    }

    /// One AdamW step with MoE's fixed betas/eps and no grad clipping — the
    /// signature the existing CLI / federated callers use. `t` is the
    /// (1-based) step index for bias correction. Delegates to the shared
    /// optimizer so the dispatch graph is cached across steps.
    pub fn adamw_step_betas(&self, t: u32, lr: f32, wd: f32, beta1: f32, beta2: f32, eps: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, beta1, beta2, eps, None, 1.0);
    }

    /// One AdamW step matching the unified [`model::Model`] signature (optional
    /// global-norm clip + grad-accum scale), using MoE's fixed betas/eps.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, MOE_BETA1, MOE_BETA2, MOE_EPS, clip, extra_scale);
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Write a parameter's weights from host data (required by gradcheck via the
    /// `Model` trait, and by any host-driven weight surgery).
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }

    /// Names of all trainable parameters.
    pub fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Zero every parameter's gradient (call once per effective batch, before the
    /// accumulating backward passes).
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Zero one parameter's gradient buffer (host-driven).
    pub fn zero_grad(&self, name: &str) {
        let numel = self.ps.numel(name);
        self.gpu.write(self.g(name), &vec![0u32; numel]);
    }

    /// Federated train-scope: freeze the shared backbone by zeroing the
    /// gradients of every parameter that is **not** part of expert `e` (call
    /// after [`Self::backward`]). Combined with `adamw_step(.., wd = 0.0, ..)`
    /// this leaves all non-expert weights exactly unchanged — AdamW with a zero
    /// gradient and no weight decay is a no-op — so a worker can train expert
    /// `e` against an immutable shared backbone, then return only its shard.
    pub fn freeze_grads_except_expert(&self, e: u32) {
        let names = self.param_names();
        for name in &names {
            if expert_id_of(name) != Some(e) {
                self.zero_grad(name);
            }
        }
    }

    /// Block until submitted device work completes (memory-aperture hygiene),
    /// matching the GPT/PID hot-loop discipline.
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// Save the current weights (incl. the tied `lm_head`) to a checkpoint.
    pub fn save(&self, path: &str) {
        save_weights(self, &self.cfg, path);
    }
}

// ---- the architecture-agnostic Model seam (ADR 0001 §4) ----
//
// MoE's `Config` already carries the layout; `ModelConfig` exposes it uniformly.
// `Trainer` already exposes the forward/backward/param surface as inherent
// methods, so the `Model` impl is a thin adapter. With `write_weight`/`zero_grads`
// added and `adamw_step` unified to `(t,lr,wd,clip,extra_scale)`, MoE is now
// gradient-checked by construction (ADR §8) and trainable by the generic `fit`.

impl model::ModelConfig for Config {
    fn param_list(&self) -> Vec<(String, usize)> {
        param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "model": "moe",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "n_experts": self.n_experts,
            "top_k": self.top_k, "d_ff": self.d_ff,
            "aux_loss_coef": self.aux_coef, "z_loss_coef": self.z_coef
        })
    }
    fn from_json(v: &serde_json::Value) -> Self {
        cfg_from_json(v)
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

impl model::Model for Trainer {
    type Config = Config;

    fn new(cfg: Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Trainer::new(cfg, b, t, init)
    }

    fn init_weights(cfg: &Config, seed: u64) -> HashMap<String, Vec<f32>> {
        init_weights(cfg, seed)
    }

    fn config(&self) -> &Config {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Trainer::set_batch(self, tokens, targets),
            _ => panic!("moe::Trainer only supports Batch::Lm"),
        }
    }

    fn forward(&self) -> f32 {
        Trainer::forward(self)
    }
    fn backward(&self) {
        Trainer::backward(self)
    }
    fn zero_grads(&self) {
        Trainer::zero_grads(self)
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Trainer::adamw_step(self, t, lr, wd, clip, extra_scale)
    }

    fn poll_wait(&self) {
        Trainer::poll_wait(self)
    }

    fn param_names(&self) -> Vec<String> {
        Trainer::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Trainer::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Trainer::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Trainer::read_grad(self, name)
    }

    /// MoE inference (per-position logits) lives in the standalone `Engine`
    /// (`moe::model`), not the `Trainer`; the generic trainer/gradcheck never
    /// call this, so the `Model` seam reports "no in-trainer token head".
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }

    fn save(&self, path: &str) {
        Trainer::save(self, path)
    }
    fn config_json(&self) -> serde_json::Value {
        model::ModelConfig::to_json(&self.cfg)
    }
}

/// Expert id `<E>` if `name` is `blocks.<L>.moe.experts.<E>.…` (the vertical
/// expert-shard key), else `None`. Matches `brain-federated`'s parser.
fn expert_id_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blocks.")?;
    let (_l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("moe.experts.")?;
    rest.split_once('.')?.0.parse().ok()
}

// ===========================================================================
// CLI: `train [flags]`
// ===========================================================================

fn cfg_from_json(c: &serde_json::Value) -> Config {
    let g = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("missing config.{k}")) as u32;
    let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
    Config {
        vocab: g("vocab_size"),
        block_size: g("block_size"),
        n_layers: g("n_layers"),
        d_model: g("d_model"),
        n_heads: g("n_heads"),
        n_experts: g("n_experts"),
        top_k: g("top_k"),
        d_ff: g("d_ff"),
        aux_coef: gf("aux_loss_coef", 0.01),
        z_coef: gf("z_loss_coef", 1e-4),
    }
}

// ---- toy corpus + init for from-scratch training (Rust side) ----

fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}
fn randf(s: &mut u64) -> f32 {
    (xorshift(s) >> 40) as f32 / (1u64 << 24) as f32
}
fn randn(s: &mut u64) -> f32 {
    // Box-Muller
    let u1 = randf(s).max(1e-7);
    let u2 = randf(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// The toy corpus and its substitution table (the rule's ground truth).
/// `data[i] = (data[i-2] + table[data[i-1]]) % vocab`, with a reset every 257th.
pub fn corpus_and_table(n: usize, vocab: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut s = seed.max(1);
    let mut table: Vec<u32> = (0..vocab).collect();
    for i in (1..vocab as usize).rev() {
        let j = (xorshift(&mut s) % (i as u64 + 1)) as usize;
        table.swap(i, j);
    }
    let mut d = vec![0u32; n];
    d[0] = (xorshift(&mut s) % vocab as u64) as u32;
    d[1] = (xorshift(&mut s) % vocab as u64) as u32;
    for i in 2..n {
        if i % 257 == 0 {
            d[i] = (xorshift(&mut s) % vocab as u64) as u32;
        } else {
            d[i] = (d[i - 2] + table[d[i - 1] as usize]) % vocab;
        }
    }
    (d, table)
}

fn make_corpus(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
    corpus_and_table(n, vocab, seed).0
}

/// A reset-free orbit of the same rule starting from (s0, s1) — used to test
/// generalisation to a never-seen trajectory.
pub fn orbit(table: &[u32], vocab: u32, n: usize, s0: u32, s1: u32) -> Vec<u32> {
    let mut d = vec![s0 % vocab, s1 % vocab];
    for i in 2..n {
        d.push((d[i - 2] + table[d[i - 1] as usize]) % vocab);
    }
    d
}

fn init_weights(cfg: &Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut s = seed.max(1);
    let mut map = HashMap::new();
    for (name, numel) in param_list(cfg) {
        let vals: Vec<f32> = if name.contains("norm") {
            vec![1.0; numel] // RMSNorm gain initialised to ones
        } else {
            (0..numel).map(|_| 0.02 * randn(&mut s)).collect()
        };
        map.insert(name, vals);
    }
    map
}

pub struct TrainArgs {
    pub steps: u32,
    pub b: u32,
    pub t: u32,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub out: String,
}

pub fn train(args: TrainArgs) {
    let cfg = Config {
        vocab: 64, block_size: args.t, n_layers: 2, d_model: 64, n_heads: 4,
        n_experts: 4, top_k: 2, d_ff: 128, aux_coef: 0.01, z_coef: 1e-4,
    };
    assert!(args.t <= 64, "T must be <= 64 (the toy corpus window)");

    // Materialise the synthetic corpus as a token dataset on disk, then drive the
    // architecture-agnostic generic trainer (ADR §3/§4) over MoE. The training
    // loop (LR schedule, grad-accum, eval/checkpoint, resume) is shared with the
    // GPT/PID models; only the corpus generation stays MoE-specific. There is no
    // meta.json, so `fit` infers vocab from the data (= 64, matching `cfg.vocab`).
    let corpus = make_corpus(20_000, cfg.vocab, 123);
    let dir = std::env::temp_dir().join(format!("brain_moe_train_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let n_val = corpus.len() / 10;
    let split = corpus.len() - n_val;
    let toks16: Vec<u16> = corpus.iter().map(|&v| v as u16).collect();
    data::binio::write_u16_bin(&dir.join("train.bin"), &toks16[..split]).expect("write train.bin");
    data::binio::write_u16_bin(&dir.join("val.bin"), &toks16[split..]).expect("write val.bin");

    let opts = model::FitOpts {
        steps: args.steps,
        batch_size: args.b,
        block_size: args.t,
        lr: args.lr,
        min_lr: args.lr,
        warmup: 0,
        decay_iters: args.steps,
        weight_decay: args.wd,
        grad_clip: 0.0,
        grad_accum: 1,
        eval_interval: 50,
        eval_batches: 10,
        seed: args.seed,
        ..Default::default()
    };
    let out = std::path::Path::new(&args.out);
    let _ = model::train::fit::<Trainer>(&dir, cfg, &opts, Some(out));
    let _ = std::fs::remove_dir_all(&dir);
    println!("saved {}", args.out);
}

/// Arguments for training a single expert with a frozen backbone.
pub struct ExpertTrainArgs {
    pub base_weights: String,
    pub expert: u32,
    pub out: String,
    pub steps: u32,
    pub b: u32,
    pub t: u32,
    pub lr: f32,
    pub seed: u64,
}

/// Train one expert against an immutable shared backbone, starting from `base`.
/// Writes a full updated `.weights` to `out`; the backbone and every other
/// expert are left bit-for-bit unchanged. This is the federated worker step:
/// load the common base, train your expert, return its shard.
pub fn train_expert(args: ExpertTrainArgs) {
    let c = checkpoint::load(&args.base_weights);
    let cfg = cfg_from_json(&c.header["config"]);
    assert!(args.t <= cfg.block_size, "T must be <= block_size");
    assert!(args.expert < cfg.n_experts, "expert {} >= n_experts {}", args.expert, cfg.n_experts);
    let init = c.by_role("");
    let trainer = Trainer::new(cfg.clone(), args.b, args.t, &init);

    let (corpus, _table) = corpus_and_table(20_000, cfg.vocab, 123);
    let mut rng = args.seed.max(1) ^ 0x9E3779B97F4A7C15;
    let tt = args.t as usize;
    for step in 1..=args.steps {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..args.b {
            let start = (xorshift(&mut rng) as usize) % (corpus.len() - tt - 1);
            xs.extend_from_slice(&corpus[start..start + tt]);
            ys.extend_from_slice(&corpus[start + 1..start + 1 + tt]);
        }
        trainer.set_batch(&xs, &ys);
        let loss = trainer.forward();
        trainer.zero_grads();
        trainer.backward();
        trainer.freeze_grads_except_expert(args.expert); // freeze the shared backbone
        trainer.adamw_step_betas(step, args.lr, 0.0, 0.9, 0.95, 1e-8); // wd=0 keeps frozen params fixed
        if step == 1 || step % 50 == 0 || step == args.steps {
            println!("expert {} | step {:5} | loss {:.4}", args.expert, step, loss);
        }
    }
    trainer.save(&args.out);
    println!("saved expert-{} checkpoint {}", args.expert, args.out);
}

fn save_weights(trainer: &Trainer, cfg: &Config, path: &str) {
    let d = cfg.d_model as u64;
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, _) in trainer.ps.params.iter() {
        let data = trainer.read_weight(name);
        tensors.push((name.clone(), vec![data.len() as u64], data));
    }
    // tied head expected by the inference loader
    let emb = trainer.read_weight("token_emb.weight");
    tensors.push(("lm_head.weight".to_string(), vec![cfg.vocab as u64, d], emb));

    let config = serde_json::json!({
        "vocab_size": cfg.vocab, "block_size": cfg.block_size, "n_layers": cfg.n_layers,
        "d_model": cfg.d_model, "n_heads": cfg.n_heads, "n_experts": cfg.n_experts,
        "top_k": cfg.top_k, "d_ff": cfg.d_ff
    });
    checkpoint::save(path, config, &tensors);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic_and_follows_rule() {
        let (a, ta) = corpus_and_table(2000, 64, 123);
        let (b, tb) = corpus_and_table(2000, 64, 123);
        assert_eq!(a, b);
        assert_eq!(ta, tb);
        // substitution table is a permutation of 0..vocab
        let mut sorted = ta.clone();
        sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        // the generating rule holds away from the periodic reset
        for i in 2..a.len() {
            if i % 257 != 0 {
                assert_eq!(a[i], (a[i - 2] + ta[a[i - 1] as usize]) % 64);
            }
        }
    }

    #[test]
    fn orbit_is_reset_free_rule() {
        let (_c, table) = corpus_and_table(500, 32, 7);
        let o = orbit(&table, 32, 100, 5, 9);
        for i in 2..o.len() {
            assert_eq!(o[i], (o[i - 2] + table[o[i - 1] as usize]) % 32);
        }
    }

    #[test]
    fn expert_id_of_parses_vertical_shard_key() {
        assert_eq!(expert_id_of("blocks.3.moe.experts.2.w_down.weight"), Some(2));
        assert_eq!(expert_id_of("blocks.0.moe.router.weight"), None);
        assert_eq!(expert_id_of("token_emb.weight"), None);
    }

    /// Federated train-scope: training expert 1 with a frozen backbone leaves
    /// the backbone and every other expert bit-for-bit unchanged, while expert
    /// 1's weights move. This is the "train experts separately" guarantee.
    #[test]
    fn train_scope_freezes_backbone_and_trains_one_expert() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = Config {
            vocab: 64, block_size: 16, n_layers: 1, d_model: 32, n_heads: 4,
            n_experts: 3, top_k: 2, d_ff: 64, aux_coef: 0.01, z_coef: 1e-4,
        };
        let init = init_weights(&cfg, 5);
        let tr = Trainer::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 2, 8, &init);
        let (corpus, _table) = corpus_and_table(4000, cfg.vocab, 7);

        let backbone0 = tr.read_weight("token_emb.weight");
        let e0_before = tr.read_weight("blocks.0.moe.experts.0.w_gate.weight");
        let e1_before = tr.read_weight("blocks.0.moe.experts.1.w_gate.weight");

        let mut rng = 99u64;
        let tt = 8usize;
        for step in 1..=30 {
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for _ in 0..2 {
                let start = (xorshift(&mut rng) as usize) % (corpus.len() - tt - 1);
                xs.extend_from_slice(&corpus[start..start + tt]);
                ys.extend_from_slice(&corpus[start + 1..start + 1 + tt]);
            }
            tr.set_batch(&xs, &ys);
            tr.forward();
            tr.zero_grads();
            tr.backward();
            tr.freeze_grads_except_expert(1); // freeze everything but expert 1
            tr.adamw_step_betas(step, 1e-2, 0.0, 0.9, 0.95, 1e-8); // wd=0 -> frozen params fixed
        }

        let backbone1 = tr.read_weight("token_emb.weight");
        let e0_after = tr.read_weight("blocks.0.moe.experts.0.w_gate.weight");
        let e1_after = tr.read_weight("blocks.0.moe.experts.1.w_gate.weight");

        assert_eq!(backbone0, backbone1, "backbone moved under train-scope");
        assert_eq!(e0_before, e0_after, "non-target expert 0 moved");
        assert!(e1_before != e1_after, "target expert 1 did not train");
    }
}
