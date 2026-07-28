// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PID event/effect transformer (port of `event_effect_transformer_pid_v3.py`).
//!
//! Architecture (matches torch, dropout disabled for parity):
//!   x = tok_emb[idx] + pos_emb[pos]
//!   per block (pre-norm): h=LN1(x); x += MHA(h); x += SwiGLU(LN2(x))
//!   logits = u_head(LN_final(x))           // U_BINS classes
//!   loss   = masked cross-entropy at DECIDE positions (ignore_index)
//!
//! LayerNorm (with bias), learned absolute positional embeddings, biased linear
//! layers, causal + key-padding attention, and a separate u_head distinguish it
//! from the sparse-MoE model. Shares all GPU plumbing (`gpu`, `paramstore`,
//! `optim`, `checkpoint`).

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{f, Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;

// ---- token schema (mirrors the Python module) ----
pub const PAD: u32 = 256;
pub const BOS: u32 = 257;
pub const EV_START: u32 = 258;
pub const EV_END: u32 = 259;
pub const FX_START: u32 = 260;
pub const FX_END: u32 = 261;
pub const DECIDE: u32 = 262;
pub const VOCAB_SIZE: u32 = 263;
pub const U_BINS: u32 = 81;
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
const ATTN_SCORES: usize = 10;
const ATTN_SOFTMAX: usize = 11;
const ATTN_APPLY: usize = 12;
const SILU: usize = 13;
const SILU_DA: usize = 14;
const SILU_DB: usize = 15;
const CE_VALUE: usize = 16;
const CE_GRAD: usize = 17;
const MATMUL_DX: usize = 18;
const MATMUL_DW: usize = 19;
const ATTN_DSCORES: usize = 20;
const ATTN_DV: usize = 21;
const ATTN_DQ: usize = 22;
const ATTN_DK: usize = 23;
const POS_BWD: usize = 24;
const EMB_BWD: usize = 25;
const ADD2: usize = 26;
const GRADNORM_SQ: usize = 27;
const GRAD_SCALE: usize = 28;
const ADAMW: usize = 29;
const CLIP_COEF: usize = 30;
const GRAD_SCALE_BUF: usize = 31;

const PIPELINES: &[(&str, &str)] = &[
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
    ("attn_scores_masked", kernels::ATTN_SCORES_MASKED),
    ("attn_softmax_masked", kernels::ATTN_SOFTMAX_MASKED),
    ("attn_apply", kernels::ATTN_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("ce_value_masked", kernels::CE_VALUE_MASKED),
    ("ce_grad_masked", kernels::CE_GRAD_MASKED),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("pos_bwd", kernels::POS_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("add2", kernels::ADD2),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

#[derive(Clone)]
pub struct PidConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_ff: u32,
    pub u_bins: u32,
}

impl PidConfig {
    pub fn default_small() -> PidConfig {
        PidConfig {
            vocab: VOCAB_SIZE,
            block_size: 256,
            n_layers: 1,
            d_model: 32,
            n_heads: 4,
            d_ff: 128,
            u_bins: U_BINS,
        }
    }
    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff, "u_bins": self.u_bins
        })
    }
    pub fn from_json(c: &Value) -> PidConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        PidConfig {
            vocab: g("vocab_size", VOCAB_SIZE),
            block_size: g("block_size", 256),
            n_layers: g("n_layers", 1),
            d_model: g("d_model", 32),
            n_heads: g("n_heads", 4),
            d_ff: g("d_ff", 128),
            u_bins: g("u_bins", U_BINS),
        }
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let u = self.u_bins as usize;
        let mut v = vec![
            ("tok.weight".to_string(), self.vocab as usize * d),
            ("pos.weight".to_string(), self.block_size as usize * d),
        ];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            v.push((p("ln1.weight"), d));
            v.push((p("ln1.bias"), d));
            v.push((p("attn.qkv.weight"), 3 * d * d));
            v.push((p("attn.qkv.bias"), 3 * d));
            v.push((p("attn.out.weight"), d * d));
            v.push((p("attn.out.bias"), d));
            v.push((p("ln2.weight"), d));
            v.push((p("ln2.bias"), d));
            v.push((p("ffn.value.weight"), ff * d));
            v.push((p("ffn.value.bias"), ff));
            v.push((p("ffn.gate.weight"), ff * d));
            v.push((p("ffn.gate.bias"), ff));
            v.push((p("ffn.down.weight"), d * ff));
            v.push((p("ffn.down.bias"), d));
        }
        v.push(("ln.weight".to_string(), d));
        v.push(("ln.bias".to_string(), d));
        v.push(("u_head.weight".to_string(), u * d));
        v.push(("u_head.bias".to_string(), u));
        v
    }
}

struct Layer {
    ln1_out: gpu_core::DeviceBuffer,
    qkv: gpu_core::DeviceBuffer,
    scores: gpu_core::DeviceBuffer,
    probs: gpu_core::DeviceBuffer,
    attn_ctx: gpu_core::DeviceBuffer,
    xmid: gpu_core::DeviceBuffer,
    ln2_out: gpu_core::DeviceBuffer,
    val: gpu_core::DeviceBuffer,
    gate: gpu_core::DeviceBuffer,
    ffn_h: gpu_core::DeviceBuffer,
}

pub struct Pid {
    pub gpu: Gpu,
    pub cfg: PidConfig,
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: gpu_core::DeviceBuffer,
    targets: gpu_core::DeviceBuffer,
    res: Vec<gpu_core::DeviceBuffer>, // n_layers+1 residual stream
    layers: Vec<Layer>,
    proj: gpu_core::DeviceBuffer,
    ffn_out: gpu_core::DeviceBuffer,
    xn_final: gpu_core::DeviceBuffer,
    logits: gpu_core::DeviceBuffer,
    ce_buf: gpu_core::DeviceBuffer,

    // backward temporaries
    dres: Vec<gpu_core::DeviceBuffer>,
    d_logits: gpu_core::DeviceBuffer,
    d_xn: gpu_core::DeviceBuffer,
    d_branch: gpu_core::DeviceBuffer,
    d_tmp: gpu_core::DeviceBuffer,
    dxmid: gpu_core::DeviceBuffer,
    d_attn_ctx: gpu_core::DeviceBuffer,
    d_scores: gpu_core::DeviceBuffer,
    d_qkv: gpu_core::DeviceBuffer,
    d_ffn_h: gpu_core::DeviceBuffer,
    d_gate: gpu_core::DeviceBuffer,
    d_val: gpu_core::DeviceBuffer,
    ln_mean: gpu_core::DeviceBuffer,
    ln_inv: gpu_core::DeviceBuffer,

    // Cached training dispatch graphs, built once and reused every step. The
    // forward graph is fully constant; the backward graph is constant except
    // the CE-grad normalisation count, which changes per batch and so lives in
    // a persistent writable uniform updated in `backward`. Rebuilding these per
    // step (the old behaviour) allocates a fresh uniform + bind group per
    // dispatch, exhausting the GPU memory aperture after a few thousand steps
    // and triggering a device reset (surfaced as `Buffer 'params' is invalid`).
    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: gpu_core::DeviceBuffer,
}

impl Pid {
    /// Native blocking constructor (CLI / training / tests). On wasm there is no
    /// blocking executor, so build the model with `new_async` instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(cfg: PidConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Pid {
        let gpu = Gpu::new(PIPELINES);
        Pid::from_gpu(gpu, cfg, b, t, init)
    }

    /// Async constructor for wasm: awaits device init, then builds the model. The
    /// buffer-allocation logic is shared with the native path via `from_gpu`.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_async(cfg: PidConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Pid {
        let gpu = Gpu::new_async(PIPELINES).await;
        Pid::from_gpu(gpu, cfg, b, t, init)
    }

    /// Build the model from an already-initialised `Gpu`. Target-agnostic: only
    /// device init differs between native and wasm.
    fn from_gpu(gpu: Gpu, cfg: PidConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Pid {
        let ps = ParamStore::new(&gpu, cfg.param_list(), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let u = cfg.u_bins as u64;
        let bht2 = (b * cfg.n_heads * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer("tokens", n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let targets = gpu.buffer("targets", n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let ce_grad_uni = gpu.uniform_dynamic(4); // [n, u, IGNORE, count]; count refreshed per batch

        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                ln1_out: st(n * d),
                qkv: st(n * 3 * d),
                scores: st(bht2),
                probs: st(bht2),
                attn_ctx: st(n * d),
                xmid: st(n * d),
                ln2_out: st(n * d),
                val: st(n * ff),
                gate: st(n * ff),
                ffn_h: st(n * ff),
            });
        }
        let mut pid = Pid {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            opt,
            tokens,
            targets,
            res,
            layers,
            proj: st(n * d),
            ffn_out: st(n * d),
            xn_final: st(n * d),
            logits: st(n * u),
            ce_buf: st(n),
            dres,
            d_logits: st(n * u),
            d_xn: st(n * d),
            d_branch: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_attn_ctx: st(n * d),
            d_scores: st(bht2),
            d_qkv: st(n * 3 * d),
            d_ffn_h: st(n * ff),
            d_gate: st(n * ff),
            d_val: st(n * ff),
            ln_mean: st(n),
            ln_inv: st(n),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            gpu,
        };
        pid.build_graphs();
        pid
    }

    /// Build the cached forward and backward dispatch graphs once, after all
    /// buffers exist. Reused on every training iteration.
    fn build_graphs(&mut self) {
        self.fwd_steps = self.forward_steps(self.b, self.t);
        self.bwd_steps = self.build_backward_steps();
    }

    /// Upload a batch. `y` uses IGNORE for masked positions. Stores the
    /// non-ignored count for loss/grad normalisation.
    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.ps.w(name)
    }

    /// Forward over the first `b_use * t_use` rows. Returns mean CE.
    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ff = c.d_ff;
        let u = c.u_bins;
        let hd = c.head_dim();
        let mut s: Vec<Step> = Vec::new();

        // embeddings
        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d));
        s.push(self.gpu.step(POS_ADD, &[&self.res[0], self.w("pos.weight")], &[n * d, d, t_use], n * d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // attention sub-layer
            s.push(self.gpu.step(LAYERNORM, &[&self.res[l], self.w(&p("ln1.weight")), self.w(&p("ln1.bias")), &lb.ln1_out], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(MATMUL, &[&lb.ln1_out, self.w(&p("attn.qkv.weight")), &lb.qkv], &[n, d, 3 * d], n * 3 * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.qkv, self.w(&p("attn.qkv.bias"))], &[n, 3 * d], n * 3 * d));
            s.push(self.gpu.step(ATTN_SCORES, &[&lb.qkv, &self.tokens, &lb.scores], &[b_use, c.n_heads, t_use, hd, 3 * d, 0, d, PAD], b_use * c.n_heads * t_use * t_use));
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&lb.scores, &lb.probs], &[b_use, c.n_heads, t_use], b_use * c.n_heads * t_use));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_ctx], &[b_use, c.n_heads, t_use, hd, 3 * d, 2 * d, d], b_use * c.n_heads * t_use * hd));
            s.push(self.gpu.step(MATMUL, &[&lb.attn_ctx, self.w(&p("attn.out.weight")), &self.proj], &[n, d, d], n * d));
            s.push(self.gpu.step(BIAS_ADD, &[&self.proj, self.w(&p("attn.out.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // SwiGLU FFN sub-layer
            s.push(self.gpu.step(LAYERNORM, &[&lb.xmid, self.w(&p("ln2.weight")), self.w(&p("ln2.bias")), &lb.ln2_out], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(MATMUL, &[&lb.ln2_out, self.w(&p("ffn.value.weight")), &lb.val], &[n, d, ff], n * ff));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.val, self.w(&p("ffn.value.bias"))], &[n, ff], n * ff));
            s.push(self.gpu.step(MATMUL, &[&lb.ln2_out, self.w(&p("ffn.gate.weight")), &lb.gate], &[n, d, ff], n * ff));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.gate, self.w(&p("ffn.gate.bias"))], &[n, ff], n * ff));
            s.push(self.gpu.step(SILU, &[&lb.gate, &lb.val, &lb.ffn_h], &[n * ff], n * ff));
            s.push(self.gpu.step(MATMUL, &[&lb.ffn_h, self.w(&p("ffn.down.weight")), &self.ffn_out], &[n, ff, d], n * d));
            s.push(self.gpu.step(BIAS_ADD, &[&self.ffn_out, self.w(&p("ffn.down.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.ffn_out, &self.res[l + 1]], &[n * d], n * d));
        }

        // final norm + head + loss
        let last = c.n_layers as usize;
        s.push(self.gpu.step(LAYERNORM, &[&self.res[last], self.w("ln.weight"), self.w("ln.bias"), &self.xn_final], &[d, n, f(1e-5)], n));
        s.push(self.gpu.step(MATMUL, &[&self.xn_final, self.w("u_head.weight"), &self.logits], &[n, d, u], n * u));
        s.push(self.gpu.step(BIAS_ADD, &[&self.logits, self.w("u_head.bias")], &[n, u], n * u));
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, u, IGNORE], n));
        s
    }

    /// Submit the forward pass to the GPU. No host readback -- the loss scalar,
    /// gradients, and weights all stay device-resident. Use `loss()` to fetch
    /// the scalar only when needed (e.g. logging), avoiding per-step PCIe traffic.
    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd_steps);
    }

    /// Mean cross-entropy of the most recently submitted forward (reads the
    /// per-position ce buffer back to the host -- call sparingly). Native only:
    /// the wasm build runs inference (logits) only, not loss/training readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn loss(&self) -> f32 {
        let n = (self.b * self.t) as usize;
        let losses = self.gpu.read(&self.ce_buf, n);
        losses.iter().sum::<f32>() / self.count.get()
    }

    /// Convenience: forward + loss. Used by validation/tests, not the hot loop.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    /// Backward; accumulates gradients (does NOT zero them — caller zeroes once
    /// per effective batch). Normalises the CE grad by the stored count.
    /// Run the (cached) backward pass, refreshing only the per-batch CE-grad
    /// normalisation count. Accumulates gradients (does NOT zero them — caller
    /// zeroes once per effective batch).
    pub fn backward(&self) {
        let n = self.b * self.t;
        let u = self.cfg.u_bins;
        self.gpu.write(&self.ce_grad_uni, &[n, u, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    /// Build the backward dispatch graph (constant across steps apart from the
    /// CE-grad count, which lives in `ce_grad_uni`).
    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let u = c.u_bins;
        let hd = c.head_dim();
        let g = |name: &str| self.ps.g(name);
        let p = |l: usize, name: &str| format!("blocks.{l}.{name}");
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final layernorm ----
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * u));
        s.push(self.gpu.step(BIAS_GRAD, &[&self.d_logits, g("u_head.bias")], &[n, u], u));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_logits, &self.xn_final, g("u_head.weight")], &[n, d, u], u * d));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_logits, self.w("u_head.weight"), &self.d_xn], &[n, d, u, 0], n * d));
        let last = c.n_layers as usize;
        s.push(self.gpu.step(LN_STATS, &[&self.res[last], &self.ln_mean, &self.ln_inv], &[d, n, f(1e-5)], n));
        s.push(self.gpu.step(LN_DGAMMA, &[&self.d_xn, &self.res[last], &self.ln_mean, &self.ln_inv, g("ln.weight")], &[d, n], d));
        s.push(self.gpu.step(LN_DBETA, &[&self.d_xn, g("ln.bias")], &[d, n], d));
        s.push(self.gpu.step(LN_DX, &[&self.res[last], self.w("ln.weight"), &self.d_xn, &self.dres[last]], &[d, n, f(1e-5)], n));

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            // ---- FFN backward (input grad = dres[l+1]) ----
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dres[l + 1], g(&p(l, "ffn.down.bias"))], &[n, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dres[l + 1], &lb.ffn_h, g(&p(l, "ffn.down.weight"))], &[n, ff, d], d * ff));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dres[l + 1], self.w(&p(l, "ffn.down.weight")), &self.d_ffn_h], &[n, ff, d, 0], n * ff));
            s.push(self.gpu.step(SILU_DA, &[&lb.gate, &lb.val, &self.d_ffn_h, &self.d_gate], &[n * ff], n * ff));
            s.push(self.gpu.step(SILU_DB, &[&lb.gate, &self.d_ffn_h, &self.d_val], &[n * ff], n * ff));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_val, g(&p(l, "ffn.value.bias"))], &[n, ff], ff));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_val, &lb.ln2_out, g(&p(l, "ffn.value.weight"))], &[n, d, ff], ff * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_val, self.w(&p(l, "ffn.value.weight")), &self.d_branch], &[n, d, ff, 0], n * d));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_gate, g(&p(l, "ffn.gate.bias"))], &[n, ff], ff));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_gate, &lb.ln2_out, g(&p(l, "ffn.gate.weight"))], &[n, d, ff], ff * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_gate, self.w(&p(l, "ffn.gate.weight")), &self.d_branch], &[n, d, ff, 1], n * d));
            s.push(self.gpu.step(LN_STATS, &[&lb.xmid, &self.ln_mean, &self.ln_inv], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &lb.xmid, &self.ln_mean, &self.ln_inv, g(&p(l, "ln2.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln2.bias"))], &[d, n], d));
            s.push(self.gpu.step(LN_DX, &[&lb.xmid, self.w(&p(l, "ln2.weight")), &self.d_branch, &self.d_tmp], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // ---- attention backward (input grad = dxmid) ----
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dxmid, g(&p(l, "attn.out.bias"))], &[n, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dxmid, &lb.attn_ctx, g(&p(l, "attn.out.weight"))], &[n, d, d], d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dxmid, self.w(&p(l, "attn.out.weight")), &self.d_attn_ctx], &[n, d, d, 0], n * d));
            s.push(self.gpu.step(ATTN_DSCORES, &[&self.d_attn_ctx, &lb.qkv, &lb.probs, &self.d_scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t));
            s.push(self.gpu.step(ATTN_DV, &[&lb.probs, &self.d_attn_ctx, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_DQ, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(ATTN_DK, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_qkv, g(&p(l, "attn.qkv.bias"))], &[n, 3 * d], 3 * d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_qkv, &lb.ln1_out, g(&p(l, "attn.qkv.weight"))], &[n, d, 3 * d], 3 * d * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_qkv, self.w(&p(l, "attn.qkv.weight")), &self.d_branch], &[n, d, 3 * d, 0], n * d));
            s.push(self.gpu.step(LN_STATS, &[&self.res[l], &self.ln_mean, &self.ln_inv], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &self.res[l], &self.ln_mean, &self.ln_inv, g(&p(l, "ln1.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln1.bias"))], &[d, n], d));
            s.push(self.gpu.step(LN_DX, &[&self.res[l], self.w(&p(l, "ln1.weight")), &self.d_branch, &self.d_tmp], &[d, n, f(1e-5)], n));
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // ---- embeddings backward ----
        s.push(self.gpu.step(POS_BWD, &[&self.dres[0], g("pos.weight")], &[self.b, self.t, d], self.t * d));
        s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], g("tok.weight")], &[n, d, c.vocab], c.vocab * d));

        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Wait for all submitted GPU work to finish so wgpu can reclaim the
    /// transient per-submit memory. Call once per training step; without it the
    /// submit-only hot loop exhausts the GPU aperture and a later allocation
    /// fails (see `Gpu::poll_wait`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    /// Overwrite a parameter's weights from host data (required by gradcheck via
    /// the `Model` trait, and by any host-driven weight surgery).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }
    /// Logits for the current batch (n_rows = b*t), shape [n_rows, u_bins].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits, (self.b * self.t * self.cfg.u_bins) as usize)
    }
    /// Debug: embedding+positional output res[0], shape [b*t, d_model].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_res0(&self) -> Vec<f32> {
        self.gpu.read(&self.res[0], (self.b * self.t * self.cfg.d_model) as usize)
    }

    /// Submit a single-sequence forward over `tokens` (B=1). Shared prologue for
    /// both the native and wasm logits paths; only the readback differs.
    fn submit_logits(&self, tokens: &[u32]) -> u32 {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        t_use
    }

    /// Inference: full logits for every position of a single sequence (B=1).
    /// `self` must have been built with b=1 and t>=tokens.len().
    #[cfg(not(target_arch = "wasm32"))]
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = self.submit_logits(tokens);
        self.gpu.read(&self.logits, (t_use * self.cfg.u_bins) as usize)
    }

    /// U-bin logits at the last position (the DECIDE token in a rollout).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn logits_last(&self, tokens: &[u32]) -> Vec<f32> {
        let all = self.logits_all(tokens);
        let u = self.cfg.u_bins as usize;
        all[all.len() - u..].to_vec()
    }

    /// Async logits for wasm: same forward pass, async buffer readback.
    #[cfg(target_arch = "wasm32")]
    pub async fn logits_all_async(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = self.submit_logits(tokens);
        self.gpu
            .read_async(&self.logits, (t_use * self.cfg.u_bins) as usize)
            .await
    }

    /// U-bin logits at the last position (the DECIDE token), async for wasm.
    #[cfg(target_arch = "wasm32")]
    pub async fn logits_last_async(&self, tokens: &[u32]) -> Vec<f32> {
        let all = self.logits_all_async(tokens).await;
        let u = self.cfg.u_bins as usize;
        all[all.len() - u..].to_vec()
    }

    /// Measure the wall-clock time of a single inference path (one forward over
    /// `tokens`, ending at the DECIDE position, including GPU readback). Each
    /// `logits_last` call is fully synchronous -- it blocks on the GPU and maps
    /// the result back -- so the timing is end-to-end and exact. Returns
    /// (mean, min, max) seconds per inference over `cycles`, after `warmup`
    /// untimed iterations (to amortise pipeline warm-up / allocator effects).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn profile_inference(&self, tokens: &[u32], cycles: usize, warmup: usize) -> (f64, f64, f64) {
        for _ in 0..warmup {
            let _ = self.logits_last(tokens);
        }
        let mut total = 0.0f64;
        let mut min = f64::INFINITY;
        let mut max = 0.0f64;
        for _ in 0..cycles.max(1) {
            let t0 = std::time::Instant::now();
            let _ = self.logits_last(tokens);
            let dt = t0.elapsed().as_secs_f64();
            total += dt;
            min = min.min(dt);
            max = max.max(dt);
        }
        (total / cycles.max(1) as f64, min, max)
    }

    #[cfg(not(target_arch = "wasm32"))]
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

// ---- the architecture-agnostic Model seam (ADR 0001 §4) ----
//
// PID already exposes nearly the whole surface as inherent methods, so these
// impls are thin adapters: `set_batch` maps `Batch::Lm` onto the inherent
// two-slice upload (`ys` already carries IGNORE at non-DECIDE positions), the
// trait `forward()` is PID's existing `forward_submit()+loss()` wrapper, and
// `logits_all` wraps the always-present u_head in `Some`. With `write_weight`
// added, PID is now gradient-checked by construction (ADR §8). The wasm
// inference path (`new_async`/`logits_*_async`) is untouched — the `Model`
// trait is native-only because its methods read back from the device.

impl model::ModelConfig for PidConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        PidConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        PidConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        PidConfig::from_json(v)
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

#[cfg(not(target_arch = "wasm32"))]
impl model::Model for Pid {
    type Config = PidConfig;

    fn new(cfg: PidConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Pid::new(cfg, b, t, init)
    }

    fn init_weights(cfg: &PidConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::data::init_weights(cfg, seed)
    }

    fn config(&self) -> &PidConfig {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => Pid::set_batch(self, tokens, targets),
            _ => panic!("pid::Pid only supports Batch::Lm"),
        }
    }

    fn forward(&self) -> f32 {
        Pid::forward(self)
    }
    fn backward(&self) {
        Pid::backward(self)
    }
    fn zero_grads(&self) {
        Pid::zero_grads(self)
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Pid::adamw_step(self, t, lr, wd, clip, extra_scale)
    }

    fn poll_wait(&self) {
        Pid::poll_wait(self)
    }

    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Pid::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Pid::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Pid::read_grad(self, name)
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(Pid::logits_all(self, tokens))
    }

    fn save(&self, path: &str) {
        Pid::save(self, path)
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
    fn param_list_names_and_sizes() {
        let cfg = PidConfig::default_small(); // d=32, layers=1, ff=128, u=81
        let p = cfg.param_list();
        // 2 embeddings + 14 per layer + 4 head/final = 20 for one layer
        assert_eq!(p.len(), 20);
        let m: std::collections::HashMap<_, _> = p.iter().cloned().collect();
        assert_eq!(m["tok.weight"], 263 * 32);
        assert_eq!(m["pos.weight"], 256 * 32);
        assert_eq!(m["blocks.0.attn.qkv.weight"], 3 * 32 * 32);
        assert_eq!(m["blocks.0.attn.qkv.bias"], 3 * 32);
        assert_eq!(m["blocks.0.ffn.value.weight"], 128 * 32);
        assert_eq!(m["blocks.0.ffn.down.weight"], 32 * 128);
        assert_eq!(m["u_head.weight"], 81 * 32);
        assert_eq!(m["u_head.bias"], 81);
        // two layers => 2 + 28 + 4
        let mut c2 = cfg.clone();
        c2.n_layers = 2;
        assert_eq!(c2.param_list().len(), 34);
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = PidConfig { vocab: 263, block_size: 128, n_layers: 2, d_model: 48, n_heads: 6, d_ff: 192, u_bins: 81 };
        let back = PidConfig::from_json(&cfg.to_json());
        assert_eq!(back.block_size, 128);
        assert_eq!(back.n_layers, 2);
        assert_eq!(back.d_model, 48);
        assert_eq!(back.n_heads, 6);
        assert_eq!(back.d_ff, 192);
        assert_eq!(back.head_dim(), 8);
    }

    #[test]
    fn forward_is_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let mut cfg = PidConfig::default_small();
        cfg.block_size = 16;
        let init = crate::data::init_weights(&cfg, 3);
        let model = Pid::from_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, 8, &init);
        // a BOS + a few event-ish tokens; label one DECIDE position
        let x: Vec<u32> = vec![BOS, EV_START, 1, 50, 50, 50, EV_END, DECIDE];
        let mut y = vec![IGNORE; 8];
        y[7] = 40;
        model.set_batch(&x, &y);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss not finite/positive: {l1}");
        assert!((l1 - l2).abs() < 1e-6, "forward not deterministic");
        let logits = model.logits_last(&x);
        assert_eq!(logits.len(), U_BINS as usize);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backward_runs_and_grads_finite() {
        if gpu_disabled() {
            return;
        }
        let mut cfg = PidConfig::default_small();
        cfg.block_size = 16;
        let init = crate::data::init_weights(&cfg, 4);
        let model = Pid::from_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, 8, &init);
        let x: Vec<u32> = vec![BOS, EV_START, 1, 50, 50, 50, EV_END, DECIDE];
        let mut y = vec![IGNORE; 8];
        y[7] = 30;
        model.set_batch(&x, &y);
        model.zero_grads();
        model.forward();
        model.backward();
        for (name, _) in model.ps.params.iter() {
            assert!(model.read_grad(name).iter().all(|v| v.is_finite()), "nan grad in {name}");
        }
    }

    #[test]
    fn profile_inference_returns_ordered_positive_timing() {
        if gpu_disabled() {
            return;
        }
        let mut cfg = PidConfig::default_small();
        cfg.block_size = 16;
        let init = crate::data::init_weights(&cfg, 5);
        let model = Pid::from_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 16, &init);
        let ctx: Vec<u32> = vec![BOS, EV_START, 1, 50, 50, 50, EV_END, DECIDE];
        let (mean, min, max) = model.profile_inference(&ctx, 3, 1);
        assert!(mean.is_finite() && mean > 0.0);
        assert!(min <= mean + 1e-9 && mean <= max + 1e-9);
    }
}
