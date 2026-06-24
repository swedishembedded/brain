// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense GPT decoder Transformer (nanogpt parity), forward + backprop as WGSL
//! compute dispatches. Shares the engine with the MoE/PID models (`gpu_core`,
//! `paramstore`, `optim`, `kernels`).
//!
//! Architecture (pre-norm, matches nanogpt `GPT`; dropout disabled):
//!   x = tok_emb[idx] + pos_emb[pos]
//!   per block: h = LN1(x); x += Wo·MHA(h) ;  h = LN2(x); x += proj·GELU(fc·h)
//!   logits = lm_head( LN_f(x) )            // over vocab; lm_head has no bias
//!   loss   = cross-entropy (ignore_index = IGNORE), so masked datasets work.
//!
//! Differences vs nanogpt, intentional and documented:
//!   * `lm_head` is **untied** from `tok.weight` (nanogpt ties them). Untied
//!     keeps each gradient written exactly once, matching the rest of the
//!     engine and the finite-difference gradient check. Tying (which needs grad
//!     accumulation into `tok.weight`) is a follow-up.
//!   * GELU uses the tanh approximation (see `kernels/wgsl/gelu.wgsl`).

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{f, Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;

/// Cross-entropy ignore index (masked target positions). The data loader emits
/// `-1` as `i32`; reinterpreted as `u32` that is exactly this value.
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
const GELU: usize = 13;
const GELU_BWD: usize = 14;
const CE_VALUE: usize = 15;
const CE_GRAD: usize = 16;
const MATMUL_DX: usize = 17;
const MATMUL_DW: usize = 18;
const ATTN_DSCORES: usize = 19;
const ATTN_DV: usize = 20;
const ATTN_DQ: usize = 21;
const ATTN_DK: usize = 22;
const POS_BWD: usize = 23;
const EMB_BWD: usize = 24;
const ADD2: usize = 25;
const GRADNORM_SQ: usize = 26;
const GRAD_SCALE: usize = 27;
const ADAMW: usize = 28;
const CLIP_COEF: usize = 29;
const GRAD_SCALE_BUF: usize = 30;

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
    ("attn_scores", kernels::ATTN_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
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

#[derive(Clone, Debug)]
pub struct GptConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_ff: u32,
}

impl GptConfig {
    /// A tiny config for tests / gradient checks.
    pub fn tiny() -> GptConfig {
        GptConfig {
            vocab: 65,
            block_size: 64,
            n_layers: 2,
            d_model: 32,
            n_heads: 4,
            d_ff: 128,
        }
    }

    /// nanogpt's `4 * d_model` feed-forward width.
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
            "model": "gpt",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff
        })
    }

    pub fn from_json(c: &Value) -> GptConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        GptConfig {
            vocab: g("vocab_size", 65),
            block_size: g("block_size", 64),
            n_layers: g("n_layers", 2),
            d_model: g("d_model", 32),
            n_heads: g("n_heads", 4),
            d_ff: g("d_ff", 128),
        }
    }

    /// Parameter list: `(name, numel)`. Order is irrelevant to correctness.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let mut out = vec![
            ("tok.weight".to_string(), v * d),
            ("pos.weight".to_string(), self.block_size as usize * d),
        ];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
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
        out.push(("ln.weight".to_string(), d));
        out.push(("ln.bias".to_string(), d));
        out.push(("lm_head.weight".to_string(), v * d)); // untied, no bias
        out
    }
}

struct Layer {
    ln1_out: wgpu::Buffer,
    qkv: wgpu::Buffer,
    scores: wgpu::Buffer,
    probs: wgpu::Buffer,
    attn_ctx: wgpu::Buffer,
    xmid: wgpu::Buffer,
    ln2_out: wgpu::Buffer,
    fc: wgpu::Buffer,   // c_fc pre-activation
    gelu: wgpu::Buffer, // GELU(fc)
}

pub struct Gpt {
    pub gpu: Gpu,
    pub cfg: GptConfig,
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: wgpu::Buffer,
    targets: wgpu::Buffer,
    res: Vec<wgpu::Buffer>,
    layers: Vec<Layer>,
    proj: wgpu::Buffer,
    ffn_out: wgpu::Buffer,
    xn_final: wgpu::Buffer,
    logits: wgpu::Buffer,
    ce_buf: wgpu::Buffer,

    // backward temporaries
    dres: Vec<wgpu::Buffer>,
    d_logits: wgpu::Buffer,
    d_xn: wgpu::Buffer,
    d_branch: wgpu::Buffer,
    d_tmp: wgpu::Buffer,
    dxmid: wgpu::Buffer,
    d_attn_ctx: wgpu::Buffer,
    d_scores: wgpu::Buffer,
    d_qkv: wgpu::Buffer,
    d_gelu: wgpu::Buffer,
    d_fc: wgpu::Buffer,
    ln_mean: wgpu::Buffer,
    ln_inv: wgpu::Buffer,

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    ce_grad_uni: wgpu::Buffer,
}

impl Gpt {
    /// Load a model from a `.weights` checkpoint, sized for batch `b` × seq `t`.
    pub fn load(path: &str, b: u32, t: u32) -> Gpt {
        let c = checkpoint::load(path);
        let cfg = GptConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Gpt::new(cfg, b, t, &init)
    }

    pub fn new(cfg: GptConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Gpt {
        let gpu = Gpu::new(PIPELINES);
        let ps = ParamStore::new(&gpu, cfg.param_list(), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let bht2 = (b * cfg.n_heads * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer(
            "tokens",
            n * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let targets = gpu.buffer(
            "targets",
            n * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                ln1_out: st(n * d),
                qkv: st(n * 3 * d),
                scores: st(bht2),
                probs: st(bht2),
                attn_ctx: st(n * d),
                xmid: st(n * d),
                ln2_out: st(n * d),
                fc: st(n * ff),
                gelu: st(n * ff),
            });
        }
        let mut m = Gpt {
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
            logits: st(n * v),
            ce_buf: st(n),
            dres,
            d_logits: st(n * v),
            d_xn: st(n * d),
            d_branch: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_attn_ctx: st(n * d),
            d_scores: st(bht2),
            d_qkv: st(n * 3 * d),
            d_gelu: st(n * ff),
            d_fc: st(n * ff),
            ln_mean: st(n),
            ln_inv: st(n),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            ce_grad_uni,
            gpu,
        };
        m.fwd_steps = m.forward_steps(m.b, m.t);
        m.bwd_steps = m.build_backward_steps();
        m
    }

    /// Upload a batch. `y` uses [`IGNORE`] for masked positions.
    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, x);
        self.gpu.write(&self.targets, y);
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    fn w(&self, name: &str) -> &wgpu::Buffer {
        self.ps.w(name)
    }

    fn forward_steps(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let mut s: Vec<Step> = Vec::new();

        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d));
        s.push(self.gpu.step(POS_ADD, &[&self.res[0], self.w("pos.weight")], &[n * d, d, t_use], n * d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            // attention
            s.push(self.gpu.step(LAYERNORM, &[&self.res[l], self.w(&p("ln1.weight")), self.w(&p("ln1.bias")), &lb.ln1_out], &[d, n], n));
            s.push(self.gpu.step(MATMUL, &[&lb.ln1_out, self.w(&p("attn.qkv.weight")), &lb.qkv], &[n, d, 3 * d], n * 3 * d));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.qkv, self.w(&p("attn.qkv.bias"))], &[n, 3 * d], n * 3 * d));
            s.push(self.gpu.step(ATTN_SCORES, &[&lb.qkv, &lb.scores], &[b_use, c.n_heads, t_use, hd, 3 * d, 0, d], b_use * c.n_heads * t_use * t_use));
            s.push(self.gpu.step(ATTN_SOFTMAX, &[&lb.scores, &lb.probs], &[b_use, c.n_heads, t_use], b_use * c.n_heads * t_use));
            s.push(self.gpu.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_ctx], &[b_use, c.n_heads, t_use, hd, 3 * d, 2 * d, d], b_use * c.n_heads * t_use * hd));
            s.push(self.gpu.step(MATMUL, &[&lb.attn_ctx, self.w(&p("attn.out.weight")), &self.proj], &[n, d, d], n * d));
            s.push(self.gpu.step(BIAS_ADD, &[&self.proj, self.w(&p("attn.out.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // MLP: fc -> GELU -> proj
            s.push(self.gpu.step(LAYERNORM, &[&lb.xmid, self.w(&p("ln2.weight")), self.w(&p("ln2.bias")), &lb.ln2_out], &[d, n], n));
            s.push(self.gpu.step(MATMUL, &[&lb.ln2_out, self.w(&p("mlp.fc.weight")), &lb.fc], &[n, d, ff], n * ff));
            s.push(self.gpu.step(BIAS_ADD, &[&lb.fc, self.w(&p("mlp.fc.bias"))], &[n, ff], n * ff));
            s.push(self.gpu.step(GELU, &[&lb.fc, &lb.gelu], &[n * ff], n * ff));
            s.push(self.gpu.step(MATMUL, &[&lb.gelu, self.w(&p("mlp.proj.weight")), &self.ffn_out], &[n, ff, d], n * d));
            s.push(self.gpu.step(BIAS_ADD, &[&self.ffn_out, self.w(&p("mlp.proj.bias"))], &[n, d], n * d));
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.ffn_out, &self.res[l + 1]], &[n * d], n * d));
        }

        let last = c.n_layers as usize;
        s.push(self.gpu.step(LAYERNORM, &[&self.res[last], self.w("ln.weight"), self.w("ln.bias"), &self.xn_final], &[d, n], n));
        s.push(self.gpu.step(MATMUL, &[&self.xn_final, self.w("lm_head.weight"), &self.logits], &[n, d, v], n * v));
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
        s
    }

    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd_steps);
    }

    pub fn loss(&self) -> f32 {
        let n = (self.b * self.t) as usize;
        let losses = self.gpu.read(&self.ce_buf, n);
        losses.iter().sum::<f32>() / self.count.get()
    }

    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    pub fn backward(&self) {
        let n = self.b * self.t;
        let v = self.cfg.vocab;
        self.gpu.write(&self.ce_grad_uni, &[n, v, IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let v = c.vocab;
        let hd = c.head_dim();
        let g = |name: &str| self.ps.g(name);
        let p = |l: usize, name: &str| format!("blocks.{l}.{name}");
        let mut s: Vec<Step> = Vec::new();

        // head (no bias) + final layernorm
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * v));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_logits, &self.xn_final, g("lm_head.weight")], &[n, d, v], v * d));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_logits, self.w("lm_head.weight"), &self.d_xn], &[n, d, v, 0], n * d));
        let last = c.n_layers as usize;
        s.push(self.gpu.step(LN_STATS, &[&self.res[last], &self.ln_mean, &self.ln_inv], &[d, n], n));
        s.push(self.gpu.step(LN_DGAMMA, &[&self.d_xn, &self.res[last], &self.ln_mean, &self.ln_inv, g("ln.weight")], &[d, n], d));
        s.push(self.gpu.step(LN_DBETA, &[&self.d_xn, g("ln.bias")], &[d, n], d));
        s.push(self.gpu.step(LN_DX, &[&self.res[last], self.w("ln.weight"), &self.d_xn, &self.dres[last]], &[d, n], n));

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            // MLP backward (input grad = dres[l+1])
            s.push(self.gpu.step(BIAS_GRAD, &[&self.dres[l + 1], g(&p(l, "mlp.proj.bias"))], &[n, d], d));
            s.push(self.gpu.step(MATMUL_DW, &[&self.dres[l + 1], &lb.gelu, g(&p(l, "mlp.proj.weight"))], &[n, ff, d], d * ff));
            s.push(self.gpu.step(MATMUL_DX, &[&self.dres[l + 1], self.w(&p(l, "mlp.proj.weight")), &self.d_gelu], &[n, ff, d, 0], n * ff));
            s.push(self.gpu.step(GELU_BWD, &[&lb.fc, &self.d_gelu, &self.d_fc], &[n * ff], n * ff));
            s.push(self.gpu.step(BIAS_GRAD, &[&self.d_fc, g(&p(l, "mlp.fc.bias"))], &[n, ff], ff));
            s.push(self.gpu.step(MATMUL_DW, &[&self.d_fc, &lb.ln2_out, g(&p(l, "mlp.fc.weight"))], &[n, d, ff], ff * d));
            s.push(self.gpu.step(MATMUL_DX, &[&self.d_fc, self.w(&p(l, "mlp.fc.weight")), &self.d_branch], &[n, d, ff, 0], n * d));
            s.push(self.gpu.step(LN_STATS, &[&lb.xmid, &self.ln_mean, &self.ln_inv], &[d, n], n));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &lb.xmid, &self.ln_mean, &self.ln_inv, g(&p(l, "ln2.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln2.bias"))], &[d, n], d));
            s.push(self.gpu.step(LN_DX, &[&lb.xmid, self.w(&p(l, "ln2.weight")), &self.d_branch, &self.d_tmp], &[d, n], n));
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // attention backward (input grad = dxmid)
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
            s.push(self.gpu.step(LN_STATS, &[&self.res[l], &self.ln_mean, &self.ln_inv], &[d, n], n));
            s.push(self.gpu.step(LN_DGAMMA, &[&self.d_branch, &self.res[l], &self.ln_mean, &self.ln_inv, g(&p(l, "ln1.weight"))], &[d, n], d));
            s.push(self.gpu.step(LN_DBETA, &[&self.d_branch, g(&p(l, "ln1.bias"))], &[d, n], d));
            s.push(self.gpu.step(LN_DX, &[&self.res[l], self.w(&p(l, "ln1.weight")), &self.d_branch, &self.d_tmp], &[d, n], n));
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embeddings backward
        s.push(self.gpu.step(POS_BWD, &[&self.dres[0], g("pos.weight")], &[self.b, self.t, d], self.t * d));
        s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], g("tok.weight")], &[n, d, c.vocab], c.vocab * d));
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

    /// Logits for every position of a single sequence (B must be 1, t>=len).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "decoder sized too small");
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.forward_steps(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab) as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn param_list_shapes() {
        let cfg = GptConfig::tiny(); // v=65 d=32 layers=2 ff=128
        let m: HashMap<_, _> = cfg.param_list().into_iter().collect();
        assert_eq!(m["tok.weight"], 65 * 32);
        assert_eq!(m["pos.weight"], 64 * 32);
        assert_eq!(m["blocks.0.mlp.fc.weight"], 128 * 32);
        assert_eq!(m["blocks.1.mlp.proj.weight"], 32 * 128);
        assert_eq!(m["lm_head.weight"], 65 * 32);
        assert!(!m.contains_key("lm_head.bias"));
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = GptConfig { vocab: 100, block_size: 128, n_layers: 3, d_model: 48, n_heads: 6, d_ff: 192 };
        let back = GptConfig::from_json(&cfg.to_json());
        assert_eq!(back.vocab, 100);
        assert_eq!(back.n_layers, 3);
        assert_eq!(back.head_dim(), 8);
    }

    #[test]
    fn forward_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 7);
        let model = Gpt::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 3 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 3 + 1) % 65) as u32).collect();
        model.set_batch(&x, &y);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss {l1}");
        assert!((l1 - l2).abs() < 1e-6, "not deterministic");
        // untrained CE should be near ln(vocab).
        assert!(l1 < 2.0 * (65f32).ln(), "loss implausibly large: {l1}");
    }

    #[test]
    fn backward_grads_finite() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 9);
        let model = Gpt::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 5 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 5 + 1) % 65) as u32).collect();
        model.set_batch(&x, &y);
        model.zero_grads();
        model.forward();
        model.backward();
        for (name, _) in model.ps.params.iter() {
            assert!(model.read_grad(name).iter().all(|v| v.is_finite()), "nan grad in {name}");
        }
    }

    #[test]
    fn one_adamw_step_reduces_loss_on_fixed_batch() {
        if gpu_disabled() {
            return;
        }
        let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
        let init = crate::init::init_weights(&cfg, 11);
        let model = Gpt::new(cfg, 2, 8, &init);
        let x: Vec<u32> = (0..16).map(|i| (i * 7 % 65) as u32).collect();
        let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 65) as u32).collect();
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
        assert!(after < before, "overfit step did not reduce loss: {before} -> {after}");
    }
}
