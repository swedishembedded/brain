// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full training pipeline (forward + backprop + AdamW) as a WGSL compute
//! pipeline. Mirrors `tiny_sparse_moe.py`'s training step exactly, so the Rust
//! executable can train the model from scratch on the GPU.
//!
//! Two entry points:
//!   * `validate(path)` — load the PyTorch golden reference (init weights, a
//!     fixed batch, per-parameter grads, post-AdamW weights), run one Rust step
//!     and report the max gradient / weight error. This is the correctness gate.
//!   * `train(args)`    — generate the toy corpus, init weights, and run the
//!     optimisation loop, then save weights in the inference engine's format.
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

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

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
    ("ce_grad", kernels::CE_GRAD),
    ("ce_value", kernels::CE_VALUE),
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
];

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

    weights: HashMap<String, DeviceBuffer>,
    grads: HashMap<String, DeviceBuffer>,
    adam_m: HashMap<String, DeviceBuffer>,
    adam_v: HashMap<String, DeviceBuffer>,
    params: Vec<(String, usize)>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,

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

    // Cached dispatch graphs. The forward/backward/AdamW step lists are
    // structurally identical on every iteration, so we build them once and
    // reuse them. This avoids allocating a fresh `params` uniform buffer and a
    // fresh bind group per dispatch (~210/step), which otherwise exhausts the
    // GPU memory aperture after a few thousand steps and triggers a device
    // reset. The bind groups keep their referenced buffers alive; only the
    // *contents* of `tokens`/`targets` (set_batch) and the AdamW uniforms (per
    // step) change in place via `write_buffer`.
    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    adamw_steps: Vec<Step>,
    adamw_uniforms: Vec<DeviceBuffer>,
}

impl Trainer {
    pub fn new(cfg: Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Trainer {
        // One shared accelerator (wgpu or native CPU, chosen at runtime). All the
        // device-init + dispatch plumbing that used to live here now lives in
        // `gpu_core`, shared with the GPT and PID models.
        let gpu = Gpu::new(PIPELINES);

        let c = cfg.clone();
        let params = param_list(&c);

        // weights / grads / adam state (grads zeroed each step; adam state zeroed
        // here since wgpu storage buffers are created uninitialised).
        let mut weights = HashMap::new();
        let mut grads = HashMap::new();
        let mut adam_m = HashMap::new();
        let mut adam_v = HashMap::new();
        for (name, numel) in &params {
            let data = init.get(name).unwrap_or_else(|| panic!("missing init weight {name}"));
            assert_eq!(data.len(), *numel, "size mismatch for {name}");
            weights.insert(name.clone(), gpu.storage_init(name, data));
            grads.insert(name.clone(), gpu.storage(*numel as u64));
            adam_m.insert(name.clone(), gpu.storage(*numel as u64));
            adam_v.insert(name.clone(), gpu.storage(*numel as u64));
        }
        for (name, numel) in &params {
            let z = vec![0u32; *numel];
            gpu.write(&adam_m[name], &z);
            gpu.write(&adam_v[name], &z);
        }

        let n = (b * t) as u64;
        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let e = c.n_experts as u64;
        let bht2 = (b * c.n_heads * t * t) as u64;

        let tokens = gpu.storage(n);
        let targets = gpu.storage(n);

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
            params,
            weights,
            grads,
            adam_m,
            adam_v,
            tokens,
            targets,
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
            adamw_steps: Vec::new(),
            adamw_uniforms: Vec::new(),
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
        let (steps, unis) = self.build_adamw();
        self.adamw_steps = steps;
        self.adamw_uniforms = unis;
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.weights.get(name).unwrap_or_else(|| panic!("no weight {name}"))
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.grads.get(name).unwrap()
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, bytemuck::cast_slice(x));
        self.gpu.write(&self.targets, bytemuck::cast_slice(y));
    }

    /// Run the (cached) forward pass and return mean cross-entropy. The
    /// `read` also polls the device, which is what lets wgpu reclaim the
    /// transient staging buffers this step allocated.
    pub fn forward(&self) -> f32 {
        let n = self.b * self.t;
        self.gpu.submit(&[], &self.fwd_steps);
        let losses = self.gpu.read(&self.ce_buf, n as usize);
        losses.iter().sum::<f32>() / n as f32
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
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, c.vocab], n));

        s
    }

    /// Run the (cached) backward pass: zero grads, then accumulate every
    /// parameter gradient in a single pass.
    pub fn backward(&self) {
        let clears: Vec<&DeviceBuffer> = self.params.iter().map(|(name, _)| self.g(name)).collect();
        self.gpu.submit(&clears, &self.bwd_steps);
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
        s.push(self.gpu.step(CE_GRAD, &[&self.logits, &self.targets, &self.d_logits], &[n, c.vocab], n * c.vocab));
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

    /// Build the AdamW dispatch graph and its per-parameter uniform buffers.
    /// The bind groups are constant, but the uniform *contents* (lr, bias
    /// corrections) change every step, so each param gets a persistent writable
    /// uniform buffer updated in `adamw_step` via `write_buffer`.
    fn build_adamw(&self) -> (Vec<Step>, Vec<DeviceBuffer>) {
        let mut steps = Vec::new();
        let mut unis = Vec::new();
        for (name, numel) in &self.params {
            let ubuf = self.gpu.uniform_dynamic(9);
            let st = self.gpu.step_buf(
                ADAMW,
                &ubuf,
                &[self.w(name), self.g(name), &self.adam_m[name], &self.adam_v[name]],
                *numel as u32,
            );
            steps.push(st);
            unis.push(ubuf);
        }
        (steps, unis)
    }

    /// One AdamW step. `t` is the (1-based) step index for bias correction.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, beta1: f32, beta2: f32, eps: f32) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        for (i, (_, numel)) in self.params.iter().enumerate() {
            let data = [*numel as u32, 0, f(lr), f(beta1), f(beta2), f(eps), f(wd), f(bc1), f(bc2)];
            self.gpu.write(&self.adamw_uniforms[i], bytemuck::cast_slice(&data));
        }
        self.gpu.submit(&[], &self.adamw_steps);
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let numel = self.params.iter().find(|(n, _)| n == name).unwrap().1;
        self.gpu.read(self.w(name), numel)
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let numel = self.params.iter().find(|(n, _)| n == name).unwrap().1;
        self.gpu.read(self.g(name), numel)
    }

    /// Names of all trainable parameters.
    pub fn param_names(&self) -> Vec<String> {
        self.params.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Zero one parameter's gradient buffer (host-driven).
    pub fn zero_grad(&self, name: &str) {
        let numel = self.params.iter().find(|(n, _)| n == name).unwrap().1;
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

    /// Save the current weights (incl. the tied `lm_head`) to a checkpoint.
    pub fn save(&self, path: &str) {
        save_weights(self, &self.cfg, path);
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
// CLI: `validate <ref.bin>`  and  `train [flags]`
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

fn max_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    // returns (max abs error, max relative error over entries with |b|>1e-3)
    let mut mae = 0.0f32;
    let mut mre = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let ae = (x - y).abs();
        mae = mae.max(ae);
        if y.abs() > 1e-3 {
            mre = mre.max(ae / y.abs());
        }
    }
    (mae, mre)
}

pub fn validate(path: &str) {
    let bytes = std::fs::read(path).expect("cannot read ref file");
    let jlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&bytes[8..8 + jlen]).unwrap(),
    )
    .unwrap();
    let data = &bytes[8 + jlen..];
    let cfg = cfg_from_json(&json["config"]);
    let bsz = json["B"].as_u64().unwrap() as u32;
    let t = json["T"].as_u64().unwrap() as u32;

    let read_tensor = |offset: usize, numel: usize| -> Vec<f32> {
        data[offset * 4..(offset + numel) * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };

    let mut init = HashMap::new();
    let mut grad = HashMap::new();
    let mut updated = HashMap::new();
    let mut batch_x = Vec::new();
    let mut batch_y = Vec::new();
    for t in json["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap().to_string();
        let role = t["role"].as_str().unwrap();
        let vals = read_tensor(t["offset"].as_u64().unwrap() as usize, t["numel"].as_u64().unwrap() as usize);
        match role {
            "init" => { init.insert(name, vals); }
            "grad" => { grad.insert(name, vals); }
            "updated" => { updated.insert(name, vals); }
            "data" if name == "batch_x" => batch_x = vals,
            "data" if name == "batch_y" => batch_y = vals,
            _ => {}
        }
    }

    let opt = &json["opt"];
    let (lr, wd) = (opt["lr"].as_f64().unwrap() as f32, opt["weight_decay"].as_f64().unwrap() as f32);
    let (beta1, beta2, eps) = (
        opt["beta1"].as_f64().unwrap() as f32,
        opt["beta2"].as_f64().unwrap() as f32,
        opt["eps"].as_f64().unwrap() as f32,
    );

    let trainer = Trainer::new(cfg, bsz, t, &init);
    let xs: Vec<u32> = batch_x.iter().map(|v| *v as u32).collect();
    let ys: Vec<u32> = batch_y.iter().map(|v| *v as u32).collect();
    trainer.set_batch(&xs, &ys);

    let ce = trainer.forward();
    let total = json["losses"]["total"].as_f64().unwrap() as f32;
    let moe = json["losses"]["moe"].as_f64().unwrap() as f32;
    println!("loss: rust_ce={:.6}  py_ce(total-moe)={:.6}  (py total={:.6})", ce, total - moe, total);

    trainer.backward();
    println!("\n== gradient check (Rust vs PyTorch autograd) ==");
    let mut g_mae = 0.0f32;
    let mut g_mre = 0.0f32;
    let mut worst = String::new();
    for (name, _) in trainer.params.iter() {
        let r = trainer.read_grad(name);
        let p = &grad[name];
        let (mae, mre) = max_err(&r, p);
        if mae > g_mae { g_mae = mae; worst = name.clone(); }
        g_mre = g_mre.max(mre);
    }
    println!("max abs grad error = {:.3e} (worst: {})", g_mae, worst);
    println!("max rel grad error = {:.3e}", g_mre);

    trainer.adamw_step(1, lr, wd, beta1, beta2, eps);
    println!("\n== weight check after one AdamW step ==");
    let mut w_mae = 0.0f32;
    let mut w_mre = 0.0f32;
    for (name, _) in trainer.params.iter() {
        let r = trainer.read_weight(name);
        let (mae, mre) = max_err(&r, &updated[name]);
        w_mae = w_mae.max(mae);
        w_mre = w_mre.max(mre);
    }
    println!("max abs weight error = {:.3e}", w_mae);
    println!("max rel weight error = {:.3e}", w_mre);

    let ok = g_mae < 2e-3 && w_mae < 2e-4;
    println!("\n{}", if ok { "VALIDATION PASSED" } else { "VALIDATION FAILED" });
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
        vocab: 64, block_size: 64, n_layers: 2, d_model: 64, n_heads: 4,
        n_experts: 4, top_k: 2, d_ff: 128, aux_coef: 0.01, z_coef: 1e-4,
    };
    assert!(args.t <= cfg.block_size, "T must be <= block_size");

    let corpus = make_corpus(20_000, cfg.vocab, 123);
    // Resume from the existing checkpoint if one is already at `out`; otherwise
    // start from a fresh random init. (Weights are warm-started; AdamW moments
    // restart at zero — they are not persisted in the checkpoint.)
    let init = if std::path::Path::new(&args.out).exists() {
        println!("resuming from existing checkpoint {}", args.out);
        checkpoint::load(&args.out).by_role("")
    } else {
        init_weights(&cfg, args.seed)
    };
    let trainer = Trainer::new(cfg.clone(), args.b, args.t, &init);

    let mut rng = args.seed.max(1) ^ 0x9E3779B97F4A7C15;
    let tt = args.t as usize;
    for step in 1..=args.steps {
        // sample B random windows
        let mut xs = Vec::with_capacity((args.b * args.t) as usize);
        let mut ys = Vec::with_capacity((args.b * args.t) as usize);
        for _ in 0..args.b {
            let start = (xorshift(&mut rng) as usize) % (corpus.len() - tt - 1);
            xs.extend_from_slice(&corpus[start..start + tt]);
            ys.extend_from_slice(&corpus[start + 1..start + 1 + tt]);
        }
        trainer.set_batch(&xs, &ys);
        let loss = trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, args.lr, args.wd, 0.9, 0.95, 1e-8);
        if step == 1 || step % 50 == 0 || step == args.steps {
            println!("step {:5} | loss {:.4}", step, loss);
        }
    }

    save_weights(&trainer, &cfg, &args.out);
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
        trainer.backward();
        trainer.freeze_grads_except_expert(args.expert); // freeze the shared backbone
        trainer.adamw_step(step, args.lr, 0.0, 0.9, 0.95, 1e-8); // wd=0 keeps frozen params fixed
        if step == 1 || step % 50 == 0 || step == args.steps {
            println!("expert {} | step {:5} | loss {:.4}", args.expert, step, loss);
        }
    }
    trainer.save(&args.out);
    println!("saved expert-{} checkpoint {}", args.expert, args.out);
}

fn save_weights(trainer: &Trainer, cfg: &Config, path: &str) {
    use std::io::Write;
    let mut tensors = Vec::new();
    let mut blob: Vec<f32> = Vec::new();
    let add = |name: &str, shape: Vec<u64>, data: &[f32], tensors: &mut Vec<serde_json::Value>, blob: &mut Vec<f32>| {
        tensors.push(serde_json::json!({
            "name": name, "shape": shape, "offset": blob.len(), "numel": data.len()
        }));
        blob.extend_from_slice(data);
    };
    let d = cfg.d_model as u64;
    for (name, _) in trainer.params.iter() {
        let data = trainer.read_weight(name);
        add(name, vec![data.len() as u64], &data, &mut tensors, &mut blob);
    }
    // tied head expected by the inference loader
    let emb = trainer.read_weight("token_emb.weight");
    add("lm_head.weight", vec![cfg.vocab as u64, d], &emb, &mut tensors, &mut blob);

    let header = serde_json::json!({
        "config": {
            "vocab_size": cfg.vocab, "block_size": cfg.block_size, "n_layers": cfg.n_layers,
            "d_model": cfg.d_model, "n_heads": cfg.n_heads, "n_experts": cfg.n_experts,
            "top_k": cfg.top_k, "d_ff": cfg.d_ff
        },
        "tensors": tensors
    });
    let hbytes = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hbytes).unwrap();
    for v in &blob {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
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
        let tr = Trainer::new(cfg.clone(), 2, 8, &init);
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
            tr.backward();
            tr.freeze_grads_except_expert(1); // freeze everything but expert 1
            tr.adamw_step(step, 1e-2, 0.0, 0.9, 0.95, 1e-8); // wd=0 -> frozen params fixed
        }

        let backbone1 = tr.read_weight("token_emb.weight");
        let e0_after = tr.read_weight("blocks.0.moe.experts.0.w_gate.weight");
        let e1_after = tr.read_weight("blocks.0.moe.experts.1.w_gate.weight");

        assert_eq!(backbone0, backbone1, "backbone moved under train-scope");
        assert_eq!(e0_before, e0_after, "non-target expert 0 moved");
        assert!(e1_before != e1_after, "target expert 1 did not train");
    }
}
