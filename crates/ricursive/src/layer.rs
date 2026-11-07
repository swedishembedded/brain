// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One Ripple decoder layer as a recorded kernel graph, f32 end to end.
//!
//! Swedish Embedded AB implements solutions for porting hybrid local/global
//! attention decoders to portable GPU and CPU backends for its clients. If
//! your team needs expertise in kernel graphs for gated attention and clipped
//! feed-forward layers then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! ```text
//! xn = input_layernorm(x)
//! q  = q_norm(q_proj(xn))         k = k_norm(k_proj(xn))         (per head)
//! v  = v_proj(lerp(x0, xn, sigmoid(value_res_lambda)))           (use_valres)
//! sliding: q, k = rope(q), rope(k)       full: no rotation
//! a  = causal_gqa_attention(q, k, v)     sliding: window keys only
//! a  = a * sigmoid(g_proj(xn))                                   (gated attention)
//! x  = x + post_attention_layernorm(o_proj(a))
//! m  = pre_feedforward_layernorm(x)
//! x  = x + post_feedforward_layernorm(down(silu(min(gate(m), c)) * clamp(up(m), -c, c)))
//! ```
//!
//! `x0` is the scaled embedding output, shared by every layer as the
//! value-residual source. Both layer types dispatch the same
//! `gqa_scores_win` / `attn_softmax` / `gqa_apply` trio: with a window of at
//! least the sequence length the windowed kernel is plain causal attention,
//! which is the global layer. The kernel's built-in `1/sqrt(head_dim)` score
//! scale is the reference's SDPA scale, so no fold is needed.
//!
//! The layer is built from explicit pieces ([`Layer::on`] for weights,
//! [`Layer::record`] for the graph) so an engine that owns the KV cache can
//! call it per layer with a layer type and window of its own choosing.

use checkpoint::TensorSource;
use gpu_core::{f, DeviceBuffer, Gpu, Step, Transient};
use paramstore::upload::Uploader;

use crate::config::{LayerType, RicursiveConfig};
use crate::import::layer_tensor;
use model::rope_table::DeviceRope;

pub(crate) const K_MATMUL: usize = 0;
pub(crate) const K_RMSNORM: usize = 1;
const K_ROPE2D: usize = 2;
const K_GQA_SCORES_WIN: usize = 3;
const K_ATTN_SOFTMAX: usize = 4;
const K_GQA_APPLY: usize = 5;
const K_SIGMOID: usize = 6;
const K_MUL: usize = 7;
const K_ADD2: usize = 8;
const K_SILU_MUL: usize = 9;
const K_CLAMP_RANGE: usize = 10;
const K_MIX: usize = 11;

/// Every kernel the forward dispatches, in the order of the `K_*` indices.
pub const KERNELS: [(&str, &str); 12] = [
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope2d", kernels::ROPE2D),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("sigmoid", kernels::SIGMOID),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("silu_mul", kernels::SILU_MUL),
    ("clamp_range", kernels::CLAMP_RANGE),
    // `y = a[row] * x + b[row] * f` with per-row coefficients: the value
    // residual's lerp, `x0 + s * (xn - x0) = (1 - s) * x0 + s * xn`.
    ("edm_mix", kernels::EDM_MIX),
];

/// Read all `n` elements of a small tensor (a gain vector or a scalar).
pub(crate) fn read_small(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(n);
    if !src.with_tensor_chunks(name, 1 << 16, &mut |_, chunk| out.extend_from_slice(chunk)) {
        return Err(format!("ricursive: source has no tensor {name}"));
    }
    if out.len() != n {
        return Err(format!("ricursive: {name} has {} elements, expected {n}", out.len()));
    }
    Ok(out)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Device-resident weights of one layer.
struct Weights {
    input_ln: DeviceBuffer,
    post_attn_ln: DeviceBuffer,
    pre_ff_ln: DeviceBuffer,
    post_ff_ln: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    o: DeviceBuffer,
    g: Option<DeviceBuffer>,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
}

/// The layer's outputs on the device.
pub struct LayerBuffers {
    /// The residual stream after the layer, `[t, hidden]`.
    pub hidden: DeviceBuffer,
    /// `o_proj`'s output before `post_attention_layernorm`, `[t, hidden]`.
    pub attn_out: DeviceBuffer,
}

/// The layer's outputs on the host.
pub struct LayerOutput {
    pub hidden: Vec<f32>,
    pub attn_out: Vec<f32>,
}

/// One decoder layer with its weights resident.
pub struct Layer {
    gpu: Gpu,
    cfg: RicursiveConfig,
    kind: LayerType,
    w: Weights,
    /// `sigmoid(value_res_lambda)`, or `None` when the model has no value residual.
    value_mix: Option<f32>,
}

impl Layer {
    /// Stream layer `layer` of `src` (canonical names, see [`crate::import`])
    /// onto `gpu`. The result is a [`Transient`]: dropping it reclaims the
    /// layer's device memory before the next layer is uploaded.
    pub fn on<'g>(gpu: &'g Gpu, cfg: &RicursiveConfig, src: &dyn TensorSource, layer: usize) -> Result<Transient<'g, Layer>, String> {
        let g = gpu.share();
        let mut up = Uploader::new(&g);
        let h = cfg.hidden_size;
        let mut load = |suffix: &str, numel: usize| -> Result<DeviceBuffer, String> {
            let name = layer_tensor(layer, suffix);
            let buf = up.tensor(src, &name, numel)?;
            src.advise_drop(&name);
            Ok(buf)
        };
        let w = Weights {
            input_ln: load("input_layernorm.weight", h)?,
            post_attn_ln: load("post_attention_layernorm.weight", h)?,
            pre_ff_ln: load("pre_feedforward_layernorm.weight", h)?,
            post_ff_ln: load("post_feedforward_layernorm.weight", h)?,
            q: load("self_attn.q_proj.weight", cfg.q_dim() * h)?,
            k: load("self_attn.k_proj.weight", cfg.kv_dim() * h)?,
            v: load("self_attn.v_proj.weight", cfg.kv_dim() * h)?,
            o: load("self_attn.o_proj.weight", h * cfg.q_dim())?,
            g: if cfg.use_gated_attention { Some(load("self_attn.g_proj.weight", cfg.q_dim() * h)?) } else { None },
            q_norm: load("self_attn.q_norm.weight", cfg.head_dim)?,
            k_norm: load("self_attn.k_norm.weight", cfg.head_dim)?,
            gate: load("mlp.gate_proj.weight", cfg.intermediate_size * h)?,
            up: load("mlp.up_proj.weight", cfg.intermediate_size * h)?,
            down: load("mlp.down_proj.weight", h * cfg.intermediate_size)?,
        };
        let value_mix = if cfg.use_valres { Some(sigmoid(read_small(src, &layer_tensor(layer, "self_attn.value_res_lambda"), 1)?[0])) } else { None };
        Ok(Transient::on(gpu, Layer { gpu: g, cfg: cfg.clone(), kind: cfg.layer_type(layer), w, value_mix }))
    }

    pub fn kind(&self) -> LayerType {
        self.kind
    }

    /// Keys a query can see: the configured window on a sliding layer, the
    /// whole prefix (`t`) on a global one.
    fn window(&self, t: u32) -> u32 {
        match self.kind {
            LayerType::Sliding => (self.cfg.sliding_window as u32).min(t),
            LayerType::Full => t,
        }
    }

    /// `out[m, n] = x[m, k] @ w[n, k]^T`.
    fn linear(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, m: u32, k: u32, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(m) * u64::from(n));
        s.push(self.gpu.step(K_MATMUL, &[x, w, &out], &[m, k, n], m * n));
        out
    }

    fn rmsnorm(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &DeviceBuffer, dim: u32, rows: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(dim) * u64::from(rows));
        s.push(self.gpu.step(K_RMSNORM, &[x, gain, &out], &[dim, rows, f(self.cfg.rms_norm_eps)], rows));
        out
    }

    fn binary(&self, s: &mut Vec<Step>, kernel: usize, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(n));
        s.push(self.gpu.step(kernel, &[a, b, &out], &[n], n));
        out
    }

    fn clamp(&self, s: &mut Vec<Step>, x: &DeviceBuffer, lo: f32, hi: f32, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(n));
        s.push(self.gpu.step(K_CLAMP_RANGE, &[x, &out], &[n, f(lo), f(hi)], n));
        out
    }

    /// Record the layer over `t` tokens starting at position 0.
    ///
    /// `x` is the residual stream entering the layer and `x0` the value
    /// residual source, both `[t, hidden]`. `rope` is required for a sliding
    /// layer and ignored for a global one.
    pub fn record(&self, s: &mut Vec<Step>, x: &DeviceBuffer, x0: &DeviceBuffer, rope: Option<&DeviceRope>, t: u32) -> LayerBuffers {
        let c = &self.cfg;
        let (hidden, heads, kv_heads, hd) = (c.hidden_size as u32, c.num_attention_heads as u32, c.num_key_value_heads as u32, c.head_dim as u32);
        let (q_dim, kv_dim) = (c.q_dim() as u32, c.kv_dim() as u32);
        let w = &self.w;

        let xn = self.rmsnorm(s, x, &w.input_ln, hidden, t);
        let q_pre = self.linear(s, &xn, &w.q, t, hidden, q_dim);
        let k_pre = self.linear(s, &xn, &w.k, t, hidden, kv_dim);
        let v_in = match self.value_mix {
            Some(mix) => {
                // (1 - mix) * x0 + mix * xn, the same coefficients on every row.
                let coeffs: Vec<f32> = (0..t).flat_map(|_| [1.0 - mix, mix]).collect();
                let ab = self.gpu.storage(coeffs.len() as u64);
                self.gpu.write_f32(&ab, &coeffs);
                let lerped = self.gpu.storage(u64::from(t) * u64::from(hidden));
                s.push(self.gpu.step(K_MIX, &[x0, &xn, &ab, &lerped], &[t * hidden, hidden], t * hidden));
                lerped
            }
            None => xn.clone(),
        };
        let v = self.linear(s, &v_in, &w.v, t, hidden, kv_dim);

        let q = self.rmsnorm(s, &q_pre, &w.q_norm, hd, t * heads);
        let k = self.rmsnorm(s, &k_pre, &w.k_norm, hd, t * kv_heads);
        if self.kind == LayerType::Sliding {
            let rope = rope.expect("a sliding layer needs its RoPE table");
            s.push(rope.apply(&self.gpu, K_ROPE2D, &q, t, heads, q_dim));
            s.push(rope.apply(&self.gpu, K_ROPE2D, &k, t, kv_heads, kv_dim));
        }

        let group = heads / kv_heads;
        let scores = self.gpu.storage(u64::from(heads) * u64::from(t) * u64::from(t));
        let probs = self.gpu.storage(u64::from(heads) * u64::from(t) * u64::from(t));
        let ctx = self.gpu.storage(u64::from(t) * u64::from(q_dim));
        s.push(self.gpu.step(K_GQA_SCORES_WIN, &[&q, &k, &scores], &[1, heads, kv_heads, t, hd, group, self.window(t)], heads * t * t));
        s.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, t], heads * t));
        s.push(self.gpu.step(K_GQA_APPLY, &[&probs, &v, &ctx], &[1, heads, kv_heads, t, hd, group], heads * t * hd));

        let ctx = match &w.g {
            Some(wg) => {
                let gate = self.linear(s, &xn, wg, t, hidden, q_dim);
                let gate = {
                    let out = self.gpu.storage(u64::from(t) * u64::from(q_dim));
                    s.push(self.gpu.step(K_SIGMOID, &[&gate, &out], &[t * q_dim], t * q_dim));
                    out
                };
                self.binary(s, K_MUL, &ctx, &gate, t * q_dim)
            }
            None => ctx,
        };
        let attn_out = self.linear(s, &ctx, &w.o, t, q_dim, hidden);
        let attn_normed = self.rmsnorm(s, &attn_out, &w.post_attn_ln, hidden, t);
        let x1 = self.binary(s, K_ADD2, x, &attn_normed, t * hidden);

        let ff_dim = c.intermediate_size as u32;
        let m = self.rmsnorm(s, &x1, &w.pre_ff_ln, hidden, t);
        let mut gate = self.linear(s, &m, &w.gate, t, hidden, ff_dim);
        let mut up = self.linear(s, &m, &w.up, t, hidden, ff_dim);
        if let Some(limit) = c.ffn_clamp_threshold {
            gate = self.clamp(s, &gate, -f32::MAX, limit, t * ff_dim);
            up = self.clamp(s, &up, -limit, limit, t * ff_dim);
        }
        let act = self.binary(s, K_SILU_MUL, &gate, &up, t * ff_dim);
        let mlp = self.linear(s, &act, &w.down, t, ff_dim, hidden);
        let mlp_normed = self.rmsnorm(s, &mlp, &w.post_ff_ln, hidden, t);
        let out = self.binary(s, K_ADD2, &x1, &mlp_normed, t * hidden);
        LayerBuffers { hidden: out, attn_out }
    }

    /// Run the layer on host activations: `x` and `x0` are `[t, hidden]`.
    pub fn forward(&self, x: &[f32], x0: &[f32], rope: Option<&DeviceRope>, t: usize) -> LayerOutput {
        let n = t * self.cfg.hidden_size;
        assert_eq!((x.len(), x0.len()), (n, n), "activations must be [t, hidden]");
        let x_buf = self.gpu.storage(n as u64);
        self.gpu.write_f32(&x_buf, x);
        let x0_buf = self.gpu.storage(n as u64);
        self.gpu.write_f32(&x0_buf, x0);
        let mut steps = Vec::new();
        let out = self.record(&mut steps, &x_buf, &x0_buf, rope, t as u32);
        self.gpu.submit(&[], &steps);
        LayerOutput { hidden: self.gpu.read(&out.hidden, n), attn_out: self.gpu.read(&out.attn_out, n) }
    }
}
