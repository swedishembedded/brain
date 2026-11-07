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
use gpu_core::{DeviceBuffer, Gpu, Step, Transient};
use model::rope_table::DeviceRope;
use paramstore::upload::Uploader;

use crate::config::{LayerType, RicursiveConfig};
use crate::import::layer_tensor;
use crate::ops::{self, FeedForward};
pub use crate::ops::KERNELS;
use crate::ops::{K_ADD2, K_GQA_APPLY, K_GQA_SCORES_WIN, K_MUL, K_ATTN_SOFTMAX};

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

/// Device-resident weights of one layer, in f32.
pub(crate) struct LayerWeights {
    pub input_ln: DeviceBuffer,
    pub post_attn_ln: DeviceBuffer,
    pub pre_ff_ln: DeviceBuffer,
    pub post_ff_ln: DeviceBuffer,
    pub q: DeviceBuffer,
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub o: DeviceBuffer,
    pub g: Option<DeviceBuffer>,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub gate: DeviceBuffer,
    pub up: DeviceBuffer,
    pub down: DeviceBuffer,
    /// `sigmoid(value_res_lambda)`, or `None` when the model has no value residual.
    pub value_mix: Option<f32>,
}

impl LayerWeights {
    /// Stream layer `layer` of `src` (canonical names, see [`crate::import`])
    /// through `up` onto its device.
    pub(crate) fn load(up: &mut Uploader<'_>, cfg: &RicursiveConfig, src: &dyn TensorSource, layer: usize) -> Result<LayerWeights, String> {
        let h = cfg.hidden_size;
        let mut load = |suffix: &str, numel: usize| -> Result<DeviceBuffer, String> {
            let name = layer_tensor(layer, suffix);
            let buf = up.tensor(src, &name, numel)?;
            src.advise_drop(&name);
            Ok(buf)
        };
        let w = LayerWeights {
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
            value_mix: None,
        };
        let value_mix = if cfg.use_valres { Some(sigmoid(read_small(src, &layer_tensor(layer, "self_attn.value_res_lambda"), 1)?[0])) } else { None };
        Ok(LayerWeights { value_mix, ..w })
    }

    /// The feed-forward sub-block's weights and flags.
    pub(crate) fn feed_forward<'a>(&'a self, cfg: &RicursiveConfig) -> FeedForward<'a> {
        FeedForward {
            pre_ln: &self.pre_ff_ln,
            post_ln: &self.post_ff_ln,
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
            hidden: cfg.hidden_size as u32,
            inter: cfg.intermediate_size as u32,
            eps: cfg.rms_norm_eps,
            clip: cfg.ffn_clamp_threshold,
        }
    }
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
    w: LayerWeights,
}

impl Layer {
    /// Stream layer `layer` of `src` onto `gpu`. The result is a
    /// [`Transient`]: dropping it reclaims the layer's device memory before the
    /// next layer is uploaded.
    pub fn on<'g>(gpu: &'g Gpu, cfg: &RicursiveConfig, src: &dyn TensorSource, layer: usize) -> Result<Transient<'g, Layer>, String> {
        let g = gpu.share();
        let w = LayerWeights::load(&mut Uploader::new(&g), cfg, src, layer)?;
        Ok(Transient::on(gpu, Layer { gpu: g, cfg: cfg.clone(), kind: cfg.layer_type(layer), w }))
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

    /// Record the layer over `t` tokens starting at position 0.
    ///
    /// `x` is the residual stream entering the layer and `x0` the value
    /// residual source, both `[t, hidden]`. `rope` is required for a sliding
    /// layer and ignored for a global one.
    pub fn record(&self, s: &mut Vec<Step>, x: &DeviceBuffer, x0: &DeviceBuffer, rope: Option<&DeviceRope>, t: u32) -> LayerBuffers {
        let c = &self.cfg;
        let (hidden, heads, kv_heads, hd) = (c.hidden_size as u32, c.num_attention_heads as u32, c.num_key_value_heads as u32, c.head_dim as u32);
        let (q_dim, kv_dim) = (c.q_dim() as u32, c.kv_dim() as u32);
        let (gpu, w) = (&self.gpu, &self.w);

        let xn = ops::rmsnorm(gpu, s, x, &w.input_ln, hidden, t, c.rms_norm_eps);
        let q_pre = ops::linear(gpu, s, &xn, &w.q, t, hidden, q_dim);
        let k_pre = ops::linear(gpu, s, &xn, &w.k, t, hidden, kv_dim);
        let v_in = match w.value_mix {
            Some(mix) => ops::value_mix(gpu, s, x0, &xn, mix, t, hidden),
            None => xn.clone(),
        };
        let v = ops::linear(gpu, s, &v_in, &w.v, t, hidden, kv_dim);

        let q = ops::rmsnorm(gpu, s, &q_pre, &w.q_norm, hd, t * heads, c.rms_norm_eps);
        let k = ops::rmsnorm(gpu, s, &k_pre, &w.k_norm, hd, t * kv_heads, c.rms_norm_eps);
        if self.kind == LayerType::Sliding {
            let rope = rope.expect("a sliding layer needs its RoPE table");
            s.push(rope.apply(gpu, ops::K_ROPE2D, &q, t, heads, q_dim));
            s.push(rope.apply(gpu, ops::K_ROPE2D, &k, t, kv_heads, kv_dim));
        }

        let group = heads / kv_heads;
        let scores = gpu.storage(u64::from(heads) * u64::from(t) * u64::from(t));
        let probs = gpu.storage(u64::from(heads) * u64::from(t) * u64::from(t));
        let ctx = gpu.storage(u64::from(t) * u64::from(q_dim));
        s.push(gpu.step(K_GQA_SCORES_WIN, &[&q, &k, &scores], &[1, heads, kv_heads, t, hd, group, self.window(t)], heads * t * t));
        s.push(gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, t], heads * t));
        s.push(gpu.step(K_GQA_APPLY, &[&probs, &v, &ctx], &[1, heads, kv_heads, t, hd, group], heads * t * hd));

        let ctx = match &w.g {
            Some(wg) => {
                let gate = ops::linear(gpu, s, &xn, wg, t, hidden, q_dim);
                let gate = ops::sigmoid(gpu, s, &gate, t * q_dim);
                ops::binary(gpu, s, K_MUL, &ctx, &gate, t * q_dim)
            }
            None => ctx,
        };
        let attn_out = ops::linear(gpu, s, &ctx, &w.o, t, q_dim, hidden);
        let attn_normed = ops::rmsnorm(gpu, s, &attn_out, &w.post_attn_ln, hidden, t, c.rms_norm_eps);
        let x1 = ops::binary(gpu, s, K_ADD2, x, &attn_normed, t * hidden);
        let out = w.feed_forward(c).record(gpu, s, &x1, t);
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
