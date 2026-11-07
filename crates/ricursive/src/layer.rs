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
use model::ops::{Ops, TierPolicy, Weight};
use model::rope_table::DeviceRope;
use paramstore::upload::Uploader;

use crate::config::{LayerType, RicursiveConfig};
use crate::import::layer_tensor;
use crate::ops::{self, FeedForward};
use crate::ops::{K_ADD2, K_ATTN_SOFTMAX, K_GQA_APPLY, K_GQA_SCORES_WIN, K_MUL};
use crate::weights::Loader;

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

/// Device-resident weights of one layer: the linears at their tier, the norms and gains in f32.
pub(crate) struct LayerWeights {
    pub input_ln: DeviceBuffer,
    pub post_attn_ln: DeviceBuffer,
    pub pre_ff_ln: DeviceBuffer,
    pub post_ff_ln: DeviceBuffer,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub g: Option<Weight>,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
    /// `sigmoid(value_res_lambda)`, or `None` when the model has no value residual.
    pub value_mix: Option<f32>,
}

impl LayerWeights {
    /// Stream layer `layer` of `src` (canonical names, see [`crate::import`])
    /// through `loader` onto its device.
    pub(crate) fn load(loader: &Loader<'_>, cfg: &RicursiveConfig, src: &dyn TensorSource, layer: usize) -> Result<LayerWeights, String> {
        let h = cfg.hidden_size;
        let mut up = Uploader::new(loader.ops().gpu());
        let mut vector = |suffix: &str, numel: usize| -> Result<DeviceBuffer, String> {
            let name = layer_tensor(layer, suffix);
            let buf = up.tensor(src, &name, numel)?;
            src.advise_drop(&name);
            Ok(buf)
        };
        let input_ln = vector("input_layernorm.weight", h)?;
        let post_attn_ln = vector("post_attention_layernorm.weight", h)?;
        let pre_ff_ln = vector("pre_feedforward_layernorm.weight", h)?;
        let post_ff_ln = vector("post_feedforward_layernorm.weight", h)?;
        let q_norm = vector("self_attn.q_norm.weight", cfg.head_dim)?;
        let k_norm = vector("self_attn.k_norm.weight", cfg.head_dim)?;
        let linear = |suffix: &str, n: usize, k: usize| -> Result<Weight, String> {
            let name = layer_tensor(layer, suffix);
            let w = loader.linear(&name, n, k)?;
            src.advise_drop(&name);
            Ok(w)
        };
        let value_mix = if cfg.use_valres { Some(sigmoid(read_small(src, &layer_tensor(layer, "self_attn.value_res_lambda"), 1)?[0])) } else { None };
        Ok(LayerWeights {
            input_ln,
            post_attn_ln,
            pre_ff_ln,
            post_ff_ln,
            q: linear("self_attn.q_proj.weight", cfg.q_dim(), h)?,
            k: linear("self_attn.k_proj.weight", cfg.kv_dim(), h)?,
            v: linear("self_attn.v_proj.weight", cfg.kv_dim(), h)?,
            o: linear("self_attn.o_proj.weight", h, cfg.q_dim())?,
            g: if cfg.use_gated_attention { Some(linear("self_attn.g_proj.weight", cfg.q_dim(), h)?) } else { None },
            q_norm,
            k_norm,
            gate: linear("mlp.gate_proj.weight", cfg.intermediate_size, h)?,
            up: linear("mlp.up_proj.weight", cfg.intermediate_size, h)?,
            down: linear("mlp.down_proj.weight", h, cfg.intermediate_size)?,
            value_mix,
        })
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
    ops: Ops,
    cfg: RicursiveConfig,
    kind: LayerType,
    w: LayerWeights,
}

impl Layer {
    /// Stream layer `layer` of `src` onto `gpu`, its linears at the tier `tiers` gives them. The result is a
    /// [`Transient`]: dropping it reclaims the layer's device memory before the
    /// next layer is uploaded. `gpu` must carry [`crate::pipelines`].
    pub fn on<'g>(gpu: &'g Gpu, cfg: &RicursiveConfig, src: &dyn TensorSource, layer: usize, tiers: &TierPolicy) -> Result<Transient<'g, Layer>, String> {
        let ops = Ops::new(gpu.share())?;
        let w = LayerWeights::load(&Loader::new(&ops, src, tiers), cfg, src, layer)?;
        Ok(Transient::on(gpu, Layer { ops, cfg: cfg.clone(), kind: cfg.layer_type(layer), w }))
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
        let (ops, gpu, w) = (&self.ops, self.ops.gpu(), &self.w);

        let xn = ops::rmsnorm(gpu, s, x, &w.input_ln, hidden, t, c.rms_norm_eps);
        let mut qk = ops::linears(ops, s, &xn, &[&w.q, &w.k], t);
        let (k_pre, q_pre) = (qk.pop().expect("k"), qk.pop().expect("q"));
        let v_in = match w.value_mix {
            Some(mix) => ops::value_mix(gpu, s, x0, &xn, mix, t, hidden),
            None => xn.clone(),
        };
        let v = ops::linear(ops, s, &v_in, &w.v, t);

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
                let gate = ops::linear(ops, s, &xn, wg, t);
                let gate = ops::sigmoid(gpu, s, &gate, t * q_dim);
                ops::binary(gpu, s, K_MUL, &ctx, &gate, t * q_dim)
            }
            None => ctx,
        };
        let attn_out = ops::linear(ops, s, &ctx, &w.o, t);
        let attn_normed = ops::rmsnorm(gpu, s, &attn_out, &w.post_attn_ln, hidden, t, c.rms_norm_eps);
        let x1 = ops::binary(gpu, s, K_ADD2, x, &attn_normed, t * hidden);
        let out = w.feed_forward(c).record(ops, s, &x1, t);
        LayerBuffers { hidden: out, attn_out }
    }

    /// Run the layer on host activations: `x` and `x0` are `[t, hidden]`.
    pub fn forward(&self, x: &[f32], x0: &[f32], rope: Option<&DeviceRope>, t: usize) -> LayerOutput {
        let n = t * self.cfg.hidden_size;
        assert_eq!((x.len(), x0.len()), (n, n), "activations must be [t, hidden]");
        let gpu = self.ops.gpu();
        let x_buf = gpu.storage(n as u64);
        gpu.write_f32(&x_buf, x);
        let x0_buf = gpu.storage(n as u64);
        gpu.write_f32(&x0_buf, x0);
        let mut steps = Vec::new();
        let out = self.record(&mut steps, &x_buf, &x0_buf, rope, t as u32);
        gpu.submit(&[], &steps);
        LayerOutput { hidden: gpu.read(&out.hidden, n), attn_out: gpu.read(&out.attn_out, n) }
    }
}
