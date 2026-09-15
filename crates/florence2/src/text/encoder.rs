// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Florence2Encoder`: learned absolute position embedding + LayerNorm,
//! then `encoder_layers` post-LN bidirectional-self-attention + FFN blocks
//! (`Florence2EncoderLayer`, `activation_function="gelu"` - exact erf GELU,
//! matched by the existing `gelu_erf` kernel, not the tanh approximation).
//! Runs over the FULL concatenated `[vision_tokens; text_prompt_tokens]`
//! sequence - no padding mask needed for this crate's single-example,
//! unpadded `ground` use case (see `crate::text::lm`'s module doc), so the
//! reference's `attention_mask` plumbing is simply never built.
//!
//! Every `add2`/`layernorm_fwd`/`gelu_erf` dispatch below writes to a
//! buffer DISTINCT from every buffer it reads: `Gpu::step`'s
//! `assert_no_output_alias` rejects a buffer bound as both an input and the
//! output in one dispatch (a real wgpu `STORAGE_READ_WRITE` exclusivity
//! rule, not just a CPU-backend nicety - caught at test time, not
//! silently). The one exception is `add_inplace` (`out += a`, a genuinely
//! single read_write binding plus one plain read-only operand), used for
//! the position-embedding add specifically because it lets that step avoid
//! a THIRD scratch buffer.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{layernorm_fwd, LayerNormIds};
use model::vit::row_index_buffer;

use super::attn::{BartAttn, BartAttnKernelIds};

pub struct EncoderKernelIds {
    pub attn: BartAttnKernelIds,
    pub embed: usize,
    pub add2: usize,
    pub add_inplace: usize,
    pub layernorm: usize,
    pub matmul_rows: usize,
    pub bias_add: usize,
    pub gelu_erf: usize,
}

impl EncoderKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> EncoderKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        EncoderKernelIds {
            attn: BartAttnKernelIds::resolve(pipelines),
            embed: k("embed"),
            add2: k("add2"),
            add_inplace: k("add_inplace"),
            layernorm: k("layernorm"),
            matmul_rows: k("matmul_rows"),
            bias_add: k("bias_add"),
            gelu_erf: k("gelu_erf"),
        }
    }
}

struct EncoderLayer {
    self_attn: BartAttn,
    sum1: DeviceBuffer,
    normed1: DeviceBuffer,
    ff_hidden: DeviceBuffer,
    ff_act: DeviceBuffer,
    ff_out: DeviceBuffer,
    sum2: DeviceBuffer,
    normed2: DeviceBuffer,
}

impl EncoderLayer {
    fn new(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, t: u32) -> EncoderLayer {
        EncoderLayer {
            self_attn: BartAttn::new(gpu, heads, d_model, false, t, t),
            sum1: gpu.storage((t * d_model) as u64),
            normed1: gpu.storage((t * d_model) as u64),
            ff_hidden: gpu.storage((t * ffn_dim) as u64),
            ff_act: gpu.storage((t * ffn_dim) as u64),
            ff_out: gpu.storage((t * d_model) as u64),
            sum2: gpu.storage((t * d_model) as u64),
            normed2: gpu.storage((t * d_model) as u64),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward<'a>(&'a self, gpu: &Gpu, k: &EncoderKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &'a DeviceBuffer, t: u32, d_model: u32, ffn_dim: u32, eps: f32) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let attn_out = self.self_attn.forward(gpu, &k.attn, ps, &format!("{prefix}.self_attn"), x, x, t, t);
        let ln1_w = ps.w(&format!("{prefix}.self_attn_layer_norm.weight"));
        let ln1_b = ps.w(&format!("{prefix}.self_attn_layer_norm.bias"));
        let s1 = vec![
            gpu.step(k.add2, &[x, attn_out, &self.sum1], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum1, ln1_w, ln1_b, &self.normed1, d_model, t, eps),
        ];
        gpu.submit(&[], &s1);

        let fc1_w = ps.w(&format!("{prefix}.fc1.weight"));
        let fc1_b = ps.w(&format!("{prefix}.fc1.bias"));
        let fc2_w = ps.w(&format!("{prefix}.fc2.weight"));
        let fc2_b = ps.w(&format!("{prefix}.fc2.bias"));
        let ln2_w = ps.w(&format!("{prefix}.final_layer_norm.weight"));
        let ln2_b = ps.w(&format!("{prefix}.final_layer_norm.bias"));
        let s2 = vec![
            gpu.step(k.matmul_rows, &[&self.normed1, fc1_w, &self.ff_hidden], &[t, d_model, ffn_dim], t.div_ceil(8) * ffn_dim),
            gpu.step(k.bias_add, &[&self.ff_hidden, fc1_b], &[t, ffn_dim], t * ffn_dim),
            gpu.step(k.gelu_erf, &[&self.ff_hidden, &self.ff_act], &[t * ffn_dim], t * ffn_dim),
            gpu.step(k.matmul_rows, &[&self.ff_act, fc2_w, &self.ff_out], &[t, ffn_dim, d_model], t.div_ceil(8) * d_model),
            gpu.step(k.bias_add, &[&self.ff_out, fc2_b], &[t, d_model], t * d_model),
            gpu.step(k.add2, &[&self.normed1, &self.ff_out, &self.sum2], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum2, ln2_w, ln2_b, &self.normed2, d_model, t, eps),
        ];
        gpu.submit(&[], &s2);
        &self.normed2
    }
}

pub struct Encoder {
    t: u32,
    d_model: u32,
    ffn_dim: u32,
    eps: f32,
    pos_idx: DeviceBuffer,
    pos_embedded: DeviceBuffer,
    embedded: DeviceBuffer,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    /// `t`: this query's FIXED `vision_tokens + text_prompt_tokens` length -
    /// the encoder never grows, unlike the decoder (see `text::attn`'s
    /// module doc), so every buffer here is sized exactly, not to a ceiling.
    pub fn new(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, layers: u32, t: u32, eps: f32) -> Encoder {
        let positions: Vec<u32> = (0..t).map(|p| p + super::config::POSITION_OFFSET).collect();
        Encoder {
            t,
            d_model,
            ffn_dim,
            eps,
            pos_idx: row_index_buffer(gpu, "florence2_enc_pos", &positions),
            pos_embedded: gpu.storage((t * d_model) as u64),
            embedded: gpu.storage((t * d_model) as u64),
            layers: (0..layers).map(|_| EncoderLayer::new(gpu, heads, d_model, ffn_dim, t)).collect(),
        }
    }

    /// `inputs_embeds`: `[t,d_model]` = `[vision_tokens; text_prompt_embeds]`
    /// already concatenated by the caller (`text::lm`). Returns the encoder's
    /// `last_hidden_state`, `[t,d_model]`.
    pub fn forward<'a>(&'a self, gpu: &Gpu, k: &EncoderKernelIds, ps: &paramstore::ParamStore, inputs_embeds: &DeviceBuffer) -> &'a DeviceBuffer {
        let (t, d) = (self.t, self.d_model);
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let pos_table = ps.w("language_model.model.encoder.embed_positions.weight");
        let ln_w = ps.w("language_model.model.encoder.layernorm_embedding.weight");
        let ln_b = ps.w("language_model.model.encoder.layernorm_embedding.bias");

        let s0 = vec![
            gpu.step(k.embed, &[&self.pos_idx, pos_table, &self.pos_embedded], &[d, t], t * d),
            gpu.step(k.add_inplace, &[&self.pos_embedded, inputs_embeds], &[t * d], t * d),
            layernorm_fwd(gpu, &ln, &self.pos_embedded, ln_w, ln_b, &self.embedded, d, t, self.eps),
        ];
        gpu.submit(&[], &s0);

        let mut cur: &DeviceBuffer = &self.embedded;
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("language_model.model.encoder.layers.{i}");
            cur = layer.forward(gpu, k, ps, &prefix, cur, t, d, self.ffn_dim, self.eps);
        }
        cur
    }
}
