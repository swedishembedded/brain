// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Florence2Decoder`: learned absolute position embedding + LayerNorm, then
//! `decoder_layers` post-LN blocks of (causal self-attention, cross-attention
//! over the FIXED encoder memory, FFN) - `Florence2DecoderLayer`. No KV
//! cache (see `text::attn`'s module doc): [`Decoder::forward`] takes the
//! CURRENT full token prefix `[0..t)` and recomputes every layer over it,
//! which is what lets buffers be sized once at `max_t` and every generation
//! step just pass a larger `t`.
//!
//! Every `add2`/`layernorm_fwd`/`gelu_erf` dispatch writes to a buffer
//! DISTINCT from every buffer it reads - see `text::encoder`'s module doc
//! for why (`Gpu::step`'s `assert_no_output_alias`, a real wgpu
//! `STORAGE_READ_WRITE` exclusivity rule).

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{layernorm_fwd, LayerNormIds};
use model::vit::row_index_buffer;

use super::attn::{BartAttn, BartAttnKernelIds};

pub struct DecoderKernelIds {
    pub attn: BartAttnKernelIds,
    pub embed: usize,
    pub add2: usize,
    pub add_inplace: usize,
    pub layernorm: usize,
    pub matmul_rows: usize,
    pub bias_add: usize,
    pub gelu_erf: usize,
}

impl DecoderKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> DecoderKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        DecoderKernelIds {
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

struct DecoderLayer {
    self_attn: BartAttn,
    cross_attn: BartAttn,
    sum1: DeviceBuffer,
    normed1: DeviceBuffer,
    sum2: DeviceBuffer,
    normed2: DeviceBuffer,
    ff_hidden: DeviceBuffer,
    ff_act: DeviceBuffer,
    ff_out: DeviceBuffer,
    sum3: DeviceBuffer,
    normed3: DeviceBuffer,
}

impl DecoderLayer {
    fn new(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, max_t: u32, t_enc: u32) -> DecoderLayer {
        DecoderLayer {
            self_attn: BartAttn::new(gpu, heads, d_model, true, max_t, max_t),
            cross_attn: BartAttn::new(gpu, heads, d_model, false, max_t, t_enc),
            sum1: gpu.storage((max_t * d_model) as u64),
            normed1: gpu.storage((max_t * d_model) as u64),
            sum2: gpu.storage((max_t * d_model) as u64),
            normed2: gpu.storage((max_t * d_model) as u64),
            ff_hidden: gpu.storage((max_t * ffn_dim) as u64),
            ff_act: gpu.storage((max_t * ffn_dim) as u64),
            ff_out: gpu.storage((max_t * d_model) as u64),
            sum3: gpu.storage((max_t * d_model) as u64),
            normed3: gpu.storage((max_t * d_model) as u64),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward<'a>(&'a self, gpu: &Gpu, k: &DecoderKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &'a DeviceBuffer, enc: &DeviceBuffer, t: u32, t_enc: u32, d_model: u32, ffn_dim: u32, eps: f32) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);

        let self_out = self.self_attn.forward(gpu, &k.attn, ps, &format!("{prefix}.self_attn"), x, x, t, t);
        let ln1_w = ps.w(&format!("{prefix}.self_attn_layer_norm.weight"));
        let ln1_b = ps.w(&format!("{prefix}.self_attn_layer_norm.bias"));
        let s1 = vec![
            gpu.step(k.add2, &[x, self_out, &self.sum1], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum1, ln1_w, ln1_b, &self.normed1, d_model, t, eps),
        ];
        gpu.submit(&[], &s1);

        let cross_out = self.cross_attn.forward(gpu, &k.attn, ps, &format!("{prefix}.encoder_attn"), &self.normed1, enc, t, t_enc);
        let ln2_w = ps.w(&format!("{prefix}.encoder_attn_layer_norm.weight"));
        let ln2_b = ps.w(&format!("{prefix}.encoder_attn_layer_norm.bias"));
        let s2 = vec![
            gpu.step(k.add2, &[&self.normed1, cross_out, &self.sum2], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum2, ln2_w, ln2_b, &self.normed2, d_model, t, eps),
        ];
        gpu.submit(&[], &s2);

        let fc1_w = ps.w(&format!("{prefix}.fc1.weight"));
        let fc1_b = ps.w(&format!("{prefix}.fc1.bias"));
        let fc2_w = ps.w(&format!("{prefix}.fc2.weight"));
        let fc2_b = ps.w(&format!("{prefix}.fc2.bias"));
        let ln3_w = ps.w(&format!("{prefix}.final_layer_norm.weight"));
        let ln3_b = ps.w(&format!("{prefix}.final_layer_norm.bias"));
        let s3 = vec![
            gpu.step(k.matmul_rows, &[&self.normed2, fc1_w, &self.ff_hidden], &[t, d_model, ffn_dim], t.div_ceil(8) * ffn_dim),
            gpu.step(k.bias_add, &[&self.ff_hidden, fc1_b], &[t, ffn_dim], t * ffn_dim),
            gpu.step(k.gelu_erf, &[&self.ff_hidden, &self.ff_act], &[t * ffn_dim], t * ffn_dim),
            gpu.step(k.matmul_rows, &[&self.ff_act, fc2_w, &self.ff_out], &[t, ffn_dim, d_model], t.div_ceil(8) * d_model),
            gpu.step(k.bias_add, &[&self.ff_out, fc2_b], &[t, d_model], t * d_model),
            gpu.step(k.add2, &[&self.normed2, &self.ff_out, &self.sum3], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum3, ln3_w, ln3_b, &self.normed3, d_model, t, eps),
        ];
        gpu.submit(&[], &s3);
        &self.normed3
    }
}

pub struct Decoder {
    max_t: u32,
    t_enc: u32,
    d_model: u32,
    ffn_dim: u32,
    eps: f32,
    pos_idx: DeviceBuffer,
    pos_embedded: DeviceBuffer,
    embedded: DeviceBuffer,
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    /// `max_t`: the ceiling this instance's scratch is sized for (see
    /// `text::attn`'s module doc - Florence-2 grounding outputs are short,
    /// so this can be a small constant). `t_enc`: the encoder's FIXED
    /// output length for this query.
    pub fn new(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, layers: u32, max_t: u32, t_enc: u32, eps: f32) -> Decoder {
        let positions: Vec<u32> = (0..max_t).map(|p| p + super::config::POSITION_OFFSET).collect();
        Decoder {
            max_t,
            t_enc,
            d_model,
            ffn_dim,
            eps,
            pos_idx: row_index_buffer(gpu, "florence2_dec_pos", &positions),
            pos_embedded: gpu.storage((max_t * d_model) as u64),
            embedded: gpu.storage((max_t * d_model) as u64),
            layers: (0..layers).map(|_| DecoderLayer::new(gpu, heads, d_model, ffn_dim, max_t, t_enc)).collect(),
        }
    }

    /// `decoder_inputs_embeds`: `[t,d_model]`, already-scaled token
    /// embeddings for positions `[0,t)` of the CURRENT prefix (`t <=
    /// max_t`). `enc`: the encoder's fixed `[t_enc,d_model]` output. Returns
    /// the decoder's final hidden state, `[t,d_model]`.
    pub fn forward<'a>(&'a self, gpu: &Gpu, k: &DecoderKernelIds, ps: &paramstore::ParamStore, decoder_inputs_embeds: &DeviceBuffer, enc: &DeviceBuffer, t: u32) -> &'a DeviceBuffer {
        assert!(t <= self.max_t, "florence2 decoder: t {t} exceeds max_t {}", self.max_t);
        let d = self.d_model;
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let pos_table = ps.w("language_model.model.decoder.embed_positions.weight");
        let ln_w = ps.w("language_model.model.decoder.layernorm_embedding.weight");
        let ln_b = ps.w("language_model.model.decoder.layernorm_embedding.bias");

        let s0 = vec![
            gpu.step(k.embed, &[&self.pos_idx, pos_table, &self.pos_embedded], &[d, t], t * d),
            gpu.step(k.add_inplace, &[&self.pos_embedded, decoder_inputs_embeds], &[t * d], t * d),
            layernorm_fwd(gpu, &ln, &self.pos_embedded, ln_w, ln_b, &self.embedded, d, t, self.eps),
        ];
        gpu.submit(&[], &s0);

        let mut cur: &DeviceBuffer = &self.embedded;
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("language_model.model.decoder.layers.{i}");
            cur = layer.forward(gpu, k, ps, &prefix, cur, enc, t, self.t_enc, d, self.ffn_dim, self.eps);
        }
        cur
    }
}
