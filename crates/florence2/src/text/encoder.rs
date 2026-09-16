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
use model::block::{layernorm_dx_bwd, layernorm_fwd, ln_stats_fwd, LayerNormIds};
use model::vit::row_index_buffer;

use super::attn::{BartAttn, BartAttnBwdIds, BartAttnBwdScratch, BartAttnKernelIds};
use super::lora::{lora_bwd, lora_fwd, trainable, LoraCtx};

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

pub struct EncoderBwdKernelIds {
    pub attn: BartAttnBwdIds,
    pub ln_stats: usize,
    pub layernorm_dx: usize,
    pub layernorm_dgamma: usize,
    pub layernorm_dbeta: usize,
    pub gelu_erf_bwd: usize,
    pub matmul_dx: usize,
    pub matmul_dw: usize,
    pub bias_grad: usize,
    pub add2: usize,
    pub emb_bwd: usize,
}

impl EncoderBwdKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> EncoderBwdKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        EncoderBwdKernelIds {
            attn: BartAttnBwdIds::resolve(pipelines),
            ln_stats: k("ln_stats"),
            layernorm_dx: k("layernorm_dx"),
            layernorm_dgamma: k("layernorm_dgamma"),
            layernorm_dbeta: k("layernorm_dbeta"),
            gelu_erf_bwd: k("gelu_erf_bwd"),
            matmul_dx: k("matmul_dx"),
            matmul_dw: k("matmul_dw"),
            bias_grad: k("bias_grad"),
            add2: k("add2"),
            emb_bwd: k("emb_bwd"),
        }
    }
}

/// Reverse-pass scratch for [`EncoderLayer::backward`], owned by [`Encoder`]
/// and reused sequentially across every layer (the backward loop processes
/// one layer at a time, in reverse) - same discipline as
/// `text::attn::BartAttnBwdScratch`.
struct EncoderLayerBwdScratch {
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    d_sum2: DeviceBuffer,
    d_ff_hidden: DeviceBuffer,
    d_ff_act: DeviceBuffer,
    d_normed1_ffn: DeviceBuffer,
    d_normed1: DeviceBuffer,
    d_sum1: DeviceBuffer,
    d_x_q: DeviceBuffer,
    d_x_kv: DeviceBuffer,
    d_attn_total: DeviceBuffer,
}

impl EncoderLayerBwdScratch {
    fn new(gpu: &Gpu, t: u32, d_model: u32, ffn_dim: u32) -> EncoderLayerBwdScratch {
        EncoderLayerBwdScratch {
            mean: gpu.storage(t as u64),
            inv: gpu.storage(t as u64),
            d_sum2: gpu.storage((t * d_model) as u64),
            d_ff_hidden: gpu.storage((t * ffn_dim) as u64),
            d_ff_act: gpu.storage((t * ffn_dim) as u64),
            d_normed1_ffn: gpu.storage((t * d_model) as u64),
            d_normed1: gpu.storage((t * d_model) as u64),
            d_sum1: gpu.storage((t * d_model) as u64),
            d_x_q: gpu.storage((t * d_model) as u64),
            d_x_kv: gpu.storage((t * d_model) as u64),
            d_attn_total: gpu.storage((t * d_model) as u64),
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

    /// Same math as [`Self::forward`], with an optional LoRA delta fused
    /// onto every attention projection and the two FFN linears.
    #[allow(clippy::too_many_arguments)]
    fn forward_train<'a>(&'a self, gpu: &Gpu, k: &EncoderKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &'a DeviceBuffer, t: u32, d_model: u32, ffn_dim: u32, eps: f32, lora: Option<&LoraCtx>) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let attn_out = self.self_attn.forward_train(gpu, &k.attn, ps, &format!("{prefix}.self_attn"), x, x, t, t, lora);
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
        let mut s2 = vec![gpu.step(k.matmul_rows, &[&self.normed1, fc1_w, &self.ff_hidden], &[t, d_model, ffn_dim], t.div_ceil(8) * ffn_dim)];
        s2.push(gpu.step(k.bias_add, &[&self.ff_hidden, fc1_b], &[t, ffn_dim], t * ffn_dim));
        if let Some(l) = lora {
            lora_fwd(&mut s2, gpu, l.ids, ps, &format!("{prefix}.fc1.weight"), l.cfg, l.scratch, &self.normed1, &self.ff_hidden, t, d_model, ffn_dim);
        }
        s2.push(gpu.step(k.gelu_erf, &[&self.ff_hidden, &self.ff_act], &[t * ffn_dim], t * ffn_dim));
        s2.push(gpu.step(k.matmul_rows, &[&self.ff_act, fc2_w, &self.ff_out], &[t, ffn_dim, d_model], t.div_ceil(8) * d_model));
        s2.push(gpu.step(k.bias_add, &[&self.ff_out, fc2_b], &[t, d_model], t * d_model));
        if let Some(l) = lora {
            lora_fwd(&mut s2, gpu, l.ids, ps, &format!("{prefix}.fc2.weight"), l.cfg, l.scratch, &self.ff_act, &self.ff_out, t, ffn_dim, d_model);
        }
        s2.push(gpu.step(k.add2, &[&self.normed1, &self.ff_out, &self.sum2], &[t * d_model], t * d_model));
        s2.push(layernorm_fwd(gpu, &ln, &self.sum2, ln2_w, ln2_b, &self.normed2, d_model, t, eps));
        gpu.submit(&[], &s2);
        &self.normed2
    }

    /// Backward of [`Self::forward_train`]. `d_out`: grad of `normed2`
    /// (this layer's returned output). Writes into `scratch.d_x_q`+`.d_x_kv`
    /// = this layer's contribution flowing into the residual stream at `x`
    /// - the caller (self-attention: `x_q == x_kv == x`) must add BOTH of
    /// those to whatever else feeds `x`'s gradient, per
    /// [`super::attn::BartAttn::backward`]'s own contract. Every LN
    /// weight/bias and projection weight/bias gradient (plus LoRA adapters,
    /// when `lora` is `Some`) accumulates through `ps`, gated on
    /// trainability exactly as `forward_train`'s own base weights are used
    /// unconditionally regardless of trainability.
    #[allow(clippy::too_many_arguments)]
    fn backward<'a>(&self, gpu: &Gpu, k: &EncoderKernelIds, bwd: &EncoderBwdKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &DeviceBuffer, d_out: &DeviceBuffer, attn_scratch: &BartAttnBwdScratch, scr: &'a EncoderLayerBwdScratch, t: u32, d_model: u32, ffn_dim: u32, eps: f32, lora: Option<&LoraCtx>) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve(gpu, k.layernorm, bwd.ln_stats, bwd.layernorm_dx);
        let ln2_w = ps.w(&format!("{prefix}.final_layer_norm.weight"));
        let ln2_b_n = format!("{prefix}.final_layer_norm.bias");
        let ln1_w = ps.w(&format!("{prefix}.self_attn_layer_norm.weight"));
        let ln1_b_n = format!("{prefix}.self_attn_layer_norm.bias");

        // ---- final_layer_norm backward: d_out -> d_sum2 ----
        let mut s1 = vec![ln_stats_fwd(gpu, &ln, &self.sum2, &scr.mean, &scr.inv, d_model, t, eps)];
        if trainable(ps, &format!("{prefix}.final_layer_norm.weight")) {
            s1.push(gpu.step(bwd.layernorm_dgamma, &[d_out, &self.sum2, &scr.mean, &scr.inv, ps.g(&format!("{prefix}.final_layer_norm.weight"))], &[d_model, t], d_model));
        }
        if trainable(ps, &ln2_b_n) {
            s1.push(gpu.step(bwd.layernorm_dbeta, &[d_out, ps.g(&ln2_b_n)], &[d_model, t], d_model));
        }
        s1.push(layernorm_dx_bwd(gpu, &ln, &self.sum2, ln2_w, d_out, &scr.d_sum2, d_model, t, eps));
        gpu.submit(&[], &s1);

        // ---- FFN backward: d_sum2 (read twice: FFN's d_out, and again
        // below as the residual branch's direct contribution to d_normed1) ----
        let fc2_w_n = format!("{prefix}.fc2.weight");
        let fc1_w_n = format!("{prefix}.fc1.weight");
        let mut s2 = Vec::new();
        if trainable(ps, &format!("{prefix}.fc2.bias")) {
            s2.push(gpu.step(bwd.bias_grad, &[&scr.d_sum2, ps.g(&format!("{prefix}.fc2.bias"))], &[t, d_model], d_model));
        }
        if trainable(ps, &fc2_w_n) {
            s2.push(gpu.step(bwd.matmul_dw, &[&scr.d_sum2, &self.ff_act, ps.g(&fc2_w_n)], &[t, ffn_dim, d_model], d_model * ffn_dim));
        }
        s2.push(gpu.step(bwd.matmul_dx, &[&scr.d_sum2, ps.w(&fc2_w_n), &scr.d_ff_act], &[t, ffn_dim, d_model, 0], t * ffn_dim));
        if let Some(l) = lora {
            lora_bwd(&mut s2, gpu, l.ids, ps, &fc2_w_n, l.cfg, l.scratch, &scr.d_sum2, &self.ff_act, &scr.d_ff_act, t, ffn_dim, d_model);
        }
        s2.push(gpu.step(bwd.gelu_erf_bwd, &[&self.ff_hidden, &scr.d_ff_act, &scr.d_ff_hidden], &[t * ffn_dim], t * ffn_dim));
        if trainable(ps, &format!("{prefix}.fc1.bias")) {
            s2.push(gpu.step(bwd.bias_grad, &[&scr.d_ff_hidden, ps.g(&format!("{prefix}.fc1.bias"))], &[t, ffn_dim], ffn_dim));
        }
        if trainable(ps, &fc1_w_n) {
            s2.push(gpu.step(bwd.matmul_dw, &[&scr.d_ff_hidden, &self.normed1, ps.g(&fc1_w_n)], &[t, d_model, ffn_dim], ffn_dim * d_model));
        }
        s2.push(gpu.step(bwd.matmul_dx, &[&scr.d_ff_hidden, ps.w(&fc1_w_n), &scr.d_normed1_ffn], &[t, d_model, ffn_dim, 0], t * d_model));
        if let Some(l) = lora {
            lora_bwd(&mut s2, gpu, l.ids, ps, &fc1_w_n, l.cfg, l.scratch, &scr.d_ff_hidden, &self.normed1, &scr.d_normed1_ffn, t, d_model, ffn_dim);
        }
        s2.push(gpu.step(bwd.add2, &[&scr.d_sum2, &scr.d_normed1_ffn, &scr.d_normed1], &[t * d_model], t * d_model));
        gpu.submit(&[], &s2);

        // ---- self_attn_layer_norm backward: d_normed1 -> d_sum1 ----
        let mut s3 = vec![ln_stats_fwd(gpu, &ln, &self.sum1, &scr.mean, &scr.inv, d_model, t, eps)];
        if trainable(ps, &format!("{prefix}.self_attn_layer_norm.weight")) {
            s3.push(gpu.step(bwd.layernorm_dgamma, &[&scr.d_normed1, &self.sum1, &scr.mean, &scr.inv, ps.g(&format!("{prefix}.self_attn_layer_norm.weight"))], &[d_model, t], d_model));
        }
        if trainable(ps, &ln1_b_n) {
            s3.push(gpu.step(bwd.layernorm_dbeta, &[&scr.d_normed1, ps.g(&ln1_b_n)], &[d_model, t], d_model));
        }
        s3.push(layernorm_dx_bwd(gpu, &ln, &self.sum1, ln1_w, &scr.d_normed1, &scr.d_sum1, d_model, t, eps));
        gpu.submit(&[], &s3);

        // ---- self-attn backward: d_sum1 read twice (attn's d_out, and
        // again as x's direct residual contribution, combined below) ----
        self.self_attn.backward(gpu, &bwd.attn, ps, &format!("{prefix}.self_attn"), x, x, &scr.d_sum1, &scr.d_x_q, &scr.d_x_kv, attn_scratch, t, t, lora);
        gpu.submit(&[], &[gpu.step(bwd.add2, &[&scr.d_x_q, &scr.d_x_kv, &scr.d_attn_total], &[t * d_model], t * d_model)]);
        gpu.submit(&[], &[gpu.step(bwd.add2, &[&scr.d_sum1, &scr.d_attn_total, &scr.d_x_q], &[t * d_model], t * d_model)]);
        // `scr.d_x_q` now holds this layer's full contribution to `d_x` -
        // reused as the output slot since attn's own read of it (inside
        // `backward`, above) is already complete.
        &scr.d_x_q
    }
}

/// Reverse-pass state for [`Encoder::backward`], allocated only by
/// [`Encoder::new_train`] - an inference build carries `None`.
struct EncoderTrain {
    lora_cfg: Option<super::lora::LoraCfg>,
    lora_scratch: Option<super::lora::LoraScratch>,
    layer_scratch: EncoderLayerBwdScratch,
    attn_scratch: BartAttnBwdScratch,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    d_pos_embedded: DeviceBuffer,
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
    train: Option<EncoderTrain>,
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
            train: None,
        }
    }

    /// [`Self::new`] plus the reverse-pass scratch [`Self::backward`] needs.
    /// `lora_cfg`: `Some` adapts every attention projection and FFN linear
    /// under this tower (the base weights must then be `Role::Frozen` in
    /// `ps` - `crate::train` builds the `ParamStore` that way); `None` is a
    /// full fine-tune (every base weight `Role::Trainable`). `lora_max_nout`:
    /// the largest `nout` any targeted linear has (`max(d_model, ffn_dim)`),
    /// sizing the shared LoRA scratch.
    pub fn new_train(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, layers: u32, t: u32, eps: f32, lora_cfg: Option<super::lora::LoraCfg>, lora_max_nout: u32) -> Encoder {
        let mut m = Encoder::new(gpu, heads, d_model, ffn_dim, layers, t, eps);
        let lora_scratch = lora_cfg.map(|c| super::lora::LoraScratch::new(gpu, t, c.rank, lora_max_nout));
        m.train = Some(EncoderTrain {
            lora_cfg,
            lora_scratch,
            layer_scratch: EncoderLayerBwdScratch::new(gpu, t, d_model, ffn_dim),
            attn_scratch: BartAttnBwdScratch::new(gpu, heads, d_model, t, t),
            mean: gpu.storage(t as u64),
            inv: gpu.storage(t as u64),
            d_pos_embedded: gpu.storage((t * d_model) as u64),
        });
        m
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

    /// Same math as [`Self::forward`], with an optional LoRA delta fused
    /// onto every layer (`new_train`'s own `lora_cfg` decides whether one
    /// applies - `lora_ids` must be `Some` whenever it does).
    pub fn forward_train<'a>(&'a self, gpu: &Gpu, k: &EncoderKernelIds, ps: &paramstore::ParamStore, inputs_embeds: &DeviceBuffer, lora_ids: Option<&super::lora::LoraKernelIds>) -> &'a DeviceBuffer {
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

        let train = self.train.as_ref().expect("forward_train: build with Encoder::new_train");
        let lora_ctx = train.lora_cfg.as_ref().map(|cfg| LoraCtx { cfg, ids: lora_ids.expect("lora_ids required when new_train's lora_cfg is Some"), scratch: train.lora_scratch.as_ref().expect("lora_scratch") });

        let mut cur: &DeviceBuffer = &self.embedded;
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("language_model.model.encoder.layers.{i}");
            cur = layer.forward_train(gpu, k, ps, &prefix, cur, t, d, self.ffn_dim, self.eps, lora_ctx.as_ref());
        }
        cur
    }

    /// The buffer [`Self::forward`]/[`Self::forward_train`] last returned -
    /// re-derivable without any GPU work, for a caller (`crate::train`) that
    /// needs the same reference again after the borrow from the original
    /// call has gone out of scope.
    pub fn last_hidden(&self) -> &DeviceBuffer {
        self.layers.last().map_or(&self.embedded, |l| &l.normed2)
    }

    /// Backward of [`Self::forward_train`]. `d_hidden_seed`: grad of the
    /// encoder's `last_hidden_state` (accumulated by the caller from every
    /// consumer - the decoder's cross-attention across all its layers, plus
    /// anything else that reads the encoder output). Returns grad of
    /// `inputs_embeds` (`&self.pos_embedded`'s add_inplace has an identity
    /// Jacobian on both operands, so this IS also the buffer this method
    /// dispatches the position-embedding-table's own `emb_bwd` from).
    pub fn backward(&self, gpu: &Gpu, k: &EncoderKernelIds, bwd: &EncoderBwdKernelIds, ps: &paramstore::ParamStore, d_hidden_seed: &DeviceBuffer, lora_ids: Option<&super::lora::LoraKernelIds>) -> &DeviceBuffer {
        let (t, d) = (self.t, self.d_model);
        let train = self.train.as_ref().expect("backward: build with Encoder::new_train");
        let lora_ctx = train.lora_cfg.as_ref().map(|cfg| LoraCtx { cfg, ids: lora_ids.expect("lora_ids required when new_train's lora_cfg is Some"), scratch: train.lora_scratch.as_ref().expect("lora_scratch") });

        let mut d_cur: &DeviceBuffer = d_hidden_seed;
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let prefix = format!("language_model.model.encoder.layers.{i}");
            let x: &DeviceBuffer = if i == 0 { &self.embedded } else { &self.layers[i - 1].normed2 };
            d_cur = layer.backward(gpu, k, bwd, ps, &prefix, x, d_cur, &train.attn_scratch, &train.layer_scratch, t, d, self.ffn_dim, self.eps, lora_ctx.as_ref());
        }

        // ---- embedding-stage LayerNorm backward: d_cur -> d_pos_embedded ----
        let ln = LayerNormIds::resolve(gpu, k.layernorm, bwd.ln_stats, bwd.layernorm_dx);
        let ln_w = ps.w("language_model.model.encoder.layernorm_embedding.weight");
        let ln_b_n = "language_model.model.encoder.layernorm_embedding.bias".to_string();
        let mut s0 = vec![ln_stats_fwd(gpu, &ln, &self.pos_embedded, &train.mean, &train.inv, d, t, self.eps)];
        if trainable(ps, "language_model.model.encoder.layernorm_embedding.weight") {
            s0.push(gpu.step(bwd.layernorm_dgamma, &[d_cur, &self.pos_embedded, &train.mean, &train.inv, ps.g("language_model.model.encoder.layernorm_embedding.weight")], &[d, t], d));
        }
        if trainable(ps, &ln_b_n) {
            s0.push(gpu.step(bwd.layernorm_dbeta, &[d_cur, ps.g(&ln_b_n)], &[d, t], d));
        }
        s0.push(layernorm_dx_bwd(gpu, &ln, &self.pos_embedded, ln_w, d_cur, &train.d_pos_embedded, d, t, self.eps));
        gpu.submit(&[], &s0);

        // ---- position-embedding table backward (if trainable) ----
        let pos_w_n = "language_model.model.encoder.embed_positions.weight";
        if trainable(ps, pos_w_n) {
            let pos_vocab = ps.numel(pos_w_n) as u32 / d;
            gpu.submit(&[], &[gpu.step(bwd.emb_bwd, &[&self.pos_idx, &train.d_pos_embedded, ps.g(pos_w_n)], &[t, d, pos_vocab], pos_vocab * d)]);
        }

        &train.d_pos_embedded
    }
}
