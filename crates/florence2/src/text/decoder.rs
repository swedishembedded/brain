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
use model::block::{layernorm_dx_bwd, layernorm_fwd, ln_stats_fwd, LayerNormIds};
use model::vit::row_index_buffer;

use super::attn::{BartAttn, BartAttnBwdIds, BartAttnBwdScratch, BartAttnKernelIds};
use super::lora::{lora_bwd, lora_fwd, trainable, LoraCtx};

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

pub struct DecoderBwdKernelIds {
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
    pub add_inplace: usize,
    pub emb_bwd: usize,
}

impl DecoderBwdKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> DecoderBwdKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        DecoderBwdKernelIds {
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
            add_inplace: k("add_inplace"),
            emb_bwd: k("emb_bwd"),
        }
    }
}

/// Reverse-pass scratch for [`DecoderLayer::backward`], owned by
/// [`Decoder`] and reused sequentially across every layer.
struct DecoderLayerBwdScratch {
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    d_sum3: DeviceBuffer,
    d_ff_hidden: DeviceBuffer,
    d_ff_act: DeviceBuffer,
    d_normed2_ffn: DeviceBuffer,
    d_normed2: DeviceBuffer,
    d_sum2: DeviceBuffer,
    d_cross_q: DeviceBuffer,
    d_cross_kv: DeviceBuffer,
    d_normed1: DeviceBuffer,
    d_sum1: DeviceBuffer,
    d_self_q: DeviceBuffer,
    d_self_kv: DeviceBuffer,
    d_attn_total: DeviceBuffer,
}

impl DecoderLayerBwdScratch {
    fn new(gpu: &Gpu, max_t: u32, t_enc: u32, d_model: u32, ffn_dim: u32) -> DecoderLayerBwdScratch {
        DecoderLayerBwdScratch {
            mean: gpu.storage(max_t as u64),
            inv: gpu.storage(max_t as u64),
            d_sum3: gpu.storage((max_t * d_model) as u64),
            d_ff_hidden: gpu.storage((max_t * ffn_dim) as u64),
            d_ff_act: gpu.storage((max_t * ffn_dim) as u64),
            d_normed2_ffn: gpu.storage((max_t * d_model) as u64),
            d_normed2: gpu.storage((max_t * d_model) as u64),
            d_sum2: gpu.storage((max_t * d_model) as u64),
            d_cross_q: gpu.storage((max_t * d_model) as u64),
            d_cross_kv: gpu.storage((t_enc * d_model) as u64),
            d_normed1: gpu.storage((max_t * d_model) as u64),
            d_sum1: gpu.storage((max_t * d_model) as u64),
            d_self_q: gpu.storage((max_t * d_model) as u64),
            d_self_kv: gpu.storage((max_t * d_model) as u64),
            d_attn_total: gpu.storage((max_t * d_model) as u64),
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

    /// Same math as [`Self::forward`], with an optional LoRA delta fused
    /// onto every attention projection and the two FFN linears.
    #[allow(clippy::too_many_arguments)]
    fn forward_train<'a>(&'a self, gpu: &Gpu, k: &DecoderKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &'a DeviceBuffer, enc: &DeviceBuffer, t: u32, t_enc: u32, d_model: u32, ffn_dim: u32, eps: f32, lora: Option<&LoraCtx>) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);

        let self_out = self.self_attn.forward_train(gpu, &k.attn, ps, &format!("{prefix}.self_attn"), x, x, t, t, lora);
        let ln1_w = ps.w(&format!("{prefix}.self_attn_layer_norm.weight"));
        let ln1_b = ps.w(&format!("{prefix}.self_attn_layer_norm.bias"));
        let s1 = vec![
            gpu.step(k.add2, &[x, self_out, &self.sum1], &[t * d_model], t * d_model),
            layernorm_fwd(gpu, &ln, &self.sum1, ln1_w, ln1_b, &self.normed1, d_model, t, eps),
        ];
        gpu.submit(&[], &s1);

        let cross_out = self.cross_attn.forward_train(gpu, &k.attn, ps, &format!("{prefix}.encoder_attn"), &self.normed1, enc, t, t_enc, lora);
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
        let mut s3 = vec![gpu.step(k.matmul_rows, &[&self.normed2, fc1_w, &self.ff_hidden], &[t, d_model, ffn_dim], t.div_ceil(8) * ffn_dim)];
        s3.push(gpu.step(k.bias_add, &[&self.ff_hidden, fc1_b], &[t, ffn_dim], t * ffn_dim));
        if let Some(l) = lora {
            lora_fwd(&mut s3, gpu, l.ids, ps, &format!("{prefix}.fc1.weight"), l.cfg, l.scratch, &self.normed2, &self.ff_hidden, t, d_model, ffn_dim);
        }
        s3.push(gpu.step(k.gelu_erf, &[&self.ff_hidden, &self.ff_act], &[t * ffn_dim], t * ffn_dim));
        s3.push(gpu.step(k.matmul_rows, &[&self.ff_act, fc2_w, &self.ff_out], &[t, ffn_dim, d_model], t.div_ceil(8) * d_model));
        s3.push(gpu.step(k.bias_add, &[&self.ff_out, fc2_b], &[t, d_model], t * d_model));
        if let Some(l) = lora {
            lora_fwd(&mut s3, gpu, l.ids, ps, &format!("{prefix}.fc2.weight"), l.cfg, l.scratch, &self.ff_act, &self.ff_out, t, ffn_dim, d_model);
        }
        s3.push(gpu.step(k.add2, &[&self.normed2, &self.ff_out, &self.sum3], &[t * d_model], t * d_model));
        s3.push(layernorm_fwd(gpu, &ln, &self.sum3, ln3_w, ln3_b, &self.normed3, d_model, t, eps));
        gpu.submit(&[], &s3);
        &self.normed3
    }

    /// Backward of [`Self::forward_train`]. `d_out`: grad of `normed3`.
    /// ASSIGNS `scr.d_self_q`/`.d_self_kv` (combined below into this
    /// layer's contribution to `d_x`, returned) and ACCUMULATES the
    /// cross-attention's key/value-side gradient into `d_enc_out` (via
    /// `add_inplace` - the caller must have cleared it before the FIRST
    /// (highest-index) layer's call, since every layer's cross-attention
    /// reads the SAME fixed encoder memory and their contributions sum).
    #[allow(clippy::too_many_arguments)]
    fn backward<'a>(&self, gpu: &Gpu, k: &DecoderKernelIds, bwd: &DecoderBwdKernelIds, ps: &paramstore::ParamStore, prefix: &str, x: &DeviceBuffer, enc: &DeviceBuffer, d_out: &DeviceBuffer, d_enc_out: &DeviceBuffer, attn_scratch: &BartAttnBwdScratch, scr: &'a DecoderLayerBwdScratch, t: u32, t_enc: u32, d_model: u32, ffn_dim: u32, eps: f32, lora: Option<&LoraCtx>) -> &'a DeviceBuffer {
        let ln = LayerNormIds::resolve(gpu, k.layernorm, bwd.ln_stats, bwd.layernorm_dx);

        // ---- final_layer_norm backward: d_out -> d_sum3 ----
        let ln3_w = ps.w(&format!("{prefix}.final_layer_norm.weight"));
        let ln3_b_n = format!("{prefix}.final_layer_norm.bias");
        let mut s1 = vec![ln_stats_fwd(gpu, &ln, &self.sum3, &scr.mean, &scr.inv, d_model, t, eps)];
        if trainable(ps, &format!("{prefix}.final_layer_norm.weight")) {
            s1.push(gpu.step(bwd.layernorm_dgamma, &[d_out, &self.sum3, &scr.mean, &scr.inv, ps.g(&format!("{prefix}.final_layer_norm.weight"))], &[d_model, t], d_model));
        }
        if trainable(ps, &ln3_b_n) {
            s1.push(gpu.step(bwd.layernorm_dbeta, &[d_out, ps.g(&ln3_b_n)], &[d_model, t], d_model));
        }
        s1.push(layernorm_dx_bwd(gpu, &ln, &self.sum3, ln3_w, d_out, &scr.d_sum3, d_model, t, eps));
        gpu.submit(&[], &s1);

        // ---- FFN backward: d_sum3 read twice ----
        let fc2_w_n = format!("{prefix}.fc2.weight");
        let fc1_w_n = format!("{prefix}.fc1.weight");
        let mut s2 = Vec::new();
        if trainable(ps, &format!("{prefix}.fc2.bias")) {
            s2.push(gpu.step(bwd.bias_grad, &[&scr.d_sum3, ps.g(&format!("{prefix}.fc2.bias"))], &[t, d_model], d_model));
        }
        if trainable(ps, &fc2_w_n) {
            s2.push(gpu.step(bwd.matmul_dw, &[&scr.d_sum3, &self.ff_act, ps.g(&fc2_w_n)], &[t, ffn_dim, d_model], d_model * ffn_dim));
        }
        s2.push(gpu.step(bwd.matmul_dx, &[&scr.d_sum3, ps.w(&fc2_w_n), &scr.d_ff_act], &[t, ffn_dim, d_model, 0], t * ffn_dim));
        if let Some(l) = lora {
            lora_bwd(&mut s2, gpu, l.ids, ps, &fc2_w_n, l.cfg, l.scratch, &scr.d_sum3, &self.ff_act, &scr.d_ff_act, t, ffn_dim, d_model);
        }
        s2.push(gpu.step(bwd.gelu_erf_bwd, &[&self.ff_hidden, &scr.d_ff_act, &scr.d_ff_hidden], &[t * ffn_dim], t * ffn_dim));
        if trainable(ps, &format!("{prefix}.fc1.bias")) {
            s2.push(gpu.step(bwd.bias_grad, &[&scr.d_ff_hidden, ps.g(&format!("{prefix}.fc1.bias"))], &[t, ffn_dim], ffn_dim));
        }
        if trainable(ps, &fc1_w_n) {
            s2.push(gpu.step(bwd.matmul_dw, &[&scr.d_ff_hidden, &self.normed2, ps.g(&fc1_w_n)], &[t, d_model, ffn_dim], ffn_dim * d_model));
        }
        s2.push(gpu.step(bwd.matmul_dx, &[&scr.d_ff_hidden, ps.w(&fc1_w_n), &scr.d_normed2_ffn], &[t, d_model, ffn_dim, 0], t * d_model));
        if let Some(l) = lora {
            lora_bwd(&mut s2, gpu, l.ids, ps, &fc1_w_n, l.cfg, l.scratch, &scr.d_ff_hidden, &self.normed2, &scr.d_normed2_ffn, t, d_model, ffn_dim);
        }
        s2.push(gpu.step(bwd.add2, &[&scr.d_sum3, &scr.d_normed2_ffn, &scr.d_normed2], &[t * d_model], t * d_model));
        gpu.submit(&[], &s2);

        // ---- encoder_attn_layer_norm backward: d_normed2 -> d_sum2 ----
        let ln2_w = ps.w(&format!("{prefix}.encoder_attn_layer_norm.weight"));
        let ln2_b_n = format!("{prefix}.encoder_attn_layer_norm.bias");
        let mut s3 = vec![ln_stats_fwd(gpu, &ln, &self.sum2, &scr.mean, &scr.inv, d_model, t, eps)];
        if trainable(ps, &format!("{prefix}.encoder_attn_layer_norm.weight")) {
            s3.push(gpu.step(bwd.layernorm_dgamma, &[&scr.d_normed2, &self.sum2, &scr.mean, &scr.inv, ps.g(&format!("{prefix}.encoder_attn_layer_norm.weight"))], &[d_model, t], d_model));
        }
        if trainable(ps, &ln2_b_n) {
            s3.push(gpu.step(bwd.layernorm_dbeta, &[&scr.d_normed2, ps.g(&ln2_b_n)], &[d_model, t], d_model));
        }
        s3.push(layernorm_dx_bwd(gpu, &ln, &self.sum2, ln2_w, &scr.d_normed2, &scr.d_sum2, d_model, t, eps));
        gpu.submit(&[], &s3);

        // ---- cross-attn backward: d_sum2 read twice (cross-attn's d_out,
        // and again as normed1's direct residual contribution) ----
        self.cross_attn.backward(gpu, &bwd.attn, ps, &format!("{prefix}.encoder_attn"), &self.normed1, enc, &scr.d_sum2, &scr.d_cross_q, &scr.d_cross_kv, attn_scratch, t, t_enc, lora);
        gpu.submit(&[], &[gpu.step(bwd.add_inplace, &[d_enc_out, &scr.d_cross_kv], &[t_enc * d_model], t_enc * d_model)]);
        gpu.submit(&[], &[gpu.step(bwd.add2, &[&scr.d_sum2, &scr.d_cross_q, &scr.d_normed1], &[t * d_model], t * d_model)]);

        // ---- self_attn_layer_norm backward: d_normed1 -> d_sum1 ----
        let ln1_w = ps.w(&format!("{prefix}.self_attn_layer_norm.weight"));
        let ln1_b_n = format!("{prefix}.self_attn_layer_norm.bias");
        let mut s4 = vec![ln_stats_fwd(gpu, &ln, &self.sum1, &scr.mean, &scr.inv, d_model, t, eps)];
        if trainable(ps, &format!("{prefix}.self_attn_layer_norm.weight")) {
            s4.push(gpu.step(bwd.layernorm_dgamma, &[&scr.d_normed1, &self.sum1, &scr.mean, &scr.inv, ps.g(&format!("{prefix}.self_attn_layer_norm.weight"))], &[d_model, t], d_model));
        }
        if trainable(ps, &ln1_b_n) {
            s4.push(gpu.step(bwd.layernorm_dbeta, &[&scr.d_normed1, ps.g(&ln1_b_n)], &[d_model, t], d_model));
        }
        s4.push(layernorm_dx_bwd(gpu, &ln, &self.sum1, ln1_w, &scr.d_normed1, &scr.d_sum1, d_model, t, eps));
        gpu.submit(&[], &s4);

        // ---- self-attn backward: d_sum1 read twice ----
        self.self_attn.backward(gpu, &bwd.attn, ps, &format!("{prefix}.self_attn"), x, x, &scr.d_sum1, &scr.d_self_q, &scr.d_self_kv, attn_scratch, t, t, lora);
        gpu.submit(&[], &[gpu.step(bwd.add2, &[&scr.d_self_q, &scr.d_self_kv, &scr.d_attn_total], &[t * d_model], t * d_model)]);
        gpu.submit(&[], &[gpu.step(bwd.add2, &[&scr.d_sum1, &scr.d_attn_total, &scr.d_self_q], &[t * d_model], t * d_model)]);
        &scr.d_self_q
    }
}

/// Reverse-pass state for [`Decoder::backward`], allocated only by
/// [`Decoder::new_train`] - an inference build carries `None`.
struct DecoderTrain {
    lora_cfg: Option<super::lora::LoraCfg>,
    lora_scratch: Option<super::lora::LoraScratch>,
    layer_scratch: DecoderLayerBwdScratch,
    attn_scratch: BartAttnBwdScratch,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    d_pos_embedded: DeviceBuffer,
    d_enc_out: DeviceBuffer,
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
    train: Option<DecoderTrain>,
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
            train: None,
        }
    }

    /// [`Self::new`] plus the reverse-pass scratch [`Self::backward`]
    /// needs, and the encoder-output gradient accumulator every layer's
    /// cross-attention adds into. See [`text::encoder::Encoder::new_train`]
    /// for `lora_cfg`/`lora_max_nout`.
    pub fn new_train(gpu: &Gpu, heads: u32, d_model: u32, ffn_dim: u32, layers: u32, max_t: u32, t_enc: u32, eps: f32, lora_cfg: Option<super::lora::LoraCfg>, lora_max_nout: u32) -> Decoder {
        let mut m = Decoder::new(gpu, heads, d_model, ffn_dim, layers, max_t, t_enc, eps);
        let lora_scratch = lora_cfg.map(|c| super::lora::LoraScratch::new(gpu, max_t, c.rank, lora_max_nout));
        m.train = Some(DecoderTrain {
            lora_cfg,
            lora_scratch,
            layer_scratch: DecoderLayerBwdScratch::new(gpu, max_t, t_enc, d_model, ffn_dim),
            attn_scratch: BartAttnBwdScratch::new(gpu, heads, d_model, max_t, max_t.max(t_enc)),
            mean: gpu.storage(max_t as u64),
            inv: gpu.storage(max_t as u64),
            d_pos_embedded: gpu.storage((max_t * d_model) as u64),
            d_enc_out: gpu.storage((t_enc * d_model) as u64),
        });
        m
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

    /// Same math as [`Self::forward`], with an optional LoRA delta fused
    /// onto every layer.
    pub fn forward_train<'a>(&'a self, gpu: &Gpu, k: &DecoderKernelIds, ps: &paramstore::ParamStore, decoder_inputs_embeds: &DeviceBuffer, enc: &DeviceBuffer, t: u32, lora_ids: Option<&super::lora::LoraKernelIds>) -> &'a DeviceBuffer {
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

        let train = self.train.as_ref().expect("forward_train: build with Decoder::new_train");
        let lora_ctx = train.lora_cfg.as_ref().map(|cfg| LoraCtx { cfg, ids: lora_ids.expect("lora_ids required when new_train's lora_cfg is Some"), scratch: train.lora_scratch.as_ref().expect("lora_scratch") });

        let mut cur: &DeviceBuffer = &self.embedded;
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("language_model.model.decoder.layers.{i}");
            cur = layer.forward_train(gpu, k, ps, &prefix, cur, enc, t, self.t_enc, d, self.ffn_dim, self.eps, lora_ctx.as_ref());
        }
        cur
    }

    /// The buffer [`Self::forward`]/[`Self::forward_train`] last returned -
    /// see `text::encoder::Encoder::last_hidden`'s doc for why this exists.
    pub fn last_hidden(&self) -> &DeviceBuffer {
        self.layers.last().map_or(&self.embedded, |l| &l.normed3)
    }

    /// Backward of [`Self::forward_train`]. `d_hidden_seed`: grad of the
    /// decoder's final hidden state (typically the lm_head's `matmul_dx`
    /// contribution). Returns `(d_decoder_inputs_embeds, d_enc_out)` -
    /// `d_enc_out` is the FULL sum of every layer's cross-attention
    /// contribution to the encoder's output, ready to seed
    /// `Encoder::backward` directly.
    pub fn backward<'a>(&'a self, gpu: &Gpu, k: &DecoderKernelIds, bwd: &DecoderBwdKernelIds, ps: &paramstore::ParamStore, enc: &DeviceBuffer, d_hidden_seed: &DeviceBuffer, t: u32, lora_ids: Option<&super::lora::LoraKernelIds>) -> (&'a DeviceBuffer, &'a DeviceBuffer) {
        let (d, t_enc) = (self.d_model, self.t_enc);
        let train = self.train.as_ref().expect("backward: build with Decoder::new_train");
        let lora_ctx = train.lora_cfg.as_ref().map(|cfg| LoraCtx { cfg, ids: lora_ids.expect("lora_ids required when new_train's lora_cfg is Some"), scratch: train.lora_scratch.as_ref().expect("lora_scratch") });

        gpu.submit(&[&train.d_enc_out], &[]);

        let mut d_cur: &DeviceBuffer = d_hidden_seed;
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let prefix = format!("language_model.model.decoder.layers.{i}");
            let x: &DeviceBuffer = if i == 0 { &self.embedded } else { &self.layers[i - 1].normed3 };
            d_cur = layer.backward(gpu, k, bwd, ps, &prefix, x, enc, d_cur, &train.d_enc_out, &train.attn_scratch, &train.layer_scratch, t, t_enc, d, self.ffn_dim, self.eps, lora_ctx.as_ref());
        }

        // ---- embedding-stage LayerNorm backward: d_cur -> d_pos_embedded ----
        let ln = LayerNormIds::resolve(gpu, k.layernorm, bwd.ln_stats, bwd.layernorm_dx);
        let ln_w = ps.w("language_model.model.decoder.layernorm_embedding.weight");
        let ln_b_n = "language_model.model.decoder.layernorm_embedding.bias".to_string();
        let mut s0 = vec![ln_stats_fwd(gpu, &ln, &self.pos_embedded, &train.mean, &train.inv, d, t, self.eps)];
        if trainable(ps, "language_model.model.decoder.layernorm_embedding.weight") {
            s0.push(gpu.step(bwd.layernorm_dgamma, &[d_cur, &self.pos_embedded, &train.mean, &train.inv, ps.g("language_model.model.decoder.layernorm_embedding.weight")], &[d, t], d));
        }
        if trainable(ps, &ln_b_n) {
            s0.push(gpu.step(bwd.layernorm_dbeta, &[d_cur, ps.g(&ln_b_n)], &[d, t], d));
        }
        s0.push(layernorm_dx_bwd(gpu, &ln, &self.pos_embedded, ln_w, d_cur, &train.d_pos_embedded, d, t, self.eps));
        gpu.submit(&[], &s0);

        // ---- position-embedding table backward (if trainable) ----
        let pos_w_n = "language_model.model.decoder.embed_positions.weight";
        if trainable(ps, pos_w_n) {
            let pos_vocab = ps.numel(pos_w_n) as u32 / d;
            gpu.submit(&[], &[gpu.step(bwd.emb_bwd, &[&self.pos_idx, &train.d_pos_embedded, ps.g(pos_w_n)], &[t, d, pos_vocab], t * d)]);
        }

        (&train.d_pos_embedded, &train.d_enc_out)
    }
}
