// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M6: LoRA + full fine-tune training for the BART text side of
//! `crates/florence2` - `Florence2Trainer`. DaViT stays entirely FROZEN and
//! out of scope for this module, matching `deepseek2ocr`'s own frozen-
//! vision-tower precedent (its SAM encoder never gets a backward pass
//! either - only its decoder trains): this trainer never builds a vision
//! tower at all, and its encoder input is the text prompt embeddings only
//! (`t_vision = 0`). That is an honest simplification, not a coverage gap -
//! the vision-token splice (`row_scatter` in `text::lm::Florence2Lm::encode`)
//! carries no trainable parameters of its own, so skipping it loses no
//! backward-pass coverage; every trainable tensor in the real `ground`
//! inference path (the full BART encoder-decoder, tied embeddings, and
//! every LoRA-targetable attention/FFN projection) is exercised here.
//!
//! LoRA targets: the four attention projections (`q_proj`/`k_proj`/
//! `v_proj`/`out_proj`) and the two FFN linears (`fc1`/`fc2`), on every
//! encoder AND decoder layer (including the decoder's `encoder_attn`) -
//! `text::lora`'s module doc has the shared forward/backward primitives.
//! Full fine-tune trains every one of those tensors directly instead.
//!
//! Swedish Embedded AB builds from-scratch GPU training stacks for
//! vision-language and encoder-decoder models. If your team needs a
//! gradient-checked LoRA or full-fine-tune path for a composite
//! architecture like this one, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};

use crate::text::attn::BartAttnKernelIds;
use crate::text::decoder::{Decoder, DecoderBwdKernelIds, DecoderKernelIds};
use crate::text::encoder::{Encoder, EncoderBwdKernelIds, EncoderKernelIds};
use crate::text::lora::{trainable, LoraCfg, LoraKernelIds};
use crate::text::{BartConfig, POSITION_OFFSET};

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("bias_add", kernels::BIAS_ADD),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("embed", kernels::EMBED),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("layernorm", kernels::LAYERNORM),
    ("gelu_erf", kernels::GELU_ERF),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("ce_value", kernels::CE_VALUE),
    ("ce_grad", kernels::CE_GRAD),
    ("matmul", kernels::MATMUL),
    ("axpy", kernels::AXPY),
    ("grad_scale", kernels::GRAD_SCALE),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

const LORA_WEIGHT_LEAVES: [&str; 6] = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight"];
const LORA_BIAS_LEAVES: [&str; 6] = ["q_proj.bias", "k_proj.bias", "v_proj.bias", "out_proj.bias", "fc1.bias", "fc2.bias"];

fn is_lora_leaf(name: &str) -> bool {
    LORA_WEIGHT_LEAVES.iter().any(|l| name.ends_with(l)) || LORA_BIAS_LEAVES.iter().any(|l| name.ends_with(l))
}

fn attn_block(v: &mut Vec<(String, usize)>, prefix: &str, d: usize) {
    for leaf in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        v.push((format!("{prefix}.{leaf}.weight"), d * d));
        v.push((format!("{prefix}.{leaf}.bias"), d));
    }
}
fn ln_block(v: &mut Vec<(String, usize)>, name: &str, d: usize) {
    v.push((format!("{name}.weight"), d));
    v.push((format!("{name}.bias"), d));
}
fn ffn_block(v: &mut Vec<(String, usize)>, prefix: &str, d: usize, ffn: usize) {
    v.push((format!("{prefix}.fc1.weight"), ffn * d));
    v.push((format!("{prefix}.fc1.bias"), ffn));
    v.push((format!("{prefix}.fc2.weight"), d * ffn));
    v.push((format!("{prefix}.fc2.bias"), d));
}

/// Every base (non-LoRA) tensor `Florence2Trainer` needs, with its element
/// count - the trainer's own shape, not the real checkpoint's: no vision
/// tower, and the position tables are sized exactly to `t_prompt`/`max_t`
/// (plus [`POSITION_OFFSET`]) rather than the real 1024-position ceiling.
fn base_tensor_names(cfg: &BartConfig, t_prompt: u32, max_t: u32) -> Vec<(String, usize)> {
    let d = cfg.d_model as usize;
    let mut v = Vec::new();
    v.push(("language_model.model.shared.weight".to_string(), cfg.vocab_size as usize * d));
    v.push(("language_model.final_logits_bias".to_string(), cfg.vocab_size as usize));
    v.push(("language_model.model.encoder.embed_positions.weight".to_string(), (t_prompt + POSITION_OFFSET) as usize * d));
    ln_block(&mut v, "language_model.model.encoder.layernorm_embedding", d);
    v.push(("language_model.model.decoder.embed_positions.weight".to_string(), (max_t + POSITION_OFFSET) as usize * d));
    ln_block(&mut v, "language_model.model.decoder.layernorm_embedding", d);

    for i in 0..cfg.encoder_layers {
        let p = format!("language_model.model.encoder.layers.{i}");
        attn_block(&mut v, &format!("{p}.self_attn"), d);
        ln_block(&mut v, &format!("{p}.self_attn_layer_norm"), d);
        ffn_block(&mut v, &p, d, cfg.encoder_ffn_dim as usize);
        ln_block(&mut v, &format!("{p}.final_layer_norm"), d);
    }
    for i in 0..cfg.decoder_layers {
        let p = format!("language_model.model.decoder.layers.{i}");
        attn_block(&mut v, &format!("{p}.self_attn"), d);
        ln_block(&mut v, &format!("{p}.self_attn_layer_norm"), d);
        attn_block(&mut v, &format!("{p}.encoder_attn"), d);
        ln_block(&mut v, &format!("{p}.encoder_attn_layer_norm"), d);
        ffn_block(&mut v, &p, d, cfg.decoder_ffn_dim as usize);
        ln_block(&mut v, &format!("{p}.final_layer_norm"), d);
    }
    v
}

/// `.lora_a`/`.lora_b` for every LoRA-targetable weight in `base` (the FFN
/// pair's `in`/`out` dims differ by leaf, recovered from `base`'s own
/// element counts rather than re-deriving encoder/decoder ffn widths here).
fn lora_tensor_names(base: &[(String, usize)], cfg: &BartConfig, rank: u32) -> Vec<(String, usize)> {
    let d = cfg.d_model as usize;
    let mut v = Vec::new();
    for (name, numel) in base {
        if !name.ends_with(".weight") || !LORA_WEIGHT_LEAVES.iter().any(|l| name.ends_with(l)) {
            continue;
        }
        let out_dim = numel / d.max(1);
        let (in_dim, out_dim) = if name.ends_with("fc1.weight") { (d, out_dim) } else if name.ends_with("fc2.weight") { (numel / d, d) } else { (d, out_dim) };
        v.push((format!("{name}.lora_a"), rank as usize * in_dim));
        v.push((format!("{name}.lora_b"), out_dim * rank as usize));
    }
    v
}

/// Full parameter list (base + LoRA adapters when `lora` is `Some`) -
/// what [`Florence2Trainer::new`] builds the [`ParamStore`] from and what
/// [`init_weights`] must produce every name for.
pub fn param_list(cfg: &BartConfig, t_prompt: u32, max_t: u32, lora: Option<LoraCfg>) -> Vec<(String, usize)> {
    let base = base_tensor_names(cfg, t_prompt, max_t);
    let mut v = base.clone();
    if let Some(l) = lora {
        v.extend(lora_tensor_names(&base, cfg, l.rank));
    }
    v
}

/// Deterministic fresh init: LayerNorm gains at identity (`1.0`)/biases at
/// `0.0`, `.lora_b` at `0.0` (a fresh adapter is an exact no-op - the same
/// convention every LoRA-adopting model in this repo holds to), everything
/// else `Normal(0, std)` via [`data::rng::Lcg`].
pub fn init_weights(names: &[(String, usize)], std: f32, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Lcg::new(seed);
    let mut sorted: Vec<&(String, usize)> = names.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = HashMap::new();
    for (name, numel) in sorted {
        let data: Vec<f32> = if name.ends_with("layer_norm.weight") || name.ends_with("layernorm_embedding.weight") {
            vec![1.0; *numel]
        } else if name.ends_with(".bias") || name.ends_with(".lora_b") || name.ends_with("layer_norm.bias") || name.ends_with("layernorm_embedding.bias") {
            vec![0.0; *numel]
        } else {
            (0..*numel).map(|_| rng.scaled(std)).collect()
        };
        out.insert(name.clone(), data);
    }
    out
}

fn role_for(name: &str, lora: bool) -> Role {
    if lora && is_lora_leaf(name) {
        Role::Frozen
    } else {
        Role::Trainable
    }
}

pub struct Florence2Trainer {
    gpu: Gpu,
    cfg: BartConfig,
    t_prompt: u32,
    max_t: u32,
    ps: ParamStore,
    encoder: Encoder,
    decoder: Decoder,
    enc_k: EncoderKernelIds,
    dec_k: DecoderKernelIds,
    enc_bwd_k: EncoderBwdKernelIds,
    dec_bwd_k: DecoderBwdKernelIds,
    lora_ids: Option<LoraKernelIds>,
    prompt_idx: DeviceBuffer,
    decoder_idx: DeviceBuffer,
    targets_idx: DeviceBuffer,
    encoder_input: DeviceBuffer,
    dec_embedded: DeviceBuffer,
    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,
    d_logits: DeviceBuffer,
    d_hidden: DeviceBuffer,
    opt: optim::Optim,
}

impl Florence2Trainer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: Gpu, cfg: BartConfig, t_prompt: u32, max_t: u32, lora: Option<LoraCfg>, init: &HashMap<String, Vec<f32>>) -> Florence2Trainer {
        let base = base_tensor_names(&cfg, t_prompt, max_t);
        let mut all = base.clone();
        if let Some(l) = lora {
            all.extend(lora_tensor_names(&base, &cfg, l.rank));
        }
        let roles: Vec<(String, usize, Role)> = all.into_iter().map(|(n, sz)| { let r = role_for(&n, lora.is_some()); (n, sz, r) }).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let d = cfg.d_model;
        let lora_max_nout = d.max(cfg.encoder_ffn_dim).max(cfg.decoder_ffn_dim);
        let encoder = Encoder::new_train(&gpu, cfg.encoder_attention_heads, d, cfg.encoder_ffn_dim, cfg.encoder_layers, t_prompt, cfg.layer_norm_eps, lora, lora_max_nout);
        let decoder = Decoder::new_train(&gpu, cfg.decoder_attention_heads, d, cfg.decoder_ffn_dim, cfg.decoder_layers, max_t, t_prompt, cfg.layer_norm_eps, lora, lora_max_nout);

        let enc_k = EncoderKernelIds { attn: BartAttnKernelIds::resolve(PIPELINES), embed: idx("embed"), add2: idx("add2"), add_inplace: idx("add_inplace"), layernorm: idx("layernorm"), matmul_rows: idx("matmul_rows"), bias_add: idx("bias_add"), gelu_erf: idx("gelu_erf") };
        let dec_k = DecoderKernelIds { attn: BartAttnKernelIds::resolve(PIPELINES), embed: idx("embed"), add2: idx("add2"), add_inplace: idx("add_inplace"), layernorm: idx("layernorm"), matmul_rows: idx("matmul_rows"), bias_add: idx("bias_add"), gelu_erf: idx("gelu_erf") };
        let enc_bwd_k = EncoderBwdKernelIds::resolve(PIPELINES);
        let dec_bwd_k = DecoderBwdKernelIds::resolve(PIPELINES);
        let lora_ids = lora.map(|_| LoraKernelIds::resolve(PIPELINES));

        let prompt_idx = gpu.buffer("florence2_train_prompt_idx", (t_prompt * 4) as u64, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let decoder_idx = gpu.buffer("florence2_train_dec_idx", (max_t * 4) as u64, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let targets_idx = gpu.buffer("florence2_train_targets", (max_t * 4) as u64, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);

        let opt = optim::Optim::new(idx("adamw"), idx("gradnorm_sq"), idx("grad_scale"), idx("clip_coef"), idx("grad_scale_buf"));

        Florence2Trainer {
            encoder_input: gpu.storage((t_prompt * d) as u64),
            dec_embedded: gpu.storage((max_t * d) as u64),
            logits: gpu.storage((max_t * cfg.vocab_size) as u64),
            ce_buf: gpu.storage(max_t as u64),
            d_logits: gpu.storage((max_t * cfg.vocab_size) as u64),
            d_hidden: gpu.storage((max_t * d) as u64),
            gpu,
            cfg,
            t_prompt,
            max_t,
            ps,
            encoder,
            decoder,
            enc_k,
            dec_k,
            enc_bwd_k,
            dec_bwd_k,
            lora_ids,
            prompt_idx,
            decoder_idx,
            targets_idx,
            opt,
        }
    }

    /// Upload one training example. `prompt_ids.len() == t_prompt`,
    /// `decoder_ids.len() == targets.len() == max_t` (this trainer runs one
    /// fixed-length teacher-forced pass per step, no padding/IGNORE - every
    /// position is supervised, matching `text::lm`'s own single-example
    /// convention).
    pub fn set_batch(&self, prompt_ids: &[u32], decoder_ids: &[u32], targets: &[u32]) {
        assert_eq!(prompt_ids.len() as u32, self.t_prompt);
        assert_eq!(decoder_ids.len() as u32, self.max_t);
        assert_eq!(targets.len() as u32, self.max_t);
        self.gpu.write(&self.prompt_idx, bytemuck::cast_slice(prompt_ids));
        self.gpu.write(&self.decoder_idx, bytemuck::cast_slice(decoder_ids));
        self.gpu.write(&self.targets_idx, bytemuck::cast_slice(targets));
    }

    fn k(&self, name: &str) -> usize {
        idx(name)
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Full forward: prompt-token embed -> encoder -> decoder-token embed ->
    /// decoder (cross-attending over the encoder output) -> tied `lm_head`
    /// -> mean cross-entropy. Returns the scalar loss.
    pub fn loss(&self) -> f32 {
        let d = self.cfg.d_model;
        let shared = self.ps.w("language_model.model.shared.weight");
        self.gpu.submit(&[], &[self.gpu.step(self.k("embed"), &[&self.prompt_idx, shared, &self.encoder_input], &[d, self.t_prompt], self.t_prompt * d)]);
        let enc_out = self.encoder.forward_train(&self.gpu, &self.enc_k, &self.ps, &self.encoder_input, self.lora_ids.as_ref());

        self.gpu.submit(&[], &[self.gpu.step(self.k("embed"), &[&self.decoder_idx, shared, &self.dec_embedded], &[d, self.max_t], self.max_t * d)]);
        let dec_hidden = self.decoder.forward_train(&self.gpu, &self.dec_k, &self.ps, &self.dec_embedded, enc_out, self.max_t, self.lora_ids.as_ref());

        let bias = self.ps.w("language_model.final_logits_bias");
        let v = self.cfg.vocab_size;
        let t = self.max_t;
        self.gpu.submit(
            &[],
            &[
                self.gpu.step(self.k("matmul_rows"), &[dec_hidden, shared, &self.logits], &[t, d, v], t.div_ceil(8) * v),
                self.gpu.step(self.k("bias_add"), &[&self.logits, bias], &[t, v], t * v),
                self.gpu.step(self.k("ce_value"), &[&self.logits, &self.targets_idx, &self.ce_buf], &[t, v], t),
            ],
        );
        self.gpu.poll_wait();
        self.gpu.read(&self.ce_buf, t as usize).iter().sum::<f32>() / t as f32
    }

    /// Backward of [`Self::loss`] - must be called immediately after (the
    /// last `loss()` call's activations are what this differentiates,
    /// exactly like every other model's `forward()`/`backward()` pair in
    /// this repo).
    pub fn backward(&self) {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let t = self.max_t;
        let shared_n = "language_model.model.shared.weight";
        let dec_hidden = self.decoder.last_hidden();

        self.gpu.submit(&[], &[self.gpu.step(self.k("ce_grad"), &[&self.logits, &self.targets_idx, &self.d_logits], &[t, v], t * v)]);

        let mut s = Vec::new();
        if trainable(&self.ps, "language_model.final_logits_bias") {
            s.push(self.gpu.step(self.k("bias_grad"), &[&self.d_logits, self.ps.g("language_model.final_logits_bias")], &[t, v], v));
        }
        if trainable(&self.ps, shared_n) {
            s.push(self.gpu.step(self.k("matmul_dw"), &[&self.d_logits, dec_hidden, self.ps.g(shared_n)], &[t, d, v], v * d));
        }
        s.push(self.gpu.step(self.k("matmul_dx"), &[&self.d_logits, self.ps.w(shared_n), &self.d_hidden], &[t, d, v, 0], t * d));
        self.gpu.submit(&[], &s);

        let enc_out = self.encoder.last_hidden();
        let (d_dec_embedded, d_enc_out) = self.decoder.backward(&self.gpu, &self.dec_k, &self.dec_bwd_k, &self.ps, enc_out, &self.d_hidden, t, self.lora_ids.as_ref());
        if trainable(&self.ps, shared_n) {
            let vocab = v;
            self.gpu.submit(&[], &[self.gpu.step(self.k("emb_bwd"), &[&self.decoder_idx, d_dec_embedded, self.ps.g(shared_n)], &[self.max_t, d, vocab], self.max_t * d)]);
        }

        let d_inputs_embeds = self.encoder.backward(&self.gpu, &self.enc_k, &self.enc_bwd_k, &self.ps, d_enc_out, self.lora_ids.as_ref());
        if trainable(&self.ps, shared_n) {
            let vocab = v;
            self.gpu.submit(&[], &[self.gpu.step(self.k("emb_bwd"), &[&self.prompt_idx, d_inputs_embeds, self.ps.g(shared_n)], &[self.t_prompt, d, vocab], self.t_prompt * d)]);
        }
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, 1.0);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn param_names(&self) -> Vec<String> {
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
}

fn idx(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("florence2 train PIPELINES: missing {name}"))
}
