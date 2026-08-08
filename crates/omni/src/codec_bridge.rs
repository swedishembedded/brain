// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loading `tts::mtp::MtpModel` (the code predictor) and `codec::Codec`
//! (Code2Wav) straight from the real HF checkpoint via `reader` — the same
//! recipe `crates/omni/tests/code_predictor_parity.rs` and `code2wav_parity.rs`
//! already validated against real weights (cosine 1.000000 each), lifted out
//! of the test files so `crate::caps`'s `speak` action can build the same
//! two models for real serving instead of only proving they're correct in
//! isolation.

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use codec::config::CodecConfig;
use codec::model::Codec;
use gpu_core::Gpu;
use tts::import::mtp_hf_to_brain;
use tts::mtp::MtpModel;

use crate::config::{Code2WavConfig, TalkerConfig};
use crate::talker_prompt::TalkerPromptSpecials;

/// Build an [`MtpModel`] from `talker.code_predictor.*` real HF tensors,
/// renamed via `tts::import::mtp_hf_to_brain` (already correct — see
/// `crate::import::map_code_predictor`'s doc, which now delegates to the
/// same function) and built directly with `MtpModel::build_on` — no
/// `ParamStore`/checkpoint-file round trip, same pattern
/// `code_predictor_parity.rs` uses.
pub fn load_mtp(reader: &WeightReader, gpu: Gpu, cfg: &tts::config::MtpConfig) -> MtpModel {
    let mut decoder: HashMap<String, Vec<f32>> = HashMap::new();
    for l in 0..cfg.n_layers {
        for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
            let hf = format!("talker.code_predictor.model.layers.{l}.{leaf}");
            let brain_name = mtp_hf_to_brain(&hf).unwrap_or_else(|| panic!("mtp_hf_to_brain rejected {hf}"));
            decoder.insert(brain_name, reader.tensor(&hf).unwrap_or_else(|| panic!("missing {hf}")));
        }
    }
    let norm_hf = "talker.code_predictor.model.norm.weight";
    decoder.insert(mtp_hf_to_brain(norm_hf).unwrap(), reader.tensor(norm_hf).unwrap_or_else(|| panic!("missing {norm_hf}")));

    let n_residual = cfg.n_residual() as usize;
    let codec_embedding: Vec<Vec<f32>> = (0..n_residual).map(|i| reader.tensor(&format!("talker.code_predictor.model.codec_embedding.{i}.weight")).unwrap()).collect();
    let lm_head: Vec<Vec<f32>> = (0..n_residual).map(|i| reader.tensor(&format!("talker.code_predictor.lm_head.{i}.weight")).unwrap()).collect();

    MtpModel::build_on(gpu, cfg.clone(), decoder, codec_embedding, lm_head)
}

/// Build a [`Codec`] from `code2wav.*` real HF tensors, prefix-stripped
/// (matching `crate::import::map_code2wav`'s now-fixed convention — see its
/// doc) and built with `Codec::from_weights` — same pattern
/// `code2wav_parity.rs` uses.
pub fn load_codec(reader: &WeightReader, oc: &Code2WavConfig) -> Codec {
    let cfg = CodecConfig {
        num_quantizers: oc.num_quantizers,
        num_semantic_quantizers: oc.num_semantic_quantizers,
        codebook_size: oc.codebook_size,
        semantic_codebook_size: oc.semantic_codebook_size,
        codebook_dim: oc.codebook_dim,
        latent_dim: oc.hidden_size,
        hidden_size: oc.hidden_size,
        intermediate_size: oc.intermediate_size,
        num_hidden_layers: oc.num_hidden_layers,
        num_attention_heads: oc.num_attention_heads,
        num_key_value_heads: oc.num_key_value_heads,
        head_dim: oc.hidden_size / oc.num_attention_heads,
        sliding_window: oc.sliding_window,
        rope_theta: oc.rope_theta,
        rms_norm_eps: oc.rms_norm_eps,
        layer_scale_initial_scale: oc.layer_scale_initial_scale,
        decoder_dim: oc.decoder_dim,
        upsample_rates: oc.upsample_rates.clone(),
        upsampling_ratios: oc.upsampling_ratios.clone(),
        input_sample_rate: oc.output_sample_rate,
        output_sample_rate: oc.output_sample_rate,
        decode_upsample_rate: oc.total_upsample(),
        enc: Default::default(),
    };

    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for name in reader.names() {
        if let Some(rest) = name.strip_prefix("code2wav.") {
            init.insert(rest.to_string(), reader.tensor(name).unwrap());
        }
    }
    Codec::from_weights(cfg, init)
}

/// `talker.text_projection`/`talker.hidden_projection` as `tts::talker::
/// TextProjection` (reused unchanged, `text_embedding: None` — see
/// `crate::talker_prompt`'s doc). `which` is `"text_projection"` or
/// `"hidden_projection"`.
pub fn load_talker_projection(reader: &WeightReader, cfg: &TalkerConfig, which: &str) -> tts::talker::TextProjection {
    let get = |leaf: &str| {
        let name = format!("talker.{which}.{leaf}");
        reader.tensor(&name).unwrap_or_else(|| panic!("missing {name}"))
    };
    tts::talker::TextProjection {
        text_embedding: None,
        fc1_w: get("linear_fc1.weight"),
        fc1_b: get("linear_fc1.bias"),
        fc2_w: get("linear_fc2.weight"),
        fc2_b: get("linear_fc2.bias"),
        in_dim: cfg.thinker_hidden_size as usize,
        inter: cfg.text.moe_intermediate as usize,
        out: cfg.text.hidden as usize,
        text_vocab: 0,
    }
}

pub fn talker_prompt_specials(oc: &crate::config::OmniConfig) -> TalkerPromptSpecials {
    TalkerPromptSpecials {
        tts_bos_id: oc.tts_bos_token_id,
        tts_eos_id: oc.tts_eos_token_id,
        tts_pad_id: oc.tts_pad_token_id,
        codec_nothink_id: oc.talker.codec_nothink_id,
        codec_think_bos_id: oc.talker.codec_think_bos_id,
        codec_think_eos_id: oc.talker.codec_think_eos_id,
        codec_pad_id: oc.talker.codec_pad_id,
        codec_bos_id: oc.talker.codec_bos_id,
    }
}
