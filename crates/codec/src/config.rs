// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Codec configuration, parsed from the official `Qwen3-TTS-Tokenizer-12Hz`
//! `config.json`. Only the fields the decode path needs are modeled here; the
//! encoder fields are kept for the from-scratch trainer (Track B).

use serde_json::Value;

/// Decoder (Mimi-style) configuration — the codes->waveform path.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    // RVQ
    pub num_quantizers: u32,          // 16 total (1 semantic + 15 acoustic)
    pub num_semantic_quantizers: u32, // 1
    pub codebook_size: u32,           // 2048 (acoustic)
    pub semantic_codebook_size: u32,  // 4096
    pub codebook_dim: u32,            // 512 (vector_quantization_hidden_dimension)
    pub latent_dim: u32,              // 1024

    // pre-transformer (sliding-window GQA decoder)
    pub hidden_size: u32,      // 512
    pub intermediate_size: u32, // 1024
    pub num_hidden_layers: u32, // 8
    pub num_attention_heads: u32, // 16
    pub num_key_value_heads: u32, // 16
    pub head_dim: u32,         // 64
    pub sliding_window: u32,   // 72
    pub rope_theta: f32,       // 10000
    pub rms_norm_eps: f32,     // 1e-5
    pub layer_scale_initial_scale: f32, // 0.01

    // SEANet upsampling decoder
    pub decoder_dim: u32,           // 1536
    pub upsample_rates: Vec<u32>,   // [8,5,4,3]
    pub upsampling_ratios: Vec<u32>, // [2,2]

    pub input_sample_rate: u32,  // 24000
    pub output_sample_rate: u32, // 24000
    pub decode_upsample_rate: u32, // 1920
}

impl CodecConfig {
    /// Parse from the top-level codec `config.json` value.
    pub fn from_json(v: &Value) -> CodecConfig {
        let d = &v["decoder_config"];
        let gu = |o: &Value, k: &str, def: u32| o[k].as_u64().map(|x| x as u32).unwrap_or(def);
        let gf = |o: &Value, k: &str, def: f32| o[k].as_f64().map(|x| x as f32).unwrap_or(def);
        let gvec = |o: &Value, k: &str| -> Vec<u32> {
            o[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default()
        };
        CodecConfig {
            num_quantizers: gu(v, "encoder_valid_num_quantizers", 16),
            num_semantic_quantizers: gu(d, "num_semantic_quantizers", 1),
            codebook_size: gu(d, "codebook_size", 2048),
            semantic_codebook_size: gu(d, "semantic_codebook_size", 4096),
            codebook_dim: gu(d, "vector_quantization_hidden_dimension", 512),
            latent_dim: gu(d, "latent_dim", 1024),
            hidden_size: gu(d, "hidden_size", 512),
            intermediate_size: gu(d, "intermediate_size", 1024),
            num_hidden_layers: gu(d, "num_hidden_layers", 8),
            num_attention_heads: gu(d, "num_attention_heads", 16),
            num_key_value_heads: gu(d, "num_key_value_heads", 16),
            head_dim: gu(d, "head_dim", 64),
            sliding_window: gu(d, "sliding_window", 72),
            rope_theta: gf(d, "rope_theta", 10000.0),
            rms_norm_eps: gf(d, "rms_norm_eps", 1e-5),
            layer_scale_initial_scale: gf(d, "layer_scale_initial_scale", 0.01),
            decoder_dim: gu(d, "decoder_dim", 1536),
            upsample_rates: { let r = gvec(d, "upsample_rates"); if r.is_empty() { vec![8, 5, 4, 3] } else { r } },
            upsampling_ratios: { let r = gvec(d, "upsampling_ratios"); if r.is_empty() { vec![2, 2] } else { r } },
            input_sample_rate: gu(v, "input_sample_rate", 24000),
            output_sample_rate: gu(v, "output_sample_rate", 24000),
            decode_upsample_rate: gu(v, "decode_upsample_rate", 1920),
        }
    }
}
