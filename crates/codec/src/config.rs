// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Codec configuration, parsed from the official `Qwen3-TTS-Tokenizer-12Hz`
//! `config.json`. Only the fields the decode path needs are modeled here; the
//! encoder fields are kept for the from-scratch trainer (Track B).

use serde_json::Value;

/// Decoder (Mimi-style) configuration — the codes->waveform path.
#[derive(Clone, Debug, Default)]
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

    // ---- encoder (HuggingFace Mimi `encoder_config`) — the encode path ----
    pub enc: EncoderConfig,
}

/// Mimi SEANet encoder + encoder-transformer + downsample hyperparameters, parsed
/// from the codec `config.json`'s `encoder_config`.
#[derive(Clone, Debug, Default)]
pub struct EncoderConfig {
    pub num_filters: u32,         // 64
    pub hidden_size: u32,         // 512
    pub kernel_size: u32,         // 7 (head conv)
    pub last_kernel_size: u32,    // 3 (tail conv)
    pub residual_kernel_size: u32, // 3
    pub compress: u32,            // 2 (resnet bottleneck divisor)
    pub dilation_growth_rate: u32, // 2
    pub num_residual_layers: u32, // 1
    pub upsampling_ratios: Vec<u32>, // [8,6,5,4]; the encoder downsamples by reversed()
    // encoder transformer
    pub num_hidden_layers: u32,   // 8
    pub num_attention_heads: u32, // 8
    pub num_key_value_heads: u32, // 8
    pub head_dim: u32,            // 64
    pub intermediate_size: u32,   // 2048
    pub rope_theta: f32,          // 10000
    pub norm_eps: f32,            // 1e-5
    pub sliding_window: u32,      // 250
    // RVQ
    pub codebook_dim: u32,        // 256 (vector_quantization_hidden_dimension)
    pub codebook_size: u32,       // 2048
    pub num_semantic_quantizers: u32, // 1
    pub downsample_stride: u32,   // 2 (frame-rate match conv)
    pub downsample_kernel: u32,   // 4 (= 2·int(encodec_frame_rate/frame_rate))
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
            enc: {
                let e = &v["encoder_config"];
                EncoderConfig {
                    num_filters: gu(e, "num_filters", 64),
                    hidden_size: gu(e, "hidden_size", 512),
                    kernel_size: gu(e, "kernel_size", 7),
                    last_kernel_size: gu(e, "last_kernel_size", 3),
                    residual_kernel_size: gu(e, "residual_kernel_size", 3),
                    compress: gu(e, "compress", 2),
                    dilation_growth_rate: gu(e, "dilation_growth_rate", 2),
                    num_residual_layers: gu(e, "num_residual_layers", 1),
                    upsampling_ratios: {
                        let r = gvec(e, "upsampling_ratios");
                        if r.is_empty() { vec![8, 6, 5, 4] } else { r }
                    },
                    num_hidden_layers: gu(e, "num_hidden_layers", 8),
                    num_attention_heads: gu(e, "num_attention_heads", 8),
                    num_key_value_heads: gu(e, "num_key_value_heads", 8),
                    head_dim: gu(e, "head_dim", 64),
                    intermediate_size: gu(e, "intermediate_size", 2048),
                    rope_theta: gf(e, "rope_theta", 10000.0),
                    norm_eps: gf(e, "norm_eps", 1e-5),
                    sliding_window: gu(e, "sliding_window", 250),
                    codebook_dim: gu(e, "vector_quantization_hidden_dimension", 256),
                    codebook_size: gu(e, "codebook_size", 2048),
                    num_semantic_quantizers: gu(e, "num_semantic_quantizers", 1),
                    downsample_stride: 2,
                    downsample_kernel: {
                        // HF: kernel_size = 2·int(encodec_frame_rate / frame_rate),
                        // with encodec_frame_rate = sampling_rate / prod(ratios).
                        let prod: u32 = {
                            let r = gvec(e, "upsampling_ratios");
                            if r.is_empty() { 960 } else { r.iter().product() }
                        };
                        let sr = gu(e, "sampling_rate", 24000) as f64;
                        let fr = e["_frame_rate"].as_f64().unwrap_or(12.5);
                        let encodec_fr = sr / prod as f64;
                        2 * (encodec_fr / fr) as u32
                    },
                }
            },
        }
    }
}
