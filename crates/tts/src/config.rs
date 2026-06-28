// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Talker + MTP code-predictor configuration, parsed from the official
//! `Qwen3-TTS-12Hz-0.6B-Base` `config.json` (`talker_config`).
//!
//! NOTE the Talker uses **M-RoPE** (`rope_scaling.interleaved = true`,
//! `mrope_section = [24,20,20]`), which differs from brain-qwen's half-split
//! NeoX RoPE — the model code must apply the interleaved/mrope convention for
//! HF parity.

use serde_json::Value;

/// Talker (Qwen3-style multi-codebook decoder) configuration.
#[derive(Clone, Debug)]
pub struct TalkerConfig {
    pub n_layers: u32,                // 28
    pub d_model: u32,                 // 1024
    pub head_dim: u32,                // 128 (note q_dim = n_heads*head_dim = 2048 != d_model)
    pub n_heads: u32,                 // 16
    pub n_kv_heads: u32,              // 8
    pub d_ff: u32,                    // 3072
    pub vocab: u32,                   // 3072 (codebook-0 + specials)
    pub num_code_groups: u32,         // 16
    pub text_hidden_size: u32,        // 2048
    pub text_vocab_size: u32,         // 151936
    pub rope_theta: f32,              // 1e6
    pub rms_norm_eps: f32,            // 1e-6
    pub max_position_embeddings: u32, // 32768
    pub mrope_section: Vec<u32>,      // [24,20,20]
    pub mrope_interleaved: bool,      // true
    pub position_id_per_seconds: u32, // 13
    // special codec token ids
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
}

/// MTP code-predictor (5-layer Qwen3 block) configuration.
#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub n_layers: u32,        // 5
    pub d_model: u32,         // 1024
    pub head_dim: u32,        // 128
    pub n_heads: u32,         // 16
    pub n_kv_heads: u32,      // 8
    pub d_ff: u32,            // 3072
    pub vocab: u32,           // 2048 (per residual codebook)
    pub num_code_groups: u32, // 16
    pub rope_theta: f32,      // 1e6
    pub rms_norm_eps: f32,    // 1e-6
}

fn gu(o: &Value, k: &str, d: u32) -> u32 {
    o[k].as_u64().map(|x| x as u32).unwrap_or(d)
}
fn gf(o: &Value, k: &str, d: f32) -> f32 {
    o[k].as_f64().map(|x| x as f32).unwrap_or(d)
}

impl TalkerConfig {
    /// Parse from the top-level talker `config.json` value (the object that
    /// contains `talker_config`).
    pub fn from_json(root: &Value) -> TalkerConfig {
        let t = &root["talker_config"];
        let mrope = t["rope_scaling"]["mrope_section"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|y| y as u32))
                    .collect()
            })
            .unwrap_or_else(|| vec![24, 20, 20]);
        TalkerConfig {
            n_layers: gu(t, "num_hidden_layers", 28),
            d_model: gu(t, "hidden_size", 1024),
            head_dim: gu(t, "head_dim", 128),
            n_heads: gu(t, "num_attention_heads", 16),
            n_kv_heads: gu(t, "num_key_value_heads", 8),
            d_ff: gu(t, "intermediate_size", 3072),
            vocab: gu(t, "vocab_size", 3072),
            num_code_groups: gu(t, "num_code_groups", 16),
            text_hidden_size: gu(t, "text_hidden_size", 2048),
            text_vocab_size: gu(t, "text_vocab_size", 151936),
            rope_theta: gf(t, "rope_theta", 1_000_000.0),
            rms_norm_eps: gf(t, "rms_norm_eps", 1e-6),
            max_position_embeddings: gu(t, "max_position_embeddings", 32768),
            mrope_interleaved: t["rope_scaling"]["interleaved"].as_bool().unwrap_or(true),
            mrope_section: mrope,
            position_id_per_seconds: gu(t, "position_id_per_seconds", 13),
            codec_bos_id: gu(t, "codec_bos_id", 2149),
            codec_eos_token_id: gu(t, "codec_eos_token_id", 2150),
            codec_pad_id: gu(t, "codec_pad_id", 2148),
        }
    }
}

impl TalkerConfig {
    /// Query projection width = `n_heads * head_dim` (q_dim = 2048 for the real
    /// model, decoupled from `d_model` = 1024).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    /// Key/Value projection width = `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// The Talker decoder is a Qwen3 dense decoder with an **untied** codec
    /// embedding/head (`tie_embeddings = false`): `tok.weight` is
    /// `talker.model.codec_embedding`, `lm_head.weight` is `talker.codec_head`.
    /// M-RoPE collapses to Qwen's half-split RoPE for an audio stream (all three
    /// mrope position-id sections share the same index — see `talker.rs`), so the
    /// shared `crate::qwen` backbone is parity-equivalent.
    pub fn to_qwen(&self, block_size: u32) -> qwen::QwenConfig {
        qwen::QwenConfig {
            vocab: self.vocab,
            block_size,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            d_ff: self.d_ff,
            rope_theta: self.rope_theta,
            rms_eps: self.rms_norm_eps,
            tie_embeddings: false,
            lora: None,
        }
        .with_defaults()
    }

    /// Reconstruct a `TalkerConfig` from the inner Qwen decoder config (the form
    /// stored in an imported Talker checkpoint). Talker-only metadata not carried
    /// by the Qwen config (`num_code_groups`, special ids, mrope) takes the
    /// real-model defaults; `text_hidden_size` is patched by the loader from the
    /// `text_projection` shape when available.
    pub fn from_qwen(q: &qwen::QwenConfig) -> TalkerConfig {
        let mut c = TalkerConfig::from_json(&Value::Null); // all real-model defaults
        c.n_layers = q.n_layers;
        c.d_model = q.d_model;
        c.head_dim = q.head_dim;
        c.n_heads = q.n_heads;
        c.n_kv_heads = q.n_kv_heads;
        c.d_ff = q.d_ff;
        c.vocab = q.vocab;
        c.rope_theta = q.rope_theta;
        c.rms_norm_eps = q.rms_eps;
        c
    }

    /// A tiny config for tests / gradient checks (GQA 4q/2kv, decoupled head_dim
    /// 8, SwiGLU ff 32, vocab 23). Mirrors `QwenConfig::tiny` but untied.
    pub fn tiny() -> TalkerConfig {
        TalkerConfig {
            n_layers: 2,
            d_model: 16,
            head_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 32,
            vocab: 23,
            num_code_groups: 16,
            text_hidden_size: 20,
            text_vocab_size: 29,
            rope_theta: 1.0e6,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 32768,
            mrope_section: vec![24, 20, 20],
            mrope_interleaved: true,
            position_id_per_seconds: 13,
            codec_bos_id: 2149,
            codec_eos_token_id: 2150,
            codec_pad_id: 2148,
        }
    }
}

impl MtpConfig {
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// Number of residual codebooks the MTP fills (and the number of input/output
    /// embedding/head tables): `num_code_groups - 1` (15 for the real model).
    pub fn n_residual(&self) -> u32 {
        self.num_code_groups - 1
    }

    /// A tiny config for tests (5→2 layers, d 16, GQA 4q/2kv, head_dim 8, ff 32,
    /// vocab 23, 4 code groups → 3 residual tables).
    pub fn tiny() -> MtpConfig {
        MtpConfig {
            n_layers: 2,
            d_model: 16,
            head_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 32,
            vocab: 23,
            num_code_groups: 4,
            rope_theta: 1.0e6,
            rms_norm_eps: 1e-6,
        }
    }

    /// Serialise to a brain checkpoint config object (consumed by
    /// [`MtpModel::load_inference`]).
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "qwen3_tts_mtp",
            "n_layers": self.n_layers, "d_model": self.d_model, "head_dim": self.head_dim,
            "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads, "d_ff": self.d_ff,
            "vocab_size": self.vocab, "num_code_groups": self.num_code_groups,
            "rope_theta": self.rope_theta, "rms_norm_eps": self.rms_norm_eps,
        })
    }

    /// Parse the config object written by [`MtpConfig::to_json`].
    pub fn from_brain_json(c: &Value) -> MtpConfig {
        MtpConfig {
            n_layers: gu(c, "n_layers", 5),
            d_model: gu(c, "d_model", 1024),
            head_dim: gu(c, "head_dim", 128),
            n_heads: gu(c, "n_heads", 16),
            n_kv_heads: gu(c, "n_kv_heads", 8),
            d_ff: gu(c, "d_ff", 3072),
            vocab: gu(c, "vocab_size", 2048),
            num_code_groups: gu(c, "num_code_groups", 16),
            rope_theta: gf(c, "rope_theta", 1_000_000.0),
            rms_norm_eps: gf(c, "rms_norm_eps", 1e-6),
        }
    }

    pub fn from_json(root: &Value) -> MtpConfig {
        let c = &root["talker_config"]["code_predictor_config"];
        MtpConfig {
            n_layers: gu(c, "num_hidden_layers", 5),
            d_model: gu(c, "hidden_size", 1024),
            head_dim: gu(c, "head_dim", 128),
            n_heads: gu(c, "num_attention_heads", 16),
            n_kv_heads: gu(c, "num_key_value_heads", 8),
            d_ff: gu(c, "intermediate_size", 3072),
            vocab: gu(c, "vocab_size", 2048),
            num_code_groups: gu(c, "num_code_groups", 16),
            rope_theta: gf(c, "rope_theta", 1_000_000.0),
            rms_norm_eps: gf(c, "rms_norm_eps", 1e-6),
        }
    }
}
