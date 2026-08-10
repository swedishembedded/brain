// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastVLM configuration. The decoder side is parsed from the released
//! `config.json` (`apple/FastVLM-*`); the FastViTHD vision-tower dims are fixed
//! (they live in the vision tower's `mci.py`, not the LLaVA config) and captured
//! as [`FastVitHdConfig::fastvithd`].

use qwen3::QwenConfig;
use serde_json::Value;

/// FastViTHD hybrid conv/attention encoder configuration (the `mobileclip_l_1024`
/// tower). Five stages: RepMixer (conv) token-mixing in stages 0–2, self-attention
/// in stages 3–4. Input 1024², total downsample 64× (stem 4× + four PatchEmbed 2×)
/// → 16×16 = 256 tokens at output dim 3072 (= 1536 · cls_ratio 2).
#[derive(Clone, Debug, PartialEq)]
pub struct FastVitHdConfig {
    /// Blocks per stage. `[2, 12, 24, 4, 2]`.
    pub layers: [u32; 5],
    /// Channels per stage. `[96, 192, 384, 768, 1536]`.
    pub embed_dims: [u32; 5],
    /// ConvFFN / MLP expansion ratio. 4.
    pub mlp_ratio: u32,
    /// Stages using self-attention token-mixing (else RepMixer conv). `[3, 4]`.
    pub attn_stages: [u32; 2],
    /// Fixed attention head dim (num_heads = stage_dim / 32). 32.
    pub head_dim: u32,
    /// Square input resolution. 1024.
    pub input_size: u32,
    /// Final `conv_exp` expansion of the last stage dim. 2.0.
    pub cls_ratio: u32,
}

impl FastVitHdConfig {
    /// The `fastvithd()` tower (`mci.py`).
    pub fn fastvithd() -> FastVitHdConfig {
        FastVitHdConfig {
            layers: [2, 12, 24, 4, 2],
            embed_dims: [96, 192, 384, 768, 1536],
            mlp_ratio: 4,
            attn_stages: [3, 4],
            head_dim: 32,
            input_size: 1024,
            cls_ratio: 2,
        }
    }
    /// Output feature dim = last stage dim × cls_ratio (3072).
    pub fn out_dim(&self) -> u32 {
        self.embed_dims[4] * self.cls_ratio
    }
    /// Total spatial downsample: stem 4× × four PatchEmbed 2× = 64.
    pub fn total_downsample(&self) -> u32 {
        64
    }
    /// Visual token count for a square input (`(input/64)²`). 256 at 1024.
    pub fn num_tokens(&self) -> u32 {
        let side = self.input_size / self.total_downsample();
        side * side
    }
}

/// Full FastVLM configuration: FastViTHD tower + `mlp2x_gelu` projector + Qwen2
/// decoder + the LLaVA image-token sentinel.
#[derive(Clone, Debug)]
pub struct FastVlmConfig {
    pub vision: FastVitHdConfig,
    /// Vision feature width feeding the projector (= `vision.out_dim()`). 3072.
    pub mm_hidden: u32,
    /// Projector type; only `mlp2x_gelu` (Linear→GELU→Linear) is supported.
    pub proj_type: String,
    /// The Qwen2 decoder config. NB: brain's `qwen` decoder currently hard-wires
    /// QK-norm ON and no bias; the Qwen2 deltas (QK-norm OFF, qkv bias ON) are
    /// applied by the decoder toggles being added next — this holds the dims.
    pub decoder: QwenConfig,
    /// LLaVA `IMAGE_TOKEN_INDEX` sentinel spliced into the text stream.
    pub image_token_index: i32,
}

impl FastVlmConfig {
    fn decoder_of(vocab: u32, layers: u32, hidden: u32, heads: u32, kv: u32, inter: u32, tie: bool) -> QwenConfig {
        // Qwen2 decoder: QK-norm off, qkv bias on.
        QwenConfig::qwen2(vocab, layers, hidden, heads, kv, inter, tie)
    }

    /// `apple/FastVLM-0.5B` (Qwen2-0.5B decoder).
    pub fn fastvlm_0_5b() -> FastVlmConfig {
        let vision = FastVitHdConfig::fastvithd();
        FastVlmConfig {
            mm_hidden: vision.out_dim(),
            proj_type: "mlp2x_gelu".into(),
            decoder: Self::decoder_of(151936, 24, 896, 14, 2, 4864, true),
            image_token_index: -200,
            vision,
        }
    }

    /// `apple/FastVLM-1.5B` (Qwen2-1.5B decoder).
    pub fn fastvlm_1_5b() -> FastVlmConfig {
        let vision = FastVitHdConfig::fastvithd();
        FastVlmConfig {
            mm_hidden: vision.out_dim(),
            proj_type: "mlp2x_gelu".into(),
            decoder: Self::decoder_of(151936, 28, 1536, 12, 2, 8960, true),
            image_token_index: -200,
            vision,
        }
    }

    /// `apple/FastVLM-7B` (Qwen2-7B decoder; untied head, vocab 152064).
    pub fn fastvlm_7b() -> FastVlmConfig {
        let vision = FastVitHdConfig::fastvithd();
        FastVlmConfig {
            mm_hidden: vision.out_dim(),
            proj_type: "mlp2x_gelu".into(),
            decoder: Self::decoder_of(152064, 28, 3584, 28, 4, 18944, false),
            image_token_index: -200,
            vision,
        }
    }

    /// Parse the decoder + multimodal fields from a released `config.json`
    /// (`LlavaQwen2ForCausalLM`). The FastViTHD tower dims are fixed defaults.
    pub fn from_hf(c: &Value) -> FastVlmConfig {
        let u = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("fastvlm config: missing {k}")) as u32;
        let hidden = u("hidden_size");
        let heads = u("num_attention_heads");
        let vision = FastVitHdConfig::fastvithd();
        FastVlmConfig {
            mm_hidden: u("mm_hidden_size"),
            proj_type: c["mm_projector_type"].as_str().unwrap_or("mlp2x_gelu").to_string(),
            decoder: QwenConfig {
                vocab: u("vocab_size"),
                block_size: 2048,
                n_layers: u("num_hidden_layers"),
                d_model: hidden,
                n_heads: heads,
                n_kv_heads: u("num_key_value_heads"),
                head_dim: hidden / heads,
                d_ff: u("intermediate_size"),
                rope_theta: c["rope_theta"].as_f64().unwrap_or(1e6) as f32,
                rms_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
                max_position_embeddings: c["max_position_embeddings"].as_u64().map(|x| x as u32).unwrap_or(2048),
                tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(true),
                qk_norm: false, // Qwen2
                attn_bias: c["attention_bias"].as_bool().unwrap_or(true),
                lora: None,
            },
            image_token_index: -200,
            vision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastvithd_token_count_and_dims() {
        let v = FastVitHdConfig::fastvithd();
        assert_eq!(v.out_dim(), 3072); // 1536 · 2
        assert_eq!(v.total_downsample(), 64);
        assert_eq!(v.num_tokens(), 256); // (1024/64)²
    }

    #[test]
    fn from_hf_matches_0_5b_preset() {
        let json = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "mm_hidden_size": 3072,
            "mm_projector_type": "mlp2x_gelu"
        });
        let parsed = FastVlmConfig::from_hf(&json);
        let preset = FastVlmConfig::fastvlm_0_5b();
        assert_eq!(parsed.decoder.d_model, preset.decoder.d_model);
        assert_eq!(parsed.decoder.n_layers, 24);
        assert_eq!(parsed.decoder.head_dim, 64); // 896/14
        assert_eq!(parsed.decoder.n_kv_heads, 2);
        assert_eq!(parsed.decoder.d_ff, 4864);
        assert_eq!(parsed.mm_hidden, 3072);
        assert_eq!(parsed.image_token_index, -200);
        assert_eq!(parsed.proj_type, "mlp2x_gelu");
    }

    #[test]
    fn projector_input_matches_vision_output() {
        let c = FastVlmConfig::fastvlm_0_5b();
        assert_eq!(c.mm_hidden, c.vision.out_dim());
    }
}
