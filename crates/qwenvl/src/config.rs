// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL-4B configuration. Every default is taken from the released
//! `config.json` (`Qwen/Qwen3-VL-4B-Instruct`); the tests parse an inline copy of
//! that structure and cross-check the [`Qwen3VlConfig::qwen3_vl_4b`] preset.

use qwen::QwenConfig;
use serde_json::Value;

/// Qwen3-VL ViT vision-encoder configuration (`vision_config`).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    /// Transformer blocks (`depth`). 24 for 4B.
    pub depth: u32,
    /// Hidden width of the ViT. 1024.
    pub hidden: u32,
    /// Attention heads (head_dim = hidden / num_heads = 64). 16.
    pub num_heads: u32,
    /// MLP intermediate width. 4096.
    pub intermediate: u32,
    /// Spatial patch size (pixels). 16.
    pub patch_size: u32,
    /// Temporal patch size (frames folded into each patch vector). 2.
    pub temporal_patch_size: u32,
    /// 2×2 spatial merge in the PatchMerger. 2.
    pub spatial_merge_size: u32,
    /// Learned positional-embedding table rows (= pos_grid²). 2304 = 48×48.
    pub num_position_embeddings: u32,
    /// Connector output width = decoder hidden size. 2560.
    pub out_hidden_size: u32,
    /// Input image channels. 3.
    pub in_channels: u32,
    /// Vision blocks whose outputs feed DeepStack (added to decoder layers
    /// 0,1,…). [5, 11, 17] for 4B.
    pub deepstack_indexes: Vec<u32>,
}

impl VisionConfig {
    /// Attention head dimension (`hidden / num_heads`).
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.num_heads
    }
    /// Patches merged into one visual token (`spatial_merge_size²`).
    pub fn merge_unit(&self) -> u32 {
        self.spatial_merge_size * self.spatial_merge_size
    }
    /// Side length of the square learned pos-embed grid (√num_position_embeddings).
    pub fn pos_grid(&self) -> u32 {
        (self.num_position_embeddings as f64).sqrt().round() as u32
    }
    /// Flattened per-patch vector width feeding the patch embed
    /// (`in_channels · temporal_patch_size · patch_size²`). 3·2·16·16 = 1536.
    pub fn patch_vec_dim(&self) -> u32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// Full Qwen3-VL configuration: vision encoder + Qwen3 text decoder + interleaved
/// M-RoPE section + the vision special-token ids.
#[derive(Clone, Debug)]
pub struct Qwen3VlConfig {
    pub vision: VisionConfig,
    /// The Qwen3 decoder config (reused wholesale from `qwen`).
    pub text: QwenConfig,
    /// Interleaved-M-RoPE per-axis channel counts (T,H,W); sums to head_dim/2.
    /// [24, 20, 20] for 4B (sum 64 = 128/2).
    pub mrope_section: [u32; 3],
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen3VlConfig {
    /// The released Qwen3-VL-4B-Instruct configuration.
    pub fn qwen3_vl_4b() -> Qwen3VlConfig {
        Qwen3VlConfig {
            vision: VisionConfig {
                depth: 24,
                hidden: 1024,
                num_heads: 16,
                intermediate: 4096,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                num_position_embeddings: 2304,
                out_hidden_size: 2560,
                in_channels: 3,
                deepstack_indexes: vec![5, 11, 17],
            },
            text: QwenConfig {
                vocab: 151936,
                block_size: 4096, // training seq len; overridden per-dataset
                n_layers: 36,
                d_model: 2560,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                d_ff: 9728,
                rope_theta: 5_000_000.0,
                rms_eps: 1e-6,
                tie_embeddings: true,
                qk_norm: true,
                attn_bias: false,
                lora: None,
            },
            mrope_section: [24, 20, 20],
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }

    /// Parse a Hugging Face `config.json` (`model_type: qwen3_vl`). Panics with a
    /// clear message on a missing/mistyped field — an import must fail loudly.
    pub fn from_hf(c: &Value) -> Qwen3VlConfig {
        let vc = &c["vision_config"];
        let tc = &c["text_config"];
        let u = |v: &Value, k: &str| -> u32 {
            v[k].as_u64().unwrap_or_else(|| panic!("qwen3-vl config: missing/!u64 field {k}")) as u32
        };
        let deepstack_indexes = vc["deepstack_visual_indexes"]
            .as_array()
            .expect("vision_config.deepstack_visual_indexes")
            .iter()
            .map(|x| x.as_u64().expect("deepstack index") as u32)
            .collect();
        let ms = tc["rope_scaling"]["mrope_section"]
            .as_array()
            .expect("text_config.rope_scaling.mrope_section");
        let mrope_section = [ms[0].as_u64().unwrap() as u32, ms[1].as_u64().unwrap() as u32, ms[2].as_u64().unwrap() as u32];
        Qwen3VlConfig {
            vision: VisionConfig {
                depth: u(vc, "depth"),
                hidden: u(vc, "hidden_size"),
                num_heads: u(vc, "num_heads"),
                intermediate: u(vc, "intermediate_size"),
                patch_size: u(vc, "patch_size"),
                temporal_patch_size: u(vc, "temporal_patch_size"),
                spatial_merge_size: u(vc, "spatial_merge_size"),
                num_position_embeddings: u(vc, "num_position_embeddings"),
                out_hidden_size: u(vc, "out_hidden_size"),
                in_channels: u(vc, "in_channels"),
                deepstack_indexes,
            },
            text: QwenConfig {
                vocab: u(tc, "vocab_size"),
                block_size: 4096,
                n_layers: u(tc, "num_hidden_layers"),
                d_model: u(tc, "hidden_size"),
                n_heads: u(tc, "num_attention_heads"),
                n_kv_heads: u(tc, "num_key_value_heads"),
                head_dim: u(tc, "head_dim"),
                d_ff: u(tc, "intermediate_size"),
                rope_theta: tc["rope_theta"].as_f64().expect("text_config.rope_theta") as f32,
                rms_eps: tc["rms_norm_eps"].as_f64().expect("text_config.rms_norm_eps") as f32,
                tie_embeddings: tc["tie_word_embeddings"].as_bool().unwrap_or(true),
                qk_norm: true,
                attn_bias: false,
                lora: None,
            },
            mrope_section,
            image_token_id: u(c, "image_token_id"),
            video_token_id: u(c, "video_token_id"),
            vision_start_token_id: u(c, "vision_start_token_id"),
            vision_end_token_id: u(c, "vision_end_token_id"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An inline copy of the released Qwen3-VL-4B-Instruct config.json structure
    /// (only the fields `from_hf` reads), so the parser test is hermetic.
    fn hf_4b_json() -> Value {
        serde_json::json!({
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "text_config": {
                "vocab_size": 151936,
                "num_hidden_layers": 36,
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 9728,
                "rope_theta": 5000000,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "rope_scaling": { "mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" }
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "num_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "in_channels": 3,
                "deepstack_visual_indexes": [5, 11, 17]
            }
        })
    }

    #[test]
    fn from_hf_matches_preset() {
        let parsed = Qwen3VlConfig::from_hf(&hf_4b_json());
        let preset = Qwen3VlConfig::qwen3_vl_4b();
        assert_eq!(parsed.vision, preset.vision);
        assert_eq!(parsed.mrope_section, preset.mrope_section);
        assert_eq!(parsed.image_token_id, preset.image_token_id);
        // Spot-check the reused Qwen3 text config.
        assert_eq!(parsed.text.d_model, 2560);
        assert_eq!(parsed.text.n_layers, 36);
        assert_eq!(parsed.text.head_dim, 128);
        assert_eq!(parsed.text.n_heads, 32);
        assert_eq!(parsed.text.n_kv_heads, 8);
        assert_eq!(parsed.text.d_ff, 9728);
        assert_eq!(parsed.text.vocab, 151936);
        assert_eq!(parsed.text.rope_theta, 5_000_000.0);
        assert!(parsed.text.tie_embeddings);
    }

    #[test]
    fn vision_derived_dims() {
        let v = Qwen3VlConfig::qwen3_vl_4b().vision;
        assert_eq!(v.head_dim(), 64); // 1024 / 16
        assert_eq!(v.merge_unit(), 4); // 2×2
        assert_eq!(v.pos_grid(), 48); // √2304
        assert_eq!(v.patch_vec_dim(), 1536); // 3·2·16·16
    }

    #[test]
    fn mrope_section_sums_to_half_head_dim() {
        let c = Qwen3VlConfig::qwen3_vl_4b();
        let sum: u32 = c.mrope_section.iter().sum();
        assert_eq!(sum, c.text.head_dim / 2, "M-RoPE section must cover head_dim/2");
    }
}
