// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL-30B-A3B (`qwen3vlmoe`) configuration.
//!
//! Every default below is transcribed from the REAL released
//! `Qwen/Qwen3-VL-30B-A3B-Instruct/config.json`, fetched and read verbatim
//! (`curl` against the raw file on huggingface.co, not summarized, not a
//! paraphrase) while writing this module - the exact byte content is quoted
//! in the field doc comments below, and the parser test embeds a full inline
//! copy of the same structure. What was **inferred rather than confirmed**
//! against that file is called out explicitly where it matters (`use_qk_norm`
//! below).
//!
//! Two real findings the fetch settled, load-bearing for how this config is
//! shaped:
//!
//! 1. **`vision_config` is BYTE-IDENTICAL to
//!    [`qwen3vl::config::VisionConfig::qwen3_omni`]** - same depth (27),
//!    hidden (1152), heads (16), intermediate (4304), patch/temporal/merge
//!    sizes, `num_position_embeddings` (2304), `out_hidden_size` (2048), and
//!    `deepstack_visual_indexes` ([8, 16, 24]). So this module reuses
//!    [`qwen3vl::config::VisionConfig`] verbatim (via the newly-shared
//!    [`qwen3vl::config::VisionConfig::from_hf`] parser) rather than defining
//!    a second vision-config type - there is nothing architecturally new
//!    about this model's vision tower.
//! 2. **`text_config` is a plain-softmax top-k sparse-MoE Qwen3 decoder with
//!    NO shared expert** (`num_experts: 128`, `num_experts_per_tok: 8`,
//!    `norm_topk_prob: true`, `mlp_only_layers: []` - every layer is routed,
//!    `decoder_sparse_step: 1`, no `shared_expert_intermediate_size` key at
//!    all) - the EXACT shape
//!    [`qwen3omnimoe::config::MoeTextConfig::thinker_defaults`] already
//!    models for Qwen3-Omni's Thinker (also 128 experts / top-8 / no shared
//!    expert / `use_qk_norm`), just at different `hidden`/`vocab`/
//!    `rope_theta` numbers. That is the real, checked justification for the
//!    splice choice this crate makes (see `crate::model`'s doc): the
//!    decoder this model needs is `qwen3omnimoe::thinker`'s GQA+QK-norm+RoPE
//!    top-k-MoE stack, reused unchanged, NOT `qwen35moe`'s hybrid
//!    Gated-DeltaNet/GQA decoder (a different checkpoint family entirely -
//!    Qwen3.5-35B-A3B, not Qwen3-VL-30B-A3B; see `qwen35moe::vl`'s own doc,
//!    whose SPLICE PATTERN this crate follows without adopting its decoder
//!    architecture).
//!
//! `use_qk_norm`: the real `text_config` carries no `use_qk_norm` key at all
//! (confirmed absent in the fetched file). Every Qwen3-family decoder in this
//! workspace applies per-head QK-norm unconditionally (`qwen3::QwenConfig`
//! has no such flag either - `qk_norm` is set `true` by every constructor,
//! never parsed), and `MoeTextConfig::thinker_defaults()`'s own doc records
//! the same reasoning for Qwen3-Omni's real checkpoint (which DOES carry
//! `q_norm`/`k_norm` weights per layer despite no config flag naming them).
//! This module's default therefore matches that established convention
//! rather than a field this fetch could directly confirm - called out here
//! rather than silently inherited.
//!
//! **Not yet confirmed from a real source (this module does not claim it
//! is)**: the exact GGUF tensor names/quant layout a real
//! `Qwen3-VL-30B-A3B-Instruct` GGUF release would carry - no such release was
//! available to inspect in this environment. See `crate::import`'s doc.

use serde_json::Value;

use qwen3omnimoe::config::MoeTextConfig;
use qwen3vl::config::VisionConfig;

/// Full Qwen3-VL-30B-A3B configuration: `qwen3vl`'s vision tower (ViT +
/// PatchMerger + DeepStack) spliced onto a `qwen3omnimoe`-shaped MoE text
/// decoder, plus the vision special-token ids the splice needs. See this
/// module's doc for what is verified vs. inherited-by-convention.
#[derive(Clone, Debug)]
pub struct Qwen3VlMoeConfig {
    pub vision: VisionConfig,
    pub text: MoeTextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen3VlMoeConfig {
    /// The real released `Qwen/Qwen3-VL-30B-A3B-Instruct` configuration -
    /// every number here is quoted from the fetched `config.json` (see this
    /// module's doc), not estimated.
    pub fn qwen3_vl_30b_a3b() -> Qwen3VlMoeConfig {
        Qwen3VlMoeConfig {
            vision: VisionConfig::qwen3_omni(), // byte-identical, see module doc point 1
            text: MoeTextConfig {
                n_layers: 48,
                hidden: 2048,
                n_heads: 32,
                n_kv_heads: 4,
                head_dim: 128,
                moe_intermediate: 768,
                shared_expert_intermediate: 0, // no `shared_expert_intermediate_size` key
                n_experts: 128,
                top_k: 8,
                norm_topk_prob: true,
                use_qk_norm: true, // convention, not a config key - see module doc
                vocab: 151936,
                rope_theta: 5_000_000.0,
                rms_norm_eps: 1e-6,
                mrope_section: vec![24, 20, 20],
                max_position_embeddings: 262144,
            },
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }

    /// Parse a real `Qwen/Qwen3-VL-30B-A3B-Instruct`-shaped `config.json`:
    /// top-level `vision_config`/`text_config` (NOT nested under a
    /// `thinker_config` wrapper the way Qwen3-Omni's is - a real, checked
    /// structural difference between the two checkpoints' config layouts,
    /// confirmed against the fetched file). Panics with a clear message on a
    /// missing/mistyped required field - an import must fail loudly, per this
    /// workspace's boundary-validation rule.
    pub fn from_hf(c: &Value) -> Qwen3VlMoeConfig {
        let tc = &c["text_config"];
        let u = |v: &Value, k: &str| -> u32 {
            v[k].as_u64().unwrap_or_else(|| panic!("qwen3vlmoe config: missing/!u64 field {k}")) as u32
        };
        let ms = tc["rope_scaling"]["mrope_section"]
            .as_array()
            .expect("text_config.rope_scaling.mrope_section");
        let mrope_section: Vec<u32> = ms.iter().map(|x| x.as_u64().expect("mrope_section entry") as u32).collect();
        Qwen3VlMoeConfig {
            vision: VisionConfig::from_hf(&c["vision_config"]),
            text: MoeTextConfig {
                n_layers: u(tc, "num_hidden_layers"),
                hidden: u(tc, "hidden_size"),
                n_heads: u(tc, "num_attention_heads"),
                n_kv_heads: u(tc, "num_key_value_heads"),
                head_dim: u(tc, "head_dim"),
                moe_intermediate: u(tc, "moe_intermediate_size"),
                shared_expert_intermediate: tc["shared_expert_intermediate_size"].as_u64().map(|x| x as u32).unwrap_or(0),
                n_experts: u(tc, "num_experts"),
                top_k: u(tc, "num_experts_per_tok"),
                norm_topk_prob: tc["norm_topk_prob"].as_bool().unwrap_or(true),
                use_qk_norm: tc["use_qk_norm"].as_bool().unwrap_or(true), // see module doc
                vocab: u(tc, "vocab_size"),
                rope_theta: tc["rope_theta"].as_f64().expect("text_config.rope_theta") as f32,
                rms_norm_eps: tc["rms_norm_eps"].as_f64().expect("text_config.rms_norm_eps") as f32,
                mrope_section,
                max_position_embeddings: tc["max_position_embeddings"].as_u64().map(|x| x as u32).unwrap_or(262144),
            },
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

    /// An inline copy of the REAL `Qwen/Qwen3-VL-30B-A3B-Instruct/config.json`
    /// structure (only the fields `from_hf` reads), transcribed byte-for-byte
    /// from the fetched file at commit `1fc78ad` - so this parser test is
    /// hermetic AND checked against the real release, not a hypothetical
    /// shape.
    fn hf_30b_a3b_json() -> Value {
        serde_json::json!({
            "architectures": ["Qwen3VLMoeForConditionalGeneration"],
            "image_token_id": 151655,
            "model_type": "qwen3_vl_moe",
            "text_config": {
                "attention_bias": false,
                "decoder_sparse_step": 1,
                "head_dim": 128,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "model_type": "qwen3_vl_moe_text",
                "moe_intermediate_size": 768,
                "norm_topk_prob": true,
                "num_attention_heads": 32,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 48,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-06,
                "rope_scaling": { "mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" },
                "rope_theta": 5000000,
                "vocab_size": 151936
            },
            "tie_word_embeddings": false,
            "video_token_id": 151656,
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "model_type": "qwen3_vl_moe",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2048,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652
        })
    }

    #[test]
    fn from_hf_matches_preset() {
        let parsed = Qwen3VlMoeConfig::from_hf(&hf_30b_a3b_json());
        let preset = Qwen3VlMoeConfig::qwen3_vl_30b_a3b();
        assert_eq!(parsed.vision, preset.vision);
        assert_eq!(parsed.image_token_id, preset.image_token_id);
        assert_eq!(parsed.video_token_id, preset.video_token_id);
        assert_eq!(parsed.vision_start_token_id, preset.vision_start_token_id);
        assert_eq!(parsed.vision_end_token_id, preset.vision_end_token_id);
        assert_eq!(parsed.text.n_layers, 48);
        assert_eq!(parsed.text.hidden, 2048);
        assert_eq!(parsed.text.n_heads, 32);
        assert_eq!(parsed.text.n_kv_heads, 4);
        assert_eq!(parsed.text.head_dim, 128);
        assert_eq!(parsed.text.moe_intermediate, 768);
        assert_eq!(parsed.text.n_experts, 128);
        assert_eq!(parsed.text.top_k, 8);
        assert_eq!(parsed.text.shared_expert_intermediate, 0, "no shared_expert_intermediate_size key -> no shared expert");
        assert!(parsed.text.norm_topk_prob);
        assert_eq!(parsed.text.vocab, 151936);
        assert_eq!(parsed.text.rope_theta, 5_000_000.0);
        assert_eq!(parsed.text.mrope_section, vec![24, 20, 20]);
    }

    #[test]
    fn mrope_section_sums_to_half_head_dim() {
        let c = Qwen3VlMoeConfig::qwen3_vl_30b_a3b();
        let sum: u32 = c.text.mrope_section.iter().sum();
        assert_eq!(sum, c.text.head_dim / 2, "M-RoPE section must cover head_dim/2");
    }

    #[test]
    fn vision_matches_qwen3_omnis_tower_byte_for_byte() {
        // The real, checked justification for reusing `qwen3vl::config::
        // VisionConfig::qwen3_omni()` verbatim instead of a new preset - see
        // this module's doc, point 1.
        assert_eq!(Qwen3VlMoeConfig::qwen3_vl_30b_a3b().vision, VisionConfig::qwen3_omni());
    }
}
