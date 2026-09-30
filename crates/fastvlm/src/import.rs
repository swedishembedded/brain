// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map an `apple/FastVLM-*` checkpoint's tensor names onto brain's layout.
//!
//! The decoder is a Qwen2 (`model.layers.*` with q/k/v biases, no QK-norm), the
//! projector is `mlp2x_gelu` (`model.mm_projector.0`/`.2`), and the vision tower is
//! FastViTHD under `model.vision_tower.vision_tower.model.*`. Tied models
//! (0.5B/1.5B) also ship a `lm_head.weight` duplicating `embed_tokens`; the tied
//! loader uses `embed_tokens` as `tok.weight` and drops `lm_head`.

/// HF FastVLM decoder name → `qwen3::Qwen` parameter key, through the one
/// `qwen3::hf` name map (`cfg` is the checkpoint's Qwen2 decoder config, which
/// is what admits the q/k/v biases); `None` for anything that is not one of the
/// decoder's parameters (the vision tower, the projector, a tied `lm_head`).
pub fn map_decoder(hf: &str, cfg: &qwen3::QwenConfig) -> Option<String> {
    match qwen3::hf::HfNames::CAUSAL_LM.to_brain(hf, cfg) {
        qwen3::hf::HfTensor::Param(p) => Some(p),
        _ => None,
    }
}

/// HF `mlp2x_gelu` projector name → projector key (`fc1`/`fc2`).
pub fn map_projector(hf: &str) -> Option<String> {
    Some(match hf {
        "model.mm_projector.0.weight" => "fc1.weight",
        "model.mm_projector.0.bias" => "fc1.bias",
        "model.mm_projector.2.weight" => "fc2.weight",
        "model.mm_projector.2.bias" => "fc2.bias",
        _ => return None,
    }
    .to_string())
}

/// HF FastViTHD vision-tower name → the tower-relative key (prefix stripped). The
/// exact mapping onto the FastViTHD block builders is finalized with the encoder;
/// this identifies tower tensors so coverage can account for them.
pub fn map_vision(hf: &str) -> Option<String> {
    hf.strip_prefix("model.vision_tower.vision_tower.model.").map(String::from)
}

#[cfg(test)]
mod tests {

use brain_testutil::model_dir;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use super::*;

    #[test]
    fn decoder_names_map_with_bias() {
        let cfg = crate::config::FastVlmConfig::fastvlm_0_5b().decoder;
        assert_eq!(map_decoder("model.embed_tokens.weight", &cfg).unwrap(), "tok.weight");
        assert_eq!(map_decoder("model.norm.weight", &cfg).unwrap(), "norm.weight");
        assert_eq!(map_decoder("model.layers.0.input_layernorm.weight", &cfg).unwrap(), "blocks.0.ln1.weight");
        assert_eq!(map_decoder("model.layers.7.self_attn.q_proj.weight", &cfg).unwrap(), "blocks.7.attn.wq.weight");
        assert_eq!(map_decoder("model.layers.7.self_attn.q_proj.bias", &cfg).unwrap(), "blocks.7.attn.wq.bias");
        assert_eq!(map_decoder("model.layers.7.self_attn.v_proj.bias", &cfg).unwrap(), "blocks.7.attn.wv.bias");
        assert_eq!(map_decoder("model.layers.23.mlp.down_proj.weight", &cfg).unwrap(), "blocks.23.mlp.down.weight");
    }

    #[test]
    fn projector_and_vision_map() {
        assert_eq!(map_projector("model.mm_projector.0.weight").unwrap(), "fc1.weight");
        assert_eq!(map_projector("model.mm_projector.2.bias").unwrap(), "fc2.bias");
        assert_eq!(
            map_vision("model.vision_tower.vision_tower.model.network.0.0.token_mixer.reparam_conv.weight").unwrap(),
            "network.0.0.token_mixer.reparam_conv.weight"
        );
    }

    /// Read only the safetensors JSON header of the real checkpoint (no tensor
    /// data) and check the decoder covers every Qwen2-0.5B parameter + the four
    /// projector tensors. Skips if the checkpoint isn't present.
    #[test]
    fn real_checkpoint_decoder_and_projector_covered() {
        let path = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        if !std::path::Path::new(&path).is_file() {
            brain_testutil::skip("FastVLM checkpoint not present");
            return;
        }
        let names: Vec<String> = checkpoint::mmap::MmapSafetensors::open(&path).unwrap().names().to_vec();

        let cfg = crate::config::FastVlmConfig::fastvlm_0_5b();
        let decoder: std::collections::HashSet<String> = names.iter().filter_map(|n| map_decoder(n, &cfg.decoder)).collect();
        let projector: Vec<String> = names.iter().filter_map(|n| map_projector(n)).collect();

        for (name, _) in cfg.decoder.param_list() {
            assert!(decoder.contains(&name), "decoder param not imported: {name}");
        }
        assert_eq!(projector.len(), 4, "projector: fc1.{{weight,bias}} + fc2.{{weight,bias}}");
        // The tower has tensors too (mapped in detail with the encoder).
        assert!(names.iter().any(|n| map_vision(n).is_some()), "vision tower present");
    }
}
