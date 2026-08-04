// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map an `apple/FastVLM-*` checkpoint's tensor names onto brain's layout.
//!
//! The decoder is a Qwen2 (`model.layers.*` with q/k/v biases, no QK-norm), the
//! projector is `mlp2x_gelu` (`model.mm_projector.0`/`.2`), and the vision tower is
//! FastViTHD under `model.vision_tower.vision_tower.model.*`. Tied models
//! (0.5B/1.5B) also ship a `lm_head.weight` duplicating `embed_tokens`; the tied
//! loader uses `embed_tokens` as `tok.weight` and drops `lm_head`.

/// HF FastVLM decoder name → `qwen::Qwen` (Qwen2 config) parameter key.
pub fn map_decoder(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("tok.weight".into()),
        "model.norm.weight" => return Some("norm.weight".into()),
        "lm_head.weight" => return Some("lm_head.weight".into()), // untied models only
        _ => {}
    }
    let (n, leaf) = hf.strip_prefix("model.layers.")?.split_once('.')?;
    let mapped = match leaf {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.q_proj.bias" => "attn.wq.bias",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.k_proj.bias" => "attn.wk.bias",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.v_proj.bias" => "attn.wv.bias",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{mapped}"))
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
        assert_eq!(map_decoder("model.embed_tokens.weight").unwrap(), "tok.weight");
        assert_eq!(map_decoder("model.norm.weight").unwrap(), "norm.weight");
        assert_eq!(map_decoder("model.layers.0.input_layernorm.weight").unwrap(), "blocks.0.ln1.weight");
        assert_eq!(map_decoder("model.layers.7.self_attn.q_proj.weight").unwrap(), "blocks.7.attn.wq.weight");
        assert_eq!(map_decoder("model.layers.7.self_attn.q_proj.bias").unwrap(), "blocks.7.attn.wq.bias");
        assert_eq!(map_decoder("model.layers.7.self_attn.v_proj.bias").unwrap(), "blocks.7.attn.wv.bias");
        assert_eq!(map_decoder("model.layers.23.mlp.down_proj.weight").unwrap(), "blocks.23.mlp.down.weight");
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
        use std::io::Read;
        let path = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        let Ok(mut f) = std::fs::File::open(path) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        let mut len = [0u8; 8];
        f.read_exact(&mut len).unwrap();
        let n = u64::from_le_bytes(len) as usize;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        let names: Vec<String> = hdr.as_object().unwrap().keys().filter(|k| *k != "__metadata__").cloned().collect();

        let decoder: std::collections::HashSet<String> = names.iter().filter_map(|n| map_decoder(n)).collect();
        let projector: Vec<String> = names.iter().filter_map(|n| map_projector(n)).collect();

        let cfg = crate::config::FastVlmConfig::fastvlm_0_5b();
        for (name, _) in cfg.decoder.param_list() {
            assert!(decoder.contains(&name), "decoder param not imported: {name}");
        }
        assert_eq!(projector.len(), 4, "projector: fc1.{{weight,bias}} + fc2.{{weight,bias}}");
        // The tower has tensors too (mapped in detail with the encoder).
        assert!(names.iter().any(|n| map_vision(n).is_some()), "vision tower present");
    }
}
