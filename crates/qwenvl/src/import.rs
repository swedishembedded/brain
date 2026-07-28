// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map a Hugging Face Qwen3-VL checkpoint's tensor names onto brain's parameter
//! layout and partition them into the four weight sets `Qwen3Vl` consumes: the
//! ViT encoder, the main PatchMerger, the per-tap DeepStack mergers, and the Qwen
//! decoder.
//!
//! The HF names are `model.visual.*` (vision) and `model.language_model.*`
//! (decoder). The patch-embed is a `Conv3d[hidden, in_ch, temporal, patch, patch]`
//! whose contiguous flatten is exactly `[hidden, in_ch·temporal·patch²]` in the
//! `[channel, temporal, patch_h, patch_w]` order our `pack_patches` produces, so
//! it maps by name only (no transpose). Decoder names mirror `qwen::import` but
//! under the extra `language_model.` prefix; tied embeddings mean `embed_tokens`
//! serves as both `tok.weight` and the head.

use std::collections::HashMap;

/// Shared merger leaf mapping (main merger + DeepStack mergers share the shape).
fn merger_leaf(leaf: &str) -> Option<&'static str> {
    Some(match leaf {
        "norm.weight" => "ln.weight",
        "norm.bias" => "ln.bias",
        "linear_fc1.weight" => "fc1.weight",
        "linear_fc1.bias" => "fc1.bias",
        "linear_fc2.weight" => "fc2.weight",
        "linear_fc2.bias" => "fc2.bias",
        _ => return None,
    })
}

/// HF vision block / patch-embed / pos-embed name → `VisionEncoder` key.
/// Returns `None` for merger / deepstack names (handled separately) and for
/// tensors with no brain counterpart (e.g. rotary inv-freq buffers).
pub fn map_vision(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("model.visual.")?;
    if s.starts_with("merger.") || s.starts_with("deepstack_merger_list.") {
        return None;
    }
    match s {
        "patch_embed.proj.weight" => return Some("patch_embed.weight".into()),
        "patch_embed.proj.bias" => return Some("patch_embed.bias".into()),
        "pos_embed.weight" => return Some("pos_embed".into()),
        _ => {}
    }
    let (n, leaf) = s.strip_prefix("blocks.")?.split_once('.')?;
    let mapped = match leaf {
        "norm1.weight" | "norm1.bias" | "norm2.weight" | "norm2.bias" => leaf,
        "attn.qkv.weight" => "qkv.weight",
        "attn.qkv.bias" => "qkv.bias",
        "attn.proj.weight" => "proj.weight",
        "attn.proj.bias" => "proj.bias",
        "mlp.linear_fc1.weight" => "fc1.weight",
        "mlp.linear_fc1.bias" => "fc1.bias",
        "mlp.linear_fc2.weight" => "fc2.weight",
        "mlp.linear_fc2.bias" => "fc2.bias",
        _ => return None,
    };
    Some(format!("blocks.{n}.{mapped}"))
}

/// HF main-merger name → PatchMerger key.
pub fn map_main_merger(hf: &str) -> Option<String> {
    let leaf = hf.strip_prefix("model.visual.merger.")?;
    merger_leaf(leaf).map(String::from)
}

/// HF DeepStack-merger name → (tap index, PatchMerger key).
pub fn map_deepstack(hf: &str) -> Option<(usize, String)> {
    let (k, leaf) = hf.strip_prefix("model.visual.deepstack_merger_list.")?.split_once('.')?;
    let idx: usize = k.parse().ok()?;
    Some((idx, merger_leaf(leaf).map(String::from)?))
}

/// HF decoder name → `qwen::Qwen` parameter key.
pub fn map_decoder(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("model.language_model.")?;
    match s {
        "embed_tokens.weight" => return Some("tok.weight".into()),
        "norm.weight" => return Some("norm.weight".into()),
        _ => {}
    }
    let (n, leaf) = s.strip_prefix("layers.")?.split_once('.')?;
    let mapped = match leaf {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "self_attn.q_norm.weight" => "attn.q_norm.weight",
        "self_attn.k_norm.weight" => "attn.k_norm.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{mapped}"))
}

/// The four brain weight sets partitioned from an HF checkpoint.
pub struct ImportedWeights {
    pub vision: HashMap<String, Vec<f32>>,
    pub main_merger: HashMap<String, Vec<f32>>,
    pub deepstack: Vec<HashMap<String, Vec<f32>>>,
    pub decoder: HashMap<String, Vec<f32>>,
}

/// Partition a name→tensor map (already dequantized to f32) into the four brain
/// weight sets. `n_deepstack` is the number of DeepStack taps
/// (`vision.deepstack_indexes.len()`). Unmapped tensors are skipped.
pub fn partition(hf: HashMap<String, Vec<f32>>, n_deepstack: usize) -> ImportedWeights {
    let mut w = ImportedWeights {
        vision: HashMap::new(),
        main_merger: HashMap::new(),
        deepstack: (0..n_deepstack).map(|_| HashMap::new()).collect(),
        decoder: HashMap::new(),
    };
    for (name, data) in hf {
        if let Some(m) = map_vision(&name) {
            w.vision.insert(m, data);
        } else if let Some(m) = map_main_merger(&name) {
            w.main_merger.insert(m, data);
        } else if let Some((k, m)) = map_deepstack(&name) {
            if k < w.deepstack.len() {
                w.deepstack[k].insert(m, data);
            }
        } else if let Some(m) = map_decoder(&name) {
            w.decoder.insert(m, data);
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_names_map() {
        assert_eq!(map_vision("model.visual.patch_embed.proj.weight").unwrap(), "patch_embed.weight");
        assert_eq!(map_vision("model.visual.patch_embed.proj.bias").unwrap(), "patch_embed.bias");
        assert_eq!(map_vision("model.visual.pos_embed.weight").unwrap(), "pos_embed");
        assert_eq!(map_vision("model.visual.blocks.5.attn.qkv.weight").unwrap(), "blocks.5.qkv.weight");
        assert_eq!(map_vision("model.visual.blocks.5.attn.proj.bias").unwrap(), "blocks.5.proj.bias");
        assert_eq!(map_vision("model.visual.blocks.11.mlp.linear_fc1.weight").unwrap(), "blocks.11.fc1.weight");
        assert_eq!(map_vision("model.visual.blocks.0.norm2.bias").unwrap(), "blocks.0.norm2.bias");
        // merger/deepstack are not vision
        assert!(map_vision("model.visual.merger.norm.weight").is_none());
        assert!(map_vision("model.visual.deepstack_merger_list.0.norm.weight").is_none());
    }

    #[test]
    fn merger_names_map() {
        assert_eq!(map_main_merger("model.visual.merger.norm.weight").unwrap(), "ln.weight");
        assert_eq!(map_main_merger("model.visual.merger.linear_fc1.weight").unwrap(), "fc1.weight");
        assert_eq!(map_main_merger("model.visual.merger.linear_fc2.bias").unwrap(), "fc2.bias");
        assert_eq!(map_deepstack("model.visual.deepstack_merger_list.2.linear_fc2.weight").unwrap(), (2, "fc2.weight".into()));
        assert_eq!(map_deepstack("model.visual.deepstack_merger_list.0.norm.bias").unwrap(), (0, "ln.bias".into()));
    }

    #[test]
    fn decoder_names_map() {
        assert_eq!(map_decoder("model.language_model.embed_tokens.weight").unwrap(), "tok.weight");
        assert_eq!(map_decoder("model.language_model.norm.weight").unwrap(), "norm.weight");
        assert_eq!(map_decoder("model.language_model.layers.0.input_layernorm.weight").unwrap(), "blocks.0.ln1.weight");
        assert_eq!(map_decoder("model.language_model.layers.7.self_attn.q_proj.weight").unwrap(), "blocks.7.attn.wq.weight");
        assert_eq!(map_decoder("model.language_model.layers.7.self_attn.q_norm.weight").unwrap(), "blocks.7.attn.q_norm.weight");
        assert_eq!(map_decoder("model.language_model.layers.35.mlp.down_proj.weight").unwrap(), "blocks.35.mlp.down.weight");
    }

    #[test]
    fn partition_routes_each_group() {
        let mut hf = HashMap::new();
        hf.insert("model.visual.patch_embed.proj.bias".to_string(), vec![1.0]);
        hf.insert("model.visual.blocks.0.norm1.weight".to_string(), vec![2.0]);
        hf.insert("model.visual.merger.linear_fc1.bias".to_string(), vec![3.0]);
        hf.insert("model.visual.deepstack_merger_list.1.norm.weight".to_string(), vec![4.0]);
        hf.insert("model.language_model.layers.0.self_attn.o_proj.weight".to_string(), vec![5.0]);
        hf.insert("model.visual.rotary_pos_emb.inv_freq".to_string(), vec![9.0]); // unmapped
        let w = partition(hf, 3);
        assert_eq!(w.vision["patch_embed.bias"], vec![1.0]);
        assert_eq!(w.vision["blocks.0.norm1.weight"], vec![2.0]);
        assert_eq!(w.main_merger["fc1.bias"], vec![3.0]);
        assert_eq!(w.deepstack[1]["ln.weight"], vec![4.0]);
        assert_eq!(w.decoder["blocks.0.attn.wo.weight"], vec![5.0]);
        assert_eq!(w.vision.len() + w.main_merger.len() + w.deepstack.iter().map(|m| m.len()).sum::<usize>() + w.decoder.len(), 5);
    }
}
