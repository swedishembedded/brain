// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map a `liuhaotian/llava-v1.5-13b` (`LlavaLlamaForCausalLM`) checkpoint's
//! tensor names onto brain's layout.
//!
//! Three families, the `apple/FastVLM` importer's own three-way split (see
//! `crates/fastvlm/src/import.rs`) - a decoder (here: LLaMA-2, no qkv/mlp
//! bias, plain MHA), an `mlp2x_gelu` projector (`model.mm_projector.0`/`.2`,
//! byte-identical convention to FastVLM's), and a CLIP-L vision tower under
//! `model.vision_tower.vision_tower.vision_model.*` (HF's own `CLIPVisionModel`
//! naming, including its `pre_layrnorm` typo - transcribed as shipped, not
//! "corrected").
//!
//! **Not yet exercised against real checkpoint bytes.** No `resources/llava/`
//! checkpoint was available this session (LLaVA-1.5-13B is a ~26 GB fp16 /
//! ~52 GB fp32 download, well past what fetching a tokenizer-sized asset
//! costs) - a stated, honest gap, not a silently skipped one. What IS
//! verified here: every mapper is
//! total over [`clip::config::ClipVisionConfig::tensor_manifest`] /
//! [`qwen3::config::QwenConfig::param_list`] / the four projector tensors on
//! a SYNTHETIC checkpoint built from those exact manifests (the
//! weight-free "mapping-units" rung), so a real safetensors header, once
//! obtained, only needs the coverage assertion re-run, not new code.

/// The vision tower's HF prefix (`model.vision_tower.vision_tower.` - the
/// outer name is the `LlavaMetaModel.vision_tower` attribute, the inner is
/// `CLIPVisionTower.vision_tower`'s own `CLIPVisionModel`).
const VISION_PREFIX: &str = "model.vision_tower.vision_tower.vision_model.";

/// One of the three q/k/v projections a CLIP block's `self_attn` carries -
/// brain fuses these into one `attn.qkv.weight`/`.bias` at import time (the
/// same fusion every other CLIP importer in this workspace performs), so this
/// mapper returns the (layer, part) coordinate rather than a brain-side name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QkvPart {
    Q,
    K,
    V,
}

/// HF LLaVA/Vicuna decoder name -> `qwen3::Qwen` parameter key, through the
/// one `qwen3::hf` name map keyed on the checkpoint's Llama decoder config;
/// `None` for anything that is not one of the decoder's parameters (the
/// vision tower, the projector, a serialized `rotary_emb.inv_freq`).
pub fn map_decoder(hf: &str, cfg: &qwen3::QwenConfig) -> Option<String> {
    match qwen3::hf::HfNames::CAUSAL_LM.to_brain(hf, cfg) {
        qwen3::hf::HfTensor::Param(p) => Some(p),
        _ => None,
    }
}

/// HF `mlp2x_gelu` projector name -> projector key (`fc1`/`fc2`). Byte-identical
/// convention to FastVLM's `map_projector` - both are `nn.Sequential(Linear,
/// GELU, Linear)`, indices 0 and 2.
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

/// HF CLIP vision-tower name -> brain `ClipVisionConfig::tensor_manifest`
/// key, for every tensor EXCEPT the three per-block attention projections
/// (see [`map_vision_qkv`] for those - they need fusing, not renaming).
pub fn map_vision(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix(VISION_PREFIX)?;
    match rest {
        "embeddings.class_embedding" => return Some("class_embed".into()),
        "embeddings.patch_embedding.weight" => return Some("patch_embed.weight".into()),
        "embeddings.position_embedding.weight" => return Some("pos_embed".into()),
        // HF's own typo, transcribed as shipped - the CLIP-L336 checkpoint's
        // pre-encoder LayerNorm (this vanilla tower's `pre_norm`).
        "pre_layrnorm.weight" => return Some("pre_norm.weight".into()),
        "pre_layrnorm.bias" => return Some("pre_norm.bias".into()),
        _ => {}
    }
    let (n, leaf) = rest.strip_prefix("encoder.layers.")?.split_once('.')?;
    let mapped = match leaf {
        "layer_norm1.weight" => "norm1.weight",
        "layer_norm1.bias" => "norm1.bias",
        "self_attn.out_proj.weight" => "attn.proj.weight",
        "self_attn.out_proj.bias" => "attn.proj.bias",
        "layer_norm2.weight" => "norm2.weight",
        "layer_norm2.bias" => "norm2.bias",
        "mlp.fc1.weight" => "mlp.fc1.weight",
        "mlp.fc1.bias" => "mlp.fc1.bias",
        "mlp.fc2.weight" => "mlp.fc2.weight",
        "mlp.fc2.bias" => "mlp.fc2.bias",
        _ => return None,
    };
    Some(format!("blocks.{n}.{mapped}"))
}

/// The `(layer, part)` an HF `self_attn.{q,k,v}_proj.{weight,bias}` tensor
/// belongs to, for the caller to collect all three before concatenating into
/// `blocks.N.attn.qkv.{weight,bias}` (`[3*d, d]` / `[3*d]`, q then k then v -
/// the same row order [`clip::config::ClipVisionConfig::tensor_manifest`]
/// declares and every other CLIP importer in this workspace fuses to).
pub fn map_vision_qkv(hf: &str) -> Option<(u32, QkvPart, bool)> {
    let rest = hf.strip_prefix(VISION_PREFIX)?;
    let (n, leaf) = rest.strip_prefix("encoder.layers.")?.split_once('.')?;
    let n: u32 = n.parse().ok()?;
    let (part, is_bias) = match leaf {
        "self_attn.q_proj.weight" => (QkvPart::Q, false),
        "self_attn.q_proj.bias" => (QkvPart::Q, true),
        "self_attn.k_proj.weight" => (QkvPart::K, false),
        "self_attn.k_proj.bias" => (QkvPart::K, true),
        "self_attn.v_proj.weight" => (QkvPart::V, false),
        "self_attn.v_proj.bias" => (QkvPart::V, true),
        _ => return None,
    };
    Some((n, part, is_bias))
}

/// Concatenate three per-head-group projections `[d, k]` (or biases `[d]`)
/// into one fused `[3*d, k]` (or `[3*d]`) tensor, q then k then v - the shape
/// `Builder::cross_attn`/`vit_block_fwd_cached` expect and every other CLIP
/// tower in this workspace is imported as.
pub fn fuse_qkv(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(q.len(), k.len());
    assert_eq!(k.len(), v.len());
    let mut out = Vec::with_capacity(q.len() * 3);
    out.extend_from_slice(q);
    out.extend_from_slice(k);
    out.extend_from_slice(v);
    out
}

/// Collect a real checkpoint's vision-tower tensors (`checkpoint::safetensors::read`'s
/// output) into brain's `ClipVisionConfig::tensor_manifest` naming, fusing
/// each block's separate `q_proj`/`k_proj`/`v_proj` into one `attn.qkv.*`
/// (see [`fuse_qkv`]) along the way - the caller ([`crate::caps::load_vision`])
/// never sees the three-tensor intermediate.
pub fn build_vision_weights(tensors: &[checkpoint::safetensors::StTensor]) -> std::collections::HashMap<String, Vec<f32>> {
    use std::collections::HashMap;

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    // (layer, part) -> the tensor's data, collected before fusing.
    let mut qkv_w: HashMap<(u32, QkvPart), &[f32]> = HashMap::new();
    let mut qkv_b: HashMap<(u32, QkvPart), &[f32]> = HashMap::new();

    for t in tensors {
        if let Some(name) = map_vision(&t.name) {
            out.insert(name, t.data.clone());
            continue;
        }
        if let Some((layer, part, is_bias)) = map_vision_qkv(&t.name) {
            if is_bias {
                qkv_b.insert((layer, part), &t.data);
            } else {
                qkv_w.insert((layer, part), &t.data);
            }
        }
    }

    let layers: std::collections::BTreeSet<u32> = qkv_w.keys().map(|(l, _)| *l).collect();
    for l in layers {
        if let (Some(q), Some(k), Some(v)) = (qkv_w.get(&(l, QkvPart::Q)), qkv_w.get(&(l, QkvPart::K)), qkv_w.get(&(l, QkvPart::V))) {
            out.insert(format!("blocks.{l}.attn.qkv.weight"), fuse_qkv(q, k, v));
        }
        if let (Some(q), Some(k), Some(v)) = (qkv_b.get(&(l, QkvPart::Q)), qkv_b.get(&(l, QkvPart::K)), qkv_b.get(&(l, QkvPart::V))) {
            out.insert(format!("blocks.{l}.attn.qkv.bias"), fuse_qkv(q, k, v));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clip::config::ClipVisionConfig;
    use qwen3::config::QwenConfig;

    #[test]
    fn decoder_names_map_with_no_bias_and_plain_mha() {
        let cfg = QwenConfig::llama2_13b();
        assert_eq!(map_decoder("model.embed_tokens.weight", &cfg).unwrap(), "tok.weight");
        assert_eq!(map_decoder("model.norm.weight", &cfg).unwrap(), "norm.weight");
        assert_eq!(map_decoder("lm_head.weight", &cfg).unwrap(), "lm_head.weight");
        assert_eq!(map_decoder("model.layers.0.input_layernorm.weight", &cfg).unwrap(), "blocks.0.ln1.weight");
        assert_eq!(map_decoder("model.layers.39.self_attn.q_proj.weight", &cfg).unwrap(), "blocks.39.attn.wq.weight");
        assert_eq!(map_decoder("model.layers.7.mlp.down_proj.weight", &cfg).unwrap(), "blocks.7.mlp.down.weight");
        // LLaMA-2 carries no attention bias - a bias tensor name (had one ever
        // appeared) must not silently map onto a weight leaf.
        assert!(map_decoder("model.layers.0.self_attn.q_proj.bias", &cfg).is_none());
        assert!(map_decoder("model.layers.0.self_attn.rotary_emb.inv_freq", &cfg).is_none());
    }

    #[test]
    fn projector_maps_the_four_mlp2x_gelu_tensors() {
        assert_eq!(map_projector("model.mm_projector.0.weight").unwrap(), "fc1.weight");
        assert_eq!(map_projector("model.mm_projector.0.bias").unwrap(), "fc1.bias");
        assert_eq!(map_projector("model.mm_projector.2.weight").unwrap(), "fc2.weight");
        assert_eq!(map_projector("model.mm_projector.2.bias").unwrap(), "fc2.bias");
        assert!(map_projector("model.mm_projector.1.something").is_none());
    }

    #[test]
    fn vision_stem_and_blocks_map_including_the_pre_layrnorm_typo() {
        let p = |s: &str| format!("{VISION_PREFIX}{s}");
        assert_eq!(map_vision(&p("embeddings.class_embedding")).unwrap(), "class_embed");
        assert_eq!(map_vision(&p("embeddings.patch_embedding.weight")).unwrap(), "patch_embed.weight");
        assert_eq!(map_vision(&p("embeddings.position_embedding.weight")).unwrap(), "pos_embed");
        assert_eq!(map_vision(&p("pre_layrnorm.weight")).unwrap(), "pre_norm.weight");
        assert_eq!(map_vision(&p("encoder.layers.0.layer_norm1.weight")).unwrap(), "blocks.0.norm1.weight");
        assert_eq!(map_vision(&p("encoder.layers.23.mlp.fc2.bias")).unwrap(), "blocks.23.mlp.fc2.bias");
        assert_eq!(map_vision(&p("encoder.layers.5.self_attn.out_proj.weight")).unwrap(), "blocks.5.attn.proj.weight");
        // The qkv projections are NOT mapped here - see map_vision_qkv.
        assert!(map_vision(&p("encoder.layers.0.self_attn.q_proj.weight")).is_none());
        // A tensor outside the vision prefix is not this mapper's business.
        assert!(map_vision("model.embed_tokens.weight").is_none());
    }

    #[test]
    fn vision_qkv_resolves_layer_part_and_bias() {
        let p = |s: &str| format!("{VISION_PREFIX}{s}");
        assert_eq!(map_vision_qkv(&p("encoder.layers.3.self_attn.q_proj.weight")).unwrap(), (3, QkvPart::Q, false));
        assert_eq!(map_vision_qkv(&p("encoder.layers.3.self_attn.k_proj.bias")).unwrap(), (3, QkvPart::K, true));
        assert_eq!(map_vision_qkv(&p("encoder.layers.22.self_attn.v_proj.weight")).unwrap(), (22, QkvPart::V, false));
        assert!(map_vision_qkv(&p("encoder.layers.0.self_attn.out_proj.weight")).is_none());
    }

    #[test]
    fn fuse_qkv_concatenates_in_qkv_order() {
        let got = fuse_qkv(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Weight-free "mapping-units" rung (the `supir::import` precedent):
    /// every declared decoder parameter and vision-tower tensor is reachable
    /// from a plausible HF name this reader would actually see, i.e. the
    /// reverse of `map_decoder`/`map_vision`/`map_vision_qkv` covers
    /// `QwenConfig::param_list()` / `ClipVisionConfig::tensor_manifest()`
    /// exactly (mm_projector is 4 fixed names, already covered above).
    #[test]
    fn decoder_manifest_is_fully_reachable_by_name() {
        let cfg = QwenConfig::llama2_13b();
        for (name, _) in cfg.param_list() {
            let hf = qwen3::hf::HfNames::CAUSAL_LM.from_brain(&name).unwrap_or_else(|| panic!("no HF name for {name}"));
            assert_eq!(map_decoder(&hf, &cfg).as_deref(), Some(name.as_str()), "round trip for {name}");
        }
    }

    #[test]
    fn vision_manifest_is_fully_reachable_by_name_including_fused_qkv() {
        let cfg = ClipVisionConfig::clip_l336();
        for (name, _) in cfg.tensor_manifest() {
            if name.ends_with("attn.qkv.weight") || name.ends_with("attn.qkv.bias") {
                // Fused tensors have no single HF source name - covered by
                // map_vision_qkv's three-part reconstruction instead.
                let n = name.strip_prefix("blocks.").unwrap().split('.').next().unwrap();
                let is_bias = name.ends_with(".bias");
                let suffix = if is_bias { "bias" } else { "weight" };
                for (part, letter) in [(QkvPart::Q, 'q'), (QkvPart::K, 'k'), (QkvPart::V, 'v')] {
                    let hf = format!("{VISION_PREFIX}encoder.layers.{n}.self_attn.{letter}_proj.{suffix}");
                    assert_eq!(map_vision_qkv(&hf), Some((n.parse().unwrap(), part, is_bias)), "qkv round trip for {hf}");
                }
                continue;
            }
            let hf = if name == "class_embed" {
                format!("{VISION_PREFIX}embeddings.class_embedding")
            } else if name == "patch_embed.weight" {
                format!("{VISION_PREFIX}embeddings.patch_embedding.weight")
            } else if name == "pos_embed" {
                format!("{VISION_PREFIX}embeddings.position_embedding.weight")
            } else if let Some(rest) = name.strip_prefix("pre_norm.") {
                format!("{VISION_PREFIX}pre_layrnorm.{rest}")
            } else {
                let (n, leaf) = name.strip_prefix("blocks.").unwrap().split_once('.').unwrap();
                let hf_leaf = match leaf {
                    "norm1.weight" => "layer_norm1.weight",
                    "norm1.bias" => "layer_norm1.bias",
                    "attn.proj.weight" => "self_attn.out_proj.weight",
                    "attn.proj.bias" => "self_attn.out_proj.bias",
                    "norm2.weight" => "layer_norm2.weight",
                    "norm2.bias" => "layer_norm2.bias",
                    "mlp.fc1.weight" => "mlp.fc1.weight",
                    "mlp.fc1.bias" => "mlp.fc1.bias",
                    "mlp.fc2.weight" => "mlp.fc2.weight",
                    "mlp.fc2.bias" => "mlp.fc2.bias",
                    other => panic!("unhandled vision leaf {other}"),
                };
                format!("{VISION_PREFIX}encoder.layers.{n}.{hf_leaf}")
            };
            assert_eq!(map_vision(&hf).as_deref(), Some(name.as_str()), "round trip for {name}");
        }
    }
}
