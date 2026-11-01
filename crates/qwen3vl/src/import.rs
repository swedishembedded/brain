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
//! it maps by name only (no transpose). Decoder names mirror `qwen3::import` but
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

/// The element count of every weight the vision encoder ([`crate::encoder::VisionEncoder`]) takes, by its key.
pub fn vision_weight_sizes(cfg: &crate::config::VisionConfig) -> HashMap<String, usize> {
    let (c, pv, mlp) = (cfg.hidden as usize, cfg.patch_vec_dim() as usize, cfg.intermediate as usize);
    let mut sizes = HashMap::from([("patch_embed.weight".to_string(), c * pv), ("patch_embed.bias".to_string(), c), ("pos_embed".to_string(), cfg.num_position_embeddings as usize * c)]);
    for b in 0..cfg.depth {
        let dims = [c, c, 3 * c * c, 3 * c, c * c, c, c, c, mlp * c, mlp, c * mlp, c];
        for (leaf, sz) in crate::encoder::BLOCK_LEAVES.iter().zip(dims) {
            sizes.insert(format!("blocks.{b}.{leaf}"), sz);
        }
    }
    sizes
}

/// The element count of every weight of the main [`crate::encoder::PatchMerger`] (which projects to the config's
/// `out_hidden_size`, the decoder width), by its key.
pub fn merger_weight_sizes(cfg: &crate::config::VisionConfig) -> HashMap<String, usize> {
    let c = cfg.hidden as usize;
    let merged = c * (cfg.spatial_merge_size * cfg.spatial_merge_size) as usize;
    let out = cfg.out_hidden_size as usize;
    HashMap::from([
        ("ln.weight".to_string(), c),
        ("ln.bias".to_string(), c),
        ("fc1.weight".to_string(), merged * merged),
        ("fc1.bias".to_string(), merged),
        ("fc2.weight".to_string(), out * merged),
        ("fc2.bias".to_string(), out),
    ])
}

/// The vision tower and its main merger read from the `model.visual.*` tensors of `r`, as the host weights
/// [`crate::encoder::VisionEncoder::new`] and [`crate::encoder::PatchMerger::new`] take. A checkpoint is hostile input:
/// every tensor must be one the config describes, of exactly the size it implies, and none may be missing - an
/// error naming the first that is not, never a tower built on what happened to be there.
pub fn load_vision(r: &checkpoint::weightio::WeightReader, cfg: &crate::config::VisionConfig) -> Result<TowerWeights, String> {
    let (want_vision, want_merger) = (vision_weight_sizes(cfg), merger_weight_sizes(cfg));
    let mut vision = HashMap::new();
    let mut merger = HashMap::new();
    for name in r.names() {
        let (key, want, into) = if let Some(k) = map_vision(name) {
            (k, &want_vision, &mut vision)
        } else if let Some(k) = map_main_merger(name) {
            (k, &want_merger, &mut merger)
        } else {
            continue;
        };
        let expect = *want.get(&key).ok_or_else(|| format!("vision tensor '{name}' is not one of the tower's"))?;
        let numel = r.shape(name).map(|s| s.iter().product::<u64>() as usize);
        if numel != Some(expect) {
            return Err(format!("vision tensor '{name}' has {numel:?} elements, the config asks for {expect}"));
        }
        let data = r.tensor(name).ok_or_else(|| format!("vision tensor '{name}' cannot be read"))?;
        into.insert(key, data);
    }
    for (want, have, what) in [(&want_vision, &vision, "tower"), (&want_merger, &merger, "merger")] {
        if let Some(missing) = want.keys().find(|k| !have.contains_key(*k)) {
            return Err(format!("the vision {what} has no tensor for '{missing}'"));
        }
    }
    Ok(TowerWeights { vision, merger })
}

/// The host weights of the vision tower and of its main merger ([`load_vision`]).
#[derive(Debug)]
pub struct TowerWeights {
    pub vision: HashMap<String, Vec<f32>>,
    pub merger: HashMap<String, Vec<f32>>,
}

/// HF decoder name → `qwen3::Qwen` parameter key (`model.language_model.*`,
/// through the one `qwen3::hf` leaf table).
pub fn map_decoder(hf: &str) -> Option<String> {
    qwen3::hf::HfNames { prefix: "model.language_model.", head: "lm_head.weight" }.body_param(hf)
}

/// A streaming [`checkpoint::remap::RemapSource`] over `r`'s decoder tensors
/// (`model.language_model.*`), for [`crate::model::Qwen3Vl::from_hf`]'s
/// large-checkpoint path. Unlike [`partition`], this never decodes a decoder
/// tensor until [`checkpoint::TensorSource::with_tensor`] asks for it by
/// name, and drops it again once the caller's callback returns - the same
/// peak-memory shape `qwen3::import::shard_source` already gives plain
/// `qwen3::Qwen`, now extended to the VL decoder (whose text encoder is the
/// dominant share of a Qwen3-VL checkpoint's bytes, so this is where
/// streaming actually matters; the vision tower stays eager in
/// [`crate::model::Qwen3Vl::from_hf`] - it is a small fraction of the
/// weights and none of the per-token bandwidth).
pub fn decoder_source(r: &checkpoint::weightio::WeightReader) -> checkpoint::remap::RemapSource<'_> {
    let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
    for name in r.names() {
        if let Some(bn) = map_decoder(name) {
            plan.insert(bn, checkpoint::remap::Fetch::Whole(name.to_string()));
        }
    }
    checkpoint::remap::RemapSource::new(r, plan)
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

use brain_testutil::model_dir;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
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

    /// Coverage against the REAL released index (names only - no 16 GB data load).
    /// Skips if the checkpoint isn't present. Asserts every decoder parameter the
    /// config expects is imported and the vision/merger groups have the right counts.
    #[test]
    fn real_index_fully_covered() {
        let path = format!("{}/model.safetensors.index.json", model_dir("Qwen/Qwen3-VL-4B-Instruct").unwrap_or_default());
        let Ok(txt) = std::fs::read_to_string(path) else {
            brain_testutil::skip("real_index_fully_covered: checkpoint not present");
            return;
        };
        let idx: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let names: Vec<String> = idx["weight_map"].as_object().unwrap().keys().cloned().collect();
        let map: HashMap<String, Vec<f32>> = names.iter().map(|n| (n.clone(), vec![])).collect();
        let total = map.len();
        let w = partition(map, 3);

        let cfg = crate::config::Qwen3VlConfig::qwen3_vl_4b();
        for (name, _) in cfg.text.param_list() {
            assert!(w.decoder.contains_key(&name), "decoder param not imported: {name}");
        }
        // 24 blocks × 12 leaves + patch_embed(w,b) + pos_embed.
        assert_eq!(w.vision.len(), 24 * 12 + 3, "vision weight count");
        assert_eq!(w.main_merger.len(), 6, "main merger weight count");
        assert_eq!(w.deepstack.len(), 3);
        for d in &w.deepstack {
            assert_eq!(d.len(), 6, "deepstack merger weight count");
        }
        // Every real tensor should be accounted for (rotary tables aren't stored).
        let mapped = w.vision.len() + w.main_merger.len() + w.deepstack.iter().map(|m| m.len()).sum::<usize>() + w.decoder.len();
        assert_eq!(mapped, total, "every checkpoint tensor must map ({mapped}/{total})");
    }

    /// [`decoder_source`]'s plan must agree with [`partition`]'s eager route
    /// for the decoder group on the SAME real checkpoint - same names in,
    /// same brain param names out - since it exists purely as a
    /// lower-peak-memory alternative path to the same weights, not a
    /// different mapping. Names only (`Fetch::Whole` targets, never read).
    #[test]
    fn decoder_source_plan_matches_partitions_decoder_route() {
        let path = format!("{}/model.safetensors.index.json", model_dir("Qwen/Qwen3-VL-4B-Instruct").unwrap_or_default());
        let Ok(txt) = std::fs::read_to_string(path) else {
            brain_testutil::skip("decoder_source_plan_matches_partitions_decoder_route: checkpoint not present");
            return;
        };
        let idx: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let names: Vec<String> = idx["weight_map"].as_object().unwrap().keys().cloned().collect();
        let map: HashMap<String, Vec<f32>> = names.iter().map(|n| (n.clone(), vec![])).collect();
        let w = partition(map, 3);

        let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
        for name in &names {
            if let Some(bn) = map_decoder(name) {
                plan.insert(bn, checkpoint::remap::Fetch::Whole(name.clone()));
            }
        }
        assert_eq!(plan.len(), w.decoder.len(), "same number of decoder params routed");
        for bn in w.decoder.keys() {
            assert!(plan.contains_key(bn), "decoder_source plan missing '{bn}'");
        }
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

    /// HF names and shapes of every tensor of a tower of `cfg`, with `data(numel)` as its content.
    fn hf_tensors(cfg: &crate::config::VisionConfig, data: impl Fn(usize) -> Vec<f32>) -> Vec<(String, Vec<u64>, Vec<f32>)> {
        let mut out = Vec::new();
        let mut add = |hf: String, numel: usize| out.push((hf, vec![numel as u64], data(numel)));
        for (key, numel) in vision_weight_sizes(cfg) {
            let hf = match key.as_str() {
                "patch_embed.weight" => "model.visual.patch_embed.proj.weight".to_string(),
                "patch_embed.bias" => "model.visual.patch_embed.proj.bias".to_string(),
                "pos_embed" => "model.visual.pos_embed.weight".to_string(),
                k => {
                    let (n, leaf) = k.strip_prefix("blocks.").unwrap().split_once('.').unwrap();
                    let leaf = match leaf {
                        "qkv.weight" | "qkv.bias" | "proj.weight" | "proj.bias" => format!("attn.{leaf}"),
                        "fc1.weight" | "fc1.bias" => format!("mlp.linear_{leaf}"),
                        "fc2.weight" | "fc2.bias" => format!("mlp.linear_{leaf}"),
                        other => other.to_string(),
                    };
                    format!("model.visual.blocks.{n}.{leaf}")
                }
            };
            add(hf, numel);
        }
        for (key, numel) in merger_weight_sizes(cfg) {
            let leaf = match key.as_str() {
                "ln.weight" => "norm.weight".to_string(),
                "ln.bias" => "norm.bias".to_string(),
                k => format!("linear_{k}"),
            };
            add(format!("model.visual.merger.{leaf}"), numel);
        }
        out
    }

    fn tiny_vision() -> crate::config::VisionConfig {
        crate::config::VisionConfig { depth: 2, hidden: 8, num_heads: 2, intermediate: 16, patch_size: 2, temporal_patch_size: 2, spatial_merge_size: 2, num_position_embeddings: 16, out_hidden_size: 12, in_channels: 3, deepstack_indexes: vec![], tokens_per_second: 2 }
    }

    fn reader(tensors: Vec<(String, Vec<u64>, Vec<f32>)>) -> checkpoint::weightio::WeightReader {
        checkpoint::weightio::WeightReader::derived(Box::new(checkpoint::weightio::InMemory::new(tensors, serde_json::json!({}))))
    }

    #[test]
    fn the_vision_tower_is_read_whole_from_the_visual_tensors() {
        let cfg = tiny_vision();
        let TowerWeights { vision, merger } = load_vision(&reader(hf_tensors(&cfg, |n| vec![0.5; n])), &cfg).expect("a complete tower loads");
        assert_eq!(vision.len(), vision_weight_sizes(&cfg).len());
        assert_eq!(merger.len(), 6);
        assert!(vision.values().all(|v| v.iter().all(|&x| x == 0.5)));
    }

    #[test]
    fn a_tower_missing_a_tensor_or_holding_a_wrong_sized_one_is_refused_by_name() {
        let cfg = tiny_vision();
        let mut missing = hf_tensors(&cfg, |n| vec![0.0; n]);
        missing.retain(|(n, _, _)| n != "model.visual.blocks.1.attn.qkv.weight");
        let err = load_vision(&reader(missing), &cfg).unwrap_err();
        assert!(err.contains("blocks.1.qkv.weight"), "{err}");
        let mut wrong = hf_tensors(&cfg, |n| vec![0.0; n]);
        let t = wrong.iter_mut().find(|(n, _, _)| n == "model.visual.merger.linear_fc1.weight").unwrap();
        t.1 = vec![5];
        t.2 = vec![0.0; 5];
        let err = load_vision(&reader(wrong), &cfg).unwrap_err();
        assert!(err.contains("linear_fc1") && err.contains("elements"), "{err}");
        let mut stray = hf_tensors(&cfg, |n| vec![0.0; n]);
        stray.push(("model.visual.blocks.7.norm1.weight".to_string(), vec![8], vec![0.0; 8]));
        assert!(load_vision(&reader(stray), &cfg).unwrap_err().contains("not one of the tower's"));
    }
}
