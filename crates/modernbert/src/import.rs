// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the real `convaiinnovations/laya` checkpoint (Apache-2.0) - one
//! combined F16 `model.safetensors` state dict covering the WHOLE
//! `DecisionModel` (encoder + head + scorer + act_head + type_emb), plus
//! `encoder/config.json` (the backbone shape) and `rl_agent_config.json`
//! (`head_layers`/`max_len`/`head_max_len`/the temperature tables) - into the
//! `HashMap<String, Vec<f32>>` init maps `ModernBert::new_on`/
//! `LayaHead::new_on` already take directly. No intermediate brain
//! `.safetensors` container is written: unlike `crates/lfm2`/`crates/decide`,
//! this crate's own constructors accept an init map straight, so writing one
//! out to disk and reading it back would be a needless round trip - the
//! model store's own copy of `model.safetensors` (mmapped, decoded one tensor
//! at a time by [`checkpoint::weightio::WeightReader`]) IS the on-disk form.
//!
//! Tensor names verified directly against the real released
//! `model.safetensors` header this session (206 tensors total: 170 encoder +
//! 35 head/scorer/act_head/type_emb + 1 `temperature`, matching
//! `ModernBertConfig::modernbert_large().tensor_manifest().len()` +
//! `laya::tensor_manifest(&LayaConfig::new(1024)).len()` + 1 exactly). Every
//! tensor is F16 except `temperature` itself, which is F32 `[3]` - a
//! serving-time calibration scalar table (per qtype), not a model parameter,
//! read and returned separately rather than fed into either `ParamStore`.
//!
//! No tensor needs transposing (HF `nn.Linear.weight` is `[out,in]`, already
//! brain's own `matmul.wgsl` convention) and no tensor needs fusing on THIS
//! side: `encoder.layers.{l}.attn.Wqkv.weight` already ships as one fused
//! `[3H,H]` matrix, and `head.layers.{l}.self_attn.in_proj_weight/bias`
//! already ships as PyTorch's own fused `nn.MultiheadAttention` QKV - unlike
//! `crates/decide`'s importer, which has to fuse three separate q/k/v
//! tensors itself. `encoder.layers.0.attn_norm.weight` is genuinely absent
//! from the checkpoint (`nn.Identity()` on the real model, matching
//! `ModernBertConfig::has_attn_norm`), not merely unmapped.
//!
//! Follows the hand-rolled `hf_to_brain(name) -> Option<Dest>` name-mapper
//! precedent (`crates/lfm2/src/import.rs`, `crates/decide/src/import.rs`):
//! [`hf_to_encoder`] feeds `ModernBertConfig::tensor_manifest`'s names,
//! [`hf_to_head`] feeds `laya::tensor_manifest`'s (a SEPARATE `ParamStore`,
//! the same split M3 already made) - both fail loudly (via
//! [`check_coverage`]'s two-way check) on any tensor neither maps and either
//! the real checkpoint or the coverage check does not expect.
//!
//! Swedish Embedded AB implements from-scratch import pipelines for released
//! transformer checkpoints, validated tensor-for-tensor against the real
//! source rather than assumed. If your team needs a model brought up this
//! way, you can procure our services by emailing info@swedishembedded.com.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::config::ModernBertConfig;
use crate::laya::LayaConfig;

/// `rl_agent_config.json`'s own fields this crate needs - the HEAD/serving
/// setup, not the backbone (that is `encoder/config.json`,
/// `ModernBertConfig::from_hf_json`'s domain).
#[derive(Clone, Debug)]
pub struct RlAgentConfig {
    /// Must equal `LayaConfig::new(d_model).head_layers` - checked by
    /// [`import_dir`], not silently ignored on a mismatch (a future released
    /// checkpoint with a different head depth must fail loudly here, not
    /// produce a coverage error three steps downstream in `LayaHead::new_on`
    /// that gives no hint why).
    pub head_layers: u32,
    /// Max packed sequence length `build_sequence` truncates the whole
    /// sequence to (512 on the released checkpoint).
    pub max_len: u32,
    /// Token budget `build_sequence` gives the instructions + option markers
    /// before the state (192 on the released checkpoint).
    pub head_max_len: u32,
    pub max_prefixes: u32,
    /// Per-qtype (`choice`/`score`/`noul`, index 0/1/2) temperature, `[3]`.
    pub temperature: Vec<f32>,
    /// Per-cardinality-bucket override, keyed like `"choice:3-5"` (the real
    /// released checkpoint's own `rl_common.py::temp_bucket` naming) -
    /// overrides `temperature[qtype]` when present for that bucket.
    pub temperature_by_options: HashMap<String, f32>,
}

/// Parse `rl_agent_config.json`. Fails loudly on a missing required field
/// rather than defaulting past it - a serving config silently missing
/// `max_len`/`head_max_len` would build sequences of the wrong length with no
/// error anywhere near the mistake.
pub fn rl_agent_config_from_json(v: &Value) -> Result<RlAgentConfig, String> {
    let u = |k: &str| -> Result<u32, String> {
        v.get(k).and_then(Value::as_u64).map(|x| x as u32).ok_or_else(|| format!("rl_agent_config.json: missing {k:?}"))
    };
    let temperature: Vec<f32> = v
        .get("temperature")
        .and_then(Value::as_array)
        .ok_or("rl_agent_config.json: missing temperature")?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| "rl_agent_config.json: temperature entry not a number".to_string()))
        .collect::<Result<_, _>>()?;
    let temperature_by_options: HashMap<String, f32> = v
        .get("temperature_by_options")
        .and_then(Value::as_object)
        .map(|o| o.iter().filter_map(|(k, val)| val.as_f64().map(|f| (k.clone(), f as f32))).collect())
        .unwrap_or_default();
    Ok(RlAgentConfig {
        head_layers: u("head_layers")?,
        max_len: u("max_len")?,
        head_max_len: u("head_max_len")?,
        max_prefixes: u("max_prefixes")?,
        temperature,
        temperature_by_options,
    })
}

/// Map one real ENCODER tensor name to `ModernBertConfig::tensor_manifest`'s
/// own short name, or `None` if it belongs to the head/scorer/act_head/
/// type_emb side instead (see [`hf_to_head`]) or to neither.
fn hf_to_encoder(name: &str) -> Option<String> {
    let rest = name.strip_prefix("encoder.")?;
    match rest {
        "embeddings.tok_embeddings.weight" => return Some("tok.weight".to_string()),
        "embeddings.norm.weight" => return Some("emb_norm.weight".to_string()),
        "final_norm.weight" => return Some("final_norm.weight".to_string()),
        _ => {}
    }
    let rest = rest.strip_prefix("layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let l: u32 = n.parse().ok()?;
    let leaf = match rest {
        "attn_norm.weight" => "attn_norm.weight",
        "attn.Wqkv.weight" => "qkv.weight",
        "attn.Wo.weight" => "proj.weight",
        "mlp_norm.weight" => "mlp_norm.weight",
        "mlp.Wi.weight" => "mlp.wi.weight",
        "mlp.Wo.weight" => "mlp.wo.weight",
        _ => return None,
    };
    Some(format!("blocks.{l}.{leaf}"))
}

/// Map one real HEAD/scorer/act_head/type_emb tensor name to
/// `laya::tensor_manifest`'s own short name - see that function's doc for the
/// exact name table this mirrors (`head.{l}.attn.in_proj.*`,
/// `scorer.{norm,fc1,fc2}.*`, `act.{fc1,fc2}.*`).
fn hf_to_head(name: &str) -> Option<String> {
    if name == "type_emb.weight" {
        return Some("type_emb.weight".to_string());
    }
    if let Some(rest) = name.strip_prefix("head.layers.") {
        let (n, rest) = rest.split_once('.')?;
        let l: u32 = n.parse().ok()?;
        let leaf = match rest {
            "self_attn.in_proj_weight" => "attn.in_proj.weight",
            "self_attn.in_proj_bias" => "attn.in_proj.bias",
            "self_attn.out_proj.weight" => "attn.out_proj.weight",
            "self_attn.out_proj.bias" => "attn.out_proj.bias",
            // PyTorch's `nn.TransformerEncoderLayer` names its FFN
            // `linear1`/`linear2` - this crate calls them `ff1`/`ff2`.
            "linear1.weight" => "ff1.weight",
            "linear1.bias" => "ff1.bias",
            "linear2.weight" => "ff2.weight",
            "linear2.bias" => "ff2.bias",
            "norm1.weight" => "norm1.weight",
            "norm1.bias" => "norm1.bias",
            "norm2.weight" => "norm2.weight",
            "norm2.bias" => "norm2.bias",
            _ => return None,
        };
        return Some(format!("head.{l}.{leaf}"));
    }
    match name {
        // scorer = nn.Sequential(LayerNorm, Linear, GELU, Linear) - index 2
        // is the parameter-free GELU, so indices 0/1/3.
        "scorer.0.weight" => Some("scorer.norm.weight".to_string()),
        "scorer.0.bias" => Some("scorer.norm.bias".to_string()),
        "scorer.1.weight" => Some("scorer.fc1.weight".to_string()),
        "scorer.1.bias" => Some("scorer.fc1.bias".to_string()),
        "scorer.3.weight" => Some("scorer.fc2.weight".to_string()),
        "scorer.3.bias" => Some("scorer.fc2.bias".to_string()),
        // act_head = nn.Sequential(Linear, GELU, Linear) - index 1 is GELU.
        "act_head.0.weight" => Some("act.fc1.weight".to_string()),
        "act_head.0.bias" => Some("act.fc1.bias".to_string()),
        "act_head.2.weight" => Some("act.fc2.weight".to_string()),
        "act_head.2.bias" => Some("act.fc2.bias".to_string()),
        _ => None,
    }
}

/// A fully-loaded Laya checkpoint: configs plus two ready-to-use init maps,
/// one per `ParamStore` (`ModernBert::new_on`'s `encoder_init`,
/// `LayaHead::new_on`'s `head_init`) - both directly usable with no further
/// transformation.
pub struct LayaCheckpoint {
    pub cfg: ModernBertConfig,
    pub laya_cfg: LayaConfig,
    pub rl: RlAgentConfig,
    pub encoder_init: HashMap<String, Vec<f32>>,
    pub head_init: HashMap<String, Vec<f32>>,
    /// The serving-time per-qtype calibration scalars - NOT fed into either
    /// `ParamStore` (see the module doc).
    pub temperature: Vec<f32>,
}

/// Import a `brain pull convaiinnovations/laya`-fetched checkpoint directory
/// (root layout: `model.safetensors`, `encoder/config.json`,
/// `rl_agent_config.json`, `tokenizer/tokenizer.json` - the English root
/// checkpoint; `multilingual/`/`typed-decisions/` are out of scope and never
/// read here). Streams the state dict one tensor at a time
/// ([`checkpoint::weightio::WeightReader::for_each`]); never partially
/// succeeds silently - any unmapped, missing, or mis-shaped tensor is an
/// `Err`, not a quietly-zero-initialized parameter.
pub fn import_dir(hf_dir: &str) -> Result<LayaCheckpoint, String> {
    let dir = Path::new(hf_dir);

    let cfg_json = std::fs::read_to_string(dir.join("encoder").join("config.json"))
        .map_err(|e| format!("read encoder/config.json: {e}"))?;
    let cfg_v: Value = serde_json::from_str(&cfg_json).map_err(|e| format!("encoder/config.json: {e}"))?;
    let mut cfg = ModernBertConfig::from_hf_json(&cfg_v)?;

    let rl_json = std::fs::read_to_string(dir.join("rl_agent_config.json"))
        .map_err(|e| format!("read rl_agent_config.json: {e}"))?;
    let rl_v: Value = serde_json::from_str(&rl_json).map_err(|e| format!("rl_agent_config.json: {e}"))?;
    let rl = rl_agent_config_from_json(&rl_v)?;

    let laya_cfg = LayaConfig::new(cfg.d_model);
    if rl.head_layers != laya_cfg.head_layers {
        return Err(format!(
            "rl_agent_config.json head_layers {} != LayaConfig::new({}).head_layers {} \
             - the released checkpoint's head depth changed and LayaConfig must follow it",
            rl.head_layers, cfg.d_model, laya_cfg.head_layers
        ));
    }

    // Special-token ids are NOT in encoder/config.json at all (see
    // ModernBertConfig::from_hf_json's own note on this) - read from the
    // real tokenizer's own added_tokens, never hardcoded, since the
    // multilingual/typed-decisions variants use a different tokenizer.
    let tok_json = std::fs::read_to_string(dir.join("tokenizer").join("tokenizer.json"))
        .map_err(|e| format!("read tokenizer/tokenizer.json: {e}"))?;
    let tok_v: Value = serde_json::from_str(&tok_json).map_err(|e| format!("tokenizer/tokenizer.json: {e}"))?;
    let special_id = |content: &str| -> Result<u32, String> {
        tok_v["added_tokens"]
            .as_array()
            .and_then(|a| a.iter().find(|t| t["content"].as_str() == Some(content)))
            .and_then(|t| t["id"].as_u64())
            .map(|x| x as u32)
            .ok_or_else(|| format!("tokenizer/tokenizer.json: added_tokens missing {content:?}"))
    };
    cfg.cls_token_id = special_id("[CLS]")?;
    cfg.sep_token_id = special_id("[SEP]")?;
    cfg.pad_token_id = special_id("[PAD]")?;
    cfg.mask_token_id = special_id("[MASK]")?;

    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {hf_dir}: {e}"))?;
    let mut encoder_brain: HashMap<String, Vec<f32>> = HashMap::new();
    let mut head_brain: HashMap<String, Vec<f32>> = HashMap::new();
    let mut temperature: Option<Vec<f32>> = None;
    let mut unmapped: Vec<String> = Vec::new();
    reader.for_each(|name, _shape, data| {
        if name == "temperature" {
            temperature = Some(data);
            return;
        }
        if let Some(bn) = hf_to_encoder(name) {
            encoder_brain.insert(bn, data);
            return;
        }
        if let Some(bn) = hf_to_head(name) {
            head_brain.insert(bn, data);
            return;
        }
        unmapped.push(name.to_string());
    });
    if !unmapped.is_empty() {
        let mut unmapped = unmapped;
        unmapped.sort();
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    let temperature = temperature.ok_or("import: missing temperature tensor")?;

    // Full coverage, two ways: every brain parameter this crate's own
    // manifests expect arrived with the right element count, and nothing
    // mapped is left over - same discipline as lfm2/decide's importers.
    let encoder_init = check_coverage("encoder", cfg.tensor_manifest(), encoder_brain)?;
    let head_init = check_coverage("head", crate::laya::tensor_manifest(&laya_cfg), head_brain)?;

    Ok(LayaCheckpoint { cfg, laya_cfg, rl, encoder_init, head_init, temperature })
}

fn check_coverage(
    which: &str,
    manifest: Vec<(String, Vec<usize>)>,
    mut brain: HashMap<String, Vec<f32>>,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut init = HashMap::new();
    for (name, shape) in manifest {
        let numel: usize = shape.iter().product();
        let data = brain.remove(&name).ok_or_else(|| format!("import: {which}: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {which}: {name} element count {} != expected {numel}", data.len()));
        }
        init.insert(name, data);
    }
    if !brain.is_empty() {
        let mut extra: Vec<&String> = brain.keys().collect();
        extra.sort();
        return Err(format!("import: {which}: {} mapped HF tensors unused: {extra:?}", brain.len()));
    }
    Ok(init)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_to_encoder_name_mapping() {
        assert_eq!(hf_to_encoder("encoder.embeddings.tok_embeddings.weight").unwrap(), "tok.weight");
        assert_eq!(hf_to_encoder("encoder.embeddings.norm.weight").unwrap(), "emb_norm.weight");
        assert_eq!(hf_to_encoder("encoder.final_norm.weight").unwrap(), "final_norm.weight");
        assert_eq!(hf_to_encoder("encoder.layers.0.attn.Wqkv.weight").unwrap(), "blocks.0.qkv.weight");
        assert_eq!(hf_to_encoder("encoder.layers.5.attn.Wo.weight").unwrap(), "blocks.5.proj.weight");
        assert_eq!(hf_to_encoder("encoder.layers.5.attn_norm.weight").unwrap(), "blocks.5.attn_norm.weight");
        assert_eq!(hf_to_encoder("encoder.layers.5.mlp.Wi.weight").unwrap(), "blocks.5.mlp.wi.weight");
        assert_eq!(hf_to_encoder("encoder.layers.5.mlp.Wo.weight").unwrap(), "blocks.5.mlp.wo.weight");
        assert_eq!(hf_to_encoder("encoder.layers.5.mlp_norm.weight").unwrap(), "blocks.5.mlp_norm.weight");
        assert_eq!(hf_to_encoder("head.layers.0.norm1.weight"), None);
        assert_eq!(hf_to_encoder("temperature"), None);
    }

    #[test]
    fn hf_to_head_name_mapping() {
        assert_eq!(hf_to_head("type_emb.weight").unwrap(), "type_emb.weight");
        assert_eq!(hf_to_head("head.layers.0.self_attn.in_proj_weight").unwrap(), "head.0.attn.in_proj.weight");
        assert_eq!(hf_to_head("head.layers.1.self_attn.in_proj_bias").unwrap(), "head.1.attn.in_proj.bias");
        assert_eq!(hf_to_head("head.layers.0.self_attn.out_proj.weight").unwrap(), "head.0.attn.out_proj.weight");
        assert_eq!(hf_to_head("head.layers.0.linear1.weight").unwrap(), "head.0.ff1.weight");
        assert_eq!(hf_to_head("head.layers.0.linear2.bias").unwrap(), "head.0.ff2.bias");
        assert_eq!(hf_to_head("head.layers.0.norm1.weight").unwrap(), "head.0.norm1.weight");
        assert_eq!(hf_to_head("head.layers.0.norm2.bias").unwrap(), "head.0.norm2.bias");
        assert_eq!(hf_to_head("scorer.0.weight").unwrap(), "scorer.norm.weight");
        assert_eq!(hf_to_head("scorer.1.bias").unwrap(), "scorer.fc1.bias");
        assert_eq!(hf_to_head("scorer.3.weight").unwrap(), "scorer.fc2.weight");
        assert_eq!(hf_to_head("act_head.0.weight").unwrap(), "act.fc1.weight");
        assert_eq!(hf_to_head("act_head.2.bias").unwrap(), "act.fc2.bias");
        assert_eq!(hf_to_head("encoder.final_norm.weight"), None);
        assert_eq!(hf_to_head("scorer.2.weight"), None); // the parameter-free GELU
    }

    #[test]
    fn rl_agent_config_parses_the_real_released_shape() {
        let v = serde_json::json!({
            "encoder": "answerdotai/ModernBERT-large",
            "head_layers": 2,
            "max_len": 512,
            "head_max_len": 192,
            "max_prefixes": 6,
            "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
            "temperature_by_options": {"choice:3-5": 1.7601518630981445, "noul:2": 1.983399510383606},
        });
        let rl = rl_agent_config_from_json(&v).unwrap();
        assert_eq!(rl.head_layers, 2);
        assert_eq!(rl.max_len, 512);
        assert_eq!(rl.head_max_len, 192);
        assert_eq!(rl.max_prefixes, 6);
        assert_eq!(rl.temperature.len(), 3);
        assert_eq!(rl.temperature_by_options.get("choice:3-5").copied(), Some(1.760_151_9_f32));
    }

    #[test]
    fn rl_agent_config_fails_loudly_on_a_missing_required_field() {
        let v = serde_json::json!({"head_layers": 2, "max_len": 512, "temperature": [1.0, 1.0, 1.0]});
        let err = rl_agent_config_from_json(&v).unwrap_err();
        assert!(err.contains("head_max_len"), "{err}");
    }

    #[test]
    fn check_coverage_rejects_a_missing_tensor() {
        let manifest = vec![("a.weight".to_string(), vec![2usize]), ("b.weight".to_string(), vec![3usize])];
        let mut brain = HashMap::new();
        brain.insert("a.weight".to_string(), vec![1.0, 2.0]);
        let err = check_coverage("t", manifest, brain).unwrap_err();
        assert!(err.contains("b.weight"), "{err}");
    }

    #[test]
    fn check_coverage_rejects_an_unused_mapped_tensor() {
        let manifest = vec![("a.weight".to_string(), vec![2usize])];
        let mut brain = HashMap::new();
        brain.insert("a.weight".to_string(), vec![1.0, 2.0]);
        brain.insert("extra.weight".to_string(), vec![9.0]);
        let err = check_coverage("t", manifest, brain).unwrap_err();
        assert!(err.contains("extra.weight"), "{err}");
    }

    #[test]
    fn check_coverage_rejects_a_wrong_element_count() {
        let manifest = vec![("a.weight".to_string(), vec![2usize])];
        let mut brain = HashMap::new();
        brain.insert("a.weight".to_string(), vec![1.0, 2.0, 3.0]);
        let err = check_coverage("t", manifest, brain).unwrap_err();
        assert!(err.contains("element count"), "{err}");
    }
}
