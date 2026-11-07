// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import: the tensor manifest, two-way coverage, and the on-demand
//! safetensors source.
//!
//! Swedish Embedded AB implements solutions for importing very large
//! checkpoints where a wrong or truncated file is refused by name before any
//! weight is read, for its clients. If your team needs expertise in
//! memory-bounded checkpoint loading then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The canonical tensor names are the checkpoint's own (`model.layers.N...`),
//! with layers numbered `0..num_hidden_layers` of the config in use. A run over
//! a few real layers of the checkpoint (a config [`truncated`] to them) maps
//! canonical layer `i` onto the file's `source_layers[i]`; the full model is the
//! identity map.
//!
//! [`open_safetensors`] validates names, shapes and dtypes against the manifest
//! in both directions (a missing tensor, an unknown tensor and a shape mismatch
//! each name the tensor) before a single weight byte is decoded, and hands back
//! a memory-mapped [`RemapSource`]: nothing is expanded to f32 on the host
//! beyond the chunk the uploader is decoding.
//!
//! [`truncated`]: crate::RicursiveConfig::truncated

use std::collections::HashMap;

use checkpoint::mmap::MmapSafetensors;
use checkpoint::remap::{Fetch, RemapSource};

use crate::config::RicursiveConfig;

/// Tensor dtypes the f32 forward can decode.
const FLOAT_DTYPES: [&str; 3] = ["F32", "BF16", "F16"];

/// Name of tensor `suffix` of layer `layer` in the checkpoint.
pub fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

/// The per-layer tensors `cfg` declares, as `(suffix, shape)`.
fn layer_manifest(cfg: &RicursiveConfig) -> Vec<(&'static str, Vec<usize>)> {
    let (h, f) = (cfg.hidden_size, cfg.intermediate_size);
    let mut m = vec![
        ("input_layernorm.weight", vec![h]),
        ("post_attention_layernorm.weight", vec![h]),
        ("pre_feedforward_layernorm.weight", vec![h]),
        ("post_feedforward_layernorm.weight", vec![h]),
        ("self_attn.q_proj.weight", vec![cfg.q_dim(), h]),
        ("self_attn.k_proj.weight", vec![cfg.kv_dim(), h]),
        ("self_attn.v_proj.weight", vec![cfg.kv_dim(), h]),
        ("self_attn.o_proj.weight", vec![h, cfg.q_dim()]),
        ("self_attn.q_norm.weight", vec![cfg.head_dim]),
        ("self_attn.k_norm.weight", vec![cfg.head_dim]),
        ("mlp.gate_proj.weight", vec![f, h]),
        ("mlp.up_proj.weight", vec![f, h]),
        ("mlp.down_proj.weight", vec![h, f]),
    ];
    if cfg.use_gated_attention {
        m.push(("self_attn.g_proj.weight", vec![cfg.q_dim(), h]));
    }
    if cfg.use_valres {
        m.push(("self_attn.value_res_lambda", vec![1]));
    }
    m
}

/// Every tensor the model declares, `(name, shape)`, with canonical layer `i`
/// named after file layer `layer_names[i]`.
pub fn manifest(cfg: &RicursiveConfig, layer_names: &[usize]) -> Vec<(String, Vec<usize>)> {
    assert_eq!(layer_names.len(), cfg.num_hidden_layers, "one source layer per model layer");
    let mut m = vec![
        ("model.embed_tokens.weight".to_string(), vec![cfg.vocab_size, cfg.hidden_size]),
        ("model.norm.weight".to_string(), vec![cfg.hidden_size]),
        ("lm_head.weight".to_string(), vec![cfg.vocab_size, cfg.hidden_size]),
    ];
    for &layer in layer_names {
        for (suffix, shape) in layer_manifest(cfg) {
            m.push((layer_tensor(layer, suffix), shape));
        }
    }
    m
}

/// Check what a checkpoint offers against what the model declares, before any
/// weight is read: every declared tensor is present with its exact shape and a
/// float dtype, and nothing else is present.
pub fn validate_layout(
    cfg: &RicursiveConfig,
    source_layers: &[usize],
    present: &[String],
    shape_of: impl Fn(&str) -> Option<Vec<usize>>,
    dtype_of: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let declared = manifest(cfg, source_layers);
    checkpoint::coverage::validate_shapes(&shape_of, present, &declared, "ricursive import")?;
    for (name, _) in &declared {
        let dtype = dtype_of(name).ok_or_else(|| format!("ricursive import: {name} has no dtype"))?;
        if !FLOAT_DTYPES.contains(&dtype.as_str()) {
            return Err(format!("ricursive import: {name} has dtype {dtype}, expected one of {FLOAT_DTYPES:?}"));
        }
    }
    Ok(())
}

/// Open the safetensors file at `path` as a source in canonical names.
///
/// `source_layers[i]` is the file layer that serves canonical layer `i`; pass
/// `0..num_hidden_layers` for a complete checkpoint.
pub fn open_safetensors(path: &str, cfg: &RicursiveConfig, source_layers: &[usize]) -> Result<RemapSource<'static>, String> {
    if source_layers.len() != cfg.num_hidden_layers {
        return Err(format!("ricursive import: {} source layers for a {}-layer config", source_layers.len(), cfg.num_hidden_layers));
    }
    let file = MmapSafetensors::open(path)?;
    validate_layout(cfg, source_layers, file.names(), |n| file.shape(n).map(<[usize]>::to_vec), |n| file.dtype(n).map(str::to_string)).map_err(|e| format!("{e} ({path})"))?;
    let identity: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let plan: HashMap<String, Fetch> = manifest(cfg, &identity)
        .into_iter()
        .zip(manifest(cfg, source_layers))
        .map(|((canonical, _), (in_file, _))| (canonical, Fetch::Whole(in_file)))
        .collect();
    Ok(RemapSource::owning(Box::new(file), plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RicursiveConfig {
        RicursiveConfig::from_json(
            r#"{"vocab_size":16,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"use_valres":true,"layer_types":["sliding_attention","sliding_attention","full_attention"]}"#,
        )
        .unwrap()
    }

    /// A checkpoint that matches the manifest, as `(name -> shape)`.
    fn offered(cfg: &RicursiveConfig, layers: &[usize]) -> HashMap<String, Vec<usize>> {
        manifest(cfg, layers).into_iter().collect()
    }

    fn check(cfg: &RicursiveConfig, layers: &[usize], offer: &HashMap<String, Vec<usize>>) -> Result<(), String> {
        let present: Vec<String> = offer.keys().cloned().collect();
        validate_layout(cfg, layers, &present, |n| offer.get(n).cloned(), |_| Some("BF16".to_string()))
    }

    #[test]
    fn the_manifest_counts_fifteen_tensors_per_layer_plus_three() {
        let c = cfg();
        assert_eq!(manifest(&c, &[0, 1, 2]).len(), 3 + 3 * 15);
        let mut plain = c.clone();
        plain.use_valres = false;
        plain.use_gated_attention = false;
        assert_eq!(manifest(&plain, &[0, 1, 2]).len(), 3 + 3 * 13);
    }

    #[test]
    fn a_complete_checkpoint_and_a_partial_one_with_its_layer_map_are_accepted() {
        let c = cfg();
        assert_eq!(check(&c, &[0, 1, 2], &offered(&c, &[0, 1, 2])), Ok(()));
        let cut = c.truncated(&[0, 2]).unwrap();
        assert_eq!(check(&cut, &[0, 2], &offered(&cut, &[0, 2])), Ok(()));
    }

    #[test]
    fn a_missing_tensor_is_refused_by_name() {
        let c = cfg();
        let mut offer = offered(&c, &[0, 1, 2]);
        offer.remove("model.layers.1.self_attn.g_proj.weight");
        let err = check(&c, &[0, 1, 2], &offer).unwrap_err();
        assert!(err.contains("missing tensor model.layers.1.self_attn.g_proj.weight"), "{err}");
    }

    #[test]
    fn an_unknown_tensor_is_refused_by_name() {
        let c = cfg();
        let mut offer = offered(&c, &[0, 1, 2]);
        offer.insert("model.layers.1.self_attn.rotary_emb.inv_freq".into(), vec![2]);
        offer.insert("model.layers.3.mlp.up_proj.weight".into(), vec![12, 8]);
        let err = check(&c, &[0, 1, 2], &offer).unwrap_err();
        assert!(err.contains("unused source tensors") && err.contains("rotary_emb.inv_freq") && err.contains("layers.3.mlp.up_proj"), "{err}");
    }

    #[test]
    fn a_shape_mismatch_names_both_shapes_and_a_non_float_dtype_is_refused() {
        let c = cfg();
        let mut offer = offered(&c, &[0, 1, 2]);
        offer.insert("lm_head.weight".into(), vec![8, 16]);
        let err = check(&c, &[0, 1, 2], &offer).unwrap_err();
        assert!(err.contains("lm_head.weight") && err.contains("[8, 16]") && err.contains("[16, 8]"), "{err}");

        let offer = offered(&c, &[0, 1, 2]);
        let present: Vec<String> = offer.keys().cloned().collect();
        let err = validate_layout(&c, &[0, 1, 2], &present, |n| offer.get(n).cloned(), |n| Some(if n == "model.norm.weight" { "I8" } else { "F32" }.to_string())).unwrap_err();
        assert!(err.contains("model.norm.weight") && err.contains("I8"), "{err}");
    }
}
