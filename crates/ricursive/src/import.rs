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
//! The checkpoint is read through its packed view ([`checkpoint::packed`]), so every tensor reaches the device in
//! the encoding it is stored in (bf16, FP8, the ModelOpt NVFP4 of the official `Ripple-65B-Preview-NVFP4`
//! release) with no conversion step and no whole-tensor f32 copy; quantization sidecars (`weight_scale`,
//! `weight_scale_2`, `input_scale`) belong to the weight they sit beside and are not tensors of their own.
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

use checkpoint::packed::{quant_sidecars, CodecRegistry, PackedReader, PackedSource, QuantEncoding};
use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;

use crate::config::RicursiveConfig;

/// Tensor dtypes the forward can decode: plain floats, and the block-scaled encodings the packed view holds
/// resident as stored (see [`dtype_label`]).
const FLOAT_DTYPES: [&str; 6] = ["F32", "BF16", "F16", "NVFP4", "F8_E4M3", "F8_E5M2"];

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

/// The dtype name [`validate_layout`] sees for a tensor the packed view describes: the safetensors dtype for
/// plain floats, `NVFP4` for ModelOpt NVFP4 (stored as `U8` codes), `F8_E4M3` / `F8_E5M2` for scaled FP8.
fn dtype_label(enc: &QuantEncoding, raw: &str) -> String {
    use checkpoint::packed::{Element, FloatFormat};
    if *enc == QuantEncoding::nvfp4() {
        return "NVFP4".to_string();
    }
    match (&enc.element, enc.scales.is_empty()) {
        (Element::Float(FloatFormat::E4M3Fn), false) => "F8_E4M3".to_string(),
        (Element::Float(FloatFormat::E5M2), false) => "F8_E5M2".to_string(),
        _ => raw.to_string(),
    }
}

/// The static activation scales a ModelOpt W4A4 / W8A8 release stores beside its quantized weights
/// (`<module>.input_scale`): what the checkpoint says its activations were calibrated to.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivationScales {
    /// Modules with a stored scale.
    pub count: usize,
    /// Quantized weights of the release that have none (a mix of W4A4 and weight-only modules is refused).
    pub quantized_without: usize,
    /// Smallest and largest stored scale, `(min, max)`; `(0, 0)` when there are none.
    pub range: (f32, f32),
}

/// Read and check the `input_scale` sidecars of the checkpoint at `path`: each is one finite, positive scalar,
/// and a release that has them has them for every quantized weight. Brain holds the activations in bf16 or
/// f32 whatever the release was calibrated for (W4A16 where the reference is W4A4, see [`crate::weights`]), so
/// the scales are validated here and not applied; nothing is ignored unread.
pub fn activation_scales(path: &str) -> Result<ActivationScales, String> {
    let p = std::path::Path::new(path);
    let reader = if p.is_dir() { WeightReader::open_hf_dir(p) } else { WeightReader::open(path) }.map_err(|e| format!("ricursive import: cannot open {path}: {e}"))?;
    let names: std::collections::HashSet<String> = reader.names().map(str::to_string).collect();
    let (mut count, mut bounds) = (0usize, (f32::INFINITY, 0f32));
    for name in names.iter().filter(|n| n.ends_with(".input_scale")) {
        let numel = reader.shape(name).map(|s| s.iter().product::<u64>());
        if numel != Some(1) {
            return Err(format!("ricursive import: {name} has {numel:?} elements, expected one static scale ({path})"));
        }
        let mut value = f32::NAN;
        reader.with_tensor_chunks(name, 1, &mut |_, v| value = v[0]);
        if !(value.is_finite() && value > 0.0) {
            return Err(format!("ricursive import: {name} is {value}, a static activation scale must be finite and positive ({path})"));
        }
        count += 1;
        bounds = (bounds.0.min(value), bounds.1.max(value));
    }
    // A quantized weight is a `<module>.weight` with a `weight_scale` beside it.
    let quantized: Vec<&String> = names.iter().filter(|n| n.ends_with(".weight") && names.contains(&format!("{n}_scale"))).collect();
    let without = quantized.iter().filter(|n| !names.contains(&format!("{}.input_scale", n.trim_end_matches(".weight")))).count();
    if count > 0 && without > 0 {
        return Err(format!("ricursive import: {without} quantized weights have no input_scale while {count} others do; refusing a release that mixes static-activation and weight-only modules ({path})"));
    }
    Ok(ActivationScales { count, quantized_without: without, range: if count == 0 { (0.0, 0.0) } else { bounds } })
}

/// Open `path` (a safetensors file or a Hugging Face directory, sharded or not) as a packed source in canonical names.
///
/// `source_layers[i]` is the file layer that serves canonical layer `i`; pass `0..num_hidden_layers` for a complete checkpoint.
fn open_packed(path: &str, cfg: &RicursiveConfig, source_layers: &[usize]) -> Result<RemapSource<'static>, String> {
    if source_layers.len() != cfg.num_hidden_layers {
        return Err(format!("ricursive import: {} source layers for a {}-layer config", source_layers.len(), cfg.num_hidden_layers));
    }
    let p = std::path::Path::new(path);
    let raw = if p.is_dir() { WeightReader::open_hf_dir(p) } else { WeightReader::open(path) }.map_err(|e| format!("ricursive import: cannot open {path}: {e}"))?;
    let packed = PackedSource::identity_lenient(Box::new(PackedReader::open(p).map_err(|e| format!("ricursive import: cannot open {path}: {e}"))?), CodecRegistry::reference());
    // The tensors of the model: everything stored except the sidecars of a quantized weight.
    let sidecars = quant_sidecars(&raw);
    let present: Vec<String> = raw.names().filter(|n| !sidecars.contains(*n)).map(str::to_string).collect();
    let shape_of = |n: &str| packed.descriptor(n).map(|d| d.shape.iter().map(|x| *x as usize).collect()).or_else(|| raw.shape(n).map(|s| s.iter().map(|d| *d as usize).collect()));
    let dtype_of = |n: &str| {
        let stored = raw.dtype(n)?;
        Some(packed.descriptor(n).map_or_else(|| stored.to_string(), |d| dtype_label(&d.encoding, stored)))
    };
    validate_layout(cfg, source_layers, &present, shape_of, dtype_of).map_err(|e| format!("{e} ({path})"))?;
    let scales = activation_scales(path)?;
    if scales.count > 0 {
        tracing::info!(count = scales.count, min = scales.range.0, max = scales.range.1, "ricursive: static activation scales validated (the engine holds activations in float; they are not applied)");
    }
    let identity: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let plan: HashMap<String, Fetch> = manifest(cfg, &identity).into_iter().zip(manifest(cfg, source_layers)).map(|((canonical, _), (in_file, _))| (canonical, Fetch::Whole(in_file))).collect();
    Ok(RemapSource::owning(Box::new(packed), plan))
}

/// Open the safetensors file at `path` as a source in canonical names.
///
/// `source_layers[i]` is the file layer that serves canonical layer `i`; pass
/// `0..num_hidden_layers` for a complete checkpoint.
pub fn open_safetensors(path: &str, cfg: &RicursiveConfig, source_layers: &[usize]) -> Result<RemapSource<'static>, String> {
    open_packed(path, cfg, source_layers)
}

/// [`open_safetensors`] for a checkpoint that is a single safetensors file or a Hugging Face directory (sharded,
/// with `model.safetensors.index.json`): the same validation, the same canonical names.
pub fn open_checkpoint(path: &str, cfg: &RicursiveConfig, source_layers: &[usize]) -> Result<RemapSource<'static>, String> {
    open_packed(path, cfg, source_layers)
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
