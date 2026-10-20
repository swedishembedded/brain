// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A lazy [`checkpoint::TensorSource`] over a Qwen3.5/3.8 HuggingFace checkpoint
//! directory (the shipped `Qwen/Qwen3.8-27B-FP8` layout: `layers-N.safetensors`
//! plus `outside.safetensors`, or any bf16/fp32 HF layout of the same tensor
//! names).
//!
//! [`crate::import::import_dir`] dequantizes the whole model into one host f32
//! map (about 108 GB for the 27B) and so cannot run under the host limit, and
//! [`crate::import::import_layer`] is per layer. This source does the same work
//! per TENSOR, on demand, under the brain-canonical names the model builders
//! ask for (`blocks.{l}.mlp.gate.weight`, `tok.weight`, ...): names go through
//! the same `import::classify`, plain RMSNorm weights get the same `(1+w)`
//! fold, and FP8 blocks are decoded by `checkpoint::packed`'s tile codec, so
//! every tensor is bit-identical to what the eager importers produce (the unit
//! tests assert exactly that). Host peak while a tensor is consumed is that one
//! tensor (`with_tensor`) or `O(chunk)` (`with_tensor_chunks`, which decodes
//! whole-row tiles so a 5 GB embedding or head never has to exist on the host
//! at once).
//!
//! The model builder pulls each tensor, uploads it at the requested tier
//! (`Weight::upload`: bf16, or NF4 quantized on the way in) and drops it, so the
//! frozen base can sit on the device without a host copy of the model.
//!
//! Swedish Embedded AB implements solutions for loading very large quantized
//! checkpoints onto a single accelerator within a small host memory budget for
//! its clients. If your team needs expertise in memory-bounded model loading or
//! low-precision fine-tuning then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::path::Path;

use checkpoint::packed::{Binding, CodecRegistry, PackedReader, PackedSource, PackedTensorReader};
use checkpoint::TensorSource;
use serde_json::Value;

use crate::config::Qwen35Config;
use crate::import::{classify, is_plain_rmsnorm_weight};

/// Open the Qwen3.5/3.8 HuggingFace checkpoint in `dir` for model `cfg` as a
/// [`checkpoint::TensorSource`] under brain-canonical names; `block` is the FP8
/// `weight_block_size` (128 for the real release, ignored by tensors without a
/// `weight_scale_inv` sibling).
///
/// This module is only the declarative part: which stored tensor is which
/// canonical parameter ([`classify`]), which norms are stored as `w` and applied
/// as `1 + w`, and the two-way coverage check against the model config.
/// Discovering FP8 scale grids, decoding, tiling and releasing file pages are
/// `checkpoint::packed`'s.
///
/// Reads only headers. Fails, naming the tensors, when the checkpoint does not
/// cover `cfg.param_list()` (LoRA adapter tensors excepted: they are not in a
/// base checkpoint), when a tensor's element count disagrees with the config,
/// when a source tensor maps to no known name, or when an FP8 weight's scale
/// plane has the wrong shape. `mtp.*` and `model.visual.*` tensors are out of
/// scope here exactly as in [`crate::import`].
pub fn open_dir(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<PackedSource, String> {
    open_dir_with(dir, cfg, block, &[])
}

/// [`open_dir`] over a checkpoint that also holds another component's tensors:
/// names under a `foreign` prefix are skipped (their owner validates them),
/// every other name is held to the same two-way coverage. A multimodal wrapper
/// around this decoder passes its vision tower and projector prefixes here.
pub fn open_dir_with(dir: &Path, cfg: &Qwen35Config, block: usize, foreign: &[&str]) -> Result<PackedSource, String> {
    let reader = PackedReader::open(dir).map_err(|e| format!("qwen35 source: {e}"))?.with_fp8_block(block as u32);
    let mut bindings: Vec<(String, Binding)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for name in reader.names() {
        // A scale plane belongs to its weight (the reader hides it); one whose
        // weight is missing reaches classify and is refused by name.
        let Some(canon) = classify(&name, cfg, foreign)? else { continue };
        if !seen.insert(canon.clone()) {
            return Err(format!("qwen35 source: two source tensors mapped onto {canon}"));
        }
        let add = if is_plain_rmsnorm_weight(&canon) { 1.0 } else { 0.0 };
        bindings.push((canon, Binding { source: name, add }));
    }
    let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), bindings).map_err(|e| format!("qwen35 source: {e}"))?;
    check_coverage(&src, cfg)?;
    Ok(src)
}

/// Two-way coverage against `cfg.param_list()` from shapes alone.
fn check_coverage(src: &PackedSource, cfg: &Qwen35Config) -> Result<(), String> {
    let expected: Vec<(String, usize)> = cfg.param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).collect();
    let mut problems = Vec::new();
    for (name, numel) in &expected {
        match src.numel(name) {
            None => problems.push(format!("missing {name}")),
            Some(n) if n != *numel => problems.push(format!("{name} has {n} elements, config expects {numel}")),
            Some(_) => {}
        }
    }
    let want: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    for name in src.names().into_iter().filter(|n| !want.contains(n)) {
        problems.push(format!("{name} is in the checkpoint but not in the config's parameter list"));
    }
    if problems.is_empty() {
        return Ok(());
    }
    problems.sort();
    let total = problems.len();
    problems.truncate(16);
    Err(format!("qwen35 source: checkpoint does not match the config ({total} problems): {}", problems.join("; ")))
}

/// The [`Qwen35Config`] of the HF checkpoint in `dir`, read from its
/// `config.json` (the `text_config` object of a multimodal release, or the
/// top level of a text-only one). `block_size` is the model's full
/// `max_position_embeddings`; a trainer narrows it to its own sequence window.
pub fn hf_config(dir: &Path) -> Result<Qwen35Config, String> {
    let path = dir.join("config.json");
    let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("qwen35 config: read {}: {e}", path.display()))?)
        .map_err(|e| format!("qwen35 config: {}: {e}", path.display()))?;
    let c = if top["text_config"].is_object() { &top["text_config"] } else { &top };
    let rope = &c["rope_parameters"];
    let pick = |keys: &[&str]| keys.iter().find_map(|k| c.get(*k).filter(|v| !v.is_null()).or_else(|| rope.get(*k).filter(|v| !v.is_null()))).cloned();
    let mut j = serde_json::Map::new();
    for (to, from) in [
        ("vocab_size", &["vocab_size"][..]),
        ("n_layers", &["num_hidden_layers"]),
        ("d_model", &["hidden_size"]),
        ("n_heads", &["num_attention_heads"]),
        ("n_kv_heads", &["num_key_value_heads"]),
        ("head_dim", &["head_dim"]),
        ("rms_norm_eps", &["rms_norm_eps"]),
        ("rope_theta", &["rope_theta"]),
        ("max_position_embeddings", &["max_position_embeddings"]),
        ("block_size", &["max_position_embeddings"]),
        ("attention_bias", &["attention_bias"]),
        ("partial_rotary_factor", &["partial_rotary_factor"]),
        ("mrope_section", &["mrope_section"]),
        ("full_attention_interval", &["full_attention_interval"]),
        ("linear_num_key_heads", &["linear_num_key_heads"]),
        ("linear_num_value_heads", &["linear_num_value_heads"]),
        ("linear_key_head_dim", &["linear_key_head_dim"]),
        ("linear_value_head_dim", &["linear_value_head_dim"]),
        ("linear_conv_kernel_dim", &["linear_conv_kernel_dim"]),
        ("intermediate_size", &["intermediate_size"]),
        ("rope_scaling", &["rope_scaling"]),
    ] {
        if let Some(v) = pick(from) {
            j.insert(to.to_string(), v);
        }
    }
    let tie = top["tie_word_embeddings"].as_bool().or_else(|| c["tie_word_embeddings"].as_bool()).unwrap_or(false);
    j.insert("tie_word_embeddings".to_string(), Value::Bool(tie));
    Qwen35Config::from_json_checked(&Value::Object(j))
}


/// The FP8 `weight_block_size` the checkpoint in `dir` declares in its
/// `quantization_config` (128 for the real release); 128 when it declares
/// none, as a bf16/fp32 release does not.
pub fn hf_fp8_block_size(dir: &Path) -> Result<usize, String> {
    let path = dir.join("config.json");
    let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("qwen35 config: read {}: {e}", path.display()))?)
        .map_err(|e| format!("qwen35 config: {}: {e}", path.display()))?;
    let q = if top["quantization_config"].is_object() { &top["quantization_config"] } else { &top["text_config"]["quantization_config"] };
    match q["weight_block_size"].as_array().map(|a| a.iter().map(|v| v.as_u64()).collect::<Vec<_>>()) {
        None => Ok(128),
        Some(b) if b.len() == 2 && b[0] == b[1] && b[0].is_some_and(|n| n > 0) => Ok(b[0].unwrap_or(128) as usize),
        Some(b) => Err(format!("qwen35 config: weight_block_size {b:?} is not a square block")),
    }
}
