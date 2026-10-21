// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements serving of HuggingFace mixture-of-experts releases as published for its clients. If
// your team needs expertise in running quantized Qwen3.5/3.6 MoE checkpoints then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A Qwen3.5/3.6 MoE HuggingFace checkpoint directory as a [`checkpoint::TensorSource`] under brain-canonical names,
//! read where it lies: no conversion, no rewritten copy, no expanded tensor.
//!
//! This module is the declarative half: which stored tensor is which canonical parameter ([`classify`]), which norm
//! gains are stored as `w` and applied as `1 + w`, how a fused-experts release (`mlp.experts.gate_up_proj`) is cut into
//! per-expert views, and the two-way coverage check against the config. What the stored bytes ARE (bf16, FP8 with a
//! per-tensor scale, ModelOpt NVFP4, ...) is `checkpoint::packed`'s: the tensor's encoding comes from the release's
//! own quantization declaration and its sibling scale tensors, so a model builder asks the source for each tensor in
//! the format it is stored in. The vision tower (`model.visual.*`) and the MTP head (`mtp.*`) are not part of the text
//! decoder and are skipped by name.

use std::path::Path;

use checkpoint::packed::{Binding, CodecRegistry, PackedReader, PackedSource, PackedTensorReader, SlicedReader};
use serde_json::Value;

use crate::config::{LayerType, Qwen35Config};

/// The [`Qwen35Config`] of the HF checkpoint in `dir`: the `text_config` object of a multimodal release or the top level
/// of a text-only one.
pub fn hf_config(dir: &Path) -> Result<Qwen35Config, String> {
    let path = dir.join("config.json");
    let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("qwen35moe config: read {}: {e}", path.display()))?)
        .map_err(|e| format!("qwen35moe config: {}: {e}", path.display()))?;
    let c = if top["text_config"].is_object() { &top["text_config"] } else { &top };
    let rope = &c["rope_parameters"];
    let pick = |keys: &[&str]| keys.iter().find_map(|k| c.get(*k).filter(|v| !v.is_null()).or_else(|| rope.get(*k).filter(|v| !v.is_null()))).cloned();
    let mut j = serde_json::Map::new();
    j.insert("model".to_string(), Value::String("qwen35moe".to_string()));
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
        ("num_experts", &["num_experts"]),
        ("num_experts_per_tok", &["num_experts_per_tok"]),
        ("moe_intermediate_size", &["moe_intermediate_size"]),
        ("shared_expert_intermediate_size", &["shared_expert_intermediate_size"]),
    ] {
        if let Some(v) = pick(from) {
            j.insert(to.to_string(), v);
        }
    }
    let tie = top["tie_word_embeddings"].as_bool().or_else(|| c["tie_word_embeddings"].as_bool()).unwrap_or(false);
    j.insert("tie_word_embeddings".to_string(), Value::Bool(tie));
    Qwen35Config::from_json_checked(&Value::Object(j))
}

/// What a stored tensor name is to this model.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// Not part of the text decoder (vision tower, MTP head).
    Skip,
    /// One canonical parameter.
    Param(String),
    /// `mlp.experts.gate_up_proj` of layer `l`: `[experts, 2 * ff, d]`, every expert's gate rows then up rows.
    FusedGateUp(usize),
    /// `mlp.experts.down_proj` of layer `l`: `[experts, d, ff]`.
    FusedDown(usize),
}

/// Which canonical parameter the stored tensor `name` is.
pub(crate) fn classify(name: &str, cfg: &Qwen35Config, file_layers: usize) -> Result<Class, String> {
    if name.starts_with("mtp.") || name.starts_with("model.visual") || name.starts_with("visual.") {
        return Ok(Class::Skip);
    }
    let Some(rest) = name.strip_prefix("model.language_model.").or_else(|| name.strip_prefix("model.")) else {
        return match name {
            "lm_head.weight" => Ok(Class::Param("lm_head.weight".to_string())),
            _ => Err(format!("qwen35moe source: unrecognized tensor {name}")),
        };
    };
    match rest {
        "embed_tokens.weight" => return Ok(Class::Param("tok.weight".to_string())),
        "norm.weight" => return Ok(Class::Param("norm.weight".to_string())),
        _ => {}
    }
    let layer = rest.strip_prefix("layers.").ok_or_else(|| format!("qwen35moe source: unrecognized tensor {name}"))?;
    let (idx, leaf) = layer.split_once('.').ok_or_else(|| format!("qwen35moe source: cannot split a layer index from {name}"))?;
    let l: usize = idx.parse().map_err(|_| format!("qwen35moe source: non-numeric layer index in {name}"))?;
    // A model truncated to its first layers (a profiling or bring-up build) skips the rest of the file's layers.
    if l >= cfg.n_layers as usize && l < file_layers {
        return Ok(Class::Skip);
    }
    let ty = *cfg.layer_types().get(l).ok_or_else(|| format!("qwen35moe source: {name} is layer {l} of a {}-layer config", cfg.n_layers))?;
    let canon = |suffix: &str| Ok(Class::Param(format!("blocks.{l}.{suffix}")));
    let unknown = || Err(format!("qwen35moe source: unrecognized leaf {leaf:?} of {name}"));
    match leaf {
        "input_layernorm.weight" => canon("ln1.weight"),
        "post_attention_layernorm.weight" => canon("ln2.weight"),
        "mlp.gate.weight" => canon("mlp.router.weight"),
        "mlp.shared_expert_gate.weight" => canon("mlp.shared_expert_gate.weight"),
        "mlp.experts.gate_up_proj" => Ok(Class::FusedGateUp(l)),
        "mlp.experts.down_proj" => Ok(Class::FusedDown(l)),
        _ => {
            if let Some(proj) = leaf.strip_prefix("mlp.shared_expert.").and_then(|p| p.strip_suffix("_proj.weight")).filter(|p| matches!(*p, "gate" | "up" | "down")) {
                return canon(&format!("mlp.shared_expert.{proj}.weight"));
            }
            if let Some(expert) = leaf.strip_prefix("mlp.experts.") {
                let parsed = expert.split_once('.').and_then(|(e, tail)| Some((e.parse::<u32>().ok()?, tail.strip_suffix("_proj.weight")?)));
                return match parsed {
                    Some((e, proj)) if matches!(proj, "gate" | "up" | "down") && e < cfg.n_experts => canon(&format!("mlp.experts.{e}.{proj}.weight")),
                    _ => unknown(),
                };
            }
            match (ty, leaf.split_once('.')) {
                (LayerType::Linear, Some(("linear_attn", sub)))
                    if matches!(sub, "in_proj_qkv.weight" | "in_proj_z.weight" | "in_proj_b.weight" | "in_proj_a.weight" | "out_proj.weight" | "A_log" | "dt_bias" | "norm.weight" | "conv1d.weight") =>
                {
                    canon(&format!("linear_attn.{sub}"))
                }
                (LayerType::Full, Some(("self_attn", sub))) if matches!(sub, "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "o_proj.weight" | "q_norm.weight" | "k_norm.weight") => {
                    canon(&format!("self_attn.{sub}"))
                }
                _ => unknown(),
            }
        }
    }
}

/// Norm gains stored as `w` and applied as `1 + w` (the zero-centred `RMSNorm` of the Qwen3.5 family). The gated
/// `linear_attn.norm` is a plain weight and is not among them.
pub(crate) fn is_plain_rmsnorm_weight(canon: &str) -> bool {
    canon == "norm.weight" || canon.ends_with(".ln1.weight") || canon.ends_with(".ln2.weight") || canon.ends_with(".self_attn.q_norm.weight") || canon.ends_with(".self_attn.k_norm.weight")
}

/// Open the checkpoint directory `dir` for model `cfg`. Reads only headers. Fails, naming the tensors, when the
/// checkpoint does not cover `cfg.param_list()` or holds a text tensor that maps to no parameter.
pub fn open_dir(dir: &Path, cfg: &Qwen35Config) -> Result<PackedSource, String> {
    open_dir_of_layers(dir, cfg, cfg.n_layers as usize)
}

/// [`open_dir`] for `cfg` truncated to its first layers: the checkpoint's later layers are left unread. The model is the
/// release's first `cfg.n_layers` layers, embedding and head included - a way to bring up and profile a release on a
/// small card layer by layer.
pub fn open_dir_truncated(dir: &Path, cfg: &Qwen35Config) -> Result<PackedSource, String> {
    open_dir_of_layers(dir, cfg, hf_config(dir)?.n_layers as usize)
}

fn open_dir_of_layers(dir: &Path, cfg: &Qwen35Config, file_layers: usize) -> Result<PackedSource, String> {
    let reader = PackedReader::open(dir).map_err(|e| format!("qwen35moe source: {e}"))?;
    let names = reader.names();
    let (ff, d, e) = (cfg.moe_intermediate_size as u64, cfg.d_model as u64, cfg.n_experts as u64);
    let mut sliced = SlicedReader::new(Box::new(reader));
    let mut bindings: Vec<(String, Binding)> = Vec::with_capacity(names.len());
    let mut seen = std::collections::HashSet::new();
    let mut bind = |canon: String, source: &str, bindings: &mut Vec<(String, Binding)>| -> Result<(), String> {
        if !seen.insert(canon.clone()) {
            return Err(format!("qwen35moe source: two stored tensors map onto {canon}"));
        }
        let add = if is_plain_rmsnorm_weight(&canon) { 1.0 } else { 0.0 };
        bindings.push((canon, Binding { source: source.to_string(), add }));
        Ok(())
    };
    for name in &names {
        match classify(name, cfg, file_layers)? {
            Class::Skip => {}
            Class::Param(canon) => bind(canon, name, &mut bindings)?,
            Class::FusedGateUp(l) => {
                for x in 0..e {
                    let base = x * 2 * ff * d;
                    for (proj, from) in [("gate", base), ("up", base + ff * d)] {
                        let view = format!("{name}#{proj}.{x}");
                        sliced = sliced.with_view(&view, name, from..from + ff * d, vec![ff, d])?;
                        bind(format!("blocks.{l}.mlp.experts.{x}.{proj}.weight"), &view, &mut bindings)?;
                    }
                }
            }
            Class::FusedDown(l) => {
                for x in 0..e {
                    let view = format!("{name}#down.{x}");
                    sliced = sliced.with_view(&view, name, x * d * ff..(x + 1) * d * ff, vec![d, ff])?;
                    bind(format!("blocks.{l}.mlp.experts.{x}.down.weight"), &view, &mut bindings)?;
                }
            }
        }
    }
    let src = PackedSource::new(Box::new(sliced), CodecRegistry::reference(), bindings).map_err(|e| format!("qwen35moe source: {e}"))?;
    check_coverage(&src, cfg)?;
    Ok(src)
}

/// Two-way coverage against `cfg.param_list()` from shapes alone.
fn check_coverage(src: &PackedSource, cfg: &Qwen35Config) -> Result<(), String> {
    let expected: Vec<(String, usize)> = cfg.param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).collect();
    let mut problems = Vec::new();
    for (name, numel) in &expected {
        match checkpoint::TensorSource::numel(src, name) {
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
    Err(format!("qwen35moe source: checkpoint does not match the config ({total} problems): {}", problems.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_names_map_to_canonical_parameters_and_the_vision_tower_and_mtp_are_skipped() {
        let cfg = Qwen35Config::tiny();
        let p = |n: &str| Ok(Class::Param(n.to_string()));
        assert_eq!(classify("model.language_model.embed_tokens.weight", &cfg, 8), p("tok.weight"));
        assert_eq!(classify("model.language_model.layers.3.self_attn.q_proj.weight", &cfg, 8), p("blocks.3.self_attn.q_proj.weight"));
        assert_eq!(classify("model.language_model.layers.0.linear_attn.A_log", &cfg, 8), p("blocks.0.linear_attn.A_log"));
        assert_eq!(classify("model.language_model.layers.2.mlp.gate.weight", &cfg, 8), p("blocks.2.mlp.router.weight"));
        assert_eq!(classify("model.language_model.layers.2.mlp.shared_expert.up_proj.weight", &cfg, 8), p("blocks.2.mlp.shared_expert.up.weight"));
        assert_eq!(classify("model.language_model.layers.2.mlp.experts.5.down_proj.weight", &cfg, 8), p("blocks.2.mlp.experts.5.down.weight"));
        assert_eq!(classify("model.layers.1.mlp.experts.gate_up_proj", &cfg, 8), Ok(Class::FusedGateUp(1)));
        assert_eq!(classify("model.visual.blocks.0.attn.qkv.weight", &cfg, 8), Ok(Class::Skip));
        assert_eq!(classify("mtp.fc.weight", &cfg, 8), Ok(Class::Skip));
        assert!(classify("model.language_model.layers.0.self_attn.q_proj.weight", &cfg, 8).is_err(), "a GDN layer has no self_attn");
        assert!(classify("model.language_model.layers.0.mlp.experts.99.up_proj.weight", &cfg, 8).is_err(), "expert out of range");
    }
}
