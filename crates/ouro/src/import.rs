// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint import for looped language models
// for its clients. If your team needs expertise in bringing a recurrent-depth
// checkpoint's exact tensor layout into a Rust engine then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Import a Hugging Face Ouro checkpoint (`config.json` + safetensors shards)
//! into a brain `.safetensors` container.
//!
//! brain's `matmul.wgsl` is `out = x @ Wᵀ` with `W:[out,in]` row-major, which
//! is exactly HF `nn.Linear.weight`; the embedding table is `[vocab, hidden]`
//! in both. **No tensor is transposed.** The released 2.6B checkpoint stores
//! one BF16 `model.safetensors`; the reader below also accepts the sharded
//! layout (`model.safetensors.index.json`) because the 1.4B release and
//! future sizes may use it.
//!
//! The exit gate is `early_exit_gate.weight` (`[1, hidden]`, flattened to
//! `[hidden]`) plus `early_exit_gate.bias` (`[1]`). The head is untied in the
//! released config and is imported as its own `lm_head.weight`.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::TensorSource;

use crate::config::OuroConfig;

/// Map an HF Ouro tensor name to its brain parameter name, or `None` when the
/// name is not part of this architecture (which the caller treats as an error,
/// never as a silent drop).
fn hf_to_brain(name: &str) -> Option<String> {
    if name == "model.embed_tokens.weight" {
        return Some("tok.weight".to_string());
    }
    if name == "model.norm.weight" {
        return Some("norm.weight".to_string());
    }
    if name == "model.early_exit_gate.weight" {
        return Some("gate.weight".to_string());
    }
    if name == "model.early_exit_gate.bias" {
        return Some("gate.bias".to_string());
    }
    if name == "lm_head.weight" {
        return Some("lm_head.weight".to_string());
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let leaf = match rest {
        // Sandwich normalization: one before and one after each sub-layer.
        "input_layernorm.weight" => "ln1.weight",
        "input_layernorm_2.weight" => "ln1_2.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "post_attention_layernorm_2.weight" => "ln2_2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{leaf}"))
}

/// Read an HF Ouro `config.json`.
pub fn config_from_hf(json: &str) -> Result<OuroConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("ouro config: {e}"))?;
    OuroConfig::from_json(&v)
}

/// The rename plan for a checkpoint whose tensors are `names`. Every tensor
/// must map - an unknown name is an error naming it, because dropping one
/// silently would leave a parameter at its initial value while the checkpoint
/// looked fully imported.
pub fn plan<'n>(names: impl IntoIterator<Item = &'n str>) -> Result<HashMap<String, Fetch>, String> {
    let mut plan = HashMap::new();
    let mut unmapped = Vec::new();
    for n in names {
        match hf_to_brain(n) {
            Some(bn) => {
                if plan.insert(bn.clone(), Fetch::Whole(n.to_string())).is_some() {
                    return Err(format!("import: duplicate mapping to {bn}"));
                }
            }
            None => unmapped.push(n.to_string()),
        }
    }
    if !unmapped.is_empty() {
        unmapped.sort();
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    Ok(plan)
}

/// The config and brain-named source of the HF checkpoint directory `dir`.
/// Shapes are checked two ways before any weight byte is read: every brain
/// parameter has a source, and every source is consumed.
pub fn open_dir(dir: &Path) -> Result<(OuroConfig, RemapSource<'static>), String> {
    let cfg = config_from_hf(
        &std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?,
    )?;
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {}: {e}", dir.display()))?;
    let plan = plan(reader.names())?;
    let src = RemapSource::owning(Box::new(reader), plan);
    src.validate(&cfg.param_list())?;
    Ok((cfg, src))
}

/// Import `<hf_dir>` into the brain checkpoint `out_path`. Never writes a
/// partial checkpoint.
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_as(hf_dir, out_path, None)
}

/// Like [`import`] with an explicit card id (defaults to the output stem).
pub fn import_as(hf_dir: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let (cfg, src) = open_dir(Path::new(hf_dir))?;
    let plan: Vec<(String, Vec<u64>)> = cfg.param_list().into_iter().map(|(name, numel)| (name, vec![numel as u64])).collect();
    let param_count: u64 = plan.iter().map(|(_, s)| s.iter().product::<u64>()).sum();
    let id = id_override.unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("ouro"));
    let mut card = checkpoint::st::ModelCard::new(id, "ouro");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(param_count);
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card))
        .map_err(|e| format!("create {out_path}: {e}"))?;
    for (name, _) in &plan {
        let mut result = Ok(());
        if !src.with_tensor(name, &mut |t| result = writer.write(name, t).map_err(|e| e.to_string())) {
            return Err(format!("ouro import: {name} has no source"));
        }
        result?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("imported {} tensors -> {out_path}", plan.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_hf_names() -> Vec<String> {
        let cfg = OuroConfig::tiny();
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "model.early_exit_gate.weight".to_string(),
            "model.early_exit_gate.bias".to_string(),
            "lm_head.weight".to_string(),
        ];
        for l in 0..cfg.n_layers {
            names.push(format!("model.layers.{l}.input_layernorm.weight"));
            names.push(format!("model.layers.{l}.input_layernorm_2.weight"));
            names.push(format!("model.layers.{l}.post_attention_layernorm.weight"));
            names.push(format!("model.layers.{l}.post_attention_layernorm_2.weight"));
            names.push(format!("model.layers.{l}.self_attn.q_proj.weight"));
            names.push(format!("model.layers.{l}.self_attn.k_proj.weight"));
            names.push(format!("model.layers.{l}.self_attn.v_proj.weight"));
            names.push(format!("model.layers.{l}.self_attn.o_proj.weight"));
            names.push(format!("model.layers.{l}.mlp.gate_proj.weight"));
            names.push(format!("model.layers.{l}.mlp.up_proj.weight"));
            names.push(format!("model.layers.{l}.mlp.down_proj.weight"));
        }
        names
    }

    #[test]
    fn every_brain_parameter_has_exactly_one_hf_source() {
        let cfg = OuroConfig::tiny();
        let plan = plan(tiny_hf_names().iter().map(String::as_str)).expect("plan");
        assert_eq!(plan.len(), cfg.param_list().len());
        for (name, _) in cfg.param_list() {
            assert!(plan.contains_key(&name), "no source for {name}");
        }
    }

    #[test]
    fn an_unknown_tensor_is_named_not_dropped() {
        let mut names = tiny_hf_names();
        names.push("model.rotary_emb.inv_freq".to_string());
        let err = plan(names.iter().map(String::as_str)).expect_err("unknown tensor must fail");
        assert!(err.contains("model.rotary_emb.inv_freq"), "{err}");
    }

    #[test]
    fn the_released_config_parses() {
        // The published `ByteDance/Ouro-2.6B-Thinking` config, key subset.
        let json = r#"{
            "vocab_size": 49152, "max_position_embeddings": 65536,
            "hidden_size": 2048, "num_hidden_layers": 48,
            "num_attention_heads": 16, "num_key_value_heads": 16,
            "head_dim": 128, "intermediate_size": 5632,
            "rope_theta": 1000000.0, "rms_norm_eps": 1e-06,
            "total_ut_steps": 4, "early_exit_threshold": 1.0,
            "tie_word_embeddings": false
        }"#;
        let cfg = config_from_hf(json).expect("released config parses");
        assert_eq!(cfg, OuroConfig::ouro_2_6b_thinking());
    }
}
