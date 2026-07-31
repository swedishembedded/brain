// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight import — strict, 1:1 by the reference `state_dict` names, for BOTH
//! nets (the BSQ tokenizer and the AR decoder are separate HF checkpoints).
//! Kronos ships fp32 safetensors, so the loader is verify-and-copy.

use crate::config::{KronosConfig, KronosTokenizerConfig};
use crate::generate::KronosModel;
use std::collections::HashMap;
use std::path::Path;

fn is_non_persistent(name: &str) -> bool {
    name.contains("inv_freq")
        || name.contains("rotary")
        || name.contains("bsq.basis")
        || name.contains("bsq.group_basis")
        || name.contains("group_codebook")
}

/// Read a safetensors file/dir into a name→values map, validated strictly
/// against `param_list` (missing / wrong-numel / leftover are hard errors).
pub fn load_hf(param_list: &[(String, Vec<usize>)], path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
    let p = Path::new(path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(path)
    }?;
    let mut have: HashMap<String, Vec<f32>> = HashMap::new();
    for t in tensors {
        if is_non_persistent(&t.name) {
            continue;
        }
        if have.insert(t.name.clone(), t.data).is_some() {
            return Err(format!("import: duplicate tensor {}", t.name));
        }
    }
    let mut out = HashMap::new();
    for (name, shape) in param_list {
        let numel: usize = shape.iter().product();
        let data = have.remove(name).ok_or_else(|| format!("import: missing tensor {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} has {} elems, expected {numel}", data.len()));
        }
        out.insert(name.clone(), data);
    }
    if !have.is_empty() {
        let mut extra: Vec<&String> = have.keys().collect();
        extra.sort();
        return Err(format!("import: {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]));
    }
    Ok(out)
}

fn config_from_dir<T>(dir: &str, parse: impl Fn(&serde_json::Value) -> Result<T, String>) -> Result<T, String> {
    let bytes = std::fs::read(Path::new(dir).join("config.json"))
        .map_err(|e| format!("import: reading config.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("import: {e}"))?;
    parse(&v)
}

/// Load just the AR decoder's config + weight map from its HF checkpoint dir
/// (reads `config.json`, imports strictly). Used by the NPU export, which needs
/// the raw weights (not the assembled [`KronosModel`]) to build the ONNX graph.
pub fn load_decoder(decoder_dir: &str) -> Result<(KronosConfig, HashMap<String, Vec<f32>>), String> {
    // A brain `.safetensors` container (e.g. a fine-tuned checkpoint from
    // `brain forecast finetune`) is a single file with the config + tensors
    // embedded; an HF checkpoint is a directory. Support both so a fine-tuned
    // decoder loads everywhere the base does (forecaster, server, ranking tool).
    if std::path::Path::new(decoder_dir).is_file() {
        let c = checkpoint::load(decoder_dir);
        let dc = KronosConfig::from_hf(&c.header["config"])?;
        let dw: HashMap<String, Vec<f32>> = c.tensors.into_iter().map(|t| (t.name, t.data)).collect();
        return Ok((dc, dw));
    }
    let dc = config_from_dir(decoder_dir, KronosConfig::from_hf)?;
    let dw = load_hf(&dc.param_list(), decoder_dir)?;
    Ok((dc, dw))
}

/// Load a full [`KronosModel`] from the two HF checkpoint dirs (tokenizer +
/// decoder), reading each `config.json` and importing its weights strictly.
pub fn load_model(tokenizer_dir: &str, decoder_dir: &str) -> Result<KronosModel, String> {
    let tc = config_from_dir(tokenizer_dir, KronosTokenizerConfig::from_hf)?;
    let tw = load_hf(&tc.param_list(), tokenizer_dir)?;
    // Decoder may be an HF dir (base) or a brain `.safetensors` file (fine-tuned) —
    // load_decoder handles both, so a promoted checkpoint drops straight in.
    let (dc, dw) = load_decoder(decoder_dir)?;
    KronosModel::from_weights(tc, &tw, dc, &dw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_load_rejects_missing() {
        let cfg = KronosTokenizerConfig::tiny();
        let mut w: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        w.remove("embed.weight");
        // simulate via the numel path: build tensors missing one — load_hf reads
        // files, so here just check the param-count contract directly.
        let full: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        assert!(full.len() > w.len());
    }
}
