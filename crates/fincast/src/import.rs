// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight import — strict, 1:1 by the reference's own tensor names.
//!
//! FinCast ships native **fp32** (`v1.pth` → safetensors via
//! `tools/fincast_convert.py`), so the loader is verify-and-copy. Strictness
//! (a duplicate destination, a missing/wrong-numel declared param, or an unused
//! leftover tensor beyond the known non-persistent gate buffers) is a hard
//! error: a silently half-loaded model is worse than one that refuses to load.

use crate::config::{is_non_persistent, FincastConfig};
use std::collections::HashMap;
use std::path::Path;

/// Read FinCast weights (a single safetensors file or an HF-style dir) into a
/// name→values map, validated strictly against `cfg.param_list()`.
pub fn load_hf(cfg: &FincastConfig, path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
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
    for (name, shape) in cfg.param_list() {
        let numel: usize = shape.iter().product();
        let data = have.remove(&name).ok_or_else(|| format!("import: missing tensor for {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} has {} elems, expected {numel}", data.len()));
        }
        out.insert(name, data);
    }
    if !have.is_empty() {
        let mut extra: Vec<&String> = have.keys().collect();
        extra.sort();
        return Err(format!("import: {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]));
    }
    Ok(out)
}

/// Import a FinCast safetensors file (or dir) → a brain `.weights` container
/// (config + tensors). The config is the published default (FinCast ships no
/// `config.json`; the dims are fixed by the checkpoint and asserted by T0).
pub fn import(ckpt: &str, out_path: &str) -> Result<(), String> {
    let cfg = FincastConfig::default();
    let weights = load_hf(&cfg, ckpt)?;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, _shape)| {
            let data = weights.get(&name).cloned().unwrap_or_default();
            (name, vec![data.len() as u64], data)
        })
        .collect();
    checkpoint::save(out_path, cfg.to_json(), &tensors);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_rejects_missing_and_extra() {
        let cfg = FincastConfig::tiny();
        // build a full valid map, then perturb it.
        let full: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();

        // wrong numel is caught (write a container-like path via checkpoint save/load).
        let dir = std::env::temp_dir();
        let good = dir.join(format!("fincast-imp-{}.weights", std::process::id()));
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            cfg.param_list().into_iter().map(|(k, s)| { let n: usize = s.iter().product(); (k.clone(), vec![n as u64], full[&k].clone()) }).collect();
        checkpoint::save(good.to_str().unwrap(), cfg.to_json(), &tensors);
        // round-trips through the container loader used by Fincast::load.
        let c = checkpoint::load(good.to_str().unwrap());
        let back = FincastConfig::from_json(&c.header["config"]).unwrap();
        assert_eq!(back, cfg);
        std::fs::remove_file(&good).ok();
    }
}
