// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load the reference WorldMirror-2 `model.safetensors` into a brain init map,
//! and convert it once into a brain `.weights` container.
//!
//! Strict 1:1 name copy (the ZipDepth precedent): `MirrorConfig::param_list`
//! carries the checkpoint's own names, gated device-free against the committed
//! header fixture, so this importer only reads, verifies shapes, and errors
//! loudly on any divergence — naming exactly what diverged. The 47 RoPE
//! `periods` aliases live in safetensors `__metadata__` (not tensors) and thus
//! never reach us; the single stored buffer imports under its own name.

use std::collections::HashMap;
use std::path::Path;

use crate::config::MirrorConfig;

/// Read a safetensors file (or a HF model dir with an optional shard index)
/// and return the init map for a model of shape `cfg`.
pub fn load(path: &str, cfg: &MirrorConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let p = Path::new(path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)?
    } else {
        checkpoint::safetensors::read(path)?
    };
    let expected: HashMap<String, Vec<usize>> = cfg.param_list().into_iter().collect();

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    let mut unexpected = Vec::new();
    for t in tensors {
        match expected.get(&t.name) {
            None => unexpected.push(t.name),
            Some(shape) => {
                if t.shape != *shape {
                    return Err(format!(
                        "tensor `{}`: file shape {:?}, the model expects {:?} — wrong \
                         checkpoint (WorldMirror-1 lacks the depth-mask channel)?",
                        t.name, t.shape, shape
                    ));
                }
                out.insert(t.name, t.data);
            }
        }
    }

    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(format!(
            "the checkpoint has {} tensor(s) the model does not declare. First few: {:?}",
            unexpected.len(),
            &unexpected[..unexpected.len().min(5)]
        ));
    }
    let mut missing: Vec<&String> = expected.keys().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "the checkpoint is missing {} tensor(s) the model needs. First few: {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        ));
    }
    Ok(out)
}

/// One-time conversion: reference safetensors → brain `.weights` with the
/// config embedded, tensors in `param_list` order. Returns the tensor count.
pub fn convert(src: &str, out_path: &str, cfg: &MirrorConfig) -> Result<usize, String> {
    let mut map = load(src, cfg)?;
    let list = cfg.param_list();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = list
        .iter()
        .map(|(name, shape)| {
            let data = map.remove(name).expect("verified by load");
            (name.clone(), shape.iter().map(|&s| s as u64).collect(), data)
        })
        .collect();
    let config = serde_json::json!({
        "model": "worldmirror2",
        "depth": cfg.depth,
        "dim": cfg.dim,
        "heads": cfg.heads,
        "mlp_ratio": cfg.mlp_ratio,
        "patch": cfg.patch,
        "img": cfg.img,
        "reg_tokens": cfg.reg_tokens,
        "tap_levels": cfg.tap_levels,
        "dpt_proj": cfg.dpt_proj,
        "dpt_feat": cfg.dpt_feat,
        "cam_blocks": cfg.cam_blocks,
        "cam_params": cfg.cam_params,
    });
    checkpoint::save(out_path, config, &tensors);
    Ok(tensors.len())
}

/// Load a converted `.weights` container back into an init map, verifying the
/// layout against `cfg` (same strictness as [`load`], minus dtype concerns).
pub fn load_weights(path: &str, cfg: &MirrorConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let c = checkpoint::load(path);
    let expected: HashMap<String, usize> =
        cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
    let mut out = HashMap::new();
    for t in c.tensors {
        match expected.get(&t.name) {
            None => return Err(format!("`{path}`: unexpected tensor `{}`", t.name)),
            Some(&numel) if t.data.len() != numel => {
                return Err(format!(
                    "`{path}`: tensor `{}` has {} elements, expected {numel}",
                    t.name,
                    t.data.len()
                ))
            }
            Some(_) => {
                out.insert(t.name, t.data);
            }
        }
    }
    if out.len() != expected.len() {
        return Err(format!(
            "`{path}`: {} tensors present, model needs {}",
            out.len(),
            expected.len()
        ));
    }
    Ok(out)
}
