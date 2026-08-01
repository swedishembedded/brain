// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load the reference WorldMirror-2 `model.safetensors` into a brain init map,
//! and convert it once into a brain `.safetensors` container.
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

/// One-time conversion: reference safetensors → brain `.safetensors` with the
/// config embedded, tensors in `param_list` order. Returns the tensor count.
/// Streams one source tensor at a time (never the whole 5 GB checkpoint in
/// memory) — mirrors [`load`]'s validation but writes straight through
/// `StWriter` instead of building the whole init map first.
pub fn convert(src: &str, out_path: &str, cfg: &MirrorConfig) -> Result<usize, String> {
    let list = cfg.param_list();
    let plan: Vec<(String, Vec<u64>)> =
        list.iter().map(|(name, shape)| (name.clone(), shape.iter().map(|&s| s as u64).collect())).collect();
    let expected: HashMap<String, Vec<usize>> = list.into_iter().collect();
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
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &config, None)
        .map_err(|e| format!("convert: creating output: {e}"))?;

    let p = Path::new(src);
    let reader = if p.is_dir() {
        checkpoint::weightio::WeightReader::open_hf_dir(p)
    } else {
        checkpoint::weightio::WeightReader::open(src)
    }
    .map_err(|e| format!("convert: opening checkpoint: {e}"))?;

    let mut unexpected: Vec<String> = Vec::new();
    let mut err: Option<String> = None;
    reader.for_each(|name, shape, data| {
        if err.is_some() {
            return;
        }
        match expected.get(name) {
            None => unexpected.push(name.to_string()),
            Some(exp_shape) => {
                let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                if &got != exp_shape {
                    err = Some(format!(
                        "tensor `{name}`: file shape {got:?}, the model expects {exp_shape:?} — wrong \
                         checkpoint (WorldMirror-1 lacks the depth-mask channel)?"
                    ));
                    return;
                }
                if let Err(e) = writer.write(name, &data) {
                    err = Some(format!("convert: {e}"));
                }
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(format!(
            "the checkpoint has {} tensor(s) the model does not declare. First few: {:?}",
            unexpected.len(),
            &unexpected[..unexpected.len().min(5)]
        ));
    }
    writer.finish().map_err(|e| format!("convert: {e}"))?;
    Ok(plan.len())
}

/// Load a converted `.safetensors` container back into an init map, verifying the
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in for the real (24-depth, 1024-dim) config: same fixed
    /// architecture shape (every dpt_head/vit_block loop still runs) but small
    /// enough dims that the fixture is a few hundred KB, not 5 GB.
    fn tiny_cfg() -> MirrorConfig {
        MirrorConfig {
            depth: 1,
            dim: 8,
            heads: 2,
            mlp_ratio: 1,
            patch: 2,
            img: 4,
            reg_tokens: 1,
            tap_levels: [0, 0, 0, 0],
            dpt_proj: [2, 2, 2, 2],
            dpt_feat: 8,
            cam_blocks: 1,
            cam_params: 2,
        }
    }

    #[test]
    fn streaming_convert_writes_every_tensor_and_rejects_unexpected() {
        let cfg = tiny_cfg();
        let list = cfg.param_list();
        let pid = std::process::id();

        // A synthetic single-file HF checkpoint holding every param_list tensor.
        let plan: Vec<(String, Vec<u64>)> =
            list.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        let src = std::env::temp_dir().join(format!("mirror-convert-src-{pid}.safetensors"));
        let mut w = checkpoint::weightio::StWriter::create(
            src.to_str().unwrap(),
            &plan,
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        let mut expect: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, (name, shape)) in list.iter().enumerate() {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|j| (i * 1000 + j) as f32 * 0.001).collect();
            w.write(name, &data).unwrap();
            expect.insert(name.clone(), data);
        }
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("mirror-convert-out-{pid}.safetensors"));
        let n = convert(src.to_str().unwrap(), out.to_str().unwrap(), &cfg).unwrap();
        assert_eq!(n, list.len());

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        for (name, data) in &expect {
            assert_eq!(reader.tensor(name).unwrap(), *data, "{name}");
        }
        assert_eq!(reader.names().count(), expect.len());
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&out).ok();

        // A checkpoint with one extra, undeclared tensor must be rejected (the
        // "unexpected" streaming check, exercised end to end).
        let mut plan2 = plan.clone();
        plan2.push(("some.extra.tensor".to_string(), vec![2]));
        let src2 = std::env::temp_dir().join(format!("mirror-convert-src2-{pid}.safetensors"));
        let mut w2 = checkpoint::weightio::StWriter::create(
            src2.to_str().unwrap(),
            &plan2,
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        for (name, shape) in &list {
            let n: usize = shape.iter().product();
            w2.write(name, &vec![0.0f32; n]).unwrap();
        }
        w2.write("some.extra.tensor", &[1.0, 2.0]).unwrap();
        w2.finish().unwrap();

        let out2 = std::env::temp_dir().join(format!("mirror-convert-out2-{pid}.safetensors"));
        let err = convert(src2.to_str().unwrap(), out2.to_str().unwrap(), &cfg).unwrap_err();
        assert!(err.contains("some.extra.tensor"), "error should name the unexpected tensor: {err}");
        std::fs::remove_file(&src2).ok();
    }
}
