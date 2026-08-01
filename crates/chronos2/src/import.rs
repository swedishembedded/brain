// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight import — strict, 1:1 by the reference's own tensor names.
//!
//! Chronos-2 ships native **fp32** safetensors (`torch_dtype: float32`), so there
//! is no bf16/f16 conversion to do — the loader is verify-and-copy. Strictness
//! (a duplicate destination, a missing/wrong-numel declared param, or an unused
//! leftover tensor beyond the known non-persistent buffers) is a hard error: a
//! silently half-loaded model is worse than one that refuses to load.

use crate::config::Chronos2Config;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Tensors the checkpoint header carries that are NOT learnable params
/// (recomputed in code): RoPE inverse frequencies, the quantile-levels buffer.
fn is_non_persistent(name: &str) -> bool {
    name.contains("inv_freq") || name.ends_with("quantiles") || name.contains("rope_embed")
}

/// Read `amazon/chronos-2` weights (a single safetensors file or an HF dir) into
/// a name→values map, validated strictly against `cfg.param_list()`.
pub fn load_hf(cfg: &Chronos2Config, path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
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
        let data = have
            .remove(&name)
            .ok_or_else(|| format!("import: missing tensor for {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} has {} elems, expected {numel}", data.len()));
        }
        out.insert(name, data);
    }
    if !have.is_empty() {
        let mut extra: Vec<&String> = have.keys().collect();
        extra.sort();
        return Err(format!(
            "import: {} unmapped tensors, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(5)]
        ));
    }
    Ok(out)
}

/// Read the HF `config.json` next to a checkpoint dir.
pub fn config_from_dir(dir: &str) -> Result<Chronos2Config, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| format!("import: reading config.json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("import: parsing config.json: {e}"))?;
    Chronos2Config::from_hf(&v)
}

/// Import an HF checkpoint dir → a brain `.safetensors` container (config +
/// tensors), streaming one source tensor at a time (never the whole checkpoint
/// in memory). Mirrors `load_hf`'s validation (strict, same error shapes) but
/// writes straight through `StWriter` instead of building an in-RAM map first.
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let cfg = config_from_dir(hf_dir)?;
    // Output shapes are flat (matches this crate's long-standing convention:
    // the container stores element counts, not param_list's real shape).
    let plan: Vec<(String, Vec<u64>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| (name, vec![shape.iter().product::<usize>() as u64]))
        .collect();
    let planned: HashSet<&str> = plan.iter().map(|(n, _)| n.as_str()).collect();
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), None)
        .map_err(|e| format!("import: creating output: {e}"))?;

    let p = Path::new(hf_dir);
    let reader = if p.is_dir() {
        checkpoint::weightio::WeightReader::open_hf_dir(p)
    } else {
        checkpoint::weightio::WeightReader::open(hf_dir)
    }
    .map_err(|e| format!("import: opening checkpoint: {e}"))?;

    let mut extra: Vec<String> = Vec::new();
    let mut err: Option<String> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() || is_non_persistent(name) {
            return;
        }
        if !planned.contains(name) {
            extra.push(name.to_string());
            return;
        }
        if let Err(e) = writer.write(name, &data) {
            err = Some(format!("import: {e}"));
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if !extra.is_empty() {
        extra.sort();
        return Err(format!(
            "import: {} unmapped tensors, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(5)]
        ));
    }
    writer.finish().map_err(|e| format!("import: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Chronos2;

    fn zero_weights(cfg: &Chronos2Config) -> HashMap<String, Vec<f32>> {
        cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect()
    }

    #[test]
    fn weights_roundtrip_through_the_checkpoint_container() {
        // save zero weights -> load -> the loaded model still forecasts the mean.
        let cfg = Chronos2Config::tiny();
        let weights = zero_weights(&cfg);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                (name.clone(), vec![n as u64], weights[&name].clone())
            })
            .collect();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("chronos2-rt-{}.safetensors", std::process::id()));
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);

        let loaded = Chronos2::load(path.to_str().unwrap()).unwrap();
        if std::env::var("MOE_SKIP_GPU_TESTS").is_err() {
            let ctx: Vec<f32> = (0..30).map(|i| 2.0 + i as f32 * 0.1).collect();
            let mean = ctx.iter().sum::<f32>() / ctx.len() as f32;
            let out = loaded.forecast_quantiles(&ctx, 4);
            assert!(out.iter().all(|&v| (v - mean).abs() < 1e-2), "loaded model lost the mean");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_import_writes_every_tensor_and_drops_non_persistent_buffers() {
        // A synthetic HF checkpoint dir: config.json + a single model.safetensors
        // holding every `tiny()` param plus one non-persistent buffer that must
        // be silently skipped (never reaches the output plan).
        let cfg = Chronos2Config::tiny();
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("chronos2-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg.to_json()).unwrap()).unwrap();

        let list = cfg.param_list();
        let mut plan: Vec<(String, Vec<u64>)> =
            list.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        plan.push(("model.rope_embed.inv_freq".to_string(), vec![4]));
        let mut w = checkpoint::weightio::StWriter::create(
            dir.join("model.safetensors").to_str().unwrap(),
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
        w.write("model.rope_embed.inv_freq", &[0.0, 0.0, 0.0, 0.0]).unwrap();
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("chronos2-import-out-{pid}.safetensors"));
        import(dir.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        for (name, data) in &expect {
            assert_eq!(reader.tensor(name).unwrap(), *data, "{name}");
        }
        assert_eq!(reader.names().count(), expect.len(), "non-persistent buffer must not be carried through");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn config_json_roundtrips_through_the_container_header() {
        // to_json is parseable back by from_hf (used by Chronos2::load).
        let cfg = Chronos2Config::default();
        let back = Chronos2Config::from_hf(&cfg.to_json()).unwrap();
        assert_eq!(cfg, back);
    }
}
