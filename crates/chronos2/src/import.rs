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
use std::collections::HashMap;
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

/// Import an HF checkpoint dir → a brain `.weights` container (config + tensors).
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let cfg = config_from_dir(hf_dir)?;
    let weights = load_hf(&cfg, hf_dir)?;
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
        let path = dir.join(format!("chronos2-rt-{}.weights", std::process::id()));
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
    fn config_json_roundtrips_through_the_container_header() {
        // to_json is parseable back by from_hf (used by Chronos2::load).
        let cfg = Chronos2Config::default();
        let back = Chronos2Config::from_hf(&cfg.to_json()).unwrap();
        assert_eq!(cfg, back);
    }
}
