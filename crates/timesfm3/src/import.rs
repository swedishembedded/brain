// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight import - strict, 1:1 by the reference's own tensor names.
//!
//! TimesFM-3 ships native fp32 `model.safetensors` directly (no `.pth`
//! conversion step, unlike FinCast), so the loader is verify-and-copy.
//! Strictness (a duplicate destination, a missing/wrong-numel declared param,
//! or an unused leftover tensor) is a hard error: a silently half-loaded
//! model is worse than one that refuses to load. Unlike FinCast/Chronos-2,
//! this checkpoint has NO non-persistent buffers to filter out - every
//! tensor its header names is a learnable param, confirmed by
//! `t0_param_layout.rs`'s live gate finding zero unmapped tensors either way.

use crate::config::Timesfm3Config;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Read TimesFM-3 weights (a single safetensors file or an HF-style dir)
/// into a name->values map, validated strictly against `cfg.param_list()`.
pub fn load_hf(cfg: &Timesfm3Config, path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
    let p = Path::new(path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(path)
    }?;

    let mut have: HashMap<String, Vec<f32>> = HashMap::new();
    for t in tensors {
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

/// Read the config that accompanies a fetched checkpoint directory
/// (`config.json`, upstream's own nested schema) - falls back to
/// [`Timesfm3Config::default`] when `path` names a bare `model.safetensors`
/// file with no sibling `config.json` to read.
pub fn load_config(path: &str) -> Result<Timesfm3Config, String> {
    let p = Path::new(path);
    let config_path = if p.is_dir() { p.join("config.json") } else { p.with_file_name("config.json") };
    if !config_path.exists() {
        return Ok(Timesfm3Config::default());
    }
    let bytes = std::fs::read(&config_path).map_err(|e| format!("import: reading {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("import: parsing {}: {e}", config_path.display()))?;
    Timesfm3Config::from_hf_config_json(&v)
}

/// Import a TimesFM-3 safetensors file (or dir) -> a brain `.safetensors`
/// container (config + tensors). Reads the checkpoint's own `config.json`
/// when present (see [`load_config`]), so a future differently-sized release
/// imports without a code change.
pub fn import(ckpt: &str, out_path: &str) -> Result<(), String> {
    let cfg = load_config(ckpt)?;
    import_cfg(&cfg, ckpt, out_path)
}

/// `import`, parametrized over the config (split out so tests can exercise
/// the real streaming path at `tiny()` scale instead of the real 330M-param
/// model). Streams one source tensor at a time (never the whole checkpoint
/// in memory); mirrors `load_hf`'s validation but writes straight through
/// `StWriter`.
fn import_cfg(cfg: &Timesfm3Config, ckpt: &str, out_path: &str) -> Result<(), String> {
    // Output shapes are flat (element counts, not param_list's real shape) -
    // the same convention fincast/chronos2's containers use.
    let plan: Vec<(String, Vec<u64>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| (name, vec![shape.iter().product::<usize>() as u64]))
        .collect();
    let planned: HashSet<&str> = plan.iter().map(|(n, _)| n.as_str()).collect();
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), None)
        .map_err(|e| format!("import: creating output: {e}"))?;

    let p = Path::new(ckpt);
    let reader = if p.is_dir() {
        checkpoint::weightio::WeightReader::open_hf_dir(p)
    } else {
        checkpoint::weightio::WeightReader::open(ckpt)
    }
    .map_err(|e| format!("import: opening checkpoint: {e}"))?;

    let mut extra: Vec<String> = Vec::new();
    let mut err: Option<String> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
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
        return Err(format!("import: {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]));
    }
    writer.finish().map_err(|e| format!("import: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_config_falls_back_to_default_when_no_config_json_is_present() {
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("timesfm3-import-nocfg-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("model.safetensors");
        std::fs::write(&weights, b"").ok();
        let c = load_config(weights.to_str().unwrap()).unwrap();
        assert_eq!(c, Timesfm3Config::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_config_reads_the_real_checkpoints_config_json() {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/hf_config.json");
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("timesfm3-import-cfg-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(src, dir.join("config.json")).unwrap();
        let c = load_config(dir.join("model.safetensors").to_str().unwrap()).unwrap();
        assert_eq!(c, Timesfm3Config::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_import_writes_every_tensor_with_two_way_coverage() {
        // A synthetic single-file HF checkpoint holding every `tiny()` param,
        // nothing more and nothing less. Goes through `import_cfg` directly
        // (real `import` is pinned to the full 330M-param default, too big
        // for a unit test).
        let cfg = Timesfm3Config::tiny();
        let pid = std::process::id();
        let src = std::env::temp_dir().join(format!("timesfm3-import-src-{pid}.safetensors"));

        let list = cfg.param_list();
        let plan: Vec<(String, Vec<u64>)> =
            list.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        let mut w =
            checkpoint::weightio::StWriter::create(src.to_str().unwrap(), &plan, &serde_json::Value::Null, None)
                .unwrap();
        let mut expect: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, (name, shape)) in list.iter().enumerate() {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|j| (i * 1000 + j) as f32 * 0.001).collect();
            w.write(name, &data).unwrap();
            expect.insert(name.clone(), data);
        }
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("timesfm3-import-out-{pid}.safetensors"));
        import_cfg(&cfg, src.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        for (name, data) in &expect {
            assert_eq!(reader.tensor(name).unwrap(), *data, "{name}");
        }
        assert_eq!(reader.names().count(), expect.len(), "two-way coverage: nothing missing, nothing extra");

        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn streaming_import_rejects_an_unmapped_source_tensor() {
        let cfg = Timesfm3Config::tiny();
        let pid = std::process::id();
        let src = std::env::temp_dir().join(format!("timesfm3-import-extra-src-{pid}.safetensors"));

        let mut plan: Vec<(String, Vec<u64>)> =
            cfg.param_list().iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        plan.push(("totally.unmapped.tensor".to_string(), vec![1]));
        let mut w =
            checkpoint::weightio::StWriter::create(src.to_str().unwrap(), &plan, &serde_json::Value::Null, None)
                .unwrap();
        for (name, shape) in cfg.param_list() {
            let n: usize = shape.iter().product();
            w.write(&name, &vec![0.0; n]).unwrap();
        }
        w.write("totally.unmapped.tensor", &[0.0]).unwrap();
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("timesfm3-import-extra-out-{pid}.safetensors"));
        let err = import_cfg(&cfg, src.to_str().unwrap(), out.to_str().unwrap()).unwrap_err();
        assert!(err.contains("unmapped"), "{err}");

        std::fs::remove_file(&src).ok();
    }

    #[test]
    fn live_import_round_trips_the_real_checkpoint() {
        let Ok(path) = std::env::var("BRAIN_TIMESFM3") else {
            brain_testutil::skip("BRAIN_TIMESFM3 unset");
            return;
        };
        let cfg = load_config(&path).unwrap();
        assert_eq!(cfg, Timesfm3Config::default());
        let weights = load_hf(&cfg, &path).unwrap();
        assert_eq!(weights.len(), cfg.param_list().len());
        let (name, shape) = &cfg.param_list()[0];
        assert_eq!(weights[name].len(), shape.iter().product::<usize>());
    }
}
