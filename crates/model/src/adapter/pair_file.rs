// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The saved file of a host-side [`crate::lora::Pair`] adapter.
//!
//! A host-Pair trainer (flow-matching and diffusion families) writes its
//! adapter as a safetensors file whose `brain.config` names the family and
//! carries `rank`/`alpha` (plus family-specific fields). Every one of those
//! families had its own copy of the write and, worse, its own read: some
//! reconstructed tensors with empty shapes, which silently downgraded
//! [`crate::lora::LoraPair::load_tensors`]'s A/B-swap check to a length
//! comparison. This module is the one read/write pair, and it always carries
//! the shapes the file records.
//!
//! The family tag is checked on read: an adapter saved by one architecture
//! refused when loaded as another is an error naming the file, not a delta
//! folded into the wrong base.
//!
//! Swedish Embedded AB implements adapter file formats that every model
//! family reads through one implementation, for its clients. If your team
//! needs expertise in parameter-efficient fine-tuning infrastructure then you
//! can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use checkpoint::safetensors::StTensor;
use serde_json::Value;

/// A host-Pair adapter file as it was saved.
pub struct PairAdapterFile {
    /// The file's `brain.config` object: the family tag plus `rank`, `alpha`
    /// and any field that family adds (`steps`, `rs`, `lr_ratio`, ...).
    pub config: Value,
    pub rank: usize,
    pub alpha: f32,
    /// The adapter's tensors, shapes included.
    pub tensors: Vec<StTensor>,
}

impl PairAdapterFile {
    /// The family tag the file was saved with (`"wan-lora"`, ...).
    pub fn family(&self) -> Option<&str> {
        self.config.get("model").and_then(Value::as_str)
    }

    /// The tensors as the name -> (shape, values) map the host-Pair loaders
    /// take.
    pub fn map(&self) -> std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> {
        self.tensors.iter().map(|t| (t.name.clone(), (t.shape.clone(), t.data.clone()))).collect()
    }
}

/// Write a host-Pair adapter: `config` is the family's own config object
/// (which must carry `rank`, and `alpha` when it is not the rank), and
/// `tensors` are `(name, shape, values)`.
///
/// The write is atomic (tmp + rename) and reports its error: a failed periodic
/// checkpoint (disk full, permissions) must reach the caller, not be
/// discovered only when the run ends and the adapter is missing.
pub fn write(path: &str, config: Value, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<(), String> {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.iter().map(|(n, s, d)| (n.clone(), s.iter().map(|&x| x as u64).collect(), d.clone())).collect();
    checkpoint::st::save_safetensors(path, &t, &config, None).map_err(|e| format!("cannot write adapter {path}: {e}"))
}

/// Read a host-Pair adapter file. `family` is the tag the file must carry
/// (`"wan-lora"`); a file saved by another family is refused by name.
///
/// A missing file is an error naming the path and a corrupt one an error from
/// the reader - never a panic: these are loaded by resident servers on a
/// caller-supplied `--adapter` path.
pub fn read(path: &str, family: &str) -> Result<PairAdapterFile, String> {
    if !Path::new(path).is_file() {
        return Err(format!("adapter file not found: {path}"));
    }
    let meta = checkpoint::st::read_metadata(path).map_err(|e| format!("cannot read adapter {path}: {e}"))?;
    let config: Value = meta
        .get(checkpoint::st::CONFIG_KEY)
        .ok_or_else(|| format!("{path}: not an adapter file (it carries no brain.config)"))
        .and_then(|s| serde_json::from_str(s).map_err(|e| format!("{path}: unreadable config: {e}")))?;
    if config.get("model").and_then(Value::as_str) != Some(family) {
        return Err(format!("{path}: not a {family} adapter (its config says {:?})", config.get("model")));
    }
    let rank = config.get("rank").and_then(Value::as_u64).ok_or_else(|| format!("{path}: adapter has no rank in its header"))? as usize;
    let alpha = config.get("alpha").and_then(Value::as_f64).unwrap_or(rank as f64) as f32;
    let tensors = checkpoint::safetensors::read(path)?;
    Ok(PairAdapterFile { config, rank, alpha, tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn save(tag: &str, family: &str, rank: usize) -> String {
        let path = std::env::temp_dir().join(format!("brain-pair-file-{tag}-{}.safetensors", std::process::id()));
        let path = path.to_str().expect("utf-8 path");
        let tensors = vec![("blocks.0.w.lora_a".to_string(), vec![rank, 4], vec![0.5; rank * 4]), ("blocks.0.w.lora_b".to_string(), vec![3, rank], vec![0.25; 3 * rank])];
        write(path, serde_json::json!({"model": family, "rank": rank, "alpha": 2.0, "steps": 7}), &tensors).expect("write");
        path.to_string()
    }

    #[test]
    fn a_saved_file_reads_back_with_its_shapes_and_config() {
        let path = save("roundtrip", "wan-lora", 2);
        let f = read(&path, "wan-lora").expect("read");
        assert_eq!(f.rank, 2);
        assert_eq!(f.alpha, 2.0);
        assert_eq!(f.family(), Some("wan-lora"));
        assert_eq!(f.config["steps"], 7);
        let map = f.map();
        assert_eq!(map["blocks.0.w.lora_a"].0, vec![2, 4]);
        assert_eq!(map["blocks.0.w.lora_b"].0, vec![3, 2]);
        assert_eq!(map["blocks.0.w.lora_a"].1, vec![0.5; 8]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn another_familys_adapter_is_refused_by_name() {
        let path = save("family", "wan-lora", 2);
        let e = read(&path, "ltxv-lora").err().expect("a wan adapter is not an ltxv adapter");
        assert!(e.contains("not a ltxv-lora adapter"), "{e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let e = read("/nonexistent/adapter.safetensors", "wan-lora").err().expect("a missing file is an error");
        assert!(e.contains("not found"), "{e}");
    }

    #[test]
    fn alpha_defaults_to_the_rank() {
        let path = std::env::temp_dir().join(format!("brain-pair-file-noalpha-{}.safetensors", std::process::id()));
        let path = path.to_str().expect("utf-8 path");
        write(path, serde_json::json!({"model": "wan-lora", "rank": 4}), &[]).expect("write");
        assert_eq!(read(path, "wan-lora").expect("read").alpha, 4.0);
        let _ = std::fs::remove_file(path);
    }
}
