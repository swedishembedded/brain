// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vertical expert sharding of MoE checkpoints (the federated-MoE core).
//!
//! Mirrors the reference `sharded_moe_example`: a worker can load an immutable
//! base, train one expert against a frozen backbone, and return only that
//! expert's shard; the coordinator overlays shards (last-wins) and reassembles.
//!
//! A **checkpoint directory** holds:
//!   shared.safetensors              — every non-expert tensor (embeddings, attn,
//!                                 norms, router, head) + the model config
//!   experts/expert_NNNN.safetensors — expert `NNNN`'s tensors across all layers
//!   manifest.json               — config hash + per-file SHA-256 + expert list
//!
//! Expert tensors are those whose name matches `blocks.<L>.moe.experts.<E>.…`;
//! `<E>` is the (vertical) expert id, spanning every layer.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use checkpoint::Container;
use serde_json::Value;

use crate::sha256;

/// Expert id `<E>` if `name` is `blocks.<L>.moe.experts.<E>.…`, else `None`.
pub fn expert_id(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blocks.")?;
    let (_layer, rest) = rest.split_once('.')?; // "<L>", "moe.experts.<E>.…"
    let rest = rest.strip_prefix("moe.experts.")?;
    let (e, _) = rest.split_once('.')?;
    e.parse().ok()
}

/// Parsed manifest of a checkpoint directory.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub base_config_sha256: String,
    pub experts: Vec<u32>,
    /// rel-path -> sha256 of that file's bytes
    pub files: BTreeMap<String, String>,
}

impl Manifest {
    fn to_json(&self) -> String {
        let files: serde_json::Map<String, Value> = self
            .files
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        serde_json::json!({
            "base_config_sha256": self.base_config_sha256,
            "experts": self.experts,
            "files": files,
        })
        .to_string()
    }

    pub fn from_json(s: &str) -> Result<Manifest, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let base_config_sha256 = v["base_config_sha256"].as_str().ok_or("manifest: base hash")?.to_string();
        let experts = v["experts"]
            .as_array()
            .ok_or("manifest: experts")?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect();
        let mut files = BTreeMap::new();
        if let Some(m) = v["files"].as_object() {
            for (k, val) in m {
                files.insert(k.clone(), val.as_str().unwrap_or("").to_string());
            }
        }
        Ok(Manifest { base_config_sha256, experts, files })
    }
}

/// Canonical SHA-256 of a config object (deterministic: serde_json sorts keys).
pub fn config_hash(config: &Value) -> String {
    sha256::hex(serde_json::to_string(config).unwrap_or_default().as_bytes())
}

fn tensors_named<'a>(c: &'a Container, want: impl Fn(&str) -> bool) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    c.tensors
        .iter()
        .filter(|t| want(&t.name))
        .map(|t| (t.name.clone(), vec![t.data.len() as u64], t.data.clone()))
        .collect()
}

/// Split a full MoE `.safetensors` checkpoint into a shard directory (all experts).
pub fn split(base_weights: &str, out_dir: &Path) -> io::Result<Manifest> {
    split_filtered(base_weights, out_dir, None)
}

/// Like [`split`], but only write the experts in `keep` (plus `shared.safetensors`).
/// `keep = None` writes every expert; `Some(&[E])` produces a single-expert
/// **overlay** directory ready to pass to [`assemble`] — this is what a
/// federated worker returns after training one expert.
pub fn split_filtered(base_weights: &str, out_dir: &Path, keep: Option<&[u32]>) -> io::Result<Manifest> {
    let c = checkpoint::load(base_weights);
    let config = c.header["config"].clone();
    fs::create_dir_all(out_dir.join("experts"))?;

    // shared = every non-expert tensor.
    let shared = tensors_named(&c, |n| expert_id(n).is_none());
    let shared_path = out_dir.join("shared.safetensors");
    checkpoint::save(shared_path.to_str().unwrap(), config.clone(), &shared);

    // per-expert shards (optionally filtered to `keep`).
    let mut expert_ids: Vec<u32> = c
        .tensors
        .iter()
        .filter_map(|t| expert_id(&t.name))
        .filter(|e| keep.is_none_or(|k| k.contains(e)))
        .collect();
    expert_ids.sort_unstable();
    expert_ids.dedup();

    let mut files = BTreeMap::new();
    files.insert("shared.safetensors".to_string(), file_hash(&shared_path)?);
    for &e in &expert_ids {
        let ts = tensors_named(&c, |n| expert_id(n) == Some(e));
        let rel = format!("experts/expert_{e:04}.safetensors");
        let path = out_dir.join(&rel);
        checkpoint::save(path.to_str().unwrap(), config.clone(), &ts);
        files.insert(rel, file_hash(&path)?);
    }

    let manifest = Manifest {
        base_config_sha256: config_hash(&config),
        experts: expert_ids,
        files,
    };
    fs::write(out_dir.join("manifest.json"), manifest.to_json())?;
    Ok(manifest)
}

fn file_hash(p: &Path) -> io::Result<String> {
    Ok(sha256::hex(&fs::read(p)?))
}

/// Verify a shard directory against its manifest: config-hash of `shared.safetensors`
/// matches, and every listed file's bytes hash as recorded. Returns the manifest.
pub fn verify(dir: &Path) -> io::Result<Manifest> {
    let manifest = Manifest::from_json(&fs::read_to_string(dir.join("manifest.json"))?)
        .map_err(io::Error::other)?;
    for (rel, want) in &manifest.files {
        let got = file_hash(&dir.join(rel))?;
        if &got != want {
            return Err(io::Error::other(format!("hash mismatch for {rel}: {got} != {want}")));
        }
    }
    let shared = checkpoint::load(dir.join("shared.safetensors").to_str().unwrap());
    let got = config_hash(&shared.header["config"]);
    if got != manifest.base_config_sha256 {
        return Err(io::Error::other("shared.safetensors config hash != manifest base hash"));
    }
    Ok(manifest)
}

/// Merge a shard directory back into a single full `.safetensors` checkpoint.
pub fn merge_to_full(dir: &Path, out_weights: &str) -> io::Result<()> {
    let shared = checkpoint::load(dir.join("shared.safetensors").to_str().unwrap());
    let config = shared.header["config"].clone();
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = tensors_named(&shared, |_| true);

    let manifest = Manifest::from_json(&fs::read_to_string(dir.join("manifest.json"))?)
        .map_err(io::Error::other)?;
    for e in &manifest.experts {
        let path = dir.join(format!("experts/expert_{e:04}.safetensors"));
        let ec = checkpoint::load(path.to_str().unwrap());
        tensors.extend(tensors_named(&ec, |_| true));
    }
    checkpoint::save(out_weights, config, &tensors);
    Ok(())
}

/// Assemble: start from `base_dir`, overlay each dir in order (last-wins per
/// expert id, and shared if the overlay carries it), and write the merged full
/// checkpoint to `out_weights`. All dirs must share the base config hash.
pub fn assemble(base_dir: &Path, overlays: &[&Path], out_weights: &str) -> io::Result<()> {
    let base_manifest = verify(base_dir)?;

    // Start from base shared + base experts, indexed by name for last-wins.
    let shared = checkpoint::load(base_dir.join("shared.safetensors").to_str().unwrap());
    let config = shared.header["config"].clone();
    let mut by_name: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    let push = |name: &str, data: &[f32], by_name: &mut BTreeMap<String, Vec<f32>>, order: &mut Vec<String>| {
        if !by_name.contains_key(name) {
            order.push(name.to_string());
        }
        by_name.insert(name.to_string(), data.to_vec());
    };
    for t in &shared.tensors {
        push(&t.name, &t.data, &mut by_name, &mut order);
    }
    for &e in &base_manifest.experts {
        let ec = checkpoint::load(base_dir.join(format!("experts/expert_{e:04}.safetensors")).to_str().unwrap());
        for t in &ec.tensors {
            push(&t.name, &t.data, &mut by_name, &mut order);
        }
    }

    // Overlays (last-wins). Each must agree on the base config.
    for ov in overlays {
        let m = verify(ov)?;
        if m.base_config_sha256 != base_manifest.base_config_sha256 {
            return Err(io::Error::other(format!(
                "overlay {} has a different base config hash",
                ov.display()
            )));
        }
        // overlay shared (if it differs from base — same names just overwrite)
        let osh = checkpoint::load(ov.join("shared.safetensors").to_str().unwrap());
        for t in &osh.tensors {
            push(&t.name, &t.data, &mut by_name, &mut order);
        }
        for e in &m.experts {
            let ec = checkpoint::load(ov.join(format!("experts/expert_{e:04}.safetensors")).to_str().unwrap());
            for t in &ec.tensors {
                push(&t.name, &t.data, &mut by_name, &mut order);
            }
        }
    }

    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = order
        .iter()
        .map(|n| (n.clone(), vec![by_name[n].len() as u64], by_name[n].clone()))
        .collect();
    checkpoint::save(out_weights, config, &tensors);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain_fed_{tag}_{}", std::process::id()))
    }

    // A fake 1-layer, 3-expert MoE checkpoint with recognizable data.
    fn make_base(path: &str) {
        let config = serde_json::json!({
            "model": "moe", "vocab_size": 8, "n_layers": 1, "d_model": 2,
            "n_heads": 1, "n_experts": 3, "top_k": 2, "d_ff": 4
        });
        let mut tensors = vec![
            ("token_emb.weight".to_string(), vec![16u64], vec![1.0; 16]),
            ("blocks.0.moe.router.weight".to_string(), vec![6], vec![2.0; 6]),
        ];
        for e in 0..3u32 {
            let v = 10.0 + e as f32;
            tensors.push((format!("blocks.0.moe.experts.{e}.w_gate.weight"), vec![8], vec![v; 8]));
            tensors.push((format!("blocks.0.moe.experts.{e}.w_up.weight"), vec![8], vec![v; 8]));
            tensors.push((format!("blocks.0.moe.experts.{e}.w_down.weight"), vec![8], vec![v; 8]));
        }
        checkpoint::save(path, config, &tensors);
    }

    #[test]
    fn expert_id_parsing() {
        assert_eq!(expert_id("blocks.2.moe.experts.7.w_gate.weight"), Some(7));
        assert_eq!(expert_id("blocks.0.moe.router.weight"), None);
        assert_eq!(expert_id("token_emb.weight"), None);
    }

    #[test]
    fn split_assemble_roundtrip_is_identity() {
        let dir = tmp("rt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.safetensors");
        make_base(base.to_str().unwrap());

        let sdir = dir.join("shards");
        let m = split(base.to_str().unwrap(), &sdir).unwrap();
        assert_eq!(m.experts, vec![0, 1, 2]);
        verify(&sdir).unwrap(); // manifest hashes check out

        let out = dir.join("reassembled.safetensors");
        assemble(&sdir, &[], out.to_str().unwrap()).unwrap();

        // Reassembled == original (tensor-for-tensor).
        let a = checkpoint::load(base.to_str().unwrap());
        let b = checkpoint::load(out.to_str().unwrap());
        for t in &a.tensors {
            let bt = b.find(&t.name, "").expect("tensor present after roundtrip");
            assert_eq!(&t.data, bt, "data mismatch for {}", t.name);
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn overlay_replaces_one_expert_last_wins() {
        let dir = tmp("ov");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.safetensors");
        make_base(base.to_str().unwrap());
        let sdir = dir.join("shards");
        split(base.to_str().unwrap(), &sdir).unwrap();

        // Build an overlay dir carrying only a retrained expert 1 (data = 99).
        let odir = dir.join("overlay");
        fs::create_dir_all(odir.join("experts")).unwrap();
        let config = checkpoint::load(sdir.join("shared.safetensors").to_str().unwrap()).header["config"].clone();
        // overlay must still carry a shared.safetensors (same config) for verify().
        checkpoint::save(odir.join("shared.safetensors").to_str().unwrap(), config.clone(), &[]);
        let e1 = vec![
            ("blocks.0.moe.experts.1.w_gate.weight".to_string(), vec![8u64], vec![99.0; 8]),
            ("blocks.0.moe.experts.1.w_up.weight".to_string(), vec![8], vec![99.0; 8]),
            ("blocks.0.moe.experts.1.w_down.weight".to_string(), vec![8], vec![99.0; 8]),
        ];
        checkpoint::save(odir.join("experts/expert_0001.safetensors").to_str().unwrap(), config.clone(), &e1);
        let mut files = BTreeMap::new();
        files.insert("shared.safetensors".to_string(), file_hash(&odir.join("shared.safetensors")).unwrap());
        files.insert("experts/expert_0001.safetensors".to_string(), file_hash(&odir.join("experts/expert_0001.safetensors")).unwrap());
        let man = Manifest { base_config_sha256: config_hash(&config), experts: vec![1], files };
        fs::write(odir.join("manifest.json"), man.to_json()).unwrap();

        let out = dir.join("assembled.safetensors");
        assemble(&sdir, &[&odir], out.to_str().unwrap()).unwrap();
        let b = checkpoint::load(out.to_str().unwrap());
        // expert 1 replaced with 99s; expert 0 and 2 unchanged (10, 12).
        assert_eq!(b.find("blocks.0.moe.experts.1.w_gate.weight", "").unwrap()[0], 99.0);
        assert_eq!(b.find("blocks.0.moe.experts.0.w_gate.weight", "").unwrap()[0], 10.0);
        assert_eq!(b.find("blocks.0.moe.experts.2.w_gate.weight", "").unwrap()[0], 12.0);
        fs::remove_dir_all(&dir).ok();
    }
}
