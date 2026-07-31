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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use checkpoint::weightio::{StWriter, WeightReader};
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

/// Split a full MoE `.safetensors` checkpoint into a shard directory (all experts).
pub fn split(base_weights: &str, out_dir: &Path) -> io::Result<Manifest> {
    split_filtered(base_weights, out_dir, None)
}

/// Like [`split`], but only write the experts in `keep` (plus `shared.safetensors`).
/// `keep = None` writes every expert; `Some(&[E])` produces a single-expert
/// **overlay** directory ready to pass to [`assemble`] — this is what a
/// federated worker returns after training one expert.
pub fn split_filtered(base_weights: &str, out_dir: &Path, keep: Option<&[u32]>) -> io::Result<Manifest> {
    let reader = WeightReader::open(base_weights)?;
    let config = reader.config();
    fs::create_dir_all(out_dir.join("experts"))?;

    // Header-only pass: group planned (name, shape) by destination without touching tensor data.
    let mut shared_plan: Vec<(String, Vec<u64>)> = Vec::new();
    let mut expert_plan: BTreeMap<u32, Vec<(String, Vec<u64>)>> = BTreeMap::new();
    for name in reader.names() {
        let shape = reader.shape(name).unwrap().to_vec();
        match expert_id(name) {
            None => shared_plan.push((name.to_string(), shape)),
            Some(e) if keep.is_none_or(|k| k.contains(&e)) => {
                expert_plan.entry(e).or_default().push((name.to_string(), shape));
            }
            Some(_) => {} // filtered out by `keep`
        }
    }

    let shared_path = out_dir.join("shared.safetensors");
    let mut shared_w = StWriter::create(shared_path.to_str().unwrap(), &shared_plan, &config, None)?;
    let mut expert_w: BTreeMap<u32, StWriter> = expert_plan
        .iter()
        .map(|(&e, plan)| {
            let path = out_dir.join(format!("experts/expert_{e:04}.safetensors"));
            Ok((e, StWriter::create(path.to_str().unwrap(), plan, &config, None)?))
        })
        .collect::<io::Result<_>>()?;

    // Stream every tensor once, routing it to its destination writer.
    let mut err: Option<io::Error> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        let res = match expert_id(name) {
            None => shared_w.write(name, &data),
            Some(e) => match expert_w.get_mut(&e) {
                Some(w) => w.write(name, &data),
                None => Ok(()), // filtered out by `keep`
            },
        };
        if let Err(e) = res {
            err = Some(e);
        }
    });
    if let Some(e) = err {
        return Err(e);
    }

    shared_w.finish()?;
    let mut expert_ids: Vec<u32> = expert_w.keys().copied().collect();
    for (_, w) in expert_w {
        w.finish()?;
    }
    expert_ids.sort_unstable();

    let mut files = BTreeMap::new();
    files.insert("shared.safetensors".to_string(), file_hash(&shared_path)?);
    for &e in &expert_ids {
        let rel = format!("experts/expert_{e:04}.safetensors");
        let path = out_dir.join(&rel);
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
    let manifest = Manifest::from_json(&fs::read_to_string(dir.join("manifest.json"))?)
        .map_err(io::Error::other)?;

    let mut paths = vec![dir.join("shared.safetensors")];
    paths.extend(manifest.experts.iter().map(|e| dir.join(format!("experts/expert_{e:04}.safetensors"))));

    // Header-only pass over every source file to build the combined output plan.
    let mut plan: Vec<(String, Vec<u64>)> = Vec::new();
    for p in &paths {
        let r = WeightReader::open(p.to_str().unwrap())?;
        plan.extend(r.names().map(|n| (n.to_string(), r.shape(n).unwrap().to_vec())));
    }
    let config = WeightReader::open(paths[0].to_str().unwrap())?.config();

    let mut writer = StWriter::create(out_weights, &plan, &config, None)?;
    let mut err: Option<io::Error> = None;
    for p in &paths {
        let r = WeightReader::open(p.to_str().unwrap())?;
        r.for_each(|name, _shape, data| {
            if err.is_none() {
                if let Err(e) = writer.write(name, &data) {
                    err = Some(e);
                }
            }
        });
    }
    if let Some(e) = err {
        return Err(e);
    }
    writer.finish()
}

/// Assemble: start from `base_dir`, overlay each dir in order (last-wins per
/// expert id, and shared if the overlay carries it), and write the merged full
/// checkpoint to `out_weights`. All dirs must share the base config hash.
pub fn assemble(base_dir: &Path, overlays: &[&Path], out_weights: &str) -> io::Result<()> {
    let base_manifest = verify(base_dir)?;

    // Same file-visit order as before: base shared, base experts, then per overlay shared + experts.
    let mut sources: Vec<PathBuf> = vec![base_dir.join("shared.safetensors")];
    sources.extend(base_manifest.experts.iter().map(|e| base_dir.join(format!("experts/expert_{e:04}.safetensors"))));
    for ov in overlays {
        let m = verify(ov)?;
        if m.base_config_sha256 != base_manifest.base_config_sha256 {
            return Err(io::Error::other(format!(
                "overlay {} has a different base config hash",
                ov.display()
            )));
        }
        sources.push(ov.join("shared.safetensors"));
        sources.extend(m.experts.iter().map(|e| ov.join(format!("experts/expert_{e:04}.safetensors"))));
    }

    // Pass 1 (header-only, no tensor data read): later sources overwrite the winner map, so
    // this resolves last-wins per name before any bytes are decoded — needed because a name
    // can recur across files and StWriter forbids writing the same planned name twice.
    let mut winner: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut winner_shape: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for path in &sources {
        let r = WeightReader::open(path.to_str().unwrap())?;
        for name in r.names() {
            if !winner.contains_key(name) {
                order.push(name.to_string());
            }
            winner.insert(name.to_string(), path.clone());
            winner_shape.insert(name.to_string(), r.shape(name).unwrap().to_vec());
        }
    }

    let plan: Vec<(String, Vec<u64>)> = order.iter().map(|n| (n.clone(), winner_shape[n].clone())).collect();
    let config = WeightReader::open(base_dir.join("shared.safetensors").to_str().unwrap())?.config();
    let mut writer = StWriter::create(out_weights, &plan, &config, None)?;

    // Pass 2: each source opened once; only its still-winning names are decoded (named
    // `tensor()` lookups, not `for_each`, which would decode a losing tensor before any
    // filter could apply) — a name overridden by a later file is never read at all.
    let mut by_source: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for (name, path) in &winner {
        by_source.entry(path.clone()).or_default().push(name.clone());
    }
    for (path, names) in &by_source {
        let r = WeightReader::open(path.to_str().unwrap())?;
        for name in names {
            let data = r.tensor(name).expect("winner name present in its own source");
            writer.write(name, &data)?;
        }
    }
    writer.finish()
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

    // Build a single-expert-1 overlay dir (mirrors the inline setup in overlay_replaces_one_expert_last_wins).
    fn make_expert1_overlay(odir: &Path, config: &Value, value: f32) {
        fs::create_dir_all(odir.join("experts")).unwrap();
        checkpoint::save(odir.join("shared.safetensors").to_str().unwrap(), config.clone(), &[]);
        let e1 = vec![
            ("blocks.0.moe.experts.1.w_gate.weight".to_string(), vec![8u64], vec![value; 8]),
            ("blocks.0.moe.experts.1.w_up.weight".to_string(), vec![8], vec![value; 8]),
            ("blocks.0.moe.experts.1.w_down.weight".to_string(), vec![8], vec![value; 8]),
        ];
        checkpoint::save(odir.join("experts/expert_0001.safetensors").to_str().unwrap(), config.clone(), &e1);
        let mut files = BTreeMap::new();
        files.insert("shared.safetensors".to_string(), file_hash(&odir.join("shared.safetensors")).unwrap());
        files.insert("experts/expert_0001.safetensors".to_string(), file_hash(&odir.join("experts/expert_0001.safetensors")).unwrap());
        let man = Manifest { base_config_sha256: config_hash(config), experts: vec![1], files };
        fs::write(odir.join("manifest.json"), man.to_json()).unwrap();
    }

    /// Same tensor name (expert 1) is touched by base + two overlays; the LAST
    /// overlay applied must win, proving winner-resolution tracks last-seen
    /// (not first-seen, and not e.g. lexical/manifest order).
    #[test]
    fn assemble_last_wins_across_three_files() {
        let dir = tmp("three");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.safetensors");
        make_base(base.to_str().unwrap()); // base expert 1 = 11.0 (X)
        let sdir = dir.join("shards");
        split(base.to_str().unwrap(), &sdir).unwrap();
        let config = checkpoint::load(sdir.join("shared.safetensors").to_str().unwrap()).header["config"].clone();

        let odir_a = dir.join("overlay_a");
        make_expert1_overlay(&odir_a, &config, 42.0); // overlay A expert 1 = Y
        let odir_b = dir.join("overlay_b");
        make_expert1_overlay(&odir_b, &config, 77.0); // overlay B (applied after A) expert 1 = Z

        let out = dir.join("assembled3.safetensors");
        assemble(&sdir, &[&odir_a, &odir_b], out.to_str().unwrap()).unwrap();
        let b = checkpoint::load(out.to_str().unwrap());
        // Z (last overlay) wins over Y (earlier overlay) and X (base).
        assert_eq!(b.find("blocks.0.moe.experts.1.w_gate.weight", "").unwrap()[0], 77.0);
        assert_eq!(b.find("blocks.0.moe.experts.1.w_up.weight", "").unwrap()[0], 77.0);
        assert_eq!(b.find("blocks.0.moe.experts.1.w_down.weight", "").unwrap()[0], 77.0);
        // Untouched experts still come from base.
        assert_eq!(b.find("blocks.0.moe.experts.0.w_gate.weight", "").unwrap()[0], 10.0);
        assert_eq!(b.find("blocks.0.moe.experts.2.w_gate.weight", "").unwrap()[0], 12.0);
        fs::remove_dir_all(&dir).ok();
    }
}
