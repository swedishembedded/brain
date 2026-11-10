// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage's rules for the model-store resolver: which directory on disk is a
//! released `Cactus-Compute/mirage` checkpoint.
//!
//! One role, `weights`: the directory that holds `config.json`,
//! `mirage.safetensors` and `mirage.cact` as published (the tokenizer and the
//! reader charset live inside the `.cact`, so it is part of the model, not an
//! optional deployment extra). A directory is recognised by what its
//! `config.json` declares (`model_type: "mirage"`), never by where it sits.
//! Both shapes the store produces classify: the bare files a hand placement
//! leaves (the safetensors file is the record) and the directory a `brain pull`
//! describes with a compound manifest.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery for
//! clients running mixed fleets of hand-placed and fetched weights. If your
//! team needs the same discipline for its own model store, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::import::{CONFIG_FILE, WEIGHTS_FILE};

/// The deployment container published beside the weights.
pub const DEPLOYMENT_FILE: &str = "mirage.cact";

/// The role this architecture resolves.
pub const ROLES: &[&str] = &["weights"];

/// The files a servable checkpoint directory holds.
pub const RELEASE_FILES: [&str; 3] = [CONFIG_FILE, WEIGHTS_FILE, DEPLOYMENT_FILE];

/// Whether `dir` is a released Mirage checkpoint directory, from its own `config.json`.
pub fn is_mirage_dir(dir: &Path) -> bool {
    if !RELEASE_FILES.iter().all(|f| dir.join(f).is_file()) {
        return false;
    }
    let Ok(bytes) = std::fs::read(dir.join(CONFIG_FILE)) else { return false };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    config.get("model_type").and_then(serde_json::Value::as_str) == Some("mirage")
}

/// Mirage's [`ArchSpec`].
pub struct MirageSpec;

impl ArchSpec for MirageSpec {
    fn arch(&self) -> &'static str {
        "mirage"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, rec)| rec.usable())
            .filter(|(_, rec)| match rec.kind {
                // A pulled repository: the manifest names the directory itself.
                ArtifactKind::Compound => rec.path.is_dir() && is_mirage_dir(&rec.path),
                // Hand-placed files: the weights file stands for its directory.
                ArtifactKind::Safetensors => rec.path.file_name().is_some_and(|n| n == WEIGHTS_FILE) && rec.path.parent().is_some_and(is_mirage_dir),
                _ => false,
            })
            // `model_type` is the checkpoint's own architecture tag.
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Declared))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("mirage assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: crate::caps::MODEL.to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let weights = assembly.roles.get("weights").ok_or("mirage validate: assembly has no weights role")?;
        let dir = if weights.is_dir() { weights.as_path() } else { weights.parent().ok_or("mirage validate: the weights role has no directory")? };
        if !is_mirage_dir(dir) {
            return Err(format!("mirage validate: {} is not a released Mirage checkpoint directory ({RELEASE_FILES:?}, model_type \"mirage\")", dir.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::{scan, Completeness};
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-mirage-spec-{tag}-{}-{n}", std::process::id()))
    }

    fn write_checkpoint(dir: &Path, model_type: &str, with_container: bool) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(CONFIG_FILE), serde_json::to_vec(&serde_json::json!({"model_type": model_type})).unwrap()).unwrap();
        checkpoint::st::save_safetensors(dir.join(WEIGHTS_FILE).to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        if with_container {
            std::fs::write(dir.join(DEPLOYMENT_FILE), b"stub").unwrap();
        }
    }

    fn record(path: &Path, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    #[test]
    fn a_hand_placed_release_resolves_by_its_declared_model_type() {
        let root = tmp("placed");
        let repo = root.join("Cactus-Compute").join("mirage");
        write_checkpoint(&repo, "mirage", true);
        let records = vec![record(&repo.join(WEIGHTS_FILE), ArtifactKind::Safetensors)];
        let specs: Vec<&dyn ArchSpec> = vec![&MirageSpec];
        match resolve("mirage", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], repo.join(WEIGHTS_FILE)),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_pulled_release_resolves_through_its_manifest_directory() {
        let root = tmp("pulled");
        let repo = root.join("Cactus-Compute").join("mirage");
        write_checkpoint(&repo, "mirage", true);
        let manifest = brain_modelstore::CompoundManifest { id: crate::caps::MODEL.to_string(), family: "mirage".to_string(), roles: BTreeMap::from([("weights".to_string(), ".".to_string())]) };
        std::fs::write(repo.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        // A second vendor keeps the scan root at `root` rather than at the single repository.
        std::fs::create_dir_all(root.join("other")).unwrap();
        std::fs::write(root.join("other").join("x.gguf"), b"not a gguf").unwrap();
        let found = scan(&root);
        let spec = MirageSpec;
        let picked = spec.classify(&found, &root);
        assert_eq!(picked.len(), 1, "{found:?}");
        assert_eq!(found[picked[0].0].path, repo);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn another_model_type_or_a_missing_container_does_not_classify() {
        let root = tmp("wrong");
        write_checkpoint(&root.join("a"), "bart", true);
        write_checkpoint(&root.join("b"), "mirage", false);
        assert!(!is_mirage_dir(&root.join("a")));
        assert!(!is_mirage_dir(&root.join("b")));
        let spec = MirageSpec;
        let records = vec![record(&root.join("a").join(WEIGHTS_FILE), ArtifactKind::Safetensors), record(&root.join("b").join(WEIGHTS_FILE), ArtifactKind::Safetensors)];
        assert!(spec.classify(&records, &root).is_empty());
        std::fs::remove_dir_all(&root).ok();
    }
}
