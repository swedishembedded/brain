// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2's rules for the model-store resolver: which `HfDir` on disk is
//! a released `microsoft/Florence-2-*` checkpoint.
//!
//! One role, `weights` - the directory `florence2::import::build_param_source`
//! reads (`config.json` + `model.safetensors` + `tokenizer.json`).
//! Identified from the checkpoint's own `config.json` `model_type` field
//! (`"florence2"`, confirmed against the real downloaded checkpoint - never
//! guessed), never from a path or filename.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role name this architecture resolves - the checkpoint directory.
pub const ROLES: &[&str] = &["weights"];

const REQUIRED_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];

/// Whether `dir` is a released Florence-2 checkpoint directory, from its own
/// `config.json`.
pub fn is_florence2_dir(dir: &Path) -> bool {
    if !REQUIRED_FILES.iter().all(|f| dir.join(f).is_file()) {
        return false;
    }
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else { return false };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    v.get("model_type").and_then(serde_json::Value::as_str) == Some("florence2")
}

/// Florence-2's [`ArchSpec`].
pub struct Florence2Spec;

impl ArchSpec for Florence2Spec {
    fn arch(&self) -> &'static str {
        "florence2"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, rec)| rec.usable() && rec.kind == ArtifactKind::HfDir && is_florence2_dir(&rec.path))
            // `Declared`: `model_type` is the checkpoint's own self-reported
            // architecture tag, not something computed from tensor shapes.
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Declared))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("florence2 assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "microsoft/Florence-2-base".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let weights = assembly.roles.get("weights").ok_or("florence2 validate: assembly has no weights role")?;
        if !is_florence2_dir(weights) {
            return Err(format!("florence2 validate: {} is not a released Florence-2 checkpoint directory", weights.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-florence2-spec-{tag}-{}-{n}", std::process::id()))
    }

    fn write_checkpoint(dir: &Path, model_type: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"model_type": model_type})).unwrap()).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"stub").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    }

    fn hfdir_record(path: &Path) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind: ArtifactKind::HfDir, completeness: Completeness::Complete }
    }

    #[test]
    fn a_florence2_checkpoint_resolves_by_its_own_model_type() {
        let dir = tmp("resolve");
        let repo = dir.join("microsoft").join("Florence-2-base");
        write_checkpoint(&repo, "florence2");

        let records = vec![hfdir_record(&repo)];
        let specs: Vec<&dyn ArchSpec> = vec![&Florence2Spec];
        match resolve("florence2", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], repo),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A same-shaped HF checkpoint directory for a DIFFERENT architecture
    /// must not classify - content, not the mere presence of the three
    /// expected filenames, is what decides it.
    #[test]
    fn a_differently_typed_checkpoint_does_not_classify() {
        let dir = tmp("wrong-type");
        let repo = dir.join("some-other-model");
        write_checkpoint(&repo, "bart");
        assert!(!is_florence2_dir(&repo));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_checkpoint_missing_the_tokenizer_does_not_classify() {
        let dir = tmp("incomplete");
        let repo = dir.join("microsoft").join("Florence-2-base");
        write_checkpoint(&repo, "florence2");
        std::fs::remove_file(repo.join("tokenizer.json")).unwrap();
        assert!(!is_florence2_dir(&repo));
        std::fs::remove_dir_all(&dir).ok();
    }
}
