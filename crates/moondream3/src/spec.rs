// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3's [`ArchSpec`]: which on-disk checkpoint directory satisfies
//! its one `dir` role - a directory holding `config.json` (declaring
//! `architectures[0] == "Moondream3ForConditionalGeneration"`), its
//! safetensors shards, and a sibling `tokenizer.json`. Header/config-only,
//! the same contract `flux2::spec::Flux2Spec`/`fastvlm::spec::FastvlmSpec`
//! already follow.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery like
//! this for clients running mixed fleets of hand-placed and fetched weights.
//! If your team needs the same discipline for its own model store, you can
//! procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct Moondream3Spec;

const ROLES: &[&str] = &["dir"];

/// The `architectures[0]` value Moondream 3's own released `config.json`
/// declares (`crates/arch/src/lib.rs`'s `hf:` list for this row).
const HF_ARCHITECTURE: &str = "Moondream3ForConditionalGeneration";

impl ArchSpec for Moondream3Spec {
    fn arch(&self) -> &'static str {
        "moondream3"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::HfDir {
                continue;
            }
            let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { continue };
            let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
            if declared_architecture(&config).as_deref() == Some(HF_ARCHITECTURE) {
                out.push((idx, "dir".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("dir").ok_or("moondream3 assemble: no dir chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/moondream3".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dir = assembly.roles.get("dir").ok_or("moondream3 validate: assembly has no dir role")?;
        if !dir.join("tokenizer.json").is_file() {
            return Err(format!("moondream3 validate: {} has no sibling tokenizer.json", dir.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-moondream3-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn write_checkpoint_dir(dir: &Path, hf_architecture: &str, with_tokenizer: bool) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": [hf_architecture]})).unwrap()).unwrap();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        if with_tokenizer {
            std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        }
    }

    #[test]
    fn classify_uses_config_content_not_directory_name() {
        let dir = tmp("content-not-name");
        let suggestive = dir.join("moondream").join("moondream3-preview");
        write_checkpoint_dir(&suggestive, "SomeUnrelatedForConditionalGeneration", true);
        let real = dir.join("moondream").join("moondream3-real");
        write_checkpoint_dir(&real, HF_ARCHITECTURE, true);

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Moondream3Spec.classify(&records, &dir);
        let dirs: Vec<_> = out.iter().filter(|(_, role, _)| role == "dir").collect();
        assert_eq!(dirs.len(), 1, "{out:?}");
        let (idx, ..) = dirs[0];
        assert_eq!(records[*idx].path, real);
    }

    #[test]
    fn resolves_cleanly_with_zero_env_vars_from_a_real_checkpoint_directory() {
        let dir = tmp("resolves-clean");
        let ckpt = dir.join("moondream").join("moondream3-preview");
        write_checkpoint_dir(&ckpt, HF_ARCHITECTURE, true);
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Moondream3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("moondream3", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(assembly) => {
                assert_eq!(assembly.roles["dir"], ckpt);
                assert_eq!(assembly.arch, "moondream3");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn a_checkpoint_with_no_tokenizer_fails_validate_not_a_silent_resolve() {
        let dir = tmp("no-tokenizer");
        let ckpt = dir.join("moondream").join("moondream3-preview");
        write_checkpoint_dir(&ckpt, HF_ARCHITECTURE, false);
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Moondream3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("moondream3", &records, &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => assert!(m.roles.iter().any(|r| r.doc.contains("tokenizer.json")), "{m:?}"),
            other => panic!("expected Missing (validate should reject a checkpoint with no tokenizer), got {other:?}"),
        }
    }

    #[test]
    fn no_candidates_on_disk_is_missing_not_a_panic() {
        let dir = tmp("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Moondream3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("moondream3", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)));
    }
}
