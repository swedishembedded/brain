// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR's [`ArchSpec`]: which on-disk HF directory satisfies the single
//! `weights` role. Real content only - an HF `config.json`'s own declared
//! `architectures[0]`, never a directory's name.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct Qwen3AsrSpec;

const ROLES: &[&str] = &["weights"];

impl ArchSpec for Qwen3AsrSpec {
    fn arch(&self) -> &'static str {
        "qwen3asr"
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
            if declared_architecture(&config).as_deref() == Some("Qwen3ASRForConditionalGeneration") {
                out.push((idx, "weights".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("qwen3asr assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/qwen3asr".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("qwen3asr validate: assembly has no weights role")?;
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
        std::env::temp_dir().join(format!("brain-qwen3asr-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_qwen3asr_hfdir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3ASRForConditionalGeneration"]})).unwrap()).unwrap();
    }

    /// A directory named exactly like the real release, but whose header says
    /// something else - proving classification reads content, not the name.
    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        let suggestive = dir.join("Qwen3-ASR-1.7B");
        std::fs::create_dir_all(&suggestive).unwrap();
        std::fs::write(suggestive.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["SomeOtherModelForCausalLM"]})).unwrap()).unwrap();

        let records = vec![complete(suggestive, ArtifactKind::HfDir)];
        let out = Qwen3AsrSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn classify_recognizes_the_real_declared_architecture() {
        let dir = tmp("real-declared");
        let repo = dir.join("Qwen").join("Qwen3-ASR-1.7B");
        write_qwen3asr_hfdir(&repo);

        let records = vec![complete(repo.clone(), ArtifactKind::HfDir)];
        let out = Qwen3AsrSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_with_exactly_one_candidate() {
        let dir = tmp("resolve-end-to-end");
        let repo = dir.join("Qwen").join("Qwen3-ASR-1.7B");
        write_qwen3asr_hfdir(&repo);

        let records = vec![complete(repo.clone(), ArtifactKind::HfDir)];
        let spec = Qwen3AsrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("qwen3asr", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], repo);
                assert_eq!(a.arch, "qwen3asr");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let repo_a = dir.join("Qwen").join("Qwen3-ASR-1.7B");
        write_qwen3asr_hfdir(&repo_a);
        let repo_b = dir.join("some-mirror").join("Qwen3-ASR-1.7B-copy");
        write_qwen3asr_hfdir(&repo_b);

        let records = vec![complete(repo_a, ArtifactKind::HfDir), complete(repo_b, ArtifactKind::HfDir)];
        let spec = Qwen3AsrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("qwen3asr", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }

    /// `--weights <path>` (the resolver's own override contract) collapses an
    /// ambiguity deterministically.
    #[test]
    fn an_explicit_override_collapses_an_ambiguity() {
        let dir = tmp("override-collapses");
        let repo_a = dir.join("Qwen").join("Qwen3-ASR-1.7B");
        write_qwen3asr_hfdir(&repo_a);
        let repo_b = dir.join("some-mirror").join("Qwen3-ASR-1.7B-copy");
        write_qwen3asr_hfdir(&repo_b);

        let records = vec![complete(repo_a.clone(), ArtifactKind::HfDir), complete(repo_b, ArtifactKind::HfDir)];
        let spec = Qwen3AsrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("weights".to_string(), repo_a.to_string_lossy().into_owned());
        let out = resolve("qwen3asr", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], repo_a),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
