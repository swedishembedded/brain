// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TimesFM-3's [`ArchSpec`]: which on-disk HF directory satisfies the single
//! `weights` role.
//!
//! The role is a released directory or a brain container (what `import` and
//! `finetune` write), told apart from other safetensors by the config in its
//! header.
//!
//! Unlike `qwen3asr`/`nemotronasr`, the upstream `config.json` carries no
//! `architectures` field at all (it is not a `transformers`-family repo -
//! see [`crate::config`]'s module doc), so there is no declared-name field to
//! read. Classification instead attempts a real, structural parse of the
//! checkpoint's own nested hyperparameter schema
//! ([`crate::config::Timesfm3Config::from_hf_config_json`]) - the same
//! "derive it from real content, never a filename" discipline every other
//! `ArchSpec` in this migration follows, just keyed on a config shape instead
//! of a declared class name or a tensor shape.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::Timesfm3Config;

pub struct Timesfm3Spec;

const ROLES: &[&str] = &["weights"];

impl ArchSpec for Timesfm3Spec {
    fn arch(&self) -> &'static str {
        "timesfm3"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            // No format-level self-declaration exists for the released schema
            // (no `architectures` field upstream) - a real, successful parse of
            // the checkpoint's own nested hyperparameters is the closest
            // equivalent, so `Derived`, not `Declared`. A brain container (what
            // `import` and `finetune` write) names its config in its header.
            let derived = match rec.kind {
                ArtifactKind::HfDir => std::fs::read(rec.path.join("config.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                    .is_some_and(|config| Timesfm3Config::from_hf_config_json(&config).is_ok()),
                ArtifactKind::Safetensors => checkpoint::mmap::MmapSafetensors::open(&rec.path).is_ok_and(|m| Timesfm3Config::from_json(&m.config()).is_ok()),
                _ => false,
            };
            if derived {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("timesfm3 assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/timesfm3".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("timesfm3 validate: assembly has no weights role")?;
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
        std::env::temp_dir().join(format!("brain-timesfm3-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// The exact upstream nested schema `from_hf_config_json` reads (see that
    /// function's own doc) - a minimal but real instance, not a guess at the
    /// shape.
    fn write_timesfm3_hfdir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let config = serde_json::json!({
            "input_patch_len": 32,
            "output_patch_len": 64,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "transformer_config": {
                "num_layers": 20,
                "transformer": { "model_dims": 1280, "hidden_dims": 1280, "num_heads": 16, "max_variates": 32 }
            }
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    }

    /// A directory named exactly like the real release, but whose
    /// `config.json` does not parse as TimesFM-3's own schema - proving
    /// classification reads real content, not the name.
    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        let suggestive = dir.join("timesfm-3.0-pytorch");
        std::fs::create_dir_all(&suggestive).unwrap();
        std::fs::write(suggestive.join("config.json"), serde_json::to_vec(&serde_json::json!({"some_other_schema": true})).unwrap()).unwrap();

        let records = vec![complete(suggestive, ArtifactKind::HfDir)];
        let out = Timesfm3Spec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn classify_recognizes_a_real_config_schema() {
        let dir = tmp("real-schema");
        let repo = dir.join("google").join("timesfm-3.0-pytorch");
        write_timesfm3_hfdir(&repo);

        let records = vec![complete(repo.clone(), ArtifactKind::HfDir)];
        let out = Timesfm3Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A brain container (what `import` and `finetune` write) is the `weights` role by the
    /// config in its header, and a safetensors file with another config is not.
    #[test]
    fn a_brain_container_is_classified_by_its_header_config() {
        let dir = tmp("container");
        std::fs::create_dir_all(&dir).unwrap();
        let ours = dir.join("ft.safetensors");
        let theirs = dir.join("other.safetensors");
        let tensor = [("w".to_string(), vec![1u64], vec![0.0f32])];
        checkpoint::save(ours.to_str().unwrap(), Timesfm3Config::tiny().to_json(), &tensor);
        checkpoint::save(theirs.to_str().unwrap(), serde_json::json!({"hidden_size": 8}), &tensor);

        let records = vec![complete(ours, ArtifactKind::Safetensors), complete(theirs, ArtifactKind::Safetensors)];
        let out = Timesfm3Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_with_exactly_one_candidate() {
        let dir = tmp("resolve-end-to-end");
        let repo = dir.join("google").join("timesfm-3.0-pytorch");
        write_timesfm3_hfdir(&repo);

        let records = vec![complete(repo.clone(), ArtifactKind::HfDir)];
        let spec = Timesfm3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("timesfm3", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], repo);
                assert_eq!(a.arch, "timesfm3");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let repo_a = dir.join("google").join("timesfm-3.0-pytorch");
        write_timesfm3_hfdir(&repo_a);
        let repo_b = dir.join("some-mirror").join("timesfm-3.0-pytorch-copy");
        write_timesfm3_hfdir(&repo_b);

        let records = vec![complete(repo_a, ArtifactKind::HfDir), complete(repo_b, ArtifactKind::HfDir)];
        let spec = Timesfm3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("timesfm3", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }

    #[test]
    fn an_explicit_override_collapses_an_ambiguity() {
        let dir = tmp("override-collapses");
        let repo_a = dir.join("google").join("timesfm-3.0-pytorch");
        write_timesfm3_hfdir(&repo_a);
        let repo_b = dir.join("some-mirror").join("timesfm-3.0-pytorch-copy");
        write_timesfm3_hfdir(&repo_b);

        let records = vec![complete(repo_a.clone(), ArtifactKind::HfDir), complete(repo_b, ArtifactKind::HfDir)];
        let spec = Timesfm3Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("weights".to_string(), repo_a.to_string_lossy().into_owned());
        let out = resolve("timesfm3", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], repo_a),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
