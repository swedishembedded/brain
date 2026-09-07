// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.8-27B's [`ArchSpec`]: which on-disk artifacts satisfy its `weights`
//! and `tokenizer` roles.
//!
//! Before this, the same checkpoint's location was spelled three different
//! ways across this codebase - `caps.rs`'s served action read
//! `BRAIN_QWEN35_WEIGHTS`/`BRAIN_QWEN35_TOKENIZER`, `crates/cli/src/
//! resident_qwen35.rs`'s scheduler adapter read the same two variables a
//! second time, and `crates/cli/src/resident.rs`'s separate
//! `multi_gpu_gguf_from_env` read a THIRD, `BRAIN_QWEN35_GGUF`, for the
//! int8/GGUF release. All three now resolve through this one `ArchSpec`
//! instead: `weights` accepts EITHER a brain-format `.safetensors` checkpoint
//! (`ModelCard.family == "qwen35"`, `checkpoint::st::read_card`) or a
//! `general.architecture == "qwen35"` GGUF - the same shipped-release-vs-
//! locally-trained-checkpoint duality `BRAIN_QWEN35_WEIGHTS`/
//! `BRAIN_QWEN35_GGUF` used to express as two separate variables a caller had
//! to know to pick between.
//!
//! `tokenizer` is classified through the shared
//! [`brain_modelstore::resolve::classify_tokenizer_role`] - real vendor
//! co-location plus vocab-size compatibility with a `weights` candidate, the
//! same contract every other tokenizer-shaped role in this workspace follows.
//! Its vocab check reads a GGUF `weights` candidate's own embedded
//! `tokenizer.ggml.tokens` KV; a bare brain-format `.safetensors` `weights`
//! candidate carries no sibling `config.json` for that check to read (its
//! config lives in its OWN header, not a directory alongside it), so a
//! same-vendor tokenizer next to a safetensors-only checkpoint is not
//! auto-matched by vocab today - state it explicitly with `--tokenizer` in
//! that case.
//!
//! A resolved `weights` is not interchangeable at the CONSUMING end just
//! because it is one role: `crate::caps::Qwen35Provider`'s in-process
//! resident and `crates/cli/src/qwen35_cli.rs::infer` only ever call
//! `checkpoint::load` (safetensors-only, panics on a GGUF), so both refuse a
//! resolved `.gguf` path cleanly before reaching that call rather than crash;
//! `crates/cli/src/resident_qwen35.rs::Qwen35Resident::from_assembly` and
//! `crate::int8_gguf_resident::Qwen35GgufResident` are the two consumers that
//! actually read each format, told apart by the SAME `.gguf` extension check.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery like
//! this for clients running mixed fleets of hand-placed and fetched weights.
//! If your team needs the same discipline for its own model store, you can
//! procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{classify_tokenizer_role, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;

pub struct Qwen35Spec;

const ROLES: &[&str] = &["weights", "tokenizer"];

/// `general.architecture` a Qwen3.8-27B GGUF release declares
/// (`crates/qwen35/src/gguf_import.rs::GGUF_ARCHITECTURE`).
const GGUF_ARCHITECTURE: &str = "qwen35";
/// `ModelCard.family` a brain-format Qwen3.8-27B checkpoint declares
/// (`Qwen35::save`).
const CARD_FAMILY: &str = "qwen35";

fn classify_gguf(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(g) = MmapGguf::open(&rec.path.to_string_lossy()) else { return };
    if g.kv().get("general.architecture").and_then(|v| v.as_str()) == Some(GGUF_ARCHITECTURE) {
        out.push((idx, "weights".to_string(), Confidence::Declared));
    }
}

fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(Some(card)) = checkpoint::st::read_card(&rec.path.to_string_lossy()) else { return };
    if card.family == CARD_FAMILY {
        out.push((idx, "weights".to_string(), Confidence::Declared));
    }
}

impl ArchSpec for Qwen35Spec {
    fn arch(&self) -> &'static str {
        "qwen35"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            match rec.kind {
                ArtifactKind::Gguf => classify_gguf(idx, rec, &mut out),
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                _ => {}
            }
        }
        let weights_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "weights").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &weights_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("qwen35 assemble: no weights chosen")?;
        chosen.get("tokenizer").ok_or("qwen35 assemble: no tokenizer chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/qwen35".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let weights = assembly.roles.get("weights").ok_or("qwen35 validate: assembly has no weights role")?;
        let tokenizer = assembly.roles.get("tokenizer").ok_or("qwen35 validate: assembly has no tokenizer role")?;
        // Only a GGUF `weights` candidate carries a header this can check
        // against without loading tensors (see this module's own doc) - a
        // safetensors checkpoint's vocab lives in its OWN header, not
        // something `checkpoint_vocab_size` can read from a bare file path.
        let Some(checkpoint_vocab) = brain_modelstore::resolve::checkpoint_vocab_size(weights) else { return Ok(()) };
        let Ok(bytes) = std::fs::read(tokenizer) else { return Ok(()) };
        let Some(tok_count) = brain_modelstore::resolve::tokenizer_vocab_count(&bytes) else { return Ok(()) };
        if !brain_modelstore::resolve::vocab_is_compatible(tok_count, checkpoint_vocab) {
            return Err(format!(
                "qwen35 validate: tokenizer vocab ({tok_count}) is not compatible with {}'s embedded vocab ({checkpoint_vocab}) - weights={}, tokenizer={}",
                weights.display(),
                weights.display(),
                tokenizer.display()
            ));
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
        std::env::temp_dir().join(format!("brain-qwen35-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    const TOY_VOCAB: usize = 32;

    fn write_gguf(path: &Path, arch: &str, vocab: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tokens = checkpoint::gguf::GgufValue::Array((0..vocab).map(|i| checkpoint::gguf::GgufValue::String(format!("t{i}"))).collect());
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[
                ("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(arch.to_string())),
                ("tokenizer.ggml.tokens".to_string(), tokens),
            ],
            &[checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
            32,
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, family: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let card = checkpoint::st::ModelCard::new("brain/qwen35", family);
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), Some(&card)).unwrap();
    }

    fn write_tokenizer_json(path: &Path, vocab: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let vocab: serde_json::Map<String, serde_json::Value> = (0..vocab).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    /// The real-world release shape: a GGUF `weights` candidate plus its
    /// sibling `tokenizer.json`, resolving with zero env vars.
    #[test]
    fn resolves_cleanly_from_a_gguf_and_its_sibling_tokenizer() {
        let dir = tmp("gguf-and-tokenizer");
        let vendor = dir.join("unsloth");
        let gguf_path = vendor.join("Qwen3.8-27B-Q8_0.gguf");
        write_gguf(&gguf_path, GGUF_ARCHITECTURE, TOY_VOCAB);
        let tok_path = vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path, TOY_VOCAB);
        // A second, unrelated vendor directory so root inference does not
        // collapse onto the one vendor dir (mirrors flux2::spec's own
        // fixture rationale).
        let records = vec![
            complete(gguf_path.clone(), ArtifactKind::Gguf),
            complete(tok_path.clone(), ArtifactKind::TokenizerJson),
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ];

        let spec = Qwen35Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("qwen35", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(assembly) => {
                assert_eq!(assembly.roles["weights"], gguf_path);
                assert_eq!(assembly.roles["tokenizer"], tok_path);
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A GGUF classifies purely on `general.architecture` content, not its
    /// filename - a suggestively-named but unrelated GGUF must never match.
    #[test]
    fn classify_uses_gguf_header_content_not_filename() {
        let dir = tmp("content-not-name");
        let real = dir.join("vendor").join("Qwen3.8-27B-Q8_0.gguf");
        write_gguf(&real, GGUF_ARCHITECTURE, TOY_VOCAB);
        let decoy = dir.join("vendor").join("totally-real-qwen35-checkpoint.gguf");
        write_gguf(&decoy, "some-other-arch", TOY_VOCAB);

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Qwen35Spec.classify(&records, &dir);
        let weights: Vec<_> = out.iter().filter(|(_, role, _)| role == "weights").collect();
        assert_eq!(weights.len(), 1, "{out:?}");
        let (idx, ..) = weights[0];
        assert_eq!(records[*idx].path, real);
    }

    /// The brain-native safetensors path: `ModelCard.family` is the real
    /// signal, not the file's own name.
    #[test]
    fn classify_recognizes_a_brain_format_safetensors_checkpoint_by_its_card() {
        let dir = tmp("safetensors-card");
        let real = dir.join("vendor").join("qwen35-run7.safetensors");
        write_safetensors(&real, CARD_FAMILY);
        let decoy = dir.join("vendor").join("unrelated.safetensors");
        write_safetensors(&decoy, "some-other-family");

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Qwen35Spec.classify(&records, &dir);
        let weights: Vec<_> = out.iter().filter(|(_, role, _)| role == "weights").collect();
        assert_eq!(weights.len(), 1, "{out:?}");
        let (idx, ..) = weights[0];
        assert_eq!(records[*idx].path, real);
    }

    /// A tokenizer whose vocab genuinely does not match the chosen GGUF's own
    /// embedded vocab must be rejected by `validate`, never silently resolved.
    #[test]
    fn a_mismatched_vocab_is_rejected_by_validate_before_any_gpu_work() {
        let dir = tmp("mismatched-vocab");
        let gguf_path = dir.join("Qwen3.8-27B-Q8_0.gguf");
        write_gguf(&gguf_path, GGUF_ARCHITECTURE, TOY_VOCAB);
        let tok_path = dir.join("tokenizer.json");
        write_tokenizer_json(&tok_path, TOY_VOCAB * 3);

        let assembly = Assembly {
            id: "local/qwen35".to_string(),
            arch: "qwen35".to_string(),
            variant: None,
            roles: BTreeMap::from([("weights".to_string(), gguf_path), ("tokenizer".to_string(), tok_path)]),
            provenance: Vec::new(),
        };
        let err = Qwen35Spec.validate(&assembly).unwrap_err();
        assert!(err.contains("vocab"), "{err}");
    }

    /// No candidates on disk at all resolves to `Missing`, not a panic.
    #[test]
    fn no_candidates_on_disk_is_missing_not_a_panic() {
        let dir = tmp("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Qwen35Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("qwen35", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)));
    }

    /// A safetensors-only checkpoint with a same-vendor tokenizer does not
    /// auto-match by vocab (documented limitation - see this module's own
    /// doc), so an explicit `--tokenizer` override is required; that
    /// override must still resolve cleanly.
    #[test]
    fn a_safetensors_only_checkpoint_needs_an_explicit_tokenizer_override() {
        let dir = tmp("safetensors-only");
        // Nested one level under `vendor` (a real "repo" directory) rather
        // than vendor-flat: a bare `tokenizer.json` sitting directly in a
        // vendor directory is not something `brain_modelstore::inventory`
        // scans as a record at all (only `.gguf`/`.safetensors` are
        // recognized vendor-flat; a `tokenizer.json` is only ever recorded
        // one level deeper, inside a repo directory).
        let weights_path = dir.join("vendor").join("repo").join("qwen35.safetensors");
        write_safetensors(&weights_path, CARD_FAMILY);
        let tok_path = dir.join("vendor").join("repo").join("tokenizer.json");
        write_tokenizer_json(&tok_path, TOY_VOCAB);
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Qwen35Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("qwen35", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)), "vocab-based auto-match should not fire for a bare safetensors checkpoint");

        let mut overrides = BTreeMap::new();
        overrides.insert("tokenizer".to_string(), tok_path.to_string_lossy().into_owned());
        match resolve("qwen35", &records, &specs, &overrides) {
            Resolution::Resolved(assembly) => {
                assert_eq!(assembly.roles["weights"], weights_path);
                assert_eq!(assembly.roles["tokenizer"], tok_path);
            }
            other => panic!("expected Resolved with an explicit tokenizer override, got {other:?}"),
        }
    }
}
