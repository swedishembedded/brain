// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder's [`ArchSpec`]: which on-disk artifacts satisfy its
//! `weights` and `tokenizer` roles.
//!
//! `weights` accepts a brain-format `.safetensors` checkpoint
//! (`ModelCard.family == "lfm"` - the family string every LFM2.5 checkpoint
//! in this workspace already writes, `crate::import::remap`'s own
//! `ModelCard::new(id, "lfm")`). Unlike `qwen3::spec::Qwen3Spec`, there is no
//! GGUF branch: nothing in this crate imports or serves a GGUF release (no
//! `gguf_import` module exists here at all), so a real LFM2.5 release is
//! ALWAYS the brain-native safetensors shape produced by `brain import` from
//! the HF checkpoint - mirrored here, not reinvented, from `Qwen3Spec`'s
//! safetensors branch.
//!
//! `tokenizer` is classified through the shared
//! [`brain_modelstore::resolve::classify_tokenizer_role`] - real vendor
//! co-location plus vocab-size compatibility with a `weights` candidate, the
//! same contract every other tokenizer-shaped role in this workspace
//! follows. LFM2.5 uses the SAME `data::qwen_tokenizer::QwenBpe` tokenizer
//! format Qwen3 does (`crates/lfm2/src/caps.rs` reads it directly), so no
//! new tokenizer classification logic is needed either.
//!
//! This is the prerequisite `crates/sdk`'s `EmbeddingPipeline` names as a
//! real, tracked gap: without an `ArchSpec` here, that pipeline has nothing
//! to resolve an LFM2.5 hub id or local checkpoint against and stays scoped
//! to the Qwen3/CLIP backbones it already has.
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

pub struct Lfm2Spec;

const ROLES: &[&str] = &["weights", "tokenizer"];

/// `ModelCard.family` a brain-format LFM2.5 checkpoint declares
/// (`crate::import::remap`'s own `ModelCard::new(id, "lfm")`).
pub const CARD_FAMILY: &str = "lfm";

fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(Some(card)) = checkpoint::st::read_card(&rec.path.to_string_lossy()) else { return };
    if card.family == CARD_FAMILY {
        out.push((idx, "weights".to_string(), Confidence::Declared));
    }
}

impl ArchSpec for Lfm2Spec {
    fn arch(&self) -> &'static str {
        "lfm2"
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
            if rec.kind == ArtifactKind::Safetensors {
                classify_safetensors(idx, rec, &mut out);
            }
        }
        let weights_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "weights").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &weights_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("lfm2 assemble: no weights chosen")?;
        chosen.get("tokenizer").ok_or("lfm2 assemble: no tokenizer chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/lfm2".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("lfm2 validate: assembly has no weights role")?;
        assembly.roles.get("tokenizer").ok_or("lfm2 validate: assembly has no tokenizer role")?;
        // No GGUF candidate ever exists for this arch (see this module's own
        // doc), so there is no header `brain_modelstore::resolve::
        // checkpoint_vocab_size` can read without loading tensors - the same
        // graceful no-op `Qwen3Spec::validate` falls back to for its own
        // safetensors-only checkpoints.
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
        std::env::temp_dir().join(format!("brain-lfm2-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    const TOY_VOCAB: usize = 32;

    fn write_safetensors(path: &Path, family: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let card = checkpoint::st::ModelCard::new("brain/lfm2", family);
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), Some(&card)).unwrap();
    }

    fn write_tokenizer_json(path: &Path, vocab: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let vocab: serde_json::Map<String, serde_json::Value> = (0..vocab).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    /// `ModelCard.family` is the real signal, not the file's own name - and
    /// NOT qwen3's own family string, even though both share a tokenizer
    /// format.
    #[test]
    fn classify_recognizes_the_card_family_not_qwen3s() {
        let dir = tmp("card-not-qwen3");
        let real = dir.join("vendor").join("lfm2-run7.safetensors");
        write_safetensors(&real, CARD_FAMILY);
        let decoy = dir.join("vendor").join("unrelated.safetensors");
        write_safetensors(&decoy, "qwen");

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Lfm2Spec.classify(&records, &dir);
        let weights: Vec<_> = out.iter().filter(|(_, role, _)| role == "weights").collect();
        assert_eq!(weights.len(), 1, "{out:?}");
        let (idx, ..) = weights[0];
        assert_eq!(records[*idx].path, real);
    }

    /// No candidates on disk at all resolves to `Missing`, not a panic.
    #[test]
    fn no_candidates_on_disk_is_missing_not_a_panic() {
        let dir = tmp("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Lfm2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("lfm2", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)));
    }

    /// A safetensors-only checkpoint with a same-vendor tokenizer does not
    /// auto-match by vocab (`classify_tokenizer_role`'s own documented
    /// limitation, same as `Qwen3Spec`'s equivalent test), so an explicit
    /// `--tokenizer` override is required; that override must still resolve
    /// cleanly.
    #[test]
    fn a_same_vendor_checkpoint_needs_an_explicit_tokenizer_override() {
        let dir = tmp("needs-override");
        let weights = dir.join("vendor").join("repo").join("lfm2.safetensors");
        write_safetensors(&weights, CARD_FAMILY);
        let tok_path = dir.join("vendor").join("repo").join("tokenizer.json");
        write_tokenizer_json(&tok_path, TOY_VOCAB);
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Lfm2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("lfm2", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)), "vocab-based auto-match should not fire for a bare safetensors checkpoint");

        let mut overrides = BTreeMap::new();
        overrides.insert("tokenizer".to_string(), tok_path.to_string_lossy().into_owned());
        match resolve("lfm2", &records, &specs, &overrides) {
            Resolution::Resolved(assembly) => {
                assert_eq!(assembly.roles["weights"], weights);
                assert_eq!(assembly.roles["tokenizer"], tok_path);
            }
            other => panic!("expected Resolved with an explicit tokenizer override, got {other:?}"),
        }
    }
}
