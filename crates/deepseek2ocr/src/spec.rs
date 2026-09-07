// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR's [`ArchSpec`]: which on-disk directory satisfies its one
//! `dir` role - a directory holding the LM GGUF (`general.architecture ==
//! "deepseek2-ocr"`) and its sibling vision-tower GGUF (`general.architecture
//! == "clip"`, `clip.projector_type == "deepseekocr"`). Header-only, the same
//! contract `flux2::spec::Flux2Spec` follows.
//!
//! `crate::import::Files` collapses this same pair down to two named paths
//! from one directory argument; this module is the other direction - finding
//! that directory from a scanned inventory in the first place. The inventory
//! scanner never collapses this pair into one directory-shaped record (no
//! `config.json`, so [`brain_modelstore::inventory`] walks it file-by-file
//! and records each GGUF on its own), so [`ArchSpec::classify`] names the LM
//! GGUF's own record as the `dir` candidate - [`dir_from_assembly`] recovers
//! the actual directory from it.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery like
//! this for clients running mixed fleets of hand-placed and fetched weights.
//! If your team needs the same discipline for its own model store, you can
//! procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;

pub struct Deepseek2ocrSpec;

const ROLES: &[&str] = &["dir"];

/// The mmproj's own declared `clip.projector_type` (`crates/gguf/src/
/// deepseek_ocr_vision.rs::PROJECTOR_TYPE`) - what tells this vision tower
/// apart from any other `general.architecture = "clip"` GGUF in the store.
const MMPROJ_PROJECTOR_TYPE: &str = "deepseekocr";
/// The LM's own declared `general.architecture`
/// (`crates/gguf/src/deepseek_ocr.rs::GGUF_ARCHITECTURE`).
const LM_ARCHITECTURE: &str = "deepseek2-ocr";
/// The mmproj's own declared `general.architecture`
/// (`crates/gguf/src/deepseek_ocr_vision.rs::GGUF_ARCHITECTURE`).
const MMPROJ_ARCHITECTURE: &str = "clip";

fn is_mmproj(g: &MmapGguf) -> bool {
    g.kv().get("general.architecture").and_then(|v| v.as_str()) == Some(MMPROJ_ARCHITECTURE) && g.kv().get("clip.projector_type").and_then(|v| v.as_str()) == Some(MMPROJ_PROJECTOR_TYPE)
}

impl ArchSpec for Deepseek2ocrSpec {
    fn arch(&self) -> &'static str {
        "deepseek2ocr"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Gguf {
                continue;
            }
            let Ok(g) = MmapGguf::open(&rec.path.to_string_lossy()) else { continue };
            if g.kv().get("general.architecture").and_then(|v| v.as_str()) != Some(LM_ARCHITECTURE) {
                continue;
            }
            // A real structural check, not filename alone: the LM's own
            // directory must also hold a real mmproj GGUF (its OWN header
            // content checked, not its name).
            let Some(dir) = rec.path.parent() else { continue };
            let has_mmproj = records.iter().any(|other| {
                other.usable() && other.kind == ArtifactKind::Gguf && other.path.parent() == Some(dir) && MmapGguf::open(&other.path.to_string_lossy()).is_ok_and(|m| is_mmproj(&m))
            });
            if has_mmproj {
                out.push((idx, "dir".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("dir").ok_or("deepseek2ocr assemble: no dir chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/deepseek2ocr".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        dir_from_assembly(assembly).map(|_| ())
    }
}

/// The checkpoint DIRECTORY a resolved [`Assembly`]'s `dir` role names - the
/// LM GGUF's own parent, since that is the record [`ArchSpec::classify`]
/// picked (see this module's doc). A missing role or a rootless path should
/// be unreachable in practice (`resolve` only returns `Resolved` once every
/// required role is filled, and `classify` only ever names a record with a
/// real parent), but this must name it rather than panic if it somehow isn't.
pub fn dir_from_assembly(assembly: &Assembly) -> Result<String, String> {
    let lm = assembly.roles.get("dir").ok_or_else(|| format!("deepseek2ocr: assembly '{}' has no dir role", assembly.id))?;
    lm.parent()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| format!("deepseek2ocr: {} has no parent directory", lm.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-deepseek2ocr-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn tiny_tensor() -> checkpoint::gguf_write::TensorOut {
        checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }
    }

    fn write_lm_gguf(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(LM_ARCHITECTURE.to_string()))],
            &[tiny_tensor()],
            32,
        )
        .unwrap();
    }

    fn write_mmproj_gguf(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[
                ("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(MMPROJ_ARCHITECTURE.to_string())),
                ("clip.projector_type".to_string(), checkpoint::gguf::GgufValue::String(MMPROJ_PROJECTOR_TYPE.to_string())),
            ],
            &[tiny_tensor()],
            32,
        )
        .unwrap();
    }

    /// A plain CLIP GGUF (a real, unrelated architecture's own vision tower)
    /// carries the exact SAME `general.architecture = "clip"` value this
    /// checkpoint's mmproj does - `clip.projector_type` is the real content
    /// that tells them apart, not the filename.
    fn write_unrelated_clip_gguf(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[
                ("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(MMPROJ_ARCHITECTURE.to_string())),
                ("clip.projector_type".to_string(), checkpoint::gguf::GgufValue::String("mlp".to_string())),
            ],
            &[tiny_tensor()],
            32,
        )
        .unwrap();
    }

    #[test]
    fn classify_requires_a_real_sibling_mmproj_not_just_the_lm_header() {
        let dir = tmp("lone-lm");
        let ckpt = dir.join("ggml-org").join("DeepSeek-OCR-GGUF");
        write_lm_gguf(&ckpt.join("DeepSeek-OCR-Q8_0.gguf"));
        // No mmproj beside it, and an unrelated CLIP tower elsewhere in the
        // store that must not be mistaken for one.
        write_unrelated_clip_gguf(&dir.join("other-vendor").join("clip-vit.gguf"));

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Deepseek2ocrSpec.classify(&records, &dir);
        assert!(out.is_empty(), "no real mmproj sibling exists: {out:?}");
    }

    #[test]
    fn classify_finds_the_lm_gguf_when_a_real_mmproj_sibling_exists() {
        let dir = tmp("real-pair");
        let ckpt_dir = dir.join("ggml-org").join("DeepSeek-OCR-GGUF");
        let lm_path = ckpt_dir.join("DeepSeek-OCR-Q8_0.gguf");
        write_lm_gguf(&lm_path);
        write_mmproj_gguf(&ckpt_dir.join("mmproj-DeepSeek-OCR-Q8_0.gguf"));
        // An unrelated same-architecture CLIP tower elsewhere must not
        // confuse the pairing (it lives in a different directory).
        write_unrelated_clip_gguf(&dir.join("other-vendor").join("clip-vit.gguf"));

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Deepseek2ocrSpec.classify(&records, &dir);
        let dirs: Vec<_> = out.iter().filter(|(_, role, _)| role == "dir").collect();
        assert_eq!(dirs.len(), 1, "{out:?}");
        let (idx, ..) = dirs[0];
        assert_eq!(records[*idx].path, lm_path);
    }

    #[test]
    fn resolves_cleanly_with_zero_env_vars_from_a_real_checkpoint_pair() {
        let dir = tmp("resolves-clean");
        let ckpt_dir = dir.join("ggml-org").join("DeepSeek-OCR-GGUF");
        write_lm_gguf(&ckpt_dir.join("DeepSeek-OCR-Q8_0.gguf"));
        write_mmproj_gguf(&ckpt_dir.join("mmproj-DeepSeek-OCR-Q8_0.gguf"));
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Deepseek2ocrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("deepseek2ocr", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(assembly) => {
                assert_eq!(assembly.arch, "deepseek2ocr");
                assert_eq!(dir_from_assembly(&assembly).unwrap(), ckpt_dir.to_string_lossy());
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn no_candidates_on_disk_is_missing_not_a_panic() {
        let dir = tmp("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Deepseek2ocrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        assert!(matches!(resolve("deepseek2ocr", &records, &specs, &BTreeMap::new()), Resolution::Missing(_)));
    }
}
