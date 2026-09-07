// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR's [`ArchSpec`]: which on-disk directory satisfies its one
//! `dir` role. Header-only, the same contract `flux2::spec::Flux2Spec`
//! follows, and content-based throughout - every check below reads what a
//! file DECLARES, never what it is named.
//!
//! `crate::import::Files::locate` turns one directory argument into the
//! specific paths a load needs; this module is the other direction - finding
//! that directory in a scanned inventory in the first place. Both published
//! releases are recognized, and the inventory represents them differently:
//!
//! * the `ggml-org/DeepSeek-OCR-GGUF` pair - the LM GGUF
//!   (`general.architecture == "deepseek2-ocr"`) beside its vision tower
//!   (`general.architecture == "clip"`, `clip.projector_type ==
//!   "deepseekocr"`). It has no `config.json`, so
//!   [`brain_modelstore::inventory`] never collapses it into a
//!   directory-shaped record: it walks the pair file-by-file, and
//!   [`ArchSpec::classify`] names the LM GGUF's own record as the `dir`
//!   candidate.
//! * the upstream `deepseek-ai/DeepSeek-OCR` `transformers` release, whose
//!   `config.json` is exactly what makes the scanner collapse it into ONE
//!   `HfDir` record - so here the record's path already IS the directory.
//!
//! [`dir_from_assembly`] reconciles the two by asking whether the role it was
//! given is itself a directory, so neither side has to know which recipe
//! fetched the checkpoint.
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
            if !rec.usable() {
                continue;
            }
            // The upstream `transformers` release. Unlike the GGUF pair this
            // IS collapsed into one directory record (it has a `config.json`),
            // so the record's own path is already the directory
            // `crate::import::Files::locate` wants - see `dir_from_assembly`.
            if rec.kind == ArtifactKind::HfDir {
                if declares_deepseek_ocr(&rec.path) {
                    out.push((idx, "dir".to_string(), Confidence::Derived));
                }
                continue;
            }
            if rec.kind != ArtifactKind::Gguf {
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

/// Whether an HF-shaped directory's `config.json` declares THIS model.
///
/// Content, not filename: a store can hold many `transformers` checkpoints,
/// and the architecture string is the same one
/// `brain_modelstore::plan` gated the download on
/// (`brain_arch`'s `deepseek2ocr` row).
fn declares_deepseek_ocr(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else { return false };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    json["architectures"].as_array().is_some_and(|a| a.iter().any(|v| v.as_str() == Some(HF_ARCHITECTURE)))
}

/// The upstream release's own `architectures[0]`, as registered in
/// `brain_arch`'s `deepseek2ocr` row.
const HF_ARCHITECTURE: &str = "DeepseekOCRForCausalLM";

/// The checkpoint DIRECTORY a resolved [`Assembly`]'s `dir` role names.
///
/// The two layouts name it differently and the difference is structural, not
/// a heuristic: an HF checkpoint is ONE directory record, so the role already
/// holds the directory, while the GGUF pair is recorded file-by-file (no
/// `config.json` for the inventory scanner to collapse on), so the role holds
/// the LM GGUF and the directory is its parent. Testing whether the role
/// itself is a directory tells the two apart without either side having to
/// know which recipe fetched it.
///
/// A missing role or a rootless path should be unreachable in practice
/// (`resolve` only returns `Resolved` once every required role is filled, and
/// `classify` only ever names a record with a real parent), but this must
/// name it rather than panic if it somehow isn't.
pub fn dir_from_assembly(assembly: &Assembly) -> Result<String, String> {
    let role = assembly.roles.get("dir").ok_or_else(|| format!("deepseek2ocr: assembly '{}' has no dir role", assembly.id))?;
    if role.is_dir() {
        return Ok(role.to_string_lossy().into_owned());
    }
    role.parent()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| format!("deepseek2ocr: {} has no parent directory", role.display()))
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

    /// A structurally valid, tensor-free safetensors file: an 8-byte
    /// little-endian header length followed by that many bytes of JSON. The
    /// inventory scanner probes a checkpoint's declared extent against its
    /// real size, so a zero-byte placeholder is (correctly) recorded as
    /// incomplete and never classified.
    fn write_empty_safetensors(path: &Path) {
        let header = b"{}";
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(path, bytes).unwrap();
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

    /// A pulled `deepseek-ai/DeepSeek-OCR` resolves with no env vars too, and
    /// its `dir` role is the DIRECTORY rather than a file inside it -- the
    /// inventory collapses an HF checkpoint into one record because it has a
    /// `config.json`, which is exactly what the GGUF pair lacks.
    #[test]
    fn resolves_the_upstream_transformers_checkpoint_to_its_own_directory() {
        let dir = tmp("resolves-hf");
        let ckpt_dir = dir.join("deepseek-ai").join("DeepSeek-OCR");
        std::fs::create_dir_all(&ckpt_dir).unwrap();
        std::fs::write(ckpt_dir.join("config.json"), br#"{"architectures":["DeepseekOCRForCausalLM"]}"#).unwrap();
        write_empty_safetensors(&ckpt_dir.join("model.safetensors"));
        // An unrelated transformers checkpoint in the same store must not be
        // claimed: the architecture, not the shape, is what decides.
        let other = dir.join("other-vendor").join("Some-LM");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("config.json"), br#"{"architectures":["Qwen3ForCausalLM"]}"#).unwrap();
        write_empty_safetensors(&other.join("model.safetensors"));

        let records = brain_modelstore::inventory::scan(&dir);
        let out = Deepseek2ocrSpec.classify(&records, &dir);
        assert_eq!(out.len(), 1, "only the DeepSeek-OCR checkpoint is claimed: {out:?}");
        assert_eq!(records[out[0].0].path, ckpt_dir);

        let spec = Deepseek2ocrSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("deepseek2ocr", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(assembly) => assert_eq!(dir_from_assembly(&assembly).unwrap(), ckpt_dir.to_string_lossy()),
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
