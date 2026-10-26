// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2's [`ArchSpec`]: which on-disk directory satisfies its one
//! `dir` role. Content-based throughout - every check reads what a file
//! DECLARES, never what it is named.
//!
//! * the upstream `deepseek-ai/DeepSeek-OCR-2` `transformers` release, whose
//!   `config.json` makes the inventory scanner collapse it into ONE `HfDir`
//!   record, so the record's path already IS the directory;
//! * a community GGUF pair: the LM GGUF (`general.architecture ==
//!   "deepseek2-ocr"`, the SAME string as DeepSeek-OCR's, because the decoder is
//!   the same) beside an mmproj whose `clip.projector_type == "deepseekocr2"` is
//!   what tells the two generations apart. It has no `config.json`, so the
//!   scanner walks it file-by-file and the LM GGUF's own record names the pair.
//!
//! [`ocrserve::dir_from_assembly`] reconciles the two.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery like this
//! for clients running mixed fleets of hand-placed and fetched weights. If your
//! team needs the same discipline for its own model store, you can procure our
//! services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;

pub struct Deepseekocr2Spec;

const ROLES: &[&str] = &["dir"];

/// The upstream release's own `architectures[0]`, as registered in `brain_arch`'s row.
const HF_ARCHITECTURE: &str = "DeepseekOCR2ForCausalLM";
/// The mmproj's own declared `clip.projector_type`: what tells this vision tower
/// apart from any other `general.architecture = "clip"` GGUF in the store.
const MMPROJ_PROJECTOR_TYPE: &str = "deepseekocr2";
/// The LM's own declared `general.architecture` (shared with DeepSeek-OCR).
const LM_ARCHITECTURE: &str = "deepseek2-ocr";
const MMPROJ_ARCHITECTURE: &str = "clip";

fn is_mmproj(g: &MmapGguf) -> bool {
    g.kv().get("general.architecture").and_then(|v| v.as_str()) == Some(MMPROJ_ARCHITECTURE) && g.kv().get("clip.projector_type").and_then(|v| v.as_str()) == Some(MMPROJ_PROJECTOR_TYPE)
}

impl ArchSpec for Deepseekocr2Spec {
    fn arch(&self) -> &'static str {
        "deepseekocr2"
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
            if rec.kind == ArtifactKind::HfDir {
                if ocrserve::declares_architecture(&rec.path, HF_ARCHITECTURE) {
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
            // A real structural check, not filename alone: the LM's own directory
            // must also hold a DeepSeek-OCR-2 mmproj (its OWN header checked).
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
        chosen.get("dir").ok_or("deepseekocr2 assemble: no dir chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/deepseekocr2".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        ocrserve::dir_from_assembly(assembly).map(|_| ())
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
        std::env::temp_dir().join(format!("brain-deepseekocr2-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    /// A structurally valid, tensor-free safetensors file (the inventory scanner
    /// probes a checkpoint's declared extent against its real size).
    fn write_empty_safetensors(path: &Path) {
        let header = b"{}";
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_hf(dir: &Path, architecture: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), format!(r#"{{"architectures":["{architecture}"]}}"#)).unwrap();
        write_empty_safetensors(&dir.join("model.safetensors"));
    }

    fn write_gguf(path: &Path, kv: &[(&str, &str)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensor = checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] };
        let kv: Vec<_> = kv.iter().map(|(k, v)| (k.to_string(), checkpoint::gguf::GgufValue::String(v.to_string()))).collect();
        checkpoint::gguf_write::write(path.to_str().unwrap(), &kv, &[tensor], 32).unwrap();
    }

    fn resolve_in(dir: &Path) -> Resolution {
        let records = brain_modelstore::inventory::scan(dir);
        let spec = Deepseekocr2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        resolve("deepseekocr2", &records, &specs, &BTreeMap::new())
    }

    /// A pulled `deepseek-ai/DeepSeek-OCR-2` resolves with no env vars, to its own
    /// DIRECTORY; DeepSeek-OCR's checkpoint beside it (the same decoder, a
    /// different `architectures` string) is not claimed.
    #[test]
    fn resolves_the_upstream_transformers_checkpoint_to_its_own_directory() {
        let dir = tmp("hf");
        let ckpt = dir.join("deepseek-ai").join("DeepSeek-OCR-2");
        write_hf(&ckpt, HF_ARCHITECTURE);
        write_hf(&dir.join("deepseek-ai").join("DeepSeek-OCR"), "DeepseekOCRForCausalLM");
        match resolve_in(&dir) {
            Resolution::Resolved(a) => assert_eq!(ocrserve::dir_from_assembly(&a).unwrap(), ckpt.to_string_lossy()),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// The community GGUF pair needs a real v2 mmproj beside its LM: a v1 mmproj
    /// (`clip.projector_type = "deepseekocr"`) next to the same LM string is
    /// DeepSeek-OCR's pair, not this model's.
    #[test]
    fn a_gguf_pair_needs_this_generations_mmproj() {
        let lm_kv = [("general.architecture", LM_ARCHITECTURE)];
        let mmproj_kv = |projector: &'static str| [("general.architecture", MMPROJ_ARCHITECTURE), ("clip.projector_type", projector)];

        let v1 = tmp("v1-pair");
        write_gguf(&v1.join("a").join("lm.gguf"), &lm_kv);
        write_gguf(&v1.join("a").join("mmproj.gguf"), &mmproj_kv("deepseekocr"));
        assert!(matches!(resolve_in(&v1), Resolution::Missing(_)), "a v1 pair is not this model");

        let v2 = tmp("v2-pair");
        write_gguf(&v2.join("a").join("lm.gguf"), &lm_kv);
        write_gguf(&v2.join("a").join("mmproj.gguf"), &mmproj_kv(MMPROJ_PROJECTOR_TYPE));
        match resolve_in(&v2) {
            Resolution::Resolved(a) => assert_eq!(ocrserve::dir_from_assembly(&a).unwrap(), v2.join("a").to_string_lossy()),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn no_candidates_on_disk_is_missing_not_a_panic() {
        let dir = tmp("empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(resolve_in(&dir), Resolution::Missing(_)));
    }
}
