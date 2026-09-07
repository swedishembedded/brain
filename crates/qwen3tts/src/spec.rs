// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS's [`ArchSpec`]: `weights_dir` (talker/mtp/codec/speaker,
//! converted by `brain qwen3tts import`) and `ckpt` (the HF checkpoint dir
//! for `config.json`/the tokenizer).
//!
//! Both roles come from a real conversion step
//! (`crates/cli/src/supply.rs::convert_qwen3tts`) that writes its own
//! two-role `brain.manifest.json` - `classify()` recognizes that manifest's
//! roles directly, at [`Confidence::Recorded`], through the shared
//! [`classify_compound_manifest`] helper every architecture with an
//! identical real-conversion shape uses, rather than re-deriving `ckpt`/
//! `weights_dir` from raw checkpoint content a second time.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::ArtifactRecord;
use brain_modelstore::resolve::{classify_compound_manifest, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use brain_modelstore::{CompoundManifest, MANIFEST_FILE};
use capability::Assembly;

pub struct Qwen3TtsSpec;

const ROLES: &[&str] = &["weights_dir", "ckpt"];

impl ArchSpec for Qwen3TtsSpec {
    fn arch(&self) -> &'static str {
        "qwen3tts"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        classify_compound_manifest(records, inventory_root, "qwen3tts", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let ckpt_idx = *chosen.get("ckpt").ok_or("qwen3tts assemble: no ckpt chosen")?;
        let ckpt_path = &records[ckpt_idx].path;
        // classify() already trusted this directory's own manifest for its
        // roles - assemble() trusts the SAME manifest's own recorded id,
        // rather than synthesizing a fresh one nothing else agrees with.
        let id = std::fs::read(ckpt_path.join(MANIFEST_FILE))
            .ok()
            .and_then(|b| serde_json::from_slice::<CompoundManifest>(&b).ok())
            .map(|m| m.id)
            .unwrap_or_else(|| "local/qwen3tts".to_string());
        Ok(AssembleOutcome::Assembled(AssembledVariant { id, variant: None }))
    }

    fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
        // Both roles are read straight off a manifest this resolver itself
        // wrote when the checkpoint was converted - there is no further
        // header-only cross-check two Recorded roles from the SAME manifest
        // could fail that the conversion step didn't already guarantee.
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
        std::env::temp_dir().join(format!("brain-qwen3tts-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn tiny_safetensors(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    }

    /// The exact on-disk shape `convert_qwen3tts` writes: an HF checkpoint
    /// dir (`config.json`) carrying a `brain_tts/` subdirectory of imported
    /// brain-format weights and its own `brain.manifest.json` naming both
    /// as this architecture's two roles.
    fn write_qwen3tts_checkpoint(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3TTSForConditionalGeneration"]})).unwrap()).unwrap();
        tiny_safetensors(&dir.join("brain_tts").join("talker.safetensors"));
        tiny_safetensors(&dir.join("brain_tts").join("mtp.safetensors"));
        tiny_safetensors(&dir.join("brain_tts").join("codec.safetensors"));
        tiny_safetensors(&dir.join("brain_tts").join("speaker.safetensors"));
        let manifest = CompoundManifest {
            id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            family: "qwen3tts".to_string(),
            roles: BTreeMap::from([("ckpt".to_string(), ".".to_string()), ("weights_dir".to_string(), "brain_tts".to_string())]),
        };
        std::fs::write(dir.join(MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    /// The whole point of this migration: a converted checkpoint resolves
    /// through the model-store resolver with ZERO `BRAIN_QWEN3TTS_*`
    /// environment variables set - the manifest `brain qwen3tts import`
    /// already wrote is the only signal this needs.
    #[test]
    fn resolves_a_converted_checkpoint_with_no_env_vars_set() {
        let dir = tmp("resolves-clean");
        let repo = dir.join("Qwen").join("Qwen3-TTS-12Hz-0.6B-Base");
        write_qwen3tts_checkpoint(&repo);
        // A second, unrelated vendor's own file - `resolve()`'s own root
        // inference (the deepest common ancestor of every record) needs
        // this present to land on `dir` rather than collapsing onto the
        // fixture's single repo directory, exactly as flux2's own fixture
        // (`nine_b_fixture`) documents.
        std::fs::create_dir_all(dir.join("other-vendor")).unwrap();
        std::fs::write(dir.join("other-vendor").join("unrelated.gguf"), b"not a real gguf").unwrap();
        let records = brain_modelstore::inventory::scan(&dir);

        let spec = Qwen3TtsSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("qwen3tts", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.id, "Qwen/Qwen3-TTS-12Hz-0.6B-Base");
                assert_eq!(a.roles["ckpt"], repo);
                assert_eq!(a.roles["weights_dir"], repo.join("brain_tts"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A tokenizer/checkpoint from an ENTIRELY unrelated, non-qwen3tts
    /// conversion (a different family's own `brain.manifest.json`) must
    /// never classify as this architecture's roles just because it is also
    /// a compound-converted checkpoint somewhere in the same store.
    #[test]
    fn a_manifest_for_a_different_family_never_classifies() {
        let dir = tmp("wrong-family");
        let repo = dir.join("Qwen").join("Other-Model");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("config.json"), b"{}").unwrap();
        let manifest = CompoundManifest { id: "Qwen/Other-Model".to_string(), family: "wan".to_string(), roles: BTreeMap::from([("ckpt".to_string(), ".".to_string())]) };
        std::fs::write(repo.join(MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let records = brain_modelstore::inventory::scan(&dir);

        let out = Qwen3TtsSpec.classify(&records, dir.as_path());
        assert!(out.is_empty(), "{out:?}");
    }
}
