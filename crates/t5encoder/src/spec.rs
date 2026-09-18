// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The T5 encoder's [`ArchSpec`]: which on-disk directory carries a text
//! tower this crate can serve, satisfying the single `root` role.
//!
//! Two released layouts satisfy it, and a directory may carry either or both
//! (a FLUX.1 checkout that has had Wan's tower dropped beside it):
//!
//! * **`flux_xxl`** - `text_encoder_2/` plus `tokenizer_2/tokenizer.json`,
//!   the HF `T5EncoderModel` layout FLUX.1/2 ship their second text encoder
//!   in;
//! * **`wan_umt5`** - `wan/models_t5_umt5-xxl-enc-bf16.pth` plus
//!   `wan/tokenizer.json`, the native Wan2.1/2.2 release layout.
//!
//! Classification asks the same two questions `caps::from_env` asked of
//! `BRAIN_T5ENCODER_DIR`, against the model store instead of a variable, so
//! what resolves and what used to be configured are the same set.
//!
//! The VARIANT is not decided here. `caps`'s `encode` action takes `variant`
//! as a parameter and a directory may hold both, so pinning one at resolve
//! time would make a served model answer for only half of what it has. The
//! assembly therefore reports no variant, and the action keeps choosing.
//!
//! Swedish Embedded AB implements model-store resolution for checkpoints that
//! ship in several vendor layouts. If your team needs one loader that accepts
//! each of them on its own terms, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role this architecture resolves - the directory holding either layout.
pub const ROLES: &[&str] = &["root"];

/// The FLUX layout: an HF `T5EncoderModel` directory and its tokenizer.
pub fn has_flux_layout(dir: &Path) -> bool {
    dir.join("text_encoder_2").is_dir() && dir.join("tokenizer_2").join("tokenizer.json").is_file()
}

/// The native Wan layout: the bf16 encoder checkpoint and its tokenizer,
/// under a `wan/` subdirectory.
pub fn has_wan_layout(dir: &Path) -> bool {
    let d = dir.join("wan");
    d.join("models_t5_umt5-xxl-enc-bf16.pth").is_file() && d.join("tokenizer.json").is_file()
}

/// Whether `dir` carries a text tower this crate can serve, in either layout.
pub fn is_t5encoder_root(dir: &Path) -> bool {
    has_flux_layout(dir) || has_wan_layout(dir)
}

pub struct T5encoderSpec;

impl ArchSpec for T5encoderSpec {
    fn arch(&self) -> &'static str {
        "t5encoder"
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
            // Either layout is a DIRECTORY the loader is handed; the scanner
            // describes a released checkout as `PipelineDir`/`HfDir`.
            if !matches!(rec.kind, ArtifactKind::PipelineDir | ArtifactKind::HfDir) {
                continue;
            }
            if is_t5encoder_root(&rec.path) {
                // `Derived`: neither layout DECLARES "I am a T5 encoder"
                // anywhere - this is a real check of what the directory holds.
                out.push((idx, "root".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("root").ok_or("t5encoder assemble: no root chosen")?;
        // No variant: a root may hold both layouts, and `encode` takes
        // `variant` as its own parameter. Deciding here would hide half of
        // what the directory can answer for.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/t5encoder".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("root").ok_or("t5encoder validate: assembly has no root role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-t5encoder-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_flux(dir: &Path) {
        std::fs::create_dir_all(dir.join("text_encoder_2")).unwrap();
        std::fs::create_dir_all(dir.join("tokenizer_2")).unwrap();
        std::fs::write(dir.join("tokenizer_2").join("tokenizer.json"), b"{}").unwrap();
    }

    fn write_wan(dir: &Path) {
        std::fs::create_dir_all(dir.join("wan")).unwrap();
        std::fs::write(dir.join("wan").join("models_t5_umt5-xxl-enc-bf16.pth"), b"").unwrap();
        std::fs::write(dir.join("wan").join("tokenizer.json"), b"{}").unwrap();
    }

    #[test]
    fn classify_recognizes_the_flux_layout() {
        let dir = tmp("flux").join("black-forest-labs").join("FLUX.1-dev");
        write_flux(&dir);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert_eq!(T5encoderSpec.classify(&records, dir.as_path()), vec![(0, "root".to_string(), Confidence::Derived)]);
    }

    #[test]
    fn classify_recognizes_the_native_wan_layout() {
        let dir = tmp("wan").join("Wan-AI").join("Wan2.1-T2V-1.3B");
        write_wan(&dir);
        let records = vec![complete(dir.clone(), ArtifactKind::HfDir)];
        assert_eq!(T5encoderSpec.classify(&records, dir.as_path()), vec![(0, "root".to_string(), Confidence::Derived)]);
    }

    /// A root carrying BOTH resolves once, and reports no variant: `encode`
    /// takes `variant` itself, so pinning one here would hide the other.
    #[test]
    fn a_root_with_both_layouts_resolves_once_and_pins_no_variant() {
        let dir = tmp("both").join("vendor").join("everything");
        write_flux(&dir);
        write_wan(&dir);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert_eq!(T5encoderSpec.classify(&records, dir.as_path()).len(), 1);
        let chosen = BTreeMap::from([("root".to_string(), 0usize)]);
        match T5encoderSpec.assemble(&chosen, &records, &BTreeMap::new()).unwrap() {
            AssembleOutcome::Assembled(v) => assert_eq!(v.variant, None),
            other => panic!("expected Assembled, got {other:?}"),
        }
    }

    /// A tokenizer without the encoder beside it is a partial download: the
    /// loader would fail, so it must not resolve.
    #[test]
    fn classify_rejects_a_half_present_layout() {
        let dir = tmp("partial").join("vendor").join("half");
        std::fs::create_dir_all(dir.join("tokenizer_2")).unwrap();
        std::fs::write(dir.join("tokenizer_2").join("tokenizer.json"), b"{}").unwrap();
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert!(T5encoderSpec.classify(&records, dir.as_path()).is_empty());
    }
}
