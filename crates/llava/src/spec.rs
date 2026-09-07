// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA's [`ArchSpec`]: one role, `"weights"` - the checkpoint DIRECTORY
//! itself (`config.json` + `model.safetensors` + `tokenizer.json`), exactly
//! what `crate::caps::load_vision`/`load_decode` already read. LLaVA carries
//! no separate vae/text_encoder/tokenizer role the way FLUX.2 does: the
//! CLIP-L/14@336 + Vicuna-1.5 splice this port reuses is entirely internal to
//! one checkpoint directory, so classifying the directory as a whole is
//! enough.
//!
//! No [`brain_arch::Arch::default_ref`] exists for this architecture -
//! deliberate (see that row's own comment: LLaVA is brought in only as
//! SUPIR's optional captioner, and its weights carry no auto-fetch story for
//! the same non-commercial-license reason SUPIR's own weights don't). So
//! `"weights"`'s only real acquisition path, once nothing on disk classifies,
//! is an explicit override - this spec's [`ArchSpec::missing_doc`] names it
//! instead of the resolver's generic "no artifact classifies" wording.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{no_default_checkpoint_doc, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct LlavaSpec;

const ROLES: &[&str] = &["weights"];

/// The original LLaVA repo's own declared architecture
/// (`liuhaotian/llava-v1.5-13b/config.json`'s `architectures[0]`) - distinct
/// from `LlavaQwen2ForCausalLM` (FastVLM's own, later `transformers`-native
/// checkpoint layout this port does not target; see `brain_arch`'s `llava`
/// row).
const HF_CLASS: &str = "LlavaLlamaForCausalLM";

impl ArchSpec for LlavaSpec {
    fn arch(&self) -> &'static str {
        "llava"
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
            if declared_architecture(&config).as_deref() == Some(HF_CLASS) {
                out.push((idx, "weights".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("llava assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/llava".to_string(), variant: None }))
    }

    fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
        // A single role, nothing to cross-check against.
        Ok(())
    }

    fn missing_doc(&self, role: &str) -> String {
        no_default_checkpoint_doc(self.arch(), role)
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
        std::env::temp_dir().join(format!("brain-llava-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_llava_hfdir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["LlavaLlamaForCausalLM"]})).unwrap()).unwrap();
    }

    /// With nothing on disk and no override, `resolve` must report Missing
    /// with a doc naming BOTH that no default checkpoint is known AND the
    /// exact `--weights` override flag - never the bare generic "no artifact
    /// classifies" wording, which leaves the caller to guess how to actually
    /// run this architecture (see `brain_arch`'s `llava` row: no
    /// `default_ref` exists at all).
    #[test]
    fn no_override_and_nothing_on_disk_names_the_escape_hatch() {
        let spec = LlavaSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("llava", &[], &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1);
                let doc = &m.roles[0].doc;
                assert!(doc.contains("no default checkpoint known"), "{doc}");
                assert!(doc.contains("--weights"), "{doc}");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// The working escape hatch: an explicit `--weights` override resolves
    /// cleanly even with nothing else on disk - this must never regress,
    /// it's the only way LLaVA is ever actually reached (SUPIR's optional
    /// captioner).
    #[test]
    fn an_explicit_override_resolves_cleanly() {
        let dir = tmp("override");
        let weights = dir.join("some-llava-checkpoint");
        let records = vec![complete(weights.clone(), ArtifactKind::HfDir)];
        let spec = LlavaSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("weights".to_string(), weights.to_string_lossy().into_owned());
        match resolve("llava", &records, &specs, &overrides) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], weights),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A real, on-disk LLaVA checkpoint directory classifies and resolves
    /// with no override at all - `classify` reads the directory's own
    /// `config.json`, real content, not a filename guess.
    #[test]
    fn classify_recognizes_a_real_llava_checkpoint_directory() {
        let dir = tmp("real-checkpoint");
        let ckpt = dir.join("llava-v1.5-13b");
        write_llava_hfdir(&ckpt);
        let records = vec![complete(ckpt.clone(), ArtifactKind::HfDir)];
        let spec = LlavaSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("llava", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], ckpt),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A directory named exactly like a real LLaVA release but declaring a
    /// DIFFERENT architecture (FastVLM's own `LlavaQwen2ForCausalLM`, a real,
    /// unrelated checkpoint layout) must never classify - content, not the
    /// name, decides.
    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        let suggestive = dir.join("llava-v1.5-13b");
        std::fs::create_dir_all(&suggestive).unwrap();
        std::fs::write(suggestive.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["LlavaQwen2ForCausalLM"]})).unwrap()).unwrap();
        let records = vec![complete(suggestive, ArtifactKind::HfDir)];
        let spec = LlavaSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("llava", &records, &specs, &BTreeMap::new()) {
            Resolution::Missing(_) => {}
            other => panic!("expected Missing (wrong declared architecture must not classify), got {other:?}"),
        }
    }
}
