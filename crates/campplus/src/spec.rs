// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CAM++'s [`ArchSpec`]: one role, `"dir"` - the directory holding the
//! released `campplus.onnx` (`crate::import::import_dir`'s own input shape).
//!
//! [`ArchSpec::classify`] here always returns nothing, deliberately: no
//! [`brain_arch::Arch::default_ref`] AND no `FilesRecipe` row exists for this
//! architecture in `crates/modelstore/src/recipe.rs` at all, so the model
//! store can never fetch a `campplus.onnx` on its own; and
//! `crates/modelstore/src/inventory.rs`'s `scan` recognizes only
//! `.gguf`/`.safetensors`/an HF checkpoint directory/`tokenizer.json`/
//! `model_index.json` (see `kind_of_extension`/`walk_repo_dir`) - a bare
//! `.onnx` file is never even turned into an [`ArtifactRecord`] in the first
//! place, whatever its own content declares. So there is genuinely nothing
//! for real-content classification to look at: the explicit `--dir` override
//! is the ONLY path this role is ever reached by, and that is correct, not a
//! gap to fix here.
use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::ArtifactRecord;
use brain_modelstore::resolve::{no_default_checkpoint_doc, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct CampplusSpec;

const ROLES: &[&str] = &["dir"];

impl ArchSpec for CampplusSpec {
    fn arch(&self) -> &'static str {
        "campplus"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, _records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        // See the module doc: no on-disk signal exists for this role at all.
        Vec::new()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("dir").ok_or("campplus assemble: no dir chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/campplus".to_string(), variant: None }))
    }

    fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
        Ok(())
    }

    fn missing_doc(&self, role: &str) -> String {
        no_default_checkpoint_doc(self.arch(), role)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::{ArtifactKind, Completeness};
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// With nothing on disk and no override, `resolve` must report Missing
    /// with a doc naming BOTH that no default checkpoint is known AND the
    /// exact `--dir` override flag - never the bare generic "no artifact
    /// classifies" wording.
    #[test]
    fn no_override_and_nothing_on_disk_names_the_escape_hatch() {
        let spec = CampplusSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("campplus", &[], &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1);
                let doc = &m.roles[0].doc;
                assert!(doc.contains("no default checkpoint known"), "{doc}");
                assert!(doc.contains("--dir"), "{doc}");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// The working escape hatch: an explicit `--dir` override resolves
    /// cleanly even with nothing else on disk.
    #[test]
    fn an_explicit_override_resolves_cleanly() {
        let dir = std::env::temp_dir().join(format!("brain-campplus-spec-test-override-{}", std::process::id()));
        let records = vec![complete(dir.clone(), ArtifactKind::Opaque)];
        let spec = CampplusSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("dir".to_string(), dir.to_string_lossy().into_owned());
        match resolve("campplus", &records, &specs, &overrides) {
            Resolution::Resolved(a) => assert_eq!(a.roles["dir"], dir),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// Even a record that LOOKS like a released `campplus.onnx` (right name,
    /// present on disk) must never classify by itself - the override is the
    /// only path, by design (see the module doc), not merely an
    /// unimplemented feature.
    #[test]
    fn classify_never_finds_a_campplus_onnx_by_itself() {
        let dir = std::env::temp_dir().join(format!("brain-campplus-spec-test-decoy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("campplus.onnx");
        std::fs::write(&onnx, b"not read by classify at all").unwrap();
        let records = vec![complete(onnx, ArtifactKind::Opaque)];
        let spec = CampplusSpec;
        assert!(spec.classify(&records, &dir).is_empty());
    }
}
