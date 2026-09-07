// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! S3Tokenizer's [`ArchSpec`]: ONE role, `"codebook"` - not two roles for the
//! two upstream releases (`speech_tokenizer_v2.onnx`/`_v3.onnx`). Both are
//! alternative publications of the same FSQ codebook (see the crate module
//! doc: a shared `1280 -> 8`-dim `quantizer.project_down` head feeding the
//! identical `3^8 = 6561`-entry finite-scalar quantizer, regardless of which
//! encoder body - 6-layer `AudioEncoderV2` or 12-layer MinMo - produced the
//! hidden state), so [`ArchSpec::classify`] tags either shape as a candidate
//! for the SAME role rather than inventing a `v2`/`v3` role split: with one
//! real file present it resolves trivially, and with both present
//! [`resolve`](brain_modelstore::resolve::resolve)'s own existing
//! same-role-ambiguity handling reports it correctly - no separate
//! "exactly one of" mechanism needed here.
//!
//! `classify_onnx` reads real content - the graph's own declared
//! `quantizer.project_down.bias` initializer SHAPE (`[8]`, the FSQ
//! dimensionality both releases share), never a value it decodes and never
//! the filename - via [`onnx::read`]'s `TensorProto::dims`, so this never
//! touches a weight byte.
//!
//! Honest reachability note: `crates/modelstore/src/inventory.rs`'s `scan`
//! does not yet recognize a bare `.onnx` file as any [`ArtifactKind`] at all
//! (`kind_of_extension` only knows `.gguf`/`.safetensors`, and
//! `walk_repo_dir` only additionally recognizes `tokenizer.json`/
//! `model_index.json`) - so today, scanning a real models directory never
//! feeds this `classify` a real on-disk candidate to recognize, regardless of
//! its own content. Teaching `scan` to see `.onnx` files is a real, separate
//! change to shared inventory code this migration does not make. Until it
//! does, `--codebook` is the only path this role is actually reached by - the
//! same practical shape as `llava`/`campplus`, and [`ArchSpec::missing_doc`]
//! is overridden the same way for exactly that reason. The classification
//! logic itself is exercised directly below against hand-built records, the
//! same way every other `ArchSpec`'s tests in this codebase do.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::ArtifactRecord;
use brain_modelstore::resolve::{no_default_checkpoint_doc, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct S3tokenizerSpec;

const ROLES: &[&str] = &["codebook"];

/// The FSQ projection head's bias tensor - present, at this exact shape,
/// in both `speech_tokenizer_v2.onnx` and `_v3.onnx` (see the module doc).
const FSQ_BIAS_TENSOR: &str = "quantizer.project_down.bias";
/// `log3(6561)` - both releases' codebook dimensionality.
const FSQ_DIMS: i64 = 8;

fn classify_onnx(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    if rec.path.extension().and_then(|e| e.to_str()) != Some("onnx") {
        return;
    }
    let Ok(m) = onnx::read_file(&rec.path) else { return };
    let Ok(g) = onnx::read::graph(&m) else { return };
    let is_s3tokenizer_codebook = g.initializer.iter().any(|t| t.name == FSQ_BIAS_TENSOR && t.dims == [FSQ_DIMS]);
    if is_s3tokenizer_codebook {
        out.push((idx, "codebook".to_string(), Confidence::Declared));
    }
}

impl ArchSpec for S3tokenizerSpec {
    fn arch(&self) -> &'static str {
        "s3tokenizer"
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
            classify_onnx(idx, rec, &mut out);
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("codebook").ok_or("s3tokenizer assemble: no codebook chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/s3tokenizer".to_string(), variant: None }))
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
    use brain_modelstore::resolve::{resolve, Question, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-s3tokenizer-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A minimal ONNX graph carrying exactly the one tensor `classify_onnx`
    /// looks for, at the real shape - no encoder blocks, no real weights.
    fn write_codebook_onnx(path: &Path) {
        let mut b = onnx::GraphBuilder::new("codebook");
        b.init_f32(FSQ_BIAS_TENSOR, &[FSQ_DIMS], vec![0.0f32; FSQ_DIMS as usize]);
        std::fs::write(path, b.finish()).unwrap();
    }

    /// A decoy ONNX file with an unrelated initializer - real content, but
    /// not the S3Tokenizer FSQ head.
    fn write_decoy_onnx(path: &Path) {
        let mut b = onnx::GraphBuilder::new("decoy");
        b.init_f32("some.other.tensor", &[4], vec![0.0f32; 4]);
        std::fs::write(path, b.finish()).unwrap();
    }

    /// With nothing on disk and no override, `resolve` must report Missing
    /// with a doc naming BOTH that no default checkpoint is known AND the
    /// exact `--codebook` override flag.
    #[test]
    fn no_override_and_nothing_on_disk_names_the_escape_hatch() {
        let spec = S3tokenizerSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("s3tokenizer", &[], &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1);
                let doc = &m.roles[0].doc;
                assert!(doc.contains("no default checkpoint known"), "{doc}");
                assert!(doc.contains("--codebook"), "{doc}");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// The working escape hatch: an explicit `--codebook` override resolves
    /// cleanly even with nothing else classified.
    #[test]
    fn an_explicit_override_resolves_cleanly() {
        let dir = tmp("override");
        let codebook = dir.join("speech_tokenizer_v2.onnx");
        let records = vec![complete(codebook.clone(), ArtifactKind::Opaque)];
        let spec = S3tokenizerSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("codebook".to_string(), codebook.to_string_lossy().into_owned());
        match resolve("s3tokenizer", &records, &specs, &overrides) {
            Resolution::Resolved(a) => assert_eq!(a.roles["codebook"], codebook),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A single real, on-disk codebook-shaped ONNX file (standing in for
    /// EITHER `speech_tokenizer_v2.onnx` or `_v3.onnx`) classifies and
    /// resolves with no override at all.
    #[test]
    fn classify_recognizes_a_single_codebook_shaped_file_and_resolves_trivially() {
        let dir = tmp("single");
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("speech_tokenizer_v2.onnx");
        write_codebook_onnx(&onnx);
        let records = vec![complete(onnx.clone(), ArtifactKind::Opaque)];
        let spec = S3tokenizerSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("s3tokenizer", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["codebook"], onnx),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// Both releases present at once (v2 AND v3) tag as two candidates for
    /// the SAME role - resolve's own existing same-role-ambiguity mechanism
    /// reports it, with both real files as selector choices, exactly as it
    /// would for any other architecture's role with two top-tier candidates.
    /// No separate "exactly one of" mechanism was written for this.
    #[test]
    fn both_releases_present_at_once_is_a_role_ambiguity_not_a_silent_pick() {
        let dir = tmp("both");
        std::fs::create_dir_all(&dir).unwrap();
        let v2 = dir.join("speech_tokenizer_v2.onnx");
        let v3 = dir.join("speech_tokenizer_v3.onnx");
        write_codebook_onnx(&v2);
        write_codebook_onnx(&v3);
        let records = vec![complete(v2, ArtifactKind::Opaque), complete(v3, ArtifactKind::Opaque)];
        let spec = S3tokenizerSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("s3tokenizer", &records, &specs, &BTreeMap::new()) {
            Resolution::Ambiguous(a) => {
                assert_eq!(a.question, Question::Role { role: "codebook".to_string() });
                assert_eq!(a.choices.len(), 2);
                for c in &a.choices {
                    assert_eq!(c.selector[0].0, "--codebook");
                }
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    /// A decoy `.onnx` file with unrelated content must never classify -
    /// real content, not the filename, decides.
    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("decoy");
        std::fs::create_dir_all(&dir).unwrap();
        let decoy = dir.join("speech_tokenizer_v2.onnx");
        write_decoy_onnx(&decoy);
        let records = vec![complete(decoy, ArtifactKind::Opaque)];
        let spec = S3tokenizerSpec;
        assert!(spec.classify(&records, &dir).is_empty());
    }
}
