// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BiSeNet's rules for the model-store resolver: which safetensors file on
//! disk is the face-parsing network PuLID's reference preprocessing needs.
//!
//! One role, `weights`. Identified by the segmentation head's own declared
//! shape - the class count is what makes a face-parsing BiSeNet a
//! face-parsing BiSeNet, and no filename can carry that.
//!
//! Swedish Embedded AB implements content-based checkpoint identification for
//! clients whose stores mix task-specific variants of one architecture. If
//! your team needs the same discipline, you can procure our services by
//! emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role name this architecture resolves.
pub const ROLES: &[&str] = &["weights"];

/// The final segmentation head's convolution.
const HEAD: &str = "conv_out.conv_out.weight";
/// The face-parsing class count the head emits (`[19, 256, 1, 1]`) - a
/// BiSeNet trained for any other segmentation task carries a different first
/// dimension here, which is exactly the distinction a name check cannot make.
const FACE_PARSING_CLASSES: usize = 19;
/// The ResNet-18 context path's deepest convolution - checked alongside the
/// head so the match rests on the whole network's shape, not one tensor.
const CONTEXT_PATH: &str = "cp.resnet.layer4.1.conv2.weight";

/// Whether `path` is a face-parsing BiSeNet checkpoint, from its own header.
pub fn is_face_parsing_bisenet(path: &Path) -> bool {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(path.to_string_lossy().as_ref()) else { return false };
    m.shape(HEAD).is_some_and(|s| s.first().copied() == Some(FACE_PARSING_CLASSES)) && m.shape(CONTEXT_PATH).is_some()
}

/// Classify every face-parsing BiSeNet in `records` under `role`, appending to
/// `out` in place. Exposed so `pulid::spec` reuses it for its own `bisenet`
/// role.
pub fn classify_weights(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::Safetensors && is_face_parsing_bisenet(&rec.path) {
            // `Derived`: computed from the head's declared shape, since a
            // bare safetensors file declares no architecture of its own.
            out.push((idx, role.to_string(), Confidence::Derived));
        }
    }
}

/// BiSeNet's [`ArchSpec`].
pub struct BisenetSpec;

impl ArchSpec for BisenetSpec {
    fn arch(&self) -> &'static str {
        "bisenet"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        classify_weights(records, "weights", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("bisenet assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/bisenet-face-parsing".to_string(), variant: Some("face-parsing".to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let path = assembly.roles.get("weights").ok_or("bisenet validate: assembly has no weights role")?;
        if !is_face_parsing_bisenet(path) {
            return Err(format!("bisenet validate: {} is not a face-parsing BiSeNet checkpoint", path.display()));
        }
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
        std::env::temp_dir().join(format!("brain-bisenet-spec-{tag}-{}-{n}", std::process::id()))
    }

    /// A real safetensors file carrying exactly the named tensors, written
    /// through the same writer the rest of this workspace uses so the header
    /// this spec reads is a genuine one.
    fn write_st(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let named: Vec<(String, Vec<u64>, Vec<f32>)> = tensors
            .iter()
            .map(|(name, shape)| ((*name).to_string(), shape.iter().map(|&d| d as u64).collect(), vec![0.0f32; shape.iter().product()]))
            .collect();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &named, &serde_json::json!({}), None).unwrap();
    }

    fn face_parsing() -> Vec<(&'static str, Vec<usize>)> {
        vec![(HEAD, vec![FACE_PARSING_CLASSES, 4, 1, 1]), (CONTEXT_PATH, vec![2, 2, 1, 1])]
    }

    fn st_record(path: &Path) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind: ArtifactKind::Safetensors, completeness: Completeness::Complete }
    }

    /// The face-parsing checkpoint classifies; a BiSeNet trained for a
    /// different task (a different class count at the head) does not, even
    /// though every other tensor matches.
    #[test]
    fn only_the_face_parsing_class_count_classifies() {
        let dir = tmp("classes");
        let face = dir.join("facexlib").join("pulid").join("parsing_bisenet.safetensors");
        write_st(&face, &face_parsing());
        let other = dir.join("vendor").join("bisenet_cityscapes.safetensors");
        write_st(&other, &[(HEAD, vec![7, 4, 1, 1]), (CONTEXT_PATH, vec![2, 2, 1, 1])]);

        assert!(is_face_parsing_bisenet(&face));
        assert!(!is_face_parsing_bisenet(&other), "a 7-class segmentation head is not face parsing");

        let mut out = Vec::new();
        classify_weights(&[st_record(&face), st_record(&other)], "weights", &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].0, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Content, not filename: the checkpoint classifies under any name, and a
    /// file merely NAMED `parsing_bisenet.safetensors` does not.
    #[test]
    fn classification_follows_content_not_the_filename() {
        let dir = tmp("content");
        let renamed = dir.join("vendor").join("hand-placed.safetensors");
        write_st(&renamed, &face_parsing());
        assert!(is_face_parsing_bisenet(&renamed));

        let impostor = dir.join("vendor").join("parsing_bisenet.safetensors");
        write_st(&impostor, &[("unrelated.weight", vec![2])]);
        assert!(!is_face_parsing_bisenet(&impostor));
        std::fs::remove_dir_all(&dir).ok();
    }
}
