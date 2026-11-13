// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2's [`ArchSpec`]: which `.safetensors` in the model store is
//! this model's `weights` role.
//!
//! The reference release and a container written by [`crate::import::convert`]
//! carry the same tensor names ([`crate::config::MirrorConfig::param_list`] is
//! the checkpoint's own naming), so one classification covers both. It reads
//! the safetensors header only: a file qualifies when it holds the camera
//! token and the Gaussian-parameter conv of the default configuration at their
//! exact shapes. A file whose name merely suggests the model never classifies.
//!
//! Swedish Embedded AB implements model stores that find the right checkpoint
//! from its content rather than its filename for its clients. If your team
//! needs expertise in reproducible model provisioning then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::MirrorConfig;

pub struct MirrorSpec;

const ROLES: &[&str] = &["weights"];

/// Tensors no other architecture carries, with the shape the default
/// configuration gives them.
fn fingerprint(cfg: &MirrorConfig) -> Vec<(String, Vec<usize>)> {
    let wanted = ["visual_geometry_transformer.cam_token", "gs_renderer.gs_head.2.weight"];
    cfg.param_list().into_iter().filter(|(name, _)| wanted.contains(&name.as_str())).collect()
}

fn matches_header(path: &Path, cfg: &MirrorConfig) -> bool {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(path) else { return false };
    let print = fingerprint(cfg);
    !print.is_empty() && print.iter().all(|(name, shape)| m.shape(name) == Some(shape.as_slice()))
}

impl ArchSpec for MirrorSpec {
    fn arch(&self) -> &'static str {
        "worldmirror2"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let cfg = MirrorConfig::default();
        records
            .iter()
            .enumerate()
            .filter(|(_, rec)| rec.usable() && rec.kind == ArtifactKind::Safetensors && matches_header(&rec.path, &cfg))
            // The shapes are computed from the header, not self-declared.
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Derived))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("worldmirror2 assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/worldmirror2".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("worldmirror2 validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};

    fn tmp(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-worldmirror2-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    /// A safetensors file holding exactly `tensors` (zero-filled): the header
    /// is all the spec reads.
    fn write_tensors(path: &Path, tensors: &[(String, Vec<usize>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let plan: Vec<(String, Vec<u64>)> = tensors.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        let mut w = checkpoint::weightio::StWriter::create(path.to_str().unwrap(), &plan, &serde_json::Value::Null, None).unwrap();
        for (name, shape) in tensors {
            w.write(name, &vec![0.0f32; shape.iter().product()]).unwrap();
        }
        w.finish().unwrap();
    }

    fn scanned(dir: &Path) -> Vec<ArtifactRecord> {
        let records = brain_modelstore::inventory::scan(dir);
        assert!(records.iter().all(|r| r.completeness == Completeness::Complete), "{records:?}");
        records
    }

    #[test]
    fn a_file_with_the_models_own_tensors_classifies_as_weights() {
        let dir = tmp("real");
        write_tensors(&dir.join("tencent").join("model.safetensors"), &fingerprint(&MirrorConfig::default()));
        let out = MirrorSpec.classify(&scanned(&dir), &dir);
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The name says nothing and the content says everything: a file named
    /// like the model but holding other tensors, or the right names at the
    /// wrong shapes (WorldMirror-1), is not this model's checkpoint.
    #[test]
    fn classification_reads_content_not_the_filename() {
        let dir = tmp("content");
        write_tensors(&dir.join("worldmirror2.safetensors"), &[("unrelated.weight".to_string(), vec![2, 2])]);
        let mut wrong_shape = fingerprint(&MirrorConfig::default());
        wrong_shape[0].1[3] += 1;
        write_tensors(&dir.join("other").join("model.safetensors"), &wrong_shape);
        let out = MirrorSpec.classify(&scanned(&dir), &dir);
        assert_eq!(out, Vec::new(), "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolves_end_to_end_to_the_one_checkpoint() {
        let dir = tmp("resolve");
        let path = dir.join("tencent").join("model.safetensors");
        write_tensors(&path, &fingerprint(&MirrorConfig::default()));
        let spec = MirrorSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("worldmirror2", &scanned(&dir), &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], path),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
