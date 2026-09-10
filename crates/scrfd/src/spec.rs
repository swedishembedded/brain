// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SCRFD's rules for the model-store resolver: which ONNX graph on disk is
//! insightface's released SCRFD-10GF `bnkps` face detector.
//!
//! One role, `weights` - the `scrfd_10g_bnkps.onnx` graph. Identified from its
//! own initializer names through `onnx::header` (which seeks past every
//! tensor's `raw_data`), never from its filename.
//!
//! Swedish Embedded AB implements content-based model identification for
//! clients running mixed detector fleets. If your team needs weight resolution
//! that never guesses which graph it loaded, you can procure our services by
//! emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role name this architecture resolves - the released ONNX graph.
pub const ROLES: &[&str] = &["weights"];

/// The keypoint head at the finest stride. The released graph names its heads
/// with the stride tuple spelled out literally, spaces included
/// (`stride_kps.(8, 8)`), and a `kps` head at all is what the `bnkps` in the
/// release name means - a plain SCRFD without keypoints has only `stride_cls`
/// and `stride_reg`, so this single name separates the variant
/// `crates/scrfd` decodes from the one it cannot.
const KPS_HEAD: &str = "bbox_head.stride_kps.(8, 8).weight";
/// The PAFPN neck's own convolutions - present across SCRFD sizes, checked so
/// classification rests on the model's structure and not on one head alone.
const NECK_CONV: &str = "neck.fpn_convs.0.conv.weight";

/// Whether `path` is a released SCRFD `bnkps` detector, from its own
/// initializer names.
pub fn is_scrfd_graph(path: &Path) -> bool {
    let Ok(inits) = onnx::header::read_initializers(path) else { return false };
    inits.iter().any(|i| i.name == KPS_HEAD) && inits.iter().any(|i| i.name == NECK_CONV)
}

/// Classify every released SCRFD graph in `records` under `role`, appending to
/// `out` in place. Exposed for reuse the same way `arcface::spec`'s own
/// classifier is.
pub fn classify_weights(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::Onnx && is_scrfd_graph(&rec.path) {
            // `Derived` for the same reason `arcface::spec` uses it: an ONNX
            // graph declares no architecture of its own.
            out.push((idx, role.to_string(), Confidence::Derived));
        }
    }
}

/// SCRFD's [`ArchSpec`].
pub struct ScrfdSpec;

impl ArchSpec for ScrfdSpec {
    fn arch(&self) -> &'static str {
        "scrfd"
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
        chosen.get("weights").ok_or("scrfd assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/scrfd-10g-bnkps".to_string(), variant: Some("10g-bnkps".to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let path = assembly.roles.get("weights").ok_or("scrfd validate: assembly has no weights role")?;
        if !is_scrfd_graph(path) {
            return Err(format!("scrfd validate: {} is not a released SCRFD bnkps graph", path.display()));
        }
        Ok(())
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
        std::env::temp_dir().join(format!("brain-scrfd-spec-{tag}-{}-{n}", std::process::id()))
    }

    fn write_graph(path: &Path, names: &[&str]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let initializer = names
            .iter()
            .map(|name| onnx::onnx::TensorProto { name: (*name).to_string(), dims: vec![4], data_type: 1, raw_data: vec![0u8; 16], ..Default::default() })
            .collect();
        let graph = onnx::onnx::GraphProto { name: "torch-jit-export".to_string(), initializer, ..Default::default() };
        let model = onnx::onnx::ModelProto { ir_version: 6, producer_name: "pytorch".to_string(), graph: Some(graph), ..Default::default() };
        std::fs::write(path, onnx::encode_model(&model)).unwrap();
    }

    fn onnx_record(path: &Path) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind: ArtifactKind::Onnx, completeness: Completeness::Complete }
    }

    /// The detector resolves on its own, and the identity embedder shipped in
    /// the SAME directory is not mistaken for it - the two really do sit side
    /// by side in insightface's antelopev2 release.
    #[test]
    fn the_detector_resolves_and_the_embedder_beside_it_does_not() {
        let dir = tmp("resolve");
        let repo = dir.join("DIAMONIK7777").join("antelopev2");
        let detector = repo.join("scrfd_10g_bnkps.onnx");
        let embedder = repo.join("glintr100.onnx");
        write_graph(&detector, &[KPS_HEAD, NECK_CONV, "bbox_head.scales.0.scale"]);
        write_graph(&embedder, &["fc.weight", "layer3.29.bn1.weight"]);

        let records = vec![onnx_record(&detector), onnx_record(&embedder)];
        let specs: Vec<&dyn ArchSpec> = vec![&ScrfdSpec];
        match resolve("scrfd", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], detector),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A keypoint-less SCRFD carries the same neck but no `stride_kps` head -
    /// `crates/scrfd` decodes five landmarks, so serving that graph would
    /// fail at the first detection. It must not classify.
    #[test]
    fn an_scrfd_without_keypoints_does_not_classify() {
        let dir = tmp("nokps");
        let plain = dir.join("vendor").join("scrfd_10g.onnx");
        write_graph(&plain, &[NECK_CONV, "bbox_head.stride_cls.(8, 8).weight", "bbox_head.stride_reg.(8, 8).weight"]);
        assert!(!is_scrfd_graph(&plain));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Content, not filename: the released graph classifies under any name.
    #[test]
    fn classification_follows_content_not_the_filename() {
        let dir = tmp("content");
        let renamed = dir.join("vendor").join("hand-placed-detector.onnx");
        write_graph(&renamed, &[KPS_HEAD, NECK_CONV]);
        assert!(is_scrfd_graph(&renamed));
        std::fs::remove_dir_all(&dir).ok();
    }
}
