// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8's [`ArchSpec`]: which on-disk `.safetensors` checkpoint satisfies
//! the single `weights` role.
//!
//! Unlike a raw upstream `.pt`/`.pth` (RRDBNet, CodeFormer), this crate's own
//! checkpoint format is brain-native: a plain safetensors file carrying its
//! own config under the `brain.config` metadata key
//! (`checkpoint::read_config`/`MmapSafetensors::config`), the same
//! self-describing convention every brain-exported checkpoint in this
//! workspace uses (`crates/yolov8/src/import.rs` produces exactly this from
//! an Ultralytics `.pt`).
//!
//! `YoloConfig::from_json` never fails - every field falls back to
//! `yolov8n`'s own default when absent, so parse SUCCESS alone proves
//! nothing (it would "succeed" on an empty `{}` or on some unrelated
//! model's config just as happily). The real signal is the same one
//! `rrdbnet::spec` uses for a shape-derived architecture:
//! [`YoloConfig::full_param_list`] names every tensor the parsed config
//! implies, with an exact element count, so classification derives a
//! candidate config and then checks that the file's REAL tensors actually
//! match it - `Confidence::Derived`, not `Declared`, for exactly that
//! reason.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::mmap::MmapSafetensors;

use crate::config::YoloConfig;

pub struct YoloSpec;

const ROLES: &[&str] = &["weights"];

/// Derive a candidate [`YoloConfig`] from `path`'s header and verify it
/// against the file's REAL tensor names/shapes - header-only, no tensor
/// bytes decoded (`MmapSafetensors::open`, mirroring `torchpt::read_shapes`'s
/// discipline for the other checkpoint format this workspace resolves).
/// `None` when the file is not a readable safetensors archive, or the
/// derived config's own tensor manifest does not actually match what is in
/// the file - the case a lenient, always-succeeding `from_json` cannot catch
/// on its own.
fn yolo_config_for(rec: &ArtifactRecord) -> Option<YoloConfig> {
    let mm = MmapSafetensors::open(&rec.path).ok()?;
    let cfg = YoloConfig::from_json(&mm.config());
    for (name, n) in cfg.full_param_list() {
        if mm.numel(&name) != Some(n) {
            return None;
        }
    }
    Some(cfg)
}

impl ArchSpec for YoloSpec {
    fn arch(&self) -> &'static str {
        "yolov8"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Safetensors {
                continue;
            }
            if !matches!(rec.path.extension().and_then(|e| e.to_str()), Some("safetensors")) {
                continue;
            }
            if yolo_config_for(rec).is_some() {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let idx = *chosen.get("weights").ok_or("yolov8 assemble: no weights chosen")?;
        let rec = &records[idx];
        let cfg = yolo_config_for(rec).ok_or_else(|| format!("yolov8 assemble: {} no longer derives a valid YOLOv8 shape", rec.path.display()))?;
        let variant = format!("nc{}", cfg.nc);
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/yolov8-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("yolov8 validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use serde_json::json;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-yolov8-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A minimal but real brain-format YOLOv8 checkpoint - exactly the
    /// tensors `cfg.full_param_list()` names, real `brain.config` header.
    fn write_yolo_st(path: &Path, cfg: &YoloConfig) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .full_param_list()
            .into_iter()
            .map(|(name, n)| (name, vec![n as u64], vec![0.0f32; n]))
            .collect();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
    }

    #[test]
    fn classify_recognizes_a_real_yolov8_checkpoint() {
        let dir = tmp("real-shape");
        let path = dir.join("local").join("yolov8-tiny.safetensors");
        write_yolo_st(&path, &YoloConfig::tiny(4));

        let records = vec![complete(path, ArtifactKind::Safetensors)];
        let out = YoloSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// `YoloConfig::from_json` never fails - it defaults every missing field
    /// - so an unrelated `.safetensors` file with SOME `brain.config` JSON
    /// but none of YOLOv8's real tensors must still be rejected: parse
    /// success alone is never enough.
    #[test]
    fn classify_rejects_a_safetensors_file_with_an_unrelated_config_and_no_real_tensors() {
        let dir = tmp("unrelated-config");
        let path = dir.join("local").join("not-yolo.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("embedding.weight".to_string(), vec![4], vec![0.0; 4])], &json!({"some_other_model": true}), None).unwrap();

        let records = vec![complete(path, ArtifactKind::Safetensors)];
        let out = YoloSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn classify_rejects_an_unreadable_safetensors_file() {
        let dir = tmp("garbage");
        let path = dir.join("local").join("garbage.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real safetensors file").unwrap();

        let records = vec![complete(path, ArtifactKind::Safetensors)];
        let out = YoloSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_and_derives_the_variant_from_the_class_count() {
        let dir = tmp("resolve-end-to-end");
        let path = dir.join("local").join("yolov8-tiny.safetensors");
        write_yolo_st(&path, &YoloConfig::tiny(7));

        let records = vec![complete(path.clone(), ArtifactKind::Safetensors)];
        let spec = YoloSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("yolov8", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant.as_deref(), Some("nc7"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let a = dir.join("vendor-a").join("model.safetensors");
        write_yolo_st(&a, &YoloConfig::tiny(4));
        let b = dir.join("vendor-b").join("model-copy.safetensors");
        write_yolo_st(&b, &YoloConfig::tiny(4));

        let records = vec![complete(a, ArtifactKind::Safetensors), complete(b, ArtifactKind::Safetensors)];
        let spec = YoloSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("yolov8", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }

    /// Regression pin, same discipline `rrdbnet::spec`/`codeformer::spec`
    /// pin: goes through the REAL scanner (`brain_modelstore::inventory::
    /// scan`) rather than a hand-built `ArtifactRecord`, so a `classify` that
    /// checked the wrong `ArtifactKind` could not pass silently the way
    /// `RrdbnetSpec`'s once did.
    #[test]
    fn classify_recognizes_a_real_safetensors_file_scanned_by_the_real_inventory_scanner() {
        let dir = tmp("real-scanner");
        let path = dir.join("local").join("yolov8-tiny.safetensors");
        write_yolo_st(&path, &YoloConfig::tiny(4));

        let records = brain_modelstore::inventory::scan(&dir);
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(records[0].kind, ArtifactKind::Safetensors, "{records:?}");

        let out = YoloSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }
}
