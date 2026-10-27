// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv5's [`ArchSpec`]: which on-disk `.safetensors` file satisfies the single
//! `weights` role.
//!
//! A file qualifies only if its `brain.config` header parses as a
//! [`crate::Yolov5Config`] (strict: unknown or missing keys fail) AND its real
//! tensor manifest matches the graph that config implies, name for name and
//! shape for shape ([`weights::read_config`], header-only: no tensor bytes are
//! decoded). Both halves are needed because neither alone proves anything: a
//! header can be well-formed for a file that is not this network, and a file can
//! carry plausible tensors under no header. `Confidence::Derived`, not
//! `Declared`, for the same reason `yolov8::spec` is.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::weights;

pub struct Yolov5Spec;

const ROLES: &[&str] = &["weights"];

fn is_yolov5_checkpoint(rec: &ArtifactRecord) -> Option<crate::Yolov5Config> {
    if !rec.usable() || rec.kind != ArtifactKind::Safetensors {
        return None;
    }
    if rec.path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return None;
    }
    weights::read_config(&rec.path).ok()
}

impl ArchSpec for Yolov5Spec {
    fn arch(&self) -> &'static str {
        "yolov5"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, rec)| is_yolov5_checkpoint(rec).is_some())
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Derived))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let idx = *chosen.get("weights").ok_or("yolov5 assemble: no weights chosen")?;
        let rec = &records[idx];
        let cfg = is_yolov5_checkpoint(rec).ok_or_else(|| format!("yolov5 assemble: {} no longer derives a valid YOLOv5 network", rec.path.display()))?;
        let variant = format!("nc{}", cfg.nc);
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/yolov5-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("yolov5 validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Activation, Yolov5Config};
    use crate::graph::Graph;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-yolov5-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn tiny_cfg(nc: u32) -> Yolov5Config {
        let mut cfg = Yolov5Config::yolov5s(Activation::Relu);
        cfg.nc = nc;
        cfg.width_multiple = 0.125;
        cfg
    }

    fn write_checkpoint(path: &Path, cfg: &Yolov5Config) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let graph = Graph::build(cfg);
        let w = crate::weights::tests::synthetic(&graph, 1);
        crate::weights::save(path, cfg, &graph, &w).unwrap();
    }

    fn record(path: PathBuf) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind: ArtifactKind::Safetensors, completeness: Completeness::Complete }
    }

    #[test]
    fn a_real_checkpoint_is_the_weights_role_and_an_unrelated_file_is_not() {
        let dir = tmp("classify");
        let good = dir.join("local").join("yolov5-tiny.safetensors");
        write_checkpoint(&good, &tiny_cfg(4));
        let other = dir.join("local").join("other.safetensors");
        checkpoint::st::save_safetensors(other.to_str().unwrap(), &[("embedding.weight".to_string(), vec![4], vec![0.0; 4])], &serde_json::json!({"some_other_model": true}), None).unwrap();
        let garbage = dir.join("local").join("garbage.safetensors");
        std::fs::write(&garbage, b"not a safetensors file").unwrap();

        let records = vec![record(good), record(other), record(garbage)];
        let out = Yolov5Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_and_derives_the_variant_from_the_class_count() {
        let dir = tmp("resolve");
        let path = dir.join("local").join("yolov5-tiny.safetensors");
        write_checkpoint(&path, &tiny_cfg(7));
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Yolov5Spec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        match resolve("yolov5", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant.as_deref(), Some("nc7"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_are_ambiguous_not_a_silent_pick() {
        let dir = tmp("ambiguous");
        write_checkpoint(&dir.join("a").join("m.safetensors"), &tiny_cfg(4));
        write_checkpoint(&dir.join("b").join("m.safetensors"), &tiny_cfg(4));
        let records = brain_modelstore::inventory::scan(&dir);
        let spec = Yolov5Spec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        assert!(matches!(resolve("yolov5", &records, &specs, &BTreeMap::new()), Resolution::Ambiguous(_)));
    }
}
