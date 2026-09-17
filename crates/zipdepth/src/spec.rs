// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's [`ArchSpec`]: which on-disk `.pt`/`.pth` checkpoint satisfies
//! the single `weights` role.
//!
//! A raw `torch.save` archive carries no self-declared architecture field the
//! way a GGUF or an HF `config.json` does, so classification reads real
//! tensor shapes instead: [`ZipConfig::from_tensors`] already derives the
//! whole net shape (encoder width, per-stage depth, decoder width, which
//! upsampler) from a checkpoint's own tensor shapes, real content
//! [`crate::import::cfg_for_checkpoint`] needs anyway at load time - reused
//! here rather than re-implemented as a lighter-weight guess, the same
//! `rrdbnet::spec::RrdbnetSpec` precedent this crate follows.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::ZipConfig;

pub struct ZipdepthSpec;

const ROLES: &[&str] = &["weights"];

fn zip_config_for(rec: &ArtifactRecord) -> Option<ZipConfig> {
    let shapes = checkpoint::torchpt::read_shapes(rec.path.to_string_lossy().as_ref()).ok()?;
    ZipConfig::from_tensors(&shapes.into_iter().collect()).ok()
}

impl ArchSpec for ZipdepthSpec {
    fn arch(&self) -> &'static str {
        "zipdepth"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            // A real `.pt`/`.pth` archive the scanner could actually read
            // classifies `Torch`, not `Opaque` - the same mistake found (and
            // fixed) in `rrdbnet::spec::RrdbnetSpec` and `sam2::spec::
            // Sam2Spec` before this spec was written; checked correctly here
            // from the start.
            if !rec.usable() || rec.kind != ArtifactKind::Torch {
                continue;
            }
            if !matches!(rec.path.extension().and_then(|e| e.to_str()), Some("pt" | "pth")) {
                continue;
            }
            if zip_config_for(rec).is_some() {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let idx = *chosen.get("weights").ok_or("zipdepth assemble: no weights chosen")?;
        let rec = &records[idx];
        let cfg = zip_config_for(rec).ok_or_else(|| format!("zipdepth assemble: {} no longer derives a valid ZipDepth shape", rec.path.display()))?;
        let upsampler = if cfg.upsample_unfold { "unfold" } else { "blend" };
        let variant = format!("d{}-{upsampler}", cfg.dims[0]);
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/zipdepth-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("zipdepth validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use checkpoint::torchpt_write::TensorOut;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-zipdepth-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A minimal but real ZipDepth checkpoint: every tensor a tiny config's
    /// `param_list()` names, all zeros - exactly what a from-scratch derive
    /// needs, at dimensions far smaller than any released preset.
    fn write_zipdepth_pt(path: &Path, cfg: &ZipConfig) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors: Vec<TensorOut> = cfg
            .param_list()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                TensorOut { name, shape, data: vec![0.0f32; n] }
            })
            .collect();
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    fn tiny() -> ZipConfig {
        ZipConfig { dims: [8, 16, 32, 64], depths: [1, 1, 1, 1], dec_ch: 6, half_dec_ch: 4, ..ZipConfig::base() }
    }

    #[test]
    fn classify_recognizes_a_real_zipdepth_shape() {
        let dir = tmp("real-shape");
        let path = dir.join("skchen1993").join("zipdepth_base.pt");
        write_zipdepth_pt(&path, &tiny());

        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = ZipdepthSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A `.pt` with no real ZipDepth tensor names must never classify. A
    /// `.pt`/`.pth` extension alone is never enough.
    #[test]
    fn classify_rejects_a_pt_with_no_real_zipdepth_shape() {
        let dir = tmp("garbage-pt");
        let path = dir.join("skchen1993").join("not-really-zipdepth.pt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real torch.save archive").unwrap();

        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = ZipdepthSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    /// The NPU (blend upsampler) variant must classify too, and `assemble`
    /// must name it distinctly from the unfold variant.
    #[test]
    fn resolves_end_to_end_and_derives_the_upsampler_variant() {
        let dir = tmp("resolve-end-to-end");
        let path = dir.join("skchen1993").join("zipdepth_base_npu.pt");
        let npu = ZipConfig { upsample_unfold: false, ..tiny() };
        write_zipdepth_pt(&path, &npu);

        let records = vec![complete(path.clone(), ArtifactKind::Torch)];
        let spec = ZipdepthSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("zipdepth", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant.as_deref(), Some("d8-blend"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let a = dir.join("skchen1993").join("zipdepth_base.pt");
        write_zipdepth_pt(&a, &tiny());
        let b = dir.join("some-mirror").join("zipdepth_base-copy.pt");
        write_zipdepth_pt(&b, &tiny());

        let records = vec![complete(a, ArtifactKind::Torch), complete(b, ArtifactKind::Torch)];
        let spec = ZipdepthSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("zipdepth", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }

    /// Regression pin, the same discipline `RrdbnetSpec`/`Sam2Spec` needed
    /// added after the fact: goes through the REAL scanner
    /// (`brain_modelstore::inventory::scan`), the same one
    /// `loader::resolve_structured` uses in production, rather than a
    /// hand-picked `ArtifactKind`.
    #[test]
    fn classify_recognizes_a_real_pt_file_scanned_by_the_real_inventory_scanner() {
        let dir = tmp("real-scanner");
        let path = dir.join("skchen1993").join("zipdepth_base.pt");
        write_zipdepth_pt(&path, &tiny());

        let records = brain_modelstore::inventory::scan(&dir);
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(records[0].kind, ArtifactKind::Torch, "a real .pt scans as Torch, not Opaque");

        let out = ZipdepthSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }
}
