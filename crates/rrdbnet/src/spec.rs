// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN's [`ArchSpec`]: which on-disk `.pt`/`.pth` checkpoint
//! satisfies the single `weights` role.
//!
//! A raw `torch.save` archive carries no self-declared architecture field the
//! way a GGUF or an HF `config.json` does, so classification reads real
//! tensor shapes instead: [`RrdbConfig::from_tensors`] already derives the
//! whole net shape (trunk width, block count, upscale factor) from a
//! checkpoint's own tensor names, real content this port needs anyway at load
//! time - reused here rather than re-implemented as a lighter-weight guess.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::RrdbConfig;

pub struct RrdbnetSpec;

const ROLES: &[&str] = &["weights"];

/// The archive's real state-dict shapes, `params_ema.`/`params.`-prefix
/// stripped - the exact convention [`crate::import::read`] applies to full
/// tensor data, mirrored here over shapes only ([`checkpoint::torchpt::read_shapes`]
/// never decodes a tensor's storage bytes). `None` when the file is not a
/// readable `torch.save` archive, or carries neither prefix at all.
fn stripped_shapes(path: &Path) -> Option<HashMap<String, Vec<usize>>> {
    let raw = checkpoint::torchpt::read_shapes(path.to_string_lossy().as_ref()).ok()?;
    let prefix = if raw.iter().any(|(n, _)| n.starts_with("params_ema.")) {
        "params_ema."
    } else if raw.iter().any(|(n, _)| n.starts_with("params.")) {
        "params."
    } else {
        return None;
    };
    Some(raw.into_iter().filter_map(|(n, s)| n.strip_prefix(prefix).map(|stripped| (stripped.to_string(), s))).collect())
}

fn rrdb_config_for(rec: &ArtifactRecord) -> Option<RrdbConfig> {
    let shapes = stripped_shapes(&rec.path)?;
    RrdbConfig::from_tensors(&shapes).ok()
}

impl ArchSpec for RrdbnetSpec {
    fn arch(&self) -> &'static str {
        "rrdbnet"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Opaque {
                continue;
            }
            if !matches!(rec.path.extension().and_then(|e| e.to_str()), Some("pt" | "pth")) {
                continue;
            }
            // Real tensor names/shapes, checked against the actual RRDBNet
            // shape grammar - no format-level self-declaration exists for a
            // raw `torch.save` state dict, so this is `Derived`, not `Declared`.
            if rrdb_config_for(rec).is_some() {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let idx = *chosen.get("weights").ok_or("rrdbnet assemble: no weights chosen")?;
        let rec = &records[idx];
        let cfg = rrdb_config_for(rec).ok_or_else(|| format!("rrdbnet assemble: {} no longer derives a valid RRDBNet shape", rec.path.display()))?;
        let variant = format!("x{}-{}b", cfg.scale, cfg.num_block);
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/rrdbnet-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("rrdbnet validate: assembly has no weights role")?;
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
        std::env::temp_dir().join(format!("brain-rrdbnet-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A minimal but real RRDBNet state dict - exactly the tensors
    /// [`RrdbConfig::from_tensors`] reads, at tiny (2-block, x2) dimensions,
    /// under the `params_ema.` prefix a real release checkpoint uses.
    fn write_rrdb_pt(path: &Path, feat: usize, grow: usize, blocks: usize, ups: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut tensors = vec![
            TensorOut { name: "params_ema.conv_first.weight".to_string(), shape: vec![feat, 3, 3, 3], data: vec![0.0; feat * 3 * 3 * 3] },
            TensorOut { name: "params_ema.conv_last.weight".to_string(), shape: vec![3, feat, 3, 3], data: vec![0.0; 3 * feat * 3 * 3] },
        ];
        for b in 0..blocks {
            tensors.push(TensorOut { name: format!("params_ema.body.{b}.rdb1.conv1.weight"), shape: vec![grow, feat, 3, 3], data: vec![0.0; grow * feat * 3 * 3] });
        }
        for u in 1..=ups {
            tensors.push(TensorOut { name: format!("params_ema.conv_up{u}.weight"), shape: vec![feat, feat, 3, 3], data: vec![0.0; feat * feat * 3 * 3] });
        }
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    #[test]
    fn classify_recognizes_a_real_rrdbnet_shape() {
        let dir = tmp("real-shape");
        let path = dir.join("schwgHao").join("RealESRGAN_x4plus.pth");
        write_rrdb_pt(&path, 8, 4, 2, 2);

        let records = vec![complete(path, ArtifactKind::Opaque)];
        let out = RrdbnetSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A `.pth` with no `params_ema.*`/`params.*` prefix at all - or real
    /// RRDBNet tensor names - must never classify. A `.pt`/`.pth` extension
    /// alone is never enough.
    #[test]
    fn classify_rejects_a_pth_with_no_real_rrdbnet_shape() {
        let dir = tmp("garbage-pth");
        let path = dir.join("schwgHao").join("not-really-esrgan.pth");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real torch.save archive").unwrap();

        let records = vec![complete(path, ArtifactKind::Opaque)];
        let out = RrdbnetSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_and_derives_the_variant_from_the_weights_shape() {
        let dir = tmp("resolve-end-to-end");
        let path = dir.join("schwgHao").join("RealESRGAN_x4plus.pth");
        write_rrdb_pt(&path, 8, 4, 2, 2);

        let records = vec![complete(path.clone(), ArtifactKind::Opaque)];
        let spec = RrdbnetSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("rrdbnet", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant.as_deref(), Some("x4-2b"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let a = dir.join("schwgHao").join("RealESRGAN_x4plus.pth");
        write_rrdb_pt(&a, 8, 4, 2, 2);
        let b = dir.join("some-mirror").join("RealESRGAN_x4plus-copy.pth");
        write_rrdb_pt(&b, 8, 4, 2, 2);

        let records = vec![complete(a, ArtifactKind::Opaque), complete(b, ArtifactKind::Opaque)];
        let spec = RrdbnetSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("rrdbnet", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }
}
