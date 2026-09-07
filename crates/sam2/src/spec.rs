// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1's [`ArchSpec`]: which on-disk checkpoint (a raw `torch.save`
//! `.pt`/`.pth`, or an equivalent `.safetensors`) satisfies the single
//! `weights` role, and which released SIZE (`tiny`/`large`) it is.
//!
//! The size is real, always-present header content, not a caller's claim: the
//! trunk's patch-embedding conv's own output-channel count
//! (`image_encoder.trunk.patch_embed.proj.weight`'s first dimension) is 96 for
//! every `hiera_tiny` release and 144 for every `hiera_large` one
//! ([`crate::config::Sam2Config::hiera_tiny`]/[`hiera_large`]), so `assemble`
//! derives it from the chosen checkpoint's own shape instead of trusting a
//! `--variant`/`BRAIN_SAM2_VARIANT` flag that could name a different file than
//! the one actually loaded - the same class of gap FLUX.2's own resolver
//! migration closed for its klein-vs-base binding.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct Sam2Spec;

const ROLES: &[&str] = &["weights"];

/// Suffix every SAM 2.1 release's trunk patch-embedding conv carries,
/// regardless of the `model.` prefix a raw `torch.save` archive wraps its
/// whole state dict in ([`crate::caps::load`]'s own doc).
const PATCH_EMBED_SUFFIX: &str = "trunk.patch_embed.proj.weight";

/// `tiny`/`large` from the patch-embedding conv's own output-channel count,
/// or `None` for a checkpoint whose trunk width matches neither released size
/// this port builds ([`crate::caps::variant_config`] only knows these two).
fn variant_from_embed_dim(embed_dim: usize) -> Option<&'static str> {
    match embed_dim {
        96 => Some("tiny"),
        144 => Some("large"),
        _ => None,
    }
}

/// The patch-embed conv's own output-channel count from a checkpoint's real
/// tensor shapes (`.pt`/`.pth` via [`checkpoint::torchpt::read_shapes`], or a
/// `.safetensors` via [`checkpoint::mmap::MmapSafetensors`]) - `None` if the
/// file cannot be read or carries no such tensor.
fn patch_embed_out_channels(rec: &ArtifactRecord) -> Option<usize> {
    let path = rec.path.to_string_lossy();
    if rec.kind == ArtifactKind::Safetensors {
        let m = checkpoint::mmap::MmapSafetensors::open(path.as_ref()).ok()?;
        // Every tensor NAME is known statically (`checkpoint::mmap` opens by
        // header, so a name lookup costs nothing extra); the suffix match
        // still needs a real name to check against, so probe both spellings
        // a safetensors export could plausibly use.
        for name in ["image_encoder.trunk.patch_embed.proj.weight", "model.image_encoder.trunk.patch_embed.proj.weight"] {
            if let Some(shape) = m.shape(name) {
                return shape.first().copied();
            }
        }
        None
    } else {
        let shapes = checkpoint::torchpt::read_shapes(path.as_ref()).ok()?;
        shapes.into_iter().find(|(name, _)| name.ends_with(PATCH_EMBED_SUFFIX)).and_then(|(_, shape)| shape.first().copied())
    }
}

impl ArchSpec for Sam2Spec {
    fn arch(&self) -> &'static str {
        "sam2"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || !matches!(rec.kind, ArtifactKind::Opaque | ArtifactKind::Safetensors) {
                continue;
            }
            let is_pt = rec.kind == ArtifactKind::Opaque && matches!(rec.path.extension().and_then(|e| e.to_str()), Some("pt" | "pth"));
            if rec.kind == ArtifactKind::Opaque && !is_pt {
                continue;
            }
            let Some(embed_dim) = patch_embed_out_channels(rec) else { continue };
            if variant_from_embed_dim(embed_dim).is_some() {
                // No format-level self-declaration exists for a raw
                // `torch.save` state dict (unlike a GGUF `general.architecture`
                // KV or an HF `config.json`'s `architectures`) - this is a
                // real shape computation, so `Derived`, not `Declared`.
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let idx = *chosen.get("weights").ok_or("sam2 assemble: no weights chosen")?;
        let rec = &records[idx];
        let embed_dim = patch_embed_out_channels(rec).ok_or_else(|| format!("sam2 assemble: {} has no readable trunk patch-embedding tensor", rec.path.display()))?;
        let variant = variant_from_embed_dim(embed_dim)
            .ok_or_else(|| format!("sam2 assemble: {} has an unrecognized trunk width ({embed_dim}; only tiny=96/large=144 are supported)", rec.path.display()))?;
        if let Some(want) = overrides.get("variant") {
            if want != variant {
                return Err(format!("sam2 assemble: --variant {want} does not match the chosen checkpoint's own size ({variant}, from its trunk width)"));
            }
        }
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/sam2-{variant}"), variant: Some(variant.to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("sam2 validate: assembly has no weights role")?;
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
        std::env::temp_dir().join(format!("brain-sam2-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// Only the one tensor this spec actually reads - real archives carry
    /// hundreds more, but a fixture only needs the trunk's own patch-embed
    /// conv to exercise real shape-derived classification.
    fn write_checkpoint_pt(path: &Path, embed_dim: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let n = embed_dim * 3 * 7 * 7;
        let tensors = vec![TensorOut { name: "model.image_encoder.trunk.patch_embed.proj.weight".to_string(), shape: vec![embed_dim, 3, 7, 7], data: vec![0.0; n] }];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    #[test]
    fn classify_recognizes_tiny_and_large_from_real_trunk_width() {
        let dir = tmp("tiny-and-large");
        let tiny = dir.join("facebook").join("sam2.1_hiera_tiny.pt");
        write_checkpoint_pt(&tiny, 96);
        let large = dir.join("facebook").join("sam2.1_hiera_large.pt");
        write_checkpoint_pt(&large, 144);

        let records = vec![complete(tiny, ArtifactKind::Opaque), complete(large, ArtifactKind::Opaque)];
        let out = Sam2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived), (1, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// An unreadable/garbage `.pt` (no real trunk tensor) must never
    /// classify - a `.pt` extension alone is never enough.
    #[test]
    fn classify_rejects_a_pt_file_with_no_real_trunk_tensor() {
        let dir = tmp("garbage-pt");
        let path = dir.join("facebook").join("not-really-sam2.pt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real torch.save archive").unwrap();

        let records = vec![complete(path, ArtifactKind::Opaque)];
        let out = Sam2Spec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    /// A trunk width matching neither released size (a hypothetical
    /// `small`/`base_plus` checkpoint this port's `variant_config` cannot
    /// build) must not classify - only real, buildable sizes count.
    #[test]
    fn classify_rejects_an_unsupported_trunk_width() {
        let dir = tmp("unsupported-width");
        let path = dir.join("facebook").join("sam2.1_hiera_small.pt");
        write_checkpoint_pt(&path, 112); // hiera_small's real embed_dim - not wired up here yet
        let records = vec![complete(path, ArtifactKind::Opaque)];
        let out = Sam2Spec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_and_derives_the_variant_from_the_weights_shape() {
        let dir = tmp("resolve-end-to-end");
        let path = dir.join("facebook").join("sam2.1_hiera_tiny.pt");
        write_checkpoint_pt(&path, 96);

        let records = vec![complete(path.clone(), ArtifactKind::Opaque)];
        let spec = Sam2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("sam2", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant.as_deref(), Some("tiny"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A `--variant` override that contradicts the chosen checkpoint's own
    /// shape is refused - relabeling a real large checkpoint as tiny (or vice
    /// versa) by a stale flag must never silently proceed.
    #[test]
    fn a_variant_override_disagreeing_with_the_real_shape_is_rejected() {
        let dir = tmp("contradicting-override");
        let path = dir.join("facebook").join("sam2.1_hiera_tiny.pt");
        write_checkpoint_pt(&path, 96);

        let records = vec![complete(path, ArtifactKind::Opaque)];
        let spec = Sam2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("variant".to_string(), "large".to_string());
        let out = resolve("sam2", &records, &specs, &overrides);
        match out {
            Resolution::Missing(m) => assert!(m.roles.iter().any(|r| r.doc.contains("does not match")), "{m:?}"),
            other => panic!("expected Missing (assemble error), got {other:?}"),
        }
    }
}
