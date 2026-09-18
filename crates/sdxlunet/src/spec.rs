// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL's [`ArchSpec`]: which on-disk directory is a released SDXL pipeline
//! root, satisfying the single `root` role.
//!
//! Identified from the pipeline's own `model_index.json` rather than from its
//! directory name or a file-shape probe, the same way
//! [`flux1::spec::is_flux1_pipeline_root`] does - a released diffusers
//! pipeline DECLARES its class, so reading that declaration is both cheaper
//! and stricter than inferring one. The transformer/UNet component's class is
//! checked as well as the pipeline's, so a re-packaged pipeline that kept the
//! outer name but swapped the backbone cannot pass.
//!
//! `Confidence::Declared`: unlike a raw `torch.save` state dict, this really
//! is a self-declaration in the artifact.
//!
//! [`classify_pipeline_root`] is public because [`crate`] is not the only
//! consumer: `controlnet::spec`'s own `sdxl` role is this same artifact, and
//! both must ask the identical question of the identical bytes rather than
//! each carrying its own idea of what SDXL looks like.
//!
//! Swedish Embedded AB implements model-store resolution that identifies a
//! checkpoint from its own declared contents. If your team needs weights
//! discovered reliably instead of configured by hand, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role this architecture resolves - the pipeline root directory.
pub const ROLES: &[&str] = &["root"];

/// The `_class_name` a released SDXL pipeline's `model_index.json` declares.
/// SD 1.x/2.x declare `StableDiffusionPipeline` and FLUX `FluxPipeline`, so
/// this one string separates them with no shape work.
const PIPELINE_CLASS: &str = "StableDiffusionXLPipeline";
/// The UNet class the same manifest names for its `unet` entry.
const UNET_CLASS: &str = "UNet2DConditionModel";

fn pipeline_manifest(dir: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(dir.join("model_index.json")).ok()?).ok()
}

/// A component entry in a `model_index.json` is a `[library, class]` pair;
/// this is the class half.
fn component_class(manifest: &serde_json::Value, component: &str) -> Option<String> {
    manifest.get(component)?.as_array()?.get(1)?.as_str().map(str::to_string)
}

/// Whether `dir` is a released SDXL pipeline root. Shared with
/// `controlnet::spec`, whose `sdxl` role is this same artifact.
///
/// The `unet/` directory is required as well as the declaration: every
/// consumer here (`SdxlProvider`, `ControlnetProvider`) joins that path, and
/// a manifest naming a component the checkout does not actually contain is a
/// real shape a partial download leaves behind.
pub fn is_sdxl_pipeline_root(dir: &Path) -> bool {
    let Some(manifest) = pipeline_manifest(dir) else { return false };
    manifest.get("_class_name").and_then(serde_json::Value::as_str) == Some(PIPELINE_CLASS)
        && component_class(&manifest, "unet").as_deref() == Some(UNET_CLASS)
        && dir.join("unet").is_dir()
}

/// Classify every SDXL pipeline root in `records` under `role`, appending to
/// `out` in place - the shape every `classify_*` helper in this workspace
/// takes. Exposed so `controlnet::spec` reuses it for its own `sdxl` role.
pub fn classify_pipeline_root(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::PipelineDir && is_sdxl_pipeline_root(&rec.path) {
            out.push((idx, role.to_string(), Confidence::Declared));
        }
    }
}

pub struct SdxlunetSpec;

impl ArchSpec for SdxlunetSpec {
    fn arch(&self) -> &'static str {
        "sdxlunet"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        classify_pipeline_root(records, "root", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("root").ok_or("sdxlunet assemble: no root chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/sdxl".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("root").ok_or("sdxlunet validate: assembly has no root role")?;
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
        std::env::temp_dir().join(format!("brain-sdxlunet-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A released SDXL checkout: the manifest's two declarations plus the
    /// `unet/` directory every consumer joins.
    fn write_sdxl_root(dir: &Path, class: &str, unet_class: &str, with_unet: bool) {
        std::fs::create_dir_all(dir).unwrap();
        if with_unet {
            std::fs::create_dir_all(dir.join("unet")).unwrap();
        }
        let manifest = serde_json::json!({ "_class_name": class, "unet": ["diffusers", unet_class] });
        std::fs::write(dir.join("model_index.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn classify_recognizes_a_released_sdxl_pipeline_root() {
        let dir = tmp("real").join("stabilityai").join("stable-diffusion-xl-base-1.0");
        write_sdxl_root(&dir, PIPELINE_CLASS, UNET_CLASS, true);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        let out = SdxlunetSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "root".to_string(), Confidence::Declared)], "{out:?}");
    }

    /// The case this spec exists to get right: a FLUX pipeline root is also a
    /// `PipelineDir` sitting in the same store, and must not satisfy SDXL's
    /// role.
    #[test]
    fn classify_rejects_another_familys_pipeline_root() {
        let dir = tmp("flux").join("black-forest-labs").join("FLUX.1-dev");
        write_sdxl_root(&dir, "FluxPipeline", "FluxTransformer2DModel", true);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert!(SdxlunetSpec.classify(&records, dir.as_path()).is_empty());
    }

    /// A manifest that declares SDXL but whose backbone was swapped.
    #[test]
    fn classify_rejects_a_repackaged_pipeline_with_a_foreign_unet() {
        let dir = tmp("swapped").join("someone").join("sdxl-ish");
        write_sdxl_root(&dir, PIPELINE_CLASS, "FluxTransformer2DModel", true);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert!(SdxlunetSpec.classify(&records, dir.as_path()).is_empty());
    }

    /// A partial download: the manifest names a `unet` the checkout has not
    /// got. Every consumer joins that path, so accepting it here would
    /// resolve a model that cannot load.
    #[test]
    fn classify_rejects_a_root_whose_declared_unet_is_absent() {
        let dir = tmp("partial").join("stabilityai").join("sdxl");
        write_sdxl_root(&dir, PIPELINE_CLASS, UNET_CLASS, false);
        let records = vec![complete(dir.clone(), ArtifactKind::PipelineDir)];
        assert!(SdxlunetSpec.classify(&records, dir.as_path()).is_empty());
    }
}
