// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which checkpoints in a model store make up SUPIR: the frozen SDXL
//! backbone (`sdxl`, a released diffusers SDXL root, as `sdxlunet` and
//! `controlnet` resolve it) and SUPIR's own delta (`supir`, a safetensors file
//! whose tensors are the GLVControl trunk and its adaptors,
//! `model.control_model.*`). Neither is ever fetched: the SUPIR weights carry
//! a non-commercial licence, so they are the ones the operator put there.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! diffusion-based image restoration then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub const ROLES: &[&str] = &["sdxl", "supir"];

/// The prefix every tensor of the SUPIR delta carries.
const CONTROL_PREFIX: &str = "model.control_model.";

/// Whether the safetensors file at `path` is a SUPIR delta.
pub fn is_supir_checkpoint(path: &Path) -> bool {
    checkpoint::mmap::MmapSafetensors::open(path).is_ok_and(|m| m.names().iter().any(|n| n.starts_with(CONTROL_PREFIX)))
}

pub struct SupirSpec;

impl ArchSpec for SupirSpec {
    fn arch(&self) -> &'static str {
        "supir"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        sdxlunet::spec::classify_pipeline_root(records, "sdxl", &mut out);
        for (idx, rec) in records.iter().enumerate() {
            if rec.usable() && rec.kind == ArtifactKind::Safetensors && is_supir_checkpoint(&rec.path) {
                out.push((idx, "supir".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("sdxl").ok_or("supir assemble: no sdxl backbone chosen")?;
        chosen.get("supir").ok_or("supir assemble: no SUPIR checkpoint chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/supir".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("sdxl").ok_or("supir validate: assembly has no sdxl role")?;
        assembly.roles.get("supir").ok_or("supir validate: assembly has no supir role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;

    fn record(path: std::path::PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A safetensors file holding the control trunk is the `supir` role; one
    /// holding anything else is not, and an SDXL root is the `sdxl` role.
    #[test]
    fn the_delta_is_found_by_its_tensors_and_the_backbone_by_its_layout() {
        let root = std::env::temp_dir().join(format!("supir-spec-{}", std::process::id()));
        let sdxl = root.join("stabilityai").join("stable-diffusion-xl-base-1.0");
        std::fs::create_dir_all(sdxl.join("unet")).unwrap();
        let index = serde_json::json!({ "_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "UNet2DConditionModel"] });
        std::fs::write(sdxl.join("model_index.json"), index.to_string()).unwrap();
        let ckpt = root.join("Kijai").join("SUPIR").join("SUPIR-v0Q.safetensors");
        let other = root.join("other").join("model.safetensors");
        for (path, name) in [(&ckpt, "model.control_model.input_blocks.0.weight"), (&other, "model.diffusion_model.out.weight")] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            checkpoint::st::save_safetensors(path.to_str().unwrap(), &[(name.to_string(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        }
        let records = vec![record(sdxl, ArtifactKind::PipelineDir), record(ckpt, ArtifactKind::Safetensors), record(other, ArtifactKind::Safetensors)];
        let out = SupirSpec.classify(&records, &root);
        assert!(out.contains(&(0, "sdxl".to_string(), Confidence::Declared)), "{out:?}");
        assert!(out.contains(&(1, "supir".to_string(), Confidence::Declared)), "{out:?}");
        assert!(!out.iter().any(|(i, ..)| *i == 2), "{out:?}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
