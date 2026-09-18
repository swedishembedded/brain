// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ControlNet's [`ArchSpec`]: the two artifacts a controlled generation needs
//! - the SDXL backbone it conditions (`sdxl`) and the trainable-copy
//! checkpoint that produces the residuals (`control`).
//!
//! The `sdxl` role is NOT re-derived here: it is
//! [`sdxlunet::spec::classify_pipeline_root`], the same predicate over the
//! same bytes the SDXL architecture itself resolves with. Two ideas of what
//! SDXL looks like is exactly how the two would drift.
//!
//! The `control` role is a diffusers `ControlNetModel` checkpoint, identified
//! by its own `config.json` declaration rather than by filename - a
//! ControlNet release is a directory of one or more `.safetensors` beside
//! that config, and `crate::import::load` already accepts either the
//! directory or one file, preferring an `fp16` variant when several are
//! present.
//!
//! Swedish Embedded AB implements model-store resolution for composed
//! pipelines, where one served model draws on several independently released
//! checkpoints. If your team needs that resolved reliably rather than
//! configured by hand, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The two roles this architecture resolves. `sdxl` first: it is the backbone
/// the control copy is only meaningful against.
pub const ROLES: &[&str] = &["sdxl", "control"];

/// The `_class_name` a released ControlNet's own `config.json` declares.
const CONTROL_CLASS: &str = "ControlNetModel";

/// Whether `dir` holds a released `ControlNetModel`: its own `config.json`
/// declaring that class, and at least one `.safetensors` beside it for
/// `crate::import::load` to choose from.
pub fn is_controlnet_dir(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else { return false };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    if cfg.get("_class_name").and_then(serde_json::Value::as_str) != Some(CONTROL_CLASS) {
        return false;
    }
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(Result::ok).any(|e| e.path().extension().is_some_and(|x| x == "safetensors")))
        .unwrap_or(false)
}

pub struct ControlnetSpec;

impl ArchSpec for ControlnetSpec {
    fn arch(&self) -> &'static str {
        "controlnet"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        // The backbone, through SDXL's own predicate rather than a second copy.
        sdxlunet::spec::classify_pipeline_root(records, "sdxl", &mut out);
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            // A ControlNet release is a DIRECTORY (config.json + shards);
            // `HfDir` is the kind the scanner gives such a checkout.
            let dir: &Path = match rec.kind {
                ArtifactKind::HfDir | ArtifactKind::PipelineDir => &rec.path,
                // A loose shard beside its own config.json still resolves -
                // the parent is what `import::load` is handed.
                ArtifactKind::Safetensors => match rec.path.parent() {
                    Some(p) => p,
                    None => continue,
                },
                _ => continue,
            };
            if is_controlnet_dir(dir) && !out.iter().any(|(i, r, _)| *i == idx && r == "control") {
                out.push((idx, "control".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("sdxl").ok_or("controlnet assemble: no sdxl backbone chosen")?;
        chosen.get("control").ok_or("controlnet assemble: no control checkpoint chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/sdxl-controlnet".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("sdxl").ok_or("controlnet validate: assembly has no sdxl role")?;
        assembly.roles.get("control").ok_or("controlnet validate: assembly has no control role")?;
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
        std::env::temp_dir().join(format!("brain-controlnet-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_controlnet(dir: &Path, class: &str, with_weights: bool) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({ "_class_name": class })).unwrap()).unwrap();
        if with_weights {
            std::fs::write(dir.join("diffusion_pytorch_model.fp16.safetensors"), b"").unwrap();
        }
    }

    fn write_sdxl_root(dir: &Path) {
        std::fs::create_dir_all(dir.join("unet")).unwrap();
        let manifest = serde_json::json!({ "_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "UNet2DConditionModel"] });
        std::fs::write(dir.join("model_index.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn classify_finds_both_roles_in_one_store() {
        let root = tmp("both");
        let sdxl = root.join("stabilityai").join("stable-diffusion-xl-base-1.0");
        let ctrl = root.join("diffusers").join("controlnet-canny-sdxl-1.0");
        write_sdxl_root(&sdxl);
        write_controlnet(&ctrl, CONTROL_CLASS, true);
        let records = vec![complete(sdxl, ArtifactKind::PipelineDir), complete(ctrl, ArtifactKind::HfDir)];
        let out = ControlnetSpec.classify(&records, root.as_path());
        assert!(out.contains(&(0, "sdxl".to_string(), Confidence::Declared)), "{out:?}");
        assert!(out.contains(&(1, "control".to_string(), Confidence::Declared)), "{out:?}");
    }

    /// The SDXL backbone must never also satisfy `control`: both are
    /// directories with JSON in them, and a spec that confused the two would
    /// assemble a model whose control copy is the thing it conditions.
    #[test]
    fn an_sdxl_root_does_not_satisfy_the_control_role() {
        let root = tmp("no-cross");
        let sdxl = root.join("stabilityai").join("sdxl");
        write_sdxl_root(&sdxl);
        let records = vec![complete(sdxl, ArtifactKind::PipelineDir)];
        let out = ControlnetSpec.classify(&records, root.as_path());
        assert!(out.iter().all(|(_, r, _)| r != "control"), "{out:?}");
    }

    /// A config declaring some other diffusers class is not a ControlNet.
    #[test]
    fn classify_rejects_a_foreign_class() {
        let root = tmp("foreign");
        let d = root.join("someone").join("not-a-controlnet");
        write_controlnet(&d, "UNet2DConditionModel", true);
        let records = vec![complete(d, ArtifactKind::HfDir)];
        assert!(ControlnetSpec.classify(&records, root.as_path()).is_empty());
    }

    /// A config with no weights beside it is a partial download, and
    /// `import::load` would fail with "no .safetensors in ...".
    #[test]
    fn classify_rejects_a_control_dir_with_no_weights() {
        let root = tmp("no-weights");
        let d = root.join("diffusers").join("controlnet-canny-sdxl-1.0");
        write_controlnet(&d, CONTROL_CLASS, false);
        let records = vec![complete(d, ArtifactKind::HfDir)];
        assert!(ControlnetSpec.classify(&records, root.as_path()).is_empty());
    }

    /// Assembling needs BOTH: a control copy alone conditions nothing.
    #[test]
    fn assemble_refuses_without_the_backbone() {
        let chosen = BTreeMap::from([("control".to_string(), 0usize)]);
        assert!(ControlnetSpec.assemble(&chosen, &[], &BTreeMap::new()).is_err());
    }
}
