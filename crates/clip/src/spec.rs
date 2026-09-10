// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP's rules for the model-store resolver.
//!
//! Two roles, because `crates/clip` really does serve two independently
//! released things:
//! - `towers` - an SDXL-layout checkpoint root (`text_encoder/`,
//!   `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`), what
//!   `clip::caps::Session::load` takes and where the CLIP-L and OpenCLIP-bigG
//!   text towers come from;
//! - `eva` - the separately released EVA-CLIP-L/336 vision checkpoint
//!   ([`crate::caps::EVA_FILE`]), a single torch `.pt` that ships on its own
//!   and is what `pulid`'s identity conditioning actually consumes. Optional:
//!   the text towers serve `embed_text` perfectly well without it.
//!
//! Both are identified from real content - a pipeline manifest's own declared
//! `_class_name`, and the EVA tower's own tensor names - never from a path.
//!
//! Swedish Embedded AB implements model-store resolution for clients whose
//! encoders arrive from several vendors in several layouts. If your team needs
//! the same, you can procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The roles this architecture resolves.
pub const ROLES: &[&str] = &["towers", "eva"];

/// The `_class_name` an SDXL pipeline root declares. Checked exactly: a
/// FLUX.1 root also carries `text_encoder/` + `tokenizer/` + `tokenizer_2/`,
/// but its `text_encoder_2` is a T5 encoder rather than OpenCLIP-bigG, so
/// serving it as this architecture's `towers` would build one working tower
/// and one broken one.
const SDXL_PIPELINE_CLASS: &str = "StableDiffusionXLPipeline";

/// EVA-02's rotary position embedding buffer. Plain CLIP and OpenCLIP have no
/// RoPE at all, so this single tensor name separates an EVA-CLIP checkpoint
/// from every other CLIP-shaped `.pt` in a store.
const EVA_ROPE: &str = "visual.rope.freqs_cos";
/// EVA-02's SwiGLU third projection - present only in the EVA block's MLP,
/// checked alongside the RoPE buffer so the match rests on the block
/// structure rather than on one buffer.
const EVA_SWIGLU: &str = "visual.blocks.0.mlp.w3.weight";

/// Whether `dir` is an SDXL-layout checkpoint root, from its own pipeline
/// manifest and the component directories the session will open.
pub fn is_sdxl_tower_root(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("model_index.json")) else { return false };
    let Ok(manifest) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    if manifest.get("_class_name").and_then(serde_json::Value::as_str) != Some(SDXL_PIPELINE_CLASS) {
        return false;
    }
    ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"].iter().all(|c| dir.join(c).is_dir())
}

/// Whether `path` is a released EVA-CLIP vision checkpoint, from its own
/// tensor names (header-only: `torchpt::read_shapes` reads the pickle's
/// structure, never a storage entry).
pub fn is_eva_checkpoint(path: &Path) -> bool {
    let Ok(shapes) = checkpoint::torchpt::read_shapes(path.to_string_lossy().as_ref()) else { return false };
    shapes.iter().any(|(n, _)| n == EVA_ROPE) && shapes.iter().any(|(n, _)| n == EVA_SWIGLU)
}

/// Classify every EVA-CLIP checkpoint in `records` under `role`, appending to
/// `out` in place. Exposed so `pulid::spec` reuses it for its own `clip` role
/// instead of carrying a second idea of what EVA-CLIP is.
pub fn classify_eva(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::Torch && is_eva_checkpoint(&rec.path) {
            // `Derived`: a torch checkpoint declares no architecture, so this
            // is computed from the tensor names its own pickle reports.
            out.push((idx, role.to_string(), Confidence::Derived));
        }
    }
}

/// CLIP's [`ArchSpec`].
pub struct ClipSpec;

impl ArchSpec for ClipSpec {
    fn arch(&self) -> &'static str {
        "clip"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn optional_roles(&self) -> &'static [&'static str] {
        // The text towers serve both of this architecture's actions on their
        // own; the EVA image tower is a genuinely separate release that a
        // store may simply not hold.
        &["eva"]
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if rec.usable() && rec.kind == ArtifactKind::PipelineDir && is_sdxl_tower_root(&rec.path) {
                out.push((idx, "towers".to_string(), Confidence::Declared));
            }
        }
        classify_eva(records, "eva", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("towers").ok_or("clip assemble: no towers chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/clip-sdxl".to_string(), variant: Some("sdxl".to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let towers = assembly.roles.get("towers").ok_or("clip validate: assembly has no towers role")?;
        if !is_sdxl_tower_root(towers) {
            return Err(format!("clip validate: {} is not an SDXL-layout checkpoint root", towers.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-clip-spec-{tag}-{}-{n}", std::process::id()))
    }

    fn write_pipeline(root: &Path, class: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("model_index.json"), serde_json::to_vec(&serde_json::json!({"_class_name": class})).unwrap()).unwrap();
        for c in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] {
            std::fs::create_dir_all(root.join(c)).unwrap();
        }
    }

    /// The distinction this spec exists to make: an SDXL root is the text
    /// towers' home, and a FLUX.1 root - which carries the same four
    /// component directories - is NOT, because its `text_encoder_2` is a T5
    /// encoder rather than OpenCLIP-bigG.
    #[test]
    fn a_flux1_root_is_not_accepted_as_the_sdxl_tower_root() {
        let dir = tmp("towers");
        let sdxl = dir.join("stabilityai").join("stable-diffusion-xl-base-1.0");
        write_pipeline(&sdxl, SDXL_PIPELINE_CLASS);
        let flux1 = dir.join("black-forest-labs").join("FLUX.1-dev");
        write_pipeline(&flux1, "FluxPipeline");

        assert!(is_sdxl_tower_root(&sdxl));
        assert!(!is_sdxl_tower_root(&flux1), "FLUX.1's text_encoder_2 is T5, not OpenCLIP-bigG");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An SDXL root missing a component the session will open does not
    /// classify - the manifest alone is not enough.
    #[test]
    fn an_incomplete_sdxl_root_does_not_classify() {
        let dir = tmp("incomplete");
        let sdxl = dir.join("stabilityai").join("sdxl");
        write_pipeline(&sdxl, SDXL_PIPELINE_CLASS);
        std::fs::remove_dir_all(sdxl.join("tokenizer_2")).unwrap();
        assert!(!is_sdxl_tower_root(&sdxl));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file that is not a torch checkpoint at all must be a clean `false`,
    /// never a panic - `classify` runs over every artifact in a store this
    /// code does not control.
    #[test]
    fn a_non_torch_file_is_not_an_eva_checkpoint() {
        let dir = tmp("junk");
        std::fs::create_dir_all(&dir).unwrap();
        let junk = dir.join("not-really.pt");
        std::fs::write(&junk, b"definitely not a torch zip container").unwrap();
        assert!(!is_eva_checkpoint(&junk));
        std::fs::remove_dir_all(&dir).ok();
    }
}
