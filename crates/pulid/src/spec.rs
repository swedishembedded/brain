// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID's rules for the model-store resolver.
//!
//! PuLID is not one checkpoint but five, from four different vendors, and
//! `pulid::caps::Session::new` takes all five: the FLUX.1 backbone, the PuLID
//! adapter itself, ArcFace, EVA-CLIP and BiSeNet. Each is a role here.
//!
//! Only ONE of those roles is classified by this file - the PuLID adapter,
//! the only artifact PuLID itself publishes. The other four are classified by
//! calling the owning architecture's own classifier
//! ([`flux1::spec::classify_pipeline_root`], [`arcface::spec::classify_weights`],
//! [`clip::spec::classify_eva`], [`bisenet::spec::classify_weights`]), so
//! "what does a released ArcFace look like" has exactly one answer in this
//! workspace no matter which architecture is asking. A second copy here would
//! be a second answer, free to drift.
//!
//! Swedish Embedded AB implements composite-model resolution like this for
//! clients whose pipelines assemble several vendors' releases into one served
//! model. If your team needs that assembled deterministically rather than by
//! convention, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// Every role `pulid::caps::Session::new` needs, in the order it takes them.
pub const ROLES: &[&str] = &["flux1", "pulid", "arcface", "clip", "bisenet"];

/// The PuLID encoder's output projection. It is stored as a BARE parameter
/// (no `.weight` suffix), which is itself unusual enough to be a signal, and
/// its `[1024, 2048]` shape is the ID-embedding width the adapter was trained
/// at.
const PROJ_OUT: &str = "pulid_encoder.proj_out";
/// The last of the 20 cross-attention blocks PuLID injects into the FLUX.1
/// backbone - present only at the released adapter's depth.
const LAST_CA_BLOCK: &str = "pulid_ca.19.to_q.weight";

/// Whether `path` is a released PuLID adapter checkpoint, from its own header.
pub fn is_pulid_adapter(path: &Path) -> bool {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(path.to_string_lossy().as_ref()) else { return false };
    m.shape(PROJ_OUT).is_some_and(|s| s == [1024, 2048]) && m.shape(LAST_CA_BLOCK).is_some()
}

/// PuLID's [`ArchSpec`].
pub struct PulidSpec;

impl ArchSpec for PulidSpec {
    fn arch(&self) -> &'static str {
        "pulid"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        // The one artifact PuLID itself publishes.
        for (idx, rec) in records.iter().enumerate() {
            if rec.usable() && rec.kind == ArtifactKind::Safetensors && is_pulid_adapter(&rec.path) {
                out.push((idx, "pulid".to_string(), Confidence::Derived));
            }
        }
        // The four borrowed from other architectures - each asked of the
        // architecture that owns it, never re-derived here.
        flux1::spec::classify_pipeline_root(records, "flux1", &mut out);
        arcface::spec::classify_weights(records, "arcface", &mut out);
        clip::spec::classify_eva(records, "clip", &mut out);
        bisenet::spec::classify_weights(records, "bisenet", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        for role in ROLES {
            chosen.get(*role).ok_or_else(|| format!("pulid assemble: no {role} chosen"))?;
        }
        // PuLID's own served variant follows the BACKBONE's - the adapter is
        // the same file either way, and `dev` vs `schnell` is what actually
        // changes the sampling path. Read through flux1's own accessor rather
        // than re-deriving it, for the same reason `classify` delegates.
        let root = &records[chosen["flux1"]].path;
        let variant = match overrides.get("variant") {
            Some(v) => v.clone(),
            None => flux1::spec::variant_of(root).ok_or_else(|| format!("pulid assemble: {} declares no guidance_embeds, so dev-vs-schnell cannot be read from it", root.display()))?.to_string(),
        };
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/flux1-pulid-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        // Every role must have survived into the assembly - a served PuLID
        // missing any one of the five builds nothing.
        for role in ROLES {
            assembly.roles.get(*role).ok_or_else(|| format!("pulid validate: assembly has no {role} role"))?;
        }
        // The backbone is the one component with a real cross-check: PuLID's
        // adapter injects into FLUX.1's blocks, so a non-FLUX.1 pipeline in
        // that role could never accept it.
        let root = &assembly.roles["flux1"];
        if !flux1::spec::is_flux1_pipeline_root(root) {
            return Err(format!("pulid validate: {} is not a FLUX.1 pipeline root", root.display()));
        }
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
        std::env::temp_dir().join(format!("brain-pulid-spec-{tag}-{}-{n}", std::process::id()))
    }

    fn write_st(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let named: Vec<(String, Vec<u64>, Vec<f32>)> = tensors
            .iter()
            .map(|(name, shape)| ((*name).to_string(), shape.iter().map(|&d| d as u64).collect(), vec![0.0f32; shape.iter().product()]))
            .collect();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &named, &serde_json::json!({}), None).unwrap();
    }

    fn adapter_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![(PROJ_OUT, vec![1024, 2048]), (LAST_CA_BLOCK, vec![2, 2])]
    }

    /// The adapter is identified by its own header, and a differently-shaped
    /// PuLID-like checkpoint (a shallower adapter, missing the 20th block) is
    /// not accepted in its place.
    #[test]
    fn the_released_adapter_is_identified_by_content() {
        let dir = tmp("adapter");
        let real = dir.join("guozinan").join("PuLID").join("pulid_flux_v0.9.1.safetensors");
        write_st(&real, &adapter_tensors());
        assert!(is_pulid_adapter(&real));

        // Content, not name.
        let renamed = dir.join("vendor").join("hand-placed.safetensors");
        write_st(&renamed, &adapter_tensors());
        assert!(is_pulid_adapter(&renamed));

        let shallow = dir.join("vendor").join("pulid_flux_v0.9.1.safetensors");
        write_st(&shallow, &[(PROJ_OUT, vec![1024, 2048]), ("pulid_ca.9.to_q.weight", vec![2, 2])]);
        assert!(!is_pulid_adapter(&shallow), "a shallower adapter must not pass as the released one");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The five roles are exactly what `pulid::caps::Session::new` takes, and
    /// all five are required - an absent one must never be quietly dropped,
    /// because a PuLID missing any component builds nothing.
    #[test]
    fn every_role_the_session_needs_is_declared_and_required() {
        let spec = PulidSpec;
        assert_eq!(spec.roles(), ["flux1", "pulid", "arcface", "clip", "bisenet"]);
        assert!(spec.optional_roles().is_empty(), "no PuLID component is optional");
    }

    /// A non-FLUX.1 pipeline in the backbone role is refused by `validate`,
    /// not discovered at the first generation request.
    #[test]
    fn a_non_flux1_backbone_is_refused() {
        let dir = tmp("backbone");
        let root = dir.join("black-forest-labs").join("FLUX.2-dev");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("model_index.json"),
            serde_json::to_vec(&serde_json::json!({"_class_name": "Flux2Pipeline", "transformer": ["diffusers", "Flux2Transformer2DModel"]})).unwrap(),
        )
        .unwrap();
        let mut roles = BTreeMap::new();
        for role in ROLES {
            roles.insert((*role).to_string(), root.clone());
        }
        let assembly = Assembly { id: "x".to_string(), arch: "pulid".to_string(), variant: None, roles, provenance: Vec::new() };
        let err = PulidSpec.validate(&assembly).unwrap_err();
        assert!(err.contains("not a FLUX.1 pipeline root"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A record that is not usable (an interrupted download) must never be
    /// classified as the adapter.
    #[test]
    fn an_interrupted_download_is_not_classified() {
        let dir = tmp("partial");
        let real = dir.join("guozinan").join("PuLID").join("pulid_flux_v0.9.1.safetensors");
        write_st(&real, &adapter_tensors());
        let records = vec![ArtifactRecord {
            path: real.clone(),
            size: 1,
            mtime_ns: 0,
            kind: ArtifactKind::Safetensors,
            completeness: Completeness::Partial { final_path: real.clone() },
        }];
        let out = PulidSpec.classify(&records, &dir);
        assert!(out.is_empty(), "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
