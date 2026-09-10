// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1's rules for the model-store resolver: which artifact on disk is a
//! released FLUX.1 pipeline, and which FLUX.1 it is.
//!
//! One role, `root` - a diffusers pipeline DIRECTORY holding `transformer/`,
//! `vae/`, `text_encoder/`, `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`,
//! exactly what `flux1::caps::Session::new` takes. Every decision here is made
//! from real file content (`model_index.json`'s own `_class_name`, the
//! transformer's own `config.json`), never from a directory's name: a store
//! holds several diffusers pipelines that are indistinguishable by name and
//! completely different models (`FluxPipeline` vs `Flux2Pipeline` vs
//! `WanPipeline` vs `ZImagePipeline` all ship the identical file layout).
//!
//! Swedish Embedded AB implements content-addressed model identification like
//! this for clients whose stores mix vendor releases, re-quantizations and
//! hand-placed checkpoints. If your team needs weight resolution that never
//! silently guesses, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role name this architecture resolves - the pipeline root directory.
pub const ROLES: &[&str] = &["root"];

/// The `_class_name` a released FLUX.1 pipeline's own `model_index.json`
/// declares. FLUX.2 declares `Flux2Pipeline`/`Flux2KleinPipeline` and Z-Image
/// `ZImagePipeline`, so this one string separates them with no shape work.
const PIPELINE_CLASS: &str = "FluxPipeline";
/// The transformer class the same manifest names for its `transformer` entry -
/// checked as well as [`PIPELINE_CLASS`] so a re-packaged pipeline that kept
/// the class name but swapped the backbone cannot pass.
const TRANSFORMER_CLASS: &str = "FluxTransformer2DModel";

/// Read `dir/model_index.json`, the diffusers pipeline manifest.
fn pipeline_manifest(dir: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(dir.join("model_index.json")).ok()?).ok()
}

/// A component entry in a `model_index.json` is a `[library, class]` pair;
/// this is the class half.
fn component_class(manifest: &serde_json::Value, component: &str) -> Option<String> {
    manifest.get(component)?.as_array()?.get(1)?.as_str().map(str::to_string)
}

/// Whether `dir` is a released FLUX.1 pipeline root - the shared predicate,
/// so [`Flux1Spec::classify`] and every other architecture that builds ON a
/// FLUX.1 backbone (`pulid::spec`, whose `flux1` role is this same artifact)
/// ask the identical question of the identical bytes instead of each carrying
/// its own idea of what FLUX.1 looks like.
pub fn is_flux1_pipeline_root(dir: &Path) -> bool {
    let Some(manifest) = pipeline_manifest(dir) else { return false };
    manifest.get("_class_name").and_then(serde_json::Value::as_str) == Some(PIPELINE_CLASS)
        && component_class(&manifest, "transformer").as_deref() == Some(TRANSFORMER_CLASS)
}

/// Classify every FLUX.1 pipeline root in `records` under `role`, appending to
/// `out` in place - the shape every `classify_*` helper in this workspace
/// takes. Exposed so `pulid::spec` reuses it for its own `flux1` role.
pub fn classify_pipeline_root(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::PipelineDir && is_flux1_pipeline_root(&rec.path) {
            // `Declared`: the manifest names the class outright, with no
            // further disambiguation needed.
            out.push((idx, role.to_string(), Confidence::Declared));
        }
    }
}

/// `dev` or `schnell`, from the transformer's own declared config rather than
/// from the directory's name.
///
/// The guidance-distilled `-dev` release carries a guidance embedder
/// (`guidance_embeds: true`) that `schnell` does not - a real, header-only
/// difference in the component's own `config.json`, so this never has to
/// trust `FLUX.1-dev` appearing in a path an operator chose.
pub fn variant_of(root: &Path) -> Option<&'static str> {
    let bytes = std::fs::read(root.join("transformer").join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    match cfg.get("guidance_embeds").and_then(serde_json::Value::as_bool) {
        Some(true) => Some("dev"),
        Some(false) => Some("schnell"),
        None => None,
    }
}

/// FLUX.1's [`ArchSpec`].
pub struct Flux1Spec;

impl ArchSpec for Flux1Spec {
    fn arch(&self) -> &'static str {
        "flux1"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        classify_pipeline_root(records, "root", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let root = &records[*chosen.get("root").ok_or("flux1 assemble: no root chosen")?].path;
        // An explicit `--variant`/`variant` override wins, as everywhere else;
        // otherwise the transformer's own config decides. Unlike FLUX.2's
        // klein-vs-base (which no weight can ever answer, hence its
        // `UnresolvedVariant`), dev-vs-schnell IS recoverable from content, so
        // this never has to ask.
        let variant = match overrides.get("variant") {
            Some(v) => v.clone(),
            None => variant_of(root).ok_or_else(|| format!("flux1 assemble: {} declares no guidance_embeds, so dev-vs-schnell cannot be read from it", root.display()))?.to_string(),
        };
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/flux1-{variant}"), variant: Some(variant) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let root = assembly.roles.get("root").ok_or("flux1 validate: assembly has no root role")?;
        // Every component `flux1::caps::Session` will open, checked before any
        // of them is opened - a pipeline missing one of these builds nothing,
        // and finding that out at the first generation request rather than at
        // resolve time is exactly what this gate exists to avoid.
        for component in ["transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] {
            if !root.join(component).is_dir() {
                return Err(format!("flux1 validate: {} holds no {component}/", root.display()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-flux1-spec-{tag}-{}-{n}", std::process::id()))
    }

    /// A pipeline root that looks exactly like a real release on disk:
    /// `model_index.json` plus every component directory, and a transformer
    /// config declaring `guidance_embeds`.
    fn write_pipeline(root: &Path, class: &str, transformer_class: &str, guidance: bool) {
        std::fs::create_dir_all(root).unwrap();
        let manifest = serde_json::json!({
            "_class_name": class,
            "transformer": ["diffusers", transformer_class],
            "vae": ["diffusers", "AutoencoderKL"],
        });
        std::fs::write(root.join("model_index.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        for c in ["transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] {
            std::fs::create_dir_all(root.join(c)).unwrap();
        }
        std::fs::write(root.join("transformer").join("config.json"), serde_json::to_vec(&serde_json::json!({"guidance_embeds": guidance})).unwrap()).unwrap();
    }

    fn dir_record(path: &Path) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind: ArtifactKind::PipelineDir, completeness: Completeness::Complete }
    }

    /// The whole point: one released FLUX.1 pipeline in a store that also
    /// holds other diffusers pipelines resolves on its own, with nothing
    /// named - and resolves to the DIRECTORY, which is what
    /// `flux1::caps::Session::new` takes.
    #[test]
    fn a_lone_flux1_pipeline_resolves_with_nothing_named() {
        let dir = tmp("lone");
        let flux1 = dir.join("black-forest-labs").join("FLUX.1-dev");
        write_pipeline(&flux1, "FluxPipeline", "FluxTransformer2DModel", true);
        // Real neighbours from the same store: neither may be mistaken for it.
        let flux2 = dir.join("black-forest-labs").join("FLUX.2-dev");
        write_pipeline(&flux2, "Flux2Pipeline", "Flux2Transformer2DModel", false);
        let wan = dir.join("Wan-AI").join("Wan2.1-T2V-1.3B-Diffusers");
        write_pipeline(&wan, "WanPipeline", "WanTransformer3DModel", false);

        let records = vec![dir_record(&flux1), dir_record(&flux2), dir_record(&wan)];
        let specs: Vec<&dyn ArchSpec> = vec![&Flux1Spec];
        match resolve("flux1", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["root"], flux1, "must resolve to the pipeline directory itself");
                assert_eq!(a.variant.as_deref(), Some("dev"), "guidance_embeds: true is the -dev release");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `dev` vs `schnell` comes from the transformer's own declared config,
    /// never from the path - a `schnell` release sitting in a directory an
    /// operator named `FLUX.1-dev` must still resolve as `schnell`.
    #[test]
    fn the_variant_is_read_from_the_transformer_config_not_the_directory_name() {
        let dir = tmp("variant");
        let misnamed = dir.join("black-forest-labs").join("FLUX.1-dev");
        write_pipeline(&misnamed, "FluxPipeline", "FluxTransformer2DModel", false);
        let records = vec![dir_record(&misnamed)];
        let specs: Vec<&dyn ArchSpec> = vec![&Flux1Spec];
        match resolve("flux1", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.variant.as_deref(), Some("schnell")),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two real FLUX.1 releases in one store is a genuine question only the
    /// operator can answer - it must be reported with both candidates, never
    /// silently resolved to whichever the walk happened to reach first.
    #[test]
    fn two_flux1_pipelines_are_ambiguous_with_both_named() {
        let dir = tmp("ambiguous");
        let a = dir.join("black-forest-labs").join("FLUX.1-dev");
        let b = dir.join("krea").join("FLUX.1-Krea-dev");
        write_pipeline(&a, "FluxPipeline", "FluxTransformer2DModel", true);
        write_pipeline(&b, "FluxPipeline", "FluxTransformer2DModel", true);
        let records = vec![dir_record(&a), dir_record(&b)];
        let specs: Vec<&dyn ArchSpec> = vec![&Flux1Spec];
        match resolve("flux1", &records, &specs, &BTreeMap::new()) {
            Resolution::Ambiguous(amb) => {
                assert_eq!(amb.choices.len(), 2, "{amb:?}");
                let named: Vec<String> = amb.choices.iter().flat_map(|c| c.selector.iter().map(|(_, v)| v.clone())).collect();
                assert!(named.iter().any(|v| v.contains("FLUX.1-dev")), "{named:?}");
                assert!(named.iter().any(|v| v.contains("FLUX.1-Krea-dev")), "{named:?}");
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A pipeline root missing a component `flux1::caps::Session` will open is
    /// refused at resolve time by `validate`, not at the first generation
    /// request.
    #[test]
    fn a_pipeline_missing_a_component_does_not_resolve() {
        let dir = tmp("incomplete");
        let root = dir.join("black-forest-labs").join("FLUX.1-dev");
        write_pipeline(&root, "FluxPipeline", "FluxTransformer2DModel", true);
        std::fs::remove_dir_all(root.join("text_encoder_2")).unwrap();
        let records = vec![dir_record(&root)];
        let specs: Vec<&dyn ArchSpec> = vec![&Flux1Spec];
        match resolve("flux1", &records, &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => assert!(m.roles[0].doc.contains("text_encoder_2"), "{m:?}"),
            other => panic!("expected Missing, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
