// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pluggable per-family fetch policy.
//!
//! [`plan::plan_base`](crate::plan) used to hardcode one family's shape
//! inline: require a root `config.json`, gate on a recognized
//! `architectures[0]`, download a fixed transformers-shaped file set. That
//! policy is now just the first, catch-all [`ArtifactRecipe`]
//! ([`TransformersRecipe`]) in an ordered registry ([`recipes`]) `plan_base`
//! consults instead. Adding a family (a diffusers-style pipeline, a flat
//! `.pt` release, eventually something p2p-sourced) means adding a recipe
//! here -- everything below it (`Hub`, `Store`, `fetch::stream_to_file`, the
//! single-flight supplier in `crates/cli/src/supply.rs`) is unchanged by
//! that, which is the whole point: this crate stays a leaf (no model-crate
//! deps), so "how do I make what I downloaded servable" stays out of scope
//! here and lives in `crates/cli/src/supply.rs::convert`, keyed by
//! [`ArtifactRecipe::id`] carried on the resulting `Step::Convert`.

use brain_modelref::ModelRef;

use crate::hub::Hub;
use crate::plan::{declared_architecture, is_supported_architecture, PlanError, REVISION};

/// One upstream file to fetch, and the name it lands under in the repo's
/// store directory.
pub struct Artifact {
    pub file: String,
    pub dest_name: String,
}

pub fn artifact(file: impl Into<String>, dest_name: impl Into<String>) -> Artifact {
    Artifact { file: file.into(), dest_name: dest_name.into() }
}

/// A family's fetch policy: does a repo's file listing look like mine, and if
/// so what needs downloading? Never "how do I make it servable" -- see the
/// module docs for why that stays out of this crate.
pub trait ArtifactRecipe: Send + Sync {
    /// Tag carried on the resulting `Step::Convert { recipe, .. }` so
    /// `crates/cli/src/supply.rs::convert` knows which family it's finishing
    /// without re-deriving it from disk a second time.
    fn id(&self) -> &'static str;
    /// Cheap, listing-only pre-filter -- no network beyond the one
    /// `list_files` call `plan_base` already made. Most repos are ruled out
    /// instantly (a flat Ultralytics-shaped repo has no `config.json`; a
    /// diffusers-shaped repo has no top-level `.pt` files).
    fn matches_listing(&self, listing: &[String]) -> bool;
    /// Full detection + the ordered artifact list to download. Only called
    /// for a repo [`matches_listing`](Self::matches_listing) accepted; may do
    /// extra [`Hub::read_file`] calls (e.g. reading and validating
    /// `config.json`'s architecture). Boxed error: `PlanError` carries a
    /// `ModelRef` in most variants (clippy's `result_large_err`), and this is
    /// the one signature in the crate that returns it from a `dyn` trait
    /// method rather than a concrete function already accounted for.
    fn artifacts(&self, reference: &ModelRef, listing: &[String], hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>>;
}

/// The registry `plan_base` walks, in order. [`TransformersRecipe`] is last
/// and always matches (the historical, still-default family) -- more
/// specific recipes get first refusal, ahead of it.
pub fn recipes() -> Vec<Box<dyn ArtifactRecipe>> {
    vec![Box::new(ZimageRecipe), Box::new(YoloRecipe), Box::new(TransformersRecipe)]
}

/// An Ultralytics-shaped release repo (`Ultralytics/YOLOv8`, confirmed live
/// via the HF API: root-level `yolov8n.pt`…`yolov8x.pt`, no `config.json`) --
/// the exact same files their GitHub-releases mirror serves. Downloads
/// exactly one file (the nano variant if present, else whichever `.pt` sorts
/// first, so a differently-curated repo still resolves to something); the
/// finish step (`crates/cli/src/supply.rs::convert`, `"yolo"` arm) runs
/// `yolov8::import::import_yolov8n` to produce `model.brain.safetensors` --
/// this recipe fits the store's existing single-file convention, unlike
/// [`ZimageRecipe`], so no compound-manifest machinery is needed here.
pub struct YoloRecipe;

impl YoloRecipe {
    /// The preferred variant when the repo offers more than one size.
    const PREFERRED: &'static str = "yolov8n.pt";
}

impl ArtifactRecipe for YoloRecipe {
    fn id(&self) -> &'static str {
        "yolo"
    }

    fn matches_listing(&self, listing: &[String]) -> bool {
        !listing.iter().any(|f| f == "config.json" || f == "model_index.json")
            && listing.iter().any(|f| f.starts_with("yolov8") && f.ends_with(".pt"))
    }

    fn artifacts(&self, reference: &ModelRef, listing: &[String], _hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>> {
        let mut candidates: Vec<&String> = listing.iter().filter(|f| f.starts_with("yolov8") && f.ends_with(".pt")).collect();
        candidates.sort();
        let file = candidates
            .iter()
            .find(|f| f.as_str() == Self::PREFERRED)
            .or_else(|| candidates.first())
            .ok_or_else(|| Box::new(PlanError::NoUpstreamArtifact(reference.clone(), "no yolov8*.pt file in repo".to_string())))?;
        Ok(vec![artifact((*file).clone(), (*file).clone())])
    }
}

/// A Z-Image-shaped diffusers pipeline repo (`Tongyi-MAI/Z-Image-Turbo` and
/// `Tongyi-MAI/Z-Image`, confirmed live via the HF API): `model_index.json`
/// at the root (no root `config.json`, so [`TransformersRecipe`] would
/// otherwise reject it) plus four role subdirectories. No tensor rewrite is
/// needed to serve one -- `s3dit::import::import_comfy` already remaps
/// tensor names in memory at load time -- so this recipe's "artifacts" are
/// just "download every file under the four role dirs plus the pipeline
/// manifest"; the finish step (`crates/cli/src/supply.rs::convert`, `"zimage"`
/// arm) writes a [`crate::CompoundManifest`] from [`ZimageRecipe::ROLES`]
/// rather than converting anything.
pub struct ZimageRecipe;

impl ZimageRecipe {
    /// Role name -> the path (relative to the repo dir) that role's loader
    /// accepts: a directory for the two sharded components
    /// (`s3dit::pipeline::read_component_tensors` is dir-or-file-aware), a
    /// specific file for the two that are always exactly one file upstream.
    /// The single source of truth for z-image's role layout -- this recipe's
    /// [`matches_listing`](ArtifactRecipe::matches_listing) probes it, and
    /// the finish-side manifest writer in `crates/cli/src/supply.rs` reuses
    /// it verbatim rather than re-deriving the same mapping.
    pub const ROLES: &'static [(&'static str, &'static str)] =
        &[("dit", "transformer"), ("vae", "vae/diffusion_pytorch_model.safetensors"), ("text_encoder", "text_encoder"), ("tokenizer", "tokenizer/tokenizer.json")];

    /// The four subdirectory prefixes [`artifacts`](ArtifactRecipe::artifacts)
    /// downloads every file under.
    const ROLE_DIRS: &'static [&'static str] = &["transformer/", "vae/", "text_encoder/", "tokenizer/"];
}

impl ArtifactRecipe for ZimageRecipe {
    fn id(&self) -> &'static str {
        "zimage"
    }

    fn matches_listing(&self, listing: &[String]) -> bool {
        listing.iter().any(|f| f == "model_index.json") && Self::ROLE_DIRS.iter().all(|prefix| listing.iter().any(|f| f.starts_with(prefix)))
    }

    fn artifacts(&self, _reference: &ModelRef, listing: &[String], _hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>> {
        let mut artifacts = vec![artifact("model_index.json", "model_index.json")];
        for f in listing {
            if Self::ROLE_DIRS.iter().any(|prefix| f.starts_with(prefix)) {
                artifacts.push(artifact(f.clone(), f.clone()));
            }
        }
        Ok(artifacts)
    }
}

/// The original (and still catch-all/default) family this crate ever
/// supported: an HF `transformers`-shaped repo -- `config.json` with a
/// recognized architecture, then single-file or sharded safetensors weights.
/// Reproduces `plan_base`'s pre-recipe logic exactly, so every existing
/// qwen/glm/lfm/gpt behavior (including exact error text) is unchanged by
/// this becoming pluggable.
pub struct TransformersRecipe;

impl ArtifactRecipe for TransformersRecipe {
    fn id(&self) -> &'static str {
        "transformers"
    }

    fn matches_listing(&self, _listing: &[String]) -> bool {
        // Catch-all: always tried, always last (see `recipes()`). Its own
        // artifacts() below produces the specific "no config.json"/
        // "unsupported architecture" errors when nothing more specific
        // claimed the repo first.
        true
    }

    fn artifacts(&self, reference: &ModelRef, listing: &[String], hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>> {
        let vendor = reference.vendor();
        let repo = reference.repo();

        // Cheap metadata before expensive bytes: fetch config.json (~KBs) and
        // gate on architecture before a single weight byte is requested.
        if !listing.iter().any(|f| f == "config.json") {
            return Err(Box::new(PlanError::NoUpstreamArtifact(reference.clone(), "no config.json in repo".to_string())));
        }
        let config_bytes = hub.read_file(vendor, repo, REVISION, "config.json").map_err(|e| Box::new(PlanError::Hub(e)))?;
        let config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| Box::new(PlanError::NoUpstreamArtifact(reference.clone(), format!("unparseable config.json: {e}"))))?;
        let arch =
            declared_architecture(&config).ok_or_else(|| Box::new(PlanError::NoUpstreamArtifact(reference.clone(), "config.json has no architecture".to_string())))?;
        if !is_supported_architecture(&arch) {
            return Err(Box::new(PlanError::UnsupportedArchitecture(reference.clone(), arch)));
        }

        let mut artifacts = vec![artifact("config.json", "config.json")];
        if listing.iter().any(|f| f == "tokenizer.json") {
            artifacts.push(artifact("tokenizer.json", "tokenizer.json"));
        }
        if listing.iter().any(|f| f == "tokenizer_config.json") {
            artifacts.push(artifact("tokenizer_config.json", "tokenizer_config.json"));
        }

        if listing.iter().any(|f| f == "model.safetensors") {
            artifacts.push(artifact("model.safetensors", "model.safetensors"));
        } else {
            let mut shards: Vec<&String> = listing.iter().filter(|f| f.starts_with("model-") && f.ends_with(".safetensors")).collect();
            if shards.is_empty() {
                return Err(Box::new(PlanError::NoUpstreamArtifact(reference.clone(), "no safetensors weights found (single file or shard set)".to_string())));
            }
            shards.sort();
            if listing.iter().any(|f| f == "model.safetensors.index.json") {
                artifacts.push(artifact("model.safetensors.index.json", "model.safetensors.index.json"));
            }
            for shard in shards {
                artifacts.push(artifact(shard.clone(), shard.clone()));
            }
        }

        Ok(artifacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_ends_in_a_catch_all() {
        let all = recipes();
        assert!(!all.is_empty());
        let last = all.last().unwrap();
        assert_eq!(last.id(), "transformers");
        assert!(last.matches_listing(&[]), "the catch-all must match even an empty listing");
        assert!(last.matches_listing(&["totally-unrelated-file.bin".to_string()]));
    }

    /// The exact `Tongyi-MAI/Z-Image-Turbo` file listing, confirmed live via
    /// the HF API this session -- not a guessed shape.
    fn zimage_turbo_listing() -> Vec<String> {
        [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn zimage_recipe_matches_a_diffusers_pipeline_repo_ahead_of_transformers() {
        let listing = zimage_turbo_listing();
        let matched = recipes().into_iter().find(|r| r.matches_listing(&listing)).unwrap();
        assert_eq!(matched.id(), "zimage", "a diffusers-pipeline repo must not fall through to the transformers catch-all");
    }

    #[test]
    fn zimage_recipe_does_not_match_a_transformers_repo() {
        let listing = vec!["config.json".to_string(), "model.safetensors".to_string()];
        assert!(!ZimageRecipe.matches_listing(&listing));
    }

    #[test]
    fn zimage_recipe_downloads_every_file_under_its_four_role_dirs_plus_the_manifest() {
        let listing = zimage_turbo_listing();
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None);
        let artifacts = ZimageRecipe.artifacts(&r, &listing, &hub).unwrap();
        let dest: Vec<&str> = artifacts.iter().map(|a| a.dest_name.as_str()).collect();

        // Every file under one of the four role dirs lands, none renamed
        // (dest_name == file), and the pipeline manifest is included even
        // though it's not under a role dir.
        for f in listing.iter().filter(|f| f.as_str() != "model_index.json") {
            let under_a_role_dir = ZimageRecipe::ROLE_DIRS.iter().any(|prefix| f.starts_with(prefix));
            assert_eq!(dest.contains(&f.as_str()), under_a_role_dir, "{f} inclusion disagrees with whether it's under a role dir");
        }
        assert!(dest.contains(&"model_index.json"));
        // `scheduler/scheduler_config.json` is real upstream content but not
        // one of the four roles this recipe's loader needs -- correctly
        // skipped, not an oversight (z-image reimplements its own flow-match
        // scheduler; it doesn't read the diffusers scheduler config).
        assert!(!dest.contains(&"scheduler/scheduler_config.json"));
    }

    #[test]
    fn zimage_recipe_roles_cover_every_role_dir_it_downloads_from() {
        for (_, rel) in ZimageRecipe::ROLES {
            // ROLES paths are extension-less directory names ("transformer")
            // or a specific file under one ("vae/…"); ROLE_DIRS are
            // trailing-slash prefixes matched against listing entries
            // ("transformer/") -- so a role IS covered when it equals the
            // bare prefix (minus the slash) or starts with the full prefix.
            let covered = ZimageRecipe::ROLE_DIRS.iter().any(|prefix| rel.starts_with(prefix) || *rel == prefix.trim_end_matches('/'));
            assert!(covered, "role path {rel:?} is not under any of ROLE_DIRS -- the manifest writer and the downloader would disagree");
        }
    }

    /// The exact `Ultralytics/YOLOv8` file listing, confirmed live via the HF
    /// API this session -- not a guessed shape.
    fn ultralytics_yolov8_listing() -> Vec<String> {
        [".gitattributes", "README.md", "yolov8l.pt", "yolov8m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8x.pt"].into_iter().map(String::from).collect()
    }

    #[test]
    fn yolo_recipe_matches_the_flat_release_repo_ahead_of_transformers_and_zimage() {
        let listing = ultralytics_yolov8_listing();
        let matched = recipes().into_iter().find(|r| r.matches_listing(&listing)).unwrap();
        assert_eq!(matched.id(), "yolo");
    }

    #[test]
    fn yolo_recipe_does_not_match_transformers_or_zimage_shaped_repos() {
        assert!(!YoloRecipe.matches_listing(&["config.json".to_string(), "model.safetensors".to_string()]));
        assert!(!YoloRecipe.matches_listing(&zimage_turbo_listing()));
    }

    #[test]
    fn yolo_recipe_downloads_only_the_nano_variant_when_present() {
        let listing = ultralytics_yolov8_listing();
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("Ultralytics", "YOLOv8", None);
        let artifacts = YoloRecipe.artifacts(&r, &listing, &hub).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].file, "yolov8n.pt");
        assert_eq!(artifacts[0].dest_name, "yolov8n.pt");
    }

    #[test]
    fn yolo_recipe_falls_back_to_the_first_variant_when_nano_is_absent() {
        let listing = vec!["yolov8x.pt".to_string(), "yolov8l.pt".to_string()];
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("Ultralytics", "YOLOv8", None);
        let artifacts = YoloRecipe.artifacts(&r, &listing, &hub).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].file, "yolov8l.pt", "sorted first among the available variants");
    }
}
