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
    vec![Box::new(TransformersRecipe)]
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
}
