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
    let mut v: Vec<Box<dyn ArtifactRecipe>> = vec![Box::new(ZimageRecipe), Box::new(WanRecipe), Box::new(YoloRecipe)];
    v.extend(FILES_RECIPES.iter().map(|r| Box::new(*r) as Box<dyn ArtifactRecipe>));
    v.push(Box::new(TransformersRecipe));
    v
}

/// A repo servable by downloading a small, fixed set of named files verbatim
/// -- no tensor rewrite, no `config.json`/architecture gate -- and exposing
/// them as a [`crate::CompoundManifest`]'s roles. This is [`ZimageRecipe`]'s
/// shape generalised past Z-Image's own four roles: most of the small
/// vision/audio checkpoints this store fetches are "grab N release files,
/// point an env var at one of them (or at the directory holding all of
/// them)", not "convert a tensor format".
///
/// `matches_listing` keys on one distinctive filename rather than the whole
/// shape (unlike [`YoloRecipe`]'s `yolov8*.pt` glob) because these are each
/// one specific upstream repo's own release artifact -- a name unlikely
/// enough that a shape-only match is not a meaningful risk, and specific
/// enough that ordering [`recipes`] ahead of [`TransformersRecipe`] cannot
/// accidentally swallow an unrelated transformers-shaped repo.
#[derive(Clone, Copy)]
pub struct FilesRecipe {
    id: &'static str,
    /// The `resident_for`-style family tag written into the manifest. Every
    /// current `FilesRecipe` row names a family with no model-dir-scan
    /// dispatch arm (`resident_for_compound` logs and skips it, harmlessly --
    /// these models are served through their own `BRAIN_*_WEIGHTS`/`_DIR` env
    /// var, not the generic scan), so this is documentation more than a live
    /// dispatch key today.
    family: &'static str,
    /// Filenames that must ALL be present in a repo's listing to claim it for
    /// this recipe. More than one entry is for repos with no single
    /// distinctively-named file (`Qwen3-ASR-1.7B` ships nothing but ordinary
    /// `transformers`-repo names -- `config.json`, `vocab.json`, `merges.txt`
    /// -- individually far too common to key on alone; the exact SET of
    /// them, keyed together, is specific enough).
    signature: &'static [&'static str],
    /// Files downloaded verbatim from the repo root, each landing under its
    /// own basename. Empty means "every file in the repo's listing" (nested
    /// paths preserved as their own `dest_name`, same as `ZimageRecipe`
    /// already does for its four role directories) -- for a repo whose
    /// consumer needs more than `TransformersRecipe`'s curated
    /// config/tokenizer/weights subset (`qwen3tts`'s nested
    /// `speech_tokenizer/` codec checkpoint, `vocab.json`/`merges.txt` with
    /// no unified `tokenizer.json`).
    files: &'static [&'static str],
    /// Role name -> path relative to the repo dir: one of `files` verbatim,
    /// or `"."` for the whole directory (when a model's env var wants a
    /// directory, not a single file -- `deepseek2ocr`'s two-GGUF pair).
    roles: &'static [(&'static str, &'static str)],
}

/// Every [`FilesRecipe`] this store knows, keyed by [`FilesRecipe::id`] --
/// the single source of truth both [`recipes`] (for planning) and
/// `crates/cli/src/supply.rs::convert_files` (for the finish-side manifest
/// write, via [`files_recipe`]) read.
const FILES_RECIPES: &[FilesRecipe] = &[
    FilesRecipe {
        id: "sam2",
        family: "sam2",
        signature: &["sam2.1_hiera_tiny.pt"],
        files: &["sam2.1_hiera_tiny.pt"],
        roles: &[("weights", "sam2.1_hiera_tiny.pt")],
    },
    FilesRecipe {
        id: "rrdbnet",
        family: "rrdbnet",
        // NOT `ai-forever/Real-ESRGAN`'s `RealESRGAN_x4.pth` -- same file
        // size (67 040 989 bytes) as the real release but a different
        // checksum, and `rrdbnet::import::read`/`validate` reject it
        // (confirmed by running it for real: `brain rrdbnet upscale` fails
        // with "set BRAIN_ESRGAN_WEIGHTS to an existing RealESRGAN_x4plus.pth"
        // even with the var pointed straight at the downloaded file). This
        // repo's copy loads and upscales correctly, verified the same way.
        signature: &["RealESRGAN_x4plus.pth"],
        files: &["RealESRGAN_x4plus.pth"],
        roles: &[("weights", "RealESRGAN_x4plus.pth")],
    },
    FilesRecipe {
        id: "deepseek2ocr-gguf",
        family: "deepseek2ocr",
        signature: &["DeepSeek-OCR-Q8_0.gguf"],
        files: &["DeepSeek-OCR-Q8_0.gguf", "mmproj-DeepSeek-OCR-Q8_0.gguf"],
        roles: &[("dir", ".")],
    },
    FilesRecipe {
        id: "qwen3tts",
        family: "qwen3tts",
        // Nested and distinctive enough that no other repo shape could match
        // it by accident (unlike a bare `config.json`, which every
        // transformers-shaped repo has).
        signature: &["speech_tokenizer/config.json"],
        // Whole repo: `TransformersRecipe`'s curated fetch (config.json,
        // tokenizer.json/tokenizer_config.json, model.safetensors) misses
        // the nested `speech_tokenizer/` codec checkpoint entirely, plus
        // `vocab.json`/`merges.txt` this repo needs since it ships no
        // unified `tokenizer.json`. `roles` is empty here on purpose --
        // `crates/cli/src/supply.rs::convert` special-cases this recipe id
        // to `convert_qwen3tts`, which runs the real Talker/MTP/codec/
        // speaker conversion and writes its OWN two-role manifest, never
        // `convert_files`.
        files: &[],
        roles: &[],
    },
    FilesRecipe {
        id: "fastvlm",
        family: "fastvlm",
        // `llava_qwen.py` (the model's own HF-hub custom modeling file) is
        // distinctive enough alone. Confirmed needed the hard way: routing
        // `apple/FastVLM-0.5B` through `TransformersRecipe`'s curated fetch
        // (config.json/tokenizer.json/tokenizer_config.json/model.safetensors)
        // downloads a checkpoint that fails at load with "read .../vocab.json:
        // No such file or directory" -- this repo has no unified
        // `tokenizer.json`, only `vocab.json`+`merges.txt`+`added_tokens.json`,
        // none of which the curated list fetches.
        signature: &["llava_qwen.py"],
        files: &[],
        roles: &[("weights", ".")],
    },
    FilesRecipe {
        id: "qwen3asr",
        family: "qwen3asr",
        // No single distinctively-named file (an ordinary transformers-repo
        // shape) -- this exact combination is specific enough. Same gap as
        // `fastvlm`, caught by inspection before running it for real:
        // `Qwen/Qwen3-ASR-1.7B` ships `vocab.json`+`merges.txt`, no unified
        // `tokenizer.json`, so `TransformersRecipe`'s curated fetch would
        // miss the tokenizer files the same way it did for fastvlm.
        signature: &["vocab.json", "merges.txt", "preprocessor_config.json"],
        files: &[],
        roles: &[("weights", ".")],
    },
];

/// The `(family, roles)` a [`FilesRecipe::id`] carries, for the finish-side
/// manifest write -- so `supply.rs::convert_files` does not hand-maintain a
/// second copy of [`FILES_RECIPES`].
pub fn files_recipe_roles(id: &str) -> Option<(&'static str, &'static [(&'static str, &'static str)])> {
    FILES_RECIPES.iter().find(|r| r.id == id).map(|r| (r.family, r.roles))
}

impl ArtifactRecipe for FilesRecipe {
    fn id(&self) -> &'static str {
        self.id
    }

    fn matches_listing(&self, listing: &[String]) -> bool {
        self.signature.iter().all(|sig| listing.iter().any(|f| f == sig))
    }

    fn artifacts(&self, _reference: &ModelRef, listing: &[String], _hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>> {
        if self.files.is_empty() {
            return Ok(listing.iter().map(|f| artifact(f.clone(), f.clone())).collect());
        }
        Ok(self.files.iter().map(|f| artifact(*f, *f)).collect())
    }
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

/// A NATIVE Wan release repo (`Wan-AI/Wan2.1-T2V-1.3B`, `-T2V-14B`,
/// `-I2V-14B-480P`, confirmed live via the HF API): four model roles in ONE
/// flat listing, which is the whole reason `wan`'s `default_ref` names this
/// repo rather than the `-Diffusers` sibling (brain's fetch plan is one
/// `ModelRef` -> one listing -> one `Plan`).
///
/// It DOES ship a root `config.json`, so ordering matters: that config
/// declares `"model_type": "t2v"` and no `architectures`, so
/// [`TransformersRecipe`] claims it and then rejects it with `unsupported
/// architecture "t2v"` -- which is exactly what a flagless `brain wan t2v`
/// used to fail with. This recipe has to get first refusal.
///
/// No tensor rewrite is needed to serve one (`wan::import::import_dit`
/// remaps names in memory at load time, and the VAE/T5 are read straight from
/// their `.pth`), so the finish step
/// (`crates/cli/src/supply.rs::convert_wan`) only writes the
/// [`crate::CompoundManifest`] naming these roles.
pub struct WanRecipe;

impl WanRecipe {
    /// Role name -> the path (relative to the repo dir) that role's loader
    /// accepts. The single source of truth for wan's role layout: this
    /// recipe's [`matches_listing`](ArtifactRecipe::matches_listing) probes
    /// it and the finish-side manifest writer reuses it verbatim.
    ///
    /// `dit` is the single-file 1.3B form; the 14B tiers ship a shard set
    /// instead, which `convert_wan` resolves to the repo directory itself
    /// (`checkpoint::safetensors::read_model_dir` follows the
    /// `diffusion_pytorch_model.safetensors.index.json` there and reads only
    /// the shards, never the two `.pth` siblings) -- so the shard case is
    /// [`SHARDED_DIT`], not a missing role.
    pub const ROLES: &'static [(&'static str, &'static str)] = &[
        ("dit", "diffusion_pytorch_model.safetensors"),
        ("vae", "Wan2.1_VAE.pth"),
        ("text_encoder", "models_t5_umt5-xxl-enc-bf16.pth"),
        // The directory, not `tokenizer.json` inside it:
        // `data::unigram::UnigramTokenizer` takes either, and the directory
        // keeps `spiece.model`/`tokenizer_config.json` reachable next to it.
        ("tokenizer", "google/umt5-xxl"),
    ];

    /// The `dit` role for a sharded (14B) checkpoint: the repo directory.
    pub const SHARDED_DIT: &'static str = ".";

    /// Files that must ALL be present for this to be a native Wan repo. The
    /// DiT is deliberately NOT in here (single file at 1.3B, a shard set at
    /// 14B); these three are byte-identical across every 2.1 release and
    /// distinctive enough that nothing else can match by accident.
    const SIGNATURE: &'static [&'static str] = &["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer.json"];

    /// Everything under here is fetched (the tokenizer's four files).
    const TOKENIZER_DIR: &'static str = "google/umt5-xxl/";

    fn is_dit(file: &str) -> bool {
        file.starts_with("diffusion_pytorch_model") && (file.ends_with(".safetensors") || file == "diffusion_pytorch_model.safetensors.index.json")
    }
}

impl ArtifactRecipe for WanRecipe {
    fn id(&self) -> &'static str {
        "wan"
    }

    fn matches_listing(&self, listing: &[String]) -> bool {
        Self::SIGNATURE.iter().all(|sig| listing.iter().any(|f| f == sig)) && listing.iter().any(|f| Self::is_dit(f))
    }

    fn artifacts(&self, reference: &ModelRef, listing: &[String], _hub: &dyn Hub) -> Result<Vec<Artifact>, Box<PlanError>> {
        // The DiT: the single file when the repo has one, else the whole
        // shard set plus its index. Never both -- a repo that shipped both
        // would otherwise download the model twice.
        let single = listing.iter().any(|f| f == "diffusion_pytorch_model.safetensors");
        let mut artifacts: Vec<Artifact> = Vec::new();
        for f in listing {
            let take = if Self::is_dit(f) {
                if single {
                    f == "diffusion_pytorch_model.safetensors"
                } else {
                    true
                }
            } else {
                f.starts_with(Self::TOKENIZER_DIR) || Self::SIGNATURE.contains(&f.as_str()) || f == "config.json"
            };
            if take {
                artifacts.push(artifact(f.clone(), f.clone()));
            }
        }
        if !artifacts.iter().any(|a| Self::is_dit(&a.file)) {
            return Err(Box::new(PlanError::NoUpstreamArtifact(reference.clone(), "no diffusion_pytorch_model*.safetensors in repo".to_string())));
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

    /// The exact `Wan-AI/Wan2.1-T2V-1.3B` file listing, confirmed live via
    /// the HF API this session -- NOT the local checkout, which is a
    /// deliberately partial `allow_patterns` download and therefore no
    /// evidence of what the repo contains.
    fn wan_t2v_1_3b_listing() -> Vec<String> {
        [
            ".gitattributes",
            "LICENSE.txt",
            "README.md",
            "Wan2.1_VAE.pth",
            "assets/comp_effic.png",
            "assets/logo.png",
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "examples/i2v_input.JPG",
            "google/umt5-xxl/special_tokens_map.json",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    /// `Wan-AI/Wan2.1-T2V-14B`, same session: identical shape except the DiT
    /// is a six-way shard set plus an index.
    fn wan_t2v_14b_listing() -> Vec<String> {
        let mut v: Vec<String> = wan_t2v_1_3b_listing().into_iter().filter(|f| f != "diffusion_pytorch_model.safetensors").collect();
        for i in 1..=6 {
            v.push(format!("diffusion_pytorch_model-0000{i}-of-00006.safetensors"));
        }
        v.push("diffusion_pytorch_model.safetensors.index.json".to_string());
        v
    }

    /// The bug this recipe exists to fix: `Wan-AI/Wan2.1-T2V-1.3B` ships a
    /// root `config.json` whose `model_type` is `"t2v"` and which declares no
    /// `architectures`, so `TransformersRecipe` claims it and then fails with
    /// `unsupported architecture "t2v"` -- which is exactly what a flagless
    /// `brain wan t2v` reported before this row existed.
    #[test]
    fn wan_recipe_claims_the_native_repo_ahead_of_transformers() {
        let listing = wan_t2v_1_3b_listing();
        let matched = recipes().into_iter().find(|r| r.matches_listing(&listing)).unwrap();
        assert_eq!(matched.id(), "wan", "the native Wan repo must not fall through to the transformers catch-all");
        assert_eq!(recipes().into_iter().find(|r| r.matches_listing(&wan_t2v_14b_listing())).unwrap().id(), "wan");
    }

    #[test]
    fn wan_recipe_does_not_match_the_other_shapes_this_store_knows() {
        assert!(!WanRecipe.matches_listing(&["config.json".to_string(), "model.safetensors".to_string()]));
        assert!(!WanRecipe.matches_listing(&zimage_turbo_listing()));
        assert!(!WanRecipe.matches_listing(&ultralytics_yolov8_listing()));
        // The three signature files alone are not enough: no DiT, no model.
        let no_dit: Vec<String> = wan_t2v_1_3b_listing().into_iter().filter(|f| !f.starts_with("diffusion_pytorch_model")).collect();
        assert!(!WanRecipe.matches_listing(&no_dit));
    }

    #[test]
    fn wan_recipe_downloads_the_four_roles_and_none_of_the_documentation() {
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("Wan-AI", "Wan2.1-T2V-1.3B", None);
        let artifacts = WanRecipe.artifacts(&r, &wan_t2v_1_3b_listing(), &hub).unwrap();
        let files: Vec<&str> = artifacts.iter().map(|a| a.file.as_str()).collect();
        for (_, rel) in WanRecipe::ROLES {
            // Every role path is either a file that landed, or the directory
            // whose files did.
            let covered = files.contains(rel) || files.iter().any(|f| f.starts_with(&format!("{rel}/")));
            assert!(covered, "role path {rel:?} is not covered by what this recipe downloads");
        }
        // Nothing renamed -- the manifest writer's role paths are the listing's
        // own paths.
        assert!(artifacts.iter().all(|a| a.file == a.dest_name));
        // 3.6 MB of README/LICENSE/screenshots is 3.6 MB nobody asked for.
        assert!(!files.iter().any(|f| f.starts_with("assets/") || f.starts_with("examples/") || *f == "README.md" || *f == "LICENSE.txt"), "{files:?}");
        // The root config.json rides along: it is 249 bytes and it is the
        // only on-disk record of which variant this checkpoint is.
        assert!(files.contains(&"config.json"));
    }

    #[test]
    fn wan_recipe_takes_the_shard_set_when_there_is_no_single_dit_file() {
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("Wan-AI", "Wan2.1-T2V-14B", None);
        let files: Vec<String> = WanRecipe.artifacts(&r, &wan_t2v_14b_listing(), &hub).unwrap().into_iter().map(|a| a.file).collect();
        assert_eq!(files.iter().filter(|f| f.ends_with(".safetensors")).count(), 6);
        assert!(files.iter().any(|f| f == "diffusion_pytorch_model.safetensors.index.json"), "the index is what makes the shard set readable");

        // ... and NOT the single-file form alongside them, which would
        // download the model twice.
        let mut both = wan_t2v_14b_listing();
        both.push("diffusion_pytorch_model.safetensors".to_string());
        let files: Vec<String> = WanRecipe.artifacts(&r, &both, &hub).unwrap().into_iter().map(|a| a.file).collect();
        assert_eq!(files.iter().filter(|f| f.ends_with(".safetensors")).collect::<Vec<_>>(), ["diffusion_pytorch_model.safetensors"]);
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

    /// The exact `facebook/sam2.1-hiera-tiny` file listing, confirmed live via
    /// the HF API this session. Deliberately includes `config.json` and
    /// `model.safetensors`, which is what makes this a real test of
    /// ordering: without `FilesRecipe` claiming it ahead of
    /// `TransformersRecipe`, this repo would fetch as a (wrong -- brain's
    /// SAM2 loader wants the raw `.pt`) transformers-shaped model instead.
    fn sam2_tiny_listing() -> Vec<String> {
        [
            ".gitattributes",
            "README.md",
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "sam2.1_hiera_t.yaml",
            "sam2.1_hiera_tiny.pt",
            "video_preprocessor_config.json",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn files_recipe_matches_sam2_ahead_of_transformers_despite_a_config_json() {
        let listing = sam2_tiny_listing();
        let matched = recipes().into_iter().find(|r| r.matches_listing(&listing)).unwrap();
        assert_eq!(matched.id(), "sam2");
    }

    #[test]
    fn files_recipe_downloads_only_its_named_files() {
        let listing = sam2_tiny_listing();
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("facebook", "sam2.1-hiera-tiny", None);
        let recipe = recipes().into_iter().find(|r| r.id() == "sam2").unwrap();
        let artifacts = recipe.artifacts(&r, &listing, &hub).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].file, "sam2.1_hiera_tiny.pt");
        assert_eq!(artifacts[0].dest_name, "sam2.1_hiera_tiny.pt");
    }

    #[test]
    fn files_recipe_roles_are_looked_up_by_id_not_duplicated() {
        let (family, roles) = files_recipe_roles("sam2").unwrap();
        assert_eq!(family, "sam2");
        assert_eq!(roles, &[("weights", "sam2.1_hiera_tiny.pt")]);
        assert!(files_recipe_roles("no-such-recipe").is_none());
    }

    #[test]
    fn deepseek2ocr_gguf_recipe_downloads_both_named_ggufs() {
        let listing: Vec<String> = ["README.md", "DeepSeek-OCR-Q8_0.gguf", "mmproj-DeepSeek-OCR-Q8_0.gguf"].into_iter().map(String::from).collect();
        let hub = crate::hub::FakeHub::new();
        let r = ModelRef::new("ggml-org", "DeepSeek-OCR-GGUF", None);
        let recipe = recipes().into_iter().find(|r| r.id() == "deepseek2ocr-gguf").unwrap();
        assert!(recipe.matches_listing(&listing));
        let artifacts = recipe.artifacts(&r, &listing, &hub).unwrap();
        let files: Vec<&str> = artifacts.iter().map(|a| a.file.as_str()).collect();
        assert_eq!(files, ["DeepSeek-OCR-Q8_0.gguf", "mmproj-DeepSeek-OCR-Q8_0.gguf"]);
        let (family, roles) = files_recipe_roles("deepseek2ocr-gguf").unwrap();
        assert_eq!(family, "deepseek2ocr");
        assert_eq!(roles, &[("dir", ".")]);
    }
}
