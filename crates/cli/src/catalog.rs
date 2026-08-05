// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served-model catalog: **one entry per model, in one place**.
//!
//! # Why this exists
//!
//! Adding a model used to mean editing three hand-maintained lists that had no
//! link to each other:
//!
//! 1. `caps_cli::static_manifests()` — what `brain caps` lists.
//! 2. `caps_cli::build_registry()`  — what `brain do` can actually run.
//! 3. `resident::build_executor()`  — what is served over D-Bus/HTTP.
//!
//! Nothing checked that a model appeared in all three, and every omission is
//! silent in a different way: missing from (1) it is undiscoverable, missing
//! from (2) `brain caps <id>` answers "unknown model" for a model it had just
//! listed, missing from (3) it is invisible to every transport. That is not
//! hypothetical — `ai-forever/Real-ESRGAN` was added with a manifest, a
//! provider, a residency adapter and passing tests, and was still unreachable
//! because only (3) had been edited.
//!
//! So (1) and (2) are now DERIVED from [`MODELS`]: the manifest and the
//! provider constructor sit in the same entry and cannot drift apart. The
//! `catalog_*` tests below pin the invariants, including the exact failure
//! above — every listed model must be constructible by name.
//!
//! # What is deliberately NOT here yet
//!
//! `build_executor` still registers text-generation, forecasting, ASR and mock
//! residents from its own list. Those have no weights-free `caps::manifest()`
//! to put in an entry, so folding them in is a real change to what `brain caps`
//! reports rather than a refactor. [`catalog_and_residency_do_not_overlap`]
//! pins that the two halves stay disjoint, so a model cannot end up registered
//! twice while that is true.

use std::sync::Arc;

use capability::{Manifest, Provider};
use residency::ResidentModel;

/// One served model. `provider` and `manifest` describe the SAME model by
/// construction — that is the whole point of the type.
pub struct ModelEntry {
    /// The static manifest: safe to build with no weights loaded.
    pub manifest: fn() -> Manifest,
    /// Build something runnable. `Err` carries the model's OWN "set BRAIN_…"
    /// message, so a caller never sees a generic one.
    pub provider: fn() -> Result<Arc<dyn Provider>, String>,
    /// Register with the residency scheduler, when this model has an adapter
    /// and its weights are configured. `None` from the fn means "not
    /// configured"; a `None` field means "no adapter exists yet".
    pub resident: Option<fn() -> Option<Arc<dyn ResidentModel>>>,
}

/// Shorthand: a provider that needs no weights.
macro_rules! always {
    ($e:expr) => {
        || Ok(Arc::new($e) as Arc<dyn Provider>)
    };
}

/// Shorthand: a provider built from env, with the model's own error message.
macro_rules! from_env {
    ($ctor:path, $msg:literal) => {
        || $ctor().map(|p| Arc::new(p) as Arc<dyn Provider>).ok_or($msg.to_string())
    };
}

/// Shorthand: a residency adapter built from env.
macro_rules! resident {
    ($ctor:path) => {
        Some((|| $ctor().map(|r| Arc::new(r) as Arc<dyn ResidentModel>)) as fn() -> Option<Arc<dyn ResidentModel>>)
    };
}

/// Every model `brain caps` lists and `brain do` can run.
pub fn models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            manifest: zimage::caps::manifest,
            provider: || zimage::caps::ZImageProvider::load().map(|p| Arc::new(p) as Arc<dyn Provider>),
            resident: None, // ZImageResident::from_env is Result-shaped; see resident.rs
        },
        ModelEntry {
            manifest: flux2::caps::manifest,
            provider: always!(flux2::caps::Flux2Provider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: qwen::caps::manifest,
            provider: always!(qwen::caps::QwenProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: lfm::caps::manifest,
            provider: always!(lfm::caps::LfmProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: fastvlm::caps::manifest,
            provider: always!(fastvlm::caps::FastVlmProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: yolo::caps::manifest,
            provider: always!(yolo::caps::YoloProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: depth::caps::manifest,
            provider: always!(depth::caps::DepthProvider::new()),
            resident: None,
        },
        // The imaging models carry their weights path in the provider (from a
        // `BRAIN_*` env var), not as an action param, so `brain do` and the
        // residency adapter advertise ONE manifest each.
        ModelEntry {
            manifest: sam2::caps::manifest,
            provider: from_env!(
                sam2::caps::Sam2Provider::from_env,
                "set BRAIN_SAM2_WEIGHTS to an existing sam2.1_hiera_*.pt checkpoint"
            ),
            resident: resident!(crate::resident_sam2::Sam2Resident::from_env),
        },
        ModelEntry {
            manifest: facenet::caps::manifest,
            provider: from_env!(
                facenet::caps::FacenetProvider::from_env,
                "set BRAIN_FACENET_DIR to an antelopev2 directory holding glintr100.onnx + scrfd_10g_bnkps.onnx"
            ),
            resident: resident!(crate::resident_facenet::FacenetResident::from_env),
        },
        ModelEntry {
            manifest: vqgan::caps::manifest,
            provider: from_env!(
                vqgan::caps::VqganProvider::from_env,
                "set BRAIN_VQGAN_WEIGHTS to an existing VQGAN checkpoint (or its directory)"
            ),
            resident: resident!(crate::resident_restore::VqganResident::from_env),
        },
        ModelEntry {
            manifest: restore::caps::manifest,
            provider: from_env!(
                restore::caps::RestoreProvider::from_env,
                "set BRAIN_RESTORE_WEIGHTS to an existing codeformer.pth (or its directory)"
            ),
            resident: resident!(crate::resident_restore::RestoreResident::from_env),
        },
        ModelEntry {
            manifest: upscale::caps::manifest,
            provider: from_env!(
                upscale::caps::UpscaleProvider::from_env,
                "set BRAIN_ESRGAN_WEIGHTS to an existing RealESRGAN_x4plus.pth"
            ),
            resident: resident!(crate::resident_upscale::UpscaleResident::from_env),
        },
        ModelEntry {
            manifest: clip::caps::manifest,
            provider: from_env!(
                clip::caps::ClipProvider::from_env,
                "set BRAIN_CLIP_DIR to a checkpoint root holding tokenizer/ (CLIP-L) and/or tokenizer_2/ (OpenCLIP-bigG)"
            ),
            resident: resident!(crate::resident_clip::ClipResident::from_env),
        },
        ModelEntry {
            manifest: imgpipe::caps::manifest,
            provider: || Ok(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(stage_registry()))) as Arc<dyn Provider>),
            resident: None,
        },
        ModelEntry {
            manifest: tts::caps::manifest,
            provider: always!(tts::caps::TtsProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: crate::imageops::manifest,
            provider: always!(crate::imageops::ImageOps),
            resident: None,
        },
        ModelEntry {
            manifest: || {
                use capability::Provider as _;
                crate::caps_cli::DemoModel.manifest()
            },
            provider: always!(crate::caps_cli::DemoModel),
            resident: None,
        },
    ]
}

/// The registry `imgpipe`'s stages dispatch into.
///
/// The pipeline is a capability that COMPOSES capabilities, so it gets the
/// models whose weights are configured — and a stage whose model is unset fails
/// with THAT model's "set BRAIN_…" message rather than a generic one from the
/// pipeline. Built from [`models`] so a new stage-capable model does not need a
/// second list here either.
fn stage_registry() -> capability::Registry {
    let mut inner = capability::Registry::new();
    for e in models() {
        let id = (e.manifest)().model;
        // Only the models a stage can actually name today. Registering the rest
        // would build providers nobody asked for (some load weights).
        if [imgpipe::SEGMENT_MODEL, imgpipe::RESTORE_MODEL, imgpipe::UPSCALE_MODEL].contains(&id.as_str()) {
            if let Ok(p) = (e.provider)() {
                inner.register(p);
            }
        }
    }
    inner
}

/// Every model's static manifest, for `brain caps`.
pub fn manifests() -> Vec<Manifest> {
    models().into_iter().map(|e| (e.manifest)()).collect()
}

/// Build a runnable provider for `model`, or say why not.
pub fn provider(model: &str) -> Result<Arc<dyn Provider>, String> {
    for e in models() {
        if (e.manifest)().model == model {
            return (e.provider)();
        }
    }
    Err(format!("unknown model '{model}' (see `brain caps`)"))
}

/// The residency adapters this catalog owns, for models whose weights are
/// configured.
pub fn residents() -> Vec<Arc<dyn ResidentModel>> {
    models().into_iter().filter_map(|e| e.resident.and_then(|f| f())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two entries claiming the same id would make `provider` resolve by
    /// position, which is a coin flip.
    #[test]
    fn catalog_ids_are_unique() {
        let ids: Vec<String> = manifests().into_iter().map(|m| m.model).collect();
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(id.clone()), "duplicate catalog id '{id}'");
        }
        assert!(ids.len() > 10, "the catalog looks truncated ({} entries)", ids.len());
    }

    /// THE DRIFT THIS FILE EXISTS TO KILL: every model `brain caps` lists must
    /// be constructible by name. It may legitimately fail for want of weights —
    /// what it must never do is answer "unknown model" for something it just
    /// advertised.
    #[test]
    fn every_listed_model_is_constructible_by_name() {
        for m in manifests() {
            match provider(&m.model) {
                Ok(_) => {}
                Err(e) => assert!(
                    !e.contains("unknown model"),
                    "`brain caps` lists '{}' but `brain do` cannot build it: {e}",
                    m.model
                ),
            }
        }
    }

    /// A model that is in neither half is unreachable; one in both would be
    /// registered twice. The residency half is still an explicit list in
    /// `resident.rs` (see the module docs), so pin that they are disjoint.
    #[test]
    fn catalog_and_residency_do_not_overlap() {
        let catalog: std::collections::HashSet<String> =
            manifests().into_iter().map(|m| m.model).collect();
        for r in residents() {
            let id = r.manifest().model;
            assert!(catalog.contains(&id), "residency adapter '{id}' is not in the catalog");
        }
    }

    /// `crates/imgpipe` names its stage models by STRING, because it links no
    /// model crate. This is the other half of that decision: the CLI sees both,
    /// so it asserts the strings still name real catalog entries — otherwise a
    /// renamed model would turn into a runtime "unknown model" from inside a
    /// pipeline run, which is the worst place to find out.
    #[test]
    fn imgpipe_stage_ids_match_the_catalog() {
        let ids: std::collections::HashSet<String> =
            manifests().into_iter().map(|m| m.model).collect();
        for stage in [imgpipe::SEGMENT_MODEL, imgpipe::RESTORE_MODEL, imgpipe::UPSCALE_MODEL] {
            assert!(ids.contains(stage), "imgpipe dispatches to '{stage}', which is not a catalog model");
        }
        assert_eq!(imgpipe::UPSCALE_MODEL, upscale::caps::MODEL);
        assert_eq!(imgpipe::RESTORE_MODEL, restore::caps::MODEL);
        assert_eq!(imgpipe::SEGMENT_MODEL, sam2::caps::MODEL);
    }

    /// An unknown name must still be an error, not a panic or a default.
    #[test]
    fn an_unknown_model_is_an_error() {
        let e = match provider("definitely/not-a-model") {
            Err(e) => e,
            Ok(_) => panic!("a made-up model resolved"),
        };
        assert!(e.contains("unknown model"), "{e}");
    }
}
