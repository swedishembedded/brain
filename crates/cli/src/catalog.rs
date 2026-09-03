// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CLI-side extension of [`catalog`] (`brain-catalog`): patches CLI-local
//! residency adapters back onto the entries `catalog::models()` deliberately
//! leaves `resident: None` on, and appends the handful of models whose
//! MANIFEST itself is CLI-local (not a model crate's `caps.rs`), so
//! `brain caps`/`brain do`/`brain serve` see the exact same list they always
//! have.
//!
//! `catalog::models()` (this workspace's new `crates/catalog` library) is now
//! the single source of truth for every model's manifest + weight-free
//! provider - see that crate's own module doc for why. It cannot also own
//! the ~20 residency adapters that wrap a `crate::resident_*` type
//! (`Sam2Resident` and its siblings): those types are CLI-local, and
//! `brain-catalog` must not depend on `brain-cli`: the crate-graph layer rule
//! this workspace follows puts `cli` at the TOP of the stack - it "aggregates
//! everything", and nothing below it may depend back on it. So this file is
//! that missing half, kept as small as the split allows:
//!
//! * [`models`] = `catalog::models()`, with [`resident_ctor_for`] patching the
//!   `resident` field on every entry whose adapter lives in this crate, plus
//!   FOUR appended entries whose `manifest` fn is itself CLI-local:
//!   `imageops`/`demo` (no residency adapter - `resident.rs::build_executor`
//!   registers those two directly as stateless residents, bypassing this
//!   list entirely, same as it always has) and the three forecasters
//!   (`chronos2`/`fincast`/`kronos` - [`crate::resident_forecast`]'s own
//!   manifest functions, `pub(crate)` and therefore unreachable from
//!   `brain-catalog` regardless of the dependency direction).
//! * [`manifests`]/[`provider`] mirror `catalog`'s own (operating on the
//!   patched+extended list here, so `brain caps`/`brain do` see imageops/
//!   demo/the forecasters too - `catalog`'s own copies do not).
//! * [`residents`]/[`multi_residents`] are new here (not in `catalog` at
//!   all): they are entirely about CLI-local `ResidentModel`/
//!   `MultiDeviceResidentModel` impls, so there is nothing for the base
//!   crate to own.
//!
//! The manifest+provider invariant `catalog`'s own tests already pin (every
//! listed model constructible by name, no duplicate ids) still holds for
//! THIS crate's `models()`, because it is `catalog::models()` plus four
//! entries that are unique by construction - see this file's own tests for
//! the residency-specific invariants layered on top (every adapter here is
//! also listed here; no model registered through both claim paths).

use std::sync::Arc;

use capability::{Manifest, Provider};
use catalog::{ModelEntry, ResidentCtor};
use residency::ResidentModel;

/// The patch table: every catalog id whose residency adapter is CLI-local,
/// mapped to the SAME `resident!`/`resident_multi!`-built [`ResidentCtor`]
/// the original single-file catalog used to build inline. One `if` per
/// model, matched on the model crate's own `caps::MODEL` constant (not a
/// string literal) so a renamed id fails to compile here instead of silently
/// leaving a model un-served.
fn resident_ctor_for(model_id: &str) -> Option<ResidentCtor> {
    if model_id == sam2::caps::MODEL {
        return catalog::resident!(crate::resident_sam2::Sam2Resident::from_env);
    }
    if model_id == scrfd::caps::MODEL {
        return catalog::resident!(crate::resident_scrfd::ScrfdResident::from_env);
    }
    if model_id == arcface::caps::MODEL {
        return catalog::resident!(crate::resident_arcface::ArcFaceResident::from_env);
    }
    if model_id == vqgan::caps::MODEL {
        return catalog::resident!(crate::resident_restore::VqganResident::from_env);
    }
    if model_id == codeformer::caps::MODEL {
        return catalog::resident!(crate::resident_restore::RestoreResident::from_env);
    }
    if model_id == rrdbnet::caps::MODEL {
        return catalog::resident!(crate::resident_upscale::UpscaleResident::from_env);
    }
    if model_id == clip::caps::MODEL {
        return catalog::resident!(crate::resident_clip::ClipResident::from_env);
    }
    if model_id == t5encoder::caps::MODEL {
        return catalog::resident!(crate::resident_t5encoder::T5encoderResident::from_env);
    }
    if model_id == sdxlunet::caps::MODEL {
        return catalog::resident!(crate::resident_sdxl::SdxlResident::from_env);
    }
    if model_id == controlnet::caps::MODEL {
        return catalog::resident!(crate::resident_controlnet::ControlnetResident::from_env);
    }
    if model_id == supir::caps::MODEL {
        return catalog::resident!(crate::resident_supir::SupirResident::from_env);
    }
    if model_id == flux1::caps::MODEL {
        return catalog::resident!(crate::resident_flux1::Flux1Resident::from_env);
    }
    if model_id == pulid::caps::MODEL {
        return catalog::resident!(crate::resident_pulid::PulidResident::from_env);
    }
    if model_id == deepseek2ocr::caps::MODEL {
        // The only MULTI-device entry: its vision tower runs on wgpu while its
        // decoder runs on the CPU backend, so it must be claimed through
        // `claim_multi` for both to be budgeted - see
        // `crate::resident_deepseekocr`'s header.
        return catalog::resident_multi!(crate::resident_deepseekocr::DeepseekOcrResident::from_env);
    }
    if model_id == moondream3::caps::MODEL {
        return catalog::resident!(crate::resident_moondream3::Moondream3Resident::from_env);
    }
    if model_id == qwen3vl::caps::MODEL {
        return catalog::resident!(crate::resident_qwen3vl::Qwen3VlResident::from_env);
    }
    if model_id == qwen3tts::caps::MODEL {
        return catalog::resident!(crate::resident_tts::TtsResident::from_env);
    }
    if model_id == minimaxmusic3::caps::MODEL {
        return catalog::resident!(crate::resident_minimaxmusic3::MinimaxMusic3Resident::from_env);
    }
    if model_id == cosyvoice::caps::MODEL {
        return catalog::resident!(crate::resident_cosyvoice::CosyVoiceResident::from_env);
    }
    if model_id == nemotronasr::caps::MODEL {
        return catalog::resident!(crate::resident_asr::NemotronResident::from_env);
    }
    if model_id == qwen3asr::caps::MODEL {
        return catalog::resident!(crate::resident_asr::QwenAsrResident::from_env);
    }
    None
}

/// Every model `brain caps` lists and `brain do` can run: `catalog::models()`
/// with CLI-local residency adapters patched back in, plus the four entries
/// whose manifest itself is CLI-local (see the module doc).
pub fn models() -> Vec<ModelEntry> {
    let mut entries = catalog::models();
    for e in &mut entries {
        let id = (e.manifest)().model;
        e.resident = resident_ctor_for(&id);
    }
    // Time-series forecasting. Discoverable (`brain caps`) and served
    // (`brain serve`, via the resident ctors) - but with no direct `brain do`
    // provider yet: the forecast run logic lives in the residency instances
    // (NPU/device placement included), so the provider says exactly how to
    // reach the model instead of "unknown model". Their manifest fns are
    // `pub(crate)` in `resident_forecast.rs`, so - unlike every entry above - 
    // they can never move into `catalog` regardless of the dependency
    // direction.
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::chronos2_manifest,
        provider: || Err("chronos-2 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_CHRONOS2 set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::Chronos2Resident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::fincast_manifest,
        provider: || Err("fincast has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_FINCAST set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::FincastResident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::kronos_manifest,
        provider: || Err("kronos has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::KronosResident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::timesfm3_manifest,
        provider: || Err("timesfm3 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_TIMESFM3 set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::Timesfm3Resident::from_env),
    });
    // No-weights utility models, listed by `brain caps` but served (over
    // D-Bus/HTTP) directly from `resident.rs::build_executor`, which pushes
    // them as stateless residents itself rather than through this list - 
    // hence `resident: None` here, exactly as before this file existed.
    entries.push(ModelEntry { manifest: crate::imageops::manifest, provider: catalog::always!(crate::imageops::ImageOps), resident: None });
    entries.push(ModelEntry {
        manifest: || {
            use capability::Provider as _;
            crate::caps_cli::DemoModel.manifest()
        },
        provider: catalog::always!(crate::caps_cli::DemoModel),
        resident: None,
    });
    entries
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

/// The SINGLE-device residency adapters this catalog owns, for models whose
/// weights are configured - what `build_executor` folds into `Executor::start`.
///
/// Multi-device models are deliberately absent: they come from
/// [`multi_residents`] instead, and registering one here as well is precisely
/// the double-registration `Executor::register_multi`'s doc forbids.
pub fn residents() -> Vec<Arc<dyn ResidentModel>> {
    models()
        .into_iter()
        .filter_map(|e| match e.resident {
            Some(ResidentCtor::Single(f)) => f(),
            _ => None,
        })
        .collect()
}

/// The MULTI-device residency adapters this catalog owns - registered after
/// `Executor::start` via `register_multi`, because the scheduler's multi-device
/// claim path is what reserves on every device such an instance occupies.
///
/// `gpus` is `build_executor`'s budgeted `(index, TOTAL bytes)` list and
/// `reserved` its per-card headroom, forwarded verbatim so each adapter picks
/// its device set against the same usable capacity the scheduler budgets.
pub fn multi_residents(gpus: &[(u32, u64)], reserved: u64) -> Vec<Arc<dyn residency::multi::MultiDeviceResidentModel>> {
    models()
        .into_iter()
        .filter_map(|e| match e.resident {
            Some(ResidentCtor::Multi(f)) => f(gpus, reserved),
            _ => None,
        })
        .collect()
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
    /// be constructible by name. It may legitimately fail for want of weights - 
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

    /// Every residency adapter reachable from this file advertises an id this
    /// file also lists, so a model cannot be schedulable but undiscoverable.
    #[test]
    fn every_residency_adapter_here_is_also_listed_here() {
        let catalog: std::collections::HashSet<String> = manifests().into_iter().map(|m| m.model).collect();
        // Both claim paths, or the multi-device half is exactly as unguarded as
        // the single-device half was before this test existed.
        let single = residents().into_iter().map(|r| r.manifest().model);
        let multi = multi_residents(&[(0, 24u64 << 30)], 2u64 << 30).into_iter().map(|r| r.manifest().model);
        for id in single.chain(multi) {
            assert!(catalog.contains(&id), "residency adapter '{id}' is not in the catalog");
        }
    }

    /// A model registered through BOTH claim paths would have its budget
    /// charged twice and its `activate` reachable through a path it correctly
    /// refuses - see `residency::Executor::register_multi`'s own doc. The two
    /// lists are derived from one field precisely so this cannot happen, and
    /// this test is what says so out loud.
    #[test]
    fn no_model_is_registered_through_both_claim_paths() {
        let single: std::collections::HashSet<String> = residents().into_iter().map(|r| r.manifest().model).collect();
        for r in multi_residents(&[(0, 24u64 << 30)], 2u64 << 30) {
            let id = r.manifest().model;
            assert!(!single.contains(&id), "'{id}' is registered as BOTH a single- and a multi-device resident");
        }
    }

    /// `crates/imgpipe` names its stage models by STRING, because it links no
    /// model crate. This is the other half of that decision: the CLI sees both,
    /// so it asserts the strings still name real catalog entries - otherwise a
    /// renamed model would turn into a runtime "unknown model" from inside a
    /// pipeline run, which is the worst place to find out.
    #[test]
    fn imgpipe_stage_ids_match_the_catalog() {
        let ids: std::collections::HashSet<String> = manifests().into_iter().map(|m| m.model).collect();
        for stage in [imgpipe::SEGMENT_MODEL, imgpipe::RESTORE_MODEL, imgpipe::UPSCALE_MODEL] {
            assert!(ids.contains(stage), "imgpipe dispatches to '{stage}', which is not a catalog model");
        }
        assert_eq!(imgpipe::UPSCALE_MODEL, rrdbnet::caps::MODEL);
        assert_eq!(imgpipe::RESTORE_MODEL, codeformer::caps::MODEL);
        assert_eq!(imgpipe::SEGMENT_MODEL, sam2::caps::MODEL);
        assert!(ids.contains(imgpipe::SUPIR_RESTORE_MODEL), "imgpipe dispatches to '{}', which is not a catalog model", imgpipe::SUPIR_RESTORE_MODEL);
        assert_eq!(imgpipe::SUPIR_RESTORE_MODEL, supir::caps::MODEL);
    }

    /// `crates/supir` links no VLM, so its optional caption auto-fill names
    /// LLaVA by STRING (`supir::caps::LLAVA_MODEL`) - this is the other half
    /// of that decision: the CLI sees both real constants, so it asserts the
    /// string still names the real catalog entry - the same drift class
    /// `imgpipe_stage_ids_match_the_catalog` guards against.
    #[test]
    fn supir_llava_model_id_matches_the_catalog() {
        assert_eq!(supir::caps::LLAVA_MODEL, llava::caps::MODEL);
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

    /// Every CLI-local resident id in [`resident_ctor_for`]'s patch table must
    /// name a real catalog entry - otherwise a typo'd `caps::MODEL` reference
    /// silently patches nothing and the model quietly stops being served.
    #[test]
    fn every_patched_id_is_a_real_catalog_entry() {
        let ids: std::collections::HashSet<String> = catalog::manifests().into_iter().map(|m| m.model).collect();
        let patched = [
            sam2::caps::MODEL,
            scrfd::caps::MODEL,
            arcface::caps::MODEL,
            vqgan::caps::MODEL,
            codeformer::caps::MODEL,
            rrdbnet::caps::MODEL,
            clip::caps::MODEL,
            t5encoder::caps::MODEL,
            sdxlunet::caps::MODEL,
            controlnet::caps::MODEL,
            supir::caps::MODEL,
            flux1::caps::MODEL,
            pulid::caps::MODEL,
            deepseek2ocr::caps::MODEL,
            moondream3::caps::MODEL,
            qwen3vl::caps::MODEL,
            qwen3tts::caps::MODEL,
            minimaxmusic3::caps::MODEL,
            cosyvoice::caps::MODEL,
            nemotronasr::caps::MODEL,
            qwen3asr::caps::MODEL,
        ];
        for id in patched {
            assert!(ids.contains(id), "resident_ctor_for patches '{id}', which is not a catalog::models() entry");
            assert!(resident_ctor_for(id).is_some(), "'{id}' must resolve to a resident ctor");
        }
    }
}
