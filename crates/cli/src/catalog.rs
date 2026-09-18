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

use brain_modelstore::resolve::ArchSpec;
use capability::{Assembly, Manifest, Provider};
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
    if model_id == florence2::caps::MODEL {
        return catalog::resident!(crate::resident_florence2::Florence2Resident::from_env);
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
        return catalog::resident_multi!(crate::resident_deepseekocr::DeepseekOcrResident::from_assembly);
    }
    if model_id == deepseekocr2::caps::MODEL {
        // Single-device (CPU) for now, unlike v1 - see
        // `crate::resident_deepseekocr2`'s header for why the wgpu split v1
        // eventually earned is not claimed here yet.
        return catalog::resident!(crate::resident_deepseekocr2::DeepseekOcr2Resident::from_env);
    }
    if model_id == moondream3::caps::MODEL {
        return None; // registered directly in build_executor with a resolved Assembly, see resident.rs
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
        provider: |_assembly: &Assembly| Err("chronos-2 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_CHRONOS2 set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::Chronos2Resident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::fincast_manifest,
        provider: |_assembly: &Assembly| Err("fincast has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_FINCAST set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::FincastResident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::kronos_manifest,
        provider: |_assembly: &Assembly| Err("kronos has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER set".to_string()),
        resident: catalog::resident!(crate::resident_forecast::KronosResident::from_env),
    });
    entries.push(ModelEntry {
        manifest: crate::resident_forecast::timesfm3_manifest,
        provider: |_assembly: &Assembly| Err("timesfm3 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_TIMESFM3 set, or the resolver (`brain timesfm3 predict ...`)".to_string()),
        resident: catalog::resident!(crate::resident_forecast::Timesfm3Resident::from_env),
    });
    // 3D Gaussian Splatting (render/fit): unlike every entry above, it needs
    // no weights at all - the scene arrives as request bytes - so, unlike the
    // truly stateless pair just below, it DOES have a resident
    // (`resident_splat.rs`, always constructible, gated on nothing): render/
    // fit genuinely allocate GPU buffers per request, worth scheduling.
    entries.push(ModelEntry {
        manifest: splat::caps::manifest,
        provider: catalog::always!(splat::caps::SplatProvider::new()),
        resident: catalog::resident!(crate::resident_splat::SplatResident::from_env),
    });
    // WorldMirror-2 multi-view 3D reconstruction. Same shape as GLM/qwen3.5
    // above (`weights` is a per-invocation action param, so `manifest` is
    // weights-free and costs nothing to build), but registered HERE rather
    // than in `crates/catalog` because its resident adapter
    // (`resident_worldmirror2.rs`) is CLI-local, same reason `splat` just
    // above lives here too - `brain-catalog` cannot depend back on `brain-cli`
    // (see this file's module doc), so a model whose resident needs
    // `crate::resident_llm::on_device` must have its WHOLE entry live here.
    entries.push(ModelEntry {
        manifest: worldmirror2::caps::manifest,
        provider: catalog::always!(worldmirror2::caps::WorldMirror2Provider::new()),
        resident: catalog::resident!(crate::resident_worldmirror2::WorldMirror2Resident::from_env),
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

/// A placeholder [`Assembly`] for a caller that has none - every entry not
/// listed in [`resolver_spec_for`] ignores the argument today (see
/// `catalog::ModelEntry::provider`'s doc).
fn empty_assembly() -> Assembly {
    Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }
}

/// `(arch name, ArchSpec)` for every catalog model id whose `ModelEntry::provider`
/// actually reads the [`Assembly`] it is called with - [`resolved_assembly_for`]
/// resolves a real one for these through the model-store resolver instead of
/// [`empty_assembly`]. FLUX.2 is not listed here: it reaches its own
/// resolver-backed weights through `crate::flux2_cli`'s dedicated command,
/// never through this generic `brain do` path, so its catalog entry keeps
/// building from `empty_assembly` here (a pre-existing gap this migration
/// does not change).
fn resolver_spec_for(model_id: &str) -> Option<(&'static str, Box<dyn ArchSpec>)> {
    if model_id == s3dit::caps::MODEL {
        return Some(("s3dit", Box::new(s3dit::spec::S3ditSpec)));
    }
    if model_id == cosyvoice::caps::MODEL {
        return Some(("cosyvoice", Box::new(cosyvoice::spec::CosyVoiceSpec)));
    }
    if model_id == minimaxmusic3::caps::MODEL {
        return Some(("minimaxmusic3", Box::new(minimaxmusic3::spec::MinimaxMusic3Spec)));
    }
    if model_id == qwen35::caps::MODEL {
        return Some(("qwen35", Box::new(qwen35::spec::Qwen35Spec)));
    }
    if model_id == qwen3vl::caps::MODEL {
        return Some(("qwen3vl", Box::new(qwen3vl::spec::Qwen3VlSpec)));
    }
    if model_id == fastvlm::caps::MODEL {
        return Some(("fastvlm", Box::new(fastvlm::spec::FastvlmSpec)));
    }
    if model_id == moondream3::caps::MODEL {
        return Some(("moondream3", Box::new(moondream3::spec::Moondream3Spec)));
    }
    if model_id == deepseek2ocr::caps::MODEL {
        return Some(("deepseek2ocr", Box::new(deepseek2ocr::spec::Deepseek2ocrSpec)));
    }
    if model_id == sam2::caps::MODEL {
        return Some(("sam2", Box::new(sam2::spec::Sam2Spec)));
    }
    if model_id == codeformer::caps::MODEL {
        return Some(("codeformer", Box::new(codeformer::spec::CodeFormerSpec)));
    }
    if model_id == rrdbnet::caps::MODEL {
        return Some(("rrdbnet", Box::new(rrdbnet::spec::RrdbnetSpec)));
    }
    if model_id == scrfd::caps::MODEL {
        return Some(("scrfd", Box::new(scrfd::spec::ScrfdSpec)));
    }
    if model_id == arcface::caps::MODEL {
        return Some(("arcface", Box::new(arcface::spec::ArcFaceSpec)));
    }
    if model_id == clip::caps::MODEL {
        return Some(("clip", Box::new(clip::spec::ClipSpec)));
    }
    if model_id == florence2::caps::MODEL {
        return Some(("florence2", Box::new(florence2::spec::Florence2Spec)));
    }
    if model_id == flux1::caps::MODEL {
        return Some(("flux1", Box::new(flux1::spec::Flux1Spec)));
    }
    if model_id == pulid::caps::MODEL {
        return Some(("pulid", Box::new(pulid::spec::PulidSpec)));
    }
    if model_id == vqgan::caps::MODEL {
        return Some(("vqgan", Box::new(vqgan::spec::VqganSpec)));
    }
    if model_id == sdxlunet::caps::MODEL {
        return Some(("sdxlunet", Box::new(sdxlunet::spec::SdxlunetSpec)));
    }
    if model_id == controlnet::caps::MODEL {
        return Some(("controlnet", Box::new(controlnet::spec::ControlnetSpec)));
    }
    if model_id == t5encoder::caps::MODEL {
        return Some(("t5encoder", Box::new(t5encoder::spec::T5encoderSpec)));
    }
    if model_id == nemotronasr::caps::MODEL {
        return Some(("nemotronasr", Box::new(nemotronasr::spec::NemotronAsrSpec)));
    }
    if model_id == qwen3asr::caps::MODEL {
        return Some(("qwen3asr", Box::new(qwen3asr::spec::Qwen3AsrSpec)));
    }
    if model_id == flux2::caps::MODEL {
        return Some(("flux2", Box::new(flux2::spec::Flux2Spec)));
    }
    if model_id == wan::caps::MODEL {
        return Some(("wan", Box::new(wan::spec::WanSpec)));
    }
    None
}

/// [`resolver_spec_for`] plus the actual resolve, collapsed to one `Result`
/// so both [`provider`] and [`multi_residents`] share the same choke point
/// instead of each re-deriving "look up the spec, then resolve it" - `Some`
/// only for a model [`resolver_spec_for`] actually names; `None` means "this
/// model's weights are not resolver-based at all", so the caller falls back
/// to [`empty_assembly`].
fn resolved_assembly_for(model: &str) -> Option<Result<Assembly, String>> {
    let (arch, spec) = resolver_spec_for(model)?;
    Some(crate::resolver_cli::try_resolve(arch, spec.as_ref(), &std::collections::BTreeMap::new()).map_err(|e| e.message().to_string()))
}

/// Build a runnable provider for `model`, or say why not.
///
/// A model listed in [`resolver_spec_for`] gets a REAL, resolver-built
/// `Assembly` (scanning the models directory, same as `brain <arch> …`'s own
/// dedicated commands) - an `Ambiguous`/`Missing` outcome, or no models
/// directory at all, becomes this function's own `Err` (never an "unknown
/// model", per `every_listed_model_is_constructible_by_name`'s own contract:
/// a listed model may legitimately fail for want of weights). Every other
/// model still gets [`empty_assembly`], unchanged.
pub fn provider(model: &str) -> Result<Arc<dyn Provider>, String> {
    let assembly = match resolved_assembly_for(model) {
        Some(r) => r?,
        None => empty_assembly(),
    };
    provider_from_assembly(model, &assembly)
}

/// [`provider`], from an already-resolved [`Assembly`] instead of resolving
/// one itself -- the entry point for a caller that already has one on hand
/// (`crate::resolver_cli::run_generic_or_exit`, called from
/// `crate::resolve::dispatch_arch`'s `ARCH_TO_MODEL` branch, or from a
/// dedicated `_cli.rs` module that forwards its own non-special verbs to the
/// generic path, e.g. `sam2_cli`), so that path never resolves twice. Every
/// other model's `provider` fn still ignores the argument (see
/// `catalog::ModelEntry::provider`'s doc).
pub fn provider_from_assembly(model: &str, assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    for e in models() {
        if (e.manifest)().model == model {
            return (e.provider)(assembly);
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
            Some(ResidentCtor::Multi(f)) => {
                let model = (e.manifest)().model;
                let assembly = match resolved_assembly_for(&model) {
                    Some(Ok(a)) => a,
                    Some(Err(err)) => {
                        eprintln!("brain: {model} not served ({err})");
                        return None;
                    }
                    None => empty_assembly(),
                };
                f(&assembly, gpus, reserved)
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SERVED half of "resolver-migrated", and the one a CLI test cannot
    /// see.
    ///
    /// `crate::resolve::dispatch_arch` hands `provider_from_assembly` a REAL
    /// assembly it resolved itself, so a migrated architecture works from the
    /// command line as soon as its `ModelEntry.provider` reads one. Every
    /// other caller - D-Bus, HTTP, `build_executor` - goes through
    /// [`provider`] instead, which resolves via [`resolver_spec_for`] and
    /// falls back to [`empty_assembly`] for anything absent from it.
    ///
    /// So an entry whose provider reads a role while its model is missing
    /// from `resolver_spec_for` is broken in exactly one direction: fine on
    /// the CLI, "assembly 'local/…' has no <role> role" when served. This
    /// asserts the two agree, by CONSTRUCTING every listed model against an
    /// empty assembly and requiring that anything which needs a role is
    /// registered to get one.
    #[test]
    fn every_assembly_reading_provider_is_registered_for_the_served_path() {
        let empty = empty_assembly();
        let mut unregistered = Vec::new();
        for m in manifests() {
            let id = m.model;
            // Does this entry's provider actually need a resolved role?
            let needs_role = match provider_from_assembly(&id, &empty) {
                Err(e) => e.contains("has no") && e.contains("role"),
                Ok(_) => false,
            };
            if needs_role && resolver_spec_for(&id).is_none() {
                unregistered.push(id);
            }
        }
        assert!(
            unregistered.is_empty(),
            "these models build their provider from an Assembly but have no resolver_spec_for entry, so every served surface hands them an empty one: {unregistered:?}"
        );
    }

    /// Every catalog model is in exactly ONE of three declared states for
    /// how it finds its weights, and a model in none of them fails here
    /// rather than at a user's first run.
    ///
    /// 1. **Resolver-backed** - the provider reads a role off the `Assembly`,
    ///    and `resolver_spec_for` supplies a real one. Nothing to configure:
    ///    the checkpoint is found by scanning the model store.
    /// 2. **Weights as an action parameter** - the provider needs none, and
    ///    the action declares a `host_env` param the caller may pass
    ///    explicitly or let `ActionSpec::validate` fill from that variable.
    /// 3. **Env-only provider** - listed by name below, with the reason it
    ///    cannot classify from disk.
    ///
    /// The point is that state 3 is a CHOICE someone made and wrote down,
    /// not a default a new model falls into by being added. Together with
    /// `every_assembly_reading_provider_is_registered_for_the_served_path`
    /// (which catches a HALF-migrated model), this makes "weights stopped
    /// being findable and nobody noticed" a compile-time-adjacent failure
    /// instead of a support question.
    #[test]
    fn every_model_declares_how_it_finds_its_weights() {
        /// Models that legitimately still take their weights from the
        /// environment, each with the reason it cannot classify from disk.
        const ENV_ONLY: &[(&str, &str)] = &[
            (
                "deepseek-ai/DeepSeek-OCR-2",
                "no vendor-published GGUF exists; community conversions vary in layout and carry no \
                 declaration to classify on, so there is nothing stable for an ArchSpec to key off yet",
            ),
            (
                "brain/ltxv",
                "`ltxv::spec` exists and the CLI resolves through it, but the served manifest declares no \
                 weights param at all - t2v's default is the tiny random-weight smoke config, and the real \
                 22B path is reached by `--dit`/$BRAIN_LTXV_DIT on the dedicated CLI only. Registering it \
                 here means giving `LtxvProvider` the roles first, which belongs with making the real-weight \
                 path servable rather than with this gate",
            ),
        ];

        /// Models that need no weights at all: pure composition over OTHER
        /// models' weights, or a deterministic mock.
        const NO_WEIGHTS: &[&str] = &[
            "brain/imgpipe",
            "brain/imageops",
            "brain/demo",
            "brain/mock",
            // A rasterizer, not a learned model: `render` takes the scene
            // itself as an `--in scene=` PLY blob and `fit` optimizes one.
            "brain/splat",
        ];

        let empty = empty_assembly();
        let mut undeclared = Vec::new();
        for m in manifests() {
            let id = m.model.clone();
            if NO_WEIGHTS.contains(&id.as_str()) || ENV_ONLY.iter().any(|(n, _)| *n == id) {
                continue;
            }
            // State 1: the provider needs a resolved role, and gets one.
            if resolver_spec_for(&id).is_some() {
                continue;
            }
            // State 2: some action names a host-side weights param.
            if m.actions.iter().any(|a| a.params.iter().any(|p| p.host_env.is_some() || p.host_resolved)) {
                continue;
            }
            // A provider that cannot even be built from an empty assembly is
            // half-migrated; that is the sibling test's business, not this one.
            if provider_from_assembly(&id, &empty).is_err() {
                continue;
            }
            undeclared.push(id);
        }
        assert!(
            undeclared.is_empty(),
            "these models are in none of the three declared weight-acquisition states - make one true, \
             or add the model to ENV_ONLY with the reason: {undeclared:?}"
        );
    }

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
            florence2::caps::MODEL,
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
            deepseekocr2::caps::MODEL,
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

    /// A resolved [`Assembly`]'s `weights` role, as a loader-ready path -
    /// no `id`/`variant`/`provenance` beyond what the assertion below reads.
    fn assembly_with_weights(arch: &str, weights: &str) -> Assembly {
        Assembly {
            id: format!("local/{arch}"),
            arch: arch.to_string(),
            variant: None,
            roles: std::collections::BTreeMap::from([("weights".to_string(), std::path::PathBuf::from(weights))]),
            provenance: Vec::new(),
        }
    }

    /// `sam2`/`qwen3asr`/`nemotronasr` all defer their real checkpoint load
    /// past construction (`Sam2Provider::new`/[`LazyProvider`] both load lazily
    /// on the first action run - see their own doc comments) - so with the
    /// weights role coming from a resolved [`Assembly`], construction alone
    /// SUCCEEDS even for a made-up path. A regression back to `from_env!`
    /// would instead answer `Err("set BRAIN_…")` here, because the relevant
    /// variable is (deliberately, via the `remove_var` calls below) unset -
    /// that flip from `Err` to `Ok` is exactly what this asserts.
    #[test]
    fn deferred_load_providers_construct_from_the_assembly_with_no_env_set() {
        let _serial = brain_testutil::env_lock();
        for var in ["BRAIN_QWEN3ASR", "BRAIN_NEMOTRONASR", "BRAIN_SAM2_WEIGHTS"] {
            std::env::remove_var(var);
        }
        for (model, arch) in [(qwen3asr::caps::MODEL, "qwen3asr"), (nemotronasr::caps::MODEL, "nemotronasr"), (sam2::caps::MODEL, "sam2")] {
            let assembly = assembly_with_weights(arch, "/nonexistent/brain-catalog-test-weights");
            provider_from_assembly(model, &assembly).unwrap_or_else(|e| panic!("{arch}: construction from a resolved assembly must succeed (the real load is deferred): {e}"));
        }
    }

    /// `rrdbnet`'s provider loads EAGERLY at construction (unlike the three
    /// above), so a made-up path is a real, immediate load error - the
    /// regression this guards is the SAME one as above, just visible as a
    /// different `Err` rather than an `Ok`/`Err` flip: `from_env!` would
    /// answer with `BRAIN_ESRGAN_WEIGHTS`'s own "set BRAIN_…" message
    /// regardless of the assembly, which this checks is no longer the
    /// failure mode.
    #[test]
    fn rrdbnets_eager_provider_reads_the_assembly_not_env() {
        let _serial = brain_testutil::env_lock();
        std::env::remove_var("BRAIN_ESRGAN_WEIGHTS");
        let assembly = assembly_with_weights("rrdbnet", "/nonexistent/brain-catalog-test-weights");
        // `Arc<dyn Provider>` (the `Ok` side) isn't `Debug`, so `expect_err` -
        // which needs that to format its own panic message - doesn't apply.
        let err = match provider_from_assembly(rrdbnet::caps::MODEL, &assembly) {
            Err(e) => e,
            Ok(_) => panic!("a nonexistent path must fail to load, not silently succeed"),
        };
        assert!(!err.contains("unknown model"), "{err}");
        assert!(!err.contains("BRAIN_ESRGAN_WEIGHTS"), "rrdbnet provider still reads BRAIN_ESRGAN_WEIGHTS instead of the assembly: {err}");
    }
}
