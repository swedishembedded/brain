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
//! `build_executor` still registers the text-generation LLMs (gpt/glm/qwen/
//! lfm), the image-generation stacks (z-image/flux2 — Result-shaped or
//! multi-var ctors), omni and the mock resident from its own list; the
//! catalog carries their manifests but not their `resident` ctors. ASR
//! (nemotron/qwen-asr), forecasting (chronos2/fincast/kronos) and TTS are
//! folded in as of the F7 cleanup — `brain caps` lists them, `brain do`
//! reaches the ASR models (lazily loaded), and `brain serve` registers them
//! through [`residents`]. [`catalog_and_residency_do_not_overlap`] pins that
//! a model registered here is never also registered from `build_executor`'s
//! own list.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Progress, Provider};
use residency::ResidentModel;

/// A provider whose heavy weight load is deferred to the FIRST action run (and
/// cached for the provider's life). Needed because the catalog contract is that
/// provider CONSTRUCTION is cheap — [`tests::every_listed_model_is_constructible_by_name`]
/// constructs every provider, and the imaging providers all hold only a path —
/// while the ASR crates' providers (`NemotronProvider::load`,
/// `QwenAsrProvider::load`) load gigabytes eagerly. This wrapper gives them the
/// same construct-cheap/load-on-run shape without touching the model crates.
struct LazyProvider {
    manifest: fn() -> Manifest,
    inner: Arc<LazyInner>,
}

struct LazyInner {
    build: Box<dyn Fn() -> Result<Arc<dyn Provider>, String> + Send + Sync>,
    cell: Mutex<Option<Arc<dyn Provider>>>,
}

impl LazyInner {
    /// The loaded provider, building (once) on first use.
    fn loaded(&self) -> Result<Arc<dyn Provider>, String> {
        let mut g = self.cell.lock().map_err(|_| "lazy provider lock poisoned".to_string())?;
        if let Some(p) = &*g {
            return Ok(p.clone());
        }
        let p = (self.build)()?;
        *g = Some(p.clone());
        Ok(p)
    }
}

impl LazyProvider {
    fn new(manifest: fn() -> Manifest, build: Box<dyn Fn() -> Result<Arc<dyn Provider>, String> + Send + Sync>) -> LazyProvider {
        LazyProvider { manifest, inner: Arc::new(LazyInner { build, cell: Mutex::new(None) }) }
    }
}

impl Provider for LazyProvider {
    fn manifest(&self) -> Manifest {
        (self.manifest)()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        // The action's SPEC comes from the static manifest (no load); the load
        // happens inside run(). An action absent from the manifest is None,
        // exactly as the eager provider would answer.
        let spec = (self.manifest)().actions.into_iter().find(|a| a.name == name)?;
        Some(Arc::new(LazyAction { inner: self.inner.clone(), name: name.to_string(), spec }))
    }
}

struct LazyAction {
    inner: Arc<LazyInner>,
    name: String,
    spec: ActionSpec,
}

impl Action for LazyAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let p = self.inner.loaded()?;
        let act = p.action(&self.name).ok_or_else(|| format!("loaded provider has no action '{}'", self.name))?;
        act.run(inv, progress)
    }
}

/// A required env-var path for a catalog provider closure: `Err` with the
/// model's own "set BRAIN_…" message when unset, empty, or nonexistent.
fn env_path(var: &str, what: &str) -> Result<String, String> {
    let p = std::env::var(var).ok().filter(|p| !p.is_empty()).ok_or_else(|| format!("set {var} to {what}"))?;
    if !std::path::Path::new(&p).exists() {
        return Err(format!("{var}={p} does not exist"));
    }
    Ok(p)
}

/// One served model. `provider` and `manifest` describe the SAME model by
/// construction — that is the whole point of the type.
/// A residency-adapter constructor: `None` when the model's weights are not
/// configured, so the scheduler simply does not serve it.
pub type ResidentCtor = fn() -> Option<Arc<dyn ResidentModel>>;

pub struct ModelEntry {
    /// The static manifest: safe to build with no weights loaded.
    pub manifest: fn() -> Manifest,
    /// Build something runnable. `Err` carries the model's OWN "set BRAIN_…"
    /// message, so a caller never sees a generic one.
    pub provider: fn() -> Result<Arc<dyn Provider>, String>,
    /// Register with the residency scheduler, when this model has an adapter
    /// and its weights are configured. `None` from the fn means "not
    /// configured"; a `None` field means "no adapter exists yet".
    pub resident: Option<ResidentCtor>,
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
            manifest: s3dit::caps::manifest,
            provider: || s3dit::caps::ZImageProvider::load().map(|p| Arc::new(p) as Arc<dyn Provider>),
            resident: None, // ZImageResident::from_env is Result-shaped; see resident.rs
        },
        ModelEntry {
            manifest: flux2::caps::manifest,
            provider: always!(flux2::caps::Flux2Provider::new()),
            resident: None,
        },
        // Wan2.1 text-to-video. Like flux2, the four weight roles live in the
        // provider (from `BRAIN_WAN_*`), not in an action param, so one
        // manifest serves `brain caps`, `brain do` and the D-Bus surface. The
        // residency adapter is registered from `resident.rs` (it is one of the
        // env-gated `from_env` families that list explains), not from here.
        ModelEntry {
            manifest: wan::caps::manifest,
            provider: always!(wan::caps::WanProvider::new()),
            resident: None,
        },
        // LTX-2.5 text-to-video: a smoke-test pipeline (real VAE +
        // tiny random-weight DiT, no real text encoder yet - see
        // `ltxv::pipeline`'s module doc). `BRAIN_LTXV_VAE` lives in the
        // provider, same shape as `wan`'s four roles above; the residency
        // adapter is registered from `resident.rs`, not from here.
        ModelEntry {
            manifest: ltxv::caps::manifest,
            provider: always!(ltxv::caps::LtxvProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: qwen3::caps::manifest,
            provider: always!(qwen3::caps::QwenProvider::new()),
            resident: None,
        },
        // GLM-5.2. Same shape as qwen3 above: `weights` is a per-invocation
        // action param, so the manifest is weights-free and `brain caps` lists
        // GLM on a box with no checkpoint. The always-hot HTTP/D-Bus path is
        // `crate::resident_llm::GlmResident`, registered directly in
        // `resident.rs::build_executor` and advertising
        // `glmdsa::caps::manifest_resident` - the same definition as this one,
        // minus the `weights` param the service supplies itself.
        ModelEntry {
            manifest: glmdsa::caps::manifest,
            provider: always!(glmdsa::caps::GlmProvider::new()),
            resident: None,
        },
        // Qwen3.5-35B-A3B: like qwen3, `weights` is a per-invocation action
        // param (not baked into the Provider at construction), so this
        // manifest is genuinely weights-free -- the same reason qwen3's own
        // entry above needs no `resident` (the HTTP/D-Bus-served, always-hot
        // path is `crate::resident_qwen35moe::Qwen35Resident`, registered
        // directly in `resident.rs::build_executor`, not through this ctor).
        ModelEntry {
            manifest: qwen35moe::caps::manifest,
            provider: always!(qwen35moe::caps::Qwen35Provider::new()),
            resident: None,
        },
        // Qwen3.8-27B dense hybrid GDN/GQA decoder: same reasoning as
        // qwen35moe above (weights is a per-invocation action param; the
        // always-hot HTTP/D-Bus path is `crate::resident_qwen35::Qwen35Resident`,
        // registered directly in `resident.rs::build_executor`).
        ModelEntry {
            manifest: qwen35::caps::manifest,
            provider: always!(qwen35::caps::Qwen35Provider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: lfm2::caps::manifest,
            provider: always!(lfm2::caps::LfmProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: fastvlm::caps::manifest,
            provider: always!(fastvlm::caps::FastVlmProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: qwen3vl::caps::manifest,
            provider: always!(qwen3vl::caps::QwenVlProvider::new()),
            resident: None, // no residency adapter yet -- brain caps/brain do only, matching fastvlm's own state
        },
        ModelEntry {
            manifest: yolov8::caps::manifest,
            provider: always!(yolov8::caps::YoloProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: zipdepth::caps::manifest,
            provider: always!(zipdepth::caps::DepthProvider::new()),
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
            manifest: scrfd::caps::manifest,
            provider: from_env!(
                scrfd::caps::ScrfdProvider::from_env,
                "set BRAIN_SCRFD_DIR to a directory holding scrfd_10g_bnkps.onnx"
            ),
            resident: resident!(crate::resident_scrfd::ScrfdResident::from_env),
        },
        ModelEntry {
            manifest: arcface::caps::manifest,
            provider: from_env!(
                arcface::caps::ArcFaceProvider::from_env,
                "set BRAIN_ARCFACE_DIR to a directory holding glintr100.onnx (+ scrfd_10g_bnkps.onnx for the default aligned path)"
            ),
            resident: resident!(crate::resident_arcface::ArcFaceResident::from_env),
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
            manifest: codeformer::caps::manifest,
            provider: from_env!(
                codeformer::caps::RestoreProvider::from_env,
                "set BRAIN_CODEFORMER_WEIGHTS to an existing codeformer.pth (or its directory)"
            ),
            resident: resident!(crate::resident_restore::RestoreResident::from_env),
        },
        ModelEntry {
            manifest: rrdbnet::caps::manifest,
            provider: from_env!(
                rrdbnet::caps::UpscaleProvider::from_env,
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
            manifest: t5encoder::caps::manifest,
            provider: from_env!(
                t5encoder::caps::T5encoderProvider::from_env,
                "set BRAIN_T5ENCODER_DIR to a checkpoint root holding text_encoder_2/+tokenizer_2/ \
                 (flux_xxl) and/or wan/ (wan_umt5)"
            ),
            resident: resident!(crate::resident_t5encoder::T5encoderResident::from_env),
        },
        ModelEntry {
            manifest: sdxlunet::caps::manifest,
            provider: from_env!(
                sdxlunet::caps::SdxlProvider::from_env,
                "set BRAIN_SDXL_DIR to a released diffusers SDXL checkpoint root holding unet/"
            ),
            resident: resident!(crate::resident_sdxl::SdxlResident::from_env),
        },
        ModelEntry {
            manifest: controlnet::caps::manifest,
            provider: from_env!(
                controlnet::caps::ControlnetProvider::from_env,
                "set BRAIN_SDXL_DIR (the backbone) and BRAIN_CONTROLNET_DIR (a released \
                 diffusers SDXL ControlNetModel checkpoint)"
            ),
            resident: resident!(crate::resident_controlnet::ControlnetResident::from_env),
        },
        ModelEntry {
            manifest: flux1::caps::manifest,
            provider: from_env!(
                flux1::caps::Flux1Provider::from_env,
                "set BRAIN_FLUX1_DIR to a released diffusers FLUX.1 checkpoint root holding transformer/"
            ),
            resident: resident!(crate::resident_flux1::Flux1Resident::from_env),
        },
        ModelEntry {
            manifest: pulid::caps::manifest,
            provider: from_env!(
                pulid::caps::PulidProvider::from_env,
                "set BRAIN_FLUX1_DIR, BRAIN_PULID_DIR, BRAIN_ARCFACE_DIR and BRAIN_CLIP_DIR"
            ),
            resident: resident!(crate::resident_pulid::PulidResident::from_env),
        },
        // DeepSeek-OCR: a document image in, decoded text out. Multi-file
        // checkpoint (mmproj + LM GGUF), so ONE directory variable, like
        // the face stack's and clip's. CPU-resident by declaration - see
        // `crate::resident_deepseekocr`'s header.
        ModelEntry {
            manifest: deepseek2ocr::caps::manifest,
            provider: from_env!(
                deepseek2ocr::caps::DeepseekOcrProvider::from_env,
                "set BRAIN_DEEPSEEK_OCR_DIR to a directory holding mmproj-DeepSeek-OCR-Q8_0.gguf + DeepSeek-OCR-Q8_0.gguf"
            ),
            resident: resident!(crate::resident_deepseekocr::DeepseekOcrResident::from_env),
        },
        ModelEntry {
            manifest: imgpipe::caps::manifest,
            provider: || Ok(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(stage_registry()))) as Arc<dyn Provider>),
            resident: None,
        },
        ModelEntry {
            manifest: qwen3tts::caps::manifest,
            provider: always!(qwen3tts::caps::TtsProvider::new()),
            resident: resident!(crate::resident_tts::TtsResident::from_env),
        },
        ModelEntry {
            manifest: minimaxmusic3::caps::manifest,
            provider: always!(minimaxmusic3::caps::MinimaxMusic3Provider::new()),
            resident: resident!(crate::resident_minimaxmusic3::MinimaxMusic3Resident::from_env),
        },
        // CosyVoice 2/3 zero-shot voice cloning TTS. Stateless like
        // MiniMax Music 3 above (`cosyvoice::pipeline::generate` loads and
        // drops all five checkpoints per call, see its own module doc) -
        // `SynthAction::run` reads `BRAIN_COSYVOICE_*`/`BRAIN_S3TOKENIZER_V2`/
        // `BRAIN_CAMPPLUS_DIR` from the environment itself, so `always!` is
        // correct here too.
        ModelEntry {
            manifest: cosyvoice::caps::manifest,
            provider: always!(cosyvoice::caps::CosyVoiceProvider::new()),
            resident: resident!(crate::resident_cosyvoice::CosyVoiceResident::from_env),
        },
        // Speech-to-text. Discovery is weight-free (the caps manifests); the
        // direct `brain do` path wraps the model crates' eager providers in
        // [`LazyProvider`] so construction stays cheap; the residency adapters
        // are the same ones `brain serve` used to register from its own list.
        ModelEntry {
            manifest: nemotronasr::caps::manifest,
            provider: || {
                let dir = env_path("BRAIN_NEMOTRONASR", "a Nemotron 3.5 ASR checkpoint dir")?;
                Ok(Arc::new(LazyProvider::new(
                    nemotronasr::caps::manifest,
                    Box::new(move || {
                        nemotronasr::caps::NemotronProvider::load(&dir, nemotronasr::NemotronConfig::nemotron_3_5_asr_0_6b())
                            .map(|p| Arc::new(p) as Arc<dyn Provider>)
                    }),
                )) as Arc<dyn Provider>)
            },
            resident: resident!(crate::resident_asr::NemotronResident::from_env),
        },
        ModelEntry {
            manifest: qwen3asr::caps::manifest,
            provider: || {
                let dir = env_path("BRAIN_QWEN3ASR", "a Qwen3-ASR checkpoint dir")?;
                Ok(Arc::new(LazyProvider::new(
                    qwen3asr::caps::manifest,
                    Box::new(move || {
                        let (window_secs, max_new) = crate::resident_asr::qwen_asr_tuning();
                        qwen3asr::caps::QwenAsrProvider::load(&dir, qwen3asr::config::QwenAsrConfig::qwen3_asr_1_7b(), window_secs, max_new)
                            .map(|p| Arc::new(p) as Arc<dyn Provider>)
                    }),
                )) as Arc<dyn Provider>)
            },
            resident: resident!(crate::resident_asr::QwenAsrResident::from_env),
        },
        // Time-series forecasting. Discoverable (`brain caps`) and served
        // (`brain serve`, via the resident ctors) — but with no direct
        // `brain do` provider yet: the forecast run logic lives in the
        // residency instances (NPU/device placement included), so the provider
        // says exactly how to reach the model instead of "unknown model".
        ModelEntry {
            manifest: crate::resident_forecast::chronos2_manifest,
            provider: || {
                Err("chronos-2 has no direct `brain do` provider yet — serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_CHRONOS2 set".to_string())
            },
            resident: resident!(crate::resident_forecast::Chronos2Resident::from_env),
        },
        ModelEntry {
            manifest: crate::resident_forecast::fincast_manifest,
            provider: || {
                Err("fincast has no direct `brain do` provider yet — serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_FINCAST set".to_string())
            },
            resident: resident!(crate::resident_forecast::FincastResident::from_env),
        },
        ModelEntry {
            manifest: crate::resident_forecast::kronos_manifest,
            provider: || {
                Err("kronos has no direct `brain do` provider yet — serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER set".to_string())
            },
            resident: resident!(crate::resident_forecast::KronosResident::from_env),
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

    /// Every residency adapter reachable from this file advertises an id this
    /// file also lists, so a model cannot be schedulable but undiscoverable.
    ///
    /// The name and doc used to say these two sets are *disjoint*, which is the
    /// opposite of what the assertion below has always checked - and disjoint
    /// is not even true: `residents()` is derived from `models()`, so every id
    /// it yields is a catalog id by construction. Read literally, the old name
    /// described a property that, if it held, would mean no catalog entry could
    /// carry a `resident` at all. The env-gated adapters registered directly in
    /// `resident.rs::build_executor` (gpt2, qwen, lfm2, flux2, wan, ltxv, ...)
    /// are a genuinely separate list that this test does not reach.
    #[test]
    fn every_residency_adapter_here_is_also_listed_here() {
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
        assert_eq!(imgpipe::UPSCALE_MODEL, rrdbnet::caps::MODEL);
        assert_eq!(imgpipe::RESTORE_MODEL, codeformer::caps::MODEL);
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
