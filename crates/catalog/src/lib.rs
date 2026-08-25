// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served-model catalog: **one entry per model, in one place**.
//!
//! # Why this exists
//!
//! Adding a model used to mean editing three hand-maintained lists that had no
//! link to each other:
//!
//! 1. `caps_cli::static_manifests()` - what `brain caps` lists.
//! 2. `caps_cli::build_registry()` - what `brain do` can actually run.
//! 3. `resident::build_executor()` - what is served over D-Bus/HTTP.
//!
//! Nothing checked that a model appeared in all three, and every omission is
//! silent in a different way: missing from (1) it is undiscoverable, missing
//! from (2) `brain caps <id>` answers "unknown model" for a model it had just
//! listed, missing from (3) it is invisible to every transport. That is not
//! hypothetical - `ai-forever/Real-ESRGAN` was added with a manifest, a
//! provider, a residency adapter and passing tests, and was still unreachable
//! because only (3) had been edited.
//!
//! So (1) and (2) are now DERIVED from [`models`]: the manifest and the
//! provider constructor sit in the same [`ModelEntry`] and cannot drift apart.
//! The tests below pin the invariant, including the exact failure above - 
//! every listed model must be constructible by name.
//!
//! # Why this crate carries manifest + provider, but not every residency adapter
//!
//! This crate is a `brain-cli`-independent library: the workspace's crate
//! graph is layered, `cli` sits at the top of the "serving & front-ends"
//! layer and "aggregates everything", and nothing below it may depend back
//! on it. So an in-process consumer - `brain-cli` itself, but also a
//! separate binary linking this crate directly - can enumerate and
//! construct every registered model's [`capability::Provider`] with **no
//! CLI, no D-Bus, no HTTP transport in the loop**.
//!
//! A model's static [`Manifest`] and its weight-free-to-construct
//! [`Provider`] never reference anything CLI-local, so every entry below
//! carries both. A model's **residency adapter** (the `ResidentModel`/
//! `MultiDeviceResidentModel` impl `brain serve` schedules onto a GPU/RAM/disk
//! budget) is a different story for about twenty of them: those adapters
//! (`crate::resident_sam2::Sam2Resident` and its siblings) are defined in
//! `crates/cli/src/resident_*.rs`, which - being CLI-local - this crate
//! cannot depend on without creating exactly the dependency cycle the layer
//! rule forbids. Their [`ModelEntry::resident`] is `None` here; `brain-cli`'s
//! own `catalog.rs` (a thin extension over this crate, not a copy of it)
//! patches those specific entries back in by model id after calling
//! [`models`], and appends the handful of models - `imageops`, `demo`, and
//! the three forecasters (`chronos2`/`fincast`/`kronos`) - whose manifest
//! itself is a CLI-local function, not a model crate's `caps.rs`. See that
//! file's own module doc for exactly which ids it patches and why.
//!
//! `Manifest` and `Provider` are never split this way - every model's
//! discovery and construction stay in this ONE list, which is the whole
//! point of the type. Only residency scheduling, an inherently CLI/serving
//! concern, is layered on top.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Progress, Provider};
use residency::ResidentModel;

/// A provider whose heavy weight load is deferred to the FIRST action run (and
/// cached for the provider's life). Needed because the catalog contract is that
/// provider CONSTRUCTION is cheap - [`tests::every_listed_model_is_constructible_by_name`]
/// constructs every provider, and the imaging providers all hold only a path - 
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

/// Qwen3-ASR's tuning knobs (window length, max new tokens), overridable via
/// `BRAIN_QWEN3ASR_WINDOW`/`BRAIN_QWEN3ASR_MAXNEW`. Deliberately duplicated
/// (not shared) with `crates/cli/src/resident_asr.rs`'s identical
/// `qwen_asr_tuning` - both are pure `std::env` reads with no CLI-local type
/// involved, one tunes this crate's direct `brain do` path, the other tunes
/// the CLI's residency adapter, and a shared helper this small is not worth a
/// dependency edge either way.
fn qwen_asr_tuning() -> (f32, usize) {
    let window_secs = std::env::var("BRAIN_QWEN3ASR_WINDOW").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0f32);
    let max_new = std::env::var("BRAIN_QWEN3ASR_MAXNEW").ok().and_then(|s| s.parse().ok()).unwrap_or(200usize);
    (window_secs, max_new)
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

/// A residency-adapter constructor: `None` from the inner `fn` when the model's
/// weights are not configured, so the scheduler simply does not serve it.
///
/// Two shapes, because the scheduler has two genuinely different claim paths and
/// a model must be registered through exactly one of them (see
/// `residency::Executor::register_multi`'s doc for why registering a
/// multi-device model through the single-device path silently leaves one of its
/// devices unbudgeted).
pub enum ResidentCtor {
    /// An ordinary adapter: one instance, one device.
    Single(SingleCtor),
    /// An adapter whose instance occupies real bytes on SEVERAL devices at once.
    Multi(MultiCtor),
}

/// Builds a single-device adapter, or `None` when its weights are not configured.
pub type SingleCtor = fn() -> Option<Arc<dyn ResidentModel>>;

/// Builds a multi-device adapter from `build_executor`'s budgeted `(index,
/// TOTAL bytes)` GPU list and its per-card reserve - such a model has to choose
/// its own device set against genuinely usable capacity, because
/// `residency::multi::pick_devices` checks the set it names but never
/// substitutes a different one.
pub type MultiCtor = fn(&[(u32, u64)], u64) -> Option<Arc<dyn residency::multi::MultiDeviceResidentModel>>;

/// One served model. `provider` and `manifest` describe the SAME model by
/// construction - that is the whole point of the type. See the module doc for
/// why `resident` is `None` here for the models whose adapter is CLI-local.
pub struct ModelEntry {
    /// The static manifest: safe to build with no weights loaded.
    pub manifest: fn() -> Manifest,
    /// Build something runnable. `Err` carries the model's OWN "set BRAIN_…"
    /// message, so a caller never sees a generic one.
    pub provider: fn() -> Result<Arc<dyn Provider>, String>,
    /// Register with the residency scheduler, when this model has an adapter
    /// and its weights are configured. `None` from the fn means "not
    /// configured"; a `None` field means "no adapter exists yet" OR "the
    /// adapter is CLI-local and patched in by `brain-cli`'s own catalog" (see
    /// the module doc) - both read identically to a caller that only wants to
    /// know whether THIS crate can serve the model.
    pub resident: Option<ResidentCtor>,
}

/// Shorthand: a provider that needs no weights.
#[macro_export]
macro_rules! always {
    ($e:expr) => {
        || Ok(std::sync::Arc::new($e) as std::sync::Arc<dyn $crate::__reexport::Provider>)
    };
}

/// Shorthand: a provider built from env, with the model's own error message.
#[macro_export]
macro_rules! from_env {
    ($ctor:path, $msg:literal) => {
        || $ctor().map(|p| std::sync::Arc::new(p) as std::sync::Arc<dyn $crate::__reexport::Provider>).ok_or($msg.to_string())
    };
}

/// Shorthand: a single-device residency adapter built from env.
#[macro_export]
macro_rules! resident {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Single((|| $ctor().map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::ResidentModel>)) as $crate::SingleCtor))
    };
}

/// Shorthand: a MULTI-device residency adapter built from env, given
/// `build_executor`'s budgeted GPU list and per-card reserve.
#[macro_export]
macro_rules! resident_multi {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Multi((|gpus: &[(u32, u64)], reserved: u64| {
            $ctor(gpus, reserved).map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::MultiDeviceResidentModel>)
        }) as $crate::MultiCtor))
    };
}

/// Re-exports [`always!`]/[`from_env!`]/[`resident!`]/[`resident_multi!`] need
/// to resolve `Provider`/`ResidentModel`/`MultiDeviceResidentModel` from a
/// caller crate (e.g. `brain-cli`) without that caller needing its own
/// `use` of `capability`/`residency` just to invoke these macros.
#[doc(hidden)]
pub mod __reexport {
    pub use capability::Provider;
    pub use residency::multi::MultiDeviceResidentModel;
    pub use residency::ResidentModel;
}

/// Every model's static manifest + weight-free-to-construct provider, in one
/// list - the "core" catalog this crate owns. See the module doc for the
/// models this list deliberately excludes (their manifest itself is
/// CLI-local) and for why `resident` is `None` on the ~20 entries whose
/// adapter is CLI-local.
pub fn models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            manifest: s3dit::caps::manifest,
            provider: || s3dit::caps::ZImageProvider::load().map(|p| Arc::new(p) as Arc<dyn Provider>),
            resident: None, // ZImageResident::from_env is Result-shaped; registered directly in crates/cli/src/resident.rs
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
        // `crates/cli/src/resident.rs`, advertising
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
        // directly in `crates/cli/src/resident.rs`, not through this ctor).
        ModelEntry {
            manifest: qwen35moe::caps::manifest,
            provider: always!(qwen35moe::caps::Qwen35Provider::new()),
            resident: None,
        },
        // Qwen3.8-27B dense hybrid GDN/GQA decoder: same reasoning as
        // qwen35moe above (weights is a per-invocation action param; the
        // always-hot HTTP/D-Bus path is `crate::resident_qwen35::Qwen35Resident`,
        // registered directly in `crates/cli/src/resident.rs`).
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
        // residency adapter advertise ONE manifest each. Every `resident: None`
        // below is a CLI-local adapter, patched in by `brain-cli`'s own
        // `catalog.rs` - see this file's module doc.
        ModelEntry {
            manifest: sam2::caps::manifest,
            provider: from_env!(
                sam2::caps::Sam2Provider::from_env,
                "set BRAIN_SAM2_WEIGHTS to an existing sam2.1_hiera_*.pt checkpoint"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: scrfd::caps::manifest,
            provider: from_env!(
                scrfd::caps::ScrfdProvider::from_env,
                "set BRAIN_SCRFD_DIR to a directory holding scrfd_10g_bnkps.onnx"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: arcface::caps::manifest,
            provider: from_env!(
                arcface::caps::ArcFaceProvider::from_env,
                "set BRAIN_ARCFACE_DIR to a directory holding glintr100.onnx (+ scrfd_10g_bnkps.onnx for the default aligned path)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: vqgan::caps::manifest,
            provider: from_env!(
                vqgan::caps::VqganProvider::from_env,
                "set BRAIN_VQGAN_WEIGHTS to an existing VQGAN checkpoint (or its directory)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: codeformer::caps::manifest,
            provider: from_env!(
                codeformer::caps::RestoreProvider::from_env,
                "set BRAIN_CODEFORMER_WEIGHTS to an existing codeformer.pth (or its directory)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: rrdbnet::caps::manifest,
            provider: from_env!(
                rrdbnet::caps::UpscaleProvider::from_env,
                "set BRAIN_ESRGAN_WEIGHTS to an existing RealESRGAN_x4plus.pth"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: clip::caps::manifest,
            provider: from_env!(
                clip::caps::ClipProvider::from_env,
                "set BRAIN_CLIP_DIR to a checkpoint root holding tokenizer/ (CLIP-L) and/or tokenizer_2/ (OpenCLIP-bigG)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: t5encoder::caps::manifest,
            provider: from_env!(
                t5encoder::caps::T5encoderProvider::from_env,
                "set BRAIN_T5ENCODER_DIR to a checkpoint root holding text_encoder_2/+tokenizer_2/ \
                 (flux_xxl) and/or wan/ (wan_umt5)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: sdxlunet::caps::manifest,
            provider: from_env!(
                sdxlunet::caps::SdxlProvider::from_env,
                "set BRAIN_SDXL_DIR to a released diffusers SDXL checkpoint root holding unet/"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: controlnet::caps::manifest,
            provider: from_env!(
                controlnet::caps::ControlnetProvider::from_env,
                "set BRAIN_SDXL_DIR (the backbone) and BRAIN_CONTROLNET_DIR (a released \
                 diffusers SDXL ControlNetModel checkpoint)"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: flux1::caps::manifest,
            provider: from_env!(
                flux1::caps::Flux1Provider::from_env,
                "set BRAIN_FLUX1_DIR to a released diffusers FLUX.1 checkpoint root holding transformer/"
            ),
            resident: None,
        },
        ModelEntry {
            manifest: pulid::caps::manifest,
            provider: from_env!(
                pulid::caps::PulidProvider::from_env,
                "set BRAIN_FLUX1_DIR, BRAIN_PULID_DIR, BRAIN_ARCFACE_DIR and BRAIN_CLIP_DIR"
            ),
            resident: None,
        },
        // DeepSeek-OCR: a document image in, decoded text out. Multi-file
        // checkpoint (mmproj + LM GGUF), so ONE directory variable, like
        // the face stack's and clip's. The only MULTI-device entry: its
        // vision tower runs on wgpu while its decoder runs on the CPU
        // backend, so it holds real bytes on two devices at once - see
        // `crate::resident_deepseekocr`'s header in `crates/cli`.
        ModelEntry {
            manifest: deepseek2ocr::caps::manifest,
            provider: from_env!(
                deepseek2ocr::caps::DeepseekOcrProvider::from_env,
                "set BRAIN_DEEPSEEK_OCR_DIR to a directory holding mmproj-DeepSeek-OCR-Q8_0.gguf + DeepSeek-OCR-Q8_0.gguf"
            ),
            resident: None,
        },
        // Moondream 3: an image in, text out. SigLIP ViT with overlap multi-crop
        // -> connector -> a parallel-block sparse-MoE decoder. int8 experts by
        // default, because the fp32 build is ~43 GiB and loads nowhere - see
        // `crate::resident_moondream3`'s header in `crates/cli`.
        ModelEntry {
            manifest: moondream3::caps::manifest,
            provider: always!(moondream3::caps::Moondream3Provider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: imgpipe::caps::manifest,
            provider: || Ok(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(stage_registry()))) as Arc<dyn Provider>),
            resident: None,
        },
        ModelEntry {
            manifest: qwen3tts::caps::manifest,
            provider: always!(qwen3tts::caps::TtsProvider::new()),
            resident: None,
        },
        ModelEntry {
            manifest: minimaxmusic3::caps::manifest,
            provider: always!(minimaxmusic3::caps::MinimaxMusic3Provider::new()),
            resident: None,
        },
        // CosyVoice 2/3 zero-shot voice cloning TTS. Stateless
        // (`cosyvoice::pipeline::generate` loads and drops all five
        // checkpoints per call, see its own module doc) - `SynthAction::run`
        // reads `BRAIN_COSYVOICE_*`/`BRAIN_S3TOKENIZER_V2`/`BRAIN_CAMPPLUS_DIR`
        // from the environment itself, so `always!` is correct here too.
        ModelEntry {
            manifest: cosyvoice::caps::manifest,
            provider: always!(cosyvoice::caps::CosyVoiceProvider::new()),
            resident: None,
        },
        // Speech-to-text. Discovery is weight-free (the caps manifests); the
        // direct `brain do` path wraps the model crates' eager providers in
        // [`LazyProvider`] so construction stays cheap; the residency adapters
        // are patched in by `brain-cli`'s own catalog.
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
            resident: None,
        },
        ModelEntry {
            manifest: qwen3asr::caps::manifest,
            provider: || {
                let dir = env_path("BRAIN_QWEN3ASR", "a Qwen3-ASR checkpoint dir")?;
                Ok(Arc::new(LazyProvider::new(
                    qwen3asr::caps::manifest,
                    Box::new(move || {
                        let (window_secs, max_new) = qwen_asr_tuning();
                        qwen3asr::caps::QwenAsrProvider::load(&dir, qwen3asr::config::QwenAsrConfig::qwen3_asr_1_7b(), window_secs, max_new)
                            .map(|p| Arc::new(p) as Arc<dyn Provider>)
                    }),
                )) as Arc<dyn Provider>)
            },
            resident: None,
        },
    ]
}

/// The registry `imgpipe`'s stages dispatch into.
///
/// The pipeline is a capability that COMPOSES capabilities, so it gets the
/// models whose weights are configured - and a stage whose model is unset fails
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

    /// THE DRIFT THIS FILE EXISTS TO KILL: every model listed here must be
    /// constructible by name. It may legitimately fail for want of weights - 
    /// what it must never do is answer "unknown model" for something it just
    /// advertised.
    #[test]
    fn every_listed_model_is_constructible_by_name() {
        for m in manifests() {
            match provider(&m.model) {
                Ok(_) => {}
                Err(e) => assert!(!e.contains("unknown model"), "'{}' is listed but cannot be built: {e}", m.model),
            }
        }
    }

    /// `crates/imgpipe` names its stage models by STRING, because it links no
    /// model crate. This is the other half of that decision: this crate sees
    /// both, so it asserts the strings still name real catalog entries - 
    /// otherwise a renamed model would turn into a runtime "unknown model"
    /// from inside a pipeline run, which is the worst place to find out.
    #[test]
    fn imgpipe_stage_ids_match_the_catalog() {
        let ids: std::collections::HashSet<String> = manifests().into_iter().map(|m| m.model).collect();
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
