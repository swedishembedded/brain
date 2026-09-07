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

use capability::{Action, ActionResult, ActionSpec, Assembly, Invocation, Manifest, Progress, Provider};
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

/// Builds a multi-device adapter from an already-resolved [`Assembly`] (see
/// [`ModelEntry::provider`]'s doc - the same contract, for the multi-device
/// claim path) plus `build_executor`'s budgeted `(index, TOTAL bytes)` GPU
/// list and its per-card reserve - such a model has to choose its own device
/// set against genuinely usable capacity, because `residency::multi::
/// pick_devices` checks the set it names but never substitutes a different
/// one.
pub type MultiCtor = fn(&Assembly, &[(u32, u64)], u64) -> Option<Arc<dyn residency::multi::MultiDeviceResidentModel>>;

/// One served model. `provider` and `manifest` describe the SAME model by
/// construction - that is the whole point of the type. See the module doc for
/// why `resident` is `None` here for the models whose adapter is CLI-local.
pub struct ModelEntry {
    /// The static manifest: safe to build with no weights loaded.
    pub manifest: fn() -> Manifest,
    /// Build something runnable from an already-resolved [`Assembly`]. `Err`
    /// carries the model's OWN "set BRAIN_…" message, so a caller never sees
    /// a generic one. FLUX.2's entry was the first to build its `Provider`
    /// from it (`crates/cli/src/catalog.rs`'s `resolved_assembly_for` is what
    /// actually resolves one for it and for every architecture migrated onto
    /// the resolver since); every entry not yet migrated still ignores the
    /// argument and reads `BRAIN_*` env vars instead, per the module doc.
    pub provider: fn(&Assembly) -> Result<Arc<dyn Provider>, String>,
    /// Register with the residency scheduler, when this model has an adapter
    /// and its weights are configured. `None` from the fn means "not
    /// configured"; a `None` field means "no adapter exists yet" OR "the
    /// adapter is CLI-local and patched in by `brain-cli`'s own catalog" (see
    /// the module doc) - both read identically to a caller that only wants to
    /// know whether THIS crate can serve the model.
    pub resident: Option<ResidentCtor>,
}

/// Shorthand: a provider that needs no weights. Ignores the [`Assembly`]
/// [`ModelEntry::provider`] is called with - nothing built this way reads
/// one yet.
#[macro_export]
macro_rules! always {
    ($e:expr) => {
        |_assembly: &$crate::__reexport::Assembly| Ok(std::sync::Arc::new($e) as std::sync::Arc<dyn $crate::__reexport::Provider>)
    };
}

/// Shorthand: a provider built from env, with the model's own error message.
/// Ignores the [`Assembly`] [`ModelEntry::provider`] is called with - still
/// env-only, same as before this parameter existed.
#[macro_export]
macro_rules! from_env {
    ($ctor:path, $msg:literal) => {
        |_assembly: &$crate::__reexport::Assembly| $ctor().map(|p| std::sync::Arc::new(p) as std::sync::Arc<dyn $crate::__reexport::Provider>).ok_or($msg.to_string())
    };
}

/// Shorthand: a single-device residency adapter built from env.
#[macro_export]
macro_rules! resident {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Single((|| $ctor().map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::ResidentModel>)) as $crate::SingleCtor))
    };
}

/// Shorthand: a MULTI-device residency adapter built from an already-resolved
/// [`Assembly`](crate::__reexport::Assembly), given `build_executor`'s
/// budgeted GPU list and per-card reserve.
#[macro_export]
macro_rules! resident_multi {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Multi((|assembly: &$crate::__reexport::Assembly, gpus: &[(u32, u64)], reserved: u64| {
            $ctor(assembly, gpus, reserved).map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::MultiDeviceResidentModel>)
        }) as $crate::MultiCtor))
    };
}

/// Re-exports [`always!`]/[`from_env!`]/[`resident!`]/[`resident_multi!`] need
/// to resolve `Provider`/`ResidentModel`/`MultiDeviceResidentModel` from a
/// caller crate (e.g. `brain-cli`) without that caller needing its own
/// `use` of `capability`/`residency` just to invoke these macros.
#[doc(hidden)]
pub mod __reexport {
    pub use capability::{Assembly, Provider};
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
        // Z-Image: the provider builds its weight paths from the resolved
        // Assembly (`s3dit::pipeline::Paths::from_assembly`) instead of
        // `BRAIN_S3DIT_*`, the same migration `flux2`/`wan` already went
        // through.
        ModelEntry {
            manifest: s3dit::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = s3dit::pipeline::Paths::from_assembly(assembly)?;
                Ok(Arc::new(s3dit::caps::ZImageProvider::from_paths(paths)) as Arc<dyn Provider>)
            },
            resident: None, // ZImageResident::from_env is Result-shaped; registered directly in crates/cli/src/resident.rs
        },
        // FLUX.2 is the first entry whose provider actually reads the
        // Assembly it is called with - `flux2::pipeline::Paths::from_assembly`
        // instead of the `BRAIN_FLUX2_*` variables every other entry here
        // still uses (see the module doc: only this one architecture is
        // migrated to the resolver in this stage).
        ModelEntry {
            manifest: flux2::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = flux2::pipeline::Paths::from_assembly(assembly)?;
                Ok(Arc::new(flux2::caps::Flux2Provider::new(paths)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        // Wan2.1 text-to-video. Like flux2, the provider builds its weight
        // paths from the resolved Assembly (`wan::pipeline::Paths::
        // from_assembly`) instead of `BRAIN_WAN_*`. The residency adapter is
        // registered from `resident.rs` (it is one of the env-gated
        // `from_env` families that list explains), not from here.
        ModelEntry {
            manifest: wan::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = wan::pipeline::Paths::from_assembly(assembly)?;
                Ok(Arc::new(wan::caps::WanProvider::from_paths(paths)) as Arc<dyn Provider>)
            },
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
        // Qwen3.8-27B dense hybrid GDN/GQA decoder: `weights`/`tokenizer` are
        // per-invocation action params, resolved through
        // `qwen35::spec::Qwen35Spec` instead of `BRAIN_QWEN35_{WEIGHTS,
        // TOKENIZER}` (see that module's doc); the always-hot HTTP/D-Bus path
        // is `crate::resident_qwen35::Qwen35Resident`, registered directly in
        // `crates/cli/src/resident.rs`.
        ModelEntry {
            manifest: qwen35::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned());
                let tokenizer = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(qwen35::caps::Qwen35Provider::new().with_defaults(weights, tokenizer)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        ModelEntry {
            manifest: lfm2::caps::manifest,
            provider: always!(lfm2::caps::LfmProvider::new()),
            resident: None,
        },
        // FastVLM: `weights` is resolved through `fastvlm::spec::FastvlmSpec`
        // (see that module's doc) instead of `BRAIN_FASTVLM_WEIGHTS` -
        // `Assembly::roles["weights"]` when the caller resolved one, `None`
        // otherwise (a caller that still names `weights` per request, or has
        // not scanned a models directory at all).
        ModelEntry {
            manifest: fastvlm::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(fastvlm::caps::FastVlmProvider::new(weights)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        ModelEntry {
            manifest: llava::caps::manifest,
            provider: always!(llava::caps::LlavaProvider::new()),
            resident: None,
        },
        // `weights` is resolved through `qwen3vl::spec::Qwen3VlSpec` (see that
        // module's doc) instead of `BRAIN_QWEN3VL_WEIGHTS` - same shape as
        // fastvlm's own entry above.
        ModelEntry {
            manifest: qwen3vl::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(qwen3vl::caps::QwenVlProvider::new(weights)) as Arc<dyn Provider>)
            },
            resident: None, // qwen3vl's own residency adapter (crates/cli/src/resident_qwen3vl.rs) is CLI-local and out of this migration's scope
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
        // SUPIR photo-realistic restoration: a frozen SDXL backbone
        // (BRAIN_SDXL_DIR, same layout `sdxlunet`/`controlnet` load) plus its
        // own 1.24B GLVControl trunk + 12 adaptors (BRAIN_SUPIR_DIR - a
        // delta checkpoint file, or a directory holding exactly one). No
        // `default_ref`/auto-fetch - the SUPIR weights carry a
        // non-commercial licence (see `supir`'s own crate doc). Optional
        // LLaVA auto-captioning dispatches through `supir_registry` below,
        // not a direct dependency - see `supir::caps`'s own module doc.
        ModelEntry {
            manifest: supir::caps::manifest,
            provider: |_assembly: &Assembly| {
                let paths = supir::pipeline::Paths::from_env()?;
                if !std::path::Path::new(&paths.backbone_root).join("unet").exists() {
                    return Err(format!("supir: {} holds no unet/", paths.backbone_root));
                }
                Ok(Arc::new(supir::caps::RestoreProvider::with_registry(paths.backbone_root, paths.supir_ckpt, Arc::new(supir_registry()))) as Arc<dyn Provider>)
            },
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
        // `dir` is resolved through `deepseek2ocr::spec::Deepseek2ocrSpec`
        // (see that module's doc) instead of `BRAIN_DEEPSEEK_OCR_DIR`.
        ModelEntry {
            manifest: deepseek2ocr::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = deepseek2ocr::spec::dir_from_assembly(assembly)?;
                deepseek2ocr::caps::DeepseekOcrProvider::new(dir).map(|p| Arc::new(p) as Arc<dyn Provider>).ok_or_else(|| "deepseek-ocr: resolved dir does not hold both shipped GGUFs".to_string())
            },
            resident: None,
        },
        // Moondream 3: an image in, text out. SigLIP ViT with overlap multi-crop
        // -> connector -> a parallel-block sparse-MoE decoder. int8 experts by
        // default, because the fp32 build is ~43 GiB and loads nowhere - see
        // `crate::resident_moondream3`'s header in `crates/cli`.
        // `dir` is resolved through `moondream3::spec::Moondream3Spec` (see
        // that module's doc) instead of `BRAIN_MOONDREAM3_WEIGHTS`.
        ModelEntry {
            manifest: moondream3::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(moondream3::caps::Moondream3Provider::new().with_default_dir(dir)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        ModelEntry {
            manifest: imgpipe::caps::manifest,
            provider: |_assembly: &Assembly| Ok(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(stage_registry()))) as Arc<dyn Provider>),
            resident: None,
        },
        ModelEntry {
            manifest: qwen3tts::caps::manifest,
            provider: always!(qwen3tts::caps::TtsProvider::new()),
            resident: None,
        },
        // MiniMax Music 3. Like flux2, its six roles live in the provider
        // (resolved through `minimaxmusic3::spec::MinimaxMusic3Spec` from
        // the Assembly this entry is called with), not `BRAIN_MINIMAXMUSIC3_*`
        // - the residency adapter (registered from `resident.rs`, not from
        // here) still reads those env vars.
        ModelEntry {
            manifest: minimaxmusic3::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = minimaxmusic3::generate::Paths::from_assembly(assembly)?;
                Ok(Arc::new(minimaxmusic3::caps::MinimaxMusic3Provider::new(paths)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        // CosyVoice 2/3 zero-shot voice cloning TTS. `llm`/`flow`/`hift`/
        // `tokenizer` come from `cosyvoice::spec::CosyVoiceSpec` via the
        // resolved Assembly this entry is called with (`s3tokenizer`/
        // `campplus` are still separate, not-yet-migrated architectures -
        // `CosyVoicePaths::from_assembly` reads those two from their own env
        // vars). `cosyvoice::pipeline::generate` still loads and drops all
        // five checkpoints per call (see its own module doc), so the bound
        // `paths` this provider holds cost nothing to keep between calls.
        ModelEntry {
            manifest: cosyvoice::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = cosyvoice::pipeline::CosyVoicePaths::from_assembly(assembly)?;
                Ok(Arc::new(cosyvoice::caps::CosyVoiceProvider::new(paths)) as Arc<dyn Provider>)
            },
            resident: None,
        },
        // Speech-to-text. Discovery is weight-free (the caps manifests); the
        // direct `brain do` path wraps the model crates' eager providers in
        // [`LazyProvider`] so construction stays cheap; the residency adapters
        // are patched in by `brain-cli`'s own catalog.
        ModelEntry {
            manifest: nemotronasr::caps::manifest,
            provider: |_assembly: &Assembly| {
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
            provider: |_assembly: &Assembly| {
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
        // would build providers nobody asked for (some load weights). None of
        // sam2/codeformer/rrdbnet reads its Assembly argument yet (all three
        // are still `from_env!`), so an empty placeholder is exactly as good
        // as a real one here.
        if [imgpipe::SEGMENT_MODEL, imgpipe::RESTORE_MODEL, imgpipe::UPSCALE_MODEL].contains(&id.as_str()) {
            if let Ok(p) = (e.provider)(&empty_assembly()) {
                inner.register(p);
            }
        }
    }
    inner
}

/// A placeholder [`Assembly`] for a caller that has none - every entry but
/// FLUX.2's own ignores the argument entirely (see [`ModelEntry::provider`]'s
/// doc), so this stands in wherever no real, resolver-built one is in hand
/// yet.
fn empty_assembly() -> Assembly {
    Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }
}

/// The registry SUPIR's optional caption auto-fill dispatches
/// [`supir::caps::LLAVA_MODEL`] through, for the direct `brain do`/D-Bus-via-
/// provider path (`crates/cli`'s own residency adapter,
/// `resident_supir.rs`, builds an equivalent registry for the served path,
/// since it cannot reach this crate's private helpers - see that file's doc).
/// A stub `LlavaProvider` costs nothing to construct (it loads weights lazily
/// per call, same as every other captioner in the tree), so this is built
/// unconditionally rather than gated on `BRAIN_LLAVA_WEIGHTS` being set - an
/// unset checkpoint just means the eventual `caption` call fails with
/// llava's own clean error, same as calling it directly would.
fn supir_registry() -> capability::Registry {
    let mut reg = capability::Registry::new();
    reg.register(Arc::new(llava::caps::LlavaProvider::new()));
    reg
}

/// Every model's static manifest, for `brain caps`.
pub fn manifests() -> Vec<Manifest> {
    models().into_iter().map(|e| (e.manifest)()).collect()
}

/// Every model's manifest as an **off-machine** consumer must see it:
/// [`manifests`] with every host-resolved param projected out
/// ([`capability::Manifest::for_serving`]).
///
/// [`manifests`] is the LOCAL surface - `brain caps`/`brain do`, run by
/// somebody standing on the machine that holds the weights, who can legitimately
/// pass `weights=/path/to/checkpoint.safetensors`. Anything that describes this
/// catalog to a caller somewhere ELSE - a scheduler placing work on a machine it
/// has never seen, a graph editor in a browser, any RPC surface - must use this
/// one instead. There is no path a remote caller could name that would be valid
/// on whichever host eventually runs the action, so it must never be asked; the
/// host answers from its own `BRAIN_*` environment at
/// [`capability::ActionSpec::validate`] time, exactly as `brain serve`'s
/// resident models already do.
pub fn serving_manifests() -> Vec<Manifest> {
    manifests().into_iter().map(Manifest::for_serving).collect()
}

/// Build a runnable provider for `model`, or say why not.
///
/// Calls the entry's provider with an [`empty_assembly`]: every entry but
/// FLUX.2's own ignores the argument (see [`ModelEntry::provider`]'s doc), so
/// this stays the plain by-name lookup every existing caller already has. A
/// caller holding a real, resolver-built [`Assembly`] for an architecture
/// that reads it uses the entry's `provider` fn directly instead.
pub fn provider(model: &str) -> Result<Arc<dyn Provider>, String> {
    for e in models() {
        if (e.manifest)().model == model {
            return (e.provider)(&empty_assembly());
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

    /// Param names that mean "a checkpoint's location on some filesystem".
    /// A remote caller shares no filesystem with the machine that will run the
    /// action, so none of these may ever appear on the served surface.
    const WEIGHT_LOCATION_PARAMS: &[&str] =
        &["weights", "weights_dir", "weights_path", "tokenizer", "ckpt", "checkpoint", "checkpoint_path", "model_path", "safetensors_path"];

    /// THE REGRESSION THIS EXISTS FOR: brain's catalog manifest is what an
    /// off-machine consumer -- a scheduler placing work, a graph editor
    /// listing what a node can do -- builds its own node/action list from.
    /// Every model that took its checkpoint as an ordinary
    /// action param therefore published a REQUIRED "path to a .safetensors
    /// checkpoint" field to a workflow author who has no idea which machine
    /// will run the graph - a question with no correct answer, at the wrong
    /// layer entirely. The location is the HOST's fact, resolved from that
    /// machine's own `BRAIN_*` variable at activation time; [`serving_manifests`]
    /// is where that guarantee is kept, so it is pinned here over the REAL
    /// catalog rather than a hand-written sample.
    #[test]
    fn no_served_manifest_asks_a_remote_caller_where_the_weights_are() {
        for m in serving_manifests() {
            for a in &m.actions {
                for p in &a.params {
                    assert!(
                        !WEIGHT_LOCATION_PARAMS.contains(&p.name.as_str()),
                        "served manifest '{}' action '{}' still advertises '{}' - mark it \
                         `.host_env(\"BRAIN_…\")` in that model's caps.rs so the host resolves it",
                        m.model,
                        a.name,
                        p.name
                    );
                }
            }
        }
    }

    /// The other side of the same coin, and what makes the test above mean
    /// something: the LOCAL surface (`brain caps`/`brain do`, run by somebody
    /// standing on the machine that holds the weights) still offers the
    /// override. If this ever went empty, the test above would be passing
    /// vacuously.
    #[test]
    fn the_local_surface_still_offers_an_explicit_checkpoint_override() {
        let overridable: Vec<String> = manifests()
            .iter()
            .flat_map(|m| m.actions.iter().flat_map(|a| a.params.iter()).map(move |p| (m.model.clone(), p)))
            .filter(|(_, p)| WEIGHT_LOCATION_PARAMS.contains(&p.name.as_str()))
            .map(|(model, p)| format!("{model}:{}", p.name))
            .collect();
        assert!(overridable.len() > 5, "the local surface lost its checkpoint overrides: {overridable:?}");
        for m in manifests() {
            for a in &m.actions {
                for p in &a.params {
                    if WEIGHT_LOCATION_PARAMS.contains(&p.name.as_str()) {
                        // `host_env` (one literal env var) and `host_resolved`
                        // (a richer host-side answer, e.g. a model-store
                        // resolver scan - see `ParamSpec::host_resolved`'s own
                        // doc) are the two ways a param declares "the host
                        // answers this, never a remote caller".
                        assert!(
                            p.host_env.is_some() || p.host_resolved,
                            "'{}':'{}' takes '{}' as a plain param - it will be published to every \
                             remote caller until it declares `.host_env(\"BRAIN_…\")` or `.host_resolved()`",
                            m.model,
                            a.name,
                            p.name
                        );
                    }
                }
            }
        }
    }

    /// The projection is total: nothing host-resolved survives on the served
    /// surface, whatever it happens to be called.
    #[test]
    fn the_served_surface_carries_no_host_resolved_param_at_all() {
        for m in serving_manifests() {
            for a in &m.actions {
                for p in &a.params {
                    assert!(p.host_env.is_none(), "'{}':'{}' leaked host-resolved param '{}'", m.model, a.name, p.name);
                }
            }
        }
    }

    /// ...and it removes ONLY that. A projection that quietly dropped a real
    /// per-request knob would make every served model subtly less capable than
    /// the same model run locally, which is exactly the drift `manifest_resident`
    /// was written to prevent in the first place.
    #[test]
    fn the_served_surface_keeps_every_param_that_is_not_host_resolved() {
        for (full, served) in manifests().into_iter().zip(serving_manifests()) {
            assert_eq!(full.model, served.model);
            assert_eq!(full.actions.len(), served.actions.len(), "'{}' lost an action", full.model);
            for (fa, sa) in full.actions.iter().zip(&served.actions) {
                let expect: Vec<&str> = fa.params.iter().filter(|p| p.host_env.is_none() && !p.host_resolved).map(|p| p.name.as_str()).collect();
                let got: Vec<&str> = sa.params.iter().map(|p| p.name.as_str()).collect();
                assert_eq!(got, expect, "'{}':'{}' params changed beyond the host-resolved ones", full.model, fa.name);
                assert_eq!(sa.inputs.len(), fa.inputs.len(), "'{}':'{}' lost an input", full.model, fa.name);
                assert_eq!(sa.outputs.len(), fa.outputs.len(), "'{}':'{}' lost an output", full.model, fa.name);
            }
        }
    }

    /// A host-resolved param must name a real `BRAIN_*` variable: the whole
    /// mechanism is "the host answers from its environment", and a param that
    /// names nothing would silently become unanswerable the moment it stopped
    /// being advertised.
    #[test]
    fn every_host_resolved_param_names_a_brain_environment_variable() {
        for m in manifests() {
            for a in &m.actions {
                for p in &a.params {
                    if let Some(var) = &p.host_env {
                        assert!(var.starts_with("BRAIN_"), "'{}':'{}' param '{}' resolves from '{var}', which is not a BRAIN_* variable", m.model, a.name, p.name);
                        // An environment variable is a string; filling a
                        // non-`Str` param from one would hand the action a
                        // value of the wrong JSON type.
                        assert_eq!(p.ty, capability::ParamType::Str, "'{}':'{}' param '{}' is host-resolved but not a Str", m.model, a.name, p.name);
                    }
                }
            }
        }
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
