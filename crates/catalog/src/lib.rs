// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served-model catalog: **one entry per model, in one place**.
//!
//! # Why this exists
//!
//! Adding a model used to mean editing three hand-maintained lists that had no
//! link to each other:
//!
//! 1. a static manifest list - what `brain caps` lists.
//! 2. a provider registry - what `brain do` can actually run.
//! 3. the serving executor's registrations - what is served over D-Bus/HTTP.
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
//! # One entry, three consumers
//!
//! A model's manifest, its weight-free provider and its residency adapter all
//! sit in the same [`ModelEntry`], so the three things a process does with a
//! model (list it, run it, schedule it onto a GPU/RAM/disk budget) cannot
//! disagree about which models exist. The adapters themselves
//! (`resident_*.rs`, `QwenResident` and its siblings) live in this crate for
//! that reason: an adapter kept in a binary crate is invisible to every other
//! embedder, which then cannot serve the model under a memory budget at all.
//!
//! [`residents`] and [`multi_residents`] turn the entries into the adapters a
//! serving process registers with its `residency::Executor`.
//!
//! The crate depends on nothing CLI-local: the workspace's crate graph is
//! layered, `brain-cli` sits at the top and nothing below it may depend back.
//! Weight resolution for a served model goes through `loader::served`, and
//! assembling an executor (budgets, discovery of a models directory) is
//! `crates/serving`'s job.

pub mod adapter_release;
pub mod demo;
pub mod imageops;
pub mod resident_arcface;
pub mod resident_asr;
pub mod resident_clip;
pub mod resident_controlnet;
pub mod resident_cosyvoice;
pub mod resident_deepseekocr;
pub mod resident_deepseekocr2;
pub mod resident_deepseekvl;
pub mod resident_depth;
pub mod resident_florence2;
pub mod resident_flux1;
pub mod resident_flux2;
pub mod resident_forecast;
pub mod resident_januspro;
pub mod resident_horizon;
pub mod resident_lfm;
pub mod resident_llm;
pub mod resident_ltxv;
pub mod resident_minimaxmusic3;
pub mod resident_mock;
pub mod resident_moondream3;
pub mod resident_omni;
pub mod resident_pulid;
pub mod resident_qwen35;
pub mod resident_qwen35moe;
pub mod resident_qwen3vl;
pub mod resident_restore;
pub mod resident_sam2;
pub mod resident_scrfd;
pub mod resident_sdxl;
pub mod resident_splat;
pub mod resident_supir;
pub mod resident_t5encoder;
pub mod resident_tts;
pub mod resident_upscale;
pub mod resident_wan;
pub mod resident_worldmirror2;
pub mod resident_yolo;
pub mod resident_zimage;
mod serving;

pub use serving::{multi_residents, residents};

use std::path::Path;
use std::sync::{Arc, Mutex};

use brain_modelstore::resolve::{describe_ambiguity, describe_missing, ArchSpec, Resolution};
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
    ///
    /// `lock_resident`, not `lock()`: a panic inside the (multi-gigabyte)
    /// build would otherwise poison this cell and make the model permanently
    /// unreachable in a process that is otherwise fine - and this cell sits in
    /// the long-lived registry every in-process consumer holds. A poisoned
    /// cell resets to "not loaded" and the next caller builds again.
    fn loaded(&self) -> Result<Arc<dyn Provider>, String> {
        let mut g = capability::lock_resident(&self.cell);
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
///
/// The argument is the serving process's already-resolved models directory
/// (its own `--models-dir`, else the resolver's documented fallback), so an
/// adapter that resolves its weights through the model store reads the SAME
/// store as every other resolver in that process. An adapter configured
/// purely from its own environment variables ignores it ([`resident!`]); one
/// that consults the store takes it ([`resident_in_store!`]).
pub type SingleCtor = fn(Option<&Path>) -> Option<Arc<dyn ResidentModel>>;

/// Builds a multi-device adapter from an already-resolved [`Assembly`] (see
/// [`ModelEntry::provider`]'s doc - the same contract, for the multi-device
/// claim path) plus `build_executor`'s budgeted `(index, TOTAL bytes)` GPU
/// list and its per-card reserve - such a model has to choose its own device
/// set against genuinely usable capacity, because `residency::multi::
/// pick_devices` checks the set it names but never substitutes a different
/// one.
///
/// Returns every model the checkpoint serves: one, or the base plus the
/// fine-tunes stored beside it, each its own model id. Empty when none can be
/// placed.
pub type MultiCtor = fn(&Assembly, &[(u32, u64)], u64) -> Vec<Arc<dyn residency::multi::MultiDeviceResidentModel>>;

/// An architecture's resolver spec, as an entry NAMES it: `(arch name,
/// spec)`. A `&'static` reference rather than a `Box`, because every
/// [`ArchSpec`] in the tree is a unit struct - an entry names the
/// architecture it belongs to, it does not own one.
pub type ArchRef = (&'static str, &'static dyn ArchSpec);

/// One served model. `provider` and `manifest` describe the SAME model by
/// construction - that is the whole point of the type.
pub struct ModelEntry {
    /// The static manifest: safe to build with no weights loaded.
    pub manifest: fn() -> Manifest,
    /// Build something runnable from an already-resolved [`Assembly`]. `Err`
    /// carries the model's OWN "set BRAIN_…" message, so a caller never sees
    /// a generic one. FLUX.2's entry was the first to build its `Provider`
    /// from it; ~26 entries read one today, single `weights`-role
    /// architectures via [`Assembly::role_path`] included, and each of those
    /// names its architecture in [`ModelEntry::spec`] so [`provider`]
    /// resolves a real assembly to hand it. An entry with no `spec` ignores
    /// the argument and reads `BRAIN_*` env vars instead, per the module doc.
    ///
    /// This fn may therefore assume its argument is either a resolved
    /// assembly or an empty one; when it is empty and a needed role is
    /// absent, the resulting role-shaped `Err` is [`provider`]'s to
    /// translate, never a by-name caller's to read.
    pub provider: fn(&Assembly) -> Result<Arc<dyn Provider>, String>,
    /// The model-store architecture this entry's weights resolve through,
    /// for an entry whose `provider` READS the [`Assembly`] it is handed:
    /// `(the resolver's arch name, that architecture's spec)`. [`provider`]
    /// resolves a real one with it, so a caller that knows only the model's
    /// name reaches the same weights the resolver-backed commands do.
    ///
    /// `None` means the entry genuinely ignores its argument (an
    /// [`always!`]/[`from_env!`] entry, or one whose checkpoint is a
    /// per-invocation action param) - such an entry is built from
    /// `empty_assembly` exactly as before.
    ///
    /// It lives ON the entry rather than in a lookup table beside it because
    /// a table beside it is precisely what drifted: for a long while
    /// [`provider`] handed EVERY entry an empty assembly, including the ~21
    /// that had since been migrated onto the resolver, so each of them
    /// answered a by-name caller with a missing-role complaint about an
    /// internal data structure instead of constructing.
    pub spec: Option<ArchRef>,
    /// Register with the residency scheduler, when this model has an adapter
    /// and its weights are configured. `None` from the fn means "not
    /// configured"; a `None` field means the model is not scheduled through
    /// this list: either it has no adapter yet, or the serving executor
    /// registers it directly because its adapter needs more than a models
    /// directory to be built (a resolved [`Assembly`], a budget).
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
/// Ignores the [`Assembly`] [`ModelEntry::provider`] is called with - an
/// entry using this one is, by definition, not resolver-migrated, so its
/// [`ModelEntry::spec`] is `None` and [`provider`] never resolves for it.
#[macro_export]
macro_rules! from_env {
    ($ctor:path, $msg:literal) => {
        |_assembly: &$crate::__reexport::Assembly| $ctor().map(|p| std::sync::Arc::new(p) as std::sync::Arc<dyn $crate::__reexport::Provider>).ok_or($msg.to_string())
    };
}

/// The DIRECTORY holding a role's resolved artifact.
///
/// A spec's role resolves to the file the resolver actually classified (an
/// `.onnx` graph, a `.pth` checkpoint), but a provider that loads SEVERAL
/// released files by their published names - antelopev2's detector and
/// embedder graphs, say - is constructed from the directory that holds them.
/// This is the one conversion between the two, so a provider never re-derives
/// it and a role never has to be declared twice to mean both.
pub fn role_dir(assembly: &Assembly, role: &str) -> Result<String, String> {
    let path = assembly.role_path(role)?;
    let p = std::path::Path::new(&path);
    // Already a directory (a spec whose role names one) is returned as-is.
    if p.is_dir() {
        return Ok(path);
    }
    p.parent()
        .map(|d| d.to_string_lossy().into_owned())
        .ok_or_else(|| format!("{role}: {path} has no parent directory"))
}

/// Shorthand: a single-device residency adapter built from env alone - it
/// never consults the model store, so the models directory is ignored.
#[macro_export]
macro_rules! resident {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Single(
            (|_models_dir: Option<&std::path::Path>| $ctor().map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::ResidentModel>)) as $crate::SingleCtor,
        ))
    };
}

/// Shorthand: a single-device residency adapter whose weights resolve
/// through the model store: `$ctor` takes the serving process's models
/// directory (see [`SingleCtor`]).
#[macro_export]
macro_rules! resident_in_store {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Single(
            (|models_dir: Option<&std::path::Path>| $ctor(models_dir).map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::ResidentModel>)) as $crate::SingleCtor,
        ))
    };
}

/// Shorthand: a MULTI-device residency adapter built from an already-resolved
/// [`Assembly`](crate::__reexport::Assembly), given `build_executor`'s
/// budgeted GPU list and per-card reserve.
#[macro_export]
macro_rules! resident_multi {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Multi((|assembly: &$crate::__reexport::Assembly, gpus: &[(u32, u64)], reserved: u64| {
            $ctor(assembly, gpus, reserved).map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::MultiDeviceResidentModel>).into_iter().collect()
        }) as $crate::MultiCtor))
    };
}

/// [`resident_multi!`] for an adapter whose `$ctor` returns a `Vec` of
/// residents (the base and its stored fine-tunes), each its own model id.
#[macro_export]
macro_rules! resident_multi_family {
    ($ctor:path) => {
        Some($crate::ResidentCtor::Multi((|assembly: &$crate::__reexport::Assembly, gpus: &[(u32, u64)], reserved: u64| {
            $ctor(assembly, gpus, reserved).into_iter().map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn $crate::__reexport::MultiDeviceResidentModel>).collect()
        }) as $crate::MultiCtor))
    };
}

/// Re-exports [`always!`]/[`from_env!`]/[`resident!`]/[`resident_in_store!`]/[`resident_multi!`] need
/// to resolve `Provider`/`ResidentModel`/`MultiDeviceResidentModel` from a
/// caller crate without that caller needing its own
/// `use` of `capability`/`residency` just to invoke these macros.
#[doc(hidden)]
pub mod __reexport {
    pub use capability::{Assembly, Provider};
    pub use residency::multi::MultiDeviceResidentModel;
    pub use residency::ResidentModel;
}

/// Every model's static manifest + weight-free-to-construct provider, in one
/// list, with the residency adapter where there is one.
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
            spec: Some(("s3dit", &s3dit::spec::S3ditSpec)),
            resident: None, // ZImageResident::from_env is Result-shaped; registered directly by the serving executor
        },
        // FLUX.2 was the FIRST entry whose provider reads the Assembly it is
        // called with - `flux2::pipeline::Paths::from_assembly` instead of the
        // `BRAIN_FLUX2_*` variables - and is no longer the only one: ~26
        // entries here are resolver-migrated now, each naming its own `spec`.
        ModelEntry {
            manifest: flux2::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = flux2::pipeline::Paths::from_assembly(assembly)?;
                Ok(Arc::new(flux2::caps::Flux2Provider::new(paths)) as Arc<dyn Provider>)
            },
            spec: Some(("flux2", &flux2::spec::Flux2Spec)),
            resident: None,
        },
        // Wan2.1 text-to-video. Like flux2, the provider builds its weight
        // paths from the resolved Assembly (`wan::pipeline::Paths::
        // from_assembly`) instead of `BRAIN_WAN_*`. The residency adapter is
        // registered directly by `serving::build_executor` (one of the
        // env-gated `from_env` families), not from here.
        ModelEntry {
            manifest: wan::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = wan::pipeline::Paths::from_assembly(assembly)?;
                Ok(Arc::new(wan::caps::WanProvider::from_paths(paths)) as Arc<dyn Provider>)
            },
            spec: Some(("wan", &wan::spec::WanSpec)),
            resident: None,
        },
        // LTX-2.5 text-to-video: a smoke-test pipeline (real VAE +
        // tiny random-weight DiT, no real text encoder yet - see
        // `ltxv::pipeline`'s module doc). `BRAIN_LTXV_VAE` lives in the
        // provider, same shape as `wan`'s four roles above; the residency
        // adapter is registered directly by `serving::build_executor`, not from here.
        ModelEntry {
            manifest: ltxv::caps::manifest,
            provider: always!(ltxv::caps::LtxvProvider::new()),
            spec: None,
            resident: None,
        },
        // `weights`/`tokenizer` are resolved through `qwen3::spec::Qwen3Spec`
        // (see that module's doc) instead of `BRAIN_QWEN_WEIGHTS`/
        // `BRAIN_QWEN_TOKENIZER` - same shape as `qwen35`'s own entry above.
        ModelEntry {
            manifest: qwen3::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned());
                let tokenizer = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(qwen3::caps::QwenProvider::new().with_defaults(weights, tokenizer)) as Arc<dyn Provider>)
            },
            spec: Some(("qwen3", &qwen3::spec::Qwen3Spec)),
            resident: None,
        },
        // GLM-5.2. Same shape as qwen3 above: `weights` is a per-invocation
        // action param, so the manifest is weights-free and `brain caps` lists
        // GLM on a box with no checkpoint. The always-hot HTTP/D-Bus path is
        // `crate::resident_llm::GlmResident`, registered directly by the
        // serving executor, advertising
        // `glmdsa::caps::manifest_resident` - the same definition as this one,
        // minus the `weights` param the service supplies itself.
        ModelEntry {
            manifest: glmdsa::caps::manifest,
            provider: always!(glmdsa::caps::GlmProvider::new()),
            spec: None,
            resident: None,
        },
        // Qwen3.5-35B-A3B: like qwen3, `weights` is a per-invocation action
        // param (not baked into the Provider at construction), so this
        // manifest is genuinely weights-free -- the same reason qwen3's own
        // entry above needs no `resident` (the HTTP/D-Bus-served, always-hot
        // path is `crate::resident_qwen35moe::Qwen35Resident`, registered
        // directly by the serving executor, not through this ctor).
        ModelEntry {
            manifest: qwen35moe::caps::manifest,
            provider: always!(qwen35moe::caps::Qwen35Provider::new()),
            spec: None,
            resident: None,
        },
        // Qwen3.8-27B dense hybrid GDN/GQA decoder: `weights`/`tokenizer` are
        // per-invocation action params, resolved through
        // `qwen35::spec::Qwen35Spec` instead of `BRAIN_QWEN35_{WEIGHTS,
        // TOKENIZER}` (see that module's doc); the always-hot HTTP/D-Bus path
        // is `crate::resident_qwen35::Qwen35Resident`, registered directly by
        // the serving executor.
        ModelEntry {
            manifest: qwen35::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned());
                let tokenizer = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(qwen35::caps::Qwen35Provider::new().with_defaults(weights, tokenizer)) as Arc<dyn Provider>)
            },
            spec: Some(("qwen35", &qwen35::spec::Qwen35Spec)),
            resident: None,
        },
        // The timeline model: a directory a caller trained and saved (no
        // released weights to resolve), named per request locally and by
        // `BRAIN_HORIZON_DIR` on a served surface.
        ModelEntry {
            manifest: horizon::caps::manifest,
            provider: always!(horizon::caps::HorizonProvider::new()),
            spec: None,
            resident: resident!(crate::resident_horizon::HorizonResident::from_env),
        },
        ModelEntry {
            manifest: lfm2::caps::manifest,
            provider: always!(lfm2::caps::LfmProvider::new()),
            spec: None,
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
            spec: Some(("fastvlm", &fastvlm::spec::FastvlmSpec)),
            resident: None,
        },
        ModelEntry {
            manifest: llava::caps::manifest,
            provider: always!(llava::caps::LlavaProvider::new()),
            spec: None,
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
            spec: Some(("qwen3vl", &qwen3vl::spec::Qwen3VlSpec)),
            resident: crate::resident!(crate::resident_qwen3vl::Qwen3VlResident::from_env),
        },
        ModelEntry {
            manifest: yolov8::caps::manifest,
            provider: always!(yolov8::caps::YoloProvider::new()),
            spec: None,
            resident: None,
        },
        // Two weights roles (`trunk`, `heads`), both resolved from the model
        // store by `lpips::spec::LpipsSpec`; the provider reads them lazily.
        ModelEntry {
            manifest: lpips::caps::manifest,
            provider: |assembly: &Assembly| {
                let (trunk, heads) = (assembly.role_path("trunk")?, assembly.role_path("heads")?);
                Ok(Arc::new(lpips::caps::LpipsProvider::new(trunk, heads)) as Arc<dyn Provider>)
            },
            spec: Some(("lpips", &lpips::spec::LpipsSpec)),
            resident: None,
        },
        ModelEntry {
            manifest: zipdepth::caps::manifest,
            provider: always!(zipdepth::caps::DepthProvider::new()),
            spec: None,
            resident: None,
        },
        // The imaging models carry their weights path in the provider (from a
        // `BRAIN_*` env var), not as an action param, so `brain do` and the
        // residency adapter advertise ONE manifest each.
        // sam2 is the first of this migration's architectures whose provider
        // reads the Assembly it is called with (`role_path`, the single-role
        // counterpart of flux2's own `Paths::from_assembly`).
        ModelEntry {
            manifest: sam2::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.role_path("weights")?;
                Ok(Arc::new(sam2::caps::Sam2Provider::new(weights)) as Arc<dyn Provider>)
            },
            spec: Some(("sam2", &sam2::spec::Sam2Spec)),
            resident: crate::resident!(crate::resident_sam2::Sam2Resident::from_env),
        },
        ModelEntry {
            manifest: scrfd::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = role_dir(assembly, "weights")?;
                Ok(Arc::new(scrfd::caps::ScrfdProvider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("scrfd", &scrfd::spec::ScrfdSpec)),
            resident: crate::resident_in_store!(crate::resident_scrfd::ScrfdResident::from_env),
        },
        ModelEntry {
            manifest: florence2::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = role_dir(assembly, "weights")?;
                Ok(Arc::new(florence2::caps::Florence2Provider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("florence2", &florence2::spec::Florence2Spec)),
            resident: crate::resident_in_store!(crate::resident_florence2::Florence2Resident::from_env),
        },
        ModelEntry {
            manifest: arcface::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = role_dir(assembly, "weights")?;
                Ok(Arc::new(arcface::caps::ArcFaceProvider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("arcface", &arcface::spec::ArcFaceSpec)),
            resident: crate::resident_in_store!(crate::resident_arcface::ArcFaceResident::from_env),
        },
        ModelEntry {
            manifest: vqgan::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.role_path("weights")?;
                Ok(Arc::new(vqgan::caps::VqganProvider::new(weights)) as Arc<dyn Provider>)
            },
            spec: Some(("vqgan", &vqgan::spec::VqganSpec)),
            resident: crate::resident!(crate::resident_restore::VqganResident::from_env),
        },
        ModelEntry {
            manifest: codeformer::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.role_path("weights")?;
                // Checked before construction, same discipline `rrdbnet`'s
                // own entry below uses - a caller handing this fn a raw
                // `Assembly` directly (a unit test, say) must not get a
                // confusing failure only surfaced on the first real call.
                if !std::path::Path::new(&weights).exists() {
                    return Err(format!("codeformer: {weights} does not exist"));
                }
                // `RestoreProvider::new` builds no GPU and imports no
                // checkpoint yet - both happen lazily on the first
                // `restore_face` call (`caps.rs`'s own doc) - so this is
                // free, unlike `rrdbnet`'s entry below which must build
                // eagerly to classify the checkpoint's derived variant.
                Ok(Arc::new(codeformer::caps::RestoreProvider::new(weights)) as Arc<dyn Provider>)
            },
            spec: Some(("codeformer", &codeformer::spec::CodeFormerSpec)),
            resident: crate::resident!(crate::resident_restore::RestoreResident::from_env),
        },
        ModelEntry {
            manifest: rrdbnet::caps::manifest,
            provider: |assembly: &Assembly| {
                let weights = assembly.role_path("weights")?;
                // Checked before `Gpu::new` (unlike the resolved path
                // itself, which the resolver already confirmed exists at
                // resolve time) - a caller handing this fn a raw Assembly
                // directly (a unit test, say) must not pay for a real
                // device just to find out the path is bad, the same
                // discipline `from_env!`'s own existence check gave every
                // other entry here.
                if !std::path::Path::new(&weights).exists() {
                    return Err(format!("rrdbnet: {weights} does not exist"));
                }
                let gpu = gpu_core::Gpu::new(&rrdbnet::KERNELS);
                rrdbnet::caps::load(&weights, gpu).map(|s| Arc::new(rrdbnet::caps::UpscaleProvider::new(s)) as Arc<dyn Provider>)
            },
            spec: Some(("rrdbnet", &rrdbnet::spec::RrdbnetSpec)),
            resident: crate::resident!(crate::resident_upscale::UpscaleResident::from_env),
        },
        ModelEntry {
            manifest: clip::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = role_dir(assembly, "towers")?;
                Ok(Arc::new(clip::caps::ClipProvider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("clip", &clip::spec::ClipSpec)),
            resident: crate::resident_in_store!(crate::resident_clip::ClipResident::from_env),
        },
        ModelEntry {
            manifest: t5encoder::caps::manifest,
            provider: |assembly: &Assembly| {
                let root = assembly.role_path("root")?;
                Ok(Arc::new(t5encoder::caps::T5encoderProvider::new(root)) as Arc<dyn Provider>)
            },
            spec: Some(("t5encoder", &t5encoder::spec::T5encoderSpec)),
            resident: crate::resident!(crate::resident_t5encoder::T5encoderResident::from_env),
        },
        ModelEntry {
            manifest: sdxlunet::caps::manifest,
            provider: |assembly: &Assembly| {
                let root = assembly.role_path("root")?;
                Ok(Arc::new(sdxlunet::caps::SdxlProvider::new(root)) as Arc<dyn Provider>)
            },
            spec: Some(("sdxlunet", &sdxlunet::spec::SdxlunetSpec)),
            resident: crate::resident!(crate::resident_sdxl::SdxlResident::from_env),
        },
        ModelEntry {
            manifest: controlnet::caps::manifest,
            provider: |assembly: &Assembly| {
                let sdxl = assembly.role_path("sdxl")?;
                let control = assembly.role_path("control")?;
                Ok(Arc::new(controlnet::caps::ControlnetProvider::new(sdxl, control)) as Arc<dyn Provider>)
            },
            spec: Some(("controlnet", &controlnet::spec::ControlnetSpec)),
            resident: crate::resident!(crate::resident_controlnet::ControlnetResident::from_env),
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
            spec: None,
            resident: crate::resident!(crate::resident_supir::SupirResident::from_env),
        },
        ModelEntry {
            manifest: flux1::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = role_dir(assembly, "root")?;
                Ok(Arc::new(flux1::caps::Flux1Provider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("flux1", &flux1::spec::Flux1Spec)),
            resident: crate::resident_in_store!(crate::resident_flux1::Flux1Resident::from_env),
        },
        ModelEntry {
            manifest: pulid::caps::manifest,
            // Five roles, and they are not all the same shape: `flux1` is a
            // released pipeline ROOT and `pulid` is the adapter FILE
            // `import::read` opens, while `arcface`/`clip`/`bisenet` are
            // directories their loaders join a published filename onto.
            provider: |assembly: &Assembly| {
                let flux1 = assembly.role_path("flux1")?;
                let pulid_w = assembly.role_path("pulid")?;
                let arcface = role_dir(assembly, "arcface")?;
                let clip = role_dir(assembly, "clip")?;
                let bisenet = role_dir(assembly, "bisenet")?;
                Ok(Arc::new(pulid::caps::PulidProvider::new(flux1, pulid_w, arcface, clip, bisenet)) as Arc<dyn Provider>)
            },
            spec: Some(("pulid", &pulid::spec::PulidSpec)),
            resident: crate::resident_in_store!(crate::resident_pulid::PulidResident::from_env),
        },
        // DeepSeek-OCR: a document image in, decoded text out. Multi-file
        // checkpoint (mmproj + LM GGUF), so ONE directory variable, like
        // the face stack's and clip's. The only MULTI-device entry: its
        // vision tower runs on wgpu while its decoder runs on the CPU
        // backend, so it holds real bytes on two devices at once - see
        // `crate::resident_deepseekocr`'s header.
        // `dir` is resolved through `deepseek2ocr::spec::Deepseek2ocrSpec`
        // (see that module's doc) instead of `BRAIN_DEEPSEEK_OCR_DIR`.
        ModelEntry {
            manifest: deepseek2ocr::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = deepseek2ocr::spec::dir_from_assembly(assembly)?;
                deepseek2ocr::caps::DeepseekOcrProvider::new(dir).map(|p| Arc::new(p) as Arc<dyn Provider>).ok_or_else(|| "deepseek-ocr: resolved dir does not hold both shipped GGUFs".to_string())
            },
            spec: Some(("deepseek2ocr", &deepseek2ocr::spec::Deepseek2ocrSpec)),
            resident: crate::resident_multi!(crate::resident_deepseekocr::DeepseekOcrResident::from_assembly),
        },
        // DeepSeek-OCR-2: v1's successor. Same decoder (`crates/deepseek2`,
        // unmodified), a new vision tower - SAM feeding a Qwen2 GQA resampler
        // under a prefix-LM mask, global view only for now (see
        // `deepseekocr2::caps`'s own header for the scope this milestone
        // shipped). Single-device (CPU) today, unlike v1's wgpu/CPU split -
        // see `crate::resident_deepseekocr2`'s header for why.
        ModelEntry {
            manifest: deepseekocr2::caps::manifest,
            provider: from_env!(
                deepseekocr2::caps::DeepseekOcr2Provider::from_env,
                "set BRAIN_DEEPSEEKOCR2_DIR to a directory holding mmproj-deepseek-ocr-2-q8_0.gguf + deepseek-ocr-2-q8_0.gguf"
            ),
            spec: None,
            resident: crate::resident!(crate::resident_deepseekocr2::DeepseekOcr2Resident::from_env),
        },
        // Moondream 3: an image in, text out. SigLIP ViT with overlap multi-crop
        // -> connector -> a parallel-block sparse-MoE decoder. int8 experts by
        // default, because the fp32 build is ~43 GiB and loads nowhere - see
        // `crate::resident_moondream3`'s header.
        // `dir` is resolved through `moondream3::spec::Moondream3Spec` (see
        // that module's doc) instead of `BRAIN_MOONDREAM3_WEIGHTS`.
        ModelEntry {
            manifest: moondream3::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(moondream3::caps::Moondream3Provider::new().with_default_dir(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("moondream3", &moondream3::spec::Moondream3Spec)),
            resident: None,
        },
        // DeepSeek-VL: images and a conversation in, text out; the decoder at
        // the checkpoint's own fp16. `dir` resolved through
        // `deepseekvl::spec::DEEPSEEK_VL`; the resident is registered directly
        // (see `crate::resident_deepseekvl`).
        ModelEntry {
            manifest: deepseekvl::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(deepseekvl::caps::DeepseekVlProvider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("deepseekvl", &deepseekvl::spec::DEEPSEEK_VL)),
            resident: crate::resident_multi_family!(crate::resident_deepseekvl::DeepseekVlResident::family_from_assembly),
        },
        // Janus-Pro: chat over images, and text to image, from one
        // checkpoint. `dir` resolved through `januspro::spec::JANUS_PRO`; the
        // resident is registered directly (see `crate::resident_januspro`).
        ModelEntry {
            manifest: januspro::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned());
                Ok(Arc::new(januspro::caps::JanusProProvider::new(dir)) as Arc<dyn Provider>)
            },
            spec: Some(("januspro", &januspro::spec::JANUS_PRO)),
            resident: None,
        },
        ModelEntry {
            manifest: imgpipe::caps::manifest,
            provider: |_assembly: &Assembly| Ok(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(stage_registry(brain_modelstore::explicit_models_root().as_deref())))) as Arc<dyn Provider>),
            spec: None,
            resident: None,
        },
        ModelEntry {
            manifest: qwen3tts::caps::manifest,
            provider: always!(qwen3tts::caps::TtsProvider::new()),
            spec: None,
            resident: crate::resident!(crate::resident_tts::TtsResident::from_env),
        },
        // MiniMax Music 3. Like flux2, its six roles live in the provider
        // (resolved through `minimaxmusic3::spec::MinimaxMusic3Spec` from
        // the Assembly this entry is called with), not `BRAIN_MINIMAXMUSIC3_*`
        // - the residency adapter (registered directly by `serving::build_executor`,
        // not from here) still reads those env vars.
        ModelEntry {
            manifest: minimaxmusic3::caps::manifest,
            provider: |assembly: &Assembly| {
                let paths = minimaxmusic3::generate::Paths::from_assembly(assembly)?;
                Ok(Arc::new(minimaxmusic3::caps::MinimaxMusic3Provider::new(paths)) as Arc<dyn Provider>)
            },
            spec: Some(("minimaxmusic3", &minimaxmusic3::spec::MinimaxMusic3Spec)),
            resident: crate::resident!(crate::resident_minimaxmusic3::MinimaxMusic3Resident::from_env),
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
            spec: Some(("cosyvoice", &cosyvoice::spec::CosyVoiceSpec)),
            resident: crate::resident!(crate::resident_cosyvoice::CosyVoiceResident::from_env),
        },
        // Speech-to-text. Discovery is weight-free (the caps manifests); the
        // direct `brain do` path wraps the model crates' eager providers in
        // [`LazyProvider`] so construction stays cheap.
        ModelEntry {
            manifest: nemotronasr::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.role_path("weights")?;
                Ok(Arc::new(LazyProvider::new(
                    nemotronasr::caps::manifest,
                    Box::new(move || {
                        nemotronasr::caps::NemotronProvider::load(&dir, nemotronasr::NemotronConfig::nemotron_3_5_asr_0_6b())
                            .map(|p| Arc::new(p) as Arc<dyn Provider>)
                    }),
                )) as Arc<dyn Provider>)
            },
            spec: Some(("nemotronasr", &nemotronasr::spec::NemotronAsrSpec)),
            resident: crate::resident!(crate::resident_asr::NemotronResident::from_env),
        },
        ModelEntry {
            manifest: qwen3asr::caps::manifest,
            provider: |assembly: &Assembly| {
                let dir = assembly.role_path("weights")?;
                Ok(Arc::new(LazyProvider::new(
                    qwen3asr::caps::manifest,
                    Box::new(move || {
                        let (window_secs, max_new) = resident_asr::qwen_asr_tuning();
                        qwen3asr::caps::QwenAsrProvider::load(&dir, qwen3asr::config::QwenAsrConfig::qwen3_asr_1_7b(), window_secs, max_new)
                            .map(|p| Arc::new(p) as Arc<dyn Provider>)
                    }),
                )) as Arc<dyn Provider>)
            },
            spec: Some(("qwen3asr", &qwen3asr::spec::Qwen3AsrSpec)),
            resident: crate::resident!(crate::resident_asr::QwenAsrResident::from_env),
        },
        // Time-series forecasting. Discoverable (`brain caps`) and served
        // (`brain serve`, via the resident ctors), but with no direct `brain do`
        // provider yet: the forecast run logic lives in the residency instances
        // (NPU/device placement included), so the provider says exactly how to
        // reach the model instead of "unknown model".
        ModelEntry {
            manifest: resident_forecast::chronos2_manifest,
            provider: |_assembly: &Assembly| Err("chronos-2 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_CHRONOS2 set".to_string()),
            spec: None,
            resident: resident!(resident_forecast::Chronos2Resident::from_env),
        },
        ModelEntry {
            manifest: resident_forecast::fincast_manifest,
            provider: |_assembly: &Assembly| Err("fincast has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_FINCAST set".to_string()),
            spec: None,
            resident: resident!(resident_forecast::FincastResident::from_env),
        },
        ModelEntry {
            manifest: resident_forecast::kronos_manifest,
            provider: |_assembly: &Assembly| Err("kronos has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER set".to_string()),
            spec: None,
            resident: resident!(resident_forecast::KronosResident::from_env),
        },
        ModelEntry {
            manifest: resident_forecast::timesfm3_manifest,
            provider: |_assembly: &Assembly| Err("timesfm3 has no direct `brain do` provider yet - serve it (`brain serve --dbus` or an HTTP surface) with BRAIN_TIMESFM3 set, or the resolver (`brain timesfm3 predict ...`)".to_string()),
            spec: None,
            resident: resident!(resident_forecast::Timesfm3Resident::from_env),
        },
        // 3D Gaussian Splatting (render/fit): needs no weights at all, the scene
        // arrives as request bytes, yet it has a resident because render and fit
        // allocate GPU buffers per request, which is worth scheduling.
        ModelEntry {
            manifest: splat::caps::manifest,
            provider: always!(splat::caps::SplatProvider::new()),
            spec: None,
            resident: resident!(resident_splat::SplatResident::from_env),
        },
        // WorldMirror-2 multi-view 3D reconstruction. `weights` is a
        // per-invocation action param, so `manifest` is weights-free.
        ModelEntry {
            manifest: worldmirror2::caps::manifest,
            provider: always!(worldmirror2::caps::WorldMirror2Provider::new()),
            spec: None,
            resident: resident!(resident_worldmirror2::WorldMirror2Resident::from_env),
        },
        // No-weights utility models, listed by `brain caps` and served as
        // stateless residents by the serving executor rather than through this
        // list, hence no `resident` here.
        ModelEntry { manifest: imageops::manifest, provider: always!(imageops::ImageOps), spec: None, resident: None },
        ModelEntry {
            manifest: || {
                use capability::Provider as _;
                demo::DemoModel.manifest()
            },
            provider: always!(demo::DemoModel),
            spec: None,
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
///
/// Each stage resolves against `models_dir`: this crate's own `imgpipe`
/// entry passes [`brain_modelstore::explicit_models_root`] (see
/// `resolved_assembly` for why), while a serving process that resolved its
/// own `--models-dir` passes that, so its pipeline reads the same store as
/// the rest of the process.
pub fn stage_registry(models_dir: Option<&Path>) -> capability::Registry {
    let mut inner = capability::Registry::new();
    for e in models() {
        let id = (e.manifest)().model;
        // Only the models a stage can actually name today, and only through
        // their OWN `ModelEntry::spec` - all three are resolver-migrated, so
        // a real Assembly (when an explicit models directory is opted into,
        // see `resolved_assembly`'s own doc) stands in for the env var each
        // used to read. A stage whose model resolves to nothing is skipped,
        // exactly as an absent `BRAIN_*` var used to make `from_env!` skip it.
        if ![imgpipe::SEGMENT_MODEL, imgpipe::UPSCALE_MODEL, imgpipe::RESTORE_MODEL].contains(&id.as_str()) {
            continue;
        }
        let Some((arch, spec)) = e.spec else { continue };
        if let Ok(a) = resolved_assembly(models_dir, arch, spec) {
            if let Ok(p) = (e.provider)(&a) {
                inner.register(p);
            }
        }
    }
    inner
}

/// A resolver-migrated model's real [`Assembly`], scanned from `models_dir` -
/// `Err` carries WHY not, in words a caller can act on (no store configured,
/// nothing published for this architecture, or more than one candidate with
/// nothing to pick between them), because by name is how most consumers of
/// this crate reach a model and "it did not resolve" is not an answer any of
/// them can do anything with.
///
/// This crate's own callers pass [`brain_modelstore::explicit_models_root`],
/// deliberately NOT [`brain_modelstore::default_root`]: this runs as a side
/// effect of constructing the `imgpipe` provider, which plain library use
/// (including `cargo test`'s own `every_listed_model_is_constructible_by_name`)
/// can reach with no CLI invocation and no opt-in in sight - falling all the
/// way to `default_root`'s bare `$HOME` tier would scan (and best-effort
/// cache-write into) a real developer's actual model store as a side effect
/// of running the test suite. This is the library's own scan/resolve, kept
/// minimal: no override flags, and silent rather than printing/exiting on
/// `Ambiguous`/`Missing`, since nothing upstream of a pipeline stage can act
/// on either outcome anyway.
///
/// Scans on every call, deliberately: the store is a directory a user adds
/// files to while a long-lived process is running, so a memoized inventory
/// would answer "no weights" for a checkpoint that is now there. The repeat
/// cost is what `brain_modelstore::inventory`'s own on-disk cache absorbs.
fn resolved_assembly(models_dir: Option<&Path>, arch: &str, spec: &dyn ArchSpec) -> Result<Assembly, String> {
    let Some(root) = models_dir else {
        return Err("no model store is configured (publish a data root, or set BRAIN_MODELS_DIR or XDG_DATA_HOME)".to_string());
    };
    let records = brain_modelstore::inventory::scan(root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    match brain_modelstore::resolve::resolve(arch, &records, &specs, &std::collections::BTreeMap::new()) {
        Resolution::Resolved(a) => Ok(*a),
        // The resolver's own rendering of both non-terminal outcomes, not a
        // reworded one: `Missing` carries each unsatisfied role's OWN doc
        // string (what that architecture wants published) and `Ambiguous`
        // every real candidate. Neither is collapsed into a pick here - only
        // a human can answer an ambiguity (see `brain_modelstore::resolve`).
        Resolution::Ambiguous(a) => Err(format!("{} (searched {})", one_line(&describe_ambiguity(&a)), root.display())),
        Resolution::Missing(m) => Err(format!("{} (searched {})", one_line(&describe_missing(&m)), root.display())),
    }
}

/// The resolver's own multi-line rendering, on one line: these reasons are
/// carried inside a single-sentence `Err` string, which is what a caller
/// logs or shows.
fn one_line(s: &str) -> String {
    s.trim().replace('\n', "; ")
}

/// The message a by-name caller gets when a resolver-migrated model's weights
/// are not on this machine.
///
/// The rule it keeps: if the weights are absent, say THAT - never that some
/// [`Assembly`] "has no `<role>` role", which describes this crate's own
/// plumbing to somebody who never passed an assembly in, and reads as "the
/// catalog is broken" rather than "the weights are missing". Named per model,
/// because the caller asked for a model.
fn weights_unavailable(model: &str, arch: &str, why: &str) -> String {
    format!("{model}: no weights available on this machine - {why}. Publish {arch}'s weights to the model store, or set BRAIN_MODELS_DIR to a store that holds them")
}

/// A placeholder [`Assembly`] for an entry that reads none: an
/// [`always!`]/[`from_env!`] entry, or one whose checkpoint is a
/// per-invocation action param. Every entry that DOES read one carries a
/// [`ModelEntry::spec`], and [`provider`] resolves a real assembly for it
/// instead of passing this.
fn empty_assembly() -> Assembly {
    Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }
}

/// The registry SUPIR's optional caption auto-fill dispatches
/// [`supir::caps::LLAVA_MODEL`] through, for the direct `brain do`/D-Bus-via-
/// provider path (`resident_supir.rs` builds an equivalent registry for the
/// served path).
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
/// This is how an embedder that holds no resolver of its own reaches a model,
/// so it does the resolving itself: an entry carrying a
/// [`ModelEntry::spec`] is built from a REAL [`Assembly`] scanned out of an
/// explicitly opted-into model store ([`resolved_assembly`]), which is what
/// makes a model whose weights ARE published constructible by name. A caller
/// that already holds a resolver-built assembly (the CLI, which resolves with
/// its own `--models-dir`/override vocabulary) calls
/// [`provider_from_assembly`] with it instead, so it never resolves twice.
///
/// When nothing resolves, an entry that needs no role at construction still
/// builds - and one that does fails with [`weights_unavailable`]'s message,
/// never with a complaint about an empty [`Assembly`]'s roles.
pub fn provider(model: &str) -> Result<Arc<dyn Provider>, String> {
    for e in models() {
        if (e.manifest)().model == model {
            let Some((arch, spec)) = e.spec else { return (e.provider)(&empty_assembly()) };
            return match resolved_assembly(brain_modelstore::explicit_models_root().as_deref(), arch, spec) {
                Ok(a) => (e.provider)(&a),
                // Nothing resolved. The entry still gets its chance with an
                // empty assembly, because "resolver-migrated" does not mean
                // "needs weights to CONSTRUCT" - the decoder LMs take their
                // checkpoint as a per-request action param and build fine
                // without one. Only an entry that genuinely needed a role
                // fails here, and for exactly one reason: the weights are
                // not on this machine. That reason replaces its role-shaped
                // complaint about a structure the caller never passed in.
                Err(why) => (e.provider)(&empty_assembly()).map_err(|_| weights_unavailable(model, arch, &why)),
            };
        }
    }
    Err(format!("unknown model '{model}' (see `brain caps`)"))
}

/// [`provider`], from an already-resolved [`Assembly`] instead of resolving
/// one itself: the entry point for a caller that holds one (the CLI, which
/// resolves with its own `--models-dir` and `--<role>` vocabulary), so it never
/// resolves twice. An entry that reads no role ignores the argument.
pub fn provider_from_assembly(model: &str, assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    for e in models() {
        if (e.manifest)().model == model {
            return (e.provider)(assembly);
        }
    }
    Err(format!("unknown model '{model}' (see `brain caps`)"))
}

/// The resolver architecture and spec a catalog model's weights come from, or
/// `None` for a model whose entry reads no [`Assembly`] at all.
///
/// The ONE place this mapping lives (it is [`ModelEntry::spec`], read by id),
/// so a caller that resolves in its own vocabulary (the CLI, with its
/// `--models-dir` flag and `--<role>` overrides) shares the catalog's table
/// instead of keeping a second one that can drift from it.
pub fn resolver_spec_for(model: &str) -> Option<ArchRef> {
    models().into_iter().find(|e| (e.manifest)().model == model).and_then(|e| e.spec)
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

    /// Serializes the tests that publish a data root: the published root is
    /// process-global (`brain_modelstore::publish_data_root`), so two tests
    /// setting it concurrently would each see the other's store.
    fn store_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A data root of this test's own, with its models directory created and
    /// empty - never a developer's real store.
    fn tmp_data_root(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("brain-catalog-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(brain_modelstore::models_dir_in(&root)).expect("create the test store's models directory");
        root
    }

    /// A SAM 2.1 `tiny` checkpoint, as far as `sam2::spec` can tell: that spec
    /// recognizes the released size from the trunk patch-embedding conv's own
    /// output-channel count (96 = tiny), read from the header, so a file
    /// carrying that ONE tensor resolves exactly as a real release does - at
    /// 56 kB instead of 150 MB, with no weights on this machine.
    fn publish_sam2_tiny(root: &std::path::Path) {
        let models = brain_modelstore::models_dir_in(root);
        let path = models.join("sam2_hiera_tiny.safetensors");
        let tensors = vec![("image_encoder.trunk.patch_embed.proj.weight".to_string(), vec![96u64, 3, 7, 7], vec![0f32; 96 * 3 * 7 * 7])];
        checkpoint::save(&path.to_string_lossy(), serde_json::json!({}), &tensors);
    }

    /// The vocabulary of this crate's own plumbing. An [`Assembly`] is how the
    /// catalog threads resolved weight paths INTO an entry; a caller that
    /// asked for a model by name never passed one and can do nothing with a
    /// complaint about its roles. Such a message reaching them means the
    /// construction failed for a STRUCTURAL reason - the catalog never
    /// resolved an assembly at all - and is describing an internal data
    /// structure instead of saying the weights are absent. Banned on the
    /// message, because the message is what a caller sees, exactly as
    /// "unknown model" already is.
    fn is_structural(e: &str) -> bool {
        e.contains("unknown model") || e.contains("assembly")
    }

    /// THE DRIFT THIS FILE EXISTS TO KILL: every model listed here must be
    /// constructible by name. It may legitimately fail for want of weights - 
    /// what it must never do is answer "unknown model" for something it just
    /// advertised, or blame its own [`Assembly`]'s missing roles: both are
    /// structural failures of the catalog rather than an honest "this machine
    /// has no weights for that model", and both read to a caller as "the
    /// model is broken". The second is what this test used to miss entirely:
    /// as entries migrated onto the resolver their providers started READING
    /// the assembly, and by-name construction - which handed them an empty
    /// one - began failing for ~21 of them with `assembly '' has no <role>
    /// role`, which this assertion happily accepted.
    #[test]
    fn every_listed_model_is_constructible_by_name() {
        // An EMPTY store of this test's own, published as the data root: the
        // hardest case (nothing resolves anywhere), and it keeps the run off a
        // developer's real model store even when BRAIN_MODELS_DIR or
        // XDG_DATA_HOME point at one.
        let _g = store_lock();
        let empty = tmp_data_root("no-weights");
        brain_modelstore::publish_data_root(Some(empty.clone()));
        let mut structural = Vec::new();
        for m in manifests() {
            if let Err(e) = provider(&m.model) {
                if is_structural(&e) {
                    structural.push(format!("{}: {e}", m.model));
                }
            }
        }
        brain_modelstore::publish_data_root(None);
        let _ = std::fs::remove_dir_all(&empty);
        assert!(structural.is_empty(), "{} listed model(s) cannot be built for a structural reason:\n{}", structural.len(), structural.join("\n"));
    }

    /// The other half of the same contract, and the one no amount of "it did
    /// not say unknown model" can stand in for: a model whose weights ARE
    /// published must actually CONSTRUCT through [`provider`], and one whose
    /// weights are absent must be told so in those words.
    ///
    /// `sam2` stands for the ~21 resolver-migrated entries here because its
    /// provider only holds the resolved path (no device, no checkpoint read),
    /// so this stays a catalog test rather than a model one.
    #[test]
    fn a_published_model_constructs_by_name_and_an_absent_one_says_the_weights_are_missing() {
        let _g = store_lock();

        let published = tmp_data_root("published");
        publish_sam2_tiny(&published);
        brain_modelstore::publish_data_root(Some(published.clone()));
        let built = provider(sam2::caps::MODEL);
        let published_err = built.err();

        let absent = tmp_data_root("absent");
        brain_modelstore::publish_data_root(Some(absent.clone()));
        let absent_err = provider(sam2::caps::MODEL).err();

        brain_modelstore::publish_data_root(None);
        let _ = std::fs::remove_dir_all(&published);
        let _ = std::fs::remove_dir_all(&absent);

        assert!(published_err.is_none(), "published sam2 weights must construct by name, got: {}", published_err.unwrap_or_default());
        let e = absent_err.expect("sam2 with nothing published must NOT construct");
        assert!(!is_structural(&e), "absent weights must not be reported structurally: {e}");
        assert!(e.contains(sam2::caps::MODEL), "the message must name the model the caller asked for: {e}");
        assert!(e.contains("weights"), "the message must say it is the WEIGHTS that are missing: {e}");
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

    /// The other side of `tests/served_paths.rs` (which reads every served
    /// manifest for a path param), and what keeps it meaningful: the LOCAL
    /// surface (`brain caps`/`brain do`, run by somebody standing on the
    /// machine that holds the weights) still offers the override. If this ever went empty, that test would be passing
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
                    assert!(p.host_env.is_none() && !p.host_resolved, "'{}':'{}' leaked host-resolved param '{}'", m.model, a.name, p.name);
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
    /// A declared range is enforced on every caller's value, so a default
    /// outside it would be refused the moment a UI sent it back.
    #[test]
    fn every_default_lies_within_its_declared_range() {
        for m in manifests() {
            for a in &m.actions {
                for p in &a.params {
                    let Some(x) = p.default.as_ref().and_then(|d| d.as_f64()) else { continue };
                    assert!(p.min.is_none_or(|lo| x >= lo) && p.max.is_none_or(|hi| x <= hi), "{}/{}: default {x} outside [{:?}, {:?}] for '{}'", m.model, a.name, p.min, p.max, p.name);
                }
            }
        }
    }

    #[test]
    fn an_unknown_model_is_an_error() {
        let e = match provider("definitely/not-a-model") {
            Err(e) => e,
            Ok(_) => panic!("a made-up model resolved"),
        };
        assert!(e.contains("unknown model"), "{e}");
    }
}
