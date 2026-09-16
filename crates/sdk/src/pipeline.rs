// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`ImagePipeline`]: `brain`'s image surface, over TWO structurally
//! different backends behind one public type -- flux2 (a per-call-sizeable
//! `flux2::Pipeline`, `generate` returning `(Vec<u8> RGB8, u32, u32)`) and
//! s3dit (a build-time-sized `s3dit::pipeline::HotPipeline`, `generate`
//! returning a float HWC `Image { hwc: Vec<f32> in [0,1], w, h }`).
//!
//! [`ImagePipelineBuilder::load`] replicates the decisions
//! `crates/cli/src/flux2_cli.rs`/`s3dit::caps::ZAction` each make inline
//! (resolve, license-gate where the architecture has one, pick the DiT
//! precision, size the pipeline) against `crates/loader`'s resolver -- the
//! SAME resolver the CLI uses, never a parallel path -- so a checkpoint that
//! resolves for `brain flux2 generate` / `brain do z-image text2image`
//! resolves here too, with no environment variable and no CLI process in the
//! loop. Which backend gets built is decided ONCE, by
//! [`resolve_arch`] reading the resolved [`capability::Assembly::arch`] back
//! -- everything past that point (`generate`/`generate_with`/`.save()`) is
//! uniform: see [`Backend`].

use std::collections::BTreeMap;

use crate::{Error, Image, Result};

// `Device` and `DType` live in lib.rs: they belong to the `device` and
// `resolve` tiers, which every surface selects, so a second surface finds them
// already in place rather than having to move them out of this module.
use crate::{DType, Device};

/// Generation knobs layered over flux2's own [`flux2::GenOpts`] defaults on
/// a flux2-backed pipeline -- every field left unset here keeps whatever
/// `GenOpts::default()` (or, on a distilled variant,
/// [`flux2::pipeline::resolved_steps`]) already says.
///
/// On an s3dit-backed pipeline the SAME type means something asymmetric,
/// because s3dit couples size to the built pipeline
/// (`s3dit::pipeline::HotPipeline::build_adapted` records its DiT/VAE graphs
/// for one `width x height` at BUILD time, unlike flux2's per-call-sizeable
/// `Pipeline::generate`): [`ImageGenerationOptions::size`] here is VALIDATED
/// against the load-time size ([`ImagePipelineBuilder::size`]), not applied
/// -- a caller who asks for a different size than the pipeline was built for
/// gets a clear [`Error::Backend`] naming both sizes and how to fix it
/// ([`check_s3dit_size`]), never a silent resize or a silent ignore. `steps`
/// and `seed` still apply, at s3dit's own defaults (8 steps, seed 42 --
/// [`s3dit::caps`]'s own `text2image` defaults) when unset; `guidance` has
/// no s3dit counterpart at all (Z-Image-Turbo is a 0-guidance distilled
/// model, like flux2's klein variants) and is silently unused on that
/// backend.
///
/// `steps` has NO upper bound anywhere in this call chain: neither this
/// type, nor flux2's [`flux2::pipeline::resolved_steps`], nor s3dit's
/// `HotPipeline::generate` (which only floors it to `.max(1)`) refuses an
/// absurd value -- a caller who passes `steps(u32::MAX)` gets a denoise loop
/// that many iterations long, not a clean refusal. `width`/`height` (via
/// [`ImagePipelineBuilder::size`] on the s3dit side) and the prompt's own
/// token length DO have real, pre-existing bounds in each backend's own
/// code (flux2's build-time forward-token ceiling; s3dit's
/// `check_build_shape`/`fit_caption`) -- `steps` is the one caller-facing
/// knob with none. Pre-existing in both backends, not introduced or masked
/// here -- deliberately not given an SDK-only clamp, which would hide the
/// same gap from every OTHER caller of `flux2`/`s3dit` directly.
#[derive(Clone, Debug, Default)]
pub struct ImageGenerationOptions {
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    seed: Option<u64>,
}

impl ImageGenerationOptions {
    pub fn new() -> ImageGenerationOptions {
        ImageGenerationOptions::default()
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn into_gen_opts(self) -> flux2::GenOpts {
        let mut o = flux2::GenOpts::default();
        if let Some(w) = self.width {
            o.width = w;
        }
        if let Some(h) = self.height {
            o.height = h;
        }
        if self.steps.is_some() {
            o.steps = self.steps;
        }
        if let Some(seed) = self.seed {
            o.seed = seed;
        }
        o
    }
}

/// Both backends' `generate` return `Result<_, String>` and signal a caller's
/// cancel token firing with the literal string `"cancelled"` (there is no
/// richer shared error type to carry it as anything else) -- this is the one
/// place that string is turned into [`Error::Cancelled`] rather than an
/// indistinguishable [`Error::Backend`], factored out so the mapping is
/// testable with no real pipeline in hand.
fn backend_err_or_cancelled(e: String) -> Error {
    if e == "cancelled" {
        Error::Cancelled
    } else {
        Error::Backend(e)
    }
}

/// [`ImagePipeline::load_lora`]'s gate, factored out so it is testable with
/// no real pipeline in hand: a literal filesystem path is accepted verbatim
/// (mirroring today's real `flux2::AdapterSpec { path, scale: 1.0 }`);
/// anything else is refused by name, naming the real, confirmed gap rather
/// than pretending an `owner/name` store reference works.
fn adapter_source_path(source: &str) -> Result<flux2::AdapterSpec> {
    if !std::path::Path::new(source).exists() {
        return Err(Error::Backend(format!(
            "{source}: not a file on disk -- store-resolved adapters (an `owner/name` reference) are not yet wired for image models (flux2/s3dit declare no adapter role); pass a literal filesystem path to a LoRA checkpoint instead"
        )));
    }
    Ok(flux2::AdapterSpec::new(source.to_string()))
}

/// The conservative sizing [`ImagePipelineBuilder::load`] builds against: no
/// reference images and flux2's own default canvas
/// ([`flux2::GenOpts::default`], 1024x1024) -- a real, useful size, not a
/// toy one, but the one ceiling this milestone's `ImagePipeline` supports.
/// [`ImageGenerationOptions::size`] past this ceiling is flux2's own to
/// refuse or accept; this facade does not re-size the built pipeline per
/// call (`crates/flux2/src/pipeline.rs`'s own `Pipeline::build_sized` doc:
/// the DiT/VAE scratch is sized once, at build time).
fn default_forward_tokens() -> u32 {
    flux2::pipeline::gen_tokens_per_forward(&flux2::GenOpts::default())
}

/// The s3dit-backed pipeline's build-time size default, when
/// [`ImagePipelineBuilder::size`] is never called: flux2's own default
/// canvas ([`flux2::GenOpts::default`]) so the two backends agree on "the
/// size you get if you never ask" even though WHEN that size is fixed
/// differs between them.
const S3DIT_DEFAULT_SIZE: (u32, u32) = (1024, 1024);
/// s3dit's own `text2image` defaults (`s3dit::caps::gen_params`) --
/// mirrored here rather than re-derived, so an unset
/// [`ImageGenerationOptions::steps`]/`seed` on an s3dit-backed pipeline
/// produces the SAME image a `brain do z-image text2image` call with no
/// `steps`/`seed` argument would.
const S3DIT_DEFAULT_STEPS: u32 = 8;
const S3DIT_DEFAULT_SEED: u64 = 42;

/// [`ImagePipeline::generate_with`]'s size gate on an s3dit-backed pipeline,
/// factored out so it is testable with no real pipeline in hand: `requested`
/// is [`ImageGenerationOptions`]'s own `(width, height)` (`None` when the
/// caller left both unset, which always matches); `built` is the size the
/// pipeline was actually constructed for
/// ([`ImagePipelineBuilder::size`]/[`S3DIT_DEFAULT_SIZE`]). A mismatch is a
/// clean, named [`Error::Backend`] -- s3dit has no per-call resize to fall
/// back to (see this module's doc), and silently generating at the WRONG
/// size, or silently ignoring the caller's request with no signal at all,
/// are both worse than refusing by name.
fn check_s3dit_size(built: (u32, u32), requested: (Option<u32>, Option<u32>)) -> Result<()> {
    let (w, h) = requested;
    if let (Some(w), Some(h)) = (w, h) {
        if (w, h) != built {
            return Err(Error::Backend(format!(
                "s3dit: this pipeline was built for {}x{} (ImagePipelineBuilder::size), but generate_with asked for {w}x{h} -- s3dit's DiT/VAE graphs are sized at BUILD time (HotPipeline::build_adapted), so a different size cannot be honored per call; build a new pipeline with .size({w}, {h}) instead",
                built.0, built.1
            )));
        }
    }
    Ok(())
}

/// [`crate::device::resolve`] plus the one thing only a `resolve`-surface
/// needs on top: installing [`loader::install_default_placer`] so automatic
/// GPU/CPU model-shard placement narrows to exactly what was asked for.
fn apply_device(device: &Device) -> Result<()> {
    let set = crate::device::resolve(device)?;
    let gpus: Option<std::collections::HashSet<u32>> = Some(set.gpus.iter().copied().collect());
    let cpu_allowed = set.cpu_enabled();
    loader::install_default_placer(gpus, cpu_allowed);
    Ok(())
}

/// The backend-specific state one resolved architecture built --
/// [`ImagePipeline`] itself carries only a [`Backend`], and every public
/// method matches on it once, at the top; nothing downstream of that match
/// (image normalization, `.save()`) differs by backend at all.
enum Backend {
    // Boxed: `Flux2Backend`/`S3ditBackend` differ enough in inline size
    // (flux2's `Pipeline`/`Flux2Config` carry far more state by value than
    // s3dit's `HotPipeline`) that leaving them unboxed makes every
    // `ImagePipeline` pay the larger variant's stack/enum size regardless of
    // which backend it resolved to (clippy::large_enum_variant).
    Flux2(Box<Flux2Backend>),
    S3dit(Box<S3ditBackend>),
}

struct Flux2Backend {
    pipe: flux2::Pipeline,
    cfg: flux2::Flux2Config,
    paths: flux2::Paths,
    precision: DType,
    /// The one adapter folded in, if any -- flux2's `Pipeline` only takes
    /// adapters at BUILD time (`Pipeline::build_sized`'s `adapters:
    /// &[AdapterSpec]`; there is no post-construction "add an adapter" call
    /// on a built `Pipeline`), so [`ImagePipeline::load_lora`] rebuilds the
    /// pipeline from `cfg`/`paths` with this adapter folded in, rather than
    /// mutating the one already built.
    adapter: Option<flux2::AdapterSpec>,
}

struct S3ditBackend {
    pipe: s3dit::pipeline::HotPipeline,
    paths: s3dit::pipeline::Paths,
    /// The size this pipeline was BUILT for -- see this module's doc for why
    /// s3dit has no per-call sizing to fall back on. Set once, in
    /// [`ImagePipelineBuilder::load`]/[`ImagePipelineBuilder::size`], and
    /// read back by [`check_s3dit_size`] on every `generate_with` call.
    width: u32,
    height: u32,
    hifi: bool,
    /// The one adapter folded in, if any -- same reason as
    /// [`Flux2Backend::adapter`]: `HotPipeline::build_adapted`'s `adapter:
    /// Option<&str>` only takes one at BUILD time.
    adapter: Option<String>,
}

/// `brain`'s image-generation pipeline: `ImagePipeline::from_pretrained(...)`
/// resolves a local model store and builds a real, resident model behind
/// this ONE type, whichever of the two structurally different backends
/// (flux2 or s3dit -- see this module's doc) the resolved store actually
/// turned out to hold.
pub struct ImagePipeline {
    backend: Backend,
}

/// Hand-written, not derived: neither `flux2::pipeline::Pipeline` nor
/// `s3dit::pipeline::HotPipeline` carries a `Debug` impl (both hold live GPU
/// device handles), so a derive here would not compile. A short summary is
/// still worth printing -- and worth having at all, since `Result<
/// ImagePipeline, Error>` needs SOME `Debug` bound to be usable with
/// `.unwrap()`/`.expect()` the way any other `Result` is.
impl std::fmt::Debug for ImagePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backend {
            Backend::Flux2(b) => f.debug_struct("ImagePipeline").field("backend", &"flux2").field("cfg", &b.cfg).field("paths", &b.paths).field("precision", &b.precision).field("adapter", &b.adapter).finish(),
            Backend::S3dit(b) => f.debug_struct("ImagePipeline").field("backend", &"s3dit").field("paths", &b.paths).field("width", &b.width).field("height", &b.height).field("hifi", &b.hifi).field("adapter", &b.adapter).finish(),
        }
    }
}

impl ImagePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<ImagePipeline> {
        ImagePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> ImagePipelineBuilder {
        ImagePipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), dtype: DType::F32, size: None }
    }

    /// What this pipeline's backend can do, per action: the SAME static
    /// `capability::Manifest` `brain caps`/D-Bus/HTTP read for this
    /// architecture (`flux2::caps::manifest`/`s3dit::caps::manifest`) --
    /// reflected, not re-described, so this can never drift into a second,
    /// weaker capability description. Every param this manifest declares is
    /// what the ARCHITECTURE supports; not every one necessarily has an
    /// `ImageGenerationOptions` knob yet (see that type's own doc for the
    /// ones that don't), so this is "what the model can do", not a promise
    /// that this facade exposes every declared param today.
    pub fn capabilities(&self) -> capability::Manifest {
        match &self.backend {
            Backend::Flux2(_) => flux2::caps::manifest(),
            Backend::S3dit(_) => s3dit::caps::manifest(),
        }
    }

    /// Fold a LoRA adapter in, from THIS milestone's one supported source: a
    /// literal filesystem path to brain's own trained checkpoint, or a
    /// third-party ai-toolkit/ComfyUI/LyCORIS `.safetensors` (mirrors
    /// today's `flux2::AdapterSpec { path, scale: 1.0 }` /
    /// `s3dit::pipeline::HotPipeline::build_adapted`'s `adapter: Option<&str>`
    /// -- exactly what `brain flux2 generate --adapter <path>` / `brain do
    /// z-image text2image adapter=<path>` build).
    ///
    /// An `owner/name`-shaped store reference is a REAL, confirmed gap, not
    /// an oversight: `crates/cli/src/model_dir.rs`'s `resident_for` only
    /// wires a store-resolved adapter for the "qwen" residency arm, and
    /// neither `flux2::spec::Flux2Spec` nor `s3dit::spec::S3ditSpec`
    /// declares an adapter role at all -- there is nothing for a store
    /// reference to resolve against yet for an image model, so this returns
    /// a clear [`Error::Backend`] naming the gap rather than silently trying
    /// (and failing) to treat it as a path.
    ///
    /// Rebuilds the pipeline (see [`Flux2Backend::adapter`]/
    /// [`S3ditBackend::adapter`]'s docs for why), so this is exactly as
    /// expensive as [`ImagePipelineBuilder::load`] itself.
    pub fn load_lora(&mut self, source: impl AsRef<str>) -> Result<()> {
        let spec = adapter_source_path(source.as_ref())?;
        match &mut self.backend {
            Backend::Flux2(b) => {
                b.adapter = Some(spec);
                let adapters: Vec<flux2::AdapterSpec> = b.adapter.iter().cloned().collect();
                let n = default_forward_tokens();
                b.pipe = flux2::Pipeline::build_sized(&b.cfg, &b.paths, n, n, &adapters, b.precision, 1).map_err(Error::Backend)?;
            }
            Backend::S3dit(b) => {
                b.adapter = Some(spec.path);
                b.pipe = s3dit::pipeline::HotPipeline::build_adapted(&b.paths, b.width, b.height, s3dit::pipeline::DEFAULT_CAP_LEN, b.hifi, b.adapter.as_deref(), |_| {}).map_err(Error::Backend)?;
            }
        }
        Ok(())
    }

    /// Generate one image from `prompt`, at every default
    /// [`ImageGenerationOptions`] leaves unset, with no progress reporting
    /// and no way to cancel mid-generation. See
    /// [`ImagePipeline::generate_with_progress`] for a caller that needs
    /// either.
    pub fn generate(&self, prompt: &str) -> Result<Image> {
        self.generate_with(prompt, ImageGenerationOptions::default())
    }

    /// [`ImagePipeline::generate`] plus [`ImageGenerationOptions`], still
    /// with no progress reporting and no cancellation -- see
    /// [`ImagePipeline::generate_with_progress`].
    pub fn generate_with(&self, prompt: &str, opts: ImageGenerationOptions) -> Result<Image> {
        self.generate_with_progress(prompt, opts, &capability::CancelToken::default(), &mut |_step, _total, _msg| {})
    }

    /// The full-control entry point [`ImagePipeline::generate`]/
    /// [`ImagePipeline::generate_with`] both delegate to at their defaults
    /// (an unarmed [`capability::CancelToken`] and a discarded progress
    /// closure): pass a token you control to cancel a multi-step denoise
    /// loop from another thread (it is polled once per step, the same
    /// cadence `capability::Action`'s own long-running actions use), and a
    /// closure to observe `(step, total, message)` as it runs -- the CLI's
    /// own progress line reads the identical three values. Neither backend's
    /// underlying `generate` takes less than this; `generate`/`generate_with`
    /// exist so the common call needs neither.
    ///
    /// This is the one place flux2's `(Vec<u8> RGB8, u32, u32)` and s3dit's
    /// float HWC `Image { hwc, w, h }` both normalize into [`Image`] -- see
    /// this module's doc and [`Image::from_hwc_unit`]'s doc.
    pub fn generate_with_progress(
        &self,
        prompt: &str,
        opts: ImageGenerationOptions,
        cancel: &capability::CancelToken,
        on_progress: &mut dyn FnMut(u32, u32, &str),
    ) -> Result<Image> {
        match &self.backend {
            Backend::Flux2(b) => {
                let o = opts.into_gen_opts();
                let (rgb, w, h) = b
                    .pipe
                    .generate(prompt, &[], &o, cancel, on_progress)
                    .map_err(backend_err_or_cancelled)?;
                Image::from_rgb8(w, h, rgb)
            }
            Backend::S3dit(b) => {
                check_s3dit_size((b.width, b.height), (opts.width, opts.height))?;
                let steps = opts.steps.unwrap_or(S3DIT_DEFAULT_STEPS);
                let seed = opts.seed.unwrap_or(S3DIT_DEFAULT_SEED);
                let img = b
                    .pipe
                    .generate(prompt, seed, steps, cancel, on_progress)
                    .map_err(backend_err_or_cancelled)?;
                Image::from_hwc_unit(img.w as u32, img.h as u32, &img.hwc)
            }
        }
    }
}

/// Which backend [`ImagePipelineBuilder::load`] should build, decided ONCE
/// by [`resolve_arch`] -- everything downstream (`generate`/`generate_with`/
/// `.save()`) is uniform past this point; see this module's doc.
enum ResolvedArch {
    Flux2(capability::Assembly),
    S3dit(capability::Assembly),
}

/// Resolve `model_id`'s components against BOTH known image architectures --
/// flux2 first (`flux2::spec::Flux2Spec`; today's only pre-D2 backend, kept
/// as the tie-break so a store that happens to satisfy both specs at once
/// behaves exactly as it did before s3dit support existed and skips the
/// second, unnecessary resolve pass entirely), then s3dit
/// (`s3dit::spec::S3ditSpec`) only when flux2 did not resolve. Whichever one
/// actually [`brain_modelstore::resolve::Resolution::Resolved`]s becomes the
/// [`ResolvedArch`] [`ImagePipelineBuilder::load`] builds against --
/// `capability::Assembly::arch` on the winning [`capability::Assembly`] is
/// what actually decided it, per this milestone's dispatch requirement.
///
/// A store that resolves NEITHER reports whichever attempt found REAL (if
/// ambiguous) evidence of its own architecture over one that found nothing
/// at all: an `Ambiguous` outcome is a strictly more useful answer to hand a
/// caller than a `Missing` one from whichever architecture happened to be
/// tried first. Two `Missing`s report flux2's -- an arbitrary tie-break
/// (flux2 was tried first), not a claim that flux2 is the more likely
/// answer; a caller who needs to know WHY neither architecture matched a
/// truly foreign store already has [`Error::Missing`]'s structured `roles`
/// to read either way.
fn resolve_arch(overrides: &BTreeMap<String, String>) -> Result<ResolvedArch> {
    use brain_modelstore::resolve::Resolution;

    let flux2_outcome = loader::resolve_structured("flux2", &flux2::spec::Flux2Spec, overrides).map_err(Error::Backend)?;
    if matches!(flux2_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(a) = flux2_outcome else { unreachable!("just matched") };
        return Ok(ResolvedArch::Flux2(*a));
    }

    let s3dit_outcome = loader::resolve_structured("s3dit", &s3dit::spec::S3ditSpec, overrides).map_err(Error::Backend)?;
    if matches!(s3dit_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(a) = s3dit_outcome else { unreachable!("just matched") };
        return Ok(ResolvedArch::S3dit(*a));
    }

    // Both `Resolved` cases already returned above; only `Ambiguous`/
    // `Missing` combinations can reach here.
    match (flux2_outcome, s3dit_outcome) {
        (Resolution::Ambiguous(a), _) => Err(Error::Ambiguous(a)),
        (_, Resolution::Ambiguous(a)) => Err(Error::Ambiguous(a)),
        (f, _) => match f {
            Resolution::Missing(m) => Err(Error::Missing(m)),
            Resolution::Resolved(_) => unreachable!("Resolved handled above"),
            Resolution::Ambiguous(_) => unreachable!("Ambiguous handled above"),
        },
    }
}

/// Builds an [`ImagePipeline`]. `.device(...)`/`.dtype(...)`/`.size(...)`
/// are the only knobs this milestone exposes; every other decision follows
/// each backend's own defaults.
pub struct ImagePipelineBuilder {
    model_id: String,
    device: Device,
    dtype: DType,
    size: Option<(u32, u32)>,
}

impl ImagePipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// The pipeline's build-time size.
    ///
    /// On a flux2-backed pipeline this is currently UNUSED: flux2 stays
    /// sized at its own conservative default ceiling
    /// ([`default_forward_tokens`]) regardless, and per-call sizing keeps
    /// working through [`ImageGenerationOptions::size`] up to that ceiling,
    /// exactly as it did before this method existed. Accepted (not refused)
    /// on a flux2-backed builder anyway, so calling code that does not yet
    /// know which backend it will get does not have to special-case it.
    ///
    /// On an s3dit-backed pipeline this IS the build shape
    /// (`HotPipeline::build_adapted`'s `width`/`height`) -- s3dit has no
    /// other way to be sized, since its DiT/VAE graphs are recorded once, at
    /// construction (see this module's doc). Defaults to
    /// [`S3DIT_DEFAULT_SIZE`] (1024x1024, flux2's own default canvas) when
    /// never called.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.size = Some((width, height));
        self
    }

    /// Resolve `model_id` and build a real [`ImagePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`apply_device`]).
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing` (the
    ///    one policy this milestone's builder offers -- see
    ///    [`loader::DownloadPolicy`]'s own doc for the other two), fetched
    ///    only when nothing local already resolves it: a checkpoint already
    ///    on disk is never re-checked against the network.
    /// 3. [`resolve_arch`] tries `crates/loader`'s resolver against each
    ///    known image architecture's roles, with no role override stated --
    ///    the same resolver `brain flux2 generate` / `brain do z-image
    ///    text2image` call with no `--dit`/`--variant`/... flags typed.
    /// 4. The resolved [`capability::Assembly`] becomes real backend
    ///    `Paths`, the license gate runs where the backend has one
    ///    ([`flux2::caps::check_license`] -- the 9B weights are
    ///    Non-Commercial-licensed; s3dit's Z-Image-Turbo has none), the
    ///    variant/config is looked up, and the executable precision is
    ///    resolved (flux2: [`flux2::pipeline::effective_dit_precision`], a
    ///    `.gguf` source always executes through FLUX.2's packed-int8 path
    ///    whatever [`DType`] was requested; s3dit: `hifi = dtype ==
    ///    DType::F32`, the literal "fp32 execution" reading of `DType::F32`).
    /// 5. The backend's own `build_sized`/`build_adapted` builds the
    ///    pipeline: flux2 at this milestone's conservative defaults (no
    ///    adapters, a 1024x1024 forward/output ceiling, `max_batch = 1`);
    ///    s3dit at [`ImagePipelineBuilder::size`]'s width/height (no
    ///    adapter, [`s3dit::pipeline::DEFAULT_CAP_LEN`] caption capacity).
    ///
    /// Every failure path returns a typed [`Error`] -- never a panic on a
    /// caller-reachable input.
    pub fn load(self) -> Result<ImagePipeline> {
        let ImagePipelineBuilder { model_id, device, dtype, size } = self;

        apply_device(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

        let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
        let store = brain_modelstore::Store::new(root);
        let hub = brain_modelstore::HfHub::new();

        // DownloadPolicy::IfMissing: a reference that already resolves
        // locally is never re-checked against the network at all.
        if store.local(&reference).is_none() {
            let plan = brain_modelstore::plan(&reference, &store, &hub)?;
            loader::supply::execute_plan(&store, &hub, &plan, &model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
        }

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        match resolve_arch(&overrides)? {
            ResolvedArch::Flux2(assembly) => {
                let paths = flux2::Paths::from_assembly(&assembly).map_err(Error::Backend)?;
                let variant_name = assembly.variant.clone().ok_or_else(|| Error::Backend(format!("flux2: resolved assembly {:?} has no variant", assembly.id)))?;
                flux2::caps::check_license(&variant_name).map_err(Error::LicenseRequired)?;
                let cfg = flux2::Flux2Config::from_name(&variant_name).map_err(Error::Backend)?;

                let precision = flux2::pipeline::effective_dit_precision(&paths.dit, dtype, false).map_err(Error::Backend)?;

                let n = default_forward_tokens();
                let pipe = flux2::Pipeline::build_sized(&cfg, &paths, n, n, &[], precision, 1).map_err(Error::Backend)?;

                Ok(ImagePipeline { backend: Backend::Flux2(Box::new(Flux2Backend { pipe, cfg, paths, precision, adapter: None })) })
            }
            ResolvedArch::S3dit(assembly) => {
                let paths = s3dit::pipeline::Paths::from_assembly(&assembly).map_err(Error::Backend)?;
                let (width, height) = size.unwrap_or(S3DIT_DEFAULT_SIZE);
                let hifi = dtype == DType::F32;

                let pipe = s3dit::pipeline::HotPipeline::build_adapted(&paths, width, height, s3dit::pipeline::DEFAULT_CAP_LEN, hifi, None, |_| {}).map_err(Error::Backend)?;

                Ok(ImagePipeline { backend: Backend::S3dit(Box::new(S3ditBackend { pipe, paths, width, height, hifi, adapter: None })) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of routing `generate_with_progress` through a
    /// caller-supplied [`capability::CancelToken`]: the one string either
    /// backend uses to signal it firing turns into [`Error::Cancelled`], not
    /// an indistinguishable [`Error::Backend`].
    #[test]
    fn backend_err_or_cancelled_recognizes_the_cancellation_sentinel() {
        assert!(matches!(backend_err_or_cancelled("cancelled".to_string()), Error::Cancelled));
    }

    /// Every other backend failure stays a named `Error::Backend`, verbatim.
    #[test]
    fn backend_err_or_cancelled_passes_every_other_message_through() {
        let err = backend_err_or_cancelled("flux2: assemble: no dit chosen".to_string());
        assert!(matches!(err, Error::Backend(ref m) if m == "flux2: assemble: no dit chosen"));
    }

    /// TDD anchor for [`ImagePipeline::load_lora`]'s dispatch: a literal
    /// path that exists on disk is accepted, at the reference-default
    /// scale -- exactly `AdapterSpec::new`'s own contract.
    #[test]
    fn adapter_source_path_accepts_a_real_file() {
        let f = std::env::temp_dir().join(format!("brain-sdk-lora-real-{}.safetensors", std::process::id()));
        std::fs::write(&f, b"not a real checkpoint, just needs to exist").unwrap();
        let spec = adapter_source_path(f.to_str().unwrap()).unwrap();
        assert_eq!(spec.path, f.to_str().unwrap());
        assert_eq!(spec.scale, 1.0);
        std::fs::remove_file(&f).ok();
    }

    /// The real, confirmed gap this milestone documents rather than
    /// pretends around: an `owner/name`-shaped store reference is refused
    /// by name, not silently treated as (and failing as) a bad path.
    #[test]
    fn adapter_source_path_refuses_a_store_style_reference_by_name() {
        let err = adapter_source_path("swedishembedded-com/generic-sft").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not yet wired"), "{msg}");
        assert!(matches!(err, Error::Backend(_)));
    }

    /// Every [`ImageGenerationOptions`] field left unset keeps flux2's own
    /// [`flux2::GenOpts::default`] value -- this facade layers on top of
    /// flux2's defaults, it does not invent its own.
    #[test]
    fn unset_generation_options_keep_flux2_s_own_defaults() {
        let o = ImageGenerationOptions::new().into_gen_opts();
        let want = flux2::GenOpts::default();
        assert_eq!(o.width, want.width);
        assert_eq!(o.height, want.height);
        assert_eq!(o.steps, want.steps);
        assert_eq!(o.seed, want.seed);
    }

    /// Every field that IS set overrides its `GenOpts` counterpart, and
    /// nothing else moves.
    #[test]
    fn set_generation_options_override_only_their_own_field() {
        let o = ImageGenerationOptions::new().size(256, 128).steps(9).seed(7).into_gen_opts();
        assert_eq!((o.width, o.height), (256, 128));
        assert_eq!(o.steps, Some(9));
        assert_eq!(o.seed, 7);
        // Untouched fields keep flux2's own default.
        assert_eq!(o.guidance, flux2::GenOpts::default().guidance);
    }

    /// [`check_s3dit_size`]: a caller who never sets
    /// [`ImageGenerationOptions::size`] always matches, at every built size.
    #[test]
    fn check_s3dit_size_always_matches_when_the_caller_leaves_size_unset() {
        assert!(check_s3dit_size((512, 768), (None, None)).is_ok());
        assert!(check_s3dit_size((1024, 1024), (None, None)).is_ok());
    }

    /// A per-call size that matches the build-time size is accepted.
    #[test]
    fn check_s3dit_size_accepts_a_matching_request() {
        assert!(check_s3dit_size((768, 768), (Some(768), Some(768))).is_ok());
    }

    /// The real, confirmed asymmetry Part 1 documents rather than hides:
    /// s3dit couples size to the built pipeline, so a per-call size that
    /// DIFFERS from what the pipeline was built for is a named, actionable
    /// [`Error::Backend`] -- never a silent resize and never a silent
    /// ignore.
    #[test]
    fn check_s3dit_size_refuses_a_mismatched_request_by_name() {
        let err = check_s3dit_size((512, 512), (Some(768), Some(768))).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("512x512"), "{msg}");
        assert!(msg.contains("768x768"), "{msg}");
        assert!(matches!(err, Error::Backend(_)));
    }
}
