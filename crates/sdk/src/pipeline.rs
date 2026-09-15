// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`ImagePipeline`]: the flux2-backed vertical slice of `brain`'s image
//! surface.
//!
//! `load()` replicates the decisions `crates/cli/src/flux2_cli.rs`'s own
//! `generate` command makes inline (resolve, license-gate, pick the DiT
//! precision, size the pipeline) against `crates/loader`'s resolver -- the
//! SAME resolver the CLI uses, never a parallel path -- so a checkpoint that
//! resolves for `brain flux2 generate` resolves here too, with no
//! environment variable and no CLI process in the loop.

use std::collections::BTreeMap;

use crate::{Error, Image, Result};

/// A device/backend selection. Re-exported, not reinvented: the SAME type
/// `--device` parses into (`gpu_core::devices::DeviceSpec`). An empty
/// [`Device::default`] is the existing "auto" concept -- schedule on
/// whatever hardware the machine actually has.
pub use gpu_core::devices::DeviceSpec as Device;
/// A DiT numeric tier. Re-exported, not reinvented: the SAME type flux2's
/// own `Pipeline::build_sized` takes (`model::dispatch::Precision`).
pub use model::dispatch::Precision as DType;

/// Generation knobs layered over flux2's own [`flux2::GenOpts`] defaults --
/// every field left unset here keeps whatever `GenOpts::default()` (or, on a
/// distilled variant, [`flux2::pipeline::resolved_steps`]) already says.
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

/// Apply a device/backend selection to this process, the same way
/// `crates/cli/src/main.rs` applies `--device` before any model builds:
/// resolve it against the real hardware inventory, publish it as the
/// process's ambient [`gpu_core::ComputeSet`], and install
/// [`loader::install_default_placer`] so automatic GPU/CPU placement narrows
/// to exactly what was asked for. An embedder has no CLI startup path to do
/// this for it, so [`ImagePipelineBuilder::load`] does it here instead.
fn apply_device(device: &Device) -> Result<()> {
    let probe = gpu_core::Inventory::probe();
    let set = device.resolve(&probe).map_err(Error::Backend)?;
    set.apply().map_err(Error::Backend)?;
    let gpus: Option<std::collections::HashSet<u32>> = Some(set.gpus.iter().copied().collect());
    let cpu_allowed = set.cpu_enabled();
    gpu_core::publish_compute_set(set);
    loader::install_default_placer(gpus, cpu_allowed);
    Ok(())
}

/// A FLUX.2 text-to-image pipeline, resolved and built from a local model
/// store.
pub struct ImagePipeline {
    pipe: flux2::Pipeline,
    cfg: flux2::Flux2Config,
    paths: flux2::Paths,
    precision: DType,
    /// The one adapter [`ImagePipeline::load_lora`] has folded in, if any --
    /// flux2's `Pipeline` only takes adapters at BUILD time
    /// (`Pipeline::build_sized`'s `adapters: &[AdapterSpec]`; there is no
    /// post-construction "add an adapter" call on a built `Pipeline`), so
    /// `load_lora` rebuilds the pipeline from `cfg`/`paths` with this
    /// adapter folded in, rather than mutating the one already built.
    adapter: Option<flux2::AdapterSpec>,
}

/// Hand-written, not derived: `flux2::pipeline::Pipeline` itself carries no
/// `Debug` impl (it holds live GPU device handles), so a derive here would
/// not compile. A short summary is still worth printing -- and worth having
/// at all, since `Result<ImagePipeline, Error>` needs SOME `Debug` bound to
/// be usable with `.unwrap()`/`.expect()` the way any other `Result` is.
impl std::fmt::Debug for ImagePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImagePipeline").field("cfg", &self.cfg).field("paths", &self.paths).field("precision", &self.precision).field("adapter", &self.adapter).finish()
    }
}

impl ImagePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<ImagePipeline> {
        ImagePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> ImagePipelineBuilder {
        ImagePipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), dtype: DType::F32 }
    }

    /// Fold a LoRA adapter in, from THIS milestone's one supported source: a
    /// literal filesystem path to brain's own trained checkpoint, or a
    /// third-party ai-toolkit/ComfyUI/LyCORIS `.safetensors` (mirrors
    /// today's `flux2::AdapterSpec { path, scale: 1.0 }`, exactly what `brain
    /// flux2 generate --adapter <path>` builds).
    ///
    /// An `owner/name`-shaped store reference is a REAL, confirmed gap, not
    /// an oversight: `loader`'s `model_dir::resident_for` only wires
    /// store-resolved adapters for the "qwen" residency arm, and neither
    /// `flux2::spec::Flux2Spec` nor `s3dit`'s own spec declares an adapter
    /// role at all -- there is nothing for a store reference to resolve
    /// against yet for an image model, so this returns a clear
    /// [`Error::Backend`] naming the gap rather than silently trying (and
    /// failing) to treat it as a path.
    ///
    /// Rebuilds the pipeline (see [`ImagePipeline::adapter`]'s doc for why),
    /// so this is exactly as expensive as [`ImagePipelineBuilder::load`]
    /// itself.
    pub fn load_lora(&mut self, source: impl AsRef<str>) -> Result<()> {
        self.adapter = Some(adapter_source_path(source.as_ref())?);
        self.rebuild()
    }

    fn rebuild(&mut self) -> Result<()> {
        let adapters: Vec<flux2::AdapterSpec> = self.adapter.iter().cloned().collect();
        let n = default_forward_tokens();
        self.pipe = flux2::Pipeline::build_sized(&self.cfg, &self.paths, n, n, &adapters, self.precision, 1).map_err(Error::Backend)?;
        Ok(())
    }

    /// Generate one image from `prompt`, at every default
    /// [`ImageGenerationOptions`] leaves unset.
    pub fn generate(&self, prompt: &str) -> Result<Image> {
        self.generate_with(prompt, ImageGenerationOptions::default())
    }

    pub fn generate_with(&self, prompt: &str, opts: ImageGenerationOptions) -> Result<Image> {
        let o = opts.into_gen_opts();
        let (rgb, w, h) = self
            .pipe
            .generate(prompt, &[], &o, &capability::CancelToken::default(), |_step, _total, _msg| {})
            .map_err(|e| if e == "cancelled" { Error::Cancelled } else { Error::Backend(e) })?;
        Image::from_rgb8(w, h, rgb)
    }
}

/// Builds an [`ImagePipeline`]. `.device(...)`/`.dtype(...)` are the only
/// knobs this milestone exposes; every other decision follows flux2's own
/// defaults.
pub struct ImagePipelineBuilder {
    model_id: String,
    device: Device,
    dtype: DType,
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

    /// Resolve `model_id` and build a real [`ImagePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`apply_device`]).
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing` (the
    ///    one policy this milestone's builder offers -- see
    ///    [`loader::DownloadPolicy`]'s own doc for the other two), fetched
    ///    only when nothing local already resolves it: a checkpoint already
    ///    on disk is never re-checked against the network.
    /// 3. `crates/loader`'s resolver ([`loader::resolve_structured`]) scans
    ///    the models directory for FLUX.2's four roles (`dit`/`vae`/
    ///    `text_encoder`/`tokenizer`), with no role override stated -- the
    ///    same resolver `brain flux2 generate` calls with no `--dit`/
    ///    `--variant`/... flags typed.
    /// 4. The resolved [`capability::Assembly`] becomes real
    ///    [`flux2::pipeline::Paths`], the license gate runs
    ///    ([`flux2::caps::check_license`] -- the 9B weights are
    ///    Non-Commercial-licensed), the variant's [`flux2::Flux2Config`] is
    ///    looked up, and the DiT's EXECUTABLE precision is resolved
    ///    ([`flux2::pipeline::effective_dit_precision`]: a `.gguf` source
    ///    always executes through FLUX.2's packed-int8 path, whatever
    ///    [`DType`] was requested).
    /// 5. [`flux2::pipeline::Pipeline::build_sized`] builds the pipeline, at
    ///    this milestone's conservative defaults: no adapters, a 1024x1024
    ///    (flux2's own default canvas) forward/output ceiling, and
    ///    `max_batch = 1`.
    ///
    /// Every failure path returns a typed [`Error`] -- never a panic on a
    /// caller-reachable input.
    pub fn load(self) -> Result<ImagePipeline> {
        let ImagePipelineBuilder { model_id, device, dtype } = self;

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
        let assembly = match loader::resolve_structured("flux2", &flux2::spec::Flux2Spec, &overrides).map_err(Error::Backend)? {
            brain_modelstore::resolve::Resolution::Resolved(assembly) => *assembly,
            brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
            brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
        };

        let paths = flux2::Paths::from_assembly(&assembly).map_err(Error::Backend)?;
        let variant_name = assembly.variant.clone().ok_or_else(|| Error::Backend(format!("flux2: resolved assembly {:?} has no variant", assembly.id)))?;
        flux2::caps::check_license(&variant_name).map_err(Error::Backend)?;
        let cfg = flux2::Flux2Config::from_name(&variant_name).map_err(Error::Backend)?;

        let precision = flux2::pipeline::effective_dit_precision(&paths.dit, dtype, false).map_err(Error::Backend)?;

        let n = default_forward_tokens();
        let pipe = flux2::Pipeline::build_sized(&cfg, &paths, n, n, &[], precision, 1).map_err(Error::Backend)?;

        Ok(ImagePipeline { pipe, cfg, paths, precision, adapter: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
