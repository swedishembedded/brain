// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`EmbeddingPipeline`]: brain's text-embedding surface, over CLIP's text
//! towers (CLIP-L / OpenCLIP-bigG). Resolved through `crates/loader`'s
//! resolver - the SAME resolver `brain do clip embed_text` uses - so a
//! checkpoint that resolves there resolves here too. Selected by the
//! `vision` Cargo feature, not `embedding` - see `Cargo.toml`'s own comment
//! on that feature for why (`brain_arch::Domain` has no `Embedding` variant;
//! CLIP's own registered domain is `Vision`).
//!
//! Scoped to TEXT embedding only: ArcFace (face embedding) takes an image
//! plus a detected-and-aligned face, not a string, so it needs a genuinely
//! different call shape (`embed_image`, composed with SCRFD detection) -
//! tracked as a real, separate extension, not folded into this type by
//! pretending the inputs are the same. T5-XXL/umT5-XXL
//! (`crates/t5encoder`) are conditioning encoders another model's pipeline
//! consumes internally (FLUX.1's text conditioning), not a
//! caller-facing embedding endpoint in their own right.
//!
//! ```no_run
//! let pipe = brain::EmbeddingPipeline::from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")?;
//! let v = pipe.embed("a whale submarine")?;
//! println!("{} dims", v.len());
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! (CLIP text towers ship inside the same released SDXL-layout directory
//! several image-generation checkpoints already carry - see
//! [`EmbeddingPipelineBuilder::load`]'s own doc for exactly which role this
//! resolves.)

use std::collections::BTreeMap;

use crate::{Device, Error, Result};

/// One embedding vector. Wraps a plain `Vec<f32>` rather than exposing one
/// directly - a domain type, per this SDK's own rule, even though today it
/// adds only a name and a `dim()` accessor; raw access stays one call away
/// (`Embedding::as_slice`).
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Embedding {
        Embedding(v)
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

/// `brain`'s text-embedding pipeline. See this module's doc for scope.
pub struct EmbeddingPipeline {
    session: clip::caps::Session,
    /// Which CLIP tower every call embeds with - a pipeline-wide choice
    /// (`.tower(...)` on the builder), not a per-call parameter: unlike
    /// `ImagePipeline`'s two structurally interchangeable backends, CLIP-L
    /// and OpenCLIP-bigG produce embeddings of DIFFERENT dimensionality, so
    /// mixing them per call on one pipeline would silently change what a
    /// caller's stored vectors mean.
    tower: String,
}

impl std::fmt::Debug for EmbeddingPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingPipeline").field("tower", &self.tower).finish()
    }
}

impl EmbeddingPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<EmbeddingPipeline> {
        EmbeddingPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> EmbeddingPipelineBuilder {
        EmbeddingPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), tower: DEFAULT_TOWER.to_string() }
    }

    /// Embed one string.
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.embed_batch(&[text])?.into_iter().next().expect("one input in, one output out"))
    }

    /// Embed `texts` in ONE batched forward - the genuine batched path
    /// `clip::caps::Session::embed_text_batch` already implements, not a
    /// serial loop over [`EmbeddingPipeline::embed`].
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let out = self.session.embed_text_batch(&self.tower, &owned).map_err(Error::Backend)?;
        Ok(out.into_iter().map(Embedding::from).collect())
    }
}

const DEFAULT_TOWER: &str = "clip_l";

/// Which architecture [`EmbeddingPipelineBuilder::load`] resolved to -
/// unlike `ImagePipeline`'s `Backend`, this exists only to carry the
/// resolved [`capability::Assembly`] through construction; today there is
/// only one arm (CLIP), kept as an enum so a second embedding architecture
/// (a future ArcFace image-embedding pipeline, say) has a documented place
/// to land rather than a rewrite.
enum ResolvedArch {
    Clip(capability::Assembly),
}

fn resolve_arch(overrides: &BTreeMap<String, String>) -> Result<ResolvedArch> {
    use brain_modelstore::resolve::Resolution;

    let outcome = loader::resolve_structured("clip", &clip::spec::ClipSpec, overrides).map_err(Error::Backend)?;
    match outcome {
        Resolution::Resolved(a) => Ok(ResolvedArch::Clip(*a)),
        Resolution::Ambiguous(a) => Err(Error::Ambiguous(a)),
        Resolution::Missing(m) => Err(Error::Missing(m)),
    }
}

/// Builds an [`EmbeddingPipeline`]. `.device(...)`/`.tower(...)` are the
/// only knobs this milestone exposes.
pub struct EmbeddingPipelineBuilder {
    model_id: String,
    device: Device,
    tower: String,
}

impl EmbeddingPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Which CLIP text tower to embed with: `"clip_l"` (the default) or
    /// `"openclip_bigg"` - see [`EmbeddingPipeline`]'s own doc for why this
    /// is fixed per pipeline rather than a per-call choice.
    pub fn tower(mut self, tower: impl Into<String>) -> Self {
        self.tower = tower.into();
        self
    }

    /// Resolve `model_id` and build a real [`EmbeddingPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing`, fetched
    ///    only when nothing local already resolves it (mirrors
    ///    [`crate::ImagePipelineBuilder::load`]/[`crate::ForecastPipelineBuilder::load`]
    ///    exactly).
    /// 3. `crates/loader`'s resolver looks for CLIP's `"towers"` role: a
    ///    released SDXL-layout directory holding `text_encoder/`,
    ///    `text_encoder_2/`, `tokenizer/` and `tokenizer_2/` - the same
    ///    directory several image-generation checkpoints in this store may
    ///    already carry (SDXL's own text conditioning). The optional
    ///    `"eva"` role (EVA-CLIP's image tower) is not needed for text
    ///    embedding and is not required to resolve.
    /// 4. `clip::caps::Session::load` opens the resolved directory's
    ///    tokenizer(s); the text tower itself builds lazily, on first
    ///    [`EmbeddingPipeline::embed`]/[`EmbeddingPipeline::embed_batch`]
    ///    call (`Session`'s own design - see that type's doc).
    pub fn load(self) -> Result<EmbeddingPipeline> {
        let EmbeddingPipelineBuilder { model_id, device, tower } = self;

        crate::device::apply(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

        let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
        let store = brain_modelstore::Store::new(root);
        let hub = brain_modelstore::HfHub::new();

        if store.local(&reference).is_none() {
            let plan = brain_modelstore::plan(&reference, &store, &hub)?;
            loader::supply::execute_plan(&store, &hub, &plan, &model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
        }

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let ResolvedArch::Clip(assembly) = resolve_arch(&overrides)?;
        let dir = assembly.roles.get("towers").ok_or_else(|| Error::Backend(format!("clip: resolved assembly {:?} has no towers role", assembly.id)))?;

        let gpu = gpu_core::Gpu::new(clip::model::TEXT_PIPELINES);
        let session = clip::caps::Session::load(&dir.to_string_lossy(), gpu).map_err(Error::Backend)?;

        Ok(EmbeddingPipeline { session, tower })
    }
}
