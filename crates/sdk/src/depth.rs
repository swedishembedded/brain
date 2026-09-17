// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`DepthPipeline`]: brain's monocular depth surface, over ZipDepth.
//! Resolved through `crates/loader`'s resolver against
//! `zipdepth::spec::ZipdepthSpec`'s one `"weights"` role - the same resolver
//! `brain do depth infer` would use once `zipdepth::caps::DepthProvider`
//! itself migrates onto it (see this module's doc note on that gap).
//!
//! A NEW public domain type, [`DepthMap`], rather than joining
//! [`crate::Image`]-shaped pipelines: depth is a per-pixel relative distance
//! grid, not a displayable image, so it does not fit `image(+opts) -> Image`
//! the way [`crate::ImagePipeline`]/[`crate::UpscalePipeline`]/
//! [`crate::RestorePipeline`] all do - the same "different pipeline shape,
//! not a fourth backend" reasoning that gave [`crate::DetectionPipeline`] its
//! own `Detection` and [`crate::SegmentPipeline`] its own `Mask`. Named for
//! the `brain_arch::Domain` it resolves under (`vision`, the same feature
//! those two joined - ZipDepth's own `arch!` row is `Vision` too).
//!
//! ```no_run
//! let pipe = brain::DepthPipeline::from_pretrained("skchen1993/ZipDepth")?;
//! let photo = brain::Image::open("street.jpg")?;
//! let depth = pipe.predict(&photo)?;
//! println!("{}x{} depth, raw range [{}, {}]", depth.width, depth.height, depth.min, depth.max);
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use crate::{Device, Error, Image, Result};

/// A dense relative inverse-depth map on the source image's own grid.
/// `values` is min-max normalized to `[0, 1]` (nearer is not fixed to either
/// end - only relative ordering is meaningful); `min`/`max` are the raw
/// (pre-normalization) bounds the map was scaled from, so the relative
/// distances stay recoverable, the same contract `zipdepth::caps::manifest`'s
/// `infer` action documents for its own `depth` blob.
#[derive(Clone, Debug)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
    pub min: f32,
    pub max: f32,
}

/// [`DepthPipeline::predict_with`]'s one knob, mirroring `zipdepth::caps`'s
/// own `input` action param.
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthOptions {
    input: Option<u32>,
}

impl DepthOptions {
    pub fn new() -> DepthOptions {
        DepthOptions::default()
    }

    /// Override the model's input side (shorter side, rounded to a multiple
    /// of 32 - the model is fully convolutional, so any such side is valid).
    /// Unset uses the checkpoint's own native input.
    pub fn input(mut self, side: u32) -> Self {
        self.input = Some(side);
        self
    }
}

/// `brain`'s monocular depth pipeline. One architecture today (ZipDepth).
pub struct DepthPipeline {
    session: zipdepth::caps::Session,
}

/// Hand-written, not derived: `zipdepth::caps::Session` holds a live GPU
/// device handle (no `Debug` impl), the same reason
/// [`crate::UpscalePipeline`]'s own `Debug` is hand-written.
impl std::fmt::Debug for DepthPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthPipeline").field("native_input", &self.session.native_input()).finish()
    }
}

impl DepthPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<DepthPipeline> {
        DepthPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> DepthPipelineBuilder {
        DepthPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`zipdepth::caps::manifest`) - reflected, not re-described,
    /// same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        zipdepth::caps::manifest()
    }

    /// Predict depth for `image` at the checkpoint's native input size.
    pub fn predict(&self, image: &Image) -> Result<DepthMap> {
        self.predict_with(image, DepthOptions::default())
    }

    /// [`DepthPipeline::predict`] plus [`DepthOptions`]. The returned
    /// [`DepthMap`] is always `image`'s own size - the model's internal grid
    /// (its `input`-side working resolution) is resized back before return.
    pub fn predict_with(&self, image: &Image, opts: DepthOptions) -> Result<DepthMap> {
        let (w, h) = (image.width(), image.height());
        let hwc = image.to_hwc_unit();
        let out = self.session.predict(&hwc, w, h, opts.input);
        Ok(DepthMap { width: out.width, height: out.height, values: out.values, min: out.min, max: out.max })
    }
}

/// Builds a [`DepthPipeline`]. `.device(...)` is the only knob this milestone
/// exposes.
pub struct DepthPipelineBuilder {
    model_id: String,
    device: Device,
}

impl DepthPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resolve `model_id` and build a real [`DepthPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing`,
    ///    fetched only when nothing local already resolves it (mirrors every
    ///    other pipeline in this crate).
    /// 3. `crates/loader`'s resolver looks for `zipdepth::spec::
    ///    ZipdepthSpec`'s one `"weights"` role: a `.pt`/`.pth` archive whose
    ///    tensor shapes derive a real ZipDepth net
    ///    ([`zipdepth::config::ZipConfig::from_tensors`]).
    /// 4. `zipdepth::caps::load` re-derives the SAME checkpoint's config and
    ///    imports its full tensor data into a fresh [`gpu_core::Gpu`]-bound
    ///    [`zipdepth::caps::Session`].
    pub fn load(self) -> Result<DepthPipeline> {
        let DepthPipelineBuilder { model_id, device } = self;

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
        let outcome = loader::resolve_structured("zipdepth", &zipdepth::spec::ZipdepthSpec, &overrides).map_err(Error::Backend)?;
        let assembly = match outcome {
            brain_modelstore::resolve::Resolution::Resolved(a) => *a,
            brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
            brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("zipdepth: resolved assembly {:?} has no weights role", assembly.id)))?;

        let session = zipdepth::caps::load(&weights.to_string_lossy()).map_err(Error::Backend)?;
        Ok(DepthPipeline { session })
    }
}
