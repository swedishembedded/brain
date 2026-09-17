// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`UpscalePipeline`]: brain's image super-resolution surface, over RRDBNet
//! (Real-ESRGAN's generator, the discriminator being training-only).
//! Resolved through `crates/loader`'s resolver against
//! `rrdbnet::spec::RrdbnetSpec`'s one `"weights"` role -- the same resolver
//! `brain rrdbnet upscale`/`brain do rrdbnet upscale` uses, never a parallel
//! path.
//!
//! A sibling of [`crate::ImagePipeline`] under the SAME `image` surface
//! (RRDBNet's `brain_arch` row is registered `Domain::Image`, like flux2/
//! s3dit) -- "generate an image" and "upscale an image" are different
//! capabilities (this SDK's own "one pipeline type per capability" rule), so
//! this is its own public type rather than a third `ImagePipeline` backend,
//! but both return the SAME [`crate::Image`] domain type.
//!
//! ```no_run
//! let pipe = brain::UpscalePipeline::from_pretrained("schwgHao/RealESRGAN_x4plus")?;
//! let small = brain::Image::open("photo.png")?;
//! pipe.upscale(&small)?.save("photo_4x.png")?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use crate::{Device, Error, Image, Result};

/// [`UpscalePipeline::upscale_with`]'s one knob, mirroring
/// `rrdbnet::caps::upscale_spec`'s own `tile` param: process in tiles of
/// this many input pixels a side (`0`, the default = the whole image in one
/// pass). Peak VRAM is quadratic in the tile side, so a large image on a
/// small card needs this; it changes the memory/latency trade-off, not the
/// task -- `rrdbnet::caps::TILE_HALO` bounds the resulting seam, it does not
/// remove it (see that constant's own doc for the measured cost).
#[derive(Clone, Debug, Default)]
pub struct UpscaleOptions {
    tile: u32,
}

impl UpscaleOptions {
    pub fn new() -> UpscaleOptions {
        UpscaleOptions::default()
    }

    pub fn tile(mut self, pixels: u32) -> Self {
        self.tile = pixels;
        self
    }
}

/// `brain`'s image super-resolution pipeline. One architecture today
/// (RRDBNet) -- see this module's doc for why it is not folded into
/// [`crate::ImagePipeline`].
pub struct UpscalePipeline {
    session: rrdbnet::caps::Session,
}

/// Hand-written, not derived: `rrdbnet::caps::Session` holds a live GPU
/// device handle (no `Debug` impl), the same reason
/// [`crate::ImagePipeline`]'s own `Debug` is hand-written.
impl std::fmt::Debug for UpscalePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cfg = self.session.config();
        f.debug_struct("UpscalePipeline")
            .field("num_feat", &cfg.num_feat)
            .field("num_block", &cfg.num_block)
            .field("scale", &cfg.scale)
            .finish()
    }
}

impl UpscalePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<UpscalePipeline> {
        UpscalePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> UpscalePipelineBuilder {
        UpscalePipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`rrdbnet::caps::manifest`) -- reflected, not re-described,
    /// same reason [`crate::ImagePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        rrdbnet::caps::manifest()
    }

    /// Upscale `image` by this pipeline's fixed scale factor (`4x` for the
    /// released `x4plus`/`x4plus_anime_6B` checkpoints, derived from the
    /// resolved checkpoint's own shape -- never hardcoded), at every
    /// [`UpscaleOptions`] left at its default.
    pub fn upscale(&self, image: &Image) -> Result<Image> {
        self.upscale_with(image, UpscaleOptions::default())
    }

    /// [`UpscalePipeline::upscale`] plus [`UpscaleOptions`].
    pub fn upscale_with(&self, image: &Image, opts: UpscaleOptions) -> Result<Image> {
        use rrdbnet::caps::Upscaler;

        let (w, h) = (image.width(), image.height());
        let hwc = image.to_hwc_unit();
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
        let (out, ow, oh) = self.session.upscale(&chw, w, h, opts.tile).map_err(Error::Backend)?;
        let hwc_out = imaging::pixels::chw_to_hwc(&out, 3, oh as usize, ow as usize);
        Image::from_hwc_unit(ow, oh, &hwc_out)
    }
}

/// Builds an [`UpscalePipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct UpscalePipelineBuilder {
    model_id: String,
    device: Device,
}

impl UpscalePipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resolve `model_id` and build a real [`UpscalePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing`,
    ///    fetched only when nothing local already resolves it (mirrors every
    ///    other pipeline in this crate).
    /// 3. `crates/loader`'s resolver looks for `rrdbnet::spec::RrdbnetSpec`'s
    ///    one `"weights"` role: a `.pt`/`.pth` archive whose tensor names
    ///    derive a real RRDBNet shape ([`rrdbnet::config::RrdbConfig::from_tensors`]).
    /// 4. `rrdbnet::caps::load` re-reads the SAME checkpoint's full tensor
    ///    data, re-derives the identical config, and builds the generator on
    ///    a fresh [`gpu_core::Gpu`].
    pub fn load(self) -> Result<UpscalePipeline> {
        let UpscalePipelineBuilder { model_id, device } = self;

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
        let outcome = loader::resolve_structured("rrdbnet", &rrdbnet::spec::RrdbnetSpec, &overrides).map_err(Error::Backend)?;
        let assembly = match outcome {
            brain_modelstore::resolve::Resolution::Resolved(a) => *a,
            brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
            brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("rrdbnet: resolved assembly {:?} has no weights role", assembly.id)))?;

        let gpu = gpu_core::Gpu::new(&rrdbnet::KERNELS);
        let session = rrdbnet::caps::load(&weights.to_string_lossy(), gpu).map_err(Error::Backend)?;

        Ok(UpscalePipeline { session })
    }
}
