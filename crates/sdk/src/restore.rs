// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`RestorePipeline`]: brain's blind face restoration surface, over
//! CodeFormer. Resolved through `crates/loader`'s resolver against
//! `codeformer::spec::CodeFormerSpec`'s one `"weights"` role -- the same
//! resolver `brain codeformer <verb>`/`brain do codeformer restore_face`
//! uses, never a parallel path.
//!
//! A sibling of [`crate::ImagePipeline`]/[`crate::UpscalePipeline`] under the
//! SAME `image` surface (CodeFormer's `brain_arch` row is registered
//! `Domain::Image`, like flux2/s3dit/rrdbnet) -- "generate an image",
//! "upscale an image" and "restore a face" are different capabilities (this
//! SDK's own "one pipeline type per capability" rule), so this is its own
//! public type, but all three return the SAME [`crate::Image`] domain type.
//!
//! # Scope this pipeline inherits from the model
//!
//! CodeFormer takes an **aligned** face and returns a fixed 512x512 restored
//! one -- any input size is resized to that on the device, and the graph is
//! forward-only (`codeformer::caps::Session::restore`'s own doc). An
//! unaligned photo still restores; it just is not the reference recipe
//! (detection + alignment is `crates/scrfd` + `crates/arcface`, not chained
//! in here).
//!
//! ```no_run
//! let pipe = brain::RestorePipeline::from_pretrained("sczhou/CodeFormer")?;
//! let face = brain::Image::open("face.png")?;
//! pipe.restore(&face)?.save("face_restored.png")?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use crate::{Device, Error, Image, Result};

/// [`RestorePipeline::restore_with`]'s one knob, mirroring `codeformer::
/// caps::restore_spec`'s own `w` param: the identity-fidelity dial. `0.0`
/// (the default) is maximum quality (no encoder feature reaches the
/// generator); `1.0` is maximum fidelity to the input. Must be in `[0, 1]`;
/// `restore_with` names the bound in its error rather than clamping
/// silently.
#[derive(Clone, Copy, Debug)]
pub struct RestoreOptions {
    fidelity: f32,
}

impl Default for RestoreOptions {
    fn default() -> RestoreOptions {
        RestoreOptions { fidelity: 0.5 }
    }
}

impl RestoreOptions {
    pub fn new() -> RestoreOptions {
        RestoreOptions::default()
    }

    pub fn fidelity(mut self, w: f32) -> Self {
        self.fidelity = w;
        self
    }
}

/// `brain`'s face restoration pipeline. One architecture today (CodeFormer)
/// -- see this module's doc for why it is not folded into
/// [`crate::ImagePipeline`]/[`crate::UpscalePipeline`].
pub struct RestorePipeline {
    session: codeformer::caps::Session,
}

/// Hand-written, not derived: `codeformer::caps::Session` holds a live GPU
/// device handle (no `Debug` impl), the same reason
/// [`crate::UpscalePipeline`]'s own `Debug` is hand-written.
impl std::fmt::Debug for RestorePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cfg = self.session.config();
        f.debug_struct("RestorePipeline")
            .field("dim_embd", &cfg.dim_embd)
            .field("n_layers", &cfg.n_layers)
            .field("img_size", &cfg.img_size())
            .finish()
    }
}

impl RestorePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<RestorePipeline> {
        RestorePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> RestorePipelineBuilder {
        RestorePipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`codeformer::caps::manifest`) -- reflected, not
    /// re-described, same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        codeformer::caps::manifest()
    }

    /// Restore `image` at [`RestoreOptions`]'s default fidelity (`0.5`).
    pub fn restore(&self, image: &Image) -> Result<Image> {
        self.restore_with(image, RestoreOptions::default())
    }

    /// [`RestorePipeline::restore`] plus [`RestoreOptions`]. The returned
    /// [`Image`] is always the graph's fixed square side (512 on the
    /// released checkpoint), regardless of `image`'s own size.
    pub fn restore_with(&self, image: &Image, opts: RestoreOptions) -> Result<Image> {
        let (w, h) = (image.width(), image.height());
        let hwc = image.to_hwc_unit();
        let out = self.session.restore(&hwc, w, h, opts.fidelity).map_err(Error::Backend)?;
        Image::from_hwc_unit(out.side, out.side, &out.pixels)
    }
}

/// Builds a [`RestorePipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct RestorePipelineBuilder {
    model_id: String,
    device: Device,
}

impl RestorePipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resolve `model_id` and build a real [`RestorePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `codeformer::spec::CodeFormerSpec`'s one `"weights"` role (a
    ///    `.pt`/`.pth` archive whose tensor names match CodeFormer's one
    ///    fixed preset), before ever consulting `Store::local`/`plan` -
    ///    deliberately the reverse of the naive "check `Store::local`, then
    ///    fetch-if-missing, then resolve" order, for the same real reason
    ///    `crate::tts::TtsPipelineBuilder::load`/`crate::depth::DepthPipelineBuilder::load`
    ///    already apply it: a real released `codeformer.pth` (no
    ///    `brain_modelstore::recipe::FilesRecipe` entry exists for it)
    ///    satisfies neither of `Store::local`'s two narrow shapes, so
    ///    `Store::local` never recognizes even an already-downloaded release
    ///    and `plan()` queries the hub for a reference that is already fully
    ///    present on disk - confirmed empirically while building this fix
    ///    (`Error::Download("not found: <id>@main")` against a real local
    ///    fixture with no network access at all).
    /// 3. Only on `Missing` does `model_id` get parsed and, under
    ///    `DownloadPolicy::IfMissing`, fetched - then resolution is retried
    ///    once.
    /// 4. `codeformer::caps::load` re-reads the SAME checkpoint's full tensor
    ///    data and builds the generator on a fresh [`gpu_core::Gpu`].
    pub fn load(self) -> Result<RestorePipeline> {
        let RestorePipelineBuilder { model_id, device } = self;

        crate::device::apply(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;
        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = match loader::resolve_structured("codeformer", &codeformer::spec::CodeFormerSpec, &overrides).map_err(Error::Backend)? {
            brain_modelstore::resolve::Resolution::Resolved(a) => *a,
            brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
            brain_modelstore::resolve::Resolution::Missing(_) => {
                let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
                let store = brain_modelstore::Store::new(root);
                let hub = brain_modelstore::HfHub::new();
                if store.local(&reference).is_none() {
                    let plan = brain_modelstore::plan(&reference, &store, &hub)?;
                    loader::supply::execute_plan(&store, &hub, &plan, &model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
                }
                match loader::resolve_structured("codeformer", &codeformer::spec::CodeFormerSpec, &overrides).map_err(Error::Backend)? {
                    brain_modelstore::resolve::Resolution::Resolved(a) => *a,
                    brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
                    brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
                }
            }
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("codeformer: resolved assembly {:?} has no weights role", assembly.id)))?;

        let gpu = gpu_core::Gpu::new(&codeformer::caps::SERVING_PIPELINES);
        let model = codeformer::caps::load(&weights.to_string_lossy(), gpu).map_err(Error::Backend)?;

        Ok(RestorePipeline { session: codeformer::caps::Session::new(model) })
    }
}
