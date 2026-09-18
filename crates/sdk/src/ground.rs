// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`GroundingPipeline`]: brain's open-vocabulary visual-grounding surface,
//! over Florence-2. Resolved through `crates/loader`'s resolver against
//! `florence2::spec::Florence2Spec`'s one `"weights"` role - the same
//! resolver `brain do florence2 ground` uses.
//!
//! A NEW public domain type, [`GroundedBox`], joining [`crate::Detection`]/
//! [`crate::Mask`] as the vision/detection bucket's third non-`Image` result
//! shape - and a genuinely different one from `Detection`, not a fourth
//! backend of [`crate::DetectionPipeline`]: YOLOv8's `class` is a fixed
//! index into a checkpoint's own trained label set, while Florence-2's
//! `phrase` is OPEN-VOCABULARY - the caller's own free-text query, not a
//! class the model was trained to name in advance
//! (`florence2::caps`'s own module doc: "screenshot-in, bounding-box-out
//! oracle... a text `target` phrase in").
//!
//! Reuses `florence2::caps::FlorenceSession::ground` directly, by building a
//! plain `capability::Invocation` the SAME way [`crate::VisionLanguagePipeline`]
//! already does - one implementation, not two.
//!
//! ```no_run
//! let pipe = brain::GroundingPipeline::from_pretrained("microsoft/Florence-2-base")?;
//! let screenshot = brain::Image::open("app.png")?;
//! for b in pipe.ground(&screenshot, "the Login button")? {
//!     println!("{:?} at ({:.0},{:.0})-({:.0},{:.0})", b.phrase, b.x1, b.y1, b.x2, b.y2);
//! }
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use capability::{Blob, Invocation};
use serde_json::json;

use crate::{Device, Error, Image, Result};

/// One grounded box, in ORIGINAL image pixel coordinates (converted from
/// Florence-2's own `0.0..=1.0`-normalized output using the source image's
/// own width/height - see this module's doc for why `phrase` is
/// open-vocabulary rather than a fixed class index).
#[derive(Clone, Debug, PartialEq)]
pub struct GroundedBox {
    pub phrase: String,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// [`GroundingPipeline::ground_with`]'s one knob, mirroring
/// `florence2::caps::ground_spec`'s own `max_new_tokens` param.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroundingOptions {
    max_new_tokens: Option<u32>,
}

impl GroundingOptions {
    pub fn new() -> GroundingOptions {
        GroundingOptions::default()
    }

    /// Cap on generated tokens. Unset uses `florence2::caps`'s own default
    /// (64) - the same "0 means default" contract the wire param itself
    /// documents.
    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = Some(n);
        self
    }
}

/// [`crate::Image`] (interleaved RGB8, [`crate::Image::pixels`]) normalized
/// to the raw HWC `f32` `[0,1]` blob shape `capability::blob::decode_image`
/// reads - the exact same encode-side helper
/// [`crate::VisionLanguagePipeline`]'s own `image_blob` uses, mirrored here
/// rather than re-derived.
fn image_blob(image: &Image) -> Blob {
    let hwc: Vec<f32> = image.pixels().iter().map(|&b| b as f32 / 255.0).collect();
    capability::blob::image_blob(&hwc, image.width(), image.height(), 3)
}

/// `outcome.outputs["boxes"]`'s wire shape (`[{"phrase": ..., "bbox": [x0,
/// y0, x1, y1]}, ...]`, each coordinate normalized `0.0..=1.0` - see
/// `florence2::grounding::GroundedBox`'s own doc) parsed back into
/// [`GroundedBox`] at `width`/`height` pixel scale. Factored out so it is
/// testable with no real pipeline in hand, the same "factored out for
/// testability" shape `crate::vlm::check_image_count` already uses.
fn boxes_from_outcome(outcome: &capability::Outcome, width: u32, height: u32) -> Vec<GroundedBox> {
    let Some(list) = outcome.outputs.get("boxes").and_then(|v| v.as_array()) else { return Vec::new() };
    list.iter()
        .filter_map(|b| {
            let phrase = b.get("phrase")?.as_str()?.to_string();
            let bbox = b.get("bbox")?.as_array()?;
            if bbox.len() != 4 {
                return None;
            }
            let n = |i: usize| bbox[i].as_f64().unwrap_or(0.0) as f32;
            Some(GroundedBox { phrase, x1: n(0) * width as f32, y1: n(1) * height as f32, x2: n(2) * width as f32, y2: n(3) * height as f32 })
        })
        .collect()
}

/// `brain`'s open-vocabulary visual-grounding pipeline. One architecture
/// today (Florence-2).
pub struct GroundingPipeline {
    session: florence2::caps::FlorenceSession,
}

/// Hand-written, not derived: `florence2::caps::FlorenceSession` holds a
/// live GPU device handle (no `Debug` impl), the same reason
/// [`crate::UpscalePipeline`]'s own `Debug` is hand-written.
impl std::fmt::Debug for GroundingPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundingPipeline").finish()
    }
}

impl GroundingPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<GroundingPipeline> {
        GroundingPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> GroundingPipelineBuilder {
        GroundingPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`florence2::caps::manifest`) - reflected, not re-described,
    /// same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        florence2::caps::manifest()
    }

    /// Locate `target` (a free-text phrase or UI element description, e.g.
    /// `"the Login button"`) in `image`, at [`GroundingOptions`]'s defaults.
    /// Empty when nothing matched (`found: false` on the wire), never an
    /// error - a phrase genuinely not present in the image is a normal,
    /// expected outcome, not a failure.
    pub fn ground(&self, image: &Image, target: &str) -> Result<Vec<GroundedBox>> {
        self.ground_with(image, target, GroundingOptions::default())
    }

    /// [`GroundingPipeline::ground`] plus [`GroundingOptions`].
    pub fn ground_with(&self, image: &Image, target: &str, opts: GroundingOptions) -> Result<Vec<GroundedBox>> {
        let mut inv = Invocation::new().set("target", json!(target)).blob("image", image_blob(image));
        if let Some(n) = opts.max_new_tokens {
            inv = inv.set("max_new_tokens", json!(n));
        }
        let outcome = self.session.ground(&inv).map_err(Error::Backend)?;
        Ok(boxes_from_outcome(&outcome, image.width(), image.height()))
    }
}

/// Builds a [`GroundingPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct GroundingPipelineBuilder {
    model_id: String,
    device: Device,
}

impl GroundingPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resolve `model_id` and build a real [`GroundingPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `florence2::spec::Florence2Spec`'s one `"weights"` role, before
    ///    ever consulting `Store::local`/`plan` - the same reordering every
    ///    other pipeline in this crate applies now (see
    ///    `crate::depth::DepthPipelineBuilder::load`'s own doc for the real,
    ///    confirmed gap this order closes). `Florence2Spec::classify` reads
    ///    a real `config.json`'s `model_type` field directly (a plain HF
    ///    `transformers`-shaped directory, `TransformersRecipe`'s own
    ///    catch-all shape), so this specific architecture likely never hits
    ///    that gap in practice - applied proactively for consistency anyway,
    ///    not because a failure was reproduced here.
    /// 3. Only on `Missing` does `model_id` get parsed and, under
    ///    `DownloadPolicy::IfMissing`, fetched - then resolution is retried
    ///    once.
    /// 4. `florence2::caps::FlorenceSession::load` imports the resolved
    ///    directory's vision tower and tokenizer onto a fresh
    ///    [`gpu_core::Gpu`]; the language model itself rebuilds per call
    ///    (its shape depends on that call's tokenized prompt length - see
    ///    `FlorenceSession::load`'s own doc).
    pub fn load(self) -> Result<GroundingPipeline> {
        let GroundingPipelineBuilder { model_id, device } = self;

        crate::device::apply(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;
        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = match loader::resolve_structured("florence2", &florence2::spec::Florence2Spec, &overrides).map_err(Error::Backend)? {
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
                match loader::resolve_structured("florence2", &florence2::spec::Florence2Spec, &overrides).map_err(Error::Backend)? {
                    brain_modelstore::resolve::Resolution::Resolved(a) => *a,
                    brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
                    brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
                }
            }
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("florence2: resolved assembly {:?} has no weights role", assembly.id)))?;

        let gpu = gpu_core::Gpu::new(&florence2::caps::SERVING_PIPELINES);
        let session = florence2::caps::FlorenceSession::load(&weights.to_string_lossy(), gpu).map_err(Error::Backend)?;

        Ok(GroundingPipeline { session })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome_with_boxes(boxes: serde_json::Value) -> capability::Outcome {
        capability::Outcome::new().set("found", json!(true)).set("boxes", boxes)
    }

    /// The whole point of [`boxes_from_outcome`]: normalized `[0,1]`
    /// coordinates scale to the SOURCE image's own pixel size, not the
    /// model's internal frame.
    #[test]
    fn boxes_from_outcome_scales_normalized_coords_to_image_pixels() {
        let outcome = outcome_with_boxes(json!([{"phrase": "the Login button", "bbox": [0.25, 0.5, 0.75, 1.0]}]));
        let boxes = boxes_from_outcome(&outcome, 200, 100);
        assert_eq!(boxes, vec![GroundedBox { phrase: "the Login button".to_string(), x1: 50.0, y1: 50.0, x2: 150.0, y2: 100.0 }]);
    }

    /// `found: false` (an empty `boxes` array) is an empty `Vec`, not an
    /// error - a phrase genuinely absent from the image is a normal outcome.
    #[test]
    fn boxes_from_outcome_is_empty_when_nothing_matched() {
        let outcome = capability::Outcome::new().set("found", json!(false)).set("boxes", json!([]));
        assert!(boxes_from_outcome(&outcome, 200, 100).is_empty());
    }

    /// A malformed `bbox` (not exactly 4 numbers) is dropped, not a panic -
    /// mirrors `florence2::grounding::parse_boxes`'s own "drop, don't error"
    /// contract for a truncated generation.
    #[test]
    fn boxes_from_outcome_drops_a_malformed_bbox_without_panicking() {
        let outcome = outcome_with_boxes(json!([{"phrase": "x", "bbox": [0.1, 0.2, 0.3]}]));
        assert!(boxes_from_outcome(&outcome, 200, 100).is_empty());
    }
}
