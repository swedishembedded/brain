// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pipeline behind the generalized [`capability`] interface.
//!
//! This is where "one `Run`, not four" stops being a design note and becomes
//! true: `brain do imgpipe run --stages '…'` and the D-Bus `Run` method both
//! take an image plus a stage list and return the composited result, with every
//! intermediate staying inside the process. Composing the same recipe
//! client-side would move a full-resolution image across the bus after each
//! stage.
//!
//! # It dispatches into the same registry it lives in
//!
//! [`PipelineAction`] holds a [`Registry`] built from the model providers, and
//! [`Pipeline::run`] calls `segment` / `restore_face` through it. So the
//! pipeline is a capability that *composes* capabilities, with no special-casing
//! anywhere: adding a stage means teaching [`crate::Stage`] one more `op` and
//! registering the model, not extending the transport.
//!
//! The registry is supplied by the caller rather than built here, because which
//! models are available is an environment question (`BRAIN_SAM2_WEIGHTS`,
//! `BRAIN_RESTORE_WEIGHTS`, …) that `crates/cli` already answers. A stage whose
//! model is not registered fails with that model's own "set BRAIN_… " message,
//! which is more useful than a generic one from here.

use std::sync::Arc;

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider, Registry,
};
use gpu_core::Gpu;
use serde_json::json;

use crate::{Pipeline, Spec};

/// The model id used on the CLI (`brain do imgpipe …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "imgpipe";

/// The kernels the pipeline itself dispatches — the mask algebra and the
/// composite. Model stages bring their own.
pub const PIPELINES: &[(&str, &str)] = imaging::PIPELINES;

fn run_spec() -> ActionSpec {
    ActionSpec::new(
        "run",
        "compose segmentation, mask refinement, restoration and an optional upscale tail into one call; \
         pixels outside the mask are returned bit-identical AT SOURCE RESOLUTION — an `upscale` tail runs \
         after the composite and resamples everything, so the guarantee is about what the composite wrote, \
         not about the final pixels",
    )
    .param(
        ParamSpec::new(
            "stages",
            ParamType::Str,
            r#"JSON stage list, e.g. {"stages":[{"op":"segment","points":[[120,80]]},{"op":"dilate","radius":4},{"op":"restore","w":0.7},{"op":"upscale"}]}; `upscale` changes the image size so it must be LAST"#,
        )
        .required(),
    )
    .input(BlobSpec::new("image", Media::Image, "the source image").required())
    .output(BlobSpec::new("image", Media::Image, "the composited result"))
    .output(BlobSpec::new(
        "mask",
        Media::Mask,
        "the mask actually composited with — the record of which pixels were authorised to move",
    ))
}

/// The full, static capability manifest — safe to build with no models loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Composed imaging pipeline: segment, refine, restore and composite in one call, leaving unselected pixels bit-identical.",
        vec![run_spec()],
    )
}

/// The pipeline as a provider.
///
/// `models` is the registry the stages dispatch into. It is deliberately NOT
/// built here — see the module docs.
pub struct PipelineProvider {
    models: Arc<Registry>,
}

impl PipelineProvider {
    pub fn new(models: Arc<Registry>) -> PipelineProvider {
        PipelineProvider { models }
    }
}

impl Provider for PipelineProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "run").then(|| Arc::new(PipelineAction { models: self.models.clone() }) as Arc<dyn Action>)
    }
}

struct PipelineAction {
    models: Arc<Registry>,
}

impl Action for PipelineAction {
    fn spec(&self) -> ActionSpec {
        run_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let stages = inv.get_str("stages").ok_or("imgpipe: 'stages' is required")?;
        let spec = Spec::parse(&stages)?;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);

        // One device for the whole pipeline. Every stage's intermediate stays
        // here; only the final image and its mask leave.
        let gpu = Gpu::new(PIPELINES);
        let out = Pipeline::new(&gpu, &self.models).run(&spec, &chw, w, h)?;

        let hwc_out = imaging::pixels::chw_to_hwc(&out.image, 3, h as usize, w as usize);
        Ok(Outcome::new()
            .set("stages", json!(spec.stages.len()))
            .set("edits", json!(spec.edits()))
            .blob("image", capability::blob::image_blob(&hwc_out, w, h, 3))
            .blob(
                "mask",
                Blob::new(Media::Mask, out.mask.iter().flat_map(|v| v.to_le_bytes()).collect())
                    .with_meta(json!({"w": w, "h": h, "c": 1, "dtype": "f32"})),
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_declares_one_action_with_both_outputs() {
        let m = manifest();
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "run");
        let outs: Vec<&str> = a.outputs.iter().map(|b| b.name.as_str()).collect();
        // The mask is not optional decoration: it is how a caller learns which
        // pixels the run was allowed to change (feathering widens that set).
        assert!(outs.contains(&"image") && outs.contains(&"mask"), "outputs were {outs:?}");
    }

    #[test]
    fn a_bad_stage_list_fails_before_any_device_work() {
        // Parsing happens first, so a typo costs no upload and no model build.
        assert!(Spec::parse(r#"{"stages":[{"op":"nope"}]}"#).is_err());
    }
}
