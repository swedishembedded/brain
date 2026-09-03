// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL as a [`captioner::Captioner`] - one of the models that satisfies
//! the model-agnostic captioning contract.
//!
//! The contract lives in `crates/captioner` and knows nothing about this model;
//! the dependency runs one way, so any vision-language model can join by adding
//! a file like this one and nothing else. This crate's own inference path is
//! its `generate` capability action, and that is what this drives: the typed
//! seam and the served surface therefore run **the same code**, and cannot
//! drift about preprocessing, prompt assembly or decoding the way two parallel
//! implementations would.
//!
//! Still images AND short video clips: [`Captioner::validate`] refuses a clip
//! longer than `max_frames` by name (`crate::caps::MAX_VIDEO_FRAMES` - a real,
//! bounded scope decision, not hours-long/streaming video). A still image
//! ([`captioner::Clip::is_still`]) takes the `image` input; a multi-frame
//! clip takes `video` + `fps` (the ONLY per-clip timing `captioner::Clip`
//! carries) so the served action can place each frame group in REAL time
//! (`crate::mrope::get_rope_index_video`) rather than by frame count - a clip
//! with more than one frame and no known `fps` is refused rather than
//! guessed at.
//!
//! Swedish Embedded AB implements vision-language model integration for its
//! clients. If your team needs expertise in deploying VLMs for captioning or
//! visual question answering then you can procure our services by sending an
//! email to info@swedishembedded.com.

use capability::{Invocation, Progress, Provider};
use captioner::{Capabilities, CaptionRequest, Captioner};
use serde_json::json;

/// A Qwen3-VL captioner bound to a checkpoint directory.
///
/// Construction is cheap: the (multi-GB, minutes-long) model build happens
/// lazily on the first caption and is then held by this crate's own resident,
/// keyed by `(dir, max_pixels)` - so labeling a folder loads once, not once per
/// image.
pub struct Qwen3VlCaptioner {
    dir: String,
    max_pixels: u32,
    precision: crate::caps::Precision,
}

impl Qwen3VlCaptioner {
    /// Caption with the checkpoint in `dir`. An empty `dir` falls back to
    /// `$BRAIN_QWEN3VL_WEIGHTS`, the same as the served path.
    pub fn new(dir: impl Into<String>) -> Qwen3VlCaptioner {
        Qwen3VlCaptioner { dir: dir.into(), max_pixels: DEFAULT_MAX_PIXELS, precision: crate::caps::Precision::F32 }
    }

    /// Build the decoder at a narrower storage tier.
    ///
    /// `int8` is LOSSY and is never the default: a captioning run is usually
    /// making training data, and a caption that is faster and subtly wrong is
    /// a worse trade there than anywhere else. `qwen3vl_bench compare` prints
    /// both tiers' captions side by side with the time and the divergence,
    /// which is the evidence this choice should be made on.
    pub fn with_precision(mut self, p: crate::caps::Precision) -> Qwen3VlCaptioner {
        self.precision = p;
        self
    }

    /// Cap the input image area. Larger images are downsampled to fit by the
    /// model's own smart-resize; this decides how much detail the vision tower
    /// gets, and therefore how specific the caption can be.
    pub fn with_max_pixels(mut self, px: u32) -> Qwen3VlCaptioner {
        self.max_pixels = px.max(1);
        self
    }
}

/// The resident capacity a caption run asks for. Larger than the served
/// default: a labeling pass is one image at a time with no concurrent tenant to
/// budget against, and detail in the caption is the whole point.
const DEFAULT_MAX_PIXELS: u32 = 1280 * 1280;

/// Room for a long, detailed caption on top of the image tokens, inside the
/// decoder context this crate builds.
const MAX_NEW_LIMIT: u32 = 1024;

impl Captioner for Qwen3VlCaptioner {
    fn capabilities(&self) -> Capabilities {
        Capabilities { model: crate::caps::MODEL.to_string(), max_frames: crate::caps::MAX_VIDEO_FRAMES, max_new_limit: MAX_NEW_LIMIT }
    }

    fn caption(&mut self, req: &CaptionRequest<'_>, on_token: &mut dyn FnMut(&str)) -> Result<String, String> {
        self.validate(req)?;
        let mut inv = Invocation::new()
            .set("prompt", json!(req.instruction))
            .set("max_new", json!(req.max_new.min(MAX_NEW_LIMIT)))
            .set("max_pixels", json!(self.max_pixels))
            .set("precision", json!(self.precision.name()));
        if !self.dir.is_empty() {
            inv = inv.set("weights", json!(self.dir));
        }
        inv = if req.clip.is_still() {
            let f = req.clip.first();
            inv.blob("image", capability::blob::image_blob(&f.hwc, f.w, f.h, 3))
        } else {
            // A multi-frame clip with no known fps cannot be placed in real
            // time -- rather than duplicate that check here, leave 'fps'
            // unset and let the SAME check `caps::GenerateAction::run` makes
            // (before it ever touches a checkpoint) produce the error, so
            // there is exactly one place this refusal is worded.
            let frames: Vec<(Vec<f32>, u32, u32)> = req.clip.frames.iter().map(|f| (f.hwc.clone(), f.w, f.h)).collect();
            let video_blob = capability::blob::video_blob(&frames)?;
            let inv = if let Some(fps) = req.clip.fps { inv.set("fps", json!(fps)) } else { inv };
            inv.blob("video", video_blob)
        };
        // The action streams real per-token deltas; forward only those, not the
        // step messages, so a caller concatenating them reconstructs the caption
        // exactly.
        let mut progress = |p: Progress| {
            if let Some(d) = &p.delta {
                on_token(d);
            }
        };
        let action = crate::caps::QwenVlProvider::new()
            .action("generate")
            .ok_or("qwen3vl: the 'generate' action is missing from its own provider")?;
        let outcome = action.run(&inv, &mut progress)?;
        outcome
            .outputs
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "qwen3vl: the generate action returned no 'text' output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use captioner::{Clip, Frame};

    /// The capability self-description is what `label_dir` and
    /// `Captioner::validate` act on, so it has to be right without a checkpoint.
    #[test]
    fn declares_a_bounded_video_window_under_its_canonical_model_id() {
        let c = Qwen3VlCaptioner::new("");
        let caps = c.capabilities();
        assert_eq!(caps.model, "brain/qwen3vl");
        assert_eq!(caps.max_frames, crate::caps::MAX_VIDEO_FRAMES, "must match caps.rs's own bound exactly, not a second copy of the number");
        assert!(caps.max_frames > 1, "the whole point of this change is that this is no longer a still-image-only captioner");
        assert!(caps.max_new_limit >= 256, "a detailed caption needs a real token budget");
    }

    /// A clip beyond this model's declared window must be refused before any
    /// checkpoint is touched -- this test would need a multi-GB model to run
    /// otherwise, which is exactly why the refusal belongs in `validate`.
    #[test]
    fn refuses_a_clip_over_the_frame_cap_without_loading_anything() {
        let c = Qwen3VlCaptioner::new("");
        let f = || Frame::new(vec![0.0; 4 * 4 * 3], 4, 4).unwrap();
        let n = (crate::caps::MAX_VIDEO_FRAMES + 1) as usize;
        let clip = Clip { frames: (0..n).map(|_| f()).collect(), fps: Some(24.0) };
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 16 };
        let err = c.validate(&req).unwrap_err();
        assert!(err.contains("brain/qwen3vl"), "{err}");
        assert!(err.contains(&format!("{n}-frame")), "{err}");
    }

    /// A multi-frame clip with NO known fps cannot be placed in real time
    /// (the whole point of this change is that the T axis is driven by REAL
    /// elapsed time, not a guessed frame rate), so it is refused BY NAME --
    /// and, since the check happens before the checkpoint dir is even read,
    /// without loading anything.
    #[test]
    fn refuses_a_multi_frame_clip_with_no_known_fps_without_loading_anything() {
        let mut c = Qwen3VlCaptioner::new("");
        let f = || Frame::new(vec![0.0; 4 * 4 * 3], 4, 4).unwrap();
        let clip = Clip { frames: vec![f(), f()], fps: None };
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 16 };
        let err = c.caption(&req, &mut |_| {}).unwrap_err();
        assert!(err.contains("fps"), "{err}");
    }

    /// A still image must keep working exactly as before -- the video path
    /// added alongside it must not disturb the `is_still()` branch.
    #[test]
    fn a_still_image_still_uses_the_image_path_without_an_fps() {
        // No checkpoint is set, so a real run isn't reachable here; this
        // only proves a still image does NOT hit the "needs a known fps"
        // refusal a multi-frame clip does -- it must fail (no checkpoint
        // dir), but for a DIFFERENT, later reason.
        let mut c = Qwen3VlCaptioner::new("");
        let clip = Clip::still(Frame::new(vec![0.0; 4 * 4 * 3], 4, 4).unwrap());
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 16 };
        let err = c.caption(&req, &mut |_| {}).unwrap_err();
        assert!(!err.contains("fps"), "a still image must not be routed through the fps-requiring video path: {err}");
    }

    /// The seam must reach the real model on a real checkpoint. Skips when
    /// there is none, like this crate's other real-weight coverage.
    #[test]
    fn captions_a_real_image_when_a_checkpoint_is_present() {
        let dir = std::env::var("BRAIN_QWEN3VL_WEIGHTS").unwrap_or_default();
        if dir.is_empty() || !std::path::Path::new(&format!("{dir}/config.json")).exists() {
            brain_testutil::skip("BRAIN_QWEN3VL_WEIGHTS not set / checkpoint absent");
            return;
        }
        let mut c = Qwen3VlCaptioner::new(dir).with_max_pixels(256 * 256);
        // A red/blue split, so a right answer is checkable without a golden.
        let (w, h) = (64u32, 64u32);
        let mut hwc = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 3;
                if x < 32 {
                    hwc[i] = 1.0;
                } else {
                    hwc[i + 2] = 1.0;
                }
            }
        }
        let clip = Clip::still(Frame::new(hwc, w, h).unwrap());
        let req = CaptionRequest { clip: &clip, instruction: "Name the two colours in this image.", max_new: 48 };
        let mut streamed = String::new();
        let text = c.caption(&req, &mut |d| streamed.push_str(d)).unwrap();
        assert!(!text.trim().is_empty(), "the model returned an empty caption");
        assert_eq!(streamed.trim(), text.trim(), "streamed deltas must reassemble into the returned text");
    }
}
