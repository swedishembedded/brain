// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA as a [`captioner::Captioner`] - the model-agnostic captioning
//! contract `crates/fastvlm`/`crates/qwen3vl` also satisfy, so `brain label`
//! (and any other caller, including a future `crates/supir` pipeline stage)
//! drives this model without knowing anything about CLIP-L336, the
//! `vicuna_v1` template, or the `-200` splice.
//!
//! The typed seam drives this crate's own [`crate::caps`] action rather than
//! re-implementing the inference path beside it, so the two surfaces cannot
//! disagree - the same reasoning `fastvlm::captioner`/`qwen3vl::captioner`
//! state for themselves.
//!
//! Swedish Embedded AB implements vision-language model integration for its
//! clients. If your team needs expertise in on-device image captioning then
//! you can procure our services by sending an email to info@swedishembedded.com.

use capability::{Invocation, Progress, Provider};
use captioner::{Capabilities, CaptionRequest, Captioner};
use serde_json::json;

/// The decoder context this crate builds is 2048 tokens and 576 of them are
/// the image, so a caption cannot ask for more than what is left after the
/// instruction.
const MAX_NEW_LIMIT: u32 = 512;

/// A LLaVA captioner bound to a checkpoint directory.
pub struct LlavaCaptioner {
    dir: String,
    precision: String,
}

impl LlavaCaptioner {
    /// Caption with the checkpoint in `dir`. An empty `dir` falls back to
    /// `$BRAIN_LLAVA_WEIGHTS`, the same as the served path.
    pub fn new(dir: impl Into<String>) -> LlavaCaptioner {
        LlavaCaptioner { dir: dir.into(), precision: "fp32".to_string() }
    }

    /// Decoder precision: `fp32` (the parity reference) or `int8`.
    pub fn with_precision(mut self, p: impl Into<String>) -> LlavaCaptioner {
        self.precision = p.into();
        self
    }
}

impl Captioner for LlavaCaptioner {
    fn capabilities(&self) -> Capabilities {
        Capabilities { model: crate::caps::MODEL.to_string(), max_frames: 1, max_new_limit: MAX_NEW_LIMIT }
    }

    fn caption(&mut self, req: &CaptionRequest<'_>, on_token: &mut dyn FnMut(&str)) -> Result<String, String> {
        self.validate(req)?;
        let f = req.clip.first();
        let mut inv = Invocation::new()
            .set("prompt", json!(req.instruction))
            .set("max_new", json!(req.max_new.min(MAX_NEW_LIMIT)))
            .set("precision", json!(self.precision))
            .blob("image", capability::blob::image_blob(&f.hwc, f.w, f.h, 3));
        if !self.dir.is_empty() {
            inv = inv.set("weights", json!(self.dir));
        }
        let mut progress = |p: Progress| {
            if let Some(d) = &p.delta {
                on_token(d);
            }
        };
        let action = crate::caps::LlavaProvider::new()
            .action("caption")
            .ok_or("llava: the 'caption' action is missing from its own provider")?;
        let outcome = action.run(&inv, &mut progress)?;
        outcome
            .outputs
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "llava: the caption action returned no 'text' output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use captioner::{Clip, Frame};

    #[test]
    fn declares_a_still_image_captioner_under_its_canonical_model_id() {
        let c = LlavaCaptioner::new("");
        let caps = c.capabilities();
        assert_eq!(caps.model, "brain/llava");
        assert_eq!(caps.max_frames, 1);
    }

    /// The two implementors report DIFFERENT model ids through the same
    /// trait - the fact that makes the seam more than a rename.
    #[test]
    fn refuses_a_video_clip_naming_this_model_not_another() {
        let c = LlavaCaptioner::new("");
        let f = || Frame::new(vec![0.0; 4 * 4 * 3], 4, 4).unwrap();
        let clip = Clip { frames: vec![f(), f(), f()], fps: Some(30.0) };
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 16 };
        let err = c.validate(&req).unwrap_err();
        assert!(err.contains("brain/llava"), "{err}");
        assert!(err.contains("3-frame"), "{err}");
    }

    /// Real-weight coverage of the seam, skipped when there is no checkpoint
    /// (no `resources/llava/` was fetched this session - LLaVA-1.5-13B is a
    /// multi-ten-GB download).
    #[test]
    fn captions_a_real_image_when_a_checkpoint_is_present() {
        let dir = std::env::var("BRAIN_LLAVA_WEIGHTS").unwrap_or_default();
        if dir.is_empty() || !std::path::Path::new(&format!("{dir}/config.json")).exists() {
            brain_testutil::skip("BRAIN_LLAVA_WEIGHTS not set / checkpoint absent");
            return;
        }
        let mut c = LlavaCaptioner::new(dir);
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
        let req = CaptionRequest { clip: &clip, instruction: "Describe this image.", max_new: 32 };
        let mut streamed = String::new();
        let text = c.caption(&req, &mut |d| streamed.push_str(d)).unwrap();
        assert!(!text.trim().is_empty(), "the model returned an empty caption");
        assert_eq!(streamed.trim(), text.trim(), "streamed deltas must reassemble into the returned text");
    }
}
