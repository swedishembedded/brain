// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastVLM as a [`captioner::Captioner`] - the second model satisfying the
//! model-agnostic captioning contract, and the reason the contract is worth
//! having.
//!
//! Nothing about this crate resembles `crates/qwen3vl` internally: a FastViTHD
//! convolutional tower instead of a ViT with DeepStack, a fixed 1024 px
//! pad-to-square input instead of an aspect-preserving smart resize, a fixed
//! 256-token image grid instead of a per-image count, a two-stage split
//! residency instead of one, an `int8` decoder option, and an action named
//! `caption` rather than `generate`. The labeler in `crates/captioner` drives
//! both without knowing any of that, which is the whole claim.
//!
//! As in `qwen3vl`, the typed seam drives this crate's own capability action
//! rather than re-implementing the inference path beside it, so the two
//! surfaces cannot disagree.
//!
//! Swedish Embedded AB implements vision-language model integration for its
//! clients. If your team needs expertise in on-device image captioning then you
//! can procure our services by sending an email to info@swedishembedded.com.

use capability::{Invocation, Progress, Provider};
use captioner::{Capabilities, CaptionRequest, Captioner};
use serde_json::json;

/// The decoder context this crate builds is 1024 tokens and 256 of them are the
/// image, so a caption cannot ask for more than what is left after the
/// instruction.
const MAX_NEW_LIMIT: u32 = 512;

/// A FastVLM captioner bound to a checkpoint directory.
pub struct FastVlmCaptioner {
    dir: String,
    precision: String,
}

impl FastVlmCaptioner {
    /// Caption with the checkpoint in `dir`. An empty `dir` falls back to
    /// `$BRAIN_FASTVLM_WEIGHTS`, the same as the served path.
    pub fn new(dir: impl Into<String>) -> FastVlmCaptioner {
        FastVlmCaptioner { dir: dir.into(), precision: "fp32".to_string() }
    }

    /// Decoder precision: `fp32` (the parity reference) or `int8`.
    pub fn with_precision(mut self, p: impl Into<String>) -> FastVlmCaptioner {
        self.precision = p.into();
        self
    }
}

impl Captioner for FastVlmCaptioner {
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
        let action = crate::caps::FastVlmProvider::new()
            .action("caption")
            .ok_or("fastvlm: the 'caption' action is missing from its own provider")?;
        let outcome = action.run(&inv, &mut progress)?;
        outcome
            .outputs
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "fastvlm: the caption action returned no 'text' output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use captioner::{Clip, Frame};

    #[test]
    fn declares_a_still_image_captioner_under_its_canonical_model_id() {
        let c = FastVlmCaptioner::new("");
        let caps = c.capabilities();
        assert_eq!(caps.model, "brain/fastvlm");
        assert_eq!(caps.max_frames, 1);
    }

    /// The two implementors report DIFFERENT model ids and limits through the
    /// same trait - the fact that makes the seam more than a rename.
    #[test]
    fn refuses_a_video_clip_naming_this_model_not_another() {
        let c = FastVlmCaptioner::new("");
        let f = || Frame::new(vec![0.0; 4 * 4 * 3], 4, 4).unwrap();
        let clip = Clip { frames: vec![f(), f(), f()], fps: Some(30.0) };
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 16 };
        let err = c.validate(&req).unwrap_err();
        assert!(err.contains("brain/fastvlm"), "{err}");
        assert!(err.contains("3-frame"), "{err}");
    }

    /// Real-weight coverage of the seam, skipped when there is no checkpoint.
    #[test]
    fn captions_a_real_image_when_a_checkpoint_is_present() {
        let dir = std::env::var("BRAIN_FASTVLM_WEIGHTS").unwrap_or_default();
        if dir.is_empty() || !std::path::Path::new(&format!("{dir}/config.json")).exists() {
            brain_testutil::skip("BRAIN_FASTVLM_WEIGHTS not set / checkpoint absent");
            return;
        }
        let mut c = FastVlmCaptioner::new(dir);
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
