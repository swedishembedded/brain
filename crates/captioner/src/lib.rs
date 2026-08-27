// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Captioning as a seam**, and the dataset labeler built on it.
//!
//! Labeling a dataset is not a property of any one vision-language model. The
//! work - walk a folder, ask a model to describe each item, write a caption
//! file a trainer can read, and be able to stop and resume - is identical
//! whether the describing is done by Qwen3-VL, FastVLM, Moondream 3 or a model
//! that does not exist yet. So what this crate exports first is the contract
//! ([`Captioner`]) and only then the driver that uses it ([`label`]). There is
//! no model code here and no dependency on any model crate; the implementors
//! live in the model crates and depend on this one.
//!
//! * [`Clip`] - the unit being captioned, and the whole of the video design.
//! * [`Captioner`] - a model that turns a [`Clip`] plus an instruction into
//!   text, with [`Capabilities`] describing what it will accept.
//! * [`label`] - the resumable folder labeler: the reusable workflow.
//!
//! # Why the unit is a clip and not an image
//!
//! A video model captions a *clip*, not a frame - "a woman turns and walks out
//! of frame" is not a statement about any single image, and a seam that passed
//! one image would force a video implementor to either re-shape the trait or
//! caption a thumbnail and call it a video caption. So the unit here is
//! [`Clip`]: an ordered run of frames plus an optional frame rate. A still
//! image is the one-frame case ([`Clip::still`]), which is what every
//! implementor in this workspace accepts today.
//!
//! **Only the image path is built.** A video captioner would differ in exactly
//! three declared places, and nowhere else:
//!
//! 1. it reports [`Capabilities::max_frames`] greater than 1, which is the
//!    single fact [`Captioner::validate`] uses to reject a clip a model cannot
//!    watch - an image model rejects a 48-frame clip by name instead of
//!    silently describing frame 0;
//! 2. it reads all of [`Clip::frames`] and [`Clip::fps`] as temporal
//!    conditioning, where an image captioner reads `frames[0]` and ignores
//!    `fps`;
//! 3. its labeler writes `data::videoset`'s `captions.json` (one caption per
//!    clip, in episode order - the format `wan::finetune::ClipSet::load_dir`
//!    already reads) instead of `data::imageset`'s `captions.yaml`.
//!
//! That third point is why [`label`] is separate from [`Captioner`]: the model
//! seam is shared, the output sink is per-medium, and both sinks already exist
//! in `crates/data`. Nothing else about the workflow - enumeration, resume,
//! the instruction, the trigger phrase - differs between the two.
//!
//! Swedish Embedded AB implements dataset labeling and vision-model integration
//! for its clients. If your team needs expertise in training-data curation for
//! image or video models then you can procure our services by sending an email
//! to info@swedishembedded.com.

pub mod label;

pub use label::{label_dir, LabelOpts, LabelReport};

/// One frame: interleaved-RGB HWC `f32` in `[0,1]` - the layout every vision
/// tower in this workspace consumes, and the one `capability::blob` carries.
pub struct Frame {
    pub hwc: Vec<f32>,
    pub w: u32,
    pub h: u32,
}

impl Frame {
    /// Build a frame, checking the buffer against the dimensions it claims.
    pub fn new(hwc: Vec<f32>, w: u32, h: u32) -> Result<Frame, String> {
        let want = (w as usize) * (h as usize) * 3;
        if hwc.len() != want {
            return Err(format!("frame: {} values for a {w}x{h} RGB image, expected {want}", hwc.len()));
        }
        Ok(Frame { hwc, w, h })
    }
}

/// The unit a [`Captioner`] describes: an ordered run of frames.
///
/// A still image is a one-frame clip. See this module's doc for what a video
/// implementor does with the rest.
pub struct Clip {
    /// The frames, in presentation order. Never empty.
    pub frames: Vec<Frame>,
    /// Frame rate, when the frames came from a timed source. `None` for a still
    /// image, and for a frame set with no meaningful rate.
    pub fps: Option<f32>,
}

impl Clip {
    /// A single still image.
    pub fn still(frame: Frame) -> Clip {
        Clip { frames: vec![frame], fps: None }
    }

    /// The first frame - what an image captioner reads.
    pub fn first(&self) -> &Frame {
        &self.frames[0]
    }

    /// Whether this is a single still image rather than a timed run of frames.
    pub fn is_still(&self) -> bool {
        self.frames.len() == 1
    }
}

/// What a captioner will accept, so a caller can check before paying for a load
/// and a model can refuse by name instead of degrading silently.
pub struct Capabilities {
    /// The canonical model id (`brain/qwen3vl`, …) this captioner speaks for.
    pub model: String,
    /// How many frames one call may carry. `1` means a still-image captioner;
    /// a video captioner reports its window.
    pub max_frames: u32,
    /// The largest `max_new` this model's context leaves room for.
    pub max_new_limit: u32,
}

/// One captioning request.
pub struct CaptionRequest<'a> {
    /// What to describe.
    pub clip: &'a Clip,
    /// The instruction. This is the whole of the prompt engineering: what the
    /// caption covers, how long it is, and any trigger phrase the adapter is
    /// meant to bind, are all decided here rather than by the model wrapper.
    pub instruction: &'a str,
    /// Token budget for the answer. A detailed caption needs a large one; the
    /// implementor clamps to [`Capabilities::max_new_limit`].
    pub max_new: u32,
}

/// A model that turns a [`Clip`] into text.
///
/// Deliberately narrow. *How* a vision-language model is built, placed on a
/// device, quantized or tokenized is model-specific, and pretending otherwise
/// would be an abstraction that fits exactly one implementation. What is
/// genuinely common - and all this trait claims - is that a loaded model can be
/// handed a clip and an instruction and will return text.
///
/// `Send` so a labeler can own one on a worker thread. Not `Sync`: brain's
/// models carry per-instance scratch, so concurrency comes from N instances,
/// not shared access.
pub trait Captioner: Send {
    /// This captioner's self-description, for validation and for the labeler's
    /// report.
    fn capabilities(&self) -> Capabilities;

    /// Describe `req.clip`. `on_token` receives decoded text deltas as they are
    /// produced, so a long caption over a slow model is not a black box; an
    /// implementor with no streaming seam may call it once with the whole
    /// answer.
    fn caption(&mut self, req: &CaptionRequest<'_>, on_token: &mut dyn FnMut(&str)) -> Result<String, String>;

    /// Check a request against [`Captioner::capabilities`] before running it.
    /// The default enforces the two facts the capabilities describe; an
    /// implementor may override to add its own. [`label_dir`] calls this, so a
    /// mismatch is an error naming the model and the limit rather than a
    /// caption of the wrong thing.
    fn validate(&self, req: &CaptionRequest<'_>) -> Result<(), String> {
        let caps = self.capabilities();
        if req.clip.frames.is_empty() {
            return Err(format!("{}: an empty clip has nothing to caption", caps.model));
        }
        let n = req.clip.frames.len() as u32;
        if n > caps.max_frames {
            return Err(format!(
                "{}: this is a {n}-frame clip but the model captions at most {} frame(s) per call{}",
                caps.model,
                caps.max_frames,
                if caps.max_frames == 1 { " - it is a still-image captioner" } else { "" }
            ));
        }
        if req.instruction.trim().is_empty() {
            return Err(format!("{}: the instruction is empty", caps.model));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract is testable with no model and no checkpoint - the point of
    /// having a contract at all.
    struct FakeCaptioner {
        max_frames: u32,
        /// What the fake echoes back, so a test can prove the request reached it.
        answer: String,
    }

    impl Captioner for FakeCaptioner {
        fn capabilities(&self) -> Capabilities {
            Capabilities { model: "test/fake".into(), max_frames: self.max_frames, max_new_limit: 128 }
        }
        fn caption(&mut self, req: &CaptionRequest<'_>, on_token: &mut dyn FnMut(&str)) -> Result<String, String> {
            self.validate(req)?;
            let text = format!("{} [{} frame(s), {}]", self.answer, req.clip.frames.len(), req.instruction);
            on_token(&text);
            Ok(text)
        }
    }

    fn frame(w: u32, h: u32) -> Frame {
        Frame::new(vec![0.5; (w * h * 3) as usize], w, h).unwrap()
    }

    #[test]
    fn frame_rejects_a_buffer_that_does_not_match_its_dimensions() {
        assert!(Frame::new(vec![0.0; 11], 2, 2).is_err());
        assert!(Frame::new(vec![0.0; 12], 2, 2).is_ok());
    }

    #[test]
    fn a_still_image_is_a_one_frame_clip() {
        let c = Clip::still(frame(4, 3));
        assert!(c.is_still());
        assert_eq!(c.first().w, 4);
        assert!(c.fps.is_none());
    }

    /// The video seam: a still-image captioner must REFUSE a multi-frame clip
    /// by name, not quietly describe the first frame. This is the one behaviour
    /// that makes `max_frames` load-bearing rather than decorative.
    #[test]
    fn a_still_image_captioner_refuses_a_multi_frame_clip() {
        let mut m = FakeCaptioner { max_frames: 1, answer: "a room".into() };
        let clip = Clip { frames: vec![frame(4, 3), frame(4, 3)], fps: Some(24.0) };
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 32 };
        let err = m.caption(&req, &mut |_| {}).unwrap_err();
        assert!(err.contains("test/fake"), "the error must name the model: {err}");
        assert!(err.contains("2-frame"), "the error must name the mismatch: {err}");
        assert!(err.contains("still-image captioner"), "{err}");

        // ... and the SAME clip through a captioner that declares a window works,
        // so the refusal is about the declared capability, not about frame count.
        let mut v = FakeCaptioner { max_frames: 16, answer: "a woman turns".into() };
        let out = v.caption(&req, &mut |_| {}).unwrap();
        assert_eq!(out, "a woman turns [2 frame(s), describe]");
    }

    #[test]
    fn an_empty_instruction_is_refused() {
        let m = FakeCaptioner { max_frames: 1, answer: String::new() };
        let clip = Clip::still(frame(2, 2));
        let req = CaptionRequest { clip: &clip, instruction: "   ", max_new: 8 };
        assert!(m.validate(&req).unwrap_err().contains("instruction is empty"));
    }

    #[test]
    fn streamed_deltas_reassemble_into_the_returned_text() {
        let mut m = FakeCaptioner { max_frames: 1, answer: "a rattan chair".into() };
        let clip = Clip::still(frame(2, 2));
        let req = CaptionRequest { clip: &clip, instruction: "describe", max_new: 8 };
        let mut seen = String::new();
        let out = m.caption(&req, &mut |d| seen.push_str(d)).unwrap();
        assert_eq!(seen, out);
    }
}
