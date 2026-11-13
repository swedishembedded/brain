// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Track one object through a clip: clicks on one frame in, the object's mask
//! on every frame out (SAM 2.1's memory bank, [`crate::video::Tracker`]).
//!
//! The result is an [`imaging::maskseq::MaskClip`] at the clip's own size,
//! one mask per frame, which a masked video editor takes as the region to
//! regenerate.
//!
//! Swedish Embedded AB implements video object tracking for its clients. If
//! your team needs expertise in segmentation and tracking models, you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use imaging::maskseq::{MaskClip, Polarity};
use imaging::{AlignCorners, Ctx, Filter, Shape};

use crate::config::Sam2Config;
use crate::import::Scope;
use crate::model::{Prompt, Sam2};
use crate::video::Tracker;

/// What to track and how to write it down.
#[derive(Debug, Clone)]
pub struct TrackRequest {
    /// `(x, y, label)` clicks in source pixels: label 1 on the object, 0 on
    /// something that is not. Several object clicks pick one object together
    /// (a person whose coat and legs SAM 2 would otherwise split).
    pub points: Vec<(f32, f32, f32)>,
    /// The frame the clicks sit on.
    pub prompt_frame: usize,
    pub object_id: u32,
    pub polarity: Polarity,
    /// Hard 0/1 masks, or SAM 2's sigmoid ramp.
    pub binary: bool,
}

impl TrackRequest {
    /// The request checked against a `frames`-long `w x h` clip: at least one
    /// click on the object, every click inside the frame, the prompt frame
    /// inside the clip.
    pub fn check(&self, frames: usize, w: u32, h: u32) -> Result<(), String> {
        if !self.points.iter().any(|p| p.2 == 1.0) {
            return Err("sam2 track: give at least one click on the object (points)".into());
        }
        if self.prompt_frame >= frames {
            return Err(format!("sam2 track: prompt_frame {} is past the {frames}-frame clip", self.prompt_frame));
        }
        if let Some(&(x, y, _)) = self.points.iter().find(|&&(x, y, _)| !(0.0..w as f32).contains(&x) || !(0.0..h as f32).contains(&y)) {
            return Err(format!("sam2 track: the click ({x}, {y}) is outside the {w}x{h} frame"));
        }
        Ok(())
    }
}

/// Read the WHOLE release checkpoint (image path and memory bank) and build
/// the tracking model on `gpu`. `Scope::Video` refuses an unmatched key in
/// either direction, so a memory bank left at random init is not reachable.
pub fn load_video(weights: &str, cfg: Sam2Config, gpu: Gpu) -> Result<Sam2, String> {
    let tensors = crate::caps::read_checkpoint(weights)?;
    let (tensors, _report) = crate::import::import_scoped(tensors, &cfg, Scope::Video)?;
    Ok(Sam2::new_video(gpu, cfg, &tensors))
}

/// Track `req`'s object through `frames` (interleaved `h * w * 3` RGB in
/// `[0,1]`), reporting `(done, total)` per frame; the run stops with
/// `"cancelled"` as soon as `progress` returns false. Frames before the prompt
/// frame take its mask (reverse tracking is not ported), so every frame has
/// one and frame indices never shift.
pub fn track_clip(m: &Sam2, frames: &[Vec<f32>], w: u32, h: u32, fps: f64, req: &TrackRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<MaskClip, String> {
    let n = frames.len();
    req.check(n, w, h)?;
    let s = m.cfg.image_size;
    // The reference resizes each frame to `image_size²` without keeping the
    // aspect ratio, so each axis scales on its own.
    let (sx, sy) = (s as f32 / w as f32, s as f32 / h as f32);
    // The 3-way ambiguity head only for a single click; several clicks
    // already disambiguate (`_use_multimask`).
    let prompt = Prompt {
        coords: req.points.iter().map(|&(x, y, _)| (x * sx, y * sy)).collect(),
        labels: req.points.iter().map(|p| p.2).collect(),
        mask_lowres: None,
        multimask_output: req.points.len() == 1,
    };
    let mut tracker = Tracker::new(m, n, req.object_id);
    let mut rows: Vec<Option<(Vec<f32>, f32, f32)>> = (0..n).map(|_| None).collect();
    let low = 4 * m.cfg.image_embedding_size();
    for f in req.prompt_frame..n {
        let ctx = Ctx::new(&m.gpu);
        let chw = imaging::pixels::hwc_to_chw(&frames[f], 3, h as usize, w as usize);
        let src = ctx.upload("sam2.track.src", &chw);
        let (dev, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), s, s, Filter::Bilinear, AlignCorners::HalfPixel);
        let enc = m.encode(&m.preprocess(&ctx.download(&dev, shape.numel())));
        let step = if f == req.prompt_frame { tracker.prompt(&enc, f, &prompt) } else { tracker.track(&enc, f) };
        // The LOW-resolution logits go straight to the video grid, as the
        // reference's `_get_orig_video_res_output` does: one bilinear step.
        let (up, ushape) = ctx.resize(&step.low_res_mask, Shape::new(1, 1, low, low), h, w, Filter::Bilinear, AlignCorners::HalfPixel);
        rows[f] = Some((ctx.download(&up, ushape.numel()), step.object_score, step.iou));
        if !progress(f + 1 - req.prompt_frame, n - req.prompt_frame) {
            return Err("cancelled".into());
        }
    }
    let seed = rows[req.prompt_frame].clone().ok_or("sam2 track: the prompt frame produced no mask")?;
    for slot in &mut rows[..req.prompt_frame] {
        *slot = Some(seed.clone());
    }
    let mut clip = MaskClip::new(w, h, fps, req.polarity, req.binary);
    clip.object_id = req.object_id;
    clip.prompt_frame = req.prompt_frame;
    clip.prompt_points = req.points.clone();
    for (f, row) in rows.into_iter().enumerate() {
        let (logits, score, iou) = row.ok_or_else(|| format!("sam2 track: frame {f} produced no mask"))?;
        clip.push_logits(&logits, score, iou)?;
    }
    Ok(clip)
}

/// Pick the frames a clip at `from_fps` shows at `to_fps`: the nearest source
/// frame to each output instant, so a resampled clip keeps its duration.
pub fn resample_indices(frames: usize, from_fps: f64, to_fps: f64) -> Vec<usize> {
    if frames == 0 || !(from_fps > 0.0 && to_fps > 0.0) {
        return (0..frames).collect();
    }
    let out = ((frames as f64) * to_fps / from_fps).round().max(1.0) as usize;
    (0..out).map(|i| (((i as f64) * from_fps / to_fps).round() as usize).min(frames - 1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(points: Vec<(f32, f32, f32)>, prompt_frame: usize) -> TrackRequest {
        TrackRequest { points, prompt_frame, object_id: 0, polarity: Polarity::ObjectWhite, binary: true }
    }

    /// A request that cannot pick an object is refused before any weight
    /// loads: no click on the object, a click off the frame, a prompt frame
    /// past the clip.
    #[test]
    fn an_unusable_request_is_refused_up_front() {
        assert!(req(vec![(1.0, 1.0, 0.0)], 0).check(4, 8, 8).unwrap_err().contains("at least one click"));
        assert!(req(vec![(9.0, 1.0, 1.0)], 0).check(4, 8, 8).unwrap_err().contains("outside"));
        assert!(req(vec![(1.0, 1.0, 1.0)], 4).check(4, 8, 8).unwrap_err().contains("past"));
        assert!(req(vec![(1.0, 1.0, 1.0), (2.0, 2.0, 0.0)], 3).check(4, 8, 8).is_ok());
    }

    /// Resampling keeps the clip's duration and picks the nearest frame.
    #[test]
    fn resampling_keeps_the_duration() {
        assert_eq!(resample_indices(6, 30.0, 15.0), vec![0, 2, 4]);
        assert_eq!(resample_indices(3, 10.0, 20.0), vec![0, 1, 1, 2, 2, 2]);
        assert_eq!(resample_indices(5, 24.0, 24.0), vec![0, 1, 2, 3, 4]);
    }
}
