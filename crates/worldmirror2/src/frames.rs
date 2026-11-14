// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From a capture to the tensor the trunk takes: which frames of a long
//! capture the model sees, which pixels of them it may use, and the grid they
//! are conformed to.
//!
//! Everything here is device-free and returns a `Result` rather than
//! aborting, so a resident process can refuse one bad request and go on
//! serving the next.
//!
//! Swedish Embedded AB implements capture-to-reconstruction pipelines that
//! cope with real footage - long videos, turntables, mixed aspect ratios - for
//! its clients. If your team needs expertise in multi-view capture handling
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.

use imaging::{host::resize_bicubic_pil, ChannelPolicy, Rgb8};

use crate::config::MirrorConfig;
use crate::preprocess;

/// How a caller narrows a long capture down to the frames the model sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSel {
    /// Keep every `stride`-th frame (1 keeps all; 0 is treated as 1).
    pub stride: usize,
    /// Keep at most this many frames, spread across the capture. 0 = no cap.
    pub max: usize,
    /// Frames per second to resample a clip to before selecting. 0 = the
    /// clip's own rate.
    pub fps: f64,
}

impl Default for FrameSel {
    fn default() -> FrameSel {
        FrameSel { stride: 1, max: 0, fps: 0.0 }
    }
}

/// The cap a clip with a frame rate of its own gets when the caller sets none:
/// the trunk's global attention is quadratic in frame count, and a minute of
/// video is thousands of frames. A set of stills is the set the caller chose
/// and is left alone.
pub const DEFAULT_CLIP_FRAME_CAP: usize = 48;

/// Take every `stride`-th of `n` frames, then thin to at most `max` of those.
///
/// The cap SPANS the sequence rather than truncating it: a 360-degree orbit
/// capped to a prefix would become a 60-degree one, which is a far worse
/// reconstruction than the same budget spread over the whole path. Never
/// returns an empty set for a non-empty input.
pub fn select_indices(n: usize, stride: usize, max: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let strided: Vec<usize> = (0..n).step_by(stride.max(1)).collect();
    if max == 0 || strided.len() <= max {
        return strided;
    }
    (0..max).map(|i| strided[i * (strided.len() - 1) / (max - 1).max(1)]).collect()
}

/// The frames of an `n`-frame clip recorded at `source_fps`, resampled to `fps`:
/// frame k of the result is the one showing at time k/`fps`. A request at or
/// above the source rate keeps every frame; resampling never invents one.
pub fn resample_indices(n: usize, source_fps: f64, fps: f64) -> Vec<usize> {
    if fps.is_nan() || fps <= 0.0 || fps >= source_fps {
        return (0..n).collect();
    }
    let step = source_fps / fps;
    (0..).map(|k| (k as f64 * step).floor() as usize).take_while(|&i| i < n).collect()
}

impl FrameSel {
    /// The indices of the `n` decoded frames to use. `source_fps` is the
    /// clip's own rate, when it states one; resampling needs it.
    pub fn indices(&self, n: usize, source_fps: Option<f64>) -> Result<Vec<usize>, String> {
        if n == 0 {
            return Err("the capture holds no frames".to_string());
        }
        let base = if self.fps > 0.0 {
            let src = source_fps
                .filter(|f| f.is_finite() && *f > 0.0)
                .ok_or("fps resampling needs a clip that states its own frame rate (a video file, not a set of stills)")?;
            resample_indices(n, src, self.fps)
        } else {
            (0..n).collect()
        };
        let cap = match (self.max, source_fps) {
            (0, Some(_)) => DEFAULT_CLIP_FRAME_CAP,
            (m, _) => m,
        };
        Ok(select_indices(base.len(), self.stride, cap).into_iter().map(|i| base[i]).collect())
    }
}

/// Black out every pixel the mask leaves unlit, in place.
///
/// The mask is resized to the frame with nearest-neighbour sampling: it is a
/// keep/drop decision per pixel, and interpolating it would invent
/// half-transparent border pixels the model would then try to explain.
///
/// This exists for TURNTABLE captures. The model assumes one rigid scene and a
/// moving camera; a turntable gives it a rotating object in front of a
/// stationary background, which is not that, and the static part is evidence
/// for a camera that never moved. Mask the background away and what is left -
/// an object turning in front of nothing - is exactly equivalent to a camera
/// orbiting a still object, which is what the model was trained on. One mask
/// covers the whole capture, because the camera does not move.
pub fn apply_mask(img: &mut Rgb8, mask: &Rgb8) {
    let (mw, mh) = (mask.w as usize, mask.h as usize);
    let (w, h) = (img.w as usize, img.h as usize);
    for y in 0..h {
        let my = y * mh / h;
        for x in 0..w {
            let mx = x * mw / w;
            if mask.px[(my * mw + mx) * 3] < 128 {
                let i = (y * w + x) * 3;
                img.px[i..i + 3].fill(0);
            }
        }
    }
}

/// An interleaved HWC `[0,1]` frame (the blob wire format) as 8-bit RGB. A
/// one-channel plane, a mask say, is rendered as grey.
pub fn rgb8_from_unit_hwc(hwc: &[f32], w: u32, h: u32, channels: usize) -> Result<Rgb8, String> {
    imaging::pixels::hwc_to_rgb8(hwc, w, h, channels, ChannelPolicy::ReplicateFirst)
}

/// The frames as the trunk takes them: raw `[0,1]` CHW, concatenated, on one
/// patch grid.
#[derive(Debug)]
pub struct Conformed {
    pub chw: Vec<f32>,
    pub frames: usize,
    /// Grid height and width in patches.
    pub hp: usize,
    pub wp: usize,
}

impl Conformed {
    /// Pixel size of one conformed frame, `(width, height)`.
    pub fn size(&self, cfg: &MirrorConfig) -> (u32, u32) {
        ((self.wp * cfg.patch) as u32, (self.hp * cfg.patch) as u32)
    }
}

/// Resize every frame to the grid the model runs at and lay them out as CHW.
///
/// `target` is the longest-edge cap (the reference's inference default, not
/// the checkpoint's native grid: the position embedding is interpolated to
/// whatever grid it is handed, which is what the normalized RoPE was added
/// for). Resampling is Pillow's bicubic, because this checkpoint's parity is
/// gated against it. Every frame must land on one grid.
pub fn conform(images: &[Rgb8], cfg: &MirrorConfig, target: usize) -> Result<Conformed, String> {
    let mut chw = Vec::new();
    let mut grid: Option<(usize, usize)> = None;
    for (i, img) in images.iter().enumerate() {
        let (iw, ih) = (img.w as usize, img.h as usize);
        let target = preprocess::adaptive_target(iw, ih, target, cfg.patch);
        let (nw, nh) = preprocess::resize_dims(iw, ih, target, cfg.patch);
        let resized = resize_bicubic_pil(img, nw, nh);
        let (cw, ch) = (nw.min(target), nh.min(target));
        let (x0, y0) = ((nw - cw) / 2, (nh - ch) / 2);
        for c in 0..3 {
            for y in 0..ch {
                for x in 0..cw {
                    chw.push(resized.px[((y0 + y) * nw + x0 + x) * 3 + c] as f32 / 255.0);
                }
            }
        }
        let g = (cw / cfg.patch, ch / cfg.patch);
        if grid.is_some_and(|prev| prev != g) {
            return Err(format!("frame {i} ({iw}x{ih}) lands on a different patch grid than the frames before it; every frame must share one size"));
        }
        grid = Some(g);
    }
    let (wp, hp) = grid.ok_or("the capture holds no frames")?;
    if wp == 0 || hp == 0 {
        return Err("frames are smaller than one patch".to_string());
    }
    Ok(Conformed { chw, frames: images.len(), hp, wp })
}

/// CHW `[0,1]` frames back to 8-bit RGB: the frames as the model saw them,
/// at the resolution it saw them.
pub fn to_rgb8(chw: &[f32], frames: usize, w: u32, h: u32) -> Result<Vec<Rgb8>, String> {
    let hw = (w * h) as usize;
    (0..frames)
        .map(|f| {
            let mut px = Vec::with_capacity(hw * 3);
            for i in 0..hw {
                for c in 0..3 {
                    px.push((chw[(f * 3 + c) * hw + i].clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            Rgb8::new(w, h, px)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame selection has to happen BEFORE the model sees the sequence: a
    /// 60-second orbit at 30fps is 1800 frames, and the trunk's global
    /// attention is quadratic in frame count.
    #[test]
    fn a_frame_sequence_can_be_subsampled_and_capped() {
        assert_eq!(select_indices(20, 1, 0).len(), 20);
        assert_eq!(select_indices(20, 4, 0), vec![0, 4, 8, 12, 16]);
        // a cap spreads its picks across the WHOLE sequence rather than taking
        // a prefix, or a 360-degree orbit would become a 60-degree one.
        let capped = select_indices(20, 1, 5);
        assert_eq!(capped.len(), 5);
        assert_eq!(capped[0], 0);
        assert!(*capped.last().unwrap() >= 15, "cap took a prefix instead of spanning: {capped:?}");
        // stride and cap compose, and neither can produce an empty set
        assert_eq!(select_indices(20, 100, 0), vec![0]);
        assert_eq!(select_indices(20, 1, 99).len(), 20);
        assert_eq!(select_indices(20, 0, 0).len(), 20, "stride 0 means every frame, not a division by zero");
        assert_eq!(select_indices(20, 1, 1), vec![0]);
        assert!(select_indices(0, 1, 5).is_empty());
    }

    #[test]
    fn a_clip_resamples_to_the_requested_rate_and_never_invents_frames() {
        // perf-number: fixture frame rates
        // 30 frames at 30 fps is one second: 10 fps keeps frames 0, 3, 6, ...
        assert_eq!(resample_indices(30, 30.0, 10.0), (0..10).map(|k| k * 3).collect::<Vec<_>>());
        // at or above the source rate, every frame
        assert_eq!(resample_indices(5, 30.0, 30.0), vec![0, 1, 2, 3, 4]);
        assert_eq!(resample_indices(5, 30.0, 120.0), vec![0, 1, 2, 3, 4]);
        assert_eq!(resample_indices(5, 30.0, 0.0), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn selection_composes_resampling_stride_and_cap() {
        let sel = FrameSel { stride: 2, max: 0, fps: 10.0 };
        // perf-number: fixture frame rates
        // 30 fps -> 10 fps gives 0,3,6,...,27; stride 2 keeps 0,6,12,18,24
        assert_eq!(sel.indices(30, Some(30.0)).unwrap(), vec![0, 6, 12, 18, 24]);
    }

    /// A video is thinned by default because the trunk is quadratic in frames;
    /// stills are the set the caller chose and are never thinned unasked.
    #[test]
    fn a_clip_gets_a_default_cap_and_stills_do_not() {
        let sel = FrameSel::default();
        assert_eq!(sel.indices(200, Some(30.0)).unwrap().len(), DEFAULT_CLIP_FRAME_CAP);
        assert_eq!(sel.indices(200, None).unwrap().len(), 200);
        let explicit = FrameSel { max: 10, ..FrameSel::default() };
        assert_eq!(explicit.indices(200, Some(30.0)).unwrap().len(), 10);
    }

    #[test]
    fn resampling_without_a_frame_rate_and_an_empty_capture_are_named_errors() {
        let sel = FrameSel { fps: 5.0, ..FrameSel::default() };
        assert!(sel.indices(10, None).unwrap_err().contains("frame rate"));
        assert!(FrameSel::default().indices(0, None).unwrap_err().contains("no frames"));
    }

    /// A mask is a keep/drop decision per pixel, so it is sampled, never
    /// interpolated - a resized mask that blends would create half-lit border
    /// pixels, and the model would dutifully reconstruct the blend.
    #[test]
    fn a_mask_blacks_out_what_it_does_not_cover_at_any_size() {
        // 2x2 mask: keep the left column, drop the right
        let mask = Rgb8::new(2, 2, vec![255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0]).unwrap();
        // a 4x4 frame of solid grey
        let mut img = Rgb8::new(4, 4, vec![200u8; 4 * 4 * 3]).unwrap();
        apply_mask(&mut img, &mask);
        for y in 0..4 {
            for x in 0..4 {
                let v = img.px[(y * 4 + x) * 3];
                let want = if x < 2 { 200 } else { 0 };
                assert_eq!(v, want, "pixel ({x},{y}) is {v}, expected {want} after a 2x2 mask on a 4x4 frame");
            }
        }
    }

    #[test]
    fn a_single_channel_mask_plane_is_read_as_grey() {
        // 2x1 plane: lit, unlit
        let m = rgb8_from_unit_hwc(&[1.0, 0.0], 2, 1, 1).unwrap();
        assert_eq!(m.px, vec![255, 255, 255, 0, 0, 0]);
    }

    /// Frames land on the 14px patch grid at any aspect ratio, capped at the
    /// target on the longest edge, and every frame of a capture shares it.
    #[test]
    fn frames_conform_to_one_patch_grid() {
        let cfg = MirrorConfig::default();
        let frame = |w: u32, h: u32| Rgb8::new(w, h, vec![128u8; (w * h * 3) as usize]).unwrap();
        let c = conform(&[frame(200, 100), frame(200, 100)], &cfg, 112).unwrap();
        // longest edge 112 = 8 patches; the short side 56 = 4 patches
        assert_eq!((c.frames, c.wp, c.hp), (2, 8, 4));
        assert_eq!(c.chw.len(), 2 * 3 * 112 * 56);
        assert_eq!(c.size(&cfg), (112, 56));
        assert!(c.chw.iter().all(|v| (*v - 128.0 / 255.0).abs() < 1e-3));
        // a frame the cap does not reach is only floored to the grid
        let small = conform(&[frame(30, 30)], &cfg, 952).unwrap();
        assert_eq!((small.wp, small.hp), (2, 2));
    }

    #[test]
    fn mixed_sizes_and_an_empty_capture_are_errors_not_panics() {
        let cfg = MirrorConfig::default();
        let frame = |w: u32, h: u32| Rgb8::new(w, h, vec![0u8; (w * h * 3) as usize]).unwrap();
        let e = conform(&[frame(112, 56), frame(56, 112)], &cfg, 112).unwrap_err();
        assert!(e.contains("one size"), "{e}");
        assert!(conform(&[], &cfg, 112).unwrap_err().contains("no frames"));
    }

    #[test]
    fn conformed_frames_round_trip_to_8_bit_rgb() {
        let cfg = MirrorConfig::default();
        let px: Vec<u8> = (0..28 * 28 * 3).map(|i| (i % 251) as u8).collect();
        let c = conform(&[Rgb8::new(28, 28, px.clone()).unwrap()], &cfg, 28).unwrap();
        let back = to_rgb8(&c.chw, 1, 28, 28).unwrap();
        assert_eq!(back[0].px, px, "a same-size frame must come back byte for byte");
    }
}
