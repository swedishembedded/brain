// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Aspect-preserving fit onto a square input, with centre padding.
//!
//! **This is the workspace's only letterbox.** It was `yolo::boxmath`'s until
//! this crate existed; `boxmath` now re-exports these items, so
//! `yolo::boxmath::{Letterbox, letterbox_rgb}` still name them and no detection
//! call site moved. The code was not *copied* here — it was moved, because the
//! pad fill, the `pad_y as usize` truncation and the half-pixel nearest rule
//! below are baked into the trained yolo weights and into every reported
//! `map50`, and a second definition free to drift from this one is precisely the
//! `rmsnorm`-was-seven-times failure `AGENTS.md` forbids.
//!
//! Still outstanding for the migrator: `cli::depth_cli::letterbox_chw` is a
//! *third* variant (CHW in, `pad = 0.5`) and has not moved, because doing so
//! changes the ZipDepth INT8 calibration inputs and must be gated on the
//! quantized-accuracy check rather than on a build passing.
//!
//! This is the one **host** resampler in the crate, and it is host for a reason
//! that is worth reading before "fixing" it.
//!
//! ## Why this cannot be dispatched today
//!
//! Its nearest-neighbour rule is *half-pixel*:
//! `src = round((dst + 0.5) / scale - 0.5)`, with `f32::round`
//! (half-away-from-zero). `crates/kernels/wgsl/resize_nearest.wgsl` implements
//! torch's `nearest`, which is ONNX `asymmetric` + `nearest_mode = floor`:
//! `src = floor(dst * in / out)`. Those two select **different source pixels for
//! most ratios**. Retargeting this function at that kernel would change every
//! YOLO detection, so it would have to be gated on `map50`, not waved through as
//! a refactor.
//!
//! The clean fix is a *mode* on `resize_nearest` (one extra `Params` word,
//! selecting half-pixel vs asymmetric) rather than a second nearest kernel.
//! The pad half is also blocked: `pad2d.wgsl` is
//! zero-fill only and cannot express the grey `114/255` fill, which likewise
//! wants a `pad_value` word rather than a new kernel.
//!
//! ## The half-pixel disagreement between pixels and boxes — LOAD-BEARING
//!
//! [`letterbox_rgb`] places content at `yi + lb.pad_y as usize`, an f32 -> usize
//! **truncation**, while [`Letterbox::apply_box`] / [`Letterbox::invert_box`]
//! use the *float* `pad_x` / `pad_y`. When `size - new_w` is odd (640 vs 479 =>
//! pad 80.5) pixels land at +80 and boxes map at +80.5: a systematic half-pixel
//! bias between an image and its own coordinate frame.
//!
//! That is preserved here **deliberately and bit-for-bit**. It is a real defect
//! (survey §6.3) but it is baked into the trained YOLO weights and into every
//! reported `map50`; correcting it silently shifts every detection. Fix it as
//! its own gated change, with the metric measured before and after — not as a
//! side effect of moving the function.

use crate::pixels::Rect;

/// An `xyxy` box in pixels.
///
/// The same `[f32; 4]` that `yolo::boxmath::Xyxy` names. Box *math* (IoU, CIoU,
/// NMS) stays in `yolo::boxmath` — `crates/eval` already re-exports yolo's IoU
/// rather than owning a second one — and only the letterbox geometry is
/// image-substrate work. A `type` alias carries no behaviour, so the two names
/// for one primitive cannot drift; a second `struct` would have.
pub type Xyxy = [f32; 4];

/// The scale + padding that maps an original `(w0, h0)` image onto a square
/// `size x size` model input while preserving aspect ratio.
///
/// Forward map (original px -> input px): `xi = x0 * scale + pad_x`,
/// `yi = y0 * scale + pad_y`, with `scale = min(size/w0, size/h0)` — the longer
/// side fills the square and the shorter side is centre-padded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    /// Scaled content size before padding: `round(w0*scale) x round(h0*scale)`.
    pub new_w: u32,
    pub new_h: u32,
    /// Target square side.
    pub size: u32,
}

impl Letterbox {
    /// Compute the transform for an original `w0 x h0` image fitted to a
    /// `size x size` square.
    pub fn compute(w0: u32, h0: u32, size: u32) -> Letterbox {
        let s = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
        let new_w = (w0 as f32 * s).round() as u32;
        let new_h = (h0 as f32 * s).round() as u32;
        let pad_x = (size as f32 - new_w as f32) * 0.5;
        let pad_y = (size as f32 - new_h as f32) * 0.5;
        Letterbox { scale: s, pad_x, pad_y, new_w, new_h, size }
    }

    /// Map an `xyxy` box from ORIGINAL-image coords into letterboxed INPUT
    /// coords.
    pub fn apply_box(&self, b: Xyxy) -> Xyxy {
        [
            b[0] * self.scale + self.pad_x,
            b[1] * self.scale + self.pad_y,
            b[2] * self.scale + self.pad_x,
            b[3] * self.scale + self.pad_y,
        ]
    }

    /// Inverse: map an `xyxy` box from letterboxed INPUT coords back to
    /// ORIGINAL coords, clamped to the original `[0,w0] x [0,h0]` frame.
    pub fn invert_box(&self, b: Xyxy, w0: u32, h0: u32) -> Xyxy {
        let inv = 1.0 / self.scale;
        [
            ((b[0] - self.pad_x) * inv).clamp(0.0, w0 as f32),
            ((b[1] - self.pad_y) * inv).clamp(0.0, h0 as f32),
            ((b[2] - self.pad_x) * inv).clamp(0.0, w0 as f32),
            ((b[3] - self.pad_y) * inv).clamp(0.0, h0 as f32),
        ]
    }

    /// Where the content actually lands in the padded square, in **pixels** —
    /// i.e. using the truncated offsets [`letterbox_rgb`] writes to, not the
    /// float ones [`Letterbox::apply_box`] uses.
    ///
    /// Exposed so a caller can see the half-pixel disagreement described in this
    /// module's header instead of rediscovering it: on an odd pad this rect's
    /// origin is half a pixel from `(pad_x, pad_y)`.
    pub fn content_rect(&self) -> Rect {
        Rect::new(self.pad_x as u32, self.pad_y as u32, self.new_w, self.new_h)
    }
}

/// Resize an interleaved-RGB HWC `src` (`[h0*w0*3]`) into a letterboxed **CHW**
/// tensor `[3*size*size]`, filling the border with `pad`.
///
/// `pad` is a parameter, never a constant: yolo's callers pass `114.0/255.0`
/// (ultralytics' grey) and `cli::depth_cli`'s INT8 calibration passes `0.5`.
/// A single hard-coded fill would silently change one of them.
///
/// Resampling is half-pixel nearest neighbour — see the module header for why it
/// is not a kernel dispatch, and why the `as usize` truncation on the paste
/// offsets must not be "corrected" here.
pub fn letterbox_rgb(src: &[f32], w0: u32, h0: u32, size: u32, pad: f32) -> (Vec<f32>, Letterbox) {
    assert_eq!(src.len(), (w0 * h0 * 3) as usize, "src must be HWC RGB [h0*w0*3]");
    let lb = Letterbox::compute(w0, h0, size);
    let sz = size as usize;
    let mut chw = vec![pad; 3 * sz * sz];
    let inv = 1.0 / lb.scale;
    for yi in 0..lb.new_h as usize {
        let sy = ((yi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, h0 as f32 - 1.0) as usize;
        // TRUNCATION, not rounding: see the module header. `apply_box` uses the
        // float pad, so on an odd pad the image and its coordinate frame differ
        // by half a pixel. Unchanged by the move out of `yolo::boxmath`.
        let dy = yi + lb.pad_y as usize;
        for xi in 0..lb.new_w as usize {
            let sx = ((xi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, w0 as f32 - 1.0) as usize;
            let dx = xi + lb.pad_x as usize;
            let s_base = (sy * w0 as usize + sx) * 3;
            for c in 0..3 {
                chw[c * sz * sz + dy * sz + dx] = src[s_base + c];
            }
        }
    }
    (chw, lb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_input_needs_no_padding() {
        let lb = Letterbox::compute(100, 100, 50);
        assert_eq!((lb.new_w, lb.new_h), (50, 50));
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 0.0));
        assert!((lb.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn wide_input_pads_vertically_and_boxes_round_trip() {
        let lb = Letterbox::compute(200, 100, 100);
        assert_eq!((lb.new_w, lb.new_h), (100, 50));
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 25.0));
        let b = [10.0f32, 20.0, 60.0, 80.0];
        let back = lb.invert_box(lb.apply_box(b), 200, 100);
        for i in 0..4 {
            assert!((back[i] - b[i]).abs() < 1e-3, "box coord {i}: {} vs {}", back[i], b[i]);
        }
    }

    /// The documented, deliberately-preserved defect. If this test starts
    /// failing, someone "fixed" the truncation — which moves every detection and
    /// must be gated on `map50`, not on a unit test passing.
    #[test]
    fn odd_pad_puts_pixels_half_a_pixel_from_where_boxes_go() {
        let lb = Letterbox::compute(640, 479, 640);
        assert_eq!(lb.new_h, 479);
        assert!((lb.pad_y - 80.5).abs() < 1e-6, "pad_y is {}", lb.pad_y);
        // Boxes map with the float pad ...
        assert!((lb.apply_box([0.0, 0.0, 1.0, 1.0])[1] - 80.5).abs() < 1e-6);
        // ... pixels land at the truncated one.
        assert_eq!(lb.content_rect().y, 80);
    }

    #[test]
    fn pad_fill_is_the_callers_choice() {
        let src = vec![1.0f32; 4 * 2 * 3]; // 4x2 all-ones RGB
        for fill in [114.0 / 255.0, 0.5, 0.0] {
            let (chw, lb) = letterbox_rgb(&src, 4, 2, 4, fill);
            assert_eq!(chw.len(), 3 * 16);
            // Row 0 is above the content (content is 4x2 centred at y=1).
            assert_eq!(lb.content_rect(), crate::pixels::Rect::new(0, 1, 4, 2));
            assert_eq!(chw[0], fill, "top-left must be the requested fill");
            // Inside the content everything is the source value (c=0, y=1, x=0).
            assert_eq!(chw[4], 1.0);
        }
    }

    #[test]
    fn content_is_copied_channel_planar_in_rgb_order() {
        // 2x1 image: pixel 0 = (1,2,3), pixel 1 = (4,5,6). Target 2x2.
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (chw, lb) = letterbox_rgb(&src, 2, 1, 2, 0.0);
        assert_eq!((lb.new_w, lb.new_h), (2, 1));
        let plane = 4usize;
        let row = lb.content_rect().y as usize; // 0 (pad_y = 0.5 truncates)
        let at = |c: usize, x: usize| chw[c * plane + row * 2 + x];
        assert_eq!((at(0, 0), at(1, 0), at(2, 0)), (1.0, 2.0, 3.0));
        assert_eq!((at(0, 1), at(1, 1), at(2, 1)), (4.0, 5.0, 6.0));
    }

    /// Nearest-neighbour source selection must stay half-pixel. The literal
    /// expectations here are what a dispatch of `resize_nearest.wgsl` would
    /// have to reproduce before this function may move to the device.
    #[test]
    fn nearest_rule_is_half_pixel_not_asymmetric_floor() {
        // 3 -> 2 downscale, scale = 2/3, inv = 1.5.
        // half-pixel: round((0+0.5)*1.5 - 0.5) = round(0.25) = 0
        //             round((1+0.5)*1.5 - 0.5) = round(1.75) = 2
        // asymmetric floor (the kernel): floor(0*3/2)=0, floor(1*3/2)=1
        let src: Vec<f32> = (0..9).map(|i| i as f32).collect(); // 3x1 RGB
        let (chw, _) = letterbox_rgb(&src, 3, 1, 2, 0.0);
        let plane = 4usize;
        assert_eq!(chw[0], 0.0, "dst x=0 -> src x=0");
        assert_eq!(chw[1], 6.0, "dst x=1 -> src x=2 (half-pixel), NOT src x=1 (floor)");
        assert_eq!(chw[plane + 1], 7.0);
    }
}
