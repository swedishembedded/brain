// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Aspect-preserving fit onto a square input, with centre padding.
//!
//! This is the workspace's only letterbox; `yolov8::boxmath` re-exports it, so
//! `yolov8::boxmath::{Letterbox, letterbox_rgb}` name these items. A second
//! definition free to drift from this one is the `rmsnorm`-was-seven-times
//! failure `AGENTS.md` forbids.
//!
//! ## One integral pad, shared by pixels and boxes
//!
//! The content is `round(w0*scale) x round(h0*scale)` with
//! `scale = min(size/w0, size/h0)`. The pad is whole pixels: the left and top
//! margins are `floor(slack / 2)` and the extra pixel of an odd slack goes to
//! the right and bottom, exactly [`crate::device::Border::centre`]. For a
//! 640x479 image on a 640 square the slack is 161 rows and the top margin is 80.
//! [`Letterbox::apply_box`], [`Letterbox::invert_box`], [`Letterbox::content_rect`]
//! and the pixels all use that one pad, so an image and its coordinate frame
//! agree at every size. It is also the upstream detectors' convention
//! (`top = round(dh - 0.1)`, which is 80 for `dh = 80.5`).
//!
//! ## The resampling rule
//!
//! Nearest neighbour, half-pixel: each output pixel takes the source pixel whose
//! centre is nearest its own,
//! `src = round_half_up((o + 0.5) * in / out - 0.5)` clamped to `[0, in - 1]`,
//! where `in` is the source extent and `out` the resized content extent
//! (`new_w` / `new_h`), not `1/scale`. In exact integer arithmetic this is
//! `min(((2*o + 1) * in) / (2 * out), in - 1)`, which [`nearest_tap`] computes
//! and `resize_nearest.wgsl` mode 1 computes identically.
//!
//! ## Device path and host path
//!
//! [`crate::device::Ctx::letterbox`] is the device path: one half-pixel nearest
//! resize and one constant-fill pad, both kernel dispatches, so the work runs
//! where `--device` says. [`letterbox_rgb`] is the same rule on the host, for
//! callers whose input and consumer both live in host memory, where a device
//! round trip would only add copies: the NPU paths (`crates/npu`,
//! `crates/cli/src/npu_cli.rs`), the board path in `yolov5`'s `npu.rs`, and
//! detectors fed host images. A test (`device_ops.rs`) proves the two bitwise
//! equal, so neither can drift from the other.
//!
//! The ZipDepth INT8 calibration does not letterbox: it preprocesses with
//! `zipdepth::predict::preprocess_chw`, the function inference uses.

use crate::pixels::Rect;

/// An `xyxy` box in pixels: [`crate::nms::Xyxy`], the one box type of the
/// detection tail (IoU, NMS, un-letterboxing), re-exported here for the
/// letterbox's own signatures.
pub use crate::nms::Xyxy;

/// The scale + padding that maps an original `(w0, h0)` image onto a square
/// `size x size` model input while preserving aspect ratio.
///
/// Forward map (original px -> input px): `xi = x0 * scale + pad_x`,
/// `yi = y0 * scale + pad_y`, with `scale = min(size/w0, size/h0)` — the longer
/// side fills the square and the shorter side is centre-padded. `pad_x` and
/// `pad_y` are whole pixels (stored as `f32` so the box maps stay one
/// multiply-add).
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
        let pad_x = ((size - new_w) / 2) as f32;
        let pad_y = ((size - new_h) / 2) as f32;
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

    /// Where the content lands in the padded square, in pixels. Its origin is
    /// exactly `(pad_x, pad_y)`, the offset [`Letterbox::apply_box`] adds.
    pub fn content_rect(&self) -> Rect {
        Rect::new(self.pad_x as u32, self.pad_y as u32, self.new_w, self.new_h)
    }
}

/// The source index half-pixel nearest neighbour selects for output index `o`
/// when `in_len` source samples are resampled to `out_len`:
/// `min(((2*o + 1) * in_len) / (2 * out_len), in_len - 1)`, the integer form of
/// `round_half_up((o + 0.5) * in/out - 0.5)` clamped to the source. The same
/// arithmetic as `resize_nearest.wgsl` mode 1, in `u64` so no extent overflows.
pub fn nearest_tap(o: u32, in_len: u32, out_len: u32) -> u32 {
    assert!(in_len > 0 && out_len > 0, "nearest_tap: extents must be non-empty");
    let tap = (2 * o as u64 + 1) * in_len as u64 / (2 * out_len as u64);
    tap.min(in_len as u64 - 1) as u32
}

/// Resize an interleaved-RGB HWC `src` (`[h0*w0*3]`) into a letterboxed **CHW**
/// tensor `[3*size*size]`, filling the border with `pad`.
///
/// `pad` is a parameter, never a constant: yolo's callers pass `114.0/255.0`
/// (ultralytics' grey), and a detector trained on another fill must be able to
/// pass its own without changing theirs.
///
/// The host twin of [`crate::device::Ctx::letterbox`]: the same [`nearest_tap`]
/// and the same [`Letterbox`] pad, bitwise equal to it. See the module header.
pub fn letterbox_rgb(src: &[f32], w0: u32, h0: u32, size: u32, pad: f32) -> (Vec<f32>, Letterbox) {
    assert_eq!(src.len(), (w0 * h0 * 3) as usize, "src must be HWC RGB [h0*w0*3]");
    let lb = Letterbox::compute(w0, h0, size);
    let sz = size as usize;
    let mut chw = vec![pad; 3 * sz * sz];
    let (pad_x, pad_y) = (lb.pad_x as usize, lb.pad_y as usize);
    for yi in 0..lb.new_h {
        let sy = nearest_tap(yi, h0, lb.new_h) as usize;
        let dy = yi as usize + pad_y;
        for xi in 0..lb.new_w {
            let sx = nearest_tap(xi, w0, lb.new_w) as usize;
            let dx = xi as usize + pad_x;
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

    /// Pixels and boxes share one pad: the pad is whole pixels (the left and
    /// top margins take the floor of half the slack, the right and bottom
    /// margins the rest), so the rectangle the content lands in starts exactly
    /// where `apply_box` maps the original origin, at every size and aspect.
    #[test]
    fn pixels_and_boxes_share_one_integral_pad() {
        let lb = Letterbox::compute(640, 479, 640);
        assert_eq!(lb.new_h, 479);
        assert_eq!(lb.pad_y, 80.0, "161 slack rows: 81 above would be the ceil, 80 the floor");
        assert_eq!(lb.apply_box([0.0, 0.0, 1.0, 1.0])[1], 80.0);
        assert_eq!(lb.content_rect().y, 80);

        for (w0, h0, size) in [(640, 479, 640), (641, 480, 640), (1920, 1080, 640), (333, 777, 416), (7, 3, 32), (1, 9, 32)] {
            let lb = Letterbox::compute(w0, h0, size);
            let r = lb.content_rect();
            let b = lb.apply_box([0.0, 0.0, w0 as f32, h0 as f32]);
            assert_eq!((r.x as f32, r.y as f32), (b[0], b[1]), "{w0}x{h0}->{size}: content origin vs mapped origin");
            assert_eq!(lb.pad_x.fract(), 0.0);
            assert_eq!(lb.pad_y.fract(), 0.0);
        }
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
        let row = lb.content_rect().y as usize; // 0 (a slack of one row puts it below)
        let at = |c: usize, x: usize| chw[c * plane + row * 2 + x];
        assert_eq!((at(0, 0), at(1, 0), at(2, 0)), (1.0, 2.0, 3.0));
        assert_eq!((at(0, 1), at(1, 1), at(2, 1)), (4.0, 5.0, 6.0));
    }

    /// Nearest-neighbour source selection is half-pixel, not torch's floor.
    #[test]
    fn nearest_rule_is_half_pixel_not_asymmetric_floor() {
        // 3 -> 2 downscale. half-pixel: taps 0 and 2; the floor rule: 0 and 1.
        assert_eq!((nearest_tap(0, 3, 2), nearest_tap(1, 3, 2)), (0, 2));
        let src: Vec<f32> = (0..9).map(|i| i as f32).collect(); // 3x1 RGB
        let (chw, _) = letterbox_rgb(&src, 3, 1, 2, 0.0);
        let plane = 4usize;
        assert_eq!(chw[0], 0.0, "dst x=0 -> src x=0");
        assert_eq!(chw[1], 6.0, "dst x=1 -> src x=2 (half-pixel), NOT src x=1 (floor)");
        assert_eq!(chw[plane + 1], 7.0);
    }

    #[test]
    fn nearest_tap_is_the_integer_form_and_stays_in_range() {
        // Upscaling a single sample clamps; the tie 0.5 of a 2 -> 1 resize rounds up.
        assert_eq!(nearest_tap(0, 2, 1), 1);
        assert_eq!((0..5).map(|o| nearest_tap(o, 1, 5)).collect::<Vec<_>>(), vec![0; 5]);
        assert_eq!((0..4).map(|o| nearest_tap(o, 2, 4)).collect::<Vec<_>>(), vec![0, 0, 1, 1]);
        for (i, o) in [(479, 640), (640, 479), (4000, 7), (7, 4000)] {
            assert!((0..o).all(|k| nearest_tap(k, i, o) < i));
        }
    }
}
