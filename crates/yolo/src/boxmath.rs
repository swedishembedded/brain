// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-Rust box geometry shared by the detection loss / assigner (P4) and the
//! later NMS + metrics (P6). All boxes here are in **pixel** coordinates unless
//! a function name says otherwise; the DFL distances and anchor centers are in
//! **feature** units (cell units), scaled to pixels by the per-anchor stride.
//!
//! The CIoU here is the SAME formula the `ciou.wgsl` kernel computes (it is the
//! assigner's alignment term `u`); it is plain `f32` Rust and does not need the
//! kernel's atan polyfill because the assigner is a non-differentiable constant
//! of the backward graph — a high-accuracy `f32::atan` is fine and slightly more
//! faithful than the kernel polyfill. The loss itself still differentiates the
//! kernel CIoU, so the two never need to agree to fp32.

/// A box as `(x1, y1, x2, y2)` (top-left / bottom-right), pixel coordinates.
pub type Xyxy = [f32; 4];

/// Decode a DFL `(l, t, r, b)` distance tuple (feature units) at the anchor
/// point `(ax, ay)` (feature units) with the given `stride` into a pixel-space
/// `xyxy` box:
///   x1 = (ax - l) * s,  y1 = (ay - t) * s,  x2 = (ax + r) * s,  y2 = (ay + b) * s.
#[inline]
pub fn dist_to_xyxy(dist: [f32; 4], ax: f32, ay: f32, stride: f32) -> Xyxy {
    let [l, t, r, b] = dist;
    [(ax - l) * stride, (ay - t) * stride, (ax + r) * stride, (ay + b) * stride]
}

/// Inverse of [`dist_to_xyxy`]: pixel `xyxy` -> DFL `(l, t, r, b)` distances in
/// feature units. (Used to build the assigner's DFL target distribution from a
/// GT box.) Distances are clamped non-negative so an off-center anchor never
/// asks the DFL bins for a negative target.
#[inline]
pub fn xyxy_to_dist(box_: Xyxy, ax: f32, ay: f32, stride: f32) -> [f32; 4] {
    let [x1, y1, x2, y2] = box_;
    [
        (ax - x1 / stride).max(0.0),
        (ay - y1 / stride).max(0.0),
        (x2 / stride - ax).max(0.0),
        (y2 / stride - ay).max(0.0),
    ]
}

/// Normalised `xywh` (center x/y, width, height; all in [0,1] of `img_size`) ->
/// pixel `xyxy`.
#[inline]
pub fn xywhn_to_xyxy(cx: f32, cy: f32, w: f32, h: f32, img_size: f32) -> Xyxy {
    let (cxp, cyp, wp, hp) = (cx * img_size, cy * img_size, w * img_size, h * img_size);
    [cxp - wp * 0.5, cyp - hp * 0.5, cxp + wp * 0.5, cyp + hp * 0.5]
}

/// Pixel `xyxy` -> `(cx, cy, w, h)` in pixels.
#[inline]
pub fn xyxy_to_xywh(b: Xyxy) -> [f32; 4] {
    [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5, b[2] - b[0], b[3] - b[1]]
}

/// Plain IoU of two pixel `xyxy` boxes.
pub fn iou(a: Xyxy, b: Xyxy) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let uni = (area_a + area_b - inter).max(1e-9);
    inter / uni
}

/// Complete-IoU of two pixel `xyxy` boxes (same definition as `ciou.wgsl`, but
/// in full-precision Rust). Returns the CIoU value in (-inf, 1]; the assigner
/// uses `u = CIoU` directly as its overlap term.
pub fn ciou(pred: Xyxy, tgt: Xyxy) -> f32 {
    let (px1, py1, px2, py2) = (pred[0], pred[1], pred[2], pred[3]);
    let (gx1, gy1, gx2, gy2) = (tgt[0], tgt[1], tgt[2], tgt[3]);
    let wp = px2 - px1;
    let hp = py2 - py1;
    let wg = gx2 - gx1;
    let hg = gy2 - gy1;

    let iw = (px2.min(gx2) - px1.max(gx1)).max(0.0);
    let ih = (py2.min(gy2) - py1.max(gy1)).max(0.0);
    let inter = iw * ih;
    let uni = (wp * hp + wg * hg - inter).max(1e-9);
    let iou = inter / uni;

    let cpx = (px1 + px2) * 0.5;
    let cpy = (py1 + py2) * 0.5;
    let cgx = (gx1 + gx2) * 0.5;
    let cgy = (gy1 + gy2) * 0.5;
    let rho2 = (cpx - cgx).powi(2) + (cpy - cgy).powi(2);

    let cw = px2.max(gx2) - px1.min(gx1);
    let ch = py2.max(gy2) - py1.min(gy1);
    let c2 = (cw * cw + ch * ch).max(1e-9);

    let atg = (wg / hg.max(1e-9)).atan();
    let atp = (wp / hp.max(1e-9)).atan();
    let diff = atg - atp;
    let k = 4.0 / (std::f32::consts::PI * std::f32::consts::PI);
    let v = k * diff * diff;
    let alpha = v / ((1.0 - iou) + v).max(1e-9);

    iou - rho2 / c2 - alpha * v
}

/// Letterbox geometry and the resize+pad+CHW pack, **re-exported** from
/// `imaging::letterbox` — the image substrate crate that owns every pixel-layout
/// operation in the workspace.
///
/// It used to be defined here, and this module is still where detection code
/// names it (`boxmath::Letterbox`, `boxmath::letterbox_rgb`), so no call site
/// moved. But the letterbox is not box *math*: it is an image transform, and
/// keeping a second definition next to `imaging`'s is exactly the drift the
/// "one implementation" rule (AGENTS.md) exists to prevent — the pad fill, the
/// `pad_y as usize` truncation and the half-pixel nearest rule are all baked
/// into the trained yolo weights and into every reported `map50`, so there must
/// be exactly one place they can be changed.
///
/// `imaging::letterbox`'s module header documents why the nearest rule is
/// half-pixel (and therefore *not* `resize_nearest.wgsl`) and why the
/// truncation must not be "fixed" without a `map50` gate.
pub use imaging::letterbox::{letterbox_rgb, Letterbox};

/// Is the pixel point `(px, py)` strictly inside the pixel `xyxy` box?
#[inline]
pub fn point_in_box(px: f32, py: f32, b: Xyxy) -> bool {
    px > b[0] && px < b[2] && py > b[1] && py < b[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_identical_is_one() {
        let b = [10.0, 20.0, 50.0, 80.0];
        assert!((iou(b, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_one_seventh_case() {
        // [0,0,10,10] vs [5,5,15,15]: inter = 25, union = 100+100-25 = 175,
        // IoU = 25/175 = 1/7.
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        assert!((iou(a, b) - 1.0 / 7.0).abs() < 1e-6, "iou = {}", iou(a, b));
    }

    #[test]
    fn ciou_identical_is_one() {
        let b = [10.0, 20.0, 50.0, 80.0];
        assert!((ciou(b, b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ciou_le_iou() {
        // For non-identical boxes CIoU <= IoU (penalty terms are non-negative).
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        assert!(ciou(a, b) <= iou(a, b) + 1e-6);
    }

    #[test]
    fn dist_xyxy_round_trip() {
        let (ax, ay, s) = (3.5f32, 4.5f32, 8.0f32);
        let dist = [1.2f32, 0.7, 2.1, 1.9];
        let b = dist_to_xyxy(dist, ax, ay, s);
        let back = xyxy_to_dist(b, ax, ay, s);
        for k in 0..4 {
            assert!((back[k] - dist[k]).abs() < 1e-4, "side {k}: {} vs {}", back[k], dist[k]);
        }
    }

    #[test]
    fn xywhn_to_xyxy_centered() {
        // A centered half-size box in a 100px image: center (50,50), size 50.
        let b = xywhn_to_xyxy(0.5, 0.5, 0.5, 0.5, 100.0);
        assert_eq!(b, [25.0, 25.0, 75.0, 75.0]);
    }

    #[test]
    fn letterbox_square_to_square_identity() {
        // A square image into a same-aspect square: scale = size/side, no pad.
        let lb = Letterbox::compute(100, 100, 200);
        assert!((lb.scale - 2.0).abs() < 1e-6);
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 0.0));
        let b = [10.0, 20.0, 30.0, 40.0];
        let fwd = lb.apply_box(b);
        let back = lb.invert_box(fwd, 100, 100);
        for k in 0..4 {
            assert!((back[k] - b[k]).abs() <= 1.0, "side {k}: {} vs {}", back[k], b[k]);
        }
    }

    #[test]
    fn letterbox_wide_to_square_recovers() {
        // Wide image: width fills, height padded (pad_y > 0, pad_x = 0).
        let (w0, h0, size) = (200u32, 100u32, 128u32);
        let lb = Letterbox::compute(w0, h0, size);
        assert!(lb.pad_x.abs() < 1e-6 && lb.pad_y > 0.0);
        let b = [20.0, 30.0, 120.0, 80.0];
        let back = lb.invert_box(lb.apply_box(b), w0, h0);
        for k in 0..4 {
            assert!((back[k] - b[k]).abs() <= 1.0, "side {k}: {} vs {}", back[k], b[k]);
        }
    }

    #[test]
    fn letterbox_tall_to_square_recovers() {
        // Tall image: height fills, width padded (pad_x > 0, pad_y = 0).
        let (w0, h0, size) = (100u32, 200u32, 128u32);
        let lb = Letterbox::compute(w0, h0, size);
        assert!(lb.pad_y.abs() < 1e-6 && lb.pad_x > 0.0);
        let b = [10.0, 40.0, 60.0, 150.0];
        let back = lb.invert_box(lb.apply_box(b), w0, h0);
        for k in 0..4 {
            assert!((back[k] - b[k]).abs() <= 1.0, "side {k}: {} vs {}", back[k], b[k]);
        }
    }

    #[test]
    fn letterbox_rgb_pads_and_places() {
        // A 2x1 image into a 4x4 square: scale = 2 (width 2 -> 4 fills), height
        // 1 -> 2, centre-padded by 1 row top/bottom.
        let src = vec![1.0, 0.0, 0.0, /*px0*/ 0.0, 1.0, 0.0 /*px1*/]; // HWC, 2 px
        let (chw, lb) = letterbox_rgb(&src, 2, 1, 4, 0.5);
        assert!((lb.scale - 2.0).abs() < 1e-6);
        assert_eq!((lb.new_w, lb.new_h), (4, 2));
        assert_eq!(lb.pad_y as u32, 1);
        // Top row (y=0) is pad.
        assert!((chw[0 * 16 + 0 * 4 + 0] - 0.5).abs() < 1e-6);
        // Content rows y=1,2 carry the (resized) image; finite + in range.
        assert!(chw.iter().all(|v| v.is_finite()));
        assert_eq!(chw.len(), 3 * 16);
    }

    #[test]
    fn point_in_box_basic() {
        let b = [10.0, 10.0, 20.0, 20.0];
        assert!(point_in_box(15.0, 15.0, b));
        assert!(!point_in_box(5.0, 15.0, b));
        assert!(!point_in_box(15.0, 25.0, b));
    }
}
