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

/// Letterbox geometry: the scale + padding that maps an original `(w0, h0)`
/// image onto a square `size x size` model input while preserving aspect ratio.
///
/// Forward map (original px -> input px):  `xi = x0 * scale + pad_x`,
/// `yi = y0 * scale + pad_y`.  `scale = min(size/w0, size/h0)` (downscale-and-
/// pad; the longer side fills the square, the shorter side is centre-padded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    /// Scaled content size (before padding), `round(w0*scale)` x `round(h0*scale)`.
    pub new_w: u32,
    pub new_h: u32,
    /// Target square side.
    pub size: u32,
}

impl Letterbox {
    /// Compute the letterbox transform for an original `w0 x h0` image fitted to
    /// a `size x size` square, preserving aspect ratio and centre-padding.
    pub fn compute(w0: u32, h0: u32, size: u32) -> Letterbox {
        let s = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
        let new_w = (w0 as f32 * s).round() as u32;
        let new_h = (h0 as f32 * s).round() as u32;
        let pad_x = (size as f32 - new_w as f32) * 0.5;
        let pad_y = (size as f32 - new_h as f32) * 0.5;
        Letterbox { scale: s, pad_x, pad_y, new_w, new_h, size }
    }

    /// Map an `xyxy` box from ORIGINAL-image coords into letterboxed INPUT coords.
    pub fn apply_box(&self, b: Xyxy) -> Xyxy {
        [
            b[0] * self.scale + self.pad_x,
            b[1] * self.scale + self.pad_y,
            b[2] * self.scale + self.pad_x,
            b[3] * self.scale + self.pad_y,
        ]
    }

    /// Inverse: map an `xyxy` box from letterboxed INPUT coords back to ORIGINAL
    /// coords, clamped to the original `[0,w0] x [0,h0]` frame.
    pub fn invert_box(&self, b: Xyxy, w0: u32, h0: u32) -> Xyxy {
        let inv = 1.0 / self.scale;
        let x1 = ((b[0] - self.pad_x) * inv).clamp(0.0, w0 as f32);
        let y1 = ((b[1] - self.pad_y) * inv).clamp(0.0, h0 as f32);
        let x2 = ((b[2] - self.pad_x) * inv).clamp(0.0, w0 as f32);
        let y2 = ((b[3] - self.pad_y) * inv).clamp(0.0, h0 as f32);
        [x1, y1, x2, y2]
    }
}

/// Resize an interleaved-RGB `src` (`w0 x h0`, row-major `[h0*w0*3]`, channel-
/// last, u8-as-f32 or already-normalised) into a letterboxed CHW float tensor
/// `[3 * size * size]` (the model input layout). Pad value is `pad` (default
/// grey 114/255 is common; caller supplies it). Uses nearest-neighbour resize
/// (adequate for the from-scratch CPU detector; bilinear is a later refinement).
/// Returns `(chw, lb)` — the CHW input buffer and the letterbox transform.
pub fn letterbox_rgb(src: &[f32], w0: u32, h0: u32, size: u32, pad: f32) -> (Vec<f32>, Letterbox) {
    assert_eq!(src.len(), (w0 * h0 * 3) as usize, "src must be HWC RGB [h0*w0*3]");
    let lb = Letterbox::compute(w0, h0, size);
    let sz = size as usize;
    let mut chw = vec![pad; 3 * sz * sz];
    let inv = 1.0 / lb.scale;
    for yi in 0..lb.new_h as usize {
        // source row for this destination row (nearest neighbour).
        let sy = ((yi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, h0 as f32 - 1.0) as usize;
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
