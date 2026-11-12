// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The networks' input and output contracts: how a frame becomes an input
//! tensor, how the detector's map becomes a box, how a box becomes the
//! landmark crop, and how the landmark map becomes 66 points.
//!
//! These follow OpenSeeFace's reference tracker (`tracker.py`, BSD-2-Clause,
//! Copyright (c) 2019 Emiliana; `Tracker.detect_faces`, the crop in
//! `Tracker.predict`, `clamp_to_im`, `Tracker.preprocess`,
//! `Tracker.landmarks` and `logit_arr`), because the released weights were
//! trained against exactly these conventions; see NOTICE.md. Two of them are
//! easy to get wrong and are kept on purpose:
//!
//! * the crop's bottom/top edges come out of the clamp one pixel lower than
//!   computed (`clamp_to_im` returns `y + 1`), and
//! * a landmark's row decodes through the crop's VERTICAL scale and its
//!   column through the horizontal one, with the heatmap cell index split as
//!   `row = idx / 28`, `col = idx % 28`.

/// Input side of every network, pixels.
pub const INPUT: u32 = 224;
/// Landmark heatmap side.
pub const HEATMAP: usize = 28;
/// Detector map side.
pub const DETECTOR_MAP: usize = 56;
/// Points per face.
pub const POINTS: usize = 66;
/// Detector peaks below this are not faces (the reference default).
pub const DETECTION_THRESHOLD: f32 = 0.6;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// The landmark offsets' logit scale (the reference's `logit_factor`).
const LOGIT_FACTOR: f32 = 16.0;

/// ImageNet-normalize an RGB image of 0..255 values, HWC, into the NCHW
/// tensor the networks read.
pub fn normalize(rgb_hwc: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(rgb_hwc.len(), w * h * 3, "normalize: {w}x{h} RGB needs {} values", w * h * 3);
    let mut out = vec![0.0f32; rgb_hwc.len()];
    for c in 0..3 {
        let (scale, offset) = (1.0 / (STD[c] * 255.0), -MEAN[c] / STD[c]);
        for i in 0..w * h {
            out[c * w * h + i] = rgb_hwc[i * 3 + c] * scale + offset;
        }
    }
    out
}

/// The detector's input for a frame: the whole frame, bilinearly resized
/// (half-pixel, the reference's cv2 `INTER_LINEAR`) to 224x224.
pub fn detector_input(rgb: &[u8], w: u32, h: u32) -> Vec<f32> {
    let f: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
    let r = imaging::host::resize_bilinear_hwc(&f, 3, w, h, INPUT, INPUT);
    normalize(&r, INPUT as usize, INPUT as usize)
}

/// A face box in frame pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

/// The strongest face in the detector's output, if any clears `threshold`:
/// a cell that survives the 3x3 max-pool (a local peak), its likelihood the
/// score and its radius channel the half-size, scaled from the 224 input to
/// the frame.
pub fn strongest_face(out: &[f32], pooled: &[f32], frame_w: u32, frame_h: u32, threshold: f32) -> Option<FaceBox> {
    let n = DETECTOR_MAP * DETECTOR_MAP;
    assert!(out.len() == 2 * n && pooled.len() == 2 * n, "strongest_face: expected two {DETECTOR_MAP}x{DETECTOR_MAP} maps");
    let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
    for i in 0..n {
        let v = if out[i] == pooled[i] { out[i] } else { 0.0 };
        if v > best {
            best = v;
            at = i;
        }
    }
    if best.is_nan() || best < threshold {
        return None;
    }
    let (y, x) = ((at / DETECTOR_MAP) as f32, (at % DETECTOR_MAP) as f32);
    let r = out[n + at] * 112.0;
    let (sx, sy) = (frame_w as f32 / INPUT as f32, frame_h as f32 / INPUT as f32);
    Some(FaceBox { x: (x * 4.0 - r) * sx, y: (y * 4.0 - r) * sy, w: 2.0 * r * sx, h: 2.0 * r * sy, score: best })
}

/// The landmark crop for a box, `(x1, y1, x2, y2)` in frame pixels: the box
/// grown by 10% each side horizontally and 12.5% vertically, clamped into the
/// frame. `None` when what is left is under 4 pixels on a side.
pub fn crop_for(b: &FaceBox, frame_w: u32, frame_h: u32) -> Option<[i32; 4]> {
    let clamp = |x: f32, y: f32| -> (i32, i32) {
        let x = x.clamp(0.0, frame_w as f32 - 1.0);
        let y = y.clamp(0.0, frame_h as f32 - 1.0);
        (x as i32, (y + 1.0) as i32)
    };
    let (gx, gy) = ((b.w * 0.1) as i32 as f32, (b.h * 0.125) as i32 as f32);
    let (x1, y1) = clamp(b.x - gx, b.y - gy);
    let (x2, y2) = clamp(b.x + b.w + gx, b.y + b.h + gy);
    (x2 - x1 >= 4 && y2 - y1 >= 4).then_some([x1, y1, x2, y2])
}

/// The landmark network's input for a crop: the crop's pixels resized to
/// 224x224 and normalized.
pub fn crop_input(rgb: &[u8], w: u32, crop: [i32; 4]) -> Vec<f32> {
    let [x1, y1, x2, y2] = crop;
    let (cw, ch) = ((x2 - x1) as u32, (y2 - y1) as u32);
    let mut f = Vec::with_capacity((cw * ch * 3) as usize);
    for y in y1..y2 {
        let row = (y as u32 * w) as usize * 3;
        for v in &rgb[row + x1 as usize * 3..row + x2 as usize * 3] {
            f.push(*v as f32);
        }
    }
    let r = imaging::host::resize_bilinear_hwc(&f, 3, cw, ch, INPUT, INPUT);
    normalize(&r, INPUT as usize, INPUT as usize)
}

/// 66 points in frame pixels, each with the confidence of its heatmap peak.
#[derive(Clone, Debug, PartialEq)]
pub struct Landmarks {
    /// `(x, y)` per point.
    pub points: Vec<[f32; 2]>,
    pub confidence: Vec<f32>,
}

impl Landmarks {
    /// The mean confidence over all points: the reference's face score.
    pub fn score(&self) -> f32 {
        self.confidence.iter().sum::<f32>() / self.confidence.len() as f32
    }

    /// The landmarks' extent `(x, y, w, h)`: the next frame's face box.
    pub fn extent(&self) -> FaceBox {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &self.points {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        FaceBox { x: x0, y: y0, w: x1 - x0, h: y1 - y0, score: self.score() }
    }
}

fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln() / LOGIT_FACTOR
}

/// Decode the landmark network's `[198, 28, 28]` output for `crop`: per
/// point, the heatmap argmax and the logit-decoded sub-cell offsets read at
/// that cell, mapped back through the crop.
pub fn landmarks(out: &[f32], crop: [i32; 4]) -> Landmarks {
    let cells = HEATMAP * HEATMAP;
    assert_eq!(out.len(), 3 * POINTS * cells, "landmarks: expected [{}, {HEATMAP}, {HEATMAP}]", 3 * POINTS);
    let [x1, y1, x2, y2] = crop;
    let res = INPUT as f32;
    let (sx, sy) = ((x2 - x1) as f32 / res, (y2 - y1) as f32 / res);
    let r = res - 1.0;
    let span = (HEATMAP - 1) as f32;
    let mut points = Vec::with_capacity(POINTS);
    let mut confidence = Vec::with_capacity(POINTS);
    for i in 0..POINTS {
        let heat = &out[i * cells..(i + 1) * cells];
        let (m, conf) = heat.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| if v > bv { (j, v) } else { (bi, bv) });
        let off_row = r * logit(out[(POINTS + i) * cells + m]);
        let off_col = r * logit(out[(2 * POINTS + i) * cells + m]);
        let y = y1 as f32 + sy * (r * (m / HEATMAP) as f32 / span + off_row);
        let x = x1 as f32 + sx * (r * (m % HEATMAP) as f32 / span + off_col);
        if x.is_finite() && y.is_finite() && conf.is_finite() {
            points.push([x, y]);
            confidence.push(conf);
        } else {
            points.push([0.0, 0.0]);
            confidence.push(0.0);
        }
    }
    Landmarks { points, confidence }
}
