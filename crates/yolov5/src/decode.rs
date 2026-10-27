// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning raw detector outputs into
// reliable detections on devices with no GPU for its clients. If your team needs
// expertise in detection post-processing for embedded targets then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Anchor decode and detection assembly on the host: raw head maps in, boxes in
//! original-image pixels out. Pure arithmetic over slices; no device, no
//! allocation beyond the candidate list, so the same code runs behind the fp32
//! reference forward and behind an NPU runtime that hands back the three head
//! tensors.
//!
//! ## Input contract
//!
//! Each head is one scale's output of the network as laid out by the model:
//! `[anchors * (5 + nc), H/stride, W/stride]` (batch 1, channel-major), **after
//! the sigmoid** - the export this crate imports applies it inside the graph, so
//! the head values are already probabilities and `0..1` box offsets. Channel
//! `a * (5 + nc) + k` holds field `k` of anchor `a`: `x, y, w, h, objectness`,
//! then the `nc` class scores.
//!
//! ## Decode (YOLOv5 v6)
//!
//! With `s` a sigmoid output, `(gx, gy)` the cell and `stride` the scale's
//! stride:
//!
//! ```text
//! cx = (2 s_x - 0.5 + gx) * stride        w = (2 s_w)^2 * anchor_w
//! cy = (2 s_y - 0.5 + gy) * stride        h = (2 s_h)^2 * anchor_h
//! score(c) = objectness * class_c         class = argmax_c score(c)
//! ```
//!
//! A candidate survives if its objectness and its best score both exceed the
//! confidence threshold (the reference post-processing filters on objectness
//! first, then on the product); survivors go through class-aware NMS and are
//! mapped back through the letterbox.

use imaging::nms::{nms, Detection};
use imaging::Letterbox;

use crate::config::{Yolov5Config, ANCHORS_PER_SCALE, BOX_FIELDS, STRIDES};
use crate::error::Error;

/// One scale's output, borrowed: `[anchors * (5 + nc), h, w]` row-major.
#[derive(Clone, Copy, Debug)]
pub struct HeadMap<'a> {
    pub data: &'a [f32],
    pub h: usize,
    pub w: usize,
}

/// Decode the three head maps into candidate boxes `[x1, y1, x2, y2, conf,
/// class]` in network-input pixels, before NMS, keeping those above
/// `conf_thresh`.
pub fn decode_candidates(cfg: &Yolov5Config, heads: [HeadMap<'_>; 3], conf_thresh: f32) -> Result<Vec<Detection>, Error> {
    let nc = cfg.nc as usize;
    let fields = BOX_FIELDS as usize + nc;
    let (h0, w0) = (heads[0].h * STRIDES[0] as usize, heads[0].w * STRIDES[0] as usize);

    let mut out = Vec::new();
    for (scale, head) in heads.iter().enumerate() {
        let stride = STRIDES[scale] as usize;
        if (head.h * stride, head.w * stride) != (h0, w0) {
            return Err(Error::Input(format!("head {scale} is {}x{} at stride {stride}, which is not the {h0}x{w0} input head 0 implies", head.h, head.w)));
        }
        let want = ANCHORS_PER_SCALE * fields * head.h * head.w;
        if head.data.len() != want {
            return Err(Error::Input(format!("head {scale} has {} values, expected {want} ({} x {} x {}x{})", head.data.len(), ANCHORS_PER_SCALE, fields, head.h, head.w)));
        }
        let plane = head.h * head.w;
        let s = stride as f32;
        for a in 0..ANCHORS_PER_SCALE {
            let (anchor_w, anchor_h) = (cfg.anchors[scale][2 * a], cfg.anchors[scale][2 * a + 1]);
            let field = |k: usize, cell: usize| head.data[(a * fields + k) * plane + cell];
            for cell in 0..plane {
                let obj = field(4, cell);
                if obj <= conf_thresh {
                    continue;
                }
                let (mut best, mut best_class) = (f32::NEG_INFINITY, 0);
                for c in 0..nc {
                    let score = obj * field(BOX_FIELDS as usize + c, cell);
                    if score > best {
                        best = score;
                        best_class = c;
                    }
                }
                if best <= conf_thresh {
                    continue;
                }
                let (gx, gy) = ((cell % head.w) as f32, (cell / head.w) as f32);
                let cx = (field(0, cell) * 2.0 - 0.5 + gx) * s;
                let cy = (field(1, cell) * 2.0 - 0.5 + gy) * s;
                let w = (field(2, cell) * 2.0).powi(2) * anchor_w;
                let h = (field(3, cell) * 2.0).powi(2) * anchor_h;
                out.push([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, best, best_class as f32]);
            }
        }
    }
    Ok(out)
}

/// The full tail: [`decode_candidates`], class-aware NMS at `iou_thresh`, at most
/// `max_det` survivors, then un-letterbox into the `orig` `(width, height)`
/// image. Detections are `[x1, y1, x2, y2, conf, class]`, highest confidence
/// first.
pub fn detections(
    cfg: &Yolov5Config,
    heads: [HeadMap<'_>; 3],
    letterbox: &Letterbox,
    orig: (u32, u32),
    conf_thresh: f32,
    iou_thresh: f32,
    max_det: usize,
) -> Result<Vec<Detection>, Error> {
    let kept = nms(&decode_candidates(cfg, heads, conf_thresh)?, iou_thresh, max_det);
    Ok(kept
        .into_iter()
        .map(|d| {
            let b = letterbox.invert_box([d[0], d[1], d[2], d[3]], orig.0, orig.1);
            [b[0], b[1], b[2], b[3], d[4], d[5]]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;

    /// A 32x32 input: head sizes 4x4, 2x2, 1x1, all-zero (obj 0) maps.
    fn blank(cfg: &Yolov5Config) -> [Vec<f32>; 3] {
        let f = (BOX_FIELDS + cfg.nc) as usize * ANCHORS_PER_SCALE;
        [vec![0.0; f * 16], vec![0.0; f * 4], vec![0.0; f]]
    }

    fn heads<'a>(m: &'a [Vec<f32>; 3]) -> [HeadMap<'a>; 3] {
        [HeadMap { data: &m[0], h: 4, w: 4 }, HeadMap { data: &m[1], h: 2, w: 2 }, HeadMap { data: &m[2], h: 1, w: 1 }]
    }

    fn set(m: &mut [f32], fields: usize, plane: usize, a: usize, cell: usize, v: &[f32]) {
        for (k, x) in v.iter().enumerate() {
            m[(a * fields + k) * plane + cell] = *x;
        }
    }

    #[test]
    fn decode_follows_the_v6_formulas() {
        let mut cfg = Yolov5Config::yolov5s(Activation::Relu);
        cfg.nc = 2;
        let fields = 7;
        let mut m = blank(&cfg);
        // P4 (stride 16), anchor 1 (62x45), cell (gx=1, gy=0): s = (0.75, 0.5, 0.5, 1.0),
        // objectness 0.8, classes (0.1, 0.9).
        set(&mut m[1], fields, 4, 1, 1, &[0.75, 0.5, 0.5, 1.0, 0.8, 0.1, 0.9]);
        let c = decode_candidates(&cfg, heads(&m), 0.25).unwrap();
        assert_eq!(c.len(), 1);
        let d = c[0];
        let (cx, cy) = ((2.0 * 0.75 - 0.5 + 1.0) * 16.0, (2.0 * 0.5 - 0.5 + 0.0) * 16.0);
        let (w, h) = ((2.0f32 * 0.5).powi(2) * 62.0, (2.0f32 * 1.0).powi(2) * 45.0);
        let want = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        for k in 0..4 {
            assert!((d[k] - want[k]).abs() < 1e-4, "coord {k}: {} vs {}", d[k], want[k]);
        }
        assert!((d[4] - 0.8 * 0.9).abs() < 1e-6);
        assert_eq!(d[5], 1.0);
    }

    #[test]
    fn the_confidence_filter_applies_to_objectness_and_to_the_product() {
        let mut cfg = Yolov5Config::yolov5s(Activation::Relu);
        cfg.nc = 1;
        let fields = 6;
        let mut m = blank(&cfg);
        set(&mut m[2], fields, 1, 0, 0, &[0.5, 0.5, 0.5, 0.5, 0.9, 0.2]); // product 0.18
        set(&mut m[2], fields, 1, 1, 0, &[0.5, 0.5, 0.5, 0.5, 0.2, 1.0]); // objectness 0.2
        set(&mut m[2], fields, 1, 2, 0, &[0.5, 0.5, 0.5, 0.5, 0.9, 0.5]); // product 0.45
        let c = decode_candidates(&cfg, heads(&m), 0.25).unwrap();
        assert_eq!(c.len(), 1);
        assert!((c[0][4] - 0.45).abs() < 1e-6);
    }

    #[test]
    fn detections_are_nmsd_per_class_and_mapped_back_through_the_letterbox() {
        let mut cfg = Yolov5Config::yolov5s(Activation::Relu);
        cfg.nc = 2;
        let fields = 7;
        let mut m = blank(&cfg);
        // A (class 0, conf 0.81) and its neighbour B (class 0, conf 0.72): 32x42 px boxes
        // one stride (8 px) apart, IoU about 0.6, so B is suppressed. C is class 1 at
        // A's cell on another anchor, overlapping A heavily but of another class: kept.
        set(&mut m[0], fields, 16, 0, 5, &[0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.0]);
        set(&mut m[0], fields, 16, 0, 6, &[0.5, 0.5, 0.9, 0.9, 0.8, 0.9, 0.0]);
        set(&mut m[0], fields, 16, 1, 5, &[0.5, 0.5, 0.5, 0.5, 0.7, 0.0, 0.9]);
        // Identity letterbox: a 32x32 image into the 32x32 input.
        let lb = Letterbox::compute(32, 32, 32);
        let d = detections(&cfg, heads(&m), &lb, (32, 32), 0.25, 0.45, 300).unwrap();
        assert_eq!(d.len(), 2, "{d:?}");
        assert_eq!((d[0][5], d[1][5]), (0.0, 1.0));
        assert!((d[0][4] - 0.81).abs() < 1e-6 && (d[1][4] - 0.63).abs() < 1e-6, "{d:?}");
        // A's box: centre (12, 12), size (2*0.9)^2 * (10, 13) = 32.4 x 42.1. The
        // un-letterbox clamps it to the 32x32 image: x1 and y1 go from negative to 0,
        // y2 from 33.05 to 32; x2 = 12 + 16.2 is inside and untouched.
        assert!((d[0][2] - 28.2).abs() < 1e-4, "{d:?}");
        assert_eq!((d[0][0], d[0][1], d[0][3]), (0.0, 0.0, 32.0), "{d:?}");
    }

    #[test]
    fn mismatched_head_sizes_are_refused() {
        let cfg = Yolov5Config::yolov5s(Activation::Relu);
        let m = blank(&cfg);
        let mut h = heads(&m);
        h[1].h = 3;
        assert!(matches!(decode_candidates(&cfg, h, 0.25), Err(Error::Input(_))));
        let short = [vec![0.0; 3], m[1].clone(), m[2].clone()];
        assert!(matches!(decode_candidates(&cfg, heads(&short), 0.25), Err(Error::Input(_))));
    }
}
