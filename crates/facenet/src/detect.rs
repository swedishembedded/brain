// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SCRFD post-processing: anchor generation, distance decoding and NMS.
//!
//! Host code by design. Every function here works on the *rows* the head already
//! produced (≤ 12 800 × 10 floats), thresholds them down to a handful, and then
//! runs a data-dependent, sequential greedy suppression. That is
//! `crates/imaging`'s "a reduction to a handful of scalars / policy" category,
//! not per-pixel image work — the per-pixel work is the network, and it is all on
//! the device.

use crate::config::ScrfdConfig;

/// A detected face: box in source-image pixels, score, and the five landmarks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    pub score: f32,
    /// `[5, 2]` as `(x, y)`, in left-eye / right-eye / nose / left-mouth /
    /// right-mouth order — the order [`crate::align::estimate_norm`] expects.
    pub kps: [[f32; 2]; 5],
}

impl Face {
    /// `(x2 - x1) * (y2 - y1)`. Used to pick the primary face; note this is the
    /// plain area, NOT the `+1` convention NMS uses (see [`nms`]).
    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]).max(0.0) * (self.bbox[3] - self.bbox[1]).max(0.0)
    }
}

/// Anchor centres for one stride, `[rows, 2]` as `(x, y)` in input pixels.
///
/// The reference builds them as
/// `np.stack(np.mgrid[:h, :w][::-1], -1) * stride`, then repeats each location
/// `num_anchors` times **contiguously**. So row `(y*W + x)*A + a` is
/// `(x*stride, y*stride)` — the same row order the head's
/// `transpose(2,3,0,1).reshape(-1, k)` produces. Interleaving the two orders
/// differently is a silent, plausible box shift.
pub fn anchor_centers(h: u32, w: u32, stride: u32, num_anchors: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((h * w * num_anchors * 2) as usize);
    for y in 0..h {
        for x in 0..w {
            for _ in 0..num_anchors {
                v.push((x * stride) as f32);
                v.push((y * stride) as f32);
            }
        }
    }
    v
}

/// `distance2bbox`: left/top/right/bottom distances from an anchor centre.
fn distance2bbox(cx: f32, cy: f32, d: &[f32]) -> [f32; 4] {
    [cx - d[0], cy - d[1], cx + d[2], cy + d[3]]
}

/// Greedy IoU NMS over score-sorted boxes, returning kept indices.
///
/// **The `+1` area convention is load-bearing.** insightface computes
/// `(x2-x1+1)*(y2-y1+1)`, inherited from the original Fast R-CNN code. Dropping
/// the `+1` changes which overlapping boxes survive on small faces — a different
/// detection set, from code that looks correct.
///
/// The `+1` also makes IoU **scale-dependent**, which is why [`decode`] runs this
/// on boxes already divided by `det_scale`: the reference thresholds in
/// source-image pixels, and the same two boxes can straddle `nms_thresh`
/// differently in detector-canvas pixels.
///
/// **Tie-break.** The reference orders by `scores.argsort()[::-1]`, and numpy's
/// argsort is a stable ASCENDING sort, so reversing it puts the *higher* index
/// first among equal scores. This sorts by `(score desc, index desc)` to match;
/// a plain stable descending sort keeps the lower index first and can suppress
/// the other member of the pair.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], thresh: f32) -> Vec<usize> {
    let n = boxes.len();
    assert_eq!(scores.len(), n, "nms: one score per box");
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a))
    });
    let areas: Vec<f32> =
        boxes.iter().map(|b| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)).collect();

    let mut keep = Vec::new();
    let mut suppressed = vec![false; n];
    for oi in 0..n {
        let i = order[oi];
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order[oi + 1..] {
            if suppressed[j] {
                continue;
            }
            let xx1 = boxes[i][0].max(boxes[j][0]);
            let yy1 = boxes[i][1].max(boxes[j][1]);
            let xx2 = boxes[i][2].min(boxes[j][2]);
            let yy2 = boxes[i][3].min(boxes[j][3]);
            let iw = (xx2 - xx1 + 1.0).max(0.0);
            let ih = (yy2 - yy1 + 1.0).max(0.0);
            let inter = iw * ih;
            let ovr = inter / (areas[i] + areas[j] - inter);
            if ovr > thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Decode the nine raw head outputs into faces.
///
/// `score[i]`, `bbox[i]`, `kps[i]` are the per-stride row-major outputs
/// (`[rows,1] / [rows,4] / [rows,10]`) in `cfg.strides` order —
/// `ScrfdTaps::out_{score,bbox,kps}`. Both `bbox` and `kps` are in **stride
/// units** and are multiplied by the stride here, exactly as the reference does
/// before decoding.
///
/// `det_scale` divides every coordinate, mapping the padded 640×640 detector
/// canvas back to source-image pixels (the reference's `det = bboxes /
/// det_scale`). Pass `1.0` to stay in detector space.
///
/// **The division happens BEFORE NMS**, exactly where insightface's
/// `SCRFD.detect` puts it (`pre = hstack([vstack(boxes)/det_scale, scores])`
/// then `nms(pre)`). That is not cosmetic: [`nms`] uses the Fast R-CNN `+1` area
/// convention, which is *not* scale-invariant, so suppressing in detector-canvas
/// pixels and scaling afterwards gives a different — plausible, wrong —
/// detection set whenever `det_scale != 1`.
pub fn decode(
    cfg: &ScrfdConfig,
    score: &[Vec<f32>; 3],
    bbox: &[Vec<f32>; 3],
    kps: &[Vec<f32>; 3],
    det_scale: f32,
) -> Vec<Face> {
    let mut boxes: Vec<[f32; 4]> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let mut kpss: Vec<[[f32; 2]; 5]> = Vec::new();

    for (si, &stride) in cfg.strides.iter().enumerate() {
        let side = cfg.image_size / stride;
        let rows = (side * side * cfg.num_anchors) as usize;
        assert_eq!(score[si].len(), rows, "scrfd decode: stride {stride} score rows");
        assert_eq!(bbox[si].len(), rows * 4, "scrfd decode: stride {stride} bbox rows");
        assert_eq!(kps[si].len(), rows * 10, "scrfd decode: stride {stride} kps rows");
        let ac = anchor_centers(side, side, stride, cfg.num_anchors);
        let sf = stride as f32;
        for r in 0..rows {
            let s = score[si][r];
            if s < cfg.det_thresh {
                continue;
            }
            let (cx, cy) = (ac[2 * r], ac[2 * r + 1]);
            let d: Vec<f32> = bbox[si][4 * r..4 * r + 4].iter().map(|v| v * sf).collect();
            let b = distance2bbox(cx, cy, &d);
            boxes.push([b[0] / det_scale, b[1] / det_scale, b[2] / det_scale, b[3] / det_scale]);
            scores.push(s);
            let mut k = [[0.0f32; 2]; 5];
            for (j, kj) in k.iter_mut().enumerate() {
                kj[0] = (cx + kps[si][10 * r + 2 * j] * sf) / det_scale;
                kj[1] = (cy + kps[si][10 * r + 2 * j + 1] * sf) / det_scale;
            }
            kpss.push(k);
        }
    }

    // Source-image pixels from here on — see the `det_scale` note above.
    let keep = nms(&boxes, &scores, cfg.nms_thresh);
    keep.into_iter()
        .map(|i| Face { bbox: boxes[i], score: scores[i], kps: kpss[i] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_rows_repeat_each_location_contiguously() {
        let a = anchor_centers(2, 3, 8, 2);
        assert_eq!(a.len(), 2 * 3 * 2 * 2);
        // location (y=0, x=1) is rows 2 and 3, both (8, 0)
        assert_eq!(&a[4..8], &[8.0, 0.0, 8.0, 0.0]);
        // location (y=1, x=0) is rows 6, 7 -> (0, 8)
        assert_eq!(&a[12..16], &[0.0, 8.0, 0.0, 8.0]);
    }

    /// The `+1` in the area convention changes the answer; pin it with a case
    /// that straddles the threshold.
    #[test]
    fn nms_uses_the_plus_one_area_convention() {
        let a = [0.0f32, 0.0, 9.0, 9.0]; // 10x10 with +1
        let b = [5.0f32, 0.0, 14.0, 9.0]; // overlap 5x10 = 50 with +1
        // +1 areas: 100 each, inter 50, iou = 50/150 = 0.3333
        let keep = nms(&[a, b], &[0.9, 0.8], 0.34);
        assert_eq!(keep.len(), 2, "iou 0.333 must not suppress at thresh 0.34");
        let keep = nms(&[a, b], &[0.9, 0.8], 0.30);
        assert_eq!(keep, vec![0], "iou 0.333 must suppress at thresh 0.30");
    }

    #[test]
    fn nms_returns_indices_in_descending_score_order() {
        let boxes = [[0.0f32, 0.0, 1.0, 1.0], [100.0, 100.0, 101.0, 101.0], [200.0, 200.0, 201.0, 201.0]];
        let keep = nms(&boxes, &[0.1, 0.9, 0.5], 0.5);
        assert_eq!(keep, vec![1, 2, 0]);
    }

    /// Equal scores must order like `numpy.argsort()[::-1]` — higher index
    /// first — because that is what decides which of the two is the survivor.
    #[test]
    fn equal_scores_break_the_tie_the_way_the_reference_does() {
        let boxes = [[0.0f32, 0.0, 1.0, 1.0], [100.0, 100.0, 101.0, 101.0]];
        assert_eq!(nms(&boxes, &[0.5, 0.5], 0.5), vec![1, 0]);
    }

    /// The `+1` area convention makes IoU scale-dependent, so `decode` MUST
    /// divide by `det_scale` before suppressing. This pair sits at IoU 0.3333 in
    /// detector pixels and 0.375 at `det_scale = 2`, straddling a 0.35
    /// threshold: the wrong order keeps both faces.
    #[test]
    fn nms_runs_in_source_pixels_not_detector_pixels() {
        let a = [0.0f32, 0.0, 9.0, 9.0];
        let b = [5.0f32, 0.0, 14.0, 9.0];
        let half = |x: [f32; 4]| [x[0] / 2.0, x[1] / 2.0, x[2] / 2.0, x[3] / 2.0];
        assert_eq!(nms(&[a, b], &[0.9, 0.8], 0.35).len(), 2, "0.3333 in detector pixels");
        assert_eq!(nms(&[half(a), half(b)], &[0.9, 0.8], 0.35).len(), 1, "0.375 at det_scale 2");

        // …and `decode` must produce the second answer.
        let cfg = ScrfdConfig {
            image_size: 32,
            num_anchors: 1,
            det_thresh: 0.5,
            nms_thresh: 0.35,
            ..ScrfdConfig::scrfd_10g_bnkps()
        };
        let rows = [16usize, 4, 1]; // (32/8)^2, (32/16)^2, (32/32)^2 at 1 anchor
        let mut score = [vec![0.0f32; rows[0]], vec![0.0; rows[1]], vec![0.0; rows[2]]];
        let mut bbox = [vec![0.0f32; rows[0] * 4], vec![0.0; rows[1] * 4], vec![0.0; rows[2] * 4]];
        let kps = [vec![0.0f32; rows[0] * 10], vec![0.0; rows[1] * 10], vec![0.0; rows[2] * 10]];
        // stride-8 row 0 sits at (0,0), row 1 at (8,0); distances are in stride units.
        score[0][0] = 0.9;
        bbox[0][0..4].copy_from_slice(&[0.0, 0.0, 9.0 / 8.0, 9.0 / 8.0]);
        score[0][1] = 0.8;
        bbox[0][4..8].copy_from_slice(&[3.0 / 8.0, 0.0, 6.0 / 8.0, 9.0 / 8.0]);

        assert_eq!(decode(&cfg, &score, &bbox, &kps, 1.0).len(), 2, "detector space keeps both");
        let faces = decode(&cfg, &score, &bbox, &kps, 2.0);
        assert_eq!(faces.len(), 1, "at det_scale 2 the pair suppresses");
        assert_eq!(faces[0].bbox, [0.0, 0.0, 4.5, 4.5]);
    }
}
