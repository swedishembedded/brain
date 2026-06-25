// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Non-Maximum Suppression over decoded detections (P6).
//!
//! A detection is `[x1, y1, x2, y2, conf, class]` (pixel `xyxy`, confidence in
//! `[0,1]`, integer class id stored as `f32`). NMS greedily keeps the highest-
//! confidence box and suppresses every lower-confidence box that overlaps it too
//! much.
//!
//! ## Class-aware vs class-agnostic
//! * **Class-aware** (the default, [`nms`]): boxes only suppress one another when
//!   they share a class. Two heavily-overlapping boxes of *different* classes are
//!   both kept (a person standing in front of a car is a valid double detection).
//! * **Class-agnostic** ([`nms_agnostic`]): any sufficiently-overlapping pair is
//!   suppressed regardless of class. Implemented by running the same greedy loop
//!   with the class-equality guard disabled.
//!
//! ## The suppression comparison: strict `>`
//! A candidate box `c` is suppressed by a kept box `k` iff
//! `iou(k, c) > iou_thresh` — a **strict** greater-than. A box whose IoU is
//! *exactly* `iou_thresh` is therefore **kept**, matching torchvision's
//! `nms` (which discards only pairs with `iou > threshold`). This makes the
//! threshold an exclusive upper bound on the IoU of two surviving same-class
//! boxes. (Choosing `>=` instead would suppress at the boundary; we deliberately
//! use `>` so a detection at exactly the threshold survives.)
//!
//! Ties in confidence are broken by original index (stable sort, lower index
//! wins) so the output is deterministic.

use crate::boxmath::{iou, Xyxy};

/// One decoded detection: `[x1, y1, x2, y2, conf, class]`.
pub type Detection = [f32; 6];

#[inline]
fn det_box(d: &Detection) -> Xyxy {
    [d[0], d[1], d[2], d[3]]
}

/// Class-aware greedy NMS. Sorts `dets` by confidence descending, then walks the
/// list keeping each box that is not suppressed (`iou > iou_thresh`) by an
/// already-kept box **of the same class**. Keeps at most `max_det` detections.
/// Returns the surviving detections in descending-confidence order.
pub fn nms(dets: &[Detection], iou_thresh: f32, max_det: usize) -> Vec<Detection> {
    nms_inner(dets, iou_thresh, max_det, /*class_aware=*/ true)
}

/// Class-agnostic greedy NMS: same as [`nms`] but boxes suppress one another
/// regardless of class.
pub fn nms_agnostic(dets: &[Detection], iou_thresh: f32, max_det: usize) -> Vec<Detection> {
    nms_inner(dets, iou_thresh, max_det, /*class_aware=*/ false)
}

fn nms_inner(dets: &[Detection], iou_thresh: f32, max_det: usize, class_aware: bool) -> Vec<Detection> {
    // Sort indices by confidence descending; ties -> lower original index first
    // (stable: sort by (-conf, idx)).
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&i, &j| {
        dets[j][4]
            .partial_cmp(&dets[i][4])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });

    let mut keep: Vec<Detection> = Vec::new();
    for &i in &order {
        if keep.len() >= max_det {
            break;
        }
        let cand = dets[i];
        let cand_box = det_box(&cand);
        let mut suppressed = false;
        for k in &keep {
            if class_aware && k[5] != cand[5] {
                continue;
            }
            // STRICT `>`: a box at exactly iou_thresh survives.
            if iou(det_box(k), cand_box) > iou_thresh {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(cand);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_same_class_keeps_highest() {
        let b = [0.0, 0.0, 10.0, 10.0];
        let dets = vec![
            [b[0], b[1], b[2], b[3], 0.9, 0.0],
            [b[0], b[1], b[2], b[3], 0.8, 0.0],
        ];
        let out = nms(&dets, 0.5, 100);
        assert_eq!(out.len(), 1);
        assert!((out[0][4] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn overlapping_different_class_aware_keeps_both() {
        let b = [0.0, 0.0, 10.0, 10.0];
        let dets = vec![
            [b[0], b[1], b[2], b[3], 0.9, 0.0],
            [b[0], b[1], b[2], b[3], 0.8, 1.0],
        ];
        assert_eq!(nms(&dets, 0.5, 100).len(), 2);
    }

    #[test]
    fn overlapping_agnostic_suppresses_lower() {
        let b = [0.0, 0.0, 10.0, 10.0];
        let dets = vec![
            [b[0], b[1], b[2], b[3], 0.9, 0.0],
            [b[0], b[1], b[2], b[3], 0.8, 1.0],
        ];
        let out = nms_agnostic(&dets, 0.5, 100);
        assert_eq!(out.len(), 1);
        assert!((out[0][4] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn non_overlapping_keeps_both() {
        let dets = vec![
            [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
            [100.0, 100.0, 110.0, 110.0, 0.8, 0.0],
        ];
        assert_eq!(nms(&dets, 0.5, 100).len(), 2);
    }
}
