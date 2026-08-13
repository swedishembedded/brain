// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Object-detection metrics, pure Rust (P6).
//!
//! The user's spec is explicit: the evaluator is golden-tested BEFORE any mAP is
//! trusted. So everything here is constructed-input, deterministic, model-free.
//!
//! A detection is `[x1, y1, x2, y2, conf, class]` (pixel `xyxy`, confidence,
//! class id as `f32`). A ground-truth box is [`GtBox`] = `(class, xyxy)`.
//!
//! ## Matching (class-aware, greedy, by confidence)
//! Per the standard VOC/COCO protocol: sort predictions by confidence
//! descending; a prediction is a **true positive** if it has IoU `>= iou_thr`
//! with an as-yet-unmatched ground-truth box **of the same class**, choosing the
//! highest-IoU available GT; otherwise it is a **false positive**. Each GT can be
//! matched at most once (extra detections of the same object are FPs).
//!
//! ## Precision / Recall
//! At a single (conf, IoU) operating point: `precision = TP / (TP+FP)`,
//! `recall = TP / n_gt`. With no predictions precision is defined as 0 (and
//! recall 0). With no GT for a class, recall is undefined and that class is
//! excluded from the mean (see below).
//!
//! ## Average Precision @0.5 (VOC/COCO all-points integration)
//! Sort that class's predictions by confidence descending, accumulate TP/FP into
//! a precision-recall curve, take the monotonically-decreasing precision envelope
//! (`p_interp(r) = max_{r'>=r} p(r')`), and integrate it over recall `[0,1]`:
//! `AP = sum_k (r_k - r_{k-1}) * p_interp(r_k)`. This is the COCO "101-point-free"
//! exact area under the interpolated curve (equivalently VOC2010+).
//!
//! ## Per-class averaging convention
//! [`map50`] averages AP over **only the classes that have at least one
//! ground-truth box** (absent classes are EXCLUDED, not counted as AP=0). This is
//! the Ultralytics/COCO convention: a class the dataset never labels does not
//! drag the mean to zero. A class that HAS ground truth but gets zero correct
//! predictions contributes AP=0 (it is included). If no class has any GT, `map50`
//! returns 0.

use yolov8::boxmath::iou;
use yolov8::Detection;

/// A ground-truth box for one image: class id + pixel `xyxy`.
#[derive(Clone, Copy, Debug)]
pub struct GtBox {
    pub class: u32,
    pub bbox: [f32; 4],
}

/// Pairwise IoU between a prediction box and a GT box (re-exported convenience;
/// uses [`yolov8::boxmath::iou`] so the metric and the detector share one IoU).
pub fn pairwise_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    iou(a, b)
}

/// Per-prediction match outcome, in the SAME order as the input `preds`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Match {
    /// True positive: matched GT index (global into the `gts` slice).
    Tp(usize),
    /// False positive (no available same-class GT above the IoU threshold).
    Fp,
}

/// Greedy class-aware matching at IoU threshold `iou_thr`. Returns, for each
/// prediction (in confidence-descending order), its [`Match`] plus the original
/// prediction index, so callers can rebuild a confidence-sorted TP/FP sequence.
/// Each GT is matched at most once.
pub fn match_dets(preds: &[Detection], gts: &[GtBox], iou_thr: f32) -> Vec<(usize, Match)> {
    // confidence-descending order, ties broken by index (stable, deterministic).
    let mut order: Vec<usize> = (0..preds.len()).collect();
    order.sort_by(|&i, &j| {
        preds[j][4]
            .partial_cmp(&preds[i][4])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });

    let mut gt_used = vec![false; gts.len()];
    let mut out = Vec::with_capacity(preds.len());
    for &pi in &order {
        let p = preds[pi];
        let pcls = p[5] as u32;
        let pbox = [p[0], p[1], p[2], p[3]];
        let mut best_iou = iou_thr;
        let mut best_gt: Option<usize> = None;
        for (gi, gt) in gts.iter().enumerate() {
            if gt_used[gi] || gt.class != pcls {
                continue;
            }
            let v = iou(pbox, gt.bbox);
            // `>=` so a prediction at exactly the IoU threshold counts as a match
            // (VOC/COCO convention: IoU >= 0.5 is a positive).
            if v >= best_iou {
                best_iou = v;
                best_gt = Some(gi);
            }
        }
        match best_gt {
            Some(gi) => {
                gt_used[gi] = true;
                out.push((pi, Match::Tp(gi)));
            }
            None => out.push((pi, Match::Fp)),
        }
    }
    out
}

/// Precision + recall at one operating point (all given predictions are taken as
/// positive at their confidence). `(precision, recall)`.
pub fn precision_recall(preds: &[Detection], gts: &[GtBox], iou_thr: f32) -> (f32, f32) {
    let matches = match_dets(preds, gts, iou_thr);
    let tp = matches.iter().filter(|(_, m)| matches!(m, Match::Tp(_))).count();
    let n_pred = preds.len();
    let n_gt = gts.len();
    let precision = if n_pred == 0 { 0.0 } else { tp as f32 / n_pred as f32 };
    let recall = if n_gt == 0 { 0.0 } else { tp as f32 / n_gt as f32 };
    (precision, recall)
}

/// Average Precision @ `iou_thr` for a SINGLE class. `preds`/`gts` may contain
/// other classes; only the boxes of `class` participate. Returns `None` if the
/// class has no ground truth (so the caller can exclude it from the mean).
pub fn ap_for_class(preds: &[Detection], gts: &[GtBox], class: u32, iou_thr: f32) -> Option<f32> {
    let cls_preds: Vec<Detection> = preds.iter().copied().filter(|p| p[5] as u32 == class).collect();
    let cls_gts: Vec<GtBox> = gts.iter().copied().filter(|g| g.class == class).collect();
    let n_gt = cls_gts.len();
    if n_gt == 0 {
        return None;
    }
    if cls_preds.is_empty() {
        return Some(0.0);
    }

    // Confidence-descending TP/FP sequence for this class.
    let matches = match_dets(&cls_preds, &cls_gts, iou_thr);
    // match_dets returns in confidence-descending order already.
    let mut tp_cum = 0u32;
    let mut fp_cum = 0u32;
    // precision-recall points as we walk the ranked list.
    let mut recalls = vec![0.0f32];
    let mut precisions = vec![1.0f32]; // (r=0, p=1) anchor for the envelope.
    for (_, m) in &matches {
        match m {
            Match::Tp(_) => tp_cum += 1,
            Match::Fp => fp_cum += 1,
        }
        let r = tp_cum as f32 / n_gt as f32;
        let p = tp_cum as f32 / (tp_cum + fp_cum) as f32;
        recalls.push(r);
        precisions.push(p);
    }

    // Monotonic precision envelope (right-to-left max), then integrate over
    // recall: AP = sum (r_k - r_{k-1}) * p_interp(r_k).
    for k in (0..precisions.len() - 1).rev() {
        precisions[k] = precisions[k].max(precisions[k + 1]);
    }
    let mut ap = 0.0f32;
    for k in 1..recalls.len() {
        ap += (recalls[k] - recalls[k - 1]) * precisions[k];
    }
    Some(ap)
}

/// mean AP @ `iou_thr` over all classes that HAVE ground truth (absent classes
/// excluded). `nc` is the class count (class ids `0..nc`). Returns 0 if no class
/// has ground truth.
pub fn map_at(preds: &[Detection], gts: &[GtBox], nc: u32, iou_thr: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for c in 0..nc {
        if let Some(ap) = ap_for_class(preds, gts, c, iou_thr) {
            sum += ap;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// mAP@0.5: [`map_at`] with `iou_thr = 0.5`.
pub fn map50(preds: &[Detection], gts: &[GtBox], nc: u32) -> f32 {
    map_at(preds, gts, nc, 0.5)
}
