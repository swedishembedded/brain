// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 Task-Aligned Assigner (TAL), plain Rust.
//!
//! For each image the assigner matches anchor points to ground-truth boxes by a
//! task-alignment metric and produces, per anchor, the foreground mask + target
//! box / class / soft score / DFL target distances that the detection loss
//! treats as **constants** (assignment is non-differentiable).
//!
//! ## Algorithm (Ultralytics `TaskAlignedAssigner`)
//! 1. **Candidates**: an anchor is a candidate for a GT only if its pixel center
//!    lies inside that GT box (`point_in_box`).
//! 2. **Alignment metric**: for every (GT, candidate anchor) pair,
//!        `t = s^alpha * u^beta`
//!    where `s` is the predicted class score (sigmoid) for the GT's class and
//!    `u = max(CIoU, 0)` between the anchor's decoded pred box and the GT box.
//!    Defaults `alpha = 0.5`, `beta = 6.0`.
//! 3. **Top-k**: each GT keeps its `topk = 10` highest-`t` candidate anchors as
//!    positives (a binary select mask).
//! 4. **De-conflict**: if an anchor is selected by >1 GT, it is assigned to the
//!    GT with the highest `u` (CIoU) at that anchor — ties broken by the larger
//!    alignment `t`, then the smaller GT index (deterministic).
//! 5. **Soft score**: per Ultralytics, the foreground BCE target is a normalised
//!    alignment. For each GT, `t` over its positive anchors is rescaled so its
//!    max equals that GT's max CIoU:
//!        `norm_t = t / max_t(GT) * max_u(GT)`.
//!    The anchor's `target_score` is the `norm_t` of the GT it ended up assigned
//!    to (placed at the GT's class channel; all other class channels target 0).
//!
//! The assigner is order-invariant in the GT list at the aggregate level: the
//! candidate/top-k/de-conflict rules depend only on per-pair `(t, u)` values and
//! a deterministic tie-break, not on iteration order (verified by a unit test).

use crate::boxmath::{ciou, point_in_box, xyxy_to_dist, Xyxy};

/// Geometry of one anchor point shared across the batch.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Pixel-space center (used for the in-box candidate test).
    pub cx: f32,
    pub cy: f32,
    /// Feature-unit anchor point (used to build the DFL target distances).
    pub ax: f32,
    pub ay: f32,
    /// This anchor's pyramid stride (pixels per feature cell).
    pub stride: f32,
}

/// A ground-truth box for one image: pixel `xyxy` + class id.
#[derive(Clone, Copy, Debug)]
pub struct Gt {
    pub cls: usize,
    pub box_: Xyxy,
}

/// Per-anchor assignment result (a constant in the backward graph).
#[derive(Clone, Copy, Debug)]
pub struct AnchorTarget {
    pub fg: bool,
    pub target_box: Xyxy,
    pub target_cls: usize,
    /// Normalised soft alignment score in [0,1] for the matched class.
    pub target_score: f32,
    /// DFL target distances `(l,t,r,b)` in feature units for this anchor.
    pub target_dist: [f32; 4],
}

impl Default for AnchorTarget {
    fn default() -> Self {
        AnchorTarget {
            fg: false,
            target_box: [0.0; 4],
            target_cls: 0,
            target_score: 0.0,
            target_dist: [0.0; 4],
        }
    }
}

/// Tunable knobs of the Task-Aligned Assigner.
#[derive(Clone, Copy, Debug)]
pub struct TalParams {
    pub topk: usize,
    pub alpha: f32,
    pub beta: f32,
    pub eps: f32,
}

impl Default for TalParams {
    fn default() -> Self {
        TalParams { topk: 10, alpha: 0.5, beta: 6.0, eps: 1e-9 }
    }
}

/// Run the Task-Aligned Assigner for ONE image.
///
/// * `anchors` — shared anchor geometry, length `A`.
/// * `pred_boxes` — decoded predicted pixel `xyxy` per anchor, length `A`.
/// * `pred_scores` — sigmoid class scores per anchor, row-major `[A, nc]`.
/// * `gts` — the image's ground-truth boxes.
///
/// Returns one [`AnchorTarget`] per anchor.
pub fn assign(
    anchors: &[Anchor],
    pred_boxes: &[Xyxy],
    pred_scores: &[f32],
    nc: usize,
    gts: &[Gt],
    p: TalParams,
) -> Vec<AnchorTarget> {
    let a = anchors.len();
    let mut out = vec![AnchorTarget::default(); a];
    if gts.is_empty() {
        return out; // empty image: every anchor is background.
    }

    // metric[g][i] = t,  overlap[g][i] = u, only for in-box candidates (else 0).
    let ng = gts.len();
    let mut metric = vec![vec![0.0f32; a]; ng];
    let mut overlap = vec![vec![0.0f32; a]; ng];
    let mut candidate = vec![vec![false; a]; ng];

    for (g, gt) in gts.iter().enumerate() {
        for i in 0..a {
            let an = anchors[i];
            if !point_in_box(an.cx, an.cy, gt.box_) {
                continue;
            }
            candidate[g][i] = true;
            let u = ciou(pred_boxes[i], gt.box_).max(0.0);
            let s = pred_scores[i * nc + gt.cls].clamp(0.0, 1.0);
            let t = s.powf(p.alpha) * u.powf(p.beta);
            metric[g][i] = t;
            overlap[g][i] = u;
        }
    }

    // Top-k positives per GT (binary mask over its candidates).
    let mut pos = vec![vec![false; a]; ng];
    for g in 0..ng {
        // Indices of candidates sorted by descending metric; deterministic
        // tie-break on anchor index keeps the selection order-independent.
        let mut idx: Vec<usize> = (0..a).filter(|&i| candidate[g][i]).collect();
        idx.sort_by(|&i, &j| {
            metric[g][j]
                .partial_cmp(&metric[g][i])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        });
        for &i in idx.iter().take(p.topk) {
            // a GT with all-zero metric still keeps its in-box candidates as
            // positives (matches Ultralytics, which top-k's the metric incl. 0).
            pos[g][i] = true;
        }
    }

    // De-conflict: each anchor goes to at most one GT. Pick the GT with the
    // highest CIoU `u`; tie-break on higher metric `t`, then lower GT index.
    // `assigned_gt[i] = Some(g)` once resolved.
    let mut assigned_gt = vec![None; a];
    for i in 0..a {
        let mut best: Option<usize> = None;
        for g in 0..ng {
            if !pos[g][i] {
                continue;
            }
            best = Some(match best {
                None => g,
                Some(bg) => {
                    let (ug, tg) = (overlap[g][i], metric[g][i]);
                    let (ub, tb) = (overlap[bg][i], metric[bg][i]);
                    if ug > ub || (ug == ub && tg > tb) {
                        g
                    } else {
                        bg
                    }
                }
            });
        }
        assigned_gt[i] = best;
    }

    // Per-GT normalisation factors for the soft target score:
    //   norm_t(i) = metric[g][i] / max_i metric[g] * max_i overlap[g].
    // Only positives that survived de-conflict contribute to a GT's maxima
    // (Ultralytics computes the maxima over the final fg mask), so gather them.
    let mut max_metric = vec![0.0f32; ng];
    let mut max_overlap = vec![0.0f32; ng];
    for i in 0..a {
        if let Some(g) = assigned_gt[i] {
            max_metric[g] = max_metric[g].max(metric[g][i]);
            max_overlap[g] = max_overlap[g].max(overlap[g][i]);
        }
    }

    for i in 0..a {
        let Some(g) = assigned_gt[i] else { continue };
        let an = anchors[i];
        let gt = gts[g];
        let denom = max_metric[g].max(p.eps);
        let score = metric[g][i] / denom * max_overlap[g];
        out[i] = AnchorTarget {
            fg: true,
            target_box: gt.box_,
            target_cls: gt.cls,
            target_score: score,
            target_dist: xyxy_to_dist(gt.box_, an.ax, an.ay, an.stride),
        };
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small synthetic anchor grid (one scale) covering a `side`-pixel image.
    fn grid(side: u32, stride: u32) -> Vec<Anchor> {
        let n = side / stride;
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let (ax, ay) = (j as f32 + 0.5, i as f32 + 0.5);
                v.push(Anchor {
                    cx: ax * stride as f32,
                    cy: ay * stride as f32,
                    ax,
                    ay,
                    stride: stride as f32,
                });
            }
        }
        v
    }

    // Decoded pred boxes: every anchor predicts a small box centered on itself.
    fn self_boxes(anchors: &[Anchor], half: f32) -> Vec<Xyxy> {
        anchors
            .iter()
            .map(|a| [a.cx - half, a.cy - half, a.cx + half, a.cy + half])
            .collect()
    }

    #[test]
    fn single_centered_object() {
        let anchors = grid(128, 8); // 16x16 = 256 anchors
        let boxes = self_boxes(&anchors, 30.0);
        let nc = 2;
        // Give class 1 a high score everywhere so s is non-degenerate.
        let scores: Vec<f32> = (0..anchors.len()).flat_map(|_| vec![0.1f32, 0.9]).collect();
        // GT: centered 64x64 box of class 1.
        let gt = Gt { cls: 1, box_: [32.0, 32.0, 96.0, 96.0] };
        let res = assign(&anchors, &boxes, &scores, nc, &[gt], TalParams::default());

        let fg: Vec<usize> = (0..res.len()).filter(|&i| res[i].fg).collect();
        assert!(!fg.is_empty(), "centered object should produce >=1 positive");
        for &i in &fg {
            assert_eq!(res[i].target_cls, 1, "fg anchor class");
            assert_eq!(res[i].target_box, gt.box_, "fg anchor target box == GT");
            assert!(res[i].target_score.is_finite() && res[i].target_score >= 0.0);
            // The GT center is inside the box, so anchors near it must be fg.
            assert!(point_in_box(anchors[i].cx, anchors[i].cy, gt.box_));
        }
    }

    #[test]
    fn no_object_image() {
        let anchors = grid(128, 8);
        let boxes = self_boxes(&anchors, 30.0);
        let scores = vec![0.5f32; anchors.len() * 2];
        let res = assign(&anchors, &boxes, &scores, 2, &[], TalParams::default());
        assert!(res.iter().all(|t| !t.fg), "no GT -> zero foreground");
        assert!(res.iter().all(|t| t.target_score.is_finite()));
    }

    #[test]
    fn two_separated_objects() {
        let anchors = grid(128, 8);
        let boxes = self_boxes(&anchors, 20.0);
        let nc = 2;
        let scores = vec![0.8f32; anchors.len() * nc];
        let gts = [
            Gt { cls: 0, box_: [8.0, 8.0, 40.0, 40.0] },     // top-left
            Gt { cls: 1, box_: [88.0, 88.0, 120.0, 120.0] }, // bottom-right
        ];
        let res = assign(&anchors, &boxes, &scores, nc, &gts, TalParams::default());
        let c0 = res.iter().filter(|t| t.fg && t.target_cls == 0).count();
        let c1 = res.iter().filter(|t| t.fg && t.target_cls == 1).count();
        assert!(c0 > 0, "object 0 got positives");
        assert!(c1 > 0, "object 1 got positives");
    }

    #[test]
    fn target_order_permutation_invariant() {
        let anchors = grid(128, 8);
        let boxes = self_boxes(&anchors, 20.0);
        let nc = 3;
        let scores: Vec<f32> = (0..anchors.len())
            .flat_map(|i| {
                let f = (i % 7) as f32 / 7.0;
                vec![0.2 + 0.5 * f, 0.7 - 0.3 * f, 0.4]
            })
            .collect();
        let gts = vec![
            Gt { cls: 0, box_: [8.0, 8.0, 48.0, 48.0] },
            Gt { cls: 1, box_: [80.0, 80.0, 120.0, 120.0] },
            Gt { cls: 2, box_: [8.0, 80.0, 48.0, 120.0] },
        ];
        let r1 = assign(&anchors, &boxes, &scores, nc, &gts, TalParams::default());

        // Permute the GT order; aggregate (per-anchor class+box+score) must match.
        let perm = vec![gts[2], gts[0], gts[1]];
        let r2 = assign(&anchors, &boxes, &scores, nc, &perm, TalParams::default());

        for i in 0..anchors.len() {
            assert_eq!(r1[i].fg, r2[i].fg, "fg mismatch at {i}");
            if r1[i].fg {
                assert_eq!(r1[i].target_cls, r2[i].target_cls, "cls mismatch at {i}");
                assert_eq!(r1[i].target_box, r2[i].target_box, "box mismatch at {i}");
                assert!(
                    (r1[i].target_score - r2[i].target_score).abs() < 1e-5,
                    "score mismatch at {i}"
                );
            }
        }
    }
}
