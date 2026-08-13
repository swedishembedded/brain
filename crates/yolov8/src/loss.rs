// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 detection loss orchestration (P4): assigner + BCE + CIoU + DFL, on the
//! CPU backend. This module is the bridge between the network's raw cls/box
//! logits and the per-head-branch grad buffers the reverse Step chain consumes.
//!
//! ## Forward
//! 1. Repack the network's flat box logits `[N,A,4*reg_max]` into the
//!    `dfl_decode` layout `[(N*A)*4, reg_max]` and run `dfl_decode` -> per-side
//!    DFL **distances** (feature units).
//! 2. `dist_to_xyxy` (anchor point + stride) -> decoded pixel `xyxy` boxes.
//! 3. `sigmoid` the cls logits -> per-anchor class scores.
//! 4. Run the Task-Aligned Assigner ([`crate::assign`]) per image -> per-anchor
//!    `{fg, target_box, target_cls, target_score, target_dist}` (CONSTANTS).
//! 5. Run the loss kernels:
//!      * `bce_logits` over ALL anchors x classes vs the soft target (the
//!        matched class gets `target_score`, every other class 0).
//!      * `ciou` over the fg anchors (decoded box vs target box).
//!      * `dfl_loss` over the fg anchors (target_dist).
//!
//!    `L = box_gain*Σciou + cls_gain*Σbce + dfl_gain*Σdfl`, with all three terms
//!    normalised by `max(Σ target_score, 1)` (Ultralytics' "target-score-sum"
//!    normaliser — the BCE is the full grid so its sum is already extensive,
//!    while box/dfl are summed over fg only; the shared denominator keeps the
//!    three gradients commensurate and the loss scale batch-size invariant).
//!
//! ## Backward
//! `dL/d(box_logits)` and `dL/d(cls_logits)` are produced flat `[N,A,·]`:
//!   * `bce_logits_grad` -> d(cls_logits) (× cls_gain / norm).
//!   * `ciou_grad` -> d(decoded box) (× box_gain / norm); chain box -> dist:
//!     x1=(ax-l)s,y1=(ay-t)s,x2=(ax+r)s,y2=(ay+b)s  =>
//!     dl = -s·dx1, dt = -s·dy1, dr = s·dx2, db = s·dy2.
//!     Feed dE=(dl,dt,dr,db) to `dfl_grad` -> d(box_logits) (box-branch path).
//!   * `dfl_loss_grad` -> d(box_logits) directly (× dfl_gain / norm), ADDED.
//!
//! Both grad tensors come back flat and are scattered into the per-branch NCHW
//! head grad buffers by the model.

use gpu_core::Gpu;

use crate::assign::{assign, Anchor, AnchorTarget, Gt, TalParams};
use crate::boxmath::{dist_to_xyxy, xywhn_to_xyxy, Xyxy};
use crate::model::GtBox;
use crate::net::{BCE_LOGITS, BCE_LOGITS_GRAD, CIOU, CIOU_GRAD, DFL_DECODE, DFL_GRAD, DFL_LOSS, DFL_LOSS_GRAD};

/// Loss gains (Ultralytics defaults).
#[derive(Clone, Copy, Debug)]
pub struct Gains {
    pub box_: f32,
    pub cls: f32,
    pub dfl: f32,
}
impl Default for Gains {
    fn default() -> Self {
        Gains { box_: 7.5, cls: 0.5, dfl: 1.5 }
    }
}

/// The frozen, per-batch assignment + decoded geometry. Computed once from a
/// forward pass and reused for every gradcheck weight perturbation so the finite
/// difference never sees the assigner's (non-differentiable) discontinuities.
#[derive(Clone, Debug)]
pub struct Assignment {
    /// Per (image, anchor) target, row-major `[N][A]`.
    pub targets: Vec<Vec<AnchorTarget>>,
}

/// Inputs the loss needs from the model each forward.
pub struct LossInput<'a> {
    pub gpu: &'a Gpu,
    pub n: usize,
    pub a: usize,
    pub nc: usize,
    pub reg_max: usize,
    pub anchors: &'a [Anchor],
    /// Raw cls logits flat `[N, A, nc]`.
    pub cls_logits: &'a [f32],
    /// Raw box logits flat `[N, A, 4*reg_max]`.
    pub box_logits: &'a [f32],
    pub gains: Gains,
}

/// Output of a forward+backward loss evaluation.
pub struct LossOutput {
    pub loss: f32,
    /// dL/d(cls_logits), flat `[N, A, nc]`.
    pub d_cls: Vec<f32>,
    /// dL/d(box_logits), flat `[N, A, 4*reg_max]`.
    pub d_box: Vec<f32>,
}

/// `sigmoid` in a numerically stable form.
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Run `dfl_decode` over flat box logits `[N,A,4*reg_max]` -> per-side distances
/// flat `[N,A,4]`. The kernel layout is `[(anchor)*4+side]*reg_max + bin`, which
/// (treating each (image,anchor) as one decode anchor) matches the flat box
/// logit layout exactly: row `(n*A+i)` is `4*reg_max` contiguous values =
/// side-major then bin, i.e. `[4][reg_max]`.
fn decode_dist(gpu: &Gpu, box_logits: &[f32], na: usize, reg_max: usize) -> Vec<f32> {
    let lb = gpu.storage_init("dfl_logits", box_logits);
    let db = gpu.storage((na * 4) as u64);
    let s = gpu.step(DFL_DECODE, &[&lb, &db], &[na as u32, reg_max as u32], (na * 4) as u32);
    gpu.submit(&[], &[s]);
    gpu.read(&db, na * 4)
}

/// Compute the frozen assignment from the current logits (one forward pass).
pub fn compute_assignment(inp: &LossInput, gts: &[GtBox], img_size: f32) -> Assignment {
    let na = inp.n * inp.a;
    let dist = decode_dist(inp.gpu, inp.box_logits, na, inp.reg_max);

    let mut targets = Vec::with_capacity(inp.n);
    for img in 0..inp.n {
        // decoded pred boxes + sigmoid scores for this image.
        let mut pred_boxes = vec![[0.0f32; 4]; inp.a];
        let mut scores = vec![0.0f32; inp.a * inp.nc];
        for i in 0..inp.a {
            let an = inp.anchors[i];
            let base = (img * inp.a + i) * 4;
            let d = [dist[base], dist[base + 1], dist[base + 2], dist[base + 3]];
            pred_boxes[i] = dist_to_xyxy(d, an.ax, an.ay, an.stride);
            for c in 0..inp.nc {
                scores[i * inp.nc + c] = sigmoid(inp.cls_logits[(img * inp.a + i) * inp.nc + c]);
            }
        }
        // GT boxes for this image -> pixel xyxy.
        let img_gts: Vec<Gt> = gts
            .iter()
            .filter(|g| g.img as usize == img)
            .map(|g| Gt {
                cls: g.cls as usize,
                box_: xywhn_to_xyxy(g.cx, g.cy, g.w, g.h, img_size),
            })
            .collect();
        let res = assign(inp.anchors, &pred_boxes, &scores, inp.nc, &img_gts, TalParams::default());
        targets.push(res);
    }
    Assignment { targets }
}

/// Forward + backward of the detection loss against a FROZEN assignment.
pub fn eval(inp: &LossInput, asg: &Assignment) -> LossOutput {
    let (n, a, nc, reg_max) = (inp.n, inp.a, inp.nc, inp.reg_max);
    let na = n * a;
    let g = inp.gains;

    // ---- decode distances + boxes for the CURRENT logits ----
    let dist = decode_dist(inp.gpu, inp.box_logits, na, reg_max);
    let mut pred_boxes = vec![[0.0f32; 4]; na];
    for img in 0..n {
        for i in 0..a {
            let an = inp.anchors[i];
            let base = (img * a + i) * 4;
            let d = [dist[base], dist[base + 1], dist[base + 2], dist[base + 3]];
            pred_boxes[img * a + i] = dist_to_xyxy(d, an.ax, an.ay, an.stride);
        }
    }

    // ---- build per-(image,anchor) target tensors from the frozen assignment ----
    // BCE soft target over the full grid [N,A,nc].
    let mut bce_tgt = vec![0.0f32; na * nc];
    // fg bookkeeping for CIoU / DFL (only fg anchors contribute those terms).
    let mut fg_idx: Vec<usize> = Vec::new(); // flat (img*a+i) of each fg anchor
    let mut fg_target_box: Vec<Xyxy> = Vec::new();
    let mut fg_target_dist: Vec<[f32; 4]> = Vec::new();
    let mut score_sum = 0.0f32;
    for img in 0..n {
        for i in 0..a {
            let t = asg.targets[img][i];
            // BCE target: matched class gets target_score (already counts the
            // soft alignment), all else 0. Background anchors -> all-zero row.
            if t.fg {
                bce_tgt[(img * a + i) * nc + t.target_cls] = t.target_score;
                score_sum += t.target_score;
                fg_idx.push(img * a + i);
                fg_target_box.push(t.target_box);
                fg_target_dist.push(t.target_dist);
            }
        }
    }
    let nfg = fg_idx.len();
    // Ultralytics normaliser: sum of assigned target scores, floored at 1.
    let norm = score_sum.max(1.0);

    // ---- BCE over the full grid ----
    let total = (na * nc) as u32;
    let clsb = inp.gpu.storage_init("bce_logits", inp.cls_logits);
    let tgtb = inp.gpu.storage_init("bce_tgt", &bce_tgt);
    let bce_out = inp.gpu.storage(total as u64);
    let s = inp.gpu.step(BCE_LOGITS, &[&clsb, &tgtb, &bce_out], &[total], total);
    inp.gpu.submit(&[], &[s]);
    let bce_sum: f32 = inp.gpu.read(&bce_out, na * nc).iter().sum();

    // BCE grad (full grid) -> scaled later.
    let dcls_buf = inp.gpu.storage(total as u64);
    let s = inp.gpu.step(BCE_LOGITS_GRAD, &[&clsb, &tgtb, &dcls_buf], &[total], total);
    inp.gpu.submit(&[], &[s]);
    let mut d_cls = inp.gpu.read(&dcls_buf, na * nc);
    let cls_scale = g.cls / norm;
    for v in d_cls.iter_mut() {
        *v *= cls_scale;
    }

    // ---- CIoU + DFL over fg anchors only ----
    let mut d_box = vec![0.0f32; na * 4 * reg_max];
    let mut ciou_sum = 0.0f32;
    let mut dfl_sum = 0.0f32;
    if nfg > 0 {
        // Gather fg decoded boxes + targets into compact [nfg,4] tensors.
        let mut pred_flat = vec![0.0f32; nfg * 4];
        let mut tgt_flat = vec![0.0f32; nfg * 4];
        for (k, &fi) in fg_idx.iter().enumerate() {
            pred_flat[k * 4..k * 4 + 4].copy_from_slice(&pred_boxes[fi]);
            tgt_flat[k * 4..k * 4 + 4].copy_from_slice(&fg_target_box[k]);
        }
        // CIoU value.
        let pb = inp.gpu.storage_init("ciou_pred", &pred_flat);
        let tb = inp.gpu.storage_init("ciou_tgt", &tgt_flat);
        let ob = inp.gpu.storage(nfg as u64);
        let s = inp.gpu.step(CIOU, &[&pb, &tb, &ob], &[nfg as u32], nfg as u32);
        inp.gpu.submit(&[], &[s]);
        ciou_sum = inp.gpu.read(&ob, nfg).iter().sum();
        // CIoU grad wrt the decoded boxes.
        let dpb = inp.gpu.storage(nfg as u64 * 4);
        let s = inp.gpu.step(CIOU_GRAD, &[&pb, &tb, &dpb], &[nfg as u32], nfg as u32);
        inp.gpu.submit(&[], &[s]);
        let dbox = inp.gpu.read(&dpb, nfg * 4);

        // Chain d(box) -> d(dist) and assemble the dE buffer for dfl_grad over
        // the fg anchors: dE = (dl,dt,dr,db) = (-s·dx1,-s·dy1, s·dx2, s·dy2),
        // already scaled by box_gain/norm.
        let box_scale = g.box_ / norm;
        let mut de = vec![0.0f32; nfg * 4];
        // Per-fg box logits (the reg_max bins) packed for dfl_grad.
        let mut fg_box_logits = vec![0.0f32; nfg * 4 * reg_max];
        for (k, &fi) in fg_idx.iter().enumerate() {
            let an = inp.anchors[fi % a];
            let s_ = an.stride;
            de[k * 4] = -s_ * dbox[k * 4] * box_scale; // dl
            de[k * 4 + 1] = -s_ * dbox[k * 4 + 1] * box_scale; // dt
            de[k * 4 + 2] = s_ * dbox[k * 4 + 2] * box_scale; // dr
            de[k * 4 + 3] = s_ * dbox[k * 4 + 3] * box_scale; // db
            let src = fi * 4 * reg_max;
            fg_box_logits[k * 4 * reg_max..(k + 1) * 4 * reg_max]
                .copy_from_slice(&inp.box_logits[src..src + 4 * reg_max]);
        }
        // dfl_grad: dE[nfg*4] -> dlogit[nfg*4*reg_max].
        let flb = inp.gpu.storage_init("dfl_fg_logits", &fg_box_logits);
        let deb = inp.gpu.storage_init("dfl_de", &de);
        let dlb = inp.gpu.storage((nfg * 4 * reg_max) as u64);
        let s = inp.gpu.step(DFL_GRAD, &[&flb, &deb, &dlb], &[nfg as u32, reg_max as u32], (nfg * 4) as u32);
        inp.gpu.submit(&[], &[s]);
        let dfl_box_grad = inp.gpu.read(&dlb, nfg * 4 * reg_max);

        // DFL loss value + grad over fg (target_dist).
        let mut tdist = vec![0.0f32; nfg * 4];
        for (k, d) in fg_target_dist.iter().enumerate() {
            tdist[k * 4..k * 4 + 4].copy_from_slice(d);
        }
        let tdb = inp.gpu.storage_init("dfl_tdist", &tdist);
        let dlo = inp.gpu.storage(nfg as u64);
        let s = inp.gpu.step(DFL_LOSS, &[&flb, &tdb, &dlo], &[nfg as u32, reg_max as u32], nfg as u32);
        inp.gpu.submit(&[], &[s]);
        dfl_sum = inp.gpu.read(&dlo, nfg).iter().sum();

        let dflg = inp.gpu.storage((nfg * 4 * reg_max) as u64);
        let s = inp.gpu.step(DFL_LOSS_GRAD, &[&flb, &tdb, &dflg], &[nfg as u32, reg_max as u32], (nfg * 4) as u32);
        inp.gpu.submit(&[], &[s]);
        let dfl_loss_grad = inp.gpu.read(&dflg, nfg * 4 * reg_max);

        // Scatter the two box-logit grad contributions into the full tensor.
        let dfl_scale = g.dfl / norm;
        for (k, &fi) in fg_idx.iter().enumerate() {
            let dst = fi * 4 * reg_max;
            for j in 0..4 * reg_max {
                d_box[dst + j] += dfl_box_grad[k * 4 * reg_max + j]
                    + dfl_loss_grad[k * 4 * reg_max + j] * dfl_scale;
            }
        }
    }

    let loss = g.box_ * ciou_sum / norm + g.cls * bce_sum / norm + g.dfl * dfl_sum / norm;
    LossOutput { loss, d_cls, d_box }
}
