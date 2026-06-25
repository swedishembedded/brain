// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 single-image / batch inference (P6): letterbox -> eval-mode forward ->
//! DFL decode -> boxes -> per-class scores -> NMS -> un-letterbox. CPU backend.
//!
//! ## The `detect` API (input form)
//! [`Yolo::detect`] takes ONE image as **interleaved-RGB HWC** floats
//! (`src[h0*w0*3]`, channel-last, already normalised the way the model was
//! trained — typically `pixel/255`) plus its original `(w0, h0)`. It returns the
//! surviving detections as `Vec<[f32;6]>` = `[x1,y1,x2,y2,conf,class]` in
//! **original-image pixel coords**. HWC-RGB is chosen because that is the natural
//! decoded-image layout; [`boxmath::letterbox_rgb`] does the resize+pad+CHW
//! transpose into the model's `[3,size,size]` input. [`Yolo::detect_batch`] runs
//! several such images (each letterboxed independently) through one forward.
//!
//! The runtime's later `DetectModel` adapter calls exactly this `detect`.
//!
//! ## Pipeline (per image)
//! 1. [`boxmath::letterbox_rgb`]: aspect-preserving resize + centre-pad to the
//!    square model input; record the [`boxmath::Letterbox`] transform.
//! 2. forward in **eval-mode BN** (we flip [`Yolo::set_eval(true)`] so BN uses
//!    running stats, not the current batch's stats).
//! 3. `dfl_decode` over the raw box logits -> per-side `ltrb` distances, then
//!    [`boxmath::dist_to_xyxy`] (anchor point + stride) -> boxes in INPUT coords.
//! 4. `sigmoid` the cls logits -> per-class scores; per anchor take the argmax
//!    class + its score; drop anchors whose best score `< conf_thresh`.
//! 5. class-aware NMS ([`crate::nms::nms`]).
//! 6. [`boxmath::Letterbox::invert_box`] -> original-image coords.

use crate::boxmath::{self, dist_to_xyxy, Letterbox};
use crate::net::DFL_DECODE;
use crate::nms::{nms, Detection};
use crate::Yolo;

/// `sigmoid`, numerically stable.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Run `dfl_decode` over flat box logits `[na, 4*reg_max]` -> per-side distances
/// flat `[na, 4]` (feature units). `na = N*A`. Mirrors the loss-module decode
/// layout: each (image,anchor) row is `4*reg_max` contiguous values =
/// side-major then bin, i.e. `[4][reg_max]`, exactly the kernel's `[A,4,reg_max]`.
pub fn dfl_decode_dist(gpu: &gpu_core::Gpu, box_logits: &[f32], na: usize, reg_max: usize) -> Vec<f32> {
    let lb = gpu.storage_init("dfl_logits_infer", box_logits);
    let db = gpu.storage((na * 4) as u64);
    let s = gpu.step(DFL_DECODE, &[&lb, &db], &[na as u32, reg_max as u32], (na * 4) as u32);
    gpu.submit(&[], &[s]);
    gpu.read(&db, na * 4)
}

impl Yolo {
    /// Detect objects in ONE interleaved-RGB HWC image (`src[h0*w0*3]`) of size
    /// `w0 x h0`. Returns `[x1,y1,x2,y2,conf,class]` in original-image coords.
    /// See the module docs for the input/output contract.
    pub fn detect(
        &self,
        src: &[f32],
        w0: u32,
        h0: u32,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Vec<Detection> {
        self.detect_batch(&[(src, w0, h0)], conf_thresh, iou_thresh).pop().unwrap_or_default()
    }

    /// Batch inference: each `(src, w0, h0)` is letterboxed independently, run
    /// through one forward, decoded and NMS'd. Returns one `Vec<Detection>` per
    /// input image (original-image coords). The batch size MUST equal the model's
    /// configured batch `b`.
    pub fn detect_batch(
        &self,
        images: &[(&[f32], u32, u32)],
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Vec<Vec<Detection>> {
        let size = self.cfg.input;
        let n = images.len();
        assert_eq!(n, self.batch() as usize, "detect_batch expects exactly model batch images");
        let prof = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let t_start = std::time::Instant::now();

        // 1. letterbox each image into the model's CHW input; stack to [N,3,H,W].
        let chw_len = (3 * size * size) as usize;
        let mut input = vec![0.0f32; n * chw_len];
        let mut lbs: Vec<Letterbox> = Vec::with_capacity(n);
        for (i, &(src, w0, h0)) in images.iter().enumerate() {
            // Grey pad (114/255) is the Ultralytics convention; with normalised
            // [0,1] inputs that is 0.447. Matches a /255 preprocessing.
            let (chw, lb) = boxmath::letterbox_rgb(src, w0, h0, size, 114.0 / 255.0);
            input[i * chw_len..(i + 1) * chw_len].copy_from_slice(&chw);
            lbs.push(lb);
        }

        let t_pre = std::time::Instant::now();

        // 2. eval-mode BN forward.
        let was_eval = self.is_eval();
        self.set_eval(true);
        self.set_image(&input);
        self.forward_net_pub();
        if !was_eval {
            self.set_eval(false);
        }
        let t_fwd = std::time::Instant::now();

        // 3-6. decode + score + NMS + un-letterbox, per image.
        let (cls, boxl) = self.raw_logits();
        let a = self.num_anchors() as usize;
        let nc = self.cfg.nc as usize;
        let reg_max = self.cfg.reg_max as usize;
        let anchors = self.anchor_geometry();

        let dist = dfl_decode_dist(&self.gpu, &boxl, n * a, reg_max);

        let mut out: Vec<Vec<Detection>> = Vec::with_capacity(n);
        for img in 0..n {
            let mut dets: Vec<Detection> = Vec::new();
            for i in 0..a {
                // Best class for this anchor. SiLU/sigmoid is monotonic, so the
                // argmax over the raw logits is the argmax over the scores —
                // compute sigmoid only once (for the winner) instead of nc times
                // per anchor (this loop runs over every anchor × class).
                let cbase = (img * a + i) * nc;
                let mut best_c = 0usize;
                let mut best_l = cls[cbase];
                for c in 1..nc {
                    let l = cls[cbase + c];
                    if l > best_l {
                        best_l = l;
                        best_c = c;
                    }
                }
                let best_s = sigmoid(best_l);
                if best_s < conf_thresh {
                    continue;
                }
                let an = anchors[i];
                let dbase = (img * a + i) * 4;
                let d = [dist[dbase], dist[dbase + 1], dist[dbase + 2], dist[dbase + 3]];
                let bx = dist_to_xyxy(d, an.ax, an.ay, an.stride); // input coords
                dets.push([bx[0], bx[1], bx[2], bx[3], best_s, best_c as f32]);
            }
            // class-aware NMS in input coords.
            let kept = nms(&dets, iou_thresh, self.max_det());
            // un-letterbox to original coords.
            let (_, w0, h0) = images[img];
            let lb = lbs[img];
            let mapped: Vec<Detection> = kept
                .into_iter()
                .map(|d| {
                    let b = lb.invert_box([d[0], d[1], d[2], d[3]], w0, h0);
                    [b[0], b[1], b[2], b[3], d[4], d[5]]
                })
                .collect();
            out.push(mapped);
        }
        if prof {
            let t_end = std::time::Instant::now();
            eprintln!(
                "[detect] preprocess {:.1} ms | forward {:.1} ms | postprocess {:.1} ms | total {:.1} ms",
                (t_pre - t_start).as_secs_f64() * 1e3,
                (t_fwd - t_pre).as_secs_f64() * 1e3,
                (t_end - t_fwd).as_secs_f64() * 1e3,
                (t_end - t_start).as_secs_f64() * 1e3,
            );
        }
        out
    }

    /// Max detections kept per image after NMS (Ultralytics default 300).
    fn max_det(&self) -> usize {
        300
    }
}
