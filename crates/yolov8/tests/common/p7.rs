// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scaffolding for the P7 detection capability + negative-control tests.
//!
//! Included (via `#[path]`) into `p7_capability.rs` and `p7_negative.rs` rather
//! than published as a crate module, because it is pure test glue: build a tiny
//! `Yolo`, train it on a fixed synthetic batch, then score a held-out split with
//! the golden `eval::detection::map50`. Everything is CPU-backend, deterministic
//! for a fixed seed, and kept to minute-scale smoke training — NOT a full
//! experiment.

#![allow(dead_code)]

use std::collections::HashMap;

use data::binio::DetectBox;
use data::gen_detect::{self, Preset};
use data::rng::Rng;
use eval::detection::GtBox as EvalGt;
use yolov8::boxmath::xywhn_to_xyxy;
use yolov8::{init_weights, Detection, GtBox, LossMode, Yolo, YoloConfig};

/// Skip unless training is explicitly requested. EMPTY `MOE_SKIP_GPU_TESTS=` does
/// NOT skip (the documented way to force training, matching `p5_overfit`); only a
/// non-empty value skips.
pub fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

/// One generated scene paired with its CHW image blob.
pub struct Sample {
    pub chw: Vec<f32>,        // [3*side*side], normalized [0,1]
    pub boxes: Vec<DetectBox>, // normalized GT for THIS image
}

/// Generate `n` independent scenes of `preset` at `side x side` with `nc` classes.
pub fn gen_split(preset: Preset, n: usize, side: u32, nc: u32, seed: u64) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let s = gen_detect::gen_scene(preset, side, side, nc, &mut rng);
            Sample { chw: gen_detect::image_to_chw(&s.image), boxes: s.boxes }
        })
        .collect()
}

/// Generate `n` GUARANTEED-empty scenes: re-roll the `Background` preset until
/// each draws zero shapes (its empty case is only ~30% of samples, so for a
/// deterministic >=50%-empty val split we synthesize empties directly).
pub fn gen_empty_scenes(n: usize, side: u32, nc: u32, seed: u64) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let s = gen_detect::gen_scene(Preset::Background, side, side, nc, &mut rng);
        if s.boxes.is_empty() {
            out.push(Sample { chw: gen_detect::image_to_chw(&s.image), boxes: Vec::new() });
        }
    }
    out
}

/// Stack a split's CHW blobs into one `[N*3*side*side]` batch buffer.
pub fn stack_images(split: &[Sample]) -> Vec<f32> {
    let mut out = Vec::new();
    for s in split {
        out.extend_from_slice(&s.chw);
    }
    out
}

/// Build the per-image `GtBox` list (with `img` index) for a split, in image
/// order — the form `set_targets` wants. `relabel` lets a caller remap classes
/// (the shuffled-label control); pass identity otherwise.
pub fn targets_for_split(split: &[Sample], relabel: impl Fn(u32) -> u32) -> Vec<GtBox> {
    let mut gts = Vec::new();
    for (img, s) in split.iter().enumerate() {
        for b in &s.boxes {
            gts.push(GtBox {
                img: img as u32,
                cls: relabel(b.class),
                cx: b.cx,
                cy: b.cy,
                w: b.w,
                h: b.h,
            });
        }
    }
    gts
}


/// A TEST-ONLY shrunk config: `YoloConfig::tiny(nc)` but at a smaller square input
/// resolution. The conv stack dominates CPU-JIT cost and scales ~quadratically
/// with the input side, so dropping 128->64 makes every forward/backward ~4x
/// cheaper — turning a multi-minute training into a ~minute one — while the
/// synthetic shapes (>=12% of the side, so >=~8px at 64) stay clearly resolvable.
/// This changes ONLY the test's config object, never the library default.
pub fn tiny_cfg(nc: u32, input: u32) -> YoloConfig {
    let mut cfg = YoloConfig::tiny(nc);
    cfg.input = input;
    cfg
}

/// The capability/control side length used everywhere below (kept in one place so
/// splits, eval and thresholds stay consistent). 64px is the smoke-budget choice.
pub const SIDE: u32 = 64;

/// Train a fresh model (config `cfg`) on `train` for `steps` AdamW steps (fixed
/// full-batch, the `p5_overfit` idiom), returning the trained model. `lr`/`seed`
/// are explicit so the controls can hold them fixed across conditions.
pub fn train_model(
    cfg: &YoloConfig,
    train: &[Sample],
    targets: &[GtBox],
    steps: u32,
    lr: f32,
    seed: u64,
) -> (Yolo, YoloConfig) {
    let cfg = cfg.clone();
    let b = train.len() as u32;
    let init = init_weights(&cfg, seed);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&stack_images(train));
    model.set_targets(targets);

    let initial = model.forward();
    let mut last = initial;
    for step in 1..=steps {
        model.zero_grads();
        let l = model.forward();
        model.backward();
        model.adamw_step(step, lr, 0.0, Some(10.0), 1.0);
        last = l;
    }
    model.poll_wait();
    println!("  train: {b} imgs, {steps} steps, loss {initial:.3} -> {last:.3}");
    (model, cfg)
}

/// Build an EVAL-mode model of batch `b` holding `model`'s trained weights, by
/// copying every param across. (A fresh model so the eval batch can differ from
/// the train batch without re-running the optimiser.)
pub fn clone_for_eval(model: &Yolo, cfg: &YoloConfig, b: u32) -> Yolo {
    let names: Vec<String> = cfg.full_param_list().iter().map(|(n, _)| n.clone()).collect();
    let weights: HashMap<String, Vec<f32>> =
        names.iter().map(|n| (n.clone(), model.read_weight(n))).collect();
    let evalm = Yolo::new(cfg.clone(), b, cfg.input, &weights);
    evalm.set_mode(LossMode::Detection);
    evalm
}

/// Run `detect` on every image of `val` (eval-mode BN, via a batch-`N` clone of
/// the trained `model`) and return per-image detections in original-image coords
/// = pixel `xyxy` of the `side x side` frame.
pub fn detect_split(
    model: &Yolo,
    cfg: &YoloConfig,
    val: &[Sample],
    side: u32,
    conf: f32,
    iou: f32,
) -> Vec<Vec<Detection>> {
    let evalm = clone_for_eval(model, cfg, val.len() as u32);
    let hwc: Vec<Vec<f32>> = val.iter().map(|s| imaging::pixels::chw_to_hwc(&s.chw, 3, side as usize, side as usize)).collect();
    let batch: Vec<(&[f32], u32, u32)> = hwc.iter().map(|h| (h.as_slice(), side, side)).collect();
    evalm.detect_batch(&batch, conf, iou)
}

/// Numerically-stable sigmoid (matches `infer.rs`).
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Pure-Rust DFL decode (matches `dfl_decode.wgsl` / p5's `decode_dist`): box
/// logits flat `[na, 4*reg_max]`, each (img,anchor) row laid out side-major
/// `[4][reg_max]`; returns per-side expectations flat `[na, 4]`.
fn dfl_decode_rust(box_logits: &[f32], na: usize, reg_max: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; na * 4];
    for idx in 0..na * 4 {
        let base = idx * reg_max;
        let mx = (0..reg_max).map(|i| box_logits[base + i]).fold(f32::MIN, f32::max);
        let sum: f32 = (0..reg_max).map(|i| (box_logits[base + i] - mx).exp()).sum();
        out[idx] = (0..reg_max)
            .map(|i| i as f32 * (box_logits[base + i] - mx).exp() / sum)
            .sum();
    }
    out
}

/// Detection over a split using the TRAIN-mode forward (current-batch BN stats),
/// decoding the raw cls/box logits exactly as the P5 overfit gate does, then
/// class-aware NMS. Returns per-image `Detection`s in `side x side` pixel coords.
///
/// Why train-mode (and not the eval-mode `Yolo::detect`): `detect` flips BN to
/// running-stats (`set_eval(true)`), but BN running stats only converge after
/// MANY steps. In minute-scale SMOKE training they are still far from the batch
/// stats, so an eval-mode forward is near-garbage even when the model has clearly
/// learned (loss collapses, train-mode decode recovers the objects) — measured:
/// eval-mode `detect` gave mAP 0 / IoU 0 after 150 steps. Decoding the train-mode
/// logits over the SAME-distribution val batch is the fair capability signal for
/// short training and matches the proven P5 recovery path. (`detect`'s own
/// eval-mode pipeline is separately smoke-tested in `p6_infer.rs`.)
pub fn detect_split_train(
    model: &Yolo,
    cfg: &YoloConfig,
    val: &[Sample],
    side: u32,
    conf: f32,
    iou_thresh: f32,
) -> Vec<Vec<Detection>> {
    let n = val.len();
    let m = clone_for_eval(model, cfg, n as u32); // train-mode BN (no set_eval).
    m.set_image(&stack_images(val));
    m.forward_net_pub();

    let (cls, boxl) = m.raw_logits();
    let anchors = m.anchor_geometry();
    let a = anchors.len();
    let nc = cfg.nc as usize;
    let reg_max = cfg.reg_max as usize;
    let dist = dfl_decode_rust(&boxl, n * a, reg_max);

    let mut out = Vec::with_capacity(n);
    for img in 0..n {
        let mut dets: Vec<Detection> = Vec::new();
        for (i, &an) in anchors.iter().enumerate() {
            // best class + score for this anchor.
            let cbase = (img * a + i) * nc;
            let mut best_c = 0usize;
            let mut best_s = 0.0f32;
            for c in 0..nc {
                let s = sigmoid(cls[cbase + c]);
                if s > best_s {
                    best_s = s;
                    best_c = c;
                }
            }
            if best_s < conf {
                continue;
            }
            let dbase = (img * a + i) * 4;
            let d = [dist[dbase], dist[dbase + 1], dist[dbase + 2], dist[dbase + 3]];
            let bx = yolov8::boxmath::dist_to_xyxy(d, an.ax, an.ay, an.stride);
            dets.push([bx[0], bx[1], bx[2], bx[3], best_s, best_c as f32]);
        }
        out.push(yolov8::nms(&dets, iou_thresh, 300));
    }
    let _ = side;
    out
}

/// Convert a split's normalized GT into the evaluator's pixel-`xyxy` `GtBox`
/// (flattened across the whole split, since `map50` takes one combined list).
pub fn eval_gts(val: &[Sample], side: u32, relabel: impl Fn(u32) -> u32) -> Vec<EvalGt> {
    let mut out = Vec::new();
    for s in val {
        for b in &s.boxes {
            out.push(EvalGt { class: relabel(b.class), bbox: xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32) });
        }
    }
    out
}

/// Flatten per-image detections into one `Vec<Detection>` (drop the image index;
/// safe here because we score each image's preds against only its own GT below).
/// For multi-image scoring we instead score per-image and average — see callers.
pub fn flatten_dets(per_img: &[Vec<Detection>]) -> Vec<Detection> {
    per_img.iter().flatten().copied().collect()
}

/// mAP@0.5 computed PER IMAGE then averaged (so detections in image i are only
/// matched against image i's GT — the correct multi-image protocol for these
/// independent scenes). Images with no GT are skipped (mAP undefined there).
pub fn map50_per_image(per_img: &[Vec<Detection>], val: &[Sample], side: u32, relabel: &impl Fn(u32) -> u32, nc: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (dets, s) in per_img.iter().zip(val) {
        if s.boxes.is_empty() {
            continue;
        }
        let gts: Vec<EvalGt> = s
            .boxes
            .iter()
            .map(|b| EvalGt { class: relabel(b.class), bbox: xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32) })
            .collect();
        sum += eval::detection::map50(dets, &gts, nc);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}
