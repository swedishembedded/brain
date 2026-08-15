// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P5 OVERFIT GATE — the most important detector integration test.
//!
//! These tests prove the WHOLE pipeline learns end-to-end:
//!   data (gen_detect) -> forward -> Task-Aligned assigner -> BCE+CIoU+DFL loss
//!   -> backward (head/neck/backbone reverse chain) -> AdamW.
//!
//! A finite-difference gradcheck (P3/P4) catches per-op numerical errors, but
//! not "the full loop fails to learn". An overfit test does: a correct detector
//! MUST be able to memorize a handful of fixed synthetic scenes. If the
//! one-image overfit fails to converge that is a real pipeline bug, not a
//! threshold to relax.
//!
//! ## Recovery check used
//! The P6 NMS/decode path is not relied on here. Instead we decode boxes
//! DIRECTLY from the network's raw logits, exactly as the loss module does:
//!   * DFL decode: per (anchor, side), softmax over `reg_max` bins then take the
//!     expectation `E = Σ i·p_i` (replicated in `decode_dist` below, matching
//!     `dfl_decode.wgsl`).
//!   * `boxmath::dist_to_xyxy` maps (l,t,r,b) + anchor point + stride -> pixel
//!     xyxy.
//!   * `sigmoid(cls_logits)` -> per-anchor class scores.
//!
//! Recovery for a GT = some anchor whose ARGMAX class equals the GT class, whose
//! sigmoid score on that class is "high", and whose decoded box has IoU>0.5 with
//! the GT. Recall = fraction of GTs recovered.
//!
//! Gated by `MOE_SKIP_GPU_TESTS` (same as the GPT convergence suite). Run WITHOUT
//! the env to actually train:
//!   `MOE_SKIP_GPU_TESTS= cargo test -p brain-yolo --test p5_overfit -- --nocapture`

use data::gen_detect::{self, Preset};
use data::rng::Rng;
use yolov8::boxmath::{dist_to_xyxy, iou};
use yolov8::{GtBox, LossMode, Yolo, YoloConfig};

/// Skip the whole test when no accelerator (CPU JIT) run is wanted. An EMPTY
/// value (`MOE_SKIP_GPU_TESTS=`) does NOT skip — that is the documented way to
/// force the overfit to actually train; only a non-empty value skips.
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// DFL decode replicated in plain Rust (matches `dfl_decode.wgsl`): box logits
/// flat `[N,A,4*reg_max]`, row `(n*A+i)` laid out side-major `[4][reg_max]`;
/// returns per-side distances flat `[N,A,4]`.
fn decode_dist(box_logits: &[f32], na: usize, reg_max: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; na * 4];
    for (idx, o) in out.iter_mut().enumerate() {
        let base = idx * reg_max;
        let mx = (0..reg_max).map(|i| box_logits[base + i]).fold(f32::MIN, f32::max);
        let sum: f32 = (0..reg_max).map(|i| (box_logits[base + i] - mx).exp()).sum();
        let e: f32 = (0..reg_max)
            .map(|i| i as f32 * (box_logits[base + i] - mx).exp() / sum)
            .sum();
        *o = e;
    }
    out
}

/// A decoded prediction for one anchor in one image.
struct Pred {
    img: usize,
    box_: [f32; 4],
    cls: usize,
    score: f32, // sigmoid score on `cls`
}

/// Decode every anchor of every image into (box, argmax-class, score).
fn decode_all(model: &Yolo, cfg: &YoloConfig, n: usize) -> Vec<Pred> {
    let (cls, boxl) = model.raw_logits();
    let anchors = model.anchor_geometry();
    let a = anchors.len();
    let nc = cfg.nc as usize;
    let reg_max = cfg.reg_max as usize;
    let dist = decode_dist(&boxl, n * a, reg_max);

    let mut out = Vec::with_capacity(n * a);
    for img in 0..n {
        for i in 0..a {
            let an = anchors[i];
            let base = (img * a + i) * 4;
            let d = [dist[base], dist[base + 1], dist[base + 2], dist[base + 3]];
            let box_ = dist_to_xyxy(d, an.ax, an.ay, an.stride);
            // argmax class + its sigmoid score.
            let mut best_c = 0usize;
            let mut best_l = f32::MIN;
            for c in 0..nc {
                let l = cls[(img * a + i) * nc + c];
                if l > best_l {
                    best_l = l;
                    best_c = c;
                }
            }
            out.push(Pred { img, box_, cls: best_c, score: sigmoid(best_l) });
        }
    }
    out
}

/// Recall over the GT set: a GT is recovered if SOME anchor in its image has the
/// correct argmax class, score >= `score_thr`, and IoU>0.5 with the GT box.
/// Returns (recovered, total).
fn recall(preds: &[Pred], gts: &[GtBox], img_size: f32, score_thr: f32) -> (usize, usize) {
    let mut recovered = 0;
    for g in gts {
        let gx = yolov8::boxmath::xywhn_to_xyxy(g.cx, g.cy, g.w, g.h, img_size);
        let hit = preds.iter().any(|p| {
            p.img == g.img as usize
                && p.cls == g.cls as usize
                && p.score >= score_thr
                && iou(p.box_, gx) > 0.5
        });
        if hit {
            recovered += 1;
        }
    }
    (recovered, gts.len())
}

/// Build GtBoxes for image `img` from a slice of normalized DetectBoxes.
fn boxes_for_img(img: u32, dboxes: &[data::binio::DetectBox]) -> Vec<GtBox> {
    dboxes
        .iter()
        .map(|b| GtBox { img, cls: b.class, cx: b.cx, cy: b.cy, w: b.w, h: b.h })
        .collect()
}

// ===========================================================================
// One-image overfit: the canonical "the whole pipeline learns" gate.
// ===========================================================================
#[test]
fn one_image_overfit_recovers_objects() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let cfg = YoloConfig::tiny(nc); // 128px input, reg_max=8, strides 8/16/32.
    let side = cfg.input;
    let b = 1u32;

    // ONE 128x128 scene with a few large, clearly separated shapes of 2-3
    // classes. We hand-build it (instead of a random preset) so the objects are
    // big and well-spread -> many fg anchors -> a strongly-conditioned signal,
    // and re-roll the seed until we get >=2 boxes spanning >=2 classes.
    let mut images = Vec::new();
    let mut gts: Vec<GtBox> = Vec::new();
    for seed in 0u64..200 {
        let mut rng = Rng::new(seed);
        let scene = gen_detect::gen_scene(Preset::MultiObject, side, side, nc, &mut rng);
        let classes: std::collections::HashSet<u32> = scene.boxes.iter().map(|b| b.class).collect();
        let big = scene.boxes.iter().all(|b| b.w >= 0.12 && b.h >= 0.12);
        if scene.boxes.len() >= 2 && scene.boxes.len() <= 3 && classes.len() >= 2 && big {
            images = gen_detect::image_to_chw(&scene.image);
            gts = boxes_for_img(0, &scene.boxes);
            break;
        }
    }
    assert!(!gts.is_empty(), "failed to sample a suitable one-image scene");
    println!("one-image overfit: {} GT boxes, classes {:?}", gts.len(), gts.iter().map(|g| g.cls).collect::<Vec<_>>());

    let init = yolov8::init_weights(&cfg, 1234);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&images);
    model.set_targets(&gts);

    // Conservative LR, no weight decay, no augmentation. The assigner is re-run
    // every step (NOT frozen) — freezing is only for gradcheck.
    let steps = 250u32;
    let lr = 1e-3f32;
    let t0 = std::time::Instant::now();

    let initial = model.forward();
    let mut last = initial;
    for step in 1..=steps {
        model.zero_grads();
        let l = model.forward();
        model.backward();
        model.adamw_step(step, lr, 0.0, Some(10.0), 1.0);
        last = l;
        if step % 50 == 0 || step == 1 {
            println!("  step {step:>3}: loss {l:.4}");
        }
    }
    model.poll_wait();

    let preds = decode_all(&model, &cfg, b as usize);
    let (rec, total) = recall(&preds, &gts, side as f32, 0.30);
    let drop = (initial - last) / initial;
    println!(
        "one-image overfit: loss {initial:.4} -> {last:.4} ({:.1}% drop), recall {rec}/{total}, {:.2?}",
        drop * 100.0,
        t0.elapsed()
    );

    assert!(last.is_finite(), "final loss non-finite");
    assert!(
        drop > 0.80,
        "one-image overfit did not learn: loss {initial:.4} -> {last:.4} ({:.1}% drop, expected >80%)",
        drop * 100.0
    );
    assert_eq!(
        rec, total,
        "one-image overfit failed to recover every object: {rec}/{total} (IoU>0.5 + correct class @ score>=0.30)"
    );
}

// ===========================================================================
// Tiny-dataset overfit: a handful of mixed-preset scenes (incl. empties).
// ===========================================================================
#[test]
fn tiny_dataset_overfit_high_recall() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let cfg = YoloConfig::tiny(nc);
    let side = cfg.input;
    let b = 8u32; // 8 images, trained as one fixed batch.

    // Build 8 fixed scenes from mixed presets, including 1-2 empties. We reuse
    // the same fixed seed sequence so the dataset is deterministic.
    let presets = [
        Preset::Localization,
        Preset::Classification,
        Preset::MultiObject,
        Preset::Scale,
        Preset::Background, // may be empty
        Preset::MultiObject,
        Preset::Background, // may be empty
        Preset::Classification,
    ];
    let mut images: Vec<f32> = Vec::new();
    let mut gts: Vec<GtBox> = Vec::new();
    let mut rng = Rng::new(2025);
    let mut n_empty = 0;
    for (img, &p) in presets.iter().enumerate() {
        let scene = gen_detect::gen_scene(p, side, side, nc, &mut rng);
        images.extend_from_slice(&gen_detect::image_to_chw(&scene.image));
        if scene.boxes.is_empty() {
            n_empty += 1;
        }
        gts.extend(boxes_for_img(img as u32, &scene.boxes));
    }
    println!("tiny-dataset overfit: {} images, {} GTs, {} empty", b, gts.len(), n_empty);

    let init = yolov8::init_weights(&cfg, 77);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&images);
    model.set_targets(&gts);

    // 200 steps is enough for full recall here (measured ~98% loss drop, 12/12
    // recovered at 300); 200 keeps the b=8 batch under a few minutes on CPU JIT.
    let steps = 200u32;
    let lr = 1e-3f32;
    let t0 = std::time::Instant::now();

    let initial = model.forward();
    let mut last = initial;
    for step in 1..=steps {
        model.zero_grads();
        let l = model.forward();
        model.backward();
        model.adamw_step(step, lr, 0.0, Some(10.0), 1.0);
        last = l;
        if step % 50 == 0 || step == 1 {
            println!("  step {step:>3}: loss {l:.4}");
        }
    }
    model.poll_wait();

    let preds = decode_all(&model, &cfg, b as usize);
    let (rec, total) = recall(&preds, &gts, side as f32, 0.25);
    let drop = (initial - last) / initial;
    println!(
        "tiny-dataset overfit: loss {initial:.4} -> {last:.4} ({:.1}% drop), recall {rec}/{total}, {:.2?}",
        drop * 100.0,
        t0.elapsed()
    );

    assert!(last.is_finite(), "final loss non-finite");
    assert!(
        drop > 0.70,
        "tiny-dataset overfit did not learn: loss {initial:.4} -> {last:.4} ({:.1}% drop, expected >70%)",
        drop * 100.0
    );
    // Most GTs recovered. CPU-JIT + tiny config: allow a small miss margin.
    let need = (total as f32 * 0.75).ceil() as usize;
    assert!(
        rec >= need,
        "tiny-dataset recall too low: {rec}/{total} (need >= {need} @ IoU>0.5, correct class, score>=0.25)"
    );
}
