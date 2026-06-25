// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P11 EVAL-INFERENCE RECOVERY GATE — proves BN running stats are usable for
//! inference through the EVAL-mode `Yolo::detect` path.
//!
//! ## The bug this guards
//! `Conv` accumulates its BN running mean/var only when `update_running` is on.
//! That toggle defaults OFF (the gradchecks need deterministic forwards) and was
//! never enabled during training, so `bn.run_mean`/`bn.run_var` stayed at their
//! init (mean=0, var=1). The train-mode forward uses *batch* stats (fine for
//! training + the P5 overfit, which decodes raw train-mode logits), but
//! `Yolo::detect` flips every Conv to EVAL-mode BN (`set_eval(true)`), which
//! reads those never-updated running stats and produces garbage. The CLI and the
//! runtime `camera_frame -> object_detected` path both go through `detect`, so
//! the shipped inference was broken.
//!
//! ## What this test does
//! Overfit ONE small synthetic scene with `set_update_running(true)` so the BN
//! running EMA tracks the data, then run the REAL eval-mode `Yolo::detect` and
//! assert it recovers every GT box (IoU>0.5, correct class, conf above
//! threshold). Unlike P5 (which decodes raw TRAIN-mode logits), this exercises
//! the production eval-mode BN path end to end. A control run with the toggle
//! OFF documents that the recovery is what the fix buys.
//!
//! Gated by `MOE_SKIP_GPU_TESTS` (same as the P5 overfit suite). Run WITHOUT the
//! env to actually train:
//!   `MOE_SKIP_GPU_TESTS= cargo test -p brain-yolo --test p11_eval_inference -- --nocapture`

use data::gen_detect::{self, Preset};
use data::rng::Rng;
use yolo::boxmath::{iou, xywhn_to_xyxy};
use yolo::{GtBox, LossMode, Yolo, YoloConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

/// Build GtBoxes for image `img` from a slice of normalized DetectBoxes.
fn boxes_for_img(img: u32, dboxes: &[data::binio::DetectBox]) -> Vec<GtBox> {
    dboxes
        .iter()
        .map(|b| GtBox { img, cls: b.class, cx: b.cx, cy: b.cy, w: b.w, h: b.h })
        .collect()
}

/// Interleaved-RGB HWC `src[h*w*3]` (the `Yolo::detect` input form), built from
/// the generator's CHW blob (`gen_detect::image_to_chw` is `[3,H,W]`, /255).
fn chw_to_hwc(chw: &[f32], side: usize) -> Vec<f32> {
    let hw = side * side;
    let mut hwc = vec![0.0f32; hw * 3];
    for p in 0..hw {
        hwc[p * 3] = chw[p];
        hwc[p * 3 + 1] = chw[hw + p];
        hwc[p * 3 + 2] = chw[2 * hw + p];
    }
    hwc
}

/// Recall over the GT set through `Yolo::detect`. A GT is recovered if some
/// returned detection has the correct class, conf >= `conf_thr`, IoU>0.5.
/// `dets` are `[x1,y1,x2,y2,conf,class]` in pixel coords. Returns (rec, total,
/// best_iou_over_recovered, best_conf_over_recovered).
fn recall(
    dets: &[[f32; 6]],
    gts: &[GtBox],
    img_size: f32,
    conf_thr: f32,
) -> (usize, usize, f32, f32) {
    let mut recovered = 0;
    let mut sum_iou = 0.0f32;
    let mut sum_conf = 0.0f32;
    for g in gts {
        let gx = xywhn_to_xyxy(g.cx, g.cy, g.w, g.h, img_size);
        let mut best: Option<(f32, f32)> = None; // (iou, conf)
        for d in dets {
            if d[5] as u32 == g.cls && d[4] >= conf_thr {
                let iv = iou([d[0], d[1], d[2], d[3]], gx);
                if iv > 0.5 && best.map(|(bi, _)| iv > bi).unwrap_or(true) {
                    best = Some((iv, d[4]));
                }
            }
        }
        if let Some((iv, cf)) = best {
            recovered += 1;
            sum_iou += iv;
            sum_conf += cf;
        }
    }
    let n = recovered.max(1) as f32;
    (recovered, gts.len(), sum_iou / n, sum_conf / n)
}

/// Sample ONE well-conditioned scene (2-3 big, clearly separated shapes spanning
/// >=2 classes), exactly like the P5 one-image overfit scaffolding.
fn sample_scene(side: u32, nc: u32) -> (Vec<f32>, Vec<GtBox>) {
    for seed in 0u64..200 {
        let mut rng = Rng::new(seed);
        let scene = gen_detect::gen_scene(Preset::MultiObject, side, side, nc, &mut rng);
        let classes: std::collections::HashSet<u32> = scene.boxes.iter().map(|b| b.class).collect();
        let big = scene.boxes.iter().all(|b| b.w >= 0.12 && b.h >= 0.12);
        if scene.boxes.len() >= 2 && scene.boxes.len() <= 3 && classes.len() >= 2 && big {
            return (gen_detect::image_to_chw(&scene.image), boxes_for_img(0, &scene.boxes));
        }
    }
    panic!("failed to sample a suitable one-image scene");
}

/// Overfit a fresh tiny model on the given scene and return it. `update_running`
/// selects whether the BN running stats are accumulated (the fix) or left at
/// init (the bug control).
fn overfit(cfg: &YoloConfig, chw: &[f32], gts: &[GtBox], update_running: bool, steps: u32) -> Yolo {
    let init = yolo::init_weights(cfg, 1234);
    let model = Yolo::new(cfg.clone(), 1, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_eval(false);
    model.set_update_running(update_running);
    model.set_image(chw);
    model.set_targets(gts);

    let lr = 1e-3f32;
    let initial = model.forward();
    let mut last = initial;
    for step in 1..=steps {
        model.zero_grads();
        let l = model.forward();
        model.backward();
        model.adamw_step(step, lr, 0.0, Some(10.0), 1.0);
        last = l;
        if step % 50 == 0 || step == 1 {
            println!("  [update_running={update_running}] step {step:>3}: loss {l:.4}");
        }
    }
    model.poll_wait();
    let drop = (initial - last) / initial;
    println!("  [update_running={update_running}] loss {initial:.4} -> {last:.4} ({:.1}% drop)", drop * 100.0);
    model
}

// ===========================================================================
// The eval-mode inference recovery gate.
// ===========================================================================
#[test]
fn eval_mode_detect_recovers_overfit_objects() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let cfg = YoloConfig::tiny(nc); // 128px input, reg_max=8, strides 8/16/32.
    let side = cfg.input;
    let steps = 300u32;
    let conf_thr = 0.30f32;

    let (chw, gts) = sample_scene(side, nc);
    println!(
        "p11 eval-inference: {} GT boxes, classes {:?}",
        gts.len(),
        gts.iter().map(|g| g.cls).collect::<Vec<_>>()
    );
    // `detect` wants interleaved-RGB HWC at the original size; at native size the
    // letterbox is an identity resize, so the eval path sees exactly the training
    // pixels (only the BN mode differs from training).
    let hwc = chw_to_hwc(&chw, side as usize);

    let t0 = std::time::Instant::now();

    // ---- THE FIX: running stats accumulated during training ----
    let model = overfit(&cfg, &chw, &gts, true, steps);
    let dets_on = model.detect(&hwc, side, side, conf_thr, 0.45);
    let (rec_on, total, iou_on, conf_on) = recall(&dets_on, &gts, side as f32, conf_thr);
    println!(
        "p11 eval-mode detect [running ON]: recall {rec_on}/{total}, mean IoU {iou_on:.3}, mean conf {conf_on:.3}, {} dets",
        dets_on.len()
    );

    // ---- THE BUG control: running stats left at init (mean=0,var=1) ----
    let model_off = overfit(&cfg, &chw, &gts, false, steps);
    let dets_off = model_off.detect(&hwc, side, side, conf_thr, 0.45);
    let (rec_off, _t, iou_off, conf_off) = recall(&dets_off, &gts, side as f32, conf_thr);
    println!(
        "p11 eval-mode detect [running OFF/bug]: recall {rec_off}/{total}, mean IoU {iou_off:.3}, mean conf {conf_off:.3}, {} dets",
        dets_off.len()
    );
    println!("p11 eval-inference: {:.2?}", t0.elapsed());

    // The hard gate: eval-mode detect through the production path recovers EVERY
    // GT (correct class, IoU>0.5, conf>=thr) once running stats are accumulated.
    assert_eq!(
        rec_on, total,
        "eval-mode detect failed to recover every object with running stats ON: \
         {rec_on}/{total} (IoU>0.5, correct class, conf>={conf_thr})"
    );
    // Sanity: the toggle is what fixes it — without accumulated running stats the
    // eval-mode detect recovers strictly fewer objects.
    assert!(
        rec_off < rec_on,
        "expected the running-stats fix to improve eval-mode recall, but OFF={rec_off} >= ON={rec_on} \
         (running stats may not be the differentiator on this seed)"
    );
}
