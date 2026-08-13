// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P7 — procedural CAPABILITY tasks.
//!
//! Each test trains the tiny `Yolo` on a generated preset and scores a HELD-OUT
//! (disjoint-seed) split of the SAME preset. The signal is train-set
//! memorization generalizing to fresh same-distribution scenes; steps are kept
//! small (minute-scale smoke training, NOT a full experiment), so thresholds are
//! calibrated CONSERVATIVELY with margin for CPU-JIT noise and tiny-data variance.
//!
//! Runnable by default (when `MOE_SKIP_GPU_TESTS` is unset): localization,
//! classification, background-suppression. The heavier `scale` and `multi_object`
//! tasks are `#[ignore]`d (run with `--ignored`); they add coverage but cost more
//! CPU than the every-PR budget warrants.
//!
//! Gate / run:
//!   MOE_SKIP_GPU_TESTS= BRAIN_DEVICE=cpu cargo test -p brain-yolo --test p7_capability -- --nocapture
//!   (+ ` -- --ignored` for scale / multi-object).

#[path = "common/p7.rs"]
mod p7;

use data::gen_detect::Preset;
use eval::detection::{map50, precision_recall, GtBox as EvalGt};
use p7::*;
use yolov8::boxmath::{iou, xywhn_to_xyxy};

const IDENT: fn(u32) -> u32 = |c| c;

// ===========================================================================
// Localization: can it put a box where the object is? (single fixed class)
// ===========================================================================
#[test]
fn localization_decent_box_map_and_iou() {
    if skip() {
        return;
    }
    let nc = 1u32; // Localization preset is fixed-class; nc=1 keeps it pure.
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    let train = gen_split(Preset::Localization, 6, side, nc, 0x10C0);
    let val = gen_split(Preset::Localization, 8, side, nc, 0xF00B); // disjoint
    let targets = targets_for_split(&train, IDENT);

    let t0 = std::time::Instant::now();
    let (model, cfg) = train_model(&cfg, &train, &targets, 150, 1e-3, 1234);
    let dets = detect_split_train(&model, &cfg, &val, side, 0.30, 0.5);
    let map = map50_per_image(&dets, &val, side, &IDENT, nc);

    // Median best-IoU over val GTs (a localization-specific quality signal that
    // is independent of the conf/NMS operating point used for mAP).
    let mut best_ious = Vec::new();
    for (d, s) in dets.iter().zip(&val) {
        for b in &s.boxes {
            let g = xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32);
            let bi = d
                .iter()
                .map(|p| iou([p[0], p[1], p[2], p[3]], g))
                .fold(0.0f32, f32::max);
            best_ious.push(bi);
        }
    }
    best_ious.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_iou = if best_ious.is_empty() { 0.0 } else { best_ious[best_ious.len() / 2] };

    println!(
        "[localization] val mAP50 = {map:.3}, median best-IoU = {median_iou:.3}, elapsed {:.1?}",
        t0.elapsed()
    );
    // Conservative bars (measured ~0.50 mAP / ~0.54 IoU after 150 steps at 64px):
    // we only assert the model CAN localize, with margin for seed/CPU-JIT variance.
    assert!(map > 0.30, "localization mAP50 too low: {map:.3} (expected > 0.30)");
    assert!(median_iou > 0.40, "localization median IoU too low: {median_iou:.3} (expected > 0.40)");
}

// ===========================================================================
// Classification: fixed centered geometry, vary color/class. Class accuracy
// high + off-diagonal low.
// ===========================================================================
#[test]
fn classification_high_accuracy_low_offdiagonal() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    let train = gen_split(Preset::Classification, 6, side, nc, 0xC0DE);
    let val = gen_split(Preset::Classification, 9, side, nc, 0xBEEF); // disjoint
    let targets = targets_for_split(&train, IDENT);

    let t0 = std::time::Instant::now();
    let (model, cfg) = train_model(&cfg, &train, &targets, 150, 1e-3, 1234);
    let dets = detect_split_train(&model, &cfg, &val, side, 0.30, 0.5);

    // Confusion: for each val GT, take the highest-conf detection that overlaps it
    // (IoU>0.5) and record its predicted class vs the true class. Geometry is
    // fixed/centered, so a localized detection is essentially guaranteed; the only
    // question is the class it assigns.
    let mut conf_mat = vec![vec![0u32; nc as usize]; nc as usize];
    let mut matched = 0u32;
    let mut total = 0u32;
    for (d, s) in dets.iter().zip(&val) {
        for b in &s.boxes {
            total += 1;
            let g = xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32);
            let mut best: Option<(f32, usize)> = None; // (conf, pred_class)
            for p in d {
                if iou([p[0], p[1], p[2], p[3]], g) > 0.5 {
                    let conf = p[4];
                    if best.map(|(bc, _)| conf > bc).unwrap_or(true) {
                        best = Some((conf, p[5] as usize));
                    }
                }
            }
            if let Some((_, pc)) = best {
                matched += 1;
                conf_mat[b.class as usize][pc] += 1;
            }
        }
    }

    let diag: u32 = (0..nc as usize).map(|c| conf_mat[c][c]).sum();
    let off: u32 = matched - diag;
    let acc = if matched == 0 { 0.0 } else { diag as f32 / matched as f32 };
    let recall = if total == 0 { 0.0 } else { matched as f32 / total as f32 };
    println!(
        "[classification] matched {matched}/{total} GTs, class acc = {acc:.3}, off-diagonal = {off}, recall = {recall:.3}, elapsed {:.1?}",
        t0.elapsed()
    );
    println!("[classification] confusion (rows=true, cols=pred): {conf_mat:?}");

    assert!(recall > 0.60, "classification localized too few GTs: recall {recall:.3} (expected > 0.60)");
    assert!(acc > 0.70, "classification accuracy too low: {acc:.3} (expected > 0.70)");
    // Off-diagonal should be the minority of matches.
    assert!(
        (off as f32) < 0.30 * matched as f32,
        "too many class confusions: {off} off-diagonal of {matched} (expected < 30%)"
    );
}

// ===========================================================================
// Background suppression: >=50% empty val images. Few/zero false positives at a
// chosen conf threshold while positive recall stays up.
// ===========================================================================
#[test]
fn background_suppression_few_false_positives() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    // Train on the Background preset (real objects MIXED with empties, so the
    // model learns both the foreground and that blank scenes carry nothing).
    let train = gen_split(Preset::Background, 8, side, nc, 0xBA56);
    // Build a val split that is >=50% empty: a few real Background scenes plus a
    // dedicated block of GUARANTEED-empty scenes (the preset's own empty rate is
    // only ~30%, so we synthesize the empties to make the fraction deterministic).
    let mut val = gen_split(Preset::Background, 4, side, nc, 0x5151);
    let empties = gen_empty_scenes(5, side, nc, 0xE3E3);
    val.extend(empties);
    let n_empty = val.iter().filter(|s| s.boxes.is_empty()).count();
    assert!(
        n_empty as f32 >= 0.5 * val.len() as f32,
        "val must be >=50% empty for this control (got {n_empty}/{})",
        val.len()
    );

    let targets = targets_for_split(&train, IDENT);
    let conf = 0.45f32; // operating point: suppress weak background activations.
    let t0 = std::time::Instant::now();
    let (model, cfg) = train_model(&cfg, &train, &targets, 150, 1e-3, 1234);
    let dets = detect_split_train(&model, &cfg, &val, side, conf, 0.5);

    // False positives = ALL detections on EMPTY images (any box is wrong there).
    let mut fp_empty = 0u32;
    let mut n_empty_imgs = 0u32;
    for (d, s) in dets.iter().zip(&val) {
        if s.boxes.is_empty() {
            n_empty_imgs += 1;
            fp_empty += d.len() as u32;
        }
    }
    // Positive recall on NON-empty images (the suppression must not kill signal).
    let mut tp = 0u32;
    let mut gt_total = 0u32;
    for (d, s) in dets.iter().zip(&val) {
        if s.boxes.is_empty() {
            continue;
        }
        let gts: Vec<EvalGt> = s
            .boxes
            .iter()
            .map(|b| EvalGt { class: b.class, bbox: xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32) })
            .collect();
        let (_p, r) = precision_recall(d, &gts, 0.5);
        tp += (r * gts.len() as f32).round() as u32;
        gt_total += gts.len() as u32;
    }
    let recall = if gt_total == 0 { 0.0 } else { tp as f32 / gt_total as f32 };
    let fp_per_empty = fp_empty as f32 / n_empty_imgs.max(1) as f32;
    println!(
        "[background] {n_empty_imgs} empty imgs, {fp_empty} false positives ({fp_per_empty:.2}/empty img) @conf={conf}, positive recall = {recall:.3}, elapsed {:.1?}",
        t0.elapsed()
    );

    // The headline property: FEW false positives on background. Measured 0/empty
    // at conf 0.45 after 150 steps, so the strict bar is < 0.5/empty img.
    assert!(
        fp_per_empty < 0.5,
        "too many false positives on empty images: {fp_per_empty:.2}/img @conf {conf} (expected < 0.5)"
    );
    // Positive recall must merely STAY NON-ZERO — i.e. the suppression (conf 0.45,
    // chosen high to kill background activations) did not silence the foreground
    // entirely. The Background preset's objects are SMALL (10-28% of a 64px side,
    // so ~6-18px) and the smoke-budget training is short, so recall at this strict
    // conf is naturally modest (measured ~0.25); the contrast that matters is
    // "0 false positives on empties WHILE still finding real objects", not a high
    // recall number. We bar it at >0.10 to guard against total signal collapse.
    assert!(
        recall > 0.10,
        "background-suppression killed ALL positive recall: {recall:.3} (expected > 0.10 — suppression \
         should not silence the foreground entirely)"
    );
}

// ===========================================================================
// Scale (IGNORED by default — heavier): small/medium/large AP. Small is the
// weakest, which is EXPECTED for an anchor-free detector at low resolution.
// Run with: ... --test p7_capability -- --ignored --nocapture
// ===========================================================================
#[test]
#[ignore = "heavier capability task; run with --ignored"]
fn scale_small_medium_large_ap() {
    if skip() {
        return;
    }
    let nc = 1u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    // More steps + data: scale is the hardest geometric task here.
    let train = gen_split(Preset::Scale, 10, side, nc, 0x5CA1);
    let val = gen_split(Preset::Scale, 12, side, nc, 0xA1E5);
    let targets = targets_for_split(&train, IDENT);

    let t0 = std::time::Instant::now();
    let (model, cfg) = train_model(&cfg, &train, &targets, 200, 1e-3, 1234);
    let dets = detect_split_train(&model, &cfg, &val, side, 0.25, 0.5);

    // Bucket each val GT by its normalized area into small/medium/large and report
    // best-IoU recall@0.5 per bucket (a per-scale AP proxy on single-object scenes).
    let mut buckets = [(0u32, 0u32); 3]; // (recovered, total) for [small, med, large]
    for (d, s) in dets.iter().zip(&val) {
        for b in &s.boxes {
            let area = b.w * b.h;
            let bk = if area < 0.04 {
                0
            } else if area < 0.20 {
                1
            } else {
                2
            };
            let g = xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32);
            let hit = d.iter().any(|p| iou([p[0], p[1], p[2], p[3]], g) > 0.5);
            buckets[bk].1 += 1;
            if hit {
                buckets[bk].0 += 1;
            }
        }
    }
    let rec = |b: (u32, u32)| if b.1 == 0 { f32::NAN } else { b.0 as f32 / b.1 as f32 };
    println!(
        "[scale] recall@0.5 small={:.3}({}) medium={:.3}({}) large={:.3}({}), elapsed {:.1?}",
        rec(buckets[0]), buckets[0].1, rec(buckets[1]), buckets[1].1, rec(buckets[2]), buckets[2].1,
        t0.elapsed()
    );
    // Large objects must be found reliably; small are allowed to lag (the point of
    // the test is to SHOW the small<large gap, not to require small parity).
    if buckets[2].1 > 0 {
        assert!(rec(buckets[2]) > 0.50, "large-object recall too low: {:.3}", rec(buckets[2]));
    }
    let overall = map50_per_image(&dets, &val, side, &IDENT, nc);
    assert!(overall > 0.25, "scale overall mAP50 too low: {overall:.3}");
}

// ===========================================================================
// Multi-object (IGNORED by default — heavier): precision/recall + count error.
// ===========================================================================
#[test]
#[ignore = "heavier capability task; run with --ignored"]
fn multi_object_precision_recall_and_count() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    let train = gen_split(Preset::MultiObject, 10, side, nc, 0xACE5);
    let val = gen_split(Preset::MultiObject, 12, side, nc, 0xD00D);
    let targets = targets_for_split(&train, IDENT);

    let t0 = std::time::Instant::now();
    let (model, cfg) = train_model(&cfg, &train, &targets, 220, 1e-3, 1234);
    let dets = detect_split_train(&model, &cfg, &val, side, 0.30, 0.5);

    let mut tp = 0u32;
    let mut n_pred = 0u32;
    let mut n_gt = 0u32;
    let mut count_abs_err = 0i64;
    for (d, s) in dets.iter().zip(&val) {
        let gts: Vec<EvalGt> = s
            .boxes
            .iter()
            .map(|b| EvalGt { class: b.class, bbox: xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, side as f32) })
            .collect();
        let (_p, r) = precision_recall(d, &gts, 0.5);
        tp += (r * gts.len() as f32).round() as u32;
        n_pred += d.len() as u32;
        n_gt += gts.len() as u32;
        count_abs_err += (d.len() as i64 - s.boxes.len() as i64).abs();
    }
    let precision = if n_pred == 0 { 0.0 } else { tp as f32 / n_pred as f32 };
    let recall = if n_gt == 0 { 0.0 } else { tp as f32 / n_gt as f32 };
    let mae_count = count_abs_err as f32 / val.len() as f32;
    let map = map50_per_image(&dets, &val, side, &IDENT, nc);
    println!(
        "[multi-object] precision = {precision:.3}, recall = {recall:.3}, count MAE = {mae_count:.2}, mAP50 = {map:.3}, elapsed {:.1?}",
        t0.elapsed()
    );
    assert!(recall > 0.40, "multi-object recall too low: {recall:.3}");
    assert!(precision > 0.40, "multi-object precision too low: {precision:.3}");
    assert!(mae_count < 2.0, "multi-object count error too high: {mae_count:.2} boxes/img");
    let _ = map50; // (map50 also re-exported; per-image variant used above)
}
