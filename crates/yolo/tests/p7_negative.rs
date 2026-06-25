// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P7 — NEGATIVE CONTROLS (the highest-value detector tests).
//!
//! A capability test that passes only tells you the model *can* score high; it
//! does NOT tell you the score MEANS anything. The controls below are the canary
//! for label/eval leakage — they MUST fail to learn:
//!
//!   * **Shuffled-label**: train on the same images/boxes but with class labels
//!     randomly PERMUTED. A correct pipeline cannot fit a contradictory mapping,
//!     so val mAP50 must COLLAPSE relative to the clean-label run. If shuffled
//!     labels still score high, the eval is leaking the answer (e.g. matching is
//!     class-agnostic, or train==val) — a real bug, NOT a threshold to relax.
//!   * **Random-image**: keep the labels, replace the IMAGES with RNG noise.
//!     There is no signal to learn the boxes from, so val mAP50 must approach
//!     chance (low).
//!
//! Thresholds carry margin for CPU-JIT fp noise + tiny-data variance; see each.
//!
//! Gated by `MOE_SKIP_GPU_TESTS` (train with `MOE_SKIP_GPU_TESTS= BRAIN_DEVICE=cpu
//! cargo test -p brain-yolo --test p7_negative -- --nocapture`).

#[path = "common/p7.rs"]
mod p7;

use data::gen_detect::Preset;
use data::rng::Rng;
use p7::*;

// ===========================================================================
// Shuffled-label control: clean mAP high, shuffled mAP collapses.
// ===========================================================================
#[test]
fn shuffled_label_control_collapses() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    // Classification preset: fixed centered geometry, class<->color only. This
    // ISOLATES the label signal — boxes are identical across classes, so the only
    // thing distinguishing a correct from a shuffled run is the color<->class map.
    // A leakage bug (class-agnostic eval, train/val overlap) would let the
    // shuffled run score just as high; a correct pipeline cannot.
    // Small split + step count + 64px input: the classification signal (fixed
    // centered shapes, class<->color only) is easy, so a few images and ~120 steps
    // suffice — and on the CPU JIT each step scales with batch * input^2, so all
    // three are kept low to stay inside a minute-scale smoke budget.
    let n_train = 6usize;
    let n_val = 8usize;
    let train = gen_split(Preset::Classification, n_train, side, nc, 0xC1EA0);
    let val = gen_split(Preset::Classification, n_val, side, nc, 0x7A11); // disjoint seed
    let ident = |c: u32| c;

    let steps = 120u32;
    let lr = 1e-3f32;
    let (conf, iou) = (0.30f32, 0.5f32);

    // --- clean labels ---
    let clean_t = targets_for_split(&train, ident);
    let t0 = std::time::Instant::now();
    println!("[shuffled-label] CLEAN run:");
    let (m_clean, cfg) = train_model(&cfg, &train, &clean_t, steps, lr, 1234);
    let d_clean = detect_split_train(&m_clean, &cfg, &val, side, conf, iou);
    let map_clean = map50_per_image(&d_clean, &val, side, &ident, nc);

    // --- shuffled labels: permute classes with a fixed derangement-ish map ---
    // A cyclic shift by 1 (0->1->2->0) is a fixed permutation with NO fixed point
    // for nc=3, so every box is mislabeled. Boxes/positions are UNCHANGED.
    let shuffle = |c: u32| (c + 1) % nc;
    let shuf_t = targets_for_split(&train, shuffle);
    println!("[shuffled-label] SHUFFLED run:");
    let (m_shuf, cfg2) = train_model(&cfg, &train, &shuf_t, steps, lr, 1234);
    // Eval against the TRUE (clean) labels: a model that learned the shuffled map
    // predicts the wrong class for each color, so it cannot match the true GT.
    let d_shuf = detect_split_train(&m_shuf, &cfg2, &val, side, conf, iou);
    let map_shuf = map50_per_image(&d_shuf, &val, side, &ident, nc);

    println!(
        "[shuffled-label] clean mAP50 = {map_clean:.3}, shuffled mAP50 = {map_shuf:.3}, elapsed {:.1?}",
        t0.elapsed()
    );

    // Clean run must actually learn something (else the comparison is vacuous).
    assert!(
        map_clean > 0.30,
        "clean-label run failed to learn (mAP50 {map_clean:.3}); the control comparison is meaningless. \
         Investigate the training/eval path before trusting the collapse."
    );
    // The canary: shuffled must collapse well below clean. < 0.3x clean is the
    // spec bar; with margin we also accept any shuffled < 0.25 absolute.
    let collapsed = map_shuf < 0.3 * map_clean || map_shuf < 0.25;
    assert!(
        collapsed,
        "LEAKAGE SUSPECTED: shuffled-label mAP50 ({map_shuf:.3}) did NOT collapse vs clean ({map_clean:.3}). \
         A correct, leak-free pipeline cannot fit randomly-permuted class labels and still score on the \
         TRUE labels. Check that eval matching is class-AWARE and that train/val are disjoint."
    );
}

// ===========================================================================
// Random-image control: labels kept, images are noise -> mAP near chance.
// ===========================================================================
#[test]
fn random_image_control_near_chance() {
    if skip() {
        return;
    }
    let nc = 3u32;
    let side = SIDE;
    let cfg = tiny_cfg(nc, side);
    let n_train = 6usize;
    let n_val = 8usize;
    // Real labels from a real preset, but we OVERWRITE the pixels with noise.
    let mut train = gen_split(Preset::MultiObject, n_train, side, nc, 0x4242);
    let val = gen_split(Preset::MultiObject, n_val, side, nc, 0x9999);
    let ident = |c: u32| c;

    // Replace every train image with deterministic RNG noise in [0,1]; KEEP boxes.
    let mut rng = Rng::new(0x0015Eu64);
    for s in train.iter_mut() {
        for v in s.chw.iter_mut() {
            *v = rng.next_f64() as f32;
        }
    }
    // Val images ALSO noise (the model has nothing real to detect at test time):
    // labels point at objects that aren't in the (noise) pixels, so any correct
    // box is pure luck.
    let mut val = val;
    let mut rng2 = Rng::new(0xBADF00D);
    for s in val.iter_mut() {
        for v in s.chw.iter_mut() {
            *v = rng2.next_f64() as f32;
        }
    }

    let steps = 120u32;
    let lr = 1e-3f32;
    let (conf, iou) = (0.30f32, 0.5f32);

    let targets = targets_for_split(&train, ident);
    let t0 = std::time::Instant::now();
    println!("[random-image] noise-image run:");
    let (model, cfg) = train_model(&cfg, &train, &targets, steps, lr, 77);
    let dets = detect_split_train(&model, &cfg, &val, side, conf, iou);
    let map_rand = map50_per_image(&dets, &val, side, &ident, nc);
    println!("[random-image] mAP50 = {map_rand:.3}, elapsed {:.1?}", t0.elapsed());

    // With no image signal, val mAP must be near chance (low). Generous ceiling
    // for CPU-JIT noise + the rare lucky overlap on tiny data.
    assert!(
        map_rand < 0.20,
        "random-image control scored too HIGH (mAP50 {map_rand:.3}); a model trained on NOISE pixels \
         should not detect objects at test time. A high score implies the score does not depend on the \
         image (eval leakage) — investigate before relaxing this bound."
    );
}
