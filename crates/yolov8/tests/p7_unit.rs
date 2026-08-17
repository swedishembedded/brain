// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P7 — cheap, EVERY-COMMIT detector correctness tests (NOT gated, no training).
//!
//! These are the fast unit-level guards that complement the (gated) training
//! tests in `p7_negative.rs` / `p7_capability.rs`. They must run — and pass —
//! under the default `cargo test` (CPU backend, `BRAIN_DEVICE=cpu`) in well under
//! a minute combined, because they perform at most a handful of forward/backward
//! passes on a tiny model (or no model at all).
//!
//! Coverage (per the P7 plan, the items not already covered by P1/P4/P6):
//!   * save/load equivalence — `save` then reload via the model's load path,
//!     identical raw logits;
//!   * one-step parameter update — weights change, grads finite, `zero_grads`
//!     clears;
//!   * invalid-label rejection / empty handling — a test-only `sanitize_gts`
//!     validator establishes the DEFINED behaviour (out-of-range class rejected,
//!     zero/negative-area rejected, out-of-range coords clipped), and the model
//!     gives FINITE loss + backward on the sanitized set, on an empty annotation,
//!     and on a degenerate-but-in-range box (assigner drops it as background);
//!   * box-format round trips not in `boxmath`'s own unit tests (full-image, 1px,
//!     border-touching, non-square);
//!   * frozen-layer / fine-tune — snapshot+restore the backbone (test-only freeze)
//!     across a few train steps, assert backbone UNCHANGED while head changes.

use std::collections::HashMap;

use yolov8::boxmath::{xywhn_to_xyxy, xyxy_to_xywh};
use yolov8::{init_weights, GtBox, LossMode, Yolo, YoloConfig};

// ---------------------------------------------------------------------------
// Shared tiny fixtures (batch=1, 128px, nc=3 — same shape as the overfit tests).
// ---------------------------------------------------------------------------

/// Deterministic LCG -> (-1,1) (matches the other yolo tests).
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// A fixed normalised [0,1] CHW image for `b` images of the tiny config.
fn fixed_image(cfg: &YoloConfig, b: u32, seed: u64) -> Vec<f32> {
    let n = (b * 3 * cfg.input * cfg.input) as usize;
    randvec(seed, n).iter().map(|v| (v + 1.0) * 0.5).collect()
}

/// One centred GT per class for a single image (well-conditioned fg signal).
fn fixed_gts() -> Vec<GtBox> {
    vec![
        GtBox { img: 0, cls: 0, cx: 0.40, cy: 0.50, w: 0.5, h: 0.6 },
        GtBox { img: 0, cls: 1, cx: 0.60, cy: 0.45, w: 0.4, h: 0.5 },
    ]
}

// ===========================================================================
// A. Save / load equivalence.
// ===========================================================================
#[test]
fn save_load_roundtrip_identical_logits() {
    let cfg = YoloConfig::tiny(3);
    let b = 1u32;
    let init = init_weights(&cfg, 4242);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);

    let img = fixed_image(&cfg, b, 0xF00D);
    model.set_image(&img);
    model.set_targets(&fixed_gts());
    model.forward_net_pub();
    let (cls0, box0) = model.raw_logits();

    // Save via the model's own `save` (checkpoint::save with role=""), then
    // reload exactly as the parity test / runtime adapter does: by_role("") ->
    // init map -> Yolo::new.
    let dir = std::env::temp_dir().join(format!("brain_p7_saveload_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("tiny.safetensors");
    let path_s = path.to_str().unwrap().to_string();
    model.save(&path_s);

    let container = checkpoint::load(&path_s);
    let weights: HashMap<String, Vec<f32>> = container.by_role("");
    // Every registered param must be present in the saved file.
    let plist = cfg.full_param_list();
    let missing: Vec<&String> =
        plist.iter().map(|(n, _)| n).filter(|n| !weights.contains_key(*n)).collect();
    assert!(missing.is_empty(), "saved file missing params: {:?}", &missing[..missing.len().min(8)]);

    let reloaded = Yolo::new(cfg.clone(), b, cfg.input, &weights);
    reloaded.set_mode(LossMode::Detection);
    reloaded.set_image(&img);
    reloaded.set_targets(&fixed_gts());
    reloaded.forward_net_pub();
    let (cls1, box1) = reloaded.raw_logits();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(cls0.len(), cls1.len());
    assert_eq!(box0.len(), box1.len());
    // Same weights + same input => bit-for-bit-ish identical logits. The save
    // path round-trips f32 losslessly; the only slack is fp non-determinism in
    // the CPU JIT reductions, which is well below 1e-4 here.
    let mut max_d = 0.0f32;
    for (a, b) in cls0.iter().zip(&cls1).chain(box0.iter().zip(&box1)) {
        assert!(a.is_finite() && b.is_finite(), "non-finite logit after reload");
        max_d = max_d.max((a - b).abs());
    }
    assert!(max_d < 1e-4, "save/load logits diverged: max abs diff {max_d:.3e}");
    println!("save/load: {} cls + {} box logits, max abs diff {max_d:.3e}", cls0.len(), box0.len());
}

// ===========================================================================
// B. One-step parameter update: weights move, grads finite, zero_grads clears.
// ===========================================================================
#[test]
fn one_step_update_moves_weights_and_clears_grads() {
    let cfg = YoloConfig::tiny(3);
    let b = 1u32;
    let init = init_weights(&cfg, 7);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&fixed_image(&cfg, b, 0x1234));
    model.set_targets(&fixed_gts());

    let names: Vec<String> = cfg.full_param_list().iter().map(|(n, _)| n.clone()).collect();
    let before: HashMap<String, Vec<f32>> =
        names.iter().map(|n| (n.clone(), model.read_weight(n))).collect();

    // One full detection train step.
    model.zero_grads();
    let l = model.forward();
    assert!(l.is_finite() && l > 0.0, "detection loss must be finite positive, got {l}");
    model.backward();

    // Every grad finite (this is the wiring guard P4 also checks; cheap to keep).
    let mut grad_l2 = 0.0f64;
    for n in &names {
        let g = model.read_grad(n);
        assert!(g.iter().all(|v| v.is_finite()), "non-finite grad in {n}");
        grad_l2 += g.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
    }
    assert!(grad_l2 > 0.0, "all grads were exactly zero — backward produced no signal");

    model.adamw_step(1, 1e-2, 0.0, Some(10.0), 1.0);
    model.poll_wait();

    // At least SOME trainable weight changed (a sizeable fraction, since AdamW's
    // first step moves every param with a non-zero grad by ~lr).
    let mut changed = 0usize;
    for n in &names {
        let after = model.read_weight(n);
        let b0 = &before[n];
        let moved = after.iter().zip(b0).any(|(a, b)| (a - b).abs() > 1e-7);
        if moved {
            changed += 1;
        }
    }
    assert!(
        changed > names.len() / 2,
        "too few params updated after one step: {changed}/{} (expected most)",
        names.len()
    );

    // zero_grads must clear the grad buffers.
    model.zero_grads();
    for n in &names {
        let g = model.read_grad(n);
        assert!(g.iter().all(|v| *v == 0.0), "zero_grads did not clear grad of {n}");
    }
    println!("one-step update: loss {l:.4}, {changed}/{} params moved, zero_grads OK", names.len());
}

// ===========================================================================
// C. Invalid-label rejection / empty handling.
// ===========================================================================

/// Test-only label validator establishing the DEFINED behaviour for malformed
/// annotations before they reach the (panic-on-out-of-range-class) assigner:
///   * class id `>= nc`              -> REJECTED (dropped),
///   * non-positive width or height  -> REJECTED (zero/negative area),
///   * center/size out of `[0,1]`    -> CLIPPED to a valid in-frame box; if the
///     clipped box collapses to zero area it is then rejected.
///
/// Returns the kept (valid) boxes. This is the contract a real loader must
/// uphold; the model's `set_targets` assumes already-valid in-range labels.
fn sanitize_gts(gts: &[GtBox], nc: u32) -> Vec<GtBox> {
    let mut out = Vec::new();
    for g in gts {
        if g.cls >= nc {
            continue; // out-of-range class: reject (would panic the assigner).
        }
        if !(g.w.is_finite() && g.h.is_finite() && g.cx.is_finite() && g.cy.is_finite()) {
            continue; // non-finite: reject (would poison the loss).
        }
        if g.w <= 0.0 || g.h <= 0.0 {
            continue; // zero/negative area: reject.
        }
        // Clip the box to the unit frame via its corners, then recompute center+wh.
        let x1 = (g.cx - g.w * 0.5).clamp(0.0, 1.0);
        let y1 = (g.cy - g.h * 0.5).clamp(0.0, 1.0);
        let x2 = (g.cx + g.w * 0.5).clamp(0.0, 1.0);
        let y2 = (g.cy + g.h * 0.5).clamp(0.0, 1.0);
        let (w, h) = (x2 - x1, y2 - y1);
        if w <= 0.0 || h <= 0.0 {
            continue; // collapsed to zero area after clipping: reject.
        }
        out.push(GtBox { img: g.img, cls: g.cls, cx: (x1 + x2) * 0.5, cy: (y1 + y2) * 0.5, w, h });
    }
    out
}

#[test]
fn sanitize_rejects_and_clips_malformed_labels() {
    let nc = 3u32;
    let raw = vec![
        GtBox { img: 0, cls: 0, cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },   // valid -> kept
        GtBox { img: 0, cls: 9, cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },   // class>=nc -> reject
        GtBox { img: 0, cls: 1, cx: 0.5, cy: 0.5, w: -0.2, h: 0.4 },  // neg width -> reject
        GtBox { img: 0, cls: 1, cx: 0.5, cy: 0.5, w: 0.0, h: 0.4 },   // zero area -> reject
        GtBox { img: 0, cls: 2, cx: 0.9, cy: 0.9, w: 0.6, h: 0.6 },   // overflows frame -> clipped, kept
        GtBox { img: 0, cls: 1, cx: 1.5, cy: 1.5, w: 0.2, h: 0.2 },   // fully outside -> reject (collapses)
        GtBox { img: 0, cls: 0, cx: f32::NAN, cy: 0.5, w: 0.3, h: 0.3 }, // NaN -> reject
    ];
    let kept = sanitize_gts(&raw, nc);
    assert_eq!(kept.len(), 2, "expected exactly the 2 valid/clippable boxes, kept {kept:?}");
    // The clipped one stays in-frame with positive area and unchanged class.
    let clipped = kept.iter().find(|g| g.cls == 2).expect("clipped box dropped");
    assert!(clipped.cx - clipped.w * 0.5 >= -1e-6 && clipped.cx + clipped.w * 0.5 <= 1.0 + 1e-6);
    assert!(clipped.w > 0.0 && clipped.h > 0.0);
    // Every kept box is now safe for the model (class < nc, positive area, in-frame).
    for g in &kept {
        assert!(g.cls < nc && g.w > 0.0 && g.h > 0.0);
    }
}

#[test]
fn empty_and_degenerate_targets_give_finite_loss_and_backward() {
    let cfg = YoloConfig::tiny(3);
    let b = 1u32;
    let init = init_weights(&cfg, 11);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&fixed_image(&cfg, b, 0xABCD));

    let names: Vec<String> = cfg.full_param_list().iter().map(|(n, _)| n.clone()).collect();
    let check_finite = |model: &Yolo, label: &str| -> f32 {
        let l = model.forward();
        assert!(l.is_finite() && l >= 0.0, "{label}: loss must be finite >= 0, got {l}");
        model.zero_grads();
        model.backward();
        for n in &names {
            let g = model.read_grad(n);
            assert!(g.iter().all(|v| v.is_finite()), "{label}: non-finite grad in {n}");
        }
        l
    };

    // 1) Empty annotation: every anchor is background. Extends the P4 smoke from
    //    the loss seam to the full `set_targets` data path.
    model.set_targets(&[]);
    let l_empty = check_finite(&model, "empty");

    // 2) Sanitized-from-malformed targets: a real loader path. The valid+clipped
    //    boxes train normally; the rejects never reach the assigner.
    let raw = vec![
        GtBox { img: 0, cls: 0, cx: 0.4, cy: 0.5, w: 0.4, h: 0.5 },
        GtBox { img: 0, cls: 7, cx: 0.5, cy: 0.5, w: 0.3, h: 0.3 }, // class>=nc — dropped
        GtBox { img: 0, cls: 1, cx: 0.7, cy: 0.6, w: -0.1, h: 0.2 }, // neg width — dropped
    ];
    let clean = sanitize_gts(&raw, cfg.nc);
    assert!(!clean.is_empty());
    model.set_targets(&clean);
    let l_clean = check_finite(&model, "sanitized");

    // 3) Degenerate-but-in-range box (positive but TINY area, valid class): the
    //    assigner finds no in-box anchor center -> treats it as background, NOT a
    //    NaN. This proves the loss tolerates a sub-pixel GT without sanitization.
    model.set_targets(&[GtBox { img: 0, cls: 2, cx: 0.5, cy: 0.5, w: 1e-4, h: 1e-4 }]);
    let l_tiny = check_finite(&model, "tiny-box");

    println!("invalid-label handling: loss empty={l_empty:.4} sanitized={l_clean:.4} tiny-box={l_tiny:.4} (all finite, grads finite)");
}

// ===========================================================================
// D. Box-format round trips NOT already in boxmath's own unit tests.
//    (boxmath covers: dist<->xyxy, xywhn centered, letterbox square/wide/tall.)
//    Here: normalized xywh <-> pixel xyxy across full-image / 1px / border /
//    non-square edge cases.
// ===========================================================================
/// One normalized-box conversion case:
/// `(case name, cx, cy, w, h, img_size, expected xyxy)`.
type XywhnCase<'a> = (&'a str, f32, f32, f32, f32, f32, [f32; 4]);

#[test]
fn xywhn_pixel_xyxy_edge_cases_round_trip() {
    // (img_w, img_h via a square `img_size` arg — xywhn_to_xyxy uses one scale,
    // so non-square is exercised by an explicit pixel build/compare below.)
    let cases: &[XywhnCase<'_>] = &[
        // name, cx, cy, w, h, img_size, expected xyxy
        ("full-image", 0.5, 0.5, 1.0, 1.0, 128.0, [0.0, 0.0, 128.0, 128.0]),
        ("centered-half", 0.5, 0.5, 0.5, 0.5, 200.0, [50.0, 50.0, 150.0, 150.0]),
        // 1px box in a 128px image: w=h=1/128 centered on pixel (64,64) center.
        ("one-pixel", 64.5 / 128.0, 64.5 / 128.0, 1.0 / 128.0, 1.0 / 128.0, 128.0, [64.0, 64.0, 65.0, 65.0]),
        // border-touching top-left corner.
        ("border-tl", 0.1, 0.1, 0.2, 0.2, 100.0, [0.0, 0.0, 20.0, 20.0]),
        // border-touching bottom-right corner.
        ("border-br", 0.9, 0.9, 0.2, 0.2, 100.0, [80.0, 80.0, 100.0, 100.0]),
    ];
    for &(name, cx, cy, w, h, sz, want) in cases {
        let got = xywhn_to_xyxy(cx, cy, w, h, sz);
        for k in 0..4 {
            assert!((got[k] - want[k]).abs() < 1e-3, "{name} xyxy[{k}] = {} want {}", got[k], want[k]);
        }
        // xyxy -> (cx,cy,w,h) in pixels, then back to normalized -> recover input.
        let pix = xyxy_to_xywh(got);
        let (rcx, rcy, rw, rh) = (pix[0] / sz, pix[1] / sz, pix[2] / sz, pix[3] / sz);
        assert!((rcx - cx).abs() < 1e-3, "{name} cx round-trip {rcx} vs {cx}");
        assert!((rcy - cy).abs() < 1e-3, "{name} cy round-trip {rcy} vs {cy}");
        assert!((rw - w).abs() < 1e-3, "{name} w round-trip {rw} vs {w}");
        assert!((rh - h).abs() < 1e-3, "{name} h round-trip {rh} vs {h}");
    }

    // Non-square: a normalized box on a 200x100 image must map with independent
    // x/y scales. xywhn_to_xyxy is single-scale, so build the pixel box directly
    // and confirm xyxy_to_xywh inverts it (the format round trip we actually use
    // when a loader carries normalized coords on a non-square image).
    let (iw, ih) = (200.0f32, 100.0f32);
    let (cx, cy, w, h) = (0.3f32, 0.7f32, 0.4f32, 0.5f32);
    let xyxy = [
        (cx - w * 0.5) * iw,
        (cy - h * 0.5) * ih,
        (cx + w * 0.5) * iw,
        (cy + h * 0.5) * ih,
    ];
    let pix = xyxy_to_xywh(xyxy);
    assert!((pix[0] / iw - cx).abs() < 1e-4 && (pix[1] / ih - cy).abs() < 1e-4);
    assert!((pix[2] / iw - w).abs() < 1e-4 && (pix[3] / ih - h).abs() < 1e-4);
}

// ===========================================================================
// E. Frozen-layer / fine-tune: backbone UNCHANGED, head changes.
//    The model has no freeze API, so per the plan we use the TEST-ONLY snapshot
//    approach: snapshot backbone weights before each adamw_step and restore them
//    after, leaving the optimiser to update only neck+head. This avoids any graph
//    change while still proving the head learns independently of a frozen trunk.
// ===========================================================================
#[test]
fn frozen_backbone_unchanged_head_changes() {
    let cfg = YoloConfig::tiny(3);
    let b = 1u32;
    let init = init_weights(&cfg, 313);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Detection);
    model.set_image(&fixed_image(&cfg, b, 0x5151));
    model.set_targets(&fixed_gts());

    let names: Vec<String> = cfg.full_param_list().iter().map(|(n, _)| n.clone()).collect();
    let is_backbone = |n: &str| n.starts_with("backbone.");
    let backbone_names: Vec<&String> = names.iter().filter(|n| is_backbone(n)).collect();
    let head_names: Vec<&String> = names.iter().filter(|n| n.starts_with("head.")).collect();
    assert!(!backbone_names.is_empty() && !head_names.is_empty());

    // Reference snapshot of the WHOLE backbone at start.
    let bb0: HashMap<String, Vec<f32>> =
        backbone_names.iter().map(|n| ((*n).clone(), model.read_weight(n))).collect();
    let head0: HashMap<String, Vec<f32>> =
        head_names.iter().map(|n| ((*n).clone(), model.read_weight(n))).collect();

    let steps = 5u32;
    for step in 1..=steps {
        // Snapshot backbone, take a normal optimiser step, then RESTORE backbone:
        // net effect is that only neck+head weights move (a test-only freeze).
        let snap: Vec<(String, Vec<f32>)> =
            backbone_names.iter().map(|n| ((*n).clone(), model.read_weight(n))).collect();
        model.zero_grads();
        let l = model.forward();
        assert!(l.is_finite(), "step {step}: non-finite loss");
        model.backward();
        model.adamw_step(step, 1e-2, 0.0, Some(10.0), 1.0);
        model.poll_wait();
        for (n, w) in &snap {
            model.write_weight(n, w); // freeze: undo the backbone update.
        }
    }

    // Backbone must be byte-identical to the start snapshot (we restored it every
    // step), head must have moved.
    let mut bb_max = 0.0f32;
    for n in &backbone_names {
        let now = model.read_weight(n);
        for (a, b) in now.iter().zip(&bb0[*n]) {
            bb_max = bb_max.max((a - b).abs());
        }
    }
    let mut head_moved = 0usize;
    for n in &head_names {
        let now = model.read_weight(n);
        if now.iter().zip(&head0[*n]).any(|(a, b)| (a - b).abs() > 1e-6) {
            head_moved += 1;
        }
    }
    assert!(bb_max == 0.0, "frozen backbone changed: max abs delta {bb_max:.3e}");
    assert!(
        head_moved > 0,
        "head did not change while backbone frozen ({}/{} head tensors moved)",
        head_moved,
        head_names.len()
    );
    println!(
        "frozen-backbone: backbone delta {bb_max:.3e} (0 expected), {head_moved}/{} head tensors moved over {steps} steps",
        head_names.len()
    );
}
