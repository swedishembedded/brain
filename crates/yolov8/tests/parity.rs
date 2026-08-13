// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P8 reference-parity test: load an exported Ultralytics `yolov8n` into the
//! brain detector and compare its forward output against PyTorch-dumped
//! activations, layer (stage) by stage, reporting the FIRST divergence.
//!
//! ## Gating
//! This test CANNOT run in the brain CI box (no torch, no GPU, and a full 640
//! forward on the CPU JIT is very slow). It is therefore gated on an env var
//! pointing at the exported weight file produced by
//! `tools/yolo_export/export_yolov8.py`:
//!
//! ```text
//! YOLO_PARITY_WEIGHTS=yolov8n.brain.safetensors \
//! YOLO_PARITY_ACTS=yolov8n.acts.safetensors \
//!     cargo test -p brain-yolo --test parity -- --nocapture
//! ```
//!
//! When `YOLO_PARITY_WEIGHTS` is unset or the file is missing the test prints a
//! skip notice and returns OK (so plain `cargo test` is green everywhere).
//!
//! `YOLO_PARITY_ACTS` (optional) is the activation dump (a safetensors file): a
//! fixed preprocessed `input` tensor plus one tensor per backbone/neck stage and
//! the head scale outputs, keyed by brain stage name. When present, the test feeds
//! the identical `input` and compares the brain forward's publicly-readable
//! output (the per-scale head logits, via [`Yolo::raw_logits`]) against the
//! dumped head-scale activations. (Finer per-internal-buffer parity needs
//! buffer-accessor hooks the model does not yet expose — see README.)
//!
//! ## IMPORTANT — known divergence
//! The current `YoloConfig::yolov8n()` is a reduced-width/-depth approximation of
//! the canonical Ultralytics yolov8n (see crates/yolo/README.md "Discrepancies").
//! Until the brain graph is reconciled to true yolov8n, the export script will
//! refuse to write a shape-correct file, so this test has nothing to load and
//! stays in the skip path. The scaffolding below is what runs once that
//! reconciliation lands.

use std::collections::HashMap;

use yolov8::{LossMode, Yolo, YoloConfig};

/// `max |a - b|` over equal-length slices; `f32::INFINITY` if lengths differ.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn yolov8n_reference_parity() {
    let weights_path = match std::env::var("YOLO_PARITY_WEIGHTS") {
        Ok(p) if std::path::Path::new(&p).exists() => p,
        _ => {
            eprintln!(
                "SKIP yolov8n_reference_parity: set YOLO_PARITY_WEIGHTS to an exported \
                 yolov8n.brain.safetensors (see crates/yolo/README.md). Nothing to do here."
            );
            return;
        }
    };

    // --- load the exported weights container ---
    let container = checkpoint::load(&weights_path);
    let weights: HashMap<String, Vec<f32>> = container.by_role(""); // export writes role=""

    // --- build the brain detector with batch 1, eval-ready ---
    let cfg = YoloConfig::yolov8n();
    let b = 1u32;
    // Init with the loaded weights directly (ParamStore takes the init map).
    let model = Yolo::new(cfg.clone(), b, cfg.input, &weights);
    model.set_mode(LossMode::Proxy); // we only read forward outputs; no loss needed

    // Sanity: every brain param must have been supplied by the export.
    let supplied: std::collections::HashSet<&String> = weights.keys().collect();
    let plist = cfg.full_param_list();
    let missing: Vec<&String> =
        plist.iter().map(|(n, _)| n).filter(|n| !supplied.contains(n)).collect();
    assert!(
        missing.is_empty(),
        "exported file is missing {} brain params, e.g. {:?}",
        missing.len(),
        missing.iter().take(8).collect::<Vec<_>>()
    );

    // --- input + reference activations ---
    let acts = std::env::var("YOLO_PARITY_ACTS").ok().and_then(|p| {
        if std::path::Path::new(&p).exists() {
            Some(checkpoint::load(&p))
        } else {
            eprintln!("YOLO_PARITY_ACTS set but file missing; using a fixed synthetic input.");
            None
        }
    });

    let n_in = (b * 3 * cfg.input * cfg.input) as usize;
    // Safetensors carries no role, so every tensor loads under role "".
    let input: Vec<f32> = match acts.as_ref().and_then(|c| c.find("input", "")) {
        Some(v) => {
            assert_eq!(v.len(), n_in, "dumped input has wrong size");
            v.clone()
        }
        None => (0..n_in).map(|i| ((i % 255) as f32) / 255.0).collect(),
    };

    model.set_image(&input);
    let _ = model.forward();
    let (cls, boxl) = model.raw_logits();

    // --- stage-by-stage comparison (head scales are the readable end stages) ---
    const TOL: f32 = 1e-3;
    let Some(acts) = acts else {
        eprintln!(
            "parity: weights loaded + forward ran ({} cls logits, {} box logits). \
             No YOLO_PARITY_ACTS dump given, so no reference comparison performed.",
            cls.len(),
            boxl.len()
        );
        return;
    };

    // The dump stores each head scale's cls/reg output under its module name; the
    // brain `raw_logits` are the concatenated per-anchor logits. We compare the
    // concatenated head outputs as the end-to-end parity signal, reporting the
    // first branch that diverges.
    let mut first_div: Option<(String, f32)> = None;
    for (name, got) in [("head.cls", &cls), ("head.reg", &boxl)] {
        if let Some(reference) = acts.find(name, "") {
            let e = max_abs_err(got, reference);
            eprintln!("stage {name}: max_abs_err = {e:.3e}");
            if e >= TOL && first_div.is_none() {
                first_div = Some((name.to_string(), e));
            }
        } else {
            eprintln!("stage {name}: no reference in acts dump (skipped)");
        }
    }

    if let Some((stage, e)) = first_div {
        panic!("FIRST divergent stage: {stage} (max_abs_err {e:.3e} >= tol {TOL:.0e})");
    }
    eprintln!("parity OK: all compared stages within tol {TOL:.0e}");
}
