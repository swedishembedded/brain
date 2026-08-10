// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware-free INT8 accuracy-parity gate. Calibrate a tiny YOLO, then compare
//! the fp32 reference head logits against the INT8 fake-quant simulation (the
//! exact arithmetic the exported INT8 ONNX runs on the NPU). High cosine
//! similarity proves the brain-native quantization scheme is faithful — measured
//! entirely on the CPU backend, no NPU / no OpenVINO.

use yolo::{Yolo, YoloConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

fn tmp(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("brain_npu_parity_{}_{}", std::process::id(), name))
        .to_string_lossy()
        .into_owned()
}

/// Deterministic pseudo-random CHW image in [0,1) — the unified LCG
/// (audit F39/F40).
fn rand_chw(side: usize, seed: u64) -> Vec<f32> {
    let mut l = data::rng::Lcg::new(seed);
    (0..3 * side * side).map(|_| l.unit()).collect()
}

#[test]
fn int8_sim_tracks_fp32() {
    if skip() {
        return;
    }
    let cfg = YoloConfig::tiny(3);
    let side = cfg.input as usize;
    let wpath = tmp("tiny.safetensors");

    // A trained-ish model would be ideal, but quantization fidelity is a property
    // of the weights' dynamic range, not their task accuracy; an init model is a
    // valid (and fast) probe. Save it.
    let init = yolo::init_weights(&cfg, 0x5151);
    Yolo::new(cfg.clone(), 1, 0, &init).save(&wpath);

    // Calibrate over a handful of images.
    let calib: Vec<Vec<f32>> = (0..6).map(|i| rand_chw(side, 100 + i)).collect();
    let quant = npu::calibrate_from_weights(&wpath, &calib);
    assert!(!quant.is_empty(), "calibration produced no activation scales");

    // Compare fp32 vs INT8-sim logits on held-out inputs.
    let mut worst_cls = 1.0f32;
    let mut worst_box = 1.0f32;
    let mut total_diff = 0.0f64;
    for i in 0..3 {
        let chw = rand_chw(side, 9000 + i);
        let (cls_f, box_f) = npu::reference_logits(&wpath, &chw);
        let (cls_q, box_q) = npu::simulate_logits(&wpath, &chw, &quant);
        let cc = npu::sim::cosine(&cls_f, &cls_q);
        let bc = npu::sim::cosine(&box_f, &box_q);
        worst_cls = worst_cls.min(cc);
        worst_box = worst_box.min(bc);
        total_diff += cls_f.iter().zip(&cls_q).map(|(a, b)| (a - b).abs() as f64).sum::<f64>();
        total_diff += box_f.iter().zip(&box_q).map(|(a, b)| (a - b).abs() as f64).sum::<f64>();
    }
    eprintln!(
        "INT8 parity: worst cls cosine = {worst_cls:.5}, worst box cosine = {worst_box:.5}, total |Δ| = {total_diff:.4}"
    );

    // The simulation must actually quantize (guard against a no-op masquerading
    // as perfect parity).
    assert!(total_diff > 1e-6, "INT8 simulation is a no-op (logits bit-identical to fp32)");
    // INT8 (per-channel weights + per-tensor activations) should track fp32
    // closely even through the full backbone+neck+head.
    assert!(worst_cls > 0.95, "cls logits diverged under INT8: cosine {worst_cls}");
    assert!(worst_box > 0.95, "box logits diverged under INT8: cosine {worst_box}");

    std::fs::remove_file(&wpath).ok();
}
