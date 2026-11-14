// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware-free INT8 accuracy-parity gate. Calibrate a tiny YOLO, then compare
//! the fp32 reference head logits against the INT8 fake-quant simulation (the
//! exact arithmetic the exported INT8 ONNX runs on the NPU). High cosine
//! similarity proves the brain-native quantization scheme is faithful — measured
//! entirely on the CPU backend, no NPU / no OpenVINO.

use yolov8::{Yolo, YoloConfig};

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
    let init = yolov8::init_weights(&cfg, 0x5151);
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

/// The first output of the ONNX graph `bytes` on `chw` under onnxruntime, the
/// reference semantics of ONNX's Q/DQ operators; `None` (after saying why)
/// when python3 or onnxruntime is not there to run it.
fn onnxruntime_first_output(bytes: &[u8], chw: &[f32], side: usize, name: &str) -> Option<Vec<f32>> {
    let model = tmp(&format!("{name}.onnx"));
    let input = tmp(&format!("{name}.bin"));
    std::fs::write(&model, bytes).expect("write model");
    std::fs::write(&input, chw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).expect("write input");
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/goldens/onnx_eval.py");
    let out = std::process::Command::new("python3").arg(script).arg(&model).arg(&input).args(["--shape", &format!("1,3,{side},{side}")]).output();
    std::fs::remove_file(&model).ok();
    std::fs::remove_file(&input).ok();
    let Ok(out) = out else {
        brain_testutil::skip_unavailable("could not run python3 - the exported INT8 graph was not run");
        return None;
    };
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("No module named") {
            brain_testutil::skip_unavailable(&format!("onnxruntime not installed - the exported INT8 graph was not run\n{err}"));
            return None;
        }
        panic!("onnx_eval.py failed:\n{err}");
    }
    Some(out.stdout.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// The exported INT8 Q/DQ graph, run by onnxruntime (the operators' reference
/// semantics), tracks the exported fp32 graph on held-out inputs - and
/// differs from it, so the export really quantizes. This checks the graph a
/// device receives (scale placement, per-channel axes, zero points), where
/// `int8_sim_tracks_fp32` checks brain's own simulation of it.
#[test]
fn the_exported_int8_graph_tracks_fp32_under_onnxruntime() {
    if skip() {
        return;
    }
    let cfg = YoloConfig::tiny(3);
    let side = cfg.input as usize;
    let wpath = tmp("int8-export.safetensors");
    Yolo::new(cfg.clone(), 1, 0, &yolov8::init_weights(&cfg, 0x5151)).save(&wpath);
    let calib: Vec<Vec<f32>> = (0..6).map(|i| rand_chw(side, 100 + i)).collect();
    let quant = npu::calibrate_from_weights(&wpath, &calib);
    let fp32 = npu::try_build_fp32_bytes(&wpath, None, onnx::DEFAULT_OPSET).expect("fp32 export");
    let int8 = npu::try_build_int8_bytes(&wpath, &quant, None, onnx::DEFAULT_OPSET).expect("int8 export");
    std::fs::remove_file(&wpath).ok();

    let mut worst = 1.0f32;
    for i in 0..3 {
        let chw = rand_chw(side, 9000 + i);
        let Some(want) = onnxruntime_first_output(&fp32, &chw, side, "fp32") else { return };
        let Some(got) = onnxruntime_first_output(&int8, &chw, side, "int8") else { return };
        assert_eq!(got.len(), want.len(), "the INT8 graph's first output has another shape");
        assert!(got.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-6), "the INT8 graph computes exactly the fp32 numbers: it does not quantize");
        worst = worst.min(npu::sim::cosine(&want, &got));
    }
    eprintln!("exported INT8 vs fp32 under onnxruntime: worst cosine {worst:.5}");
    assert!(worst > 0.95, "the exported INT8 graph diverges from fp32: cosine {worst}");
}
