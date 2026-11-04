// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware-gated tier-B test: run the exported fp32 ONNX on a real OpenVINO
//! device and assert its raw head tensors match brain's own engine forward. This
//! validates the ONNX *topology numerics* (Resize/Split/Concat/fold) end-to-end —
//! the one thing the hardware-free tests can't, since there is no ONNX runtime in
//! CI.
//!
//! Skipped unless `BRAIN_NPU_AVAILABLE` is set (and OpenVINO + a device are
//! present). Choose the device with `BRAIN_NPU_DEVICE` (default `NPU`); set it to
//! `CPU` to validate the graph on the OpenVINO CPU plugin without an NPU.

use yolov8::head::repack_heads_to_flat;
use yolov8::{Yolo, YoloConfig};

use npu::openvino::{NpuConfig, NpuDevice, NpuSession};

fn enabled() -> bool {
    std::env::var("BRAIN_NPU_AVAILABLE").map(|v| !v.is_empty()).unwrap_or(false)
}

use model::hostmath::cosine;

fn rand_chw(side: usize, seed: u64) -> Vec<f32> {
    // The unified deterministic LCG (audit F39/F40).
    let mut l = data::rng::Lcg::new(seed);
    (0..3 * side * side).map(|_| l.unit()).collect()
}

#[test]
fn npu_fp32_matches_engine() {
    if !enabled() {
        return;
    }
    let cfg = YoloConfig::tiny(3);
    let side = cfg.input as usize;
    let wpath = std::env::temp_dir().join(format!("brain_npu_tierb_{}.safetensors", std::process::id()));
    let wpath = wpath.to_string_lossy().into_owned();

    let init = yolov8::init_weights(&cfg, 0x1234);
    Yolo::new(cfg.clone(), 1, 0, &init).save(&wpath);

    let chw = rand_chw(side, 42);

    // Engine reference.
    let (cls_ref, box_ref) = npu::reference_logits(&wpath, &chw);

    // NPU/OpenVINO path.
    let device = std::env::var("BRAIN_NPU_DEVICE")
        .ok()
        .and_then(|s| NpuDevice::parse(&s))
        .unwrap_or(NpuDevice::Npu);
    let bytes = npu::build_fp32_bytes(&wpath, None, onnx::DEFAULT_OPSET);
    let mut session = NpuSession::load_bytes(
        &bytes,
        &NpuConfig { device, allow_fallback: true, ..Default::default() },
    )
    .expect("compile ONNX on device");
    let heads = session.run(&chw, [1, 3, side, side]).expect("infer");

    // Repack the NPU per-scale outputs the same way the engine flattens.
    let nc = cfg.nc as usize;
    let four_rm = 4 * cfg.reg_max as usize;
    let pick = |which: &str, c: usize| -> Vec<f32> {
        let scales: Vec<(Vec<f32>, u32, u32)> = (0..3)
            .map(|s| {
                let name = format!("head.{s}.{which}");
                let (_, shape, data) =
                    heads.tensors.iter().find(|(n, _, _)| *n == name).expect("output");
                (data.clone(), shape[2] as u32, shape[3] as u32)
            })
            .collect();
        let a: usize = scales.iter().map(|(_, h, w)| (h * w) as usize).sum();
        let refs: Vec<(&[f32], u32, u32)> = scales.iter().map(|(d, h, w)| (d.as_slice(), *h, *w)).collect();
        repack_heads_to_flat(&refs, 1, c, a)
    };
    let cls_npu = pick("cls", nc);
    let box_npu = pick("reg", four_rm);

    let cc = cosine(&cls_ref, &cls_npu);
    let bc = cosine(&box_ref, &box_npu);
    eprintln!("tier-B NPU vs engine ({}): cls cosine {cc:.5}, box cosine {bc:.5}", session.device());
    // Allow for the device running fp16 internally.
    assert!(cc > 0.99, "cls topology mismatch NPU vs engine: cosine {cc}");
    assert!(bc > 0.99, "box topology mismatch NPU vs engine: cosine {bc}");

    std::fs::remove_file(&wpath).ok();
}
