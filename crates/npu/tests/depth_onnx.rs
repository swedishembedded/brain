// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth ONNX export parity: brain's CPU forward vs the exported graph run on
//! OpenVINO (CPU first — always available — then the NPU when present).
//!
//! Env-gated on the real checkpoint (ZIPDEPTH_NPU_PTH points at zipdepth_base_npu.pth,
//! the blend/where_conv variant this exporter emits). Skips cleanly when unset.
use std::collections::HashMap;

use depth::{import, Predictor, ZipConfig};
use gpu_core::Gpu;
use npu::build_depth_graph;
use npu::openvino::{NpuConfig, NpuDevice, NpuSession};
use onnx::GraphBuilder;

fn npu_pth() -> Option<String> {
    std::env::var("ZIPDEPTH_NPU_PTH").ok()
}

/// cosine similarity of two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn export_and_run(device: NpuDevice) {
    let Some(path) = npu_pth() else {
        eprintln!("SKIP: set ZIPDEPTH_NPU_PTH=<zipdepth_base_npu.pth>");
        return;
    };
    let cfg = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
    let sz = cfg.input;

    // brain's own weights + CPU reference depth (letterbox-free: a centred square
    // input at exactly the model resolution, so the predictor's letterbox is the
    // identity and the ONNX 'input' gets the same CHW).
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let init: HashMap<String, Vec<f32>> = import::load(&path, &cfg).expect("import npu checkpoint");
    let ps = import::load_into(&gpu, &path, &cfg).expect("load_into");
    let predictor = Predictor::new(&gpu, cfg.clone(), ps);

    // A deterministic square RGB image [H=W=sz].
    let hwc: Vec<f32> = (0..(sz * sz * 3)).map(|i| ((i * 37 % 251) as f32) / 251.0).collect();
    let ref_depth = predictor.predict(&hwc, sz, sz); // [sz*sz], frame grid == model grid

    // Export ONNX and run it on OpenVINO.
    let mut g = GraphBuilder::new("zipdepth");
    build_depth_graph(&cfg, &init, &mut g);
    let bytes = g.finish();

    let ncfg = NpuConfig { device, allow_fallback: false, ..Default::default() };
    let mut sess = match NpuSession::load_bytes(&bytes, &ncfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: {device:?} unavailable ({e})");
            return;
        }
    };
    // The ONNX input is CHW; predict() letterboxes internally, but for a square
    // sz×sz frame the letterbox is identity, so feed the same CHW the model saw.
    let chw = imaging::pixels::hwc_to_chw(&hwc, 3, sz as usize, sz as usize);
    let out = sess.run(&chw, [1, 3, sz as usize, sz as usize]).expect("inference");
    let (_, shape, data) = &out.tensors[0];
    assert_eq!(shape, &vec![1, 1, sz as usize, sz as usize], "output must be [1,1,H,W]");

    let cos = cosine(&ref_depth, data);
    let max_abs = ref_depth.iter().zip(data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean_ref = ref_depth.iter().sum::<f32>() / ref_depth.len() as f32;
    println!(
        "{device:?}: cosine(brain-CPU, ONNX) = {cos:.5}   max|Δ| = {max_abs:.4}   mean_depth = {mean_ref:.4}   dev={}",
        sess.device()
    );
    assert!(cos > 0.999, "{device:?}: exported graph diverges from brain's forward (cosine {cos:.5})");
}

/// OpenVINO CPU: always available where OpenVINO is installed. Isolates "is the
/// exported GRAPH correct" from "does the NPU run it".
#[test]
fn export_matches_brain_on_openvino_cpu() {
    export_and_run(NpuDevice::Cpu);
}

/// The real Intel NPU.
#[test]
fn export_matches_brain_on_npu() {
    export_and_run(NpuDevice::Npu);
}
