// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 network the Rockchip NPU runs, simulated on the host with its exact
//! integers (`npu::simulate_detect`), must find what the float network finds
//! on a real photograph: the same objects, the same classes, boxes that
//! overlap the reference's and confidences close to them.
//!
//! Needs the converted checkpoint (`local/yolov5s-relu`), its NPU calibration
//! (`make crates/yolov5/examples/npu-calibrate/run`) and the goldens from
//! `tools/goldens/yolov5_dump_reference.py` (the photograph and onnxruntime's
//! detections at 320). A missing fixture is a named skip.

use std::path::Path;

use checkpoint::mmap::MmapSafetensors;

fn iou(a: &[f32], b: &[f32]) -> f32 {
    let (x1, y1, x2, y2) = (a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area = |d: &[f32]| (d[2] - d[0]) * (d[3] - d[1]);
    inter / (area(a) + area(b) - inter)
}

#[test]
fn int8_detections_match_the_float_network_on_a_photograph() {
    let ckpt = Path::new(&brain_testutil::model_dir("local/yolov5s-relu").unwrap_or_default()).join("model.safetensors");
    if !ckpt.exists() || !yolov5::npu::calibration_path(&ckpt).exists() {
        brain_testutil::skip("local/yolov5s-relu/model.safetensors or its .npu-calibration.json absent: convert it, then run make crates/yolov5/examples/npu-calibrate/run");
        return;
    }
    let golden_path = Path::new(&brain_testutil::testdata("golden/yolov5")).join("golden.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip("testdata/golden/yolov5 absent: run tools/goldens/yolov5_dump_reference.py");
        return;
    }
    let golden = MmapSafetensors::open(&golden_path).unwrap();
    let shape = golden.shape("image_rgb").unwrap().to_vec();
    let (h, w) = (shape[0] as u32, shape[1] as u32);
    // The dumper stores the decoded photograph as 0..255 floats.
    let rgb: Vec<u8> = golden.tensor_f32("image_rgb").unwrap().iter().map(|v| v.round() as u8).collect();
    let want = golden.tensor_f32("img320_dets").unwrap();

    let got = yolov5::npu::simulate_detect(&ckpt, &rgb, w, h, 0.25, 0.45, 300).unwrap();
    println!("int8: {got:?}\nfloat: {:?}", want.chunks(6).collect::<Vec<_>>());
    let strong: Vec<&[f32]> = want.chunks(6).filter(|d| d[4] > 0.5).collect();
    assert!(!strong.is_empty(), "the golden photograph has confident float detections");
    for f in strong {
        let best = got.iter().filter(|d| d[5] == f[5]).map(|d| (iou(d, f), d[4])).fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        assert!(best.0 > 0.8, "float detection {f:?} has no int8 match of its class (best IoU {})", best.0);
        assert!((best.1 - f[4]).abs() < 0.15, "float detection {f:?}: int8 confidence {} differs by more than 0.15", best.1);
    }
}
