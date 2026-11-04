// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity of the fp32 reference forward and the host decode against
//! onnxruntime, over the real `yolov5s_relu` export.
//!
//! Fixtures (none are committed):
//!   * the converted checkpoint, `local/yolov5s-relu` in the model store, made by
//!     `tools/yolov5_export/export_yolov5.py`;
//!   * the goldens under `testdata/golden/yolov5/`, made by
//!     `tools/goldens/yolov5_dump_reference.py` from the same ONNX file.
//!
//! A missing fixture is a named skip (`brain_testutil::skip`), fatal under
//! `BRAIN_REQUIRE_FIXTURES=1`.
//!
//! ## Tolerances
//!
//! Head outputs are sigmoid outputs in `[0, 1]`, so an absolute tolerance is the
//! natural measure. fp32 accumulates differently here (the WGSL/CPU conv kernels)
//! than in onnxruntime's MLAS, across 60 layers; [`HEAD_TOL`] is that bound,
//! measured with margin (the observed maxima are about 2e-5, at 640 on a real
//! image), and the test prints the per-output maxima. Detections are compared
//! field by field: identical count and classes, confidence within [`CONF_TOL`], boxes within [`BOX_TOL_PX`]
//! original-image pixels.

use std::path::{Path, PathBuf};

use checkpoint::mmap::MmapSafetensors;
use yolov5::decode::{detections, HeadMap};
use yolov5::{Checkpoint, Yolov5};

const DUMPER: &str = "tools/goldens/yolov5_dump_reference.py";
const HEAD_TOL: f32 = 1e-4;
const CONF_TOL: f32 = 1e-3;
const BOX_TOL_PX: f32 = 0.1;
const CONF: f32 = 0.25;
const IOU: f32 = 0.45;
const MAX_DET: usize = 300;

struct Fixtures {
    ckpt: Checkpoint,
    ckpt_path: PathBuf,
    golden: MmapSafetensors,
}

impl Fixtures {
    fn tensor(&self, name: &str) -> Vec<f32> {
        self.golden.tensor_f32(name).unwrap_or_else(|| panic!("golden tensor {name} is missing; re-run {DUMPER}"))
    }
}

fn fixtures() -> Option<Fixtures> {
    let ckpt_path = Path::new(&brain_testutil::model_dir("local/yolov5s-relu").unwrap_or_default()).join("model.safetensors");
    if !ckpt_path.exists() {
        brain_testutil::skip("local/yolov5s-relu/model.safetensors absent: run tools/yolov5_export/export_yolov5.py --onnx yolov5s_relu.onnx");
        return None;
    }
    let dir = PathBuf::from(brain_testutil::testdata("golden/yolov5"));
    let src = brain_testutil::golden::Source::open(&dir, DUMPER)?;
    let ckpt = yolov5::weights::open(&ckpt_path).expect("the converted checkpoint must load");
    let params: usize = ckpt.graph.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    if !src.require(&[("nc", ckpt.config.nc as i64), ("head_channels", ckpt.config.head_channels() as i64), ("param_count", params as i64)]) {
        return None;
    }
    let golden = MmapSafetensors::open(dir.join("golden.safetensors")).expect("golden.safetensors");
    Some(Fixtures { ckpt, ckpt_path, golden })
}

fn model(fx: &Fixtures, size: u32) -> Yolov5 {
    let gpu = gpu_core::testgpu::dev(yolov5::model::PIPELINES);
    Yolov5::new_on(gpu, fx.ckpt.config.clone(), &fx.ckpt.weights, size, size).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Run `chw` and compare the three head maps with the goldens `<prefix>_out{0,1,2}`.
fn check_heads(fx: &Fixtures, m: &Yolov5, chw: &[f32], prefix: &str, size: u32) -> [f32; 3] {
    let out = m.forward(chw).unwrap();
    let diffs: [f32; 3] = std::array::from_fn(|i| {
        let side = size as usize / (8 << i);
        assert_eq!(out.dims[i], (side, side));
        max_abs_diff(&out.maps[i], &fx.tensor(&format!("{prefix}_out{i}")))
    });
    println!("{prefix}: max |brain - onnxruntime| per head = {diffs:?}");
    for (i, d) in diffs.iter().enumerate() {
        assert!(*d < HEAD_TOL, "{prefix} head {i}: max abs diff {d} exceeds {HEAD_TOL}");
    }
    diffs
}

#[test]
fn raw_heads_match_onnxruntime_on_noise_at_320_and_640() {
    let Some(fx) = fixtures() else { return };
    let _device = gpu_core::testgpu::exclusive();
    for size in [320, 640] {
        let m = model(&fx, size);
        check_heads(&fx, &m, &fx.tensor(&format!("noise{size}")), &format!("noise{size}"), size);
    }
}

#[test]
fn raw_heads_match_onnxruntime_on_a_real_image_at_320_and_640() {
    let Some(fx) = fixtures() else { return };
    let _device = gpu_core::testgpu::exclusive();
    for size in [320, 640] {
        let m = model(&fx, size);
        check_heads(&fx, &m, &fx.tensor(&format!("img{size}")), &format!("img{size}"), size);
    }
}

fn image_hwc(fx: &Fixtures) -> (Vec<f32>, u32, u32) {
    let shape = fx.golden.shape("image_rgb").unwrap().to_vec();
    let (h0, w0) = (shape[0] as u32, shape[1] as u32);
    let hwc = fx.tensor("image_rgb").into_iter().map(|v| v / 255.0).collect();
    (hwc, w0, h0)
}

fn assert_same_detections(got: &[[f32; 6]], want: &[f32], what: &str) {
    assert_eq!(got.len() * 6, want.len(), "{what}: detection count {} vs golden {}\n{got:?}", got.len(), want.len() / 6);
    for (i, (g, w)) in got.iter().zip(want.chunks(6)).enumerate() {
        assert_eq!(g[5], w[5], "{what} #{i}: class");
        assert!((g[4] - w[4]).abs() < CONF_TOL, "{what} #{i}: conf {} vs {}", g[4], w[4]);
        for k in 0..4 {
            assert!((g[k] - w[k]).abs() < BOX_TOL_PX, "{what} #{i}: box coord {k} {} vs {}", g[k], w[k]);
        }
    }
}

/// The network runs on the golden's own letterboxed input, so this isolates the
/// host decode + NMS + un-letterbox from the forward and from preprocessing.
#[test]
fn decode_and_nms_over_the_forward_match_the_reference_detections() {
    let Some(fx) = fixtures() else { return };
    let _device = gpu_core::testgpu::exclusive();
    let (_, w0, h0) = image_hwc(&fx);
    for size in [320u32, 640] {
        let m = model(&fx, size);
        let out = m.forward(&fx.tensor(&format!("img{size}"))).unwrap();
        let lb = imaging::Letterbox::compute(w0, h0, size);
        let got = detections(&fx.ckpt.config, out.heads(), &lb, (w0, h0), CONF, IOU, MAX_DET).unwrap();
        assert_same_detections(&got, &fx.tensor(&format!("img{size}_dets")), &format!("{size}"));
    }
}

/// The whole path from the decoded image: the Rust letterbox must reproduce the
/// reference's pixels, and `detect` must then land on the same detections.
#[test]
fn detect_from_the_image_matches_the_reference_end_to_end() {
    let Some(fx) = fixtures() else { return };
    let _device = gpu_core::testgpu::exclusive();
    let (hwc, w0, h0) = image_hwc(&fx);
    for size in [320u32, 640] {
        let (chw, _) = imaging::letterbox_rgb(&hwc, w0, h0, size, 114.0 / 255.0);
        let d = max_abs_diff(&chw, &fx.tensor(&format!("img{size}")));
        println!("letterbox {size}: max |rust - reference| = {d}");
        assert!(d < 1e-6, "the letterbox must feed the network the reference's pixels");
        let m = model(&fx, size);
        let got = m.detect(&hwc, w0, h0, CONF, IOU).unwrap();
        assert_same_detections(&got, &fx.tensor(&format!("img{size}_dets")), &format!("detect {size}"));
    }
}

/// Loading from disk through the public path (`Yolov5::load`) builds the same
/// network as the in-memory one: same config, same graph, same output.
#[test]
fn loading_from_the_checkpoint_path_matches_the_in_memory_build() {
    let Some(fx) = fixtures() else { return };
    let _device = gpu_core::testgpu::exclusive();
    let loaded = Yolov5::load(&fx.ckpt_path, 320).unwrap();
    assert_eq!(loaded.input_size(), (320, 320));
    assert_eq!(loaded.config(), &fx.ckpt.config);
    let chw = fx.tensor("noise320");
    let a = loaded.forward(&chw).unwrap();
    let b = model(&fx, 320).forward(&chw).unwrap();
    assert!(max_abs_diff(&a.maps[0], &b.maps[0]) < 1e-6);
}

#[test]
fn head_maps_feed_the_decode_without_a_device() {
    let Some(fx) = fixtures() else { return };
    // The board-side path: golden head tensors straight into the host decode,
    // no `Yolov5`, no device.
    let (_, w0, h0) = image_hwc(&fx);
    let size = 640u32;
    let maps: Vec<Vec<f32>> = (0..3).map(|i| fx.tensor(&format!("img{size}_out{i}"))).collect();
    let heads = std::array::from_fn(|i| HeadMap { data: &maps[i], h: (size as usize) / (8 << i), w: (size as usize) / (8 << i) });
    let lb = imaging::Letterbox::compute(w0, h0, size);
    let got = detections(&fx.ckpt.config, heads, &lb, (w0, h0), CONF, IOU, MAX_DET).unwrap();
    // Decoding onnxruntime's own outputs reproduces the reference exactly (same fp32 inputs).
    assert_same_detections(&got, &fx.tensor("img640_dets"), "device-free decode");
}
