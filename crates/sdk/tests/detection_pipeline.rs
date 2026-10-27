// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "vision")]

//! End-to-end coverage of `DetectionPipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture - mirroring
//! `tests/upscale_pipeline.rs`'s fixture pattern.
//!
//! ## Why this DOES reach a real, successful `.detect()`
//!
//! Like `RrdbConfig` (`tests/upscale_pipeline.rs`'s own doc) and unlike
//! `CodeFormerConfig` (`tests/restore_pipeline.rs`'s own doc), YOLOv8's
//! shape is DERIVED from the checkpoint - `YoloConfig::from_json` reads
//! `channels`/`backbone_ch`/`nc`/... straight from the safetensors header,
//! and `YoloConfig::tiny(nc)` is a genuinely small, real preset (not the
//! canonical `yolov8n` layout) - so a fixture built from
//! `cfg.full_param_list()` at TINY dimensions is a genuinely complete,
//! genuinely buildable checkpoint. All zeros, so the "detections" are not
//! meaningful, but every kernel dispatch on the real path (backbone, neck,
//! decoupled head, DFL decode, NMS) runs for real.

use std::path::{Path, PathBuf};

use yolov8::YoloConfig;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

fn scratch_root(tag: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-detection-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

fn mark_locally_present(root: &Path, vendor: &str, repo: &str) {
    let dir = root.join(vendor).join(repo);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "yolov8", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// A COMPLETE, real brain-format checkpoint at `cfg`'s shape: every tensor
/// `cfg.full_param_list()` names, all zeros, with a real `brain.config`
/// header - the exact contract `yolov8::spec::YoloSpec::classify` and
/// `yolov8::Yolo::load` both read.
fn write_complete_yolo_checkpoint(path: &Path, cfg: &YoloConfig) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.full_param_list().into_iter().map(|(name, n)| (name, vec![n as u64], vec![0.0f32; n])).collect();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::DetectionPipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "yolov8-empty-test");

    let err = with_models_dir(&root, || brain::DetectionPipeline::from_pretrained("local/yolov8-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "yolov8");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            assert!(roles.contains(&"weights"), "{roles:?}");
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

fn rgb8_gradient(w: u32, h: u32) -> brain::Image {
    let px: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
    brain::Image::from_rgb8(w, h, px).unwrap()
}

/// The full facade path against a real, COMPLETE local fixture, with no
/// network access at any point, all the way to a real `.detect()`:
/// reference parses, `Store::local` resolves it, `loader::resolve_structured`
/// finds exactly one candidate for `"weights"`, `yolov8::Yolo::load`
/// re-derives the identical tiny config and builds a real detector, and
/// `.detect()` runs a genuine forward pass through backbone/neck/head, DFL
/// decode and NMS.
#[test]
fn from_pretrained_detects_over_a_real_tiny_fixture_end_to_end() {
    let root = scratch_root("detect");
    write_complete_yolo_checkpoint(&root.join("local").join("yolov8-tiny.safetensors"), &YoloConfig::tiny(4));
    mark_locally_present(&root, "local", "yolov8-sdk-test");

    let pipe = with_models_dir(&root, || brain::DetectionPipeline::from_pretrained("local/yolov8-sdk-test")).expect("a complete, real tiny checkpoint must build");

    let input = rgb8_gradient(64, 64);
    let dets = pipe.detect(&input).expect("a real forward pass over a complete checkpoint must succeed");

    // All-zero weights is not a meaningful detector, but every returned box
    // must still be finite and class-indexed within the fixture's `nc = 4`.
    for d in &dets {
        assert!(d.x1.is_finite() && d.y1.is_finite() && d.x2.is_finite() && d.y2.is_finite(), "{d:?}");
        assert!((0.0..=1.0).contains(&d.confidence), "{d:?}");
        assert!(d.class < 4, "{d:?}");
    }
}

/// `DetectOptions::confidence` at `1.1` (above any real sigmoid output)
/// filters every candidate out - proving the option is actually threaded
/// through, over the SAME real fixture.
#[test]
fn detect_with_a_confidence_above_any_real_score_returns_nothing() {
    let root = scratch_root("high-confidence");
    write_complete_yolo_checkpoint(&root.join("local").join("yolov8-tiny.safetensors"), &YoloConfig::tiny(4));
    mark_locally_present(&root, "local", "yolov8-confidence-test");

    let pipe = with_models_dir(&root, || brain::DetectionPipeline::from_pretrained("local/yolov8-confidence-test")).expect("a complete, real tiny checkpoint must build");

    let input = rgb8_gradient(64, 64);
    let dets = pipe.detect_with(&input, brain::DetectOptions::new().confidence(1.1)).expect("a real forward pass must succeed");
    assert!(dets.is_empty(), "{dets:?}");
}

/// `DownloadPolicy::Offline` never reaches the network -- same proof shape
/// as `crates/sdk/tests/image_pipeline.rs`'s own
/// `download_policy_offline_never_touches_the_network`: point `HfHub` at a
/// loopback port nothing listens on, and show a reference resolving neither
/// locally nor from any real hub still comes back the resolver's own clean
/// `Error::Missing`, never a connection-error-flavored `Error::Download`.
#[test]
fn download_policy_offline_never_touches_the_network() {
    let root = scratch_root("offline");
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", &*root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::DetectionPipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}

// ---- YOLOv5: the same pipeline type, the architecture read from the checkpoint ----

/// A complete tiny YOLOv5 checkpoint with deterministic weights.
fn write_tiny_yolov5_checkpoint(path: &Path, nc: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut cfg = yolov5::Yolov5Config::yolov5s(yolov5::Activation::Relu);
    cfg.nc = nc;
    cfg.width_multiple = 0.125;
    let graph = yolov5::Graph::build(&cfg);
    let mut rng = data::rng::Lcg::new(9);
    let tensors = graph
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            (name, (0..n).map(|_| rng.scaled(0.2)).collect())
        })
        .collect();
    let weights = yolov5::Weights::new(&graph, tensors).unwrap();
    yolov5::weights::save(path, &cfg, &graph, &weights).unwrap();
}

/// The facade resolves a YOLOv5 checkpoint to the YOLOv5 model with no flag
/// naming the architecture, runs a real forward and decode, and reflects the
/// YOLOv5 capability manifest.
#[test]
fn from_pretrained_resolves_a_yolov5_checkpoint_and_detects_over_it() {
    let root = scratch_root("yolov5-detect");
    write_tiny_yolov5_checkpoint(&root.join("local").join("yolov5-sdk-test").join("model.safetensors"), 3);

    let pipe = with_models_dir(&root, || brain::DetectionPipeline::from_pretrained("local/yolov5-sdk-test")).expect("a complete YOLOv5 checkpoint must build");
    assert_eq!(pipe.capabilities().model, yolov5::caps::MODEL);

    // The tiny network's own default input is 640; any photo-sized image is letterboxed into it.
    let dets = pipe.detect_with(&rgb8_gradient(96, 64), brain::DetectOptions::new().confidence(0.0)).expect("a real forward pass must succeed");
    assert!(!dets.is_empty(), "a zero threshold keeps candidates");
    for d in &dets {
        assert!(d.x1 >= 0.0 && d.x2 <= 96.0 && d.y1 >= 0.0 && d.y2 <= 64.0 && d.x1 <= d.x2 && d.y1 <= d.y2, "{d:?}");
        assert!(d.class < 3, "{d:?}");
    }
}

/// A YOLOv8 checkpoint still resolves to YOLOv8 (and says so in its manifest)
/// when YOLOv5 is also a candidate architecture.
#[test]
fn a_yolov8_checkpoint_still_resolves_to_yolov8() {
    let root = scratch_root("yolov8-still");
    write_complete_yolo_checkpoint(&root.join("local").join("yolov8-both").join("model.safetensors"), &YoloConfig::tiny(4));
    let pipe = with_models_dir(&root, || brain::DetectionPipeline::from_pretrained("local/yolov8-both")).expect("must build");
    assert_eq!(pipe.capabilities().model, yolov8::caps::MODEL);
}

/// The real Rockchip ReLU export through the facade, against the onnxruntime
/// goldens (see `crates/yolov5/tests/parity.rs` for the fixtures).
#[test]
fn the_real_yolov5s_relu_checkpoint_detects_what_onnxruntime_detects() {
    let ckpt = std::path::PathBuf::from(brain_testutil::model_dir("local/yolov5s-relu").unwrap_or_default()).join("model.safetensors");
    if !ckpt.exists() {
        brain_testutil::skip("local/yolov5s-relu/model.safetensors absent: run tools/yolov5_export/export_yolov5.py");
        return;
    }
    let golden_dir = std::path::PathBuf::from(brain_testutil::testdata("golden/yolov5"));
    let Some(_src) = brain_testutil::golden::Source::open(&golden_dir, "tools/goldens/yolov5_dump_reference.py") else { return };
    let golden = checkpoint::mmap::MmapSafetensors::open(golden_dir.join("golden.safetensors")).expect("golden.safetensors");

    let shape = golden.shape("image_rgb").unwrap().to_vec();
    let (h0, w0) = (shape[0] as u32, shape[1] as u32);
    let rgb8: Vec<u8> = golden.tensor_f32("image_rgb").unwrap().into_iter().map(|v| v as u8).collect();
    let image = brain::Image::from_rgb8(w0, h0, rgb8).unwrap();

    // Default model store; the reference is a bare checkpoint file under a store repo directory.
    let pipe = brain::DetectionPipeline::builder("local/yolov5s-relu").download_policy(brain::DownloadPolicy::Offline).load().expect("the converted checkpoint must resolve");
    let got = pipe.detect(&image).unwrap();

    let want = golden.tensor_f32("img640_dets").unwrap();
    assert_eq!(got.len() * 6, want.len(), "{got:?}");
    for (g, w) in got.iter().zip(want.chunks(6)) {
        assert_eq!(g.class as f32, w[5], "{g:?}");
        assert!((g.confidence - w[4]).abs() < 1e-3 && (g.x1 - w[0]).abs() < 0.1 && (g.y1 - w[1]).abs() < 0.1 && (g.x2 - w[2]).abs() < 0.1 && (g.y2 - w[3]).abs() < 0.1, "{g:?} vs {w:?}");
    }
}
