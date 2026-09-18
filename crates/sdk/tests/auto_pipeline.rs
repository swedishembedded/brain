// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "auto")]

//! End-to-end coverage of `AutoPipeline::from_pretrained`'s dispatch, over
//! two DIFFERENT real, synthetic, fully offline fixtures - proving this is
//! genuine content-based dispatch, not a hardcoded single-architecture
//! path. Reuses `tests/depth_pipeline.rs`'s and `tests/detection_pipeline.
//! rs`'s own fixture helpers (`ZipConfig`/`YoloConfig`-derived tiny, complete
//! checkpoints) rather than reinventing them, since both already prove the
//! SAME tiny shapes build and run for real.

use std::path::{Path, PathBuf};

use yolov8::YoloConfig;
use zipdepth::ZipConfig;

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-auto-pipeline-{tag}-{}-{n}", std::process::id()));
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

/// Same tiny, complete, real ZipDepth shape `tests/depth_pipeline.rs`'s own
/// `tiny_cfg` builds - far smaller than any released preset.
fn tiny_zipdepth_cfg() -> ZipConfig {
    ZipConfig { dims: [8, 16, 32, 64], depths: [1, 1, 1, 1], dec_ch: 6, half_dec_ch: 4, ..ZipConfig::base() }
}

fn write_complete_zipdepth_checkpoint(path: &Path, cfg: &ZipConfig) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let tensors: Vec<checkpoint::torchpt_write::TensorOut> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            checkpoint::torchpt_write::TensorOut { name, shape, data: vec![0.0f32; n] }
        })
        .collect();
    checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
}

/// Same tiny, complete, real YOLOv8 shape `tests/detection_pipeline.rs`'s
/// own `write_complete_yolo_checkpoint` builds.
fn write_complete_yolo_checkpoint(path: &Path, cfg: &YoloConfig) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.full_param_list().into_iter().map(|(name, n)| (name, vec![n as u64], vec![0.0f32; n])).collect();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
}

fn rgb8_gradient(w: u32, h: u32) -> brain::Image {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = (x * 255 / w.max(1)) as u8;
            pixels[i + 1] = (y * 255 / h.max(1)) as u8;
            pixels[i + 2] = 128;
        }
    }
    brain::Image::from_rgb8(w, h, pixels).expect("a well-formed buffer must build")
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::AutoPipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A store holding ONLY a real ZipDepth fixture dispatches to
/// `AutoPipeline::Depth`, and the wrapped pipeline is genuinely usable - not
/// just correctly classified but actually reaches a real `.predict()`, the
/// SAME ceiling `tests/depth_pipeline.rs`'s own end-to-end test proves for
/// `DepthPipeline::from_pretrained` directly.
#[test]
fn from_pretrained_dispatches_to_depth_for_a_real_zipdepth_fixture() {
    let root = scratch_root("depth");
    write_complete_zipdepth_checkpoint(&root.join("skchen1993").join("ZipDepth").join("model.pt"), &tiny_zipdepth_cfg());

    let pipe = with_models_dir(&root, || brain::AutoPipeline::from_pretrained("skchen1993/ZipDepth")).expect("a real, complete ZipDepth fixture must dispatch and build");

    let brain::AutoPipeline::Depth(depth) = pipe else { panic!("expected AutoPipeline::Depth, got {pipe:?}") };
    let input = rgb8_gradient(64, 64);
    let opts = brain::DepthOptions::new().input(32);
    let map = depth.predict_with(&input, opts).expect("a real forward pass over a complete checkpoint must succeed");
    assert_eq!((map.width, map.height), (64, 64));
}

/// The SAME dispatch, over a DIFFERENT real fixture (YOLOv8, not ZipDepth) -
/// the actual "auto" claim: this is not a hardcoded single-architecture
/// path, `detect` genuinely tells the two apart and routes each to its own
/// pipeline TYPE. Reaches a real `.detect()`, the SAME ceiling
/// `tests/detection_pipeline.rs`'s own end-to-end test proves for
/// `DetectionPipeline::from_pretrained` directly - and proves the boxed
/// `AutoPipeline::Detection(Box<DetectionPipeline>)` variant is transparent
/// to use (no extra deref ceremony past the pattern match itself).
#[test]
fn from_pretrained_dispatches_to_detection_for_a_real_yolov8_fixture() {
    let root = scratch_root("detect");
    write_complete_yolo_checkpoint(&root.join("local").join("yolov8-tiny.safetensors"), &YoloConfig::tiny(4));

    let pipe = with_models_dir(&root, || brain::AutoPipeline::from_pretrained("local/yolov8-sdk-test")).expect("a real, complete YOLOv8 fixture must dispatch and build");

    let brain::AutoPipeline::Detection(det) = pipe else { panic!("expected AutoPipeline::Detection, got {pipe:?}") };
    let input = rgb8_gradient(64, 64);
    let dets = det.detect(&input).expect("a real forward pass over a complete checkpoint must succeed");
    for d in &dets {
        assert!(d.x1.is_finite() && d.y1.is_finite() && d.x2.is_finite() && d.y2.is_finite(), "{d:?}");
        assert!(d.class < 4, "{d:?}");
    }
}

/// An empty store (nothing any known architecture's resolver classifies)
/// is a clean, named error - never a panic, never a network attempt.
#[test]
fn from_pretrained_names_no_known_architecture_when_the_store_is_empty() {
    let root = scratch_root("empty");
    std::fs::write(root.join("unrelated.txt"), b"not a checkpoint").unwrap();

    let err = with_models_dir(&root, || brain::AutoPipeline::from_pretrained("local/nothing-here").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        brain::Error::Missing(_) => {}
        other => panic!("expected Error::Backend or Error::Missing, got {other:?}"),
    }
}
