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
