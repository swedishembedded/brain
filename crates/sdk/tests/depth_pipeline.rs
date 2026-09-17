// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "vision")]

//! End-to-end coverage of `DepthPipeline::from_pretrained` against a local,
//! synthetic, fully offline models-directory fixture - mirroring
//! `tests/detection_pipeline.rs`'s fixture pattern.
//!
//! ## Why this DOES reach a real, successful `.predict()`, fast
//!
//! Like `RrdbConfig`/`YoloConfig` and unlike `CodeFormerConfig`/SAM 2.1's
//! own config, ZipDepth's shape is DERIVED from the checkpoint's own tensors
//! (`zipdepth::config::ZipConfig::from_tensors`, added alongside
//! `zipdepth::spec::ZipdepthSpec` this milestone) rather than pinned to one
//! released preset - so a fixture at TINY encoder widths is a genuinely
//! complete, genuinely buildable checkpoint, not a full-size one. The model's
//! input SIDE (`ZipConfig::input`) is a runtime knob, not a stored weight, so
//! it is not shrunk by the fixture's tiny widths alone - `DepthOptions::
//! input(32)` overrides it down to the smallest valid (x32) size for this
//! test, the same reason `tests/restore_pipeline.rs`/`segment_pipeline.rs`
//! pay a real fixed cost their model has no smaller preset for, but this one
//! does not have to.

use std::path::{Path, PathBuf};

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-depth-pipeline-{tag}-{}-{n}", std::process::id()));
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
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "zipdepth", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// A tiny but real ZipDepth shape - far smaller than any released preset,
/// the same "derive, don't hardcode" property `zipdepth::spec::tests` relies
/// on. `global_mode: Balanced` (via `..ZipConfig::base()`) so the fixture
/// exercises the StripPoolingAttention/GlobalContextBlock branches too.
fn tiny_cfg() -> ZipConfig {
    ZipConfig { dims: [8, 16, 32, 64], depths: [1, 1, 1, 1], dec_ch: 6, half_dec_ch: 4, ..ZipConfig::base() }
}

/// A COMPLETE, real `torch.save` checkpoint at `cfg`'s shape: every tensor
/// `cfg.param_list()` names, all zeros - the exact contract
/// `zipdepth::spec::ZipdepthSpec::classify` and `zipdepth::caps::load` both
/// read.
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

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::DepthPipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "zipdepth-empty-test");

    let err = with_models_dir(&root, || brain::DepthPipeline::from_pretrained("local/zipdepth-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "zipdepth");
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
/// network access at any point, all the way to a real `.predict()`:
/// reference parses, `Store::local` resolves it, `loader::resolve_structured`
/// finds exactly one candidate for `"weights"` and derives the tiny shape
/// from the checkpoint's own tensors, `zipdepth::caps::load` re-derives the
/// identical config and imports the full tensor data, and `.predict()` runs
/// a genuine forward pass through the whole graph (encoder stages 1-4 incl.
/// StripPoolingAttention/GlobalContextBlock, SPPF, cross-scale fusion,
/// decoder, FastConvexUpsample) at a small overridden input size.
#[test]
fn from_pretrained_predicts_depth_over_a_real_tiny_fixture_end_to_end() {
    let root = scratch_root("predict");
    write_complete_zipdepth_checkpoint(&root.join("local").join("zipdepth-tiny.pt"), &tiny_cfg());
    mark_locally_present(&root, "local", "zipdepth-sdk-test");

    let pipe = with_models_dir(&root, || brain::DepthPipeline::from_pretrained("local/zipdepth-sdk-test")).expect("a complete, real tiny checkpoint must build");

    let input = rgb8_gradient(48, 32);
    let depth = pipe.predict_with(&input, brain::DepthOptions::new().input(32)).expect("a real forward pass over a complete checkpoint must succeed");

    assert_eq!(depth.width, 48, "the source image's own size, not the model's internal working grid");
    assert_eq!(depth.height, 32);
    assert_eq!(depth.values.len(), 48 * 32);
    assert!(depth.values.iter().all(|v| (0.0..=1.0).contains(v)), "{:?}", depth.values);
    assert!(depth.min <= depth.max, "min {} > max {}", depth.min, depth.max);
}
