// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "image")]

//! End-to-end coverage of `UpscalePipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture -- mirroring
//! `tests/image_pipeline.rs`'s fixture pattern.
//!
//! ## Why this DOES reach a real, successful `.upscale()?.save()`
//!
//! Unlike `ImagePipeline`'s flux2/s3dit backends (`tests/image_pipeline.rs`'s
//! own doc explains why those stop at construction), RRDBNet's shape is
//! DERIVED from the checkpoint, not hardcoded to one multi-billion-parameter
//! release (`rrdbnet::config::RrdbConfig::from_tensors`,
//! `rrdbnet::spec::RrdbnetSpec`). `RrdbConfig::param_list()` already names
//! every tensor a given config's forward pass reads
//! (`crates/rrdbnet/src/import.rs::validate` requires an exact match, both
//! directions), so a fixture at TINY dimensions (`num_feat=8`,
//! `num_grow_ch=4`, 2 blocks, x2) is a genuinely complete, genuinely
//! buildable checkpoint -- all zeros, so the output is not a meaningful
//! image, but every kernel dispatch, buffer size and layout permutation on
//! the real path runs for real. This is the pipeline family's first
//! `from_pretrained -> task call -> inspect the domain result -> save`
//! integration test that does not have to stop at a clean construction
//! error (this SDK's own "every public pipeline needs an end-to-end test,
//! and a real one" rule).

use std::path::{Path, PathBuf};

use checkpoint::torchpt_write::TensorOut;
use rrdbnet::RrdbConfig;

/// A fixture models directory that deletes itself when the test ends.
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
    let dir = std::env::temp_dir().join(format!("brain-sdk-upscale-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// Run `f` with `BRAIN_MODELS_DIR` pointed at `root`, serialized against
/// every other test in this binary that also touches process environment
/// (mirrors `tests/image_pipeline.rs`'s own helper).
fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

/// Drop a `brain.manifest.json` so `brain_modelstore::Store::local` resolves
/// this exact reference with no network access -- mirrors
/// `tests/image_pipeline.rs`'s identically-named helper.
fn mark_locally_present(root: &Path, vendor: &str, repo: &str) {
    let dir = root.join(vendor).join(repo);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "rrdbnet", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// The tiny config every fixture in this file agrees on: small enough to
/// build and run in milliseconds, still a genuine `x2`, 2-block RRDBNet
/// (`crates/rrdbnet/src/spec.rs`'s own `write_rrdb_pt` test helper uses the
/// same dimensions for its classification-only fixture).
fn tiny_cfg() -> RrdbConfig {
    RrdbConfig { in_channels: 3, out_channels: 3, num_feat: 8, num_grow_ch: 4, num_block: 2, scale: 2 }
}

/// A COMPLETE, real `torch.save` checkpoint at `cfg`'s shape: every tensor
/// `RrdbConfig::param_list` names (not just the classification subset
/// `rrdbnet::spec`'s own test fixture writes), all zeros. `import::validate`
/// requires an exact match in both directions, so this is what actually
/// lets construction succeed rather than fail on a missing/extra tensor.
fn write_complete_rrdb_checkpoint(path: &Path, cfg: &RrdbConfig) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let tensors: Vec<TensorOut> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            TensorOut { name: format!("params_ema.{name}"), shape, data: vec![0.0f32; n] }
        })
        .collect();
    checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
}

/// An unparseable reference is refused before any filesystem or network
/// access at all.
#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::UpscalePipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A reference that resolves locally but whose store has nothing
/// classifiable as RRDBNet is a clean `Error::Missing` naming the one
/// `"weights"` role, never a panic and never a network fetch attempt.
#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "rrdbnet-empty-test");

    let err = with_models_dir(&root, || brain::UpscalePipeline::from_pretrained("local/rrdbnet-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "rrdbnet");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            assert!(roles.contains(&"weights"), "{roles:?}");
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

/// Two equally-valid candidates for the one `"weights"` role is an
/// `Ambiguous` question, never a silent pick -- the same resolver guarantee
/// `rrdbnet::spec`'s own `two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick`
/// test pins at the spec layer; this pins it through the public facade.
#[test]
fn from_pretrained_reports_two_equally_real_candidates_as_ambiguous() {
    let root = scratch_root("ambiguous");
    let cfg = tiny_cfg();
    write_complete_rrdb_checkpoint(&root.join("vendor-a").join("model.pth"), &cfg);
    write_complete_rrdb_checkpoint(&root.join("vendor-b").join("model-copy.pth"), &cfg);
    mark_locally_present(&root, "local", "rrdbnet-ambiguous-test");

    let err = with_models_dir(&root, || brain::UpscalePipeline::from_pretrained("local/rrdbnet-ambiguous-test").unwrap_err());
    assert!(matches!(err, brain::Error::Ambiguous(_)), "{err:?}");
}

fn rgb8_gradient(w: u32, h: u32) -> brain::Image {
    let px: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
    brain::Image::from_rgb8(w, h, px).unwrap()
}

/// The full facade path against a real, COMPLETE local fixture, with no
/// network access at any point, all the way to a real `.upscale()?.save()`:
/// reference parses, `Store::local` resolves it, `loader::resolve_structured`
/// finds exactly one candidate for `"weights"`, `rrdbnet::caps::load`
/// re-derives the identical tiny config and builds a real generator on a
/// real `Gpu`, and `.upscale()` runs a genuine forward pass -- 2x2 in,
/// 4x4 out (this fixture's `scale: 2`), a real PNG on disk.
#[test]
fn from_pretrained_upscales_a_real_tiny_fixture_end_to_end() {
    let root = scratch_root("upscale");
    write_complete_rrdb_checkpoint(&root.join("schwgHao").join("RealESRGAN_tiny.pth"), &tiny_cfg());
    mark_locally_present(&root, "local", "rrdbnet-sdk-test");

    let pipe = with_models_dir(&root, || brain::UpscalePipeline::from_pretrained("local/rrdbnet-sdk-test")).expect("a complete, real tiny checkpoint must build");

    let input = rgb8_gradient(2, 2);
    let out = pipe.upscale(&input).expect("a real forward pass over a complete checkpoint must succeed");

    assert_eq!(out.width(), 4, "scale: 2 over a 2px input");
    assert_eq!(out.height(), 4);
    assert_eq!(out.pixels().len(), 4 * 4 * 3);

    let png = std::env::temp_dir().join(format!("brain-sdk-upscale-e2e-{}.png", std::process::id()));
    out.save(&png).expect("Image::save must write a real PNG");
    let bytes = std::fs::read(&png).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "must be a real PNG signature");
    std::fs::remove_file(&png).ok();
}

/// `UpscaleOptions::tile` takes the SAME fixture through the tiled path
/// (`rrdbnet::caps::Session::upscale_with_halo`'s `tile != 0` branch)
/// instead of the whole-image one -- a second real forward pass, through
/// different code, over the same checkpoint.
#[test]
fn from_pretrained_upscales_tiled_over_the_same_real_fixture() {
    let root = scratch_root("upscale-tiled");
    write_complete_rrdb_checkpoint(&root.join("schwgHao").join("RealESRGAN_tiny.pth"), &tiny_cfg());
    mark_locally_present(&root, "local", "rrdbnet-sdk-tiled-test");

    let pipe = with_models_dir(&root, || brain::UpscalePipeline::from_pretrained("local/rrdbnet-sdk-tiled-test")).expect("a complete, real tiny checkpoint must build");

    let input = rgb8_gradient(4, 4);
    let out = pipe.upscale_with(&input, brain::UpscaleOptions::new().tile(2)).expect("a real tiled forward pass must succeed");

    assert_eq!((out.width(), out.height()), (8, 8));
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

    let err = brain::UpscalePipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}
