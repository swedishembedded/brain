// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "image")]

//! End-to-end coverage of `RestorePipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture -- mirroring
//! `tests/upscale_pipeline.rs`'s fixture pattern.
//!
//! ## Why this stops at construction, and two real gaps found reaching that far
//!
//! Unlike `RrdbConfig` (`tests/upscale_pipeline.rs`'s own doc), CodeFormer's
//! shape is NOT derived from the checkpoint - `codeformer::CodeFormerConfig`
//! is one fixed preset (`inference_codeformer.py`'s hardcoded constructor
//! call), so there is no "tiny" variant to shrink to the way RRDBNet's is.
//! `CodeFormerConfig::tensor_manifest()` still names every tensor the
//! forward graph reads with an exact-match contract
//! (`crates/codeformer/src/import.rs::import`), so an all-zero checkpoint AT
//! THE REAL RELEASED SIZE is a genuinely complete, genuinely importable one -
//! `from_pretrained` resolves, classifies and imports all 515 real tensors
//! and builds the whole real 512x512 graph (VQGAN encoder/generator, the
//! 9-layer code-prediction Transformer, the controllable feature
//! transformation), further than `tests/image_pipeline.rs`'s flux2/s3dit
//! backends reach at all (their weights are too large to fixture even at
//! construction).
//!
//! `.restore()` itself - the actual forward dispatch - is NOT called here,
//! because it does not complete on either backend available in this
//! environment, for two independent, real, pre-existing reasons this
//! fixture surfaced (neither is new code from this milestone; both are in
//! shared forward-pass/backend infrastructure this crate does not own):
//!
//! * **wgpu**: `backend-wgpu` aborts with "2.37 GiB of device buffers were
//!   dropped without an intervening `poll_wait()`", over this device's 2 GiB
//!   ceiling - the exact failure mode `gpu_core::transient`'s own module doc
//!   describes ("a real 12B text encoder and a real 22B DiT both reached
//!   multiple gigabytes of abandoned-but-live device memory this way"),
//!   here from CodeFormer's ~59-block encoder+transformer+generator walk
//!   never calling `gpu_core::reclaiming`/`Transient` anywhere in its own or
//!   `vqgan::model::run_blocks`'s forward path.
//! * **CPU (`wgsl-cpu`)**: `matmul_reg3` "was not JIT-compiled (unsupported
//!   work-group structure)" - a kernel-coverage gap in the CPU JIT backend
//!   the code-prediction Transformer's attention/FFN matmuls hit.
//!
//! Both are plausibly never exercised anywhere else in this workspace
//! either: `crates/codeformer/tests/parity.rs`'s own real-forward-pass tests
//! all gate on `BRAIN_CODEFORMER_WEIGHTS` (a license-gated real checkpoint),
//! silently skipped in any environment that has not fetched one - this
//! all-zero-but-complete synthetic fixture is the first thing in this
//! workspace to force CodeFormer's real graph to actually dispatch with no
//! real weights required. Tracked in the SDK design-sweep roadmap as two
//! real, named, NOT-fixed gaps - fixing either safely needs a device this
//! environment does not have (more VRAM, or a working CPU JIT path to
//! cross-check a memory-management change against),
//! not a fix attempted blind against the one small, already-panicking
//! device available here.

use std::path::{Path, PathBuf};

use checkpoint::torchpt_write::TensorOut;
use codeformer::CodeFormerConfig;

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-restore-pipeline-{tag}-{}-{n}", std::process::id()));
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
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "codeformer", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// A COMPLETE, real `torch.save` checkpoint at the released `codeformer()`
/// preset's shape: every tensor `CodeFormerConfig::tensor_manifest` names,
/// all zeros. `import::import` requires an exact match in both directions,
/// so this is what actually lets construction succeed.
fn write_complete_codeformer_checkpoint(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let cfg = CodeFormerConfig::codeformer();
    let tensors: Vec<TensorOut> = cfg
        .tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            TensorOut { name: format!("params_ema.{name}"), shape, data: vec![0.0f32; n] }
        })
        .collect();
    checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::RestorePipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "codeformer-empty-test");

    let err = with_models_dir(&root, || brain::RestorePipeline::from_pretrained("local/codeformer-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "codeformer");
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
/// network access at any point: reference parses, `Store::local` resolves
/// it, `loader::resolve_structured` finds exactly one candidate for
/// `"weights"`, and `codeformer::caps::load` imports all 515 real tensors
/// and builds the whole real 512x512 graph on a real `Gpu`. This module's
/// own doc explains why `.restore()` itself is not called here - two real,
/// separately-tracked gaps in shared forward-pass/backend infrastructure,
/// neither of which is new code from this milestone.
#[test]
fn from_pretrained_builds_a_real_complete_fixture() {
    let root = scratch_root("restore");
    write_complete_codeformer_checkpoint(&root.join("sczhou").join("codeformer.pth"));
    mark_locally_present(&root, "local", "codeformer-sdk-test");

    let pipe = with_models_dir(&root, || brain::RestorePipeline::from_pretrained("local/codeformer-sdk-test"))
        .expect("a complete, real checkpoint must import and build");

    let cfg = format!("{pipe:?}");
    assert!(cfg.contains("RestorePipeline"), "{cfg}");
}

/// `RestoreOptions::fidelity` out of `[0, 1]` is a clean, named error over
/// the SAME real fixture, never a panic.
#[test]
fn restore_with_rejects_fidelity_out_of_range() {
    let root = scratch_root("bad-fidelity");
    write_complete_codeformer_checkpoint(&root.join("sczhou").join("codeformer.pth"));
    mark_locally_present(&root, "local", "codeformer-fidelity-test");

    let pipe = with_models_dir(&root, || brain::RestorePipeline::from_pretrained("local/codeformer-fidelity-test")).expect("a complete, real checkpoint must build");

    let input = rgb8_gradient(4, 4);
    let err = pipe.restore_with(&input, brain::RestoreOptions::new().fidelity(2.0)).unwrap_err();
    assert!(matches!(err, brain::Error::Backend(_)), "{err:?}");
}
