// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "image")]

//! End-to-end coverage of `RestorePipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture -- mirroring
//! `tests/upscale_pipeline.rs`'s fixture pattern.
//!
//! ## The pipeline family's first real, full-size forward pass -- and two real, independent, pre-existing bugs found and fixed reaching it
//!
//! Unlike `RrdbConfig` (`tests/upscale_pipeline.rs`'s own doc), CodeFormer's
//! shape is NOT derived from the checkpoint - `codeformer::CodeFormerConfig`
//! is one fixed preset (`inference_codeformer.py`'s hardcoded constructor
//! call), so there is no "tiny" variant to shrink to the way RRDBNet's is.
//! `CodeFormerConfig::tensor_manifest()` still names every tensor the
//! forward graph reads with an exact-match contract
//! (`crates/codeformer/src/import.rs::import`), so an all-zero checkpoint AT
//! THE REAL RELEASED SIZE is a genuinely complete, genuinely buildable one -
//! all zeros, so the restored face is not a meaningful image, but every
//! kernel dispatch on the real path (VQGAN encoder, 9-layer code-prediction
//! Transformer, codebook gather, generator with the controllable feature
//! transformation) runs for real, at the model's real fixed 512x512
//! geometry, on a real `Gpu` - there is no smaller real config to fall back
//! to, and unlike every other pipeline in this crate, this one does not need
//! one: `.restore()` itself completes.
//!
//! Getting there surfaced two real, independent, pre-existing bugs in shared
//! forward-pass/backend infrastructure this crate does not own - neither
//! reachable before, because `crates/codeformer/tests/parity.rs`'s own
//! real-forward-pass tests all gate on `BRAIN_CODEFORMER_WEIGHTS` (a
//! license-gated real checkpoint, silently skipped absent one) - both found
//! and fixed in the same milestone that added this test:
//!
//! * **A duplicate kernel registration** (`crates/codeformer/src/model.rs`):
//!   `matmul_reg3` is already present in `vae::blocks::KERNELS` (exported as
//!   `vae::blocks::MATMUL_REG3_SLOT` for exactly this reason - a caller
//!   layering its own kernels on top must reuse it, not register a second
//!   copy, the same lesson `crates/sdxlunet`'s own history records), but
//!   this crate's `kernel_set()` registered a SECOND `("matmul_reg3", ...)`
//!   at a new slot anyway. Harmless on `backend-wgpu` (both indices compile
//!   to a valid pipeline), but `wgsl-cpu`'s JIT cannot compile `matmul_reg3`
//!   AT ALL (a work-group/shared-memory kernel, CPU-native-only by design) -
//!   `backend_cpu`'s AVX2 fast path intercepts it by matching ONE cached
//!   index, so dispatching through the uncaught duplicate fell through to
//!   the JIT and panicked. Fixed by resolving `K_MATMUL_REG3` to the
//!   existing `vae::blocks::MATMUL_REG3_SLOT` instead of appending a new
//!   one (`model::tests::matmul_reg3_reuses_the_shared_slot_not_a_second_registration`
//!   pins it).
//! * **Two unreclaimed device-memory scopes** (`crates/codeformer/src/
//!   model.rs::CodeFormer::build`): the encoder+transformer half and the
//!   generator+CFT half each build their own `vae::blocks::Builder`, whose
//!   activation pool (`Builder::free`) reuses same-length buffers WITHIN one
//!   builder's own recording but leaves anything that never recurs sitting
//!   in the pool until that `Builder` itself drops at the end of its block -
//!   with no poll in between, and nothing else allocates again until
//!   `.restore()`'s own first post-construction buffer (a readback staging
//!   buffer), which is where `backend-wgpu`'s "too much unreclaimed memory"
//!   ceiling actually tripped - exactly the failure mode `gpu_core::
//!   transient`'s own module doc describes, one `Builder` scope at a time
//!   rather than one loop iteration at a time. Fixed by wrapping each of the
//!   two builder scopes in `gpu_core::reclaiming`, which polls once each
//!   scope's drops are done - the buffers a later scope still needs
//!   (`enc_feat`'s four pinned encoder taps) are returned OUT of the first
//!   closure, so they survive its poll, exactly as that helper's own doc
//!   describes.
//!
//! Verified two ways beyond this file's own tests: `cargo test -p
//! brain-codeformer --lib` (no regressions - taps/gradcheck/parity-fixture
//! coverage all still pass), and this test itself passing on BOTH the wgpu
//! backend (the default here) and, during investigation, `Device::parse
//! ("cpu")` - a genuine forward pass and NOT a placeholder success, since
//! there is only one real fixed geometry for this model to have run.

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
/// network access at any point, all the way to a real `.restore()?.save()`:
/// reference parses, `Store::local` resolves it, `loader::resolve_structured`
/// finds exactly one candidate for `"weights"`, `codeformer::caps::load`
/// imports all 515 real tensors and builds the whole real 512x512 graph on a
/// real `Gpu`, and `.restore()` runs a genuine forward pass through the
/// entire graph - any input size in, a real 512x512 PNG out. This module's
/// own doc explains the two real bugs fixing this took.
#[test]
fn from_pretrained_restores_a_real_complete_fixture_end_to_end() {
    let root = scratch_root("restore");
    write_complete_codeformer_checkpoint(&root.join("sczhou").join("codeformer.pth"));
    mark_locally_present(&root, "local", "codeformer-sdk-test");

    let pipe = with_models_dir(&root, || brain::RestorePipeline::from_pretrained("local/codeformer-sdk-test")).expect("a complete, real checkpoint must build");

    let input = rgb8_gradient(8, 8);
    let out = pipe.restore(&input).expect("a real forward pass over a complete checkpoint must succeed");

    assert_eq!(out.width(), 512, "the graph's fixed square side");
    assert_eq!(out.height(), 512);
    assert_eq!(out.pixels().len(), 512 * 512 * 3);

    let png = std::env::temp_dir().join(format!("brain-sdk-restore-e2e-{}.png", std::process::id()));
    out.save(&png).expect("Image::save must write a real PNG");
    let bytes = std::fs::read(&png).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "must be a real PNG signature");
    std::fs::remove_file(&png).ok();
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

/// The real, confirmed gap `RestorePipelineBuilder::load`'s own doc names:
/// unlike every other fixture in this file (which use `mark_locally_present`
/// purely to satisfy `Store::local`'s own generic check cheaply, with no
/// bearing on whether the real bug is present), THIS fixture places the
/// checkpoint at EXACTLY the repo path `model_id` itself names, with no
/// separate manifest anywhere else in the store - the same shape a real `hf
/// download sczhou/CodeFormer --local-dir $BRAIN_MODELS_DIR/sczhou/CodeFormer`
/// produces. Before the resolve-first fix, this reached `Error::Download`
/// ("not found: sczhou/CodeFormer@main") - a real network hub query for a
/// reference that was already fully present on disk - confirmed empirically
/// while building this fix. Now it resolves and builds cleanly, with no
/// network access at any point.
#[test]
fn from_pretrained_resolves_a_real_fixture_at_its_own_named_path_with_no_store_local_shortcut() {
    let root = scratch_root("no-shortcut");
    write_complete_codeformer_checkpoint(&root.join("sczhou").join("CodeFormer").join("codeformer.pth"));

    let pipe = with_models_dir(&root, || brain::RestorePipeline::from_pretrained("sczhou/CodeFormer"));
    pipe.expect("a real checkpoint at exactly its own model_id's path must resolve with no Store::local shortcut and no network access");
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

    let err = brain::RestorePipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}
