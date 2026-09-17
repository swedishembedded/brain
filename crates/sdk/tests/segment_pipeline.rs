// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "vision")]

//! End-to-end coverage of `SegmentPipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture - mirroring
//! `tests/upscale_pipeline.rs`'s fixture pattern.
//!
//! Like CodeFormer (`tests/restore_pipeline.rs`) and unlike RRDBNet/YOLOv8,
//! SAM 2.1 has no smaller synthetic config to shrink to for this fixture:
//! `sam2::spec::Sam2Spec::classify` derives the released variant from the
//! trunk's own patch-embedding width, and only `96` (tiny) or `144` (large)
//! classify at all - there is no synthetic in-between size the real resolver
//! would accept. This fixture builds a COMPLETE `hiera_tiny` checkpoint (via
//! `sam2::import::manifest_for`, the same tensor list `sam2::caps::load`
//! itself validates against) and reaches a real `.segment()` forward pass at
//! the model's real 1024x1024 geometry, unlike `RestorePipeline`'s own
//! CodeFormer fixture, which found - and this milestone's own bug-hunting
//! discipline found and fixed, in `sam2::spec::Sam2Spec` itself, one layer
//! before ever reaching the forward pass: `classify` checked `ArtifactKind::
//! Opaque`, but a real `.pt`/`.pth` scanned by `brain_modelstore::inventory::
//! scan` classifies `Torch` - the exact bug `rrdbnet::spec::RrdbnetSpec` had
//! before Phase 2.5 fixed it, now found a second time in a spec this
//! session did not otherwise touch. Every one of `Sam2Spec`'s own pre-existing
//! unit tests passed anyway, because - like `RrdbnetSpec`'s once did - they
//! all hand-build an `ArtifactRecord` with an explicitly chosen `kind`,
//! never going through the real scanner; `brain sam2 track`/`brain do sam2
//! segment` (the resolver-migrated CLI paths) could never actually resolve
//! a real installed SAM2 checkpoint either, until this fix.

use std::path::{Path, PathBuf};

use sam2::Sam2Config;

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-segment-pipeline-{tag}-{}-{n}", std::process::id()));
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
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "sam2", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// A COMPLETE, real `torch.save` checkpoint at the released `hiera_tiny`
/// preset's shape: every tensor `sam2::import::manifest_for(cfg, Scope::
/// Image)` names, plus `no_mem_embed` (the image path reads it too, per
/// `crate::import`'s own doc), all zeros.
fn write_complete_sam2_checkpoint(path: &Path, cfg: &Sam2Config) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut tensors: Vec<checkpoint::torchpt_write::TensorOut> = sam2::import::manifest_for(cfg, sam2::import::Scope::Image)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            checkpoint::torchpt_write::TensorOut { name, shape, data: vec![0.0f32; n] }
        })
        .collect();
    tensors.push(checkpoint::torchpt_write::TensorOut { name: sam2::import::NO_MEM_EMBED.to_string(), shape: vec![1, 1, cfg.d_model as usize], data: vec![0.0f32; cfg.d_model as usize] });
    checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::SegmentPipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "sam2-empty-test");

    let err = with_models_dir(&root, || brain::SegmentPipeline::from_pretrained("local/sam2-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "sam2");
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
/// network access at any point, all the way to a real `.segment()`:
/// reference parses, `Store::local` resolves it, `loader::resolve_structured`
/// finds exactly one candidate for `"weights"` and derives the `tiny`
/// variant from the checkpoint's own trunk width, `sam2::caps::load` imports
/// the image-path tensors and builds the whole real graph on a real `Gpu`,
/// and `.segment()` runs a genuine forward pass through the trunk, FPN neck,
/// prompt encoder and mask decoder. All-zero weights make the mask
/// meaningless, but every kernel dispatch on the real path runs for real, at
/// the model's real fixed 1024x1024 geometry - SAM 2.1 has no smaller real
/// config to fall back to, the same ceiling `RestorePipeline`'s own ~70-90s
/// fixture accepts for the same reason.
#[test]
fn from_pretrained_segments_a_real_complete_fixture_end_to_end() {
    let root = scratch_root("segment");
    write_complete_sam2_checkpoint(&root.join("facebook").join("sam2.1_hiera_tiny.pt"), &Sam2Config::hiera_tiny());
    mark_locally_present(&root, "local", "sam2-sdk-test");

    let pipe = with_models_dir(&root, || brain::SegmentPipeline::from_pretrained("local/sam2-sdk-test")).expect("a complete, real checkpoint must import and build");

    let input = rgb8_gradient(32, 32);
    let mask = pipe.segment(&input, &brain::Prompt::new().point(16.0, 16.0, true)).expect("a real forward pass over a complete checkpoint must succeed");

    assert_eq!(mask.width, 32, "the source image's own size, not the model's internal 1024x1024 frame");
    assert_eq!(mask.height, 32);
    assert_eq!(mask.probabilities.len(), 32 * 32);
    assert!(mask.probabilities.iter().all(|p| (0.0..=1.0).contains(p)), "{:?}", mask.probabilities);
    assert!((0.0..=1.0).contains(&mask.confidence), "{}", mask.confidence);
}

/// An empty [`brain::Prompt`] is a clean, named error before any GPU work,
/// never a panic.
#[test]
fn segment_rejects_an_empty_prompt() {
    let root = scratch_root("empty-prompt");
    write_complete_sam2_checkpoint(&root.join("facebook").join("sam2.1_hiera_tiny.pt"), &Sam2Config::hiera_tiny());
    mark_locally_present(&root, "local", "sam2-prompt-test");

    let pipe = with_models_dir(&root, || brain::SegmentPipeline::from_pretrained("local/sam2-prompt-test")).expect("a complete, real checkpoint must import and build");

    let input = rgb8_gradient(8, 8);
    let err = pipe.segment(&input, &brain::Prompt::new()).unwrap_err();
    assert!(matches!(err, brain::Error::MissingArgument(_)), "{err:?}");
}
