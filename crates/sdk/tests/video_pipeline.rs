// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "video")]

//! End-to-end coverage of `VideoPipeline::from_pretrained`'s resolution path
//! against a real local, synthetic, fully offline fixture reproducing
//! `crates/wan/src/spec.rs`'s own (private) `t2v_1_3b_fixture` shape -
//! mirroring `tests/tts_pipeline.rs`'s established pattern for this exact
//! class of ceiling.
//!
//! ## Why this stops short of a successful `.generate(...)`
//!
//! Wan's resolver only needs enough real tensor SHAPES to derive a variant
//! and satisfy `WanSpec::validate`'s text_dim cross-check
//! (`crate::import::dit_config_from_shapes` counts marker tensors and reads
//! `patch_embedding.weight`'s shape; nothing about `classify`/`validate`
//! reads a block's real weight content) - so resolution succeeds against a
//! MUCH smaller fixture than a real 1.3B-parameter checkpoint. But
//! `wan::pipeline::generate_hot` needs every block's real weights to build
//! the transformer at all, which this fixture does not carry (only one
//! marker tensor per block, the same discipline `wan::spec::tests`' own
//! fixture uses to stay fast) - so `.generate()` fails cleanly
//! (`Error::Backend`, never a panic) once it tries to build the DiT from an
//! incomplete tensor set, the same "resolution proven, full forward pass
//! not" scope `TtsPipeline`'s/`TranscribePipeline`'s own tests already
//! accepted for their own (different) ceiling.

use std::path::{Path, PathBuf};

use wan::config::WanConfig;

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-video-pipeline-{tag}-{}-{n}", std::process::id()));
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

/// Only the tensors `wan::import::dit_config_from_shapes` actually reads
/// (mirrors `wan::spec::tests::write_dit_safetensors`) at real t2v-1.3B
/// dimensions, NOT the full ~800-tensor manifest.
fn write_dit_safetensors(path: &Path, cfg: &WanConfig) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let (dim, in_ch, pt, ph, pw) = (cfg.dim, cfg.in_channels, cfg.patch_size.0, cfg.patch_size.1, cfg.patch_size.2);
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        vec![("patch_embedding.weight".to_string(), vec![dim as u64, in_ch as u64, pt as u64, ph as u64, pw as u64], vec![0.0f32; dim * in_ch * pt * ph * pw])];
    for l in 0..cfg.num_layers {
        tensors.push((format!("blocks.{l}.marker"), vec![1], vec![0.0f32]));
    }
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
}

/// Only the two tensors `wan::spec`'s own `is_wan_vae` checks, at a tiny (not
/// real) channel width - real dims would write hundreds of MB of zeros.
fn write_vae_pth(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            checkpoint::torchpt_write::TensorOut { name: "encoder.conv1.weight".to_string(), shape: vec![4, 3, 3, 3, 3], data: vec![0.0; 4 * 3 * 3 * 3 * 3] },
            checkpoint::torchpt_write::TensorOut { name: "decoder.conv1.weight".to_string(), shape: vec![4, 4, 3, 3, 3], data: vec![0.0; 4 * 4 * 3 * 3 * 3] },
        ],
    )
    .unwrap();
}

const TOY_VOCAB: usize = 100;

fn write_t5_pth(path: &Path, d_model: usize) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            checkpoint::torchpt_write::TensorOut { name: "token_embedding.weight".to_string(), shape: vec![TOY_VOCAB, d_model], data: vec![0.0; TOY_VOCAB * d_model] },
            checkpoint::torchpt_write::TensorOut { name: "blocks.0.attn.q.weight".to_string(), shape: vec![d_model, d_model], data: vec![0.0; d_model * d_model] },
        ],
    )
    .unwrap();
}

fn write_tokenizer_json(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
}

/// The exact on-disk shape `wan::spec::WanSpec::classify` reads for a
/// t2v-1.3B assembly, real dims for everything `dit_config_from_shapes`/
/// `validate` check, toy content everywhere else. The DiT file is named
/// `diffusion_pytorch_model.safetensors` - a REAL Wan release's own
/// diffusers-convention filename (`wan::spec::tests::t2v_1_3b_fixture`'s own
/// naming, mirrored here rather than reinvented), deliberately NOT
/// `brain_modelstore::Store::BASE_WEIGHTS_FILE`'s magic fallback name: this
/// repo has no `brain.manifest.json` either (a manifest's presence collapses
/// the WHOLE directory into one `ArtifactKind::Compound` record, real
/// scanner behavior found while building this fixture, which would hide
/// every other file in it from `WanSpec::classify`'s own content-based,
/// per-file checks), so a REAL Wan release's own layout genuinely satisfies
/// neither of `Store::local`'s two recognized shapes - proving
/// `VideoPipelineBuilder::load`'s resolve-first ordering (see that
/// function's own doc) actually matters, not just a convenient coincidence
/// of this fixture's file naming.
fn write_wan_checkpoint(repo: &Path) {
    let cfg = WanConfig::t2v_1_3b();
    write_dit_safetensors(&repo.join("diffusion_pytorch_model.safetensors"), &cfg);
    write_vae_pth(&repo.join("Wan2.1_VAE.pth"));
    write_t5_pth(&repo.join("models_t5_umt5-xxl-enc-bf16.pth"), cfg.text_dim);
    write_tokenizer_json(&repo.join("tokenizer.json"));
}

/// `resolve()`'s own root inference needs a second, unrelated vendor - a
/// SIBLING of the fixture's own vendor directory, under `root` - to land on
/// `root` rather than collapsing onto the fixture's single repo directory,
/// the same requirement `wan::spec::tests`' own fixture documents.
fn write_unrelated_sibling(root: &Path) {
    let dir = root.join("other-vendor");
    std::fs::create_dir_all(&dir).unwrap();
    // `.gguf`, not `.bin`: `inventory::scan` only emits a record for an
    // extension `kind_of_extension` recognizes - a real gap `wan::spec::
    // tests`' own fixture never caught, since it hand-builds `ArtifactRecord`s
    // directly rather than going through the real scanner (the SAME class of
    // "never verified via the real scanner" gap this campaign has found and
    // fixed in three separate `ArchSpec`s already).
    std::fs::write(dir.join("unrelated.gguf"), b"not a real gguf").unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::VideoPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A locally-present repo (satisfying `Store::local`'s own
/// `BASE_WEIGHTS_FILE` fallback with a real but content-unrelated
/// safetensors file - no `patch_embedding.weight`, so `WanSpec::classify`
/// never recognizes it as `dit`, and no `vae`/`text_encoder`/`tokenizer`
/// files at all) surfaces as `Error::Missing`, naming the arch and every
/// missing role.
#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    let dir = root.join("local").join("wan-empty-test");
    std::fs::create_dir_all(&dir).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.brain.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    write_unrelated_sibling(&root);

    let err = with_models_dir(&root, || brain::VideoPipeline::from_pretrained("local/wan-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "wan");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            assert!(roles.contains(&"dit") && roles.contains(&"vae") && roles.contains(&"text_encoder") && roles.contains(&"tokenizer"), "{roles:?}");
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

/// The full facade path against a real, content-classified local fixture,
/// with no network access at any point: reference parses, `loader::
/// resolve_structured` (tried FIRST - see `VideoPipelineBuilder::load`'s own
/// doc for why) finds all four roles via `WanSpec::classify`'s real content
/// checks against the fixture's REAL-WORLD filenames (see
/// `write_wan_checkpoint`'s own doc - `Store::local` itself would find
/// nothing here, since neither of its two recognized shapes matches a real
/// Wan release's own layout) and derives the `t2v-1.3B` variant from the
/// DiT's own shapes, and `VideoPipelineBuilder::load` builds a real
/// `VideoPipeline`. `.generate()` then reaches `wan::pipeline::generate_hot`,
/// which fails cleanly on the incomplete (marker-only) DiT tensor set rather
/// than resolution itself failing.
#[test]
fn from_pretrained_resolves_wan_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("resolve");
    let dir = root.join("local").join("wan-sdk-test");
    write_wan_checkpoint(&dir);
    write_unrelated_sibling(&root);

    let pipe = with_models_dir(&root, || brain::VideoPipeline::from_pretrained("local/wan-sdk-test"));
    let pipe = pipe.expect("a content-classifiable checkpoint (fake tensor content, real shapes) must resolve and build");

    let err = pipe.generate("a whale submarine").unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from the incomplete DiT tensor set, got {other:?}"),
    }
}
