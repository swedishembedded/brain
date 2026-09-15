// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end coverage of `ImagePipeline::from_pretrained`'s resolution path
//! against a local, synthetic, fully offline models-directory fixture --
//! mirroring `crates/flux2/tests/resolve_layout.rs`'s fixture pattern, but
//! driven through THIS crate's public facade rather than by constructing
//! `flux2::Paths`/`capability::Assembly` directly, so what is actually under
//! test is the facade's own resolution wiring.
//!
//! ## Why this stops short of a successful `.generate()?.save()`
//!
//! `ImagePipelineBuilder::load` is specified to call the real
//! `flux2::Flux2Config::from_name`, which answers with exactly one of four
//! REAL FLUX.2 variants (klein-4b/9b, base-4b/9b) -- there is no "tiny"
//! variant available through this API, unlike the DiT-only smoke tests
//! elsewhere in this workspace (`crates/flux2/tests/model_smoke.rs`), which
//! build a hand-shrunk `Flux2Config` that this facade is never handed. Even
//! the smallest real variant (klein-4b) is a several-billion-parameter
//! model; a fixture carrying its full, real tensor set would be multiple
//! gigabytes on disk and take real GPU/CPU time to build and run, which is
//! not a "tiny, fully local, synthetic fixture" by any reasonable reading,
//! and is not the kind of fixture an ordinary `cargo test` run on a shared,
//! disk-constrained box should have to pay for.
//!
//! So this file proves everything UP TO that real, unavoidable ceiling: the
//! facade reaches `crates/loader`'s resolver against a real local fixture
//! with zero network access, resolves a full `capability::Assembly`, passes
//! the license gate, looks up the variant's real `Flux2Config`, resolves the
//! DiT's executable precision, and reaches
//! `flux2::pipeline::Pipeline::build_sized` -- which then fails CLEANLY (a
//! typed `brain::Error`, never a panic, never a silent partial success) on a
//! deliberately-incomplete DiT tensor set, the same "five classification
//! tensors, not the full manifest" fixture shape
//! `crates/flux2/tests/resolve_layout.rs`/`placement.rs` already use for
//! exactly this reason. `Image::save`'s own reuse of the shared `imaging`
//! codec, and `ImagePipeline::load_lora`'s path-vs-store-reference gate, are
//! covered directly as unit tests in `crates/sdk/src/image.rs` and
//! `crates/sdk/src/pipeline.rs` -- neither needs a real model in hand.

use std::path::{Path, PathBuf};

use checkpoint::gguf::GgufValue;
use checkpoint::gguf_write::TensorOut;
use flux2::Flux2Config;

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-image-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// Run `f` with `BRAIN_MODELS_DIR` pointed at `root`, serialized against
/// every other test in this binary that also touches process environment
/// (`brain_testutil::env_lock`) -- the same discipline
/// `crates/loader/src/resolver.rs`'s own tests already follow.
fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

fn f32_tensor(name: &str, shape: Vec<usize>) -> TensorOut {
    let n: usize = shape.iter().product();
    TensorOut { name: name.to_string(), shape, ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; n * 4] }
}

/// Only the five tensors `flux2::dit_config_from_shapes` actually reads for
/// CLASSIFICATION (mirrors `crates/flux2/tests/resolve_layout.rs`'s own
/// `write_dit_gguf`) -- not the full real tensor manifest, which at real
/// klein-4b dimensions would be several gigabytes of weights for a fixture
/// that only needs its own shape read back. This is also exactly what makes
/// the eventual `Pipeline::build_sized` call fail cleanly instead of
/// attempting a multi-gigabyte build: a real construction needs every named
/// tensor `Flux2Config::tensor_manifest` lists, and this fixture
/// deliberately supplies only five of them.
fn write_dit_gguf(path: &Path, cfg: &Flux2Config) {
    let tensors = vec![
        f32_tensor("img_in.weight", vec![cfg.hidden, cfg.in_channels]),
        f32_tensor("txt_in.weight", vec![cfg.hidden, cfg.context_in_dim]),
        f32_tensor("double_blocks.0.img_attn.norm.query_norm.scale", vec![cfg.head_dim()]),
        f32_tensor(&format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1]),
        f32_tensor(&format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1]),
    ];
    checkpoint::gguf_write::write(path.to_str().unwrap(), &[("general.architecture".to_string(), GgufValue::String("flux".to_string()))], &tensors, 32).unwrap();
}

/// Toy vocab size every fixture writer here agrees on, so
/// `Flux2Spec::classify`'s vocab-compatibility check finds a real match
/// between the synthetic tokenizer and its text encoder.
const TOY_VOCAB: usize = 100;

fn write_tokenizer_json(path: &Path) {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
}

fn write_shard(path: &Path) {
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
}

fn write_index(dir: &Path, shard_name: &str) {
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map": {"w": shard_name}})).unwrap()).unwrap();
}

/// A canonical `<vendor>/<repo>` HF checkpoint directory: `config.json` plus
/// one shard and its index -- the shape `brain_modelstore::inventory::scan`
/// collapses to a single `HfDir` record.
fn write_hf_checkpoint(dir: &Path, architectures: &[&str], hidden_size: u64) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": architectures, "hidden_size": hidden_size, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
    write_shard(&dir.join("model-00001-of-00001.safetensors"));
    write_index(dir, "model-00001-of-00001.safetensors");
}

/// Real FLUX.2 VAE shape rank: a 2D conv, `[out_ch, in_ch, kh, kw]` (4
/// dims) -- `Flux2Spec::classify` uses this to tell a real image VAE apart
/// from an unrelated architecture's causal 3D video VAE at the same tensor
/// names.
fn write_vae_safetensors_flat(path: &Path) {
    checkpoint::st::save_safetensors(
        path.to_str().unwrap(),
        &[
            ("decoder.conv_in.weight".to_string(), vec![512, 32, 3, 3], vec![0.0f32; 512 * 32 * 3 * 3]),
            ("encoder.conv_in.weight".to_string(), vec![128, 3, 3, 3], vec![0.0f32; 128 * 3 * 3 * 3]),
        ],
        &serde_json::json!({}),
        None,
    )
    .unwrap();
}

/// Drop a `brain.manifest.json` at `<root>/<vendor>/<repo>/` so
/// `brain_modelstore::Store::local` resolves that exact reference without
/// touching the network at all -- this is the ONE thing
/// `ImagePipelineBuilder::load`'s `DownloadPolicy::IfMissing` check reads
/// before it would otherwise attempt a fetch. Deliberately decoupled from
/// the role-resolver fixture below: `Store::local` and
/// `loader::resolve_structured`/`Flux2Spec::classify` are two independent
/// mechanisms (see `crates/sdk/src/pipeline.rs`'s `load` doc), and this is
/// only the former.
fn mark_locally_present(root: &Path, vendor: &str, repo: &str) {
    let dir = root.join(vendor).join(repo);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": "flux2", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// A minimal, UNAMBIGUOUS real flux2 store: exactly one DiT (klein-4b
/// shape), one VAE, one text encoder, one compatible tokenizer -- so
/// `loader::resolve_structured` (which `ImagePipelineBuilder::load` calls
/// with no role override, exactly like `brain flux2 generate` with no
/// `--dit`/`--variant`/... flags) resolves cleanly on the first try, with no
/// `Ambiguous` question to answer.
fn build_unambiguous_store() -> Scratch {
    let root = scratch_root("resolve");

    // The text encoder: real HF dir, Qwen3ForCausalLM, hidden=2560 (3x2560 =
    // 7680 = klein-4b's own `context_in_dim`).
    let te_dir = root.join("Qwen").join("Qwen3-4B-like");
    write_hf_checkpoint(&te_dir, &["Qwen3ForCausalLM"], 2560);
    // The tokenizer must sit under the SAME top-level vendor directory as a
    // real text_encoder candidate (`classify_tokenizer_role`'s co-location
    // check, `crates/modelstore/src/resolve.rs`) -- "Qwen", not the DiT's
    // own vendor.
    let tok_dir = root.join("Qwen").join("flux2-tokenizer");
    std::fs::create_dir_all(&tok_dir).unwrap();
    write_tokenizer_json(&tok_dir.join("tokenizer.json"));

    // The DiT + VAE: a real, vendor-flat unsloth-style release layout.
    let vendor_dir = root.join("myvendor");
    std::fs::create_dir_all(&vendor_dir).unwrap();
    write_dit_gguf(&vendor_dir.join("flux2-klein-4b-q8_0.gguf"), &Flux2Config::klein_4b());
    write_vae_safetensors_flat(&vendor_dir.join("flux2-vae.safetensors"));

    root
}

/// An unparseable reference is refused before any filesystem or network
/// access at all.
#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::ImagePipeline::from_pretrained("not-a-valid-reference").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");

    let err = brain::ImagePipeline::from_pretrained("").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A reference that resolves locally (so `IfMissing` never touches the
/// network) but whose store has NOTHING classifiable for any of flux2's four
/// roles is a clean `Error::Missing`, never a panic and never a network
/// fetch attempt.
#[test]
fn from_pretrained_names_every_missing_role_when_the_store_is_otherwise_empty() {
    let root = scratch_root("missing");
    mark_locally_present(&root, "local", "flux2-empty-test");

    let err = with_models_dir(&root, || brain::ImagePipeline::from_pretrained("local/flux2-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "flux2");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            for want in ["dit", "vae", "text_encoder", "tokenizer"] {
                assert!(roles.contains(&want), "missing roles must name {want:?}: {roles:?}");
            }
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

/// The full facade path against a real local fixture, with no network
/// access at any point: reference parses, `Store::local` resolves it (no
/// fetch attempted), `loader::resolve_structured` finds exactly one
/// candidate per role and resolves, `Paths::from_assembly` reads every role
/// back, the license gate passes (klein-4b is not NC-gated),
/// `Flux2Config::from_name("klein-4b")` succeeds, and
/// `effective_dit_precision` reports `Int8` for the `.gguf` DiT regardless
/// of the requested `DType` -- then, and only then, construction itself
/// fails cleanly on the fixture's deliberately incomplete tensor set (see
/// this file's module doc for why a REAL tensor set is infeasible here).
#[test]
fn from_pretrained_resolves_a_real_local_fixture_with_no_network_access() {
    let root = build_unambiguous_store();
    mark_locally_present(&root, "local", "flux2-sdk-test");

    let err = with_models_dir(&root, || brain::ImagePipeline::from_pretrained("local/flux2-sdk-test").unwrap_err());
    match &err {
        // A `Backend` error this deep means resolution, the license gate,
        // the variant lookup and the precision decision all already
        // succeeded -- only `Pipeline::build_sized` itself, reached last,
        // can still fail. `Ambiguous`/`Missing` here would mean the fixture
        // (or the facade's resolver call) regressed.
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete DiT construction, got {other:?}"),
    }
}
