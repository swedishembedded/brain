// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `image` surface, so it compiles only with it.
// Without this, `cargo test -p brain --no-default-features --features
// <anything else>` fails to build a test that was never meant to run - which
// looks like the other surface being broken.
#![cfg(feature = "image")]

//! End-to-end coverage of `ImagePipeline::from_pretrained`'s resolution +
//! backend-dispatch path against a local, synthetic, fully offline
//! models-directory fixture -- mirroring `crates/flux2/tests/
//! resolve_layout.rs`'s fixture pattern (and, for the s3dit half,
//! `crates/cli/src/resident.rs`'s own `ZImageResident::from_store` synthetic
//! fixture), but driven through THIS crate's public facade rather than by
//! constructing `flux2::Paths`/`s3dit::pipeline::Paths`/`capability::Assembly`
//! directly, so what is actually under test is the facade's own resolution
//! and backend-dispatch (`pipeline::resolve_arch`, private to `brain`)
//! wiring.
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
//! s3dit's ceiling is the SAME shape, for an even more rigid reason:
//! `s3dit::pipeline`'s `dit_config`/`QwenConfig::qwen3_4b` are hardcoded to
//! the one shipped Z-Image-Turbo/Qwen3-4B shape at every `HotPipeline::build*`
//! call site (`crates/s3dit/src/pipeline.rs`'s own `dit_config` doc, audit
//! F44) -- there is no config-injection seam AT ALL, not even the kind
//! flux2's `Pipeline::build_with(cfg: &Flux2Config, ...)` offers past
//! `from_name`'s four names. A real DiT+encoder pair at that shape is tens
//! of gigabytes; there is no way to make s3dit's OWN construction "tiny"
//! today, SDK or no SDK.
//!
//! So this file proves everything UP TO that real, unavoidable ceiling, for
//! BOTH backends: the facade reaches `crates/loader`'s resolver against a
//! real local fixture with zero network access, `resolve_arch` picks the
//! right backend off the resolved `capability::Assembly::arch`, the
//! architecture-specific `Paths::from_assembly` reads every role back, the
//! license gate runs where the backend has one (flux2 only), the
//! variant/config is looked up, the executable precision/hifi decision is
//! made, and the backend's own `build_sized`/`build_adapted` is reached --
//! which then fails CLEANLY (a typed `brain::Error`, never a panic, never a
//! silent partial success) on a deliberately-incomplete weight-tensor set,
//! the same "just the classification tensors, not the full manifest"
//! fixture shape `crates/flux2/tests/resolve_layout.rs`/`crates/s3dit/src/
//! spec.rs`'s own `turbo_fixture` test helper already use for exactly this
//! reason. `Image::save`'s own reuse of the shared `imaging` codec,
//! `Image::from_hwc_unit`'s float-HWC normalization, and
//! `ImagePipeline::load_lora`'s path-vs-store-reference gate, are covered
//! directly as unit tests in `crates/sdk/src/image.rs` and
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
/// `loader::resolve_structured`/`Flux2Spec::classify`/`S3ditSpec::classify`
/// are two independent mechanisms (see `crates/sdk/src/pipeline.rs`'s `load`
/// doc), and this is only the former -- `family` is unread by `Store::local`
/// itself, so it is a plain label here, not a second dispatch key.
fn mark_locally_present(root: &Path, vendor: &str, repo: &str, family: &str) {
    let dir = root.join(vendor).join(repo);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": family, "roles": {"weights": "weights.stub"}})).unwrap(),
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
    mark_locally_present(&root, "local", "flux2-empty-test", "flux2");

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

/// `DownloadPolicy::Offline` never reaches the network, proven the same way
/// `crates/modelstore/src/hub.rs`'s own tests prove `BRAIN_HUB_ENDPOINT`
/// wins over the real `huggingface.co` default: point `HfHub` at a loopback
/// port nothing listens on, then show a reference that resolves NEITHER
/// locally nor from any real hub still comes back as the resolver's own
/// clean `Error::Missing` rather than an `Error::Download` wrapping a
/// connection failure -- the ONLY way `Missing` (not a connection error) can
/// come back here is if `ImagePipelineBuilder::load`'s `Offline` arm never
/// called `brain_modelstore::plan`/`execute_plan` at all. The default
/// `DownloadPolicy::IfMissing` is deliberately NOT exercised against this
/// same bogus endpoint here -- that would need a real connect-refused round
/// trip to observe the contrast, which is exactly the network dependency
/// this test exists to avoid.
#[test]
fn download_policy_offline_never_touches_the_network() {
    let root = scratch_root("offline");
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", &*root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::ImagePipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
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
    mark_locally_present(&root, "local", "flux2-sdk-test", "flux2");

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

// ===================== s3dit-backed dispatch =====================
//
// Mirrors `crates/cli/src/resident.rs`'s own `ZImageResident::from_store`
// synthetic-store test (`write_from_store_dit`/`_text_encoder`/`_vae`/
// `_tokenizer`) and `crates/s3dit/src/spec.rs`'s own `turbo_fixture` test
// helper: only the tensors `s3dit::import::dit_config_from_shapes` (DiT
// classification) and `S3ditSpec::validate` (the text-encoder hidden-size
// cross-check) actually read, at real `ZImageConfig::turbo()` dimensions --
// not the full per-layer weight manifest, which at real dimensions would be
// tens of gigabytes (see this file's module doc).

/// Only the tensors `s3dit::import::dit_config_from_shapes` actually reads
/// for CLASSIFICATION, at real `ZImageConfig::turbo()` dimensions.
fn write_s3dit_dit(path: &Path) {
    let cfg = s3dit::ZImageConfig::turbo();
    let (dim, cap_feat_dim, head_dim) = (cfg.dim as usize, cfg.cap_feat_dim as usize, (cfg.dim / cfg.n_heads) as usize);
    let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
        // The DISCRIMINATOR tensor `S3ditSpec::classify` gates a `dit`
        // candidate on (`cap_embedder.0.weight` --
        // `crate::import::DISCRIMINATOR_TENSOR`, `pub(crate)` to `s3dit`, so
        // named literally here rather than imported).
        ("cap_embedder.0.weight".to_string(), vec![1], vec![0.0f32]),
        ("cap_embedder.1.weight".to_string(), vec![dim as u64, cap_feat_dim as u64], vec![0.0f32; dim * cap_feat_dim]),
        ("layers.0.attention.q_norm.weight".to_string(), vec![head_dim as u64], vec![0.0f32; head_dim]),
        ("x_embedder.weight".to_string(), vec![dim as u64, patch_dim as u64], vec![0.0f32; dim * patch_dim]),
    ];
    for prefix in ["layers", "noise_refiner", "context_refiner"] {
        let n = if prefix == "layers" { cfg.n_layers } else { cfg.n_refiner_layers };
        for l in 0..n {
            tensors.push((format!("{prefix}.{l}.attention.qkv.weight"), vec![1], vec![0.0f32]));
        }
    }
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
}

/// A canonical `<vendor>/<repo>` HF text-encoder directory: `config.json`
/// declaring `Qwen3ForCausalLM` at `hidden` (must equal the DiT's own
/// `cap_feat_dim` for `S3ditSpec::validate` to accept the pair), plus a real
/// (tiny) shard and its index -- the shape `brain_modelstore::inventory::
/// scan` collapses to one `HfDir` record.
fn write_s3dit_text_encoder(dir: &Path, hidden: u64) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
    write_shard(&dir.join("model-00001-of-00001.safetensors"));
    write_index(dir, "model-00001-of-00001.safetensors");
}

/// Real Z-Image VAE shape: the same generic `decoder`/`encoder`.conv_in.weight
/// autoencoder names FLUX.2's own VAE carries, at a 2D-conv (4-dim) shape --
/// `S3ditSpec::classify_safetensors`'s rank guard is what tells it apart
/// from an unrelated causal 3D video VAE at the same tensor names.
fn write_s3dit_vae(path: &Path) {
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

/// A minimal, UNAMBIGUOUS real s3dit store: exactly one DiT (turbo shape),
/// one VAE, one compatible text encoder, one co-located tokenizer -- so
/// `loader::resolve_structured("s3dit", ...)` (which `resolve_arch` tries
/// once flux2 does not resolve, exactly like `brain do z-image text2image`
/// with no role override typed) resolves cleanly on the first try, with no
/// `Ambiguous` question to answer, and nothing flux2-shaped anywhere in the
/// store for `resolve_arch`'s flux2-first attempt to pick up instead.
fn build_unambiguous_s3dit_store() -> Scratch {
    let root = scratch_root("s3dit-resolve");
    let cap_feat_dim = s3dit::ZImageConfig::turbo().cap_feat_dim as u64;

    let vendor = root.join("Tongyi-MAI");
    std::fs::create_dir_all(&vendor).unwrap();
    write_s3dit_dit(&vendor.join("dit.safetensors"));
    write_s3dit_text_encoder(&vendor.join("Qwen3-4B"), cap_feat_dim);
    write_s3dit_vae(&vendor.join("vae.safetensors"));
    // A vendor-flat loose `tokenizer.json` sitting one level under a
    // repo-shaped directory (`classify_tokenizer_role`'s co-location check,
    // and `walk_vendor_dir`'s "one level deeper" rule -- see
    // `crates/cli/src/resident.rs`'s own `from_store` fixture test, which
    // notes the exact same rule).
    let tok_dir = vendor.join("Qwen3-4B-tokenizer");
    std::fs::create_dir_all(&tok_dir).unwrap();
    write_tokenizer_json(&tok_dir.join("tokenizer.json"));

    root
}

/// The full facade path against a real local s3dit fixture, with no network
/// access at any point: reference parses, `Store::local` resolves it (no
/// fetch attempted), `resolve_arch` tries flux2 first (finds nothing --
/// there is nothing flux2-shaped in this store), then tries s3dit and finds
/// exactly one candidate per role, `s3dit::pipeline::Paths::from_assembly`
/// reads every role back (s3dit has no license gate to run), `hifi = dtype
/// == DType::F32` resolves -- then, and only then, `HotPipeline::
/// build_adapted` itself fails cleanly on the fixture's deliberately
/// incomplete tensor set (see this file's module doc for why a REAL tensor
/// set is infeasible here, for s3dit even more rigidly than for flux2).
#[test]
fn from_pretrained_dispatches_to_s3dit_and_resolves_a_real_local_fixture_with_no_network_access() {
    let root = build_unambiguous_s3dit_store();
    mark_locally_present(&root, "local", "s3dit-sdk-test", "zimage");

    let err = with_models_dir(&root, || brain::ImagePipeline::from_pretrained("local/s3dit-sdk-test").unwrap_err());
    match &err {
        // A `Backend` error this deep means `resolve_arch` correctly picked
        // s3dit (not flux2, and not an `Ambiguous`/`Missing` regression) and
        // `Paths::from_assembly` succeeded -- only `HotPipeline::
        // build_adapted` itself, reached last, can still fail.
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete DiT construction, got {other:?}"),
    }
}

/// [`ImagePipelineBuilder::load_with_progress`]'s `on_build` closure: on the
/// SAME fixture and failure as the test above, `HotPipeline::build_adapted`
/// reports its first stage ("loading tokenizer") before it ever reaches the
/// fixture's deliberately incomplete DiT tensor set -- so a caller watching
/// build progress sees at least that one message even though the call still
/// ends in the same clean `Error::Backend`. `on_download` never fires here:
/// `Store::local` resolves the fixture with no network access at all (see
/// the test above's own doc), the same "never called when nothing was
/// fetched" contract [`ImagePipelineBuilder::load_with_progress`]'s own doc
/// states.
#[test]
fn load_with_progress_reports_s3dit_build_stages_before_the_same_clean_failure() {
    let root = build_unambiguous_s3dit_store();
    mark_locally_present(&root, "local", "s3dit-sdk-test", "zimage");

    let mut build_msgs: Vec<String> = Vec::new();
    let mut download_calls = 0u32;
    let err = with_models_dir(&root, || {
        brain::ImagePipeline::builder("local/s3dit-sdk-test")
            .load_with_progress(&mut |_name, _got, _total| download_calls += 1, &mut |msg| build_msgs.push(msg.to_string()))
            .unwrap_err()
    });
    assert!(matches!(err, brain::Error::Backend(_)), "expected the same clean Error::Backend the no-progress test gets, got {err:?}");
    assert!(!build_msgs.is_empty(), "HotPipeline::build_adapted must report at least its first stage before failing on the incomplete tensor set");
    assert_eq!(download_calls, 0, "a locally-resolved fixture must never invoke on_download");
}
