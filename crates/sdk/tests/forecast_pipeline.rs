// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `forecast` surface, so it compiles only with
// it - the same reason `tests/image_pipeline.rs` gates itself on `image`.
#![cfg(feature = "forecast")]

//! End-to-end coverage of `ForecastPipeline::from_pretrained`'s resolution +
//! backend-dispatch path against local, synthetic, fully offline
//! models-directory fixtures - mirroring `tests/image_pipeline.rs`'s own
//! fixture pattern and, for the exact classification schemas, `crates/
//! kronos/src/spec.rs`/`crates/timesfm3/src/spec.rs`'s own test fixtures
//! (which are private to those crates, so reproduced here rather than
//! reused).
//!
//! ## Why this stops short of a successful `.forecast(...)`
//!
//! Neither `kronos::KronosForecaster::load` nor
//! `timesfm3::Timesfm3Forecaster::load` takes an injectable tiny config -
//! both read a real checkpoint's own `config.json` and then expect every
//! weight tensor that shape implies. A fixture carrying the full real
//! tensor manifest for either model is not a "tiny, fully local, synthetic
//! fixture" by any reasonable reading (the same ceiling `tests/
//! image_pipeline.rs` documents for flux2/s3dit). So these tests prove
//! everything UP TO that real, unavoidable ceiling: the facade reaches
//! `crates/loader`'s resolver against a real local fixture with zero
//! network access, `resolve_arch` picks the right architecture off the
//! resolved `capability::Assembly::arch`, every role is read back, and the
//! architecture's own `Forecaster::load` is reached - which then fails
//! CLEANLY (a typed `brain::Error`, never a panic) on a deliberately
//! incomplete weight-tensor set.

use std::path::{Path, PathBuf};

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-forecast-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// Run `f` with `BRAIN_MODELS_DIR` pointed at `root`, serialized against
/// every other test in this binary that also touches process environment.
fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

/// Marks `<vendor>/<repo>` as already-locally-present under the reserved
/// `"local"` vendor, so `Store::local` resolves the reference with no fetch
/// attempted - the resolution/dispatch under test happens via a full-store
/// content scan regardless of this id, same as `tests/image_pipeline.rs`'s
/// identical helper.
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

/// The exact nested schema `timesfm3::config::Timesfm3Config::from_hf_config_json`
/// reads - mirrors `crates/timesfm3/src/spec.rs`'s own private test fixture.
/// Unlike that fixture (which hands `classify` a manually-built
/// `ArtifactRecord`), this one goes through a REAL `inventory::scan`, which
/// needs a loose `model*.safetensors` shard beside `config.json` to collapse
/// the directory into an `HfDir` record at all - same reason
/// `write_kronos_hfdir` below carries one.
fn write_timesfm3_hfdir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = serde_json::json!({
        "input_patch_len": 32,
        "output_patch_len": 64,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "transformer_config": {
            "num_layers": 20,
            "transformer": { "model_dims": 1280, "hidden_dims": 1280, "num_heads": 16, "max_variates": 32 }
        }
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

/// A real HF-shaped directory whose `config.json` classifies against
/// `config` - mirrors `crates/kronos/src/spec.rs`'s own private
/// `write_hfdir`. `hfdir_record` needs a loose `model*.safetensors` shard
/// beside `config.json` to collapse to an `HfDir` record at all.
fn write_kronos_hfdir(dir: &Path, config: &serde_json::Value) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

fn kronos_decoder_json() -> serde_json::Value {
    serde_json::json!({"d_model": 512, "n_layers": 8, "n_heads": 8, "ff_dim": 1024, "s1_bits": 10, "s2_bits": 10, "learn_te": true, "dep_n_heads": 4, "max_context": 512})
}

fn kronos_tokenizer_json() -> serde_json::Value {
    serde_json::json!({"d_in": 6, "d_model": 256, "n_heads": 4, "ff_dim": 512, "n_enc_layers": 4, "n_dec_layers": 4, "s1_bits": 10, "s2_bits": 10, "group_size": 4})
}

/// The full facade path against a real local timesfm3 fixture, with no
/// network access at any point: reference parses, `Store::local` resolves
/// it (no fetch attempted), `resolve_arch` tries kronos first (finds
/// nothing - there is nothing kronos-shaped in this store), then timesfm3
/// and finds exactly one candidate for its one `weights` role, then
/// `timesfm3::Timesfm3Forecaster::load` itself fails cleanly on the
/// fixture's deliberately incomplete weight-tensor set (config only, no
/// real checkpoint tensors - see this file's module doc).
#[test]
fn from_pretrained_resolves_timesfm3_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("timesfm3");
    write_timesfm3_hfdir(&root.join("google").join("timesfm-3.0-pytorch"));
    mark_locally_present(&root, "local", "forecast-sdk-test-timesfm3", "timesfm3");

    let err = with_models_dir(&root, || brain::ForecastPipeline::from_pretrained("local/forecast-sdk-test-timesfm3").unwrap_err());
    match &err {
        // A `Backend` error this deep means resolution and role extraction
        // both already succeeded - only `Timesfm3Forecaster::load` itself,
        // reached last, can still fail.
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete weight construction, got {other:?}"),
    }
}

/// The full facade path against a real local kronos fixture (two repos:
/// tokenizer + decoder), with no network access at any point: `resolve_arch`
/// tries kronos first and finds exactly one candidate per role, then
/// `kronos::KronosForecaster::load` itself fails cleanly on the fixture's
/// deliberately incomplete weight-tensor set.
#[test]
fn from_pretrained_resolves_kronos_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("kronos");
    write_kronos_hfdir(&root.join("NeoQuasar").join("Kronos-base"), &kronos_decoder_json());
    write_kronos_hfdir(&root.join("NeoQuasar").join("Kronos-Tokenizer-base"), &kronos_tokenizer_json());
    mark_locally_present(&root, "local", "forecast-sdk-test-kronos", "kronos");

    let err = with_models_dir(&root, || brain::ForecastPipeline::from_pretrained("local/forecast-sdk-test-kronos").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete weight construction, got {other:?}"),
    }
}

/// An unparseable reference is refused before any filesystem/network work -
/// mirrors `tests/image_pipeline.rs`'s identical case.
#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::ForecastPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}
