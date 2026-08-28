// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain models {list,list-adapters,info}` over a fixture store: no GPU
//! (never triggers device work by itself), no network (never calls the hub).
//!
//! The one property every plain-mode assertion below checks is the reason
//! this command exists in this shape: piping the output through `grep` for a
//! quant tag, a repo, or an adapter tag must return a complete, self
//! -explanatory line, never a fragment that only makes sense next to a parent
//! row above it.
//!
//! Swedish Embedded AB implements dependable command-line tooling for
//! embedded and edge-AI systems. If your team needs a model-fleet inventory
//! that is provably honest about what is and is not on disk, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::process::Command;

use brain_modelstore::Store;
use checkpoint::st::{Adapter, ModelCard};

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

fn run(models_dir: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin()).arg("models").args(args).env("BRAIN_MODELS_DIR", models_dir).env_remove("BRAIN_PIPELINE_CACHE_DIR").output().expect("run brain models");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-cli-models-list-test-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A minimal-but-real qwen3 checkpoint: small enough for `checkpoint::st`'s
/// eager writer, real enough for `WeightReader::open` (header-only) to parse
/// its config and card exactly the way a real pulled model would.
fn write_base(store: &Store, vendor: &str, repo: &str, quant: Option<&str>) -> PathBuf {
    let dir = store.repo_dir(&brain_modelref::ModelRef::new(vendor, repo, None));
    std::fs::create_dir_all(&dir).unwrap();
    let file = match quant {
        None => dir.join("model.brain.safetensors"),
        Some(q) => dir.join(format!("{q}.gguf")),
    };
    let card = ModelCard::for_ref(&format!("{vendor}/{repo}"), vendor, repo, quant, "qwen3");
    // Real `QwenConfig::from_json` key spellings (`vocab_size`, `d_ff`,
    // `rope_theta`, `rms_norm_eps`) - `QwenConfig::from_json_checked` (now
    // what `modelcost`'s pricers call) refuses anything less, so a fixture
    // using the wrong names would misprice itself exactly the way a bug
    // this session traced back to a mis-keyed fixture did.
    let config = serde_json::json!({
        "n_layers": 2, "d_model": 8, "n_heads": 2, "n_kv_heads": 1, "head_dim": 4,
        "vocab_size": 16, "block_size": 32, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "d_ff": 32,
    });
    let tensors = vec![("tok_embeddings.weight".to_string(), vec![2, 4], vec![0.0f32; 8])];
    if quant.is_none() {
        checkpoint::st::save_safetensors(file.to_str().unwrap(), &tensors, &config, Some(&card)).unwrap();
    } else {
        // `Store::scan` opens every `.gguf` it finds (`WeightReader::open`) to
        // read its card - a file that fails to parse as GGUF is silently
        // skipped, same as any other unreadable entry. So the fixture must be
        // a REAL (if minimal) GGUF, not just a `.gguf`-named blob: real magic,
        // header, KV (`general.architecture` is what `resolve_arch` reads
        // back via `brain_arch::by_gguf`), and one real F32 tensor.
        const T_F32: u32 = 0;
        let data: Vec<u8> = [0.0f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        checkpoint::gguf_write::write(
            file.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String("qwen3".to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "tok_embeddings.weight".to_string(), shape: vec![2, 4], ty: T_F32, data }],
            32,
        )
        .unwrap();
    }
    file
}

fn write_adapter(store: &Store, vendor: &str, repo: &str, owner: &str, name: &str, tag: &str) {
    let adapter_ref = brain_modelref::ModelRef::new_adapter(vendor, repo, None, brain_modelref::AdapterRef::new(owner, name, tag));
    let path = store.adapter_weights_path(&adapter_ref).unwrap();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut card = ModelCard::for_ref(&adapter_ref.to_string(), vendor, repo, None, "qwen3");
    card.adapter = Some(Adapter {
        kind: "lora".to_string(),
        rank: Some(8),
        base: Some(format!("{vendor}/{repo}")),
        alpha: Some(16.0),
        targets: Some(vec!["q".to_string(), "k".to_string(), "v".to_string(), "o".to_string()]),
        dataset_id: Some("sql-2026".to_string()),
    });
    let tensors = vec![("blocks.0.attn.wq.lora_a".to_string(), vec![4, 1], vec![1.0f32; 4])];
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), Some(&card)).unwrap();
}

// ------------------------------------------------------------------- list --

#[test]
fn list_plain_leaf_lines_are_self_contained_and_greppable() {
    let dir = scratch("leaf-lines");
    let store = Store::new(dir.clone());
    write_base(&store, "Qwen", "Qwen3-0.6B", None);
    write_base(&store, "Qwen", "Qwen3-0.6B", Some("Q4_K_M"));

    let (ok, stdout, stderr) = run(&dir, &["list", "--arch", "qwen3", "--plain"]);
    assert!(ok, "brain models list failed: {stderr}");

    // "Q4_K_M" alone is too loose a needle here - Qwen3-0.6B has no OFFICIAL
    // Q4_K_M variant (only the 4B+ sizes do; see `crates/arch`'s qwen3 row),
    // so this local file is a genuinely custom quant, and several OTHER
    // repos' declared-but-unpulled Q4_K_M rows also contain that substring.
    // The full canonical id is the needle a real user would actually grep
    // for, and it is unambiguous.
    let q4_lines: Vec<&str> = stdout.lines().filter(|l| l.contains("Qwen/Qwen3-0.6B-Q4_K_M")).collect();
    assert_eq!(q4_lines.len(), 1, "expected exactly one Qwen/Qwen3-0.6B-Q4_K_M line in:\n{stdout}");
    let line = q4_lines[0];
    assert!(line.contains("qwen3"), "leaf line must name its architecture on its own: {line:?}");
    assert!(line.contains("Qwen/Qwen3-0.6B-Q4_K_M"), "leaf line must carry the full canonical id, not just the quant token: {line:?}");
    assert!(line.contains("local"), "a pulled quant must say so on its own line: {line:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn list_declared_but_absent_variant_shows_not_pulled_never_a_fabricated_size() {
    let dir = scratch("declared-absent");
    // Nothing written - Qwen/Qwen3-8B is a real declared variant with no
    // local files at all.
    let (ok, stdout, stderr) = run(&dir, &["list", "--arch", "qwen3", "--plain"]);
    assert!(ok, "brain models list failed: {stderr}");
    let eightb: Vec<&str> = stdout.lines().filter(|l| l.contains("Qwen3-8B-Q4_K_M")).collect();
    assert_eq!(eightb.len(), 1);
    assert!(eightb[0].contains("not pulled"), "an undeclared-locally variant must say not pulled: {:?}", eightb[0]);
    assert!(!eightb[0].contains("GiB") && !eightb[0].contains("MiB"), "an unpulled row must never print a fabricated size: {:?}", eightb[0]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn list_local_flag_drops_declared_but_absent_rows() {
    let dir = scratch("local-flag");
    let store = Store::new(dir.clone());
    write_base(&store, "Qwen", "Qwen3-0.6B", None);

    let (ok, stdout, _) = run(&dir, &["list", "--arch", "qwen3", "--plain", "--local"]);
    assert!(ok);
    assert!(!stdout.contains("not pulled"), "--local must drop every declared-but-absent row:\n{stdout}");
    assert!(stdout.contains("Qwen/Qwen3-0.6B"), "the one real local repo must still be present:\n{stdout}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn list_unprofiled_local_model_reports_not_profiled_not_a_missing_column() {
    let dir = scratch("not-profiled");
    let store = Store::new(dir.clone());
    write_base(&store, "Qwen", "Qwen3-0.6B", None);
    // A dedicated, never-populated cache dir - this local model has genuinely
    // never been priced.
    let cache = scratch("not-profiled-cache");

    let out = Command::new(bin())
        .args(["models", "list", "--arch", "qwen3", "--plain"])
        .env("BRAIN_MODELS_DIR", &dir)
        .env("BRAIN_PIPELINE_CACHE_DIR", &cache)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The LEAF line specifically (arch-prefixed), not the repo header above
    // it - the header also carries the word "local" as its pulled/not-pulled
    // status and would satisfy a looser match by accident.
    let base_line = stdout.lines().find(|l| l.contains("qwen3 Qwen/Qwen3-0.6B ")).expect("the base leaf row must be present");
    assert!(base_line.contains("not profiled"), "a local model with an empty cost cache must say not profiled: {base_line:?}");

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&cache).ok();
}

#[test]
fn list_empty_or_missing_store_is_not_an_error() {
    // A store directory that does not exist on disk is not the same as "no
    // store configured at all" (`crate::model_dir::resolve` returning
    // `None`, which DOES render nothing - see `run_list`'s early return):
    // `BRAIN_MODELS_DIR` is set, so `list` still has an answer to "what
    // COULD be pulled here" from the declared registry alone. What it must
    // never do is claim anything is local, or invent a size, when nothing is
    // actually on disk.
    let missing = std::env::temp_dir().join("brain-cli-models-list-test-genuinely-missing-dir-xyz");
    std::fs::remove_dir_all(&missing).ok();
    let (ok, stdout, stderr) = run(&missing, &["list", "--arch", "qwen3", "--plain"]);
    assert!(ok, "a missing store must not fail: stderr={stderr}");
    assert!(!stdout.contains(" local "), "an empty store must claim nothing is local: {stdout:?}");
    assert!(stdout.contains("not pulled"), "the declared catalog should still render: {stdout:?}");
}

#[test]
fn models_profile_refuses_a_config_that_would_be_silently_mispriced() {
    // The real bug this session's investigation traced the "suspiciously low
    // FLOP count" report back to: a config using the WRONG key name for a
    // shape field (`"vocab"` instead of `"vocab_size"`) used to be silently
    // priced against a hardcoded fallback instead of the checkpoint's real
    // shape, with `brain models profile` reporting total confidence in the
    // wrong number. It must now be refused, loudly, naming the real key.
    let dir = scratch("mis-keyed-config");
    let store = Store::new(dir.clone());
    let file = store.repo_dir(&brain_modelref::ModelRef::new("Qwen", "Qwen3-0.6B", None)).join("model.brain.safetensors");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    let card = ModelCard::for_ref("Qwen/Qwen3-0.6B", "Qwen", "Qwen3-0.6B", None, "qwen3");
    let bad_config = serde_json::json!({"vocab": 16, "block_size": 32, "n_layers": 2, "d_model": 8, "n_heads": 2, "n_kv_heads": 1, "head_dim": 4});
    let tensors = vec![("tok_embeddings.weight".to_string(), vec![2, 4], vec![0.0f32; 8])];
    checkpoint::st::save_safetensors(file.to_str().unwrap(), &tensors, &bad_config, Some(&card)).unwrap();

    let (ok, stdout, stderr) = run(&dir, &["profile", "Qwen/Qwen3-0.6B"]);
    assert!(!ok, "a mis-keyed config must not report success");
    assert!(stdout.is_empty(), "nothing should print to stdout on refusal: {stdout:?}");
    assert!(stderr.contains("vocab_size"), "the error must name the real missing key: {stderr:?}");

    std::fs::remove_dir_all(&dir).ok();
}

// --------------------------------------------------------- list-adapters --

#[test]
fn list_adapters_reports_rank_alpha_base_from_the_adapters_own_card() {
    let dir = scratch("adapters");
    let store = Store::new(dir.clone());
    write_base(&store, "Qwen", "Qwen3-0.6B", None);
    write_adapter(&store, "Qwen", "Qwen3-0.6B", "acme", "sql", "v1");

    let (ok, stdout, stderr) = run(&dir, &["list-adapters", "--plain"]);
    assert!(ok, "brain models list-adapters failed: {stderr}");
    assert!(stdout.contains("qwen3"), "the adapter must be grouped under its architecture:\n{stdout}");
    assert!(stdout.contains("Qwen/Qwen3-0.6B"), "the adapter must be grouped under its base variant:\n{stdout}");
    let adapter_line = stdout.lines().find(|l| l.contains("acme:sql:v1")).expect("the adapter's own line must be present");
    assert!(adapter_line.contains("rank=8"), "{adapter_line:?}");
    assert!(adapter_line.contains("alpha=16"), "{adapter_line:?}");
    assert!(adapter_line.contains("dataset=sql-2026"), "{adapter_line:?}");
    assert!(adapter_line.contains("qwen3") && adapter_line.contains("Qwen/Qwen3-0.6B"), "the adapter's own line must be self-contained: {adapter_line:?}");

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------- info --

#[test]
fn info_lists_every_tensor_with_its_own_dtype_and_marks_adapter_tensors() {
    let dir = scratch("info");
    let store = Store::new(dir.clone());
    write_base(&store, "Qwen", "Qwen3-0.6B", None);
    write_adapter(&store, "Qwen", "Qwen3-0.6B", "acme", "sql", "v1");

    let (ok, stdout, stderr) = run(&dir, &["info", "Qwen/Qwen3-0.6B"]);
    assert!(ok, "brain models info failed: {stderr}");
    assert!(stdout.contains("tok_embeddings.weight"), "the real base tensor must be listed:\n{stdout}");
    assert!(stdout.contains("F32"), "every tensor line must carry its own dtype:\n{stdout}");
    assert!(stdout.contains("[2, 4]"), "every tensor line must carry its own shape:\n{stdout}");
    assert!(stdout.contains("adapters: acme:sql:v1"), "a pulled adapter for this repo must be named in the header:\n{stdout}");
    assert!(stdout.contains("blocks.0.attn.wq.lora_a"), "the adapter's own tensor must appear in the tree:\n{stdout}");
    let lora_line = stdout.lines().find(|l| l.contains("lora_a")).unwrap();
    assert!(lora_line.contains('+'), "an adapter tensor line must carry the '+' marker: {lora_line:?}");

    std::fs::remove_dir_all(&dir).ok();
}

// -------------------------------------------------- flops/models sharing --

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| v != "0").unwrap_or(false)
}

/// A FULLY real, loadable qwen3 checkpoint - every tensor `Qwen::new_shard`
/// actually needs, generated the same way `QwenConfig::tiny()` + zero-init
/// weights are generated anywhere else in this workspace
/// (`qwen3::init_weights`), not a hand-picked subset. `write_base` above is
/// deliberately lighter (one tensor) because every OTHER test in this file
/// only reads a config/card/tensor header, never builds a real `Model` - this
/// one does (`brain flops --weights` does), so it needs the real thing.
fn write_real_qwen3_base(store: &Store, vendor: &str, repo: &str) -> PathBuf {
    use qwen3::{init_weights, QwenConfig};
    let dir = store.repo_dir(&brain_modelref::ModelRef::new(vendor, repo, None));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("model.brain.safetensors");
    let cfg = QwenConfig::tiny();
    let weights = init_weights(&cfg, 0);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = weights.into_iter().map(|(name, data)| (name, vec![data.len() as u64], data)).collect();
    let card = ModelCard::for_ref(&format!("{vendor}/{repo}"), vendor, repo, None, "qwen3");
    checkpoint::st::save_safetensors(file.to_str().unwrap(), &tensors, &cfg.to_json(), Some(&card)).unwrap();
    file
}

/// The concrete proof `modelcost` is one engine, not two: `brain flops`
/// pricing a REAL on-disk model must write into the SAME cache `brain models
/// list` reads, with no `--reprofile` in between.
#[test]
fn brain_flops_on_a_real_checkpoint_populates_the_cache_brain_models_list_reads() {
    if skip_gpu() {
        return;
    }
    let dir = scratch("flops-populates-cache");
    let cache = scratch("flops-populates-cache-cache");
    let store = Store::new(dir.clone());
    let path = write_real_qwen3_base(&store, "Qwen", "Qwen3-0.6B");

    let flops_out = Command::new(bin())
        .args(["flops", "--model", "qwen", "--weights", path.to_str().unwrap(), "--batch", "1"])
        .env("BRAIN_PIPELINE_CACHE_DIR", &cache)
        .output()
        .expect("run brain flops");
    assert!(flops_out.status.success(), "brain flops --weights failed: {}", String::from_utf8_lossy(&flops_out.stderr));

    let list_out = Command::new(bin())
        .args(["models", "list", "--arch", "qwen3", "--plain"])
        .env("BRAIN_MODELS_DIR", &dir)
        .env("BRAIN_PIPELINE_CACHE_DIR", &cache)
        .output()
        .expect("run brain models list");
    assert!(list_out.status.success());
    let stdout = String::from_utf8_lossy(&list_out.stdout);
    let base_line = stdout.lines().find(|l| l.contains("qwen3 Qwen/Qwen3-0.6B ")).expect("the base leaf row must be present");
    assert!(base_line.contains("exact"), "brain flops just priced this exact model - brain models list must read that back without --reprofile: {base_line:?}");

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&cache).ok();
}

/// `brain models profile --measure` on a real checkpoint: builds it, runs
/// it, times it - a genuinely different code path than the dry `price_*`
/// tier tested above, so covered separately.
#[test]
fn models_profile_measure_reports_real_positive_timings_for_a_real_checkpoint() {
    if skip_gpu() {
        return;
    }
    let dir = scratch("measure");
    let store = Store::new(dir.clone());
    write_real_qwen3_base(&store, "Qwen", "Qwen3-0.6B");

    let (ok, stdout, stderr) = run(&dir, &["profile", "Qwen/Qwen3-0.6B", "--measure", "--reps", "2"]);
    assert!(ok, "brain models profile --measure failed: {stderr}");
    assert!(stdout.contains("load:"), "must report load time: {stdout:?}");
    assert!(stdout.contains("cold:"), "must report the cold (first) pass separately: {stdout:?}");
    assert!(stdout.contains("hot:"), "must report the hot (steady-state) pass separately: {stdout:?}");
    assert!(stdout.contains("per layer:"), "must report a per-layer breakdown: {stdout:?}");

    std::fs::remove_dir_all(&dir).ok();
}

/// A REAL, fully-KV'd qwen3 GGUF fixture - not `write_base`'s minimal one
/// (which only sets `general.architecture` and one tensor, deliberately too
/// thin for `qwen3::gguf_import::config_from_gguf` to succeed against, since
/// that codepath is not what `write_base`'s callers exercise). This reuses
/// `qwen3::gguf_import::testing::write_synthetic_gguf` - the exact fixture
/// builder `crate::gguf_import`'s own tests already trust to round-trip
/// through the real GGUF importer - rather than a second, drift-prone copy of
/// its KV/tensor set.
fn write_real_qwen3_gguf(store: &Store, vendor: &str, repo: &str, quant: &str) -> PathBuf {
    let dir = store.repo_dir(&brain_modelref::ModelRef::new(vendor, repo, None));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{quant}.gguf"));
    qwen3::gguf_import::testing::write_synthetic_gguf(file.to_str().unwrap(), false);
    file
}

/// The regression test for the bug this fix addresses: `read_config` used to
/// hand a GGUF's raw llama.cpp-keyed KV map straight to `modelcost` under
/// EVERY format, including GGUF - which uses completely different key names
/// (`qwen3.block_count` vs. brain's own `n_layers`, ...) than brain's
/// canonical schema, so `QwenConfig::from_json_checked` correctly refused it
/// as missing every shape key. `brain models profile`/`list` on a REAL,
/// pulled GGUF quant must price it, not error - this is the exact case
/// `brain models profile Qwen/Qwen3-0.6B-Q8_0` failed on before this fix.
#[test]
fn models_profile_and_list_read_a_real_gguf_config_correctly() {
    if skip_gpu() {
        return;
    }
    let dir = scratch("real-gguf-config");
    let store = Store::new(dir.clone());
    write_real_qwen3_gguf(&store, "Qwen", "Qwen3-0.6B", "Q8_0");

    let (ok, stdout, stderr) = run(&dir, &["profile", "Qwen/Qwen3-0.6B-Q8_0"]);
    assert!(ok, "brain models profile on a real GGUF must succeed: stdout={stdout:?} stderr={stderr:?}");
    assert!(stdout.contains("exact"), "a real GGUF checkpoint must be priced at the exact tier, not a fabricated default: {stdout:?}");

    let (list_ok, list_stdout, list_stderr) = run(&dir, &["list", "--arch", "qwen3", "--plain"]);
    assert!(list_ok, "brain models list failed: {list_stderr}");
    let q8_line = list_stdout.lines().find(|l| l.contains("Qwen/Qwen3-0.6B-Q8_0")).expect("the Q8_0 GGUF leaf row must be present");
    assert!(q8_line.contains("exact"), "brain models profile just cached an exact price for this GGUF - list must read it back, not say not profiled: {q8_line:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn info_on_a_ref_that_is_not_local_fails_and_says_to_pull_it() {
    let dir = scratch("info-missing");
    let (ok, _stdout, stderr) = run(&dir, &["info", "Qwen/Qwen3-0.6B"]);
    assert!(!ok, "info on an unpulled ref must not report success");
    assert!(stderr.contains("brain pull"), "the error must say how to fix it: {stderr:?}");
    std::fs::remove_dir_all(&dir).ok();
}
