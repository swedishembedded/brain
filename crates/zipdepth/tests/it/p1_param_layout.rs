// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1 LAYOUT GATE: `ZipConfig::param_list` must reproduce the released
//! checkpoint's `state_dict` exactly — every key, every shape.
//!
//! This is the cheapest high-value gate in the whole depth workstream, and it is
//! deliberately the FIRST thing built. It needs no GPU, no forward pass and no
//! kernels: it is pure structure. Getting it green means the config, the channel
//! derivations, `_pick_groups`, the BN placement, the bias-vs-no-bias decisions
//! and the module naming are all right BEFORE any arithmetic exists to debug on
//! top of them. A shape bug found here is a one-line fix; the same bug found via
//! a wrong depth map is a day.
//!
//! The counts are asserted unconditionally in `config.rs`'s unit tests (235
//! tensors / 6,802,884 elements for base). Counts can coincide, so this diffs the
//! actual names and shapes against a real file when one is available.
//!
//! ```text
//! ZIPDEPTH_PTH=.../checkpoints/zipdepth_base.pth \
//!   cargo test -p brain-depth --test p1_param_layout -- --nocapture
//! ```
//! Skips OK when unset, per the `YOLO_PARITY_WEIGHTS` convention.

use std::collections::BTreeMap;

use zipdepth::ZipConfig;

/// int64 BatchNorm step counters: bookkeeping no path here reads (BN is folded
/// into the conv for export), so `param_list` omits them by design.
fn is_counter(name: &str) -> bool {
    name.ends_with("num_batches_tracked")
}

fn check(path: &str, cfg: ZipConfig) {
    let report = checkpoint::torchpt::read_report(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));

    let mut actual: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut counters = 0usize;
    for t in &report.tensors {
        if is_counter(&t.name) {
            counters += 1;
            continue;
        }
        actual.insert(t.name.clone(), t.shape.clone());
    }
    let expected: BTreeMap<String, Vec<usize>> = cfg.param_list().into_iter().collect();

    let missing: Vec<&String> = expected.keys().filter(|k| !actual.contains_key(*k)).collect();
    let extra: Vec<&String> = actual.keys().filter(|k| !expected.contains_key(*k)).collect();
    let mismatched: Vec<String> = expected
        .iter()
        .filter_map(|(k, want)| {
            actual.get(k).filter(|got| *got != want).map(|got| format!("  {k}: want {want:?}, got {got:?}"))
        })
        .collect();

    println!(
        "{path}: {} float tensors + {counters} num_batches_tracked; \
         param_list has {} — missing {}, extra {}, shape-mismatched {}",
        actual.len(),
        expected.len(),
        missing.len(),
        extra.len(),
        mismatched.len()
    );
    assert!(missing.is_empty(), "param_list declares keys the checkpoint lacks:\n  {missing:#?}");
    assert!(extra.is_empty(), "checkpoint has keys param_list never declares:\n  {extra:#?}");
    assert!(mismatched.is_empty(), "shape mismatches:\n{}", mismatched.join("\n"));

    // Every BN in the layout must have exactly one counter in the file — i.e. the
    // omission is complete and deliberate, not an accidental undercount.
    let bns = expected.keys().filter(|k| k.ends_with(".running_var")).count();
    assert_eq!(counters, bns, "num_batches_tracked count != BatchNorm count");

    // ...and the BUILT graph must reproduce the SAME set. The config is written
    // from the reference source; the graph is emitted block-by-block from the
    // actual modules. Checking both against the file closes the gap where the two
    // agree with each other but neither matches reality.
    let gpu = gpu_core::Gpu::new_cpu(zipdepth::net::PIPELINES);
    let built = zipdepth::ZipDepth::new(&gpu, cfg, 1, true).param_list();
    let built_names: BTreeMap<String, usize> = built.into_iter().collect();
    let file_numel: BTreeMap<String, usize> = actual.iter().map(|(k, s)| (k.clone(), s.iter().product())).collect();
    let g_missing: Vec<&String> = file_numel.keys().filter(|k| !built_names.contains_key(*k)).collect();
    let g_extra: Vec<&String> = built_names.keys().filter(|k| !file_numel.contains_key(*k)).collect();
    assert!(g_missing.is_empty(), "the built ZipDepth graph is missing checkpoint keys:\n  {g_missing:#?}");
    assert!(g_extra.is_empty(), "the built ZipDepth graph has keys the checkpoint lacks:\n  {g_extra:#?}");
    let g_bad: Vec<String> = file_numel
        .iter()
        .filter_map(|(k, want)| built_names.get(k).filter(|g| *g != want).map(|g| format!("  {k}: file {want}, graph {g}")))
        .collect();
    assert!(g_bad.is_empty(), "the built graph's element counts diverge from the file:\n{}", g_bad.join("\n"));
    println!("{path}: built ZipDepth graph matches the file exactly ({} tensors)", built_names.len());
}

#[test]
fn base_param_list_matches_the_released_checkpoint() {
    let Ok(path) = std::env::var("ZIPDEPTH_PTH") else {
        brain_testutil::skip("set ZIPDEPTH_PTH=<zipdepth_base.pth> to run");
        return;
    };
    check(&path, ZipConfig::base());
}

#[test]
fn npu_param_list_matches_the_released_npu_checkpoint() {
    let Ok(path) = std::env::var("ZIPDEPTH_NPU_PTH") else {
        brain_testutil::skip("set ZIPDEPTH_NPU_PTH=<zipdepth_base_npu.pth> to run");
        return;
    };
    check(&path, ZipConfig { upsample_unfold: false, ..ZipConfig::base() });
}
