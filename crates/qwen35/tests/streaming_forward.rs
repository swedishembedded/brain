// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint gates for `qwen35::stream` - the sliding-window streaming
//! forward pass over all 64 real decoder layers, holding only a small
//! [`weightset::WeightSet`]-scheduled window of layers' weights resident at
//! once (see that module's own doc for the full design).
//!
//! Two distinct gates, not to be conflated:
//!
//! 1. [`layer_0_streamed_matches_the_real_reference`] /
//!    `layer_3_.../layer_63_...` - per-layer NUMERICAL parity: the exact
//!    three layers `tests/real_weight_streaming.rs` already validates in
//!    isolation (one Gated DeltaNet, two GQA - first and last full-attention
//!    layer), using the SAME real weights and the SAME golden `x_in`/`out`
//!    fixtures that file reads, but driven through `stream::StreamState`'s
//!    own `load_layer`/`layer_forward` (the NEW streaming/int8 machinery -
//!    `WeightSet`-shaped loading, drop/rebuild slots, `Weight::upload(...,
//!    Dtype::I8)`) instead of `real_weight_streaming.rs`'s own plain-fp32,
//!    non-streaming replay. A high cosine against the real Python reference
//!    here proves the new plumbing did not silently change the numerical
//!    result versus the already-proven path - modulo the SAME int8
//!    quantization error M14's `int8_real_weight_sanity.rs` already measured
//!    per-leaf (cosine in [0.9999283, 0.9999519] there), now propagated
//!    through one whole layer's chain of 8-12 quantized linears plus the
//!    mixer nonlinearities instead of measured leaf-by-leaf.
//! 2. [`full_chain_streams_all_64_real_layers_within_budget`] - a bounded-
//!    correctness-and-memory SMOKE gate, not a numerical parity gate: there
//!    is no whole-model reference on any machine this workspace has access
//!    to (the approved plan says so explicitly), so this test can only
//!    confirm the streaming loop actually completes across all 64 real
//!    layers, produces finite output, and stays under a measured, justified
//!    peak-RSS budget - never that the output value is "correct" against
//!    anything external.
//!
//! Self-skips loudly (never silently) without `BRAIN_QWEN35_DIR` - same
//! pattern as `real_weight_streaming.rs`. Run with:
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//!     cargo test -p brain-qwen35 --test streaming_forward -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use checkpoint::mmap::MmapSafetensors;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::stream::{run, StreamState};

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

fn golden_dir(l: usize) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/qwen35").join(format!("real_layer_{l}"))
}

/// Current process `VmHWM` (peak resident set), GiB - same `/proc/self/status`
/// source `brain_testutil::mem` prints, read here as a number so the smoke
/// gate can assert a real budget instead of only eyeballing printed output.
fn peak_rss_gib() -> f64 {
    let st = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    st.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .map(|kb| kb / 1_048_576.0)
        .unwrap_or(0.0)
}

/// Gate 1: layer `l`'s streamed/int8 forward against the real Python
/// reference's own `out`, using the same golden `x_in` `real_weight_streaming.
/// rs` reads for the identical layer. `window_budget` is irrelevant to a
/// single-layer call (no eviction ever happens with one layer loaded), so
/// this exercises `load_layer`/`layer_forward` directly - the same two
/// building blocks `stream::run`'s own windowed loop calls on every miss.
fn run_layer_gate(l: usize) {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let shard = dir.join(format!("layers-{l}.safetensors"));
    if !shard.exists() {
        brain_testutil::skip_unavailable(&format!("{} not present under BRAIN_QWEN35_DIR", shard.display()));
        return;
    }
    let golden_path = golden_dir(l).join("layer.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run: python tools/goldens/qwen35_dump_real_layer_reference.py --dir {} --layer {l}",
            golden_path.display(),
            dir.display()
        ));
        return;
    }

    brain_testutil::mem(&format!("layer {l}: before load_layer"));
    let cfg = Qwen35Config::qwen38_27b();
    let d = cfg.d_model;

    let golden = MmapSafetensors::open(&golden_path).unwrap_or_else(|e| panic!("open {}: {e}", golden_path.display()));
    let t = golden.shape("x_in").unwrap()[0] as u32;
    let x_in = golden.tensor_f32("x_in").unwrap();
    let want_out = golden.tensor_f32("out").unwrap();
    assert_eq!(x_in.len(), (t * d) as usize);
    assert_eq!(want_out.len(), (t * d) as usize);

    let state = StreamState::new(Gpu::new(qwen35::model::pipelines()), &cfg, t);
    let layer = state.load_layer(&dir, &cfg, l);
    brain_testutil::mem(&format!("layer {l}: after load_layer"));

    let x = state.gpu.storage_init("streaming_forward_test.x_in", &x_in);
    let out = state.layer_forward(&cfg, &layer, &x, t);
    let got = state.gpu.read(&out, (t * d) as usize);
    brain_testutil::mem(&format!("layer {l}: after layer_forward"));

    assert!(got.iter().all(|v| v.is_finite()), "layer {l}: streamed forward produced a non-finite value");
    let (cos, max_abs) = brain_testutil::parity::compare(&got, &want_out);
    let rel = brain_testutil::parity::rel_l2(&got, &want_out);
    eprintln!("layer {l} (streamed, int8): cosine={cos:.9} rel_l2={rel:.6} max_abs={max_abs:.4}");
    // int8 quantizes every one of this layer's 8-12 mixer/MLP linears (see
    // `is_i8_linear`) - M14's own leaf-level measurement on these same real
    // weights put per-leaf int8-vs-fp32 cosine in [0.9999283, 0.9999519];
    // chained through a whole layer (multiple quantized linears in series
    // plus the mixer's nonlinearities) some further drift is expected, but
    // nowhere near enough to threaten a 0.99 floor - well short of
    // `real_weight_streaming.rs`'s own fp32-vs-reference 0.999 floor, which
    // is the right relationship: streaming introduces no NEW error source of
    // its own (same `import_layer`, same mixer math), only int8's already-
    // measured one.
    assert!(cos > 0.99, "layer {l}: streamed cosine={cos:.9} too low (want > 0.99)");
}

#[test]
#[ignore]
fn layer_0_streamed_matches_the_real_reference() {
    run_layer_gate(0);
}

#[test]
#[ignore]
fn layer_3_streamed_matches_the_real_reference() {
    run_layer_gate(3);
}

#[test]
#[ignore]
fn layer_63_streamed_matches_the_real_reference() {
    run_layer_gate(63);
}

/// Gate 2: the full 64-real-layer streaming chain. NOT a numerical parity
/// gate (see this file's own doc) - no whole-model reference exists to
/// compare against on any machine this workspace has access to. Asserts
/// only: the loop completes across all 64 real layers without panicking,
/// every output value is finite, and peak RSS stays under a budget derived
/// below, not merely asserted.
///
/// Budget derivation: `crate::import::import_layer`'s own measured peak is
/// 2.37-2.45 GiB per layer (`real_weight_streaming.rs`'s doc, M10) - that
/// figure already reflects the FP8-dequantize working set for one shard, and
/// since layers are imported strictly sequentially here (never two `import_
/// layer` calls overlapping), the process-wide high-water mark should track
/// that single-layer peak, not `window_budget` times it. On top of that:
/// this iGPU's "device" buffers are drawn from the SAME shared system RAM
/// (see this crate's own docs on this box's hardware), so the window's own
/// resident layers also count against RSS - at most `window_budget` (4)
/// layers' quantized weights, each well under 400 MB (see `stream`'s own
/// per-leaf size accounting), under 1.6 GiB. Plus baseline process/driver/
/// compiled-kernel overhead (typically a few hundred MB on this stack, per
/// every other real-weight test in this crate). 2.5 + 1.6 + 1.0 headroom
/// rounds to a 6 GiB ceiling - comfortably under this box's ~11 GiB
/// available RAM, and generous enough that a real regression (not just
/// measurement noise) is what would trip it.
#[test]
#[ignore]
fn full_chain_streams_all_64_real_layers_within_budget() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let cfg = Qwen35Config::qwen38_27b();
    for l in 0..cfg.n_layers as usize {
        if !dir.join(format!("layers-{l}.safetensors")).exists() {
            brain_testutil::skip_unavailable(&format!("layers-{l}.safetensors missing under BRAIN_QWEN35_DIR - need all 64 shards for this gate"));
            return;
        }
    }

    brain_testutil::mem("full chain: before run");
    // n=4: a handful of rows (this is a bounded-correctness-and-memory
    // smoke gate, not a throughput benchmark - see this file's own doc), a
    // multiple of every GDN chunk size `model::gdn::gdn_chunk_size` can pick
    // for t=4 (chunk=4, dividing evenly). window_budget=4: comfortably under
    // this box's available RAM per the budget derivation above, matching
    // `stream`'s own doc ("3-4 slots... comfortably safe").
    let out = run(&dir, &cfg, 4, 4, 20260819);
    brain_testutil::mem("full chain: after run");

    assert_eq!(out.len(), (4 * cfg.d_model) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "full chain: streamed forward produced a non-finite value");

    let peak = peak_rss_gib();
    eprintln!("full chain (64 real layers, window_budget=4): peak RSS = {peak:.2} GiB");
    assert!(peak < 6.0, "full chain: peak RSS {peak:.2} GiB exceeds the 6 GiB budget derived in this test's own doc comment");
}
