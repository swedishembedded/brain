// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixture-free structural check, following `crates/diamond/tests/
//! graph_snapshot.rs`'s precedent: a tiny `ModernBert` with random weights
//! must run without panicking and produce finite, non-all-zero output over a
//! multi-span packed batch. Unlike `graph_snapshot.rs`, this does not pin
//! exact numbers - `tests/parity.rs` is the real numeric gate, and it SKIPS
//! without a fixture; this is what a bare `cargo test -p brain-modernbert`
//! proves even before that fixture is generated.
//!
//! Ragged multi-span, past `2*window`, so both attention rungs
//! (`chunked_bidir_fwd`/`chunked_bidir_fwd_win`) and the no-attn_norm layer 0
//! path all actually run.

use modernbert::config::ModernBertConfig;
use modernbert::kern::PIPELINES;
use modernbert::model::ModernBert;

/// `tiny()` alternates Full/Local every other layer with `window: 3`; spans
/// of 12 and 9 rows are both well past `2*window = 6`, so the local layers'
/// window genuinely cuts off part of each span rather than covering it
/// vacuously.
const SPANS: &[(u32, u32)] = &[(0, 12), (12, 9)];

#[test]
fn a_tiny_random_weight_model_runs_and_produces_finite_nonzero_output() {
    let cfg = ModernBertConfig::tiny();
    let init = modernbert::init::init_weights(&cfg, 11);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().unwrap();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut m = ModernBert::new_on(gpu, cfg.clone(), rows, max_span, &init);

    let ids: Vec<u32> = (0..rows).map(|i| i % cfg.vocab).collect();
    m.set_batch(&ids, SPANS);
    m.forward();

    let hidden = m.hidden();
    assert_eq!(hidden.len(), (rows * cfg.d_model) as usize);
    assert!(hidden.iter().all(|v| v.is_finite()), "forward produced a non-finite value");
    assert!(hidden.iter().any(|&v| v != 0.0), "forward produced an all-zero output");

    let pooled = m.pooled_mean();
    assert_eq!(pooled.len(), SPANS.len() * cfg.d_model as usize);
    assert!(pooled.iter().all(|v| v.is_finite()), "pooled mean produced a non-finite value");
    assert!(pooled.iter().any(|&v| v != 0.0), "pooled mean produced an all-zero output");
}

/// Re-`set_batch`ing at the SAME shape must not panic and must keep producing
/// finite output - the packed-span rebuild path exercised a second time.
#[test]
fn a_second_batch_at_the_same_shape_runs_cleanly() {
    let cfg = ModernBertConfig::tiny();
    let init = modernbert::init::init_weights(&cfg, 12);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().unwrap();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut m = ModernBert::new_on(gpu, cfg.clone(), rows, max_span, &init);

    let ids: Vec<u32> = (0..rows).map(|i| i % cfg.vocab).collect();
    m.set_batch(&ids, SPANS);
    m.forward();
    let first = m.hidden();

    let ids2: Vec<u32> = (0..rows).map(|i| (i + 3) % cfg.vocab).collect();
    m.set_batch(&ids2, SPANS);
    m.forward();
    let second = m.hidden();

    assert!(second.iter().all(|v| v.is_finite()));
    assert_ne!(first, second, "different token ids must produce a different forward");
}

/// A packed batch whose spans straddle the local band (`tiny()`: window 3, so
/// the band is 7 wide): the 12-row span takes the banded rung and the 5-row
/// span the dense windowed one. Each span's hidden states must equal the same
/// span run alone, so neither rung leaks into the other's rows and the split
/// covers every span exactly once.
#[test]
fn spans_longer_and_shorter_than_the_local_band_do_not_disturb_each_other() {
    let cfg = ModernBertConfig::tiny();
    assert_eq!(cfg.window, 3, "the spans below are chosen against a band of 2*3+1");
    let init = modernbert::init::init_weights(&cfg, 13);
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let ids: Vec<u32> = (0..17u32).map(|i| (i * 5 + 1) % cfg.vocab).collect();

    let mixed: &[(u32, u32)] = &[(0, 12), (12, 5)];
    let mut both = ModernBert::new_on(gpu.share(), cfg.clone(), 17, 12, &init);
    both.set_batch(&ids, mixed);
    both.forward();
    let together = both.hidden();

    let d = cfg.d_model as usize;
    for &(row0, len) in mixed {
        let (row0, len) = (row0 as usize, len as usize);
        let mut alone = ModernBert::new_on(gpu.share(), cfg.clone(), len as u32, len as u32, &init);
        alone.set_batch(&ids[row0..row0 + len], &[(0, len as u32)]);
        alone.forward();
        let solo = alone.hidden();
        let worst = solo.iter().zip(&together[row0 * d..(row0 + len) * d]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "span of {len} rows differs by {worst:e} between a packed batch and a run on its own");
    }
}
