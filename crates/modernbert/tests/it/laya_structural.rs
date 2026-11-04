// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixture-free structural check for [`modernbert::laya::LayaHead`],
//! mirroring `tests/structural.rs`'s precedent for the trunk: a tiny head
//! with random weights must run without panicking and produce finite output
//! over a multi-question, variable-arity packed batch. `tests/laya_parity.rs`
//! is the real numeric gate against the reference `DecisionModel`; this is
//! what a bare `cargo test -p brain-modernbert` proves even before that
//! fixture is generated.
//!
//! Built directly on top of a real (random-weight) `ModernBert` trunk, not a
//! synthetic hidden buffer - `LayaHead::set_call` takes the encoder's own
//! `hidden_buf()`, and this test exercises that exact seam.

use modernbert::config::ModernBertConfig;
use modernbert::kern::PIPELINES;
use modernbert::laya::{tensor_manifest, LayaConfig, LayaHead};
use modernbert::model::ModernBert;

/// Two questions, three and two options respectively - different arities so
/// the host-side feature math (`k.clamp(min=2)`, `top2` when only one real
/// option would exist) is exercised on a real (if small) spread. Spans well
/// past `2*window` so the trunk's own local-attention layers are exercised
/// too, same convention as `tests/structural.rs`.
const SPANS: &[(u32, u32)] = &[(0, 12), (12, 9)];
const QTYPE: &[u32] = &[0, 2];
/// Absolute packed rows of each question's option markers - arbitrary rows
/// within each span, not the span's own [CLS] row (row0), which is reserved
/// for the pooled readout.
const MARKER_ROWS: &[u32] = &[1, 4, 7, 13, 16];
const ARITY: &[usize] = &[3, 2];

#[test]
fn a_tiny_random_weight_head_runs_and_produces_finite_output() {
    let cfg = ModernBertConfig::tiny();
    let enc_init = modernbert::init::init_weights(&cfg, 21);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().unwrap();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = ModernBert::new_on(gpu.share(), cfg.clone(), rows, max_span, &enc_init);

    let ids: Vec<u32> = (0..rows).map(|i| i % cfg.vocab).collect();
    enc.set_batch(&ids, SPANS);
    enc.forward();
    enc.gpu.poll_wait();

    let laya_cfg = LayaConfig::new(cfg.d_model);
    assert_eq!(laya_cfg.head_layers, 2, "the released checkpoint's own head_layers");
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, 22);
    let n_markers = MARKER_ROWS.len() as u32;
    let n_questions = SPANS.len() as u32;
    let mut head = LayaHead::new_on(gpu, laya_cfg.clone(), rows, max_span, n_markers, n_questions, &head_init);

    head.set_call(enc.hidden_buf(), SPANS, QTYPE, MARKER_ROWS, ARITY);
    let (logits, act_logits) = head.forward();

    assert_eq!(logits.len(), MARKER_ROWS.len());
    assert!(logits.iter().all(|v| v.is_finite()), "option logits produced a non-finite value: {logits:?}");
    assert!(logits.iter().any(|&v| v != 0.0), "option logits were all zero");

    assert_eq!(act_logits.len(), SPANS.len() * laya_cfg.n_act as usize);
    assert!(act_logits.iter().all(|v| v.is_finite()), "act logits produced a non-finite value: {act_logits:?}");

    // The tensor manifest's own coverage must match what the init map
    // produced - the same discipline `tests/parity.rs` runs against a real
    // fixture, checked here against the structural init instead.
    let expected = tensor_manifest(&laya_cfg);
    let total: usize = expected.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    let got: usize = head_init.values().map(|v| v.len()).sum();
    assert_eq!(got, total, "init map vs tensor_manifest total parameter count");
}

/// A second call at a DIFFERENT shape (fewer questions, fewer markers) must
/// rebuild cleanly and produce a different, still-finite result - the
/// packed-call rebuild path `LayaHead::set_call` documents as unconditional.
#[test]
fn a_second_call_at_a_different_shape_runs_cleanly() {
    let cfg = ModernBertConfig::tiny();
    let enc_init = modernbert::init::init_weights(&cfg, 31);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().unwrap();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = ModernBert::new_on(gpu.share(), cfg.clone(), rows, max_span, &enc_init);
    let ids: Vec<u32> = (0..rows).map(|i| i % cfg.vocab).collect();
    enc.set_batch(&ids, SPANS);
    enc.forward();
    enc.gpu.poll_wait();

    let laya_cfg = LayaConfig::new(cfg.d_model);
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, 32);
    let n_markers = MARKER_ROWS.len() as u32;
    let n_questions = SPANS.len() as u32;
    let mut head = LayaHead::new_on(gpu, laya_cfg, rows, max_span, n_markers, n_questions, &head_init);

    head.set_call(enc.hidden_buf(), SPANS, QTYPE, MARKER_ROWS, ARITY);
    let (first, _) = head.forward();

    // One question only, two options - a strictly smaller call than the one
    // the buffers were sized for.
    let one_span = &SPANS[..1];
    let one_qtype = &QTYPE[..1];
    let one_markers = &MARKER_ROWS[..2];
    let one_arity = &[2usize];
    head.set_call(enc.hidden_buf(), one_span, one_qtype, one_markers, one_arity);
    let (second, act2) = head.forward();

    assert_eq!(second.len(), 2);
    assert!(second.iter().all(|v| v.is_finite()));
    assert_eq!(act2.len(), 2);
    // Markers 1 and 4 sit inside the SAME span both calls process (question
    // 0's own), so this shrunk call reproduces them exactly - the shape
    // change only drops question 1 and marker 7, it does not perturb what
    // question 0's own self-attention computed.
    assert_eq!(&first[..2], &second[..], "the first question's own markers must be call-shape-independent");
}
