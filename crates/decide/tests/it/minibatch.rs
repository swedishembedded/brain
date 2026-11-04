// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Accumulating a minibatch must equal summing its examples' gradients.
//!
//! The reason this needs a test rather than a reading of the code: parameter
//! gradients accumulate ONLY if nothing in the path assigns instead of adding,
//! and that is a property of every kernel in the reverse pass, not of the loop
//! that calls them. If one of them assigns, a minibatch silently becomes "the
//! last example in it" - which trains, converges, and is not what was asked
//! for.

use std::collections::HashMap;

use decide::decide::{Decide, Limits};
use decide::primitives::{Opt, Question};

fn model_with(frozen: bool) -> Option<Decide> {
    let tok_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/decide/tokenizer/tokenizer.json");
    let tok = match data::wordpiece::WordPiece::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            brain_testutil::skip(&format!("{tok_path}: {e} - run scripts/data/fetch-testdata.sh"));
            return None;
        }
    };
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    let enc: HashMap<String, Vec<f32>> = decide::init::init_weights(&cfg, 1);
    let head: HashMap<String, Vec<f32>> = decide::init::init_head(&cfg, 2);
    let gpu = gpu_core::testgpu::dev(decide::kern::PIPELINES);
    let limits = Limits { cap_rows: 512, cap_slots: 8, max_span: 128, overlap: 16 };
    let mut m = Decide::new_on(gpu, cfg, tok, limits, &enc, &head, true);
    m.set_encoder_frozen(frozen);
    Some(m)
}

/// The configuration a control run uses.
fn model() -> Option<Decide> {
    model_with(true)
}

fn q() -> Question {
    Question::Choice {
        instructions: "which action".into(),
        options: vec![Opt::new("advance"), Opt::new("retreat"), Opt::new("reload")],
    }
}

const STATES: [&str; 3] = [
    "hp 90 ammo 6 | imp range 4",
    "hp 40 ammo 0 | demon range 1",
    "hp 70 ammo 3 | imp range 2 | medkit range 1",
];

fn grads(m: &Decide) -> Vec<f32> {
    // One parameter is enough: the accumulation property is per-buffer.
    m.head.read_grad("head.score.weight")
}

#[test]
fn a_minibatch_gradient_is_the_sum_of_its_examples() {
    let Some(mut m) = model() else { return };
    let ce = decide::loss::LossConfig::cross_entropy();
    let question = q();

    // Each example on its own, summed by hand.
    let mut want: Vec<f32> = Vec::new();
    for (i, state) in STATES.iter().enumerate() {
        m.zero_grads();
        m.accumulate(state, &question, |s| decide::loss::decision_loss(s, i % 3, &ce)).expect("step");
        let g = grads(&m);
        if want.is_empty() {
            want = vec![0.0; g.len()];
        }
        for (w, v) in want.iter_mut().zip(&g) {
            *w += v;
        }
    }

    // All three accumulated into one gradient.
    m.zero_grads();
    for (i, state) in STATES.iter().enumerate() {
        m.accumulate(state, &question, |s| decide::loss::decision_loss(s, i % 3, &ce)).expect("step");
    }
    let got = grads(&m);

    assert_eq!(got.len(), want.len());
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(
        worst <= 2e-4 * scale.max(1.0),
        "accumulated gradient differs from the sum of its parts by {worst:.3e} (scale {scale:.3e})"
    );
    // ...and it must not merely equal the LAST example, which is what an
    // assigning kernel anywhere in the path would produce.
    m.zero_grads();
    m.accumulate(STATES[2], &question, |s| decide::loss::decision_loss(s, 2, &ce)).expect("step");
    let last = grads(&m);
    let diff = got.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff > 1e-5 * scale, "the minibatch gradient is indistinguishable from its last example");
}

/// Scaling the accumulated gradient by `1/n` must be what makes one learning
/// rate mean the same thing at any batch size.
#[test]
fn scaling_turns_an_accumulated_sum_into_a_mean() {
    let Some(mut m) = model() else { return };
    let ce = decide::loss::LossConfig::cross_entropy();
    let question = q();
    let before = m.head.weights();

    // One example, full step.
    m.zero_grads();
    m.accumulate(STATES[0], &question, |s| decide::loss::decision_loss(s, 0, &ce)).expect("step");
    m.adamw_scaled(0.0, 1e-3, 1.0);
    let single: f32 = m
        .head
        .weights()
        .iter()
        .zip(&before)
        .map(|((_, a), (_, b))| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>())
        .sum::<f32>()
        .sqrt();

    // The SAME example three times, scaled by 1/3: the mean gradient is
    // identical, so the step should be too.
    let Some(mut m) = model() else { return };
    m.zero_grads();
    for _ in 0..3 {
        m.accumulate(STATES[0], &question, |s| decide::loss::decision_loss(s, 0, &ce)).expect("step");
    }
    m.adamw_scaled(0.0, 1e-3, 1.0 / 3.0);
    let averaged: f32 = m
        .head
        .weights()
        .iter()
        .zip(&before)
        .map(|((_, a), (_, b))| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>())
        .sum::<f32>()
        .sqrt();

    assert!(single > 0.0, "the single-example step did not move the weights");
    let rel = (averaged - single).abs() / single;
    assert!(rel < 0.05, "averaging three copies moved {averaged} where one moved {single}");
}

/// A minibatch over ONE encoder pass must produce the SAME gradient as the
/// same examples accumulated one pass at a time - in BOTH halves.
///
/// This is the whole claim of `Decide::accumulate_batch`, and it is not
/// something a loss curve can show: a shared pass that dropped every example
/// but the last, or that let one example's state leak into another's
/// attention, would still fall and still converge to something. What makes it
/// true is that the examples occupy disjoint rows of one pack and that
/// attention never crosses a span - properties of the packer and of the
/// encoder, not of this loop - so it is checked against the sequential path
/// directly.
///
/// The ENCODER IS LIVE here. Its gradient is the half a shared pass actually
/// changes (the head is called once per example either way), and it is the
/// half that would be wrong if the seed buffer were cleared per example
/// instead of per pass.
#[test]
fn a_batched_pass_is_the_same_gradient_as_separate_passes() {
    let ce = decide::loss::LossConfig::cross_entropy();
    let question = q();
    let batch: Vec<(&str, &Question)> = STATES.iter().map(|s| (*s, &question)).collect();

    // Sequential: one encoder pass per example, accumulating.
    let Some(mut seq) = model_with(false) else { return };
    let mut want_loss = Vec::new();
    seq.zero_grads();
    for (i, state) in STATES.iter().enumerate() {
        let l = seq
            .accumulate(state, &question, |s| decide::loss::decision_loss(s, i % 3, &ce))
            .expect("step");
        want_loss.push(l);
    }

    // Batched: one encoder pass for all of them.
    let Some(mut bat) = model_with(false) else { return };
    bat.zero_grads();
    let got_loss = bat
        .accumulate_batch(&batch, |i, s| decide::loss::decision_loss(s, i % 3, &ce))
        .expect("batch");

    assert_eq!(got_loss.len(), want_loss.len());
    for (i, (a, b)) in got_loss.iter().zip(&want_loss).enumerate() {
        assert!((a - b).abs() < 2e-4, "example {i} scored {a} batched and {b} alone");
    }

    // One parameter from each half. The head's is the path through the
    // gathered state rows; the encoder's is only reachable through the seed
    // buffer every example in the pack wrote into.
    for name in ["head.wkv.weight", "head.score.weight"] {
        close(&bat.head.read_grad(name), &seq.head.read_grad(name), name);
    }
    for name in ["blocks.0.fc1.weight", "blocks.5.qkv.weight", "tok.weight"] {
        close(&bat.enc.read_grad(name), &seq.enc.read_grad(name), name);
    }
}

/// A batched pass is not bit-identical to a sequential one - the GEMMs see a
/// different row count and so a different tiling - so this is a relative
/// bound, the same one the accumulation test uses.
fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(scale > 0.0, "{what} has an all-zero reference gradient - the test would prove nothing");
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(
        worst <= 2e-4 * scale,
        "{what}: batched gradient differs from the sequential one by {worst:.3e} (scale {scale:.3e})"
    );
}

/// ...and one optimizer step over a batch moves the weights the way that
/// gradient says, which is what `Decide::train_batch` adds on top.
#[test]
fn a_batch_step_takes_the_mean_gradient() {
    let Some(mut m) = model_with(true) else { return };
    let ce = decide::loss::LossConfig::cross_entropy();
    let question = q();
    let before = m.head.weights();

    // The SAME example three times is its own mean, so a batch of three must
    // move exactly as far as a batch of one on that example.
    let batch: Vec<(&str, &Question)> = vec![(STATES[0], &question); 3];
    m.train_batch(&batch, &ce, &[0, 0, 0], 0.0, 1e-3).expect("batch step");
    let batched = moved(&before, &m.head.weights());

    let Some(mut one) = model_with(true) else { return };
    one.train_batch(&[(STATES[0], &question)], &ce, &[0], 0.0, 1e-3).expect("batch step");
    let single = moved(&before, &one.head.weights());

    assert!(single > 0.0, "the single-example step did not move the weights");
    let rel = (batched - single).abs() / single;
    assert!(rel < 0.05, "a batch of three copies moved {batched} where one moved {single}");
}

fn moved(before: &[(String, Vec<f32>)], after: &[(String, Vec<f32>)]) -> f32 {
    after
        .iter()
        .zip(before)
        .map(|((_, a), (_, b))| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>())
        .sum::<f32>()
        .sqrt()
}
