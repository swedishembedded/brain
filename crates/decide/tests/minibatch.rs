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

fn model() -> Option<Decide> {
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
    // The configuration a control run uses.
    m.set_encoder_frozen(true);
    Some(m)
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
