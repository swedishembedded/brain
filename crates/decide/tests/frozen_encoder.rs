// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Freezing the encoder must actually freeze it, and must not break the head.
//!
//! Both halves matter and they fail differently. An encoder that still moves
//! is a stability problem that shows up as a training run that will not
//! converge; a head that stops learning is a model that trains to nothing. The
//! first is the reason the switch exists, the second is what a careless
//! implementation of it costs.

use std::collections::HashMap;

use decide::decide::{Decide, Limits};
use decide::primitives::{Opt, Question};

fn tiny_model(frozen: bool) -> Option<Decide> {
    let tok_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/decide/tokenizer/tokenizer.json");
    let tok = match data::wordpiece::WordPiece::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            brain_testutil::skip(&format!("{tok_path}: {e} - run scripts/data/fetch-testdata.sh"));
            return None;
        }
    };
    // The real tier, with random weights. `EncoderConfig::tiny` cannot be used
    // here: its vocabulary is 23 tokens, for the gradient checker's synthetic
    // ids, and a real tokenizer emits ids far outside it.
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    let enc: HashMap<String, Vec<f32>> = decide::init::init_weights(&cfg, 1);
    let head: HashMap<String, Vec<f32>> = decide::init::init_head(&cfg, 2);
    let gpu = gpu_core::testgpu::dev(decide::kern::PIPELINES);
    let limits = Limits { cap_rows: 512, cap_slots: 8, max_span: 128, overlap: 16 };
    let mut m = Decide::new_on(gpu, cfg, tok, limits, &enc, &head, true);
    m.set_encoder_frozen(frozen);
    Some(m)
}

fn question() -> Question {
    Question::Choice {
        instructions: "which action".into(),
        options: vec![Opt::new("advance"), Opt::new("retreat")],
    }
}

/// Train a few steps and report how far the encoder and the head moved.
fn drift(frozen: bool) -> Option<(f32, f32)> {
    let mut m = tiny_model(frozen)?;
    let before_enc = m.enc.read_weight("blocks.0.fc1.weight");
    let before_head = m.head.weights();
    let q = question();
    for i in 0..6 {
        let gold = i % 2;
        m.train_step_with("hp 40 ammo 2 | demon range 1", &q, 1e-3, 1e-2, |scores| {
            decide::loss::decision_loss(scores, gold, &decide::loss::LossConfig::cross_entropy())
        })
        .expect("step");
    }
    let after_enc = m.enc.read_weight("blocks.0.fc1.weight");
    let after_head = m.head.weights();
    let l2 = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    };
    let enc_moved = l2(&before_enc, &after_enc);
    let head_moved: f32 =
        before_head.iter().zip(&after_head).map(|((_, a), (_, b))| l2(a, b)).sum();
    Some((enc_moved, head_moved))
}

#[test]
fn a_frozen_encoder_does_not_move_and_the_head_still_learns() {
    let Some((enc_frozen, head_frozen)) = drift(true) else { return };
    let Some((enc_live, head_live)) = drift(false) else { return };

    assert_eq!(enc_frozen, 0.0, "a frozen encoder moved by {enc_frozen}");
    assert!(enc_live > 0.0, "the control case did not move the encoder either ({enc_live}) - the test proves nothing");
    assert!(
        head_frozen > 0.0,
        "freezing the encoder also stopped the head learning (moved {head_frozen})"
    );
    // The head must learn at a comparable rate either way: freezing removes a
    // gradient path the head does not use, not one it depends on.
    let ratio = head_frozen / head_live.max(1e-9);
    assert!(
        (0.5..2.0).contains(&ratio),
        "the head learned very differently when frozen: {head_frozen} vs {head_live}"
    );
}
