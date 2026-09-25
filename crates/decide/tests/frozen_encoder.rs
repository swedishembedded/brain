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

/// Learning from KEPT features is the same computation as learning from the
/// text, and the encoder is the part that is skipped.
///
/// The whole claim of `Decide::score_keeping` is that a frozen encoder's
/// output for a state and its options cannot change, so it may be computed
/// once and reused - which is only worth anything if it is exactly the same
/// number. An approximation here would be a silent one: training would still
/// run, still converge to something, and be wrong.
#[test]
fn kept_features_score_exactly_as_the_text_does() {
    let Some(mut m) = tiny_model(true) else { return };
    let q = question();
    let state = "hp 40 ammo 2 | demon range 1";

    let from_text = m.score(state, std::slice::from_ref(&q)).expect("score")[0].clone();
    let (from_keeping, kept) = m.score_keeping(state, &q).expect("score_keeping");
    assert_eq!(from_text.len(), from_keeping.len());
    for (a, b) in from_text.iter().zip(&from_keeping) {
        assert!((a - b).abs() < 1e-5, "scoring twice disagreed: {a} vs {b}");
    }

    // And the head, run from the kept rows alone, sees the same scores.
    let mut seen = Vec::new();
    m.accumulate_kept(&kept, |sc| {
        seen = sc.to_vec();
        (0.0, vec![0.0; sc.len()])
    })
    .expect("accumulate_kept");
    assert_eq!(seen.len(), from_text.len());
    for (a, b) in from_text.iter().zip(&seen) {
        assert!((a - b).abs() < 1e-4, "the head disagreed with the encoder path: {a} vs {b}");
    }
}

/// And it refuses to run when the encoder is NOT frozen, where the features
/// would be one update stale and the ratio PPO clips would be wrong.
#[test]
fn kept_features_are_refused_on_a_trainable_encoder() {
    let Some(mut m) = tiny_model(true) else { return };
    let q = question();
    let (_, kept) = m.score_keeping("hp 40 ammo 2", &q).expect("score_keeping");
    m.set_encoder_frozen(false);
    assert!(m.accumulate_kept(&kept, |sc| (0.0, vec![0.0; sc.len()])).is_err());
}

/// `Features::from_parts` round-trips through the same three fields
/// [`decide::decide::Features`] keeps privately, and a caller that rebuilds
/// one from a kept `Features`'s own getters must see `accumulate_kept` score
/// it identically - the whole reason `from_parts` exists is to let a caller
/// EDIT specific rows (splicing in rows the encoder never produced) between
/// reading a `Features` apart and handing a new one back.
#[test]
fn features_from_parts_round_trips_and_edited_rows_change_the_score() {
    let Some(mut m) = tiny_model(true) else { return };
    let q = question();
    let state = "hp 40 ammo 2 | demon range 1";
    let (from_text, kept) = m.score_keeping(state, &q).expect("score_keeping");

    // Round-trip through the getters unchanged: same scores.
    let rebuilt = decide::decide::Features::from_parts(kept.hidden().to_vec(), kept.state_rows(), kept.n_slots());
    let rebuilt_scores = m.score_kept(&rebuilt).expect("score_kept on a round-tripped Features");
    assert_eq!(from_text.len(), rebuilt_scores.len());
    for (a, b) in from_text.iter().zip(&rebuilt_scores) {
        assert!((a - b).abs() < 1e-5, "round-tripping Features changed the score: {a} vs {b}");
    }

    // Overwrite state row 0 with noise: the score must move, proving the
    // head's cross-attention actually reads that row rather than some cached
    // copy - the property a spliced-in row (an image patch, say) depends on.
    let h = kept.hidden().len() / (kept.state_rows() + kept.n_slots()) as usize;
    let mut edited = kept.hidden().to_vec();
    for (i, v) in edited[..h].iter_mut().enumerate() {
        *v = if i % 2 == 0 { 5.0 } else { -5.0 };
    }
    let edited = decide::decide::Features::from_parts(edited, kept.state_rows(), kept.n_slots());
    let edited_scores = m.score_kept(&edited).expect("score_kept on an edited Features");
    let moved: f32 = rebuilt_scores.iter().zip(&edited_scores).map(|(a, b)| (a - b).abs()).sum();
    assert!(moved > 1e-3, "editing a state row did not move the score at all: {moved}");
}

/// A head-only checkpoint is refused exactly when the ENCODER MOVED, which is
/// not the same question as which way the freeze switch is pointing now.
///
/// Both directions have bitten. Refusing on the switch alone refused a model
/// that had never been trained at all - an inference build cannot move its
/// encoder, so its head is a perfectly reproducible artifact - and it would
/// equally have ACCEPTED a run that fine-tuned the encoder and then froze it,
/// which is the case the refusal exists for.
#[test]
fn a_head_only_save_is_refused_only_once_the_encoder_has_actually_moved() {
    let Some(mut m) = tiny_model(false) else { return };
    let path = std::env::temp_dir().join(format!("decide-head-{}.safetensors", std::process::id()));
    let path = path.to_str().expect("utf-8 temp path");

    assert!(!m.encoder_was_trained());
    m.save_head(path).expect("an untrained encoder is still the published one");

    // Freezing the encoder keeps that true however much the HEAD learns.
    let ce = decide::loss::LossConfig::cross_entropy();
    let q = question();
    m.set_encoder_frozen(true);
    m.train_step(&decide::Example { state: "hp 40 ammo 2", question: &q, gold: 0 }, &ce, 2e-5, 1e-3)
        .expect("train step");
    assert!(!m.encoder_was_trained());
    m.save_head(path).expect("a frozen encoder is still the published one");

    // One step with it live, and the file can no longer reproduce the model -
    // freezing again afterwards does not launder that.
    m.set_encoder_frozen(false);
    m.train_step(&decide::Example { state: "hp 40 ammo 2", question: &q, gold: 0 }, &ce, 2e-5, 1e-3)
        .expect("train step");
    assert!(m.encoder_was_trained());
    assert!(m.save_head(path).is_err());
    m.set_encoder_frozen(true);
    assert!(m.save_head(path).is_err(), "freezing after the fact does not restore the encoder");

    let _ = std::fs::remove_file(path);
}

/// The gradient `accumulate_kept` leaves in the seed buffer must describe THIS
/// call, not this call plus every call before it.
///
/// `Features::from_parts` exists so a caller can splice in rows the encoder
/// never produced and train whatever produced them on the gradient this pass
/// leaves behind. That caller reads the slot rows as readily as the state
/// rows, and the two are written by different mechanisms: the state path
/// ASSIGNS (`row_scatter`), so a stale row is overwritten and the defect is
/// invisible there, while the `[CLS]` path ACCUMULATES (`emb_bwd`). Without a
/// clear, the second identical step reports twice the first step's slot
/// gradient, the tenth reports ten times, and an external projector trained on
/// it is following a number that grows without bound.
///
/// Running the SAME features through the SAME objective twice is what makes
/// that visible: nothing between the two calls changes any weight, so the two
/// gradients have to be identical.
#[test]
fn the_kept_path_leaves_this_steps_gradient_not_a_running_sum() {
    let Some(mut m) = tiny_model(true) else { return };
    let q = question();
    let (_, kept) = m.score_keeping("hp 40 ammo 2 | demon range 1", &q).expect("score_keeping");

    let h = m.cfg.d_model as usize;
    let rows = (kept.state_rows() + kept.n_slots()) as usize;
    let seed_of = |m: &mut Decide| -> Vec<f32> {
        m.accumulate_kept(&kept, |sc| {
            // A fixed, non-zero score gradient, so the reverse pass has
            // something definite to propagate on both calls.
            (0.0, (0..sc.len()).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect())
        })
        .expect("accumulate_kept");
        m.enc.gpu().read(m.enc.seed_buf(), rows * h)
    };

    let first = seed_of(&mut m);
    let second = seed_of(&mut m);

    let slots = kept.state_rows() as usize * h;
    let magnitude: f32 = first[slots..].iter().map(|v| v.abs()).sum();
    assert!(magnitude > 1e-6, "the slot rows got no gradient at all, so this test proves nothing");

    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        let (row, kind) = if i < slots { (i / h, "state") } else { ((i - slots) / h + kept.state_rows() as usize, "slot") };
        assert!(
            (a - b).abs() <= 1e-5 * (a.abs() + 1.0),
            "{kind} row {row}: repeating an identical step changed its gradient, {a} -> {b}"
        );
    }
}
