// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does an option's own text reach its score at all?
//!
//! The whole model rests on this: if every option of a question gets the same
//! score, the thing still trains (the loss falls, because the shared score
//! moves) and still answers, but the answer is whichever option argmax
//! happens to break the tie toward. That failure is invisible to parity, to
//! the gradient check, and to a falling training loss.

use data::wordpiece::WordPiece;
use decide::config::EncoderConfig;
use decide::decide::{Decide, Limits};
use decide::kern::PIPELINES;
use decide::primitives::{Opt, Question};

fn model(cfg: &EncoderConfig) -> Option<Decide> {
    let tok_path = brain_testutil::testdata_path("decide/tokenizer/tokenizer.json");
    if !tok_path.exists() {
        brain_testutil::skip(&format!("{} missing - run `make fetch/testdata`", tok_path.display()));
        return None;
    }
    let tok = WordPiece::from_file(tok_path.to_str().expect("path")).expect("tokenizer");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let limits = Limits { cap_rows: 512, cap_slots: 32, max_span: 128, overlap: 16 };
    Some(Decide::new_on(
        gpu,
        cfg.clone(),
        tok,
        limits,
        &decide::init::init_weights(cfg, 3),
        &decide::init::init_head(cfg, 7),
        false,
    ))
}

fn question(names: &[&str]) -> Question {
    Question::Choice {
        instructions: "which banking intent does this message express".into(),
        options: names.iter().map(|n| Opt::new(*n)).collect(),
    }
}

const NAMES: [&str; 6] =
    ["card arrival", "exchange rate", "pin blocked", "lost or stolen card", "top up limits", "age limit"];

/// Different options must get DIFFERENT scores. Equal scores mean the option
/// text never reached the scorer.
#[test]
fn an_options_own_text_changes_its_score() {
    let cfg = EncoderConfig::mini_lm_l6();
    let Some(mut m) = model(&cfg) else { return };
    let scores = m.score("I am still waiting on my card", &[question(&NAMES)]).expect("score");
    let s = &scores[0];
    eprintln!("  scores {s:?}");
    let spread = s.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - s.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-4,
        "every option scored within {spread:.3e} of the others, so the option text is not reaching the score: {s:?}"
    );
}

/// Permuting the option list must permute the scores the same way. This is
/// the property that makes the option space genuinely runtime-defined: an
/// option's score may not depend on where in the list it was supplied.
#[test]
fn scores_follow_the_options_when_the_list_is_permuted() {
    let cfg = EncoderConfig::mini_lm_l6();
    let Some(mut m) = model(&cfg) else { return };
    let state = "I am still waiting on my card";
    let forward = m.score(state, &[question(&NAMES)]).expect("score")[0].clone();

    let mut reversed: Vec<&str> = NAMES.to_vec();
    reversed.reverse();
    let back = m.score(state, &[question(&reversed)]).expect("score")[0].clone();

    for (i, &f) in forward.iter().enumerate() {
        let b = back[NAMES.len() - 1 - i];
        assert!(
            (f - b).abs() <= 1e-3,
            "option {:?} scored {f} in one order and {b} in the other - position is leaking into the score",
            NAMES[i]
        );
    }
}

/// Which stage of the head is dead, when one is.
#[test]
fn every_forward_stage_of_the_head_produces_something() {
    let cfg = EncoderConfig::mini_lm_l6();
    let Some(mut m) = model(&cfg) else { return };
    m.score("I am still waiting on my card", &[question(&NAMES)]).expect("score");
    let hid = m.enc.hidden();
    let hn = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!("  encoder hidden rows={} |.| {hn:.6}", hid.len() / cfg.d_model as usize);
    assert!(hn > 0.0, "the encoder produced no hidden states at all");
    let norms = m.head.stage_norms();
    for (name, v) in &norms {
        eprintln!("  {name:8} |.| {v:.6}");
    }
    let dead: Vec<&str> = norms.iter().filter(|(_, v)| *v == 0.0).map(|(n, _)| *n).collect();
    assert!(dead.is_empty(), "these head stages produced all zeros: {dead:?}");
}
