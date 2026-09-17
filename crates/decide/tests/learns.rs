// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End to end: does the whole stack actually LEARN a decision?
//!
//! Parity proves the encoder reproduces its reference and the gradient check
//! proves the adjoint is the adjoint. Neither proves the parts are wired to
//! each other: an encoder whose output never reaches the head, a head whose
//! gradient never reaches the encoder, or an optimizer stepping the wrong
//! store all pass both of those and train to nothing.
//!
//! So this trains on real BANKING77 utterances with real MiniLM weights and
//! asserts the model ends up better than it started, on examples it was
//! trained on and on ones it was not.
//!
//! Deliberately small - a handful of intents and a few hundred steps - because
//! what is being gated here is that learning HAPPENS, not how well. The
//! accuracy this produces is not a benchmark number and must not be quoted as
//! one.

use std::path::Path;

use data::rng::Rng;
use data::wordpiece::WordPiece;
use decide::banking77::{Banking77, OptionSampler};
use decide::decide::{Decide, Example, Limits};
use decide::kern::PIPELINES;
use decide::loss::LossConfig;
use decide::primitives::{Answer, Opt, Question};

/// Intents the run trains on. Few enough to be quick, enough that chance is
/// clearly below what learning should reach.
const N_INTENTS: usize = 8;
const STEPS: usize = 400;
const ENC_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

struct Fixture {
    data: Banking77,
    tok: WordPiece,
    enc_init: std::collections::HashMap<String, Vec<f32>>,
}

fn fixture() -> Option<Fixture> {
    let dir = brain_testutil::testdata_path("decide/banking77");
    if !dir.join("train.csv").exists() {
        brain_testutil::skip(&format!("{} missing - run `make fetch/testdata`", dir.display()));
        return None;
    }
    let tok_path = brain_testutil::testdata_path("decide/tokenizer/tokenizer.json");
    if !tok_path.exists() {
        brain_testutil::skip(&format!("{} missing - run `make fetch/testdata`", tok_path.display()));
        return None;
    }
    let hf = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2")?;
    let weights = Path::new(&hf).join("model.safetensors");
    if !weights.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run `brain pull sentence-transformers/all-MiniLM-L6-v2`",
            weights.display()
        ));
        return None;
    }
    let cfg = decide::import::config_from_hf(
        &std::fs::read_to_string(Path::new(&hf).join("config.json")).expect("config.json"),
    )
    .expect("config");
    let tensors = checkpoint::safetensors::read(weights.to_str().expect("path")).expect("weights");
    let enc_init = decide::import::brain_init_from_hf(tensors, &cfg).expect("import");
    Some(Fixture {
        data: Banking77::load(&dir).expect("banking77"),
        tok: WordPiece::from_file(tok_path.to_str().expect("path")).expect("tokenizer"),
        enc_init,
    })
}

/// A choice over the given intents, in the given order.
fn question(data: &Banking77, options: &[usize]) -> Question {
    Question::Choice {
        instructions: "which banking intent does this message express".into(),
        options: options.iter().map(|&l| Opt::new(data.option_text(l))).collect(),
    }
}

/// Accuracy over `rows`, each scored against ALL the run's intents at once -
/// the hardest version of the task, and the one that does not depend on which
/// distractors happened to be drawn.
fn accuracy(model: &mut Decide, data: &Banking77, rows: &[&decide::banking77::Row], intents: &[usize]) -> f32 {
    let q = question(data, intents);
    let mut hit = 0usize;
    for r in rows {
        let answers = model.decide(&r.text, std::slice::from_ref(&q)).expect("decide");
        let Answer::Choice { choice, .. } = &answers[0] else { panic!("wrong answer variant") };
        if *choice == data.option_text(r.label) {
            hit += 1;
        }
    }
    hit as f32 / rows.len().max(1) as f32
}

#[test]
fn the_model_learns_a_banking_intent_decision() {
    let Some(fx) = fixture() else { return };
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    let intents: Vec<usize> = (0..N_INTENTS).collect();

    let train: Vec<&decide::banking77::Row> =
        fx.data.train.iter().filter(|r| intents.contains(&r.label)).collect();
    // Up to ten rows PER INTENT, not the first eighty overall: the test split
    // is ordered by intent, so taking a flat prefix would have measured only
    // the first two of the eight and called a two-way decision an eight-way
    // one.
    let mut per_intent = vec![0usize; N_INTENTS];
    let test: Vec<&decide::banking77::Row> = fx
        .data
        .test
        .iter()
        .filter(|r| intents.contains(&r.label))
        .filter(|r| {
            let n = &mut per_intent[r.label];
            *n += 1;
            *n <= 10
        })
        .collect();
    assert!(train.len() > 100, "the intent filter left too little training data");
    assert!(
        per_intent.iter().all(|&n| n >= 10),
        "every intent must be represented in the evaluation set, got {per_intent:?}"
    );

    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let limits = Limits { cap_rows: 1024, cap_slots: 64, max_span: 128, overlap: 16 };
    let mut model = Decide::new_on(
        gpu,
        cfg.clone(),
        fx.tok,
        limits,
        &fx.enc_init,
        &decide::init::init_head(&cfg, 7),
        true,
    );

    let before = accuracy(&mut model, &fx.data, &test, &intents);

    let mut rng = Rng::new(11);
    let sampler = OptionSampler { min: 2, max: N_INTENTS };
    let loss_cfg = LossConfig::cross_entropy();
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for step in 0..STEPS {
        let row = train[(rng.next_u64() % train.len() as u64) as usize];
        let (options, gold) = sampler.draw(row.label, &intents, &mut rng);
        let q = question(&fx.data, &options);
        let l = model
            .train_step(&Example { state: &row.text, question: &q, gold }, &loss_cfg, ENC_LR, HEAD_LR)
            .expect("train step");
        assert!(l.is_finite(), "step {step} produced a non-finite loss {l}");
        if step < 20 {
            first += l / 20.0;
        }
        if step >= STEPS - 20 {
            last += l / 20.0;
        }
    }

    let after = accuracy(&mut model, &fx.data, &test, &intents);
    eprintln!("  loss {first:.4} -> {last:.4}   accuracy {before:.3} -> {after:.3} (chance {:.3})", 1.0 / N_INTENTS as f32);

    assert!(last < first, "loss did not fall: {first:.4} -> {last:.4}");
    // TWICE chance, measured: 400 single-example steps from a fresh head
    // reached 0.287 against a chance of 0.125. The bar is deliberately near
    // what this actually produces rather than near what the task is worth,
    // because what is gated here is that the halves are connected - a model
    // that cannot see its own input sat at 0.0 while its loss fell for all
    // 400 steps, which is the failure this exists to catch.
    //
    // This is NOT a benchmark number and must not be quoted as one. It is 400
    // examples of a batch-of-one over eight intents.
    let chance = 1.0 / N_INTENTS as f32;
    assert!(
        after > 2.0 * chance,
        "accuracy {after:.3} is not clearly above chance {chance:.3} - the halves may not be connected"
    );
    assert!(after > before, "accuracy did not improve: {before:.3} -> {after:.3}");
}
