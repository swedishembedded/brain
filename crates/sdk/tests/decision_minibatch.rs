// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One optimizer step per BATCH, not per example.
//!
//! `train_choices` used to take one example per step, and a single decision's
//! gradient is a very noisy estimate of the one the objective actually has:
//! the loss wanders instead of descending, and AdamW - whose whole business
//! is a running estimate of the gradient's first two moments - is chasing
//! that noise. `crates/decide` has had the machinery to do better since it
//! was written (`zero_grads` / `accumulate` / `adamw_scaled`, gradient
//! accumulation proved correct by `crates/decide/tests/minibatch.rs`); what
//! was missing was the SDK passing a batch size down to it.
//!
//! These tests pin the WIRING, which is the part that can silently be wrong:
//! the gradient maths is already gated one layer down, but a loop that reads
//! `batch` and then steps the optimizer per example - or accumulates without
//! rescaling - trains, converges, and is not what was asked for.
//!
//! Swedish Embedded AB implements the training loops behind model SDKs -
//! batching, gradient accumulation, and the schedules that make one learning
//! rate mean the same thing at any batch size - for its clients. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use brain::decision::TrainSpec;
use brain::{DecisionPipeline, Stages};

const INSTRUCTIONS: &str = "which banking intent does this message express";

fn options() -> Vec<String> {
    ["card arrival", "exchange rate", "pin blocked", "top up failed"].iter().map(|s| s.to_string()).collect()
}

/// Enough distinct examples that a batch really draws several of them.
fn examples() -> Vec<(String, usize)> {
    let phrases = [
        ("when will my new card get here", 0),
        ("my replacement card has not arrived", 0),
        ("what rate do you use for euros", 1),
        ("how is the exchange rate decided", 1),
        ("my pin is blocked after three tries", 2),
        ("the machine blocked my pin", 2),
        ("my top up did not go through", 3),
        ("topping up my account failed again", 3),
    ];
    phrases.iter().map(|(t, l)| (t.to_string(), *l)).collect()
}

fn pipeline() -> Option<DecisionPipeline> {
    let dir = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2")?;
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!(
            "{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"
        ));
        return None;
    }
    Some(DecisionPipeline::builder(&dir).load().expect("a real MiniLM directory loads through the decide arm"))
}

/// THE property: `steps` is a count of OPTIMIZER UPDATES, and a batch of `b`
/// spends `b` examples on each one.
///
/// Asserted through `steps_taken`, which is AdamW's own time index - the
/// number the bias correction is computed from - so a loop that stepped per
/// example rather than per batch could not pass it.
#[test]
fn a_batch_is_one_optimizer_step_over_several_examples() {
    let Some(mut pipe) = pipeline() else { return };
    let opts = options();
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();

    let before = pipe.steps_taken();
    let steps = 4;
    let batch = 5;
    let mut logged = Vec::new();
    pipe.train_choices(&ex, &opts, INSTRUCTIONS, steps, batch, 0xB_A7C4, &mut |step, l| {
        logged.push((step, l));
    })
    .expect("training runs");

    assert_eq!(
        pipe.steps_taken() - before,
        steps as u32,
        "a batch of {batch} must advance the optimizer once, not {batch} times"
    );
    // One log line per optimizer step, carrying the batch's own mean loss -
    // not one per example, which would make a caller's progress output and
    // its step budget disagree.
    assert_eq!(logged.len(), steps, "expected one log line per step");
    for (i, (step, l)) in logged.iter().enumerate() {
        assert_eq!(*step, i, "log lines must arrive in step order");
        assert!(l.is_finite(), "step {step} logged a non-finite loss {l}");
    }
}

/// A batch of one is still exactly the loop it always was, so the numbers
/// this repo has already published for the `Decide` arm keep meaning what
/// they meant.
#[test]
fn a_batch_of_one_is_the_per_example_loop_it_replaced() {
    let Some(mut pipe) = pipeline() else { return };
    let opts = options();
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();

    let before = pipe.steps_taken();
    let mut n = 0usize;
    pipe.train_choices(&ex, &opts, INSTRUCTIONS, 6, 1, 0x51_1, &mut |_, _| n += 1).expect("training runs");
    assert_eq!(pipe.steps_taken() - before, 6);
    assert_eq!(n, 6);
}

/// A batch size of zero is a caller mistake, and it is named rather than
/// silently rounded up to one - which would make a run report a batch it did
/// not use.
#[test]
fn a_zero_batch_is_refused_by_name() {
    let Some(mut pipe) = pipeline() else { return };
    let opts = options();
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();
    let err = pipe
        .train_choices(&ex, &opts, INSTRUCTIONS, 2, 0, 1, &mut |_, _| {})
        .expect_err("a zero batch must be refused");
    assert!(format!("{err}").contains("batch"), "unhelpful refusal: {err}");
}

/// The level-2 surface carries the same knob, so a caller using `Flow`/
/// `TrainSpec` is not forced down to the slice-and-callback call to batch.
#[test]
fn the_train_spec_carries_the_batch_size() {
    let Some(mut pipe) = pipeline() else { return };
    let before = pipe.steps_taken();
    let spec = TrainSpec::new(INSTRUCTIONS).options(options()).examples(examples()).steps(3).batch(4).seed(7);
    let report = pipe.run_train(&spec, &mut |_, _| {}).expect("run_train");
    assert_eq!(report.steps, 3);
    assert_eq!(pipe.steps_taken() - before, 3, "TrainSpec::batch must reach the optimizer loop");
    assert!(report.final_loss.is_finite());
}

/// The default is the batch a caller who says nothing should get: more than
/// one, because one is the setting that made the loss wander.
#[test]
fn the_default_batch_is_not_one() {
    assert!(
        TrainSpec::new(INSTRUCTIONS).batch > 1,
        "the default batch size is what a caller who does not think about it gets"
    );
}
