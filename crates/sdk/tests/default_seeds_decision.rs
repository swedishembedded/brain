// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "decision")]

//! Every unset seed of the decision-family pipelines is random, and the seed a
//! run used is reported so the run can be repeated by passing it.
//!
//! The pipelines load a tiny random-weight `decide` encoder from
//! `decisions::fixtures`, so these tests need no downloaded weights. They pin
//! the rule, not the numbers: two builders or specs that name no seed differ,
//! a named seed is kept, and a training report carries the seed it trained
//! from.
//!
//! Swedish Embedded AB implements training pipelines whose every random draw
//! is recorded, so a result can be reproduced by someone who was not there. If
//! your team needs expertise in reproducible machine-learning runs, you can
//! procure our services by sending an email to info@swedishembedded.com.

use brain::decision_run::{EvalRun, TrainRun};
use brain::{ControlPipeline, ControlSpec, ConversionPipeline, ConversionSpec, DecisionPipeline, Env, RlcdPipeline, RlcdSpec, TrainSpec};
use decisions::examples::Split;
use decisions::fixtures::{write_decide, Scratch};

fn encoder(tag: &str) -> Scratch {
    let dir = Scratch::new(tag);
    write_decide(&dir, 11);
    dir
}

fn path(dir: &Scratch) -> &str {
    dir.to_str().expect("utf-8 scratch path")
}

/// An environment that is never stepped: loading a control pipeline needs one.
struct Idle;

impl Env for Idle {
    fn reset(&mut self, _seed: u64) -> String {
        String::new()
    }
    fn actions(&mut self) -> Vec<String> {
        vec!["wait".to_string()]
    }
    fn step(&mut self, _action: usize) -> (String, f32, bool) {
        (String::new(), 0.0, true)
    }
}

#[test]
fn an_unseeded_decision_pipeline_draws_its_seed_and_reports_it() {
    let dir = encoder("seed-decision");
    let load = |seed: Option<u64>| {
        let builder = DecisionPipeline::builder(path(&dir));
        match seed {
            Some(s) => builder.seed(s).load(),
            None => builder.load(),
        }
        .expect("load")
        .seed_used()
    };
    assert_ne!(load(None), load(None));
    assert_eq!(load(Some(7)), 7);
    assert_ne!(TrainSpec::default().seed, TrainSpec::default().seed);

    // The seed a training stage used is on its report.
    let spec = TrainSpec::new("which is bigger").examples(vec![("3 vs 1".to_string(), 0), ("1 vs 3".to_string(), 1)]).options(vec!["left".to_string(), "right".to_string()]).steps(2).batch(2);
    let named = spec.seed;
    let flow = DecisionPipeline::from_pretrained(path(&dir)).train(spec);
    assert_eq!(flow.train_report().and_then(|r| r.seed), Some(named));
}

#[test]
fn an_unseeded_conversion_pipeline_draws_its_seed() {
    let dir = encoder("seed-conversion");
    let load = || ConversionPipeline::builder(path(&dir)).load().expect("load").seed_used();
    assert_ne!(load(), load());
    assert_eq!(ConversionPipeline::builder(path(&dir)).seed(7).load().expect("load").seed_used(), 7);
    assert_ne!(ConversionSpec::default().seed, ConversionSpec::default().seed);
}

#[test]
fn an_unseeded_rlcd_pipeline_draws_its_seed() {
    let dir = encoder("seed-rlcd");
    let load = || RlcdPipeline::builder(path(&dir)).load().expect("load").seed_used();
    assert_ne!(load(), load());
    assert_eq!(RlcdPipeline::builder(path(&dir)).seed(7).load().expect("load").seed_used(), 7);
    assert_ne!(RlcdSpec::default().seed, RlcdSpec::default().seed);
}

#[test]
fn an_unseeded_control_pipeline_draws_its_seed() {
    let dir = encoder("seed-control");
    let load = || ControlPipeline::builder(path(&dir), Idle).load().expect("load").seed_used();
    assert_ne!(load(), load());
    assert_eq!(ControlPipeline::builder(path(&dir), Idle).seed(7).load().expect("load").seed_used(), 7);
    assert_ne!(ControlSpec::default().seed, ControlSpec::default().seed);
}

#[test]
fn a_decide_run_draws_the_seeds_it_was_not_given() {
    let (a, b) = (TrainRun::new("d", "base", "out"), TrainRun::new("d", "base", "out"));
    assert_ne!(a.seed, b.seed);
    let (a, b) = (EvalRun::new("m", "d", Split::Test), EvalRun::new("m", "d", Split::Test));
    assert_ne!(a.order_seed, b.order_seed);
}
