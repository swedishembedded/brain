// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple through the generic fit loop: a run learns a tiny corpus, and a run stopped half way and resumed from its
//! state file ends bit for bit where the uninterrupted run ends.
//!
//! Swedish Embedded AB implements solutions for resuming and verifying long training runs exactly, with the
//! optimizer state wherever the memory is. If your team needs expertise in fault-tolerant fine-tuning of large
//! language models then you can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::{Model, TrainPrecision};
use ricursive::{Ripple, TrainConfig, TrainOpts};

const VOCAB: usize = 32;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-ricursive-fit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A corpus of one repeated sentence whose last token is the only supervised one.
fn write_dataset(dir: &Path) {
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB)).unwrap();
}

fn opts(steps: u32) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: 8,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: steps,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: data::rng::random_seed(),
        precision: TrainPrecision::parse("state=f32").unwrap(),
        ..Default::default()
    }
}

fn objective(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<Ripple> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<Ripple>(train, val, bcfg, itos)
}

fn build(opts: &model::FitOpts) -> Ripple {
    let cfg = TrainConfig::tiny();
    let init = cfg.init_weights(7);
    Ripple::build(cfg, opts.batch_size, opts.block_size, &init, &TrainOpts::default(), true).unwrap()
}

fn weights(m: &Ripple) -> Vec<(String, Vec<f32>)> {
    let mut names = m.param_names();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

/// Spec: AdamW through the fit loop learns the corpus: the loss falls to a fraction of its start.
#[test]
fn a_run_learns_a_tiny_corpus() {
    let dir = scratch("learn");
    write_dataset(&dir);
    let run = opts(40);
    let mut first = None;
    let mut record = |s: &model::StepReport| {
        first.get_or_insert(s.loss);
        true
    };
    let control = model::FitControl { on_step: Some(&mut record), ..Default::default() };
    let (report, _) = model::fit_controlled(build(&run), objective(&dir, &run), &run, None, control).unwrap();
    let (start, end) = (first.unwrap(), report.final_loss.unwrap());
    eprintln!("loss {start:.4} -> {end:.4} over {} steps", run.steps);
    assert!(end < 0.25 * start, "loss {start} -> {end}: a repeated sentence is learned in 40 steps");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: a run stopped at step 4 and resumed in a fresh model ends exactly where the uninterrupted run ends: the
/// last loss and every parameter, bit for bit (the weights and both AdamW moments are in the state file).
#[test]
fn a_resumed_run_ends_exactly_where_the_uninterrupted_run_does() {
    let dir = scratch("resume");
    write_dataset(&dir);
    let identity = serde_json::json!({"test": "ricursive-resume"});
    let run = opts(8);
    let (straight_report, straight) = model::fit_controlled(build(&run), objective(&dir, &run), &run, None, model::FitControl::default()).unwrap();

    let state = dir.join("run.state");
    let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
    let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = model::fit_controlled(build(&run), objective(&dir, &run), &run, None, control).unwrap();
    assert!(first.interrupted && state.is_file(), "stopped at step 4 with a state file");

    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(build(&run), objective(&dir, &run), &run, None, control).unwrap();
    assert_eq!(second.resumed_at, Some(4));
    assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits(), "the last loss is identical");
    for ((name, a), (_, b)) in weights(&straight).iter().zip(weights(&resumed)) {
        assert_eq!(a, &b, "{name}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
