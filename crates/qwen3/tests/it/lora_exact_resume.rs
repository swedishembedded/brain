// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements adapter fine-tuning that can stop and resume without changing its result for its
// clients. If your team needs expertise in resumable training runs then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a LoRA fine-tune stopped half way and resumed from its state file in a fresh model, through the shared fit
//! loop, ends bit for bit where the uninterrupted run ends: the loss, every adapter factor and every factor's AdamW
//! moments (the optimizer state the resume must carry, not only the weights). A continuation that keeps only the
//! weights (moments, schedule position and data order started afresh, as a weight warm start does) is the mutant the
//! comparison must catch.

use std::path::Path;

use model::adapter::device::is_adapter_param;
use model::Model;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

const VOCAB: u32 = 24;
const BLOCK: u32 = 8;
const STEPS: u32 = 8;
const STOP: u32 = 4;

fn cfg() -> QwenConfig {
    let targets = qwen3::finetune::default_lora_targets();
    QwenConfig { vocab: VOCAB, block_size: BLOCK, max_position_embeddings: BLOCK, lora: Some(LoraCfg::new(3, 6.0, targets)), ..QwenConfig::tiny() }
}

fn write_dataset(dir: &Path) {
    let mut split = data::chat::EncodedSplit::default();
    for i in 0..12u32 {
        split.push(&[2 + i % 7, 9, 3 + i % 5, 11, 12 + (i * 5) % 11], &[false, false, false, false, true]);
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &data::chat::EncodedSplit::default()).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(VOCAB as usize)).unwrap();
}

fn opts(steps: u32, seed: u64) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 2,
        block_size: BLOCK,
        lr: 2e-2,
        min_lr: 2e-3,
        warmup: 2,
        decay_iters: steps,
        weight_decay: 0.1,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        checkpoint_secs: 0,
        seed,
        ..Default::default()
    }
}

fn objective(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<Qwen> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<Qwen>(train, val, bcfg, itos)
}

fn build(opts: &model::FitOpts, init: &std::collections::HashMap<String, Vec<f32>>) -> Qwen {
    qwen3::finetune::build_for_training(cfg(), opts, init, Dtype::F32).unwrap()
}

/// Every adapter factor's bits, with its AdamW moments' bits.
fn adapter_state(m: &Qwen) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    let mut names: Vec<String> = m.param_names().into_iter().filter(|n| is_adapter_param(n)).collect();
    names.sort();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    names
        .into_iter()
        .map(|n| {
            let (mm, vv) = m.read_moments(&n).unwrap_or_else(|| panic!("{n}: the LoRA build exposes its AdamW moments"));
            let moments = bits(&mm).into_iter().chain(bits(&vv)).collect();
            (n.clone(), bits(&m.read_weight(&n)), moments)
        })
        .collect()
}

/// The first factor whose weights or moments differ between two runs, or `None`.
fn first_difference(a: &[(String, Vec<u32>, Vec<u32>)], b: &[(String, Vec<u32>, Vec<u32>)]) -> Option<String> {
    a.iter().zip(b).find_map(|((n, wa, ma), (_, wb, mb))| {
        if wa != wb {
            Some(format!("{n}: weights"))
        } else if ma != mb {
            Some(format!("{n}: AdamW moments"))
        } else {
            None
        }
    })
}

#[test]
fn a_resumed_lora_run_ends_with_the_uninterrupted_runs_weights_and_optimizer_state() {
    let scratch = brain_testutil::Scratch::dir("qwen3-lora-exact-resume");
    write_dataset(scratch.path());
    let run = opts(STEPS, data::rng::random_seed());
    let init = qwen3::init_weights(&cfg(), run.seed);
    let identity = serde_json::json!({"test": "lora-exact-resume"});

    let (straight_report, straight) = model::fit_controlled(build(&run, &init), objective(scratch.path(), &run), &run, None, model::FitControl::default()).unwrap();
    let want = adapter_state(&straight);
    assert!(want.iter().all(|(_, _, m)| m.iter().any(|&b| f32::from_bits(b) != 0.0)), "every factor carries optimizer state, so comparing it means something");

    let state = scratch.path().join("run.state");
    let mut stop = |s: &model::StepReport| s.step < STOP;
    let control = model::FitControl { on_step: Some(&mut stop), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, stopped) = model::fit_controlled(build(&run, &init), objective(scratch.path(), &run), &run, None, control).unwrap();
    assert!(first.interrupted && state.is_file(), "stopped at step {STOP} with a state file");
    // The frozen base, with the stopped run's trained factors over it.
    let mut stopped_weights = init.clone();
    stopped_weights.extend(stopped.param_names().into_iter().map(|n| (n.clone(), stopped.read_weight(&n))));

    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(build(&run, &init), objective(scratch.path(), &run), &run, None, control).unwrap();
    assert_eq!(second.resumed_at, Some(STOP));
    assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits(), "the last loss is identical");
    if let Some(what) = first_difference(&want, &adapter_state(&resumed)) {
        panic!("the resumed run differs from the uninterrupted one in {what}");
    }

    // Mutant: continue from the stopped run's weights alone for the remaining steps, the moments, schedule position
    // and data order started afresh.
    let rest = opts(STEPS - STOP, run.seed);
    let (_, warm) = model::fit_controlled(build(&rest, &stopped_weights), objective(scratch.path(), &rest), &rest, None, model::FitControl::default()).unwrap();
    let why = first_difference(&want, &adapter_state(&warm)).expect("a weights-only continuation must not pass as an exact resume");
    eprintln!("mutant \"weights-only continuation\" killed: differs in {why}");
}
