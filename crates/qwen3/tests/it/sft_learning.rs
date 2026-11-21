// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that is shown to learn a rule rather than its examples for its clients.
// If your team needs expertise in validating that a training stack generalises then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Learning effectiveness of supervised fine-tuning, through the shared training loop (`model::fit`) the trainers
//! run, on the CPU backend.
//!
//! * A rule is learned, not its table: a tiny decoder trained from scratch on single-digit additions answers
//!   additions it never saw (held-out pairs), not only the ones it was trained on.
//! * The labels are what it learns: the same run with each answer replaced by an unrelated token answers held-out
//!   pairs no better than chance, so the held-out score above measures the data, not the architecture.
//! * One optimiser step moves every trained tensor against its gradient and lowers the batch's loss; a step at
//!   learning rate zero changes nothing. A step whose gradient never reached the optimiser, or whose rate is zero,
//!   fails the first check (the mutants below).

use std::path::Path;

use model::{Batch, Model};
use qwen3::{Qwen, QwenConfig, IGNORE};

/// A single-digit addition `(a, b)`.
type Pair = (u32, u32);

/// Digits are ids `0..10`; `PLUS`, `EQ`; a sum `s` is the single id `SUM0 + s`.
const PLUS: u32 = 10;
const EQ: u32 = 11;
const SUM0: u32 = 12;
const VOCAB: u32 = SUM0 + 19;
const BLOCK: u32 = 8;

fn config() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: BLOCK, max_position_embeddings: BLOCK, d_model: 32, n_heads: 4, n_kv_heads: 2, head_dim: 8, d_ff: 64, ..QwenConfig::tiny() }
}

/// `a + b = s`, the sum the only supervised token.
fn example(a: u32, b: u32, answer: u32) -> (Vec<u32>, Vec<bool>) {
    (vec![a, PLUS, b, EQ, answer], vec![false, false, false, false, true])
}

/// Every pair `(a, b)`, split into train and held-out by a fixed hash of the pair (about a fifth held out, with no
/// pattern in `a` and `b` for a split to follow). A sum only one or two pairs reach (0, 1, 17, 18) is never held out:
/// an answer no training example shows cannot be learned, by any rule.
fn split() -> (Vec<Pair>, Vec<Pair>) {
    let pairs: Vec<Pair> = (0..10).flat_map(|a| (0..10).map(move |b| (a, b))).collect();
    let held = |a: u32, b: u32| ((a * 10 + b).wrapping_mul(2_654_435_761) >> 7).is_multiple_of(5) && (2..=16).contains(&(a + b));
    pairs.into_iter().partition(|&(a, b)| !held(a, b))
}

fn write_dataset(dir: &Path, train: &[Pair], answer: &dyn Fn(u32, u32) -> u32) {
    let mut split = data::chat::EncodedSplit::default();
    for &(a, b) in train {
        let (ids, mask) = example(a, b, answer(a, b));
        split.push(&ids, &mask);
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &data::chat::EncodedSplit::default()).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(VOCAB as usize)).unwrap();
}

fn opts(steps: u32, lr: f32) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 16,
        block_size: BLOCK,
        lr,
        min_lr: lr / 10.0,
        warmup: steps / 20,
        decay_iters: steps,
        weight_decay: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        checkpoint_secs: 0,
        seed: data::rng::random_seed(),
        ..Default::default()
    }
}

/// The fraction of `pairs` whose greedy answer is `answer(a, b)`.
fn accuracy(m: &Qwen, pairs: &[Pair], answer: &dyn Fn(u32, u32) -> u32) -> f64 {
    let v = VOCAB as usize;
    let right = pairs
        .iter()
        .filter(|&&(a, b)| {
            let logits = m.logits_all(&[a, PLUS, b, EQ]);
            let last = &logits[3 * v..4 * v];
            let best = (0..v).max_by(|&i, &j| last[i].total_cmp(&last[j])).unwrap() as u32;
            best == answer(a, b)
        })
        .count();
    right as f64 / pairs.len() as f64
}

fn true_sum(a: u32, b: u32) -> u32 {
    SUM0 + a + b
}

/// Train from scratch through the shared fit loop on the training pairs labelled by `labels`; the trained model and
/// the split.
fn train(tag: &str, labels: &dyn Fn(u32, u32) -> u32) -> Trained {
    let scratch = brain_testutil::Scratch::dir(&format!("qwen3-sft-learning-{tag}"));
    let (train, held_out) = split();
    write_dataset(scratch.path(), &train, labels);
    let out = scratch.path().join("model.ckpt");
    model::fit::<Qwen>(scratch.path(), config(), &opts(STEPS, 3e-3), Some(&out)).expect("the shared fit loop trains");
    Trained { model: Qwen::load(out.to_str().unwrap(), 1, BLOCK), train, held_out, _scratch: scratch }
}

/// A trained model, the pairs it trained on and the pairs held out; the scratch directory lives as long as it does.
struct Trained {
    model: Qwen,
    train: Vec<Pair>,
    held_out: Vec<Pair>,
    _scratch: brain_testutil::Scratch,
}

/// Steps of both learning runs: about 15 s each on the CPU backend. Measured held-out accuracy 0.81-0.86 over
/// repeated runs (random seeds), chance 0.05.
const STEPS: u32 = 8000;

/// Trained on 79 of the 100 single-digit additions, the model answers most of the 21 it never saw: it learned the
/// rule, not the table. Chance is 1 in 19 sums.
#[test]
fn a_tiny_decoder_learns_addition_and_answers_held_out_pairs() {
    let run = train("clean", &true_sum);
    let (train, held) = (accuracy(&run.model, &run.train, &true_sum), accuracy(&run.model, &run.held_out, &true_sum));
    eprintln!("clean labels: train accuracy {train:.2}, held-out accuracy {held:.2}");
    assert!(train >= 0.95, "the training pairs are learned ({train:.2})");
    assert!(held >= 0.6, "held-out pairs are answered by the rule ({held:.2})");
}

/// The same run with every answer replaced by an unrelated sum (a fixed scramble of the pair) is learned just as
/// well on its training pairs, and transfers nothing: held-out pairs are answered with the true sum at about chance.
/// The held-out score of the clean run therefore measures what its labels teach, not the architecture or the loop.
#[test]
fn corrupted_labels_are_memorised_and_teach_nothing_that_transfers() {
    let scramble = |a: u32, b: u32| SUM0 + (a * 7 + b * 3 + a * b) % 19;
    let run = train("corrupt", &scramble);
    let (memorised, held) = (accuracy(&run.model, &run.train, &scramble), accuracy(&run.model, &run.held_out, &true_sum));
    eprintln!("corrupted labels: train accuracy on its own labels {memorised:.2}, held-out accuracy against the true sum {held:.2}");
    assert!(memorised >= 0.9, "the run trained: its own (corrupted) labels are learned ({memorised:.2})");
    assert!(held <= 0.25, "corrupted labels still produced the rule ({held:.2})");
}

/// A two-row batch of additions, the trainer's layout: position `t` predicts token `t + 1`, only the sum supervised.
fn step_inputs() -> (Vec<u32>, Vec<u32>) {
    let t = BLOCK as usize;
    let (mut x, mut y) = (vec![0u32; 2 * t], vec![IGNORE; 2 * t]);
    for (r, (a, b)) in [(3u32, 4u32), (8, 9)].into_iter().enumerate() {
        let (ids, mask) = example(a, b, SUM0 + a + b);
        for (j, &id) in ids.iter().enumerate() {
            x[r * t + j] = id;
            if j + 1 < ids.len() && mask[j + 1] {
                y[r * t + j] = ids[j + 1];
            }
        }
    }
    (x, y)
}

/// What a step did: for each trained tensor, whether it moved and the inner product of its move with its gradient;
/// and the batch loss before and after.
struct StepOutcome {
    moves: Vec<(String, bool, f64)>,
    before: f32,
    after: f32,
}

/// One AdamW step on the fixed batch. `between` runs after the backward and before the update: the mutants use it to
/// lose the gradient.
fn one_step(lr: f32, between: impl Fn(&Qwen)) -> StepOutcome {
    let cfg = config();
    let m = Qwen::new(cfg.clone(), 2, BLOCK, &qwen3::init_weights(&cfg, 11));
    let (x, y) = step_inputs();
    Model::set_batch(&m, Batch::Lm { tokens: &x, targets: &y });
    m.zero_grads();
    let before = Model::forward(&m);
    Model::backward(&m);
    m.poll_wait();
    let names = Model::optimized_params(&m).expect("a full fine-tune names its trained tensors");
    let start: Vec<(Vec<f32>, Vec<f32>)> = names.iter().map(|n| (m.read_weight(n), m.read_grad(n))).collect();
    between(&m);
    Model::adamw_step(&m, 1, lr, 0.0, model::Adam::default(), None, 1.0);
    m.poll_wait();
    let moves = names
        .iter()
        .zip(&start)
        .map(|(n, (w0, g))| {
            let w1 = m.read_weight(n);
            let moved = w1.iter().zip(w0).any(|(a, b)| a.to_bits() != b.to_bits());
            let along: f64 = w1.iter().zip(w0).zip(g).map(|((a, b), g)| f64::from(a - b) * f64::from(*g)).sum();
            (n.clone(), moved, along)
        })
        .collect();
    let after = Model::forward(&m);
    StepOutcome { moves, before, after }
}

/// Every trained tensor moved against its gradient, and the loss fell; `Err` names the first violation.
fn descends(s: &StepOutcome) -> Result<(), String> {
    if let Some((n, _, _)) = s.moves.iter().find(|(_, moved, _)| !moved) {
        return Err(format!("{n} did not move"));
    }
    if let Some((n, _, along)) = s.moves.iter().find(|(_, _, along)| *along >= 0.0) {
        return Err(format!("{n} moved along its gradient, not against it ({along:e})"));
    }
    if s.after >= s.before {
        return Err(format!("the batch loss did not fall ({} to {})", s.before, s.after));
    }
    Ok(())
}

#[test]
fn one_step_moves_every_trained_tensor_against_its_gradient_and_lowers_the_loss() {
    let s = one_step(1e-3, |_| {});
    descends(&s).unwrap_or_else(|why| panic!("{why}"));
    // Mutant: the gradient lost between the backward and the update.
    let lost = one_step(1e-3, |m| m.zero_grads());
    let why = descends(&lost).expect_err("a step without its gradient must not pass as descent");
    eprintln!("mutant \"gradient lost before the update\" killed: {why}");
    // Mutant: updates disabled (a zero rate).
    let frozen = one_step(0.0, |_| {});
    let why = descends(&frozen).expect_err("a step at rate zero must not pass as descent");
    eprintln!("mutant \"updates disabled\" killed: {why}");
}

/// At learning rate zero a step, weight decay included, changes no weight by a single bit.
#[test]
fn a_zero_learning_rate_changes_no_weight() {
    let cfg = config();
    let init = qwen3::init_weights(&cfg, 11);
    let m = Qwen::new(cfg.clone(), 2, BLOCK, &init);
    let (x, y) = step_inputs();
    for step in 1..=3 {
        Model::set_batch(&m, Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        Model::forward(&m);
        Model::backward(&m);
        Model::adamw_step(&m, step, 0.0, 0.1, model::Adam::default(), Some(1.0), 1.0);
    }
    m.poll_wait();
    for (name, _) in cfg.param_list() {
        let w = m.read_weight(&name);
        assert!(w.iter().zip(&init[&name]).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} changed at learning rate zero");
    }
}
