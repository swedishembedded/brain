// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-efficiency work that provably leaves every trained result where it was for
// its clients. If your team needs expertise in optimising feedback fine-tuning without regressing it then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tier-1 step parity of KTO (`brain_testutil::step_parity`): one optimiser step over examples made of the eight real
//! A2 records (each record's prompt and answer, its mismatched row the prompt with the next record's answer), from one
//! fixed adapter state.
//!
//! The old path is the two-row step KTO took before each row became its own pass, written out here: every example
//! packed as one `b = 2` batch (its own row, then the mismatched one), the reference scored by one forward of both,
//! a step forwarding every drawn example once for the reference point and again to train both rows at row 1's weight
//! zero. The new path is [`Feedback`] itself at `b = 1`.

use std::collections::HashMap;

use brain_testutil::step_parity::{self, Observation};
use data::rng::Rng;
use gpu_core::provider::library::GemmPrecision;
use model::{Batch, Model, Objective, IGNORE};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};
use rl::objective::feedback::{feedback_term, reference_point, Feedback, FeedbackConfig, PackedFeedback, FEEDBACK_ROWS};

const BLOCK: usize = 1024;
const STEP: u32 = 4;
const SEED: u64 = 11;
const CFG: FeedbackConfig = FeedbackConfig { beta: 0.5, desirable_weight: 1.0, undesirable_weight: 1.3, seq_len: BLOCK };

fn config(vocab: u32) -> QwenConfig {
    let targets = qwen3::finetune::LORA_TARGETS.iter().map(|s| s.to_string()).collect();
    QwenConfig { vocab, block_size: BLOCK as u32, max_position_embeddings: BLOCK as u32, d_model: 32, d_ff: 64, lora: Some(LoraCfg::new(4, 8.0, targets)), ..QwenConfig::tiny() }
}

/// The base from one seed, `lora_b` drawn non-zero: the start policy is not the base, and every adapter has a gradient.
fn init(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 3);
    let mut rng = Rng::new(17);
    let mut names: Vec<String> = w.keys().filter(|n| n.ends_with(".lora_b")).cloned().collect();
    names.sort();
    for name in names {
        w.get_mut(&name).unwrap().iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * 0.2);
    }
    w
}

/// An example: its own `(ids, mask)`, its mismatched `(ids, mask)`, desirable or not.
type Example = ((Vec<u32>, Vec<bool>), (Vec<u32>, Vec<bool>), bool);

fn examples() -> (Vec<Example>, u32) {
    let records = step_parity::a2_records();
    let mut rows: Vec<Vec<u32>> = records.iter().flat_map(|r| {
        let (prompt, answer) = r.first_answer();
        [prompt, answer]
    }).collect();
    let vocab = step_parity::compact_vocab(&mut rows);
    let turns: Vec<(Vec<u32>, Vec<u32>)> = rows.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect();
    let row = |prompt: &[u32], answer: &[u32]| {
        let ids: Vec<u32> = prompt.iter().chain(answer).copied().collect();
        let mask = (0..ids.len()).map(|i| i >= prompt.len()).collect();
        (ids, mask)
    };
    let n = turns.len();
    let examples = (0..n).map(|i| (row(&turns[i].0, &turns[i].1), row(&turns[i].0, &turns[(i + 1) % n].1), i % 3 != 1)).collect();
    (examples, vocab)
}

fn build(cfg: &QwenConfig, b: u32) -> Qwen {
    Qwen::new_lora_dt_gemm(cfg.clone(), b, BLOCK as u32, &init(cfg), Dtype::F32, GemmPrecision::Fp32Reference)
}

fn drawn(n: usize) -> Vec<usize> {
    let mut rng = Rng::new(SEED);
    (0..STEP).map(|_| rng.gen_range_inclusive(0, n as i64 - 1) as usize).collect()
}

/// Both rows of an example packed as one `b = 2` batch.
fn two_rows(e: &Example) -> (Vec<u32>, Vec<u32>) {
    let (mut x, mut y) = (vec![0u32; 2 * BLOCK], vec![IGNORE; 2 * BLOCK]);
    for (r, (ids, mask)) in [&e.0, &e.1].into_iter().enumerate() {
        x[r * BLOCK..r * BLOCK + ids.len()].copy_from_slice(ids);
        for t in 1..ids.len() {
            if mask[t] {
                y[r * BLOCK + t - 1] = ids[t];
            }
        }
    }
    (x, y)
}

/// The log-probability sum, the reference's sum and the supervised count of row `r`.
fn row_sum(lp: &[f32], reference: &[f32], y: &[u32], r: usize) -> (f32, f32, usize) {
    (r * BLOCK..(r + 1) * BLOCK).filter(|&i| y[i] != IGNORE).fold((0.0, 0.0, 0), |(a, b, c), i| (a + lp[i], b + reference[i], c + 1))
}

fn forward(m: &Qwen, x: &[u32], y: &[u32]) -> Vec<f32> {
    Model::set_batch(m, Batch::Lm { tokens: x, targets: y });
    m.forward();
    m.batch_token_logprobs()
}

fn finish(m: &Qwen, obs: &mut Observation) {
    let names = Model::optimized_params(m).expect("a LoRA build trains its adapters");
    obs.grads = names.iter().map(|n| (n.clone(), m.read_grad(n))).collect();
    m.adamw_step(1, 1e-3, 0.0, model::Adam::default(), Some(1.0), 1.0);
    obs.weights = names.iter().map(|n| (n.clone(), m.read_weight(n))).collect();
}

/// The step KTO took when an example was the two rows of one batch.
fn old_step() -> Observation {
    let (examples, vocab) = examples();
    let cfg = config(vocab);
    let reference = build(&cfg, 2);
    let packed: Vec<(Vec<u32>, Vec<u32>)> = examples.iter().map(two_rows).collect();
    let ref_lp: Vec<Vec<f32>> = packed.iter().map(|(x, y)| forward(&reference, x, y).iter().zip(y).map(|(&lp, &t)| if t == IGNORE { 0.0 } else { lp }).collect()).collect();
    let mut m = build(&cfg, 2);
    m.enable_weighted_loss();
    let mut obs = Observation::default();
    for (e, (_, y)) in packed.iter().enumerate() {
        obs.logprobs.extend((0..BLOCK).filter(|&i| y[i] != IGNORE).map(|i| ref_lp[e][i]));
    }
    let drawn = drawn(examples.len());
    m.zero_grads();
    let mismatched: Vec<f32> = drawn.iter().map(|&e| {
        let (x, y) = &packed[e];
        let (new, old, _) = row_sum(&forward(&m, x, y), &ref_lp[e], y, 1);
        new - old
    }).collect();
    let z = reference_point(&mismatched);
    for &e in &drawn {
        let (x, y) = &packed[e];
        let lp = forward(&m, x, y);
        let (new, old, count_own) = row_sum(&lp, &ref_lp[e], y, 0);
        let (_, _, count_mis) = row_sum(&lp, &ref_lp[e], y, 1);
        let term = feedback_term(&CFG, examples[e].2, new - old, z, (count_own + count_mis) as f32);
        let weights: Vec<f32> = (0..2 * BLOCK).map(|i| if i < BLOCK && y[i] != IGNORE { term.weight } else { 0.0 }).collect();
        m.write_weights(&weights);
        m.backward();
        obs.loss += f64::from(term.loss);
    }
    finish(&m, &mut obs);
    for (e, (x, y)) in packed.iter().enumerate() {
        let (new, old, _) = row_sum(&forward(&m, x, y), &ref_lp[e], y, 0);
        obs.logprobs.push(new - old);
    }
    obs
}

/// The step [`Feedback`] takes.
fn new_step() -> Observation {
    let (examples, vocab) = examples();
    let cfg = config(vocab);
    let packed: Vec<PackedFeedback> = examples.iter().map(|(own, mis, desirable)| PackedFeedback::from_masked(BLOCK, (&own.0, &own.1), (&mis.0, &mis.1), *desirable).unwrap()).collect();
    let mut m = build(&cfg, FEEDBACK_ROWS);
    let mut kto = Feedback::try_from_dataset(CFG, packed).unwrap();
    Objective::<Qwen>::prepare(&mut kto, &mut m).unwrap();
    Objective::<Qwen>::check_data(&mut kto, &m).unwrap();
    let mut obs = Observation::default();
    m.zero_grads();
    let mut rng = Rng::new(SEED);
    obs.loss = f64::from(Objective::<Qwen>::micro_steps(&mut kto, &m, &mut rng, STEP));
    finish(&m, &mut obs);
    obs.logprobs = kto.reference_logprobs();
    obs.logprobs.extend(kto.logratios(&m));
    obs
}

#[test]
fn a_kto_step_of_one_row_passes_trains_as_the_two_row_batches_did() {
    let (old, old_again, new) = (old_step(), old_step(), new_step());
    let floor = brain_testutil::training::step_bound(2 * BLOCK + STEP as usize);
    step_parity::assert_parity("kto one-row passes", &old, &old_again, &new, floor);
}
