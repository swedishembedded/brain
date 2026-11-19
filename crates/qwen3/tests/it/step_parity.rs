// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-efficiency work that provably leaves every trained result where it was for
// its clients. If your team needs expertise in optimising training pipelines without regressing them then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tier-1 step parity of the Qwen3 trainer (`brain_testutil::step_parity`): one optimiser step over the eight real
//! A2 records, from one fixed adapter state, through the old path (run twice, for its band) and the new one.
//!
//! **Weighted rows are trimmed.** A weighted-loss batch (the reduction GRPO, policy gradient, KTO and DPO train
//! through) computes only as far as its last supervised position, as an unweighted one does. The old path is the same
//! batch computed over the whole block: a sentinel target at each row's last position, of weight zero, with the step's
//! denominator pinned to the real supervised count, adds nothing to the loss or the gradient and keeps the whole block
//! computed, as every weighted batch was before rows were trimmed under it.

use std::collections::HashMap;

use brain_testutil::step_parity::{self, Observation};
use data::rng::Rng;
use gpu_core::provider::library::GemmPrecision;
use model::Model;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig, IGNORE};

/// The block every record is padded to: past the longest record (705 ids), so every batch has a tail to trim.
pub const BLOCK: u32 = 1024;
/// Rows per micro-batch.
pub const ROWS: usize = 2;

pub fn config(vocab: u32, t: u32) -> QwenConfig {
    let targets = qwen3::finetune::LORA_TARGETS.iter().map(|s| s.to_string()).collect();
    QwenConfig { vocab, block_size: t, max_position_embeddings: t, d_model: 32, d_ff: 64, lora: Some(LoraCfg::new(4, 8.0, targets)), ..QwenConfig::tiny() }
}

/// One record's token ids and supervision mask.
pub type Record = (Vec<u32>, Vec<bool>);

/// The records' `(ids, mask)` on their compact vocabulary, and its size.
pub fn records() -> (Vec<Record>, u32) {
    let encoded = step_parity::a2_records();
    let mut ids: Vec<Vec<u32>> = encoded.iter().map(|r| r.ids.clone()).collect();
    let vocab = step_parity::compact_vocab(&mut ids);
    (ids.into_iter().zip(encoded.iter().map(|r| r.mask())).collect(), vocab)
}

/// The fixed adapter state every path starts from: the base drawn from one seed, `lora_b` drawn non-zero so every
/// adapter tensor has a gradient.
pub fn init(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 3);
    let mut rng = Rng::new(17);
    let mut names: Vec<String> = w.keys().filter(|n| n.ends_with(".lora_b")).cloned().collect();
    names.sort();
    for name in names {
        w.get_mut(&name).unwrap().iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * 0.2);
    }
    w
}

pub fn build(cfg: &QwenConfig, b: u32, init: &HashMap<String, Vec<f32>>, gemm: GemmPrecision) -> Qwen {
    Qwen::new_lora_dt_gemm(cfg.clone(), b, cfg.block_size, init, Dtype::F32, gemm)
}

/// Rows `(ids, mask)` packed at `t`: tokens, and targets at each position whose next token is supervised.
pub fn pack(rows: &[&Record], t: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut x, mut y) = (vec![0u32; rows.len() * t], vec![IGNORE; rows.len() * t]);
    for (r, (ids, mask)) in rows.iter().enumerate() {
        x[r * t..r * t + ids.len()].copy_from_slice(ids);
        for i in 1..ids.len() {
            if mask[i] {
                y[r * t + i - 1] = ids[i];
            }
        }
    }
    (x, y)
}

/// A non-uniform per-position weight, signed by row (an advantage's shape): what a policy-gradient step installs.
fn weight(row: usize, pos: usize) -> f32 {
    let sign = if row.is_multiple_of(2) { 1.0 } else { -0.6 };
    sign * (1.0 + 0.25 * (pos % 3) as f32)
}

/// One optimiser step of weighted-loss training over the records, `ROWS` a micro-batch, from `init`. `untrimmed` adds
/// the weight-zero sentinel that keeps the whole block computed. Returns the observation and the tokens the
/// micro-batches computed.
pub fn weighted_step(untrimmed: bool, gemm: GemmPrecision) -> (Observation, u64) {
    let (recs, vocab) = records();
    let cfg = config(vocab, BLOCK);
    let mut m = build(&cfg, ROWS as u32, &init(&cfg), gemm);
    m.enable_weighted_loss();
    let t = BLOCK as usize;
    let batches: Vec<(Vec<u32>, Vec<u32>)> = recs.chunks(ROWS).map(|rows| pack(&rows.iter().collect::<Vec<_>>(), t)).collect();
    // One denominator for the step: its supervised positions, so the micro-batches accumulate a mean.
    let count = batches.iter().map(|(_, y)| y.iter().filter(|&&v| v != IGNORE).count()).sum::<usize>() as f32;
    let mut obs = Observation::default();
    let mut computed = 0;
    m.zero_grads();
    for (x, y) in &batches {
        let mut y = y.clone();
        let mut w: Vec<f32> = (0..y.len()).map(|i| if y[i] == IGNORE { 0.0 } else { weight(i / t, i % t) }).collect();
        if untrimmed {
            for r in 0..ROWS {
                y[r * t + t - 1] = 1;
                w[r * t + t - 1] = 0.0;
            }
        }
        Model::set_batch(&m, model::Batch::LmWeighted { tokens: x, targets: &y, weights: &w });
        m.set_loss_denominator(count);
        obs.loss += f64::from(m.forward());
        computed += Model::computed_tokens(&m).expect("qwen3 counts its computed tokens");
        let lp = m.batch_token_logprobs();
        let real = |i: usize| y[i] != IGNORE && !(untrimmed && i % t == t - 1);
        obs.logprobs.extend((0..y.len()).filter(|&i| real(i)).map(|i| lp[i]));
        m.backward();
    }
    let names = Model::optimized_params(&m).expect("a LoRA build trains its adapters");
    obs.grads = names.iter().map(|n| (n.clone(), m.read_grad(n))).collect();
    m.adamw_step(1, 1e-3, 0.0, model::Adam::default(), Some(1.0), 1.0);
    obs.weights = names.iter().map(|n| (n.clone(), m.read_weight(n))).collect();
    (obs, computed)
}

/// The binary32 agreement bound of the step's longest reduction: an adapter's weight gradient summed over every
/// position of a micro-batch, across the step's micro-batches.
pub fn floor(positions: usize) -> f64 {
    brain_testutil::training::step_bound(positions)
}

fn weighted_rows_are_trimmed(gemm: GemmPrecision) {
    let (old, whole) = weighted_step(true, gemm);
    let (old_again, _) = weighted_step(true, gemm);
    let (new, trimmed) = weighted_step(false, gemm);
    let positions = ROWS * BLOCK as usize + 4;
    step_parity::assert_parity("weighted rows trimmed", &old, &old_again, &new, floor(positions));
    eprintln!("weighted rows trimmed: computed {trimmed} tokens a step against {whole} untrimmed ({:.2}x fewer)", whole as f64 / trimmed as f64);
    assert_eq!(whole, 4 * ROWS as u64 * u64::from(BLOCK), "the old path computes the whole block");
    assert!(trimmed < whole * 3 / 4, "a weighted batch is trimmed to its records: {trimmed} of {whole} tokens");
}

#[test]
fn a_weighted_batch_trains_trimmed_as_over_the_whole_block() {
    weighted_rows_are_trimmed(GemmPrecision::Fp32Reference);
}

#[test]
#[ignore = "on a card: run with BRAIN_DEVICE=cuda (or another accelerator), the device's own GEMM selection"]
fn a_weighted_batch_trains_trimmed_as_over_the_whole_block_on_a_card() {
    weighted_rows_are_trimmed(GemmPrecision::Auto);
}
