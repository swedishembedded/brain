// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The trained fixture the optimizer's tests search and prune: a tiny qwen3
//! that has learned a task needing two tokens of context
//! (`x[t+1] = x[t] + x[t-1] mod vocab`), so its linears carry real, unevenly
//! important structure.

use model::{Batch, Model, IGNORE};
use modelopt::held_out_loss;

pub const ROWS: u32 = 8;

/// `ROWS` sequences of the recurrence from random starting pairs; the first
/// target of each row is unpredictable and masked.
pub fn recurrence(rng: &mut data::rng::Rng, vocab: u32, seq: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    for _ in 0..ROWS {
        let mut x = vec![rng.gen_range_inclusive(0, vocab as i64 - 1) as u32, rng.gen_range_inclusive(0, vocab as i64 - 1) as u32];
        while x.len() < seq as usize + 1 {
            let next = (x[x.len() - 1] + x[x.len() - 2]) % vocab;
            x.push(next);
        }
        tokens.extend_from_slice(&x[..seq as usize]);
        targets.push(IGNORE);
        targets.extend_from_slice(&x[2..]);
    }
    (tokens, targets)
}

pub fn batches(rng: &mut data::rng::Rng, count: usize, vocab: u32, seq: u32) -> Vec<(Vec<u32>, Vec<u32>)> {
    (0..count).map(|_| recurrence(rng, vocab, seq)).collect()
}

pub fn as_batches(owned: &[(Vec<u32>, Vec<u32>)]) -> Vec<Batch<'_>> {
    owned.iter().map(|(tokens, targets)| Batch::Lm { tokens, targets }).collect()
}

/// A tiny qwen3 trained on the recurrence until it has learned it: until its
/// loss on training batches is under half the untrained loss, in rounds of
/// 200 steps, at most 3000.
pub fn trained(rng: &mut data::rng::Rng) -> qwen3::Qwen {
    let cfg = qwen3::QwenConfig::tiny_i8();
    let (vocab, seq) = (cfg.vocab, cfg.block_size);
    let m = qwen3::Qwen::new(cfg, ROWS, seq, &qwen3::init_weights(&qwen3::QwenConfig::tiny_i8(), rng.next_u64()));
    let train = batches(rng, 64, vocab, seq);
    let first = held_out_loss(&m, &as_batches(&train[..4]));
    let mut step = 0u32;
    let mut learned = first;
    while learned >= 0.5 * first && step < 3000 {
        for _ in 0..200 {
            let (tokens, targets) = &train[step as usize % train.len()];
            Model::zero_grads(&m);
            Model::set_batch(&m, Batch::Lm { tokens, targets });
            Model::forward(&m);
            Model::backward(&m);
            step += 1;
            Model::adamw_step(&m, step, 1e-2, 0.0, Default::default(), Some(1.0), 1.0);
        }
        learned = held_out_loss(&m, &as_batches(&train[..4]));
    }
    assert!(learned < 0.5 * first, "the fixture must learn the task before it is compressed: loss {first} -> {learned}");
    m
}

