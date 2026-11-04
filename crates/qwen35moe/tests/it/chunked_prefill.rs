// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunked prefill must leave a sequence in exactly the state a token-by-token
//! replay leaves it in, whatever the round sizes - including rounds that start
//! mid-sequence (a recurrent state and a KV prefix that earlier rounds built)
//! and rounds mixed with single-token decode steps - and a pooled round (its
//! per-layer scratch recycled through the arena) must agree with an unpooled
//! one.
//!
//! Swedish Embedded AB implements fast prompt ingestion for large sparse-MoE
//! models. If your team needs expertise in prefill that matches decode to the
//! last bit of state then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Both paths are the same fp32 kernels in a different order, so the bound is
//! rounding, not quantisation. `Qwen35Config::tiny()` has both mixer types and a
//! multi-chunk Gated-DeltaNet sequence, so a round boundary that fails to carry
//! the recurrent state or the conv window shows up here.

use gpu_core::Gpu;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Replay `tokens` one at a time, then decode `extra` more tokens; returns every
/// returned hidden state.
fn replay(m: &Qwen35, tokens: &[u32], extra: &[u32]) -> Vec<Vec<f32>> {
    m.reset_decode_cache();
    tokens.iter().chain(extra).map(|&t| m.step(t)).collect()
}

fn run(gpu: Gpu, round_sets: &[&[usize]], pooled: bool) {
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let init = qwen35moe::init::init_weights(&cfg, 11);
    let m = Qwen35::new_on(gpu, cfg.clone(), 1, t, &init);
    m.set_chunk_arena_min_rows(if pooled { 1 } else { u32::MAX });
    for rounds in round_sets {
        check(&m, &cfg, rounds, pooled);
    }
}

fn check(m: &Qwen35, cfg: &Qwen35Config, rounds: &[usize], pooled: bool) {
    let total: usize = rounds.iter().sum();
    let tokens: Vec<u32> = (0..total as u32).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let extra: Vec<u32> = (0..4u32).map(|i| (i * 3 + 5) % cfg.vocab).collect();
    let want = replay(m, &tokens, &extra);

    // Under test: the same prompt in the given rounds, then the same decode steps.
    m.reset_decode_cache();
    let mut pos = 0usize;
    let mut last = Vec::new();
    for &r in rounds {
        last = m.prefill_chunked(&tokens[pos..pos + r], r as u32);
        pos += r;
    }
    assert_eq!(m.decode_pos() as usize, total);
    let e = rel_l2(&last, &want[total - 1]);
    assert!(e < 1e-4, "rounds {rounds:?} (pooled={pooled}): last prompt hidden differs from the replay's, rel_l2 {e:e}");
    for (i, &tok) in extra.iter().enumerate() {
        let h = m.step(tok);
        let e = rel_l2(&h, &want[total + i]);
        assert!(e < 1e-4, "rounds {rounds:?} (pooled={pooled}): decode step {i} after the prompt differs from the replay's, rel_l2 {e:e}");
    }
}

#[test]
fn chunked_rounds_leave_the_state_a_token_replay_leaves_default_backend() {
    run(gpu_core::testgpu::dev(pipelines()), &[&[16], &[8, 8], &[4, 12], &[4, 4, 8]], false);
}

#[test]
fn a_pooled_round_agrees_with_an_unpooled_one_default_backend() {
    run(gpu_core::testgpu::dev(pipelines()), &[&[16], &[8, 8]], true);
}

#[test]
fn chunked_rounds_leave_the_state_a_token_replay_leaves_cpu() {
    run(Gpu::new_cpu(pipelines()), &[&[8, 8], &[4, 12]], false);
}
