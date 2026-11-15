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

/// On a device that selects the dense engine's Gated-DeltaNet prefill form (`gpu_core::provider::cuda::
/// gdn_segment_kernels_served`: tensor cores, the bf16 form measured faster, head widths 128, a 4-tap convolution), a
/// prefill round runs the round-wide segment kernels on the sequence's state row - the kernels the dense `qwen35`
/// serving rounds run - and still leaves the state a token replay leaves, across rounds that continue a sequence and
/// the decode steps after them. Measured 3.8e-7 on gpu3 (GH200); with these random weights the mixer is a small share of
/// the hidden state, so a round that leaves half its rows to no kernel moves it by only 1.2e-3, and the bound sits
/// between the two.
#[test]
fn a_round_on_the_segment_kernels_leaves_the_state_a_token_replay_leaves() {
    let gpu = gpu_core::testgpu::dev(pipelines());
    let cfg = Qwen35Config {
        block_size: 256,
        max_position_embeddings: 256,
        n_layers: 4,
        d_model: 64,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        ..Qwen35Config::tiny()
    };
    let init = qwen35moe::init::init_weights(&cfg, 3);
    let m = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &init);
    if !gpu_core::provider::cuda::tensor_core_kernels_enabled(&m.gpu) {
        return brain_testutil::skip_unavailable("no tensor-core kernels on this device");
    }
    // Placing a model with 128-wide recurrent heads measures the prefill form, as the dense model's placement does.
    m.gpu.gdn_prefill_choice().expect("the model was placed without measuring its Gated-DeltaNet prefill form");
    if !gpu_core::provider::cuda::gdn_segment_kernels_served(&m.gpu) {
        return brain_testutil::skip_unavailable("this device measured the exact Gated-DeltaNet prefill form faster than the bf16 one");
    }
    let tokens: Vec<u32> = (0..192u32).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let extra: Vec<u32> = (0..4u32).map(|i| (i * 3 + 5) % cfg.vocab).collect();
    let want = replay(&m, &tokens, &extra);

    m.reset_decode_cache();
    m.gpu.set_kernel_timing(true);
    m.gpu.reset_kernel_times();
    m.prefill_chunked(&tokens[..128], 128);
    let last = m.prefill_chunked(&tokens[128..], 64);
    let kernels = m.gpu.kernel_times().expect("this device times its kernels");
    m.gpu.set_kernel_timing(false);
    assert!(kernels.iter().any(|(name, _, _)| name.contains("gdn_chunk_loop_seg")), "the rounds did not run the segment recurrence: {kernels:?}");
    let mut worst = rel_l2(&last, &want[tokens.len() - 1]);
    for (i, &tok) in extra.iter().enumerate() {
        worst = worst.max(rel_l2(&m.step(tok), &want[tokens.len() + i]));
    }
    println!("segment rounds against the token replay: worst rel_l2 {worst:e}");
    assert!(worst < 1e-4, "the segment rounds differ from the token replay by rel_l2 {worst:e}");
}
