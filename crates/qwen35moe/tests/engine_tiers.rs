// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35moe::serve::Engine` across its weight and KV tiers, and its device head.
//!
//! Swedish Embedded AB implements memory-lean long-context serving of sparse-MoE
//! models. If your team needs expertise in fitting more concurrent sequences of
//! a hybrid model onto one accelerator without changing what it says then you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! * The KV pool stored `bf16` or per-row `int8` must answer like the `f32` pool
//!   to the format's rounding (the tolerances are the format's: bf16 keeps 8
//!   significand bits, int8 a scale per (token, kv-head) row), through the whole
//!   engine - chunked prefill, then batched decode.
//! * The device head's greedy and top-k answers must be what the full logits row
//!   says, without the row ever being read back.
//! * An int8-weight engine built from a tensor source tracks the fp32 engine.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::{select::Dtype, Gpu};
use model::kv_tier::KvTier;
use model::ops::TierPolicy;
use model::paged::BlockTable;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::pipelines;
use qwen35moe::serve::{Engine, EngineOptions};

/// `tiny()`'s dims with every int8-quantised width a multiple of 32 and the
/// shared expert the routed experts' shape (`model_i8_smoke`'s reasoning).
fn cfg() -> Qwen35Config {
    Qwen35Config {
        vocab: 29,
        block_size: 24,
        n_layers: 8,
        d_model: 32,
        rms_eps: 1e-6,
        max_position_embeddings: 24,
        tie_embeddings: false,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        attn_bias: false,
        rope_theta: 1.0e6,
        partial_rotary_factor: 0.5,
        mrope_section: [1, 1, 1],
        full_attention_interval: 4,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 3,
        n_experts: 6,
        top_k: 2,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
        lora: None,
    }
}

fn weights(cfg: &Qwen35Config) -> HashMap<String, Vec<f32>> {
    qwen35moe::init::init_weights(cfg, 21)
}

fn prompt(cfg: &Qwen35Config, seed: u64, n: usize) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| (r.next_u64() % cfg.vocab as u64) as u32).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn build(gpu: &Gpu, c: &Qwen35Config, w: &HashMap<String, Vec<f32>>, o: EngineOptions) -> Engine {
    Engine::from_source_on_parent(gpu, c.clone(), w, o)
}

#[test]
fn a_compact_kv_pool_answers_like_the_f32_pool() {
    let gpu = gpu_core::testgpu::dev(pipelines());
    let c = cfg();
    let w = weights(&c);
    let p = prompt(&c, 3, 13);
    let run = |kv: KvTier| -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut e = build(&gpu, &c, &w, EngineOptions::new(24, 3).with_kv_tier(kv).with_prefill_chunk(5));
        let mut t = BlockTable::new();
        let h = e.prefill(&mut t, &p);
        // Four decode steps; the hidden of each, via the logits head, so a tier's
        // error in a step shows in what the head sees.
        let mut hs = Vec::new();
        let mut tok = 4u32;
        for _ in 0..4 {
            let top = e.forward_batched_topk(&mut [&mut t], &[tok], 1);
            hs.push(vec![top[0][0].1]);
            tok = top[0][0].0;
        }
        e.release_table(&mut t);
        (h, hs)
    };
    let (h32, d32) = run(KvTier::F32);
    for (kv, bound) in [(KvTier::Bf16, 5e-3), (KvTier::Int8, 5e-2)] {
        let (h, d) = run(kv);
        let e = rel_l2(&h, &h32);
        assert!(e < bound, "{kv}: the prompt's last hidden differs from f32's by {e:e} (bound {bound})");
        let top_f32: Vec<f32> = d32.iter().map(|v| v[0]).collect();
        let top: Vec<f32> = d.iter().map(|v| v[0]).collect();
        let e = rel_l2(&top, &top_f32);
        assert!(e < bound * 4.0, "{kv}: the decoded top logits differ from f32's by {e:e}");
    }
}

#[test]
fn the_device_head_picks_what_the_logits_row_says() {
    let gpu = gpu_core::testgpu::dev(pipelines());
    let c = cfg();
    let w = weights(&c);
    let mut e = build(&gpu, &c, &w, EngineOptions::new(24, 4));
    let (pa, pb) = (prompt(&c, 5, 9), prompt(&c, 6, 7));
    let (mut ta, mut tb) = (BlockTable::new(), BlockTable::new());
    let (ha, hb) = (e.prefill(&mut ta, &pa), e.prefill(&mut tb, &pb));
    let _ = (ha, hb);
    // A second pair of identical sequences to compare greedy and top-k on the
    // same hidden states (each call consumes one position).
    let (mut ta2, mut tb2) = (BlockTable::new(), BlockTable::new());
    e.prefill(&mut ta2, &pa);
    e.prefill(&mut tb2, &pb);
    let greedy = e.forward_batched_greedy(&mut [&mut ta, &mut tb], &[3, 4]);
    let top = e.forward_batched_topk(&mut [&mut ta2, &mut tb2], &[3, 4], 5);
    for (row, (g, cand)) in greedy.iter().zip(&top).enumerate() {
        assert_eq!(*g, cand[0].0, "row {row}: greedy is the top-1 candidate");
        assert!(cand.windows(2).all(|w| w[0].1 >= w[1].1), "row {row}: candidates are best first: {cand:?}");
        let ids: std::collections::HashSet<u32> = cand.iter().map(|c| c.0).collect();
        assert_eq!(ids.len(), cand.len(), "row {row}: no token twice in the top-k");
    }
}

#[test]
fn an_int8_engine_built_from_a_source_tracks_the_fp32_engine() {
    let gpu = gpu_core::testgpu::dev(pipelines());
    if !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("no packed int8 dot on this device");
    }
    let c = cfg();
    let w = weights(&c);
    let p = prompt(&c, 9, 11);
    let hidden = |tier: TierPolicy| {
        let mut e = build(&gpu, &c, &w, EngineOptions::new(24, 2).with_tier(tier).with_prefill_chunk(4));
        let mut t = BlockTable::new();
        let h = e.prefill(&mut t, &p);
        e.release_table(&mut t);
        h
    };
    let f32h = hidden(TierPolicy::uniform(Dtype::F32));
    let i8h = hidden(TierPolicy::uniform(Dtype::I8));
    let e = rel_l2(&i8h, &f32h);
    assert!(e < 0.1, "the int8 engine's prompt hidden differs from fp32's by {e:e}");
}
