// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified incremental decoding for transformer models for its clients. If your team
// needs expertise in KV-cache inference that provably matches a full recompute then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The KV-cache decode must produce the logits, and so the greedy tokens, of a full recompute of the prefix.
//!
//! The cached step and the recompute run different kernels and sum in different orders, so their logits differ by
//! round-off. How much is bounded, not measured: [`logit_bounds`] derives it from the binary32 round-off bound for
//! this model's shapes and weights, and the greedy test asserts the margin that bound needs before it claims the two
//! paths pick the same token.

use std::collections::HashMap;

use data::rng::Rng;
use gpt2::model::{Gpt, GptConfig, PIPELINES};
use gpt2::sample::{generate, generate_kv};

use crate::logit_bound::logit_bounds;

/// Weights for `cfg`: LayerNorm gains one, everything else Gaussian with standard deviation `std`.
fn weights(cfg: &GptConfig, rng: &mut Rng, std: f32) -> HashMap<String, Vec<f32>> {
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| {
            let v = if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "ln.weight" {
                vec![1.0f32; numel]
            } else {
                (0..numel).map(|_| rng.next_gaussian() as f32 * std).collect()
            };
            (name, v)
        })
        .collect()
}

/// Largest distance of a logit from its bound-implied tolerance: `|a - b| - bound` at the worst position.
fn worst_excess(a: &[f32], b: &[f32], bounds: &[f32]) -> f32 {
    a.iter().zip(b).enumerate().map(|(i, (x, y))| (x - y).abs() - bounds[i % bounds.len()]).fold(f32::NEG_INFINITY, f32::max)
}

/// The head applied on the host to the cached step's final-norm hidden state, against the device recompute of every
/// prefix, within the derived bound at every position.
#[test]
fn kv_step_matches_full_recompute_within_the_roundoff_bound() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    // d_model=32 (=> the fused-qkv weight-slice offsets d² are 256B-aligned), biases random so the fused-QKV bias
    // split is genuinely exercised.
    let cfg = GptConfig { vocab: 65, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, d_ff: 64 };
    let (d, v, seq) = (cfg.d_model as usize, cfg.vocab as usize, 10usize);
    // seed: a fixed weight draw so the reported worst difference reproduces; the bound holds for any draw
    let mut rng = Rng::new(4321);
    let init = weights(&cfg, &mut rng, 0.08);
    let m = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, cfg.block_size, &init);
    let head = m.read_weight("tok.weight"); // [v, d], the tied head, no bias
    let tokens: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % v as u64) as u32).collect();
    let bounds = logit_bounds(&cfg, &init, seq);

    let mut worst = 0.0f32;
    m.reset_cache();
    for (i, &tok) in tokens.iter().enumerate() {
        assert_eq!(m.cache_pos(), i as u32);
        let h = m.step(tok); // [d] final-LayerNorm hidden
        let cached: Vec<f32> = (0..v).map(|o| head[o * d..(o + 1) * d].iter().zip(&h).map(|(w, x)| w * x).sum()).collect();
        let full = m.logits_all(&tokens[..=i]); // [(i+1)*v]
        let excess = worst_excess(&cached, &full[i * v..(i + 1) * v], &bounds);
        assert!(excess <= 0.0, "prefix {i}: the cached step is {excess:e} beyond the round-off bound of the full recompute");
        worst = cached.iter().zip(&full[i * v..(i + 1) * v]).fold(worst, |m, (x, y)| m.max((x - y).abs()));
    }
    println!("cached step vs full recompute: largest difference {worst:e}, derived bound {:e}", bounds.iter().copied().fold(0.0f32, f32::max));
}

/// Greedy generation by cache equals greedy generation by recompute, and the test says why it may: at every step the
/// best logit leads the runner-up by more than the two paths can differ, so no draw of the weights is trusted to
/// "have no near tie", the margin is asserted.
#[test]
fn greedy_kv_generation_equals_recompute_with_a_margin_the_bound_needs() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = GptConfig::tiny();
    let vocab = cfg.vocab as usize;
    // seed: a fixed weight draw so the report reproduces; the margin assertion below is what makes the draw admissible
    let mut rng = Rng::new(1);
    let init = weights(&cfg, &mut rng, 0.05);
    let model = Gpt::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 1, cfg.block_size, &init);
    let prompt = vec![1u32, 5, 3];
    let new_tokens = 16;

    let recompute = generate(&model, &prompt, new_tokens, 0.0, 0, &mut Rng::new(data::rng::random_seed()), &mut |_, _| true);
    let kv = generate_kv(&model, &prompt, new_tokens, 0.0, 0, &mut Rng::new(data::rng::random_seed()), &mut |_, _| true);
    assert_eq!(recompute, kv, "KV greedy generation must equal recompute generation");
    // A caller that says stop ends either decode at the token it was shown.
    let stop_at = |i: usize, _: u32| i < 2;
    assert_eq!(generate(&model, &prompt, new_tokens, 0.0, 0, &mut Rng::new(1), &mut { stop_at }), recompute[..3]);
    assert_eq!(generate_kv(&model, &prompt, new_tokens, 0.0, 0, &mut Rng::new(1), &mut { stop_at }), kv[..3]);

    // Each path's logit is within the agreement bound of the other's, so a leading logit survives whenever its lead
    // exceeds twice that bound (once for the leader's move, once for the runner-up's).
    let bounds = logit_bounds(&cfg, &init, prompt.len() + new_tokens);
    let tolerance = 2.0 * bounds.iter().copied().fold(0.0f32, f32::max);
    println!("greedy margin: tolerance {tolerance:e}");
    let mut context = prompt;
    let mut smallest_gap = f32::INFINITY;
    for (step, &chosen) in recompute.iter().enumerate() {
        let logits = model.logits_all(&context);
        let mut row = logits[logits.len() - vocab..].to_vec();
        row.sort_by(|a, b| b.total_cmp(a));
        let gap = row[0] - row[1];
        assert!(gap > tolerance, "step {step}: top-2 logit gap {gap:e} is not beyond the {tolerance:e} the paths may differ by");
        smallest_gap = smallest_gap.min(gap);
        context.push(chosen);
    }
    println!("greedy margin: smallest top-2 gap {smallest_gap:e}");
}
