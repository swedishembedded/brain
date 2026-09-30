// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LM head materialises its logits a row chunk at a time once `rows·vocab`
//! passes the binding budget, and the backward then recomputes each chunk.
//! That must be invisible: the same loss, the same per-position logprobs and
//! the same gradients as one whole-batch chunk - weighted loss included.
//!
//! Its own test binary with a single test: the chunking is forced through the
//! process-wide `BRAIN_TILE_BUDGET_WORDS`, which a concurrently running test
//! would also see.

use std::collections::HashMap;

use qwen3::{Qwen, QwenConfig};

struct Run {
    loss: f32,
    logprobs: Vec<f32>,
    logits: Vec<f32>,
    grads: HashMap<String, Vec<f32>>,
}

fn run(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>, x: &[u32], y: &[u32], weights: &[f32]) -> Run {
    let t = x.len() as u32;
    let mut m = Qwen::new(cfg.clone(), 1, t, init);
    m.enable_weighted_loss();
    m.set_batch(x, y);
    m.write_weights(weights);
    m.zero_grads();
    let loss = m.forward();
    let logprobs = m.batch_token_logprobs();
    m.backward();
    m.poll_wait();
    let grads = cfg.param_list().into_iter().map(|(name, _)| (name.clone(), m.read_grad(&name))).collect();
    let logits = m.logits_all(x);
    Run { loss, logprobs, logits, grads }
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn a_chunked_head_trains_exactly_like_a_whole_one() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    // 200 rows: three full 64-row chunks and a partial one.
    let t = 200u32;
    let cfg = QwenConfig { block_size: t, max_position_embeddings: t, ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    // Every fifth target ignored, so masking crosses chunk boundaries.
    let y: Vec<u32> = (0..t).map(|i| if i % 5 == 3 { qwen3::IGNORE } else { (i * 7 + 2) % cfg.vocab }).collect();
    let weights: Vec<f32> = (0..t).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();

    std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");
    let whole = run(&cfg, &init, &x, &y, &weights);
    // One 64-row chunk of logits fits; the embedding and head weight
    // (23 x 16 words) still take a single vocab tile.
    std::env::set_var("BRAIN_TILE_BUDGET_WORDS", (64 * cfg.vocab).to_string());
    let chunked = run(&cfg, &init, &x, &y, &weights);
    std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");

    assert!((chunked.loss - whole.loss).abs() <= 1e-5 * whole.loss.abs(), "loss {} vs {}", chunked.loss, whole.loss);
    assert!(max_rel(&chunked.logprobs, &whole.logprobs) <= 1e-5, "per-position logprobs differ");
    assert_eq!(chunked.logits.len(), whole.logits.len());
    assert!(max_rel(&chunked.logits, &whole.logits) <= 1e-5, "logits_all differs");
    for (name, want) in &whole.grads {
        let got = &chunked.grads[name];
        let rel = max_rel(got, want);
        assert!(rel <= 1e-4, "{name}: chunked-head gradient differs by {rel:e} (relative)");
    }
}
