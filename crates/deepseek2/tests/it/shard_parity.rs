// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic pipeline sharding (`model::Pipeline<DeepseekV2>`) is bit-exact
//! against the single-device decoder - proving the sharding seam this crate
//! implements (`crate::shard`, shared by both DeepSeek-OCR and DeepSeek-OCR-2,
//! see the crate's own lib doc) is correct, not just present. `head_weight()`
//! is untied for this decoder, so stages hold fully disjoint parameters (no
//! replicated gradient to sum). The cut is placed automatically by
//! `plan_balanced`. Mirrors `gpt2::tests::shard_parity`/
//! `qwen35moe::tests::shard_parity` exactly.
//!
//! Two stages on the first two selected cards (BRAIN_DEVICE) by default; `SHARD_TEST_GPUS=1,1` pins both to
//! one card. Skips (does not fail) when this box has fewer discrete GPUs than
//! the stages need - single-GPU hardware cannot exercise the cross-device
//! cut, only the single-device path every other test in this crate already
//! gates.

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::DeepseekV2;
use model::{Batch, Pipeline};

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    if let Err(why) = stage_cards() {
        brain_testutil::skip_unavailable(&format!("deepseekv2 shard parity {why}"));
        return true;
    }
    false
}
fn stage_gpus() -> Vec<usize> {
    stage_cards().unwrap_or_default()
}

/// The two cards the stages go on: `SHARD_TEST_GPUS` when set (each one a card the selection names), else the first two
/// cards `BRAIN_DEVICE` selects; never cards the selection leaves to someone else.
fn stage_cards() -> Result<Vec<usize>, String> {
    let requested = std::env::var("SHARD_TEST_GPUS").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>()).filter(|v| !v.is_empty());
    gpu_core::testgpu::stage_cards(requested.as_deref(), 2)
}

#[test]
fn shard_forward_and_grad_parity_deepseekv2() {
    if gpu_disabled() {
        return;
    }
    let cfg = DeepseekV2Config::tiny();
    let init = deepseek2::init::init_weights(&cfg, 11);
    let (b, t) = (2u32, 8u32);
    let vocab = cfg.vocab();
    let x: Vec<u32> = (0..b * t).map(|i| i * 7 % vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 7 + 1) % vocab).collect();

    // Single-device reference - the SAME `new_on` every non-sharded caller
    // (deepseek2ocr, deepseekocr2) already uses, so this is the exact path
    // sharding must not regress.
    let single = DeepseekV2::new_on(gpu_core::testgpu::dev(deepseek2::model::PIPELINES), cfg.clone(), b, t, &init, true);
    single.set_batch(&x, &y);
    single.zero_grads();
    let l_single = single.forward();
    single.backward();
    single.poll_wait();

    // Auto-placed two-stage pipeline from the SAME weights.
    let pipe = Pipeline::<DeepseekV2>::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(pipe.n_stages(), 2);
    eprintln!("auto-placed shards: {:?}", pipe.shards());
    pipe.zero_grads();
    let l_pipe = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    pipe.backward();

    let dl = (l_single - l_pipe).abs() / l_single.abs().max(1e-6);
    eprintln!("deepseekv2 loss  single={l_single:.6}  pipe={l_pipe:.6}  rel={dl:.2e}");
    assert!(dl < 1e-4, "deepseekv2 sharded loss mismatch: {l_single} vs {l_pipe}");

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = pipe.reduced_grad(&name);
        let (mut num, mut den) = (0f32, 1e-6f32);
        for (p, q) in a.iter().zip(&g) {
            num = num.max((p - q).abs());
            den = den.max(p.abs());
        }
        let rel = num / den;
        if rel > worst {
            worst = rel;
            worst_name = name.clone();
        }
        assert!(rel < 1e-3, "deepseekv2 sharded grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("deepseekv2 worst grad rel {worst:.2e} ({worst_name})");
}
