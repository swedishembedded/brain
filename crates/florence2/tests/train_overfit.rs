// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overfit proofs for `florence2::train::Florence2Trainer` (M6): forward,
//! backward, optimizer step, and weight update are actually wired together
//! end to end, on a checkpoint-free tiny fixture - no real checkpoint
//! needed, since capacity to memorize a synthetic example is a property of
//! the WIRING, not the pretrained weights (the gradcheck gate in
//! `crates/gradcheck/tests/florence2_fd.rs` already proves the gradient
//! itself is correct; this proves the optimizer loop actually uses it).
//!
//! Swedish Embedded AB builds from-scratch GPU training stacks for
//! vision-language and encoder-decoder models. If your team needs an
//! overfit-verified LoRA or full-fine-tune training loop for a composite
//! architecture like this one, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::HashMap;

use florence2::text::{BartConfig, LoraCfg};
use florence2::train::{self, Florence2Trainer};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

struct Example {
    prompt_ids: Vec<u32>,
    decoder_ids: Vec<u32>,
    targets: Vec<u32>,
}

fn example(cfg: &BartConfig, t_prompt: u32, max_t: u32, seed: u32) -> Example {
    let v = cfg.vocab_size;
    Example {
        prompt_ids: (0..t_prompt).map(|i| (i * 5 + seed) % v).collect(),
        decoder_ids: (0..max_t).map(|i| (i * 3 + seed + 1) % v).collect(),
        targets: (0..max_t).map(|i| (i * 7 + seed + 2) % v).collect(),
    }
}

fn build(cfg: &BartConfig, t_prompt: u32, max_t: u32, lora: Option<LoraCfg>, init: &HashMap<String, Vec<f32>>) -> Florence2Trainer {
    Florence2Trainer::new(gpu_core::testgpu::dev(train::PIPELINES), cfg.clone(), t_prompt, max_t, lora, init)
}

/// Full fine-tune: every base tensor moves, and the wiring alone (matching
/// `deepseekocr2::tests::train_overfit`'s own reasoning) drives a single
/// example's loss to near zero.
#[test]
fn full_finetune_overfits_a_single_example() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("florence2 overfit: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let cfg = BartConfig::tiny();
    let (t_prompt, max_t) = (4u32, 4u32);
    let ex = example(&cfg, t_prompt, max_t, 3);
    let names = train::param_list(&cfg, t_prompt, max_t, None);
    let init = train::init_weights(&names, 0.15, 101);
    let m = build(&cfg, t_prompt, max_t, None, &init);
    m.set_batch(&ex.prompt_ids, &ex.decoder_ids, &ex.targets);

    let lr = 5e-2f32;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for epoch in 1..=200u32 {
        m.zero_grads();
        let loss = m.loss();
        m.backward();
        m.adamw_step(epoch, lr, 0.0, None);
        if epoch == 1 {
            first = loss;
        }
        last = loss;
    }
    m.poll_wait();
    eprintln!("full_finetune_overfits_a_single_example: loss {first:.6} -> {last:.6}");
    assert!(first > 0.5, "the fixture's initial loss ({first}) is suspiciously low for a random init - not exercising real learning");
    assert!(last < 0.05, "full fine-tune did not drive a single example's loss near zero: {first} -> {last}");
}

/// Full fine-tune, several distinct examples cycled once per epoch: a
/// broken gradient path could not drive multiple different targets' losses
/// down together by coincidence.
#[test]
fn full_finetune_overfits_a_small_batch() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("florence2 overfit: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let cfg = BartConfig::tiny();
    let (t_prompt, max_t) = (4u32, 4u32);
    let examples: Vec<Example> = [3u32, 11].iter().map(|&s| example(&cfg, t_prompt, max_t, s)).collect();
    let names = train::param_list(&cfg, t_prompt, max_t, None);
    let init = train::init_weights(&names, 0.15, 202);
    let m = build(&cfg, t_prompt, max_t, None, &init);

    let lr = 5e-2f32;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for epoch in 1..=600u32 {
        let mut mean = 0.0f32;
        for ex in &examples {
            m.set_batch(&ex.prompt_ids, &ex.decoder_ids, &ex.targets);
            m.zero_grads();
            let loss = m.loss();
            m.backward();
            m.adamw_step(epoch, lr, 0.0, None);
            mean += loss;
        }
        mean /= examples.len() as f32;
        if epoch == 1 {
            first = mean;
        }
        last = mean;
    }
    m.poll_wait();
    eprintln!("full_finetune_overfits_a_small_batch: loss {first:.6} -> {last:.6} over {} examples", examples.len());
    assert!(last < 0.1, "full fine-tune did not drive the batch's mean loss near zero: {first} -> {last}");
}

/// LoRA against a REAL base (mirrors `deepseekocr2::tests::train_overfit`'s
/// own two-phase reasoning: a from-scratch random network is not a fair
/// target for a rank-limited adapter, since memorizing an arbitrary
/// next-token mapping needs the WHOLE network's capacity). Phase 1 full-
/// fine-tunes one example to near zero; phase 2 freezes that base, merges a
/// fresh (`B=0`) adapter, and confirms the adapter mechanism makes large,
/// real progress on a SECOND, different example.
#[test]
fn lora_makes_large_progress_against_a_real_base() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("florence2 overfit: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let cfg = BartConfig::tiny();
    let (t_prompt, max_t) = (4u32, 4u32);
    let base_ex = example(&cfg, t_prompt, max_t, 5);
    let lora_ex = example(&cfg, t_prompt, max_t, 41);

    let base_names = train::param_list(&cfg, t_prompt, max_t, None);
    let base_init = train::init_weights(&base_names, 0.15, 303);
    let base = build(&cfg, t_prompt, max_t, None, &base_init);
    base.set_batch(&base_ex.prompt_ids, &base_ex.decoder_ids, &base_ex.targets);
    let mut base_loss = f32::NAN;
    for epoch in 1..=200u32 {
        base.zero_grads();
        base_loss = base.loss();
        base.backward();
        base.adamw_step(epoch, 5e-2, 0.0, None);
    }
    base.poll_wait();
    assert!(base_loss < 0.05, "phase 1 (building a real base) did not converge: {base_loss}");

    let trained: HashMap<String, Vec<f32>> = base.param_names().into_iter().map(|n| { let v = base.read_weight(&n); (n, v) }).collect();

    let lora_cfg = LoraCfg { rank: 2, alpha: 4.0 };
    let lora_names = train::param_list(&cfg, t_prompt, max_t, Some(lora_cfg));
    let mut lora_init = train::init_weights(&lora_names, 0.15, 909);
    for (name, data) in trained {
        lora_init.insert(name, data);
    }
    let lora_m = build(&cfg, t_prompt, max_t, Some(lora_cfg), &lora_init);
    lora_m.set_batch(&lora_ex.prompt_ids, &lora_ex.decoder_ids, &lora_ex.targets);

    let lr = 3e-2f32;
    let mut first = f32::NAN;
    let mut best = f32::INFINITY;
    for epoch in 1..=400u32 {
        lora_m.zero_grads();
        let loss = lora_m.loss();
        lora_m.backward();
        lora_m.adamw_step(epoch, lr, 0.0, None);
        if epoch == 1 {
            first = loss;
        }
        best = best.min(loss);
    }
    lora_m.poll_wait();
    eprintln!("lora_makes_large_progress_against_a_real_base: loss {first:.6} -> best {best:.6}");
    assert!(first > 1.0, "the phase-1 base's starting confusion on the NEW example ({first}) is suspiciously low - phase 1 may not have produced a genuinely confident (and thus hard-to-correct) wrong base");
    assert!(best < first * 0.5, "LoRA against a real base did not make real progress: {first} -> best {best}");
}

/// A freshly-built LoRA composite (`B=0`) is a bit-for-bit no-op against
/// the equivalent un-adapted forward - the house pattern every
/// LoRA-adopting model in this repo holds to.
#[test]
fn a_fresh_adapter_is_a_no_op() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("florence2 overfit: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let cfg = BartConfig::tiny();
    let (t_prompt, max_t) = (4u32, 4u32);
    let ex = example(&cfg, t_prompt, max_t, 7);
    let seed = 55u64;

    let base_names = train::param_list(&cfg, t_prompt, max_t, None);
    let base_init = train::init_weights(&base_names, 0.15, seed);
    let lora_cfg = LoraCfg { rank: 2, alpha: 4.0 };
    let lora_names = train::param_list(&cfg, t_prompt, max_t, Some(lora_cfg));
    let lora_init = train::init_weights(&lora_names, 0.15, seed);
    for (name, data) in &base_init {
        assert_eq!(lora_init.get(name), Some(data), "{name}: base tensor value drifted between the plain and LoRA-configured init");
    }

    let base = build(&cfg, t_prompt, max_t, None, &base_init);
    base.set_batch(&ex.prompt_ids, &ex.decoder_ids, &ex.targets);
    let loss_base = base.loss();

    let lora_m = build(&cfg, t_prompt, max_t, Some(lora_cfg), &lora_init);
    lora_m.set_batch(&ex.prompt_ids, &ex.decoder_ids, &ex.targets);
    let loss_lora = lora_m.loss();

    assert_eq!(loss_base.to_bits(), loss_lora.to_bits(), "a fresh (B=0) adapter changed the loss: {loss_base} vs {loss_lora}");
}
