// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35::sample::generate_kv` - greedy decoding must be deterministic,
//! stop at any EOS id, and (since it is a thin loop around the
//! already-validated `Qwen35::step`, `crates/qwen35/tests/decode_step.rs`)
//! must match a hand-rolled step-by-step argmax over the same prompt
//! exactly. Mirrors `qwen35moe/tests/sample_generate.rs` exactly.

use data::rng::Rng;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use qwen35::sample::generate_kv;

fn tiny_model() -> (Qwen35, Vec<u32>) {
    let cfg = Qwen35Config::tiny();
    let init = qwen35::init::init_weights(&cfg, 11);
    let cap = cfg.block_size;
    let model = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cap, &init);
    let prompt: Vec<u32> = (0..4).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    (model, prompt)
}

fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi as u32
}

#[test]
fn greedy_generation_is_deterministic() {
    let (model, prompt) = tiny_model();
    let mut rng1 = Rng::new(1);
    let mut rng2 = Rng::new(2); // greedy ignores the seed entirely
    let out1 = generate_kv(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut rng1);
    let out2 = generate_kv(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut rng2);
    assert_eq!(out1, out2, "greedy decoding must not depend on the rng seed");
    assert!(out1.iter().all(|&t| t < model.cfg.vocab), "generated a token outside the vocab");
}

#[test]
fn generation_stops_at_eos() {
    let (model, prompt) = tiny_model();
    let mut rng = Rng::new(1);
    // Run once, unrestricted, to find a real token this model actually
    // produces at some position -- using it as the "eos" id proves the stop
    // check is load-bearing (not just "eos never fires in this tiny model").
    let free = generate_kv(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut rng);
    assert!(!free.is_empty(), "greedy generation produced nothing to test against");
    let eos = free[0];
    let mut rng = Rng::new(1);
    let stopped = generate_kv(&model, &prompt, 6, 0.0, 0, 1.0, &[eos], &mut rng);
    assert!(stopped.is_empty(), "generation should stop before emitting the eos id, got {stopped:?}");
}

#[test]
fn greedy_generation_matches_hand_rolled_step_argmax() {
    let (model, prompt) = tiny_model();
    let head = model.read_weight(model.cfg.head_weight());
    let (vocab, d) = (model.cfg.vocab as usize, model.cfg.d_model as usize);
    let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(&head, hidden, vocab, d) };

    // Reference: reset, feed the prompt, then greedily step + argmax by hand.
    model.reset_decode_cache();
    let mut hidden = Vec::new();
    for &t in &prompt {
        hidden = model.step(t);
    }
    let mut want = Vec::new();
    for _ in 0..5 {
        let next = argmax(&logits_of(&hidden));
        want.push(next);
        hidden = model.step(next);
    }

    let mut rng = Rng::new(42);
    let got = generate_kv(&model, &prompt, 5, 0.0, 0, 1.0, &[], &mut rng);
    assert_eq!(got, want);
}
