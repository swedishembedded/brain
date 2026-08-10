// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35moe::sample::{generate_kv_stream, generate_kv_stream_with_head}` —
//! the per-token-callback streaming variants added for `qwen35moe::caps`
//! (P13's `capability` wiring). Mirrors `qwen3::sample`'s own
//! `eos_stops_on_any_id_in_the_slice` / `with_head_matches_the_self_reading_wrapper`
//! coverage for the analogous claims on this crate's streaming entry points:
//! streaming must produce the SAME tokens as the non-streaming `generate_kv`
//! (the callback is purely an observability hook, not a behavior change), an
//! `eos` slice must stop generation at the FIRST id it contains, and the
//! caller-supplied-head variant must be bit-for-bit identical to the
//! self-reading wrapper.

use data::rng::Rng;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::Qwen35;
use qwen35moe::sample::{generate_kv, generate_kv_stream, generate_kv_stream_with_head};

fn tiny_model() -> (Qwen35, Vec<u32>) {
    let cfg = Qwen35Config::tiny();
    let init = qwen35moe::init::init_weights(&cfg, 11);
    let cap = cfg.block_size;
    let model = Qwen35::new(cfg.clone(), 1, cap, &init);
    let prompt: Vec<u32> = (0..4).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    (model, prompt)
}

#[test]
fn streaming_matches_non_streaming_and_calls_back_once_per_token() {
    let (model, prompt) = tiny_model();
    let mut r1 = Rng::new(3);
    let plain = generate_kv(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut r1);

    let mut calls = Vec::new();
    let mut r2 = Rng::new(3);
    let streamed = generate_kv_stream(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut r2, &mut |i, t| {
        calls.push((i, t));
        true
    });
    assert_eq!(streamed, plain, "streaming must produce the same tokens as generate_kv");
    assert_eq!(calls.len(), plain.len(), "on_token must fire exactly once per accepted token");
    for (i, (idx, tok)) in calls.iter().enumerate() {
        assert_eq!(*idx, i, "callback index must be the token's position in the output");
        assert_eq!(*tok, plain[i]);
    }
}

#[test]
fn eos_stops_on_any_id_in_the_slice() {
    let (model, prompt) = tiny_model();
    let mut r0 = Rng::new(1);
    let full = generate_kv(&model, &prompt, 8, 0.0, 0, 1.0, &[], &mut r0);
    assert!(!full.is_empty(), "greedy generation produced nothing to test against");
    let never_occurs = 999_999u32;
    let stop_at_id = full[0];

    let mut r1 = Rng::new(1);
    let truncated = generate_kv_stream(&model, &prompt, 8, 0.0, 0, 1.0, &[never_occurs, stop_at_id], &mut r1, &mut |_, _| true);
    assert!(truncated.is_empty(), "must stop before emitting the eos id (the very first sampled token here)");

    // Order in the slice must not matter.
    let mut r2 = Rng::new(1);
    let truncated2 = generate_kv_stream(&model, &prompt, 8, 0.0, 0, 1.0, &[stop_at_id, never_occurs], &mut r2, &mut |_, _| true);
    assert_eq!(truncated2, truncated);

    // An empty eos slice never stops early.
    let mut r3 = Rng::new(1);
    let no_stop = generate_kv_stream(&model, &prompt, 8, 0.0, 0, 1.0, &[], &mut r3, &mut |_, _| true);
    assert_eq!(no_stop, full, "empty eos slice must not stop generation");
}

#[test]
fn on_token_returning_false_stops_generation_early_but_keeps_the_token() {
    let (model, prompt) = tiny_model();
    let mut rng = Rng::new(5);
    let mut seen = 0usize;
    let out = generate_kv_stream(&model, &prompt, 10, 0.0, 0, 1.0, &[], &mut rng, &mut |_, _| {
        seen += 1;
        seen < 3 // stop after the 3rd accepted token
    });
    assert_eq!(out.len(), 3, "generation must stop as soon as on_token returns false, keeping that token");
}

#[test]
fn with_head_matches_the_self_reading_wrapper() {
    let (model, prompt) = tiny_model();
    let head = model.read_weight(model.cfg.head_weight());

    let mut r1 = Rng::new(7);
    let a = generate_kv_stream(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut r1, &mut |_, _| true);
    let mut r2 = Rng::new(7);
    let b = generate_kv_stream_with_head(&model, &prompt, 6, 0.0, 0, 1.0, &[], &mut r2, &head, &mut |_, _| true);
    assert_eq!(a, b, "generate_kv_stream_with_head must match generate_kv_stream given the same head");
}
