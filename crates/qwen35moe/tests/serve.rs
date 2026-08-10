// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35moe::serve::Engine`/`Scheduler` must reproduce `Qwen35::step`'s
//! single-sequence decode EXACTLY -- going through the paged
//! `Scheduler`/`Engine`/`BlockTable` machinery must not change the actual
//! numbers versus the already-proven-correct P11b decode path
//! (`decode_step.rs` gates `Qwen35::step` itself against whole-sequence
//! `logits_all`; this test's job is only to gate the NEW paged wiring this
//! module adds on top of it).
//!
//! Admits one request, drives it to completion via
//! `model::serve::Scheduler<qwen35moe::serve::Engine>`, and compares its
//! generated tokens token-for-token against `qwen35moe::sample::generate_kv`
//! (greedy, `temperature=0.0`) run directly over a second `Qwen35` instance
//! built from the SAME weights -- the same "one model instance for the
//! reference, one independent path for the thing under test" structure
//! `decode_step.rs` itself uses. Runs on both the CPU JIT and the default
//! GPU backend (`docs/lessons.md` #5).

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use model::serve::Request;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{Qwen35, PIPELINES};
use qwen35moe::serve::{Engine, Scheduler};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    qwen35moe::init::init_weights(cfg, seed)
}

fn run(make_gpu: fn(&[(&str, &str)]) -> Gpu) {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 11);
    let prompt = vec![1u32, 5, 3, 9, 2, 7];
    let max_new = 6usize;
    let max_seq_len = (prompt.len() + max_new) as u32;

    // Reference: `Qwen35::step`'s own single-sequence decode, driven by
    // `crate::sample::generate_kv` (greedy) -- P11b, already proven correct
    // against `logits_all` by `decode_step.rs`. `t_ref` just needs to be
    // positive (`gdn_chunk_size` always returns a divisor of its own input),
    // so the prompt+max_new length is as good a choice as any.
    let reference = Qwen35::new_on(make_gpu(PIPELINES), cfg.clone(), 1, max_seq_len, &init);
    let mut rng = Rng::new(1);
    let want = qwen35moe::sample::generate_kv(&reference, &prompt, max_new, 0.0, 0, 1.0, &[], &mut rng);
    assert_eq!(want.len(), max_new, "greedy decode with no eos must always produce exactly max_new tokens");

    // Under test: the SAME prompt, on the SAME weights, through the paged
    // Scheduler/Engine/BlockTable machinery this file adds.
    // `max_concurrent=2`: exercises a pool wider than the one sequence this
    // test actually admits, so a bug that only shows up when
    // `blocks()[0] != 0` (i.e. hard-coded to the first physical block) would
    // have a chance to surface if a later change reordered allocation.
    let engine = Engine::from_map_on(&make_gpu(PIPELINES), cfg, &init, max_seq_len, 2);
    println!("kv_pool_bytes={} kv_pool_capacity_tokens={}", engine.kv_pool_bytes(), engine.kv_pool_capacity_tokens());
    let mut sched = Scheduler::new(engine, 1);
    let id = sched.submit(Request { prompt: prompt.clone(), max_new, eos: None });
    let out = sched.run();
    let got = out.get(&id).expect("the admitted request must complete");

    assert_eq!(got, &want, "paged Scheduler/Engine decode must exactly match Qwen35::step's single-sequence decode");
    println!("serve engine matches Qwen35::step over {max_new} greedy tokens: {got:?}");
}

#[test]
fn scheduler_decode_matches_step_cpu() {
    run(Gpu::new_cpu);
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise -- run this under both `BRAIN_DEVICE=cpu` and unset
/// (the default GPU backend) per `docs/lessons.md` #5.
/// `scheduler_decode_matches_step_cpu` above pins the CPU JIT explicitly
/// regardless of `BRAIN_DEVICE` so the CPU path is always exercised even when
/// this one runs against the GPU.
#[test]
fn scheduler_decode_matches_step_default_backend() {
    run(Gpu::new);
}
