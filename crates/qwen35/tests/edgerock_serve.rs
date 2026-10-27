// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context language
// models at a fixed memory budget for its clients. If your team needs
// expertise in bounded-state inference engines and prefix reuse for hybrid
// recurrent/attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! EdgeRock served through the generic scheduler at tiny dims (window 11, commit blocks of 4, a 6-slot bank,
//! positions up to 48): the ring engine ([`qwen35::serve_ring::RingEngine`]) passes the shared conformance suite,
//! a scheduler-driven batch of sequences generates exactly what each sequence decoded alone with `Qwen35::step`
//! does, a prefix hit reproduces the cold run while computing only the suffix, and the cache keeps its
//! namespace, LRU and reclaim rules.
//!
//! Every scenario runs on the CPU JIT; the default device (CUDA under the build environment) runs the same
//! scenarios with the engine and the reference on the same device.

use std::collections::HashMap;

use gpu_core::Gpu;
use model::paged::BlockTable;
use model::serve::conformance::{self, Conformance, MIN_SEQ_TOKENS};
use model::serve::{PagedDecoder, Request, Scheduler};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use qwen35::serve_ring::RingEngine;

const MAX_CONCURRENT: u32 = 4;
const PREFIX_SLOTS: u32 = 3;

fn weights() -> HashMap<String, Vec<f32>> {
    qwen35::init::init_weights(&Qwen35Config::edgerock_tiny(), 11)
}

fn f32_tier() -> model::ops::TierPolicy {
    model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32)
}

fn engine_on(gpu: &Gpu, prefix_slots: u32) -> RingEngine {
    let cfg = Qwen35Config::edgerock_tiny();
    RingEngine::from_source_tier_on(gpu, cfg.clone(), &weights(), cfg.max_position_embeddings, MAX_CONCURRENT, prefix_slots, &f32_tier())
}

fn cpu_engine() -> RingEngine {
    engine_on(&Gpu::new_cpu(pipelines()), PREFIX_SLOTS)
}

fn default_engine() -> RingEngine {
    engine_on(&Gpu::new(pipelines()), PREFIX_SLOTS)
}

fn tokens(seed: usize, n: usize) -> Vec<u32> {
    (0..n).map(|i| 1 + ((seed * 7 + i * 13 + i * i) % 27) as u32).collect()
}

/// Greedy continuation of `prompt` by `max_new` tokens, one sequence on the model's own `step` path.
fn reference(gpu: Gpu, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let cfg = Qwen35Config::edgerock_tiny();
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &weights(), &f32_tier());
    let head = m.read_weight("tok.weight");
    let mut hidden = Vec::new();
    for &t in prompt {
        hidden = m.step(t);
    }
    let mut out = Vec::new();
    for _ in 0..max_new {
        let next = head.chunks(hidden.len()).enumerate().max_by(|a, b| a.1.iter().zip(&hidden).map(|(x, y)| x * y).sum::<f32>().total_cmp(&b.1.iter().zip(&hidden).map(|(x, y)| x * y).sum::<f32>())).unwrap().0 as u32;
        out.push(next);
        hidden = m.step(next);
    }
    out
}

fn run_to_end(sched: &mut Scheduler<RingEngine>) -> HashMap<u64, Vec<u32>> {
    let mut done = HashMap::new();
    while sched.pending() {
        done.extend(sched.step());
    }
    done
}

fn submit(sched: &mut Scheduler<RingEngine>, prompt: &[u32], max_new: usize) -> u64 {
    sched.submit(Request { prompt: prompt.to_vec(), max_new, stop: Vec::new() })
}

#[test]
fn the_ring_engine_passes_the_conformance_suite() {
    conformance::run_all(&Conformance { make: cpu_engine, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}

#[test]
fn the_ring_engine_passes_the_conformance_suite_on_the_default_device() {
    conformance::run_all(&Conformance { make: default_engine, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}

#[test]
fn scheduler_batched_generation_equals_each_sequence_decoded_alone() {
    for (name, gpu) in [("cpu", Gpu::new_cpu(pipelines())), ("default", Gpu::new(pipelines()))] {
        // Different lengths, so the batch holds sequences at different positions, ring phases and bank fills.
        let jobs: Vec<(Vec<u32>, usize)> = vec![(tokens(1, 30), 14), (tokens(2, 19), 20), (tokens(3, 12), 9)];
        let mut sched = Scheduler::new(engine_on(&gpu, PREFIX_SLOTS), MAX_CONCURRENT as usize);
        let ids: Vec<u64> = jobs.iter().map(|(p, n)| submit(&mut sched, p, *n)).collect();
        let done = run_to_end(&mut sched);
        for ((prompt, max_new), id) in jobs.iter().zip(ids) {
            assert_eq!(done[&id], reference(gpu.new_like(pipelines()), prompt, *max_new), "{name}: a batched sequence differs from its own step decoding");
        }
    }
}

#[test]
fn a_prefix_hit_reproduces_the_cold_run_and_prefills_only_the_suffix() {
    let gpu = Gpu::new_cpu(pipelines());
    let first = tokens(5, 24);
    let mut extended = first.clone();
    extended.extend(tokens(6, 8));
    let cold = {
        let mut sched = Scheduler::new(engine_on(&gpu, PREFIX_SLOTS), 2);
        let id = submit(&mut sched, &extended, 6);
        run_to_end(&mut sched).remove(&id).unwrap()
    };

    let mut sched = Scheduler::new(engine_on(&gpu, PREFIX_SLOTS), 2);
    submit(&mut sched, &first, 4);
    run_to_end(&mut sched);
    assert_eq!(sched.prefix_stats().0, 0, "nothing was cached before the first prompt");

    let id2 = submit(&mut sched, &extended, 6);
    let warm = run_to_end(&mut sched).remove(&id2).unwrap();
    let (hits, looked_up, cached) = sched.prefix_stats();
    assert_eq!(hits as usize, first.len(), "exactly the cached prompt is reused; only the {} suffix tokens are prefilled", extended.len() - first.len());
    assert_eq!(looked_up as usize, first.len() + extended.len());
    assert_eq!(cached, 1, "the longer prompt superseded the shorter entry");
    assert_eq!(warm, cold, "a prefix hit must reproduce the cold run's tokens");

    // An exact repeat is served from the entry alone.
    let id3 = submit(&mut sched, &extended, 6);
    let again = run_to_end(&mut sched).remove(&id3).unwrap();
    assert_eq!(sched.prefix_stats().0 as usize, first.len() + extended.len());
    assert_eq!(again, cold);
}

#[test]
fn cached_prompts_obey_namespaces_lru_and_reclaim() {
    let mut engine = engine_on(&Gpu::new_cpu(pipelines()), 3);
    let prompts: Vec<Vec<u32>> = (0..3).map(|s| tokens(10 + s, 20)).collect();
    let run = |engine: &mut RingEngine, namespace: u32, prompt: &[u32]| {
        let mut table = BlockTable::with_namespace(namespace);
        let hidden = engine.prefill(&mut table, prompt);
        engine.release_table(&mut table);
        hidden
    };

    // A prompt cached under one namespace is invisible to another and to the base.
    let cold = run(&mut engine, 7, &prompts[0]);
    assert_eq!(engine.prefix_stats().2, 1);
    run(&mut engine, 0, &prompts[0]);
    run(&mut engine, 8, &prompts[0]);
    assert_eq!(engine.prefix_stats().0, 0, "no other namespace may reuse the entry");
    assert_eq!(run(&mut engine, 7, &prompts[0]), cold, "its own namespace does, to the bit");
    assert_eq!(engine.prefix_stats().0 as usize, prompts[0].len());

    // Reclaiming blocks never drops an entry: the cache holds snapshot slots, not pool blocks.
    let entries = engine.prefix_stats().2;
    assert_eq!(engine.reclaim_prefix(MAX_CONCURRENT), 0);
    assert_eq!(engine.prefix_stats().2, entries);

    // Two snapshot slots: the third distinct prompt evicts the least recently used one.
    let mut engine = engine_on(&Gpu::new_cpu(pipelines()), 2);
    for p in &prompts {
        run(&mut engine, 0, p);
    }
    assert_eq!(engine.prefix_stats().2, 2);
    let hits = engine.prefix_stats().0;
    run(&mut engine, 0, &prompts[0]);
    assert_eq!(engine.prefix_stats().0, hits, "the oldest prompt was evicted");
    run(&mut engine, 0, &prompts[2]);
    assert_eq!(engine.prefix_stats().0 as usize, hits as usize + prompts[2].len(), "the newest is still cached");
}

#[test]
fn a_released_slot_leaks_nothing_into_the_next_sequence() {
    let gpu = Gpu::new_cpu(pipelines());
    let victim = tokens(20, 30);
    let next = tokens(21, 25);
    let alone = {
        let mut sched = Scheduler::new(engine_on(&gpu, 0), 1);
        let id = submit(&mut sched, &next, 8);
        run_to_end(&mut sched).remove(&id).unwrap()
    };
    // One live slot, so the second request reuses the first's slot; the first is cancelled mid-decode.
    let mut sched = Scheduler::new(
        RingEngine::from_source_tier_on(&gpu, Qwen35Config::edgerock_tiny(), &weights(), 48, 1, 0, &f32_tier()),
        1,
    );
    let a = submit(&mut sched, &victim, 12);
    for _ in 0..5 {
        sched.step();
    }
    assert!(sched.cancel(a).is_some());
    let b = submit(&mut sched, &next, 8);
    assert_eq!(run_to_end(&mut sched).remove(&b).unwrap(), alone, "a reused slot must start empty");
}

#[test]
fn state_bytes_are_fixed_and_match_the_analytic_size() {
    let cfg = Qwen35Config::edgerock_tiny();
    let mut engine = engine_on(&Gpu::new_cpu(pipelines()), PREFIX_SLOTS);
    let slots = u64::from(MAX_CONCURRENT + PREFIX_SLOTS);
    assert_eq!(engine.state_bytes(), qwen35::model::slot_state_bytes(&cfg) * slots);
    assert_eq!(engine.kv_pool_bytes(), engine.state_bytes());
    let before = engine.state_bytes();
    let mut table = BlockTable::new();
    engine.prefill(&mut table, &tokens(30, 40));
    engine.forward_batched_greedy(&mut [&mut table], &[3]);
    engine.release_table(&mut table);
    assert_eq!(engine.state_bytes(), before);
    assert!(qwen35::serve_ring::memory_param_bytes(&cfg) > 0);
    assert_eq!(engine.max_seq_len(), cfg.max_position_embeddings as usize);
    assert!(engine.max_seq_len() >= MIN_SEQ_TOKENS);
}
