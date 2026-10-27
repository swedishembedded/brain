// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for profiling and tuning
// long-context inference on a specific accelerator for its clients. If your
// team needs expertise in making a model decode at the speed of its memory
// system then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Per-kernel device profile of one EdgeRock decode step (`RingEngine::forward_batched_greedy`) at a decode
//! position where the ring and the anchor bank are full: the steady state of a long conversation.
//!
//! Usage: `edgerock_decode_profile (<model dir> | synth[:<layers>]) [--batch N] [--position P] [--reps R]
//! [--tape] [--device gpu1] [--backend cuda]`. `--tape` replays a recorded step instead of building every step. `synth` is the published shape with `<layers>` layers (default 8) on random
//! weights, for measuring the hardware without the checkpoint. Prints the state bytes per sequence, the
//! wall time of a step, and the per-kernel device table.

use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::config::Qwen35Config;
use qwen35::serve_ring::RingEngine;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let flag = |name: &str| a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned();
    let num = |name: &str, default: u32| flag(name).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name} takes a number, got {v:?}")));
    let (batch, position, reps) = (num("--batch", 1), num("--position", 2048), num("--reps", 3));
    let set = gpu_core::select_compute(flag("--device").as_deref().unwrap_or("gpu"), flag("--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));
    let _ = set;
    let Some(target) = a.get(1).filter(|s| !s.starts_with("--")).cloned() else {
        eprintln!("usage: edgerock_decode_profile (<model dir> | synth[:layers]) [--batch N] [--position P] [--reps R] [--tape] [--device gpu1]");
        std::process::exit(2);
    };
    // `--weights` takes the generic tier grammar (`bf16`, `f16`, `f32`, `i8`, `q4`, `nf4`, `f8e4m3`, `f4e2m1`, with
    // `pattern=tier` rules); the checkpoint's bf16 tensors are quantized group-wise on load.
    let tier = model::ops::TierPolicy::parse(flag("--weights").as_deref().unwrap_or("bf16")).unwrap_or_else(|e| panic!("{e}"));
    println!("weight tier: {}", tier.summary());
    let (cfg, source): (Qwen35Config, Box<dyn checkpoint::TensorSource>) = if let Some(layers) = target.strip_prefix("synth") {
        let layers: u32 = layers.strip_prefix(':').map_or(8, |l| l.parse().expect("synth:<layers>"));
        let cfg = Qwen35Config { n_layers: layers, ..Qwen35Config::edgerock_4b() };
        let weights = qwen35::init::init_weights(&cfg, data::rng::random_seed());
        (cfg, Box::new(weights))
    } else {
        let dir = std::path::Path::new(&target);
        let cfg = qwen35::fp8_source::hf_config(dir).unwrap_or_else(|e| panic!("{e}"));
        let src = qwen35::fp8_source::open_dir(dir, &cfg, 128).unwrap_or_else(|e| panic!("{e}"));
        (cfg, Box::new(src))
    };
    let max_seq = (position + 64).min(cfg.max_position_embeddings);
    println!("EdgeRock decode profile: {} layers, batch {batch}, position {position}, backend {}", cfg.n_layers, gpu_core::backend_name());
    let ring = model::kv_tier::KvTier::parse(flag("--ring").as_deref().unwrap_or("bf16")).unwrap_or_else(|e| panic!("{e}"));
    println!("state per sequence: {:.1} MiB (ring {ring})", qwen35::model::slot_state_bytes(&cfg, ring) as f64 / (1u64 << 20) as f64);
    let t0 = std::time::Instant::now();
    let mut eng = RingEngine::from_source_tier(cfg, &*source, max_seq, batch, 0, &tier, ring);
    println!("load: {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(u) = eng.gpu().memory_usage() {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        println!("device live after load: {:.2} GiB ({:.2} GiB of it weights and tables, {:.2} GiB the slot pool)", gib(u.live), gib(u.live.saturating_sub(eng.state_bytes())), gib(eng.state_bytes()));
    }
    eng.set_decode_tape(a.iter().any(|x| x == "--tape"));
    let mut tables: Vec<BlockTable> = (0..batch).map(|_| BlockTable::new()).collect();
    let t0 = std::time::Instant::now();
    let profile_prefill = a.iter().any(|x| x == "--prefill-profile");
    if profile_prefill {
        eng.seed_synthetic(&mut tables[..1], 256); // warm-up: pipelines compile on first use
        eng.gpu().poll_wait();
        eng.release_table(&mut tables[0]);
        eng.gpu().set_kernel_timing(true);
        eng.gpu().reset_kernel_times();
    }
    eng.seed_synthetic(&mut tables, position);
    if profile_prefill {
        eng.gpu().poll_wait();
        let rows = eng.gpu().kernel_times();
        eng.gpu().set_kernel_timing(false);
        gpu_core::profile::print_device_table("prefill", t0.elapsed().as_secs_f64() * 1e3, rows);
        return;
    }
    let _ = eng.admit_greedy(&vec![0.0; eng.model().cfg.d_model as usize]);
    println!("seed to position {position}: {:.1} s ({:.0} tok/s prefill incl. warm-up)", t0.elapsed().as_secs_f64(), position as f64 / t0.elapsed().as_secs_f64());
    let mut input = vec![1u32; batch as usize];
    let mut step = |eng: &mut RingEngine| {
        let mut rows: Vec<&mut BlockTable> = tables.iter_mut().collect();
        input = eng.forward_batched_greedy(&mut rows, &input);
        eng.gpu().poll_wait();
    };
    step(&mut eng); // warm-up: pipelines, allocator growth
    let before = eng.gpu().stats();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        step(&mut eng);
        best = best.min(t.elapsed().as_secs_f64());
    }
    if let (Some(a), Some(b)) = (before, eng.gpu().stats()) {
        let n = u64::from(reps).max(1);
        println!("per step: {} submits, {} dispatches, {} readbacks, {} host writes, {} uniform allocs", (b.submits - a.submits) / n, (b.dispatches - a.dispatches) / n, (b.readbacks - a.readbacks) / n, (b.writes - a.writes) / n, (b.uniform_allocs - a.uniform_allocs) / n);
    }
    if let Some(u) = eng.gpu().memory_usage() {
        println!("device live after decoding: {:.2} GiB", u.live as f64 / (1u64 << 30) as f64);
    }
    println!("wall per step (best of {reps}): {:.2} ms  = {:.1} tok/s/stream, {:.0} tok/s aggregate", best * 1e3, 1.0 / best, batch as f64 / best);
    let recorded = eng.time_recorded_step(20);
    if let Some(ms) = recorded {
        println!("recorded layers replayed back to back, no host in the loop: {ms:.3} ms per step (head and readback excluded)");
    }
    // The recorded step as one program: each kernel timed from the previous one's completion, so the table carries no
    // launch latency and its total is what the card spends on a step.
    if let Some(rows) = eng.profile_recorded_step(5) {
        let device_ms: f64 = rows.iter().map(|r| r.1).sum();
        println!("recorded step: {device_ms:.3} ms device time over {} dispatches", rows.iter().map(|r| r.2).sum::<u64>());
        gpu_core::profile::print_device_table("recorded decode step", best * 1e3, Some(rows));
        return;
    }
    let timed = eng.gpu().set_kernel_timing(true);
    if timed {
        eng.gpu().reset_kernel_times();
    }
    step(&mut eng);
    let rows = eng.gpu().kernel_times();
    eng.gpu().set_kernel_timing(false);
    gpu_core::profile::print_device_table("decode step", best * 1e3, rows);
}
