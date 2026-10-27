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
//! [--device gpu1] [--backend cuda]`. `synth` is the published shape with `<layers>` layers (default 8) on random
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
        eprintln!("usage: edgerock_decode_profile (<model dir> | synth[:layers]) [--batch N] [--position P] [--reps R] [--device gpu1]");
        std::process::exit(2);
    };
    let tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::BF16);
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
    println!("state per sequence: {:.1} MiB", qwen35::model::slot_state_bytes(&cfg) as f64 / (1u64 << 20) as f64);
    let t0 = std::time::Instant::now();
    let mut eng = RingEngine::from_source_tier(cfg, &*source, max_seq, batch, 0, &tier);
    println!("load: {:.1} s", t0.elapsed().as_secs_f64());
    let mut tables: Vec<BlockTable> = (0..batch).map(|_| BlockTable::new()).collect();
    let t0 = std::time::Instant::now();
    eng.seed_synthetic(&mut tables, position);
    let _ = eng.admit_greedy(&vec![0.0; eng.model().cfg.d_model as usize]);
    println!("seed to position {position}: {:.1} s ({:.0} tok/s prefill incl. warm-up)", t0.elapsed().as_secs_f64(), position as f64 / t0.elapsed().as_secs_f64());
    let mut input = vec![1u32; batch as usize];
    let mut step = |eng: &mut RingEngine| {
        let mut rows: Vec<&mut BlockTable> = tables.iter_mut().collect();
        input = eng.forward_batched_greedy(&mut rows, &input);
        eng.gpu().poll_wait();
    };
    step(&mut eng); // warm-up: pipelines, allocator growth
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        step(&mut eng);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("wall per step (best of {reps}): {:.2} ms  = {:.1} tok/s/stream, {:.0} tok/s aggregate", best * 1e3, 1.0 / best, batch as f64 / best);
    let timed = eng.gpu().set_kernel_timing(true);
    if timed {
        eng.gpu().reset_kernel_times();
    }
    step(&mut eng);
    let rows = eng.gpu().kernel_times();
    eng.gpu().set_kernel_timing(false);
    gpu_core::profile::print_device_table("decode step", best * 1e3, rows);
}
