// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models
// at high aggregate throughput on a fixed GPU budget for its clients. If your
// team needs expertise in LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Batch-size sweep of the REAL Qwen3.8-27B served from its released GGUF:
//! aggregate decode tokens per second and per-sequence latency as the number
//! of sequences decoding together grows, at a real context length.
//!
//! It drives the same `Qwen35GgufEngine` the server's continuous-batching
//! scheduler drives (`PagedDecoder::prefill` into a slot, then one
//! `forward_batched_greedy` step for all live sequences per iteration), so a
//! number here is a number `brain serve --qwen35-max-batch N` can reach, minus
//! the host-side streaming work per token.
//!
//! One activation at the largest batch; each row of the table then decodes
//! `batch` sequences, each prefilled with a real prompt of `--prompt-tokens`
//! tokens, for `--steps` timed steps after a warm-up step.
//!
//! Usage:
//!   qwen35_batch_sweep <Qwen3.8-27B*.gguf> [--ctx 4096] [--kv f32|bf16|int8|fp8]
//!       [--prompt-tokens 512] [--steps 24] [--batches 1,4,8,16,32,64]
//!       [--sampled] [--e2e 8,32] [--e2e-new 64]
//!
//! `--e2e` additionally serves that many concurrent requests (the prompt, `--e2e-new`
//! greedy tokens each) through the resident exactly as the server does
//! (`Instance::run_batch` over the generic scheduler), so the host-side work per
//! token (streaming, stop-string scanning) is in the number.
//!
//! Run it under the GPU wrapper with `--exclusive`; see `make bench/batch-sweep`.

use std::time::Instant;

use data::tokenizer::Tokenizer;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::int8_gguf_resident::Qwen35GgufResident;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card: `brain serve`'s own default `--reserve-gb 2`.
const RESERVE: u64 = 2 << 30;

const PROMPT: &str = include_str!("../../tests/fixtures/real_prompt.txt");

struct Args {
    gguf: String,
    ctx: u32,
    kv: KvTier,
    prompt_tokens: usize,
    steps: usize,
    batches: Vec<u32>,
    /// Step through the top-k head (what a sampling request uses) instead of the
    /// greedy argmax head.
    sampled: bool,
    /// Concurrent-request counts for the end-to-end rows (empty: none).
    e2e: Vec<u32>,
    e2e_new: u32,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut args = Args { gguf: String::new(), ctx: 4096, kv: KvTier::F32, prompt_tokens: 512, steps: 24, batches: vec![1, 4, 8, 16, 32, 64], sampled: false, e2e: Vec::new(), e2e_new: 64 };
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match flag.as_str() {
            "--ctx" => args.ctx = value("--ctx").parse().expect("--ctx N"),
            "--kv" => args.kv = KvTier::parse(&value("--kv")).unwrap_or_else(|e| panic!("{e}")),
            "--prompt-tokens" => args.prompt_tokens = value("--prompt-tokens").parse().expect("--prompt-tokens N"),
            "--sampled" => args.sampled = true,
            "--e2e" => args.e2e = value("--e2e").split(',').map(|s| s.trim().parse().expect("--e2e a,b,c")).collect(),
            "--e2e-new" => args.e2e_new = value("--e2e-new").parse().expect("--e2e-new N"),
            "--steps" => args.steps = value("--steps").parse().expect("--steps N"),
            "--batches" => args.batches = value("--batches").split(',').map(|s| s.trim().parse().expect("--batches a,b,c")).collect(),
            other if !other.starts_with("--") && args.gguf.is_empty() => args.gguf = other.to_string(),
            other => panic!("unknown argument {other}"),
        }
    }
    if args.gguf.is_empty() || args.batches.is_empty() {
        eprintln!("usage: qwen35_batch_sweep <Qwen3.8-27B*.gguf> [--ctx N] [--kv f32|bf16|int8|fp8] [--prompt-tokens N] [--steps N] [--batches a,b,c] [--sampled]");
        std::process::exit(2);
    }
    args
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn main() {
    let args = parse_args();
    let max_batch = args.batches.iter().chain(&args.e2e).copied().max().expect("non-empty");
    assert!(args.prompt_tokens as u64 + (args.steps as u64).max(args.e2e_new as u64) + 2 <= args.ctx as u64, "prompt + steps must fit --ctx");
    let devices = qwen35::int8_gguf_resident::budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }
    let tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
    let r = Qwen35GgufResident::new(args.gguf.clone(), devices, args.ctx, tier).with_kv_tier(args.kv).with_max_batch(max_batch);
    let cost = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new()));
    let placed: Vec<Device> = cost.devices().collect();
    if placed.is_empty() {
        eprintln!("ctx {} x max_batch {max_batch} ({}) does not fit the budgeted devices (see the placement message above)", args.ctx, args.kv.as_str());
        std::process::exit(1);
    }
    println!("qwen35 batch sweep: ctx {}, kv {}, prompt {} tokens, {} timed steps, activating max_batch {max_batch} on {:?}", args.ctx, args.kv.as_str(), args.prompt_tokens, args.steps, placed);
    let t0 = Instant::now();
    let (mut engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");
    println!("cold load: {:.1} s; resident decode state for {max_batch} slots: {:.2} GiB", t0.elapsed().as_secs_f64(), gib(engine.kv_pool_bytes()));

    // A real prompt of the requested length from the model's own tokenizer.
    let mut ids = tok.encode(PROMPT);
    while ids.len() < args.prompt_tokens {
        let more = ids.clone();
        ids.extend(more);
    }
    ids.truncate(args.prompt_tokens);

    println!();
    println!("{:>5} {:>12} {:>14} {:>13} {:>14} {:>12}", "batch", "agg tok/s", "ms/step (seq)", "tok/s per seq", "prefill s/seq", "KV+state GiB");
    for &b in &args.batches {
        let mut tables: Vec<BlockTable> = (0..b).map(|_| BlockTable::new()).collect();
        let t = Instant::now();
        let mut inputs: Vec<u32> = Vec::with_capacity(b as usize);
        for table in tables.iter_mut() {
            let hidden = engine.prefill(table, &ids);
            inputs.push(engine.admit_greedy(&hidden));
        }
        let prefill_s = t.elapsed().as_secs_f64() / b as f64;
        let step = |engine: &mut qwen35::int8_gguf_resident::Qwen35GgufEngine, tables: &mut Vec<BlockTable>, inputs: &mut Vec<u32>| {
            let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
            *inputs = if args.sampled {
                // The sampling head's candidate extraction; the best candidate
                // keeps the run deterministic without an RNG in the timing.
                let k = engine.topk_capacity();
                engine.forward_batched_topk(&mut refs, inputs, k).iter().map(|c| c[0].0).collect()
            } else {
                engine.forward_batched_greedy(&mut refs, inputs)
            };
        };
        step(&mut engine, &mut tables, &mut inputs); // warm-up: pipeline compile and graph capture
        let t = Instant::now();
        for _ in 0..args.steps {
            step(&mut engine, &mut tables, &mut inputs);
        }
        let wall = t.elapsed().as_secs_f64();
        let step_s = wall / args.steps as f64;
        println!(
            "{:>5} {:>12.1} {:>14.1} {:>13.2} {:>14.1} {:>12.2}",
            b,
            b as f64 / step_s,
            step_s * 1e3,
            1.0 / step_s,
            prefill_s,
            gib(engine.kv_pool_bytes()) * b as f64 / max_batch as f64
        );
        for table in tables.iter_mut() {
            engine.release_table(table);
        }
    }
    if let Some(stats) = engine.device_stats() {
        println!("\ndevice stats at exit: {stats:?}");
    }
    if args.e2e.is_empty() {
        return;
    }
    drop(engine);
    let text = tok.decode(&ids);
    let mut inst = r.activate_multi(&r.instance_key("generate", &capability::Invocation::new()), &placed).expect("activate the serving instance");
    println!("\nend to end through the resident (prompt {} tokens, {} new tokens each, greedy):", ids.len(), args.e2e_new);
    println!("{:>9} {:>14} {:>12} {:>12}", "requests", "generated tok", "wall s", "agg tok/s");
    for &n in &args.e2e {
        let invs: Vec<capability::Invocation> = (0..n)
            .map(|i| {
                capability::Invocation::new()
                    .set("prompt", serde_json::json!(text))
                    .set("chat", serde_json::json!(false))
                    .set("max_new", serde_json::json!(args.e2e_new))
                    .set("temp", serde_json::json!(0.0))
                    .set("seed", serde_json::json!(i))
            })
            .collect();
        let t = Instant::now();
        let results = inst.run_batch("generate", &invs, &mut |_, _| {});
        let wall = t.elapsed().as_secs_f64();
        let generated: i64 = results.iter().filter_map(|r| r.as_ref().ok()).filter_map(|o| o.outputs["completion_tokens"].as_i64()).sum();
        let failed = results.iter().filter(|r| r.is_err()).count();
        println!("{:>9} {:>14} {:>12.1} {:>12.1}{}", n, generated, wall, generated as f64 / wall, if failed > 0 { format!("  ({failed} FAILED)") } else { String::new() });
    }
}
