// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel DEVICE-time profile of the Qwen3.6-35B-A3B GGUF resident
//! (`qwen35moe::serve::Engine`) on one GPU: a decode step at a chosen batch and
//! synthetic context, or a cold prefill - on the real Q8_0 weights, through the
//! real engine.
//!
//! Swedish Embedded AB implements profiling-driven optimisation of sparse-MoE
//! inference for its clients. If your team needs expertise in finding what a
//! served mixture-of-experts model spends its time on, and in removing it, then
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Reports what the repo's kernel checklist asks for: the whole-pass rate (the
//! only figure an optimisation may be judged by), the per-kernel table (which
//! RANKS - each dispatch is timed on the device with timestamp queries, so a busy
//! host cannot distort it the way it stretches the wall clock), and the
//! weight-streaming roofline the pass is read against.
//!
//! Usage:
//!   BRAIN_QWEN35MOE_GGUF=<Q8_0.gguf> qwen35moe_decode_profile [passes]
//!
//! Environment:
//!   BRAIN_PROFILE_CONTEXT  synthetic decode context, tokens (default 1024)
//!   BRAIN_PROFILE_BATCH    sequences decoded together (default 1)
//!   BRAIN_PROFILE_PREFILL  profile a cold prefill of this many tokens instead
//!   BRAIN_PROFILE_CHUNK    prefill round size (default 256)
//!   BRAIN_QWEN35MOE_TIER   weight tier policy (default `i8`)
//!   BRAIN_QWEN35_KV        KV tier: `f32` (default), `bf16`, `int8`

use std::time::Instant;

use checkpoint::gguf::MmapGguf;
use qwen35moe::gguf_load;
use qwen35moe::serve::{Engine, EngineOptions, KernelProfile};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let passes: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let Ok(path) = std::env::var(gguf_load::GGUF_ENV) else {
        eprintln!("set {} to the Qwen3.6-35B-A3B Q8_0 GGUF", gguf_load::GGUF_ENV);
        std::process::exit(2);
    };
    let context = env_u32("BRAIN_PROFILE_CONTEXT", 1024);
    let batch = env_u32("BRAIN_PROFILE_BATCH", 1);
    let prefill = std::env::var("BRAIN_PROFILE_PREFILL").ok().and_then(|s| s.parse::<u32>().ok());
    let chunk = env_u32("BRAIN_PROFILE_CHUNK", 256);
    let tier = gguf_load::tier_from_env();
    let kv = gguf_load::kv_tier_from_env().unwrap_or_else(|e| panic!("{e}"));

    let capacity = prefill.unwrap_or(context) + 2 * passes + 8;
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let cfg = gguf_load::resident_config(&mg, capacity).unwrap_or_else(|e| panic!("{e}"));
    let src = gguf_load::source(&mg, &cfg).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let opts = EngineOptions::new(capacity, if prefill.is_some() { 1 } else { batch }).with_tier(tier.clone()).with_kv_tier(kv).with_prefill_chunk(chunk);
    let mut engine = Engine::from_source(cfg.clone(), &src, opts);
    println!("weight tier {}, kv {kv}, backend {}", tier.describe(), gpu_core::backend_name());
    println!("cold load {:.1} s, KV pool + recurrent state {:.2} GiB", t0.elapsed().as_secs_f64(), engine.kv_pool_bytes() as f64 / (1u64 << 30) as f64);

    let (profile, what, per_pass_tokens) = match prefill {
        Some(n) => (engine.profile_prefill(n, passes), format!("prefill of {n} tokens (rounds of {chunk})"), n as f64),
        None => (engine.profile_decode_at(batch, context, passes), format!("decode, batch {batch} at context {context}"), batch as f64),
    };
    let p = profile.unwrap_or_else(|e| panic!("{e}"));
    report(&p, &what, per_pass_tokens);
    if let Some((hits, misses, live)) = engine.gpu().step_cache_stats() {
        println!("  step cache: {hits} hits, {misses} misses, {live} entries");
    }
}

fn report(p: &KernelProfile, what: &str, tokens_per_pass: f64) {
    println!();
    println!("=== {what}: whole pass (production path) ===");
    let wall_per_pass = p.wall_s / p.passes as f64;
    println!(
        "  mean {:.3} ms/pass ({:.2} tok/s), median {:.3} ms ({:.2} tok/s), best {:.3} ms ({:.2} tok/s)",
        1e3 * wall_per_pass,
        tokens_per_pass / wall_per_pass,
        1e3 * p.median_pass_s,
        tokens_per_pass / p.median_pass_s,
        1e3 * p.best_pass_s,
        tokens_per_pass / p.best_pass_s
    );
    if p.host_run_s > 0.0 {
        println!(
            "  launching thread per pass: {:.3} ms running (host cost), {:.3} ms runnable but off-core (other tenants), {:.3} ms asleep (waiting on the device)",
            1e3 * p.host_run_s,
            1e3 * p.host_wait_s,
            1e3 * (wall_per_pass - p.host_run_s - p.host_wait_s).max(0.0)
        );
    }
    if let Some((submits, dispatches, writes, uniforms)) = p.host_ops_per_pass {
        println!("  host per pass: {submits:.1} submissions, {dispatches:.0} dispatches, {writes:.1} writes, {uniforms:.1} uniform blocks allocated");
    }
    if p.rows.is_empty() {
        println!("(per-kernel device timing unavailable on this backend - the whole-pass number above is the honest one)");
        return;
    }
    let total = p.device_ms();
    println!();
    println!("=== per-kernel device time ({} pass(es), total {total:.1} ms = {:.3} ms/pass = {:.2} tok/s device-timed) ===", p.passes, p.device_ms_per_pass(), 1e3 * tokens_per_pass / p.device_ms_per_pass());
    println!("{:<44} {:>10} {:>12} {:>8} {:>7}", "kernel", "ms", "ms/pass", "calls", "%");
    for (name, ms, calls) in p.rows.iter().take(24) {
        println!("{name:<44} {ms:>10.3} {:>12.3} {:>8} {:>6.1}%", ms / p.passes as f64, calls / p.passes as u64, 100.0 * ms / total.max(1e-9));
    }
    println!("  dispatches/pass: {}", p.rows.iter().map(|(_, _, c)| c).sum::<u64>() / p.passes as u64);
}
