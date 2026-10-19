// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefill-mode per-kernel profile of the REAL INT8 GGUF resident
//! (`qwen35::int8_gguf_resident`): the chunk tape (`MAX_PREFILL_TOKENS`-row
//! rounds) over every layer of the real checkpoint, with device-timed
//! per-kernel totals.
//!
//! Companion to `qwen35_decode_profile` (the `n = 1` token loop) and to
//! `qwen35_prefill_profile` (synthetic fp32 weights at a reduced depth). This
//! one answers "where does a prompt token go on the model that is served":
//! the table is device time from events around each launch, so it is immune
//! to the host load a shared box has, and it ranks the kernels a prefill
//! optimisation has to move.
//!
//! Swedish Embedded AB implements solutions for time-to-first-token on large
//! quantised models for its clients. If your team needs expertise in
//! profiling and optimising GPU prefill throughput then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Usage:
//!   qwen35_gguf_prefill_profile <path to Qwen3.8-27B*.gguf> [depth] [rows] [rounds]
//!       [--device gpu0,gpu1] [--backend wgpu|vulkan|cuda] [--kv f32|bf16|int8] [--flash-prefill v1|v2]
//!
//! `depth` (default 256) prompt tokens establish the KV depth first, `rows`
//! (default 256) is the round width, `rounds` (default 4) the rounds per
//! measured region; the fastest of them is reported.

use checkpoint::gguf::MmapGguf;
use qwen35::int8_gguf_resident::{resident_config, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card: `brain serve`'s own default `--reserve-gb 2`.
const RESERVE: u64 = 2 << 30;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let depth: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let rows: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let rounds: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let cap = depth + 2 * rows * rounds;
    let flag = |name: &str| a.iter().position(|arg| arg == name).and_then(|i| a.get(i + 1)).cloned();
    // The head_dim 256 prefill attention over a bf16 or int8 cache: version 2 by default, `v1` for the original kernel.
    match flag("--flash-prefill").as_deref() {
        None | Some("v2") => {}
        Some("v1") => gpu_core::provider::cuda::set_flash_prefill_v2(false),
        Some(other) => panic!("--flash-prefill takes v1 or v2, not {other}"),
    }
    let set = gpu_core::select_compute(flag("--device").as_deref().unwrap_or("gpu"), flag("--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));

    let Some(path) = a.get(1).filter(|s| s.ends_with(".gguf") || std::path::Path::new(s).is_file()).cloned() else {
        eprintln!("usage: qwen35_gguf_prefill_profile <path to Qwen3.8-27B*.gguf> [depth] [rows] [rounds]");
        std::process::exit(2);
    };
    let devices = qwen35::int8_gguf_resident::budgeted_gpus(&set.gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }

    let tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let cfg = resident_config(&mg, cap).expect("resident_config");
    drop(mg);
    let kv = model::kv_tier::KvTier::parse(flag("--kv").as_deref().unwrap_or("f32")).unwrap_or_else(|e| panic!("{e}"));
    let r = Qwen35GgufResident::new(path, devices, cap, tier.clone()).with_kv_tier(kv);
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    println!("qwen35 prefill profile: {} layers, weight tier {}, kv {kv}, {} stage(s), backend {}", cfg.n_layers, tier.describe(), placed.len(), gpu_core::backend_name());
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

    // A/B switch: `BRAIN_PREFILL_ARENA=0` runs the rounds without the scratch arena
    // (and so without its per-layer drain), `1` forces it on.
    match std::env::var("BRAIN_PREFILL_ARENA").as_deref() {
        Ok("0") => inst.set_chunk_arena_min_rows(u32::MAX),
        Ok("1") => inst.set_chunk_arena_min_rows(1),
        _ => {}
    }

    let seed = inst.tokenize("The quick brown fox jumps over the lazy dog while a kalman filter estimates the state of a noisy system. ");
    let prompt: Vec<u32> = seed.iter().cycle().take(depth as usize).copied().collect();
    let p = inst.profile_prefill_rounds(&prompt, rows, rounds);

    println!();
    println!("depth {depth}, {rows} rows, {rounds} round(s) per region - fastest round reported (a shared card only ever adds time)");
    let walls: Vec<String> = p.round_s.iter().map(|s| format!("{:.0}", s * 1e3)).collect();
    println!("  whole rounds (ms)   : {}", walls.join(" "));
    println!("  best round          : {:9.2} ms   {:9.1} tok/s", p.best_round_s() * 1e3, p.best_tok_per_s());
    if p.table.is_empty() {
        println!("(per-kernel device timing unavailable on this backend)");
        return;
    }
    let total = p.device_ms();
    println!();
    println!("=== per-kernel device time, layer stack, least-interfered round (total {total:.1} ms; launches serialised by the timers) ===");
    println!("{:<34} {:>10} {:>8} {:>7}", "kernel", "ms", "calls", "%");
    for (name, ms, calls) in &p.table {
        println!("{name:<34} {ms:>10.3} {calls:>8} {:>6.1}%", 100.0 * ms / total.max(1e-9));
    }
}
