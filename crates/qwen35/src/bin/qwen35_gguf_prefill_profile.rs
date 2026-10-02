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
//!   BRAIN_QWEN35_GGUF=<path to Qwen3.8-27B*.gguf> qwen35_gguf_prefill_profile [depth] [rows] [rounds]
//!
//! `depth` (default 256) prompt tokens establish the KV depth first, `rows`
//! (default 256) is the round width, `rounds` (default 2) the rounds per
//! measured region.

use checkpoint::gguf::MmapGguf;
use qwen35::int8_gguf_resident::{resident_config, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card: `brain serve`'s own default `--reserve-gb 2`.
const RESERVE: u64 = 2 << 30;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let depth: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let rows: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let rounds: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let cap = depth + 2 * rows * rounds;

    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        eprintln!("set BRAIN_QWEN35_GGUF to a downloaded Qwen3.8-27B*.gguf");
        std::process::exit(2);
    };
    let devices: Vec<(Device, u64)> = gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect();
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }

    let tier = Qwen35GgufResident::tier_from_env();
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let cfg = resident_config(&mg, cap).expect("resident_config");
    drop(mg);
    let r = Qwen35GgufResident::new(path, devices, cap, tier.clone());
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    println!("qwen35 prefill profile: {} layers, weight tier {}, {} stage(s), backend {}", cfg.n_layers, tier.describe(), placed.len(), gpu_core::backend_name());
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

    let seed = inst.tokenize("The quick brown fox jumps over the lazy dog while a kalman filter estimates the state of a noisy system. ");
    let prompt: Vec<u32> = seed.iter().cycle().take(depth as usize).copied().collect();
    let p = inst.profile_chunk_round(&prompt, rows, rounds);

    println!();
    println!("depth {depth}, {rows} rows x {rounds} rounds");
    println!("  layer stack : {:9.2} ms/round  {:9.1} tok/s", p.carry_ms(), rows as f64 * 1e3 / p.carry_ms());
    println!("  head        : {:9.2} ms/round", p.head_ms());
    println!("  whole round : {:9.2} ms/round  {:9.1} tok/s", p.round_ms(), rows as f64 * 1e3 / p.round_ms());
    if p.table.is_empty() {
        println!("(per-kernel device timing unavailable on this backend)");
        return;
    }
    let total = p.device_ms();
    println!();
    println!("=== per-kernel device time, layer stack ({rounds} round(s), total {total:.1} ms) ===");
    println!("{:<34} {:>10} {:>12} {:>8} {:>7}", "kernel", "ms", "ms/round", "calls", "%");
    for (name, ms, calls) in &p.table {
        println!("{name:<34} {ms:>10.3} {:>12.3} {calls:>8} {:>6.1}%", ms / rounds as f64, 100.0 * ms / total.max(1e-9));
    }
}
