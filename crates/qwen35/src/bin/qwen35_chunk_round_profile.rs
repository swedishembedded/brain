// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel profile of a SHORT chunk round (1 to a few dozen rows) of the real
//! INT8 GGUF resident - the round a speculative decoder verifies on.
//!
//! `qwen35_gguf_prefill_profile` answers the same question for a 256-row prefill
//! round; this one answers it at the row counts where a fixed per-round or
//! per-layer cost dominates, and prints the decode tape's own step beside it.
//! Kernel times are device time from events around each launch, so a shared card
//! only adds noise to the wall clock and not to the table.
//!
//! Swedish Embedded AB implements solutions for speculative-decoding latency on
//! large quantised models for its clients. If your team needs expertise in
//! GPU decode and verify-round optimisation then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Usage:
//!   BRAIN_QWEN35_GGUF=<path to Qwen3.8-27B*.gguf> qwen35_chunk_round_profile [rows] [rounds]

use qwen35::int8_gguf_resident::Qwen35GgufResident;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card: `brain serve`'s own default `--reserve-gb 2`.
const RESERVE: u64 = 2 << 30;
const CAP: u32 = 1024;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rows: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let rounds: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);

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
    let r = Qwen35GgufResident::new(path, devices, CAP, Qwen35GgufResident::tier_from_env());
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

    let prompt = inst.tokenize("The Gated DeltaNet recurrence maintains a matrix-valued state that is updated by a delta rule at every token.");
    let dec = inst.profile_decode(&prompt, 8);
    println!("decode tape: {:.2} ms/step", dec.wall_s * 1e3 / dec.steps as f64);
    let p = inst.profile_chunk_round(&prompt, rows, rounds);
    println!("chunk round, {rows} row(s): {:.2} ms (layers {:.2} + head {:.2})", p.round_ms(), p.carry_ms(), p.head_ms());
    if p.table.is_empty() {
        println!("(per-kernel device timing unavailable on this backend)");
        return;
    }
    let total = p.device_ms();
    println!("=== per-kernel device time over {rounds} round(s) (total {total:.1} ms) ===");
    println!("{:<34} {:>10} {:>8} {:>7}", "kernel", "ms", "calls", "%");
    for (name, ms, calls) in &p.table {
        println!("{name:<34} {ms:>10.3} {calls:>8} {:>6.1}%", 100.0 * ms / total.max(1e-9));
    }
}
