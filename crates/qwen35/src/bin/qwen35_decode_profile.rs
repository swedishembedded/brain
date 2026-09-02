// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decode-mode per-kernel profile of the REAL two-card INT8 GGUF resident
//! (`qwen35::int8_gguf_resident`) - the `n = 1` steady-state token loop that
//! actually serves this model, at its real 64-layer shape, on real weights.
//!
//! This is the companion to `qwen35_bench`, and the split is deliberate:
//! `qwen35_bench` prices ONE layer at prefill widths on random weights, which
//! is the right shape for asking "is this kernel well written". It cannot
//! answer "where does a served token go", because at `T = 1` the whole model
//! is memory-bound on its own weights and the ranking is a different ranking.
//! Every number here comes from the real checkpoint through the real resident.
//!
//! Reports, per the optimization discipline in this repo's kernel checklist:
//! the whole-pass decode rate (the only figure an optimization may be judged
//! by), the per-kernel table (which RANKS, and whose total is an upper bound
//! because each dispatch is drained alone), and the weight-streaming roofline
//! the whole-pass number should be read against.
//!
//! Usage:
//!   BRAIN_QWEN35_GGUF=<path to Qwen3.8-27B*.gguf> qwen35_decode_profile [steps] [json-out]
//!
//! `steps` defaults to 8 profiled decode passes per region. `json-out`, when
//! given, writes a `brain perf`-shaped baseline artifact to that path.

use std::time::Instant;

use checkpoint::gguf::MmapGguf;
use qwen35::int8_gguf_resident::{layer_cost, resident_config, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// `prompt + max_new` capacity to build the caches at - the same figure
/// `tests/gguf_resident_real.rs` plans against, so the profile describes the
/// same instance that gate does.
const CAP: u32 = 512;

/// Bytes kept free per card: `brain serve`'s own default `--reserve-gb 2`.
const RESERVE: u64 = 2 << 30;

fn real_devices() -> Vec<(Device, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let steps: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let json_out = a.get(2).cloned();

    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        eprintln!("set BRAIN_QWEN35_GGUF to a downloaded Qwen3.8-27B*.gguf");
        std::process::exit(2);
    };
    let devices = real_devices();
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }

    let tier = Qwen35GgufResident::tier_from_env();
    println!("weight tier: {}", tier.describe());

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let cfg = resident_config(&mg, CAP).expect("resident_config");
    let cost = layer_cost(&cfg, CAP, &tier);
    let weight_bytes = cost.total();
    drop(mg);

    let r = Qwen35GgufResident::new(path, devices.clone(), CAP, tier.clone());
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    println!("qwen35 decode profile: {} layers, {} stage(s) over {:?}", cfg.n_layers, placed.len(), placed);

    let t0 = Instant::now();
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");
    println!("cold load: {:.1} s", t0.elapsed().as_secs_f64());

    // A real prompt through the model's own embedded tokenizer: the profiled
    // steps then run at a realistic position with a realistic KV occupancy,
    // rather than at position 0 where the GQA layers attend over nothing.
    let prompt = inst.tokenize("Give one short sentence explaining what a Kalman filter does.");
    println!("prompt: {} tokens, profiling {steps} decode step(s) per region", prompt.len());

    let p = inst.profile_decode(&prompt, steps);

    println!();
    println!("=== whole pass (production flush path) ===");
    println!("  decode              {:.3} tok/s  ({:.1} ms/token over {} step(s))", p.tok_per_s(), 1e3 * p.wall_s / p.steps as f64, p.steps);
    println!("  timed-region wall   {:.3} tok/s  (timestamp queries armed - inflated, for the table only)", p.steps as f64 / p.timed_wall_s.max(1e-9));

    // The roofline this whole-pass number is judged against. Decode reads every
    // resident byte once per token; the stages run in SERIES (stage k+1 needs
    // stage k's residual), so the model's own bytes, not one card's, set the
    // floor - which is exactly why "add a card" does not make this faster.
    println!();
    println!("=== weight-streaming roof ===");
    println!("  resident bytes      {:.2} GiB (read once per token, stages in series)", weight_bytes as f64 / (1u64 << 30) as f64);
    println!("  measured            {:.1} GB/s effective", weight_bytes as f64 / 1e9 / (p.wall_s / p.steps as f64));

    println!();
    if p.rows.is_empty() {
        println!("(per-kernel device timing unavailable on this backend - the whole-pass number above is the honest one)");
    } else {
        let total = p.device_ms();
        println!("=== per-kernel device time ({steps} step(s), all stages merged, total {total:.1} ms) ===");
        println!("{:<28} {:>10} {:>12} {:>8} {:>7}", "kernel", "ms", "ms/token", "calls", "%");
        for (name, ms, calls) in &p.rows {
            println!("{name:<28} {ms:>10.3} {:>12.3} {calls:>8} {:>6.1}%", ms / steps as f64, 100.0 * ms / total.max(1e-9));
        }
        println!("  calls/token: {}", p.rows.iter().map(|(_, _, c)| c).sum::<u64>() / steps as u64);
    }

    if let Some(out) = json_out {
        let doc = baseline_json(&p, weight_bytes, prompt.len() as u32, &placed, &tier);
        std::fs::write(&out, serde_json::to_string_pretty(&doc).expect("serialize")).unwrap_or_else(|e| panic!("write {out}: {e}"));
        println!("\nwrote baseline artifact: {out}");
    }
}

/// The measured run as a `brain perf`-shaped artifact: the same top-level
/// sections `crates/perf`'s own runs emit, with every field this target cannot
/// honestly fill left `null` rather than zeroed. A zero in a perf artifact is a
/// measurement; a null is "not measured here", and `brain perf compare`
/// distinguishes them. `target` carries the tier so `qwen35:i8-gguf-resident`
/// and `qwen35:q4-gguf-resident` artifacts never get compared as if they
/// measured the same thing.
fn baseline_json(
    p: &qwen35::int8_gguf_resident::DecodeProfile,
    weight_bytes: u64,
    prompt_tokens: u32,
    placed: &[Device],
    tier: &model::ops::TierPolicy,
) -> serde_json::Value {
    let ms_per_token = 1e3 * p.wall_s / p.steps as f64;
    serde_json::json!({
        "scenario": "latency",
        "target": format!("qwen35:{}-gguf-resident", tier.describe()),
        "notes": format!("weight tier: {}", tier.describe()),
        "workload": "decode",
        "best_of_n": 1,
        "env": {
            "backend": "wgpu",
            "build": "release",
            "device": placed.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>().join(" + "),
            "cpu": { "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0) },
        },
        "performance": {
            "completed": p.steps,
            "requests": p.steps,
            "failed": 0,
            "errors": [],
            "output_artifacts_per_s": p.tok_per_s(),
            "wall_s": p.wall_s,
            "tpoa_ms": { "n": p.steps, "mean": ms_per_token, "p50": ms_per_token, "p95": null, "p99": null, "p999": null, "min": null, "max": null },
        },
        "memory": {
            "peak_device_mb": weight_bytes / (1 << 20),
            "bytes_moved_per_artifact": weight_bytes,
        },
        "notes": format!(
            "qwen35 int8 GGUF two-card resident, n=1 decode, {} prompt tokens of warm-up, {} profiled steps; \
             per-kernel table alongside. Correctness of the TEXT is separately gated and currently red \
             (see the crate's own gguf_resident_real.rs) - this artifact measures throughput only.",
            prompt_tokens, p.steps
        ),
        "correctness": { "gate": null, "passed": null },
        "per_kernel": p.rows.iter().map(|(n, ms, c)| serde_json::json!({
            "kernel": n, "ms_total": ms, "ms_per_token": ms / p.steps as f64, "calls": c,
        })).collect::<Vec<_>>(),
    })
}
