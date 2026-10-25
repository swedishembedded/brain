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
//!   qwen35_decode_profile <path to Qwen3.8-27B*.gguf> [steps] [json-out]
//!       [--device gpu0,gpu1] [--backend wgpu|vulkan|cuda] [--weight-format SPEC]
//!
//! `steps` defaults to 8 profiled decode passes per region. `--weight-format`
//! takes `brain serve`'s weight-tier grammar (default: each tensor as the checkpoint
//! stores it, `held_tiers`). `json-out`, when
//! given, writes a `brain perf`-shaped baseline artifact to that path.
//! `--device` and `--backend` take `brain serve`'s grammar and default to every
//! GPU on the backend the machine prefers.
//!
//! **Batches and long contexts** - `--batch <n>` or `--batches a,b,c` and/or
//! `--context <tokens>` switch to a SYNTHETIC context (fresh caches, decode
//! work at a real position, as `brain perf run longctx` does) with `n` rows
//! (default 1) at that context (default 512); the KV cache storage comes from
//! `--kv <f32|bf16|int8|fp8>` (default f32). One activation serves every batch
//! in the list, so a sweep pays the model load once.
//!
//! The report is the production GREEDY tape's wall time per step (the only
//! figure an optimization may be judged by), the per-kernel DEVICE time
//! (which RANKS, and is an upper bound - each dispatch is drained alone), the
//! step rolled up into phases with the head measured on its own, and the
//! weight bandwidth that step achieved. `--kernels` prints the full per-kernel
//! table behind the phase rollup.

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

fn real_devices(allowed: &[u32]) -> Vec<(Device, u64)> {
    qwen35::int8_gguf_resident::budgeted_gpus(allowed, RESERVE)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let steps: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let json_out = a.get(3).filter(|s| !s.starts_with("--")).cloned();
    let mut kv: Option<model::kv_tier::KvTier> = None;
    let flag = |name: &str| a.iter().position(|arg| arg == name).and_then(|i| a.get(i + 1)).cloned();
    if let Some(v) = flag("--kv") {
        kv = Some(model::kv_tier::KvTier::parse(&v).unwrap_or_else(|e| panic!("{e}")));
    }
    let batch: Option<u32> = flag("--batch").map(|v| v.parse().unwrap_or_else(|_| panic!("--batch takes a count, got {v:?}")));
    let batches: Option<Vec<u32>> = flag("--batches").map(|v| {
        let list: Vec<u32> = v.split(',').map(|s| s.trim().parse().unwrap_or_else(|_| panic!("--batches takes counts, got {v:?}"))).collect();
        assert!(!list.is_empty(), "--batches needs at least one count");
        list
    });
    let batches = batches.or_else(|| batch.map(|b| vec![b]));
    let context: Option<u32> = flag("--context").map(|v| v.parse().unwrap_or_else(|_| panic!("--context takes a token count, got {v:?}")));
    let set = gpu_core::select_compute(flag("--device").as_deref().unwrap_or("gpu"), flag("--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));

    let Some(path) = a.get(1).filter(|s| s.ends_with(".gguf") || std::path::Path::new(s).is_file()).cloned() else {
        eprintln!("usage: qwen35_decode_profile <path to Qwen3.8-27B*.gguf> [steps] [json-out] [--kv f32|bf16|int8|fp8] [--weight-format int8|q4|...] [--batches 1,8,12] [--context N] [--kernels]");
        std::process::exit(2);
    };
    let devices = real_devices(&set.gpus);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }

    let mut policy = model::policy::ServePolicy::default();
    if let Some(v) = flag("--weight-format") {
        policy.set("--weight-format", &v).unwrap_or_else(|e| panic!("{e}"));
    }
    // Unset, every tensor as the checkpoint stores it - `brain serve`'s default.
    let tier = policy.weights.unwrap_or_else(|| qwen35::int8_gguf_resident::held_tiers_at(&path).unwrap_or_else(|e| panic!("{e}")));
    println!("weight tier: {}", tier.summary());
    if batches.is_some() || context.is_some() {
        let kernels = a.iter().any(|s| s == "--kernels");
        return synthetic_sweep(&path, devices, &tier, kv, context.unwrap_or(CAP), &batches.unwrap_or_else(|| vec![1]), steps, kernels);
    }

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let cfg = resident_config(&mg, CAP).expect("resident_config");
    let cost = layer_cost(&cfg, CAP, &tier, model::kv_tier::KvTier::F32, 1);
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
    println!(
        "  per token           best {:.1} / median {:.1} / worst {:.1} ms",
        p.step_ms_at(0.0),
        p.step_ms_at(0.5),
        p.step_ms_at(1.0)
    );
    println!(
        "  host calls/token    {:.1} cuMemAlloc, {:.1} individual launches, {:.1} graph replays",
        p.alloc_calls as f64 / p.steps as f64,
        p.host_launches as f64 / p.steps as f64,
        p.graph_replays as f64 / p.steps as f64
    );
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

/// Bytes one decode step streams from the weights: every layer's weights plus
/// the head, read once for the whole batch. The per-sequence decode state is
/// excluded - it scales with the batch and is not what batching amortises.
fn weights_streamed_per_step(path: &str, context: u32, tier: &model::ops::TierPolicy) -> u64 {
    let cfg = resident_config(&MmapGguf::open(path).expect("open the checkpoint"), context).expect("resident_config");
    let head = layer_cost(&cfg, context, tier, model::kv_tier::KvTier::F32, 1).head;
    head + cfg.layer_types().into_iter().map(|ty| cfg.layer_weight_bytes(ty, tier)).sum::<u64>()
}

/// The head's resident bytes, the part of [`weights_streamed_per_step`] the
/// separately-profiled head reads.
fn head_bytes(path: &str, context: u32, tier: &model::ops::TierPolicy) -> u64 {
    let cfg = resident_config(&MmapGguf::open(path).expect("open the checkpoint"), context).expect("resident_config");
    layer_cost(&cfg, context, tier, model::kv_tier::KvTier::F32, 1).head
}

/// One batch's measured decode step, as a row of the sweep table.
struct BatchStep {
    batch: u32,
    /// MEDIAN wall milliseconds of one step on the production greedy tape. The
    /// median, not the mean: the first measured passes of each batch size pay
    /// graph capture, which no served token pays, and the min/max are printed
    /// beside it so the spread is visible rather than hidden.
    step_ms: f64,
    best_ms: f64,
    worst_ms: f64,
    /// Summed device time per step (a per-dispatch-timed upper bound; see
    /// [`qwen35::int8_gguf_resident::DecodeProfile`]).
    device_ms: f64,
    /// The head alone, measured on its own (ms/step): the epilogue the merged
    /// table cannot split out, because it dispatches the same matmul kernel
    /// the layer linears do.
    head_ms: f64,
    /// Host-side calls per step: allocations, individual launches and graph
    /// replays. A steady-state step should show no allocation, no individual
    /// launch and one replay; anything else is a host- or graph-bound step and
    /// is the first thing to read when a batch is slower than its kernel table
    /// explains.
    allocs_per_step: f64,
    launches_per_step: f64,
    replays_per_step: f64,
    phases: Vec<qwen35::int8_gguf_resident::PhaseCost>,
    /// `(kernel, ms/step, calls/step)` of the whole step, for the full table.
    rows: Vec<(String, f64, u64)>,
}

/// Batches decoding at a synthetic `context`: per-step wall and device time,
/// effective weight bandwidth, and the step split into phases with the head
/// measured on its own. One activation, one table per batch.
fn synthetic_sweep(path: &str, devices: Vec<(Device, u64)>, tier: &model::ops::TierPolicy, kv: Option<model::kv_tier::KvTier>, context: u32, batches: &[u32], steps: u32, kernels: bool) {
    let kv = kv.unwrap_or(model::kv_tier::KvTier::F32);
    let max_batch = batches.iter().copied().max().expect("a non-empty batch list");
    let r = Qwen35GgufResident::new(path.to_string(), devices, context, tier.clone()).with_kv_tier(kv).with_max_batch(max_batch);
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    if placed.is_empty() {
        eprintln!("batch {max_batch} x {context} tokens with a {kv} KV cache does not fit these GPUs");
        std::process::exit(3);
    }
    let t0 = Instant::now();
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");
    println!("synthetic context {context}, batches {batches:?}, kv {kv}: cold load {:.1} s over {:?}", t0.elapsed().as_secs_f64(), placed);

    let weight_bytes = weights_streamed_per_step(path, context, tier);
    let head_weight_bytes = head_bytes(path, context, tier);
    let position = context - 2 * steps - 2;
    let mut measured: Vec<BatchStep> = Vec::with_capacity(batches.len());
    for &batch in batches {
        let p = inst.profile_decode_batch_greedy_at(batch, position, steps).unwrap_or_else(|e| panic!("{e}"));
        let h = inst.profile_head_at(batch, position, steps).unwrap_or_else(|e| panic!("{e}"));
        measured.push(BatchStep {
            batch,
            step_ms: p.step_ms_at(0.5),
            best_ms: p.step_ms_at(0.0),
            worst_ms: p.step_ms_at(1.0),
            device_ms: p.device_ms() / p.steps as f64,
            head_ms: h.step_ms_at(0.5),
            allocs_per_step: p.alloc_calls as f64 / p.steps as f64,
            launches_per_step: p.host_launches as f64 / p.steps as f64,
            replays_per_step: p.graph_replays as f64 / p.steps as f64,
            phases: qwen35::int8_gguf_resident::decode_phases(&p.rows),
            rows: p.rows.iter().map(|(name, ms, calls)| (name.clone(), ms / p.steps as f64, calls / p.steps as u64)).collect(),
        });
        // The head's own table names the kernels, so a change there is visible.
        let head_kernels: Vec<String> = h.rows.iter().map(|(name, ms, calls)| format!("{name} {:.3}/{:.0}", ms / h.steps as f64, *calls as f64 / h.steps as f64)).collect();
        if let Some(last) = measured.last_mut() {
            last.rows.push((format!("# head: {}", head_kernels.join(", ")), 0.0, 0));
        }
    }

    println!();
    println!("backend {}; {steps} profiled step(s) per batch; weights per step {:.2} GiB (head {:.2} GiB)", gpu_core::backend_name(), weight_bytes as f64 / (1u64 << 30) as f64, head_weight_bytes as f64 / (1u64 << 30) as f64);
    println!();
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12} {:>14} {:>9} {:>9}",
        "batch", "ms/step", "best ms", "worst ms", "agg tok/s", "tok/s/stream", "dev ms/step", "GB/s weights", "head ms/step", "launch/step", "replay/step", "calls/step"
    );
    for m in &measured {
        println!(
            "{:>5} {:>10.2} {:>10.2} {:>10.2} {:>10.1} {:>10.1} {:>12.2} {:>12.0} {:>12.2} {:>14.1} {:>9.2} {:>9.0}",
            m.batch,
            m.step_ms,
            m.best_ms,
            m.worst_ms,
            m.batch as f64 * 1e3 / m.step_ms,
            1e3 / m.step_ms,
            m.device_ms,
            weight_bytes as f64 / 1e6 / m.step_ms,
            m.head_ms,
            m.launches_per_step,
            m.replays_per_step,
            m.phases.iter().map(|p| p.calls).sum::<u64>() as f64 / steps as f64
        );
    }

    // The step split into phases. `linear` includes the head (same kernel
    // family), so the layer-only share is derived by subtraction - the head is
    // measured on its own above, on the production greedy tape.
    let layer_weight_bytes = weight_bytes.saturating_sub(head_weight_bytes);
    for m in &measured {
        if m.phases.is_empty() {
            println!("\nbatch {}: (per-kernel device timing unavailable on this backend)", m.batch);
            continue;
        }
        println!();
        println!(
            "=== batch {}: phases ({steps} step(s), device time {:.2} ms/step; per step {:.1} alloc / {:.1} launch / {:.2} replay) ===",
            m.batch, m.device_ms, m.allocs_per_step, m.launches_per_step, m.replays_per_step
        );
        println!("{:<24} {:>10} {:>8} {:>10} {:>12}", "phase", "ms/step", "%dev", "calls/step", "GB/s weights");
        for ph in &m.phases {
            let ms = ph.ms_per_step(steps);
            let gbs = match ph.phase {
                qwen35::int8_gguf_resident::DecodePhase::Linear => weight_bytes as f64 / 1e6 / ms.max(1e-9),
                _ => f64::NAN,
            };
            println!(
                "{:<24} {:>10.2} {:>7.1}% {:>10.0} {:>12}",
                ph.phase.label(),
                ms,
                100.0 * ms / m.device_ms.max(1e-9),
                ph.calls as f64 / steps as f64,
                if gbs.is_finite() { format!("{gbs:.0}") } else { "-".to_string() }
            );
        }
        let linear_ms = m.phases.iter().find(|p| p.phase == qwen35::int8_gguf_resident::DecodePhase::Linear).map_or(0.0, |p| p.ms_per_step(steps));
        let layer_linear_ms = linear_ms - m.head_ms;
        if layer_linear_ms > 0.0 {
            println!(
                "{:<24} {:>10.2} {:>7.1}% {:>10} {:>12.0}",
                "  of which layers",
                layer_linear_ms,
                100.0 * layer_linear_ms / m.device_ms.max(1e-9),
                "-",
                layer_weight_bytes as f64 / 1e6 / layer_linear_ms
            );
        }
        println!("(linear includes the head; layers = linear - head, measured apart)");
        if kernels {
            println!("\n{:<44} {:>10} {:>8}", "kernel", "ms/step", "calls/step");
            for (name, ms, calls) in &m.rows {
                println!("{name:<44} {ms:>10.3} {calls:>8}");
            }
        }
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
        "target": format!("qwen35:{}-gguf-resident", tier.summary()),
        "notes": format!("weight tier: {}", tier.summary()),
        "workload": "decode",
        "best_of_n": 1,
        "env": {
            "backend": gpu_core::backend_name(),
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
