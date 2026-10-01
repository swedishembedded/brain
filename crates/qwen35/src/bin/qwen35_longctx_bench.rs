// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decode throughput of the real Qwen3.8-27B GGUF resident at a long context,
//! swept over batch size until the model no longer fits in GPU memory.
//!
//! Swedish Embedded AB implements long-context inference serving and the
//! measurement that shows what a given GPU can really sustain for its clients.
//! If your team needs expertise in sizing and tuning LLM serving on a specific
//! accelerator then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! GPU only, by construction: the resident is built from GPU devices alone, and
//! a batch size whose weights plus per-sequence caches do not fit is reported as
//! the out-of-memory boundary, never spilled to host memory.
//!
//! **What is measured, and what is not.** Each row is one real batched decode
//! step - every weight read, every GDN state update, every paged-attention pass
//! over the context - at a position near the end of a `--ctx`-token context.
//! The context itself is *synthetic*: the caches are fresh (zeros), because
//! prefilling `ctx` tokens for every sequence of a batch would take far longer
//! than the measurement, and decode throughput does not depend on what the
//! cached keys and values hold, only on how many there are. The logits are
//! therefore not meaningful and are not checked. Real prefill speed is a
//! separate figure (`qwen35_prefill_profile`).
//!
//! Usage:
//!   BRAIN_QWEN35_GGUF=<Qwen3.8-27B GGUF> qwen35_longctx_bench \
//!       [--ctx 131072] [--ladder 1,2,3,4] [--steps 8] [--position N] [--json out.json]
//!
//! The weight tier comes from `BRAIN_QWEN35_GGUF_TIER` (default INT8).

use std::time::Instant;

use capability::Invocation;
use checkpoint::gguf::MmapGguf;
use qwen35::int8_gguf_resident::{layer_cost, resident_config, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card for the driver and for activations the placement
/// model does not count: `brain serve`'s own default reserve.
const RESERVE: u64 = 2 << 30;

struct Opts {
    ctx: u32,
    ladder: Vec<u32>,
    steps: u32,
    position: Option<u32>,
    json: Option<String>,
}

fn parse_args() -> Opts {
    let mut o = Opts { ctx: 131_072, ladder: vec![1, 2, 3, 4], steps: 8, position: None, json: None };
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let need = |i: usize, flag: &str| -> String {
        a.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(2);
        })
    };
    while i < a.len() {
        match a[i].as_str() {
            "--ctx" => o.ctx = need(i, "--ctx").parse().expect("--ctx: an integer"),
            "--ladder" => o.ladder = need(i, "--ladder").split(',').map(|s| s.trim().parse().expect("--ladder: integers")).collect(),
            "--steps" => o.steps = need(i, "--steps").parse().expect("--steps: an integer"),
            "--position" => o.position = Some(need(i, "--position").parse().expect("--position: an integer")),
            "--json" => o.json = Some(need(i, "--json")),
            other => {
                eprintln!("unknown argument {other:?}\nusage: qwen35_longctx_bench [--ctx N] [--ladder a,b,c] [--steps N] [--position N] [--json FILE]");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    o
}

fn gpu_devices() -> Vec<(Device, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect()
}

/// Device memory in use right now, in MiB, as the driver's own tool reports it
/// (the same source the capacity probe uses), or `None` where it is absent.
fn device_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8(out.stdout).ok()?.lines().next()?.trim().parse().ok()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn main() {
    let o = parse_args();
    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        eprintln!("set BRAIN_QWEN35_GGUF to a downloaded Qwen3.8-27B*.gguf");
        std::process::exit(2);
    };
    let devices = gpu_devices();
    if devices.is_empty() {
        eprintln!("no GPU with queryable memory: this benchmark is GPU-only and has no host fallback");
        std::process::exit(2);
    }
    let tier = Qwen35GgufResident::tier_from_env();
    let position = o.position.unwrap_or(o.ctx.saturating_sub(o.steps + 8));
    let backend = gpu_core::backend_name();
    println!("backend {backend}, weight tier {}, context capacity {} tokens, decode position {position}", tier.describe(), o.ctx);
    println!("devices: {:?} (GPU only, no host offload)", devices.iter().map(|(d, c)| format!("{d:?} {:.1} GiB usable", gib(*c))).collect::<Vec<_>>());
    println!("context is SYNTHETIC (fresh caches): decode work is that of a real {position}-token context, logits are not meaningful\n");

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let cfg = resident_config(&mg, o.ctx).expect("resident_config");
    drop(mg);

    println!("{:>5} {:>10} {:>10} {:>11} {:>11} {:>13} {:>13}", "batch", "plan GiB", "used GiB", "ms/step", "tok/s total", "tok/s/stream", "kv GiB/seq");
    let mut rows = Vec::new();
    let mut boundary: Option<serde_json::Value> = None;
    for &batch in &o.ladder {
        let cost = layer_cost(&cfg, o.ctx, &tier, batch);
        let planned = cost.total();
        let kv_per_seq: u64 = cfg.layer_types().into_iter().map(|ty| cfg.layer_decode_state_bytes(ty, o.ctx)).sum();
        let r = Qwen35GgufResident::new(path.clone(), devices.clone(), o.ctx, tier.clone()).with_max_batch(batch);
        let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &Invocation::new())).devices().collect();
        if placed.is_empty() {
            println!("{batch:>5} {:>10.1} {:>10} -- does not fit in GPU memory (needs {:.1} GiB of weights and caches): out-of-memory boundary", gib(planned), "-", gib(planned));
            boundary = Some(serde_json::json!({ "batch": batch, "needed_bytes": planned, "usable_bytes": devices.iter().map(|(_, c)| c).sum::<u64>() }));
            break;
        }
        let inst = match r.activate_owned(&placed) {
            Ok(i) => i,
            Err(e) => {
                println!("{batch:>5} {:>10.1} {:>10} -- failed to load: {e}", gib(planned), "-");
                boundary = Some(serde_json::json!({ "batch": batch, "load_error": e, "needed_bytes": planned }));
                break;
            }
        };
        let tokens = vec![1u32; batch as usize];
        let mut times = Vec::new();
        let result = (|| -> Result<(), String> {
            // Two warm-up steps: kernel compilation and graph capture are not
            // decode time.
            for w in 0..2u32 {
                inst.decode_batch_at(&tokens, &vec![position + w; batch as usize])?;
            }
            inst.poll_wait();
            for s in 0..o.steps {
                let t = Instant::now();
                // Reading the logits back is part of a real step, and it is also
                // what makes the wall time cover the device's work.
                inst.decode_batch_at(&tokens, &vec![position + 2 + s; batch as usize])?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            Ok(())
        })();
        let used = device_used_mib();
        if let Err(e) = result {
            println!("{batch:>5} {:>10.1} {:>10} -- decode failed: {e}", gib(planned), "-");
            boundary = Some(serde_json::json!({ "batch": batch, "decode_error": e, "needed_bytes": planned }));
            break;
        }
        let ms = median(&mut times);
        let total = batch as f64 * 1e3 / ms;
        println!(
            "{batch:>5} {:>10.1} {:>10} {ms:>11.2} {total:>11.1} {:>13.2} {:>13.2}",
            gib(planned),
            used.map(|m| format!("{:.1}", m as f64 / 1024.0)).unwrap_or_else(|| "-".into()),
            total / batch as f64,
            gib(kv_per_seq)
        );
        rows.push(serde_json::json!({
            "batch": batch,
            "planned_bytes": planned,
            "device_used_mib": used,
            "ms_per_step_median": ms,
            "ms_per_step_min": times.iter().cloned().fold(f64::INFINITY, f64::min),
            "tok_per_s_total": total,
            "tok_per_s_per_stream": total / batch as f64,
            "kv_bytes_per_sequence": kv_per_seq,
        }));
        drop(inst);
    }

    if let Some(best) = rows.iter().max_by(|a, b| a["tok_per_s_total"].as_f64().partial_cmp(&b["tok_per_s_total"].as_f64()).unwrap()) {
        println!("\nmaximum sustained throughput: {:.1} tok/s total at batch {}", best["tok_per_s_total"].as_f64().unwrap(), best["batch"]);
    }
    if let Some(b) = &boundary {
        println!("stopped at batch {}: the next size does not fit or failed (see above)", b["batch"]);
    }
    if let Some(out) = o.json {
        let doc = serde_json::json!({
            "scenario": "longctx-decode-sweep",
            "target": format!("qwen35:{}-gguf-resident", tier.describe()),
            "env": { "backend": backend, "build": "release", "device": devices.iter().map(|(d, _)| format!("{d:?}")).collect::<Vec<_>>(), "host_offload": false },
            "workload": { "context_capacity": o.ctx, "decode_position": position, "context": "synthetic: fresh caches, real decode work", "steps_measured": o.steps },
            "rows": rows,
            "out_of_memory_boundary": boundary,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&doc).expect("serialize")).unwrap_or_else(|e| panic!("write {out}: {e}"));
        println!("wrote {out}");
    }
}
