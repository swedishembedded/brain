// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoder-LM profiler: where a PREFILL and a DECODE step actually spend their
//! time, per kernel kind, graded against the device's measured roofline.
//!
//! Why this exists. `vqgan_bench`, `unet_bench` and `flux2_bench` cover the
//! conv/diffusion datapaths; the LLM datapath had no per-kernel-kind profiler at
//! all, so every recorded qwen number in `docs/performance/` came from
//! `BRAIN_PROFILE`'s timestamp table — and, more importantly, all of them were
//! measured on `qwen-synth:8x512x8` (47 M params, vocab 32 k). That target
//! **structurally cannot express Qwen3-0.6B**: it forces `head_dim = d/h` and
//! `d_ff = 4d`, while the real model has `head_dim` 128 (so `q_dim` 2048 ≠
//! `d_model` 1024) and `d_ff` 3072. Its vocab is 151936, not 32000 — 4.75× —
//! and the head is tied, so the LM head alone is 155.6 M params = 622 MB fp32,
//! about a quarter of every decode step's bytes. None of that was ever profiled.
//!
//! Random weights throughout: cost depends on shape, not values. Every number
//! this produces is **valid for cost and meaningless for output quality**.
//!
//! The roofline the rows are graded against is `gpu_core::roof`'s MEASURED one,
//! not a hardcoded peak — so the same table is meaningful on any device.
//!
//! Usage:
//!   qwen_bench                      # prefill T=512 + decode, fp32, at 0.6B
//!   qwen_bench prefill [T] [reps]
//!   qwen_bench decode  [ctx] [reps]
//!   qwen_bench head    [reps]       # the tied 151936x1024 LM head alone
//!   qwen_bench cost                 # offline FLOP/byte accounting, no device

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::Gpu;
use qwen::{init_weights, Qwen, QwenConfig};

/// One §F.1 pass profile against the device's measured roofline, plus the rows
/// that sit below their class's floor.
fn report(gpu: &Gpu, label: &str, steps: &[gpu_core::Step], reps: usize, roofs: Option<Roofs>) -> f64 {
    let p = gpu_core::profile::profile(gpu, label, steps, reps);
    p.print_top(roofs, 16);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<26} {:>5.1}% of its {} roof (floor {:.0}%) — {:.1}% of this pass",
                row.name,
                pct,
                bound.as_str(),
                bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
    p.total_secs
}

fn banner(gpu: &Gpu) -> Option<Roofs> {
    let roofs = gpu_core::roof::ensure(gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s, ridge {:.1} FLOP/byte",
            r.gflops,
            r.gbs,
            r.ridge()
        ),
        None => println!("roofline unmeasured — utilisation columns print '-' rather than a guess"),
    }
    roofs
}

/// Weight bytes and the decode-step bandwidth ceiling they imply.
///
/// Decode reads every weight once per token, so on a memory-bound device the
/// token rate has a hard ceiling of `bandwidth / weight_bytes` no matter how
/// good the kernels are. Stating it first is what makes the profile
/// interpretable: a kernel-level fix that cannot move this number is not a
/// throughput fix.
fn weight_budget(cfg: &QwenConfig, roofs: Option<Roofs>) {
    let (d, ff, l) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.n_layers as u64);
    let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
    let attn = d * hq + 2 * (d * hkv) + hq * d;
    let mlp = 2 * (d * ff) + ff * d;
    let per_layer = attn + mlp;
    let embed = cfg.vocab as u64 * d;
    // Tied embeddings mean the head IS the embedding table — counted once as
    // resident bytes, but read twice per step (gather + head GEMV).
    let total = l * per_layer + embed * if cfg.tie_embeddings { 1 } else { 2 };

    println!(
        "\n{} params = {:.2} GB fp32 / {:.2} GB int8   (layers {:.1} M x {l}, embed/head {:.1} M{})",
        total,
        total as f64 * 4.0 / 1e9,
        total as f64 / 1e9,
        per_layer as f64 / 1e6,
        embed as f64 / 1e6,
        if cfg.tie_embeddings { ", tied" } else { "" },
    );
    if let Some(r) = roofs {
        let bw = r.gbs as f64 * 1e9;
        println!(
            "decode is weight-bandwidth bound: ceiling {:.0} tok/s fp32, {:.0} tok/s int8 \
             (the head alone is {:.0}% of the bytes)",
            bw / (total as f64 * 4.0),
            bw / total as f64,
            100.0 * embed as f64 / total as f64,
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("all");

    let cfg = QwenConfig::qwen3_0_6b();

    if mode == "cost" {
        // Offline accounting only — no device, no probe, no weights uploaded.
        let t = 128u32;
        let init = init_weights(&cfg, 7);
        let m = Qwen::new(cfg.clone(), 1, t, &init);
        let c = m.cost_fwd();
        println!("{c}");
        return;
    }

    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    match mode {
        "decode" => {
            let ctx: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
            eprintln!("qwen_bench decode: Qwen3-0.6B, ctx {ctx}, {reps} reps (random weights)");
            let t0 = Instant::now();
            let init = init_weights(&cfg, 7);
            let m = Qwen::from_tensors_decode(cfg.clone(), &init, ctx);
            eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());
            let gpu = m.gpu().share();
            let roofs = banner(&gpu);
            weight_budget(&cfg, roofs);

            // Profile the step at the END of the context, where the attention
            // reduction is longest — the honest worst case for a decode step,
            // and the one a served request spends most of its tokens near.
            let steps = m.decode_steps(Some(1), ctx - 1);
            let secs = report(&gpu, &format!("DECODE @pos {}", ctx - 1), &steps, reps, roofs);
            println!(
                "\none decode step: {:.3} ms  ->  {:.1} tok/s (single stream, no LM head — \
                 the head is applied host-side on this path)",
                secs * 1e3,
                1.0 / secs
            );
        }
        "prefill" | "all" | _ => {
            let t: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
            eprintln!("qwen_bench prefill: Qwen3-0.6B, T={t}, {reps} reps (random weights)");
            let t0 = Instant::now();
            let init = init_weights(&cfg, 7);
            let m = Qwen::new(cfg.clone(), 1, t, &init);
            eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());
            let gpu = m.gpu().share();
            let roofs = banner(&gpu);
            weight_budget(&cfg, roofs);

            let x: Vec<u32> = (0..t).map(|i| (i * 131 + 7) % cfg.vocab).collect();
            m.set_batch(&x, &x);
            let secs = report(&gpu, &format!("PREFILL T={t}"), m.fwd_steps(), reps, roofs);
            println!(
                "\nprefill T={t}: {:.2} ms  ->  {:.0} tok/s",
                secs * 1e3,
                t as f64 / secs
            );
        }
    }
}
