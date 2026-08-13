// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated CPU microbench for the per-frame HOST work of the NPU TTS path:
//! the MTP residual generation (`CpuMtp::generate_residuals`) and the cb0
//! codec-head logits (`TalkerTables::codec_head_logits`). No NPU involvement —
//! measures steady-state host cost, uncontended, so we can separate it from the
//! NPU/codec time and from thermal/contention noise.
//!
//!   cargo build --release --example mtp_bench -p brain-tts
//!   ./target/release/examples/mtp_bench out/tts-1b7 [iters]

use data::rng::Lcg;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "out/tts-1b7".to_string());
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);

    let mtp_path = format!("{dir}/mtp.safetensors");
    let talker_path = format!("{dir}/talker.safetensors");
    eprintln!("loading MTP {mtp_path} + TalkerTables {talker_path} …");
    let mut mtp = qwen3tts::CpuMtp::load(&mtp_path);
    let tables = qwen3tts::npu_gen::TalkerTables::load(&talker_path);
    let emb = mtp.cfg.embedding_dim as usize;
    let d = tables.d();
    eprintln!("MTP embedding_dim={emb} d_model(MTP)={} ; talker d={d} vocab={}", mtp.cfg.d_model, tables.cfg.vocab);

    // Deterministic pseudo-random inputs (cost is data-independent).
    let talker_hidden = Lcg::new(1).vec_scaled(emb, 0.5);
    let cb0_embed = Lcg::new(2).vec_scaled(emb, 0.5);

    // ---- MTP generate_residuals ----
    for _ in 0..5 {
        let _ = mtp.generate_residuals(&talker_hidden, &cb0_embed);
    }
    let mut ts = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mtp.generate_residuals(&talker_hidden, &cb0_embed);
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    let mean = ts.iter().sum::<f64>() / ts.len() as f64;
    println!(
        "MTP generate_residuals: min={:.1}ms p50={:.1}ms mean={:.1}ms max={:.1}ms (n={})",
        ts[0], med, mean, ts[ts.len() - 1], iters
    );

    // ---- cb0 codec head ----
    let hidden = Lcg::new(3).vec_scaled(d, 0.5);
    for _ in 0..5 {
        let _ = tables.codec_head_logits(&hidden);
    }
    let mut hs = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = tables.codec_head_logits(&hidden);
        hs.push(t.elapsed().as_secs_f64() * 1e3);
    }
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "cb0 codec_head_logits: min={:.1}ms p50={:.1}ms max={:.1}ms",
        hs[0], hs[hs.len() / 2], hs[hs.len() - 1]
    );
    println!("threads={}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
}
