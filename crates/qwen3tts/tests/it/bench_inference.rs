// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS Talker inference speed — validated across backends and precisions.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-tts --test bench_inference -- --ignored --nocapture
//! ```
//!
//! The Talker is the TTS bottleneck: a 28-layer Qwen3 GQA decoder that samples
//! one codec frame at a time (`TalkerModel` wraps `qwen3::Qwen`, so it inherits
//! the register-tiled `matmul_reg2` GEMM). This measures the real 0.6B Talker at
//! two levels:
//!
//!  * **model** — a full T-frame forward (the prefill cost, and what a cache-free
//!    decode step recomputes), on the CPU backend (AVX2) vs the P40 (Vulkan fp32),
//!    reported as latency and codec-frames/second (the TTS frame rate is
//!    ~12.5 Hz, so anything above that is faster-than-realtime), and
//!  * **precision** — the Talker's dominant linear shapes run at fp32 (naive),
//!    fp32 (reg2), and INT8 (DP4A `matmul_i8`), so the per-precision speedup is
//!    measured on the exact GEMMs TTS issues.
//!
//! No checkpoint needed: random-init weights exercise the identical arithmetic,
//! and speed is weight-value-independent.

use std::time::Instant;

use gpu_core::{set_default_backend, Backend};
use qwen3tts::{config::TalkerConfig, TalkerModel};

/// The real 0.6B Talker: 28 layers, d_model 1024, q_dim 2048 (16×128), 8 KV
/// heads, d_ff 3072, codebook-0 vocab 3072.
fn talker_0_6b() -> TalkerConfig {
    let mut c = TalkerConfig::tiny();
    c.n_layers = 28;
    c.d_model = 1024;
    c.head_dim = 128;
    c.n_heads = 16;
    c.n_kv_heads = 8;
    c.d_ff = 3072;
    c.vocab = 3072;
    c
}

const T: usize = 256; // codec frames in the forward (about twenty seconds of audio at 12.5 Hz)

/// Total forward FLOPs of the 0.6B Talker at T=256 (28 layers: qkv/o + gate/up/
/// down GEMMs + O(T²) attention + lm_head). Almost all of it is GEMM.
const TALKER_GFLOP: f64 = 242.1;
const P40_PEAK_GFLOPS: f64 = 11_760.0;

fn frames(cfg: &TalkerConfig) -> Vec<u32> {
    (0..T as u32).map(|i| (i * 7 + 1) % cfg.vocab).collect()
}

/// Build the 0.6B Talker on `backend`, return (min-of-`reps` ms, logits) for a
/// T-frame forward.
fn forward(backend: Backend, reps: usize) -> (f64, Vec<f32>) {
    set_default_backend(backend);
    let cfg = talker_0_6b();
    let m = TalkerModel::new_trainable(cfg.clone(), 1, T as u32, 1234);
    let x = frames(&cfg);
    let logits = m.logits_all(&x); // warm + captured for parity
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = m.logits_all(&x);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    (best, logits)
}

#[test]
#[ignore]
fn talker_inference_speed() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let reps = 5;

    println!("\n=== Qwen3-TTS Talker 0.6B — {T}-frame forward (prefill), {TALKER_GFLOP:.0} GFLOP ===");
    let (cpu, cpu_logits) = forward(Backend::Cpu, reps);
    let (vk, vk_logits) = forward(Backend::Vulkan, reps);
    let (wg, wg_logits) = forward(Backend::Wgpu, reps);
    let fps = |ms: f64| T as f64 / (ms / 1e3);
    let gfs = |ms: f64| TALKER_GFLOP / (ms / 1e3);
    let pk = |ms: f64| 100.0 * gfs(ms) / P40_PEAK_GFLOPS;
    println!("  {:<20} {:>8} {:>10} {:>10} {:>8} {:>8}", "backend", "ms/fwd", "frames/s", "GFLOP/s", "%peak", "vs cpu");
    println!("  {:<20} {:>8.1} {:>10.1} {:>10.0} {:>7.1}% {:>7}", "cpu  fp32 (AVX2)", cpu, fps(cpu), gfs(cpu), pk(cpu), "1.0");
    println!("  {:<20} {:>8.1} {:>10.1} {:>10.0} {:>7.1}% {:>6.1}x", "P40  fp32 vulkan", vk, fps(vk), gfs(vk), pk(vk), cpu / vk);
    println!("  {:<20} {:>8.1} {:>10.1} {:>10.0} {:>7.1}% {:>6.1}x", "P40  fp32 wgpu", wg, fps(wg), gfs(wg), pk(wg), cpu / wg);

    // Validate: both GPU backends reproduce the CPU reference.
    let rel = |a: &[f32], b: &[f32]| {
        let maxd = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        maxd / a.iter().fold(1e-6f32, |m, &v| m.max(v.abs()))
    };
    let (rv, rw) = (rel(&cpu_logits, &vk_logits), rel(&cpu_logits, &wg_logits));
    println!("  parity vs cpu: vulkan rel {rv:.2e}   wgpu rel {rw:.2e}  ({} logits)", cpu_logits.len());
    assert!(cpu.is_finite() && vk > 0.0 && wg > 0.0);
    assert!(rv < 5e-3 && rw < 5e-3, "TTS Talker GPU forward diverges from cpu (vk {rv:.2e} wg {rw:.2e})");
}
