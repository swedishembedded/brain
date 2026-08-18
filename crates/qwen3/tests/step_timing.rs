// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One-step timing on the real imported Qwen3-0.6B (diagnose finetune speed).
//! ```text
//! QWEN3_DIR=<qwen3-0.6b checkpoint dir> DISPLAY= \
//!   cargo test --release -p brain-qwen --test step_timing -- --ignored --nocapture
//! ```
use std::time::Instant;

use gpu_core::{set_default_backend, Backend};
use qwen3::{Qwen, QwenConfig};

#[test]
#[ignore]
fn step_timing() {
    let Some(d) = std::env::var("QWEN3_DIR").ok() else {
        return brain_testutil::skip("set QWEN3_DIR to a real Qwen3 checkpoint dir");
    };
    set_default_backend(Backend::Wgpu);
    if std::env::var("BRAIN_TILE_BUDGET_WORDS").is_err() {
        std::env::set_var("BRAIN_TILE_BUDGET_WORDS", "200000000");
    }
    let w = format!("{d}/brain/qwen3-0.6b-ft512.safetensors");
    let c = checkpoint::load(&w);
    let mut cfg = QwenConfig::from_json(&c.header["config"]);
    let lora = std::env::var("LORA").is_ok();
    if lora {
        cfg.lora = Some(qwen3::LoraCfg { rank: 16, alpha: 32.0,
            targets: ["wq","wk","wv","wo","gate","up","down"].iter().map(|s| s.to_string()).collect() });
    }
    println!("lora={lora}");
    let block = std::env::var("BLK").ok().and_then(|v| v.parse().ok()).unwrap_or(512u32);
    println!("cfg: L={} d={} vocab={} block={}", cfg.n_layers, cfg.d_model, cfg.vocab, block);
    let init = c.by_role("");
    let t0 = Instant::now();
    let m = Qwen::new(cfg.clone(), 1, block, &init);
    m.gpu.poll_wait();
    println!("model build: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let x: Vec<u32> = (0..block).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..block).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    m.zero_grads();
    m.forward();
    m.backward();
    m.gpu.poll_wait(); // warm

    let mut fbest = f64::INFINITY;
    let mut sbest = f64::INFINITY;
    for _ in 0..3 {
        let tf = Instant::now();
        m.forward();
        m.gpu.poll_wait();
        fbest = fbest.min(tf.elapsed().as_secs_f64() * 1e3);

        let ts = Instant::now();
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(1, 1e-4, 0.0, Some(1.0), 1.0);
        m.gpu.poll_wait();
        sbest = sbest.min(ts.elapsed().as_secs_f64() * 1e3);
    }
    println!("forward: {fbest:.0} ms   full step (fwd+bwd+adamw): {sbest:.0} ms");
}
