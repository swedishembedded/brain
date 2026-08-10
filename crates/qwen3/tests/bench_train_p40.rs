// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen training step (fwd+bwd) on the P40: tiled backward vs naive + grad parity.
//! ```text
//! DISPLAY= cargo test --release -p brain-qwen --test bench_train_p40 -- --ignored --nocapture
//! ```
use std::time::Instant;
use gpu_core::{set_default_backend, Backend};
use qwen3::{Qwen, QwenConfig};

fn cfg() -> QwenConfig {
    // 0.6B-shaped but shallow for a quick step: d1024, ff3072, GQA 16/8, L4.
    let mut c = QwenConfig::tiny();
    c.vocab = 4096; c.block_size = 256; c.n_layers = 4; c.d_model = 1024;
    c.n_heads = 16; c.n_kv_heads = 8; c.head_dim = 128; c.d_ff = 3072;
    c.with_defaults()
}
const B: usize = 2; const T: usize = 256;

fn run(naive: bool, reps: usize) -> (f64, Vec<f32>) {
    if naive { std::env::set_var("BRAIN_QWEN_NAIVE_MM", "1"); } else { std::env::remove_var("BRAIN_QWEN_NAIVE_MM"); }
    set_default_backend(Backend::Wgpu);
    let c = cfg();
    let init = qwen3::init_weights(&c, 7);
    let m = Qwen::new(c.clone(), B as u32, T as u32, &init);
    let x: Vec<u32> = (0..B*T).map(|i| ((i*131+7) as u32) % c.vocab).collect();
    let y: Vec<u32> = (0..B*T).map(|i| ((i*131+8) as u32) % c.vocab).collect();
    m.set_batch(&x, &y);
    m.zero_grads(); m.forward(); m.backward(); m.gpu.poll_wait();
    let grad = m.read_grad("blocks.0.mlp.gate.weight");
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        m.zero_grads(); m.forward(); m.backward(); m.gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64()*1e3);
    }
    (best, grad)
}

#[test]
#[ignore]
fn qwen_train_step() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let reps = 12;
    let (nm, gn) = run(true, reps);
    let (rm, gr) = run(false, reps);
    let maxd = gn.iter().zip(&gr).fold(0f32,|m,(a,b)| m.max((a-b).abs()));
    let rel = maxd / gn.iter().fold(1e-6f32,|m,&v| m.max(v.abs()));
    println!("\n=== Qwen train step (fwd+bwd) d1024 ff3072 L4 GQA16/8 (P40 wgpu) ===");
    println!("  naive backward: {nm:8.1} ms   tiled: {rm:8.1} ms   {:.2}x", nm/rm);
    println!("  grad parity (dW gate): max-abs {maxd:.2e} rel {rel:.2e}");
    assert!(rel < 1e-4 && rm < nm);
}
