// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT training step (forward + backward) on the P40: tiled backward GEMMs
//! (matmul_dx_reg / matmul_dw_reg) vs naive, with gradient parity.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-gpt --test bench_train_p40 -- --ignored --nocapture
//! ```
//! The backward is ~2/3 of a training step and ran entirely on the naive GEMMs
//! (~0.5% of peak). This measures the whole-step effect and checks the tiled
//! backward produces the same gradients.

use std::time::Instant;

use gpt::{Gpt, GptConfig};
use gpu_core::{set_default_backend, Backend};

fn cfg() -> GptConfig {
    GptConfig { vocab: 8192, block_size: 256, n_layers: 6, d_model: 384, n_heads: 6, d_ff: 1536 }
        .with_ff_default()
}
const B: usize = 4;
const T: usize = 256;

fn run(naive: bool, reps: usize) -> (f64, Vec<f32>) {
    if naive { std::env::set_var("BRAIN_GPT_NAIVE_MM", "1"); }
    else { std::env::remove_var("BRAIN_GPT_NAIVE_MM"); }
    set_default_backend(Backend::Wgpu);
    let c = cfg();
    let init = gpt::init_weights(&c, 1234);
    let m = Gpt::new_on(gpu_core::testgpu::dev(gpt::model::PIPELINES), c.clone(), B as u32, T as u32, &init);
    let x: Vec<u32> = (0..B * T).map(|i| ((i * 131 + 7) as u32) % c.vocab).collect();
    let y: Vec<u32> = (0..B * T).map(|i| ((i * 131 + 8) as u32) % c.vocab).collect();
    m.set_batch(&x, &y);

    m.zero_grads(); m.forward(); m.backward(); m.gpu.poll_wait(); // warm
    let grad = m.read_grad("blocks.0.mlp.fc.weight"); // a representative dW

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        m.zero_grads();
        m.forward();
        m.backward();
        m.gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, grad)
}

#[test]
#[ignore]
fn train_step_speed() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let reps = 15;
    let (naive_ms, gn) = run(true, reps);
    let (reg_ms, gr) = run(false, reps);

    let maxd = gn.iter().zip(&gr).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = gn.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let rel = maxd / scale;

    println!("\n=== GPT train step (fwd+bwd) B={B} T={T} d=384 L=6 (P40 wgpu) ===");
    println!("  naive backward:  {naive_ms:8.1} ms/step");
    println!("  tiled backward:  {reg_ms:8.1} ms/step   {:.2}x faster", naive_ms / reg_ms);
    println!("  gradient parity (dW mlp.fc): max-abs {maxd:.2e}  rel {rel:.2e}");
    assert!(rel < 1e-4, "tiled backward changes gradients (rel {rel:.2e})");
    assert!(reg_ms < naive_ms, "tiled backward not faster");
}
