// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end GPT forward: CPU reference vs GPU, with the size-adaptive GEMM
//! picker (`linear_kernel`) live. This is the model-level version of
//! `brain-gpu-core`'s `bench_matmul` — it answers the same two questions one
//! layer up, where the naive/register GEMM choice actually runs inside a real
//! forward pass rather than in isolation:
//!
//!   * **equivalence** — the GPU logits match the CPU reference (fp32
//!     reduction-order differences only), AND the register-tiled forward
//!     matches the naive-GEMM forward on the SAME backend (isolating the kernel
//!     swap from any CPU/GPU difference), and
//!   * **speedup** — GPU forward vs the 48-thread AVX2 CPU forward, min-of-N.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-gpt --test bench_forward_p40 -- --ignored --nocapture
//! ```

use std::time::Instant;

use gpt2::{Gpt, GptConfig};
use gpu_core::{set_default_backend, Backend};

/// A gpt-small-ish config: d_model 384, 6 layers, ff 1536 — the size where every
/// forward linear clears the 128×128 tile threshold and takes the register GEMM.
fn cfg() -> GptConfig {
    GptConfig { vocab: 8192, block_size: 256, n_layers: 6, d_model: 384, n_heads: 6, d_ff: 1536 }
        .with_ff_default()
}

const B: usize = 4;
const T: usize = 256;

fn tokens(cfg: &GptConfig) -> (Vec<u32>, Vec<u32>) {
    let n = B * T;
    let x: Vec<u32> = (0..n).map(|i| ((i * 131 + 7) as u32) % cfg.vocab).collect();
    let y: Vec<u32> = (0..n).map(|i| ((i * 131 + 8) as u32) % cfg.vocab).collect();
    (x, y)
}

/// Run `reps` forwards, return (logits, best-ms). `env` is applied before the
/// model is built so `linear_kernel` sees it at dispatch-record time.
fn run(backend: Backend, naive_mm: bool, reps: usize) -> (Vec<f32>, f64) {
    if naive_mm {
        std::env::set_var("BRAIN_GPT_NAIVE_MM", "1");
    } else {
        std::env::remove_var("BRAIN_GPT_NAIVE_MM");
    }
    set_default_backend(backend);
    let c = cfg();
    let init = gpt2::init_weights(&c, 1234);
    let m = Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), c.clone(), B as u32, T as u32, &init);
    let (x, y) = tokens(&c);

    m.set_batch(&x, &y);
    m.forward_submit();
    let logits = m.logits_host(); // warm

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        m.forward_submit();
        m.gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    (logits, best)
}

fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let maxd = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let scale = a.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    (maxd, maxd / scale)
}

#[test]
#[ignore]
fn bench_forward_p40() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let reps: usize = std::env::var("BRAIN_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);

    // CPU reference (naive GEMM — the tiled kernels route to the same AVX2 path
    // on CPU anyway, so this is the honest `--device cpu` number).
    let (cpu_logits, cpu_ms) = run(Backend::Cpu, true, reps);

    // GPU, naive GEMM: isolates the CPU-vs-GPU reduction-order difference.
    let (gpu_naive_logits, gpu_naive_ms) = run(Backend::Vulkan, true, reps);
    // GPU, register GEMM via linear_kernel: the production path.
    let (gpu_reg_logits, gpu_reg_ms) = run(Backend::Vulkan, false, reps);

    let (d_cpu_abs, d_cpu_rel) = rel(&cpu_logits, &gpu_reg_logits);
    // Same backend, only the GEMM kernel differs → this is the tight equivalence
    // gate: the register tile must reproduce the naive kernel's own output.
    let (d_ker_abs, d_ker_rel) = rel(&gpu_naive_logits, &gpu_reg_logits);

    println!("\n=== GPT forward B={B} T={T} d=384 L=6 (P40) ===");
    println!("cpu (avx2, naive gemm)   {cpu_ms:8.2} ms");
    println!("gpu (naive gemm)         {gpu_naive_ms:8.2} ms   {:.1}x vs cpu", cpu_ms / gpu_naive_ms);
    println!("gpu (register gemm)      {gpu_reg_ms:8.2} ms   {:.1}x vs cpu", cpu_ms / gpu_reg_ms);
    println!("register vs naive gemm:  {:.2}x faster on the same GPU", gpu_naive_ms / gpu_reg_ms);
    println!("equivalence:");
    println!("  gpu-reg vs cpu ref     max-abs {d_cpu_abs:.2e}  rel {d_cpu_rel:.2e}");
    println!("  gpu-reg vs gpu-naive   max-abs {d_ker_abs:.2e}  rel {d_ker_rel:.2e}");

    // Kernel swap on one backend: pure fp32 associativity, ~1e-5.
    assert!(d_ker_rel < 1e-4, "register gemm changes the forward result (rel {d_ker_rel:.2e})");
    // CPU vs GPU: reduction order across two different reducers.
    assert!(d_cpu_rel < 5e-3, "gpu forward diverges from cpu reference (rel {d_cpu_rel:.2e})");
    // The whole point: the register GEMM makes the GPU forward faster than CPU.
    assert!(gpu_reg_ms < cpu_ms, "gpu-reg forward ({gpu_reg_ms:.1}ms) not faster than cpu ({cpu_ms:.1}ms)");
}
