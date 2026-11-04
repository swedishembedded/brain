// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Throughput of the library GEMMs and the blockwise attention at the 27B's
//! training shapes, as the minimum device time over repeated runs (the least
//! preempted run is the closest a shared card gets to an idle one). A
//! benchmark, not a gate: it prints and asserts only that the work ran.
//!
//! Swedish Embedded AB implements tensor-core training stacks that degrade
//! gracefully on hardware without them. If your team needs expertise in
//! measuring GPU kernels on a card you do not own exclusively, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Run: `cargo test --release -p brain-gpu-core --test cuda_library_bench -- --ignored --nocapture`

use gpu_core::provider::attention::AttnShape;
use gpu_core::provider::library::{GemmPrecision, LibraryGemmProvider, Operand};
use gpu_core::{Gpu, Step};

static NO_KERNELS: &[(&str, &str)] = &[];

/// Minimum wall time over `reps` runs of `steps` (each followed by a drain).
fn best_ms(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore = "benchmark: needs a CUDA card and prints a table"]
fn library_gemm_and_attention_throughput_at_the_27b_shapes() {
    let gpu = Gpu::try_new_cuda(NO_KERNELS).expect("a CUDA device");
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    assert!(provider.is_active(), "{:?}", provider.path());
    // (what, m, n, k): the real projections at 8192 rows, an MLP chunk, the head tile.
    for (what, m, n, k) in [
        ("mlp gate/up fwd", 8192u32, 17408u32, 5120u32),
        ("mlp down fwd", 8192, 5120, 17408),
        ("gdn in_proj_qkv fwd", 8192, 10240, 5120),
        ("gqa q_proj fwd", 8192, 12288, 5120),
        ("gqa k_proj fwd", 8192, 1024, 5120),
        ("gdn out_proj fwd", 8192, 5120, 6144),
        ("mlp chunk gate fwd", 2048, 17408, 5120),
        ("head tile fwd", 1024, 62080, 5120),
    ] {
        let w = gpu.storage((n as u64 * k as u64) / 2);
        let x = gpu.storage(m as u64 * k as u64);
        let y = gpu.storage(m as u64 * n as u64);
        let op = |buf| Operand { buf, range: (0, 0) };
        let mut steps = Vec::new();
        assert!(provider.forward(&gpu, &mut steps, &op(&x), &op(&w), &op(&y), m, n, k), "{what} was declined");
        let ms = best_ms(&gpu, &steps, 8);
        // The cast step is part of the measured cost: it is part of the step.
        let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
        println!("BENCH gemm {what:24} {m:5}x{n:5}x{k:5}: {ms:8.3} ms (cast included) {tflops:7.1} TFLOP/s");
    }
    for (t, hd) in [(4096u32, 256u32), (8192, 256)] {
        let sh = AttnShape { b: 1, t, n_heads: 24, n_kv_heads: 4, head_dim: hd, block_rows: None };
        let (n, qd, kvd) = (t as u64, 24 * hd as u64, 4 * hd as u64);
        let (q, k, v) = (gpu.storage(n * qd), gpu.storage(n * kvd), gpu.storage(n * kvd));
        let (ctx, lse, dctx) = (gpu.storage(n * qd), gpu.storage(24 * n), gpu.storage(n * qd));
        let fwd = provider.attention_fwd(&gpu, &sh, &q, &k, &v, &ctx, &lse).expect("forward");
        let f = best_ms(&gpu, &fwd, 5);
        let (dq, dk, dv) = (gpu.storage(n * qd), gpu.storage(n * kvd), gpu.storage(n * kvd));
        let bwd = provider.attention_bwd(&gpu, &sh, &q, &k, &v, &ctx, &lse, &dctx, &dq, &dk, &dv).expect("backward");
        let b = best_ms(&gpu, &bwd, 5);
        println!("BENCH attention T={t} hd={hd}: forward {f:8.2} ms, backward {b:8.2} ms ({} + {} steps)", fwd.len(), bwd.len());
    }
}
