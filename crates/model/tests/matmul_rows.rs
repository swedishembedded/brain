// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! matmul_rows (8-row-blocked) must match the naive matmul kernel exactly —
//! same per-output accumulation order → bitwise equality, including ragged
//! row-tails.

use gpu_core::Gpu;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
}

#[test]
fn matches_naive_matmul_bitwise() {
    // Pin BOTH kernels to the scalar JIT. This test states that the row-blocked
    // WGSL reorders nothing per output — a claim about the two KERNELS, which
    // bitwise equality can only witness if both actually run as written. The
    // backend's AVX2 fast path (which `matmul` otherwise routes to) sums in a
    // different order; its accuracy has its own tests in backend-cpu.
    std::env::set_var("BRAIN_NO_FASTCONV", "1");
    let gpu = Gpu::new_cpu(&[
        ("matmul", kernels::MATMUL),
        ("matmul_rows", kernels::MATMUL_ROWS),
    ]);
    std::env::remove_var("BRAIN_NO_FASTCONV");
    let mut seed = 0xC0FFEE;
    for (m, k, n) in [(1usize, 7usize, 5usize), (8, 16, 3), (13, 32, 20), (64, 24, 8), (17, 5, 1)] {
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let xb = gpu.storage_init("x", &x);
        let wb = gpu.storage_init("w", &w);
        let naive = gpu.storage((m * n) as u64);
        let rows = gpu.storage((m * n) as u64);
        let params = [m as u32, k as u32, n as u32];
        let steps = vec![
            gpu.step(0, &[&xb, &wb, &naive], &params, (m * n) as u32),
            gpu.step(1, &[&xb, &wb, &rows], &params, (m.div_ceil(8) * n) as u32),
        ];
        gpu.submit(&[], &steps);
        let a = gpu.read(&naive, m * n);
        let b = gpu.read(&rows, m * n);
        assert_eq!(a, b, "mismatch at m={m} k={k} n={n}");
    }
}
