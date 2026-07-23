// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tensor parallelism (Megatron-style) validated bit-exact against a single GPU.
//!
//! The MLP `Z = (GeLU(X·Wfc))·Wproj` is split across `k` ranks: `Wfc` is
//! **column-parallel** (each rank owns `ff/k` of the hidden columns, no
//! communication), GeLU is elementwise on the shard, and `Wproj` is
//! **row-parallel** (each rank produces a partial `[m,d]`), combined by **one
//! all-reduce** through the transport-agnostic [`model::Collective`]. The
//! intermediate stays sharded, so no layout-aware gather is needed — exactly the
//! Megatron design (`resources/dp/megatron-lm-tensor-parallel-1909.08053.pdf`).
//!
//! This proves the core TP mechanic (split one op across devices + collective
//! combine) on the real 2×P40 hardware. Two ranks on GPUs 0 and 1 by default;
//! `SHARD_TEST_GPUS=1,1` pins both to one card. Skipped under `MOE_SKIP_GPU_TESTS`.

use gpu_core::Gpu;
use model::{Collective, HostCollective};

const MATMUL: usize = 0;
const GELU: usize = 1;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![("matmul", kernels::MATMUL), ("gelu", kernels::GELU)]
}

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}
fn stage_gpus() -> Vec<usize> {
    std::env::var("SHARD_TEST_GPUS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1])
}

/// `out[m,n] = GeLU?(x[m,k] · w[n,k]^T)` on `gpu`.
fn matmul(gpu: &Gpu, x: &[f32], w: &[f32], m: usize, k: usize, n: usize, gelu: bool) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let ob = gpu.storage((m * n) as u64);
    let mut steps = vec![gpu.step(MATMUL, &[&xb, &wb, &ob], &[m as u32, k as u32, n as u32], (m * n) as u32)];
    let gb = gpu.storage((m * n) as u64);
    if gelu {
        steps.push(gpu.step(GELU, &[&ob, &gb], &[(m * n) as u32], (m * n) as u32));
    }
    gpu.submit(&[], &steps);
    gpu.read(if gelu { &gb } else { &ob }, m * n)
}

#[test]
fn tensor_parallel_mlp_matches_single_gpu() {
    if gpu_disabled() {
        return;
    }
    let gpus = stage_gpus();
    let world = gpus.len();
    let (m, d, ff) = (8usize, 16usize, 32usize);
    assert_eq!(ff % world, 0, "ff must divide across ranks");
    let half = ff / world;

    // Deterministic inputs/weights.
    let x: Vec<f32> = (0..m * d).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
    let w_fc: Vec<f32> = (0..ff * d).map(|i| ((i * 5 % 17) as f32 / 17.0) - 0.5).collect(); // [ff, d]
    let w_proj: Vec<f32> = (0..d * ff).map(|i| ((i * 3 % 11) as f32 / 11.0) - 0.5).collect(); // [d, ff]

    // --- single-GPU reference: Z = GeLU(X·Wfc^T) · Wproj^T ---
    std::env::remove_var("BRAIN_GPU_INDEX");
    let g0 = Gpu::new(&kernels());
    let y = matmul(&g0, &x, &w_fc, m, d, ff, true); // [m, ff]
    let z_single = matmul(&g0, &y, &w_proj, m, ff, d, false); // [m, d]
    drop(g0);

    // Build one Gpu per rank **sequentially** (BRAIN_GPU_INDEX is process-global
    // and wgpu device init is not concurrency-safe — the Pipeline/DataParallel do
    // the same). Then the threads only compute + collective, never touch env.
    let rank_gpus: Vec<Gpu> = gpus
        .iter()
        .map(|&gi| {
            std::env::set_var("BRAIN_GPU_INDEX", gi.to_string());
            Gpu::new(&kernels())
        })
        .collect();
    std::env::remove_var("BRAIN_GPU_INDEX");

    // --- tensor-parallel across `world` ranks ---
    let coll = HostCollective::new(world);
    let results: Vec<std::sync::Mutex<Vec<f32>>> = (0..world).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    std::thread::scope(|s| {
        for (rank, g) in rank_gpus.iter().enumerate() {
            let (coll, results, x, w_fc, w_proj) = (&coll, &results, &x, &w_fc, &w_proj);
            s.spawn(move || {
                // column-parallel Wfc: rows [rank*half .. ) of [ff, d] -> [half, d]
                let w_fc_r: Vec<f32> = w_fc[rank * half * d..(rank + 1) * half * d].to_vec();
                let y_r = matmul(g, x, &w_fc_r, m, d, half, true); // [m, half]
                // row-parallel Wproj: column block [rank*half ..) of [d, ff] -> [d, half]
                let mut w_proj_r = vec![0f32; d * half];
                for i in 0..d {
                    w_proj_r[i * half..(i + 1) * half]
                        .copy_from_slice(&w_proj[i * ff + rank * half..i * ff + rank * half + half]);
                }
                let z_partial = matmul(g, &y_r, &w_proj_r, m, half, d, false); // [m, d] partial
                // combine partials: one all-reduce (Megatron `g` operator).
                let z = coll.all_reduce(rank, z_partial);
                *results[rank].lock().unwrap() = z;
            });
        }
    });

    let z_tp = results[0].lock().unwrap().clone();
    let (mut num, mut den) = (0f32, 1e-6f32);
    for (a, b) in z_single.iter().zip(&z_tp) {
        num = num.max((a - b).abs());
        den = den.max(a.abs());
    }
    let rel = num / den;
    eprintln!("TP MLP ({world}-way) vs single-GPU: rel {rel:.2e}");
    assert!(rel < 1e-4, "tensor-parallel MLP diverged: rel {rel:.2e}");
    // every rank must hold the identical reduced result
    for r in 1..world {
        assert_eq!(*results[r].lock().unwrap(), z_tp, "rank {r} disagrees after all-reduce");
    }
}
