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
const ATTN_SCORES: usize = 2;
const ATTN_SOFTMAX: usize = 3;
const ATTN_APPLY: usize = 4;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![("matmul", kernels::MATMUL), ("gelu", kernels::GELU)]
}
fn attn_kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("matmul", kernels::MATMUL),
        ("gelu", kernels::GELU),
        ("attn_scores", kernels::ATTN_SCORES),
        ("attn_softmax", kernels::ATTN_SOFTMAX),
        ("attn_apply", kernels::ATTN_APPLY),
    ]
}

const MATMUL_DX: usize = 2;
const MATMUL_DW: usize = 3;
const GELU_BWD: usize = 4;

fn bwd_kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("matmul", kernels::MATMUL),
        ("gelu", kernels::GELU),
        ("matmul_dx", kernels::MATMUL_DX),
        ("matmul_dw", kernels::MATMUL_DW),
        ("gelu_bwd", kernels::GELU_BWD),
    ]
}

/// One MLP forward+backward on `gpu` with local hidden width `ffl`
/// (`w_fc` is `[ffl,d]`, `w_proj` is `[d,ffl]`). Given upstream grad `dz` `[m,d]`
/// returns `(z, dx, dw_fc, dw_proj)` — `z`/`dx` are **partial** (all-reduce across
/// TP ranks for the full values); `dw_fc`/`dw_proj` are the local weight-shard
/// gradients. With `ffl==ff` this is the whole single-GPU MLP.
// The trailing (m, d, ffl) are the GEMM dims this oracle must be told
// explicitly — it exists to be dimension-swept against the sharded path.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn mlp_fwd_bwd(gpu: &Gpu, x: &[f32], w_fc: &[f32], w_proj: &[f32], dz: &[f32], m: usize, d: usize, ffl: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mu, du, fu) = (m as u32, d as u32, ffl as u32);
    let xb = gpu.storage_init("x", x);
    let wfc = gpu.storage_init("wfc", w_fc);
    let wproj = gpu.storage_init("wproj", w_proj);
    let dzb = gpu.storage_init("dz", dz);
    let fc_pre = gpu.storage((m * ffl) as u64);
    let y = gpu.storage((m * ffl) as u64);
    let z = gpu.storage((m * d) as u64);
    let dw_proj = gpu.storage((d * ffl) as u64);
    let dy = gpu.storage((m * ffl) as u64);
    let dpre = gpu.storage((m * ffl) as u64);
    let dw_fc = gpu.storage((ffl * d) as u64);
    let dx = gpu.storage((m * d) as u64);
    let steps = [
        // forward
        gpu.step(MATMUL, &[&xb, &wfc, &fc_pre], &[mu, du, fu], m as u32 * ffl as u32),
        gpu.step(GELU, &[&fc_pre, &y], &[(m * ffl) as u32], (m * ffl) as u32),
        gpu.step(MATMUL, &[&y, &wproj, &z], &[mu, fu, du], (m * d) as u32),
        // backward
        gpu.step(MATMUL_DW, &[&dzb, &y, &dw_proj], &[mu, fu, du], (d * ffl) as u32),
        gpu.step(MATMUL_DX, &[&dzb, &wproj, &dy], &[mu, fu, du, 0], (m * ffl) as u32),
        gpu.step(GELU_BWD, &[&fc_pre, &dy, &dpre], &[(m * ffl) as u32], (m * ffl) as u32),
        gpu.step(MATMUL_DW, &[&dpre, &xb, &dw_fc], &[mu, du, fu], (ffl * d) as u32),
        gpu.step(MATMUL_DX, &[&dpre, &wfc, &dx], &[mu, du, fu, 0], (m * d) as u32),
    ];
    gpu.submit(&[], &steps);
    (gpu.read(&z, m * d), gpu.read(&dx, m * d), gpu.read(&dw_fc, ffl * d), gpu.read(&dw_proj, d * ffl))
}

/// Multi-head self-attention **context** `[m, d_local]` for `nh_local` heads:
/// qkv = X·Wqkv^T, causal scores, softmax, apply. `Wqkv` is `[3·d_local, d]`.
/// (No output projection — the caller does that row-parallel.)
// (b, t, d, nh_local, hd) is the attention shape, swept by the caller.
#[allow(clippy::too_many_arguments)]
fn attn_ctx(gpu: &Gpu, x: &[f32], w_qkv: &[f32], b: usize, t: usize, d: usize, nh_local: usize, hd: usize) -> Vec<f32> {
    let m = b * t;
    let dl = nh_local * hd; // local hidden
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("wqkv", w_qkv);
    let qkv = gpu.storage((m * 3 * dl) as u64);
    let scores = gpu.storage((b * nh_local * t * t) as u64);
    let probs = gpu.storage((b * nh_local * t * t) as u64);
    let ctx = gpu.storage((m * dl) as u64);
    let (bb, nn, tt, hh) = (b as u32, nh_local as u32, t as u32, hd as u32);
    let s3 = (3 * dl) as u32;
    let steps = [
        gpu.step(MATMUL, &[&xb, &wb, &qkv], &[m as u32, d as u32, 3 * dl as u32], (m * 3 * dl) as u32),
        gpu.step(ATTN_SCORES, &[&qkv, &scores], &[bb, nn, tt, hh, s3, 0, dl as u32], bb * nn * tt * tt),
        gpu.step(ATTN_SOFTMAX, &[&scores, &probs], &[bb, nn, tt], bb * nn * tt),
        gpu.step(ATTN_APPLY, &[&probs, &qkv, &ctx], &[bb, nn, tt, hh, s3, 2 * dl as u32, dl as u32], bb * nn * tt * hh),
    ];
    gpu.submit(&[], &steps);
    gpu.read(&ctx, m * dl)
}

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    // Skip rather than fault when this box lacks the cards the test pins to.
    // Without the check the multi-GPU paths assume cards 0..n exist and die
    // inside the driver on a single-GPU or GPU-less machine, which reads as a
    // real regression and masks actual ones.
    let need = stage_gpus().iter().copied().max().unwrap_or(0) + 1;
    let have = gpu_core::discrete_gpu_count();
    if have < need {
        brain_testutil::skip_unavailable(&format!("needs {need} discrete GPU(s), found {have}"));
        return true;
    }
    false
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

/// These tests build REAL devices on purpose — device lifecycle/sharding is
/// the thing under test, so the pooled test device would defeat them. They
/// must therefore not run concurrently with EACH OTHER: several fresh devices
/// on one card is the exact driver deadlock the rest of the suite avoids via
/// gpu_core::testgpu. One lock, held for each test's whole body.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn tensor_parallel_mlp_matches_single_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let g0 = Gpu::new(&kernels());
    let y = matmul(&g0, &x, &w_fc, m, d, ff, true); // [m, ff]
    let z_single = matmul(&g0, &y, &w_proj, m, ff, d, false); // [m, d]
    drop(g0);

    // Build one Gpu per rank **sequentially** (wgpu device init is not
    // concurrency-safe — the Pipeline/DataParallel do the same), each pinned to
    // its canonical card through the device registry. The threads then only
    // compute + collective.
    let rank_gpus: Vec<Gpu> = gpus
        .iter()
        .map(|&gi| Gpu::new_on_index(gi as u32, &kernels()).expect("rank placement"))
        .collect();

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
    for (r, res) in results.iter().enumerate().skip(1) {
        assert_eq!(*res.lock().unwrap(), z_tp, "rank {r} disagrees after all-reduce");
    }
}

#[test]
fn tensor_parallel_mlp_training_matches_single_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if gpu_disabled() {
        return;
    }
    let gpus = stage_gpus();
    let world = gpus.len();
    let (m, d, ff) = (8usize, 16usize, 32usize);
    assert_eq!(ff % world, 0);
    let ffl = ff / world;

    let x: Vec<f32> = (0..m * d).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
    let w_fc: Vec<f32> = (0..ff * d).map(|i| ((i * 5 % 17) as f32 / 17.0) - 0.5).collect(); // [ff, d]
    let w_proj: Vec<f32> = (0..d * ff).map(|i| ((i * 3 % 11) as f32 / 11.0) - 0.5).collect(); // [d, ff]
    let dz: Vec<f32> = (0..m * d).map(|i| ((i * 2 % 7) as f32 / 7.0) - 0.5).collect(); // upstream grad

    // --- single-GPU reference gradients ---
    let g0 = Gpu::new(&bwd_kernels());
    let (_z0, dx0, dwfc0, dwproj0) = mlp_fwd_bwd(&g0, &x, &w_fc, &w_proj, &dz, m, d, ff);
    drop(g0);

    let rank_gpus: Vec<Gpu> = gpus
        .iter()
        .map(|&gi| Gpu::new_on_index(gi as u32, &bwd_kernels()).expect("rank placement"))
        .collect();

    // --- tensor-parallel forward+backward ---
    let coll = HostCollective::new(world);
    #[allow(clippy::type_complexity)]
    let out: Vec<std::sync::Mutex<(Vec<f32>, Vec<f32>, Vec<f32>)>> =
        (0..world).map(|_| std::sync::Mutex::new((Vec::new(), Vec::new(), Vec::new()))).collect();
    std::thread::scope(|s| {
        for (rank, g) in rank_gpus.iter().enumerate() {
            let (coll, out, x, w_fc, w_proj, dz) = (&coll, &out, &x, &w_fc, &w_proj, &dz);
            s.spawn(move || {
                let w_fc_r: Vec<f32> = w_fc[rank * ffl * d..(rank + 1) * ffl * d].to_vec(); // [ffl, d]
                let mut w_proj_r = vec![0f32; d * ffl];
                for i in 0..d {
                    w_proj_r[i * ffl..(i + 1) * ffl].copy_from_slice(&w_proj[i * ff + rank * ffl..i * ff + rank * ffl + ffl]);
                }
                let (_z, dx_p, dwfc_r, dwproj_r) = mlp_fwd_bwd(g, x, &w_fc_r, &w_proj_r, dz, m, d, ffl);
                // f operator: all-reduce the input gradient.
                let dx = coll.all_reduce(rank, dx_p);
                *out[rank].lock().unwrap() = (dx, dwfc_r, dwproj_r);
            });
        }
    });

    // dX: all-reduced, identical on every rank, == single-GPU.
    let dx_tp = out[0].lock().unwrap().0.clone();
    let rel = |a: &[f32], b: &[f32]| {
        let (mut n, mut den) = (0f32, 1e-6f32);
        for (p, q) in a.iter().zip(b) {
            n = n.max((p - q).abs());
            den = den.max(p.abs());
        }
        n / den
    };
    let rdx = rel(&dx0, &dx_tp);
    // dW_fc: rank shards stack over ff -> [ff, d].
    let mut dwfc_tp = Vec::new();
    for o in out.iter() {
        dwfc_tp.extend_from_slice(&o.lock().unwrap().1);
    }
    let rwfc = rel(&dwfc0, &dwfc_tp);
    // dW_proj: rank shards are column blocks of [d, ff]; reassemble per row.
    let mut dwproj_tp = vec![0f32; d * ff];
    for (r, o) in out.iter().enumerate() {
        let shard = &o.lock().unwrap().2; // [d, ffl]
        for i in 0..d {
            dwproj_tp[i * ff + r * ffl..i * ff + r * ffl + ffl].copy_from_slice(&shard[i * ffl..(i + 1) * ffl]);
        }
    }
    let rwproj = rel(&dwproj0, &dwproj_tp);
    eprintln!("TP training grads vs single-GPU: dX {rdx:.2e}  dWfc {rwfc:.2e}  dWproj {rwproj:.2e}");
    assert!(rdx < 1e-4 && rwfc < 1e-4 && rwproj < 1e-4, "TP training gradients diverged");
}

#[test]
fn tensor_parallel_attention_matches_single_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if gpu_disabled() {
        return;
    }
    let gpus = stage_gpus();
    let world = gpus.len();
    let (b, t, nh, hd) = (1usize, 8usize, 4usize, 4usize);
    let d = nh * hd;
    let m = b * t;
    assert_eq!(nh % world, 0, "heads must divide across ranks");
    let nh_r = nh / world;
    let d_r = nh_r * hd;

    let x: Vec<f32> = (0..m * d).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
    let w_qkv: Vec<f32> = (0..3 * d * d).map(|i| ((i * 5 % 17) as f32 / 17.0) - 0.5).collect(); // [3d, d]
    let w_o: Vec<f32> = (0..d * d).map(|i| ((i * 3 % 11) as f32 / 11.0) - 0.5).collect(); // [d, d]

    // --- single-GPU reference: MHA then output projection ---
    let g0 = Gpu::new(&attn_kernels());
    let ctx = attn_ctx(&g0, &x, &w_qkv, b, t, d, nh, hd); // [m, d]
    let z_single = matmul(&g0, &ctx, &w_o, m, d, d, false); // [m, d]
    drop(g0);

    let rank_gpus: Vec<Gpu> = gpus
        .iter()
        .map(|&gi| Gpu::new_on_index(gi as u32, &attn_kernels()).expect("rank placement"))
        .collect();

    // --- tensor-parallel: split heads; row-parallel output proj + all-reduce ---
    let coll = HostCollective::new(world);
    let results: Vec<std::sync::Mutex<Vec<f32>>> = (0..world).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    std::thread::scope(|s| {
        for (rank, g) in rank_gpus.iter().enumerate() {
            let (coll, results, x, w_qkv, w_o) = (&coll, &results, &x, &w_qkv, &w_o);
            s.spawn(move || {
                // QKV column-parallel by head: gather this rank's q,k,v rows
                // ([r*d_r..) within each of the q/k/v blocks of [3d, d]) -> [3*d_r, d].
                let mut w_qkv_r = Vec::with_capacity(3 * d_r * d);
                for blk in 0..3 {
                    let base = (blk * d + rank * d_r) * d;
                    w_qkv_r.extend_from_slice(&w_qkv[base..base + d_r * d]);
                }
                let ctx_r = attn_ctx(g, x, &w_qkv_r, b, t, d, nh_r, hd); // [m, d_r]
                // Row-parallel Wo: input columns [rank*d_r ..) of [d, d] -> [d, d_r].
                let mut w_o_r = vec![0f32; d * d_r];
                for i in 0..d {
                    w_o_r[i * d_r..(i + 1) * d_r].copy_from_slice(&w_o[i * d + rank * d_r..i * d + rank * d_r + d_r]);
                }
                let z_partial = matmul(g, &ctx_r, &w_o_r, m, d_r, d, false); // [m, d] partial
                let z = coll.all_reduce(rank, z_partial); // one all-reduce (Megatron g)
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
    eprintln!("TP attention ({world}-way, {nh_r} heads/rank) vs single-GPU: rel {rel:.2e}");
    assert!(rel < 1e-4, "tensor-parallel attention diverged: rel {rel:.2e}");
}
