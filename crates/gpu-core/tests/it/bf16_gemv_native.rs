// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate and the measurement for the dense bf16 decode GEMV (`kernels_cuda`, `cu/bf16_gemv.cu`) that
//! `gpu_core::native_upgrade` substitutes for `matmul_bf16_gemv.wgsl` on a CUDA device.
//!
//! Swedish Embedded AB implements native kernels for bf16 LLM decode. If your team needs expertise in streaming bf16
//! weights at the speed of the memory system then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The native kernel widens bf16 exactly and sums with fused multiply-adds in a different order than the WGSL tier, so the
//! claim is agreement with an f64 oracle of the stored weights to f32 rounding of the running sum, for the native kernel
//! and for the WGSL tier alike (`_ref` is the same source under a name the upgrade table does not know, so both run on
//! the same device from the same inputs). The selection assertion makes the comparison honest: the native kernel must
//! be the one that ran.

use data::rng::Lcg;
use backend_api::DeviceBuffer;
use gpu_core::{Dispatch, Fused, Gpu};
use std::sync::OnceLock;

fn kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    LIST.get_or_init(|| {
        let mut list: Vec<_> = kernels::template::GEMV_ROW_CAPACITIES.iter().map(|&cap| kernels::template::interned("matmul_bf16_gemv", kernels::MATMUL_BF16_GEMV, &[("MAXM", cap)]).expect("MAXM")).collect();
        list.push(("matmul_bf16_gemv_ref", list[3].1));
        list
    })
}

/// The pipeline index of the build that carries `m` rows (`GEMV_ROW_CAPACITIES` order), and the reference's.
fn kind_for(m: usize) -> usize {
    kernels::template::GEMV_ROW_CAPACITIES.iter().position(|&cap| cap as usize >= m).expect("m is at most 8")
}
const REF: usize = 4;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Finite bf16 bit patterns of ordinary magnitude.
fn bf16_bits(n: usize, seed: u64) -> Vec<u16> {
    let mut r = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let v = r.next_u32();
            (((v >> 31) as u16) << 15) | ((0x78 + (v >> 8) % 8) as u16) << 7 | (v & 0x7f) as u16
        })
        .collect()
}

fn run(gpu: &Gpu, (m, k, count, first, xr0): (usize, usize, usize, usize, usize), seed: u64) {
    let n = first + count + 3;
    let w = bf16_bits(n * k, seed);
    let words: Vec<u32> = w.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16).collect();
    let x = Lcg::new(seed + 1).vec_scaled((xr0 + m) * k, 1.0);
    let (xb, wb) = (gpu.storage_init("x", &x), gpu.storage(words.len() as u64));
    gpu.write(&wb, &words);
    let params = [m as u32, k as u32, count as u32, first as u32, xr0 as u32, 1, 0, 0];
    let (native, reference) = (gpu.storage((m * count) as u64), gpu.storage((m * count) as u64));
    gpu.write(&native, &vec![0x7fc0_dead; m * count]);
    gpu.write(&reference, &vec![0x7fc0_dead; m * count]);
    let kind = kind_for(m);
    let expect = format!("matmul_bf16_gemv_m{}", kernels::template::GEMV_ROW_CAPACITIES[kind]);
    assert_eq!(gpu.native_kernel_for(kind, &params), Some(expect.as_str()), "the native kernel must be the one that runs");
    assert_eq!(gpu.native_kernel_for(REF, &params), None);
    let step = |kind: usize, y: &backend_api::DeviceBuffer, blocks: u32| gpu.dispatch(kind, &[&xb, &wb, y], &params, Dispatch::Workgroups(blocks));
    gpu.submit(&[], &[step(kind, &native, count.div_ceil(4) as u32), step(REF, &reference, count.div_ceil(4) as u32)]);
    gpu.poll_wait();
    let mut want = Vec::new();
    for r in 0..m {
        for col in 0..count {
            let row = first + col;
            let terms: Vec<f64> = (0..k).map(|i| f64::from(x[(xr0 + r) * k + i]) * f64::from(bf16_value(w[row * k + i]))).collect();
            want.push((terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    for (label, got) in [("native", gpu.read(&native, m * count)), ("wgsl", gpu.read(&reference, m * count))] {
        for (i, (g, (w, scale))) in got.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "{label} m={m} k={k}: output {i} not written");
            assert!((f64::from(*g) - w).abs() <= 2e-5 * scale + 1e-12, "{label} m={m} k={k} count={count} first={first}: output {i} is {g}, oracle {w} (scale {scale})");
        }
    }
}

#[test]
fn the_native_bf16_gemv_matches_the_oracle_across_ragged_shapes() {
    let gpu = Gpu::new(kernel_list());
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the native bf16 GEMV runs on a CUDA device");
        return;
    }
    // (m, k, count, first, xr0): one row, k from one chunk to a few hundred, counts that leave a partial
    // block of 16 rows, a window that starts inside the weight, activation rows that start after rows that must not be read.
    for (i, shape) in [(1, 8, 1, 0, 0), (1, 2560, 33, 5, 2), (1, 4096, 4096, 0, 0), (1, 520, 17, 0, 1), (1, 9216, 100, 3, 2), (1, 2568, 21, 1, 0), (2, 2560, 70, 7, 1), (3, 1032, 45, 0, 0), (4, 2560, 128, 0, 2), (5, 3000, 33, 1, 1), (8, 9216, 100, 3, 2), (8, 520, 17, 0, 0), (6, 4104, 31, 2, 0)].into_iter().enumerate() {
        run(&gpu, shape, 100 + i as u64);
    }
}

/// Up to four matrices that read one activation, in one launch, against each matrix's own single-matrix launch: every output
/// bit for bit (the multi kernel runs the single kernel's block bodies).
#[test]
fn the_multi_gemv_equals_the_single_launches_bit_for_bit() {
    let gpu = Gpu::new(kernel_list());
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the native bf16 GEMV runs on a CUDA device");
        return;
    }
    // (m, k, the rows of each matrix): the GDN projections (one big, one skinny), three of a GQA layer, two of an MLP.
    for (case, (m, k, ns)) in [(1usize, 2560usize, vec![300usize, 33, 32, 280]), (1, 2560, vec![512, 128, 128]), (2, 1032, vec![70, 41]), (5, 2560, vec![200, 32, 32, 100]), (8, 520, vec![33, 64, 17])].into_iter().enumerate() {
        let x = Lcg::new(500 + case as u64).vec_scaled(m * k, 1.0);
        let xb = gpu.storage_init("x", &x);
        let mats: Vec<DeviceBuffer> = ns
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let w = bf16_bits(n * k, 600 + (case * 4 + i) as u64);
                let words: Vec<u32> = w.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16).collect();
                let b = gpu.storage(words.len() as u64);
                gpu.write(&b, &words);
                b
            })
            .collect();
        let single: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage((m * n) as u64)).collect();
        let multi: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage((m * n) as u64)).collect();
        let kind = kind_for(m);
        for (i, &n) in ns.iter().enumerate() {
            let params = [m as u32, k as u32, n as u32, 0, 0, 1, 0, 0];
            gpu.submit(&[], &[gpu.dispatch(kind, &[&xb, &mats[i], &single[i]], &params, Dispatch::Workgroups((n as u32).div_ceil(4)))]);
        }
        let unused: Vec<DeviceBuffer> = (ns.len()..4).map(|_| gpu.storage(1)).collect();
        let mut bufs: Vec<&DeviceBuffer> = vec![&xb];
        let mut params = vec![m as u32, k as u32];
        for i in 0..4 {
            if i < ns.len() {
                bufs.extend([&mats[i], &multi[i]]);
                params.push(ns[i] as u32);
            } else {
                bufs.extend([&xb, &unused[i - ns.len()]]);
                params.push(0);
            }
        }
        let step = gpu.fused_step(Fused::Bf16GemvMulti, &bufs, &params).expect("the multi GEMV was declined on a CUDA device");
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        for (i, &n) in ns.iter().enumerate() {
            let (a, b) = (gpu.read(&multi[i], m * n), gpu.read(&single[i], m * n));
            assert!(a.iter().all(|v| v.is_finite()) && a.iter().any(|v| *v != 0.0), "matrix {i}: an output was left unwritten (m={m} k={k} ns={ns:?})");
            assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "matrix {i} differs from its own single-matrix launch (m={m} k={k} ns={ns:?})");
        }
    }
}

/// Device-timed cost at the EdgeRock decode shapes, in GB/s of weight bytes (`#[ignore]`d: a measurement, not a gate).
/// `cargo test --release -p brain-gpu-core --test bf16_gemv_native -- --ignored --nocapture`. Time is the kernel's own, from
/// the backend's device events; each shape is timed with the weights cold (a buffer larger than L2 is rotated through).
#[test]
#[ignore]
fn decode_bf16_gemv_cost_at_the_edgerock_shapes() {
    let gpu = Gpu::new(kernel_list());
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let x = gpu.storage_init("x", &vec![0.5f32; 16384]);
    println!("{:<28} {:>10} {:>10}", "shape [n, k] rows", "best us", "GB/s");
    for m in [1u32, 2, 4, 8] {
        for (n, k) in [(9216u32, 2560u32), (8192, 2560), (2560, 9216), (2560, 4096), (4096, 2560), (1024, 2560), (2560, 2560), (32, 2560), (248_320, 2560)] {
            // Rotate through enough copies that every call meets cold weights.
            let bytes = u64::from(n) * u64::from(k) * 2;
            let copies = (128u64 << 20).div_ceil(bytes).max(2) as usize;
            let ws: Vec<_> = (0..copies).map(|_| gpu.storage(bytes / 4)).collect();
            let y = gpu.storage(u64::from(m) * u64::from(n));
            let params = [m, k, n, 0, 0, 1, 0, 0];
            let kind = kind_for(m as usize);
            let mut best = f64::INFINITY;
            for t in 0..12 {
                gpu.reset_kernel_times();
                let steps: Vec<_> = ws.iter().map(|w| gpu.dispatch(kind, &[&x, w, &y], &params, Dispatch::Workgroups(n.div_ceil(16)))).collect();
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                if t > 0 {
                    let rows = gpu.kernel_times().expect("device timing");
                    let (_, ms, calls) = rows.iter().find(|(name, _, _)| name.contains("matmul_bf16_gemv")).expect("the kernel ran");
                    best = best.min(ms / *calls as f64 * 1e3);
                }
            }
            println!("[{n:>6}, {k:>5}] m={m:<2}          {:>10.1} {:>10.0}", best, bytes as f64 / (best * 1e-6) / 1e9);
        }
    }
}
