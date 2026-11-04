// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate and the measurement for the split-K dense bf16 projection of a decode step (`kernels_cuda`, `cu/bf16_gemv_splitk.cu`,
//! `Fused::Bf16GemvSplitK`): up to four thin bf16 matrices that read one f32 activation of 2 to 16 rows, in one launch.
//!
//! Swedish Embedded AB implements native kernels for latency-bound LLM decode. If your team needs expertise in thin projections
//! that fill the card without extra reduction launches then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The claim: each output is the f64 product of the activation rounded to bf16 (what `cast_f32_bf16` makes of it, the operand the
//! library's tensor-core product multiplies) and the bf16 weight, to f32 accumulation; the result is the same bits launch after
//! launch (ascending reduction order); every output of the `m` rows is written and nothing else is.

use backend_api::DeviceBuffer;
use data::rng::Lcg;
use gpu_core::{Fused, Gpu};

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `cast_f32_bf16`'s rounding: nearest even, kept as an f32.
fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16)
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

struct Case {
    x: Vec<f32>,
    w: Vec<Vec<u16>>,
    xb: DeviceBuffer,
    wb: Vec<DeviceBuffer>,
    yb: Vec<DeviceBuffer>,
    unused: Vec<DeviceBuffer>,
    m: usize,
    k: usize,
    ns: Vec<usize>,
}

impl Case {
    fn new(gpu: &Gpu, m: usize, k: usize, ns: &[usize], seed: u64) -> Self {
        let x = Lcg::new(seed).vec_scaled(m * k, 1.0);
        let w: Vec<Vec<u16>> = ns.iter().enumerate().map(|(i, &n)| bf16_bits(n * k, seed + 1 + i as u64)).collect();
        let wb = w
            .iter()
            .map(|w| {
                let words: Vec<u32> = w.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16).collect();
                let b = gpu.storage(words.len() as u64);
                gpu.write(&b, &words);
                b
            })
            .collect();
        let yb = ns
            .iter()
            .map(|&n| {
                let y = gpu.storage((m * n) as u64);
                gpu.write(&y, &vec![0x7fc0_dead; m * n]);
                y
            })
            .collect();
        let unused = (ns.len()..4).map(|_| gpu.storage(1)).collect();
        Self { xb: gpu.storage_init("x", &x), x, w, wb, yb, unused, m, k, ns: ns.to_vec() }
    }

    fn step(&self, gpu: &Gpu) -> gpu_core::Step {
        let mut bufs: Vec<&DeviceBuffer> = vec![&self.xb];
        let mut params = vec![self.m as u32, self.k as u32];
        for i in 0..4 {
            if i < self.ns.len() {
                bufs.extend([&self.wb[i], &self.yb[i]]);
                params.push(self.ns[i] as u32);
            } else {
                bufs.extend([&self.xb, &self.unused[i - self.ns.len()]]);
                params.push(0);
            }
        }
        gpu.fused_step(Fused::Bf16GemvSplitK, &bufs, &params).expect("the split-K GEMV was declined on a CUDA device with block clusters")
    }

    fn read(&self, gpu: &Gpu) -> Vec<Vec<u32>> {
        self.ns.iter().enumerate().map(|(i, &n)| gpu.read(&self.yb[i], self.m * n).iter().map(|v| v.to_bits()).collect()).collect()
    }
}

#[test]
fn the_split_k_gemv_matches_the_oracle_of_the_bf16_operands() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) || !gpu.has_fused(Fused::Bf16GemvSplitK) {
        brain_testutil::skip_unavailable("the split-K bf16 GEMV needs a CUDA device with block clusters");
        return;
    }
    // (m, k, rows of each matrix): the GDN gate pair at the widths served, a K that leaves some of the 8 slices without a group, one
    // that leaves a warp without one, a single tile, four matrices.
    let cases: [(usize, usize, &[usize]); 9] = [
        (16, 5120, &[48, 48]),
        (2, 5120, &[48, 48]),
        (7, 2560, &[48, 48]),
        (13, 96, &[16]),
        (16, 32, &[16, 32]),
        (5, 8192, &[64, 16, 48, 32]),
        (3, 1056, &[48, 16]),
        (16, 4096, &[256]),
        (9, 640, &[32, 32, 32]),
    ];
    for (case, (m, k, ns)) in cases.into_iter().enumerate() {
        let c = Case::new(&gpu, m, k, ns, 1000 + 10 * case as u64);
        gpu.submit(&[], &[c.step(&gpu)]);
        gpu.poll_wait();
        let first = c.read(&gpu);
        for (i, &n) in ns.iter().enumerate() {
            for r in 0..m {
                for col in 0..n {
                    let terms: Vec<f64> = (0..k).map(|j| f64::from(round_bf16(c.x[r * k + j])) * f64::from(bf16_value(c.w[i][col * k + j]))).collect();
                    let (want, scale) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
                    let got = f64::from(f32::from_bits(first[i][r * n + col]));
                    assert!(got.is_finite(), "m={m} k={k} ns={ns:?}: output ({i}, {r}, {col}) was not written");
                    assert!((got - want).abs() <= 1e-5 * scale + 1e-12, "m={m} k={k} ns={ns:?}: output ({i}, {r}, {col}) is {got}, oracle {want} (scale {scale})");
                }
            }
        }
        // The same bits again (the reduction order is fixed).
        gpu.submit(&[], &[c.step(&gpu)]);
        gpu.poll_wait();
        assert_eq!(c.read(&gpu), first, "m={m} k={k} ns={ns:?}: a second launch returned other bits");
    }
}

#[test]
fn the_split_k_gemv_declines_what_it_does_not_serve() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the split-K bf16 GEMV needs a CUDA device with block clusters");
        return;
    }
    let ok = [4u32, 5120, 48, 48, 0, 0];
    assert!(Fused::Bf16GemvSplitK.serves(&ok));
    for (what, params) in [
        ("17 rows", [17u32, 5120, 48, 48, 0, 0]),
        ("0 rows", [0, 5120, 48, 48, 0, 0]),
        ("k not a multiple of 32", [4, 5128, 48, 48, 0, 0]),
        ("k past 8192", [4, 8224, 48, 48, 0, 0]),
        ("n not a multiple of 16", [4, 5120, 48, 40, 0, 0]),
    ] {
        assert!(!Fused::Bf16GemvSplitK.serves(&params), "{what}");
    }
}

/// 3,000 back-to-back launches of the gate pair at a decode step's rows and at the extremes of the shape, each returning the first
/// launch's bits (`#[ignore]`: a stress; knowledge 410 and 420 - a cluster kernel's fault showed only under launch pressure).
///
/// ```text
/// make test/one P=brain-gpu-core T=bf16_gemv_splitk_native ARGS="--ignored --nocapture back_to_back"
/// ```
#[test]
#[ignore]
fn back_to_back_launches_return_the_first_launch_s_bits() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) || !gpu.has_fused(Fused::Bf16GemvSplitK) {
        brain_testutil::skip_unavailable("the split-K bf16 GEMV needs a CUDA device with block clusters");
        return;
    }
    for (m, k, ns) in [(16usize, 5120usize, vec![48usize, 48]), (2, 5120, vec![48, 48]), (16, 5120, vec![4096, 48]), (8, 32, vec![16])] {
        let c = Case::new(&gpu, m, k, &ns, 77);
        gpu.submit(&[], &[c.step(&gpu)]);
        gpu.poll_wait();
        let first = c.read(&gpu);
        for chunk in 0..30 {
            let steps: Vec<_> = (0..100).map(|_| c.step(&gpu)).collect();
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            assert!(c.read(&gpu) == first, "m={m} k={k} ns={ns:?}: launch block {chunk} returned other bits than the first launch");
        }
    }
}

/// Device-timed cost of the gate pair (`#[ignore]`d: a measurement), the split-K kernel against the library's product of each
/// matrix (activation cast, split-K GEMM and reduction a matrix). Run on a CUDA box:
///
/// ```text
/// make test/one P=brain-gpu-core T=bf16_gemv_splitk_native ARGS="--ignored --nocapture gate_pair_cost"
/// ```
#[test]
#[ignore]
fn gate_pair_cost() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) || !gpu.has_fused(Fused::Bf16GemvSplitK) {
        brain_testutil::skip_unavailable("the split-K bf16 GEMV needs a CUDA device with block clusters");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    println!("{:<24} {:>10}", "m, k, matrices", "us");
    for m in [2usize, 4, 8, 16] {
        // Rotate through copies of the weights so each launch meets them as a decode step does.
        let cases: Vec<Case> = (0..8).map(|i| Case::new(&gpu, m, 5120, &[48, 48], 300 + i)).collect();
        let mut best = f64::INFINITY;
        for t in 0..12 {
            gpu.reset_kernel_times();
            let steps: Vec<_> = cases.iter().map(|c| c.step(&gpu)).collect();
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            if t > 0 {
                let total: f64 = gpu.kernel_times().expect("device timing").iter().map(|r| r.1).sum();
                best = best.min(total / cases.len() as f64 * 1e3);
            }
        }
        println!("{:<24} {:>10.2}", format!("m={m} k=5120 [48, 48]"), best);
    }
}
