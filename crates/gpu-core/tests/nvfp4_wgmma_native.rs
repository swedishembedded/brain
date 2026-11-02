// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `matmul_nvfp4_wgmma` (`kernels_cuda`, `cu/matmul_nvfp4_wgmma.cu`): an NVFP4 weight against E4M3 activations on the
//! Hopper warpgroup tensor cores, the weight tile decoded to E4M3 in shared memory.
//!
//! Swedish Embedded AB implements tensor-core inference kernels for 4-bit checkpoints for its clients. If your team needs
//! expertise in serving NVFP4 releases at prefill speed on Hopper without converting them, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The kernel's operand precision is stated, not hidden: the weight the tensor core sees is `RN_e4m3(e2m1(code) * e4m3(scale) / 8)` (one
//! rounding, relative error at most 2^-4, subnormal range included), the activation is the E4M3 byte it is handed. The oracle is the f64
//! product of exactly those operands; the claim is agreement with it to the accumulation precision of the FP8 tensor core
//! (`ACC_REL` of the sum of absolute terms), on ragged shapes, with every output written (the output buffer starts NaN).
//! The second test reports how far the E4M3 weight is from the exact decoded weight (the operand-precision bound).

use data::rng::Lcg;
use gpu_core::{Fused, Gpu};

/// The FP8 tensor core accumulates with fewer than 24 mantissa bits inside one instruction; the bound is relative to the sum of |terms|.
const ACC_REL: f64 = 2e-3;

fn is_hopper(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::WGMMA_MIN_CC)
}

fn e4m3(b: u8) -> f64 {
    let (e, m) = (i32::from(b >> 3) & 15, f64::from(b & 7));
    let v = if e == 0 { m / 512.0 } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) };
    if b & 128 != 0 { -v } else { v }
}

fn e2m1(c: u32) -> f64 {
    let v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// Round `v` to the nearest finite E4M3 value, ties to the even mantissa, saturating at 448.
fn round_e4m3(v: f64) -> f64 {
    let a = v.abs().min(448.0);
    let mut best = (f64::MAX, 0u8);
    for b in 0u8..0x7f {
        let d = (e4m3(b) - a).abs();
        if d < best.0 || (d == best.0 && b % 2 == 0) {
            best = (d, b);
        }
    }
    let r = e4m3(best.1);
    if v < 0.0 { -r } else { r }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, b)| w | u32::from(*b) << (8 * i))).collect()
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> backend_api::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

struct Problem {
    codes: Vec<u32>,
    scales: Vec<u8>,
    xq: Vec<u8>,
    sx: Vec<f32>,
    global: f32,
}

fn problem(m: usize, k: usize, n: usize, seed: u64, wide_scales: bool) -> Problem {
    let mut r = Lcg::new(seed);
    let codes = (0..n * k / 8).map(|_| r.next_u32()).collect();
    // Positive E4M3 scale bytes; `wide_scales` covers the whole range a checkpoint can store (subnormals to 448).
    let scales = (0..n * k / 16).map(|_| if wide_scales { 1 + (r.next_u32() % 0x7e) as u8 } else { 0x30 + (r.next_u32() % 0x20) as u8 }).collect();
    let xq = (0..m * k).map(|_| loop { let b = r.next_u32() as u8; if b & 0x7f != 0x7f { break b; } }).collect();
    let sx = (0..m).map(|_| 0.001 + (r.next_u32() % 1000) as f32 / 1000.0).collect();
    Problem { codes, scales, xq, sx, global: 0.0071 }
}

fn weight(p: &Problem, k: usize, row: usize, i: usize, rounded: bool) -> f64 {
    let exact = e2m1(p.codes[row * k / 8 + i / 8] >> (4 * (i % 8))) * e4m3(p.scales[row * k / 16 + i / 16]);
    if rounded { round_e4m3(exact / 8.0) * 8.0 } else { exact }
}

fn run(gpu: &Gpu, p: &Problem, m: usize, k: usize, n: usize) -> Vec<f32> {
    let (xq, w, s, sx) = (upload_u32(gpu, &words(&p.xq)), upload_u32(gpu, &p.codes), upload_u32(gpu, &words(&p.scales)), gpu.storage_init("sx", &p.sx));
    let y = gpu.storage_init("y", &vec![f32::NAN; m * n]);
    let params = [m as u32, k as u32, n as u32, p.global.to_bits()];
    let step = gpu.fused_step(Fused::MatmulNvfp4Wgmma, &[&xq, &w, &s, &sx, &y], &params).expect("served shape");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    gpu.read(&y, m * n)
}

#[test]
fn the_wgmma_product_matches_the_oracle_of_its_operands() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_hopper(&gpu) || !gpu.has_fused(Fused::MatmulNvfp4Wgmma) {
        return brain_testutil::skip_unavailable("the NVFP4 wgmma product needs a CUDA device with warpgroup tensor cores");
    }
    let base = data::rng::random_seed();
    let shapes = [(1usize, 256usize, 2usize), (5, 256, 130), (127, 512, 128), (128, 256, 128), (129, 768, 254), (300, 5120, 300), (33, 1024, 62), (200, 17408, 258), (1000, 1280, 5120)];
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let p = problem(m, k, n, base.wrapping_add(i as u64 * 31), i % 2 == 1);
        let got = run(&gpu, &p, m, k, n);
        let mut worst = 0.0f64;
        let (row_step, col_step) = (if m > 64 { 37 } else { 1 }, if n > 64 { 29 } else { 1 });
        for a in (0..m).step_by(row_step) {
            for c in (0..n).step_by(col_step) {
                let terms: Vec<f64> = (0..k).map(|j| e4m3(p.xq[a * k + j]) * weight(&p, k, c, j, true)).collect();
                let f = f64::from(p.sx[a]) * f64::from(p.global);
                let (sum, mag) = (terms.iter().sum::<f64>() * f, terms.iter().map(|t| t.abs()).sum::<f64>() * f);
                let g = f64::from(got[a * n + c]);
                assert!(got[a * n + c].is_finite(), "m={m} k={k} n={n}: output ({a},{c}) not written");
                let err = (g - sum).abs() / (mag + 1e-30);
                worst = worst.max(err);
                assert!(err <= ACC_REL, "m={m} k={k} n={n}: output ({a},{c}) is {g}, oracle {sum} (sum of |terms| {mag})");
            }
        }
        eprintln!("m={m} k={k} n={n}: worst error {worst:.2e} of the sum of |terms|");
    }
}

#[test]
fn the_e4m3_weight_is_within_a_sixteenth_of_the_exact_decoded_weight() {
    let p = problem(1, 256, 64, data::rng::random_seed(), true);
    let (mut worst, mut inexact) = (0.0f64, 0usize);
    for row in 0..64 {
        for i in 0..256 {
            let (exact, rounded) = (weight(&p, 256, row, i, false), weight(&p, 256, row, i, true));
            if exact != rounded {
                inexact += 1;
            }
            // Relative to the value, except where it falls under the E4M3 subnormal step (2^-9, times 8).
            if exact.abs() >= 8.0 * 2f64.powi(-6) {
                worst = worst.max(((rounded - exact) / exact).abs());
            }
        }
    }
    eprintln!("weights rounded: {inexact} of {}; worst relative error {worst:.4}", 64 * 256);
    assert!(worst <= 1.0 / 16.0, "worst relative error {worst}");
}

/// Median-free minimum of device-event times of `Fused::MatmulNvfp4Wgmma` and, where it serves, the 64-row `mma` build, against the best FP8 candidate,
/// at the 27B's MLP and head shapes. Weights are cycled past the L2 for the streaming row counts. Run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing table, not a gate"]
fn kernel_times_against_fp8() {
    let gpu = gpu_core::testgpu::dev(&[("matmul_fp8_gemv", kernels::MATMUL_FP8_GEMV)]);
    if !is_hopper(&gpu) || !gpu.has_fused(Fused::MatmulNvfp4Wgmma) {
        return brain_testutil::skip_unavailable("the NVFP4 wgmma product needs a CUDA device with warpgroup tensor cores");
    }
    let timing = gpu.set_kernel_timing(true);
    assert!(timing, "device event timing");
    let mut r = Lcg::new(7);
    for (label, k, n) in [("gate/up", 5120usize, 17408usize), ("down", 17408, 5120), ("lm_head", 5120, 248320)] {
        let weight_bytes = n * k / 2;
        let copies = (192usize << 20).div_ceil(weight_bytes).clamp(1, 4);
        let ws: Vec<_> = (0..copies).map(|_| upload_u32(&gpu, &(0..n * k / 8).map(|_| r.next_u32()).collect::<Vec<_>>())).collect();
        let ss: Vec<_> = (0..copies).map(|_| upload_u32(&gpu, &words(&(0..n * k / 16).map(|_| 0x30 + (r.next_u32() % 0x20) as u8).collect::<Vec<_>>()))).collect();
        for m in [16usize, 32, 64, 256, 1024, 4096, 8192] {
            let xq = upload_u32(&gpu, &(0..m * k / 4).map(|_| 0x3838_3838u32 ^ (r.next_u32() & 0x0707_0707)).collect::<Vec<_>>());
            let sx = gpu.storage_init("sx", &vec![0.01f32; m]);
            let xb = upload_u32(&gpu, &(0..m * k / 2).map(|_| 0x3f80_3f80u32).collect::<Vec<_>>());
            let y = gpu.storage(m as u64 * n as u64);
            let time = |which: Fused, params: &[u32], bufs: &dyn Fn(usize) -> Vec<backend_api::DeviceBuffer>, name: &str| -> Option<f64> {
                let steps: Vec<_> = (0..12).map(|i| { let b = bufs(i % copies); gpu.fused_step(which, &b.iter().collect::<Vec<_>>(), params) }).collect::<Option<Vec<_>>>()?;
                gpu.set_kernel_timing(true);
                gpu.submit(&[], &steps[..2]);
                gpu.poll_wait();
                gpu.reset_kernel_times();
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                let t = gpu.kernel_times()?;
                let (ms, calls) = t.iter().filter(|e| e.0 == name).fold((0.0, 0u64), |a, e| (a.0 + e.1, a.1 + e.2));
                (calls > 0).then(|| ms / calls as f64 * 1e3)
            };
            let wg = time(
                Fused::MatmulNvfp4Wgmma,
                &[m as u32, k as u32, n as u32, 0.0071f32.to_bits()],
                &|i| vec![xq.clone(), ws[i].clone(), ss[i].clone(), sx.clone(), y.clone()],
                "native:matmul_nvfp4_wgmma",
            );
            let mma = (m <= 64).then(|| {
                let which = match m { 0..=8 => Fused::MatmulNvfp4Mma8, 9..=16 => Fused::MatmulNvfp4Mma16, 17..=32 => Fused::MatmulNvfp4Mma32, _ => Fused::MatmulNvfp4Mma64 };
                let name = format!("native:matmul_nvfp4_mma{}", match m { 0..=8 => 8, 9..=16 => 16, 17..=32 => 32, _ => 64 });
                time(which, &[m as u32, k as u32, n as u32, 0, 0, 0.0071f32.to_bits(), 0, 0], &|i| vec![xb.clone(), ws[i].clone(), ss[i].clone(), y.clone()], &name)
            }).flatten();
            let verdicts = gpu_core::native_time_fp8_shape(&gpu, [m as u32, (k / 4) as u32, n as u32]);
            if verdicts.is_empty() { eprintln!("no FP8 candidate times {m} x {k} x {n}"); }
            for (name, v) in &verdicts { eprintln!("   fp8 {name}: {v:?}"); }
            let fp8 = verdicts
                .into_iter()
                .filter_map(|(name, v)| if let gpu_core::NativeVerdict::Timed { us, .. } = v { Some((us, name)) } else { None })
                .fold(None, |b: Option<(f64, String)>, c| if b.as_ref().is_none_or(|b| c.0 < b.0) { Some(c) } else { b });
            eprintln!("{label:8} m={m:5}  wgmma {:>9.1} us   mma {:>9}   fp8 best {}", wg.unwrap_or(f64::NAN), mma.map_or("-".into(), |v| format!("{v:.1}")), fp8.map_or("-".into(), |(us, n)| format!("{us:.1} ({n})")));
        }
    }
}
