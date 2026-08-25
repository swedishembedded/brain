// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPU conv microbench: `conv_act_reg` (and friends) on ZipDepth's dominant
//! layer shapes, reported as min-of-N GFLOP/s. Ignored by default; run with
//!
//!   DISPLAY= cargo test --release -p brain-backend-wgpu -- --ignored --nocapture bench_conv_gpu
//!
//! Purpose: a tight iterate loop for kernel tuning — editing a .wgsl and
//! re-running this rebuilds two crates, not the whole release binary. The
//! numbers are also the calibration for "how far from roofline are we": compare
//! them against the running box's own datasheet fp32 peak and memory bandwidth
//! (an MTL Arc iGPU shares LPDDR5 with the host, so both are modest).

use backend_wgpu::WgpuBackend;

struct Shape {
    label: &'static str,
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    pad: u32,
}

/// ZipDepth base @ 608x384 — the layers that dominate the frame.
const SHAPES: &[Shape] = &[
    Shape { label: "stage1 48x48 3x3 @152x96", cin: 48, cout: 48, h: 96, w: 152, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage2 96x96 3x3 @76x48", cin: 96, cout: 96, h: 48, w: 76, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage3 192x192 3x3 @38x24", cin: 192, cout: 192, h: 24, w: 38, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage4 384x384 3x3 @19x12", cin: 384, cout: 384, h: 12, w: 19, k: 3, stride: 1, pad: 1 },
    Shape { label: "stem 24x48 3x3 s2 @304x192", cin: 24, cout: 48, h: 192, w: 304, k: 3, stride: 2, pad: 1 },
    Shape { label: "1x1 48x48 @152x96", cin: 48, cout: 48, h: 96, w: 152, k: 1, stride: 1, pad: 0 },
];

#[test]
#[ignore]
fn bench_conv_gpu() {
    let gpu = WgpuBackend::new(&[
        ("conv_act", kernels::CONV_ACT),
        ("conv_act_reg", kernels::CONV_ACT_REG),
    ]);
    let reps = 20u32;
    for s in SHAPES {
        let ho = (s.h + 2 * s.pad - s.k) / s.stride + 1;
        let wo = (s.w + 2 * s.pad - s.k) / s.stride + 1;
        let x: Vec<f32> = (0..(s.cin * s.h * s.w) as usize).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
        let w: Vec<f32> = (0..(s.cout * s.cin * s.k * s.k) as usize).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
        let sb: Vec<f32> = (0..2 * s.cout as usize).map(|i| if i % 2 == 0 { 1.0 } else { 0.1 }).collect();
        let xb = gpu.storage_init("x", &x);
        let wb = gpu.storage_init("w", &w);
        let sbb = gpu.storage_init("sb", &sb);
        let yb = WgpuBackend::storage(&gpu, (s.cout * ho * wo) as u64);

        let flops = 2.0 * (s.cout as f64) * (s.cin as f64) * (s.k as f64) * (s.k as f64) * (ho as f64) * (wo as f64);
        let params = [1, s.cin, s.h, s.w, s.cout, s.k, s.stride, s.pad, ho, wo, 2u32];

        // (kernel index, threads) per variant.
        let ntc = s.cout.div_ceil(8);
        let npq = (ho * wo).div_ceil(4);
        for (name, kind, threads) in
            [("naive", 0usize, s.cout * ho * wo), ("reg", 1usize, ntc * npq)]
        {
            // Warm once, then min-of-5 batches of `reps` back-to-back dispatches.
            let step = WgpuBackend::step(&gpu, kind, &[&xb, &wb, &sbb, &yb], &params, threads);
            WgpuBackend::submit(&gpu, &[], &[step]);
            gpu.poll_wait();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let steps: Vec<_> = (0..reps)
                    .map(|_| WgpuBackend::step(&gpu, kind, &[&xb, &wb, &sbb, &yb], &params, threads))
                    .collect();
                let t0 = std::time::Instant::now();
                WgpuBackend::submit(&gpu, &[], &steps);
                gpu.poll_wait();
                best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
            }
            println!(
                "{:<28} {:>6}: {:8.3} ms  {:8.1} GFLOP/s",
                s.label,
                name,
                best * 1e3,
                flops / best / 1e9
            );
        }
    }
}

/// The question the flat bench cannot answer: does a DEPENDENT chain (each
/// dispatch reads the previous one's output, like a real forward) pay a
/// pipeline-drain per barrier? Ping-pongs x<->y through the same reg conv;
/// compare per-dispatch time against `bench_conv_gpu`'s independent runs.
#[test]
#[ignore]
fn bench_conv_gpu_chain() {
    let gpu = WgpuBackend::new(&[("conv_act_reg", kernels::CONV_ACT_REG)]);
    // stage1-like same-size in/out: 48ch 152x96, 3x3 s1 p1.
    let (c, h, w) = (48u32, 96u32, 152u32);
    let n = (c * h * w) as usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01).collect();
    let a = gpu.storage_init("a", &x);
    let b = gpu.storage_init("b", &x);
    let wt: Vec<f32> = (0..(c * c * 9) as usize).map(|i| (i % 13) as f32 * 0.001).collect();
    let wb = gpu.storage_init("w", &wt);
    let sb: Vec<f32> = (0..2 * c as usize).map(|i| if i % 2 == 0 { 0.05 } else { 0.0 }).collect();
    let sbb = gpu.storage_init("sb", &sb);
    let params = [1, c, h, w, c, 3, 1, 1, h, w, 2u32];
    let threads = c.div_ceil(8) * (h * w).div_ceil(4);
    let flops = 2.0 * (c as f64) * (c as f64) * 9.0 * (h as f64) * (w as f64);

    let reps = 40;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let steps: Vec<_> = (0..reps)
            .map(|i| {
                let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
                WgpuBackend::step(&gpu, 0, &[src, &wb, &sbb, dst], &params, threads)
            })
            .collect();
        let t0 = std::time::Instant::now();
        WgpuBackend::submit(&gpu, &[], &steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
    }
    println!(
        "dependent chain 48ch @152x96: {:8.3} ms/dispatch  {:8.1} GFLOP/s",
        best * 1e3,
        flops / best / 1e9
    );
}

/// Pure per-hop overhead: a dependent chain of TINY dispatches (1x1 conv on
/// 64 values). If this costs ~the same per hop as the big chain's delta, the
/// hop cost is a FIXED pipeline drain (dispatch-count bound — fuse harder);
/// if it is ~microseconds, the big chain's delta is lost execution overlap
/// (work-bound — fusing helps less than making kernels faster).
#[test]
#[ignore]
fn bench_chain_overhead() {
    let gpu = WgpuBackend::new(&[("conv_act_reg", kernels::CONV_ACT_REG)]);
    let (c, h, w) = (8u32, 4u32, 2u32);
    let n = (c * h * w) as usize;
    let a = gpu.storage_init("a", &vec![0.01f32; n]);
    let b = gpu.storage_init("b", &vec![0.01f32; n]);
    let wt = gpu.storage_init("w", &vec![0.001f32; (c * c) as usize]);
    let sb = gpu.storage_init("sb", &{
        let mut v = vec![0.0f32; 2 * c as usize];
        v.iter_mut().step_by(2).for_each(|x| *x = 0.5);
        v
    });
    let params = [1, c, h, w, c, 1, 1, 0, h, w, 2u32];
    let threads = c.div_ceil(8) * (h * w).div_ceil(4);

    let reps = 200;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let steps: Vec<_> = (0..reps)
            .map(|i| {
                let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
                WgpuBackend::step(&gpu, 0, &[src, &wt, &sb, dst], &params, threads)
            })
            .collect();
        let t0 = std::time::Instant::now();
        WgpuBackend::submit(&gpu, &[], &steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
    }
    println!("tiny dependent chain: {:8.1} us/dispatch", best * 1e6);
}
